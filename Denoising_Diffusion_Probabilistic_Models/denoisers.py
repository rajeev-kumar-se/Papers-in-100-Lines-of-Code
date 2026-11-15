# Drop-in replacements for the UNet in unet.py.
#
# The ONLY contract diffusion_models.py needs from the denoiser is:
#
#     eps_pred = model(x, t)
#         x: [B, C, 32, 32]  noisy image, float32
#         t: [B]             timestep, torch.long, **0-based** (caller passes t - 1)
#     eps_pred: [B, C, 32, 32]  predicted noise, same shape as x
#
# Nothing else is used: no .sample(), no .ch, no attention maps. So any
# nn.Module with that signature can be swapped in, e.g.
#
#     denoiser_model = DiTDenoiser()            # instead of UNet()
#     diffusion_model = DiffusionModel(1000, denoiser_model).to(device)
#
# What actually matters for quality is the same in every architecture:
#   1. the timestep must reach every layer (otherwise one net cannot act
#      like T different denoisers),
#   2. the output must be able to take any real value (eps ~ N(0, I)),
#   3. the receptive field must cover the structures in the image.
import math

import torch
import torch.nn as nn

# reuse the sinusoidal embedding from the UNet file instead of rewriting it
from unet import get_timestep_embedding


def _groups(ch):
    """GroupNorm needs num_groups to divide ch; unet.py hardcodes 32, which
    breaks as soon as you shrink the width below 32. Stay safe here."""
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


# ---------------------------------------------------------------- 1. ConvNet
class ConvNetDenoiser(nn.Module):
    """
    Plain fully-convolutional stack, no downsampling. The cheapest thing that
    still works: every conv is 3x3 at full resolution, so 6 convs give a
    13x13 receptive field, enough for MNIST strokes.

    Timestep -> scale/shift of every block (FiLM). That is the whole trick:
    "denoise this much" becomes a per-channel affine change of the features.

    [B, 1, 32, 32] -> conv -> [B, ch, 32, 32] -> n x (FiLM + conv + SiLU)
                   -> conv -> [B, 1, 32, 32]
    """

    def __init__(self, ch=64, in_ch=1, depth=6, t_dim=128):
        super().__init__()
        self.in_ch = in_ch

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim * 4),
        )
        self.t_dim = t_dim

        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            # one (scale, shift) pair per block, per channel
            [nn.Linear(t_dim * 4, ch * 2) for _ in range(depth)]
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(ch, ch, 3, padding=1) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(_groups(ch), ch) for _ in range(depth)])

        self.norm_out = nn.GroupNorm(_groups(ch), ch)
        # no activation after this conv: eps is Gaussian noise, any real value
        self.conv_out = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.t_mlp(get_timestep_embedding(t, self.t_dim))  # [B, 4*t_dim]

        h = self.conv_in(x)  # [B, 1, 32, 32] -> [B, ch, 32, 32]
        for proj, conv, norm in zip(self.blocks, self.convs, self.norms):
            scale, shift = proj(temb).chunk(2, dim=-1)  # [B, ch], [B, ch]
            h = norm(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = torch.nn.functional.silu(h)
            h = conv(h)

        h = torch.nn.functional.silu(self.norm_out(h))
        return self.conv_out(h)  # [B, ch, 32, 32] -> [B, 1, 32, 32]


# ---------------------------------------------------------------- 2. TinyUNet
class TinyUNet(nn.Module):
    """
    Same skeleton as UNet, stripped down: 2 down/up stages, 1 ResNet block per
    level, no attention, width 32 instead of 128.

    Useful because the reference UNet is ~50M parameters and this is ~0.5M, so
    it overfits/underfits differently and trains far faster. Skips are kept:
    they are what makes a UNet beat a plain encoder-decoder at this size.
    """

    def __init__(self, ch=32, in_ch=1, t_dim=None):
        super().__init__()
        t_dim = t_dim or ch
        self.t_dim = t_dim
        self.in_ch = in_ch

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, 4 * t_dim),
            nn.SiLU(),
            nn.Linear(4 * t_dim, 4 * t_dim),
        )

        class Block(nn.Module):
            """GroupNorm -> SiLU -> Conv -> + temb -> GroupNorm -> SiLU -> Conv -> + x"""

            def __init__(self, cin, cout, tdim):
                super().__init__()
                self.norm1 = nn.GroupNorm(_groups(cin), cin)
                self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
                self.emb = nn.Linear(tdim, cout)
                self.norm2 = nn.GroupNorm(_groups(cout), cout)
                self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
                self.skip = (
                    nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
                )

            def forward(self, x, temb):
                h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
                h = h + self.emb(torch.nn.functional.silu(temb))[:, :, None, None]
                h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
                return self.skip(x) + h

        def down(c):
            return nn.Conv2d(c, c, 3, stride=2, padding=1)

        def up(c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c, c, 3, padding=1),
            )

        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        # 32x32 -> 16x16 -> 8x8
        self.d1 = Block(ch, ch, 4 * t_dim)
        self.dn1 = down(ch)
        self.d2 = Block(ch, 2 * ch, 4 * t_dim)
        self.dn2 = down(2 * ch)
        self.mid = Block(2 * ch, 2 * ch, 4 * t_dim)
        # 8x8 -> 16x16 -> 32x32, each level concatenates its skip (+2ch)
        self.up2 = up(2 * ch)
        self.u2 = Block(4 * ch, 2 * ch, 4 * t_dim)
        self.up1 = up(2 * ch)
        self.u1 = Block(3 * ch, ch, 4 * t_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(_groups(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        temb = self.t_mlp(get_timestep_embedding(t, self.t_dim))

        h0 = self.conv_in(x)          # [B, ch,   32, 32]
        h1 = self.d1(h0, temb)        # [B, ch,   32, 32]
        h2 = self.d2(self.dn1(h1), temb)  # [B, 2ch, 16, 16]
        h = self.mid(self.dn2(h2), temb)  # [B, 2ch,  8, 8]

        h = self.u2(torch.cat((self.up2(h), h2), dim=1), temb)  # [B, 2ch, 16, 16]
        h = self.u1(torch.cat((self.up1(h), h1), dim=1), temb)  # [B, ch,  32, 32]
        return self.out(h)            # [B, in_ch, 32, 32]


# ------------------------------------------------------------------- 3. DiT
class DiTDenoiser(nn.Module):
    """
    Diffusion Transformer, the architecture behind SD3 / Sora (Peebles & Xie,
    2023). No convolutions at all:

      1. cut the image into patch x patch squares      [B,1,32,32] -> [B,n,n,1*p*p]
      2. linear-embed each patch                        -> [B, N, dim]  (N = 256 for p=2)
      3. add position embedding + timestep embedding    -> [B, N, dim]
      4. transformer blocks with adaLN-Zero:
           every block asks the timestep for (shift, scale, gate) x 2
           gates are zero-init, so each block starts as the identity
      5. unpatchify back to an image                     -> [B,1,32,32]

    Attention is global from layer 1, so the receptive field is the whole image
    immediately -- the opposite trade-off from the UNet, where it grows one
    downsample at a time. Cost is O(N^2) per layer.
    """

    def __init__(self, dim=192, depth=6, heads=6, patch=2, in_ch=1, t_dim=None):
        super().__init__()
        t_dim = t_dim or dim
        self.patch = patch
        self.in_ch = in_ch
        # 32 / 2 = 16 -> 256 tokens (+1 class-free token below)
        self.n = 32 // patch

        self.patch_embed = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n * self.n, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        class Block(nn.Module):
            """adaLN-Zero: modulate, then gate the residual."""

            def __init__(self, dim, heads):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
                self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
                self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

                def mlp():
                    return nn.Sequential(
                        nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
                    )

                self.mlp = mlp()
                # shift, scale, gate for attention + shift, scale, gate for mlp
                self.mod = nn.Linear(dim, 6 * dim)
                nn.init.zeros_(self.mod.weight)  # <- Zero: blocks start as identity
                nn.init.zeros_(self.mod.bias)

            def forward(self, x, c):
                s1, sc1, g1, s2, sc2, g2 = self.mod(c).chunk(6, dim=-1)
                h = self.norm1(x) * (1 + sc1[:, None]) + s1[:, None]
                h, _ = self.attn(h, h, h, need_weights=False)
                x = x + g1[:, None] * h
                h = self.norm2(x) * (1 + sc2[:, None]) + s2[:, None]
                x = x + g2[:, None] * self.mlp(h)
                return x

        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])

        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod_out = nn.Linear(dim, 2 * dim)
        self.head = nn.Linear(dim, patch * patch * in_ch)

        # final adaLN + head start at zero too
        for lin in (self.mod_out, self.head):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, t):
        B, C, H, W = x.shape

        h = self.patch_embed(x)                       # [B, dim, 16, 16]
        h = h.flatten(2).transpose(1, 2)              # [B, 256, dim]
        h = h + self.pos

        c = self.t_mlp(get_timestep_embedding(t, self.t_mlp[0].in_features))
        for blk in self.blocks:
            h = blk(h, c)

        shift, scale = self.mod_out(torch.nn.functional.silu(c)).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        eps = self.head(h)                            # [B, 256, p*p*C]

        # unpatchify: [B, n*n, p*p*C] -> [B, C, n*p, n*p]
        n, p = self.n, self.patch
        eps = eps.reshape(B, n, n, p, p, C)
        eps = eps.permute(0, 5, 1, 3, 2, 4).reshape(B, C, n * p, n * p)
        assert eps.shape == (B, C, H, W)
        return eps


# ---------------------------------------------------------------- 4. MLP / SIREN
class MLPDenoiser(nn.Module):
    """
    The "no spatial inductive bias at all" baseline: flatten the image, run a
    plain MLP, reshape back. This is close to what the original 2015 score
    matching work used, and it still trains -- it just needs many more
    parameters to express what one 3x3 conv does for free.

    Keep it around as the sanity-check baseline: if a UNet does not beat this
    on MNIST, something in the schedule or loss is wrong.
    """

    def __init__(self, in_ch=1, img_size=32, hidden=1024, depth=4, t_dim=128):
        super().__init__()
        self.in_ch = in_ch
        self.img_size = img_size
        self.t_dim = t_dim
        flat = in_ch * img_size * img_size

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        layers = [nn.Linear(flat, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, flat)]
        self.net = nn.Sequential(*layers)

        # timestep injected into every hidden layer: weight * x + (a * temb + b)
        self.emb = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])

    def forward(self, x, t):
        B = x.shape[0]
        h = x.reshape(B, -1)                          # [B, 1*32*32]
        temb = self.t_mlp(get_timestep_embedding(t, self.t_dim))  # [B, hidden]

        i = 0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                h = layer(h) + self.emb[i](temb)      # same temb at every depth
                i += 1
            else:
                h = layer(h)
        return h.reshape(B, self.in_ch, self.img_size, self.img_size)


# ------------------------------------------------------------------- factory
def build(name: str = "tiny_unet", **kwargs) -> nn.Module:
    """Name -> denoiser, for `python denoisers.py` experiments."""
    table = {
        "convnet": ConvNetDenoiser,
        "tiny_unet": TinyUNet,
        "dit": DiTDenoiser,
        "mlp": MLPDenoiser,
    }
    return table[name](**kwargs)


if __name__ == "__main__":

    # smoke test: every denoiser must accept [B,1,32,32] + 0-based t and return
    # the same shape, in both training (grad) and sampling (no_grad) mode.
    import time

    B = 8
    x = torch.randn(B, 1, 32, 32)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)

    for name in ("convnet", "tiny_unet", "dit", "mlp"):
        model = build(name)
        eps = model(x, t)
        assert eps.shape == x.shape, f"{name}: {eps.shape} != {x.shape}"
        n_params = sum(p.numel() for p in model.parameters())

        # timing, CPU, one forward + backward
        model.zero_grad()
        start = time.perf_counter()
        model(x, t).square().mean().backward()
        ms = (time.perf_counter() - start) * 1000

        print(f"{name:>10}  params={n_params/1e6:7.2f}M  fwd+bwd={ms:7.1f}ms  out={tuple(eps.shape)}")
