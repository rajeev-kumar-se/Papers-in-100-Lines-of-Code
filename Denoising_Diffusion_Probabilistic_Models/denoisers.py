import math

import torch
import torch.nn as nn

# reuse the sinusoidal embedding from the UNet file instead of rewriting it
from unet import get_timestep_embedding

# Drop-in replacements for the UNet in unet.py.
#
# Every class here is the same thing: the noise predictor epsilon_theta(xt, t)
# that diffusion_models.py calls. The ONLY contract you have to satisfy is
#
#     eps_pred = model(x, t)
#         x        : [B, in_ch, 32, 32]  noisy image, float32
#         t        : [B]                 timestep, torch.long, **0-based**
#                                        (diffusion_models.py passes t - 1)
#         eps_pred : [B, in_ch, 32, 32]  predicted noise, same shape as x
#
# nothing else is used: no .sample(), no .ch, no attention maps, no loss. So
# swapping architectures is one line in diffusion_models.py:
#
#     from denoisers import build
#     denoiser_model = build("dit")        # instead of UNet()
#
# THE FOUR ARCHITECTURES
#
#     ConvNetDenoiser   0.95M   full-resolution convs, no downsampling, FiLM
#     TinyUNet          0.48M   the unet.py skeleton with the fat trimmed
#     DiTDenoiser       4.20M   patch tokens + self-attention (no convs)
#     MLPDenoiser       9.58M   flatten + MLP, no spatial structure at all
#
# THE THREE THINGS THAT MATTER IN ALL OF THEM
#
#     1. TIMESTEP REACHES EVERY LAYER.
#        training calls the net once per step, sampling calls it T times, so a
#        single set of weights must act like T different denoisers. That only
#        works if t is injected deep, not just at the input.
#           unet.py      : + temb inside every ResNetBlock
#           ConvNet      : FiLM, h = h * (1 + scale) + shift
#           TinyUNet     : + temb inside every Block (same as unet.py)
#           DiT          : adaLN-Zero, scale/shift/gate from c
#           MLP          : + A_i c at every hidden layer
#
#     2. THE OUTPUT IS UNCONSTRAINED.
#        eps ~ N(0, I) takes any real value, so the last layer has no
#        activation and often zero-initialised weights. Every head below ends
#        in a plain Linear/Conv and returns raw numbers.
#
#     3. THE RECEPTIVE FIELD COVERS THE IMAGE STRUCTURE.
#        an MNIST digit is ~20 pixels tall, so a 3x3 receptive field cannot
#        see a whole stroke. Each architecture reaches the needed range
#        differently: stacking convs (ConvNet), downsampling (UNet), or global
#        attention from layer 1 (DiT).


def _groups(ch):
    """
    GroupNorm with the largest group count that divides ch.

    unet.py hardcodes num_groups=32 everywhere, which is correct for ch=128 but
    crashes as soon as the width drops below 32 (16 channels cannot be split
    into 32 groups). Every block here uses this helper instead, so TinyUNet and
    ConvNet can be shrunk to 16 channels without touching the code.
    """
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


# ConvNetDenoiser: the "no downsampling" control. The receptive field only grows
# by stacking 3x3 convs, nothing is ever pooled, so the same weights work at
# 32x32 or 64x64 without touching the code -- the opposite trade-off from the
# UNet, which spends its depth on resolution changes.
#
# the timestep is injected by FiLM: instead of ADDING the embedding (UNet's
# choice) it produces a per-channel (scale, shift) that multiplies and offsets
# the normalized features. That gives the timestep a way to turn individual
# channels up or down, not just offset them.
#
# 1. TIMESTEP EMBEDDING:
#     a. Sinusoidal embedding of t:            [B]        -> [B, t_dim]
#     b. 2-layer MLP:                          [B, t_dim] -> [B, 4*t_dim]
# 2. STEM:
#     32x32, in_ch  conv1 3x3                  -> [B, ch, 32, 32]  h
# 3. BODY (n blocks, resolution never changes), per block:
#     a. temb -> Linear -> chunk(2)            [B, 4*t_dim] -> [B, ch] scale,
#                                                             [B, ch] shift
#     b. GroupNorm(h)                          -> [B, ch, 32, 32]
#     c. h = h * (1 + scale) + shift           -> [B, ch, 32, 32]
#     d. SiLU -> 3x3 conv                      -> [B, ch, 32, 32]
# 4. OUTPUT:
#     GroupNorm -> SiLU -> 3x3 conv            -> [B, in_ch, 32, 32]
#
# receptive field after n blocks = 1 + 2*n pixels, so the default n=6 gives
# 13x13: enough for a digit stroke, nowhere near the UNet's full image view.
# no activation on the last conv: eps is Gaussian noise, any real value.
# (1 + scale) rather than scale so a zero-initialised projection means
# "pass the features through unchanged", the same reason unet.py's Nin uses
# scale=1e-10 for residual branches.
class ConvNetDenoiser(nn.Module):

    def __init__(self, ch=64, in_ch=1, depth=6, t_dim=128):
        super(ConvNetDenoiser, self).__init__()

        self.in_ch = in_ch
        self.t_dim = t_dim

        # t [B, t_dim] -> [B, 4*t_dim]; width kept equal to the UNet's so the
        # embedding costs the same no matter which architecture uses it
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim * 4),
        )

        # [B, in_ch, 32, 32] -> [B, ch, 32, 32]
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)

        # one (scale, shift) projection per block: [B, 4*t_dim] -> [B, 2*ch]
        self.blocks = nn.ModuleList(
            [nn.Linear(t_dim * 4, ch * 2) for _ in range(depth)]
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(ch, ch, 3, padding=1) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(_groups(ch), ch) for _ in range(depth)])

        # GroupNorm -> SiLU -> conv, same tail as unet.py's final_conv
        self.norm_out = nn.GroupNorm(_groups(ch), ch)
        self.conv_out = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        """
        :param x: (torch.Tensor) batch of noisy images [B, in_ch, 32, 32]
        :param t: (torch.Tensor) 0-based timesteps (torch.long) [B]
        :return: (torch.Tensor) predicted noise eps [B, in_ch, 32, 32]
        """
        temb = self.t_mlp(get_timestep_embedding(t, self.t_dim))  # [B, 4*t_dim]

        h = self.conv_in(x)  # [B, in_ch, 32, 32] -> [B, ch, 32, 32]

        # body: FiLM-conditioned conv, resolution stays 32x32 the whole way
        for proj, conv, norm in zip(self.blocks, self.convs, self.norms):
            # split the per-block projection into the affine pair
            scale, shift = proj(temb).chunk(2, dim=-1)  # [B, ch], [B, ch]

            # normalize, then modulate per channel; [:, :, None, None] broadcasts
            # [B, ch] over the 32x32 grid
            h = norm(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = torch.nn.functional.silu(h)  # swish, the activation used throughout DDPM
            h = conv(h)  # [B, ch, 32, 32] -> [B, ch, 32, 32]

        h = torch.nn.functional.silu(self.norm_out(h))  # final norm + activation
        return self.conv_out(h)  # [B, ch, 32, 32] -> [B, in_ch, 32, 32]


# TinyUNet: the reference UNet with the fat trimmed.
#
# same skeleton as unet.py -- encode, bottleneck, decode with skip connections
# -- but 2 downsamplings instead of 3, one ResNet block per level instead of
# 2-3, no attention blocks, and ch=32 instead of ch=128. 0.48M parameters
# instead of 35.7M, so it fits and trains on hardware where the reference does
# not, and it keeps the thing that makes a UNet beat a plain encoder-decoder:
# the skip connections.
#
# 1. TIMESTEP EMBEDDING:
#     a. Sinusoidal embedding of t:            [B]        -> [B, t_dim]
#     b. 2-layer MLP:                          [B, t_dim] -> [B, 4*t_dim]
# 2. ENCODER (down):
#     32x32, in_ch  conv1                     -> [B, ch,  32, 32]  h0
#     32x32, ch     Block(ch -> ch)           -> [B, ch,  32, 32]  h1  (skip)
#     Downsample                              -> [B, ch,  16, 16]
#     16x16, ch     Block(ch -> 2ch)          -> [B, 2ch, 16, 16]  h2  (skip)
#     Downsample                              -> [B, 2ch,  8, 8]
# 3. MIDDLE (bottleneck):
#     8x8,  2ch     Block(2ch -> 2ch)         -> [B, 2ch,  8, 8]
# 4. DECODER (up):
#     a. Concatenate the encoder feature with the same resolution
#        (skip connection), so channels grow before the Block.
#     b. Nearest-neighbour 2x resize, then a 3x3 conv (no transposed conv:
#        DDPM resamples first, then convolves).
#     8x8 -> 16x16:  cat(h, h2) = 2ch + 2ch   -> Block(4ch -> 2ch) -> [B, 2ch, 16, 16]
#     16x16 -> 32x32: cat(h, h1) = 2ch + ch   -> Block(3ch -> ch)  -> [B, ch,  32, 32]
# 5. OUTPUT:
#     GroupNorm -> SiLU -> 3x3 conv           -> [B, in_ch, 32, 32]
#
# the Block recipe is unet.py's ResNetBlock, written inline:
#     GroupNorm -> SiLU -> Conv -> + temb -> GroupNorm -> SiLU -> Conv -> + x
# with a 1x1 conv on x when the channel count changes, so the residual add is
# legal. Two differences from unet.py's version: the GroupNorm group count comes
# from _groups(ch) instead of a hardcoded 32, and the timestep projection is
# sized from t_dim instead of a hardcoded 512 (= 4*128).
class TinyUNet(nn.Module):

    def __init__(self, ch=32, in_ch=1, t_dim=None):
        super(TinyUNet, self).__init__()

        # t_dim defaults to ch; a wider t_dim costs almost nothing because the
        # embedding only touches [B] tensors, never the [B, C, H, W] feature maps
        t_dim = t_dim or ch
        self.t_dim = t_dim
        self.in_ch = in_ch

        # ch -> 4*ch -> 4*ch, no activation after linear2 (same as unet.py)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, 4 * t_dim),
            nn.SiLU(),
            nn.Linear(4 * t_dim, 4 * t_dim),
        )

        class Block(nn.Module):
            """GroupNorm -> SiLU -> Conv -> + temb -> GroupNorm -> SiLU -> Conv -> + x"""

            def __init__(self, cin, cout, tdim):
                super(Block, self).__init__()

                self.norm1 = nn.GroupNorm(_groups(cin), cin)
                self.conv1 = nn.Conv2d(cin, cout, 3, stride=1, padding=1)

                # temb [B, 4*t_dim] -> [B, cout], broadcast over H, W
                self.emb = nn.Linear(tdim, cout)

                self.norm2 = nn.GroupNorm(_groups(cout), cout)
                self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1)

                # 1x1 conv so x + h has matching channels (unet.py's Nin)
                self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

            def forward(self, x, temb):
                """
                :param x: (B, cin, H, W)
                :param temb: (B, 4*t_dim)
                """
                h = torch.nn.functional.silu(self.norm1(x))  # [B, cin, H, W]
                h = self.conv1(h)  # [B, cin, H, W] -> [B, cout, H, W]

                # add in the timestep: every block sees t, not just the input
                h = h + self.emb(torch.nn.functional.silu(temb))[:, :, None, None]

                h = torch.nn.functional.silu(self.norm2(h))
                h = self.conv2(h)  # [B, cout, H, W] -> [B, cout, H, W]

                return self.skip(x) + h  # residual: the block only learns the change

        def down(c):
            # 3x3 stride-2 conv: H -> H/2, W -> W/2 (learned, no pooling)
            return nn.Conv2d(c, c, 3, stride=2, padding=1)

        def up(c):
            # resize first, convolve second: same order as unet.py's Upsample
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c, c, 3, padding=1),
            )

        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)  # 1 -> ch, keeps 32x32

        # encoder: 32x32 -> 16x16 -> 8x8
        self.d1 = Block(ch, ch, 4 * t_dim)       # 32x32
        self.dn1 = down(ch)                      # 32x32 -> 16x16
        self.d2 = Block(ch, 2 * ch, 4 * t_dim)   # 16x16, ch -> 2ch
        self.dn2 = down(2 * ch)                  # 16x16 -> 8x8

        # bottleneck
        self.mid = Block(2 * ch, 2 * ch, 4 * t_dim)  # 8x8

        # decoder: 8x8 -> 16x16 -> 32x32, each level eats its skip (+2ch / +ch)
        self.up2 = up(2 * ch)                          # 8x8 -> 16x16
        self.u2 = Block(4 * ch, 2 * ch, 4 * t_dim)     # 16x16, 2ch + 2ch
        self.up1 = up(2 * ch)                          # 16x16 -> 32x32
        self.u1 = Block(3 * ch, ch, 4 * t_dim)         # 32x32, 2ch + ch

        # output head, identical shape to unet.py's final_conv path
        self.out = nn.Sequential(
            nn.GroupNorm(_groups(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),  # ch -> in_ch, no activation
        )

    def forward(self, x, t):
        """
        :param x: (torch.Tensor) batch of noisy images [B, in_ch, 32, 32]
        :param t: (torch.Tensor) 0-based timesteps (torch.long) [B]
        :return: (torch.Tensor) predicted noise eps [B, in_ch, 32, 32]
        """
        temb = self.t_mlp(get_timestep_embedding(t, self.t_dim))  # [B, 4*t_dim]

        h0 = self.conv_in(x)  # [B, in_ch, 32, 32] -> [B, ch, 32, 32]
        h1 = self.d1(h0, temb)  # 32x32, kept as a skip
        h2 = self.d2(self.dn1(h1), temb)  # [B, ch, 16, 16] -> [B, 2ch, 16, 16]
        h = self.mid(self.dn2(h2), temb)  # [B, 2ch, 16, 16] -> [B, 2ch, 8, 8]

        # decoder: upsample, concatenate the skip, then mix. the cat doubles the
        # channels, which is why u2 takes 4ch in and u1 takes 3ch in.
        h = self.u2(torch.cat((self.up2(h), h2), dim=1), temb)  # [B, 2ch, 16, 16]
        h = self.u1(torch.cat((self.up1(h), h1), dim=1), temb)  # [B, ch,  32, 32]

        return self.out(h)  # [B, ch, 32, 32] -> [B, in_ch, 32, 32]


# DiTDenoiser: the Diffusion Transformer (Peebles & Xie, 2023), the architecture
# behind SD3 and Sora. No convolutions and no U shape at all: the image is cut
# into patches, treated as a token sequence, and mixed by self-attention.
#
# against the UNet the trade-off flips:
#     UNet: receptive field grows one downsample at a time (local -> global)
#     DiT:  every token sees every other token in layer 1 (global immediately)
# so the cost becomes O(N^2) attention instead of O(H*W) convolutions, and the
# patch size is the knob that controls it.
#
# 1. TIMESTEP EMBEDDING:
#     a. Sinusoidal embedding of t:            [B]        -> [B, t_dim]
#     b. 2-layer MLP:                          [B, t_dim] -> [B, dim]  c
# 2. PATCHIFY:
#     a. split into 32/p x 32/p non-overlapping p x p patches
#        32x32, in_ch, p=2                   -> 256 patches
#     b. conv with kernel=stride=p, i.e. one linear layer per patch
#        [B, in_ch, 32, 32]                   -> [B, dim, 16, 16]
#     c. flatten the grid, transpose           -> [B, 256, dim]
#     d. + learned position embedding          -> [B, 256, dim]
# 3. TRANSFORMER x depth, adaLN-Zero in every block:
#     a. c -> Linear -> chunk(6)               [B, dim] -> [B, 6*dim]
#        (shift, scale, gate) for attention + (shift, scale, gate) for the MLP
#     b. x = x + gate1 * Attn(adaLN(x, scale1, shift1))
#     c. x = x + gate2 * MLP (adaLN(x, scale2, shift2))
#     both gates and both shifts are zero-initialised, so every block starts as
#     the identity and the network begins as a shallow one.
# 4. OUTPUT HEAD:
#     a. final adaLN                           -> [B, 256, dim]
#     b. Linear dim -> p*p*in_ch               -> [B, 256, p*p*in_ch]
# 5. UNPATCHIFY:
#     [B, 256, p*p*in_ch] -> [B, 16, 16, p, p, in_ch] -> [B, in_ch, 32, 32]
#
# notes:
#   - no activation on the head: eps is Gaussian noise, any real value.
#   - the zero-init trick is DDPM's (unet.py's Nin uses scale=0 for its output
#     projection); adaLN-Zero applies it to the residual gates instead.
#   - attention logits = 256^2 = 65536 per layer at p=2, which is fine. Above
#     32x32 use p=4 or windowed attention, or the O(N^2) term takes over.
class DiTDenoiser(nn.Module):

    def __init__(self, dim=192, depth=6, heads=6, patch=2, in_ch=1, t_dim=None):
        super(DiTDenoiser, self).__init__()

        t_dim = t_dim or dim
        self.patch = patch
        self.in_ch = in_ch
        self.n = 32 // patch  # 32/2 = 16 -> 16*16 = 256 tokens

        # patch embedding: kernel = stride = patch means every p x p block of
        # pixels is flattened and multiplied by one weight matrix. that is a
        # "linear patchify" with the pixels kept in a conv instead of a reshape.
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, stride=patch)

        # position embedding: attention is permutation-invariant, so without
        # this the model cannot tell pixel (0,0) from pixel (15,15)
        self.pos = nn.Parameter(torch.zeros(1, self.n * self.n, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # t [B, t_dim] -> [B, dim]; also fixes in_features for the caller below
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        class Block(nn.Module):
            """pre-norm transformer block, but the norm's affine params come from t."""

            def __init__(self, dim, heads):
                super(Block, self).__init__()

                # elementwise_affine=False: the scale/shift normally learned by
                # LayerNorm is produced from c instead, per timestep
                self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
                # batch_first=True so tensors stay [B, N, dim] like the rest
                self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
                self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

                def mlp():
                    # the usual transformer 4x expansion
                    return nn.Sequential(
                        nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
                    )

                self.mlp = mlp()

                # 6 pieces in one projection: shift, scale, gate for attention
                # and shift, scale, gate for the MLP
                self.mod = nn.Linear(dim, 6 * dim)
                nn.init.zeros_(self.mod.weight)  # <- Zero in adaLN-Zero
                nn.init.zeros_(self.mod.bias)

            def forward(self, x, c):
                """
                :param x: (B, N, dim) token sequence
                :param c: (B, dim)   timestep conditioning vector
                """
                # [B, dim] -> six [B, dim] pieces, [:, None] broadcasts over tokens
                s1, sc1, g1, s2, sc2, g2 = self.mod(c).chunk(6, dim=-1)

                # attention branch: modulate the normalized tokens, not the input
                h = self.norm1(x) * (1 + sc1[:, None]) + s1[:, None]
                h, _ = self.attn(h, h, h, need_weights=False)  # self-attention
                x = x + g1[:, None] * h  # gated residual

                # MLP branch, same pattern
                h = self.norm2(x) * (1 + sc2[:, None]) + s2[:, None]
                x = x + g2[:, None] * self.mlp(h)

                return x

        # depth blocks, attention only: no downsampling anywhere, so every
        # block runs at the same 256-token length
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])

        # final adaLN + head, both zero-initialised (see note above)
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod_out = nn.Linear(dim, 2 * dim)
        self.head = nn.Linear(dim, patch * patch * in_ch)

        for lin in (self.mod_out, self.head):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, t):
        """
        :param x: (torch.Tensor) batch of noisy images [B, in_ch, 32, 32]
        :param t: (torch.Tensor) 0-based timesteps (torch.long) [B]
        :return: (torch.Tensor) predicted noise eps [B, in_ch, 32, 32]
        """
        B, C, H, W = x.shape  # [B, in_ch, 32, 32]

        # patchify: conv does the "cut into patches + linear" in one step
        h = self.patch_embed(x)  # [B, in_ch, 32, 32] -> [B, dim, 16, 16]
        h = h.flatten(2).transpose(1, 2)  # [B, dim, 256] -> [B, 256, dim]
        h = h + self.pos  # tell the tokens where they are

        # timestep conditioning vector, reused by every block
        c = self.t_mlp(get_timestep_embedding(t, self.t_mlp[0].in_features))  # [B, dim]

        for blk in self.blocks:
            h = blk(h, c)  # [B, 256, dim] -> [B, 256, dim]

        # final modulation, then project each token back to its patch's pixels
        shift, scale = self.mod_out(torch.nn.functional.silu(c)).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        eps = self.head(h)  # [B, 256, dim] -> [B, 256, p*p*in_ch]

        # unpatchify: [B, n*n, p*p*C] -> [B, C, n*p, n*p]
        # the permute interleaves the p x p pixels back into the right place;
        # getting this order wrong gives a correctly shaped but scrambled image,
        # which is why the assert below is not enough on its own -- look at it.
        n, p = self.n, self.patch
        eps = eps.reshape(B, n, n, p, p, C)  # patch grid -> pixel block
        eps = eps.permute(0, 5, 1, 3, 2, 4).reshape(B, C, n * p, n * p)
        assert eps.shape == (B, C, H, W)
        return eps


# MLPDenoiser: no inductive bias at all. Flatten the image, run an MLP, reshape.
# This is roughly the score network of the 2015 score matching work, and it is
# kept as the baseline: a conv or attention net that cannot beat this on MNIST
# is not actually learning anything about images.
#
# what dropping the spatial structure costs:
#     a 3x3 conv reuses 9 weights at every position; an MLP needs a separate
#     weight for every (pixel, neighbour) pair, so the first layer alone is
#     1024 x 1024 = 1M parameters just to mix pixels. That is why 9.6M
#     parameters here still underperform 0.5M in TinyUNet.
#
# 1. TIMESTEP EMBEDDING:
#     a. Sinusoidal embedding of t:            [B]        -> [B, t_dim]
#     b. 2-layer MLP:                          [B, t_dim] -> [B, hidden]  c
# 2. FLATTEN:
#     [B, in_ch, 32, 32]                       -> [B, in_ch*32*32]
# 3. BODY (depth Linear layers, SiLU between):
#     every hidden layer gets the timestep added in:
#         h = W_i h + b_i + A_i c              -> [B, hidden]
#     A_i c is what makes the net timestep-dependent; without it there is only
#     one denoiser, not T.
# 4. OUTPUT:
#     Linear hidden -> in_ch*32*32             -> [B, in_ch*32*32]
#     reshape                                  -> [B, in_ch, 32, 32]
#
# the whole forward pass is a matmul chain, which is why it has the lowest CPU
# time in the smoke test despite having the most parameters: no convolutions,
# and every "pixel" is handled independently by the same weight matrix.
class MLPDenoiser(nn.Module):

    def __init__(self, in_ch=1, img_size=32, hidden=1024, depth=4, t_dim=128):
        super(MLPDenoiser, self).__init__()

        self.in_ch = in_ch
        self.img_size = img_size
        self.t_dim = t_dim
        flat = in_ch * img_size * img_size  # 1 * 32 * 32 = 1024 for MNIST

        # t [B, t_dim] -> [B, hidden]: one vector per sample, same width as the
        # hidden layers so it can simply be added
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        # `depth` Linear layers total: 1 in, depth-2 hidden, 1 out. the MLP in
        # a transformer block would have the same shape.
        layers = [nn.Linear(flat, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, flat)]  # no activation: eps is real-valued
        self.net = nn.Sequential(*layers)

        # per-layer projection of the timestep, added after each Linear.
        # one vector of scale factors per layer would do too, but a Linear lets
        # the timestep choose which hidden units to favour.
        self.emb = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])

    def forward(self, x, t):
        """
        :param x: (torch.Tensor) batch of noisy images [B, in_ch, 32, 32]
        :param t: (torch.Tensor) 0-based timesteps (torch.long) [B]
        :return: (torch.Tensor) predicted noise eps [B, in_ch, 32, 32]
        """
        B = x.shape[0]
        h = x.reshape(B, -1)  # [B, in_ch, 32, 32] -> [B, 1024]

        temb = self.t_mlp(get_timestep_embedding(t, self.t_dim))  # [B, hidden]

        # hand-walked instead of nn.Sequential so the timestep can be added
        # between layers; i indexes which emb projection belongs to this Linear
        i = 0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                h = layer(h) + self.emb[i](temb)  # [B, hidden] + [B, hidden]
                i += 1
            else:
                h = layer(h)  # SiLU

        return h.reshape(B, self.in_ch, self.img_size, self.img_size)


# build: name -> denoiser, so diffusion_models.py can pick one by string.
#
#     denoiser_model = build("tiny_unet")
#
# kwargs pass straight through, e.g. build("dit", depth=12, dim=384) for a
# bigger transformer or build("tiny_unet", ch=16) for an even smaller UNet.
def build(name: str = "tiny_unet", **kwargs) -> nn.Module:
    """Name -> denoiser class, with kwargs forwarded to its __init__."""
    table = {
        "convnet": ConvNetDenoiser,
        "tiny_unet": TinyUNet,
        "dit": DiTDenoiser,
        "mlp": MLPDenoiser,
    }
    return table[name](**kwargs)


if __name__ == "__main__":

    # smoke test: every denoiser must accept [B, 1, 32, 32] plus 0-based t and
    # return the same shape, and must be differentiable.
    #
    # why B=8 and not 64: this is a shape check, not a training run. Nothing
    # here proves the samples look like digits -- for that, train for a while.
    import time

    B = 8
    x = torch.randn(B, 1, 32, 32)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)

    for name in ("convnet", "tiny_unet", "dit", "mlp"):
        model = build(name)

        # forward pass: the shape contract from the top of this file
        eps = model(x, t)
        assert eps.shape == x.shape, f"{name}: {eps.shape} != {x.shape}"

        # parameter count: the number that decides whether this fits on your GPU
        n_params = sum(p.numel() for p in model.parameters())

        # one forward + backward, timed on CPU. do not read this as a training
        # speed: the sampling loop calls the model T=1000 times per image, so a
        # 3x slower forward costs 3x the generation time.
        model.zero_grad()
        start = time.perf_counter()
        model(x, t).square().mean().backward()
        ms = (time.perf_counter() - start) * 1000

        print(f"{name:>10}  params={n_params/1e6:7.2f}M  fwd+bwd={ms:7.1f}ms  out={tuple(eps.shape)}")
