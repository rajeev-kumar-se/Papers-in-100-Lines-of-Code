import torch
import torch.nn as nn

# reuse the sinusoidal embedding from unet.py rather than rewriting it
from unet import get_timestep_embedding

# SimpleUNet: the bare-minimum U-Net noise predictor.
#
# this is the "does the fancy stuff actually matter?" baseline. unet.py is 35.7M
# parameters with attention at 16x16 and 4x4, residual blocks, FiLM-style
# timestep injection and zero-initialised output projections. this file is
# ~1.2M parameters: one DoubleConv per level, plain MaxPool down, plain
# Upsample up, BatchNorm, nothing else.
#
# it is still a valid denoiser epsilon_theta(xt, t), so it drops straight into
# diffusion_models.py:
#
#     from unet_simple import SimpleUNet
#     denoiser_model = SimpleUNet()
#
# 1. TIMESTEP EMBEDDING:
#     a. Sinusoidal embedding of t:         [B]        -> [B, t_dim]
#     b. 2-layer MLP:                       [B, t_dim] -> [B, 4*t_dim]  temb
# 2. ENCODER (down):
#     32x32, 1ch   DoubleConv                -> [B, 32,  32, 32]  x1  (skip)
#     MaxPool 2x2                            -> [B, 32,  16, 16]
#     16x16, 32ch  DoubleConv                -> [B, 64,  16, 16]  x2  (skip)
#     MaxPool 2x2                            -> [B, 64,   8, 8]
#     8x8,   64ch  DoubleConv                -> [B, 128,  8, 8]   x3  (skip)
#     MaxPool 2x2                            -> [B, 128,  4, 4]
# 3. BOTTLENECK:
#     4x4,  128ch  DoubleConv                -> [B, 256,  4, 4]
# 4. DECODER (up):
#     a. Upsample nearest 2x, concatenate the skip, then one DoubleConv.
#     4x4   -> 8x8:   cat(256, 128)          -> [B, 128,  8, 8]
#     8x8   -> 16x16: cat(128, 64)           -> [B, 64,  16, 16]
#     16x16 -> 32x32: cat(64, 32)            -> [B, 32,  32, 32]
# 5. OUTPUT:
#     1x1 conv                               -> [B, 1,   32, 32]
#
# how the timestep gets in: every DoubleConv ends with + Linear(temb), broadcast
# over H and W. no per-block MLPs, no learned per-channel scale, no gates.
#
# what is deliberately missing, and why it matters for the FID you will measure:
#   - no attention. attention is what lets the UNet see the whole 32x32 image at
#     once at 16x16 and 4x4. without it the receptive field only grows through
#     the bottleneck, so global shape (is this a 3 or an 8?) is harder.
#   - no residual blocks. each block has to relearn the identity, so deeper is
#     not automatically better and training is slower per unit of quality.
#   - no zero-init on the output. at step 0 the net predicts loud noise instead
#     of zero noise, so the first few hundred steps are spent undoing that.
#   - MaxPool, not a strided 3x3 conv. pooling is not learned and throws away
#     information; the skip connection is the only thing putting it back.
#
# one practical trap that unet.py does not have:
#   BatchNorm keeps running statistics, so the model MUST be in eval() when you
#   sample. in train() mode a batch of 81 samples gets normalized by that batch's
#   own statistics, and the images come out different from what a batch of 1
#   produces. the runner does model.eval() before sampling for this reason.


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, t_dim):
        """
        :param in_ch (int): input channels
        :param out_ch (int): output channels
        :param t_dim (int): width of the timestep embedding (4 * t_dim from the MLP)
        """
        super(DoubleConv, self).__init__()

        # (conv 3x3 -> BN -> ReLU) x 2: the whole block, no residual path
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),  # H, W unchanged
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # temb [B, t_dim] -> [B, out_ch], broadcast over the feature map
        self.temb = nn.Linear(t_dim, out_ch)

    def forward(self, x, temb):
        """
        :param x: (B, in_ch, H, W)
        :param temb: (B, t_dim) timestep embedding
        """
        h = self.net(x)  # [B, in_ch, H, W] -> [B, out_ch, H, W]

        # added after the last ReLU, so it is not squashed by the activation;
        # the next conv (or the 1x1 output conv) consumes it directly
        return h + self.temb(temb)[:, :, None, None]


class SimpleUNet(nn.Module):

    def __init__(self, in_ch=1, base=32, t_dim=64):
        """
        :param in_ch (int): image channels (1 for MNIST)
        :param base (int): width of the first level; doubles at every level
        :param t_dim (int): sinusoidal embedding width
        """
        super(SimpleUNet, self).__init__()

        self.in_ch = in_ch
        te = 4 * t_dim  # width of temb, consumed by every DoubleConv

        # t [B, t_dim] -> [B, 4*t_dim]; no activation after linear2, same as unet.py
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, te),
            nn.SiLU(),
            nn.Linear(te, te),
        )

        # channels per level: 32 -> 64 -> 128 -> 256
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        # encoder: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.enc1 = DoubleConv(in_ch, c1, te)  # 32x32
        self.enc2 = DoubleConv(c1, c2, te)     # 16x16
        self.enc3 = DoubleConv(c2, c3, te)     # 8x8
        self.pool = nn.MaxPool2d(2)            # H, W -> H/2, W/2

        # bottleneck: the only place where a pixel can influence a far pixel
        self.bottom = DoubleConv(c3, c4, te)   # 4x4

        # decoder: nearest-neighbour 2x, then concat the skip, then DoubleConv
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # the cat doubles the channels, so each decoder block takes c(i+1) + ci
        self.dec3 = DoubleConv(c4 + c3, c3, te)  # 4x4  -> 8x8
        self.dec2 = DoubleConv(c3 + c2, c2, te)  # 8x8  -> 16x16
        self.dec1 = DoubleConv(c2 + c1, c1, te)  # 16x16 -> 32x32

        # 1x1 conv back to image channels; no activation, eps is real-valued
        self.out = nn.Conv2d(c1, in_ch, 1)

    def forward(self, x, t):
        """
        :param x: (torch.Tensor) batch of noisy images [B, in_ch, 32, 32]
        :param t: (torch.Tensor) 0-based timesteps (torch.long) [B]
        :return: (torch.Tensor) predicted noise eps [B, in_ch, 32, 32]
        """
        temb = self.t_mlp(get_timestep_embedding(t, self.t_mlp[0].in_features))  # [B, 4*t_dim]

        # encoder, keeping every level for the skip concatenation
        x1 = self.enc1(x, temb)             # [B, 32,  32, 32]
        x2 = self.enc2(self.pool(x1), temb)  # [B, 64,  16, 16]
        x3 = self.enc3(self.pool(x2), temb)  # [B, 128,  8, 8]
        h = self.bottom(self.pool(x3), temb)  # [B, 256,  4, 4]

        # decoder: upsample, concat the skip from the same resolution, convolve.
        # the shape comment on the right is what the cat has to produce, and it
        # is the only reason the encoder kept x1/x2/x3 around.
        h = self.dec3(torch.cat((self.up(h), x3), dim=1), temb)   # [B, 128,  8, 8]
        h = self.dec2(torch.cat((self.up(h), x2), dim=1), temb)   # [B, 64,  16, 16]
        h = self.dec1(torch.cat((self.up(h), x1), dim=1), temb)   # [B, 32,  32, 32]

        return self.out(h)  # [B, 32, 32, 32] -> [B, 1, 32, 32]


if __name__ == "__main__":

    # shape check only: [B, 1, 32, 32] + 0-based t in, same shape out
    B = 4
    x = torch.randn(B, 1, 32, 32)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)

    model = SimpleUNet()
    eps = model(x, t)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleUNet  params={n_params/1e6:.2f}M  in={tuple(x.shape)} out={tuple(eps.shape)}")
    assert eps.shape == x.shape
