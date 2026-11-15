import torch
import torch.nn as nn
import math
import numpy as np


# UNet: the noise predictor epsilon_theta(xt, t) used by DDPM.
#
# training calls it once per step, sampling calls it T times, so the same
# weights must handle every noise level. t is injected through the timestep
# embedding, which lets one network act like T different denoisers.
#
# 1. TIMESTEP EMBEDDING:
#     a. Sinusoidal embedding of t:             [B]      -> [B, ch]
#     b. 2-layer MLP:                           [B, ch]  -> [B, 4*ch]
# 2. ENCODER (down):
#     32x32, 1ch   conv1                     -> [B, ch,  32, 32]  x1
#     32x32, ch    ResNet x2                 -> [B, ch,  32, 32]  x2, x3
#     Downsample                              -> [B, ch,  16, 16]  x4
#     16x16, ch    ResNet (ch -> 2ch), Attn  -> [B, 2ch, 16, 16]  x5, x6
#     16x16, 2ch   ResNet, Attn              -> [B, 2ch, 16, 16]  x7, x8
#     Downsample                              -> [B, 2ch,  8, 8]   x9
#     8x8,  2ch    ResNet x2                 -> [B, 2ch,  8, 8]   x10, x11
#     Downsample                              -> [B, 2ch,  4, 4]   x12
#     4x4,  2ch    ResNet x2                 -> [B, 2ch,  4, 4]   x13, x14
# 3. MIDDLE:
#     4x4,  2ch    ResNet, Attn, ResNet      -> [B, 2ch,  4, 4]
# 4. DECODER (up):
#     a. Concatenate the encoder feature with the same resolution
#        (skip connection), so channels double before the ResNet.
#     b. ResNet (+ Attention at 16x16).
#     c. Upsample after every 3rd block:
#        4x4 -> 8x8 -> 16x16 -> 32x32
# 5. OUTPUT:
#     GroupNorm -> SiLU -> 3x3 conv
#     eps_pred = [B, in_ch, 32, 32]
#
# no activation after the last conv: eps is Gaussian noise, any real value.
def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    Retrieved from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#LL90C1-L109C13
    """
    # timesteps [B] -> sinusoidal embedding [B, embedding_dim]
    # same idea as Transformer positional encoding, but the "position" here
    # is the diffusion step, so the network knows how noisy its input is.
    #
    # freq_i = exp(-log(10000) * i / (half_dim - 1))
    # out    = [sin(t * freq_i), cos(t * freq_i)]
    #
    # no learned weights: pure function, so any T works without retraining.
    assert len(timesteps.shape) == 1  # one timestep per batch element: [B]

    half_dim = embedding_dim // 2  # sin and cos each get half the dims
    emb = math.log(10000) / (half_dim - 1)  # step between frequencies
    # geometric frequencies: exp(-i * log(10000) / (half_dim - 1))
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # outer product: t [B, 1] * freq [1, half_dim] -> [B, half_dim]
    emb = timesteps.type(torch.float32)[:, None] * emb[None, :]
    # concat sin and cos: [B, half_dim] + [B, half_dim] -> [B, 2 * half_dim]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))  # odd dim: one zero column

    assert emb.shape == (timesteps.shape[0], embedding_dim), f"{emb.shape}"
    return emb


class Downsample(nn.Module):

    def __init__(self, C):
        """
        :param C (int): number of input and output channels
        """
        super(Downsample, self).__init__()
        # 3x3 stride-2 conv: H -> H/2, W -> W/2 (learned, no pooling)
        self.conv = nn.Conv2d(C, C, 3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape  # [B, C, H, W]
        x = self.conv(x)  # [B, C, H, W] -> [B, C, H//2, W//2]
        assert x.shape == (B, C, H // 2, W // 2)
        return x


class Upsample(nn.Module):

    def __init__(self, C):
        """
        :param C (int): number of input and output channels
        """
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(C, C, 3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape  # [B, C, H, W]

        # nearest-neighbor 2x resize: [B, C, H, W] -> [B, C, 2H, 2W]
        # (no transposed conv; DDPM resamples first, then convolves)
        x = nn.functional.interpolate(x, size=None, scale_factor=2, mode='nearest')

        x = self.conv(x)  # 3x3 smooths the blocky nearest-neighbor result
        assert x.shape == (B, C, H * 2, W * 2)
        return x


class Nin(nn.Module):

    def __init__(self, in_dim, out_dim, scale=1e-10):
        super(Nin, self).__init__()

        # 1x1 conv written as an einsum: a per-pixel linear layer over channels
        # [B, in_dim, H, W] -> [B, out_dim, H, W]
        #
        # used twice:
        #   a. ResNetBlock: match channels before the residual add x + h
        #   b. AttentionBlock: project to Q, K, V (and to the output)
        #
        # uniform weights in [-limit, limit] give variance = scale / n
        # scale = 1e-10 -> tiny  -> branch starts near zero
        # scale = 0     -> exactly zero -> block starts as identity
        n = (in_dim + out_dim) / 2
        limit = np.sqrt(3 * scale / n)
        self.W = torch.nn.Parameter(torch.zeros((in_dim, out_dim), dtype=torch.float32
                                                ).uniform_(-limit, limit))  # [in_dim, out_dim]
        self.b = torch.nn.Parameter(torch.zeros((1, out_dim, 1, 1), dtype=torch.float32))  # broadcast over H, W

    def forward(self, x):
        # 'bchw, co->bohw': same W at every pixel, i.e. a 1x1 conv
        return torch.einsum('bchw, co->bohw', x, self.W) + self.b


class ResNetBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super(ResNetBlock, self).__init__()

        # h = SiLU(GroupNorm(x)) -> Conv -> + timestep embedding
        #     -> SiLU(GroupNorm(h)) -> Dropout -> Conv
        # out = x + h  (1x1 conv on x if channels differ)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.dense = nn.Linear(512, out_ch)  # temb [B, 4*ch] -> [B, out_ch]; 512 hardcoded = 4 * 128
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)

        if not (in_ch == out_ch):
            self.nin = Nin(in_ch, out_ch)  # 1x1 conv so x + h has matching channels

        self.dropout_rate = dropout_rate
        self.nonlinearity = torch.nn.SiLU()  # swish, the activation used throughout DDPM

    def forward(self, x, temb):
        """
        :param x: (B, C, H, W)
        :param temb: (B, dim)
        """
        # GroupNorm(32 groups) -> SiLU -> 3x3 conv
        h = self.nonlinearity(nn.functional.group_norm(x, num_groups=32))
        h = self.conv1(h)

        # add in timestep embedding
        # temb [B, 4*ch] -> dense -> [B, out_ch] -> broadcast over H, W
        h += self.dense(self.nonlinearity(temb))[:, :, None, None]

        h = self.nonlinearity(nn.functional.group_norm(h, num_groups=32))
        h = nn.functional.dropout(h, p=self.dropout_rate)
        h = self.conv2(h)

        if not (x.shape[1] == h.shape[1]):
            x = self.nin(x)  # match channels before the residual add

        assert x.shape == h.shape
        return x + h  # residual: the block only has to learn the change


class AttentionBlock(nn.Module):

    def __init__(self, ch):
        super(AttentionBlock, self).__init__()

        # global self-attention over spatial positions: every pixel attends to
        # every other pixel at the same resolution, cost O((H*W)^2).
        # that is why attention sits at 16x16 and 4x4, never at 32x32.
        self.Q = Nin(ch, ch)  # query: 1x1 conv
        self.K = Nin(ch, ch)  # key:   1x1 conv
        self.V = Nin(ch, ch)  # value: 1x1 conv

        self.ch = ch

        self.nin = Nin(ch, ch, scale=0.)  # output projection: zero init -> block starts as identity

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.ch

        # normalize, then project to Q, K, V
        h = nn.functional.group_norm(x, num_groups=32)
        q = self.Q(h)
        k = self.K(h)
        v = self.V(h)

        # attention logits: q . k over channels, for every pair of pixels
        # [B, C, H, W] x [B, C, H, W] -> [B, H, W, H, W]
        # / sqrt(C) keeps the logits from growing with the channel count
        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * (int(C) ** (-0.5))  # [B, H, W, H, W]
        w = torch.reshape(w, [B, H, W, H * W])  # flatten the keys for softmax
        w = torch.nn.functional.softmax(w, dim=-1)  # each pixel over all H*W pixels
        w = torch.reshape(w, [B, H, W, H, W])

        h = torch.einsum('bhwHW,bcHW->bchw', w, v)  # weighted sum of values: [B, C, H, W]
        h = self.nin(h)  # 1x1 conv back to ch channels

        assert h.shape == x.shape
        return x + h  # residual (zero-init nin: starts as identity)


class UNet(nn.Module):

    def __init__(self, ch=128, in_ch=1):
        super(UNet, self).__init__()

        # ch = base width; the bottleneck doubles it:
        #   32x32 -> ch, 16x16 -> 2ch, 8x8 -> 2ch, 4x4 -> 2ch
        self.ch = ch
        # timestep MLP: ch -> 4*ch -> 4*ch (no activation after linear2)
        self.linear1 = nn.Linear(ch, 4 * ch)
        self.linear2 = nn.Linear(4 * ch, 4 * ch)

        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride=1, padding=1)  # 1 -> ch, keeps 32x32

        self.down = nn.ModuleList([ResNetBlock(ch, 1 * ch),       # 32x32
                                   ResNetBlock(1 * ch, 1 * ch),   # 32x32
                                   Downsample(1 * ch),            # 32x32 -> 16x16
                                   ResNetBlock(1 * ch, 2 * ch),   # 16x16, ch -> 2ch
                                   AttentionBlock(2 * ch),        # 16x16
                                   ResNetBlock(2 * ch, 2 * ch),   # 16x16
                                   AttentionBlock(2 * ch),        # 16x16
                                   Downsample(2 * ch),            # 16x16 -> 8x8
                                   ResNetBlock(2 * ch, 2 * ch),   # 8x8
                                   ResNetBlock(2 * ch, 2 * ch),   # 8x8
                                   Downsample(2 * ch),            # 8x8 -> 4x4
                                   ResNetBlock(2 * ch, 2 * ch),   # 4x4
                                   ResNetBlock(2 * ch, 2 * ch)])  # 4x4

        self.middle = nn.ModuleList([ResNetBlock(2 * ch, 2 * ch),    # 4x4
                                     AttentionBlock(2 * ch),         # 4x4
                                     ResNetBlock(2 * ch, 2 * ch)])   # 4x4

        # input is 4*ch because the encoder skip (2ch) is concatenated
        self.up = nn.ModuleList([ResNetBlock(4 * ch, 2 * ch),    # 4x4,   2ch + 2ch
                                 ResNetBlock(4 * ch, 2 * ch),    # 4x4
                                 ResNetBlock(4 * ch, 2 * ch),    # 4x4
                                 Upsample(2 * ch),               # 4x4 -> 8x8
                                 ResNetBlock(4 * ch, 2 * ch),    # 8x8
                                 ResNetBlock(4 * ch, 2 * ch),    # 8x8
                                 ResNetBlock(4 * ch, 2 * ch),    # 8x8
                                 Upsample(2 * ch),               # 8x8 -> 16x16
                                 ResNetBlock(4 * ch, 2 * ch),    # 16x16
                                 AttentionBlock(2 * ch),         # 16x16
                                 ResNetBlock(4 * ch, 2 * ch),    # 16x16
                                 AttentionBlock(2 * ch),         # 16x16
                                 ResNetBlock(3 * ch, 2 * ch),    # 16x16, 2ch + 1ch
                                 AttentionBlock(2 * ch),         # 16x16
                                 Upsample(2 * ch),               # 16x16 -> 32x32
                                 ResNetBlock(3 * ch, ch),        # 32x32, 2ch + 1ch
                                 ResNetBlock(2 * ch, ch),        # 32x32, ch + 1ch
                                 ResNetBlock(2 * ch, ch)])       # 32x32, ch + ch

        self.final_conv = nn.Conv2d(ch, in_ch, 3, stride=1, padding=1)  # ch -> in_ch

    def forward(self, x, t):
        """
        :param x: (torch.Tensor) batch of images [B, C, H, W]
        :param t: (torch.Tensor) tensor of time steps (torch.long) [B]
        """
        # t arrives 0-based (diffusion_models.py passes t - 1 everywhere)

        temb = get_timestep_embedding(t, self.ch)  # [B] -> [B, ch]
        temb = torch.nn.functional.silu(self.linear1(temb))  # [B, ch] -> [B, 4*ch]
        temb = self.linear2(temb)  # [B, 4*ch] -> [B, 4*ch]
        assert temb.shape == (t.shape[0], self.ch * 4)

        x1 = self.conv1(x)  # [B, in_ch, 32, 32] -> [B, ch, 32, 32]

        # Down: 32x32 -> 16x16 -> 8x8 -> 4x4
        x2 = self.down[0](x1, temb)  # 32x32
        x3 = self.down[1](x2, temb)  # 32x32
        x4 = self.down[2](x3)        # 32x32 -> 16x16
        x5 = self.down[3](x4, temb)  # 16x16, ch -> 2ch
        x6 = self.down[4](x5)  # Attention
        x7 = self.down[5](x6, temb)  # 16x16
        x8 = self.down[6](x7)  # Attention
        x9 = self.down[7](x8)        # 16x16 -> 8x8
        x10 = self.down[8](x9, temb)  # 8x8
        x11 = self.down[9](x10, temb)  # 8x8
        x12 = self.down[10](x11)      # 8x8 -> 4x4
        x13 = self.down[11](x12, temb)  # 4x4
        x14 = self.down[12](x13, temb)  # 4x4

        # Middle: 4x4, ResNet + Attention + ResNet
        x = self.middle[0](x14, temb)  # 4x4
        x = self.middle[1](x)  # Attention
        x = self.middle[2](x, temb)  # 4x4

        # Up: 4x4 -> 8x8 -> 16x16 -> 32x32
        # each cat doubles the channels, so up blocks take 4*ch (or 3*ch / 2*ch)
        x = self.up[0](torch.cat((x, x14), dim=1), temb)  # 4x4, 2ch + 2ch
        x = self.up[1](torch.cat((x, x13), dim=1), temb)  # 4x4, 2ch + 2ch
        x = self.up[2](torch.cat((x, x12), dim=1), temb)  # 4x4, 2ch + 2ch
        x = self.up[3](x)                                 # 4x4 -> 8x8
        x = self.up[4](torch.cat((x, x11), dim=1), temb)  # 8x8, 2ch + 2ch
        x = self.up[5](torch.cat((x, x10), dim=1), temb)  # 8x8, 2ch + 2ch
        x = self.up[6](torch.cat((x, x9), dim=1), temb)   # 8x8, 2ch + 2ch
        x = self.up[7](x)                                 # 8x8 -> 16x16
        x = self.up[8](torch.cat((x, x8), dim=1), temb)   # 16x16, 2ch + 2ch
        x = self.up[9](x)  # Attention
        x = self.up[10](torch.cat((x, x6), dim=1), temb)  # 16x16, 2ch + 2ch
        x = self.up[11](x)  # Attention
        x = self.up[12](torch.cat((x, x4), dim=1), temb)  # 16x16, 2ch + 1ch = 3ch
        x = self.up[13](x)  # Attention
        x = self.up[14](x)                                # 16x16 -> 32x32
        x = self.up[15](torch.cat((x, x3), dim=1), temb)  # 32x32, 2ch + 1ch = 3ch
        x = self.up[16](torch.cat((x, x2), dim=1), temb)  # 32x32, ch + 1ch
        x = self.up[17](torch.cat((x, x1), dim=1), temb)  # 32x32, ch + ch

        x = nn.functional.silu(nn.functional.group_norm(x, num_groups=32))  # final norm + activation
        x = self.final_conv(x)  # [B, ch, 32, 32] -> [B, in_ch, 32, 32]

        return x  # predicted noise eps
