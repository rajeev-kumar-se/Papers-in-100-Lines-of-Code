"""SimpleUNet as one chain: every line is x = layer(x), nothing else declared.

Forward-pass demo only: PyTorch keeps the weights inside the layer object, so a
layer built inline is freed after the call -- there is nothing left to train or
to hand to an optimizer. The trainable version is unet_simple.py.
https://www.youtube.com/watch?v=o5gPABcGZwc
https://www.learnpytorch.io/pytorch_cheatsheet/
https://rstudio.github.io/cheatsheets/torch.pdf
Explaining over lunch: Rule
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

# 64 images, 1 channel, 32x32. 1 channel because MNIST is grayscale (28x28 is
# resized to 32x32 in diffusion_models.sample_batch so that three 2x pools land
# exactly on 4x4; 28 is not divisible by 8). 64 is a normal training batch.
x = torch.randn(64, 1, 32, 32)                                                   # [B,1,32,32]: MNIST is grayscale
# One integer step per image, uniform over 0..999 (T = 1000 in this repo).
# Note the 64s here and in (64,) are the BATCH size -- they have nothing to do
# with the 64-wide embedding below; that one is 2 * 32 frequencies.
# randint, not randn: t is an index into the noise schedule, so it must be a
# whole number, and uniform so every noise level is seen equally often.
t = torch.randint(0, 1000, (64,))                                                # [B] random step 0..999

# Sinusoidal frequencies: freq_i = exp(-i * log(10000) / (half - 1)), i = 0..31.
# This is the Transformer positional encoding applied to the scalar t: 32
# frequencies spread geometrically from 1 down to 1e-4, so low i changes fast
# with t and high i changes slowly -- together they give t a unique fingerprint.
freq = torch.exp(torch.arange(32.) * -(math.log(10000) / 31))                    # [32]
# sin and cos of t * each frequency, concatenated -> 64 numbers describing t.
# Both are needed: sin alone repeats every 2*pi / freq, cos breaks the tie.
e = torch.cat([torch.sin(t[:, None] * freq), torch.cos(t[:, None] * freq)], 1)   # [B,64]
# Two widths appear below, and they are easy to mix up:
#   64  = this raw code here (32 sines + 32 cosines, not learned yet)
#   256 = the learned temb that comes out of the MLP (4 * t_dim, t_dim = 64)
# every block adds the 256-wide temb, never the raw 64.
# MLP widens 64 -> 256; this 4x factor is convention from unet.py (512 = 4*128),
# not part of the diffusion math. The net decides which frequencies matter
# instead of using them raw.
e = nn.Linear(64, 256)(e)
# SiLU (x * sigmoid(x)): smooth, used in DDPM; keeps negative values, unlike ReLU.
e = F.silu(e)
# Second linear, deliberately with no activation after it, same as unet.py.
e = nn.Linear(256, 256)(e)                                                       # [B,256] temb

# 3x3 conv, padding=1 keeps H,W; 1 -> 32 channels (the "base" width).
x = nn.Conv2d(1, 32, 3, padding=1)(x)
# BatchNorm: per-channel mean/var over batch+H+W, so activations stay O(1) and
# the 3e-4 learning rate works. (This is why sampling must be done in eval mode.)
x = nn.BatchNorm2d(32)(x)
x = F.relu(x)
# Second conv of the "DoubleConv" block: keeps 32 channels, H,W unchanged.
x = nn.Conv2d(32, 32, 3, padding=1)(x)
x = nn.BatchNorm2d(32)(x)
# Add the timestep *after* the last ReLU, so the activation cannot squash it.
# Linear(256 -> 32) projects temb to 32 channels; [:, :, None, None] makes it
# [B,32,1,1] so the same 32 numbers are broadcast over every pixel.
x = F.relu(x) + nn.Linear(256, 32)(e)[:, :, None, None]                          # +t
# Keep the full-resolution result: the decoder concatenates it later. Without
# skips, detail lost at 4x4 could never come back.
s1 = x                                                                           # 32x32x32 skip
# 2x2 max pool: 32->16, no weights, widens the receptive field by 2.
x = F.max_pool2d(x, 2)

# Level 2: channels double to 64 because H,W just halved (keeps cost per level flat).
x = nn.Conv2d(32, 64, 3, padding=1)(x)
x = nn.BatchNorm2d(64)(x)
x = F.relu(x)
x = nn.Conv2d(64, 64, 3, padding=1)(x)
x = nn.BatchNorm2d(64)(x)
x = F.relu(x) + nn.Linear(256, 64)(e)[:, :, None, None]                          # +t
s2 = x                                                                           # 16x16x64 skip
x = F.max_pool2d(x, 2)

# Level 3: 128 channels at 8x8.
x = nn.Conv2d(64, 128, 3, padding=1)(x)
x = nn.BatchNorm2d(128)(x)
x = F.relu(x)
x = nn.Conv2d(128, 128, 3, padding=1)(x)
x = nn.BatchNorm2d(128)(x)
x = F.relu(x) + nn.Linear(256, 128)(e)[:, :, None, None]                         # +t
s3 = x                                                                           # 8x8x128 skip
x = F.max_pool2d(x, 2)

# Bottleneck: 256 channels at 4x4, the only place where a pixel can influence a
# far pixel, since the receptive field finally covers the whole image.
x = nn.Conv2d(128, 256, 3, padding=1)(x)
x = nn.BatchNorm2d(256)(x)
x = F.relu(x)
x = nn.Conv2d(256, 256, 3, padding=1)(x)
x = nn.BatchNorm2d(256)(x)
x = F.relu(x) + nn.Linear(256, 256)(e)[:, :, None, None]                         # +t

# Upsample nearest x2 (no weights), then concat the matching skip:
# 256 upsampled + 128 skip = 384 input channels, conv halves back to 128.
x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s3], 1)
x = nn.Conv2d(384, 128, 3, padding=1)(x)
x = nn.BatchNorm2d(128)(x)
x = F.relu(x)
x = nn.Conv2d(128, 128, 3, padding=1)(x)
x = nn.BatchNorm2d(128)(x)
x = F.relu(x) + nn.Linear(256, 128)(e)[:, :, None, None]                         # +t

# Next level: 128 upsampled + 64 skip = 192 in, 64 out.
x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s2], 1)
x = nn.Conv2d(192, 64, 3, padding=1)(x)
x = nn.BatchNorm2d(64)(x)
x = F.relu(x)
x = nn.Conv2d(64, 64, 3, padding=1)(x)
x = nn.BatchNorm2d(64)(x)
x = F.relu(x) + nn.Linear(256, 64)(e)[:, :, None, None]                          # +t

# Last level, back at full resolution: 64 upsampled + 32 skip = 96 in, 32 out.
x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s1], 1)
x = nn.Conv2d(96, 32, 3, padding=1)(x)
x = nn.BatchNorm2d(32)(x)
x = F.relu(x)
x = nn.Conv2d(32, 32, 3, padding=1)(x)
x = nn.BatchNorm2d(32)(x)
x = F.relu(x) + nn.Linear(256, 32)(e)[:, :, None, None]                          # +t

# 1x1 conv: mixes the 32 channels at each pixel without looking at neighbours,
# and gives 1 output channel because eps has the shape of the image. No sigmoid:
# predicted noise is signed, roughly N(0,1), not a probability.
x = nn.Conv2d(32, 1, 1)(x)                                                       # [B,1,32,32], no sigmoid

print("output", tuple(x.shape))
