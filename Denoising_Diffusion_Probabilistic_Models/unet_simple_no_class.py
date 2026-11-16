"""SimpleUNet as one chain: every line is x = layer(x), nothing else declared.

Forward-pass demo only: PyTorch keeps the weights inside the layer object, so a
layer built inline is freed after the call -- there is nothing left to train or
to hand to an optimizer. The trainable version is unet_simple.py.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn(64, 1, 32, 32)                                                   # [B,1,32,32]: MNIST is grayscale
t = torch.randint(0, 1000, (len(x),))                                            # [B] random step 0..999
# A timestep t is the index of a row in the noise schedule (√ᾱₜ, βₜ, …), so it has to be an integer in [0, T−1]

freq = torch.exp(torch.arange(32.) * -(math.log(10000) / 31))                    # [32]
e = torch.cat([torch.sin(t[:, None] * freq), torch.cos(t[:, None] * freq)], 1)   # [B,64]
e = nn.Linear(64, 256)(e)
e = F.silu(e)
e = nn.Linear(256, 256)(e)                                                       # [B,256] temb

x = nn.Conv2d(1, 32, 3, padding=1)(x)
x = nn.BatchNorm2d(32)(x)
x = F.relu(x)
x = nn.Conv2d(32, 32, 3, padding=1)(x)
x = nn.BatchNorm2d(32)(x)
x = F.relu(x) + nn.Linear(256, 32)(e)[:, :, None, None]                          # +t
s1 = x                                                                           # 32x32x32 skip
x = F.max_pool2d(x, 2)

x = nn.Conv2d(32, 64, 3, padding=1)(x)
x = nn.BatchNorm2d(64)(x)
x = F.relu(x)
x = nn.Conv2d(64, 64, 3, padding=1)(x)
x = nn.BatchNorm2d(64)(x)
x = F.relu(x) + nn.Linear(256, 64)(e)[:, :, None, None]                          # +t
s2 = x                                                                           # 16x16x64 skip
x = F.max_pool2d(x, 2)

x = nn.Conv2d(64, 128, 3, padding=1)(x)
x = nn.BatchNorm2d(128)(x)
x = F.relu(x)
x = nn.Conv2d(128, 128, 3, padding=1)(x)
x = nn.BatchNorm2d(128)(x)
x = F.relu(x) + nn.Linear(256, 128)(e)[:, :, None, None]                         # +t
s3 = x                                                                           # 8x8x128 skip
x = F.max_pool2d(x, 2)

x = nn.Conv2d(128, 256, 3, padding=1)(x)
x = nn.BatchNorm2d(256)(x)
x = F.relu(x)
x = nn.Conv2d(256, 256, 3, padding=1)(x)
x = nn.BatchNorm2d(256)(x)
x = F.relu(x) + nn.Linear(256, 256)(e)[:, :, None, None]                         # +t

x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s3], 1)
x = nn.Conv2d(384, 128, 3, padding=1)(x)
x = nn.BatchNorm2d(128)(x)
x = F.relu(x)
x = nn.Conv2d(128, 128, 3, padding=1)(x)
x = nn.BatchNorm2d(128)(x)
x = F.relu(x) + nn.Linear(256, 128)(e)[:, :, None, None]                         # +t

x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s2], 1)
x = nn.Conv2d(192, 64, 3, padding=1)(x)
x = nn.BatchNorm2d(64)(x)
x = F.relu(x)
x = nn.Conv2d(64, 64, 3, padding=1)(x)
x = nn.BatchNorm2d(64)(x)
x = F.relu(x) + nn.Linear(256, 64)(e)[:, :, None, None]                          # +t

x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s1], 1)
x = nn.Conv2d(96, 32, 3, padding=1)(x)
x = nn.BatchNorm2d(32)(x)
x = F.relu(x)
x = nn.Conv2d(32, 32, 3, padding=1)(x)
x = nn.BatchNorm2d(32)(x)
x = F.relu(x) + nn.Linear(256, 32)(e)[:, :, None, None]                          # +t

x = nn.Conv2d(32, 1, 1)(x)                                                       # [B,1,32,32], no sigmoid

print("output", tuple(x.shape))
