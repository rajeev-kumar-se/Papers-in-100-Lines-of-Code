"""Is eval/tensors/feat_*.npy still what the extractor produces today?"""
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import torch

import evaluate as ev

for name in ("real_a", "fake_100_seed0"):
    cached = np.load(f"eval/tensors/feat_{name}.npy")
    imgs = torch.load(f"eval/tensors/{name}.pt", map_location="cpu")
    fresh = ev.inception_features(imgs)
    d = np.abs(cached - fresh)
    print(
        f"{name:16s} cached {cached.shape} fresh {fresh.shape} "
        f"max|d| {d.max():.3e}  mean|d| {d.mean():.3e}"
    )

real = np.load("eval/tensors/feat_real_a.npy")
fake_cached = np.load("eval/tensors/feat_fake_100_seed0.npy")
fake_fresh = ev.inception_features(torch.load("eval/tensors/fake_100_seed0.pt", map_location="cpu"))
print(f"FID with cached features {ev.fid(real, fake_cached):.4f}")
print(f"FID with fresh  features {ev.fid(real, fake_fresh):.4f}")
print(f"sqrt paths used {ev._METHODS}")
