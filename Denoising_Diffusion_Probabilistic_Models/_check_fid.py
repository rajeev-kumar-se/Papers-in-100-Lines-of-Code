"""Ad-hoc check: our frechet() vs pytorch-fid's, at every n in the bias curve."""
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance

import evaluate as ev

fa = np.load("eval/tensors/feat_real_a.npy")
fb = np.load("eval/tensors/feat_real_b.npy")
fg = np.load("eval/tensors/feat_fake_100_seed0.npy")
fn = np.load("eval/tensors/feat_noise.npy")


def stats(f):
    return f.mean(0), np.cov(f, rowvar=False)


print("n     set     ours        lib         abs diff")
for n in (100, 250, 500, 1000):
    a = stats(fa[:n])
    for name, v in (("floor", fb), ("model", fg), ("noise", fn)):
        s = stats(v[:n])
        ours = ev.frechet(a[0], a[1], s[0], s[1])
        try:
            ref = calculate_frechet_distance(a[0], a[1], s[0], s[1])
            d = abs(ours - ref)
            print(f"{n:4d}  {name:6s}  {ours:10.4f}  {ref:10.4f}  {d:.2e}")
        except Exception as exc:
            print(f"{n:4d}  {name:6s}  {ours:10.4f}  LIB RAISED {type(exc).__name__}")
    print()
