"""Compare the three ways of computing FID on identical PNGs at n=1000."""
import glob
import shutil
import subprocess
import sys

import torch

import evaluate as ev
from pytorch_fid.fid_score import calculate_fid_given_paths
from run_simple_unet import dump_pngs
from diffusion_models import device

fake = torch.load("eval/tensors/fake_100_seed0.pt", map_location="cpu")
real = torch.load("eval/tensors/real_a.pt", map_location="cpu")

tmp = "eval/_xc3"
shutil.rmtree(tmp, ignore_errors=True)
dump_pngs(fake, f"{tmp}/fake", "f")
dump_pngs(real, f"{tmp}/real", "r")
print("files", len(glob.glob(tmp + "/fake/*.png")), len(glob.glob(tmp + "/real/*.png")))

inproc = calculate_fid_given_paths([f"{tmp}/fake", f"{tmp}/real"], 50, device, 2048, 0)
print(f"in-process library      {inproc:.4f}")

for nw in ("0", "1"):
    out = subprocess.run(
        [sys.executable, "-m", "pytorch_fid", f"{tmp}/fake", f"{tmp}/real",
         "--device", "cpu", "--batch-size", "50", "--num-workers", nw],
        capture_output=True, text=True,
    )
    line = [ln for ln in (out.stdout or "").splitlines() if "FID" in ln]
    print(f"CLI num_workers={nw}       {line}  rc={out.returncode}")
    if out.returncode:
        print("   stderr tail:", out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "")

print(f"our features            {ev.fid(__import__('numpy').load('eval/tensors/feat_real_a.npy'), __import__('numpy').load('eval/tensors/feat_fake_100_seed0.npy')):.4f}")
shutil.rmtree(tmp, ignore_errors=True)
