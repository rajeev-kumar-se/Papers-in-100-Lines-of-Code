"""Reproduce the cross-check at n=1000 and locate any difference."""
import glob
import shutil
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

import evaluate as ev
from pytorch_fid.inception import InceptionV3
from run_simple_unet import dump_pngs
from diffusion_models import device

fake = torch.load("eval/tensors/fake_100_seed0.pt", map_location="cpu")
real = torch.load("eval/tensors/real_a.pt", map_location="cpu")

tmp = "eval/_xc2"
shutil.rmtree(tmp, ignore_errors=True)
dump_pngs(fake, f"{tmp}/fake", "f")
dump_pngs(real, f"{tmp}/real", "r")

out = subprocess.run(
    [sys.executable, "-m", "pytorch_fid", f"{tmp}/fake", f"{tmp}/real", "--device", "cpu", "--batch-size", "50"],
    capture_output=True, text=True,
)
print("CLI says:", [ln for ln in out.stdout.splitlines() if "FID" in ln])

def feats_from_pngs(folder):
    files = sorted(glob.glob(folder + "/*.png"))
    print(folder, "n files", len(files))
    model = InceptionV3([3]).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(files), 50):
            x = torch.stack(
                [torch.from_numpy(np.array(Image.open(f).convert("RGB"))).permute(2, 0, 1) for f in files[i:i+50]]
            ).float() / 255
            out.append(model(x.to(device))[0].mean([2, 3]).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)

pf = feats_from_pngs(f"{tmp}/fake")
pr = feats_from_pngs(f"{tmp}/real")
tf = np.load("eval/tensors/feat_fake_100_seed0.npy")
tr = np.load("eval/tensors/feat_real_a.npy")

print("feature drift fake", np.abs(pf - tf).max(), " real", np.abs(pr - tr).max())
print("FID from png features   ", ev.fid(pr, pf))
print("FID from cached features", ev.fid(tr, tf))
shutil.rmtree(tmp, ignore_errors=True)
