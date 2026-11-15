"""Feature vectors from PNGs vs from our tensors: same pixels, same numbers?"""
import glob
import shutil

import numpy as np
import torch
from PIL import Image

import evaluate as ev
from pytorch_fid.inception import InceptionV3
from run_simple_unet import dump_pngs
from diffusion_models import device

fake = torch.load("eval/tensors/fake_100_seed0.pt", map_location="cpu")[:100]
real = torch.load("eval/tensors/real_a.pt", map_location="cpu")[:100]

tmp = "eval/_fc"
shutil.rmtree(tmp, ignore_errors=True)
dump_pngs(fake, f"{tmp}/fake", "f")
dump_pngs(real, f"{tmp}/real", "r")


def feats_from_pngs(folder):
    files = sorted(glob.glob(folder + "/*.png"))
    model = InceptionV3([3]).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(files), 50):
            x = torch.stack(
                [torch.from_numpy(np.array(Image.open(f).convert("RGB"))).permute(2, 0, 1) for f in files[i : i + 50]]
            ).float() / 255
            out.append(model(x.to(device))[0].mean([2, 3]).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


pf = feats_from_pngs(f"{tmp}/fake")
pr = feats_from_pngs(f"{tmp}/real")
tf = ev.inception_features(fake)
tr = ev.inception_features(real)

print("png-feat vs tensor-feat  fake max|d|", np.abs(pf - tf).max(), " mean|d|", np.abs(pf - tf).mean())
print("png-feat vs tensor-feat  real max|d|", np.abs(pr - tr).max(), " mean|d|", np.abs(pr - tr).mean())
print("FID pngs   ", ev.fid(pr, pf))
print("FID tensors", ev.fid(tr, tf))
shutil.rmtree(tmp, ignore_errors=True)
