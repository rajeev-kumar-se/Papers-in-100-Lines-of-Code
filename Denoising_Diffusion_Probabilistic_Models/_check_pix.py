"""Do the PNGs the library reads contain the same pixels as our tensors?"""
import glob
import os
import shutil

import numpy as np
import torch
from PIL import Image

import evaluate as ev
from run_simple_unet import dump_pngs

fake = torch.load("eval/tensors/fake_100_seed0.pt", map_location="cpu")
tmp = "eval/_pixcheck"
shutil.rmtree(tmp, ignore_errors=True)
dump_pngs(fake[:40], tmp, "f")

files = sorted(glob.glob(tmp + "/*.png"))
back = np.stack([np.array(Image.open(f))[:, :, 0] for f in files])
want = np.round(fake[:40, 0].numpy() * 255).astype(np.int32)

print("pixels equal to round(x*255):", np.array_equal(back.astype(np.int32), want))
print("max |png - round| =", np.abs(back.astype(np.int32) - want).max())
print("max |png - trunc| =", np.abs(back.astype(np.int32) - (fake[:40, 0].numpy() * 255).astype(np.int32)).max())

# and the tensor the library would build, vs ours
lib_t = torch.stack([torch.from_numpy(np.array(Image.open(f))).permute(2, 0, 1) for f in files]).float() / 255
ours = ev.quantize(fake[:40])
print("library tensor matches our quantize():", torch.equal(lib_t[:, :1], time_check := ours[:, :1]) if False else torch.allclose(lib_t[:, :1], ours, atol=0))
print("max |lib tensor - our quantize| =", (lib_t[:, :1] - ours).abs().max().item())
shutil.rmtree(tmp, ignore_errors=True)
