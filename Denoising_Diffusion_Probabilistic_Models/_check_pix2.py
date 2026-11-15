import glob
import shutil

import numpy as np
import torch
from PIL import Image

from run_simple_unet import dump_pngs

fake = torch.load("eval/tensors/fake_100_seed0.pt", map_location="cpu")[:4]
tmp = "eval/_pixcheck"
shutil.rmtree(tmp, ignore_errors=True)
dump_pngs(fake, tmp, "f")

for i, f in enumerate(sorted(glob.glob(tmp + "/*.png"))):
    a = np.array(Image.open(f))
    print(f, a.shape, a.dtype, "modes ok")
    want = np.round(fake[i, 0].numpy() * 255).astype(np.int32)
    got = a[:, :, 0].astype(np.int32)
    d = np.abs(got - want)
    print("  max|d|", d.max(), "argmax", np.unravel_index(d.argmax(), d.shape))
    if d.max() > 0:
        r, c = np.unravel_index(d.argmax(), d.shape)
        print("  got", got[r, c], "want", want[r, c], "float", fake[i, 0, r, c].item())
    print("  channel spread R-G-G-B:", np.abs(a[:, :, 0].astype(int) - a[:, :, 1]).max(),
          np.abs(a[:, :, 1].astype(int) - a[:, :, 2]).max())
shutil.rmtree(tmp, ignore_errors=True)
