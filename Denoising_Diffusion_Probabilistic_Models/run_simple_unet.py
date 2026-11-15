# Train the bare-minimum SimpleUNet from unet_simple.py on MNIST, then measure
# how good the samples actually are:
#
#   1. TRAIN          -- same diffusion_models.py loop, just a different model
#   2. SAMPLE         -- 81 images per milestone, saved as a 9x9 grid
#   3. FID            -- 1000 generated vs 1000 real, InceptionV3 pool3 features
#   4. CALIBRATION    -- two reference FIDs so the number means something:
#                          real vs real   = the floor at this sample size
#                          noise vs real  = what "no signal at all" scores
#
# run:
#     python run_simple_unet.py                    # full run, ~25 min on a 4070
#     python run_simple_unet.py --steps 2000       # quick look
#
# why FID needs calibration here: FID is biased upward when the sample count is
# small, so "FID = 40" is meaningless without knowing that this image set scores
# 5 for real-vs-real. the two reference numbers are what the model's FID should
# be compared against, not zero.
import argparse
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import diffusion_models as dm_module
from diffusion_models import DiffusionModel, device, testX
from unet_simple import SimpleUNet

# diffusion_models.sampling() wraps its loop in tqdm; we call it once per chunk,
# so that would print ten progress bars per milestone. silence it here instead of
# touching the paper implementation.
dm_module.tqdm = lambda iterable, **kwargs: iterable


def build_model(name):
    """
    Name -> denoiser. Anything here satisfies forward(x, t) -> same shape as x,
    which is the only contract diffusion_models.py has.

    imports are inside the branches so a run only pays for what it uses (the
    reference UNet alone is 35.7M parameters).
    """
    if name == "simple":
        from unet_simple import SimpleUNet

        return SimpleUNet()
    if name == "unet":
        from unet import UNet

        return UNet()
    if name in ("tiny_unet", "dit", "convnet", "mlp"):
        from denoisers import build

        return build(name)
    raise ValueError(f"unknown model {name!r}")


def sample_images(model, diffusion, n_samples, chunk=100):
    """Generate n_samples images, a chunk at a time so activations stay small."""
    # BatchNorm uses running stats, so eval() is mandatory: in train() mode the
    # batch statistics of this chunk would leak into the output.
    model.eval()
    with torch.no_grad():
        out = [
            diffusion.sampling(n_samples=min(chunk, n_samples - i))
            for i in range(0, n_samples, chunk)
        ]
    model.train()
    return torch.cat(out, dim=0)


def save_grid(samples, path, n=81, cols=9):
    """9x9 grid, same layout as the Imgs/samples.png in the README."""
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols, rows))
    for i in range(n):
        plt.subplot(rows, cols, 1 + i)
        plt.imshow(samples[i, 0].clip(0, 1).cpu(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def save_real_vs_generated(samples, path="Imgs/simple_unet_real_vs_gen.png"):
    """
    Top 4 rows: real training data. Bottom 4 rows: generated.
    Same 32x32 nearest-neighbour grid the model was trained on, so the two rows
    are directly comparable instead of one looking soft for the wrong reason.
    """
    rows, cols = 8, 4  # 32 tiles: 16 real on top, 16 generated below
    real = torch.nn.functional.interpolate(testX[: rows * cols].unsqueeze(1), 32)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            if r < rows // 2:
                ax.imshow(real[r * cols + c, 0].cpu(), cmap="gray")
            else:
                idx = (r - rows // 2) * cols + c
                ax.imshow(samples[idx, 0].clip(0, 1).cpu(), cmap="gray")
    fig.text(0.01, 0.72, "real", rotation=90, fontsize=14, va="center")
    fig.text(0.01, 0.28, "generated", rotation=90, fontsize=14, va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def dump_pngs(images, folder, prefix):
    """
    Write [N,1,H,W] floats in [0,1] out as 3-channel PNGs.

    pytorch-fid resizes to 299x299 with TF and expects RGB, so convert here and
    keep the pipeline simple. the 1->3 channel change does not add information,
    it just satisfies Inception's first conv.

    np.round before uint8: .astype(np.uint8) TRUNCATES, so 0.999 becomes 254/255
    and every image comes out slightly darker than the tensor. that single
    detail moved FID by ~1.8, which is larger than the effect of 2,500 training
    steps. quantize with round, the way every image library does it.
    """
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        arr = np.round(img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).convert("RGB").save(os.path.join(folder, f"{prefix}_{i:04d}.png"))


def compute_fid(fake_dir, real_dir, batch_size=50):
    # imported here so the training loop does not pay the inception import cost
    from pytorch_fid.fid_score import calculate_fid_given_paths

    # num_workers=0: Windows spawn-based workers inside a script are trouble,
    # and inception on 2000 images is fast enough without them
    return calculate_fid_given_paths(
        [fake_dir, real_dir],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--fid-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model", type=str, default="simple",
        help="simple | unet | tiny_unet | dit | convnet | mlp",
    )
    parser.add_argument("--tag", type=str, default="simple_unet")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs("Imgs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    model = build_model(args.model).to(device)
    diffusion = DiffusionModel(args.T, model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}  params={n_params/1e6:.2f}M  device={device}")
    print(f"config: steps={args.steps} batch={args.batch} lr={args.lr} T={args.T}")

    # milestones as fractions of the run, so --steps still works for quick runs
    milestones = {int(args.steps * f): int(f * 100) for f in (0.25, 0.5, 0.75, 1.0)}
    losses = []

    start = time.perf_counter()
    for step in tqdm(range(1, args.steps + 1), ncols=80):
        loss = diffusion.train_step(args.batch, optimizer)
        losses.append(loss)

        if step % 500 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"step {step:6d}  loss {np.mean(losses[-500:]):.4f}"
                f"  {elapsed/60:5.1f} min  ({elapsed/step*1000:.0f} ms/step)"
            )

        if step in milestones:
            pct = milestones[step]
            print(f"-- sampling at {pct}% ({step} steps) --")
            samples = sample_images(model, diffusion, 81)
            save_grid(samples, f"Imgs/{args.tag}_{pct:03d}.png")
            torch.save(
                {"model": model.state_dict(), "steps": step, "args": vars(args)},
                f"checkpoints/{args.tag}_{pct:03d}.pt",
            )
            torch.cuda.empty_cache()

    train_min = (time.perf_counter() - start) / 60

    # loss curve, log scale, so the first hundred steps do not flatten it
    plt.figure(figsize=(6, 3))
    plt.plot(losses, linewidth=0.4)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE noise loss")
    plt.title(f"SimpleUNet, {args.steps} steps")
    plt.tight_layout()
    plt.savefig(f"Imgs/{args.tag}_loss.png", dpi=110)
    plt.close()

    # ---------------------------------------------------------------- sampling
    print(f"-- generating {args.fid_n} images for FID --")
    samples = sample_images(model, diffusion, args.fid_n)
    save_grid(samples, f"Imgs/{args.tag}_samples.png")
    save_real_vs_generated(samples, path=f"Imgs/{args.tag}_real_vs_gen.png")

    # -------------------------------------------------------------------- FID
    # real and fake must live on the same pixel grid, otherwise FID measures the
    # 28->32 resampling difference instead of the model. sample_batch() trains on
    # nearest-neighbour-upscaled 32x32, so mirror exactly that for the real set.
    tmp = "fid_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    n = args.fid_n

    real32 = torch.nn.functional.interpolate(testX[: 2 * n].unsqueeze(1), 32)
    dump_pngs(real32[:n], f"{tmp}/real_a", "real")
    dump_pngs(real32[n:], f"{tmp}/real_b", "real")
    dump_pngs(samples, f"{tmp}/fake", "fake")

    noise = torch.rand(n, 1, 32, 32)
    dump_pngs(noise, f"{tmp}/noise", "noise")

    print("-- FID --")
    fid_model = compute_fid(f"{tmp}/fake", f"{tmp}/real_a")
    fid_floor = compute_fid(f"{tmp}/real_b", f"{tmp}/real_a")
    fid_noise = compute_fid(f"{tmp}/noise", f"{tmp}/real_a")
    shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"  SimpleUNet vs real      FID = {fid_model:7.2f}   <- the model")
    print(f"  real vs real            FID = {fid_floor:7.2f}   <- floor at n={n}")
    print(f"  uniform noise vs real   FID = {fid_noise:7.2f}   <- no signal")
    print(f"  train time {train_min:.1f} min, {n_params/1e6:.2f}M params")
    print()
    print("  Imgs/", f"{args.tag}_samples.png       81 samples")
    print("  Imgs/", f"{args.tag}_real_vs_gen.png    real (top) vs generated (bottom)")
    print("  Imgs/", f"{args.tag}_loss.png          training curve")

    with open(f"Imgs/{args.tag}_fid.txt", "w") as f:
        f.write(f"model_params={n_params}\n")
        f.write(f"config={vars(args)}\n")
        f.write(f"train_minutes={train_min:.1f}\n")
        f.write(f"fid_model={fid_model:.4f}\n")
        f.write(f"fid_real_vs_real_floor={fid_floor:.4f}\n")
        f.write(f"fid_noise={fid_noise:.4f}\n")


if __name__ == "__main__":
    main()
