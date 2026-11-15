# Bring the checkpoints we already have up to a resumable format.
#
#     python checkpoints_meta.py
#
# the training run saved only `{"model": state_dict}` at 4 milestones, which is
# enough to sample but not enough to resume: no optimizer state, no step counter,
# no config. this rewrites each file in place, keeping the `model` key so
# evaluate.py and anything else that loads them still works, and adds the rest.
#
# honest limitation: optimizer state CANNOT be reconstructed after the fact. the
# field is present and null, so a future run knows it has to start from scratch
# rather than silently resuming with a fresh Adam.
import json
import os
import pickle

import torch

CKPTS = (25, 50, 75, 100)
STEPS = {25: 2500, 50: 5000, 75: 7500, 100: 10000}


def load_ckpt(path):
    """torch.load that survives both our own old and new checkpoint layouts.

    torch 2.6 defaults to weights_only=True, which rejects the TorchVersion
    object that `torch.__version__` pickles as (hence the string cast below).
    falling back to weights_only=False is fine here: these are files this script
    wrote itself, not downloads.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)

# the exact configuration the 10k-step run used (from logs/simple_unet.log)
CONFIG = {
    "architecture": "SimpleUNet",
    "params": 2_212_481,
    "in_ch": 1,
    "base": 32,
    "t_dim": 64,
    "T": 1000,
    "beta": "linear 1e-4 -> 0.02",
    "objective": "epsilon-prediction MSE",
    "optimizer": "Adam",
    "lr": 3e-4,
    "batch": 256,
    "steps": 10000,
    "seed": 0,
    "dataset": "MNIST, 28x28 nearest-upscaled to 32x32, scaled to [0,1]",
}


def main():
    manifest = []
    for pct in CKPTS:
        path = f"checkpoints/simple_unet_{pct:03d}.pt"
        if not os.path.exists(path):
            print(f"skip {path} (missing)")
            continue

        old = load_ckpt(path)
        state = old["model"] if isinstance(old, dict) and "model" in old else old

        torch.save(
            {
                "model": state,
                "optimizer": None,  # not recoverable, see header
                "step": STEPS[pct],
                "pct": pct,
                "args": CONFIG,
                # str() on purpose: torch.__version__ is a TorchVersion instance,
                # which torch>=2.6 weights_only=True refuses to unpickle
                "torch_version": str(torch.__version__),
                "notes": "weights only; no optimizer/RNG state from the original run",
            },
            path,
        )

        manifest.append(
            {
                "file": path,
                "pct": pct,
                "step": STEPS[pct],
                # exclude BN running stats / num_batches_tracked: those are
                # buffers, and counting them would misstate the model size
                "params": sum(
                    v.numel()
                    for k, v in state.items()
                    if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))
                ),
                "bytes": os.path.getsize(path),
                "resumable": False,
            }
        )
        print(f"rewrote {path}  (step {STEPS[pct]}, {os.path.getsize(path)/1e6:.1f} MB)")

    # no best.pt copy: it would be byte-identical to the 100% checkpoint below,
    # since nothing was trained after it. a duplicate is not a checkpoint.
    with open("checkpoints/manifest.json", "w") as f:
        json.dump(
            {
                "best": "checkpoints/simple_unet_100.pt",
                "checkpoints": manifest,
                "policy": {
                    "keep_permanent": ["*_025.pt", "*_050.pt", "*_075.pt", "*_100.pt"],
                    "keep_rolling": "last 3 if a run saves every 1k steps",
                    "mnist_cnn.pt": "evaluation classifier, not part of the diffusion model",
                },
                "storage_note": "2.2M params = 8.8 MB weights, ~27 MB with Adam state",
            },
            f,
            indent=2,
        )
    print("wrote checkpoints/manifest.json")


if __name__ == "__main__":
    main()
