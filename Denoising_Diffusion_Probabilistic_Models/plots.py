# Build every graph from eval/metrics.json + the cached tensors.
#
#     python plots.py
#
# CPU only, no model, no GPU: this just draws what evaluate.py measured, so it can
# be re-run while tweaking labels without paying for the evaluation again.
#
# output: eval/graphs/g01..g11 and eval/samples/*.png
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DIR = "eval"
STEPS_AT = {25: 2500, 50: 5000, 75: 7500, 100: 10000}  # milestone -> training steps
CKPTS = (25, 50, 75, 100)


def load():
    with open(f"{DIR}/metrics.json") as f:
        return json.load(f)


def _read_text(path):
    """Read a log whatever encoding PowerShell decided to use.

    `Tee-Object` on Windows writes UTF-16LE with a BOM, while Python's run logs
    are UTF-8. Decoding UTF-16 bytes as UTF-8 gives NUL-separated garbage that
    silently matches nothing -- which is how g03 ended up with an empty loss
    curve. Detect the BOM instead of assuming.
    """
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8", errors="ignore")


def parse_loss_log(path="logs/simple_unet.log"):
    """Pull the 'step  500  loss 0.0688' lines the trainer printed every 500 steps."""
    steps, losses = [], []
    for line in _read_text(path).splitlines():
        m = re.search(r"step\s+(\d+)\s+loss\s+([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return steps, losses


def save(fig, name):
    fig.savefig(f"{DIR}/graphs/{name}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# ------------------------------------------------------------------ g01
def g01_fid_kid_vs_steps(m):
    steps = [STEPS_AT[p] for p in CKPTS]
    fid = [m["fid"][f"{p}_0"] for p in CKPTS]
    kid = [m["kid"][f"{p}_0"] for p in CKPTS]

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(steps, fid, "o-", label="FID (n=1000)", color="#1f77b4")
    ax.axhline(m["fid_real_vs_real"], ls="--", lw=1, color="green",
               label=f"real-vs-real floor {m['fid_real_vs_real']:.1f}")

    # error bar from the 3 sampling seeds: the metric's own noise at the last point
    s = m["fid_stats"]
    ax.errorbar([steps[-1]], [s["mean"]], yerr=[s["std"]], fmt="o",
                capsize=4, color="#1f77b4")
    ax.annotate(f"{s['mean']:.2f} ± {s['std']:.2f}", (steps[-1], s["mean"]),
                textcoords="offset points", xytext=(-60, 8), fontsize=8)

    ax.set_xlabel("training step")
    ax.set_ylabel("FID", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax.twinx()
    ax2.plot(steps, kid, "s--", color="#d62728", alpha=0.8)
    ax2.set_ylabel("KID (MMD², unbiased)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax.set_title("Quality vs training: FID (biased) and KID (unbiased)")
    ax.grid(alpha=0.2)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.86), fontsize=8)
    save(fig, "g01_fid_kid_vs_steps.png")


# ------------------------------------------------------------------ g02
def g02_fid_vs_n(m):
    ns = sorted(int(k) for k in m["fid_vs_n"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for key, style, label in (
        ("model", "o-", "SimpleUNet vs real"),
        ("floor", "s--", "real vs real (floor)"),
        ("noise", "^:", "uniform noise"),
    ):
        ax.plot(ns, [m["fid_vs_n"][str(n)][key] for n in ns], style, label=label)
    # log y: the noise curve sits ~10x above the model curve and would otherwise
    # compress the model/floor gap -- the only part of this plot that matters
    ax.set_yscale("log")
    for key, color in (("model", "#1f77b4"), ("floor", "#ff7f0e")):
        for n in (ns[0], ns[-1]):
            v = m["fid_vs_n"][str(n)][key]
            ax.annotate(f"{v:.1f}", (n, v), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color=color)
    ax.set_xlabel("samples n per set")
    ax.set_ylabel("FID (log)")
    ax.set_title("FID is biased at small n — this is why we report the floor")
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, "g02_fid_vs_n.png")


# ------------------------------------------------------------------ g03
def g03_loss_and_fid(m):
    steps, losses = parse_loss_log()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(steps, losses, lw=0.8, color="#333")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("ε-MSE loss (log)")

    ax2 = ax.twinx()
    ax2.plot([STEPS_AT[p] for p in CKPTS], [m["fid"][f"{p}_0"] for p in CKPTS],
             "o-", color="#d62728")
    ax2.set_ylabel("FID", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    for p in CKPTS:
        ax2.annotate(f"{m['fid'][f'{p}_0']:.1f}",
                     (STEPS_AT[p], m["fid"][f"{p}_0"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8,
                     color="#d62728")

    ax.set_title("Loss falls 4.5x, FID falls 2x — loss is not a quality metric")
    ax.grid(alpha=0.2)
    save(fig, "g03_loss_and_fid.png")


# ------------------------------------------------------------------ g04/g05
def _grid(ax_row, images, cols):
    for c in range(cols):
        ax_row[c].axis("off")
        if c < len(images):
            ax_row[c].imshow(images[c, 0].clip(0, 1), cmap="gray")


def g04_progression(m, cols=12):
    rows = []
    for p in CKPTS:
        imgs = torch.load(f"{DIR}/tensors/fake_{p:03d}_seed0.pt", map_location="cpu")
        rows.append((p, m["fid"][f"{p}_0"], imgs[:cols]))

    fig, axes = plt.subplots(len(rows), cols, figsize=(cols * 0.85, len(rows) * 1.05))
    for r, (p, f, imgs) in enumerate(rows):
        _grid(axes[r], imgs.numpy(), cols)
        axes[r][0].text(-0.45, 0.5, f"{p}%  FID {f:.1f}", rotation=90, fontsize=9,
                        va="center", ha="right", transform=axes[r][0].transAxes)
    plt.suptitle("Samples through training (fixed seed)", y=0.98)
    save(fig, "g04_progression.png")


def g05_real_vs_generated(m, n=16):
    fake = torch.load(f"{DIR}/tensors/fake_100_seed0.pt", map_location="cpu")[:n]
    real = torch.load(f"{DIR}/tensors/real_a.pt", map_location="cpu")[:n]

    fig, axes = plt.subplots(2, n // 2 if n >= 8 else n, figsize=(11, 3))
    axes = np.atleast_2d(axes)
    for i in range(n // 2):
        axes[0][i].imshow(real[i, 0], cmap="gray"); axes[0][i].axis("off")
        axes[1][i].imshow(fake[i, 0].clip(0, 1), cmap="gray"); axes[1][i].axis("off")
    axes[0][0].set_ylabel("real")
    fig.text(0.02, 0.74, "real", rotation=90, va="center", fontsize=11)
    fig.text(0.02, 0.28, "generated", rotation=90, va="center", fontsize=11)
    save(fig, "g05_real_vs_generated.png")


# ------------------------------------------------------------------ g06
def g06_precision_recall(m):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    pts = [
        ("SimpleUNet", m["precision_recall"]["precision"], m["precision_recall"]["recall"], "#1f77b4"),
        ("real vs real", *m["precision_recall"]["real_vs_real"], "green"),
        ("noise", *m["precision_recall"]["noise"], "grey"),
    ]
    for label, p, r, c in pts:
        ax.scatter(r, p, s=110, color=c, label=label, zorder=3)
        ax.annotate(label, (r, p), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_xlabel("recall (coverage of the real distribution)")
    ax.set_ylabel("precision (sample fidelity)")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Precision / recall — separates fidelity from mode collapse")
    ax.grid(alpha=0.25)
    save(fig, "g06_precision_recall.png")


# ------------------------------------------------------------------ g07/g08
def g07_class_hist(m):
    gen = m["classifier"]["generated"]["hist"]
    real = m["classifier"]["real"]["hist"]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - 0.2, real, 0.4, label="real", color="#1f77b4")
    ax.bar(x + 0.2, gen, 0.4, label="generated (classified)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("fraction")
    ax.set_title("All 10 digits appear, but 1s are over-generated (26% vs 13%)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    save(fig, "g07_class_hist.png")


def g08_confidence(m):
    gen = np.array(m["classifier"]["generated"]["conf_hist"], dtype=float)
    real = np.array(m["classifier"]["real"]["conf_hist"], dtype=float)
    edges = np.linspace(0, 1, len(gen) + 1)[:-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(edges, real / real.sum(), width=0.04, label="real", color="#1f77b4", align="edge")
    ax.bar(edges, gen / gen.sum(), width=0.04, label="generated", color="#d62728",
           align="edge", alpha=0.65)
    ax.set_xlabel("classifier confidence (max softmax)")
    ax.set_ylabel("fraction of samples")
    ax.set_title(f"Confidence: real {m['classifier']['real']['conf_mean']:.3f} vs "
                 f"generated {m['classifier']['generated']['conf_mean']:.3f}")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "g08_confidence_hist.png")


# ------------------------------------------------------------------ g09
def g09_nn(m):
    g = np.array(m["nn_distance"]["generated"])
    r = np.array(m["nn_distance"]["real"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, max(g.max(), r.max()), 40)
    ax.hist(r, bins=bins, alpha=0.6, label="real test -> train", color="#1f77b4", density=True)
    ax.hist(g, bins=bins, alpha=0.6, label="generated -> train", color="#d62728", density=True)
    ax.set_xlabel("L2 distance to nearest training image")
    ax.set_ylabel("density")
    ax.set_title("Memorization check: generated samples are not copies")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "g09_nn_distance.png")


# ------------------------------------------------------------------ g10
def g10_per_timestep(m):
    t = m["per_timestep"]["t"]
    mse = m["per_timestep"]["mse"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(t, mse, "o-", ms=3, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("timestep t (noise level)")
    ax.set_ylabel("ε-MSE (log)")
    ax.set_title("Denoising loss across the schedule")
    ax.grid(alpha=0.2)
    save(fig, "g10_per_timestep_loss.png")


# ------------------------------------------------------------------ g11
def g11_schedule(T=1000):
    beta = torch.linspace(1e-4, 0.02, T)
    alpha_bar = torch.cumprod(1 - beta, 0)
    t = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, beta.numpy(), label="β_t (noise added per step)")
    ax.plot(t, alpha_bar.numpy(), label="ᾱ_t = ∏α (signal kept)")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("value")
    ax.set_title("Linear β schedule: signal dies, noise fills in")

    ax2 = ax.twinx()
    ax2.plot(t, torch.sqrt(alpha_bar).numpy(), "--", lw=1, label="√ᾱ_t (signal coeff)")
    ax2.plot(t, torch.sqrt(1 - alpha_bar).numpy(), ":", lw=1, label="√(1-ᾱ_t) (noise coeff)")
    ax2.set_ylabel("coefficient in x_t = √ᾱ x₀ + √(1-ᾱ) ε")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    ax.grid(alpha=0.2)
    save(fig, "g11_noise_schedule.png")


# ------------------------------------------------------------------ samples
def sample_grids():
    """The 9x9 grids, one per milestone, plus the final 81-image grid."""
    for p in CKPTS:
        imgs = torch.load(f"{DIR}/tensors/fake_{p:03d}_seed0.pt", map_location="cpu")
        fig = plt.figure(figsize=(9, 9))
        for i in range(81):
            plt.subplot(9, 9, i + 1)
            plt.imshow(imgs[i, 0].clip(0, 1), cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        fig.savefig(f"{DIR}/samples/grid_{p:03d}.png", dpi=100)
        plt.close(fig)
        print(f"  samples/grid_{p:03d}.png")


def main():
    os.makedirs(f"{DIR}/graphs", exist_ok=True)
    os.makedirs(f"{DIR}/samples", exist_ok=True)
    m = load()

    print("graphs:")
    g01_fid_kid_vs_steps(m)
    g02_fid_vs_n(m)
    g03_loss_and_fid(m)
    g04_progression(m)
    g05_real_vs_generated(m)
    g06_precision_recall(m)
    g07_class_hist(m)
    g08_confidence(m)
    g09_nn(m)
    g10_per_timestep(m)
    g11_schedule()

    print("samples:")
    sample_grids()


if __name__ == "__main__":
    main()
