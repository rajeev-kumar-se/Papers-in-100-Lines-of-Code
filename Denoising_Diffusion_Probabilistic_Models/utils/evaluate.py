# One evaluation run over the checkpoints we already trained.
#
#     python utils/evaluate.py --tag simple_unet
#
# produces eval/metrics.json + everything plots.py needs. no training here, so
# nothing in this file can improve the model -- it only describes it.
#
# the 6 metric families, and the objection each one answers:
#
#   FID            how far are the generated features from real features
#                  (+ 3 sampling seeds -> the metric's own noise floor)
#   FID vs n       how much of the score is small-sample bias
#   KID            same question without FID's Gaussian/unbiased assumptions
#   precision/recall  do samples look real (precision) and cover all 10 digits
#                    (recall) -- this is what catches mode collapse
#   classifier     are they recognizable as digits, and which ones
#   NN distance    are they novel, or copied from the training set
#
# plus two diagnostics that need no sampling: per-timestep denoising loss, and a
# check that the forward process q(x_t|x_0) has the mean/variance it should.
import gc
import json
import os
import re
import sys
import time

# this file lives in utils/; unet.py and diffusion_models.py are one level up,
# so put the project root on the path before the imports below run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# silences joblib's Windows core-count probe (it shells out to wmic, which is
# removed on newer Windows and prints a scary traceback for no reason)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import diffusion_models as dm_module
from diffusion_models import DiffusionModel, device, testX, testy, trainX, trainy
from pytorch_fid.inception import InceptionV3
from run_simple_unet import build_model

dm_module.tqdm = lambda iterable, **kwargs: iterable  # silence per-chunk bars

N_SAMPLES = 1000        # images per set; every metric below shares these
SEEDS = (0, 1, 2)       # one sample draw per seed -> FID variance
CKPTS = (25, 50, 75, 100)
DIR = "eval"


def upscale32(x):
    """28x28 MNIST -> the 32x32 nearest-neighbour grid the model was trained on.

    must match diffusion_models.sample_batch() exactly, otherwise real and
    generated live on different pixel grids and FID measures the resampling."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return F.interpolate(x, 32)


def quantize(x):
    """Round to the 8-bit grid a PNG round-trip would produce.

    pytorch-fid reads PNGs, so its inputs are quantized. mirroring that here is
    what makes our cached features reproduce calculate_fid_given_paths() exactly.
    """
    return (x.clamp(0, 1) * 255).round() / 255


# --------------------------------------------------------------------- sampling
def sample_set(pct, seed, n=N_SAMPLES):
    """Generate n images from checkpoint <pct>% with a fixed seed, cached on disk."""
    path = f"{DIR}/tensors/fake_{pct:03d}_seed{seed}.pt"
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")

    ckpt = torch.load(f"checkpoints/{TAG}_{pct:03d}.pt", map_location=device)
    model = build_model(MODEL).to(device)
    model.load_state_dict(ckpt["model"])
    diffusion = DiffusionModel(T, model).to(device)

    torch.manual_seed(seed)  # sampling is stochastic: seed it or the FID is noise
    model.eval()             # BatchNorm running stats, dropout off
    with torch.no_grad():
        out = [
            diffusion.sampling(n_samples=min(100, n - i))
            for i in range(0, n, 100)
        ]
    samples = torch.cat(out).cpu()
    torch.save(samples, path)
    del model, diffusion
    torch.cuda.empty_cache()
    return samples


# -------------------------------------------------------------------- features
def inception_features(images, batch=50):
    """
    2048-d pool3 features, exactly as pytorch-fid computes them:
        [0,1] -> repeat to RGB -> bilinear 299 -> 2x-1 -> block3 -> mean over H,W
    so FID from these features equals calculate_fid_given_paths on the same PNGs.
    """
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = quantize(images[i : i + batch]).float()
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            out = model(x.to(device))[0]
            feats.append(out.mean(dim=[2, 3]).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)


def cached_features(name, images=None):
    path = f"{DIR}/tensors/feat_{name}.npy"
    if os.path.exists(path):
        return np.load(path)
    assert images is not None, f"missing {path} and no images given"
    f = inception_features(images)
    np.save(path, f)
    return f


# ---------------------------------------------------------------------- metrics
# how many times each square-root path was used; recorded in metrics.json so the
# headline number can be attributed to the exact library-matching code.
_METHODS = {"sqrtm": 0, "eigh": 0}


def _sqrtm_psd_eigh(a):
    """
    Symmetric square root of a PSD matrix, via eigh.

    Fallback only. scipy's sqrtm (below) is the primary path because it is the
    same code pytorch-fid runs, so it reproduces the library bit-for-bit. This
    version exists for the cases sqrtm refuses: it raises LinAlgError -101 when
    the covariance is rank-deficient (n samples in 2048 dims always is), and it
    allocates several 2048x2048 complex128 arrays (64 MB each) at peak.

    For A = V diag(w) V^T symmetric PSD, sqrt(A) = V diag(sqrt(w)) V^T, which is
    real and needs one 32 MB eigendecomposition. Tiny negative eigenvalues from
    roundoff are clamped to 0, the standard treatment. Accurate to ~1e-8 in
    relative terms -- enough that its effect on FID is ~0.05, i.e. visible only
    in the fourth significant figure, which is why it is not the default.
    """
    w, v = np.linalg.eigh(a)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def _trace_covmean(s1, s2):
    """Tr(sqrt(S1 S2)), preferring the same sqrtm call pytorch-fid makes."""
    import warnings

    from scipy import linalg

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
        if np.iscomplexobj(cm):
            cm = cm.real
        if np.isfinite(cm).all():
            _METHODS["sqrtm"] += 1
            return float(np.trace(cm))
    except (np.linalg.LinAlgError, ValueError, MemoryError):
        pass

    # algebraic identity: Tr((S1 S2)^1/2) = Tr((S1^1/2 S2 S1^1/2)^1/2), and the
    # right-hand side only needs symmetric square roots, so it survives a
    # singular covariance.
    _METHODS["eigh"] += 1
    s1_half = _sqrtm_psd_eigh(s1)
    return float(np.trace(_sqrtm_psd_eigh(s1_half @ s2 @ s1_half)))


def frechet(mu1, s1, mu2, s2, eps=1e-6):
    """
        d^2 = ||mu1-mu2||^2 + Tr(S1 + S2 - 2 sqrt(S1 S2))

    identical to pytorch_fid.fid_score.calculate_frechet_distance on the primary
    path, with an algebraically equal fallback (see _sqrtm_psd_eigh) so the
    FID-vs-n curve has a value at n=100 instead of an exception.

    main() re-checks the equality by shelling out to the library's own CLI.
    """
    diff = mu1 - mu2
    tr = np.trace(s1) + np.trace(s2) - 2 * _trace_covmean(s1, s2)
    return float(diff.dot(diff) + tr)


def fid(f_real, f_fake):
    mu1, s1 = f_real.mean(0), np.cov(f_real, rowvar=False)
    mu2, s2 = f_fake.mean(0), np.cov(f_fake, rowvar=False)
    return frechet(mu1, s1, mu2, s2)


def kid(f_real, f_fake):
    """
    Unbiased squared MMD with the polynomial kernel (x.y/d + 1)^3, which is the
    KID default. unbiased = the diagonal is excluded, so identical distributions
    score ~0 instead of +something.
    """
    d = f_real.shape[1]

    def poly(a, b):
        return (a @ b.T / d + 1.0) ** 3

    k_rr = poly(f_real, f_real)
    k_ff = poly(f_fake, f_fake)
    k_rf = poly(f_real, f_fake)

    n, m = k_rr.shape[0], k_ff.shape[0]
    # exclude self-similarity: that is what makes the estimator unbiased
    s_rr = (k_rr.sum() - np.trace(k_rr)) / (n * (n - 1))
    s_ff = (k_ff.sum() - np.trace(k_ff)) / (m * (m - 1))
    return float(s_rr + s_ff - 2 * k_rf.mean())


def precision_recall(f_real, f_fake, k=3):
    """
    Kynkaanniemi et al. 2019, k-NN version.
      precision: fraction of fake samples inside a real k-NN ball
      recall:    fraction of real samples inside a fake k-NN ball
    real-vs-real should give ~(1, 1); noise should give ~(0, 0).
    """
    real_radii = NearestNeighbors(n_neighbors=k + 1).fit(f_real).kneighbors(f_real)[0][:, k]
    fake_radii = NearestNeighbors(n_neighbors=k + 1).fit(f_fake).kneighbors(f_fake)[0][:, k]

    d_fr, i_fr = NearestNeighbors(n_neighbors=1).fit(f_real).kneighbors(f_fake)
    precision = float(np.mean(d_fr[:, 0] <= real_radii[i_fr[:, 0]]))

    d_rf, i_rf = NearestNeighbors(n_neighbors=1).fit(f_fake).kneighbors(f_real)
    recall = float(np.mean(d_rf[:, 0] <= fake_radii[i_rf[:, 0]]))

    return precision, recall


# ------------------------------------------------------------- mnist classifier
class MnistCNN(nn.Module):
    """Small classifier, trained on the same 32x32 grid as the samples."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier():
    """Train a small MNIST CNN on the same 32x32 grid as the samples."""
    path = "checkpoints/mnist_cnn.pt"
    model = MnistCNN().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

    x = upscale32(trainX).to(device)  # [60000, 1, 32, 32]
    y = trainy.to(device)
    x_te = upscale32(testX[:2000]).to(device)
    y_te = testy[:2000].to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=512, shuffle=True
    )
    for _ in range(3):
        for xb, yb in dl:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(x_te).argmax(1) == y_te).float().mean().item()
    print(f"  classifier trained: {acc*100:.2f}% on 2000 held-out images")
    torch.save(model.state_dict(), path)
    return model


def classifier_stats(model, images, batch=500):
    """Predicted label distribution + confidence, for any set of 32x32 images."""
    labels, conf = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            logits = model(images[i : i + batch].to(device))
            p = logits.softmax(1)
            conf.append(p.max(1).values.cpu().numpy())
            labels.append(p.argmax(1).cpu().numpy())
    labels = np.concatenate(labels)
    conf = np.concatenate(conf)
    hist = np.bincount(labels, minlength=10) / len(labels)
    return {
        "hist": hist.tolist(),
        "conf_mean": float(conf.mean()),
        "conf_std": float(conf.std()),
        "conf_hist": np.histogram(conf, bins=20, range=(0, 1))[0].tolist(),
        "frac_confident_90": float((conf > 0.9).mean()),
    }


# ----------------------------------------------------------------- diagnostics
def per_timestep_loss(model, diffusion, n_t=50):
    """MSE at 50 points of the schedule: is the denoiser good at every noise level?"""
    model.eval()
    x0 = upscale32(testX[:64]).to(device)
    ts = torch.linspace(1, T, n_t).long().to(device)
    out = []
    with torch.no_grad():
        for t in ts:
            losses = []
            for _ in range(4):  # average a few noise draws per t
                eps = torch.randn_like(x0)
                ab = diffusion.alpha_bar[t - 1]
                xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
                pred = model(xt, torch.full((64,), t - 1, dtype=torch.long, device=device))
                losses.append(F.mse_loss(eps, pred).item())
            out.append(float(np.mean(losses)))
    return ts.cpu().tolist(), out


def moment_check(diffusion, n_draws=500):
    """
    Unit-test the forward process: for a FIXED x0, q(x_t|x_0) must satisfy
        E[x_t] = sqrt(abar_t) * x0        Var[x_t] = (1 - abar_t)
    So hold x0 still, draw many eps, and check the empirical moments over the
    draws. (Averaging over a batch of different x0 instead would mostly measure
    how varied MNIST is, not whether the formula is right.)

    The mean error is the verdict; the max is reported too but it is bounded
    below by Monte-Carlo noise (~1/sqrt(n_draws) per pixel, and we take a max
    over 8192 pixels), so it is only useful as a "nothing exploded" check.
    """
    x0 = upscale32(testX[:8]).to(device)  # 8 fixed images, [8,1,32,32]
    rows = []
    for t in (1, 100, 500, 1000):
        ab = diffusion.alpha_bar[t - 1]
        eps = torch.randn(n_draws, *x0.shape, device=device)
        xt = torch.sqrt(ab) * x0[None] + torch.sqrt(1 - ab) * eps

        mean_err = (xt.mean(0) - torch.sqrt(ab) * x0).abs()
        std_err = (xt.std(0) - torch.sqrt(1 - ab)).abs()
        rows.append(
            {
                "t": t,
                "sqrt_abar": float(torch.sqrt(ab)),
                "mean_abs_err": float(mean_err.mean()),
                "mean_max_abs_err": float(mean_err.max()),
                "std_abs_err": float(std_err.mean()),
                "std_max_abs_err": float(std_err.max()),
                "n_draws": n_draws,
                # what pure sampling noise alone would produce for the max
                "mc_scale": float(1.0 / np.sqrt(n_draws)),
            }
        )
    return rows


def nn_distance_to_train(query, chunk=100, ref_chunk=5000):
    """
    Min L2 distance from each query image to the 60k training images, in pixel
    space. Streams the reference set in chunks instead of materializing
    60000x1024 floats, which is what blew up the CPU allocator on the first try.
    """
    q = query.reshape(len(query), -1)
    best = torch.full((len(q),), float("inf"))

    for i in range(0, len(trainX), ref_chunk):
        r = upscale32(trainX[i : i + ref_chunk]).reshape(-1, 32 * 32).to(device)
        for j in range(0, len(q), chunk):
            d = torch.cdist(q[j : j + chunk].to(device), r).min(1).values.cpu()
            best[j : j + chunk] = torch.minimum(best[j : j + chunk], d)
        del r
        torch.cuda.empty_cache()

    return best.numpy()


# ------------------------------------------------------------------------ main
def main():
    global TAG, MODEL, T
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="simple_unet")
    p.add_argument("--model", default="simple")
    p.add_argument("--T", type=int, default=1000)
    args = p.parse_args()
    TAG, MODEL, T = args.tag, args.model, args.T

    for sub in ("", "/tensors", "/graphs", "/samples", "/logs"):
        os.makedirs(DIR + sub, exist_ok=True)

    start = time.perf_counter()
    results = {"config": vars(args), "n_samples": N_SAMPLES, "seeds": list(SEEDS)}
    print(f"== {TAG}: {N_SAMPLES} samples/set, seeds {SEEDS} ==")

    # ---- 1. reference sets ------------------------------------------------
    real_a = upscale32(testX[:N_SAMPLES])
    real_b = upscale32(testX[N_SAMPLES : 2 * N_SAMPLES])
    torch.save(real_a, f"{DIR}/tensors/real_a.pt")
    torch.save(real_b, f"{DIR}/tensors/real_b.pt")
    noise = torch.rand(N_SAMPLES, 1, 32, 32)

    f_real_a = cached_features("real_a", real_a)
    f_real_b = cached_features("real_b", real_b)
    f_noise = cached_features("noise", noise)

    # ---- 2. sample every checkpoint --------------------------------------
    print("-- sampling --")
    fake = {}
    for pct in CKPTS:
        seeds = SEEDS if pct == 100 else (0,)  # variance only where it matters
        for s in seeds:
            t0 = time.perf_counter()
            fake[(pct, s)] = sample_set(pct, s)
            print(f"  {pct:3d}% seed {s}: {time.perf_counter()-t0:.0f}s")

    # ---- 3. features ------------------------------------------------------
    print("-- inception features --")
    feats = {}
    for (pct, s), imgs in fake.items():
        feats[(pct, s)] = cached_features(f"fake_{pct:03d}_seed{s}", imgs)

    main_set = feats[(100, 0)]

    # ---- 4. metrics -------------------------------------------------------
    print("-- metrics --")
    results["fid"] = {}
    for (pct, s), f in feats.items():
        results["fid"][f"{pct}_{s}"] = fid(f_real_a, f)

    results["fid_stats"] = {
        "mean": float(np.mean([results["fid"][f"100_{s}"] for s in SEEDS])),
        "std": float(np.std([results["fid"][f"100_{s}"] for s in SEEDS])),
        "values": [results["fid"][f"100_{s}"] for s in SEEDS],
    }
    results["fid_real_vs_real"] = fid(f_real_a, f_real_b)
    results["fid_noise"] = fid(f_real_a, f_noise)

    results["kid"] = {f"{pct}_{s}": kid(f_real_a, f) for (pct, s), f in feats.items()}
    results["kid_real_vs_real"] = kid(f_real_a, f_real_b)
    results["kid_noise"] = kid(f_real_a, f_noise)

    pr = precision_recall(f_real_a, main_set)
    results["precision_recall"] = {
        "precision": pr[0],
        "recall": pr[1],
        "real_vs_real": precision_recall(f_real_a, f_real_b),
        "noise": precision_recall(f_real_a, f_noise),
    }

    # ---- 5. FID vs n (bias curve) ----------------------------------------
    results["fid_vs_n"] = {}
    for n in (100, 250, 500, 1000):
        results["fid_vs_n"][n] = {
            "model": fid(f_real_a[:n], main_set[:n]),
            "floor": fid(f_real_a[:n], f_real_b[:n]),
            "noise": fid(f_real_a[:n], f_noise[:n]),
        }

    # ---- 7. classifier ---------------------------------------------------
    # after every numpy metric on purpose: fitting this puts a CUDA context and a
    # 60k-image tensor in memory, and with no pagefile on this box that is enough
    # to make the 2048x2048 eigendecompositions above fail to allocate.
    print("-- classifier --")
    clf = train_classifier()
    results["classifier"] = {
        "generated": classifier_stats(clf, fake[(100, 0)]),
        "real": classifier_stats(clf, real_a),
    }
    del clf
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 8. memorization -------------------------------------------------
    print("-- nn distance --")
    results["nn_distance"] = {
        "generated": nn_distance_to_train(fake[(100, 0)]).tolist(),
        "real": nn_distance_to_train(real_a).tolist(),
    }
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 9. diagnostics --------------------------------------------------
    print("-- diagnostics --")
    ckpt = torch.load(f"checkpoints/{TAG}_100.pt", map_location=device)
    model = build_model(MODEL).to(device)
    model.load_state_dict(ckpt["model"])
    diffusion = DiffusionModel(T, model).to(device)
    ts, losses = per_timestep_loss(model, diffusion)
    results["per_timestep"] = {"t": ts, "mse": losses}
    results["moment_check"] = moment_check(diffusion)

    results["wall_clock_s"] = time.perf_counter() - start
    results["fid_sqrt_method"] = dict(_METHODS)  # exact sqrtm vs eigh fallback
    del model, diffusion
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 10. cross-check: our cached features vs the library's own CLI -----
    # same 1000 images: one number from our npz features, one from `python -m
    # pytorch_fid` reading the PNGs off disk in its own process. if these agree,
    # the FID is not a bug. a subprocess also keeps the library's memory use out
    # of ours, which is why the first attempt died in the dataloader collate.
    print("-- fid cross-check vs pytorch-fid (subprocess) --")
    import shutil
    import subprocess
    import sys

    from run_simple_unet import dump_pngs

    gc.collect()
    torch.cuda.empty_cache()

    tmp = f"{DIR}/_xcheck"
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        dump_pngs(fake[(100, 0)], f"{tmp}/fake", "fake")
        dump_pngs(real_a, f"{tmp}/real", "real")

        def run_cli(dev):
            out = subprocess.run(
                [
                    sys.executable, "-m", "pytorch_fid",
                    f"{tmp}/fake", f"{tmp}/real",
                    "--device", dev,
                    "--batch-size", "50",
                    "--dims", "2048",
                    "--num-workers", "0",
                ],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            match = re.search(r"FID:\s*([0-9.]+)", out.stdout)
            if match is None:
                raise RuntimeError(out.stderr[-800:])
            return float(match.group(1))

        # same device as our own feature extraction: this is the apples-to-apples
        # check. cuDNN and the CPU convolution kernels differ in the last float32
        # bits, and FID's covariance arithmetic turns that into ~0.05, so comparing
        # a GPU feature set against a CPU run would measure kernel noise, not our code.
        our_dev = "cuda" if torch.cuda.is_available() else "cpu"
        lib_fid = run_cli(our_dev)
        ours = results["fid"]["100_0"]
        results["fid_crosscheck"] = {
            "device": our_dev,
            "library": lib_fid,
            "ours": ours,
            "abs_diff": abs(lib_fid - ours),
        }
        print(f"  {our_dev}: library {lib_fid:.4f}   ours {ours:.4f}"
              f"   diff {abs(lib_fid - ours):.2e}")

        # cpu run, for the record: shows the size of the device effect
        if our_dev != "cpu":
            lib_cpu = run_cli("cpu")
            results["fid_crosscheck"]["library_cpu"] = lib_cpu
            results["fid_crosscheck"]["cpu_vs_gpu"] = abs(lib_cpu - lib_fid)
            print(f"  cpu: library {lib_cpu:.4f}   (cpu-vs-gpu kernel difference "
                  f"{abs(lib_cpu - lib_fid):.2e})")
    except Exception as exc:  # a hiccup here must not lose the metrics
        results["fid_crosscheck"] = {"error": repr(exc)}
        print(f"  cross-check skipped: {exc!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    with open(f"{DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- 9. print the summary -------------------------------------------
    print("\n=========== SUMMARY ===========")
    print(f"FID (n={N_SAMPLES})      {results['fid_stats']['mean']:.2f} "
          f"+/- {results['fid_stats']['std']:.2f}   (seeds {SEEDS})")
    print(f"FID floor (real/real) {results['fid_real_vs_real']:.2f}")
    print(f"FID noise             {results['fid_noise']:.2f}")
    print(f"KID                    {results['kid']['100_0']:.5f}  (floor {results['kid_real_vs_real']:.5f})")
    print(f"precision / recall     {pr[0]:.3f} / {pr[1]:.3f}")
    print(f"classifier conf        {results['classifier']['generated']['conf_mean']:.3f} "
          f"(real {results['classifier']['real']['conf_mean']:.3f})")
    print(f"fid sqrt path          {results['fid_sqrt_method']}  (sqrtm = same call as pytorch-fid)")
    cc = results.get("fid_crosscheck", {})
    if "abs_diff" in cc:
        print(f"fid cross-check        {cc['abs_diff']:.2e} vs pytorch-fid CLI")
    print(f"wall clock             {results['wall_clock_s']/60:.1f} min")


if __name__ == "__main__":
    main()
