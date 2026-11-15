# Evaluation Plan — SimpleUNet / DDPM on MNIST

Budget: **one GPU run** (evaluation only, no retraining) + one short CPU/GPU job to
train the MNIST classifier. Target wall clock: **~15 min**.

Everything below runs on the checkpoints already in `checkpoints/`. Nothing here
changes the model, so no metric here can "improve" the FID — the goal is to
**describe** quality correctly, which is what the interview will probe.

---

## 0. Inventory — what we actually have

| artifact | status |
|---|---|
| `checkpoints/simple_unet_{025,050,075,100}.pt` | model weights only, no optimizer state, no step, no RNG → **cannot resume** |
| FID at 4 checkpoints | 85.74 / 57.16 / 49.50 / 43.01 (n=1000) |
| FID floors | real-vs-real 12.19, uniform noise 414.15 |
| images | 9x9 grids at each milestone, real-vs-generated, progression, loss curve |
| `denoisers.py` (4 alt architectures) | shape/timing/param verified only — **no metrics** |
| reference `unet.py` (35.7M) | never trained — **no metrics** |

So the project currently rests on one metric with one sample of uncertainty.

---

## 1. Metrics to compute

Grouped by which objection each one kills. All computed from **one set of 1000
generated images** (fixed seed, saved to disk as tensors) so every metric shares
the same samples.

### A. FID, done properly
| metric | definition | why |
|---|---|---|
| FID @ n=1000 (re-run) | 2048-d pool3, Gaussian fit, Frechet distance | reproduce 43.01 on the saved tensor set so all metrics below are consistent |
| FID mean ± std over 3 sample draws | 3×1000 fresh samples, different seeds | **the metric's own noise floor.** Without it, "43.01 vs 43.47" looks like a finding instead of sampling noise |
| FID vs n ∈ {100, 250, 500, 1000} | same samples, prefix subsets | plots the **bias curve** — the single most persuasive slide for "why I report the floor" |

### B. Unbiased distance — KID
| metric | definition | why |
|---|---|---|
| KID (MMD², polynomial kernel, unbiased) | squared MMD between Inception features | FID assumes Gaussians and is biased at small n; KID is neither. If KID and FID rank the checkpoints the same way, the FID claim survives scrutiny |
| KID @ n=1000 + KID at each milestone | same features | second curve for the "is it still improving" question |

### C. Quality vs coverage — Precision / Recall
| metric | definition | why |
|---|---|---|
| Precision (fidelity) | fraction of generated features inside the real manifold (k-NN ball, k=3) | "do the samples look like digits" |
| Recall (coverage) | fraction of real features inside the generated manifold | "did it produce all 10 classes or just the easy ones" |

This is the pair that exposes **mode collapse**, which FID structurally cannot
isolate. Plot as a single point against two reference points: real-vs-real
(≈ upper right) and uniform-noise (bottom left).

### D. Semantic check — MNIST classifier
| metric | definition | why |
|---|---|---|
| generated-digit accuracy / confidence | train a small CNN (~99.2% on MNIST test), classify the 1000 generated samples | "are these actually digits" — an objective version of eyeballing |
| predicted-label histogram | counts over 0–9 vs real histogram | proves all 10 classes appear, in roughly the right proportion |
| max-prob confidence histogram | real vs generated | generated confidence lower than real = blurred/ambiguous strokes; equal = indistinguishable structurally |

Note: with **unconditional** generation there is no "correct" digit per sample,
so this measures *realism*, not correctness. Say it that way in the interview.

### E. Memorization check
| metric | definition | why |
|---|---|---|
| nearest-neighbour L2 distance, generated → train set | per sample, min over 60k train images, pixel space | rules out "it just copied MNIST" before it's raised |
| same for real test → train | reference distribution | the two histograms should overlap, not collapse |

### F. Model diagnostics (no sampling needed)
| metric | definition | why |
|---|---|---|
| per-timestep MSE | loss evaluated at t = 1..1000 on the test set | is the denoiser uniformly good, or only on easy (low-t) steps? a spike at high t explains blurry global structure |
| forward-process sanity test | empirical mean/var of q(x_t|x_0) vs the closed form | unit-test that sqrt(ā)·x0 + sqrt(1-ā)·ε is implemented correctly — cheap, and shows you verified the math |

### G. Explicitly NOT doing (and why)
- **Inception Score** — redundant with FID, worse behaved, no longer recommended.
- **Per-class conditioning metrics** — model is unconditional.
- **FID vs published numbers** — invalid: 32×32 grayscale upscaled to 299, different protocol.
- **Head-to-head vs `unet.py` / `denoisers.py`** — would need ~1-2h of training, out of budget. State it as a known gap.

---

## 2. Graphs to build

| # | graph | question it answers |
|---|---|---|
| 1 | **FID & KID vs steps** (two lines, shaded floor band at 12.19, error bars from the 3-seed draws) | "is the model still improving, and by how much" |
| 2 | **FID vs n** (100→1000, model + floor on same axes) | "how much of your FID is metric bias" |
| 3 | **Loss curve** (log y, milestone markers, right axis = FID at those points) | ties the training objective to the sample metric |
| 4 | **Progression grid** (rows = milestones, FID in the row label) — regenerate cleanly | the one image people actually look at |
| 5 | **Real vs generated** (top real, bottom generated, same 32×32 grid) | qualitative side-by-side |
| 6 | **Precision–Recall scatter** (model point, real-vs-real, noise) | quality vs coverage on one plot |
| 7 | **Class histogram** real vs generated (grouped bars, 10 bins) | "are all digits generated" |
| 8 | **Confidence histogram** real vs generated (overlaid) | stroke sharpness, objectively |
| 9 | **NN-distance histograms** generated→train vs test→train | memorization vs novelty |
| 10 | **Per-timestep loss curve** (MSE vs t, log y) | where in the noise schedule the model is weak |
| 11 | **Noise schedule plot** (β_t, ᾱ_t, and sqrt(ᾱ), sqrt(1-ᾱ) on twin axes) | pure explanation, no GPU; belongs in the README |

Graphs 1, 2, 6, 7 are the four that earn their place in a resume/portfolio.
11 is documentation.

---

## 3. Checkpoint strategy

Current scheme is the weak part: 4 files, weights only, no resume.

**Format** — every checkpoint becomes a dict:
```
{
  "model": state_dict,
  "optimizer": state_dict,          # needed to actually resume
  "step": int,
  "args": vars(args),
  "rng": {torch, cuda},
  "torch_version", "git_commit",
}
```

**Schedule** — save every 1,000 steps, but only keep:
- a **rolling window of the last 3** (`latest.pt` always the newest),
- **milestones** at 25/50/75/100% (kept permanently, they make the curves),
- **best by FID** (`best.pt`), written only after evaluation, never overwritten by training.

**Storage math** (so rotation is a choice, not an accident):
| model | weights | + Adam states | per ckpt | 10 kept |
|---|---|---|---|---|
| SimpleUNet 2.2M | 8.8 MB | ~27 MB | 27 MB | 270 MB |
| reference UNet 35.7M | 143 MB | ~430 MB | 430 MB | 4.3 GB |

So: keep everything for SimpleUNet, rotate to a window of 3 for the 35.7M model.

**Do not overwrite the final checkpoint until the metrics are computed.** Order of
operations: evaluate → copy `simple_unet_100.pt` → `simple_unet_best.pt` → prune.

---

## 4. What we missed (beyond metrics)

1. **EMA weights** — still the biggest missing thing, worth 20-40% of FID at zero training cost. Not fixable in an evaluation run; list it as "known next step", don't claim it.
2. **Seed hygiene** — sampling is unseeded, so FID moves ±0.5 between draws. Fix: seed before every sample call, report mean ± std.
3. **Config is not saved** — the only record of hyperparameters is a `.txt` line. Needs a JSON dump per run.
4. **No resume** — see checkpoint section.
5. **Protocol mismatch, undocumented** — 28×28 MNIST nearest-upscaled to 32×32 while FID upscales both to 299. This must be stated wherever the number appears, or the number is misleading.
6. **`requirements.txt` is stale** — pins keras 2.8 / torch 1.12, but the code now uses torchvision + torch 2.x + pytorch-fid. Anyone cloning this cannot run it.
7. **README results section is empty** — has a `## Results` heading and nothing under it.
8. **`denoisers.py` has no numbers** — either train one of them or don't mention them as results. Currently they're architecture + cost only.
9. **No test suite** — the forward-process moment check (F1) is a 20-line pytest that would let you say "I unit-tested the diffusion math".
10. **No DDIM comparison** — 1000-step sampling costs ~90 s per 1000 images. A 50-step DDIM sampler is ~20 lines and gives a "quality vs sampling cost" trade-off curve. Nice-to-have, not this run.

---

## 5. Execution order for the one run

| step | what | GPU time |
|---|---|---|
| 1 | Train MNIST classifier (3 epochs, tiny CNN), save `classifier.pt` | ~40 s |
| 2 | Sample 1000 images × 3 seeds from `simple_unet_100.pt`, save `.pt` + PNGs | ~4.5 min |
| 3 | Sample 1000 × 1 seed from `_050` and `_075` (for the curves) | ~3 min |
| 4 | Extract Inception features once per set, cache to `features/*.npz` | ~2 min |
| 5 | FID (all n), KID, precision/recall from cached features — no GPU | ~1 min |
| 6 | Classifier + NN-distance + per-timestep loss on the saved tensors | ~2 min |
| 7 | Build all 11 graphs, write `RESULTS.md` | ~1 min |

Step 4 is the trick: **extract features once, derive every metric from them.**

---

## 6. Deliverables

- `evaluate.py` — one script, one run, writes `results/metrics.json`
- `plots.py` — all graphs into `Imgs/eval/`
- `RESULTS.md` — the table you paste into the README and the resume
- `results/metrics.json` — machine-readable, so future runs append instead of overwrite
- `checkpoints/` — resumed-capable format, best.pt preserved
