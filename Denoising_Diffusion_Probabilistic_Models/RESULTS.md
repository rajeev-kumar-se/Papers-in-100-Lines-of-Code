# DDPM on MNIST — results

One line: a 2.21M-parameter convolutional U-Net, trained for 10,000 steps in 11.4 minutes,
reaches **FID 45.57 ± 0.26** (n=1000) against a **12.19 real-vs-real floor** measured the same way.
Every number here is in `eval/metrics.json`; every graph is in `eval/graphs/`.

---

## 1. What was trained

| | |
|---|---|
| Model | `SimpleUNet` (`unet_simple.py`) — conv3×3 + BatchNorm + ReLU, MaxPool down, `Upsample`+skip-concat up, timestep as a per-channel bias ([layer diagram](RESULTS.html#arch)) |
| Parameters | 2,212,481 |
| Not used | attention, residual blocks, transformer, zero-init — none of it |
| Diffusion | T = 1000, linear β from 1e-4 to 0.02, ε-prediction, plain MSE |
| Optimizer | Adam, lr 3e-4, batch 256, 10,000 steps |
| Data | MNIST 28×28 nearest-upscaled to 32×32, scaled to [0,1] |
| Time | 11.4 min (68 ms/step at the end), single GPU |

Loss fell 0.0688 → 0.0151 (4.5×). FID fell 86.3 → 45.3 (2×), which is the point of
[`g03_loss_and_fid.png`](eval/graphs/g03_loss_and_fid.png): **training loss is not a quality metric.**

## 2. How one image is made

```
x = torch.randn(1, 1, 32, 32)              # start from pure noise  (t = 1000)
for t in range(1000, 0, -1):               # 1000 reverse steps
    eps = model(x, t)                      # predict the noise in x
    x = 1/sqrt(α_t) * (x - (1-α_t)/sqrt(1-ᾱ_t) * eps) + sqrt(β_t) * z
return x.clamp(0, 1)                       # a 32×32 digit
```

The schedule is in [`g11_noise_schedule.png`](eval/graphs/g11_noise_schedule.png): signal √ᾱ dies
from 1 to 0.006 as noise √(1−ᾱ) rises to 1.

## 3. Headline metrics (n = 1000 images per set)

| metric | SimpleUNet | real-vs-real floor | uniform noise |
|---|---|---|---|
| **FID** | **45.57 ± 0.26** (45.31 / 45.92 / 45.48, 3 sampling seeds) | 12.19 | 413.18 |
| **KID** (unbiased) | **0.0391** | −0.0001 | 0.629 |
| precision / recall (k=3) | 0.113 / 0.135 | 0.573 / 0.640 | 0.000 / 0.000 |
| classifier confidence | 0.927 | 0.969 | — |

The classifier is a small CNN trained on the same 32×32 grid: **97.85%** on 2,000 held-out images.

Read this table as: FID 45.6 is *not* "close to real", it is *better than noise by 9× and worse
than real data by 3.7×*. Same for KID. Precision/recall must be compared to their own floor —
0.573/0.640 is what real data scores against itself at n=1000 in 2048 dimensions, not 1.0.

## 4. Did it improve with training?

| step | 2,500 | 5,000 | 7,500 | 10,000 |
|---|---|---|---|---|
| FID | 86.30 | 60.97 | 50.74 | **45.31** |
| KID | 0.0936 | 0.0571 | 0.0462 | **0.0391** |

[`g01_fid_kid_vs_steps.png`](eval/graphs/g01_fid_kid_vs_steps.png) ·
[`g04_progression.png`](eval/graphs/g04_progression.png) (same seed across all four checkpoints)

## 5. The number that makes FID honest

FID is **biased at small n**. Measuring real-vs-real with the same n exposes the bias:

| n | model | floor | noise |
|---|---|---|---|
| 100 | 86.67 | 53.64 | 430.9 |
| 250 | 61.84 | 32.56 | 422.5 |
| 500 | 54.95 | 20.36 | 415.5 |
| 1000 | **45.31** | **12.19** | 413.2 |

At n=100 the floor alone (53.64) is *higher* than the model's score at n=1000 (45.31).
Any FID quoted without its n and a matched floor is uninterpretable.
[`g02_fid_vs_n.png`](eval/graphs/g02_fid_vs_n.png)

## 6. Why I trust the FID (five checks)

1. **Same preprocessing as `pytorch-fid`**: PIL→RGB→ToTensor→bilinear 299→`2x−1`→block3→mean over H,W.
   Inception weights are the library's own cached checkpoint.
2. **Same square root as the library**: 20/20 FID computations used `scipy.linalg.sqrtm` on
   `S1·S2`, exactly the call `calculate_frechet_distance` makes. Zero fallbacks
   (`fid_sqrt_method: {sqrtm: 20, eigh: 0}`).
3. **Cross-check against the library's CLI, on the same device**:
   ours **45.3064** vs `python -m pytorch_fid` **45.3064**, difference **4.7e-06**.
4. **Device matters, and that is measured too**: the same CLI on CPU gives 45.2507, i.e. a
   **0.056** gap from cuDNN vs CPU float32 convolution. Comparing a GPU feature set to a CPU run
   measures kernel noise, not methodology — so the check runs both.
5. **A real bug was found this way and fixed**: PNG export originally used
   `.astype(np.uint8)`, which truncates instead of rounding, darkening every pixel by up to
   1/255. That alone moved FID by **~1.8**, more than 2,500 training steps were worth. The
   cross-check is what caught it; the code now rounds.

## 7. Failure modes I tested

No catastrophic failure (no memorized copies, no collapsed classes), but two real weaknesses show up.

| check | result | graph |
|---|---|---|
| **Memorization** — L2 to nearest training image | generated median 4.78 vs real 5.01; **min 1.47**, no mass near 0 → no copies | [g09](eval/graphs/g09_nn_distance.png) |
| **Mode collapse** — predicted class histogram | all 10 digits present, but **1s over-generated: 26.6% vs 12.7%** → imbalance, not collapse | [g07](eval/graphs/g07_class_hist.png) |
| **Coverage** — precision/recall | 0.113 / 0.135: low precision *and* low recall vs floor, i.e. blurry and not fully covering | [g06](eval/graphs/g06_precision_recall.png) |
| **Classifier confidence** | 0.927 vs real 0.969 → recognizable digits, slightly less crisp | [g08](eval/graphs/g08_confidence_hist.png) |
| **Forward process unit test** — fixed x₀, 500 noise draws | mean and std of x_t match √ᾱ·x₀ and √(1−ᾱ) to ≤0.036 per pixel, within Monte-Carlo noise (0.045) | — |
| **Denoising loss vs t** | 0.368 at t=1 → 0.0008 at t=1000 | [g10](eval/graphs/g10_per_timestep_loss.png) |

The per-timestep curve is high at low t on purpose: at t=1 almost every pixel of an MNIST image is
background (x₀≈0), so x_t ≈ 0.01·ε and the noise is recoverable; at t=1000 x_t ≈ ε, so predicting
zero error is trivial. It is not a sign of undertraining.

## 8. Where everything is

```
eval/
  metrics.json          all numbers (source of truth)
  graphs/g01..g11.png   11 graphs
  samples/grid_{025,050,075,100}.png   81-sample grids per checkpoint
checkpoints/
  simple_unet_{025,050,075,100}.pt   the four checkpoints; _100 is the best one
  manifest.json         what each file is, and why they are not resumable
logs/
  simple_unet.log       training log (loss every 500 steps)
  evaluate.log          evaluation log, incl. the FID cross-check

diffusion_models.py     the paper implementation: forward process, training loop, sampling
unet.py                 the U-Net the paper describes (35.7M params)
unet_simple.py          the small U-Net that was actually trained (2.2M params)
denoisers.py            four alternative noise predictors, shape-checked only
utils/                  the scripts around them; nothing here is imported by the files above
  run_simple_unet.py    train + sample + FID + calibration
  evaluate.py           the full evaluation run -> eval/metrics.json
  plots.py              eval/metrics.json -> eval/graphs + eval/samples
  checkpoints_meta.py   rewrites the four checkpoints into a resumable format
```

Run the scripts from the project root (`python utils/evaluate.py`); they read and write the
relative paths above. Sample and feature caches are not kept, so `python utils/evaluate.py`
regenerates them in ~5 minutes.

Key images: [final samples](eval/samples/grid_100.png) ·
[real vs generated](eval/graphs/g05_real_vs_generated.png) ·
[progression](eval/graphs/g04_progression.png)

## 9. What I do **not** claim

- Not "U-Net with attention", not a transformer, not latent diffusion, not DDIM. Plain ε-prediction DDPM.
- 45.6 is **not comparable** to published MNIST FIDs: different splits, n, resampling and preprocessing.
  It is only comparable to the 12.19 floor measured here, under this protocol.
- Only one architecture was trained. `unet.py` (35.7M), `TinyUNet`, `DiT`, `ConvNet`, `MLP` have
  parameter counts and timings in `denoisers.py`, **no trained metrics**.
- 3 sampling seeds only at the final checkpoint; 25/50/75% use seed 0.
- No Inception Score, no FID-vs-published-baseline table, no resume-from-checkpoint (the original
  run did not save optimizer state; the field is null rather than fake).

## 10. Interview version, in four sentences

> I trained a 2.2M-parameter conv U-Net as the DDPM noise predictor on MNIST, 10k steps in 11 minutes.
> FID is 45.6 ± 0.3 at n=1000 against a 12.2 real-vs-real floor measured with the same code and sample count,
> and it improves monotonically from 86.3 at 2.5k steps.
> I verified the metric against `pytorch-fid`'s own CLI to 5e-06, which caught a real uint8-truncation bug in my
> PNG export, and I report the floor because FID at n=100 is 53.6 even for real data.
> The model covers all ten digits with no memorized copies, but it over-produces 1s and its precision/recall
> are both below the floor — so it is a correct, modest model, not a state-of-the-art one.
