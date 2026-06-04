# Project Journal — Diffusion Inpainting of Ocean Current Vector Fields

---

## Workspace Structure (as of June 4, 2026)

```
diffusionInpaintingVectorFields - try 2/
├── data.pickle               ← dataset (shared, stays at root)
├── dataset.py                ← dataset loader
├── paths.py                  ← shared robot path generators
├── plot_utils.py             ← shared quiver-plot helper
├── journal.md
├── vector_field_sample0.png
└── Basic DDPM/               ← DDPM model/inference code
    ├── model.py
    ├── diffusion.py
    ├── train.py
    ├── repaint_infer.py
    ├── visualize_infer.py
    ├── batch_infer.py
    ├── requirements.txt
    ├── inference_result.png
    ├── checkpoints/
    │   └── best_model.pt
    └── batch_results/
        └── result_01.png … result_10.png
└── GP Baseline/              ← GP inpainting baseline
    ├── gp_infer.py
    ├── visualize_infer.py
    ├── batch_infer.py
    ├── requirements.txt
    └── batch_results/
        └── result_01.png … result_10.png
```

**Note on paths:** The Python scripts default to `data.pickle` (same directory).
On the remote server (`~/ocean_diffusion/`) everything is flat so defaults work.
If running locally from `Basic DDPM/`, pass `--pickle ../data.pickle`.

---

## Project Goal

A slow autonomous underwater robot traverses a small random-walk path through a coastal
ocean domain, collecting local current measurements at only the cells it visits (~3% of
the ocean area).  The objective is to reconstruct the **full 2D vector field** of ocean
currents from those sparse observations, using a diffusion-model-based inpainting
approach inspired by the **RePaint** paper (Lugmayr et al., CVPR 2022).

---

## What We Know About the Data (`data.pickle`)

### Structure
- A Python `list` of **3 NumPy arrays** representing train / validation / test splits.
- Split sizes: **9180 train**, **1965 val**, **1965 test** samples (~82 / 9 / 9 %).

### Array Layout
Each array has shape **(94, 44, C=2, N)** where:

| Dimension | Meaning |
|---|---|
| 94 | **X axis** (east-west, longitude-like) — runs 0–93 |
| 44 | **Y axis** (north-south, latitude-like) — runs 0–43 |
| C=2  | Vector field components: channel 0 = **u** (east-west), channel 1 = **v** (north-south) |
| N    | Number of timestep snapshots (9180 / 1965) |

After transposing for PyTorch: **(N, 2, 94, 44)**.

**Display convention:** for plotting, arrays are transposed to shape (44, 94) so that
rows correspond to Y (0–43) and columns to X (0–93), giving a landscape-orientation image.

### Values
| Split | Min | Max | Mean | Std | NaN % |
|---|---|---|---|---|---|
| Train | -0.8973 | 1.0860 | -0.051 | 0.116 | 8.8% |
| Val   | -0.6952 | 0.6305 | -0.049 | 0.119 | 8.8% |
| Test  | -0.6980 | 0.8431 | -0.047 | 0.115 | 8.8% |

- Values appear **normalised** (roughly centred near zero, std ~0.12).
- The mean is slightly negative in u, suggesting a weak mean westward background current.

### Land Mask
- **8.8% of all grid cells are NaN** — these are land pixels.
- The NaN pattern is **identical across all timesteps and all splits**, confirming it is a
  fixed spatial land/ocean mask.
- After transposing for display (X=0–93, Y=0–43), the land forms a coastal peninsula
  extending from the top edge down to approximately Y=24, centred around X=5–51.
- **3787 ocean cells** out of 94×44=4136 total grid cells.

### Physical Interpretation
- The domain is a **coastal ocean region** (likely a model output or reanalysis product).
- The flow shows a dominant large-scale current (mostly eastward/rightward in the upper
  half) with a recirculation / boundary-layer feature near the coast.
- Speeds reach up to ~0.9 normalised units; typical ocean speeds are ~0.05–0.15.

---

## What Has Been Built

### 1. `dataset.py` — `OceanCurrentDataset`
- Loads the pickle and exposes PyTorch `Dataset` interface.
- Transposes to (N, 2, H, W), replaces NaN → 0, stores the boolean `land_mask`.
- Supports `split=0/1/2` (train/val/test).

### 2. `model.py` — `UNet`
- 2D UNet with **sinusoidal timestep embeddings** and **ResNet-style blocks**.
- Input/output: (B, 2, 94, 44) — two-channel vector field.
- Internal padding to (96, 48) allows clean factor-of-2 downsampling through 5 levels.
- **14,957,958 parameters** (~15M).
- Time conditioning via `t → sinusoidal embedding → MLP → added to every ResBlock`.

### 3. `diffusion.py` — `DDPM`
- **Cosine noise schedule** (Nichol & Dhariwal 2021) with T=1000 steps.
- Implements:
  - `q_sample` — forward noising x_0 → x_t
  - `training_loss` — epsilon-prediction MSE, masked to ocean pixels only
  - `p_sample_step` — single DDPM reverse step p(x_{t-1} | x_t)
  - `q_sample_from_prev` — one forward step for RePaint resampling

### 4. `repaint_infer.py` — RePaint inference engine
- `biased_walk_path(land_mask, n_steps, seed, straight_bias=0.75)` — simulates a robot
  path that is a continuous connected walk on ocean cells.  Three design constraints:
  1. **Directional persistence** — ~75% weight to continuing straight, ~12.5% each
     perpendicular turn, ~0.4% to reversing.
  2. **Novelty bonus** — each candidate cell's weight is scaled by `1/(1+visit_count)`,
     so the robot strongly avoids backtracking and spreads across the domain.
  3. **Land avoidance** — only ocean neighbours are considered at each step.
  With 150 steps this typically covers ~130–270 unique ocean cells (3–7% of 3787).
  **Path evolution:**
  - v1: Pure random walk (equal probability all 4 directions)
  - v2: Straight-line scan (deterministic per-column snap to nearest ocean cell)
  - v3 (current): Biased walk with directional persistence + novelty exploration
- `repaint(model, diffusion, x0_known, path_mask, land_mask, r)` — runs the full
  RePaint algorithm:
  1. Start from Gaussian noise (land=0).
  2. For each t from T→0, repeated r times:
     - Model reverse step for unknown cells.
     - Forward-diffuse true observations to t−1 for known (path) cells.
     - Merge: known cells ← observed, unknown cells ← model prediction.
     - If not last iteration: go forward one step and repeat (resampling).
  3. Return final x_0 prediction.

### 5. `train.py` — Training script
- AdamW optimiser, cosine LR decay.
- Gradient clipping at 1.0.
- Saves `best_model.pt` (lowest val loss) and rolling checkpoints every 10 epochs.
- CLI flags: `--epochs`, `--batch`, `--lr`, `--base_ch`, `--T`, etc.

### 6. `visualize_infer.py` — Inference + visualisation
- Loads checkpoint, picks a val sample, runs `biased_walk_path`, runs `repaint`.
- Arrays are transposed before plotting so X=0–93 runs horizontally and Y=0–43 vertically.
- Quiver plots use `scale=12`, `cmap='cool'`, white ocean, black land.
- Produces a **2×2 figure** (18×10 inches):
  1. Ground truth quiver plot
  2. Robot path overlay
  3. Reconstructed field
  4. Pointwise speed error heatmap
- Default: `--path_steps 150`, `--resample 10`, `--seed 42`

### 7. `batch_infer.py` — Batch evaluation
- Runs N inference evaluations on the val set with different path seeds.
- Seeds: `seed = i*7 + 1` for i=0..N-1.
- Saves individual 2×2 PNGs to `batch_results/result_01.png` … `result_NN.png`.
- Prints per-run RMSE, mean, and std at the end.

---

## Training Run (June 2–3, 2026)

| Setting | Value |
|---|---|
| Server | vast.ai — NVIDIA TITAN X (Pascal), 12 GB VRAM |
| Epochs | 200 |
| Batch size | 32 |
| Learning rate | 2e-4 (cosine decay) |
| T (diffusion steps) | 1000 |
| Best val loss | **0.00327** (epoch 155) |
| Final train loss | ~0.0035 |

Loss dropped from ~0.05 (epoch 1) to ~0.0035 (epoch 200), converging around epoch 100.
Best checkpoint saved locally at `checkpoints/best_model.pt` (57 MB).

---

## Inference Results (June 3–4, 2026)

### Single-run results (`visualize_infer.py`, val sample 0, seed 1)

| Date | Path type | Steps | Cells covered | RMSE | Notes |
|---|---|---|---|---|---|
| Jun 3 | Random walk | 300 | ~113 (3.0%) | 0.1361 | Baseline |
| Jun 3 | Straight scan | 94 | 94 (2.5%) | 0.0930 | One cell per X column |
| Jun 3 | Biased walk v1 | 300 | 236 (6.2%) | 0.1091 | Persistence, no novelty |
| Jun 3 | Biased walk v2 | 300 | 269 (7.1%) | 0.1105 | + novelty bonus |
| Jun 4 | Biased walk v2 | 150 | 143 (3.8%) | **0.0801** | Final config |

### Batch results (`batch_infer.py`, 10 val samples, seeds 1/8/15/…/64)

Comparing old random-walk path vs current biased-walk path (both 150 steps default,
old runs used 300 steps):

| Run | Val idx | Seed | Old RMSE | New RMSE |
|---|---|---|---|---|
| 1 | 0 | 1 | 0.1598 | 0.1576 |
| 2 | 1 | 8 | 0.1452 | 0.1015 |
| 3 | 2 | 15 | 0.1855 | 0.1082 |
| 4 | 3 | 22 | 0.1323 | 0.0940 |
| 5 | 4 | 29 | 0.1054 | 0.1391 |
| 6 | 5 | 36 | 0.0802 | 0.1086 |
| 7 | 6 | 43 | 0.1417 | 0.0923 |
| 8 | 7 | 50 | 0.0863 | 0.1125 |
| 9 | 8 | 57 | 0.0962 | 0.0927 |
| 10 | 9 | 64 | **0.7320** | **0.1393** |
| **Mean** | | | **0.1865** | **0.1146** |
| **Std** | | | **0.1847** | **0.0217** |

The most significant improvement was run 10 (RMSE 0.73→0.14): the old random walk got
trapped near the top land boundary covering almost no open ocean; the biased walk with
novelty penalty spreads across the domain regardless of seed.  The standard deviation
collapsed from 0.18 to 0.02, showing the new path is highly consistent.

### Observations
- The model successfully recovers the large-scale flow structure from ~3–7% of cells.
- Highest errors in regions far from the robot path and near the coast.
- The biased walk with novelty exploration is substantially more robust than pure random
  walk because it guarantees good domain coverage independent of starting position.
- Directional persistence produces a realistic robot-like trajectory (smooth curves rather
  than dense clusters).

### Known Limitations / Next Steps
- RePaint results are stochastic — running multiple samples and averaging would give a
  more stable reconstruction and allow uncertainty quantification.
- RePaint with r=10, T=1000 takes several minutes even on the TITAN X; DDIM or fewer
  timesteps could speed this up dramatically.
- No quantitative comparison against baselines (e.g. kriging, nearest-neighbour).
- The model was trained unconditionally; a conditional model (path as input channel) might
  produce sharper reconstructions in the observed region.

---

## GP Inpainting Baseline (June 4, 2026)

A Gaussian Process baseline was implemented in `GP Baseline/` to compare against the
DDPM approach.  Two independent GPs (one for u, one for v) are fit to the robot path
observations and used to predict the full ocean field.

### Setup
- **Kernel:** Matérn ν = 2.5 + WhiteKernel
- **Inputs:** (row, col) normalised to [0, 1]
- **Hyperparameters:** optimised per snapshot via log-marginal-likelihood (L-BFGS-B,
  2 random restarts)
- **Training data used at inference:** only the ~130–150 robot path observations
- **No training phase** — the GP is purely non-parametric; it uses the 9180 training
  snapshots only indirectly if hyperparameters were pre-fit (currently fit per-snapshot)

### Single-run results (`visualize_infer.py`, test sample 0, seed 42)

| Path steps | Cells covered | RMSE | Notes |
|---|---|---|---|
| 300 | 274 (7.2%) | 0.1208 | — |
| 150 | 140 (3.7%) | 0.1145 | Slight improvement — luckier path placement |

### Batch results (`batch_infer.py`, 10 val samples, path steps = 150)

| Run | Val sample | Path cells | RMSE |
|---|---|---|---|
| 01 | 0 | 142 (3.7%) | 0.1173 |
| 02 | 1 | 140 (3.7%) | 0.1356 |
| 03 | 2 | 144 (3.8%) | 0.1257 |
| 04 | 3 | 130 (3.4%) | 0.1692 |
| 05 | 4 | 134 (3.5%) | 0.1340 |
| 06 | 5 | 124 (3.3%) | 0.2006 |
| 07 | 6 | 142 (3.7%) | 0.1970 |
| 08 | 7 | 143 (3.8%) | 0.3171 |
| 09 | 8 | 123 (3.2%) | 0.3474 |
| 10 | 9 | 148 (3.9%) | 0.2670 |
| **Mean** | | | **0.2011 ± 0.0786** |
| **Min / Max** | | | **0.1173 / 0.3474** |

### GP vs DDPM comparison (both using 150-step biased walk, val set)

| Method | Mean RMSE | Std | Notes |
|---|---|---|---|
| DDPM (RePaint, r=10) | **0.1146** | 0.0217 | 10 runs on val set |
| GP (Matérn 2.5) | 0.2011 | 0.0786 | 10 runs on val set |

The GP mean RMSE is ~75% higher than the DDPM and its variance is ~3.6× larger.

### Qualitative observations

**Where the GP does well:**
- In the upper-right quadrant of the domain (X = 30–93, Y = 15–43) where currents are
  relatively smooth and uniform — the Matérn kernel's stationarity assumption is a
  reasonable approximation there.
- The posterior uncertainty (std) is well-calibrated: it is low along the robot path
  corridor and increases smoothly with distance, correctly flagging the unobserved
  bottom-left region as uncertain.

**Where the GP fails:**
- Runs 8–9 have RMSE 0.32–0.35, nearly 3× worse than the best runs.  In these cases the
  robot path happened to miss the high-speed coastal current jet in the bottom-left corner
  entirely.  With no nearby observations the GP simply regresses toward the global mean,
  producing a near-zero prediction where there are strong fast currents.
- The DDPM does significantly better in those cases because it has internalised the
  typical spatial structure of ocean currents from 9180 training snapshots and can
  plausibly extrapolate structure even into completely unobserved regions.
- The GP reconstruction is visibly over-smoothed compared to the ground truth: it cannot
  reproduce the sharp gradient at the current boundary near Y=5–10.

**High variance across runs:**
- The DDPM std across 10 runs was 0.022; the GP std is 0.079 — 3.6× more variable.
  The GP performance is strongly path-dependent: a lucky path that crosses the high-speed
  jet gives RMSE ~0.12, while a path that misses it entirely gives RMSE ~0.35.

**Convergence warnings:**
- Several runs triggered a scikit-learn `ConvergenceWarning` indicating the optimised
  noise level hit its lower bound (1e-7).  This suggests the GP is fitting the
  observations nearly exactly (near-zero noise), which makes sense since the ocean
  current values are deterministic model output with no measurement noise.  The bound
  could be lowered further but is unlikely to significantly affect the mean prediction.

### Takeaways
- The GP is a reasonable baseline for ~3.5% path coverage when the path happens to
  cross the main flow features, but it is brittle — a single unlucky path placement
  causes a large degradation that the DDPM is immune to.
- The DDPM's learned prior acts as a strong regulariser that the GP lacks.
- The GP has one key advantage: it provides **analytic uncertainty estimates** that are
  well-calibrated spatially, which could be useful for adaptive path planning (directing
  the robot toward high-uncertainty regions on subsequent passes).
- A natural extension would be to use the GP uncertainty map to seed the next robot
  path, iteratively reducing uncertainty — something the DDPM cannot easily do without
  ensemble sampling.

---

## Topology-Regularised DDPM Variants (June 4, 2026)

### Motivation

The Basic DDPM loss is pure epsilon-MSE on ocean pixels — it has no explicit incentive
to reproduce the rotational/divergent structure of the vector field.  Two new model
families were created to explore structural regularisation losses.

---

### `Topo DDPM/` — Curl + Divergence Regularisation

A subclass `TopoDDPM(DDPM)` adds a topological auxiliary loss computed on the
reconstructed denoised field $\hat{x}_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon) / \sqrt{\bar\alpha_t}$:

$$L = \underbrace{\text{MSE}(\hat\epsilon, \epsilon)}_{\text{epsilon loss}} + \lambda \cdot \underbrace{\text{MSE}(\text{topo}(\hat{x}_0),\, \text{topo}(x_0))}_{\text{topo loss}}$$

where $\text{topo}(\cdot) = [\text{curl}, \text{divergence}]$ computed via central-difference
finite differences:

$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}, \qquad D = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

Both terms are masked to ocean pixels only.  `training_loss` returns
`(total, eps_loss, topo_loss)` so all three are logged separately.

#### Training runs

| Run | λ (topo_weight) | Outcome |
|---|---|---|
| 1 | 0.1 | Topo loss highly unstable — val swings from 0.006 to 113 epoch-to-epoch.  λ too large. |
| 2 | 0.01 | More stable but still noisy; total loss dominated by topo term. |
| 3 | 0.0002 | Current run — eps and topo well-balanced; training looks healthy. |

The instability at large λ is expected: the topo loss operates on second-order features
of the reconstructed $\hat{x}_0$, which is itself a noisy estimate early in training and
at high timesteps.

**File layout:**
```
Topo DDPM/
├── diffusion.py    ← TopoDDPM(DDPM) subclass; loads base DDPM via importlib
├── train.py        ← --topo_weight arg, logs eps + topo separately
└── _patch_server.py
```

---

### `Multi-Loss DDPM/` — Switchable Structural Loss

A generalised `MultiLossDDPM(DDPM)` subclass with a `--loss` flag to select from five
auxiliary loss modes.  All modes share the same epsilon-MSE backbone and the same
$\hat{x}_0$ reconstruction; only the auxiliary term changes.

| `--loss` | Auxiliary term | Notes |
|---|---|---|
| `eps` | None | Identical to Basic DDPM |
| `curl_div` | MSE on [curl, divergence] of $\hat{x}_0$ | Same as Topo DDPM |
| `spectral` | MSE on rfft2 magnitude of both velocity components | Land-masked before FFT so land contributes zero energy |
| `okubo_weiss` | MSE on $W = s_n^2 + s_s^2 - \omega^2$ | Eddy boundaries: $W<0$ rotation-dominated, $W>0$ strain-dominated |
| `wasserstein` | Sinkhorn–Wasserstein distance between vorticity point clouds | Penalises eddies in wrong location; requires `geomloss` |

#### Wasserstein mode details

The vorticity field $|\omega(x,y)|$ is normalised to a probability distribution over the
$(row, col)$ grid and represented as a weighted point cloud.  The Sinkhorn approximation
(entropic regularisation, `geomloss.SamplesLoss`) is used for a differentiable
approximation.  Key parameters:

- `--aux_weight`: should be ~1.0 for Wasserstein (distance is in coordinate units
  ~0–1, vs. squared magnitudes for the other modes which need small λ)
- `--sinkhorn_blur`: entropic regularisation radius (default 0.05); smaller = more
  accurate but slower

**File layout:**
```
Multi-Loss DDPM/
├── diffusion.py      ← MultiLossDDPM(DDPM); LOSS_MODES tuple exported
├── train.py          ← --loss, --aux_weight, --sinkhorn_blur args
└── _patch_server.py  ← also pip-installs geomloss if needed
```

#### Recommended starting weights

| Mode | Suggested λ | Rationale |
|---|---|---|
| `curl_div` | 0.0002 | Curl/div values are large squared; needs small λ |
| `spectral` | 0.0002 | FFT magnitudes are similar scale to curl/div |
| `okubo_weiss` | 0.001 | OW squares derivatives twice; even larger scale |
| `wasserstein` | 1.0 | Sinkhorn distance is in [0, 1] coordinate units |

---

### Discussion: Structural Loss Comparison

| Loss | Differentiable | Captures displacement | Compute overhead | Extra deps |
|---|---|---|---|---|
| curl_div | Yes | No | Negligible | None |
| spectral | Yes | Partial (multi-scale) | Negligible | None |
| okubo_weiss | Yes | No (local) | Negligible | None |
| wasserstein | Via Sinkhorn | Yes — key advantage | ~5–10× slower | geomloss |

The **spectral loss** is the most likely to improve inpainting quality for the least
cost: it penalises the model for concentrating energy at the wrong spatial scales,
which is the main failure mode of pure MSE (over-smoothing).

The **Wasserstein loss** is theoretically the most compelling for this problem —
an eddy predicted 5 grid cells from its true location gets a large MSE loss but a small
Wasserstein loss — but the extra compute and dependency make it harder to validate
quickly.

---

### Next Steps

- Complete the `topo_weight=0.0002` Topo DDPM run (100 epochs) and compare val eps-loss
  against the Basic DDPM baseline.
- Run Multi-Loss DDPM with `--loss spectral --aux_weight 0.0002` and compare.
- After training, run `batch_infer.py` on both new models and compare RMSE with the
  Basic DDPM (mean 0.1146 ± 0.022).
- Consider whether `okubo_weiss` or `wasserstein` is worth the extra complexity based
  on spectral results.
