# Project Journal — Diffusion Inpainting of Ocean Current Vector Fields

---

## Workspace Structure (as of June 10, 2026)

```
diffusionInpaintingVectorFields - try 2/
├── data.pickle               ← dataset (shared, stays at root)
├── dataset.py                ← dataset loader
├── paths.py                  ← shared robot path generators (biased_walk_path, random_walk_path)
├── plot_utils.py             ← shared quiver-plot helper
├── journal.md
├── README.md
├── train_loss_comparison.sh  ← server script to compare loss functions
├── vector_field_sample0.png  ← sample visualisation
├── DDPM/                     ← DDPM model, training, inference
│   ├── requirements.txt
│   ├── model/
│   │   ├── diffusion.py      ← DDPM class (all loss modes)
│   │   ├── model.py          ← UNet architecture
│   │   └── train.py          ← training script (--loss, --weights)
│   ├── testing/
│   │   ├── repaint/
│   │   │   └── repaint_infer.py  ← RePaint inference engine
│   │   ├── visualize_infer.py    ← single-sample inference + figure
│   │   └── batch_infer.py        ← batch evaluation
│   ├── models/
│   │   ├── best_model.pt                              ← eps-only baseline (cosine schedule)
│   │   ├── model_ddpm_curl_div_gaussian_cosine1.pt
│   │   ├── model_ddpm_eps_gaussian_cosine1.pt
│   │   ├── model_ddpm_okubo_weiss_gaussian_cosine1.pt
│   │   ├── model_ddpm_spectral_gaussian_cosine1.pt
│   │   └── model_ddpm_wasserstein_gaussian_cosine1.pt
│   └── results/
│       ├── inference_result.png
│       └── best_model_results/
│           └── result_01.png … result_10.png
├── GP Baseline/              ← GP inpainting baseline
│   ├── gp_infer.py
│   ├── visualize_infer.py
│   ├── batch_infer.py
│   ├── requirements.txt
│   └── GP_results/
│       ├── gp_result.png
│       └── result_01.png … result_10.png
├── Model Parameters/
│   ├── Loss Function/        ← loss function ablation study
│   │   ├── loss_functions.py         ← curl_div, spectral, okubo_weiss, wasserstein, stream_fn, strain_rate
│   │   ├── batch_eval_loss.py        ← Run 1 eval: 7 models × 10 val samples, 7 metrics
│   │   ├── batch_eval_v2.py          ← Run 2 eval: 6 models × 10 val samples, 3 metrics
│   │   ├── batch_eval_loss_visual.py ← (legacy visual batch eval)
│   │   ├── loss_eval_visual.py       ← variance eval: 7 models × 50 samples × 10 seeds
│   │   ├── train_loss_comparison.sh  ← server training script
│   │   ├── train_new_losses.sh       ← server script for additional loss variants
│   │   ├── models/
│   │   │   ├── model_ddpm_curl_div_gaussian_cosine.pt
│   │   │   ├── model_ddpm_eps_gaussian_cosine.pt
│   │   │   ├── model_ddpm_okubo_weiss_gaussian_cosine.pt
│   │   │   ├── model_ddpm_spectral_gaussian_cosine.pt
│   │   │   ├── model_ddpm_strain_rate_gaussian_cosine.pt
│   │   │   ├── model_ddpm_stream_function_gaussian_cosine.pt
│   │   │   └── model_ddpm_wasserstein_gaussian_cosine.pt
│   │   └── results/
│   │       ├── eval_loss/                  ← Run 1 results (seed 42, 7 models)
│   │       │   ├── loss_eval_visual_report.txt
│   │       │   ├── loss_eval_visual_results.csv
│   │       │   ├── loss_eval_visual_summary.csv
│   │       │   └── sample_0168.png … sample_1683.png  (10 PNGs)
│   │       ├── eval_run2/                  ← Run 2 results (seed 99, 6 models)
│   │       │   ├── report.txt
│   │       │   ├── results.csv
│   │       │   ├── summary.csv
│   │       │   └── sample_0348.png … sample_1908.png  (10 PNGs)
│   │       └── eval_variance_loss/         ← variance/mean PNGs (16/50 downloaded)
│   │           └── sample_124.png … sample_831.png  (16 PNGs)
│   └── NoiseSchedule/        ← noise schedule ablation study
│       ├── diffusion.py          ← DDPM with pluggable schedule (linear/cosine/quadratic/sigmoid)
│       ├── repaint_model.py      ← Repaint UNet (same arch as DDPM/model/model.py)
│       ├── repaint_infer.py      ← RePaint inference (biased walk + repaint loop)
│       ├── train_repaint.py      ← training script
│       ├── test_repaint.py       ← test-set evaluation + 2×2 visualisation
│       ├── batch_repaint.py      ← batch evaluation (10 val samples)
│       ├── batch_eval.py         ← batch evaluation script
│       ├── batch_eval_visual.py  ← visual batch evaluation
│       ├── multi_run_grid.py     ← multi-run grid comparison
│       ├── run_repaint.sh        ← bash: train all four schedules
│       ├── requirements.txt
│       ├── checkpoints/
│       │   ├── checkpoints_repaint_cosine/    ← cosine_out.txt + best_model_cosine.pt
│       │   ├── checkpoints_repaint_linear/    ← linear_out.txt + best_model_linear.pt
│       │   ├── checkpoints_repaint_quadratic/ ← quadratic_out.txt + best_model_quadratic.pt
│       │   └── checkpoints_repaint_sigmoid/   ← sigmoid_out.txt + best_model_sigmoid.pt
│       └── results/
│           ├── model_comparison.txt
│           ├── cosine_multi_run_grid.png
│           ├── learning_curve.png
│           ├── eval__method_compare/        ← per-schedule batch eval comparison
│           │   ├── eval_results.csv
│           │   ├── eval_summary.csv
│           │   ├── model_comparison.txt
│           │   ├── cosine/   result_01.png … result_10.png
│           │   ├── linear/   result_01.png … result_10.png
│           │   ├── quadratic/ result_01.png … result_10.png
│           │   └── sigmoid/  result_01.png … result_10.png
│           ├── model_cosine_results/     result_01.png … result_10.png
│           ├── model_geometric_results/  result_01.png … result_10.png + batch_geometric.log
│           ├── model_linear_results/     result_01.png … result_10.png
│           ├── model_quadratic_results/  result_01.png … result_10.png
│           └── model_sigmoid_results/    result_01.png … result_10.png
└── Voronoi/                  ← Voronoi tessellation baseline (Fukami et al. 2021)
    ├── model/
    │   ├── voronoi_model.py      ← VoronoiNet (VoronoiLayer + U-Net)
    │   └── train_voronoi.py      ← training script (--sensor_mode scattered|walk)
    ├── testing/
    │   ├── test_voronoi.py       ← test-set evaluation + 2×2 visualisation
    │   ├── batch_voronoi.py      ← batch evaluation with scattered sensors
    │   └── batch_voronoi_walk.py ← batch evaluation with biased-walk sensors
    ├── models/
    │   ├── checkpoints_voronoi_scattered/  ← voronoi_train.log (no saved .pt)
    │   └── checkpoints_voronoi_walk/       ← best_model_walk.pt + voronoi_walk_train.log
    └── results/
        ├── model_comparison.txt
        ├── scattered_out.txt
        ├── walk_out.txt
        ├── voronoi_test_result.png
        ├── model_scattered_test_scattered/  result_01.png … result_10.png
        ├── model_scattered_test_walk/       result_01.png … result_10.png
        ├── model_walk_test_scattered/       result_01.png … result_10.png
        └── model_walk_test_walk/            result_01.png … result_10.png
```

**Note on paths:** All scripts use `sys.path.insert` to add the workspace root to the Python
path so that `dataset.py` and `paths.py` are importable from any subdirectory. Run all
scripts from the workspace root, e.g.:
```
python "Voronoi/model/train_voronoi.py" --pickle data.pickle
python "Model Parameters/NoiseSchedule/train_repaint.py" --schedule cosine --pickle data.pickle
```

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


---

## Structural Loss Integration (June 4–5, 2026)

### Motivation

The base DDPM loss is pure epsilon-MSE on ocean pixels — it has no explicit incentive
to reproduce the rotational/divergent structure of the vector field.  Structural
regularisation losses were developed and ultimately consolidated directly into the
main DDPM class.

### Development history

1. **Topo DDPM** (June 4) — subclass adding curl + divergence loss on $\hat{x}_0$.
   Trained with λ = 0.1, 0.01, 0.0002.  λ = 0.1 caused extreme instability (val loss
   swinging 0.006 → 113); λ = 0.0002 stable.  Kept as proof-of-concept but superseded.

2. **Multi-Loss DDPM** (June 4) — generalised subclass with `--loss` flag covering
   five modes: `eps`, `curl_div`, `spectral`, `okubo_weiss`, `wasserstein`.

3. **Consolidation** (June 5) — all auxiliary loss functions moved to
   `Model Parameters/loss_functions.py` as standalone functions and the base `DDPM`
   class in `DDPM/model/diffusion.py` updated to import them.  The separate
   Multi-Loss DDPM is now a thin backward-compatible alias; `Topo DDPM/` removed.

### Unified loss API (as of June 5)

Training any combination of losses is done directly via `DDPM/model/train.py`:

```bash
python model/train.py --loss spectral --pickle ../data.pickle
python model/train.py --loss spectral okubo_weiss --weights 0.0002 0.001 --pickle ../data.pickle
```

`training_loss` always returns `(total, eps_loss, indiv)` where `indiv` is a dict
mapping each active loss name to its unweighted value.

### Loss modes

All auxiliary terms operate on $\hat{x}_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon) / \sqrt{\bar\alpha_t}$,
masked to ocean pixels.

$$L_\text{total} = L_\text{eps} + \sum_i \lambda_i \cdot L_i$$

| `--loss` | Term $L_i$ | Default $\lambda_i$ |
|---|---|---|
| `curl_div` | $\text{MSE}([\omega, D]_{\hat{x}_0},\, [\omega, D]_{x_0})$ | 0.0002 |
| `spectral` | $\text{MSE}(|\text{rfft2}(\hat{x}_0)|,\, |\text{rfft2}(x_0)|)$ | 0.0002 |
| `okubo_weiss` | $\text{MSE}(W_{\hat{x}_0},\, W_{x_0})$, $W = s_n^2 + s_s^2 - \omega^2$ | 0.001 |
| `wasserstein` | Sinkhorn–Wasserstein between $|\omega|$ point clouds | 1.0 |

### Next Steps
- Train with `--loss spectral` and compare val RMSE against eps-only baseline (0.1146 ± 0.022).
- Train with `--loss okubo_weiss` and `--loss spectral okubo_weiss` combinations.
- Consider Wasserstein loss once simpler modes are benchmarked.

---

## Noise Schedule Ablation (June 5, 2026)

### Motivation

The base DDPM uses a **cosine** noise schedule throughout.  Four schedules are now compared
to understand whether the choice of β_t affects inpainting quality under RePaint:

| Schedule | β_t formula |
|---|---|
| `linear`    | linearly from β_min to β_max |
| `cosine`    | Nichol & Dhariwal 2021 cosine schedule (default) |
| `quadratic` | quadratic from β_min to β_max |
| `sigmoid`   | sigmoid-shaped from β_min to β_max |

### Module layout (`Model Parameters/NoiseSchedule/`)

Each schedule trains an **identical Repaint UNet** (same architecture as `DDPM/model/model.py`,
`base_ch=64`, `time_dim=256`, T=1000) via `train_repaint.py`.

| File | Purpose |
|---|---|
| `diffusion.py` | `DDPM` class with `beta_schedule` argument |
| `repaint_model.py` | `Repaint` UNet (drop-in for DDPM UNet) |
| `repaint_infer.py` | biased walk path + RePaint inference loop |
| `train_repaint.py` | training (100 epochs, batch 32, AdamW + cosine LR) |
| `test_repaint.py` | test-set evaluation + 2×2 single-sample visualisation |
| `batch_repaint.py` | batch evaluation (10 val samples per schedule) |
| `run_repaint.sh` | trains all four schedules sequentially |

Checkpoints save to `checkpoints/checkpoints_repaint_{schedule}/best_model_{schedule}.pt`.
Results save to `results/model_{schedule}_results/`.

### Usage
```bash
# train all four (from workspace root)
bash "Model Parameters/NoiseSchedule/run_repaint.sh"

# evaluate one schedule
python3 "Model Parameters/NoiseSchedule/batch_repaint.py" --schedule cosine --pickle data.pickle
```

### Results
TBD — training not yet run on server.

---

## Voronoi Tessellation Baseline (June 5, 2026)

### Motivation

Implements **VoronoiNet** (Fukami et al., *Nature Machine Intelligence* 2021).  Instead of
the diffusion-based RePaint pipeline, sparse sensor readings are mapped onto the full
spatial grid via Voronoi tessellation (nearest-neighbour assignment), and a U-Net
encoder-decoder reconstructs the full field directly.

This is a **deterministic feed-forward model** (no iterative reverse diffusion), so inference
is orders of magnitude faster than RePaint but lacks a generative prior over the full field.

### Architecture (`Voronoi/model/voronoi_model.py`)

| Component | Description |
|---|---|
| `VoronoiLayer` | Maps K sensor (pos, value) pairs → (C+1, H, W) structured grid via nearest-neighbour; extra channel = binary sensor-presence mask |
| `VoronoiUNet` | Encoder-decoder UNet, (C+1, H, W) → (C, H, W), no time conditioning |
| `VoronoiNet` | Convenience wrapper combining both stages; exposes `forward(x0, n_sensors, land_mask)` |

### Sensor modes

| Mode | Description |
|---|---|
| `scattered` | K random ocean cells chosen independently per batch (default K=50) |
| `walk` | K cells from a 150-step biased robot walk (same path as RePaint) |

Training uses a different sensor pattern per batch/epoch to prevent overfitting to a
fixed sensor layout.

### Module layout (`Voronoi/`)

```
model/
  voronoi_model.py        ← VoronoiNet architecture
  train_voronoi.py        ← training (--sensor_mode scattered|walk)
testing/
  test_voronoi.py         ← test-set evaluation + 2×2 visualisation
  batch_voronoi.py        ← 10-run batch eval, scattered sensors
  batch_voronoi_walk.py   ← 10-run batch eval, walk sensors
models/
  checkpoints_voronoi_scattered/  ← best_model_scattered.pt
  checkpoints_voronoi_walk/       ← best_model_walk.pt
results/
  model_scattered_test_scattered/
  model_scattered_test_walk/
  model_walk_test_scattered/
  model_walk_test_walk/
```

### Usage
```bash
# train (from workspace root)
python "Voronoi/model/train_voronoi.py" --pickle data.pickle --epochs 100
python "Voronoi/model/train_voronoi.py" --sensor_mode walk --pickle data.pickle \
    --save_dir Voronoi/models/checkpoints_voronoi_walk

# evaluate
python "Voronoi/testing/test_voronoi.py" --pickle data.pickle
python "Voronoi/testing/batch_voronoi.py" --pickle data.pickle
python "Voronoi/testing/batch_voronoi_walk.py" --pickle data.pickle
```

### Results
TBD — training not yet run on server.

---

## Repository Path Fixes (June 5, 2026)

All Python scripts were updated to correctly add the **workspace root** to `sys.path`
so that `dataset.py` and `paths.py` are importable from any subdirectory.

| File | Old `sys.path` insert | Fixed insert |
|---|---|---|
| `Voronoi/model/train_voronoi.py` | `".."` → `Voronoi/` | `"../.."`  → root |
| `Voronoi/testing/test_voronoi.py` | `".."` → `Voronoi/` | `"../.."`  → root |
| `Voronoi/testing/batch_voronoi.py` | `".."` → `Voronoi/` | `"../.."`  → root |
| `Voronoi/testing/batch_voronoi_walk.py` | `".."` → `Voronoi/` | `"../.."`  → root |
| `Model Parameters/NoiseSchedule/*.py` | `".."` → `Model Parameters/` | `"../.."`  → root |

Additional fixes applied:
- `train_voronoi.py`: removed wrong `DDPM`-relative `sys.path` insert; replaced
  `from repaint_infer import biased_walk_path` → `from paths import biased_walk_path`
  (the canonical location at root).  Changed `from Voronoi.model.voronoi_model import VoronoiNet`
  → `from voronoi_model import VoronoiNet` (same-directory import).
- `batch_voronoi_walk.py`: same `biased_walk_path` fix.
- Default checkpoint paths updated to use `Voronoi/models/` (matching workspace layout).
- Default output dirs updated: `Voronoi/results/model_scattered_test_scattered/`,
  `Voronoi/results/model_walk_test_walk/`.
- `NoiseSchedule/train_repaint.py`, `test_repaint.py`, `batch_repaint.py`: checkpoint
  defaults updated to include the `checkpoints/` subdirectory; output dirs updated to
  include `results/` subdirectory.
- `run_repaint.sh`: updated script path to `"Model Parameters/NoiseSchedule/train_repaint.py"`,
  `cd` updated to `/root/ocean_diffusion`, log files routed to checkpoint subdirectories.

---

## Loss Function Ablation — Evaluation Runs (June 5–10, 2026)

### New evaluation scripts

Three new scripts were added to `Model Parameters/Loss Function/`:

| Script | Purpose |
|---|---|
| `batch_eval_loss.py` | Original batch eval: 7 models × 10 val samples, 3 metrics |
| `batch_eval_v2.py` | Run 2 eval: 6 models (no spectral) × 10 samples, 3 metrics, saves 3×3 grid PNGs |
| `loss_eval_visual.py` | Visual eval: 7 models × 50 samples × 10 diffusion seeds per sample; saves per-sample 4×4 PNGs with per-model **mean reconstruction** + **variance heatmap** |
| `train_new_losses.sh` | Server script to train remaining loss-function variants |

`loss_eval_visual.py` is the most comprehensive evaluation: for each val sample it runs
every model 10 times (same robot path, different diffusion noise seeds) and plots the
**pixelwise variance** across those runs as a heatmap, alongside the mean reconstruction.
Outputs go to `results/eval_variance_loss/sample_{idx:03d}.png`.

### Run 1 — seed 42, 7 models, 10 val samples (`eval_loss/`)

Samples: [168, 174, 185, 395, 848, 859, 1281, 1368, 1514, 1683]

| Model | RMSE (field) | Spectral RMSE | Wasserstein |
|---|---|---|---|
| **strain_rate** | **0.0674** | **21.28** | 0.0751 |
| **curl_div** | **0.0699** | **21.98** | 0.0785 |
| eps | 0.0837 | 31.29 | 0.0805 |
| stream_function | 0.0847 | 29.97 | 0.0735 |
| wasserstein | 0.0966 | 39.74 | 0.0929 |
| okubo_weiss | 0.0973 | 24.55 | **0.0405** |
| spectral | 0.1175 | 71.56 | 0.1055 |

- `strain_rate` wins 5/7 metrics; `curl_div` wins the remaining 2 and is very close on spectral.
- `spectral` is the worst performer overall and was excluded from Run 2.
- `okubo_weiss` wins Wasserstein distance by a large margin.

### Run 2 — seed 99, 6 models (no spectral), 10 val samples (`eval_run2/`)

Samples: [348, 990, 1003, 1107, 1208, 1483, 1848, 1874, 1895, 1908]

| Model | RMSE (field) | Spectral RMSE | Wasserstein |
|---|---|---|---|
| **eps** | **0.0962** | **21.42** | **0.0463** |
| curl_div | 0.1029 | 30.00 | 0.0495 |
| wasserstein | 0.1059 | 41.52 | 0.0623 |
| stream_function | 0.1106 | 38.13 | 0.0716 |
| okubo_weiss | 0.1296 | 48.46 | 0.0661 |
| strain_rate | 0.1666 | 62.10 | 0.0691 |

- **eps wins all 3 metrics on this sample set** — a complete reversal of Run 1.
- `strain_rate` collapses to worst performer (RMSE 0.067 → 0.167); its Run 1 dominance was
  sample-set-dependent.
- `curl_div` remains consistently 2nd best across both runs — the most robust structural loss.

### Cross-run comparison summary

| Model | Run1 RMSE | Run2 RMSE | Run1 Std | Run2 Std |
|---|---|---|---|---|
| eps | 0.0837 | 0.0962 | 0.0383 | 0.0425 |
| curl_div | 0.0699 | 0.1029 | 0.0177 | 0.0923 |
| okubo_weiss | 0.0973 | 0.1296 | 0.1078 | 0.0993 |
| wasserstein | 0.0966 | 0.1059 | 0.0244 | 0.0672 |
| stream_function | 0.0847 | 0.1106 | 0.0285 | 0.0409 |
| strain_rate | 0.0674 | 0.1666 | 0.0248 | 0.1212 |

**Key takeaway:** No single structural loss consistently outperforms eps-only across both
sample sets. The eps model is the **most stable** (moderate mean, low variance in both
runs). `curl_div` has low variance in Run 1 but degrades in Run 2. `strain_rate` shows
high sensitivity to which samples are evaluated.

### Visual / variance evaluation (`eval_variance_loss/`, June 9–10, 2026)

`loss_eval_visual.py` was launched on the server to produce per-sample 4×4 grid plots
(ground truth | robot path | model avg | model variance heatmap) for 50 val samples ×
7 models × 10 diffusion seeds. Results are being downloaded incrementally from the
server (`182.224.239.168`). As of June 10, **16 of 50 sample PNGs** are available locally
(`sample_124.png` … `sample_831.png`).

This evaluation will reveal which spatial regions have high stochastic variance under
each model — i.e., where the model is "uncertain" independent of the robot path.


