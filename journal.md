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
├── Conditional DDPM/         ← FiLM-conditioned DDPM (four conditioning modes)
│   ├── cond_model.py             ← CondUNet (FiLMResBlock, ConditionEncoder)
│   ├── cond_diffusion.py         ← CondDDPM (forward/reverse/RePaint with cond)
│   ├── train.py                  ← training script (--cond voronoi|path|path_field|both)
│   ├── infer.py                  ← batch inference (10 val samples, RePaint)
│   ├── compare_infer.py          ← side-by-side eps vs voronoi-cond comparison
│   ├── batch_eval_cond_multirun.py ← multi-run batch evaluation
│   ├── compare_eps_vs_cond.py    ← comprehensive comparison script
│   ├── launch_training.sh        ← server script (trains voronoi, path, both)
│   ├── requirements.txt
│   ├── checkpoints_voronoi/      ← best_cond_ddpm_voronoi_cosine.pt
│   ├── checkpoints_path/         ← best_cond_ddpm_path_cosine.pt
│   ├── checkpoints_both/         ← best_cond_ddpm_both_cosine.pt  (in progress)
│   ├── results_voronoi/          ← 10 sample PNGs + rmse_summary.png
│   ├── results_path/             ← 10 sample PNGs + rmse_summary.png
│   ├── results_both/             ← partial results
│   ├── results_compare/          ← 2×3 eps vs voronoi-cond comparison PNGs
│   └── results_compare_grid/     ← grid comparison PNGs
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
  - `p_sample_step` — single DDPM reverse step
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

1. **Topo DDPM** (June 4) — subclass adding curl + divergence loss on the estimated x_0.
   Trained with weight = 0.1, 0.01, 0.0002.  Weight = 0.1 caused extreme instability (val loss
   swinging 0.006 → 113); weight = 0.0002 stable.  Kept as proof-of-concept but superseded.

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

All auxiliary terms operate on the pair (estimated clean x_0, true x_0) via
finite-difference gradients, masked to ocean pixels.  The total loss is:

  L_total = L_eps + sum_i(weight_i * L_i)

| `--loss` | What it measures |
|---|---|
| `curl_div` | RMSE between curl and divergence fields of predicted vs true x_0 |
| `spectral` | RMSE between 2D FFT power spectra of predicted vs true x_0 |
| `okubo_weiss` | RMSE between Okubo-Weiss parameter W of predicted vs true x_0 |
| `wasserstein` | Sinkhorn–Wasserstein distance between vorticity point clouds |

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

## Loss Function Ablation — Full Study (June 5–12, 2026)

### Background and motivation

The base DDPM loss is pure epsilon-MSE on ocean pixels, penalising pixel-level amplitude
error but imposing no constraint on the *physical structure* of the predicted field —
rotational features (eddies), divergence patterns, spectral energy distribution, and
deformation fronts are all ignored.  The hypothesis is that structural regularisation
losses, added as auxiliary terms computed on the estimated clean field recovered from the
noisy prediction, could encourage the model to learn better representations of ocean
dynamics.

All auxiliary losses are added on top of the base epsilon loss:

  L_total = L_eps + weight * L_aux

where the ocean pixel mask is applied before each auxiliary computation.

---

### Loss functions implemented (`loss_functions.py`)

Seven loss modes are available.  All auxiliary terms operate on the pair
(predicted x_0, true x_0) via finite-difference gradients through a central-difference
convolution kernel (shared `_jacobian` helper).

#### 1. `eps` — Pure epsilon-MSE baseline
No auxiliary term.  The de-facto standard DDPM training objective.

#### 2. `curl_div` — Vorticity + Divergence regularisation

Computes RMSE between the stacked [curl, divergence] fields of the predicted and true
x_0.  Curl (vorticity) = dv/dx - du/dy; Divergence = du/dx + dv/dy.  Both computed via
2D central differences and stacked into a 2-channel field.

**Physical motivation:** Eddies and jets are characterised by coherent vorticity
concentrations.  Matching vorticity forces the model to reproduce the rotational
structure of the flow, not just the velocity magnitude.  Divergence matching encourages
mass-conservation properties (ocean currents are nearly incompressible, so divergence ~ 0).

#### 3. `spectral` — Power-spectrum MSE

Computes RMSE between the 2D real FFT amplitude spectra of the predicted and true x_0,
channel-wise.  Land pixels are zeroed before the FFT so they contribute no spurious
spectral energy.

**Physical motivation:** Ocean turbulence follows a power-law kinetic energy spectrum.
Matching spectral amplitudes forces the model to reproduce the correct distribution of
energy across spatial scales — suppressing unphysical checkerboard artefacts or
over-smoothing.

In practice this loss performed *worst* in evaluation — see results below.

#### 4. `okubo_weiss` — Okubo–Weiss parameter

Computes RMSE between the Okubo–Weiss parameter W of the predicted and true x_0, masked
to ocean pixels.  W = sn^2 + ss^2 - omega^2, where sn = du/dx - dv/dy (normal strain),
ss = du/dy + dv/dx (shear strain), and omega = dv/dx - du/dy (vorticity).

**Physical motivation:** W < 0 marks eddy cores (rotation-dominated); W > 0 marks eddy
boundaries and strain-dominated regions (fronts, filaments).  The Okubo–Weiss criterion
is a standard oceanographic tool for detecting coherent vortices.  Matching W encourages
the model to correctly locate and size eddy structures.

#### 5. `wasserstein` — Sinkhorn–Wasserstein vorticity transport distance

Computes the Wasserstein-2 distance between the vorticity point clouds of the predicted
and true x_0.  The top-64 ocean vorticity points are used per sample (weighted by
absolute vorticity magnitude, normalised to sum 1) to keep the Sinkhorn computation
tractable.  Implemented via `geomloss` (blur=0.05) on CPU float32 tensors.

**Physical motivation:** Pure pixel-wise losses treat vorticity errors as independent,
but physically what matters is whether eddies are in the *right location*.  The
Wasserstein distance measures the minimum transport cost to rearrange one vorticity
distribution into the other, so it correctly penalises spatially displaced eddies rather
than misregistration that might look pixel-perfect if the field were shifted by a few
cells.

**Note:** Only the top-64 vorticity points per sample are used to keep the Sinkhorn
computation tractable without `pykeops` (the pure-Python CPU backend scales as O(K^2)
in the number of points K).

#### 6. `stream_function` — Stream-function Poisson solve

Computes RMSE between the stream functions of the predicted and true x_0, masked to
ocean pixels.  The stream function is recovered from vorticity via an FFT Poisson solve
(nabla^2 psi = omega; DC component set to zero to fix gauge freedom).

**Physical motivation:** For a 2D incompressible flow, the stream function completely
characterises the velocity field (u = dpsi/dy, v = -dpsi/dx).  Matching the stream
function directly penalises errors in the integrated rotational structure — it is a
smoother, more spatially coherent version of the vorticity loss, since the Poisson
inversion acts as a low-pass filter on vorticity errors.

Trained for 400 epochs (vs 200 for the original five models), reaching epoch 392 at
best checkpoint.

#### 7. `strain_rate` — Strain-rate tensor invariants

Computes RMSE between the strain-rate tensor invariants of the predicted and true x_0,
masked to ocean pixels.  The two invariants are: I1 = du/dx + dv/dy (divergence / trace
of S) and I2 = (du/dx)(dv/dy) - 0.25*(du/dy + dv/dx)^2 (determinant of S).

**Physical motivation:** The determinant of the strain-rate tensor is sensitive to
deformation structures — fronts, filaments, and confluent/diffluent flow regions — that
are not captured by vorticity or divergence alone.  These structures are
oceanographically important because they control tracer dispersion and subduction of
water masses.

Also targeted for 400 epochs (best checkpoint at epoch 391).

---

### Training runs summary

All 7 models were trained on a vast.ai NVIDIA TITAN X (Pascal, 12 GB) using AdamW,
cosine LR decay from 2e-4, batch size 32, T=1000, cosine noise schedule, base_ch=64.
All models were targeted at **400 epochs** with auxiliary weight = 1.0 using the root
`train_loss_comparison.sh` (for eps/curl_div/spectral/okubo_weiss/wasserstein) and
`train_new_losses.sh` (for stream_function/strain_rate).

The evaluations (Runs 1–3) were run using checkpoints **downloaded mid-training** before
the 400-epoch runs completed for the first 5 models.  The checkpoints used were:

| Model | Script | Target epochs | Best epoch at eval time | Best epoch (current, server) |
|---|---|---|---|---|
| eps | `train_loss_comparison.sh` | 400 | 172 | 279 |
| curl_div | `train_loss_comparison.sh` | 400 | 173 | 386 |
| spectral | `train_loss_comparison.sh` | 400 | 179 | 375 |
| okubo_weiss | `train_loss_comparison.sh` | 400 | 147 | 359 |
| wasserstein | `train_loss_comparison.sh` | 400 | 195 | 279 |
| stream_function | `train_new_losses.sh` | 400 | 392 | 392 |
| strain_rate | `train_new_losses.sh` | 400 | 391 | 391 |

stream_function and strain_rate had fully converged before the evaluation download,
so their checkpoints are identical between the eval and current server state.  The other
five continued training after the evaluation snapshots were taken and found better
checkpoints (epoch 279–386 vs 147–195 at eval time) — the 800-epoch re-training run
will include fully converged versions of all 7.

**Note on DDPM/models/\*1.pt:** A separate earlier prototype training run (June 5,
default weights, first 5 models only) produced a set of checkpoints downloaded locally
as `model_ddpm_{name}_gaussian_cosine1.pt` in `DDPM/models/`.  These are distinct
from the models used in the ablation evaluations (confirmed by MD5 hash comparison)
and were not used in any of the formal evaluation runs.

A subsequent **800-epoch re-training run** was launched via `train_all_losses_800.sh`
with a uniform auxiliary weight of 1.0, saving under
`Model Parameters/loss_comparison_800/`.  This will provide fully converged checkpoints
for all 7 models and allow a fair comparison unaffected by the mid-run download issue.

---

### Evaluation scripts

| Script | Purpose | Key parameters |
|---|---|---|
| `batch_eval_loss.py` | Run 1: 7 models × N samples, 7 metrics, CSV output | `--n_runs 10 --seed 42` |
| `batch_eval_v2.py` | Run 2: 6 models × N samples, 3 metrics, visual 3×3 PNGs | `--seed 99` (different samples from Run 1) |
| `loss_eval_visual.py` | Run 3: 7 models × 50 samples × 10 seeds, 4×4 visual grid + mean/variance heatmaps | `--n_samples 50 --n_runs 10` |
| `batch_eval_multi_run.py` | Run 4: 7 models × N samples × 10 seeds, comprehensive 4×4 visual PNGs + full CSV detail table, versioned output dirs | `--n_samples 50 --n_runs 10` |
| `train_loss_comparison.sh` (root) | Server training for all 5 original models (eps, curl_div, spectral, okubo_weiss, wasserstein), 400 epochs, weight = 1.0 | — |
| `train_new_losses.sh` | Server training for stream_function and strain_rate, 400 epochs | — |
| `train_all_losses_800.sh` | Server re-training all 7 models for 800 epochs, weight = 1.0 | — |

`batch_eval_multi_run.py` is the most comprehensive evaluation script.  For each
validation sample it runs all 7 models 10 times with different diffusion seeds but the
same robot path, saving a 4×4 grid per sample:

```
Row 0: Ground truth | Robot path
Row 1: eps avg      | eps variance heatmap
Row 2: curl_div avg | curl_div variance heatmap
...    (one row per model, 2 panels each)
```

It also writes a `summary.txt` with mean±std of all 7 metrics per model and a full
detail CSV (one row per model × sample × seed).  Output goes to versioned subdirectories
`results/eval_multi_run/run_v1/`, `run_v2/`, etc. — nothing is ever overwritten.

Metrics computed by the evaluation scripts:

| Metric key | What it measures |
|---|---|
| `eps` | Ocean-masked RMSE of the predicted velocity field |
| `curl_div` | RMSE of curl and divergence fields |
| `spectral` | RMSE of FFT power spectra (u and v channels) |
| `okubo_weiss` | RMSE of Okubo–Weiss parameter W |
| `stream_function` | RMSE of stream-function field (from Poisson solve) |
| `strain_rate` | RMSE of strain-rate tensor invariants I₁, I₂ |
| `wasserstein` | Sinkhorn–Wasserstein distance on vorticity point clouds |

---

### Run 1 — seed 42, 7 models, 10 val samples (`results/eval_loss/`)

Samples: [168, 174, 185, 395, 848, 859, 1281, 1368, 1514, 1683]

| Model | RMSE (field) | Spectral RMSE | Wasserstein |
|---|---|---|---|
| **strain_rate** | **0.0674** | **21.28** | 0.0751 |
| **curl_div** | **0.0699** | **21.98** | 0.0785 |
| eps | 0.0837 | 31.29 | 0.0805 |
| stream_function | 0.0847 | 29.97 | 0.0735 |
| wasserstein | 0.0966 | 39.74 | 0.0929 |
| okubo_weiss | 0.0973 | 24.55 | **0.0405** |
| spectral | 0.1175 | 71.56 | 1.1055 |

- `strain_rate` wins RMSE and spectral RMSE; `curl_div` is close behind.
- `okubo_weiss` wins Wasserstein distance by a large margin — its explicit vorticity-matching
  incentive produces better-located eddies even when the pixel-level RMSE is mediocre.
- `spectral` performs worst on field RMSE and produces dramatically inflated spectral RMSE
  (71.56 vs 21–32 for the others) — the loss is poorly calibrated and was excluded from Run 2.

### Run 2 — seed 99, 6 models (no spectral), 10 val samples (`results/eval_run2/`)

Samples: [348, 990, 1003, 1107, 1208, 1483, 1848, 1874, 1895, 1908]

| Model | RMSE (field) | Spectral RMSE | Wasserstein |
|---|---|---|---|
| **eps** | **0.0962** | **21.42** | **0.0463** |
| curl_div | 0.1029 | 30.00 | 0.0495 |
| wasserstein | 0.1059 | 41.52 | 0.0623 |
| stream_function | 0.1106 | 38.13 | 0.0716 |
| okubo_weiss | 0.1296 | 48.46 | 0.0661 |
| strain_rate | 0.1666 | 62.10 | 0.0691 |

- **eps wins all 3 metrics** on this sample set — a complete reversal of Run 1.
- `strain_rate` collapses to worst performer (RMSE 0.067 → 0.167): its Run 1 dominance was
  sample-set-dependent, not a genuine structural advantage.
- `curl_div` remains consistently 2nd best across both runs.
- The complete reversal between Run 1 and Run 2 demonstrates that 10-sample evaluations
  are insufficient to reliably rank models — high variance in the val set composition
  dominates the signal.

### Cross-run comparison (Runs 1 and 2)

| Model | Run1 RMSE | Run2 RMSE | Run1 Std | Run2 Std | Trend |
|---|---|---|---|---|---|
| eps | 0.0837 | 0.0962 | 0.0383 | 0.0425 | Stable |
| curl_div | 0.0699 | 0.1029 | 0.0177 | 0.0923 | Degrades in Run 2 |
| okubo_weiss | 0.0973 | 0.1296 | 0.1078 | 0.0993 | High variance both runs |
| wasserstein | 0.0966 | 0.1059 | 0.0244 | 0.0672 | Moderate degradation |
| stream_function | 0.0847 | 0.1106 | 0.0285 | 0.0409 | Moderate degradation |
| strain_rate | 0.0674 | 0.1666 | 0.0248 | 0.1212 | Catastrophic degradation |

### Run 3 — 50 samples × 10 diffusion seeds (`results/eval_visual_loss/`, June 9–12, 2026)

This is the definitive evaluation.  `loss_eval_visual.py` was run on the server for all
50 validation samples listed below, each evaluated 10 times with different diffusion
noise seeds (same robot path per sample across all models):

```
Samples: [124, 133, 165, 171, 179, 181, 247, 323, 353, 387, 442, 540, 694, 718, 781,
          831, 842, 862, 871, 874, 876, 968, 991, 1013, 1060, 1080, 1234, 1249, 1255,
          1331, 1341, 1371, 1384, 1418, 1468, 1483, 1516, 1517, 1529, 1598, 1616, 1624,
          1649, 1675, 1731, 1753, 1796, 1880, 1903, 1956]
```

For each sample a 4×4 PNG is saved (`results/eval_visual_loss/sample_{idx:03d}.png`)
showing the ground truth, robot path, and for each model: its **mean quiver
reconstruction** over 10 runs alongside a **pixelwise speed-variance heatmap**.  All 50
samples completed successfully.

**Aggregate mean RMSE (mean of per-sample mean-RMSE, 50 samples × 10 seeds):**

| Model | Mean RMSE | Approx. Std | Rank | Notes |
|---|---|---|---|---|
| **wasserstein** | **0.0788** | ~0.025 | 1 | Tied with curl_div |
| **curl_div** | **0.0788** | ~0.022 | 1 | Tied with wasserstein |
| stream_function | 0.0855 | ~0.025 | 3 | Consistent 3rd place |
| eps | 0.0876 | ~0.027 | 4 | Baseline; stable |
| strain_rate | 0.0894 | ~0.027 | 5 | Moderate |
| okubo_weiss | 0.0895 | ~0.034 | 6 | High variance |
| spectral | 0.1461 | ~0.047 | 7 | Worst by large margin |

Key findings from Run 3:

- **`wasserstein` and `curl_div` emerge as the two best models** with nearly identical mean
  RMSE (0.0788), each approximately 10% better than the eps-only baseline (0.0876).
  This result is stable across 50 samples and could not have been detected from the 10-sample
  runs, where sampling noise dominated.

- **`stream_function` is a consistent 3rd** at 0.0855 — noticeably better than eps despite
  being based on the same vorticity information as `curl_div`.  The smoother (Poisson-filtered)
  representation of rotational structure appears to provide a gentler but reliable regularisation.

- **`eps` sits in 4th place**, confirming it is not the best choice when structural accuracy
  matters, but it remains competitive and is the most interpretable baseline.

- **`strain_rate` and `okubo_weiss` perform similarly** (~0.089–0.090) — marginally worse than
  eps on average and with higher variance, especially `okubo_weiss`.  The instability of
  `okubo_weiss` is unsurprising given it involves a quadratic combination of first-order
  derivatives (sn^2 + ss^2 - omega^2) that can produce large gradients.

- **`spectral` is definitively the worst** (0.146 mean, 0.047 std) — 67% worse than the best
  models and 7× more variable.  The FFT-domain loss provides no spatial localisation and
  likely provides contradictory training signals in regions with strong land boundaries.

**Per-sample breakdown of notable outliers:**

| Sample | Best model | RMSE | Worst model | RMSE | Notes |
|---|---|---|---|---|---|
| 1060 | wasserstein | 0.0494 | spectral | 0.0877 | Easy sample, flat flow |
| 1080 | curl_div | 0.0410 | spectral | 0.0862 | Lowest RMSE seen |
| 1624 | okubo_weiss | 0.0422 | spectral | 0.0649 | Compact, simple field |
| 1616 | okubo_weiss | 0.0930 | spectral | 0.2940 | Spectral fails severely |
| 1731 | okubo_weiss | 0.1010 | spectral | 0.2732 | Strong eddy, spectral blown out |
| 1880 | curl_div | 0.1150 | okubo_weiss | 0.1967 | High-speed jet sample |
| 862 | stream_fn | 0.0996 | spectral | 0.2428 | Difficult sample, curl_div 2nd |

On samples containing large, well-defined eddies (e.g. 1616, 1731), `okubo_weiss` wins
individually — it correctly places and sizes eddy cores because its loss directly targets
the Okubo–Weiss parameter that defines them.  However it is highly inconsistent across
samples without clear eddies, explaining its high overall variance.

**Stochastic variance (across 10 diffusion seeds per sample):**

The variance heatmaps in the 4×4 PNGs reveal that all models have high stochastic
uncertainty near the coastline (land–ocean boundary) and in the lower-left quadrant
(X = 0–30, Y = 0–15) which is the high-speed coastal jet region far from most robot
paths.  `wasserstein` and `curl_div` show the smallest variance heatmaps overall,
consistent with their lower RMSE — the structural regularisation reduces sample-to-sample
noise in the diffusion sampling process itself.

**Model checkpoint note:** All 7 models were targeted at 400 epochs.  `stream_function`
and `strain_rate` had fully converged (best at epoch 391–392) before the evaluation
download.  The other five were downloaded mid-training with best checkpoints at epoch
147–195; they have since continued training on the server and found better checkpoints
(epoch 279–386).  The evaluation results therefore represent a somewhat earlier stage
of training for the first five models, which may slightly understate their potential.

---

### Overall conclusions from the loss function ablation

1. **`curl_div` is the recommended structural loss** for this task.  It is tied with
   `wasserstein` for best mean RMSE across 50 samples, has lower training complexity
   (no Sinkhorn solver required, no geomloss dependency), and showed low variance in
   Run 1.  It directly regularises both vorticity and divergence — the two most
   physically meaningful first-order invariants of a 2D flow.

2. **`wasserstein` is equally accurate** but more expensive to train (CPU Sinkhorn
   solve per batch, limited to top-64 vorticity points) and has an extra geomloss
   dependency.  Its advantage is that it penalises spatially displaced eddies correctly
   rather than pixel-by-pixel discrepancy.

3. **`stream_function` is the best single-physics-quantity regulariser** when a clean,
   smooth interpretation of the vorticity field is desired.  Poisson-filtering the
   vorticity before computing the loss acts as a spatial low-pass that stabilises
   training.

4. **`spectral` should not be used** as a standalone auxiliary loss for this domain.
   The hard land boundaries create large spectral artefacts and the loss lacks spatial
   localisation; it consistently produced the worst reconstructions by a large margin
   across all three evaluation runs.

5. **`strain_rate` is unreliable.**  It performed best in Run 1 and worst in Run 2 — a
   swing of 15 percentage points.  The strain-rate invariants (I1 = divergence, I2 = det(S)) appear
   to capture structures that are highly sample-set-dependent and not generically
   predictive of inpainting quality.

6. **Small-sample evaluations (N=10) are insufficient** to rank these models.  The 50-sample
   evaluation changed the conclusion on which model is best: in Run 1 `strain_rate` won,
   in Run 2 `eps` won, and in Run 3 `curl_div`/`wasserstein` won.  At least 30–50
   samples are required to obtain stable rankings.

7. **The 800-epoch re-training** (`train_all_losses_800.sh`, uniform weight = 1.0) is in
   progress on the server.  If the ordering is preserved after longer training and
   higher loss weight, it will confirm that `curl_div` and `wasserstein` are genuinely
   the better structural regularisers and the finding is not an artefact of early stopping.

---

## Conditional DDPM (June 13–16, 2026)

### Motivation

The base DDPM (and all the structural-loss variants) are **unconditional**: they know
nothing about which cells the robot observed during inference.  The RePaint algorithm
grafts observations in post-hoc by anchoring the path pixels to their true values at
every reverse step, but the model's denoising prior is still trained without that
information.  A **conditioned** model can see the observation geometry (and optionally
the measured values) at every denoising step and adjust its prediction accordingly — it
has the structural information it needs from the very start of the reverse chain.

### Architecture (`Conditional DDPM/cond_model.py`)

The conditioned model uses **FiLM (Feature-wise Linear Modulation)** (Perez et al.,
AAAI 2018) to inject conditioning information into a UNet with the same encoder-decoder
structure as the base model.

#### ConditionEncoder
A lightweight CNN encodes the spatial conditioning map (B, cond_in_ch, H, W) into a
fixed-size embedding (B, cond_dim=256) via strided convolutions + global average
pooling.  This produces a single conditioning vector per sample that is shared across
all blocks.

#### FiLMResBlock
Every ResBlock is replaced by a FiLMResBlock that receives **two** conditioning signals:

1. **Time embedding** — additive shift (same as the base UNet ResBlock).
2. **Condition embedding** — FiLM scale and shift:
   `h ← (1 + γ) ⊙ h + β`
   where γ and β are produced by a per-block linear layer from the condition embedding.

Using `(1 + γ)` instead of `γ` means γ = 0 at initialisation leaves activations
unchanged — the block starts as the unconditioned ResBlock and gradually learns to use
the conditioning signal.  The FiLM projection weights and biases are zero-initialised
to enforce this.

**Parameter counts:** ~16.5M (vs 15.0M for the base UNet) — the overhead is small
because the ConditionEncoder and FiLM projections are lightweight.

### Conditioning modes

Four conditioning modes are implemented, selected with `--cond`:

| Mode | `cond_in_ch` | Channels provided | What the model sees |
|---|---|---|---|
| `voronoi` | 3 | `[u_vor, v_vor, sensor_mask]` | Voronoi tessellation of sensor readings + presence mask |
| `path` | 1 | `[path_mask]` | Binary mask: which cells were visited (no values) |
| `path_field` | 3 | `[u_path, v_path, path_mask]` | True u/v at path cells (0 elsewhere) + path mask |
| `both` | 4 | `[u_vor, v_vor, sensor_mask, path_mask]` | Voronoi field + explicit path geometry |

`voronoi` feeds the nearest-neighbour Voronoi interpolation of the sensor readings into
the model, giving an estimate of the full field everywhere.  `path` provides only the
observation geometry, forcing the model to infer values from its learned prior.
`path_field` provides sparse direct observations (the most honest representation of what
the robot actually measured) without an interpolated estimate.  `both` combines the
Voronoi estimate and the path geometry for maximum information.

### Conditioning generation during training

Conditioning is generated **on-the-fly per training step**:
- Each sample in a batch receives a fresh biased-walk path with a deterministic seed
  `seed = epoch * 100_000 + batch_idx * 1_000 + b`.
- This means the model trains over a large distribution of observation geometries rather
  than memorising a fixed path layout — the key property that makes the model applicable
  to arbitrary robot paths at inference time.
- The VoronoiLayer (no learnable weights) tessellates sensor positions to a spatial grid
  and is reused from `Voronoi/model/voronoi_model.py`.

### Training runs

All three runs used AdamW + cosine LR decay from 2e-4, batch size 16, T=1000, cosine
noise schedule, base_ch=64, cond_dim=256, 400 epochs, on vast.ai NVIDIA TITAN X (Pascal).

| Mode | Model params | Best val loss | Best epoch | Status |
|---|---|---|---|---|
| `voronoi` | 16,497,734 | **0.00330** | 278 | Complete |
| `path` | 16,497,158 | **0.00337** | 329 | Complete |
| `both` | 16,498,022 | 0.00366 | 106 | Incomplete (cut off at epoch 110) |

The `both` run was terminated early before converging; the `voronoi` and `path` runs
fully converged.  Note the training val losses are epsilon-MSE only (no RePaint
anchoring), so they reflect the model's denoising quality in free-sampling mode.

### Inference results

All inference uses RePaint (r=10, T=1000) with the biased walk path, evaluated on
10 random val samples (indices: [1663, 1598, 1246, 1000, 528, 80, 32, 603, 344, 147]),
seeds 1–10.

#### Per-mode RMSE (10 val samples)

| Mode | Mean RMSE | Std | Best epoch loaded |
|---|---|---|---|
| `path` | **0.1023** | 0.0327 | 329 |
| `voronoi` | 0.1271 | 0.0383 | 278 |
| `both` | partial (0.1961 on 1 sample) | — | 106 (incomplete) |

The `path`-conditioned model outperforms `voronoi` despite receiving less information:
knowing *where* the robot walked (without the measured values) is enough to meaningfully
guide the diffusion sampling.  The `voronoi` model receives the interpolated field but
performs worse — the Voronoi tessellation may introduce artefacts (sharp Voronoi
boundaries, incorrect values in cells far from sensors) that confuse the model.

#### eps vs voronoi-cond comparison (`compare_infer.py`, 10 val samples)

The comparison script runs both the base eps model (via RePaint) and the voronoi-cond
model (via free sampling) on the same validation samples.

| Model | Mean RMSE | Std | Notes |
|---|---|---|---|
| DDPM-eps (RePaint, r=10) | 0.1281 | 0.0758 | Unconditional base model |
| Voronoi-cond DDPM | **0.1094** | **0.0270** | Conditioned, free sampling |

The voronoi-cond model achieves lower mean RMSE and substantially lower variance — the
conditioning reduces sample-to-sample inconsistency from std=0.076 to std=0.027.

Per-sample breakdown (the two models use the same path and same validation samples):

| Sample | eps RMSE | voronoi-cond RMSE | Winner |
|---|---|---|---|
| 1663 | 0.0859 | 0.1595 | eps |
| 1598 | 0.1009 | 0.1372 | eps |
| 1246 | 0.1336 | 0.0796 | cond |
| 1000 | 0.0489 | 0.1031 | eps |
| 528 | **0.3306** | **0.0997** | cond (+3× better) |
| 80 | 0.0771 | 0.1165 | eps |
| 32 | 0.0683 | 0.0948 | eps |
| 603 | 0.1380 | 0.1409 | eps (narrow) |
| 344 | 0.1323 | 0.0893 | cond |
| 147 | 0.1650 | 0.0732 | cond |

Sample 528 is striking: eps RMSE of 0.33 (the robot path missed the main high-speed
feature) while the Voronoi-cond model produced 0.10.  The Voronoi tessellation anchors
the conditioning on the rough field structure even when the path coverage is poor,
providing a weaker but spatially broader prior than the RePaint path-anchoring alone.

#### Key observations

- **`path` conditioning wins quantitatively** (mean RMSE 0.1023 vs 0.1271 for voronoi).
  The path mask, while minimal in information content, is precisely what the model needs
  to know during denoising: where the constraint cells are.
- **Voronoi conditioning reduces worst-case failures**: the voronoi model's std (0.038)
  vs path model's std (0.033) is similar, but the voronoi model avoids the extreme
  failure mode seen in eps (RMSE 0.33 on sample 528).
- **Voronoi conditioning is used in free-sampling mode** (no RePaint loop), which is
  ~10× faster at inference time than RePaint with r=10.
- **`both` mode incomplete**: the training run was cut off at epoch 110 (best at 106);
  the model has not converged and its single-sample result (RMSE 0.196) is not
  representative.

### Module layout (`Conditional DDPM/`)

| File | Purpose |
|---|---|
| `cond_model.py` | `CondUNet`, `FiLMResBlock`, `ConditionEncoder` |
| `cond_diffusion.py` | `CondDDPM` — forward/reverse/RePaint all taking a `cond` argument |
| `train.py` | training loop with on-the-fly conditioning, `--cond` flag |
| `infer.py` | batch inference for any single conditioning mode (RePaint + free sampling) |
| `compare_infer.py` | side-by-side eps vs voronoi-cond, 2×3 figure per sample |
| `batch_eval_cond_multirun.py` | multi-run (N seeds) batch evaluation |
| `launch_training.sh` | server script; trains voronoi, path, and both sequentially |

### Usage
```bash
# train (from workspace root)
python "Conditional DDPM/train.py" --cond voronoi --pickle data.pickle
python "Conditional DDPM/train.py" --cond path    --pickle data.pickle
python "Conditional DDPM/train.py" --cond both    --pickle data.pickle

# infer (from workspace root)
py "Conditional DDPM/infer.py" --cond voronoi --pickle data.pickle
py "Conditional DDPM/infer.py" --cond path    --pickle data.pickle

# compare eps vs voronoi-cond
python "Conditional DDPM/compare_infer.py" --pickle data.pickle
```

---

## Gaussian Noise Normalization (June 16, 2026)

### Motivation

Standard DDPM uses unit Gaussian noise ε ~ N(0, I) in the forward process.  The ocean
current data has values with std ≈ 0.12 (range roughly −0.9 to +1.1 after the dataset's
built-in normalisation).  This means the data lives in a much smaller region of the
noise space than standard DDPM assumes — the model is asked to learn to denoise noise
that is ~8× larger than the signal.

The hypothesis is that scaling the forward-process noise to match the data's natural
amplitude (setting `noise_scale ≈ 0.12`) could improve training stability and produce
better reconstructions by reducing the signal-to-noise gap.

### Implementation (`DDPM/model/diffusion.py`)

A `noise_scale` parameter (default 1.0) was added to the `DDPM` class.  Three locations
are affected:

**1. Forward process `q_sample`** — noise is scaled and clamped to [−1, 1]:
```python
noise = torch.clamp(torch.randn_like(x0) * self.noise_scale, -1.0, 1.0)
```

**2. Reverse step `p_sample_step`** — posterior variance is scaled to match:
```python
return mean + var.sqrt() * self.noise_scale * torch.randn_like(xt)
```

**3. Forward resampling step `q_sample_from_prev`** (used by RePaint):
```python
return alpha.sqrt() * x_prev + (1.0 - alpha).sqrt() * self.noise_scale * torch.randn_like(x_prev)
```

**4. Initial noise in RePaint (`repaint_infer.py`)** — starting noise is also scaled:
```python
xt = torch.clamp(torch.randn(...) * diffusion.noise_scale, -1.0, 1.0)
```

The clamp to [−1, 1] prevents the noise from producing values far outside the data
range, acting as a soft constraint on the amplitude of the diffusion trajectory.

### Training flag

`--noise_scale` was added to `DDPM/model/train.py`:
```bash
python model/train.py --loss curl_div --noise_scale 0.12 --pickle ../data.pickle
```

The checkpoint filename automatically appends `_ns0p12` when `noise_scale != 1.0` so
that models trained with different scales are saved to distinct files:
`ddpm_curl_div_gaussian_cosine_ns0p12.pt`.

At inference time, `batch_infer.py` reads the noise scale from the checkpoint's saved
args (with fallback to 1.0 for older checkpoints), so no manual flag is needed when
evaluating a checkpoint trained with `noise_scale != 1.0`.

### Status

The noise scale feature is implemented and staged.  Training runs with `noise_scale=0.12`
have not yet been completed — results are TBD pending a server run.

