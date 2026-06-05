# Diffusion Inpainting of Ocean Current Vector Fields

Reconstructing full 2D ocean current vector fields from sparse robot path observations using diffusion-model-based inpainting.

---

## Project Goal

A slow autonomous underwater robot traverses a short random-walk path through a coastal ocean domain, collecting local current measurements at only the grid cells it visits (~3–4% of the ocean area). The objective is to reconstruct the **complete 2D vector field** (east-west *u* and north-south *v* velocity components) from those sparse observations.

The approach is based on the **RePaint** algorithm (Lugmayr et al., CVPR 2022): a pre-trained unconditional DDPM is used as a learned prior over ocean current fields, and at inference time the known observations are blended in at each denoising step to condition the reconstruction.

---

## Data (`data.pickle`)

### Source
The dataset contains snapshots of a coastal ocean current simulation (or reanalysis product). Values are normalised and represent 2D velocity vector fields over a fixed spatial grid.

### Structure

The pickle file is a Python `list` of 3 NumPy arrays, one per split:

| Split | Samples | Array shape |
|---|---|---|
| Train | 9,180 | (94, 44, 2, 9180) |
| Val   | 1,965 | (94, 44, 2, 1965) |
| Test  | 1,965 | (94, 44, 2, 1965) |

Dimensions: `(X=94, Y=44, C=2, N)` where C=0 is *u* (east-west) and C=1 is *v* (north-south).  
After transposing for PyTorch: **(N, 2, H=94, W=44)**.

### Value Statistics

| Split | Min | Max | Mean | Std | NaN % |
|---|---|---|---|---|---|
| Train | −0.8973 | 1.0860 | −0.051 | 0.116 | 8.8% |
| Val   | −0.6952 | 0.6305 | −0.049 | 0.119 | 8.8% |
| Test  | −0.6980 | 0.8431 | −0.047 | 0.115 | 8.8% |

Values are normalised (roughly zero-centred, std ≈ 0.12). The slightly negative mean *u* suggests a weak mean westward background current.

### Land Mask

- **8.8% of all grid cells are NaN** — these are land pixels.
- The NaN pattern is **identical across all timesteps and splits**: it is a fixed spatial mask.
- After transposing for display (X=0–93 horizontal, Y=0–43 vertical), land forms a coastal peninsula extending from the top edge down to approximately Y=24, centred around X=5–51.
- **3,787 ocean cells** out of 94×44 = 4,136 total.
- `dataset.py` replaces NaN → 0 and exposes the mask as a `(H, W)` boolean tensor (`True` = land).

### Physical Interpretation

- Coastal ocean region with a dominant large-scale current (mostly eastward in the upper half) and a recirculation/boundary-layer feature near the coast.
- Speeds reach ~0.9 normalised units; typical ocean speeds are 0.05–0.15 normalised units.

---

## Models

### DDPM (`DDPM/`)
Unconditional DDPM with configurable structural regularisation losses:
- **UNet** with sinusoidal timestep embeddings and ResNet blocks, 14.9 M parameters, input (B, 2, 94, 44) padded internally to (96, 48).
- **Cosine noise schedule**, T = 1000 steps, epsilon-prediction.
- **Training loss**: epsilon-MSE (default) plus any combination of auxiliary structural losses, each with its own independent weight λ.
- **RePaint inference**: robot path observations are blended in at every reverse-diffusion step with r=10 resamplings per timestep.

Best val loss (eps-only): **0.00327** (epoch 155/200). Mean RMSE on 10 val samples: **0.1146 ± 0.022**.

#### Loss modes (`--loss`, combinable)

All auxiliary losses operate on the denoised reconstruction $\hat{x}_0$ and are masked to ocean pixels.

| `--loss` | What it penalises | Default λ |
|---|---|---|
| `eps` | Epsilon-MSE only — no auxiliary term (default) | — |
| `curl_div` | MSE between curl and divergence of $\hat{x}_0$ and $x_0$ | 0.0002 |
| `spectral` | MSE between FFT power spectra | 0.0002 |
| `okubo_weiss` | MSE between Okubo-Weiss eddy criterion $W = s_n^2 + s_s^2 - \omega^2$ | 0.001 |
| `wasserstein` | Sinkhorn–Wasserstein distance between vorticity point clouds | 1.0 |

All loss functions are defined in `Model Parameters/loss_functions.py`.

### GP Baseline (`GP Baseline/`)
Two independent Gaussian Processes (Matérn ν=2.5 + WhiteKernel), one per velocity component, fit directly to the robot path observations at inference time. No training phase. Mean RMSE on 10 val samples: **0.2011 ± 0.079** — ~75% higher than the DDPM.

### Noise Schedule Ablation (`Model Parameters/NoiseSchedule/`)
Same DDPM architecture trained under four noise schedules: **linear**, **cosine**, **quadratic**, **sigmoid**. Each is trained independently for 100 epochs and evaluated with the RePaint algorithm. All four share the same inference engine (`repaint_infer.py`). Results stored in `results/model_{schedule}_results/`.

Run all four: `bash "Model Parameters/NoiseSchedule/run_repaint.sh"` from the project root.

### Voronoi Tessellation Baseline (`Voronoi/`)
Implementation of **VoronoiNet** (Fukami et al., 2021 — *Nature Machine Intelligence*): sparse sensor readings are mapped onto the grid via nearest-neighbour Voronoi tessellation and fed into a U-Net encoder-decoder to reconstruct the full field.

- **Two sensor modes**: scattered (random ocean cells) and walk (biased robot path).
- Trained independently for each mode; checkpoints in `Voronoi/models/`.
- Evaluation across all four train/test mode combinations in `Voronoi/results/`.

---

## Repository Layout

```
data.pickle               ← dataset (shared, gitignored)
dataset.py                ← OceanCurrentDataset (PyTorch Dataset)
paths.py                  ← robot path generators (biased walk, random walk)
plot_utils.py             ← quiver-plot helper

DDPM/
  requirements.txt
  model/
    model.py              ← UNet architecture
    diffusion.py          ← DDPM (cosine schedule, q_sample, RePaint)
    train.py              ← training script
  testing/
    repaint/
      repaint_infer.py    ← RePaint inference engine
    visualize_infer.py    ← single-sample inference + 2×2 figure
    batch_infer.py        ← batch evaluation over N val samples
  models/
    best_model.pt         ← trained checkpoint (gitignored)
  results/
    best_model_results/
      result_01.png … result_10.png

GP Baseline/
  gp_infer.py
  visualize_infer.py
  batch_infer.py
  requirements.txt
  GP_results/
    result_01.png … result_10.png

Model Parameters/
  loss_functions.py       ← all auxiliary loss functions (curl_div, spectral, okubo_weiss, wasserstein)
  NoiseSchedule/
    diffusion.py          ← DDPM with pluggable noise schedules (linear/cosine/quadratic/sigmoid)
    repaint_model.py      ← Repaint UNet (same architecture as DDPM/model/model.py)
    repaint_infer.py      ← RePaint inference engine (biased walk + repaint loop)
    train_repaint.py      ← training script  (--schedule cosine|linear|quadratic|sigmoid)
    test_repaint.py       ← test-set evaluation + 2×2 visualisation
    batch_repaint.py      ← batch evaluation (10 val samples per schedule)
    run_repaint.sh        ← bash script to train all four schedules
    requirements.txt
    checkpoints/
      checkpoints_repaint_cosine/
      checkpoints_repaint_linear/
      checkpoints_repaint_quadratic/
      checkpoints_repaint_sigmoid/
    results/
      model_comparison.txt
      model_cosine_results/
      model_linear_results/
      model_quadratic_results/
      model_sigmoid_results/

Voronoi/
  model/
    voronoi_model.py      ← VoronoiNet (VoronoiLayer + U-Net encoder-decoder)
    train_voronoi.py      ← training script (--sensor_mode scattered|walk)
  testing/
    test_voronoi.py       ← test-set evaluation + 2×2 visualisation
    batch_voronoi.py      ← batch evaluation with scattered sensors
    batch_voronoi_walk.py ← batch evaluation with biased-walk sensors
  models/
    checkpoints_voronoi_scattered/  ← checkpoints from scattered-sensor training
    checkpoints_voronoi_walk/       ← checkpoints from walk-sensor training
  results/
    model_comparison.txt
    model_scattered_test_scattered/
    model_scattered_test_walk/
    model_walk_test_scattered/
    model_walk_test_walk/
```

---

## Quick Start

### Training — epsilon-MSE only (default)
```bash
cd DDPM
pip install -r requirements.txt
python model/train.py --epochs 200 --batch 32 --pickle ../data.pickle
```

### Training — with structural losses
```bash
cd DDPM
# single auxiliary loss
python model/train.py --loss spectral --pickle ../data.pickle

# multiple losses with explicit weights
python model/train.py --loss spectral okubo_weiss --weights 0.0002 0.001 --pickle ../data.pickle

# all available modes
python model/train.py --loss curl_div spectral okubo_weiss --pickle ../data.pickle
```

### Inference (Basic DDPM)
```bash
cd DDPM
python testing/visualize_infer.py --checkpoint models/best_model.pt \
    --pickle ../data.pickle --sample 0 --path_steps 150 --resample 10
```

### Batch Evaluation
```bash
cd DDPM
python testing/batch_infer.py --checkpoint models/best_model.pt \
    --pickle ../data.pickle --n 10 --path_steps 150
```

### Noise Schedule Ablation — train all four schedules
```bash
bash "Model Parameters/NoiseSchedule/run_repaint.sh"
# or individually:
python3 "Model Parameters/NoiseSchedule/train_repaint.py" --schedule cosine --pickle data.pickle
```

### Noise Schedule — batch evaluation
```bash
python3 "Model Parameters/NoiseSchedule/batch_repaint.py" --schedule cosine --pickle data.pickle
```

### VoronoiNet — training
```bash
# scattered sensors (default)
python "Voronoi/model/train_voronoi.py" --pickle data.pickle --epochs 100

# robot walk sensors
python "Voronoi/model/train_voronoi.py" --sensor_mode walk --pickle data.pickle \
    --save_dir Voronoi/models/checkpoints_voronoi_walk
```

### VoronoiNet — evaluation
```bash
# single-sample test visualisation
python "Voronoi/testing/test_voronoi.py" --pickle data.pickle

# batch evaluation (scattered sensors)
python "Voronoi/testing/batch_voronoi.py" --pickle data.pickle

# batch evaluation (walk sensors)
python "Voronoi/testing/batch_voronoi_walk.py" --pickle data.pickle
```

---

## Results Summary

| Method | Mean RMSE | Std | Path coverage |
|---|---|---|---|
| DDPM (RePaint, r=10) | **0.1146** | 0.022 | ~3.8% (150-step biased walk) |
| GP (Matérn 2.5) | 0.2011 | 0.079 | ~3.5% (150-step biased walk) |
| NoiseSchedule ablation | TBD | TBD | 150-step biased walk |
| VoronoiNet (scattered) | TBD | TBD | ~50 random ocean sensors |
| VoronoiNet (walk) | TBD | TBD | ~150-step biased walk |

The DDPM's learned prior enables robust reconstruction even when the robot path misses the high-speed coastal jet — a scenario where the GP degrades to near-zero predictions and RMSE > 0.3.
