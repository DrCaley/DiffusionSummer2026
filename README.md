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

### Basic DDPM (`Basic DDPM/`)
Baseline unconditional DDPM:
- **UNet** with sinusoidal timestep embeddings and ResNet blocks, 14.9 M parameters, input (B, 2, 94, 44) padded internally to (96, 48).
- **Cosine noise schedule**, T = 1000 steps, epsilon-prediction, MSE loss masked to ocean pixels.
- **RePaint inference**: robot path observations are blended in at every reverse-diffusion step with r=10 resamplings per timestep.

Best val loss: **0.00327** (epoch 155/200). Mean RMSE on 10 val samples: **0.1146 ± 0.022**.


### GP Baseline (`GP Baseline/`)
Two independent Gaussian Processes (Matérn ν=2.5 + WhiteKernel), one per velocity component, fit directly to the robot path observations at inference time. No training phase. Mean RMSE on 10 val samples: **0.2011 ± 0.079** — ~75% higher than the DDPM.

---

---

## Quick Start

### Training (Basic DDPM)
```bash
cd "Basic DDPM"
pip install -r requirements.txt
python train.py --epochs 200 --batch 32 --pickle ../data.pickle
```

### Inference (Basic DDPM)
```bash
cd "Basic DDPM"
python visualize_infer.py --checkpoint checkpoints/best_model.pt \
    --pickle ../data.pickle --sample 0 --path_steps 150 --resample 10
```

### Batch Evaluation
```bash
python batch_infer.py --checkpoint checkpoints/best_model.pt \
    --pickle ../data.pickle --n 10 --path_steps 150
```

---

## Results Summary

| Method | Mean RMSE | Std | Path coverage |
|---|---|---|---|
| DDPM (RePaint, r=10) | **0.1146** | 0.022 | ~3.8% (150-step biased walk) |
| GP (Matérn 2.5) | 0.2011 | 0.079 | ~3.5% (150-step biased walk) |

The DDPM's learned prior enables robust reconstruction even when the robot path misses the high-speed coastal jet — a scenario where the GP degrades to near-zero predictions and RMSE > 0.3.
