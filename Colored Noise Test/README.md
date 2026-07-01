# Colored Noise Test

Experiments comparing how the **spectral color of the diffusion noise prior** affects RePaint-based ocean current reconstruction. Six model variants are trained, each identical in architecture but differing only in how forward-process noise is generated.

---

## Models

| Folder | Noise type | Description |
|---|---|---|
| `white_noise/` | White | i.i.d. Gaussian; flat power spectrum S(f) ∝ f⁰ |
| `pink_noise/` | Pink | 1/f power spectrum (α = 1); moderate spatial correlation |
| `red_noise/` | Red | 1/f² power spectrum (α = 2); smooth, large-scale correlated |
| `pink_noise_full/` | Pink (full) | Pink noise applied at **all** timesteps (no annealing) |
| `red_noise_full/` | Red (full) | Red noise applied at **all** timesteps (no annealing) |
| `annealed_noise/` | Annealed | Spectral exponent α(t) = 2·t/T; red at t=T, white at t=0 |

All models share:
- **Architecture**: Repaint UNet, `base_ch=64`, `time_dim=256`, ~14.9 M parameters
- **Schedule**: Linear beta schedule, β₁ = 1×10⁻⁴ → β_T = 0.02, T = 1000
- **Loss**: ε-MSE + `0.002 × curl_div_loss`
- **Data**: `/root/model_pink_noise/data.pickle`, ocean current grid (94 × 44), split 2 = test

Each model subfolder contains:
```
<model>/
    diffusion.py       # DDPM class with noise-specific _make_noise()
    loss_functions.py  # curl_div_loss
    repaint_model.py   # Repaint wrapper
    train.py           # Training script
    checkpoints/
        best_model.pt  # Best validation-loss checkpoint
```

---

## Scripts

### Training

| Script | Purpose |
|---|---|
| `<model>/train.py` | Train a single model variant |
| `launch_full_noise_training.sh` | Launch all 6 training runs sequentially on the server |
| `wait_then_train_full_200.sh` | Wait for a prior job, then train for 200 epochs |

**Example:**
```bash
python "Colored Noise Test/white_noise/train.py" \
    --pickle /root/model_pink_noise/data.pickle \
    --resume "Colored Noise Test/white_noise/checkpoints/best_model.pt"
```

### Inference & Comparison

| Script | Layout | Purpose |
|---|---|---|
| `infer_compare.py` | 1 × 5 (GT \| White \| Pink \| Red \| RMSE bar) | Single-sample comparison across white/pink/red |
| `batch_infer_all.py` | 2 × 4 per sample | Batch inference across all 6 models |
| `batch_white_vs_red.py` | 1 × 3 (GT \| White \| Red) | White vs Red, N seeds |
| `batch_white_vs_annealed.py` | 1 × 3 (GT \| White \| Annealed) | White vs Annealed, N seeds |
| `batch_white_red_annealed.py` | 1 × 4 (GT \| White \| Red \| Annealed) | Combined three-way comparison, N seeds |

**Key inference parameters:**
- `--pickle` — path to `data.pickle`
- `--ckpt` — `best` (best validation loss) or `epoch100`
- `--out_dir` — output directory
- `--n_samples` — number of test samples (default 20)
- `--seed` — base random seed (default 42)
- `--r` — RePaint resampling rounds (default 1)
- `--stride` — timestep stride (default 10)
- `--n_steps` — RePaint denoising steps (default 150)

**Example (batch all-6 on server):**
```bash
python "Colored Noise Test/batch_infer_all.py" \
    --pickle /root/model_pink_noise/data.pickle \
    --ckpt best \
    --out_dir "Colored Noise Test/outputs/batch_best"
```

### Analysis & Plotting

| Script | Purpose |
|---|---|
| `plot_rmse_bars.py` | Bar charts for `batch_best` and `batch_epoch100` runs |
| `merge_r1_results.py` | Bar chart from `r1_combined/summary.txt` |
| `plot_comparison.py` | Additional comparison plots |

---

## Results

All outputs land in `outputs/`.

### `outputs/batch_best/` — all 6 models, best checkpoint, r=10, 10 samples
| Model | Mean RMSE |
|---|---|
| White | 0.117 |
| Pink | 0.198 |
| Red | 0.108 |
| Pink (full) | 0.213 |
| Red (full) | 0.137 |
| Annealed | 0.112 |

### `outputs/batch_epoch100/` — all 6 models, epoch-100 checkpoint, r=10, 10 samples
| Model | Mean RMSE |
|---|---|
| White | 0.132 |
| Pink | 0.219 |
| Red | 0.114 |
| Pink (full) | 0.232 |
| Red (full) | 0.131 |
| Annealed | 0.258 |

### `outputs/r1_combined/` — White / Red / Annealed, best checkpoint, r=1, 20 seeds
| Model | Mean RMSE |
|---|---|
| White | 0.108 |
| Red | 0.200 |
| Annealed | 0.428 |

Bar charts: `outputs/rmse_bars_best.png`, `outputs/rmse_bars_epoch100.png`, `outputs/rmse_bars_combined.png`, `outputs/r1_combined/rmse_bars_r1.png`

---

## Image Orientation

All output images are rendered **90° counter-clockwise** from the raw data array so that land appears at the top. In code, every `plot_field` call applies:

```python
land_mask = np.rot90(land_mask, k=3)
u_r       = np.rot90(u, k=3)
v_r       = np.rot90(v, k=3)
u         =  v_r   # component transform for CCW rotation
v         = -u_r
```

`k=3` (90° CW array rotation) produces a visual 90° CCW rotation when displayed with `origin="lower"`, moving land from the right edge to the top edge.


---

## Server Setup

Training and batch inference run on a vast.ai GPU instance.

```
Host : 74.48.140.178   Port : 29628
Key  : ~/.ssh/vastai_new_key
Env  : /root/ocean_ddpm/venv
Data : /root/model_pink_noise/data.pickle
Code : /root/DiffusionSummer2026/
```

Upload a script:
```bash
scp -P 29628 -i ~/.ssh/vastai_new_key \
    "Colored Noise Test/batch_white_red_annealed.py" \
    root@74.48.140.178:"/root/DiffusionSummer2026/Colored Noise Test/"
```
