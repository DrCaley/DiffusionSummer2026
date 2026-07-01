# Repaint vs DPS

Comparison of diffusion-based inpainting methods for ocean current reconstruction. A pre-trained DDPM is given sparse satellite-track observations (a biased random-walk path mask) and asked to reconstruct the full 2-channel (u, v) velocity field. Three algorithmic families are benchmarked across several hyperparameter settings.

---

## Methods

| Label | Description |
|---|---|
| **RePaint r=10** | Standard RePaint — 10 resampling passes per diffusion step |
| **RePaint r=1** | RePaint with a single UNet call per step (no resampling loop) |
| **DPS ζ=0.5 / 0.04** | Diffusion Posterior Sampling (Chung et al. 2022) — gradient step on $x_t$ toward the observation each step. ζ controls step size. |
| **RePaint+DPS r=10/r=1, ζ=0.5/0.04** | Hybrid: RePaint resampling loop *plus* a DPS gradient correction |

**DPS update rule:**

$$x_{t-1} \leftarrow \hat{x}_{t-1} - \frac{\zeta}{\|r\|} \nabla_{x_t} \|r\|^2, \quad r = \text{mask}(\hat{x}_{0|t} - y)$$

---

## Model

| Field | Value |
|---|---|
| Architecture | Repaint UNet (`base_ch=64`, `time_dim=256`) |
| Checkpoint | `best_model_linear.pt` — epoch 147, val_loss = 0.00019 |
| Schedule | Linear beta, `noise_std = 0.11619` |
| Data | `OceanCurrentDataset`, test split = 2 |
| Path mask | `biased_walk_path`, 150 steps, seeded per sample |

---

## Scripts

| Script | Purpose |
|---|---|
| `run_all_methods.py` | **Main entry point.** Runs all 8 method configs in one pass. Saves `summary.txt`, `results.csv`, `bar_chart.png`. |
| `compare_methods.py` | Runs 4 methods (RePaint r=10/r=1, DPS, RePaint+DPS r=10) with a configurable ζ. |
| `run_repaint_compare.py` | RePaint r=10 and r=1 only. |
| `run_rpdps_r1.py` | RePaint+DPS r=1 only (no resampling loop), supports `--dps_step`. |

### `run_all_methods.py` usage

```bash
python run_all_methods.py \
    --pickle   /path/to/data.pickle \
    --checkpoint /path/to/best_model_linear.pt \
    --T 1000 --stride 1 \
    --n_seeds 2 \
    --out_dir  results/all8_T1000_s1
```

| Argument | Default | Description |
|---|---|---|
| `--T` | 1000 | Total diffusion timesteps |
| `--stride` | 1 | Step size through the reverse process |
| `--n_seeds` | 20 | Number of test samples to evaluate |
| `--out_dir` | required | Output directory |

---

## Results (T=1000, stride=10, 20 seeds)

Best results across all runs; lower RMSE is better.

| Method | Mean RMSE | Mean Time (s) |
|---|---|---|
| DPS ζ=0.04 | **0.0499** | 3.3 |
| RePaint+DPS r=1 ζ=0.04 | 0.0508 | 3.3 |
| RePaint+DPS r=10 ζ=0.04 | 0.0618 | 33.6 |
| RePaint+DPS r=10 ζ=0.5 | 0.0619 | 37.5 |
| RePaint r=10 | 0.0752 | 8.7 |
| RePaint r=1 | 0.0772 | 1.2 |
| DPS ζ=0.5 | 0.1033 | 3.7 |
| RePaint+DPS r=1 ζ=0.5 | 0.1139 | 3.4 |

**Key findings:**
- DPS with a small step size (ζ=0.04) gives the best reconstruction accuracy.
- RePaint+DPS r=1 ζ=0.04 matches DPS accuracy with similar speed — the resampling loop adds no benefit at r=1.
- Large ζ=0.5 degrades both DPS and hybrid methods significantly.
- RePaint r=1 is the fastest option (1.2 s/sample) with moderate accuracy.

---

## Outputs

```
outputs/
├── T1000_bar_chart.png       # RMSE bar chart across all methods
├── T1000_summary.txt         # Detailed per-seed results for T=1000 runs
├── T1000/                    # Per-method result folders (stride=10 runs)
├── T1000_4methods/
├── repaint_T1000_s10/
└── result_01_all_strides.png
```

Each run folder contains `summary.txt`, `results.csv`, and comparison images for seed 0.
