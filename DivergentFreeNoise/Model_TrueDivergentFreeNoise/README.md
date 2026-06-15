# Divergence-Free DDPM for Ocean Current Inpainting

This workspace now contains a PyTorch DDPM project for reconstructing 2D ocean current vector fields from sparse robot paths.

The project does three things:

1. Cleans the source pickle into a divergence-free version while preserving the land mask as `NaN`.
2. Trains a cosine-schedule DDPM on the cleaned fields with masked MSE loss over ocean cells only.
3. Uses divergence-free Gaussian noise during both diffusion training and RePaint-style inference.

## Layout

- `Basic DDPM/` contains the runnable project.
- `Basic DDPM/model/` contains the UNet, diffusion, plotting, metrics, and path utilities.
- `Basic DDPM/model_parameters/` contains the noise types, schedules, and loss helpers.
- `Basic DDPM/preprocess_divfree.py` rewrites `data.pickle` into a divergence-free version.
- `Basic DDPM/train.py` trains the model and writes checkpoints.
- `Basic DDPM/visualize_infer.py` generates actual/predicted/loss figures with vector arrows and robot path overlays.
- `Basic DDPM/batch_infer.py` runs batch evaluation.

## Quick Start

From inside `Basic DDPM/`:

```powershell
pip install -r requirements.txt
python preprocess_divfree.py --source ..\data.pickle --target ..\data_divfree.pickle
python train.py --epochs 200 --batch 32 --pickle ..\data_divfree.pickle
python visualize_infer.py --checkpoint checkpoints\best_model.pt --pickle ..\data_divfree.pickle --sample 0 --path_steps 150 --resample 10
python batch_infer.py --checkpoint checkpoints\best_model.pt --pickle ..\data_divfree.pickle --n 10 --path_steps 150
```

## Vast.ai

The provided SSH target can be used once the remote box is free:

```powershell
ssh -p 50919 root@220.82.52.202 -L 8080:localhost:8080
```

If the server is available, copy `Basic DDPM/` plus `data.pickle` or `data_divfree.pickle` before training.
