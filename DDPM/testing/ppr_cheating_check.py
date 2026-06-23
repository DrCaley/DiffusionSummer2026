"""
Diagnostic: is PPR-old "cheating" by predicting a near-constant climatology field?

Tests on the OLD 7% model (data.pickle), the suspicious low-RMSE config.

For a set of test samples we compute:
  1. PPR RMSE                       — actual reconstruction error
  2. Climatology RMSE               — error of predicting the TRAIN-MEAN field
                                      (the trivial "ignore observations" baseline)
  3. Zero RMSE                      — error of predicting zeros (sanity floor)
  4. Prediction diversity vs GT     — std of PPR predictions across samples vs
                                      std of ground truths across samples. If PPR
                                      outputs ~identical fields it is climatology.
  5. Path-swap control              — feed sample i's OBSERVATIONS but score
                                      against sample j's GT. If PPR genuinely uses
                                      the observations, swapped RMSE >> matched RMSE.

If PPR RMSE << climatology RMSE AND swapped >> matched, PPR is genuinely using the
sparse observations (not cheating).
"""

import os
import sys
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..")
for _p in [_root, os.path.join(_root, "utils"),
           os.path.join(_here, "..", "model"),
           os.path.join(_here, "repaint"), os.path.join(_here, "ppr")]:
    sys.path.insert(0, _p)

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from model         import UNet
from repaint_infer import biased_walk_path
from ppr_infer     import ppr

CKPT    = "Models/Div_Free_DDPM_7%.pt"
PICKLE  = "Datasets/data.pickle"
SAMPLES = [0, 5, 12, 3, 7]      # test-split indices
SEEDS   = {0: 0, 5: 5, 12: 12, 3: 3, 7: 7}
STEPS   = 200                   # fewer inference steps for speed (fair across all)
PROJ_IT = 20

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ---- Data ----
test_ds   = OceanCurrentDataset(PICKLE, split=2)
train_ds  = OceanCurrentDataset(PICKLE, split=0)
land_np   = test_ds.land_mask.numpy()
ocean     = ~land_np
ocean_t   = torch.from_numpy(ocean)

# Climatology = mean field over the TRAIN split (per-cell mean of u, v)
clim = train_ds.data.mean(dim=0)        # (2, H, W)
clim[:, ~ocean_t] = 0.0

# ---- Model ----
ckpt      = torch.load(CKPT, map_location=device, weights_only=False)
ckpt_args = ckpt.get("args", {})
T         = ckpt_args.get("T", 1000)
noise_type      = ckpt_args.get("noise_type", "gaussian")
spectral_filter = ckpt.get("spectral_filter", None)
data_mean = ckpt.get("data_mean", None)
data_std  = ckpt.get("data_std", None)

model = UNet(in_ch=2, base_ch=ckpt_args.get("base_ch", 64),
             time_dim=ckpt_args.get("time_dim", 256)).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
diffusion = DDPM(T=T, beta_schedule="cosine", device=device,
                 noise_type=noise_type, spectral_filter=spectral_filter)
print(f"Model: epoch {ckpt.get('epoch','?')}, noise={noise_type}, "
      f"normalize={'yes' if data_mean is not None else 'no'}, steps={STEPS}\n")


def rmse(a, b):
    d = (a - b)[:, ocean]
    return float(np.sqrt((d ** 2).mean()))


def run_ppr(sample_idx, path_mask):
    """Return PPR prediction (2,H,W) physical units for given sample + path."""
    x0_true = test_ds[sample_idx]
    x0_obs  = x0_true.clone()
    x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0
    x0_in   = ((x0_obs - data_mean) / data_std) if data_mean is not None else x0_obs
    pred = ppr(model, diffusion, x0_in, path_mask, land_np,
               r=1, proj_iter=PROJ_IT, device=device,
               inference_steps=STEPS, data_mean=data_mean, data_std=data_std,
               projector="pocs")
    if data_mean is not None:
        pred = pred * data_std + data_mean
    pred[:, ~ocean_t] = 0.0
    return pred.numpy()


# ---- Per-sample PPR + baselines ----
gts, preds, paths = {}, {}, {}
print(f"{'sample':>6} | {'PPR':>8} | {'climatology':>11} | {'zeros':>8}")
print("-" * 44)
clim_np = clim.numpy()
zeros_np = np.zeros_like(clim_np)
for s in SAMPLES:
    pm = biased_walk_path(land_np, n_steps=150, seed=SEEDS[s])
    gt = test_ds[s].numpy()
    pr = run_ppr(s, pm)
    gts[s], preds[s], paths[s] = gt, pr, pm
    print(f"{s:>6} | {rmse(pr, gt):>8.4f} | {rmse(clim_np, gt):>11.4f} | {rmse(zeros_np, gt):>8.4f}")

ppr_mean  = np.mean([rmse(preds[s], gts[s]) for s in SAMPLES])
clim_mean = np.mean([rmse(clim_np, gts[s]) for s in SAMPLES])
print("-" * 44)
print(f"  MEAN | {ppr_mean:>8.4f} | {clim_mean:>11.4f} |")

# ---- Diversity: std across samples (ocean cells) ----
gt_stack   = np.stack([gts[s][:, ocean] for s in SAMPLES])     # (S, 2, n_ocean)
pred_stack = np.stack([preds[s][:, ocean] for s in SAMPLES])
gt_div   = float(gt_stack.std(axis=0).mean())
pred_div = float(pred_stack.std(axis=0).mean())
print(f"\nCross-sample diversity (std over samples, ocean cells):")
print(f"  ground truth : {gt_div:.4f}")
print(f"  PPR preds    : {pred_div:.4f}   "
      f"(ratio pred/gt = {pred_div/gt_div:.2f}; ~1 = as diverse as data, ~0 = constant)")

# ---- Path-swap control: sample 0's observations, sample 5's GT ----
print(f"\nPath-swap control (does the prediction follow the OBSERVATIONS?):")
print(f"  matched  : PPR(obs=s0) vs GT s0  = {rmse(preds[0], gts[0]):.4f}")
swapped = rmse(preds[0], gts[5])
print(f"  swapped  : PPR(obs=s0) vs GT s5  = {swapped:.4f}")
print(f"  -> swapped should be MUCH larger if PPR uses obs; "
      f"ratio swapped/matched = {swapped/rmse(preds[0], gts[0]):.2f}")
