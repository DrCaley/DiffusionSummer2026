"""
repaint_infer_fno.py
====================
RePaint-r1 inference using the trained FNO autoencoder as a diffusion prior.

The FNO was trained as an autoencoder (clean field → clean field).  At each
denoising step we use it as an *x0-predictor*:

    x0_pred  = FNO(x_t)            # project noisy sample to clean manifold
    ε_pred   = (x_t − √ᾱ_t · x0_pred) / √(1−ᾱ_t)   # convert to noise pred

This is then fed into the standard DDPM posterior (p_sample_step) and the
RePaint known-pixel merge.

Usage (from NeuralOperator/ directory on the remote):
    python3 repaint_infer_fno.py \\
        --ckpt        ./checkpoints/best_fno.pth \\
        --data-path   /root/ocean_ddpm/data_local.pickle \\
        --n-samples   100 \\
        --T           1000 \\
        --n-steps     200 \\
        --walk-steps  150 \\
        --out         ./repaint_results.npz
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

# ── local imports ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from dataset import PickleFieldDataset
from model_fno import FNO2d

# Reuse DDPM schedule + repaint helpers — check several candidate dirs
def _find_diffusion_dir():
    candidates = [
        os.path.join(SCRIPT_DIR, '..', 'Stride'),
        os.path.join(SCRIPT_DIR, '..', 'Repaint_vs_DPS'),
        '/root/Repaint_vs_DPS',
        '/root/Stride',
    ]
    for d in candidates:
        d = os.path.abspath(d)
        if os.path.isfile(os.path.join(d, 'diffusion.py')):
            return d
    raise RuntimeError('Cannot find diffusion.py — tried: ' + str(candidates))

_diff_dir = _find_diffusion_dir()
print(f'Using diffusion helpers from: {_diff_dir}')
sys.path.insert(0, _diff_dir)
from diffusion import DDPM
from repaint_infer import random_walk_path


# ─────────────────────────────────────────────────────────────────────────────
# FNO wrapper: makes the autoencoder look like a DDPM noise predictor
# ─────────────────────────────────────────────────────────────────────────────

class FNOx0Predictor(nn.Module):
    """
    Wraps a trained FNO (H,W,C) autoencoder so that it can be called as:
        model(xt, t) -> pred_noise   (B, C, H, W)
    which is the interface expected by DDPM.p_sample_step.

    Internally:
        x0_pred  = FNO(xt permuted to (B,H,W,C), permute back)
        eps_pred = (xt − sqrt(ab_t) · x0_pred) / sqrt(1−ab_t)
    """

    def __init__(self, fno: nn.Module, alpha_bar: torch.Tensor):
        super().__init__()
        self.fno = fno
        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # xt: (B, C, H, W);  t: (B,) long
        x_hwc   = xt.permute(0, 2, 3, 1)            # (B, H, W, C)
        x0_pred = self.fno(x_hwc).permute(0, 3, 1, 2)  # (B, C, H, W)
        x0_pred = x0_pred.clamp(-1.5, 1.5)

        ab = self.alpha_bar[t][:, None, None, None]  # (B,1,1,1)
        eps = (xt - ab.sqrt() * x0_pred) / (1.0 - ab).clamp(min=1e-8).sqrt()
        return eps


# ─────────────────────────────────────────────────────────────────────────────
# Minimal RePaint r=1 (no resampling)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def repaint_r1(
    model:      nn.Module,
    diffusion:  DDPM,
    x0_known:   torch.Tensor,   # (1, C, H, W)  known values at path (0 elsewhere)
    path_mask:  torch.Tensor,   # (1, 1, H, W)  float, 1 = known
    ocean_mask: torch.Tensor,   # (1, 1, H, W)  float, 1 = ocean
    timesteps:  list,
    device:     str,
) -> torch.Tensor:
    """Single-pass RePaint (r=1) using strided timesteps."""
    C, H, W = x0_known.shape[1], x0_known.shape[2], x0_known.shape[3]
    xt = torch.randn(1, C, H, W, device=device) * ocean_mask

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        # 1. Model reverse step for unknown pixels
        xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)

        # 2. Forward-diffuse x0_known to t_prev
        t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
        xt_known, _ = diffusion.q_sample(x0_known, t_prev_t)

        # 3. Merge: known path uses diffused truth, unknown uses model
        xt = path_mask * xt_known + (1.0 - path_mask) * xt_unknown
        xt = xt * ocean_mask  # keep land at 0

    return xt.squeeze(0).cpu()  # (C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        type=str,   default='./checkpoints/best_fno.pth')
    parser.add_argument('--data-path',   type=str,   default='/root/ocean_ddpm/data_local.pickle')
    parser.add_argument('--n-samples',   type=int,   default=100)
    parser.add_argument('--T',           type=int,   default=1000,
                        help='DDPM timesteps')
    parser.add_argument('--n-steps',     type=int,   default=200,
                        help='Number of inference steps (stride = T/n-steps)')
    parser.add_argument('--walk-steps',  type=int,   default=150,
                        help='Random walk path length')
    parser.add_argument('--val-frac',    type=float, default=0.1)
    parser.add_argument('--out',         type=str,   default='./repaint_results.npz')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Load dataset (val split for evaluation) ───────────────────────────────
    print('Loading dataset...')
    val_ds = PickleFieldDataset(args.data_path, split='val', val_fraction=args.val_frac)
    print(f'Val samples: {len(val_ds)}')

    H, W, C = val_ds[0].shape
    land_mask_hwc = val_ds.land_mask          # (H, W, C) bool
    land_mask_hw  = land_mask_hwc[..., 0]     # (H, W) bool (same for all channels)
    print(f'Field shape: H={H} W={W} C={C}')

    # ── Load trained FNO ──────────────────────────────────────────────────────
    print(f'Loading FNO from {args.ckpt} ...')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    fno  = FNO2d(in_channels=C, out_channels=C).to(device)
    fno.load_state_dict(ckpt['model_state'])
    fno.eval()

    # ── DDPM with linear schedule ─────────────────────────────────────────────
    diffusion = DDPM(T=args.T, beta_schedule='linear', device=device,
                     curl_div_weight=0.0)

    # ── Wrap FNO as noise predictor ───────────────────────────────────────────
    model = FNOx0Predictor(fno, diffusion.alpha_bar).to(device)
    model.eval()

    # ── Strided timestep list ─────────────────────────────────────────────────
    stride    = max(1, args.T // args.n_steps)
    timesteps = list(range(0, args.T, stride))
    print(f'Inference steps: {len(timesteps)} (stride={stride})')

    # ── Masks as tensors ─────────────────────────────────────────────────────
    ocean_t = torch.from_numpy(~land_mask_hw).float()[None, None].to(device)  # (1,1,H,W)

    # ── Evaluate over n_samples ───────────────────────────────────────────────
    rmse_list  = []
    rng        = np.random.default_rng(42)

    print(f'Running {args.n_samples} RePaint-r1 samples...')
    for i in range(args.n_samples):
        sample_idx = int(rng.integers(len(val_ds)))
        field_hwc  = val_ds[sample_idx]                       # (H, W, C) float32
        x0_chw     = torch.from_numpy(field_hwc).permute(2, 0, 1)  # (C, H, W)

        # Random walk path (seed = i)
        path_hw = random_walk_path(land_mask_hw, n_steps=args.walk_steps, seed=i)

        # Build x0_known: true values at path cells, 0 elsewhere
        x0_known_chw = x0_chw * torch.from_numpy(path_hw).float()  # (C, H, W)
        x0_known_t   = x0_known_chw.unsqueeze(0).to(device)        # (1, C, H, W)

        path_t = torch.from_numpy(path_hw).float()[None, None].to(device)  # (1,1,H,W)

        # Run RePaint r=1
        x0_pred_chw = repaint_r1(
            model, diffusion,
            x0_known   = x0_known_t,
            path_mask  = path_t,
            ocean_mask = ocean_t,
            timesteps  = timesteps,
            device     = device,
        )  # (C, H, W)

        # RMSE on ocean cells (excluding land and path — evaluate unseen regions)
        ocean_np  = ~land_mask_hw                          # (H, W)
        unknown   = ocean_np & ~path_hw                    # (H, W) — cells to evaluate

        gt   = x0_chw.numpy()                              # (C, H, W)
        pred = x0_pred_chw.numpy()                         # (C, H, W)

        diff = (pred - gt)[:, unknown]                     # (C, n_ocean_unknown)
        rmse = float(np.sqrt((diff ** 2).mean()))
        rmse_list.append(rmse)

        if (i + 1) % 10 == 0:
            running_mean = np.mean(rmse_list)
            print(f'  [{i+1:3d}/{args.n_samples}] RMSE={rmse:.6f}  running_mean={running_mean:.6f}')

    rmse_arr = np.array(rmse_list)
    print(f'\n=== Results ({args.n_samples} samples, RePaint r=1) ===')
    print(f'  Mean RMSE : {rmse_arr.mean():.6f}')
    print(f'  Std  RMSE : {rmse_arr.std():.6f}')
    print(f'  Min  RMSE : {rmse_arr.min():.6f}')
    print(f'  Max  RMSE : {rmse_arr.max():.6f}')

    np.savez(args.out, rmse=rmse_arr)
    print(f'Saved results to {args.out}')


if __name__ == '__main__':
    main()
