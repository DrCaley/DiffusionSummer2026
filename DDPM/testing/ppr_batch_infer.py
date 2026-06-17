"""
Batch comparison: RePaint vs PPR over N validation samples.

Reports per-run and aggregate:
  - RMSE (ocean cells)
  - Mean |divergence| over ocean cells  ← headline metric; PPR should be ~0
  - Radially-averaged KE spectral error (low vs high wavenumber split)

Usage (from workspace root):
    python DDPM/testing/ppr_batch_infer.py --checkpoint DDPM/checkpoints/best_model.pt
    python DDPM/testing/ppr_batch_infer.py --n_runs 5 --path_steps 150 --resample 5
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_here  = os.path.dirname(os.path.abspath(__file__))
_root  = os.path.join(_here, "..", "..")
_model = os.path.join(_here, "..", "model")
_repaint = os.path.join(_here, "repaint")
_ppr   = os.path.join(_here, "ppr")
for _p in [_root, os.path.join(_root, "utils"), _model, _repaint, _ppr]:
    sys.path.insert(0, _p)

from dataset              import OceanCurrentDataset
from diffusion            import DDPM
from model                import UNet
from divfree_projection   import divergence as compute_divergence
from repaint_infer        import biased_walk_path, repaint
from ppr_infer            import ppr


# ---------------------------------------------------------------------------
# Spectral metric helper
# ---------------------------------------------------------------------------

def _ke_spectrum_error(pred_np: np.ndarray, true_np: np.ndarray,
                       ocean_mask_np: np.ndarray) -> tuple[float, float]:
    """
    Radially-averaged kinetic energy power-spectrum error.

    Returns
    -------
    low_err  : mean absolute spectrum error at |k| < k_med  (large scales)
    high_err : mean absolute spectrum error at |k| >= k_med (small scales)
    """
    def _ke_spec(uv_np):
        # Zero land, compute per-channel rfft2
        ocean_f = (~ocean_mask_np).astype(np.float32)
        u = uv_np[0] * ocean_f
        v = uv_np[1] * ocean_f
        H, W = u.shape
        fu = np.fft.rfft2(u)
        fv = np.fft.rfft2(v)
        ke = (np.abs(fu) ** 2 + np.abs(fv) ** 2) / 2.0   # (H, W//2+1)

        # Radial wavenumber grid
        kx = np.fft.fftfreq(H)[:, None]
        ky = np.fft.rfftfreq(W)[None, :]
        k  = np.sqrt(kx ** 2 + ky ** 2)                   # (H, W//2+1)

        # Bin into N_bins radial shells
        N_bins = min(H, W) // 2
        k_max  = k.max()
        bins   = np.linspace(0, k_max, N_bins + 1)
        spec   = np.zeros(N_bins)
        for i in range(N_bins):
            mask = (k >= bins[i]) & (k < bins[i + 1])
            if mask.any():
                spec[i] = ke[mask].mean()
        return spec, bins

    spec_pred, bins = _ke_spec(pred_np)
    spec_true, _    = _ke_spec(true_np)

    diff    = np.abs(spec_pred - spec_true)
    k_mid   = len(diff) // 2
    return float(diff[:k_mid].mean()), float(diff[k_mid:].mean())


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def _run_one(model, diffusion, val_ds, land_mask_np, sample_idx, seed, args, device,
             method: str, data_mean=None, data_std=None):
    x0_true = val_ds[sample_idx]                       # (2, H, W) clean field
    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

    x0_observed = x0_true.clone()
    x0_observed[:, ~torch.from_numpy(path_mask)] = 0.0

    # Normalize to training scale if checkpoint was trained with --normalize
    _normalize = data_mean is not None
    x0_obs_infer = ((x0_observed - data_mean) / data_std) if _normalize else x0_observed

    if method == "repaint":
        x0_pred_norm = repaint(
            model, diffusion, x0_obs_infer, path_mask, land_mask_np,
            r=args.resample, device=device, inference_steps=args.inference_steps,
        )
    else:  # ppr
        x0_pred_norm = ppr(
            model, diffusion, x0_obs_infer, path_mask, land_mask_np,
            r=args.ppr_resample, proj_iter=args.proj_iter, device=device,
            inference_steps=args.inference_steps,
        )

    # Denormalize prediction back to original units
    x0_pred = (x0_pred_norm * data_std + data_mean) if _normalize else x0_pred_norm

    # --- RMSE (ocean cells) ---
    u_pred, v_pred = x0_pred[0].numpy(),  x0_pred[1].numpy()
    u_true, v_true = x0_true[0].numpy(),  x0_true[1].numpy()
    ocean          = ~land_mask_np
    err_sq         = (u_pred - u_true) ** 2 + (v_pred - v_true) ** 2
    rmse           = float(np.sqrt(err_sq[ocean].mean()))

    # --- Mean |divergence| (ocean cells) ---
    ocean_mask_t = torch.from_numpy(ocean)
    x0_pred_t    = x0_pred.unsqueeze(0)                     # (1, 2, H, W)
    div          = compute_divergence(x0_pred_t, ocean_mask_t)  # (1, H, W)
    mean_div     = float(div[0][ocean_mask_t].abs().mean().item())

    # --- Spectral error ---
    low_err, high_err = _ke_spectrum_error(x0_pred.numpy(), x0_true.numpy(), land_mask_np)

    return rmse, mean_div, low_err, high_err, int(path_mask.sum())


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch comparison: RePaint vs PPR"
    )
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--checkpoint", default="DDPM/checkpoints/best_model.pt")
    p.add_argument("--n_runs",     type=int, default=10,  help="runs per method")
    p.add_argument("--path_steps",   type=int, default=150)
    p.add_argument("--resample",     type=int, default=10, help="RePaint resamples per step")
    p.add_argument("--ppr_resample", type=int, default=1,  help="PPR resamples per step (1 = single pass)")
    p.add_argument("--proj_iter",    type=int, default=20, help="POCS iterations")
    p.add_argument("--T",            type=int, default=1000)
    p.add_argument("--base_ch",      type=int, default=64)
    p.add_argument("--time_dim",     type=int, default=256)
    p.add_argument("--inference_steps", type=int, default=100,
                   help="Denoising steps at inference (default 100).")
    p.add_argument("--device",       default=None, help="cuda / mps / cpu (auto-detect if omitted)")
    p.add_argument("--out_dir",      default="DDPM/best_model_results/ppr_batch")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device : {device}")
    print(f"Runs   : {args.n_runs}  path_steps={args.path_steps}  "
          f"resample={args.resample}  proj_iter={args.proj_iter}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Data ----
    val_ds        = OceanCurrentDataset(args.pickle, split=1)
    land_mask_np  = val_ds.land_mask.numpy()

    # ---- Model ----
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch    = ckpt_args.get("base_ch",   args.base_ch)
    time_dim   = ckpt_args.get("time_dim",  args.time_dim)
    T          = ckpt_args.get("T",         args.T)
    noise_type       = ckpt_args.get("noise_type", "gaussian")
    spectral_filter  = ckpt.get("spectral_filter", None)
    data_mean        = ckpt.get("data_mean", None)
    data_std         = ckpt.get("data_std",  None)

    model = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sf_str   = "yes" if spectral_filter is not None else "no"
    norm_str = f"mean={data_mean:.4f} std={data_std:.4f}" if data_mean is not None else "no"
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}), T={T}, noise={noise_type}, "
          f"spectral_filter={sf_str}, normalize={norm_str}")

    diffusion = DDPM(T=T, beta_schedule="cosine", device=device, noise_type=noise_type,
                     spectral_filter=spectral_filter)

    # ---- Run both methods ----
    results = {"repaint": [], "ppr": []}

    header = f"{'Run':>4}  {'Sample':>7}  {'Seed':>6}  {'Cells':>6}  " \
             f"{'RMSE':>8}  {'|div|':>9}  {'SpecLo':>8}  {'SpecHi':>8}"

    for method in ("repaint", "ppr"):
        print(f"\n{'=' * 70}")
        print(f"  Method: {method.upper()}")
        print(f"{'=' * 70}")
        print(header)

        for i in range(args.n_runs):
            sample_idx = i % len(val_ds)
            seed       = i * 7 + 1

            rmse, mean_div, low_err, high_err, path_cells = _run_one(
                model, diffusion, val_ds, land_mask_np,
                sample_idx, seed, args, device, method,
                data_mean=data_mean, data_std=data_std,
            )
            results[method].append((rmse, mean_div, low_err, high_err))

            print(
                f"{i+1:>4}  {sample_idx:>7}  {seed:>6}  {path_cells:>6}  "
                f"{rmse:>8.4f}  {mean_div:>9.6f}  {low_err:>8.4f}  {high_err:>8.4f}"
            )

    # ---- Summary table ----
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY  ({args.n_runs} runs each)")
    print(f"{'=' * 70}")
    row_fmt = f"  {{:<10}}  {{:>8}}  {{:>8}}  {{:>9}}  {{:>8}}  {{:>8}}"
    print(row_fmt.format("Method", "RMSE", "Std", "|div|", "SpecLo", "SpecHi"))
    print(f"  {'-'*64}")

    for method in ("repaint", "ppr"):
        arr = np.array(results[method])        # (N, 4)
        rmse_m, div_m, lo_m, hi_m = arr.mean(0)
        rmse_s                     = arr[:, 0].std()
        print(row_fmt.format(
            method,
            f"{rmse_m:.4f}", f"{rmse_s:.4f}",
            f"{div_m:.6f}", f"{lo_m:.4f}", f"{hi_m:.4f}",
        ))

    print()


if __name__ == "__main__":
    main()
