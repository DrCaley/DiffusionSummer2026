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
from divfree_projection   import divergence as compute_divergence, joint_project
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
             method: str, clim=None, data_mean=None, data_std=None):
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
            inference_steps=args.inference_steps, projector=args.projector,
            data_mean=data_mean, data_std=data_std,
        )

    # Optional post-hoc data-consistency projection (applied ONCE to the final
    # prediction). Cleans the divergence seam without the per-step energy drift
    # of in-loop PPR. n_iter=0 = off. Operates in the model's (normalized) space.
    if args.final_project > 0:
        ocean_mask_t = torch.from_numpy(~land_mask_np)
        obs_mask_t   = torch.from_numpy(path_mask)
        x0_pred_norm = joint_project(
            x0_pred_norm.unsqueeze(0), ocean_mask_t, obs_mask_t,
            x0_obs_infer.unsqueeze(0), n_iter=args.final_project,
        ).squeeze(0)

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

    # --- Anomaly metrics (true flow-structure skill, DC-offset removed) ---
    # The raw speed ratio is fooled by the data mean: a blank normalized field
    # denormalizes to a constant offset with nonzero "speed" (~climatology).
    # Subtracting the climatology field isolates the per-sample FLUCTUATION that
    # the model must actually reconstruct.
    #   AnomRatio = RMS(pred anomaly) / RMS(true anomaly)  (~1 = right energy)
    #   ACC       = anomaly correlation coeff pred vs true (1 = perfect pattern,
    #               0 = no skill / climatology, <0 = anti-correlated)
    if clim is not None:
        pa = (x0_pred.numpy() - clim)[:, ocean].reshape(-1)   # (2*n_ocean,)
        ta = (x0_true.numpy() - clim)[:, ocean].reshape(-1)
        anom_ratio = float(np.sqrt((pa ** 2).mean()) / (np.sqrt((ta ** 2).mean()) + 1e-12))
        denom      = np.sqrt((pa ** 2).sum()) * np.sqrt((ta ** 2).sum()) + 1e-12
        acc        = float((pa * ta).sum() / denom)
    else:
        anom_ratio, acc = float("nan"), float("nan")

    # --- Spectral error ---
    low_err, high_err = _ke_spectrum_error(x0_pred.numpy(), x0_true.numpy(), land_mask_np)

    return rmse, mean_div, anom_ratio, acc, low_err, high_err, int(path_mask.sum())


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
    p.add_argument("--projector",    default="pocs", choices=["pocs", "snap_x0"],
                   help="PPR data-consistency projector (pocs = joint div-free+obs; snap_x0 = obs-only on x0-hat)")
    p.add_argument("--final_project", type=int, default=0,
                   help="POCS iters applied ONCE to the final prediction (0=off). "
                        "Cleans divergence post-hoc without per-step energy drift.")
    p.add_argument("--methods", nargs="+", default=["repaint", "ppr"],
                   choices=["repaint", "ppr"], help="which methods to run")
    p.add_argument("--random", action="store_true",
                   help="draw n_runs random distinct val samples (seeded by --seed) instead of 0,1,2,...")
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for --random sample selection")
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

    # ---- Sample indices (random distinct, or sequential) ----
    if args.random:
        rng         = np.random.default_rng(args.seed)
        sample_idxs = rng.choice(len(val_ds), size=min(args.n_runs, len(val_ds)),
                                 replace=False).tolist()
    else:
        sample_idxs = [i % len(val_ds) for i in range(args.n_runs)]
    print(f"Val samples ({'random' if args.random else 'sequential'}): {sample_idxs}")

    # ---- Climatology baseline (train-mean field) for skill reference ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    ocean    = ~land_mask_np
    clim     = train_ds.data.mean(dim=0).numpy()      # (2, H, W)
    clim[:, ~ocean] = 0.0

    # ---- Run methods ----
    results = {"repaint": [], "ppr": []}

    header = f"{'Run':>4}  {'Sample':>7}  {'Seed':>6}  {'Cells':>6}  " \
             f"{'RMSE':>8}  {'|div|':>9}  {'AnomRat':>8}  {'ACC':>7}  {'SpecLo':>8}  {'SpecHi':>8}"

    clim_rmses = []
    for method in args.methods:
        print(f"\n{'=' * 88}")
        print(f"  Method: {method.upper()}")
        print(f"{'=' * 88}")
        print(header)

        for i, sample_idx in enumerate(sample_idxs):
            seed       = i * 7 + 1

            rmse, mean_div, anom_ratio, acc, low_err, high_err, path_cells = _run_one(
                model, diffusion, val_ds, land_mask_np,
                sample_idx, seed, args, device, method, clim=clim,
                data_mean=data_mean, data_std=data_std,
            )
            results[method].append((rmse, mean_div, anom_ratio, acc, low_err, high_err))

            if method == args.methods[0]:
                gt        = val_ds[sample_idx].numpy()
                cr        = float(np.sqrt((((clim - gt) ** 2).sum(0))[ocean].mean()))
                clim_rmses.append(cr)

            print(
                f"{i+1:>4}  {sample_idx:>7}  {seed:>6}  {path_cells:>6}  "
                f"{rmse:>8.4f}  {mean_div:>9.6f}  {anom_ratio:>8.3f}  {acc:>7.3f}  "
                f"{low_err:>8.4f}  {high_err:>8.4f}"
            )

    # ---- Summary table ----
    print(f"\n{'=' * 88}")
    print(f"  SUMMARY  ({len(sample_idxs)} runs each)")
    print(f"{'=' * 88}")
    row_fmt = f"  {{:<11}}  {{:>8}}  {{:>8}}  {{:>9}}  {{:>8}}  {{:>7}}  {{:>8}}  {{:>8}}"
    print(row_fmt.format("Method", "RMSE", "Std", "|div|", "AnomRat", "ACC", "SpecLo", "SpecHi"))
    print(f"  {'-'*82}")

    for method in args.methods:
        arr = np.array(results[method])        # (N, 6)
        rmse_m, div_m, anom_m, acc_m, lo_m, hi_m = arr.mean(0)
        rmse_s                                   = arr[:, 0].std()
        print(row_fmt.format(
            method,
            f"{rmse_m:.4f}", f"{rmse_s:.4f}",
            f"{div_m:.6f}", f"{anom_m:.3f}", f"{acc_m:.3f}", f"{lo_m:.4f}", f"{hi_m:.4f}",
        ))

    # Climatology baseline = trivial 'predict train-mean field' (ignores observations).
    # By construction climatology has AnomRat=0 and ACC=0 (it IS the anomaly origin).
    clim_m = float(np.mean(clim_rmses))
    print(row_fmt.format(
        "climatology", f"{clim_m:.4f}", "-", "-", "0.000", "0.000", "-", "-"))
    print(f"\n  AnomRat = anomaly-RMS / true-anomaly-RMS (DC-offset removed; ~1 = right energy).")
    print(f"  ACC     = anomaly pattern correlation vs GT (1 = perfect, 0 = climatology, <0 = worse).")
    print(f"  REAL skill needs RMSE < climatology ({clim_m:.4f}) AND ACC > 0 (positive pattern match).")

    print()


if __name__ == "__main__":
    main()
