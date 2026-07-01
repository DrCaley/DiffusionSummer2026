"""
Compute pointwise divergence of generated fields (raw diffusion vs coupled-fused)
across a sample of test frames, to quantify how much magnitude fusion breaks
the div-free guarantee from stream-function parameterization.

div(u) = du_x/dx + du_y/dy  (central finite differences, pixel spacing = 1)
Measured on interior ocean cells only.

Usage (run from /workspace/DiffusionSummer2026):
  python "Conditional DDPM/testing/_probe_divergence.py" \
      --n_frames 20 --n_draws 4 --seed 0
"""
import argparse
import os
import sys
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", ".."))
for _p in (_here, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), _root):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import infer_cond as IC
from _probe_calib_mag import load_magnitude_model, EPS
from _probe_multidraw import (
    load_hetero_magnitude_model, predict_speed_mean_sigma, coupled_magnitude,
)


def divergence(field):
    """Central-difference divergence of (2, H, W) field. Returns (H, W)."""
    ux, uy = field[0], field[1]
    dux_dx = np.zeros_like(ux)
    dux_dx[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / 2.0
    dux_dx[:, 0]    = ux[:, 1]  - ux[:, 0]
    dux_dx[:, -1]   = ux[:, -1] - ux[:, -2]
    duy_dy = np.zeros_like(uy)
    duy_dy[1:-1, :] = (uy[2:, :] - uy[:-2, :]) / 2.0
    duy_dy[0, :]    = uy[1, :]  - uy[0, :]
    duy_dy[-1, :]   = uy[-1, :] - uy[-2, :]
    return dux_dx + duy_dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",        default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--mag_checkpoint",    default="Models/Cond_Magnitude_UNet.pt")
    ap.add_argument("--hetero_checkpoint", default="Magnitude/checkpoints_cond_mag_hetero/best_cond_magnitude_hetero.pt")
    ap.add_argument("--pickle",            default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split",     type=int, default=2)
    ap.add_argument("--n_frames",  type=int, default=20)
    ap.add_argument("--n_draws",   type=int, default=4)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--path_steps",      type=int, default=90)
    ap.add_argument("--inference_steps", type=int, default=100)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)

    ds = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=ckpt.get("data_mean", 0.0), data_std=data_std,
        path_steps=args.path_steps, deterministic=True)
    land_np = ds.land_mask.cpu().numpy().astype(bool)
    ocean_np = ~land_np

    model = IC.StreamFunctionUNet(in_ch=2, base_ch=ca.get("base_ch", 64),
        time_dim=ca.get("time_dim", 256), cond_ch=cond_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = IC.DDPM(T=ca.get("T", 1000), beta_schedule=ca.get("schedule", "cosine"),
        device=device, noise_type=ca.get("noise_type", "div_free"),
        spectral_filter=ckpt.get("spectral_filter", None))

    het_net, hsm, hss, het_clip = load_hetero_magnitude_model(
        args.hetero_checkpoint, device)

    sargs = argparse.Namespace(pred_type=pred_type,
        inference_steps=args.inference_steps, capture_every=10**9,
        n_ensemble=args.n_draws)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds.valid), size=min(args.n_frames, len(ds.valid)), replace=False)

    raw_divs, fused_divs = [], []
    # interior mask: exclude 1-pixel border where finite diff is one-sided
    interior = np.zeros(ocean_np.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    interior_ocean = interior & ocean_np

    for i, src_idx in enumerate(indices):
        src_idx = int(src_idx)
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)

        _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                          sargs, device, base_seed=src_idx)

        mu_n, sig_n = predict_speed_mean_sigma(
            het_net, hsm, hss, land_np, data_std, device, b["cond"], het_clip)
        draws_fused = coupled_magnitude(members, mu_n, sig_n, ocean_np)

        for raw, fused in zip(members, draws_fused):
            raw_np   = raw if isinstance(raw, np.ndarray) else raw.cpu().numpy()
            fused_np = fused if isinstance(fused, np.ndarray) else fused.cpu().numpy()
            raw_divs.append(  np.abs(divergence(raw_np)  [interior_ocean]))
            fused_divs.append(np.abs(divergence(fused_np)[interior_ocean]))

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(indices)} frames done")

    raw_all   = np.concatenate(raw_divs)
    fused_all = np.concatenate(fused_divs)

    def stats(arr, label):
        print(f"\n{label}  (n={len(arr):,} cells)")
        print(f"  mean |div|  : {arr.mean():.6f}")
        print(f"  median |div|: {np.median(arr):.6f}")
        print(f"  95th pctile : {np.percentile(arr, 95):.6f}")
        print(f"  99th pctile : {np.percentile(arr, 99):.6f}")
        print(f"  max |div|   : {arr.max():.6f}")

    print(f"\n=== Divergence stats ({len(indices)} frames × {args.n_draws} draws) ===")
    stats(raw_all,   "Raw diffusion (stream-fn, should be ~0)")
    stats(fused_all, "Coupled-fused (magnitude rescaled)")
    ratio = fused_all.mean() / (raw_all.mean() + 1e-12)
    print(f"\nFusion increases mean |div| by {ratio:.1f}×")


if __name__ == "__main__":
    main()
