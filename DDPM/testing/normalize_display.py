"""
Direction-only display: normalize every vector to unit magnitude and compare the
predicted flow direction against the ground-truth direction.

For the angle-loss model, magnitude is intentionally ignored at training time —
only the *direction* of each (u, v) vector matters. This script visualizes that:
it runs PPR (old un-normalized pipeline) to reconstruct the field, then unit-
normalizes every ocean vector so all arrows have the same length. Colour encodes
the arrow's compass direction (cyclic colormap), so a correct reconstruction
matches the ground-truth colours/arrows regardless of speed.

Per sample it produces a 1x3 panel:
    [ Predicted direction | Ground-truth direction | Angle error (deg) ]
with the robot-path cells outlined, and prints the mean/median angular error and
the mean cosine similarity (== anomaly-free ACC of unit vectors).

Usage (from workspace root):
    python DDPM/testing/normalize_display.py \
        --checkpoint Models/Div_Free_DDPM_7%.pt \
        --pickle     Datasets/data.pickle \
        --n_samples  10 --random --seed 1234 \
        --inference_steps 100 --ppr_resample 10 \
        --out_dir    DDPM/best_model_results/direction
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_here  = os.path.dirname(os.path.abspath(__file__))
_root  = os.path.join(_here, "..", "..")
_model = os.path.join(_here, "..", "model")
_repaint = os.path.join(_here, "repaint")
_ppr   = os.path.join(_here, "ppr")
for _p in [_root, os.path.join(_root, "utils"), _model, _repaint, _ppr]:
    sys.path.insert(0, _p)

from dataset            import OceanCurrentDataset
from diffusion          import DDPM
from model              import UNet
from divfree_projection import joint_project
from repaint_infer      import biased_walk_path, repaint
from ppr_infer          import ppr


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def unit_normalize(field_np: np.ndarray, ocean_np: np.ndarray, eps: float = 1e-8):
    """
    Unit-normalize every vector of a (2, H, W) field.

    Returns (u_hat, v_hat, mag) where (u_hat, v_hat) have magnitude 1 at ocean
    cells with non-negligible speed, 0 where the original vector is ~0 or land.
    """
    u, v = field_np[0], field_np[1]
    mag  = np.sqrt(u ** 2 + v ** 2)
    safe = mag > eps
    u_hat = np.zeros_like(u)
    v_hat = np.zeros_like(v)
    u_hat[safe] = u[safe] / mag[safe]
    v_hat[safe] = v[safe] / mag[safe]
    # Zero out land
    u_hat[~ocean_np] = 0.0
    v_hat[~ocean_np] = 0.0
    return u_hat, v_hat, mag


def angle_error_deg(pred_np: np.ndarray, true_np: np.ndarray,
                    ocean_np: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-cell angular error in degrees [0, 180] between predicted and true vectors.
    Returns (H, W), NaN at land or where either vector is ~0.
    """
    up, vp = pred_np[0], pred_np[1]
    ut, vt = true_np[0], true_np[1]
    dot    = up * ut + vp * vt
    mp     = np.sqrt(up ** 2 + vp ** 2)
    mt     = np.sqrt(ut ** 2 + vt ** 2)
    cos    = dot / (mp * mt + eps)
    cos    = np.clip(cos, -1.0, 1.0)
    err    = np.degrees(np.arccos(cos))
    valid  = ocean_np & (mp > eps) & (mt > eps)
    out    = np.full(err.shape, np.nan, dtype=np.float32)
    out[valid] = err[valid]
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _direction_quiver(ax, u_hat, v_hat, land_np, path_np, title, step=2):
    """Draw unit-length arrows coloured by compass direction (cyclic colormap)."""
    H, W = u_hat.shape
    # Land in black
    ax.imshow(
        land_np, origin="lower",
        cmap=mcolors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u_hat[::step, ::step]
    vq = v_hat[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    land_q = land_np[::step, ::step]
    mask = (mq > 1e-6) & (~land_q)
    # Direction angle in [0, 2pi) -> cyclic colormap
    ang = (np.arctan2(vq, uq) % (2 * np.pi))
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], ang[mask],
        cmap="twilight", clim=(0, 2 * np.pi),
        scale=30, width=0.004, pivot="mid", zorder=2,
    )
    cb = plt.colorbar(q, ax=ax, label="Direction (rad)", shrink=0.7)
    cb.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cb.set_ticklabels(["E", "N", "W", "S", "E"])
    # Robot-path overlay (red dots)
    if path_np is not None:
        py, px = np.where(path_np)
        ax.scatter(px, py, s=6, c="red", marker="s", zorder=3,
                   linewidths=0, label="path")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def _angle_err_panel(ax, err_map, land_np, title):
    """Heatmap of per-cell angular error in degrees."""
    H, W = err_map.shape
    masked = np.ma.masked_invalid(err_map)
    im = ax.imshow(
        masked, origin="lower", cmap="inferno", vmin=0, vmax=180,
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=1,
    )
    ax.imshow(
        land_np, origin="lower",
        cmap=mcolors.ListedColormap(["none", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=2,
    )
    plt.colorbar(im, ax=ax, label="Angle error (deg)", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# Single sample: run PPR, normalize, plot
# ---------------------------------------------------------------------------

def run_sample(model, diffusion, val_ds, land_np, sample_idx, seed, args, device,
               data_mean=None, data_std=None):
    x0_true   = val_ds[sample_idx]                          # (2, H, W)
    path_mask = biased_walk_path(land_np, n_steps=args.path_steps, seed=seed)

    x0_observed = x0_true.clone()
    x0_observed[:, ~torch.from_numpy(path_mask)] = 0.0

    _normalize   = data_mean is not None
    x0_obs_infer = ((x0_observed - data_mean) / data_std) if _normalize else x0_observed

    if args.method == "repaint":
        x0_pred_norm = repaint(
            model, diffusion, x0_obs_infer, path_mask, land_np,
            r=args.resample, device=device, inference_steps=args.inference_steps,
        )
    else:  # ppr
        x0_pred_norm = ppr(
            model, diffusion, x0_obs_infer, path_mask, land_np,
            r=args.ppr_resample, proj_iter=args.proj_iter, device=device,
            inference_steps=args.inference_steps, projector=args.projector,
            data_mean=data_mean, data_std=data_std,
        )

    if args.final_project > 0:
        ocean_mask_t = torch.from_numpy(~land_np)
        obs_mask_t   = torch.from_numpy(path_mask)
        x0_pred_norm = joint_project(
            x0_pred_norm.unsqueeze(0), ocean_mask_t, obs_mask_t,
            x0_obs_infer.unsqueeze(0), n_iter=args.final_project,
        ).squeeze(0)

    x0_pred = (x0_pred_norm * data_std + data_mean) if _normalize else x0_pred_norm

    pred_np = x0_pred.numpy()
    true_np = x0_true.numpy()
    ocean_np = ~land_np

    # Direction-only fields
    up_hat, vp_hat, _ = unit_normalize(pred_np, ocean_np)
    ut_hat, vt_hat, _ = unit_normalize(true_np, ocean_np)
    err_map = angle_error_deg(pred_np, true_np, ocean_np)

    # Metrics (unit-vector cosine == direction-only ACC)
    valid = ~np.isnan(err_map)
    mean_err = float(np.nanmean(err_map))
    med_err  = float(np.nanmedian(err_map))
    cos_sim  = float(np.mean(np.cos(np.radians(err_map[valid])))) if valid.any() else float("nan")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _direction_quiver(axes[0], up_hat, vp_hat, land_np, path_mask,
                      f"Predicted direction ({args.method.upper()})", step=args.step)
    _direction_quiver(axes[1], ut_hat, vt_hat, land_np, None,
                      "Ground-truth direction", step=args.step)
    _angle_err_panel(axes[2], err_map, land_np,
                     f"Angle error\nmean={mean_err:.1f}°  med={med_err:.1f}°  cos={cos_sim:.3f}")
    fig.suptitle(
        f"Sample {sample_idx}  |  path cells={int(path_mask.sum())}  |  "
        f"unit-normalized vectors (magnitude discarded)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(args.out_dir, f"direction_val{sample_idx}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return mean_err, med_err, cos_sim, int(path_mask.sum()), out_path


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Direction-only display: unit-normalize vectors, compare "
                    "predicted vs ground-truth flow direction."
    )
    p.add_argument("--pickle",     default="Datasets/data.pickle")
    p.add_argument("--checkpoint", default="Models/Div_Free_DDPM_7%.pt")
    p.add_argument("--n_samples",  type=int, default=10)
    p.add_argument("--random",     action="store_true",
                   help="draw n_samples random distinct val samples (seeded by --seed)")
    p.add_argument("--seed",       type=int, default=1234)
    p.add_argument("--method",     default="ppr", choices=["ppr", "repaint"])
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--inference_steps", type=int, default=100)
    p.add_argument("--resample",     type=int, default=10, help="RePaint resamples per step")
    p.add_argument("--ppr_resample", type=int, default=10, help="PPR resamples per step")
    p.add_argument("--proj_iter",    type=int, default=20)
    p.add_argument("--projector",    default="pocs", choices=["pocs", "snap_x0"])
    p.add_argument("--final_project", type=int, default=0,
                   help="POCS iters applied ONCE to the final prediction (0=off).")
    p.add_argument("--step",       type=int, default=2, help="quiver subsample step")
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--device",     default=None, help="cuda / mps / cpu (auto if omitted)")
    p.add_argument("--out_dir",    default="DDPM/best_model_results/direction")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device : {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Data ----
    val_ds  = OceanCurrentDataset(args.pickle, split=1)
    land_np = val_ds.land_mask.numpy()

    # ---- Model ----
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)
    noise_type      = ckpt_args.get("noise_type", "gaussian")
    spectral_filter = ckpt.get("spectral_filter", None)
    data_mean       = ckpt.get("data_mean", None)
    data_std        = ckpt.get("data_std",  None)

    model = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm_str = f"mean={data_mean:.4f} std={data_std:.4f}" if data_mean is not None else "no"
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}), T={T}, "
          f"noise={noise_type}, normalize={norm_str}")
    print(f"Method : {args.method}  inference_steps={args.inference_steps}  "
          f"projector={args.projector}")

    diffusion = DDPM(T=T, beta_schedule="cosine", device=device,
                     noise_type=noise_type, spectral_filter=spectral_filter)

    # ---- Sample indices ----
    if args.random:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(val_ds), size=min(args.n_samples, len(val_ds)),
                          replace=False).tolist()
    else:
        idxs = [i % len(val_ds) for i in range(args.n_samples)]
    print(f"Val samples ({'random' if args.random else 'sequential'}): {idxs}\n")

    header = f"{'Run':>4}  {'Sample':>7}  {'Cells':>6}  {'MeanErr°':>9}  {'MedErr°':>9}  {'CosSim':>8}"
    print(header)

    results = []
    for i, sidx in enumerate(idxs):
        seed = i * 7 + 1
        mean_err, med_err, cos_sim, cells, out_path = run_sample(
            model, diffusion, val_ds, land_np, sidx, seed, args, device,
            data_mean=data_mean, data_std=data_std,
        )
        results.append((mean_err, med_err, cos_sim))
        print(f"{i+1:>4}  {sidx:>7}  {cells:>6}  {mean_err:>9.1f}  {med_err:>9.1f}  {cos_sim:>8.3f}")

    arr = np.array(results)
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY ({len(idxs)} samples)")
    print(f"{'=' * 60}")
    print(f"  Mean angular error : {arr[:, 0].mean():.1f}°  (± {arr[:, 0].std():.1f})")
    print(f"  Median angular err : {arr[:, 1].mean():.1f}°")
    print(f"  Mean cosine sim    : {arr[:, 2].mean():.3f}  (1 = perfect direction)")
    print(f"\n  Figures saved to: {args.out_dir}/direction_val*.png")


if __name__ == "__main__":
    main()
