"""
Batch GP *magnitude* (speed) inpainting on the validation set.

Fits a single GP on the known robot-path speeds |v| = sqrt(u^2 + v^2) and
predicts the dense speed field, then reports speed RMSE and saves a 1×3 plot
per run (ground-truth speed | predicted speed | absolute error).

This is the magnitude half of the direction/magnitude decomposition; it is
fully independent of the DDPM (the DDPM's magnitudes are meaningless under a
pure angle loss, so they are deliberately not used here).

Usage:
    python "GP Baseline/batch_magnitude.py" --pickle Datasets/data.pickle \
        --n_runs 10 --random --seed 1234 --path_steps 150
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "utils"))

from dataset      import OceanCurrentDataset
from gp_magnitude import gp_reconstruct_magnitude, climatology_speed
from paths        import biased_walk_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",       default="Datasets/data.pickle")
    p.add_argument("--n_runs",       type=int,   default=10,  help="number of runs")
    p.add_argument("--random",       action="store_true",     help="random validation samples")
    p.add_argument("--seed",         type=int,   default=1234, help="RNG seed for sample/path selection")
    p.add_argument("--path_steps",   type=int,   default=150, help="robot walk length")
    p.add_argument("--length_scale", type=float, default=0.15, help="initial GP length scale")
    p.add_argument("--n_restarts",   type=int,   default=2,   help="hyperparameter optimizer restarts")
    p.add_argument("--target_space", default="linear", choices=["linear", "log"], help="GP target space")
    p.add_argument("--prior",        default="flat", choices=["flat", "climatology"],
                   help="GP prior mean: 'flat' (constant) or 'climatology' (training-set mean speed field)")
    p.add_argument("--out_dir",      default="GP Baseline/GP_magnitude_results")
    return p.parse_args()


def run_one(val_ds, land_mask_np, sample_idx, seed, args):
    x0_true   = val_ds[sample_idx].numpy()    # (2, H, W) float32
    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

    mag_pred, mag_std, mag_true = gp_reconstruct_magnitude(
        x0_true, path_mask, land_mask_np,
        length_scale=args.length_scale,
        n_restarts=args.n_restarts,
        target_space=args.target_space,
        climatology=args._climatology,
    )

    err = np.abs(mag_pred - mag_true)
    err[land_mask_np] = np.nan
    ocean_err = err[~land_mask_np]
    rmse      = float(np.sqrt(np.nanmean(ocean_err ** 2)))
    mae       = float(np.nanmean(ocean_err))

    # Relative error vs the true mean speed (scale-free skill indicator).
    mean_speed = float(np.nanmean(mag_true[~land_mask_np]))
    rel_rmse   = rmse / (mean_speed + 1e-8)

    return mag_true, mag_pred, err, land_mask_np, path_mask, rmse, mae, rel_rmse, int(path_mask.sum())


def save_plot(mag_true, mag_pred, err, land_mask, path_mask,
              rmse, path_cells, label, sample_idx, seed, out_path):
    # Display transpose: (94, 44) → (44, 94)
    mag_true_d = mag_true.T
    mag_pred_d = mag_pred.T
    err_d      = err.T
    land_d     = land_mask.T
    path_d     = path_mask.T

    vmax = float(np.nanmax(np.ma.masked_where(land_d, mag_true_d)))
    extent = [-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Ground-truth speed (with robot path overlay)
    gt = np.ma.masked_where(land_d, mag_true_d)
    im0 = axes[0].imshow(gt, origin="lower", cmap="viridis", aspect="auto",
                         extent=extent, vmin=0, vmax=vmax)
    path_display = np.ma.masked_where(~path_d, np.ones_like(land_d, dtype=float))
    axes[0].imshow(path_display, origin="lower", cmap="Reds", alpha=0.9,
                   aspect="auto", extent=extent, vmin=0, vmax=1, zorder=2)
    axes[0].imshow(land_d, origin="lower",
                   cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
                   aspect="auto", extent=extent, zorder=1)
    plt.colorbar(im0, ax=axes[0], label="speed |v|", shrink=0.8)
    axes[0].set_title(f"Ground-truth speed  (path = {path_cells} cells)")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")

    # 2. Predicted speed
    pr = np.ma.masked_where(land_d, mag_pred_d)
    im1 = axes[1].imshow(pr, origin="lower", cmap="viridis", aspect="auto",
                         extent=extent, vmin=0, vmax=vmax)
    axes[1].imshow(land_d, origin="lower",
                   cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
                   aspect="auto", extent=extent, zorder=1)
    plt.colorbar(im1, ax=axes[1], label="speed |v|", shrink=0.8)
    axes[1].set_title("Predicted speed (GP mean)")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")

    # 3. Absolute error
    er = np.ma.masked_where(land_d, err_d)
    im2 = axes[2].imshow(er, origin="lower", cmap="hot_r", aspect="auto", extent=extent)
    axes[2].imshow(land_d, origin="lower",
                   cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
                   aspect="auto", extent=extent, zorder=1)
    plt.colorbar(im2, ax=axes[2], label="|error|", shrink=0.8)
    axes[2].set_title(f"Speed error  (RMSE = {rmse:.4f})")
    axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")

    plt.suptitle(f"Run {label} — Val sample {sample_idx}, path seed {seed}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    val_ds    = OceanCurrentDataset(args.pickle, split=1)
    land_mask = val_ds.land_mask.numpy()    # (H, W) bool
    n_ocean   = int((~land_mask).sum())

    # Climatology prior is estimated from the TRAINING split (split=0) so it
    # never peeks at validation ground truth.
    args._climatology = None
    if args.prior == "climatology":
        print("Computing training-set climatology (mean speed field)...")
        train_ds = OceanCurrentDataset(args.pickle, split=0)
        args._climatology = climatology_speed(train_ds, land_mask)
        print(f"  climatology mean speed = {args._climatology[~land_mask].mean():.4f}")

    if args.random:
        rng         = np.random.default_rng(args.seed)
        sample_idxs = rng.integers(0, len(val_ds), size=args.n_runs)
        seeds       = rng.integers(0, 1_000_000, size=args.n_runs)
    else:
        sample_idxs = [i % len(val_ds) for i in range(args.n_runs)]
        seeds       = [1000 + i for i in range(args.n_runs)]

    rmse_list, mae_list, rel_list = [], [], []

    for i in range(args.n_runs):
        sample_idx = int(sample_idxs[i])
        seed       = int(seeds[i])

        print(f"\nRun {i + 1}/{args.n_runs}  (val sample {sample_idx}, seed {seed})")

        (mag_true, mag_pred, err, land_d, path_d,
         rmse, mae, rel_rmse, path_cells) = run_one(val_ds, land_mask, sample_idx, seed, args)

        pct = 100 * path_cells / n_ocean
        print(f"  Path cells: {path_cells} ({pct:.1f}%)   "
              f"RMSE: {rmse:.4f}   MAE: {mae:.4f}   relRMSE: {rel_rmse:.3f}")
        rmse_list.append(rmse); mae_list.append(mae); rel_list.append(rel_rmse)

        label    = f"{i + 1:02d}"
        out_path = os.path.join(args.out_dir, f"mag_val{sample_idx}_{label}.png")
        save_plot(mag_true, mag_pred, err, land_d, path_d,
                  rmse, path_cells, label, sample_idx, seed, out_path)
        print(f"  Saved: {out_path}")

    print(f"\n{'=' * 56}")
    print(f"Magnitude GP over {args.n_runs} runs (target_space={args.target_space}, prior={args.prior})")
    print(f"  Mean RMSE:    {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"  Mean MAE:     {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"  Mean relRMSE: {np.mean(rel_list):.3f} ± {np.std(rel_list):.3f}")
    print(f"  RMSE min/max: {np.min(rmse_list):.4f} / {np.max(rmse_list):.4f}")


if __name__ == "__main__":
    main()
