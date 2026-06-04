"""
Batch GP inpainting: run N times on the validation set with different random
robot paths and save individual 2×2 plots labelled 1–N.

Usage:
    python3 batch_infer.py
    python3 batch_infer.py --n_runs 10 --path_steps 300 --out_dir batch_results

Notes on paths:
    On the remote server (~/ocean_diffusion/ flat layout):  --pickle data.pickle
    When running locally from GP Inpainting/:               (default works)
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dataset    import OceanCurrentDataset
from gp_infer   import gp_reconstruct
from paths      import biased_walk_path
from plot_utils import plot_field


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",        default="../data.pickle")
    p.add_argument("--n_runs",        type=int,   default=10,   help="number of runs")
    p.add_argument("--path_steps",    type=int,   default=300,  help="robot walk length")
    p.add_argument("--length_scale",  type=float, default=0.15, help="initial GP length scale")
    p.add_argument("--n_restarts",    type=int,   default=2,    help="hyperparameter optimizer restarts")
    p.add_argument("--out_dir",       default="batch_results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_one(val_ds, land_mask_np, sample_idx, seed, args):
    x0_true   = val_ds[sample_idx].numpy()    # (2, H, W) float32
    u_true    = x0_true[0]
    v_true    = x0_true[1]

    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

    u_pred, v_pred, u_std, v_std = gp_reconstruct(
        x0_true, path_mask, land_mask_np,
        length_scale=args.length_scale,
        n_restarts=args.n_restarts,
    )

    err = np.sqrt((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)
    err[land_mask_np] = np.nan
    ocean_err = err[~land_mask_np]
    rmse      = float(np.sqrt(np.nanmean(ocean_err ** 2)))

    # Transpose for display: (94, 44) → (44, 94)
    return (
        u_true.T, v_true.T,
        u_pred.T, v_pred.T,
        land_mask_np.T, path_mask.T,
        err.T,
        rmse,
        int(path_mask.sum()),
    )


# ---------------------------------------------------------------------------
# Save one figure
# ---------------------------------------------------------------------------

def save_plot(u_true, v_true, u_pred, v_pred, land_mask, path_mask,
              err, rmse, path_cells, label, sample_idx, seed, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Ground truth
    plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth")

    # 2. Robot path
    axes[1].imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    path_display = np.zeros_like(land_mask, dtype=float)
    path_display[path_mask] = 1.0
    axes[1].imshow(
        path_display, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    axes[1].set_title(f"Robot Path ({path_cells} cells, seed={seed})", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    axes[1].legend(handles=[
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
        mpatches.Patch(facecolor="#d62728",                 label="Path"),
        mpatches.Patch(facecolor="black",                   label="Land"),
    ], loc="upper right", fontsize=8)

    # 3. Reconstruction
    plot_field(axes[2], u_pred, v_pred, land_mask, "Reconstructed (GP mean)", cmap="cool")

    # 4. Error map
    err_plot = np.ma.masked_where(land_mask, err)
    im = axes[3].imshow(
        err_plot, origin="lower", cmap="hot_r", aspect="auto",
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
    )
    axes[3].imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=1,
    )
    plt.colorbar(im, ax=axes[3], label="|error| speed", shrink=0.7)
    axes[3].set_title(f"Error  (RMSE = {rmse:.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    plt.suptitle(
        f"Run {label}  —  Val sample {sample_idx}, path seed {seed}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    val_ds    = OceanCurrentDataset(args.pickle, split=1)
    land_mask = val_ds.land_mask.numpy()    # (H, W) bool
    n_ocean   = int((~land_mask).sum())

    rmse_list = []

    for i in range(args.n_runs):
        sample_idx = i % len(val_ds)
        seed       = 1000 + i

        print(f"\nRun {i + 1}/{args.n_runs}  (val sample {sample_idx}, seed {seed})")

        (u_true, v_true,
         u_pred, v_pred,
         land_d, path_d,
         err_d, rmse, path_cells) = run_one(val_ds, land_mask, sample_idx, seed, args)

        pct = 100 * path_cells / n_ocean
        print(f"  Path cells: {path_cells} ({pct:.1f}%)   RMSE: {rmse:.4f}")
        rmse_list.append(rmse)

        label    = f"{i + 1:02d}"
        out_path = os.path.join(args.out_dir, f"result_{label}.png")
        save_plot(
            u_true, v_true, u_pred, v_pred,
            land_d, path_d, err_d, rmse, path_cells,
            label, sample_idx, seed, out_path,
        )
        print(f"  Saved: {out_path}")

    print(f"\n{'='*50}")
    print(f"Mean RMSE over {args.n_runs} runs: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"Min: {np.min(rmse_list):.4f}   Max: {np.max(rmse_list):.4f}")


if __name__ == "__main__":
    main()
