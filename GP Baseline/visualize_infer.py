"""
Visualise GP inpainting on a single test sample.

Loads a test sample, simulates a robot path, runs GP reconstruction,
and plots a 2×3 grid:
  1. Ground truth field
  2. Robot path (observed cells)
  3. Reconstructed field  (GP posterior mean)
  4. Error magnitude
  5. GP uncertainty       (posterior std, average of u and v)
  6. Run summary text

Usage:
    py visualize_infer.py
    py visualize_infer.py --sample 5 --path_steps 150 --seed 7

Notes on paths:
    On the remote server (~/ocean_diffusion/ flat layout), all files are in
    the same directory, so use:  --pickle data.pickle
    When running locally from GP Inpainting/, use the default: --pickle ../data.pickle
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Allow importing dataset.py from the repo root
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
    p.add_argument("--sample",        type=int,   default=0,    help="test set index")
    p.add_argument("--path_steps",    type=int,   default=150,  help="robot walk length")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out",           default="gp_result.png")
    p.add_argument("--length_scale",  type=float, default=0.15, help="initial GP length scale")
    p.add_argument("--n_restarts",    type=int,   default=2,    help="hyperparameter optimizer restarts")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load data ----
    test_ds   = OceanCurrentDataset(args.pickle, split=2)
    land_mask = test_ds.land_mask.numpy()           # (H, W) bool
    x0_true   = test_ds[args.sample].numpy()        # (2, H, W) float32
    u_true    = x0_true[0]
    v_true    = x0_true[1]

    # ---- Simulate robot path ----
    path_mask = biased_walk_path(land_mask, n_steps=args.path_steps, seed=args.seed)
    n_obs     = path_mask.sum()
    n_ocean   = int((~land_mask).sum())
    print(f"Path covers {n_obs} / {n_ocean} ocean cells ({100 * n_obs / n_ocean:.1f}%)")

    # ---- GP reconstruction ----
    print("Fitting GPs and predicting …")
    u_pred, v_pred, u_std, v_std = gp_reconstruct(
        x0_true, path_mask, land_mask,
        length_scale=args.length_scale,
        n_restarts=args.n_restarts,
    )

    # ---- Metrics ----
    err       = np.sqrt((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)
    err[land_mask] = np.nan
    ocean_err = err[~land_mask]
    rmse      = float(np.sqrt(np.nanmean(ocean_err ** 2)))
    print(f"Mean speed error: {np.nanmean(ocean_err):.4f}")
    print(f"RMSE:             {rmse:.4f}")

    # ---- Transpose for display: (94, 44) → (44, 94) ----
    u_true_d = u_true.T
    v_true_d = v_true.T
    u_pred_d = u_pred.T
    v_pred_d = v_pred.T
    land_d   = land_mask.T
    path_d   = path_mask.T
    err_d    = err.T
    std_d    = ((u_std + v_std) / 2).T

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    axes = axes.flatten()

    # 1. Ground truth
    plot_field(axes[0], u_true_d, v_true_d, land_d, "Ground Truth")

    # 2. Robot path
    axes[1].imshow(
        land_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    path_display = np.zeros_like(land_d, dtype=float)
    path_display[path_d] = 1.0
    axes[1].imshow(
        path_display, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    axes[1].set_title(f"Robot Path ({n_obs} cells)", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    axes[1].legend(handles=[
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
        mpatches.Patch(facecolor="#d62728",                 label="Path"),
        mpatches.Patch(facecolor="black",                   label="Land"),
    ], loc="upper right", fontsize=8)

    # 3. Reconstructed field
    plot_field(axes[2], u_pred_d, v_pred_d, land_d, "Reconstructed (GP mean)", cmap="cool")

    # 4. Error map
    err_plot = np.ma.masked_where(land_d, err_d)
    im = axes[3].imshow(
        err_plot, origin="lower", cmap="hot_r", aspect="auto",
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
    )
    axes[3].imshow(
        land_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=1,
    )
    plt.colorbar(im, ax=axes[3], label="|error| speed", shrink=0.7)
    axes[3].set_title(f"Error  (RMSE = {rmse:.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    # 5. GP uncertainty (posterior std, average of u and v channels)
    std_plot = np.ma.masked_where(land_d, std_d)
    im2 = axes[4].imshow(
        std_plot, origin="lower", cmap="viridis", aspect="auto",
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
    )
    axes[4].imshow(
        land_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=1,
    )
    plt.colorbar(im2, ax=axes[4], label="Posterior std (avg u, v)", shrink=0.7)
    axes[4].set_title("GP Uncertainty", fontsize=11)
    axes[4].set_xlabel("X"); axes[4].set_ylabel("Y")

    # 6. Run summary
    axes[5].axis("off")
    axes[5].text(
        0.5, 0.5,
        f"Test sample:    {args.sample}\n"
        f"Path steps:     {args.path_steps}\n"
        f"Observed cells: {n_obs}  ({100 * n_obs / n_ocean:.1f} %)\n"
        f"Kernel:         Matérn  ν = 2.5\n"
        f"Length scale:   {args.length_scale}  (initial)\n"
        f"RMSE:           {rmse:.4f}",
        ha="center", va="center", fontsize=12,
        transform=axes[5].transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", edgecolor="gray"),
    )

    plt.suptitle(
        f"Ocean Current GP Inpainting — Test sample {args.sample}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
