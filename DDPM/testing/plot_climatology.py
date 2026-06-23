"""
Visualize the climatology baseline = per-cell mean of the TRAIN split.

This is the "predict the historical average everywhere, ignore observations"
baseline. A reconstruction method only has real skill if it beats this in RMSE
AND reproduces per-sample structure (positive anomaly correlation, ACC > 0).

Renders a 1x3 panel:
    [climatology field]  [a sample ground truth]  [that sample's anomaly = GT - clim]

Usage (from workspace root):
    python DDPM/testing/plot_climatology.py --pickle Datasets/data.pickle --sample 1915
    python DDPM/testing/plot_climatology.py --pickle Datasets/data_divfree.pickle \
        --sample 1915 --split 1 --out DDPM/best_model_results/climatology.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..")
for _p in [_root, os.path.join(_root, "utils")]:
    sys.path.insert(0, _p)

from dataset import OceanCurrentDataset


def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool", vmax=None):
    H, W = u.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u[::step, ::step]
    vq = v[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    mask = ~np.isnan(uq) & ~land_mask[::step, ::step]
    clim_max = vmax if vmax is not None else (np.nanpercentile(mq[mask], 98) if mask.any() else 1)
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
        cmap=cmap, clim=(0, clim_max),
        scale=12, width=0.003, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def parse_args():
    p = argparse.ArgumentParser(description="Plot the climatology (train-mean) field.")
    p.add_argument("--pickle", default="Datasets/data.pickle")
    p.add_argument("--sample", type=int, default=1915,
                   help="Dataset index to compare against the climatology field.")
    p.add_argument("--split",  type=int, default=1,
                   help="Split of the comparison sample (0=train,1=val,2=test).")
    p.add_argument("--out",    default="DDPM/best_model_results/climatology.png")
    return p.parse_args()


def main():
    args = parse_args()

    # Climatology = per-cell mean over the TRAIN split (physical units).
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    land     = train_ds.land_mask.numpy()
    ocean    = ~land
    clim     = train_ds.data.mean(dim=0).numpy()   # (2, H, W)
    clim[:, ~ocean] = 0.0

    # A comparison sample + its anomaly (what the model actually has to predict).
    cmp_ds = OceanCurrentDataset(args.pickle, split=args.split)
    idx    = args.sample % len(cmp_ds)
    gt     = cmp_ds[idx].numpy()
    anom   = gt - clim
    anom[:, ~ocean] = 0.0

    # Shared colour scale from the ground-truth sample.
    gt_speed = np.sqrt(gt[0] ** 2 + gt[1] ** 2)
    gt_speed[land] = np.nan
    vmax = float(np.nanpercentile(gt_speed, 98)) or 1.0

    # Stats for the title / console.
    clim_speed = np.sqrt(clim[0] ** 2 + clim[1] ** 2)[ocean].mean()
    gt_meansp  = np.sqrt(gt[0] ** 2 + gt[1] ** 2)[ocean].mean()
    anom_rms   = float(np.sqrt((anom[:, ocean] ** 2).sum(0).mean()))
    print(f"Climatology mean speed : {clim_speed:.4f}")
    print(f"Sample {idx} mean speed : {gt_meansp:.4f}")
    print(f"Sample {idx} anomaly RMS: {anom_rms:.4f}  "
          f"(this is the structure a model must reconstruct)")

    land_d = land.T
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_field(axes[0], clim[0].T, clim[1].T, land_d,
               f"Climatology (train mean)\nmean speed {clim_speed:.3f}", vmax=vmax)
    plot_field(axes[1], gt[0].T, gt[1].T, land_d,
               f"Ground truth (sample {idx})\nmean speed {gt_meansp:.3f}", vmax=vmax)
    plot_field(axes[2], anom[0].T, anom[1].T, land_d,
               f"Anomaly = GT - climatology\nRMS {anom_rms:.3f}", vmax=vmax)
    fig.suptitle("Climatology baseline vs. ground truth vs. the anomaly to reconstruct",
                 fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
