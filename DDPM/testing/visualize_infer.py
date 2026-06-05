"""
Visualise RePaint inference on a test sample.

Loads the best checkpoint, picks a test sample, simulates a robot path,
runs RePaint, and plots:
  - Ground truth field
  - Robot path (observed cells)
  - Reconstructed field
  - Error magnitude

Usage:
    py visualize_infer.py
    py visualize_infer.py --sample 5 --path_steps 400 --resample 10
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from model         import UNet
from plot_utils    import plot_field
from repaint_infer import biased_walk_path, repaint


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--checkpoint",  default="checkpoints/best_model.pt")
    p.add_argument("--sample",      type=int,   default=0,   help="test set index")
    p.add_argument("--path_steps",  type=int,   default=150, help="robot walk length")
    p.add_argument("--resample",    type=int,   default=10,  help="RePaint r parameter")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--out",         default="inference_result.png")
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--time_dim",    type=int,   default=256)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load data ----
    test_ds   = OceanCurrentDataset(args.pickle, split=2)
    land_mask = test_ds.land_mask.numpy()        # (H, W) bool

    x0_true = test_ds[args.sample]               # (2, H, W)
    u_true  = x0_true[0].numpy()
    v_true  = x0_true[1].numpy()

    # ---- Simulate robot path ----
    path_mask = biased_walk_path(land_mask, n_steps=args.path_steps, seed=args.seed)
    print(f"Path covers {path_mask.sum()} / {(~land_mask).sum()} ocean cells "
          f"({100 * path_mask.sum() / (~land_mask).sum():.1f}%)")

    # Observed field: true values at path, 0 elsewhere
    x0_observed = x0_true.clone()
    path_t = torch.from_numpy(path_mask)
    x0_observed[:, ~path_t] = 0.0

    # ---- Load model ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch  = ckpt_args.get("base_ch",  args.base_ch)
    time_dim = ckpt_args.get("time_dim", args.time_dim)
    T        = ckpt_args.get("T",        args.T)

    model = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    diffusion = DDPM(T=T, beta_schedule="cosine", device=device)

    # ---- Run RePaint ----
    print(f"Running RePaint (T={T}, r={args.resample}) …")
    x0_pred = repaint(
        model, diffusion, x0_observed,
        path_mask, land_mask,
        r=args.resample, device=device,
    )
    u_pred = x0_pred[0].numpy()
    v_pred = x0_pred[1].numpy()

    # ---- Compute error ----
    err = np.sqrt((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)
    err[land_mask] = np.nan

    ocean_err = err[~land_mask]
    print(f"Mean speed error: {np.nanmean(ocean_err):.4f}")
    print(f"RMSE:             {np.sqrt(np.nanmean(ocean_err**2)):.4f}")

    # ---- Transpose for display: (94×44) → (44×94) so X=0-94, Y=0-44 ----
    u_true    = u_true.T
    v_true    = v_true.T
    u_pred    = u_pred.T
    v_pred    = v_pred.T
    land_mask = land_mask.T
    path_mask = path_mask.T
    err       = err.T

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Ground truth
    plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth")

    # 2. Observed path
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
    axes[1].set_title(f"Robot Path ({path_mask.sum()} cells)", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    ocean_p = mpatches.Patch(color="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(color="#d62728",                 label="Path")
    land_p  = mpatches.Patch(color="black",                   label="Land")
    axes[1].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

    # 3. Reconstructed field
    plot_field(axes[2], u_pred, v_pred, land_mask, "Reconstructed (RePaint)", cmap="cool")

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
    axes[3].set_title(f"Error  (RMSE={np.sqrt(np.nanmean(ocean_err**2)):.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    plt.suptitle(f"Ocean Current Inpainting — Test sample {args.sample}", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
