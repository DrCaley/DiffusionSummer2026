"""
Test VoronoiNet on the held-out test set and visualise one sample.

Output matches the style of visualize_infer.py:
  2x2 quiver layout -
      (1) Ground truth          (2) Voronoi sensor input
      (3) Reconstruction        (4) Speed-error heatmap

Usage
-----
    py "Voronoi/testing/test_voronoi.py"
    py "Voronoi/testing/test_voronoi.py" --sample 5 --n_sensors 30 --max_samples 10
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset import OceanCurrentDataset
from Voronoi.model.voronoi_model import VoronoiNet


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--checkpoint",  default="Voronoi/models/checkpoints_voronoi_scattered/best_model_scattered.pt")
    p.add_argument("--sample",      type=int,   default=0,    help="Test set index to visualise")
    p.add_argument("--n_sensors",   type=int,   default=50,   help="Number of sparse sensors")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--max_samples", type=int,   default=None, help="Limit evaluation to N samples")
    p.add_argument("--out",         default="Voronoi/voronoi_test_result.png")
    p.add_argument("--base_ch",     type=int,   default=64)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper  (matches visualize_infer.py exactly)
# ---------------------------------------------------------------------------

def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool"):
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
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
        cmap=cmap, clim=(0, np.nanpercentile(mq[mask], 98) if mask.any() else 1),
        scale=12, width=0.003, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

    # ---- Data ----------------------------------------------------------------
    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    land_mask_np = test_ds.land_mask.numpy()           # (H, W) bool
    land_mask    = test_ds.land_mask.to(device)

    H, W       = test_ds.data.shape[2], test_ds.data.shape[3]
    ocean_mask = (~land_mask).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    print(f"Test set size : {len(test_ds)} samples")
    print(f"Grid          : {H} x {W}  ({int(ocean_mask.sum().item())} ocean pixels)")

    # ---- Model ---------------------------------------------------------------
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = VoronoiNet(H=H, W=W, n_sensors=args.n_sensors,
                       in_ch=2, base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    saved_epoch = ckpt.get("epoch", "?")
    saved_val   = ckpt.get("val_loss", float("nan"))
    print(f"Checkpoint    : epoch {saved_epoch}, val_loss={saved_val:.5f}")

    # ---- Test-set evaluation -------------------------------------------------
    from torch.utils.data import DataLoader, Subset
    eval_ds = (Subset(test_ds, list(range(min(args.max_samples, len(test_ds)))))
               if args.max_samples is not None else test_ds)
    loader  = DataLoader(eval_ds, batch_size=16, shuffle=False)

    total_mse = 0.0
    total_rel = 0.0
    n_batches  = 0

    with torch.no_grad():
        for x0 in loader:
            x0   = x0.to(device)
            pred = model(x0, n_sensors=args.n_sensors, land_mask=land_mask)
            mse  = F.mse_loss(pred * ocean_mask, x0 * ocean_mask).item()
            num  = ((pred - x0) * ocean_mask).pow(2).sum().item()
            den  = (x0          * ocean_mask).pow(2).sum().item() + 1e-8
            total_mse += mse
            total_rel += (num / den) ** 0.5
            n_batches  += 1

    mean_mse  = total_mse / n_batches
    mean_rmse = mean_mse ** 0.5
    mean_rel  = total_rel / n_batches

    n_eval = len(eval_ds)
    print(f"\n--- Test-set metrics ({n_eval} samples, {args.n_sensors} sensors) ---")
    print(f"  MSE            : {mean_mse:.6f}")
    print(f"  RMSE           : {mean_rmse:.6f}  (same units as u/v)")
    print(f"  Relative L2    : {mean_rel*100:.2f}%")

    # ---- Single-sample visualisation ----------------------------------------
    idx       = args.sample % len(test_ds)
    x0_single = test_ds[idx].unsqueeze(0).to(device)  # (1, 2, H, W)

    with torch.no_grad():
        voronoi_input = model.voronoi.sample_and_tessellate(
            x0_single, n_sensors=args.n_sensors, avoid_land=land_mask
        )                                               # (1, C+1, H, W)
        pred_single = model.unet(voronoi_input)         # (1, 2, H, W)

    x0_np          = x0_single[0].cpu().numpy()        # (2, H, W)
    pred_np        = pred_single[0].cpu().numpy()
    vor_np         = voronoi_input[0, :2].cpu().numpy()
    sensor_mask_np = voronoi_input[0, 2].cpu().numpy()

    # speed error magnitude (H, W)
    err = np.sqrt((pred_np[0] - x0_np[0]) ** 2 + (pred_np[1] - x0_np[1]) ** 2)
    err[land_mask_np] = np.nan
    ocean_err = err[~land_mask_np]

    # Transpose to (W, H) so X=cols, Y=rows -- matches visualize_infer.py
    u_true   = x0_np[0].T;    v_true   = x0_np[1].T
    u_vor    = vor_np[0].T;    v_vor    = vor_np[1].T
    u_pred   = pred_np[0].T;   v_pred   = pred_np[1].T
    land_T   = land_mask_np.T
    err_T    = err.T
    sensor_T = sensor_mask_np.T

    # ---- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Ground truth
    plot_field(axes[0], u_true, v_true, land_T, "Ground Truth")

    # 2. Voronoi sensor input  (styled like the Robot Path panel)
    axes[1].imshow(
        land_T, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_T.shape[1] - 0.5, -0.5, land_T.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    sensor_disp = np.zeros_like(land_T, dtype=float)
    sensor_disp[sensor_T > 0.5] = 1.0
    axes[1].imshow(
        sensor_disp, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, land_T.shape[1] - 0.5, -0.5, land_T.shape[0] - 0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    axes[1].set_title(f"Voronoi Sensor Input ({args.n_sensors} sensors)", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    ocean_p  = mpatches.Patch(color="white",   edgecolor="gray", label="Ocean")
    sensor_p = mpatches.Patch(color="#d62728",                   label="Sensor")
    land_p   = mpatches.Patch(color="black",                     label="Land")
    axes[1].legend(handles=[ocean_p, sensor_p, land_p], loc="upper right", fontsize=8)

    # 3. Reconstruction
    plot_field(axes[2], u_pred, v_pred, land_T,
               "Reconstructed (VoronoiNet)", cmap="cool")

    # 4. Speed-error heatmap
    err_plot = np.ma.masked_where(land_T, err_T)
    im = axes[3].imshow(
        err_plot, origin="lower", cmap="hot_r", aspect="auto",
        extent=[-0.5, land_T.shape[1] - 0.5, -0.5, land_T.shape[0] - 0.5],
    )
    axes[3].imshow(
        land_T, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=[-0.5, land_T.shape[1] - 0.5, -0.5, land_T.shape[0] - 0.5],
        aspect="auto", zorder=1,
    )
    plt.colorbar(im, ax=axes[3], label="|error| speed", shrink=0.7)
    axes[3].set_title(f"Error  (RMSE={np.sqrt(np.nanmean(ocean_err**2)):.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    plt.suptitle(
        f"VoronoiNet Reconstruction — Test sample {idx}  |  "
        f"{args.n_sensors} sensors  |  epoch {saved_epoch}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
