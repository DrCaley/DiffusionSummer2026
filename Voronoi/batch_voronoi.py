"""
Run VoronoiNet inference 10 times with different test samples and sensor patterns.
Saves individual 2x2 plots labelled result_01.png ... result_10.png.

Mirrors the structure of batch_infer.py exactly.

Usage:
    py "Voronoi/batch_voronoi.py"
    py "Voronoi/batch_voronoi.py" --n_runs 10 --n_sensors 50
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import OceanCurrentDataset
from voronoi_model import VoronoiNet


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--checkpoint", default="Voronoi/checkpoints_voronoi_scattered/best_model_scattered.pt")
    p.add_argument("--n_runs",     type=int, default=10,  help="number of runs")
    p.add_argument("--n_sensors",  type=int, default=50,  help="sensors per run")
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--out_dir",    default="Voronoi/voronoi_results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper  (identical to batch_infer.py)
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
    ax.set_xlabel("X"); ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# One inference run
# ---------------------------------------------------------------------------

def run_one(model, test_ds, land_mask_tensor, land_mask_np,
            sample_idx, seed, n_sensors, device):
    torch.manual_seed(seed)

    x0_true = test_ds[sample_idx].unsqueeze(0).to(device)  # (1, 2, H, W)

    with torch.no_grad():
        voronoi_input = model.voronoi.sample_and_tessellate(
            x0_true, n_sensors=n_sensors, avoid_land=land_mask_tensor
        )                                                    # (1, C+1, H, W)
        pred = model.unet(voronoi_input)                     # (1, 2, H, W)

    x0_np          = x0_true[0].cpu().numpy()               # (2, H, W)
    pred_np        = pred[0].cpu().numpy()
    sensor_mask_np = voronoi_input[0, 2].cpu().numpy()       # (H, W)

    err = np.sqrt((pred_np[0] - x0_np[0]) ** 2 + (pred_np[1] - x0_np[1]) ** 2)
    err[land_mask_np] = np.nan
    rmse = float(np.sqrt(np.nanmean(err[~land_mask_np] ** 2)))

    # Transpose to (W, H) for display -- matches batch_infer.py convention
    u_true_d   = x0_np[0].T
    v_true_d   = x0_np[1].T
    u_pred_d   = pred_np[0].T
    v_pred_d   = pred_np[1].T
    land_d     = land_mask_np.T
    sensor_d   = sensor_mask_np.T
    err_d      = err.T

    return u_true_d, v_true_d, u_pred_d, v_pred_d, land_d, sensor_d, err_d, rmse


# ---------------------------------------------------------------------------
# Save one 2x2 plot
# ---------------------------------------------------------------------------

def save_plot(u_true, v_true, u_pred, v_pred, land_mask, sensor_mask, err,
              rmse, label, sample_idx, seed, n_sensors, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Ground truth
    plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth")

    # 2. Voronoi sensor input
    axes[1].imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    sensor_disp = np.zeros_like(land_mask, dtype=float)
    sensor_disp[sensor_mask > 0.5] = 1.0
    axes[1].imshow(
        sensor_disp, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    n_sensors_shown = int((sensor_mask > 0.5).sum())
    axes[1].set_title(f"Voronoi Sensor Input ({n_sensors_shown} sensors, seed={seed})", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    ocean_p  = mpatches.Patch(facecolor="white",   edgecolor="gray", label="Ocean")
    sensor_p = mpatches.Patch(facecolor="#d62728",                   label="Sensor")
    land_p   = mpatches.Patch(facecolor="black",                     label="Land")
    axes[1].legend(handles=[ocean_p, sensor_p, land_p], loc="upper right", fontsize=8)

    # 3. Reconstruction
    plot_field(axes[2], u_pred, v_pred, land_mask,
               "Reconstructed (VoronoiNet)", cmap="cool")

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
    axes[3].set_title(f"Error  (RMSE={rmse:.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    plt.suptitle(
        f"Run {label}  —  Test sample {sample_idx}, sensor seed {seed}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}  (RMSE={rmse:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Data ----------------------------------------------------------------
    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    land_mask_np = test_ds.land_mask.numpy()
    land_mask    = test_ds.land_mask.to(device)
    print(f"Test set size: {len(test_ds)} samples")

    # ---- Model ---------------------------------------------------------------
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = VoronoiNet(H=test_ds.data.shape[2], W=test_ds.data.shape[3],
                       n_sensors=args.n_sensors, in_ch=2, base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # ---- 10 runs: test samples 0-9, distinct seeds --------------------------
    rmse_list = []
    for i in range(args.n_runs):
        label      = i + 1
        sample_idx = i          # test sample index
        seed       = i * 7 + 1  # distinct seeds spread apart (same pattern as batch_infer)

        print(f"\n[Run {label}/{args.n_runs}]  test sample={sample_idx}, seed={seed}")
        (u_true, v_true, u_pred, v_pred,
         land_d, sensor_d, err_d, rmse) = run_one(
            model, test_ds, land_mask, land_mask_np,
            sample_idx, seed, args.n_sensors, device,
        )
        rmse_list.append(rmse)
        out_path = os.path.join(args.out_dir, f"result_{label:02d}.png")
        save_plot(u_true, v_true, u_pred, v_pred, land_d, sensor_d, err_d,
                  rmse, label, sample_idx, seed, args.n_sensors, out_path)

    print(f"\nAll done.")
    print(f"RMSE per run : {[f'{r:.4f}' for r in rmse_list]}")
    print(f"Mean RMSE    : {np.mean(rmse_list):.4f}  Std: {np.std(rmse_list):.4f}")


if __name__ == "__main__":
    main()
