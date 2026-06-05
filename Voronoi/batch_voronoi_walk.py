"""
Run VoronoiNet inference 10 times using biased-walk robot paths as sensor locations
(instead of random scattered points), matching the sensor pattern from RePaint.

Saves result_01.png ... result_10.png in Voronoi/voronoi_results_walk/
using the same 2x2 quiver layout as batch_results/.

Usage:
    py "Voronoi/batch_voronoi_walk.py"
    py "Voronoi/batch_voronoi_walk.py" --path_steps 150 --n_runs 10
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DDPM"))
from repaint_infer import biased_walk_path


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--checkpoint",  default="Voronoi/checkpoints_voronoi_scattered/best_model_scattered.pt")
    p.add_argument("--n_runs",      type=int,   default=10,  help="number of runs")
    p.add_argument("--path_steps",  type=int,   default=150, help="robot walk length")
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--out_dir",     default="Voronoi/voronoi_results_walk")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper  (identical to batch_infer.py / batch_voronoi.py)
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

def run_one(model, test_ds, land_mask_np, sample_idx, seed, path_steps, device):
    """
    Generate a biased-walk path mask, use those cells as Voronoi sensors,
    run VoronoiNet, return arrays ready for plotting.
    """
    # --- build sensor positions from the walk path ---
    path_mask = biased_walk_path(land_mask_np, n_steps=path_steps, seed=seed)
    sensor_rows, sensor_cols = np.where(path_mask)
    K = len(sensor_rows)

    x0_true = test_ds[sample_idx].unsqueeze(0).to(device)  # (1, 2, H, W)
    H, W    = land_mask_np.shape

    # Normalise positions to [-1, 1]
    rows_n = torch.tensor(sensor_rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
    cols_n = torch.tensor(sensor_cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
    sensor_positions = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)  # (1, K, 2)

    # Extract ground-truth values at path cells
    x0_flat = x0_true.reshape(1, 2, H * W)
    flat_idx = torch.tensor(
        sensor_rows * W + sensor_cols, dtype=torch.long, device=device
    ).unsqueeze(0).unsqueeze(0).expand(1, 2, K)             # (1, 2, K)
    sensor_values = torch.gather(x0_flat, 2, flat_idx)      # (1, 2, K)

    with torch.no_grad():
        voronoi_grid = model.voronoi.tessellate(sensor_values, sensor_positions)
        pred = model.unet(voronoi_grid)                      # (1, 2, H, W)

    x0_np   = x0_true[0].cpu().numpy()
    pred_np = pred[0].cpu().numpy()

    err = np.sqrt((pred_np[0] - x0_np[0]) ** 2 + (pred_np[1] - x0_np[1]) ** 2)
    err[land_mask_np] = np.nan
    rmse = float(np.sqrt(np.nanmean(err[~land_mask_np] ** 2)))

    # Transpose to (W, H) for display -- matches batch_infer.py convention
    return (
        x0_np[0].T, x0_np[1].T,
        pred_np[0].T, pred_np[1].T,
        land_mask_np.T,
        path_mask.T,
        err.T,
        rmse,
        K,
    )


# ---------------------------------------------------------------------------
# Save one 2x2 plot
# ---------------------------------------------------------------------------

def save_plot(u_true, v_true, u_pred, v_pred, land_mask, path_mask,
              err, rmse, label, sample_idx, seed, path_cells, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Ground truth
    plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth")

    # 2. Robot path (biased walk)  — same style as batch_infer.py
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
    ocean_p = mpatches.Patch(facecolor="white",   edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728",                   label="Path")
    land_p  = mpatches.Patch(facecolor="black",                     label="Land")
    axes[1].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

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
        f"Run {label}  —  Test sample {sample_idx}, path seed {seed}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}  (RMSE={rmse:.4f}, path_cells={path_cells})")


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
    print(f"Test set size: {len(test_ds)} samples")

    # ---- Model ---------------------------------------------------------------
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = VoronoiNet(H=test_ds.data.shape[2], W=test_ds.data.shape[3],
                       n_sensors=50, in_ch=2, base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    print(f"Path steps: {args.path_steps}")

    # ---- 10 runs: test samples 0-9, seeds matching batch_infer.py -----------
    rmse_list = []
    for i in range(args.n_runs):
        label      = i + 1
        sample_idx = i
        seed       = i * 7 + 1   # same seeds as batch_infer.py / batch_voronoi.py

        print(f"\n[Run {label}/{args.n_runs}]  test sample={sample_idx}, seed={seed}")
        (u_true, v_true, u_pred, v_pred,
         land_d, path_d, err_d, rmse, path_cells) = run_one(
            model, test_ds, land_mask_np,
            sample_idx, seed, args.path_steps, device,
        )
        rmse_list.append(rmse)
        out_path = os.path.join(args.out_dir, f"result_{label:02d}.png")
        save_plot(u_true, v_true, u_pred, v_pred, land_d, path_d, err_d,
                  rmse, label, sample_idx, seed, path_cells, out_path)

    print(f"\nAll done.")
    print(f"RMSE per run : {[f'{r:.4f}' for r in rmse_list]}")
    print(f"Mean RMSE    : {np.mean(rmse_list):.4f}  Std: {np.std(rmse_list):.4f}")


if __name__ == "__main__":
    main()
