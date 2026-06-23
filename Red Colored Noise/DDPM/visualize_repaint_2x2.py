"""2x2 RePaint visualization matching the attached figure layout.

This script renders:
  - Ground Truth
  - Robot Path
  - Reconstructed field
  - Error magnitude

It uses the trained colored-noise DDPM with RePaint sampling and defaults to
the same sample/seed pair shown in the attached reference image.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_data
from model.unet import UNet
from testing.repaint.repaint_infer import repaint_sample


def simulate_robot_path(ocean_mask: np.ndarray, n_steps: int, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    H, W = ocean_mask.shape
    ocean_cells = np.argwhere(ocean_mask)
    pos = ocean_cells[rng.integers(len(ocean_cells))].copy()
    path = [pos.copy()]
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

    for _ in range(n_steps - 1):
        rng.shuffle(directions)
        moved = False
        for d in directions:
            npos = pos + d
            r, c = npos
            if 0 <= r < H and 0 <= c < W and ocean_mask[r, c]:
                pos = npos.copy()
                moved = True
                break
        if not moved:
            pos = ocean_cells[rng.integers(len(ocean_cells))].copy()
        path.append(pos.copy())

    return np.array(path)


def transpose_for_display(u, v, land_mask):
    return u.T, v.T, land_mask.T


def transpose_path_for_display(path):
    return path


def plot_vector_field(ax, u, v, land_mask, title, vmax=None, quiver_scale=0.35):
    speed = np.hypot(u, v)
    if vmax is None:
        vmax = float(np.nanpercentile(speed[~land_mask], 98))

    H, W = u.shape
    step = max(1, min(H, W) // 10)
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    ocean = ~land_mask[yy, xx]

    land_rgba = np.zeros((H, W, 4), dtype=np.float32)
    land_rgba[land_mask] = [0.0, 0.0, 0.0, 1.0]
    ax.imshow(land_rgba, origin="lower", aspect="auto", extent=[-0.5, W - 0.5, -0.5, H - 0.5])
    ax.set_facecolor("white")

    ax.quiver(
        (xx + 0.5)[ocean],
        (yy + 0.5)[ocean],
        u[yy, xx][ocean],
        v[yy, xx][ocean],
        speed[yy, xx][ocean],
        cmap="cool",
        norm=Normalize(vmin=0, vmax=vmax),
        scale=quiver_scale,
        scale_units="xy",
        angles="xy",
        pivot="mid",
        headwidth=3,
        headlength=4,
        width=0.006,
        alpha=0.95,
    )

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap="cool")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speed", fontsize=8)
    cbar.ax.tick_params(labelsize=8)



def plot_robot_path(ax, u, v, land_mask, path, title, vmax=None):
    H, W = u.shape
    land_rgba = np.zeros((H, W, 4), dtype=np.float32)
    land_rgba[land_mask] = [0.0, 0.0, 0.0, 1.0]
    ax.imshow(land_rgba, origin="lower", aspect="auto", extent=[-0.5, W - 0.5, -0.5, H - 0.5])
    ax.set_facecolor("white")

    ax.plot(path[:, 0], path[:, 1], color="#8c2d3a", linewidth=5.0, solid_capstyle="round", zorder=5)
    ax.scatter(path[:, 0], path[:, 1], s=14, marker="s", color="#8c2d3a", edgecolors="none", zorder=6)

    legend_handles = [
        Patch(facecolor="white", edgecolor="#9c9c9c", label="Ocean"),
        Patch(facecolor="#d62728", edgecolor="#d62728", label="Path"),
        Patch(facecolor="black", edgecolor="black", label="Land"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True, framealpha=1.0)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plot_error(ax, diff_u, diff_v, land_mask, title, vmax=None):
    err = np.hypot(diff_u, diff_v)
    err[land_mask] = np.nan
    if vmax is None:
        vmax = float(np.nanpercentile(err, 98))

    H, W = err.shape
    im = ax.imshow(err, origin="lower", aspect="auto", cmap="hot_r", vmin=0, vmax=vmax, extent=[-0.5, W - 0.5, -0.5, H - 0.5])

    land_rgba = np.zeros((H, W, 4), dtype=np.float32)
    land_rgba[land_mask] = [0.0, 0.0, 0.0, 1.0]
    ax.imshow(land_rgba, origin="lower", aspect="auto", extent=[-0.5, W - 0.5, -0.5, H - 0.5])

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|error| speed", fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits, land_mask = load_data(args.pickle)
    land_np = land_mask.numpy()
    ocean_np = ~land_np

    val_data = splits[1]
    idx = args.sample % len(val_data)
    x0_np = val_data[idx]

    rng = np.random.default_rng(args.seed)
    path = simulate_robot_path(ocean_np, args.path_steps, rng=rng)

    known_mask_np = np.zeros(x0_np.shape[1:], dtype=bool)
    for r, c in path:
        known_mask_np[r, c] = True

    pct = 100 * known_mask_np.sum() / ocean_np.sum()
    print(f"Robot path: {known_mask_np.sum()} pixels ({pct:.1f}% of ocean)")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = UNet(
        in_channels=2,
        base_ch=saved_args.get("base_ch", 128),
        ch_mults=tuple(saved_args.get("ch_mults", [1, 2, 2])),
        time_dim=saved_args.get("time_dim", 512),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    noise_alpha = saved_args.get("noise_alpha", 2.0)
    print(f"Loaded epoch {ckpt.get('epoch', '?')}  val_loss={ckpt.get('val_loss', float('nan')):.5f}")

    x0_t = torch.from_numpy(x0_np).unsqueeze(0).to(device)
    km_t = torch.from_numpy(known_mask_np)
    x_pred_t = repaint_sample(model, x0_t, km_t, T=args.T, device=device, resample=args.resample, noise_alpha=noise_alpha)
    x0_pred_np = x_pred_t[0].cpu().numpy()

    diff = x0_pred_np - x0_np
    diff[:, land_np] = 0.0
    ocean_diff = diff[:, ocean_np]
    rmse = float(np.sqrt((ocean_diff ** 2).mean()))
    mae = float(np.abs(ocean_diff).mean())
    print(f"RMSE = {rmse:.4f}   MAE = {mae:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor="white")
    title = os.path.splitext(os.path.basename(args.out))[0]
    fig.suptitle(title, fontsize=14)

    u_gt, v_gt = x0_np[0], x0_np[1]
    u_pr, v_pr = x0_pred_np[0], x0_pred_np[1]
    u_gt_disp, v_gt_disp, land_disp = transpose_for_display(u_gt, v_gt, land_np)
    u_pr_disp, v_pr_disp, _ = transpose_for_display(u_pr, v_pr, land_np)
    diff_u_disp, diff_v_disp, _ = transpose_for_display(diff[0], diff[1], land_np)
    path_disp = transpose_path_for_display(path)
    ocean_disp = ~land_disp
    vmax_gt = float(np.nanpercentile(np.hypot(u_gt_disp[ocean_disp], v_gt_disp[ocean_disp]), 98))

    plot_vector_field(axes[0, 0], u_gt_disp, v_gt_disp, land_disp, "Ground Truth", vmax=vmax_gt, quiver_scale=args.quiver_scale)
    plot_robot_path(
        axes[0, 1],
        u_gt_disp,
        v_gt_disp,
        land_disp,
        path_disp,
        f"Robot Path ({known_mask_np.sum()} cells, seed={args.seed})",
        vmax=vmax_gt,
    )
    plot_vector_field(axes[1, 0], u_pr_disp, v_pr_disp, land_disp, "Reconstructed (RePaint)", vmax=vmax_gt, quiver_scale=args.quiver_scale)
    plot_error(axes[1, 1], diff_u_disp, diff_v_disp, land_disp, f"Error (RMSE={rmse:.4f})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pickle", default="../data.pickle")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--path_steps", type=int, default=150)
    parser.add_argument("--resample", type=int, default=10)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiver_scale", type=float, default=0.22)
    parser.add_argument("--out_dir", default="testing/results")
    parser.add_argument("--out", default="infer_repaint_T1000_2x2.png")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())