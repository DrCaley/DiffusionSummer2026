"""
visualize_infer.py – Single-sample RePaint inference + visualisation.

Generates a 3-panel figure:
  Panel 1 – Ground truth vector field
  Panel 2 – Reconstructed field (robot path overlaid)
  Panel 3 – Absolute error |predicted − truth|

Land is oriented at the TOP of each panel (origin="upper", row 0 = top,
land peninsula is in rows 0–~50).
RMSE and MAE are printed and annotated on the figure title.
Output is saved to DDPM/testing/results/.

Usage
-----
cd DDPM
python visualize_infer.py \\
    --checkpoint checkpoints/model_DDPM_MSE_coloredGaussian_cosine.pt \\
    --pickle ../data.pickle \\
    --sample 0 \\
    --method repaint \\
    --path_steps 150 \\
    --resample 10
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_data
from model.unet import UNet
from testing.repaint.repaint_infer import repaint_sample
from testing.DPS.dps_infer import dps_sample


# ---------------------------------------------------------------------------
# Robot path simulation
# ---------------------------------------------------------------------------

def simulate_robot_path(ocean_mask: np.ndarray, n_steps: int, rng=None) -> np.ndarray:
    """Random walk on ocean pixels; returns (n_steps, 2) int array of (row, col)."""
    if rng is None:
        rng = np.random.default_rng()

    H, W = ocean_mask.shape
    ocean_cells = np.argwhere(ocean_mask)
    pos  = ocean_cells[rng.integers(len(ocean_cells))].copy()
    path = [pos.copy()]

    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

    for _ in range(n_steps - 1):
        rng.shuffle(directions)
        moved = False
        for d in directions:
            npos = pos + d
            r, c = npos
            if 0 <= r < H and 0 <= c < W and ocean_mask[r, c]:
                pos   = npos.copy()
                moved = True
                break
        if not moved:
            pos = ocean_cells[rng.integers(len(ocean_cells))].copy()
        path.append(pos.copy())

    return np.array(path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_field(ax, u, v, land_mask, title, cmap="RdBu_r", vmax=None, path=None):
    """Speed colour map + quiver arrows.  Land pixels drawn in grey."""
    speed = np.hypot(u, v)
    if vmax is None:
        vmax = float(np.nanpercentile(speed[~land_mask], 98))

    H, W = u.shape
    step = max(1, min(H, W) // 20)
    ys   = np.arange(0, H, step)
    xs   = np.arange(0, W, step)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")

    ax.imshow(speed, origin="upper", aspect="auto", cmap=cmap,
              vmin=0, vmax=vmax, extent=[0, W, H, 0])

    land_rgba = np.zeros((H, W, 4), dtype=np.float32)
    land_rgba[land_mask] = [0.5, 0.5, 0.5, 1.0]
    ax.imshow(land_rgba, origin="upper", aspect="auto", extent=[0, W, H, 0])

    ax.quiver(
        XX + 0.5, YY + 0.5,
        u[YY, XX], -v[YY, XX],
        scale=15, scale_units="inches",
        headwidth=3, headlength=4, width=0.003,
        color="k", alpha=0.6,
    )

    if path is not None:
        ax.plot(path[:, 1] + 0.5, path[:, 0] + 0.5,
                "y-", linewidth=0.8, alpha=0.7, label="robot path")
        ax.plot(path[0, 1] + 0.5, path[0, 0] + 0.5,
                "g^", markersize=5, label="start")
        ax.plot(path[-1, 1] + 0.5, path[-1, 0] + 0.5,
                "rs", markersize=5, label="end")
        ax.legend(fontsize=7, loc="lower right")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("W axis (N-S, 44 cols)")
    ax.set_ylabel("H axis (E-W, 94 rows)  [land at top]")


def plot_error(ax, diff_u, diff_v, land_mask, title, vmax=None):
    """Absolute error magnitude as colour map."""
    err = np.hypot(diff_u, diff_v)
    err[land_mask] = np.nan
    if vmax is None:
        vmax = float(np.nanpercentile(err, 98))

    H, W = err.shape
    im = ax.imshow(err, origin="upper", aspect="auto", cmap="hot_r",
                   vmin=0, vmax=vmax, extent=[0, W, H, 0])

    land_rgba = np.zeros((H, W, 4), dtype=np.float32)
    land_rgba[land_mask] = [0.5, 0.5, 0.5, 1.0]
    ax.imshow(land_rgba, origin="upper", aspect="auto", extent=[0, W, H, 0])

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("W axis (N-S, 44 cols)")
    ax.set_ylabel("H axis (E-W, 94 rows)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|error|")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    splits, land_mask = load_data(args.pickle)
    land_np  = land_mask.numpy()
    ocean_np = ~land_np

    val_data = splits[1]
    idx      = args.sample % len(val_data)
    x0_np    = val_data[idx]                  # (2, H, W)

    # ---- Robot path ----
    rng  = np.random.default_rng(args.seed)
    path = simulate_robot_path(ocean_np, args.path_steps, rng=rng)

    known_mask_np = np.zeros(x0_np.shape[1:], dtype=bool)
    for r, c in path:
        known_mask_np[r, c] = True
    pct = 100 * known_mask_np.sum() / ocean_np.sum()
    print(f"Robot path: {known_mask_np.sum()} pixels ({pct:.1f}% of ocean)")

    # ---- Load model ----
    ckpt       = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    print(f"Loaded epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.5f}")

    # ---- Inference ----
    x0_t  = torch.from_numpy(x0_np).unsqueeze(0).to(device)
    km_t  = torch.from_numpy(known_mask_np)

    method = args.method.lower()
    print(f"Running {method.upper()} inference …")

    if method == "repaint":
        x_pred_t = repaint_sample(
            model, x0_t, km_t, T=args.T, device=device,
            resample=args.resample, noise_alpha=noise_alpha,
        )
    elif method == "dps":
        x_pred_t = dps_sample(
            model, x0_t, km_t, T=args.T, device=device,
            step_size=args.dps_step, noise_alpha=noise_alpha,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'repaint' or 'dps'.")

    x0_pred_np = x_pred_t[0].cpu().numpy()   # (2, H, W)

    # ---- Metrics ----
    diff = x0_pred_np - x0_np                 # (2, H, W)
    diff[:, land_np] = 0.0
    ocean_diff = diff[:, ocean_np]
    rmse = float(np.sqrt((ocean_diff ** 2).mean()))
    mae  = float(np.abs(ocean_diff).mean())
    print(f"RMSE = {rmse:.4f}   MAE = {mae:.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        f"Colored Gaussian Noise DDPM ({method.upper()})  |  Val sample {idx}\n"
        f"RMSE = {rmse:.4f}   MAE = {mae:.4f}   "
        f"Path coverage = {pct:.1f}%",
        fontsize=12,
    )

    u_gt, v_gt  = x0_np[0],      x0_np[1]
    u_pr, v_pr  = x0_pred_np[0], x0_pred_np[1]
    vmax_gt     = float(np.nanpercentile(np.hypot(u_gt[ocean_np], v_gt[ocean_np]), 98))

    plot_field(axes[0], u_gt, v_gt, land_np, "Ground Truth", vmax=vmax_gt)
    plot_field(axes[1], u_pr, v_pr, land_np,
               f"Prediction ({method.upper()}) + robot path",
               vmax=vmax_gt, path=path)
    plot_error(axes[2], diff[0], diff[1], land_np,
               f"|Error|  (RMSE={rmse:.4f}, MAE={mae:.4f})")

    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pickle",     default="../data.pickle")
    p.add_argument("--sample",     type=int,   default=0)
    p.add_argument("--method",     default="repaint", choices=["repaint", "dps"])
    p.add_argument("--path_steps", type=int,   default=150)
    p.add_argument("--resample",   type=int,   default=10)
    p.add_argument("--dps_step",   type=float, default=1.0)
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out_dir",    default="testing/results")
    p.add_argument("--out",        default="infer_sample.png")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
