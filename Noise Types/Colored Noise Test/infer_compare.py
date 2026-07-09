"""
Repaint inference comparison for white / pink / red noise models.

Loads all three best_model.pt checkpoints, runs RePaint on one test sample,
and saves a single comparison PNG to:
    outputs/comparison_<sample_idx>_seed<seed>.png

Layout (1 row × 5 columns):
    [Ground Truth]  [White pred]  [Pink pred]  [Red pred]  [RMSE bar chart]

Usage (run from workspace root):
    python "Colored Noise Test/infer_compare.py" --pickle data.pickle
    python "Colored Noise Test/infer_compare.py" --pickle data.pickle --sample 42 --seed 7
    python "Colored Noise Test/infer_compare.py" --pickle data.pickle --stride 10 --r 5
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "utils"))
# repaint_infer.py lives in each noise subfolder (identical); use white_noise copy
sys.path.insert(0, os.path.join(_HERE, "white_noise"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset       import OceanCurrentDataset
from repaint_infer import biased_walk_path, repaint   # from white_noise/
from repaint_model import Repaint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_diffusion(ckpt_path: str, noise_subdir: str, device: str):
    """Load a Repaint model + matching DDPM from a checkpoint."""
    # Import the correct diffusion module for this noise type
    import importlib.util
    diff_path = os.path.join(_HERE, noise_subdir, "diffusion.py")
    spec      = importlib.util.spec_from_file_location(f"diffusion_{noise_subdir}", diff_path)
    diff_mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diff_mod)

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    noise_std = ckpt.get("noise_std", 1.0)

    model     = Repaint(in_ch=2, base_ch=64, time_dim=256).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = diff_mod.DDPM(T=1000, device=device, noise_std=noise_std)
    return model, diffusion, ckpt


def rmse(pred: np.ndarray, true: np.ndarray, ocean_mask: np.ndarray) -> float:
    """Ocean-pixel RMSE between (2, H, W) fields."""
    diff = (pred - true)[:, ocean_mask]   # (2, N)
    return float(np.sqrt((diff ** 2).mean()))


def plot_field(ax, u, v, land_mask, title, vmax=None):
    """Quiver plot coloured by speed."""
    land_mask = np.rot90(land_mask, k=3)
    u_r       = np.rot90(u, k=3)
    v_r       = np.rot90(v, k=3)
    u         =  v_r
    v         = -u_r
    H, W      = land_mask.shape
    step = 2
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u[::step, ::step]
    vq = v[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    mask = ~land_mask[::step, ::step]
    if vmax is None:
        vmax = float(np.nanpercentile(mq[mask], 98)) if mask.any() else 1.0
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
        cmap="cool", clim=(0, vmax),
        scale=12, width=0.003, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    return vmax


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",  default="data.pickle")
    p.add_argument("--sample",  type=int, default=0,   help="Test-set sample index")
    p.add_argument("--seed",    type=int, default=42,  help="Path RNG seed")
    p.add_argument("--r",       type=int, default=10,  help="RePaint resampling iterations")
    p.add_argument("--stride",  type=int, default=10,  help="Timestep stride (10 = ~100 steps)")
    p.add_argument("--n_steps", type=int, default=150, help="Robot path length")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = args.out_dir or os.path.join(_HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Dataset ----
    ds        = OceanCurrentDataset(args.pickle, split=2)   # test split
    land_mask = ds.land_mask.numpy()   # (H, W) bool
    ocean_mask = ~land_mask

    x0_norm = ds[args.sample]          # (2, H, W) normalised tensor
    x0_np   = x0_norm.numpy()

    # ---- Robot path ----
    path_mask = biased_walk_path(land_mask, n_steps=args.n_steps, seed=args.seed)
    x0_known  = x0_norm.clone()
    x0_known[:, ~path_mask] = 0.0     # zero out unobserved cells

    print(f"Sample: {args.sample}  |  path cells: {path_mask.sum()}  |  seed: {args.seed}")

    # ---- Load all three models and run repaint ----
    models_cfg = [
        ("white_noise", "White noise"),
        ("pink_noise",  "Pink noise"),
        ("red_noise",   "Red noise"),
    ]

    preds = {}
    rmses = {}

    for subdir, label in models_cfg:
        ckpt_path = os.path.join(_HERE, subdir, "checkpoints", "best_model.pt")
        print(f"Loading {label} from {ckpt_path} ...")
        model, diffusion, ckpt = load_model_and_diffusion(ckpt_path, subdir, device)
        print(f"  epoch={ckpt.get('epoch','?')}  best_val={ckpt.get('val_loss', float('nan')):.5f}")

        pred = repaint(
            model, diffusion, x0_known,
            path_mask=path_mask,
            land_mask=land_mask,
            r=args.r,
            device=device,
            stride=args.stride,
        )
        pred_np = pred.numpy()   # (2, H, W)

        # Zero land
        pred_np[:, land_mask] = 0.0

        preds[subdir] = pred_np
        rmses[subdir] = rmse(pred_np, x0_np, ocean_mask)
        print(f"  RMSE = {rmses[subdir]:.5f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    fig.suptitle(
        f"Repaint comparison — sample {args.sample}, seed {args.seed}\n"
        f"stride={args.stride}, r={args.r}, path_steps={args.n_steps}",
        fontsize=12,
    )

    # Shared speed scale across all panels
    gt_speed = np.sqrt(x0_np[0] ** 2 + x0_np[1] ** 2)
    vmax     = float(np.nanpercentile(gt_speed[ocean_mask], 98))

    plot_field(axes[0], x0_np[0], x0_np[1], land_mask, "Ground truth", vmax=vmax)

    # Overlay path on ground-truth panel
    path_rot     = np.rot90(path_mask, k=3)
    path_overlay = np.ma.masked_where(~path_rot, np.ones(path_rot.shape, dtype=float))
    axes[0].imshow(
        path_overlay, origin="lower", cmap="autumn", alpha=0.45,
        extent=[-0.5, land_mask.shape[0] - 0.5, -0.5, land_mask.shape[1] - 0.5],
        zorder=1,
    )

    labels = ["White noise", "Pink noise", "Red noise"]
    subdirs = ["white_noise", "pink_noise", "red_noise"]
    for ax, subdir, label in zip(axes[1:4], subdirs, labels):
        r_val = rmses[subdir]
        plot_field(ax, preds[subdir][0], preds[subdir][1], land_mask,
                   f"{label}\nRMSE={r_val:.5f}", vmax=vmax)

    # RMSE bar chart
    ax_bar = axes[4]
    bar_labels = ["White", "Pink", "Red"]
    bar_vals   = [rmses[s] for s in subdirs]
    colors     = ["#4c9be8", "#e87d4c", "#c44ce8"]
    bars = ax_bar.bar(bar_labels, bar_vals, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, bar_vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_vals) * 0.01,
            f"{val:.5f}", ha="center", va="bottom", fontsize=9,
        )
    ax_bar.set_ylabel("RMSE (normalised units)")
    ax_bar.set_title("RMSE comparison")
    ax_bar.set_ylim(0, max(bar_vals) * 1.2)
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax_bar.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"comparison_s{args.sample}_seed{args.seed}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
