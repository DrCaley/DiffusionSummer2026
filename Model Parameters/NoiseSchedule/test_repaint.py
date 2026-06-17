"""
Test Repaint on the held-out test set and visualise one sample.

Mirrors the structure of Voronoi/test_voronoi.py.

Evaluation runs RePaint inference over --max_samples test samples using
biased-walk robot paths (default 10, since each run requires T reverse steps).

Output:
    Prints per-sample RMSE and summary stats to stdout.
    Saves a 2x2 visualisation for --sample to model_{schedule}_results/test_result.png

Usage (run from workspace root):
    python3 "Model Parameters/NoiseSchedule/test_repaint.py" --schedule cosine
    python3 "Model Parameters/NoiseSchedule/test_repaint.py" --schedule linear --sample 3 --max_samples 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset        import OceanCurrentDataset
from diffusion      import DDPM
from repaint_model  import Repaint
from repaint_infer  import biased_walk_path, repaint


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test Repaint on the held-out test set."
    )
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--schedule",    default="cosine",
                   choices=["linear", "cosine", "cosine_s0001", "cosine_s02", "cosine_s10", "quadratic", "sigmoid", "geometric"])
    p.add_argument("--checkpoint",  default=None,
                   help="Defaults to checkpoints_repaint_{schedule}/best_model_{schedule}.pt")
    p.add_argument("--sample",      type=int,   default=0,
                   help="Test set index to visualise.")
    p.add_argument("--path_steps",  type=int,   default=150)
    p.add_argument("--resample",    type=int,   default=10,  help="RePaint r parameter")
    p.add_argument("--seed",        type=int,   default=42,  help="Seed for visualisation path")
    p.add_argument("--max_samples", type=int,   default=10,
                   help="Number of test samples used for metric evaluation.")
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--time_dim",    type=int,   default=256)
    p.add_argument("--out",         default=None,
                   help="Defaults to model_{schedule}_results/test_result.png")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper  (matches test_voronoi.py exactly)
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

    script_dir = os.path.dirname(__file__)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            script_dir,
            "checkpoints",
            f"checkpoints_repaint_{args.schedule}",
            f"best_model_{args.schedule}.pt",
        )
    out_dir = os.path.join(script_dir, "results", f"model_{args.schedule}_results")
    if args.out is None:
        args.out = os.path.join(out_dir, "test_result.png")

    os.makedirs(out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Schedule   : {args.schedule}")
    print(f"Checkpoint : {args.checkpoint}")

    # ---- Data ----------------------------------------------------------------
    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    land_mask_np = test_ds.land_mask.numpy()
    H = test_ds.data.shape[2]
    W = test_ds.data.shape[3]
    n_ocean = int((~test_ds.land_mask).sum().item())

    print(f"Test set size : {len(test_ds)} samples")
    print(f"Grid          : {H} x {W}  ({n_ocean} ocean pixels)")

    # ---- Model ---------------------------------------------------------------
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)
    schedule  = ckpt_args.get("schedule", args.schedule)

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    saved_epoch = ckpt.get("epoch", "?")
    saved_val   = ckpt.get("val_loss", float("nan"))
    print(f"Checkpoint    : epoch {saved_epoch}, val_loss={saved_val:.5f}")

    noise_std = ckpt.get("noise_std", None)
    if noise_std is None:
        train_ds  = OceanCurrentDataset(args.pickle, split=0)
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())
        print(f"noise_std     : {noise_std:.5f}  (computed from training data)")
    else:
        print(f"noise_std     : {noise_std:.5f}  (from checkpoint)")

    diffusion = DDPM(T=T, beta_schedule=schedule, device=device, noise_std=noise_std)

    # ---- Test-set evaluation -------------------------------------------------
    n_eval   = min(args.max_samples, len(test_ds))
    rmse_all = []

    print(f"\n--- Evaluating {n_eval} test samples "
          f"(path_steps={args.path_steps}, r={args.resample}) ---")

    for i in range(n_eval):
        seed      = i * 7 + 1
        path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

        x0_true = test_ds[i]
        x0_obs  = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        x0_pred = repaint(
            model, diffusion, x0_obs,
            path_mask, land_mask_np,
            r=args.resample, device=device,
        )
        pred_np = x0_pred.numpy()
        true_np = x0_true.numpy()

        err  = np.sqrt((pred_np[0] - true_np[0]) ** 2 + (pred_np[1] - true_np[1]) ** 2)
        err[land_mask_np] = np.nan
        rmse = float(np.sqrt(np.nanmean(err[~land_mask_np] ** 2)))
        rmse_all.append(rmse)
        print(f"  Sample {i:3d}  RMSE={rmse:.5f}")

    mean_rmse = float(np.mean(rmse_all))
    std_rmse  = float(np.std(rmse_all))

    print(f"\n--- Test-set metrics ({n_eval} samples, schedule={schedule}) ---")
    print(f"  RMSE per sample : {[f'{r:.4f}' for r in rmse_all]}")
    print(f"  Mean RMSE       : {mean_rmse:.6f}")
    print(f"  Std RMSE        : {std_rmse:.6f}")

    # ---- Single-sample visualisation ----------------------------------------
    idx      = args.sample % len(test_ds)
    seed_viz = args.seed

    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed_viz)
    print(f"\nVisualising test sample {idx}  "
          f"(path seed={seed_viz}, {path_mask.sum()} / {n_ocean} ocean cells = "
          f"{100 * path_mask.sum() / n_ocean:.1f}%)")

    x0_true = test_ds[idx]
    x0_obs  = x0_true.clone()
    x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

    x0_pred = repaint(
        model, diffusion, x0_obs,
        path_mask, land_mask_np,
        r=args.resample, device=device,
    )
    pred_np = x0_pred.numpy()
    true_np = x0_true.numpy()

    err       = np.sqrt((pred_np[0] - true_np[0]) ** 2 + (pred_np[1] - true_np[1]) ** 2)
    err[land_mask_np] = np.nan
    ocean_err = err[~land_mask_np]
    viz_rmse  = float(np.sqrt(np.nanmean(ocean_err ** 2)))

    # Transpose to (W, H) for display — matches batch_infer.py convention
    u_true_d = true_np[0].T;  v_true_d = true_np[1].T
    u_pred_d = pred_np[0].T;  v_pred_d = pred_np[1].T
    land_d   = land_mask_np.T
    path_d   = path_mask.T
    err_d    = err.T

    # ---- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
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
    axes[1].set_title(f"Robot Path ({path_mask.sum()} cells, seed={seed_viz})", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728",                 label="Path")
    land_p  = mpatches.Patch(facecolor="black",                   label="Land")
    axes[1].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

    # 3. Reconstruction
    plot_field(axes[2], u_pred_d, v_pred_d, land_d,
               f"Reconstructed (RePaint — {schedule})", cmap="cool")

    # 4. Speed-error heatmap
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
    axes[3].set_title(f"Error  (RMSE={viz_rmse:.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    plt.suptitle(
        f"Repaint Reconstruction — Test sample {idx}  |  "
        f"schedule={schedule}  |  epoch {saved_epoch}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
