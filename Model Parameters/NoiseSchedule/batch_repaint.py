"""
Run RePaint inference 10 times on the validation set with different random robot paths
and save individual 2x2 plots labelled result_01.png ... result_10.png.

Mirrors the structure of DDPM/batch_infer.py exactly.

Usage (run from workspace root):
    python3 "Model Parameters/NoiseSchedule/batch_repaint.py" --schedule cosine
    python3 "Model Parameters/NoiseSchedule/batch_repaint.py" --schedule linear --n_runs 10 --path_steps 150
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # workspace root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # prefer local diffusion.py

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset        import OceanCurrentDataset
from diffusion      import DDPM
from repaint_model  import Repaint
from repaint_infer  import biased_walk_path, repaint


# ---------------------------------------------------------------------------
# Stdout tee  — ensures log is always written as batch_{schedule}.log
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror stdout to a file, preserving the exact same output."""
    def __init__(self, path):
        self._file   = open(path, "w", buffering=1)
        self._stdout = sys.stdout
        sys.stdout   = self
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._stdout
        self._file.close()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--schedule",    default="cosine",
                   choices=["linear", "cosine", "cosine_s0001", "cosine_s02", "cosine_s10",
                            "quadratic", "sigmoid", "geometric"])
    p.add_argument("--checkpoint",  default=None,
                   help="Path to checkpoint. Defaults to "
                        "checkpoints_repaint_{schedule}/best_model_{schedule}.pt")
    p.add_argument("--n_runs",      type=int, default=10,  help="number of runs")
    p.add_argument("--path_steps",  type=int, default=150, help="robot walk length")
    p.add_argument("--resample",    type=int, default=10,  help="RePaint r parameter")
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--time_dim",    type=int, default=256)
    p.add_argument("--out_dir",     default=None,
                   help="Output directory. Defaults to model_{schedule}_results/")
    p.add_argument("--start_run",   type=int, default=1,
                   help="Run number to start from (1-indexed, default=1)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper  (identical to DDPM/batch_infer.py)
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

def run_one(model, diffusion, val_ds, land_mask_np, sample_idx, seed, args, device):
    x0_true = val_ds[sample_idx]
    u_true  = x0_true[0].numpy()
    v_true  = x0_true[1].numpy()

    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

    x0_observed = x0_true.clone()
    path_t = torch.from_numpy(path_mask)
    x0_observed[:, ~path_t] = 0.0

    x0_pred = repaint(
        model, diffusion, x0_observed,
        path_mask, land_mask_np,
        r=args.resample, device=device,
    )
    u_pred = x0_pred[0].numpy()
    v_pred = x0_pred[1].numpy()

    err = np.sqrt((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)
    err[land_mask_np] = np.nan
    ocean_err = err[~land_mask_np]
    rmse = float(np.sqrt(np.nanmean(ocean_err ** 2)))

    # Transpose for display: (94,44) -> (44,94)
    return (
        u_true.T, v_true.T,
        u_pred.T, v_pred.T,
        land_mask_np.T,
        path_mask.T,
        err.T,
        rmse,
        path_mask.sum(),
    )


# ---------------------------------------------------------------------------
# Save one 2x2 plot
# ---------------------------------------------------------------------------

def save_plot(u_true, v_true, u_pred, v_pred, land_mask, path_mask, err, rmse,
              path_cells, label, sample_idx, seed, schedule, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Ground truth
    plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth")

    # 2. Robot path
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
    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728",                 label="Path")
    land_p  = mpatches.Patch(facecolor="black",                   label="Land")
    axes[1].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

    # 3. Reconstruction
    plot_field(axes[2], u_pred, v_pred, land_mask,
               f"Reconstructed (RePaint — {schedule})", cmap="cool")

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
        f"Run {label}  —  Val sample {sample_idx}, path seed {seed}  "
        f"[schedule={schedule}]",
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

    script_dir = os.path.dirname(__file__)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            script_dir,
            "checkpoints",
            f"checkpoints_repaint_{args.schedule}",
            f"best_model_{args.schedule}.pt",
        )
    if args.out_dir is None:
        args.out_dir = os.path.join(script_dir, "results", f"model_{args.schedule}_results")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"batch_{args.schedule}.log")
    tee = _Tee(log_path)

    print(f"Device     : {device}")
    print(f"Schedule   : {args.schedule}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output dir : {args.out_dir}")

    # ---- Load data ----
    val_ds    = OceanCurrentDataset(args.pickle, split=1)
    land_mask = val_ds.land_mask.numpy()

    # ---- Load model ----
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)
    schedule  = ckpt_args.get("schedule", args.schedule)

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}), T={T}, schedule={schedule}")

    noise_std = ckpt.get("noise_std", None)
    if noise_std is None:
        train_ds  = OceanCurrentDataset(args.pickle, split=0)
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())
        print(f"noise_std : {noise_std:.5f}  (computed from training data)")
    else:
        print(f"noise_std : {noise_std:.5f}  (from checkpoint)")

    diffusion = DDPM(T=T, beta_schedule=schedule, device=device, noise_std=noise_std)

    # ---- 10 runs: val samples 0–9, seeds spread apart ----
    rmse_list = []
    for i in range(args.n_runs):
        label      = i + 1
        if label < args.start_run:
            continue
        sample_idx = i
        seed       = i * 7 + 1

        print(f"\n[Run {label}/{args.n_runs}]  val sample={sample_idx}, seed={seed}")
        (u_true, v_true, u_pred, v_pred,
         land_d, path_d, err_d, rmse, path_cells) = run_one(
            model, diffusion, val_ds, land_mask,
            sample_idx, seed, args, device,
        )
        rmse_list.append(rmse)
        out_path = os.path.join(args.out_dir, f"result_{label:02d}.png")
        save_plot(u_true, v_true, u_pred, v_pred, land_d, path_d, err_d,
                  rmse, path_cells, label, sample_idx, seed, schedule, out_path)

    print(f"\nAll done.")
    print(f"RMSE per run : {[f'{r:.4f}' for r in rmse_list]}")
    print(f"Mean RMSE    : {np.mean(rmse_list):.4f}   Std: {np.std(rmse_list):.4f}")
    tee.close()


if __name__ == "__main__":
    main()
