"""
Diffusion-variance analysis for RePaint noise schedules.

Runs batch_repaint exactly (sample_idx=i, seed=i*7+1) N_REPS times.
The path is identical each repeat (same seed); variance comes from the
diffusion model's internal stochasticity.

Seed scheme: seed = sample_idx * 7 + 1  (identical to batch_repaint, fixed)

Output:
  variance_{schedule}_results/result_{sample+1:02d}_rep{rep+1:02d}.png  (100 plots)
  variance_results/variance_path_{schedule}.png                          (summary)
  variance_results/variance_path_{schedule}.txt                          (stats)

Usage (run from workspace root):
    python3 NoiseSchedule/variance_repaint.py --schedule cosine
"""

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path, repaint


# ---------------------------------------------------------------------------
# Quiver helper  (identical to batch_repaint.py)
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


def save_plot(u_true, v_true, u_pred, v_pred, land_mask, path_mask, err, rmse,
              path_cells, sample_label, run_label, seed, schedule, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth")

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

    plot_field(axes[2], u_pred, v_pred, land_mask,
               f"Reconstructed (RePaint — {schedule})", cmap="cool")

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
        f"Sample {sample_label}, Run {run_label}  —  path seed {seed}  "
        f"[schedule={schedule}]",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}  (RMSE={rmse:.4f})")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--schedule",    default="cosine",
                   choices=["linear", "cosine", "quadratic", "sigmoid", "geometric"])
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--n_samples",   type=int, default=10,
                   help="Val samples to process (0..n_samples-1), matches batch_repaint")
    p.add_argument("--n_runs",      type=int, default=10,
                   help="Number of repetitions of the full batch_repaint pass")
    p.add_argument("--path_steps",  type=int, default=150)
    p.add_argument("--resample",    type=int, default=5,
                   help="RePaint r (default 5 for speed)")
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--time_dim",    type=int, default=256)
    p.add_argument("--out_dir",     default=None,
                   help="Directory for summary files. Defaults to variance_results/")
    return p.parse_args()


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
            f"checkpoints_repaint_{args.schedule}",
            f"best_model_{args.schedule}.pt",
        )
    if args.out_dir is None:
        args.out_dir = os.path.join(script_dir, "variance_results")
    os.makedirs(args.out_dir, exist_ok=True)

    # Per-schedule folder for the batch_repaint-style plots
    run_dir = os.path.join(script_dir, f"variance_{args.schedule}_results")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Schedule   : {args.schedule}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Samples    : {args.n_samples}   Repetitions: {args.n_runs}   r={args.resample}")
    print(f"Run plots  : {run_dir}\n")

    val_ds       = OceanCurrentDataset(args.pickle, split=1)
    land_mask_np = val_ds.land_mask.numpy()
    H, W         = val_ds.data.shape[2], val_ds.data.shape[3]

    model = Repaint(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt.get('epoch','?')}\n")

    diffusion = DDPM(T=args.T, beta_schedule=args.schedule, device=device)

    # rmse_grid[sample_idx, rep_idx]
    rmse_grid  = np.zeros((args.n_samples, args.n_runs), dtype=np.float32)
    preds_last = None  # store preds for last sample for the heatmap

    for ri in range(args.n_runs):
        print(f"\n[Repeat {ri+1}/{args.n_runs}]")

        for si in range(args.n_samples):
            sample_idx = si
            # exact batch_repaint seed formula — path is identical every repeat
            seed    = sample_idx * 7 + 1
            x0_true = val_ds[sample_idx]
            u_true  = x0_true[0].numpy()
            v_true  = x0_true[1].numpy()

            if ri == 0:
                preds_last = np.zeros((args.n_runs, 2, H, W), dtype=np.float32)

            path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)
            x0_obs    = x0_true.clone()
            x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

            x0_pred = repaint(model, diffusion, x0_obs, path_mask, land_mask_np,
                              r=args.resample, device=device)
            u_pred  = x0_pred[0].numpy()
            v_pred  = x0_pred[1].numpy()

            # accumulate preds for sample si across repeats (for heatmap of last sample)
            if si == args.n_samples - 1:
                if preds_last is None:
                    preds_last = np.zeros((args.n_runs, 2, H, W), dtype=np.float32)
                preds_last[ri, 0] = u_pred
                preds_last[ri, 1] = v_pred

            err  = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
            err[land_mask_np] = np.nan
            rmse = float(np.sqrt(np.nanmean(err[~land_mask_np]**2)))
            rmse_grid[si, ri] = rmse

            out_path = os.path.join(run_dir, f"result_{si+1:02d}_rep{ri+1:02d}.png")
            save_plot(
                u_true.T, v_true.T, u_pred.T, v_pred.T,
                land_mask_np.T, path_mask.T, err.T,
                rmse, int(path_mask.sum()),
                si + 1, ri + 1, seed, args.schedule, out_path,
            )

        per = rmse_grid[:, ri]
        print(f"  -> repeat mean={per.mean():.4f}  std={per.std():.4f}")

    # ---- Summary: box plot per sample + per-pixel std heatmap (last sample) -
    u_std   = preds_last[:, 0, :, :].std(axis=0)
    v_std   = preds_last[:, 1, :, :].std(axis=0)
    vec_std = np.sqrt(u_std**2 + v_std**2)
    vec_std[land_mask_np] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].boxplot(
        [rmse_grid[si] for si in range(args.n_samples)],
        labels=[f"S{si+1}" for si in range(args.n_samples)],
    )
    axes[0].set_ylabel("RMSE")
    axes[0].set_xlabel("Val sample")
    axes[0].set_title(
        f"{args.schedule} — RMSE distribution ({args.n_runs} repeats, r={args.resample})"
    )
    axes[0].grid(axis="y", alpha=0.3)

    vmax = np.nanpercentile(vec_std, 98)
    im   = axes[1].imshow(vec_std.T, origin="lower", cmap="hot",
                          vmin=0, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=axes[1], label="Prediction std")
    axes[1].set_title(
        f"{args.schedule} — Per-pixel std (sample {args.n_samples})\n"
        f"mean ocean std={np.nanmean(vec_std):.4f}"
    )
    axes[1].axis("off")

    plt.suptitle(f"Path variance — {args.schedule} schedule", fontsize=13)
    plt.tight_layout()
    out = os.path.join(args.out_dir, f"variance_path_{args.schedule}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")

    # ---- Save txt -----------------------------------------------------------
    txt = os.path.join(args.out_dir, f"variance_path_{args.schedule}.txt")
    with open(txt, "w") as f:
        f.write(f"Schedule   : {args.schedule}\n")
        f.write(f"N_samples  : {args.n_samples}\n")
        f.write(f"N_repeats  : {args.n_runs}\n")
        f.write(f"r          : {args.resample}\n\n")
        for si in range(args.n_samples):
            seed = si * 7 + 1
            f.write(f"Sample {si+1} (val_idx={si}, seed={seed}):\n")
            for ri in range(args.n_runs):
                f.write(f"  rep{ri+1:02d}  RMSE={rmse_grid[si,ri]:.4f}\n")
            m, s = rmse_grid[si].mean(), rmse_grid[si].std()
            f.write(f"  -> mean={m:.4f}  std={s:.4f}  "
                    f"min={rmse_grid[si].min():.4f}  max={rmse_grid[si].max():.4f}\n\n")
        overall = rmse_grid.flatten()
        f.write(f"Overall: mean={overall.mean():.4f}  std={overall.std():.4f}  "
                f"min={overall.min():.4f}  max={overall.max():.4f}\n")
    print(f"Saved: {txt}\nDone.")


if __name__ == "__main__":
    main()
