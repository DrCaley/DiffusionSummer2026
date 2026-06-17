"""
Per-sample variance comparison across all 5 noise schedules.

For each of the 10 validation samples (val_idx 0-9, seeds 1,8,15,...,64)
runs 10 RePaint repeats per schedule (5 schedules), then saves one PNG:

  Layout per PNG
  ──────────────
  Row 1 (2 panels):
    [A] Robot path  (same for all schedules — seed fixed per sample)
    [B] RMSE boxplot — all 5 schedules side-by-side, 10 reps each

  Row 2 (5 panels):
    [C–G] Per-pixel std heatmap for each schedule
          cmap = hot_r  (matches test_repaint.py error maps)
          land = black overlay  (matches test_repaint.py convention)

Output:
    NoiseSchedule/variance_results/per_sample/sample_{si+1:02d}.png  (10 files)

Usage (run from workspace root):
    py NoiseSchedule/variance_by_sample.py
    py NoiseSchedule/variance_by_sample.py --resample 5 --n_runs 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path, repaint


SCHEDULES = ["linear", "cosine", "quadratic", "sigmoid", "geometric"]
COLORS    = {
    "linear":    "#d62728",
    "cosine":    "#1f77b4",
    "quadratic": "#2ca02c",
    "sigmoid":   "#ff7f0e",
    "geometric": "#9467bd",
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--n_samples",  type=int, default=10)
    p.add_argument("--n_runs",     type=int, default=10)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--resample",   type=int, default=5,
                   help="RePaint r parameter")
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--out_dir",    default=None,
                   help="Output directory. Defaults to "
                        "NoiseSchedule/variance_results/per_sample/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load one model
# ---------------------------------------------------------------------------

def load_model(schedule, script_dir, base_ch, time_dim, device):
    ckpt_path = os.path.join(
        script_dir,
        f"checkpoints_repaint_{schedule}",
        f"best_model_{schedule}.pt",
    )
    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded {schedule:10s}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}")
    return model


# ---------------------------------------------------------------------------
# Plot helpers  (matching test_repaint.py conventions)
# ---------------------------------------------------------------------------

def plot_path(ax, land_mask, path_mask, seed):
    H, W = land_mask.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    pd = np.zeros_like(land_mask, dtype=float)
    pd[path_mask] = 1.0
    ax.imshow(
        pd, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto",
        zorder=1, vmin=0, vmax=1,
    )
    n_path   = int(path_mask.sum())
    n_ocean  = int((~land_mask).sum())
    ax.set_title(
        f"Robot Path\nseed={seed}  {n_path}/{n_ocean} ocean cells "
        f"({100*n_path/n_ocean:.1f}%)",
        fontsize=10,
    )
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728",                 label="Path")
    land_p  = mpatches.Patch(facecolor="black",                   label="Land")
    ax.legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=7)


def plot_std_heatmap(ax, vec_std, land_mask, schedule, vmax=None):
    """Per-pixel prediction std — hot_r + black land, matching test_repaint.py."""
    H, W = vec_std.shape
    if vmax is None:
        vmax = np.nanpercentile(vec_std, 98)

    std_masked = np.ma.masked_where(land_mask, vec_std)
    im = ax.imshow(
        std_masked, origin="lower", cmap="hot_r", aspect="auto",
        vmin=0, vmax=vmax,
        extent=[-0.5, W - 0.5, -0.5, H - 0.5],
    )
    # Black land overlay — identical to test_repaint.py error panel
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5],
        aspect="auto", zorder=1,
    )
    plt.colorbar(im, ax=ax, label="Pred std", shrink=0.8)
    mean_std = float(np.nanmean(vec_std))
    ax.set_title(f"{schedule}\nmean std={mean_std:.4f}", fontsize=9)
    ax.axis("off")
    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.out_dir is None:
        args.out_dir = os.path.join(script_dir, "variance_results", "per_sample")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device   : {device}")
    print(f"Samples  : {args.n_samples}   Runs per sample per schedule: {args.n_runs}")
    print(f"Out dir  : {args.out_dir}\n")

    # ---- Data ----------------------------------------------------------------
    val_ds       = OceanCurrentDataset(args.pickle, split=1)
    land_mask_np = val_ds.land_mask.numpy()       # (H, W)
    H = val_ds.data.shape[2]
    W = val_ds.data.shape[3]

    # ---- Load all models -----------------------------------------------------
    print("Loading checkpoints...")
    models     = {}
    diffusions = {}
    for sched in SCHEDULES:
        models[sched]     = load_model(sched, script_dir, args.base_ch, args.time_dim, device)
        diffusions[sched] = DDPM(T=args.T, beta_schedule=sched, device=device)

    # ---- Inference -----------------------------------------------------------
    # preds[sched][si, ri, ch, H, W]
    preds     = {s: np.zeros((args.n_samples, args.n_runs, 2, H, W), dtype=np.float32)
                 for s in SCHEDULES}
    rmse_grid = {s: np.zeros((args.n_samples, args.n_runs), dtype=np.float32)
                 for s in SCHEDULES}
    path_masks = {}  # path_masks[si] — fixed per sample

    total = len(SCHEDULES) * args.n_samples * args.n_runs
    done  = 0

    for si in range(args.n_samples):
        seed    = si * 7 + 1
        x0_true = val_ds[si]
        u_true  = x0_true[0].numpy()
        v_true  = x0_true[1].numpy()

        path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)
        path_masks[si] = path_mask

        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        for sched in SCHEDULES:
            for ri in range(args.n_runs):
                x0_pred = repaint(
                    models[sched], diffusions[sched], x0_obs,
                    path_mask, land_mask_np,
                    r=args.resample, device=device,
                )
                u_pred = x0_pred[0].numpy()
                v_pred = x0_pred[1].numpy()

                preds[sched][si, ri, 0] = u_pred
                preds[sched][si, ri, 1] = v_pred

                err  = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
                err[land_mask_np] = np.nan
                rmse = float(np.sqrt(np.nanmean(err[~land_mask_np]**2)))
                rmse_grid[sched][si, ri] = rmse

                done += 1
                print(f"  [{done}/{total}] sample={si+1}  {sched:10s}  "
                      f"rep={ri+1}  RMSE={rmse:.4f}")

    # ---- Generate per-sample PNGs -------------------------------------------
    print("\nGenerating per-sample figures...")

    for si in range(args.n_samples):
        seed      = si * 7 + 1
        path_mask = path_masks[si]

        # Compute per-pixel std for each schedule (transposed to W×H for imshow)
        std_maps = {}
        vmaxes   = []
        for sched in SCHEDULES:
            u_std = preds[sched][si, :, 0, :, :].std(axis=0)   # (H, W)
            v_std = preds[sched][si, :, 1, :, :].std(axis=0)
            vs    = np.sqrt(u_std**2 + v_std**2)
            vs[land_mask_np] = np.nan
            std_maps[sched] = vs.T   # (W, H) for imshow origin="lower"
            vmaxes.append(np.nanpercentile(vs, 98))

        # Shared vmax across all 5 heatmaps so they're comparable
        shared_vmax = float(np.nanmax(vmaxes))

        # ---- Figure layout --------------------------------------------------
        # Row 0: path (col 0-1) | boxplot (col 2-4)
        # Row 1: 5 heatmaps (one per schedule)
        fig = plt.figure(figsize=(26, 10))
        gs  = gridspec.GridSpec(
            2, 5,
            figure=fig,
            hspace=0.35, wspace=0.3,
        )

        ax_path = fig.add_subplot(gs[0, 0:2])
        ax_box  = fig.add_subplot(gs[0, 2:5])

        land_d = land_mask_np.T
        plot_path(ax_path, land_d, path_mask.T, seed)

        # Boxplot: all 5 schedules for this sample
        box_data = [rmse_grid[s][si] for s in SCHEDULES]
        bp = ax_box.boxplot(
            box_data,
            positions=range(len(SCHEDULES)),
            widths=0.4,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, sched in zip(bp["boxes"], SCHEDULES):
            patch.set_facecolor(COLORS[sched])
            patch.set_alpha(0.6)

        # Strip jitter overlay
        rng = np.random.default_rng(42)
        for i_s, sched in enumerate(SCHEDULES):
            jitter = rng.uniform(-0.15, 0.15, args.n_runs)
            ax_box.scatter(
                i_s + jitter, rmse_grid[sched][si],
                color=COLORS[sched], alpha=0.8, zorder=3, s=25,
            )

        ax_box.set_xticks(range(len(SCHEDULES)))
        ax_box.set_xticklabels(SCHEDULES, fontsize=9)
        ax_box.set_ylabel("RMSE")
        ax_box.set_title(
            f"RMSE across {args.n_runs} runs  "
            f"(same path, different diffusion noise)",
            fontsize=10,
        )
        ax_box.grid(axis="y", alpha=0.3)

        # Annotate each box with mean ± std
        for i_s, sched in enumerate(SCHEDULES):
            m = rmse_grid[sched][si].mean()
            s = rmse_grid[sched][si].std()
            ax_box.text(
                i_s, ax_box.get_ylim()[0],
                f"μ={m:.3f}\nσ={s:.3f}",
                ha="center", va="bottom", fontsize=7, color="black",
            )

        # Heatmaps row
        for i_s, sched in enumerate(SCHEDULES):
            ax_h = fig.add_subplot(gs[1, i_s])
            plot_std_heatmap(ax_h, std_maps[sched], land_d, sched, vmax=shared_vmax)

        fig.suptitle(
            f"Model stochasticity — Sample {si+1}  (val_idx={si}, path seed={seed})  "
            f"[{args.n_runs} runs, r={args.resample}]",
            fontsize=13,
        )

        out_path = os.path.join(args.out_dir, f"sample_{si+1:02d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone. {args.n_samples} figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
