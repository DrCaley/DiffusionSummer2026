"""
infer_batch_repaint.py
======================
Runs RePaint (r=10 and r=1) on N seeds and saves per-seed results as
  - <out_dir>/seed<S>_r<R>.pt      (format identical to infer_single.py)
  - <out_dir>/seed<S>_r<R>.png     (visualize_infer.py-style 2x2 figure)

Path model: v_shape_path() — two straight legs meeting at a common vertex.

Usage:
    python infer_batch_repaint.py \\
        --pickle   /root/ocean_ddpm/data.pickle \\
        --checkpoint /root/Repaint_vs_DPS/checkpoints_linear/best_model_linear.pt \\
        --out_dir  /root/Repaint_vs_DPS/results/infer_batch_r10_r1 \\
        --n_seeds  10 --segment_len 10 --T 1000 --stride 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint


# ── Robot path: V-shape ───────────────────────────────────────────────────────

def v_shape_path(land_mask, segment_len=10, seed=None):
    """
    V-shape path: two straight legs of `segment_len` steps each, starting from
    a common vertex and heading in directions ~135° apart (e.g. NE + NW = upward V).
    The vertex cell is shared; total unique cells ≤ 2*segment_len + 1.
    """
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape

    ALL_DIRS = [
        (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
        ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
    ]

    ocean_cells = list(zip(*np.where(~land_mask)))
    if not ocean_cells:
        raise ValueError("No ocean cells found.")

    for _ in range(10_000):
        vertex  = ocean_cells[rng.integers(len(ocean_cells))]
        dir1    = int(rng.integers(8))
        # V opening angle: ±3 steps = 135°, ±4 = 180° (straight line, skip)
        offset  = int(rng.choice([3, -3]))
        dir2    = (dir1 + offset) % 8

        pm = np.zeros((H, W), dtype=bool)
        pm[vertex[0], vertex[1]] = True

        def walk_leg(start_r, start_c, direction, n):
            r, c = start_r, start_c
            for _ in range(n):
                dr, dc = ALL_DIRS[direction]
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not land_mask[nr, nc]:
                    r, c = nr, nc
                    pm[r, c] = True
                else:
                    return False
            return True

        leg1_ok = walk_leg(vertex[0], vertex[1], dir1, segment_len)
        leg2_ok = walk_leg(vertex[0], vertex[1], dir2, segment_len)

        if leg1_ok and leg2_ok:
            return pm

    return pm   # fallback


# ── RePaint inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def repaint_infer(model, diffusion, x0_known, path_mask, land_mask,
                  r=10, device="cpu", stride=1):
    H, W     = x0_known.shape[1:]
    x0_known = x0_known.unsqueeze(0).to(device)
    known_t  = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t   = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t  = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0
        for j in range(r):
            xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)
            t_prev_t   = torch.full((1,), t_prev_int, device=device,
                                    dtype=torch.long)
            xt_known_noisy, _ = diffusion.q_sample(x0_known, t_prev_t)
            xt_merged  = known_t * xt_known_noisy + (1.0 - known_t) * xt_unknown
            xt_merged  = xt_merged * ocean_t
            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int,
                                                   t_prev_int) * ocean_t
            else:
                xt = xt_merged

    return xt.squeeze(0).cpu().numpy()   # (2, H, W)


# ── Visualisation (visualize_infer.py style) ──────────────────────────────────

def _plot_field(ax, u, v, land_mask, title, step=2, vmax=None):
    H, W  = u.shape
    speed = np.ma.masked_where(land_mask, np.sqrt(u ** 2 + v ** 2))
    if vmax is None:
        vmax = float(np.nanpercentile(speed.compressed(), 98)) if speed.count() else 1

    im = ax.imshow(speed, origin="lower", cmap="cool", vmin=0, vmax=vmax,
                   extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0)
    ax.imshow(land_mask, origin="lower",
              cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
              extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=1)

    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    mask   = ~land_mask[::step, ::step]
    ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask],
              color="black", scale=12, width=0.003, zorder=2)

    plt.colorbar(im, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def save_png(result, title, out_path):
    # All arrays stored as (H, W); transpose for display to get X=col, Y=row
    u_true    = result["u_true"].T
    v_true    = result["v_true"].T
    u_pred    = result["u_pred"].T
    v_pred    = result["v_pred"].T
    land_mask = result["land_mask"].T
    path_mask = result["path_mask"].T
    err       = result["err"].T

    rmse = float(np.sqrt(np.nanmean(err[~land_mask] ** 2)))

    speed_true = np.ma.masked_where(land_mask, np.sqrt(u_true ** 2 + v_true ** 2))
    speed_pred = np.ma.masked_where(land_mask, np.sqrt(u_pred ** 2 + v_pred ** 2))
    vmax = float(np.nanpercentile(
        np.ma.concatenate([speed_true, speed_pred]).compressed(), 98
    )) if (speed_true.count() + speed_pred.count()) else 1

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    _plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth", vmax=vmax)

    H, W = land_mask.shape
    ext  = [-0.5, W - 0.5, -0.5, H - 0.5]
    axes[1].imshow(land_mask, origin="lower",
                   cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
                   extent=ext, aspect="auto", zorder=0)
    path_disp = np.zeros_like(land_mask, dtype=float)
    path_disp[path_mask] = 1.0
    axes[1].imshow(path_disp, origin="lower", cmap="Reds", alpha=0.8,
                   extent=ext, aspect="auto", zorder=1, vmin=0, vmax=1)
    axes[1].set_title(
        f"Robot Path ({result['path_cells']} cells, seed={result['seed']})",
        fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    axes[1].legend(handles=[
        mpatches.Patch(facecolor="white",   edgecolor="gray", label="Ocean"),
        mpatches.Patch(facecolor="#d62728",                   label="Path"),
        mpatches.Patch(facecolor="black",                     label="Land"),
    ], loc="upper right", fontsize=8)

    _plot_field(axes[2], u_pred, v_pred, land_mask,
                f"RePaint r={result['r']}  (inference)", vmax=vmax)

    err_plot = np.ma.masked_where(land_mask, err)
    im = axes[3].imshow(err_plot, origin="lower", cmap="hot_r",
                        extent=ext, aspect="auto")
    axes[3].imshow(land_mask, origin="lower",
                   cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
                   extent=ext, aspect="auto", zorder=1)
    plt.colorbar(im, ax=axes[3], label="|error| speed", shrink=0.7)
    axes[3].set_title(f"Error  (RMSE={rmse:.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PNG: {out_path}  (RMSE={rmse:.4f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--out_dir",     default="infer_batch_vshape")
    p.add_argument("--n_seeds",     type=int, default=10)
    p.add_argument("--segment_len", type=int, default=10)
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--stride",      type=int, default=10)
    p.add_argument("--r_values",    type=int, nargs="+", default=[10, 1])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── data ──
    test_ds    = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = test_ds.land_mask.numpy()
    ocean_mask = ~land_mask
    n_test     = len(test_ds)

    seeds = list(range(0, args.n_seeds * 7, 7))[:args.n_seeds]   # 0,7,14,...
    print(f"Seeds: {seeds}")

    # ── model ──
    ckpt      = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model     = Repaint(in_ch=2, base_ch=64, time_dim=256).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    schedule  = ckpt.get("schedule", "linear")
    noise_std = ckpt.get("noise_std", 1.0)
    diffusion = DDPM(T=args.T, beta_schedule=schedule,
                     noise_std=noise_std, device=device)
    print(f"Loaded: epoch={ckpt.get('epoch','?')}  "
          f"schedule={schedule}  noise_std={noise_std:.5f}")

    # ── run ──
    for seed in seeds:
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]          # (2, H, W)
        true_np    = x0_true.numpy()
        path_mask  = v_shape_path(land_mask, segment_len=args.segment_len,
                                     seed=seed)
        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"\nSeed={seed}  sample={sample_idx}  "
              f"path_cells={int(path_mask.sum())}")

        for r in args.r_values:
            print(f"  r={r} ...", end="", flush=True)
            pred = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask,
                                 r=r, device=device, stride=args.stride)

            err = np.sqrt((pred[0] - true_np[0]) ** 2
                          + (pred[1] - true_np[1]) ** 2)
            err[land_mask] = float("nan")
            rmse = float(np.sqrt(np.nanmean(err[ocean_mask] ** 2)))
            print(f"  RMSE={rmse:.4f}")

            result = {
                "u_true":    true_np[0],
                "v_true":    true_np[1],
                "u_pred":    pred[0],
                "v_pred":    pred[1],
                "land_mask": land_mask,
                "path_mask": path_mask,
                "err":       err,
                "path_cells": int(path_mask.sum()),
                "sample_idx": sample_idx,
                "seed":       seed,
                "segment_len": args.segment_len,
                "r":           r,
                "rmse":        rmse,
            }

            stem = f"seed{seed:03d}_r{r}"
            torch.save(result, os.path.join(args.out_dir, stem + ".pt"))

            ckpt_name = os.path.basename(args.checkpoint)
            title = (f"RePaint r={r}  —  "
                     f"Test sample {sample_idx}, seed={seed}  |  {ckpt_name}")
            save_png(result, title,
                     os.path.join(args.out_dir, stem + ".png"))

    print(f"\nDone. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
