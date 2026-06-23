"""
Run the cosine RePaint model N times on a single sample with the SAME robot
path (fixed path seed) but different diffusion seeds.  Produce a 3×4 grid:

  [ ground truth | robot path  | run 1       | run 2 ]
  [ run 3        | run 4       | run 5       | run 6 ]
  [ run 7        | run 8       | run 9       | average ]

Each run panel includes its RMSE (eps) in the title.

Usage (from workspace root or NoiseSchedule/):
    python3 "Model Parameters/NoiseSchedule/multi_run_grid.py"
    python3 "Model Parameters/NoiseSchedule/multi_run_grid.py" \
        --sample 0 --n_runs 9 --path_seed 1 --path_steps 150 --resample 10 \
        --out results/cosine_multi_run_grid.png
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup (works from workspace root or from NoiseSchedule/)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.append(_ROOT)

_LOSS_DIR_LOCAL  = os.path.join(os.path.dirname(_HERE), "Loss Function")
_LOSS_DIR_SERVER = os.path.dirname(_HERE)
for _d in (_LOSS_DIR_LOCAL, _LOSS_DIR_SERVER):
    if _d not in sys.path:
        sys.path.append(_d)

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path, repaint


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--schedule",   default="cosine",
                   choices=["cosine", "linear", "quadratic", "sigmoid"])
    p.add_argument("--sample",     type=int, default=0,
                   help="Validation-set index to use.")
    p.add_argument("--n_runs",     type=int, default=9,
                   help="Number of independent inference runs (default 9).")
    p.add_argument("--path_seed",  type=int, default=1,
                   help="RNG seed for the robot path (fixed across all runs).")
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--resample",   type=int, default=10,
                   help="RePaint r parameter.")
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--out",        default=None,
                   help="Output image path. Defaults to results/cosine_multi_run_grid.png")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver plot helper
# ---------------------------------------------------------------------------

def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool", eps=None):
    """Draw a quiver plot on transposed (W, H) arrays — matches batch_infer convention."""
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
    if mask.any():
        q = ax.quiver(
            xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
            cmap=cmap,
            clim=(0, np.nanpercentile(mq[mask], 98) or 1),
            scale=12, width=0.003, zorder=2,
        )
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.65)

    full_title = title if eps is None else f"{title}\nRMSE={eps:.5f}"
    ax.set_title(full_title, fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Resolve paths ----
    pickle_path = args.pickle
    if not os.path.isabs(pickle_path) and not os.path.exists(pickle_path):
        pickle_path = os.path.join(_ROOT, pickle_path)

    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        out_dir, f"{args.schedule}_multi_run_grid.png"
    )

    # ---- Data ----
    val_ds       = OceanCurrentDataset(pickle_path, split=1)
    land_mask_np = val_ds.land_mask.numpy()         # (H, W) bool
    land_mask_t  = val_ds.land_mask.to(device)

    x0_true = val_ds[args.sample]                   # (2, H, W) CPU tensor

    # ---- Fixed robot path ----
    path_mask = biased_walk_path(
        land_mask_np, n_steps=args.path_steps, seed=args.path_seed
    )

    # Observed field: real values on path, zeros elsewhere
    x0_observed = x0_true.clone()
    path_t = torch.from_numpy(path_mask)
    x0_observed[:, ~path_t] = 0.0

    # ---- Load model ----
    ckpt_path = os.path.join(
        _HERE, "checkpoints",
        f"checkpoints_repaint_{args.schedule}",
        f"best_model_{args.schedule}.pt",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)
    sched     = ckpt_args.get("schedule", args.schedule)

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch','?')}, T={T}, schedule={sched}")

    diffusion = DDPM(T=T, beta_schedule=sched, device=device)

    # ---- Ocean mask for RMSE ----
    ocean = (~land_mask_t).float()[None, None].to(device)   # (1, 1, H, W)
    true_dev = x0_true.unsqueeze(0).to(device)              # (1, 2, H, W)

    # ---- Run inference N times ----
    runs = []
    eps_values = []
    for i in range(args.n_runs):
        # Each run uses a different torch seed so diffusion noise differs,
        # but the robot path (path_mask) is identical every time.
        torch.manual_seed(i * 13 + 7)
        if device == "cuda":
            torch.cuda.manual_seed_all(i * 13 + 7)

        print(f"  Run {i+1}/{args.n_runs} ...", end=" ", flush=True)
        x0_pred = repaint(
            model, diffusion, x0_observed,
            path_mask, land_mask_np,
            r=args.resample, device=device,
        )   # (2, H, W) CPU

        pred_dev = x0_pred.unsqueeze(0).to(device)
        eps = F.mse_loss(pred_dev * ocean, true_dev * ocean).sqrt().item()
        eps_values.append(eps)
        runs.append(x0_pred.numpy())
        print(f"RMSE={eps:.5f}")

    # ---- Average of all runs ----
    avg_run = np.mean(runs, axis=0)   # (2, H, W)
    avg_dev = torch.from_numpy(avg_run).unsqueeze(0).to(device)
    avg_eps = F.mse_loss(avg_dev * ocean, true_dev * ocean).sqrt().item()
    print(f"  Average RMSE={avg_eps:.5f}")

    # ---- Build 4×3 grid (portrait) ----
    #  Row 0: ground truth | robot path  | run 1
    #  Row 1: run 2        | run 3       | run 4
    #  Row 2: run 5        | run 6       | run 7
    #  Row 3: run 8        | run 9       | average
    n_cols  = 3
    n_rows  = 4
    n_slots = n_rows * n_cols   # 12

    # Transpose everything to (W, H) for display — matches batch_infer.py convention
    x0_np     = x0_true.numpy()
    x0_obs_np = x0_observed.numpy()
    land_d    = land_mask_np.T
    path_d    = path_mask.T

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 6, n_rows * 4),
        constrained_layout=True,
    )
    axes = axes.flatten()

    def _draw(slot, u, v, title, eps=None):
        plot_field(axes[slot], u.T, v.T, land_d, title, eps=eps)

    # Slot 0: ground truth
    _draw(0, x0_np[0], x0_np[1], "Ground Truth")

    # Slot 1: robot path (observed — hide unobserved ocean cells)
    obs_u = x0_obs_np[0].copy()
    obs_v = x0_obs_np[1].copy()
    obs_u[~path_mask & ~land_mask_np] = np.nan
    obs_v[~path_mask & ~land_mask_np] = np.nan
    # Draw as quiver, then overlay the path in red like test_repaint.py
    plot_field(axes[1], obs_u.T, obs_v.T, land_d, "Robot Path (observed)")
    Wd, Hd = land_d.shape[1], land_d.shape[0]
    path_display = np.zeros_like(land_d, dtype=float)
    path_display[path_d] = 1.0
    axes[1].imshow(
        path_display, origin="lower", cmap="Reds", alpha=0.7,
        extent=[-0.5, Wd - 0.5, -0.5, Hd - 0.5],
        aspect="auto", zorder=3, vmin=0, vmax=1,
    )

    # Slots 2–10: individual runs (no path overlay)
    for i, (run, eps) in enumerate(zip(runs, eps_values)):
        _draw(2 + i, run[0], run[1], f"Run {i+1}", eps=eps)

    # Slot 11: average (no path overlay)
    _draw(n_slots - 1, avg_run[0], avg_run[1],
          f"Average ({args.n_runs} runs)", eps=avg_eps)

    # ---- Global title ----
    fig.suptitle(
        f"Cosine schedule — {args.n_runs} independent runs, val sample {args.sample}\n"
        f"path_steps={args.path_steps}  resample={args.resample}  path_seed={args.path_seed}",
        fontsize=13,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
