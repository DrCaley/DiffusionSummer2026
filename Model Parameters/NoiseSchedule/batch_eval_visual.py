"""
Run all 4 noise-schedule models on N validation samples, compute all 7 loss
metrics per run, and save a 2x2 image for each run.

Layout of each 2x2 image:
  [ Ground Truth     | Robot Path       ]
  [ Reconstruction   | Metrics table    ]

The metrics table (panel 4) displays the 7 evaluation scores:
    eps, curl_div, spectral, okubo_weiss, wasserstein,
    stream_function, strain_rate

Images are saved to:
    results/eval_visual/{schedule}/result_{run:02d}.png

Usage (from workspace root or NoiseSchedule/):
    python3 "Model Parameters/NoiseSchedule/batch_eval_visual.py"
    python3 "Model Parameters/NoiseSchedule/batch_eval_visual.py" \
        --n_runs 10 --path_steps 150 --resample 10
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
# Path setup
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

from dataset        import OceanCurrentDataset
from diffusion      import DDPM
from repaint_model  import Repaint
from repaint_infer  import biased_walk_path, repaint
from loss_functions import (
    curl_div_loss, spectral_loss, okubo_weiss_loss,
    wasserstein_loss, stream_function_loss, strain_rate_loss,
    LOSS_MODES,
)

try:
    from geomloss import SamplesLoss
    _sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)
    _HAS_GEOMLOSS = True
except ImportError:
    _HAS_GEOMLOSS = False
    print("  [warn] geomloss not installed — wasserstein metric will be NaN")

SCHEDULES = ["cosine", "linear", "quadratic", "sigmoid"]
# Seeds identical to batch_eval.py: seed = i*7 + 1
SEEDS = [i * 7 + 1 for i in range(20)]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--schedules",  nargs="+", default=SCHEDULES,
                   choices=SCHEDULES, help="Which schedules to evaluate.")
    p.add_argument("--n_runs",     type=int, default=10)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--resample",   type=int, default=10)
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--out_dir",    default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper  (matches batch_repaint.py exactly)
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
# Metrics table panel
# ---------------------------------------------------------------------------

def plot_metrics(ax, metrics: dict, schedule: str, sample_idx: int, seed: int):
    """Render a clean text table of all 7 metric values on ax."""
    ax.axis("off")

    metric_labels = {
        "eps":             "RMSE (field)",
        "curl_div":        "Curl/Div RMSE",
        "spectral":        "Spectral RMSE",
        "okubo_weiss":     "Okubo-Weiss RMSE",
        "wasserstein":     "Wasserstein dist.",
        "stream_function": "Stream fn. RMSE",
        "strain_rate":     "Strain rate RMSE",
    }

    rows = []
    for key in LOSS_MODES:
        val = metrics.get(key, float("nan"))
        label = metric_labels.get(key, key)
        val_str = f"{val:.6f}" if not np.isnan(val) else "N/A"
        rows.append([label, val_str])

    col_labels = ["Metric", "Value"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style header row
    for col in range(2):
        table[(0, col)].set_facecolor("#2c5f8a")
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for row in range(1, len(rows) + 1):
        bg = "#e8f0f7" if row % 2 == 0 else "white"
        for col in range(2):
            table[(row, col)].set_facecolor(bg)
        # Bold the value column
        table[(row, 1)].set_text_props(fontfamily="monospace")

    ax.set_title("Evaluation Metrics", fontsize=11, pad=8)


# ---------------------------------------------------------------------------
# Save one 2x2 plot
# ---------------------------------------------------------------------------

def save_plot(
    u_true_d, v_true_d,
    u_pred_d, v_pred_d,
    land_d, path_d,
    metrics,
    path_cells, sample_idx, seed, run_num, schedule, out_path,
):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # Panel 1: Ground truth
    plot_field(axes[0], u_true_d, v_true_d, land_d, "Ground Truth")

    # Panel 2: Robot path
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
    axes[1].set_title(f"Robot Path ({path_cells} cells, seed={seed})", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728",                 label="Path")
    land_p  = mpatches.Patch(facecolor="black",                   label="Land")
    axes[1].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

    # Panel 3: Reconstruction
    plot_field(axes[2], u_pred_d, v_pred_d, land_d,
               f"Reconstructed (RePaint — {schedule})", cmap="cool")

    # Panel 4: Metrics table
    plot_metrics(axes[3], metrics, schedule, sample_idx, seed)

    plt.suptitle(
        f"Run {run_num:02d}  —  Val sample {sample_idx}, path seed {seed}"
        f"  [schedule={schedule}]",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Resolve paths
    pickle_path = args.pickle
    if not os.path.isabs(pickle_path) and not os.path.exists(pickle_path):
        pickle_path = os.path.join(_ROOT, pickle_path)

    base_out = args.out_dir or os.path.join(_HERE, "results", "eval_visual")

    val_ds       = OceanCurrentDataset(pickle_path, split=1)
    land_mask_np = val_ds.land_mask.numpy()
    land_mask_t  = val_ds.land_mask.to(device)
    ocean        = (~land_mask_t).float()[None, None]   # (1,1,H,W)

    for schedule in args.schedules:
        ckpt_path = os.path.join(
            _HERE, "checkpoints",
            f"checkpoints_repaint_{schedule}",
            f"best_model_{schedule}.pt",
        )
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {schedule}: checkpoint not found at {ckpt_path}")
            continue

        out_dir = os.path.join(base_out, schedule)
        os.makedirs(out_dir, exist_ok=True)

        print(f"{'='*60}")
        print(f"Schedule: {schedule}")

        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        base_ch   = ckpt_args.get("base_ch",  args.base_ch)
        time_dim  = ckpt_args.get("time_dim", args.time_dim)
        T         = ckpt_args.get("T",        args.T)
        sched     = ckpt_args.get("schedule", schedule)

        model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"  Loaded epoch {ckpt.get('epoch','?')}, T={T}, schedule={sched}")

        diffusion = DDPM(T=T, beta_schedule=sched, device=device)

        for i in range(args.n_runs):
            sample_idx = i
            seed       = SEEDS[i]
            print(f"  Run {i+1:2d}/{args.n_runs}  (val sample={sample_idx}, seed={seed})", end="  ", flush=True)

            x0_true = val_ds[sample_idx]
            path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

            x0_observed = x0_true.clone()
            path_t = torch.from_numpy(path_mask)
            x0_observed[:, ~path_t] = 0.0

            x0_pred = repaint(
                model, diffusion, x0_observed,
                path_mask, land_mask_np,
                r=args.resample, device=device,
            )  # (2, H, W) CPU

            pred_dev = x0_pred.unsqueeze(0).to(device)
            true_dev = x0_true.unsqueeze(0).to(device)

            # Compute all 7 metrics
            metrics = {}
            metrics["eps"] = F.mse_loss(pred_dev * ocean, true_dev * ocean).sqrt().item()
            metrics["curl_div"]        = curl_div_loss(pred_dev, true_dev, ocean).item()
            metrics["spectral"]        = spectral_loss(pred_dev, true_dev, ocean).item()
            metrics["okubo_weiss"]     = okubo_weiss_loss(pred_dev, true_dev, ocean).item()
            metrics["stream_function"] = stream_function_loss(pred_dev, true_dev, ocean).item()
            metrics["strain_rate"]     = strain_rate_loss(pred_dev, true_dev, ocean).item()
            if _HAS_GEOMLOSS:
                metrics["wasserstein"] = wasserstein_loss(pred_dev, true_dev, ocean, _sinkhorn).item()
            else:
                metrics["wasserstein"] = float("nan")

            metric_str = "  ".join(
                f"{k}={v:.5f}" if not np.isnan(v) else f"{k}=NaN"
                for k, v in metrics.items()
            )
            print(metric_str)

            # Transpose for display: (H,W) -> (W,H) matches batch_repaint.py
            pred_np = x0_pred.numpy()
            true_np = x0_true.numpy()
            land_d  = land_mask_np.T
            path_d  = path_mask.T

            out_path = os.path.join(out_dir, f"result_{i+1:02d}.png")
            save_plot(
                true_np[0].T, true_np[1].T,
                pred_np[0].T, pred_np[1].T,
                land_d, path_d,
                metrics,
                path_cells=int(path_mask.sum()),
                sample_idx=sample_idx,
                seed=seed,
                run_num=i + 1,
                schedule=schedule,
                out_path=out_path,
            )
            print(f"    -> {out_path}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
