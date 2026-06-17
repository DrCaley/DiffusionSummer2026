"""
Evaluate all 7 loss-function models on N randomly chosen validation samples,
running each model 10 times per sample to capture stochastic variance.

For each sample a 4×4 figure is saved:
  Row 0:  [ Ground Truth | Robot Path   | eps avg-quiver    | eps std-heatmap    ]
  Row 1:  [ curl_div avg | curl_div hm  | spectral avg      | spectral heatmap   ]
  Row 2:  [ okubo avg    | okubo hm     | wasserstein avg   | wasserstein hm     ]
  Row 3:  [ stream avg   | stream hm    | strain_rate avg   | strain_rate hm     ]

  "avg" panels  → mean prediction quiver over 10 runs
  "hm" panels   → pixel-wise std of speed across 10 runs (where runs differ most)

No existing file is ever overwritten.  A fresh versioned sub-directory is
created each invocation: results/eval_multi_run/run_v1/, run_v2/, …

A single text file summary.txt is written alongside the images containing:
  1. Summary table: mean (±std over samples×runs) per (model, metric)
  2. Full detail table: every (model, sample, run, seed, 7 metrics)

Usage (run from workspace root or this directory):
    python3 "Model Parameters/Loss Function/batch_eval_multi_run.py"
    python3 "Model Parameters/Loss Function/batch_eval_multi_run.py" \\
        --n_samples 50 --n_runs 10 --path_steps 150 --resample 10 --seed 42

Model checkpoints
-----------------
The 7 models are loaded from --ckpt_dir (default: <workspace>/Model Parameters/
Loss Function/models/).  Two models have newer versions available:
  --okubo_weiss_ckpt  path to the updated okubo-weiss .pt file
  --spectral_ckpt     path to the updated spectral .pt file
These default to the same pattern inside --ckpt_dir if not supplied.
"""

import argparse
import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_MODEL_PARAMS = os.path.dirname(_HERE)
_ROOT         = os.path.dirname(_MODEL_PARAMS)

sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# NoiseSchedule/ contains repaint_infer.py; model.py/diffusion.py may be at
# DDPM/ (server layout) or DDPM/model/ (local layout)
_NS_DIR    = os.path.join(_MODEL_PARAMS, "NoiseSchedule")
_DDPM_DIR  = os.path.join(_ROOT, "DDPM")
_DDPM_DIR2 = os.path.join(_ROOT, "DDPM", "model")
for _d in (_NS_DIR, _DDPM_DIR, _DDPM_DIR2):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

from dataset import OceanCurrentDataset

try:
    from model    import UNet
    from diffusion import DDPM
except ImportError:
    from DDPM.model.model     import UNet
    from DDPM.model.diffusion import DDPM

try:
    from repaint_infer import biased_walk_path, repaint
except ImportError:
    from NoiseSchedule.repaint_infer import biased_walk_path, repaint

from loss_functions import (
    curl_div_loss, spectral_loss, okubo_weiss_loss,
    wasserstein_loss, stream_function_loss, strain_rate_loss,
    LOSS_MODES,
)

try:
    from geomloss import SamplesLoss
    _sinkhorn     = SamplesLoss("sinkhorn", p=2, blur=0.05)
    _HAS_GEOMLOSS = True
except ImportError:
    _HAS_GEOMLOSS = False
    print("  [warn] geomloss not installed — wasserstein metric will be NaN")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_LABELS = [
    "eps",
    "curl_div",
    "spectral",
    "okubo_weiss",
    "wasserstein",
    "stream_function",
    "strain_rate",
]

METRIC_SHORT = {
    "eps":             "rmse",
    "curl_div":        "cdiv",
    "spectral":        "spec",
    "okubo_weiss":     "ow",
    "wasserstein":     "wass",
    "stream_function": "strm",
    "strain_rate":     "strn",
}

# 4×4 layout: panels 0-1 are GT/Path, then pairs (avg, heatmap) per model
# Order must match MODEL_LABELS
_PANEL_ORDER = MODEL_LABELS  # 7 models → panels 2,3 … 14,15

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _next_versioned_dir(base: str, prefix: str) -> str:
    """Return <base>/<prefix>_v1/, _v2/, … choosing the first that doesn't exist."""
    v = 1
    while True:
        candidate = os.path.join(base, f"{prefix}_v{v}")
        if not os.path.exists(candidate):
            return candidate
        v += 1


def _next_versioned_file(path: str) -> str:
    """If *path* already exists, insert _v2, _v3 … before the extension."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    v = 2
    while True:
        candidate = f"{root}_v{v}{ext}"
        if not os.path.exists(candidate):
            return candidate
        v += 1


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_quiver(ax, u, v, land_mask, title, step=2, cmap="cool", clim=None):
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
        vmax = clim if clim is not None else (np.nanpercentile(mq[mask], 98) or 1.0)
        q = ax.quiver(
            xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
            cmap=cmap, clim=(0, vmax), scale=12, width=0.003, zorder=2,
        )
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.65)
    ax.set_title(title, fontsize=6, pad=3)
    ax.set_xlabel("X", fontsize=6)
    ax.set_ylabel("Y", fontsize=6)
    ax.tick_params(labelsize=5)


def _plot_heatmap(ax, std_speed, land_mask, title):
    """Overlay pixel-wise std-of-speed on the land mask."""
    display = std_speed.copy()
    display[land_mask] = np.nan
    H, W = land_mask.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    im = ax.imshow(
        display, origin="lower", cmap="hot_r",
        extent=[-0.5, W - 0.5, -0.5, H - 0.5],
        aspect="auto", zorder=1,
        vmin=0, vmax=np.nanpercentile(display, 98) if not np.all(np.isnan(display)) else 1.0,
    )
    plt.colorbar(im, ax=ax, label="Speed std", shrink=0.65)
    ax.set_title(title, fontsize=6, pad=3)
    ax.set_xlabel("X", fontsize=6)
    ax.set_ylabel("Y", fontsize=6)
    ax.tick_params(labelsize=5)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(pred_dev, true_dev, ocean):
    m = {}
    m["eps"]             = F.mse_loss(pred_dev * ocean, true_dev * ocean).sqrt().item()
    m["curl_div"]        = curl_div_loss(pred_dev, true_dev, ocean).item()
    m["spectral"]        = spectral_loss(pred_dev, true_dev, ocean).item()
    m["okubo_weiss"]     = okubo_weiss_loss(pred_dev, true_dev, ocean).item()
    m["stream_function"] = stream_function_loss(pred_dev, true_dev, ocean).item()
    m["strain_rate"]     = strain_rate_loss(pred_dev, true_dev, ocean).item()
    if _HAS_GEOMLOSS:
        m["wasserstein"] = wasserstein_loss(pred_dev, true_dev, ocean, _sinkhorn).item()
    else:
        m["wasserstein"] = float("nan")
    return m


# ---------------------------------------------------------------------------
# Per-sample figure (4×4)
# ---------------------------------------------------------------------------

def _save_sample_figure(
    sample_idx,
    path_seed,
    x0_true_np,      # (2, H, W)
    path_mask,       # (H, W) bool
    land_mask_np,    # (H, W) bool
    model_avg_preds, # label -> (2, H, W) mean prediction
    model_std_speed, # label -> (H, W) pixel-wise std of speed
    model_avg_metrics, # label -> dict metric -> float (mean over runs)
    active_labels,
    out_path,        # already guaranteed not to exist
):
    land_d = land_mask_np.T   # (W, H) for imshow origin='lower'
    path_d = path_mask.T

    speed_gt = np.sqrt(x0_true_np[0] ** 2 + x0_true_np[1] ** 2)
    speed_gt[land_mask_np] = np.nan
    clim = float(np.nanpercentile(speed_gt, 98)) if not np.all(np.isnan(speed_gt)) else 1.0

    fig, axes = plt.subplots(4, 4, figsize=(4 * 6, 4 * 5), constrained_layout=True)
    ax = axes.flatten()   # 16 panels

    # ── Panel 0: Ground Truth ──────────────────────────────────────────────
    _plot_quiver(
        ax[0], x0_true_np[0].T, x0_true_np[1].T, land_d,
        f"Ground Truth  (val sample {sample_idx})",
        cmap="cool", clim=clim,
    )

    # ── Panel 1: Robot Path ────────────────────────────────────────────────
    ax[1].imshow(
        land_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    path_display = np.full(land_d.shape, np.nan)
    path_display[path_d] = 1.0
    ax[1].imshow(
        path_display, origin="lower", cmap="Reds",
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    ax[1].set_title(
        f"Robot Path  ({int(path_mask.sum())} cells  seed={path_seed})",
        fontsize=6, pad=3,
    )
    ax[1].set_xlabel("X", fontsize=6)
    ax[1].set_ylabel("Y", fontsize=6)
    ax[1].tick_params(labelsize=5)
    ax[1].legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
            mpatches.Patch(facecolor="#d62728", label="Path"),
            mpatches.Patch(facecolor="black", label="Land"),
        ],
        loc="upper right", fontsize=5,
    )

    # ── Panels 2–15: pairs (avg quiver, std heatmap) per model ────────────
    for mi, label in enumerate(active_labels):
        avg_idx = 2 + mi * 2       # 2, 4, 6, 8, 10, 12, 14
        hm_idx  = avg_idx + 1      # 3, 5, 7, 9, 11, 13, 15

        if avg_idx >= 16 or hm_idx >= 16:
            break

        avg_pred = model_avg_preds[label]    # (2, H, W)
        std_spd  = model_std_speed[label]    # (H, W)
        m        = model_avg_metrics[label]

        # compact two-line metric summary
        row1 = "  ".join(
            f"{METRIC_SHORT[k]}={m[k]:.4f}" if not np.isnan(m[k]) else f"{METRIC_SHORT[k]}=NaN"
            for k in ["eps", "curl_div", "spectral", "okubo_weiss"]
        )
        row2 = "  ".join(
            f"{METRIC_SHORT[k]}={m[k]:.4f}" if not np.isnan(m[k]) else f"{METRIC_SHORT[k]}=NaN"
            for k in ["wasserstein", "stream_function", "strain_rate"]
        )
        avg_title = f"{label}  (avg of {len(active_labels)} runs)\n{row1}\n{row2}"

        _plot_quiver(
            ax[avg_idx], avg_pred[0].T, avg_pred[1].T, land_d,
            avg_title, cmap="cool", clim=clim,
        )
        _plot_heatmap(
            ax[hm_idx], std_spd.T, land_d,
            f"{label}  run variance (speed std)",
        )

    # hide any remaining blank panels
    for i in range(2 + len(active_labels) * 2, 16):
        ax[i].axis("off")

    fig.suptitle(
        f"Val sample {sample_idx}  —  7 loss-function models  "
        f"(avg + variance over 10 runs, same robot path)",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-run evaluation of 7 loss-function models on val data."
    )
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument(
        "--ckpt_dir", default=None,
        help="Dir with model_ddpm_*_gaussian_cosine.pt files. "
             "Defaults to <workspace_root>/Model Parameters/Loss Function/models/",
    )
    p.add_argument(
        "--okubo_weiss_ckpt", default=None,
        help="Path to the updated okubo-weiss .pt file. "
             "Defaults to model_ddpm_okubo_weiss_gaussian_cosine.pt in --ckpt_dir.",
    )
    p.add_argument(
        "--spectral_ckpt", default=None,
        help="Path to the updated spectral .pt file. "
             "Defaults to model_ddpm_spectral_gaussian_cosine.pt in --ckpt_dir.",
    )
    p.add_argument("--n_samples",   type=int, default=50)
    p.add_argument("--n_runs",      type=int, default=10)
    p.add_argument("--path_steps",  type=int, default=150)
    p.add_argument("--resample",    type=int, default=10)
    p.add_argument("--seed",        type=int, default=42,
                   help="Master RNG seed (sample selection + path seeds).")
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--time_dim",    type=int, default=256)
    p.add_argument(
        "--out_dir", default=None,
        help="Parent directory for output. A versioned sub-directory is created "
             "automatically. Defaults to <this_dir>/results/eval_multi_run.",
    )
    p.add_argument(
        "--labels", nargs="+", default=MODEL_LABELS, choices=MODEL_LABELS,
        help="Subset of models to evaluate (default: all 7).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Resolve directories ────────────────────────────────────────────────
    if args.ckpt_dir is None:
        ckpt_dir = os.path.join(_HERE, "models")
    else:
        ckpt_dir = args.ckpt_dir

    parent_out = args.out_dir or os.path.join(_HERE, "results", "eval_multi_run")
    os.makedirs(parent_out, exist_ok=True)
    out_dir = _next_versioned_dir(parent_out, "run")
    os.makedirs(out_dir)
    print(f"Output directory : {out_dir}\n")

    # ── Resolve data.pickle ────────────────────────────────────────────────
    pickle_path = args.pickle
    if not os.path.isabs(pickle_path) and not os.path.exists(pickle_path):
        pickle_path = os.path.join(_ROOT, pickle_path)

    val_ds       = OceanCurrentDataset(pickle_path, split=1)
    land_mask_np = val_ds.land_mask.numpy()
    land_mask_t  = val_ds.land_mask.to(device)
    ocean        = (~land_mask_t).float()[None, None]   # (1,1,H,W)

    n_val = len(val_ds)
    rng   = np.random.default_rng(args.seed)
    sample_indices = sorted(
        rng.choice(n_val, size=min(args.n_samples, n_val), replace=False).tolist()
    )
    # Deterministic path seed per sample
    path_seeds = {idx: int(rng.integers(0, 99999)) for idx in sample_indices}
    # Diffusion seeds: n_runs per sample, deterministic
    run_seeds_for = {
        idx: [int(rng.integers(0, 999999)) for _ in range(args.n_runs)]
        for idx in sample_indices
    }

    print(f"Val set size : {n_val}")
    print(f"Seed         : {args.seed}")
    print(f"Samples      : {sample_indices}\n")

    # ── Build per-model checkpoint map ────────────────────────────────────
    ckpt_paths = {}
    for label in args.labels:
        ckpt_paths[label] = os.path.join(
            ckpt_dir, f"model_ddpm_{label}_gaussian_cosine.pt"
        )
    # Override okubo_weiss / spectral if explicit paths were given
    if args.okubo_weiss_ckpt:
        ckpt_paths["okubo_weiss"] = args.okubo_weiss_ckpt
    if args.spectral_ckpt:
        ckpt_paths["spectral"] = args.spectral_ckpt

    # ── Load all models ────────────────────────────────────────────────────
    models_loaded = {}
    diffusions    = {}
    active_labels = []
    for label in args.labels:
        cp = ckpt_paths.get(label, "")
        if not os.path.exists(cp):
            print(f"[SKIP] {label}: not found at {cp}")
            continue
        ckpt      = torch.load(cp, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        base_ch   = ckpt_args.get("base_ch",  args.base_ch)
        time_dim  = ckpt_args.get("time_dim", args.time_dim)
        T         = ckpt_args.get("T",        args.T)
        schedule  = ckpt_args.get("schedule", "cosine")
        net = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        models_loaded[label] = net
        diffusions[label]    = DDPM(T=T, beta_schedule=schedule, device=device)
        active_labels.append(label)
        print(
            f"  Loaded '{label}'  epoch={ckpt.get('epoch','?')}  "
            f"T={T}  sched={schedule}  ← {os.path.basename(cp)}"
        )
    print(f"\n{len(active_labels)} models loaded.\n")

    # ── Data accumulator for text report ──────────────────────────────────
    # all_rows: list of dicts with keys:
    #   model, sample, run, seed, eps, curl_div, spectral,
    #   okubo_weiss, wasserstein, stream_function, strain_rate
    all_rows = []

    # ── Main loop: samples ─────────────────────────────────────────────────
    total = len(sample_indices)
    for si, sample_idx in enumerate(sample_indices):
        path_seed  = path_seeds[sample_idx]
        run_seeds  = run_seeds_for[sample_idx]

        print(
            f"[{si+1:2d}/{total}] Val sample {sample_idx:4d}  "
            f"path_seed={path_seed}"
        )

        x0_true     = val_ds[sample_idx]                  # (2, H, W)
        path_mask   = biased_walk_path(
            land_mask_np, n_steps=args.path_steps, seed=path_seed
        )

        x0_observed = x0_true.clone()
        path_t      = torch.from_numpy(path_mask)
        x0_observed[:, ~path_t] = 0.0

        true_dev = x0_true.unsqueeze(0).to(device)        # (1,2,H,W)

        # per-model accumulators
        # model_run_preds[label] = list of (2,H,W) numpy arrays, one per run
        # model_run_metrics[label] = list of metric dicts, one per run
        model_run_preds   = {lbl: [] for lbl in active_labels}
        model_run_metrics = {lbl: [] for lbl in active_labels}

        for run_i, rseed in enumerate(run_seeds):
            torch.manual_seed(rseed)
            if device == "cuda":
                torch.cuda.manual_seed_all(rseed)

            for label in active_labels:
                # Each model gets a distinct seed offset within this run so
                # different models don't share identical noise draws
                model_seed = rseed + hash(label) % 100000
                torch.manual_seed(model_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(model_seed)

                x0_pred = repaint(
                    models_loaded[label], diffusions[label], x0_observed,
                    path_mask, land_mask_np,
                    r=args.resample, device=device,
                )   # (2,H,W) CPU

                pred_dev = x0_pred.unsqueeze(0).to(device)
                m        = _compute_metrics(pred_dev, true_dev, ocean)

                model_run_preds[label].append(x0_pred.numpy())
                model_run_metrics[label].append(m)

                all_rows.append({
                    "model":  label,
                    "sample": sample_idx,
                    "run":    run_i + 1,
                    "seed":   model_seed,
                    **m,
                })

            metric_preview = "  ".join(
                f"{METRIC_SHORT[k]}={model_run_metrics[active_labels[0]][-1][k]:.4f}"
                for k in ["eps", "spectral"]
            )
            print(
                f"    run {run_i+1:2d}/{args.n_runs}  "
                f"seed={rseed}  [{active_labels[0]}] {metric_preview}"
            )

        # ── Per-model averages & heatmaps ──────────────────────────────────
        model_avg_preds   = {}
        model_std_speed   = {}
        model_avg_metrics = {}

        for label in active_labels:
            preds_np = np.stack(model_run_preds[label], axis=0)  # (R,2,H,W)
            avg_pred = preds_np.mean(axis=0)                     # (2,H,W)

            # pixel-wise std of speed across runs
            speeds   = np.sqrt(preds_np[:, 0] ** 2 + preds_np[:, 1] ** 2)  # (R,H,W)
            std_spd  = speeds.std(axis=0)                                    # (H,W)
            std_spd[land_mask_np] = np.nan

            model_avg_preds[label]   = avg_pred
            model_std_speed[label]   = std_spd

            # average metrics over runs (ignoring NaN)
            avg_m = {}
            for k in LOSS_MODES:
                vals = [r[k] for r in model_run_metrics[label] if not np.isnan(r[k])]
                avg_m[k] = float(np.mean(vals)) if vals else float("nan")
            model_avg_metrics[label] = avg_m

            metric_str = "  ".join(
                f"{k}={avg_m[k]:.5f}" if not np.isnan(avg_m[k]) else f"{k}=NaN"
                for k in LOSS_MODES
            )
            print(f"    {label:<16}  avg→  {metric_str}")

        # ── Save 4×4 figure ───────────────────────────────────────────────
        img_path = _next_versioned_file(
            os.path.join(out_dir, f"sample_{sample_idx:04d}.png")
        )
        _save_sample_figure(
            sample_idx, path_seed,
            x0_true.numpy(),
            path_mask, land_mask_np,
            model_avg_preds,
            model_std_speed,
            model_avg_metrics,
            active_labels,
            img_path,
        )
        print(f"    → {img_path}\n")

    # ── Build text report ─────────────────────────────────────────────────
    # Summary: per (model, metric): mean and std over all (sample × run) values
    summary_stats = {}  # label -> metric -> (mean, std)
    for label in active_labels:
        rows_m = [r for r in all_rows if r["model"] == label]
        summary_stats[label] = {}
        for k in LOSS_MODES:
            vals = [r[k] for r in rows_m if not np.isnan(r[k])]
            if vals:
                summary_stats[label][k] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                summary_stats[label][k] = (float("nan"), float("nan"))

    # Column widths
    metrics_list = list(LOSS_MODES)
    col_w = 15
    model_w = 18

    sep_width = model_w + col_w * len(metrics_list) + 2 * len(metrics_list)
    sep  = "=" * sep_width
    dsep = "-" * sep_width

    lines = []
    title = "LOSS MODEL EVALUATION SUMMARY"
    params = (
        f"n_samples={args.n_samples}  n_runs_per_sample={args.n_runs}  "
        f"path_steps={args.path_steps}  resample={args.resample}"
    )
    lines.append(sep)
    lines.append(title.center(sep_width))
    lines.append(params.center(sep_width))
    lines.append(sep)

    header = f"{'model':<{model_w}}" + "".join(
        f"  {m:>{col_w}}" for m in metrics_list
    )
    lines.append(header)
    lines.append(dsep)

    for label in active_labels:
        mean_line = f"{label:<{model_w}}" + "".join(
            f"  {summary_stats[label][k][0]:>{col_w}.6f}"
            if not np.isnan(summary_stats[label][k][0])
            else f"  {'NaN':>{col_w}}"
            for k in metrics_list
        )
        std_str = f"{'':>{model_w}}" + "".join(
            f"  {('(±'+f'{summary_stats[label][k][1]:.5f}'+')'):>{col_w}}"
            if not np.isnan(summary_stats[label][k][1])
            else f"  {'(±NaN)':>{col_w}}"
            for k in metrics_list
        )
        lines.append(mean_line)
        lines.append(std_str)
        lines.append("")

    lines.append(sep)
    lines.append("")

    # ── Full detail table ─────────────────────────────────────────────────
    lines.append("FULL DETAIL  (model × sample × run)")
    lines.append(sep)

    detail_col_w = 15
    detail_header = (
        f"{'model':<{model_w}}  {'sample':>6}  {'run':>3}  {'seed':>8}"
        + "".join(f"  {m:>{detail_col_w}}" for m in metrics_list)
    )
    lines.append(detail_header)
    lines.append("-" * len(detail_header))

    for row in all_rows:
        vals_str = "".join(
            f"  {row[m]:>{detail_col_w}.6f}"
            if not np.isnan(row[m])
            else f"  {'NaN':>{detail_col_w}}"
            for m in metrics_list
        )
        lines.append(
            f"{row['model']:<{model_w}}  "
            f"{row['sample']:>6}  "
            f"{row['run']:>3}  "
            f"{row['seed']:>8}"
            + vals_str
        )

    report_path = _next_versioned_file(os.path.join(out_dir, "summary.txt"))
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nSummary report → {report_path}")

    # ── Pretty-print summary to stdout ────────────────────────────────────
    print()
    for l in lines[:lines.index(sep) + 4]:
        print(l)
    for label in active_labels:
        mean_line = f"{label:<{model_w}}" + "".join(
            f"  {summary_stats[label][k][0]:>{col_w}.6f}"
            if not np.isnan(summary_stats[label][k][0])
            else f"  {'NaN':>{col_w}}"
            for k in metrics_list
        )
        print(mean_line)
    print(sep)
    print(f"\nImages saved in : {out_dir}")
    print(f"Report saved to : {report_path}")


if __name__ == "__main__":
    main()
