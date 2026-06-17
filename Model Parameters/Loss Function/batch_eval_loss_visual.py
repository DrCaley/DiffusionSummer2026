"""
Evaluate all 7 loss-function models on N randomly chosen validation samples.

For each sample this script:
  1. Draws a random robot path (biased walk).
  2. Runs all 7 models once each (same path, same diffusion seed → fair comparison).
  3. Saves a 3×3 quiver image:

       [ Ground Truth  |  Robot Path     |  eps output         ]
       [ curl_div      |  spectral       |  okubo_weiss output  ]
       [ wasserstein   |  stream_fn      |  strain_rate output  ]

     Each reconstruction panel title shows all 7 evaluation metric values.

  4. Writes CSV results:
       results/loss_eval_visual_results.csv   — one row per (model, sample)
       results/loss_eval_visual_summary.csv   — mean ± std per model

Samples are drawn uniformly at random from the full validation set (1965
samples) using --seed for reproducibility.

Usage (run from workspace root or this directory):
    python3 "Model Parameters/Loss Function/batch_eval_loss_visual.py"
    python3 "Model Parameters/Loss Function/batch_eval_loss_visual.py" \\
        --n_samples 10 --path_steps 150 --resample 10 --seed 42
"""

import argparse
import csv
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
_HERE         = os.path.dirname(os.path.abspath(__file__))   # .../Loss Function/
_MODEL_PARAMS = os.path.dirname(_HERE)                        # .../Model Parameters/
_ROOT         = os.path.dirname(_MODEL_PARAMS)                # workspace root

sys.path.insert(0, _ROOT)   # dataset.py, diffusion.py, model.py, repaint_infer.py
sys.path.insert(0, _HERE)   # loss_functions.py

from dataset        import OceanCurrentDataset
from diffusion      import DDPM
from model          import UNet
from repaint_infer  import biased_walk_path, repaint
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

MODEL_LABELS = [
    "eps",
    "curl_div",
    "spectral",
    "okubo_weiss",
    "wasserstein",
    "stream_function",
    "strain_rate",
]

# Layout: GT + path in row 0, then 7 model outputs filling a 3×3 grid.
# Panel indices:   0=GT  1=path  2=eps  3=curl_div  4=spectral
#                  5=okubo_weiss  6=wasserstein  7=stream_fn  8=strain_rate
_PANEL_LABELS = ["eps", "curl_div", "spectral",
                 "okubo_weiss", "wasserstein", "stream_function", "strain_rate"]

METRIC_SHORT = {
    "eps":             "rmse",
    "curl_div":        "cdiv",
    "spectral":        "spec",
    "okubo_weiss":     "ow",
    "wasserstein":     "wass",
    "stream_function": "strm",
    "strain_rate":     "strn",
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--ckpt_dir",   default=None,
                   help="Dir with model_ddpm_*_gaussian_cosine.pt files. "
                        "Defaults to <workspace_root>/Model Parameters/loss_comparison/")
    p.add_argument("--n_samples",  type=int, default=10,
                   help="Number of random val samples (default 10).")
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--resample",   type=int, default=10)
    p.add_argument("--seed",       type=int, default=42,
                   help="RNG seed for sample selection and diffusion.")
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--out_dir",    default=None)
    p.add_argument("--labels",     nargs="+", default=MODEL_LABELS,
                   choices=MODEL_LABELS,
                   help="Subset of models to run (default: all 7).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper
# ---------------------------------------------------------------------------

def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool", clim=None):
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
            cmap=cmap, clim=(0, vmax),
            scale=12, width=0.003, zorder=2,
        )
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.65)
    ax.set_title(title, fontsize=7, pad=3)
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
    ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_dev, true_dev, ocean):
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
# Per-sample 3×3 image
# ---------------------------------------------------------------------------

def save_sample_image(
    sample_idx, path_seed,
    x0_true_np,     # (2, H, W)
    path_mask,      # (H, W) bool
    land_mask_np,   # (H, W) bool
    model_preds,    # dict label -> (2, H, W) numpy
    model_metrics,  # dict label -> dict metric -> float
    active_labels,
    out_path,
):
    land_d = land_mask_np.T
    path_d = path_mask.T

    # Compute a shared speed color limit from the ground truth for consistency
    speed_gt = np.sqrt(x0_true_np[0] ** 2 + x0_true_np[1] ** 2)
    speed_gt[land_mask_np] = np.nan
    clim = float(np.nanpercentile(speed_gt, 98)) if not np.all(np.isnan(speed_gt)) else 1.0

    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5),
                             constrained_layout=True)
    ax = axes.flatten()

    # ── Panel 0: Ground Truth ──────────────────────────────────────────────
    plot_field(ax[0], x0_true_np[0].T, x0_true_np[1].T, land_d,
               f"Ground Truth  (val sample {sample_idx})",
               cmap="cool", clim=clim)

    # ── Panel 1: Robot Path ────────────────────────────────────────────────
    ax[1].imshow(
        land_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    path_display = np.zeros_like(land_d, dtype=float)
    path_display[path_d] = 1.0
    ax[1].imshow(
        path_display, origin="lower", cmap="Reds", alpha=0.85,
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    ax[1].set_title(
        f"Robot Path\n{int(path_mask.sum())} cells  |  seed={path_seed}",
        fontsize=7, pad=3,
    )
    ax[1].set_xlabel("X", fontsize=7); ax[1].set_ylabel("Y", fontsize=7)
    ax[1].tick_params(labelsize=6)
    ax[1].legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
            mpatches.Patch(facecolor="#d62728", label="Path"),
            mpatches.Patch(facecolor="black",   label="Land"),
        ],
        loc="upper right", fontsize=6,
    )

    # ── Panels 2–8: one per model ─────────────────────────────────────────
    panel_order = [l for l in _PANEL_LABELS if l in active_labels]
    for pi, label in enumerate(panel_order):
        pidx = pi + 2
        if pidx >= len(ax):
            break
        pred_np  = model_preds[label]    # (2, H, W)
        m        = model_metrics[label]

        # Build a compact metric string (two rows)
        row1 = "  ".join(
            f"{METRIC_SHORT[k]}={m[k]:.4f}" if not np.isnan(m[k]) else f"{METRIC_SHORT[k]}=NaN"
            for k in ["eps", "curl_div", "spectral", "okubo_weiss"]
        )
        row2 = "  ".join(
            f"{METRIC_SHORT[k]}={m[k]:.4f}" if not np.isnan(m[k]) else f"{METRIC_SHORT[k]}=NaN"
            for k in ["wasserstein", "stream_function", "strain_rate"]
        )
        title = f"Model: {label}\n{row1}\n{row2}"

        plot_field(ax[pidx], pred_np[0].T, pred_np[1].T, land_d,
                   title, cmap="cool", clim=clim)

    # Hide any unused panels
    for pidx in range(2 + len(panel_order), len(ax)):
        ax[pidx].axis("off")

    fig.suptitle(
        f"Val sample {sample_idx}  —  7 loss-function models, same robot path",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Resolve paths
    if args.ckpt_dir is None:
        ckpt_dir = os.path.join(_MODEL_PARAMS, "loss_comparison")
    else:
        ckpt_dir = args.ckpt_dir

    if args.out_dir is None:
        out_dir = os.path.join(_HERE, "results", "eval_visual_loss")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

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
    print(f"Val set size : {n_val}")
    print(f"Random seed  : {args.seed}")
    print(f"Samples      : {sample_indices}\n")

    # Path seed per sample: deterministic from val index so reruns are consistent
    path_seeds = {idx: int(rng.integers(0, 99999)) for idx in sample_indices}

    # -------------------------------------------------------------------------
    # Load all 7 models
    # -------------------------------------------------------------------------
    models_loaded = {}
    diffusions    = {}
    active_labels = []
    for label in args.labels:
        ckpt_path = os.path.join(ckpt_dir, f"model_ddpm_{label}_gaussian_cosine.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {label}: not found at {ckpt_path}")
            continue
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        print(f"  Loaded '{label}'  epoch={ckpt.get('epoch','?')}  T={T}  sched={schedule}")
    print(f"\n{len(active_labels)} models loaded.\n")

    # -------------------------------------------------------------------------
    # CSV output setup
    # -------------------------------------------------------------------------
    detail_path  = os.path.join(_HERE, "results", "loss_eval_visual_results.csv")
    summary_path = os.path.join(_HERE, "results", "loss_eval_visual_summary.csv")
    os.makedirs(os.path.dirname(detail_path), exist_ok=True)

    detail_cols = ["model", "sample", "path_seed"] + list(LOSS_MODES)
    all_rows    = []

    # -------------------------------------------------------------------------
    # Main loop: samples
    # -------------------------------------------------------------------------
    for si, sample_idx in enumerate(sample_indices):
        path_seed = path_seeds[sample_idx]
        print(f"[{si+1:2d}/{len(sample_indices)}] Val sample {sample_idx}  path_seed={path_seed}")

        x0_true     = val_ds[sample_idx]                       # (2,H,W)
        path_mask   = biased_walk_path(land_mask_np, n_steps=args.path_steps,
                                       seed=path_seed)

        x0_observed = x0_true.clone()
        path_t      = torch.from_numpy(path_mask)
        x0_observed[:, ~path_t] = 0.0

        true_dev = x0_true.unsqueeze(0).to(device)             # (1,2,H,W)

        # Fix diffusion seed for this sample so all models use the same noise
        torch.manual_seed(args.seed + sample_idx)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed + sample_idx)

        model_preds   = {}
        model_metrics = {}

        for label in active_labels:
            # Re-fix seed per model so each has a deterministic but distinct draw
            torch.manual_seed(args.seed + sample_idx + hash(label) % 10000)
            if device == "cuda":
                torch.cuda.manual_seed_all(args.seed + sample_idx + hash(label) % 10000)

            x0_pred  = repaint(
                models_loaded[label], diffusions[label], x0_observed,
                path_mask, land_mask_np,
                r=args.resample, device=device,
            )   # (2,H,W) CPU

            pred_dev = x0_pred.unsqueeze(0).to(device)
            m        = compute_metrics(pred_dev, true_dev, ocean)

            model_preds[label]   = x0_pred.numpy()
            model_metrics[label] = m

            row = {"model": label, "sample": sample_idx, "path_seed": path_seed, **m}
            all_rows.append(row)

            metric_str = "  ".join(
                f"{k}={v:.5f}" if not np.isnan(v) else f"{k}=NaN"
                for k, v in m.items()
            )
            print(f"    {label:<16}  {metric_str}")

        # Save 3×3 image
        img_path = os.path.join(out_dir, f"sample_{sample_idx:04d}.png")
        save_sample_image(
            sample_idx, path_seed,
            x0_true.numpy(),
            path_mask, land_mask_np,
            model_preds, model_metrics, active_labels,
            img_path,
        )
        print(f"    -> {img_path}\n")

    # -------------------------------------------------------------------------
    # Write CSVs
    # -------------------------------------------------------------------------
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_cols)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Detail CSV  -> {detail_path}")

    summary_cols = (
        ["model"]
        + [f"{m}_mean" for m in LOSS_MODES]
        + [f"{m}_std"  for m in LOSS_MODES]
    )
    summary_rows = []
    for label in active_labels:
        rows = [r for r in all_rows if r["model"] == label]
        sr   = {"model": label}
        for m in LOSS_MODES:
            vals = [r[m] for r in rows if not np.isnan(r[m])]
            sr[f"{m}_mean"] = np.mean(vals) if vals else float("nan")
            sr[f"{m}_std"]  = np.std(vals)  if vals else float("nan")
        summary_rows.append(sr)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV -> {summary_path}")

    # -------------------------------------------------------------------------
    # Pretty-print summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY — mean across all samples")
    print(f"{'='*60}")
    header = f"{'Model':<18}" + "".join(f"  {m:>15}" for m in LOSS_MODES)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        line = f"{row['model']:<18}"
        for m in LOSS_MODES:
            val = row[f"{m}_mean"]
            line += f"  {val:>15.5f}" if not np.isnan(val) else f"  {'NaN':>15}"
        print(line)
    print()


if __name__ == "__main__":
    main()
