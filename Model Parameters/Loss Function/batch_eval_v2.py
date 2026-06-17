"""
Evaluate 6 loss-function models (no spectral model) on N randomly chosen
validation samples, scored by 3 metrics only: RMSE (field), Spectral RMSE,
and Wasserstein distance.

Layout of each saved image (3×3 grid, last slot empty):
  [ Ground Truth  |  Robot Path       |  eps output         ]
  [ curl_div      |  okubo_weiss      |  wasserstein output  ]
  [ stream_fn     |  strain_rate      |  (empty)             ]

Each model panel title shows: rmse=X  spec=X  wass=X

Outputs go into results/eval_run2/ — nothing in that folder is overwritten
by a previous run.

Usage:
    python3 "Model Parameters/Loss Function/batch_eval_v2.py"
    python3 "Model Parameters/Loss Function/batch_eval_v2.py" \\
        --n_samples 10 --path_steps 150 --resample 10 --seed 99
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

sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from dataset        import OceanCurrentDataset
from diffusion      import DDPM
from model          import UNet
from repaint_infer  import biased_walk_path, repaint
from loss_functions import spectral_loss, wasserstein_loss

try:
    from geomloss import SamplesLoss
    _sinkhorn     = SamplesLoss("sinkhorn", p=2, blur=0.05)
    _HAS_GEOMLOSS = True
except ImportError:
    _HAS_GEOMLOSS = False
    print("  [warn] geomloss not installed — wasserstein metric will be NaN")

# 6 models: spectral removed
MODEL_LABELS = [
    "eps",
    "curl_div",
    "okubo_weiss",
    "wasserstein",
    "stream_function",
    "strain_rate",
]

# Only 3 metrics
ACTIVE_METRICS = ["eps", "spectral", "wasserstein"]

METRIC_SHORT = {
    "eps":       "rmse",
    "spectral":  "spec",
    "wasserstein": "wass",
}

# Panel order for model outputs (matches MODEL_LABELS)
_PANEL_LABELS = MODEL_LABELS[:]   # same order


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--ckpt_dir",   default=None,
                   help="Dir with model_ddpm_*_gaussian_cosine.pt files.")
    p.add_argument("--n_samples",  type=int, default=10)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--resample",   type=int, default=10)
    p.add_argument("--seed",       type=int, default=99,
                   help="RNG seed for sample selection (default 99, different from run1).")
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--out_dir",    default=None,
                   help="Output image directory. Defaults to results/eval_run2/")
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
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Metrics  (only 3)
# ---------------------------------------------------------------------------

def compute_metrics(pred_dev, true_dev, ocean):
    m = {}
    m["eps"]      = F.mse_loss(pred_dev * ocean, true_dev * ocean).sqrt().item()
    m["spectral"] = spectral_loss(pred_dev, true_dev, ocean).item()
    if _HAS_GEOMLOSS:
        m["wasserstein"] = wasserstein_loss(pred_dev, true_dev, ocean, _sinkhorn).item()
    else:
        m["wasserstein"] = float("nan")
    return m


# ---------------------------------------------------------------------------
# Per-sample 3×3 image  (6 models + GT + path, last slot empty)
# ---------------------------------------------------------------------------

def save_sample_image(
    sample_idx, path_seed,
    x0_true_np,
    path_mask,
    land_mask_np,
    model_preds,
    model_metrics,
    active_labels,
    out_path,
):
    land_d = land_mask_np.T
    path_d = path_mask.T

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
    ax[1].set_xlabel("X", fontsize=7)
    ax[1].set_ylabel("Y", fontsize=7)
    ax[1].tick_params(labelsize=6)
    ax[1].legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
            mpatches.Patch(facecolor="#d62728", label="Path"),
            mpatches.Patch(facecolor="black",   label="Land"),
        ],
        loc="upper right", fontsize=6,
    )

    # ── Panels 2–7: one per model ─────────────────────────────────────────
    panel_order = [l for l in _PANEL_LABELS if l in active_labels]
    for pi, label in enumerate(panel_order):
        pidx = pi + 2
        if pidx >= len(ax):
            break
        pred_np = model_preds[label]
        m       = model_metrics[label]

        metric_line = "  ".join(
            f"{METRIC_SHORT[k]}={m[k]:.4f}" if not np.isnan(m[k]) else f"{METRIC_SHORT[k]}=NaN"
            for k in ACTIVE_METRICS
        )
        title = f"Model: {label}\n{metric_line}"
        plot_field(ax[pidx], pred_np[0].T, pred_np[1].T, land_d,
                   title, cmap="cool", clim=clim)

    # Hide unused slots (last panel with 6 models + 2 header = 8/9 used)
    for pidx in range(2 + len(panel_order), len(ax)):
        ax[pidx].axis("off")

    fig.suptitle(
        f"Val sample {sample_idx}  —  6 loss-function models (no spectral), same robot path",
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

    if args.ckpt_dir is None:
        ckpt_dir = os.path.join(_MODEL_PARAMS, "loss_comparison")
    else:
        ckpt_dir = args.ckpt_dir

    if args.out_dir is None:
        out_dir = os.path.join(_HERE, "results", "eval_run2")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # CSVs go into the same new folder
    detail_path  = os.path.join(out_dir, "results.csv")
    summary_path = os.path.join(out_dir, "summary.csv")

    pickle_path = args.pickle
    if not os.path.isabs(pickle_path) and not os.path.exists(pickle_path):
        pickle_path = os.path.join(_ROOT, pickle_path)

    val_ds       = OceanCurrentDataset(pickle_path, split=1)
    land_mask_np = val_ds.land_mask.numpy()
    land_mask_t  = val_ds.land_mask.to(device)
    ocean        = (~land_mask_t).float()[None, None]

    n_val = len(val_ds)
    rng   = np.random.default_rng(args.seed)
    sample_indices = sorted(
        rng.choice(n_val, size=min(args.n_samples, n_val), replace=False).tolist()
    )
    print(f"Val set size : {n_val}")
    print(f"Random seed  : {args.seed}")
    print(f"Samples      : {sample_indices}\n")

    path_seeds = {idx: int(rng.integers(0, 99999)) for idx in sample_indices}

    # -------------------------------------------------------------------------
    # Load 6 models
    # -------------------------------------------------------------------------
    models_loaded = {}
    diffusions    = {}
    active_labels = []
    for label in MODEL_LABELS:
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

    detail_cols = ["model", "sample", "path_seed"] + ACTIVE_METRICS
    all_rows    = []

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    for si, sample_idx in enumerate(sample_indices):
        path_seed = path_seeds[sample_idx]
        print(f"[{si+1:2d}/{len(sample_indices)}] Val sample {sample_idx}  path_seed={path_seed}")

        x0_true   = val_ds[sample_idx]
        path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=path_seed)

        x0_observed = x0_true.clone()
        path_t      = torch.from_numpy(path_mask)
        x0_observed[:, ~path_t] = 0.0

        true_dev = x0_true.unsqueeze(0).to(device)

        model_preds   = {}
        model_metrics = {}

        for label in active_labels:
            torch.manual_seed(args.seed + sample_idx + hash(label) % 10000)
            if device == "cuda":
                torch.cuda.manual_seed_all(args.seed + sample_idx + hash(label) % 10000)

            x0_pred  = repaint(
                models_loaded[label], diffusions[label], x0_observed,
                path_mask, land_mask_np,
                r=args.resample, device=device,
            )

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
            print(f"    {label:<18}  {metric_str}")

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

    summary_cols = ["model"] + [f"{m}_mean" for m in ACTIVE_METRICS] + [f"{m}_std" for m in ACTIVE_METRICS]
    summary_rows = []
    for label in active_labels:
        rows = [r for r in all_rows if r["model"] == label]
        sr   = {"model": label}
        for m in ACTIVE_METRICS:
            vals = [r[m] for r in rows if not np.isnan(r[m])]
            sr[f"{m}_mean"] = np.mean(vals) if vals else float("nan")
            sr[f"{m}_std"]  = np.std(vals)  if vals else float("nan")
        summary_rows.append(sr)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV -> {summary_path}")

    print(f"\n{'='*60}")
    print("SUMMARY — mean across all samples")
    print(f"{'='*60}")
    header = f"{'Model':<18}" + "".join(f"  {m:>15}" for m in ACTIVE_METRICS)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        line = f"{row['model']:<18}"
        for m in ACTIVE_METRICS:
            val = row[f"{m}_mean"]
            line += f"  {val:>15.5f}" if not np.isnan(val) else f"  {'NaN':>15}"
        print(line)
    print()


if __name__ == "__main__":
    main()
