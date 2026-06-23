"""
Evaluate every loss-function model on 10 validation samples using all 7 loss
metrics, then write CSV results and a printed summary.

Models evaluated (from the loss_comparison/ directory):
    model_ddpm_eps_gaussian_cosine.pt
    model_ddpm_curl_div_gaussian_cosine.pt
    model_ddpm_spectral_gaussian_cosine.pt
    model_ddpm_okubo_weiss_gaussian_cosine.pt
    model_ddpm_wasserstein_gaussian_cosine.pt
    model_ddpm_stream_function_gaussian_cosine.pt
    model_ddpm_strain_rate_gaussian_cosine.pt

Metrics computed per (model, sample):
    eps              ocean-masked RMSE of the predicted vs true field
    curl_div         RMSE of curl + divergence fields
    spectral         RMSE of FFT power spectra
    okubo_weiss      RMSE of Okubo-Weiss parameter W
    wasserstein      Sinkhorn-Wasserstein distance on vorticity clouds
    stream_function  RMSE of approximate stream-function fields
    strain_rate      RMSE of strain-rate tensor invariants

Outputs (relative to this script's directory):
    results/loss_eval_results.csv   — one row per (model, sample)
    results/loss_eval_summary.csv   — mean ± std per model across 10 samples

Usage (run from workspace root or this directory):
    python3 "Model Parameters/Loss Function/batch_eval_loss.py"
    python3 "Model Parameters/Loss Function/batch_eval_loss.py" \
        --path_steps 150 --resample 10 --n_runs 10
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))              # .../Loss Function/
_MODEL_PARAMS = os.path.dirname(_HERE)                          # .../Model Parameters/
_ROOT = os.path.dirname(_MODEL_PARAMS)                          # workspace root

# Root has: dataset.py, diffusion.py, model.py, repaint_infer.py
sys.path.insert(0, _ROOT)
# Loss Function dir has: loss_functions.py
sys.path.insert(0, _HERE)

from dataset        import OceanCurrentDataset
from diffusion      import DDPM
from model          import UNet
from repaint_infer  import biased_walk_path, repaint
from loss_functions import (
    curl_div_loss, spectral_loss, okubo_weiss_loss,
    wasserstein_loss, stream_function_loss, strain_rate_loss,
    LOSS_MODES,
)

# Optional: geomloss for Wasserstein
try:
    from geomloss import SamplesLoss
    _sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)
    _HAS_GEOMLOSS = True
except ImportError:
    _HAS_GEOMLOSS = False
    print("  [warn] geomloss not installed — wasserstein metric will be NaN")

# The 7 model labels (matched to filenames in loss_comparison/)
LOSS_LABELS = [
    "eps",
    "curl_div",
    "spectral",
    "okubo_weiss",
    "wasserstein",
    "stream_function",
    "strain_rate",
]

# Seeds identical to batch_eval.py: seed = i*7 + 1
N_RUNS_DEFAULT = 10
SEEDS = [i * 7 + 1 for i in range(20)]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",       default="data.pickle")
    p.add_argument("--ckpt_dir",     default=None,
                   help="Directory containing the 7 model .pt files. "
                        "Defaults to <workspace_root>/Model Parameters/loss_comparison/")
    p.add_argument("--n_runs",       type=int, default=N_RUNS_DEFAULT)
    p.add_argument("--path_steps",   type=int, default=150)
    p.add_argument("--resample",     type=int, default=10)
    p.add_argument("--T",            type=int, default=1000)
    p.add_argument("--base_ch",      type=int, default=64)
    p.add_argument("--time_dim",     type=int, default=256)
    p.add_argument("--out_dir",      default=None)
    p.add_argument("--labels",       nargs="+", default=LOSS_LABELS,
                   choices=LOSS_LABELS,
                   help="Subset of models to evaluate (default: all 7).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    model, diffusion, val_ds, land_mask_np, land_mask_t,
    sample_idx, seed, args, device,
):
    """Run inference for one sample and return a dict of 7 metric values."""
    x0_true = val_ds[sample_idx]   # (2, H, W)

    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

    x0_observed = x0_true.clone()
    path_t = torch.from_numpy(path_mask)
    x0_observed[:, ~path_t] = 0.0

    x0_pred = repaint(
        model, diffusion, x0_observed,
        path_mask, land_mask_np,
        r=args.resample, device=device,
    )  # (2, H, W) on CPU

    pred  = x0_pred.unsqueeze(0).to(device)    # (1, 2, H, W)
    true_ = x0_true.unsqueeze(0).to(device)    # (1, 2, H, W)
    ocean = (~land_mask_t).float()[None, None]  # (1, 1, H, W)

    metrics = {}

    metrics["eps"] = F.mse_loss(pred * ocean, true_ * ocean).sqrt().item()
    metrics["curl_div"]        = curl_div_loss(pred, true_, ocean).item()
    metrics["spectral"]        = spectral_loss(pred, true_, ocean).item()
    metrics["okubo_weiss"]     = okubo_weiss_loss(pred, true_, ocean).item()
    metrics["stream_function"] = stream_function_loss(pred, true_, ocean).item()
    metrics["strain_rate"]     = strain_rate_loss(pred, true_, ocean).item()

    if _HAS_GEOMLOSS:
        metrics["wasserstein"] = wasserstein_loss(
            pred, true_, ocean, _sinkhorn
        ).item()
    else:
        metrics["wasserstein"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Resolve checkpoint directory
    if args.ckpt_dir is None:
        ckpt_dir = os.path.join(_MODEL_PARAMS, "loss_comparison")
    else:
        ckpt_dir = args.ckpt_dir
    print(f"Checkpoint dir : {ckpt_dir}")

    # Resolve output directory
    if args.out_dir is None:
        out_dir = os.path.join(_HERE, "results")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Resolve data.pickle
    pickle_path = args.pickle
    if not os.path.isabs(pickle_path) and not os.path.exists(pickle_path):
        pickle_path = os.path.join(_ROOT, pickle_path)

    val_ds       = OceanCurrentDataset(pickle_path, split=1)
    land_mask_np = val_ds.land_mask.numpy()       # (H, W) bool
    land_mask_t  = val_ds.land_mask.to(device)    # (H, W) bool tensor

    # CSV paths
    detail_path  = os.path.join(out_dir, "loss_eval_results.csv")
    summary_path = os.path.join(out_dir, "loss_eval_summary.csv")

    detail_cols = ["model", "sample", "seed"] + list(LOSS_MODES)
    all_rows    = []

    # -------------------------------------------------------------------------
    # Outer loop: models
    # -------------------------------------------------------------------------
    for label in args.labels:
        ckpt_path = os.path.join(
            ckpt_dir, f"model_ddpm_{label}_gaussian_cosine.pt"
        )
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"{'='*60}")
        print(f"Model: {label}   checkpoint: {ckpt_path}")

        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        base_ch   = ckpt_args.get("base_ch",  args.base_ch)
        time_dim  = ckpt_args.get("time_dim", args.time_dim)
        T         = ckpt_args.get("T",        args.T)
        schedule  = ckpt_args.get("schedule", "cosine")

        model = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"  Loaded epoch {ckpt.get('epoch', '?')}, T={T}, schedule={schedule}")

        diffusion = DDPM(T=T, beta_schedule=schedule, device=device)

        model_metrics = {m: [] for m in LOSS_MODES}

        # Inner loop: validation samples
        for i in range(args.n_runs):
            sample_idx = i
            seed       = SEEDS[i]
            print(
                f"  Run {i+1:2d}/{args.n_runs}"
                f"  (val sample={sample_idx}, seed={seed})",
                end="  ", flush=True,
            )

            metrics = evaluate_sample(
                model, diffusion, val_ds, land_mask_np, land_mask_t,
                sample_idx, seed, args, device,
            )

            row = {"model": label, "sample": sample_idx, "seed": seed, **metrics}
            all_rows.append(row)

            for m in LOSS_MODES:
                model_metrics[m].append(metrics[m])

            metric_str = "  ".join(
                f"{m}={metrics[m]:.5f}" if not np.isnan(metrics[m]) else f"{m}=NaN"
                for m in LOSS_MODES
            )
            print(metric_str)

        # Per-model summary
        print(f"\n  --- {label} summary (n={args.n_runs}) ---")
        for m in LOSS_MODES:
            vals = [v for v in model_metrics[m] if not np.isnan(v)]
            if vals:
                print(f"    {m:15s}: mean={np.mean(vals):.5f}  std={np.std(vals):.5f}")
            else:
                print(f"    {m:15s}: NaN")
        print()

    # -------------------------------------------------------------------------
    # Write detail CSV
    # -------------------------------------------------------------------------
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_cols)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Detail results  -> {detail_path}")

    # -------------------------------------------------------------------------
    # Write summary CSV (mean ± std per model)
    # -------------------------------------------------------------------------
    summary_cols = (
        ["model"]
        + [f"{m}_mean" for m in LOSS_MODES]
        + [f"{m}_std"  for m in LOSS_MODES]
    )
    summary_rows = []
    for label in args.labels:
        rows = [r for r in all_rows if r["model"] == label]
        if not rows:
            continue
        summary_row = {"model": label}
        for m in LOSS_MODES:
            vals = [r[m] for r in rows if not np.isnan(r[m])]
            summary_row[f"{m}_mean"] = np.mean(vals) if vals else float("nan")
            summary_row[f"{m}_std"]  = np.std(vals)  if vals else float("nan")
        summary_rows.append(summary_row)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary results -> {summary_path}")

    # -------------------------------------------------------------------------
    # Pretty-print summary table
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY — mean across all runs")
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
