"""
Evaluate every noise-schedule model on 10 validation samples using all 5 loss
functions as metrics.

For each of the 4 schedules (cosine, linear, quadratic, sigmoid):
  - Load the corresponding best_model checkpoint
  - Run RePaint inference on val samples 0-9 with the same seeds used by batch_repaint.py
  - Compute 5 metrics per prediction vs ground truth:
      eps          -> ocean-masked MSE between predicted and true u/v fields
      curl_div     -> MSE of curl + divergence fields
      spectral     -> MSE of FFT power spectra
      okubo_weiss  -> MSE of Okubo-Weiss parameter W
      wasserstein  -> Sinkhorn-Wasserstein distance on vorticity clouds
                      (skipped with NaN if geomloss is not installed)

Outputs:
  results/eval_results.csv   - one row per (schedule, sample)
  results/eval_summary.csv   - mean ± std per schedule across 10 samples (printed too)

Usage (run from workspace root or NoiseSchedule/):
    python3 "Model Parameters/NoiseSchedule/batch_eval.py"
    python3 "Model Parameters/NoiseSchedule/batch_eval.py" --path_steps 150 --resample 10
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(os.path.dirname(_HERE))   # workspace root
sys.path.insert(0, _HERE)   # repaint_infer, repaint_model, diffusion (local versions)
sys.path.append(_ROOT)      # dataset.py

# loss_functions.py: locally in ../Loss Function/, on server directly in ../
_LOSS_DIR_LOCAL  = os.path.join(os.path.dirname(_HERE), "Loss Function")
_LOSS_DIR_SERVER = os.path.dirname(_HERE)   # one level up (Model Parameters/)
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

# Optional: geomloss for Wasserstein
try:
    from geomloss import SamplesLoss
    _sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)
    _HAS_GEOMLOSS = True
except ImportError:
    _HAS_GEOMLOSS = False
    print("  [warn] geomloss not installed — wasserstein metric will be NaN")

SCHEDULES = ["cosine", "linear", "quadratic", "sigmoid"]

# Identical seeds to batch_repaint.py: seed = i*7 + 1 for run i (0-indexed)
N_RUNS = 10
SEEDS  = [i * 7 + 1 for i in range(N_RUNS)]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--resample",   type=int, default=10)
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--out_dir",    default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    model, diffusion, val_ds, land_mask_np, land_mask_t,
    sample_idx, seed, args, device,
):
    """
    Run inference for one sample and return a dict of 5 metric values.
    """
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

    # Expand to (1, 2, H, W) for loss functions
    pred  = x0_pred.unsqueeze(0).to(device)
    true_ = x0_true.unsqueeze(0).to(device)
    ocean = (~land_mask_t).float()[None, None]   # (1, 1, H, W)

    metrics = {}

    # eps: ocean-masked RMSE on the field values themselves
    import torch.nn.functional as F
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

    # Resolve output dir relative to this script
    if args.out_dir is None:
        args.out_dir = os.path.join(_HERE, "results")
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve data.pickle relative to workspace root if not found locally
    pickle_path = args.pickle
    if not os.path.isabs(pickle_path) and not os.path.exists(pickle_path):
        pickle_path = os.path.join(_ROOT, args.pickle)

    val_ds       = OceanCurrentDataset(pickle_path, split=1)
    land_mask_np = val_ds.land_mask.numpy()       # (H, W) bool
    land_mask_t  = val_ds.land_mask.to(device)    # (H, W) bool tensor

    # ---------------------------------------------------------------------------
    # CSV header
    # ---------------------------------------------------------------------------
    detail_path  = os.path.join(args.out_dir, "eval_results.csv")
    summary_path = os.path.join(args.out_dir, "eval_summary.csv")

    detail_cols = ["schedule", "sample", "seed"] + list(LOSS_MODES)
    all_rows    = []

    # ---------------------------------------------------------------------------
    # Outer loop: schedules
    # ---------------------------------------------------------------------------
    for schedule in SCHEDULES:
        ckpt_path = os.path.join(
            _HERE, "checkpoints",
            f"checkpoints_repaint_{schedule}",
            f"best_model_{schedule}.pt",
        )
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {schedule}: checkpoint not found at {ckpt_path}")
            continue

        print(f"{'='*60}")
        print(f"Schedule: {schedule}   checkpoint: {ckpt_path}")

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

        schedule_metrics = {m: [] for m in LOSS_MODES}

        # Inner loop: 10 val samples
        for i in range(N_RUNS):
            sample_idx = i
            seed       = SEEDS[i]
            print(f"  Run {i+1:2d}/10  (val sample={sample_idx}, seed={seed})", end="  ", flush=True)

            metrics = evaluate_sample(
                model, diffusion, val_ds, land_mask_np, land_mask_t,
                sample_idx, seed, args, device,
            )

            row = {
                "schedule": schedule,
                "sample":   sample_idx,
                "seed":     seed,
                **metrics,
            }
            all_rows.append(row)

            for m in LOSS_MODES:
                schedule_metrics[m].append(metrics[m])

            metric_str = "  ".join(
                f"{m}={metrics[m]:.5f}" if not np.isnan(metrics[m]) else f"{m}=NaN"
                for m in LOSS_MODES
            )
            print(metric_str)

        # Per-schedule summary
        print(f"\n  --- {schedule} summary (n=10) ---")
        for m in LOSS_MODES:
            vals = [v for v in schedule_metrics[m] if not np.isnan(v)]
            if vals:
                print(f"    {m:15s}: mean={np.mean(vals):.5f}  std={np.std(vals):.5f}")
            else:
                print(f"    {m:15s}: NaN")
        print()

    # ---------------------------------------------------------------------------
    # Write detail CSV
    # ---------------------------------------------------------------------------
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_cols)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Detail results -> {detail_path}")

    # ---------------------------------------------------------------------------
    # Write summary CSV (mean ± std per schedule)
    # ---------------------------------------------------------------------------
    summary_cols = ["schedule"] + [f"{m}_mean" for m in LOSS_MODES] + \
                                   [f"{m}_std"  for m in LOSS_MODES]
    summary_rows = []
    for schedule in SCHEDULES:
        rows = [r for r in all_rows if r["schedule"] == schedule]
        if not rows:
            continue
        summary_row = {"schedule": schedule}
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

    # ---------------------------------------------------------------------------
    # Build summary table string (printed + written to .txt)
    # ---------------------------------------------------------------------------
    col_w   = 14
    divider = "=" * (12 + col_w * len(LOSS_MODES))
    lines   = []
    lines.append(divider)
    lines.append(f"{'EVALUATION SUMMARY':^{len(divider)}}")
    lines.append(f"path_steps={args.path_steps}  resample={args.resample}  n_runs={N_RUNS}")
    lines.append(divider)
    header_line = f"{'schedule':<12}" + "".join(f"{m:>{col_w}}" for m in LOSS_MODES)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in summary_rows:
        mean_line = f"{row['schedule']:<12}"
        std_line  = f"{'':12}"
        for m in LOSS_MODES:
            mv = row[f"{m}_mean"]
            sv = row[f"{m}_std"]
            mean_line += f"{mv:>{col_w}.5f}" if not np.isnan(mv) else f"{'NaN':>{col_w}}"
            std_line  += f"{'(±'+f'{sv:.5f}'+')':>{col_w}}" if not np.isnan(sv) else f"{'':>{col_w}}"
        lines.append(mean_line)
        lines.append(std_line)
        lines.append("")
    lines.append(divider)

    # Also append per-sample detail for each schedule
    lines.append("")
    lines.append("PER-SAMPLE DETAIL")
    lines.append(divider)
    detail_header = f"{'schedule':<12}{'sample':>8}{'seed':>7}" + \
                    "".join(f"{m:>{col_w}}" for m in LOSS_MODES)
    lines.append(detail_header)
    lines.append("-" * len(detail_header))
    for row in all_rows:
        detail_line = f"{row['schedule']:<12}{row['sample']:>8}{row['seed']:>7}"
        for m in LOSS_MODES:
            v = row[m]
            detail_line += f"{v:>{col_w}.5f}" if not np.isnan(v) else f"{'NaN':>{col_w}}"
        lines.append(detail_line)
    lines.append(divider)

    table = "\n".join(lines)
    print("\n" + table)

    txt_path = os.path.join(args.out_dir, "eval_summary.txt")
    with open(txt_path, "w") as f:
        f.write(table + "\n")
    print(f"\nText summary  -> {txt_path}")


if __name__ == "__main__":
    main()
