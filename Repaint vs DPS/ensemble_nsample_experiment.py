"""
Ensemble n_samples experiment.

For each ensemble size N in [1, 5, 10, 15, 20]:
  - Run RePaint N times per test case and average the predictions
  - Record per-seed inference time and RMSE
  - Report mean time and mean RMSE across all 50 seeds

Settings: T=1000, stride=10, resample=1 (no resampling)
Seeds: 50 nonconsecutive test indices evenly spread across the test set

Usage (from /root/Repaint_vs_DPS/):
    python ensemble_nsample_experiment.py
    python ensemble_nsample_experiment.py --checkpoint checkpoints_linear/best_model_linear.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path, repaint


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--checkpoint",  default="checkpoints_linear/best_model_linear.pt")
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--stride",      type=int,   default=10)
    p.add_argument("--resample",    type=int,   default=1,
                   help="RePaint r parameter (1 = no resampling).")
    p.add_argument("--path_steps",  type=int,   default=150)
    p.add_argument("--n_seeds",     type=int,   default=50,
                   help="Number of nonconsecutive test indices to evaluate.")
    p.add_argument("--seed_shift",  type=int,   default=0,
                   help="Offset added to each linspace seed index (use to get non-overlapping second batch).")
    p.add_argument("--ensemble_sizes", type=int, nargs="+",
                   default=[1, 5, 10, 15, 20],
                   help="List of ensemble sizes to test.")
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--time_dim",    type=int,   default=256)
    p.add_argument("--out",         default=None,
                   help="Output txt path. Defaults to results/ensemble_nsample_results_shift<N>.txt")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.out is None:
        args.out = f"results/ensemble_nsample_results_shift{args.seed_shift}.txt"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Device         : {device}")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"T={args.T}, stride={args.stride}, resample={args.resample}")
    print(f"Ensemble sizes : {args.ensemble_sizes}")
    print(f"N seeds        : {args.n_seeds}")

    # ---- Dataset ----
    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    land_mask_np = test_ds.land_mask.numpy()
    ocean_mask   = ~land_mask_np
    n_test       = len(test_ds)
    print(f"Test set size  : {n_test}")

    # nonconsecutive seeds evenly spread across the test set, with optional shift
    seed_indices = [
        int(np.clip(i + args.seed_shift, 0, n_test - 1))
        for i in np.linspace(0, n_test - 1, args.n_seeds, dtype=int).tolist()
    ]
    print(f"Seed indices   : {seed_indices}\n")

    # ---- Model ----
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    noise_std = ckpt.get("noise_std", None)
    if noise_std is None:
        train_ds  = OceanCurrentDataset(args.pickle, split=0)
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())

    beta_schedule = ckpt.get("schedule", "linear")

    print(f"Loaded  : epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}, "
          f"schedule={beta_schedule}, noise_std={noise_std:.5f}\n")

    diffusion = DDPM(T=T, beta_schedule=beta_schedule, device=device, noise_std=noise_std)

    n_steps_per_run = len(range(0, T, args.stride))
    print(f"Steps per run  : {n_steps_per_run}  (T={T}, stride={args.stride})\n")

    # ---- Experiment ----
    # results[n] = list of (seed_idx, time_sec, rmse) for ensemble size n
    results = {n: [] for n in args.ensemble_sizes}

    for seed_number, test_idx in enumerate(seed_indices):
        x0_true  = test_ds[test_idx]
        true_np  = x0_true.numpy()

        path_mask = biased_walk_path(
            land_mask_np, n_steps=args.path_steps, seed=test_idx
        )
        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"=== Seed {seed_number+1:02d}/{args.n_seeds}  (test_idx={test_idx}) ===")

        for n in args.ensemble_sizes:
            t_start = time.perf_counter()

            # Run repaint n times and accumulate
            preds = []
            for run_i in range(n):
                pred = repaint(
                    model, diffusion, x0_obs, path_mask, land_mask_np,
                    r=args.resample, device=device, stride=args.stride,
                )
                # repaint returns a torch.Tensor; convert to numpy
                if hasattr(pred, "cpu"):
                    pred = pred.cpu().numpy()
                preds.append(pred)

            elapsed = time.perf_counter() - t_start
            ensemble_pred = np.mean(np.stack(preds, axis=0), axis=0)

            rmse = float(np.sqrt(np.mean(
                (ensemble_pred[:, ocean_mask] - true_np[:, ocean_mask]) ** 2
            )))

            results[n].append((test_idx, elapsed, rmse))
            print(f"  n_samples={n:2d} | {elapsed:6.1f}s | RMSE={rmse:.4f}")

    # ---- Summary ----
    lines = []
    lines.append("Ensemble n_samples experiment")
    lines.append(f"Checkpoint    : {args.checkpoint}")
    lines.append(f"T={T}, stride={args.stride}, resample={args.resample}")
    lines.append(f"noise_std     : {noise_std:.5f}")
    lines.append(f"schedule      : {beta_schedule}")
    lines.append(f"n_seeds       : {args.n_seeds}")
    lines.append(f"steps/run     : {n_steps_per_run}")
    lines.append("")
    lines.append(f"{'n_samples':>10}  {'mean_time(s)':>13}  {'std_time':>9}  "
                 f"{'mean_RMSE':>10}  {'std_RMSE':>9}  {'min_RMSE':>9}  {'max_RMSE':>9}")
    lines.append("-" * 80)

    for n in args.ensemble_sizes:
        times = [r[1] for r in results[n]]
        rmses = [r[2] for r in results[n]]
        lines.append(
            f"{n:>10}  {np.mean(times):>13.2f}  {np.std(times):>9.2f}  "
            f"{np.mean(rmses):>10.4f}  {np.std(rmses):>9.4f}  "
            f"{np.min(rmses):>9.4f}  {np.max(rmses):>9.4f}"
        )

    lines.append("")
    lines.append("Per-seed breakdown:")
    header = f"{'test_idx':>10}" + "".join(
        f"  t(n={n})  RMSE(n={n})" for n in args.ensemble_sizes
    )
    lines.append(header)
    lines.append("-" * (10 + 22 * len(args.ensemble_sizes)))

    for i, test_idx in enumerate(seed_indices):
        row = f"{test_idx:>10}"
        for n in args.ensemble_sizes:
            t_s, rmse = results[n][i][1], results[n][i][2]
            row += f"  {t_s:6.1f}s  {rmse:.4f}"
        lines.append(row)

    report = "\n".join(lines)
    print("\n\n" + report)

    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
