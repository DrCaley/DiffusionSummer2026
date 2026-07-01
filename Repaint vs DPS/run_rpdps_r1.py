"""
run_rpdps_r1.py
===============
Runs only RePaint+DPS-r1 (DPS gradient + hard mask merge, no resampling loop)
on the same 20 seeds used in compare_methods.py, then saves a standalone CSV.

Usage:
    python run_rpdps_r1.py --pickle data.pickle --checkpoint ckpt.pt \
        --T 1000 --stride 10 --out_dir results/rpdps_r1_T1000_s10
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path

SEEDS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63,
         70, 77, 84, 91, 98, 105, 112, 119, 126, 133]


def repaint_dps_r1_infer(model, diffusion, x0_known, path_mask, land_mask,
                         device="cpu", stride=1, step_size=0.5):
    """
    RePaint+DPS with r=1: DPS gradient correction + hard mask merge, no resampling.
    Equivalent to repaint_dps_infer(..., r=1).
    """
    H, W       = x0_known.shape[1:]
    x0_known_t = x0_known.unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t     = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        pred_noise = model(xt_in, t_vec)
        ab = diffusion.alpha_bar[t_int]
        x0_hat = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        residual = known_t * (x0_hat - x0_known_t)
        norm_sq  = (residual ** 2).sum()
        grad     = torch.autograd.grad(norm_sq, xt_in)[0]

        with torch.no_grad():
            xt_unknown = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)
            norm = norm_sq.sqrt().item() + 1e-8
            xt_unknown = xt_unknown - (step_size / norm) * grad.detach()

            t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_known_noisy, _ = diffusion.q_sample(x0_known_t, t_prev_t)
            xt = known_t * xt_known_noisy + (1.0 - known_t) * xt_unknown
            xt = xt * ocean_t

    return xt.squeeze(0).cpu().numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--stride",     type=int, default=None)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--dps_step",   type=float, default=0.5)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--n_seeds",    type=int, default=None)
    p.add_argument("--out_dir",    default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = args.stride if args.stride is not None else max(1, args.T // 100)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.out_dir is None:
        args.out_dir = os.path.join(script_dir, "results",
                                    f"rpdps_r1_T{args.T}_s{stride}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"T          : {args.T}  stride={stride}  "
          f"({len(range(0, args.T, stride))} diffusion steps)")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output dir : {args.out_dir}")

    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    train_ds     = OceanCurrentDataset(args.pickle, split=0)
    land_mask_np = test_ds.land_mask.numpy()
    ocean_mask   = ~land_mask_np

    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    schedule  = ckpt.get("schedule", "linear")

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    noise_std = ckpt.get("noise_std", None)
    if noise_std is None:
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())

    diffusion = DDPM(T=args.T, beta_schedule=schedule, device=device,
                     noise_std=noise_std)

    print(f"Loaded     : epoch {ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}  "
          f"schedule={schedule}  noise_std={noise_std:.5f}\n")

    seeds   = SEEDS[:args.n_seeds] if args.n_seeds is not None else SEEDS
    n_total = len(seeds)
    n_test  = len(test_ds)

    rows   = []
    rmses  = []
    times  = []

    for run_i, seed in enumerate(seeds):
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]
        true_np    = x0_true.numpy()
        path_mask  = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"[{run_i+1:02d}/{n_total:02d}]  seed={seed}  test_idx={sample_idx}", flush=True)

        t0   = time.perf_counter()
        pred = repaint_dps_r1_infer(model, diffusion, x0_obs, path_mask,
                                    land_mask_np, device=device,
                                    stride=stride, step_size=args.dps_step)
        elapsed = time.perf_counter() - t0

        rmse = float(np.sqrt(np.mean(
            (pred[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        rmses.append(rmse)
        times.append(elapsed)
        rows.append((seed, sample_idx, rmse, elapsed))
        print(f"  RePaint+DPS-r1 : RMSE={rmse:.4f}  t={elapsed:.1f}s", flush=True)

    mean_rmse = float(np.mean(rmses))
    std_rmse  = float(np.std(rmses))
    mean_time = float(np.mean(times))

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results_rpdps_r1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "test_idx", "RD1-RMSE", "RD1-t"])
        w.writerows(rows)
    print(f"\nCSV saved : {csv_path}")

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary_rpdps_r1.txt")
    with open(summary_path, "w") as f:
        f.write(f"RePaint+DPS-r1  —  T={args.T}  stride={stride}\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Schedule   : {schedule}   noise_std={noise_std:.5f}\n")
        f.write(f"N seeds    : {n_total}\n\n")
        f.write(f"Mean RMSE  : {mean_rmse:.4f}   Std: {std_rmse:.4f}\n")
        f.write(f"Mean Time  : {mean_time:.2f}s\n\n")
        f.write(f"{'Seed':>6}  {'idx':>5}  {'RMSE':>9}  {'Time(s)':>9}\n")
        f.write("-" * 40 + "\n")
        for seed, idx, rmse, t in rows:
            f.write(f"{seed:6d}  {idx:5d}  {rmse:9.4f}  {t:9.2f}\n")
    print(f"Summary saved : {summary_path}")


if __name__ == "__main__":
    main()
