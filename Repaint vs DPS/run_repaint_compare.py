"""
run_repaint_compare.py
======================
Compares only RePaint-r10 and RePaint-r1 on the same 20 seeds.

Usage:
    python run_repaint_compare.py --pickle data.pickle --checkpoint ckpt.pt \
        --T 1000 --stride 10 --out_dir results/repaint_T1000_s10
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


@torch.no_grad()
def repaint_infer(model, diffusion, x0_known, path_mask, land_mask,
                  r=10, device="cpu", stride=1):
    H, W      = x0_known.shape[1:]
    x0_known  = x0_known.unsqueeze(0).to(device)
    known_t   = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t    = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t   = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    T  = diffusion.T
    timesteps = list(range(0, T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        for j in range(r):
            with torch.no_grad():
                xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)
                t_prev_t   = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
                xt_known_noisy, _ = diffusion.q_sample(x0_known, t_prev_t)
                xt_merged  = known_t * xt_known_noisy + (1.0 - known_t) * xt_unknown
                xt_merged  = xt_merged * ocean_t

                if j < r - 1 and t_int > 0:
                    xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int) * ocean_t
                else:
                    xt = xt_merged

    return xt.squeeze(0).cpu().numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--stride",     type=int, default=None)
    p.add_argument("--path_steps", type=int, default=150)
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
                                    f"repaint_T{args.T}_s{stride}")
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

    all_rmse  = {"RePaint-r10": [], "RePaint-r1": []}
    all_times = {"RePaint-r10": [], "RePaint-r1": []}
    rows = []

    for run_i, seed in enumerate(seeds):
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]
        true_np    = x0_true.numpy()
        path_mask  = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"\n[{run_i+1:02d}/{n_total:02d}]  seed={seed}  test_idx={sample_idx}", flush=True)

        t0 = time.perf_counter()
        pred_r10 = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask_np,
                                 r=10, device=device, stride=stride)
        t_r10 = time.perf_counter() - t0
        rmse_r10 = float(np.sqrt(np.mean((pred_r10[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        all_rmse["RePaint-r10"].append(rmse_r10)
        all_times["RePaint-r10"].append(t_r10)
        print(f"  RePaint-r10 : RMSE={rmse_r10:.4f}  t={t_r10:.1f}s", flush=True)

        t0 = time.perf_counter()
        pred_r1 = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask_np,
                                r=1, device=device, stride=stride)
        t_r1 = time.perf_counter() - t0
        rmse_r1 = float(np.sqrt(np.mean((pred_r1[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        all_rmse["RePaint-r1"].append(rmse_r1)
        all_times["RePaint-r1"].append(t_r1)
        print(f"  RePaint-r1  : RMSE={rmse_r1:.4f}  t={t_r1:.1f}s", flush=True)

        rows.append((seed, sample_idx, rmse_r10, t_r10, rmse_r1, t_r1))

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "test_idx", "RPr10-RMSE", "RPr10-t", "RPr1-RMSE", "RPr1-t"])
        w.writerows(rows)

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"RePaint Comparison  —  T={args.T}  stride={stride}\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Schedule   : {schedule}   noise_std={noise_std:.5f}\n")
        f.write(f"N seeds    : {n_total}\n\n")
        f.write(f"{'Method':<15} {'Mean RMSE':>10} {'Std RMSE':>10} "
                f"{'Min':>8} {'Max':>8} {'Mean Time(s)':>13}\n")
        f.write("-" * 68 + "\n")
        for method in ["RePaint-r10", "RePaint-r1"]:
            r = all_rmse[method]
            t = all_times[method]
            f.write(f"{method:<15} {np.mean(r):>10.4f} {np.std(r):>10.4f} "
                    f"{np.min(r):>8.4f} {np.max(r):>8.4f} {np.mean(t):>13.2f}\n")
        f.write("\nPer-seed breakdown:\n")
        f.write(f"  {'Seed':>6}  {'idx':>5}  {'RPr10-RMSE':>11}  {'RPr10-t':>8}  "
                f"{'RPr1-RMSE':>10}  {'RPr1-t':>8}\n")
        f.write("-" * 68 + "\n")
        for seed, idx, rr10, tr10, rr1, tr1 in rows:
            f.write(f"  {seed:6d}  {idx:5d}  {rr10:11.4f}  {tr10:8.2f}  "
                    f"{rr1:10.4f}  {tr1:8.2f}\n")

    print(f"\nCSV saved     : {csv_path}")
    print(f"Summary saved : {summary_path}")


if __name__ == "__main__":
    main()
