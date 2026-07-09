"""
infer_batch_magangle.py
========================
For N seeds using biased_walk_path (random-walk observation path), runs:
  - RePaint r=10
  - RePaint r=1
  - DPS z=0.04

Instead of RMSE, reports two per-pixel error metrics (ocean cells only),
averaged over all pixels and all seeds:

  - Magnitude error : |‖pred‖ - ‖truth‖|            (speed error)
  - Angle error      : arccos( (pred·truth) / (‖pred‖‖truth‖ + eps) )
                        i.e. the angle between predicted and true vectors,
                        in degrees (matches the `angle_loss` convention
                        already used in loss_functions.py, reported here as
                        an actual angle rather than 1 - cos).

Saves results.csv, summary.txt, and bar_chart.png.

Usage:
    python infer_batch_magangle.py \\
        --pickle /root/ocean_ddpm/data.pickle \\
        --checkpoint /root/Repaint_vs_DPS/checkpoints_linear/best_model_linear.pt \\
        --out_dir /root/Repaint_vs_DPS/results/magangle_T1000_s10_50seeds \\
        --n_seeds 50 --T 1000 --stride 10 --path_steps 150
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path

SEEDS = list(range(0, 700, 7))  # 100 seeds: 0,7,14,...,693

METHODS = ["RePaint r=10", "RePaint r=1", "DPS z=0.04"]


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def repaint_infer(model, diffusion, x0_known, path_mask, land_mask,
                  r=10, device="cpu", stride=1):
    H, W     = x0_known.shape[1:]
    x0_known = x0_known.unsqueeze(0).to(device)
    known_t  = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t  = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0
        for j in range(r):
            xt_unk = diffusion.p_sample_step(model, xt, t_int, t_prev_int)
            t_prev = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_kn, _ = diffusion.q_sample(x0_known, t_prev)
            xt = (known_t * xt_kn + (1 - known_t) * xt_unk) * ocean_t
            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt, t_int, t_prev_int) * ocean_t

    return xt.squeeze(0).cpu().numpy()


def dps_infer(model, diffusion, x0_known, path_mask, land_mask,
              device="cpu", stride=1, step_size=0.04):
    H, W       = x0_known.shape[1:]
    x0_kn_t    = x0_known.unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        eps   = model(xt_in, t_vec)
        ab    = diffusion.alpha_bar[t_int]
        x0h   = ((xt_in - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.5, 1.5)
        nsq   = (known_t * (x0h - x0_kn_t) ** 2).sum()
        grad  = torch.autograd.grad(nsq, xt_in)[0]

        with torch.no_grad():
            xt = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)
            xt = (xt - (step_size / (nsq.sqrt().item() + 1e-8)) * grad.detach()) * ocean_t

    return xt.squeeze(0).cpu().numpy()


# ── metrics ───────────────────────────────────────────────────────────────────

def magnitude_angle_error(pred, truth, ocean_mask, eps=1e-8):
    """
    pred, truth: (2, H, W) numpy arrays (u, v channels)
    ocean_mask:  (H, W) bool, True = ocean

    Returns (mean_mag_error, mean_angle_error_deg) over ocean pixels.
    """
    pu, pv = pred[0][ocean_mask],  pred[1][ocean_mask]
    tu, tv = truth[0][ocean_mask], truth[1][ocean_mask]

    pred_mag = np.sqrt(pu**2 + pv**2)
    true_mag = np.sqrt(tu**2 + tv**2)
    mag_err  = float(np.mean(np.abs(pred_mag - true_mag)))

    dot = pu * tu + pv * tv
    cos = dot / (pred_mag * true_mag + eps)
    cos = np.clip(cos, -1.0, 1.0)
    angle_err_deg = float(np.degrees(np.mean(np.arccos(cos))))

    return mag_err, angle_err_deg


# ── bar chart ─────────────────────────────────────────────────────────────────

def save_bar_chart(all_mag, all_ang, all_times, T, stride, n_seeds, out_path):
    methods = list(all_mag.keys())
    mag   = [np.mean(all_mag[m])   for m in methods]
    mstd  = [np.std(all_mag[m])    for m in methods]
    ang   = [np.mean(all_ang[m])   for m in methods]
    astd  = [np.std(all_ang[m])    for m in methods]
    times = [np.mean(all_times[m]) for m in methods]

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    x = np.arange(len(methods))
    w = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(f"T={T} / stride={stride}  —  {n_seeds} seeds", fontsize=11)

    ax = axes[0]
    bars = ax.bar(x, mag, w, yerr=mstd, capsize=5, color=colors, alpha=0.85)
    ax.set_title("Mean Magnitude Error (± 1 std)", fontsize=10)
    ax.set_ylabel("|Δ speed|")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, max(mag) * 1.5 + 1e-6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, mag):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, ang, w, yerr=astd, capsize=5, color=colors, alpha=0.85)
    ax.set_title("Mean Angle Error (± 1 std)", fontsize=10)
    ax.set_ylabel("Degrees")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, max(ang) * 1.5 + 1e-6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, ang):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}°", ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    bars = ax.bar(x, times, w, color=colors, alpha=0.85)
    ax.set_title("Mean Inference Time per Seed (s)", fontsize=10)
    ax.set_ylabel("Seconds")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, max(times) * 1.35 + 1e-6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--out_dir",     default="magangle_results")
    p.add_argument("--n_seeds",     type=int, default=50)
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--stride",      type=int, default=10)
    p.add_argument("--path_steps",  type=int, default=150)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_ds    = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = test_ds.land_mask.numpy()
    ocean_mask = ~land_mask
    n_test     = len(test_ds)

    ckpt      = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model     = Repaint(in_ch=2, base_ch=64, time_dim=256).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    schedule  = ckpt.get("schedule", "linear")
    noise_std = ckpt.get("noise_std", 1.0)
    diffusion = DDPM(T=args.T, beta_schedule=schedule,
                     noise_std=noise_std, device=device)
    ckpt_name = os.path.basename(args.checkpoint)
    print(f"Loaded: epoch={ckpt.get('epoch','?')}  schedule={schedule}  "
          f"noise_std={noise_std:.5f}")

    seeds = SEEDS[:args.n_seeds]
    n_total = len(seeds)
    print(f"Seeds ({n_total}): {seeds}\n", flush=True)

    all_mag   = {m: [] for m in METHODS}
    all_ang   = {m: [] for m in METHODS}
    all_times = {m: [] for m in METHODS}
    rows = []

    for run_i, seed in enumerate(seeds):
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]
        true_np    = x0_true.numpy()
        path_mask  = biased_walk_path(land_mask, n_steps=args.path_steps, seed=seed)
        x0_obs     = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"[{run_i+1:02d}/{n_total:02d}] seed={seed}  sample={sample_idx}  "
              f"path_cells={int(path_mask.sum())}", flush=True)
        row = [seed, sample_idx]

        def run_method(name, fn, *fn_args, **fn_kwargs):
            t0   = time.perf_counter()
            pred = fn(*fn_args, **fn_kwargs)
            t    = time.perf_counter() - t0
            mag_err, ang_err = magnitude_angle_error(pred, true_np, ocean_mask)
            all_mag[name].append(mag_err)
            all_ang[name].append(ang_err)
            all_times[name].append(t)
            print(f"  {name:<14}: mag_err={mag_err:.4f}  angle_err={ang_err:6.2f}°  t={t:.1f}s",
                  flush=True)
            return mag_err, ang_err, t

        m_, a_, t_ = run_method("RePaint r=10", repaint_infer, model, diffusion,
                                x0_obs, path_mask, land_mask, r=10, device=device,
                                stride=args.stride)
        row += [m_, a_, t_]

        m_, a_, t_ = run_method("RePaint r=1", repaint_infer, model, diffusion,
                                x0_obs, path_mask, land_mask, r=1, device=device,
                                stride=args.stride)
        row += [m_, a_, t_]

        m_, a_, t_ = run_method("DPS z=0.04", dps_infer, model, diffusion,
                                x0_obs, path_mask, land_mask, device=device,
                                stride=args.stride, step_size=0.04)
        row += [m_, a_, t_]

        rows.append(row)

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    header = ["seed", "test_idx"]
    for m in METHODS:
        key = m.replace(" ", "_").replace("=", "")
        header += [f"{key}_mag_err", f"{key}_angle_err_deg", f"{key}_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Magnitude / Angle Error Comparison  —  T={args.T}  stride={args.stride}\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Schedule   : {schedule}   noise_std={noise_std:.5f}\n")
        f.write(f"N seeds    : {n_total}\n")
        f.write(f"Path steps : {args.path_steps}\n\n")
        f.write(f"{'Method':<14} {'Mean MagErr':>12} {'Std MagErr':>11} "
                f"{'Mean AngErr(deg)':>17} {'Std AngErr':>11} {'Mean Time(s)':>13}\n")
        f.write("-" * 85 + "\n")
        for m in METHODS:
            ms, as_, ts = all_mag[m], all_ang[m], all_times[m]
            f.write(f"{m:<14} {np.mean(ms):>12.4f} {np.std(ms):>11.4f} "
                    f"{np.mean(as_):>17.2f} {np.std(as_):>11.2f} {np.mean(ts):>13.2f}\n")

        f.write("\nPer-seed breakdown:\n")
        hdr = f"  {'Seed':>6}  {'idx':>4}"
        for m in METHODS:
            hdr += f"  {m[:10]+'-Mag':>10}  {m[:10]+'-Ang':>10}  {'t':>5}"
        f.write(hdr + "\n")
        f.write("-" * (len(hdr) + 2) + "\n")
        for row in rows:
            seed, idx = row[0], row[1]
            line = f"  {seed:6d}  {idx:4d}"
            for k in range(len(METHODS)):
                mag_v = row[2 + k*3]
                ang_v = row[3 + k*3]
                t_v   = row[4 + k*3]
                line += f"  {mag_v:10.4f}  {ang_v:10.2f}  {t_v:5.1f}"
            f.write(line + "\n")

    print(f"\nCSV saved     : {csv_path}")
    print(f"Summary saved : {summary_path}")

    # ── Bar chart
    chart_path = os.path.join(args.out_dir, "bar_chart.png")
    save_bar_chart(all_mag, all_ang, all_times, args.T, args.stride, n_total, chart_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
