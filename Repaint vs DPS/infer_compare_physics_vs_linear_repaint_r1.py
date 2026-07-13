"""
infer_compare_physics_vs_linear_repaint_r1.py
===============================================
Head-to-head comparison of two Repaint-UNet checkpoints under RePaint r=1
inference (a single UNet call per diffusion step, no resampling loop):

  - Physics-Informed model  (Physics Informed/best_model_physics_ocean.pt)
  - Linear/CurlDiv baseline (Repaint vs DPS/best-model-linear-curldiv-gaussian/
                              checkpoints_linear/best_model_linear.pt)

Same paired-seed setup as infer_compare_physics_vs_linear.py: for N seeds
using biased_walk_path (random-walk observation path), each seed's sample
index and path mask are shared across both models. Reports RMSE, magnitude
error, and angle error per model, averaged over ocean pixels and seeds.

Saves results.csv, summary.txt, and bar_chart.png.

Usage:
    python infer_compare_physics_vs_linear_repaint_r1.py \\
        --pickle /workspace/DiffusionSummer2026/data.pickle \\
        --physics_checkpoint /workspace/DiffusionSummer2026/PhysicsInformed/best_model_physics_ocean.pt \\
        --linear_checkpoint  /workspace/DiffusionSummer2026/RepaintVsDPS/checkpoints_linear/best_model_linear.pt \\
        --out_dir /workspace/DiffusionSummer2026/RepaintVsDPS/results/physics_vs_linear_repaint_r1_s10_50seeds \\
        --n_seeds 50 --T 1000 --stride 10 --path_steps 150
"""

import argparse
import csv
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "best-model-linear-curldiv-gaussian"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "utils"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path

SEEDS = list(range(0, 700, 7))  # up to 100 seeds: 0,7,14,...,693

MODEL_NAMES = ["Physics-Informed", "Linear-CurlDiv"]


# ── inference (RePaint, r=1) ──────────────────────────────────────────────────

@torch.no_grad()
def repaint_infer(model, diffusion, x0_known, path_mask, land_mask,
                   r=1, device="cpu", stride=1):
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


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_errors(pred, truth, ocean_mask, eps=1e-8):
    """Returns (rmse, mag_err, angle_err_deg) over ocean pixels."""
    pu, pv = pred[0][ocean_mask],  pred[1][ocean_mask]
    tu, tv = truth[0][ocean_mask], truth[1][ocean_mask]

    rmse = float(np.sqrt(np.mean((pu - tu)**2 + (pv - tv)**2)))

    pred_mag = np.sqrt(pu**2 + pv**2)
    true_mag = np.sqrt(tu**2 + tv**2)
    mag_err  = float(np.mean(np.abs(pred_mag - true_mag)))

    dot = pu * tu + pv * tv
    cos = np.clip(dot / (pred_mag * true_mag + eps), -1.0, 1.0)
    angle_err_deg = float(np.degrees(np.mean(np.arccos(cos))))

    return rmse, mag_err, angle_err_deg


# ── bar chart ─────────────────────────────────────────────────────────────────

def save_bar_chart(all_rmse, all_mag, all_ang, all_times, T, stride, n_seeds, out_path):
    keys   = list(all_rmse.keys())
    rmse   = [np.mean(all_rmse[k]) for k in keys]
    rstd   = [np.std(all_rmse[k])  for k in keys]
    mag    = [np.mean(all_mag[k])  for k in keys]
    mstd   = [np.std(all_mag[k])   for k in keys]
    ang    = [np.mean(all_ang[k])  for k in keys]
    astd   = [np.std(all_ang[k])   for k in keys]
    times  = [np.mean(all_times[k]) for k in keys]

    colors = ["#4C72B0", "#C44E52"]
    x = np.arange(len(keys))
    w = 0.5

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f"Physics-Informed vs Linear-CurlDiv  —  RePaint r=1  "
                 f"T={T} / stride={stride}  —  {n_seeds} seeds", fontsize=12)

    def _bar(ax, vals, errs, title, ylabel, fmt):
        bars = ax.bar(x, vals, w, yerr=errs, capsize=4, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.5 + 1e-9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=8)

    _bar(axes[0], rmse,  rstd, "Mean RMSE (± 1 std)",         "RMSE",   "{:.4f}")
    _bar(axes[1], mag,   mstd, "Mean Magnitude Error (± 1 std)", "|Δ speed|", "{:.4f}")
    _bar(axes[2], ang,   astd, "Mean Angle Error (± 1 std)",  "Degrees","{:.1f}°")
    _bar(axes[3], times, None, "Mean Inference Time / seed",  "Seconds","{:.1f}s")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved: {out_path}")


# ── model loading ──────────────────────────────────────────────────────────────

def load_repaint(checkpoint_path, device):
    ckpt      = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  64)
    time_dim  = ckpt_args.get("time_dim", 256)
    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    schedule  = ckpt.get("schedule", "linear")
    noise_std = ckpt.get("noise_std", 1.0)
    return model, schedule, noise_std


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",              required=True)
    p.add_argument("--physics_checkpoint",  required=True)
    p.add_argument("--linear_checkpoint",   required=True)
    p.add_argument("--out_dir",             default="compare_physics_vs_linear_repaint_r1")
    p.add_argument("--n_seeds",             type=int, default=50)
    p.add_argument("--T",                   type=int, default=1000)
    p.add_argument("--stride",              type=int, default=10)
    p.add_argument("--path_steps",          type=int, default=150)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_ds    = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = test_ds.land_mask.numpy()
    ocean_mask = ~land_mask
    n_test     = len(test_ds)

    physics_model, physics_schedule, physics_noise_std = load_repaint(args.physics_checkpoint, device)
    linear_model,  linear_schedule,  linear_noise_std  = load_repaint(args.linear_checkpoint, device)
    print(f"Physics-Informed: schedule={physics_schedule}  noise_std={physics_noise_std:.5f}")
    print(f"Linear-CurlDiv  : schedule={linear_schedule}  noise_std={linear_noise_std:.5f}")

    physics_diffusion = DDPM(T=args.T, beta_schedule=physics_schedule,
                              noise_std=physics_noise_std, device=device)
    linear_diffusion  = DDPM(T=args.T, beta_schedule=linear_schedule,
                              noise_std=linear_noise_std, device=device)

    MODELS = {
        "Physics-Informed": (physics_model, physics_diffusion),
        "Linear-CurlDiv":   (linear_model,  linear_diffusion),
    }

    all_rmse  = {k: [] for k in MODEL_NAMES}
    all_mag   = {k: [] for k in MODEL_NAMES}
    all_ang   = {k: [] for k in MODEL_NAMES}
    all_times = {k: [] for k in MODEL_NAMES}
    rows = []

    seeds = SEEDS[:args.n_seeds]
    n_total = len(seeds)
    print(f"Seeds ({n_total}): {seeds}\n", flush=True)

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

        for model_name in MODEL_NAMES:
            model, diffusion = MODELS[model_name]

            t0   = time.perf_counter()
            pred = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask,
                                  r=1, device=device, stride=args.stride)
            t    = time.perf_counter() - t0
            rmse, mag_err, ang_err = compute_errors(pred, true_np, ocean_mask)

            all_rmse[model_name].append(rmse)
            all_mag[model_name].append(mag_err)
            all_ang[model_name].append(ang_err)
            all_times[model_name].append(t)
            row += [rmse, mag_err, ang_err, t]
            print(f"  {model_name:<18}: rmse={rmse:.4f}  mag_err={mag_err:.4f}  "
                  f"angle_err={ang_err:6.2f}°  t={t:.1f}s", flush=True)

        rows.append(row)

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    header = ["seed", "test_idx"]
    for model_name in MODEL_NAMES:
        key = model_name.replace(" ", "_").replace("-", "_")
        header += [f"{key}_rmse", f"{key}_mag_err", f"{key}_angle_err_deg", f"{key}_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Physics-Informed vs Linear-CurlDiv  —  RePaint r=1  "
                f"T={args.T}  stride={args.stride}\n")
        f.write(f"Physics checkpoint : {args.physics_checkpoint}\n")
        f.write(f"Linear  checkpoint : {args.linear_checkpoint}\n")
        f.write(f"N seeds            : {n_total}\n")
        f.write(f"Path steps         : {args.path_steps}\n\n")
        f.write(f"{'Model':<20} {'Mean RMSE':>10} {'Std RMSE':>9} "
                f"{'Mean MagErr':>12} {'Mean AngErr(deg)':>17} {'Mean Time(s)':>13}\n")
        f.write("-" * 88 + "\n")
        for k in MODEL_NAMES:
            rs, ms, as_, ts = all_rmse[k], all_mag[k], all_ang[k], all_times[k]
            f.write(f"{k:<20} {np.mean(rs):>10.4f} {np.std(rs):>9.4f} "
                    f"{np.mean(ms):>12.4f} {np.mean(as_):>17.2f} {np.mean(ts):>13.2f}\n")

    print(f"\nCSV saved     : {csv_path}")
    print(f"Summary saved : {summary_path}")

    # ── Bar chart
    chart_path = os.path.join(args.out_dir, "bar_chart.png")
    save_bar_chart(all_rmse, all_mag, all_ang, all_times, args.T, args.stride,
                    n_total, chart_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
