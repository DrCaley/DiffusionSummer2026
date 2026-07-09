"""
run_ddim_inpaint.py
====================
Implements DDIM inpainting (Song et al. 2021, arXiv:2010.02502).

The DDIM reverse step is:
    x_{t-1} = sqrt(ab_prev) * x0_hat
            + sqrt(1 - ab_prev - sigma_t^2) * eps_hat
            + sigma_t * eps

where  x0_hat    = (x_t - sqrt(1-ab_t) * eps_hat) / sqrt(ab_t)
       sigma_t   = eta * sqrt((1-ab_prev)/(1-ab_t)) * sqrt(1 - ab_t/ab_prev)
       eta = 0   -> fully deterministic (DDIM)
       eta = 1   -> maximum stochasticity (DDPM-like variance)

For inpainting, at every step we paste the noise-corrupted ground truth into
the known-pixel region, identical to RePaint r=1 — but using the DDIM reverse
step instead of the DDPM posterior.

Runs two configurations:
  DDIM eta=0  (deterministic)
  DDIM eta=1  (stochastic)

Usage:
    python run_ddim_inpaint.py \\
        --pickle   /root/ocean_ddpm/data.pickle \\
        --checkpoint /root/Repaint_vs_DPS/checkpoints_linear/best_model_linear.pt \\
        --T 1000 --stride 10 --n_seeds 20 --out_dir results/ddim_T1000_s10
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

SEEDS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63,
         70, 77, 84, 91, 98, 105, 112, 119, 126, 133]

METHODS = ["DDIM eta=0", "DDIM eta=1"]


# ── DDIM inpainting ───────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_infer(model, diffusion, x0_known, path_mask, land_mask,
               device="cpu", stride=1, eta=0.0):
    """
    DDIM inpainting (arXiv:2010.02502, eq. 12 + inpainting mask merge).

    At each reverse step t -> t_prev:
      1. Predict eps_hat from model(x_t, t)
      2. Recover x0_hat = (x_t - sqrt(1-ab_t)*eps_hat) / sqrt(ab_t)
      3. DDIM step: x_t_prev_unknown = sqrt(ab_prev)*x0_hat
                                      + sqrt(1-ab_prev-sigma^2)*eps_hat
                                      + sigma*noise
      4. Sample q(x_{t_prev} | x_0) for known pixels
      5. Merge: x_{t_prev} = mask * x_known_noisy + (1-mask) * x_unknown
    """
    H, W     = x0_known.shape[1:]
    x0_known = x0_known.unsqueeze(0).to(device)
    known_t  = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t   = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t  = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        t_vec    = torch.full((1,), t_int,      device=device, dtype=torch.long)
        t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)

        eps_hat = model(xt, t_vec)

        ab      = diffusion.alpha_bar[t_int]
        ab_prev = (diffusion.alpha_bar[t_prev_int]
                   if t_prev_int > 0 else torch.tensor(1.0, device=device))

        # Predict x0
        x0_hat = (xt - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        if t_int == 0:
            # Final step: just return x0_hat merged
            xt_known_noisy, _ = diffusion.q_sample(x0_known, t_prev_t)
            xt = known_t * xt_known_noisy + (1.0 - known_t) * x0_hat
            xt = xt * ocean_t
            break

        # DDIM sigma (eq. 16 in arXiv:2010.02502)
        #   sigma_t = eta * sqrt((1-ab_prev)/(1-ab)) * sqrt(1 - ab/ab_prev)
        ratio    = (1.0 - ab_prev) / (1.0 - ab)
        coeff    = (1.0 - ab / ab_prev).clamp(min=0.0)
        sigma_t  = eta * (ratio * coeff).sqrt()

        # Direction pointing toward x_t (eps coefficient)
        eps_coeff = (1.0 - ab_prev - sigma_t ** 2).clamp(min=0.0).sqrt()

        # DDIM reverse step for unknown region
        x_unknown = ab_prev.sqrt() * x0_hat + eps_coeff * eps_hat
        if eta > 0.0:
            x_unknown = x_unknown + sigma_t * torch.randn_like(xt) * diffusion.noise_std

        # Paste known pixels at level t_prev
        xt_known_noisy, _ = diffusion.q_sample(x0_known, t_prev_t)
        xt = known_t * xt_known_noisy + (1.0 - known_t) * x_unknown
        xt = xt * ocean_t

    return xt.squeeze(0).cpu().numpy()


# ── Bar chart ─────────────────────────────────────────────────────────────────

def save_bar_chart(all_rmse, all_times, T, stride, n_seeds, out_path):
    methods = list(all_rmse.keys())
    rmse    = [np.mean(all_rmse[m]) for m in methods]
    std     = [np.std(all_rmse[m])  for m in methods]
    times   = [np.mean(all_times[m]) for m in methods]

    colors = ["#4C72B0", "#C44E52"]
    x = np.arange(len(methods))
    w = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle(f"DDIM Inpainting  —  T={T}/stride={stride}, {n_seeds} seeds",
                 fontsize=11)

    ax = axes[0]
    bars = ax.bar(x, rmse, w, yerr=std, capsize=5, color=colors, alpha=0.85)
    ax.set_title("Mean RMSE (± 1 std)")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0, max(rmse) * 1.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bars = ax.bar(x, times, w, color=colors, alpha=0.85)
    ax.set_title("Mean Inference Time per Seed (s)")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0, max(times) * 1.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

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

    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.out_dir = os.path.join(script_dir, "results",
                                    f"ddim_T{args.T}_s{stride}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"T          : {args.T}  stride={stride}  "
          f"({len(range(0, args.T, stride))} diffusion steps)")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output dir : {args.out_dir}", flush=True)

    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    train_ds     = OceanCurrentDataset(args.pickle, split=0)
    land_mask_np = test_ds.land_mask.numpy()
    ocean_mask   = ~land_mask_np
    n_test       = len(test_ds)

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
          f"schedule={schedule}  noise_std={noise_std:.5f}\n", flush=True)

    seeds   = SEEDS[:args.n_seeds] if args.n_seeds is not None else SEEDS
    n_total = len(seeds)

    all_rmse  = {m: [] for m in METHODS}
    all_times = {m: [] for m in METHODS}
    rows = []

    for run_i, seed in enumerate(seeds):
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]
        true_np    = x0_true.numpy()
        path_mask  = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"\n[{run_i+1:02d}/{n_total:02d}]  seed={seed}  test_idx={sample_idx}",
              flush=True)
        row = [seed, sample_idx]

        def run_method(name, fn, **kwargs):
            t0   = time.perf_counter()
            pred = fn(model, diffusion, x0_obs, path_mask, land_mask_np,
                      device=device, stride=stride, **kwargs)
            t    = time.perf_counter() - t0
            rmse = float(np.sqrt(np.mean(
                (pred[:, ocean_mask] - true_np[:, ocean_mask]) ** 2)))
            all_rmse[name].append(rmse)
            all_times[name].append(t)
            print(f"  {name:<16}: RMSE={rmse:.4f}  t={t:.1f}s", flush=True)
            return rmse, t

        r, t = run_method("DDIM eta=0", ddim_infer, eta=0.0)
        row += [r, t]
        r, t = run_method("DDIM eta=1", ddim_infer, eta=1.0)
        row += [r, t]

        rows.append(row)

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    header = ["seed", "test_idx",
              "DDIM_eta0_rmse", "DDIM_eta0_time",
              "DDIM_eta1_rmse", "DDIM_eta1_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\nCSV saved     : {csv_path}")

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"DDIM Inpainting  —  T={args.T}  stride={stride}\n")
        f.write(f"Reference: arXiv:2010.02502 (Song et al. 2021)\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Schedule   : {schedule}   noise_std={noise_std:.5f}\n")
        f.write(f"N seeds    : {n_total}\n\n")
        f.write(f"{'Method':<16} {'Mean RMSE':>10} {'Std RMSE':>10} "
                f"{'Min':>8} {'Max':>8} {'Mean Time(s)':>13}\n")
        f.write("-" * 65 + "\n")
        for m in METHODS:
            rs = all_rmse[m]; ts = all_times[m]
            f.write(f"{m:<16} {np.mean(rs):>10.4f} {np.std(rs):>10.4f} "
                    f"{np.min(rs):>8.4f} {np.max(rs):>8.4f} "
                    f"{np.mean(ts):>13.2f}\n")
        f.write("\nPer-seed breakdown:\n")
        f.write(f"  {'Seed':>6}  {'idx':>4}  "
                f"{'DDIM eta=0':>10}  {'t':>5}  "
                f"{'DDIM eta=1':>10}  {'t':>5}\n")
        f.write("-" * 55 + "\n")
        for row in rows:
            f.write(f"  {row[0]:6d}  {row[1]:4d}  "
                    f"{row[2]:10.4f}  {row[3]:5.1f}  "
                    f"{row[4]:10.4f}  {row[5]:5.1f}\n")
    print(f"Summary saved : {summary_path}")

    # ── Bar chart
    chart_path = os.path.join(args.out_dir, "bar_chart.png")
    save_bar_chart(all_rmse, all_times, args.T, stride, n_total, chart_path)


if __name__ == "__main__":
    main()
