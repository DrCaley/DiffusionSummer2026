"""
run_12methods.py
=================
Runs 12 inpainting method configurations on n_seeds samples and saves
summary.txt, results.csv, and bar_chart.png.

Methods (excludes RD r=10 variants):
  1.  RePaint r=10
  2.  RePaint r=1
  3.  DPS z=0.5
  4.  DPS z=0.04
  5.  RD r=1 z=0.5
  6.  RD r=1 z=0.04
  7.  DDIM eta=0
  8.  DDIM eta=1
  9.  DDIM+DPS e0 z=0.5
  10. DDIM+DPS e0 z=0.04
  11. DDIM+DPS e1 z=0.5
  12. DDIM+DPS e1 z=0.04

Usage:
    python run_12methods.py --pickle data.pickle --checkpoint ckpt.pt \\
        --T 1000 --stride 10 --n_seeds 100 --out_dir results/12m_T1000_s10
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

SEEDS = list(range(0, 700, 7))   # 100 seeds: 0, 7, 14, ..., 693

METHODS = [
    "RePaint r=10",
    "RePaint r=1",
    "DPS z=0.5",
    "DPS z=0.04",
    "RD r=1 z=0.5",
    "RD r=1 z=0.04",
    "DDIM eta=0",
    "DDIM eta=1",
    "DDIM+DPS e0 z=0.5",
    "DDIM+DPS e0 z=0.04",
    "DDIM+DPS e1 z=0.5",
    "DDIM+DPS e1 z=0.04",
]


# ── RePaint ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def repaint_infer(model, diffusion, x0_known, path_mask, land_mask,
                  r=10, device="cpu", stride=1):
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
        for j in range(r):
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


# ── DPS ───────────────────────────────────────────────────────────────────────

def dps_infer(model, diffusion, x0_known, path_mask, land_mask,
              device="cpu", stride=1, step_size=0.5):
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
        ab     = diffusion.alpha_bar[t_int]
        x0_hat = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        residual = known_t * (x0_hat - x0_known_t)
        norm_sq  = (residual ** 2).sum()
        grad     = torch.autograd.grad(norm_sq, xt_in)[0]

        with torch.no_grad():
            xt_next = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)
            norm    = norm_sq.sqrt().item() + 1e-8
            xt_next = xt_next - (step_size / norm) * grad.detach()
            xt      = xt_next * ocean_t

    return xt.squeeze(0).cpu().numpy()


# ── RePaint + DPS r=1 ─────────────────────────────────────────────────────────

def repaint_dps_infer(model, diffusion, x0_known, path_mask, land_mask,
                      device="cpu", stride=1, step_size=0.5):
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
        ab     = diffusion.alpha_bar[t_int]
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


# ── DDIM ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_infer(model, diffusion, x0_known, path_mask, land_mask,
               device="cpu", stride=1, eta=0.0):
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

        x0_hat = (xt - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        if t_int == 0:
            xt_known_noisy, _ = diffusion.q_sample(x0_known, t_prev_t)
            xt = known_t * xt_known_noisy + (1.0 - known_t) * x0_hat
            xt = xt * ocean_t
            break

        ratio    = (1.0 - ab_prev) / (1.0 - ab)
        coeff    = (1.0 - ab / ab_prev).clamp(min=0.0)
        sigma_t  = eta * (ratio * coeff).sqrt()
        eps_coeff = (1.0 - ab_prev - sigma_t ** 2).clamp(min=0.0).sqrt()

        x_unknown = ab_prev.sqrt() * x0_hat + eps_coeff * eps_hat
        if eta > 0.0:
            x_unknown = x_unknown + sigma_t * torch.randn_like(xt) * diffusion.noise_std

        xt_known_noisy, _ = diffusion.q_sample(x0_known, t_prev_t)
        xt = known_t * xt_known_noisy + (1.0 - known_t) * x_unknown
        xt = xt * ocean_t

    return xt.squeeze(0).cpu().numpy()


# ── DDIM + DPS ────────────────────────────────────────────────────────────────

def ddim_dps_infer(model, diffusion, x0_known, path_mask, land_mask,
                   device="cpu", stride=1, eta=0.0, step_size=0.5):
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

        t_vec    = torch.full((1,), t_int,      device=device, dtype=torch.long)
        t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)

        xt_in = xt.detach().requires_grad_(True)
        eps_hat = model(xt_in, t_vec)

        ab      = diffusion.alpha_bar[t_int]
        ab_prev = (diffusion.alpha_bar[t_prev_int]
                   if t_prev_int > 0 else torch.tensor(1.0, device=device))

        x0_hat = (xt_in - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        residual = known_t * (x0_hat - x0_known_t)
        norm_sq  = (residual ** 2).sum()
        grad     = torch.autograd.grad(norm_sq, xt_in)[0]

        with torch.no_grad():
            if t_int == 0:
                x_unknown = x0_hat.detach()
            else:
                ratio    = (1.0 - ab_prev) / (1.0 - ab)
                coeff    = (1.0 - ab / ab_prev).clamp(min=0.0)
                sigma_t  = eta * (ratio * coeff).sqrt()
                eps_coeff = (1.0 - ab_prev - sigma_t ** 2).clamp(min=0.0).sqrt()

                x_unknown = (ab_prev.sqrt() * x0_hat.detach()
                             + eps_coeff * eps_hat.detach())
                if eta > 0.0:
                    x_unknown = (x_unknown
                                 + sigma_t * torch.randn_like(xt) * diffusion.noise_std)

            norm = norm_sq.sqrt().item() + 1e-8
            x_unknown = x_unknown - (step_size / norm) * grad.detach()

            xt_known_noisy, _ = diffusion.q_sample(x0_known_t, t_prev_t)
            xt = known_t * xt_known_noisy + (1.0 - known_t) * x_unknown
            xt = xt * ocean_t

    return xt.squeeze(0).cpu().numpy()


# ── Bar chart ─────────────────────────────────────────────────────────────────

def save_bar_chart(all_rmse, all_times, T, stride, n_seeds, out_path):
    methods = list(all_rmse.keys())
    rmse    = [np.mean(all_rmse[m]) for m in methods]
    std     = [np.std(all_rmse[m])  for m in methods]
    times   = [np.mean(all_times[m]) for m in methods]

    colors = ["#4C72B0","#55A868","#C44E52","#E08B3A",
              "#8172B2","#937860","#DA8BC3","#8C8C8C",
              "#1F77B4","#FF7F0E","#2CA02C","#D62728"]
    x = np.arange(len(methods))
    w = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle(f"T={T}/stride={stride}  —  12 Methods, {n_seeds} seeds", fontsize=12)

    ax = axes[0]
    bars = ax.bar(x, rmse, w, yerr=std, capsize=4, color=colors, alpha=0.85)
    ax.set_title("Mean RMSE (± 1 std)", fontsize=10)
    ax.set_ylabel("RMSE")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, max(rmse) * 1.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=6.5)

    ax = axes[1]
    bars = ax.bar(x, times, w, color=colors, alpha=0.85)
    ax.set_title("Mean Inference Time per Seed (s)", fontsize=10)
    ax.set_ylabel("Seconds")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, max(times) * 1.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=6.5)

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
                                    f"12m_T{args.T}_s{stride}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"T          : {args.T}  stride={stride}  "
          f"({len(range(0, args.T, stride))} diffusion steps)")
    print(f"Methods    : {len(METHODS)}")
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

        print(f"\n[{run_i+1:03d}/{n_total:03d}]  seed={seed}  test_idx={sample_idx}",
              flush=True)
        row = [seed, sample_idx]

        def run_m(name, fn, *fn_args, **fn_kwargs):
            t0   = time.perf_counter()
            pred = fn(*fn_args, **fn_kwargs)
            t    = time.perf_counter() - t0
            rmse = float(np.sqrt(np.mean(
                (pred[:, ocean_mask] - true_np[:, ocean_mask]) ** 2)))
            all_rmse[name].append(rmse)
            all_times[name].append(t)
            print(f"  {name:<24}: RMSE={rmse:.4f}  t={t:.1f}s", flush=True)
            return rmse, t

        r, t = run_m("RePaint r=10",       repaint_infer,    model, diffusion, x0_obs, path_mask, land_mask_np, r=10, device=device, stride=stride)
        row += [r, t]
        r, t = run_m("RePaint r=1",        repaint_infer,    model, diffusion, x0_obs, path_mask, land_mask_np, r=1,  device=device, stride=stride)
        row += [r, t]
        r, t = run_m("DPS z=0.5",          dps_infer,        model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, step_size=0.5)
        row += [r, t]
        r, t = run_m("DPS z=0.04",         dps_infer,        model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, step_size=0.04)
        row += [r, t]
        r, t = run_m("RD r=1 z=0.5",       repaint_dps_infer, model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, step_size=0.5)
        row += [r, t]
        r, t = run_m("RD r=1 z=0.04",      repaint_dps_infer, model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, step_size=0.04)
        row += [r, t]
        r, t = run_m("DDIM eta=0",          ddim_infer,       model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, eta=0.0)
        row += [r, t]
        r, t = run_m("DDIM eta=1",          ddim_infer,       model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, eta=1.0)
        row += [r, t]
        r, t = run_m("DDIM+DPS e0 z=0.5",  ddim_dps_infer,   model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, eta=0.0, step_size=0.5)
        row += [r, t]
        r, t = run_m("DDIM+DPS e0 z=0.04", ddim_dps_infer,   model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, eta=0.0, step_size=0.04)
        row += [r, t]
        r, t = run_m("DDIM+DPS e1 z=0.5",  ddim_dps_infer,   model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, eta=1.0, step_size=0.5)
        row += [r, t]
        r, t = run_m("DDIM+DPS e1 z=0.04", ddim_dps_infer,   model, diffusion, x0_obs, path_mask, land_mask_np, device=device, stride=stride, eta=1.0, step_size=0.04)
        row += [r, t]

        rows.append(row)

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    header = ["seed", "test_idx"]
    for m in METHODS:
        key = m.replace(" ", "_").replace("=","").replace("+","_")
        header += [f"{key}_rmse", f"{key}_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\nCSV saved     : {csv_path}")

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"12 Methods  —  T={args.T}  stride={stride}\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Schedule   : {schedule}   noise_std={noise_std:.5f}\n")
        f.write(f"N seeds    : {n_total}\n\n")
        f.write(f"{'Method':<24} {'Mean RMSE':>10} {'Std RMSE':>10} "
                f"{'Min':>8} {'Max':>8} {'Mean Time(s)':>13}\n")
        f.write("-" * 75 + "\n")
        for m in METHODS:
            rs = all_rmse[m]; ts = all_times[m]
            f.write(f"{m:<24} {np.mean(rs):>10.4f} {np.std(rs):>10.4f} "
                    f"{np.min(rs):>8.4f} {np.max(rs):>8.4f} {np.mean(ts):>13.2f}\n")
        f.write("\nPer-seed breakdown:\n")
        hdr = f"  {'Seed':>6}  {'idx':>4}"
        for m in METHODS:
            hdr += f"  {m[:10]:>10}  {'t':>5}"
        f.write(hdr + "\n")
        f.write("-" * (len(hdr) + 2) + "\n")
        for row in rows:
            line = f"  {row[0]:6d}  {row[1]:4d}"
            for k in range(len(METHODS)):
                line += f"  {row[2+k*2]:10.4f}  {row[3+k*2]:5.1f}"
            f.write(line + "\n")
    print(f"Summary saved : {summary_path}")

    # ── Bar chart
    chart_path = os.path.join(args.out_dir, "bar_chart.png")
    save_bar_chart(all_rmse, all_times, args.T, stride, n_total, chart_path)


if __name__ == "__main__":
    main()
