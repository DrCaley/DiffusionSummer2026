"""
infer_compare_unet_fno.py
==========================
Head-to-head comparison of the UNet epsilon-predictor (repaint_model.Repaint)
against the noise-conditioned FNO (Neural Operator/model_fno_ddpm.FNO2dDDPM),
across the same 3 inference methods used in infer_batch_3methods.py /
infer_batch_magangle.py:
  - RePaint r=10
  - RePaint r=1
  - DPS z=0.04

For N seeds using biased_walk_path (random-walk observation path), reports
three metrics per (model, method) combination, averaged over ocean pixels
and seeds:
  - RMSE
  - Magnitude error : |‖pred‖ - ‖truth‖|
  - Angle error      : arccos( (pred·truth) / (‖pred‖‖truth‖ + eps) )  [degrees]

Saves results.csv, summary.txt, and bar_chart.png (6 bars: 2 models x 3 methods).

Usage:
    python infer_compare_unet_fno.py \\
        --pickle /root/ocean_ddpm/data_local.pickle \\
        --unet_checkpoint /root/Repaint_vs_DPS/checkpoints_linear/best_model_linear.pt \\
        --fno_checkpoint  /root/NeuralOperator/checkpoints_fno_ddpm/best_fno_ddpm_linear.pt \\
        --out_dir /root/Repaint_vs_DPS/results/compare_unet_fno_T1000_s10_50seeds \\
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
MODEL_NAMES = ["UNet", "FNO"]


def _find_neural_operator_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "Neural Operator"),
        os.path.join(script_dir, "..", "NeuralOperator"),
        "/root/NeuralOperator",
    ]
    for d in candidates:
        d = os.path.abspath(d)
        if os.path.isfile(os.path.join(d, "model_fno_ddpm.py")):
            return d
    raise RuntimeError("Cannot find model_fno_ddpm.py — tried: " + str(candidates))


sys.path.insert(0, _find_neural_operator_dir())
from model_fno_ddpm import FNO2dDDPM


# ── inference (model-agnostic: works for any model(x, t) -> eps) ─────────────

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
    keys   = list(all_rmse.keys())   # "UNet / RePaint r=10", etc.
    rmse   = [np.mean(all_rmse[k]) for k in keys]
    rstd   = [np.std(all_rmse[k])  for k in keys]
    mag    = [np.mean(all_mag[k])  for k in keys]
    mstd   = [np.std(all_mag[k])   for k in keys]
    ang    = [np.mean(all_ang[k])  for k in keys]
    astd   = [np.std(all_ang[k])   for k in keys]
    times  = [np.mean(all_times[k]) for k in keys]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]
    x = np.arange(len(keys))
    w = 0.6

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.suptitle(f"UNet vs FNO-DDPM  —  T={T} / stride={stride}  —  {n_seeds} seeds", fontsize=12)

    def _bar(ax, vals, errs, title, ylabel, fmt):
        bars = ax.bar(x, vals, w, yerr=errs, capsize=4, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=7, rotation=20, ha="right")
        ax.set_ylim(0, max(vals) * 1.5 + 1e-9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=7)

    _bar(axes[0], rmse,  rstd, "Mean RMSE (± 1 std)",         "RMSE",   "{:.4f}")
    _bar(axes[1], mag,   mstd, "Mean Magnitude Error (± 1 std)", "|Δ speed|", "{:.4f}")
    _bar(axes[2], ang,   astd, "Mean Angle Error (± 1 std)",  "Degrees","{:.1f}°")
    _bar(axes[3], times, None, "Mean Inference Time / seed",  "Seconds","{:.1f}s")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved: {out_path}")


# ── model loading ──────────────────────────────────────────────────────────────

def load_unet(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Repaint(in_ch=2, base_ch=64, time_dim=256).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    schedule  = ckpt.get("schedule", "linear")
    noise_std = ckpt.get("noise_std", 1.0)
    return model, schedule, noise_std


def load_fno(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = FNO2dDDPM(
        in_ch=2,
        width=ckpt_args.get("width", 64),
        modes1=ckpt_args.get("modes1", 16),
        modes2=ckpt_args.get("modes2", 16),
        time_dim=ckpt_args.get("time_dim", 256),
        n_layers=ckpt_args.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    schedule  = ckpt.get("schedule", "linear")
    noise_std = ckpt.get("noise_std", 1.0)
    return model, schedule, noise_std


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",          required=True)
    p.add_argument("--unet_checkpoint", required=True)
    p.add_argument("--fno_checkpoint",  required=True)
    p.add_argument("--out_dir",         default="compare_unet_fno")
    p.add_argument("--n_seeds",         type=int, default=50)
    p.add_argument("--T",               type=int, default=1000)
    p.add_argument("--stride",          type=int, default=10)
    p.add_argument("--path_steps",      type=int, default=150)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_ds    = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = test_ds.land_mask.numpy()
    ocean_mask = ~land_mask
    n_test     = len(test_ds)

    unet_model, unet_schedule, unet_noise_std = load_unet(args.unet_checkpoint, device)
    fno_model,  fno_schedule,  fno_noise_std  = load_fno(args.fno_checkpoint, device)
    print(f"UNet: schedule={unet_schedule}  noise_std={unet_noise_std:.5f}")
    print(f"FNO : schedule={fno_schedule}  noise_std={fno_noise_std:.5f}")

    unet_diffusion = DDPM(T=args.T, beta_schedule=unet_schedule,
                          noise_std=unet_noise_std, device=device)
    fno_diffusion  = DDPM(T=args.T, beta_schedule=fno_schedule,
                          noise_std=fno_noise_std, device=device)

    MODELS = {
        "UNet": (unet_model, unet_diffusion),
        "FNO":  (fno_model,  fno_diffusion),
    }

    keys = [f"{m} / {meth}" for m in MODEL_NAMES for meth in METHODS]
    all_rmse  = {k: [] for k in keys}
    all_mag   = {k: [] for k in keys}
    all_ang   = {k: [] for k in keys}
    all_times = {k: [] for k in keys}
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

            def run(method_name, fn, *fn_args, **fn_kwargs):
                key  = f"{model_name} / {method_name}"
                t0   = time.perf_counter()
                pred = fn(*fn_args, **fn_kwargs)
                t    = time.perf_counter() - t0
                rmse, mag_err, ang_err = compute_errors(pred, true_np, ocean_mask)
                all_rmse[key].append(rmse)
                all_mag[key].append(mag_err)
                all_ang[key].append(ang_err)
                all_times[key].append(t)
                print(f"  {key:<20}: rmse={rmse:.4f}  mag_err={mag_err:.4f}  "
                      f"angle_err={ang_err:6.2f}°  t={t:.1f}s", flush=True)
                return rmse, mag_err, ang_err, t

            r_, m_, a_, t_ = run("RePaint r=10", repaint_infer, model, diffusion,
                                 x0_obs, path_mask, land_mask, r=10, device=device,
                                 stride=args.stride)
            row += [r_, m_, a_, t_]

            r_, m_, a_, t_ = run("RePaint r=1", repaint_infer, model, diffusion,
                                 x0_obs, path_mask, land_mask, r=1, device=device,
                                 stride=args.stride)
            row += [r_, m_, a_, t_]

            r_, m_, a_, t_ = run("DPS z=0.04", dps_infer, model, diffusion,
                                 x0_obs, path_mask, land_mask, device=device,
                                 stride=args.stride, step_size=0.04)
            row += [r_, m_, a_, t_]

        rows.append(row)

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    header = ["seed", "test_idx"]
    for model_name in MODEL_NAMES:
        for meth in METHODS:
            key = f"{model_name}_{meth}".replace(" ", "_").replace("=", "")
            header += [f"{key}_rmse", f"{key}_mag_err", f"{key}_angle_err_deg", f"{key}_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"UNet vs FNO-DDPM Comparison  —  T={args.T}  stride={args.stride}\n")
        f.write(f"UNet checkpoint : {args.unet_checkpoint}\n")
        f.write(f"FNO  checkpoint : {args.fno_checkpoint}\n")
        f.write(f"N seeds         : {n_total}\n")
        f.write(f"Path steps      : {args.path_steps}\n\n")
        f.write(f"{'Model / Method':<22} {'Mean RMSE':>10} {'Std RMSE':>9} "
                f"{'Mean MagErr':>12} {'Mean AngErr(deg)':>17} {'Mean Time(s)':>13}\n")
        f.write("-" * 90 + "\n")
        for k in keys:
            rs, ms, as_, ts = all_rmse[k], all_mag[k], all_ang[k], all_times[k]
            f.write(f"{k:<22} {np.mean(rs):>10.4f} {np.std(rs):>9.4f} "
                    f"{np.mean(ms):>12.4f} {np.mean(as_):>17.2f} {np.mean(ts):>13.2f}\n")

    print(f"\nCSV saved     : {csv_path}")
    print(f"Summary saved : {summary_path}")

    # ── Bar chart
    chart_path = os.path.join(args.out_dir, "bar_chart.png")
    save_bar_chart(all_rmse, all_mag, all_ang, all_times, args.T, args.stride, n_total, chart_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
