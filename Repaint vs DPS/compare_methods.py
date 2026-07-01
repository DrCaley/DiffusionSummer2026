"""
compare_methods.py
==================
Compares three inpainting methods on 20 non-consecutive test samples:

  1. RePaint  (r=10 resampling steps per diffusion step)
  2. RePaint1 (r=1  — single UNet call per diffusion step, no resampling loop)
  3. DPS      (Diffusion Posterior Sampling, Song et al. 2023)

For each method × sample:
  - RMSE on ocean cells (u,v)
  - Wall-clock time

For seed 0 (first sample) also saves a side-by-side comparison image.

Summary CSVs and human-readable text reports are written to --out_dir.

Usage:
    python compare_methods.py --pickle data.pickle --checkpoint ckpt.pt --T 100
    python compare_methods.py --pickle data.pickle --checkpoint ckpt.pt --T 1000
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path

# ──────────────────────────────────────────────────────────────────────────────
# Seed list: 20 non-consecutive seeds spread across the test set
# ──────────────────────────────────────────────────────────────────────────────
SEEDS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63,
         70, 77, 84, 91, 98, 105, 112, 119, 126, 133]


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def repaint_infer(model, diffusion, x0_known, path_mask, land_mask,
                  r=10, device="cpu", stride=1):
    """Standard RePaint inference (r resampling iterations per step)."""
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
            xt_unknown   = diffusion.p_sample_step(model, xt, t_int, t_prev_int)
            t_prev_t     = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_known_t, _ = diffusion.q_sample(x0_known, t_prev_t)
            xt_merged    = known_t * xt_known_t + (1.0 - known_t) * xt_unknown
            xt_merged    = xt_merged * ocean_t

            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int) * ocean_t
            else:
                xt = xt_merged

    return xt.squeeze(0).cpu().numpy()


def dps_infer(model, diffusion, x0_known, path_mask, land_mask,
              device="cpu", stride=1, step_size=0.5):
    """
    Diffusion Posterior Sampling (DPS) for inpainting.

    At each reverse step t:
      1. Forward pass (with grad) to get x0_hat from model
      2. Take the standard DDPM reverse step (no grad) to get x_{t-1}
      3. Compute measurement residual: ||y - A(x0_hat)||^2  where A = path mask
      4. Subtract gradient of residual w.r.t. xt, scaled by step_size / ||residual||

    Reference: Chung et al. (2022) "Diffusion Posterior Sampling for General
    Noisy Inverse Problems", Algorithm 1.
    """
    H, W       = x0_known.shape[1:]
    x0_known_t = x0_known.unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t     = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    T  = diffusion.T
    timesteps = list(range(0, T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        # ── (1) Predicted noise and x0_hat, keeping grad for DPS correction
        pred_noise = model(xt_in, t_vec)
        ab = diffusion.alpha_bar[t_int]
        x0_hat = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        # ── (2) Measurement residual on path cells only
        residual = known_t * (x0_hat - x0_known_t)   # (1, 2, H, W)
        norm_sq  = (residual ** 2).sum()

        # ── (3) Gradient of ||residual||^2 w.r.t. xt_in
        grad = torch.autograd.grad(norm_sq, xt_in)[0]

        # ── (4) Standard DDPM reverse step (no grad)
        with torch.no_grad():
            xt_next = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)

            # DPS correction: subtract scaled gradient
            norm = norm_sq.sqrt().item() + 1e-8
            xt_next = xt_next - (step_size / norm) * grad.detach()
            xt_next = xt_next * ocean_t

        xt = xt_next

    return xt.squeeze(0).cpu().numpy()


def repaint_dps_infer(model, diffusion, x0_known, path_mask, land_mask,
                      r=10, device="cpu", stride=1, step_size=0.5):
    """
    Combined RePaint + DPS.

    At each reverse step t, repeated r times:
      1. Enable gradients on xt to compute the DPS correction.
      2. Get pred_noise and x0_hat from model.
      3. Compute DPS residual on known path cells and its gradient.
      4. Take standard DDPM reverse step, apply DPS gradient correction → xt_unknown.
      5. Merge: known cells get q(x_{t-1}|x_0_known); unknown cells keep xt_unknown.
      6. Resample (RePaint) if not the last iteration: re-noise back to t and repeat.
    """
    H, W       = x0_known.shape[1:]
    x0_known_t = x0_known.unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t     = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    T  = diffusion.T
    timesteps = list(range(0, T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        for j in range(r):
            # ── DPS gradient on xt
            xt_in = xt.detach().requires_grad_(True)
            t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

            pred_noise = model(xt_in, t_vec)
            ab = diffusion.alpha_bar[t_int]
            x0_hat = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
            x0_hat = x0_hat.clamp(-1.5, 1.5)

            residual = known_t * (x0_hat - x0_known_t)
            norm_sq  = (residual ** 2).sum()
            grad     = torch.autograd.grad(norm_sq, xt_in)[0]

            # ── DDPM reverse step + DPS correction → candidate for unknown cells
            with torch.no_grad():
                xt_unknown = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)
                norm = norm_sq.sqrt().item() + 1e-8
                xt_unknown = xt_unknown - (step_size / norm) * grad.detach()

                # ── RePaint merge: replace known cells with noisy ground truth
                t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
                xt_known_noisy, _ = diffusion.q_sample(x0_known_t, t_prev_t)
                xt_merged = known_t * xt_known_noisy + (1.0 - known_t) * xt_unknown
                xt_merged = xt_merged * ocean_t

                # ── Resample (RePaint) unless last iteration
                if j < r - 1 and t_int > 0:
                    xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int) * ocean_t
                else:
                    xt = xt_merged

    return xt.squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool"):
    H, W = u.shape
    ax.imshow(land_mask, origin="lower",
              cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
              extent=[-0.5, W-0.5, -0.5, H-0.5], aspect="auto", zorder=0)
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    mq     = np.sqrt(uq**2 + vq**2)
    mask   = ~np.isnan(uq) & ~land_mask[::step, ::step]
    if mask.any():
        q = ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
                      cmap=cmap, clim=(0, np.nanpercentile(mq[mask], 98)),
                      scale=12, width=0.003, zorder=2)
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def save_comparison_image(true_np, land_mask_np, path_mask,
                          preds, method_names, rmses, times,
                          T, stride, seed, out_path):
    """
    One figure: Ground Truth | Path | Repaint | Repaint-r1 | DPS
    Row 0: quiver fields
    Row 1: error maps
    """
    n = len(method_names)
    fig, axes = plt.subplots(2, n + 2, figsize=(5 * (n + 2), 10))

    # Ground truth
    plot_field(axes[0, 0], true_np[0].T, true_np[1].T, land_mask_np.T, "Ground Truth")
    axes[1, 0].axis("off")

    # Path
    H, W = land_mask_np.shape
    axes[0, 1].imshow(land_mask_np, origin="lower",
                      cmap=plt.matplotlib.colors.ListedColormap(["white","black"]),
                      extent=[-0.5,W-0.5,-0.5,H-0.5], aspect="auto")
    pd = np.zeros_like(land_mask_np, dtype=float)
    pd[path_mask] = 1.0
    axes[0, 1].imshow(pd, origin="lower", cmap="Reds", alpha=0.8,
                      extent=[-0.5,W-0.5,-0.5,H-0.5], aspect="auto", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Robot Path ({int(path_mask.sum())} cells)", fontsize=10)
    axes[1, 1].axis("off")

    ocean = ~land_mask_np
    for col, (name, pred, rmse, t) in enumerate(zip(method_names, preds, rmses, times), start=2):
        plot_field(axes[0, col], pred[0].T, pred[1].T, land_mask_np.T,
                   f"{name}\nRMSE={rmse:.4f}  t={t:.1f}s")
        err = np.sqrt((pred[0] - true_np[0])**2 + (pred[1] - true_np[1])**2)
        err[land_mask_np] = np.nan
        err_m = np.ma.masked_where(land_mask_np, err)
        im = axes[1, col].imshow(err_m, origin="lower", cmap="hot_r", aspect="auto",
                                 extent=[-0.5,W-0.5,-0.5,H-0.5])
        axes[1, col].imshow(land_mask_np, origin="lower",
                            cmap=plt.matplotlib.colors.ListedColormap(["none","black"]),
                            extent=[-0.5,W-0.5,-0.5,H-0.5], aspect="auto", zorder=1)
        plt.colorbar(im, ax=axes[1, col], label="|error|", shrink=0.7)
        axes[1, col].set_title(f"Error — {name}", fontsize=10)

    plt.suptitle(f"Method comparison  T={T}  stride={stride}  seed={seed}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Image saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--T",           type=int, default=100)
    p.add_argument("--stride",      type=int, default=None,
                   help="Diffusion stride. Defaults to T//100 (gives ~100 UNet calls).")
    p.add_argument("--path_steps",  type=int, default=150)
    p.add_argument("--dps_step",    type=float, default=0.5,
                   help="DPS gradient step size.")
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--time_dim",    type=int, default=256)
    p.add_argument("--out_dir",     default=None)
    p.add_argument("--n_seeds",     type=int, default=None,
                   help="Limit to first N seeds (default: all 20).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stride = args.stride if args.stride is not None else max(1, args.T // 100)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.out_dir is None:
        args.out_dir = os.path.join(script_dir, "results", f"compare_T{args.T}_s{stride}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"T          : {args.T}  stride={stride}  ({len(range(0,args.T,stride))} diffusion steps)")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output dir : {args.out_dir}")

    # ── Dataset
    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    land_mask_np = test_ds.land_mask.numpy()
    ocean_mask   = ~land_mask_np
    train_ds     = OceanCurrentDataset(args.pickle, split=0)

    # ── Model
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = args.T                               # use requested T, not ckpt's T
    schedule  = ckpt.get("schedule", "linear")

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    noise_std = ckpt.get("noise_std", None)
    if noise_std is None:
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())

    diffusion = DDPM(T=T, beta_schedule=schedule, device=device, noise_std=noise_std)

    print(f"Loaded     : epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss',float('nan')):.5f}"
          f"  schedule={schedule}  noise_std={noise_std:.5f}")

    methods      = ["RePaint-r10", "RePaint-r1",  "DPS", "RePaint+DPS"]
    all_rmse     = {m: [] for m in methods}
    all_times    = {m: [] for m in methods}

    n_test  = len(test_ds)
    seeds   = SEEDS[:args.n_seeds] if args.n_seeds is not None else SEEDS
    n_total = len(seeds)

    for run_i, seed in enumerate(seeds):
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]
        true_np    = x0_true.numpy()
        path_mask  = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"\n[{run_i+1:02d}/{n_total:02d}]  seed={seed}  test_idx={sample_idx}")

        preds = []
        rmses = []
        times = []

        # ── RePaint r=10
        t0 = time.perf_counter()
        pred_r10 = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask_np,
                                 r=10, device=device, stride=stride)
        elapsed_r10 = time.perf_counter() - t0
        rmse_r10 = float(np.sqrt(np.mean((pred_r10[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        all_rmse["RePaint-r10"].append(rmse_r10)
        all_times["RePaint-r10"].append(elapsed_r10)
        preds.append(pred_r10); rmses.append(rmse_r10); times.append(elapsed_r10)
        print(f"  RePaint-r10 : RMSE={rmse_r10:.4f}  t={elapsed_r10:.1f}s")

        # ── RePaint r=1
        t0 = time.perf_counter()
        pred_r1 = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask_np,
                                r=1, device=device, stride=stride)
        elapsed_r1 = time.perf_counter() - t0
        rmse_r1 = float(np.sqrt(np.mean((pred_r1[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        all_rmse["RePaint-r1"].append(rmse_r1)
        all_times["RePaint-r1"].append(elapsed_r1)
        preds.append(pred_r1); rmses.append(rmse_r1); times.append(elapsed_r1)
        print(f"  RePaint-r1  : RMSE={rmse_r1:.4f}  t={elapsed_r1:.1f}s")

        # ── DPS
        t0 = time.perf_counter()
        pred_dps = dps_infer(model, diffusion, x0_obs, path_mask, land_mask_np,
                             device=device, stride=stride, step_size=args.dps_step)
        elapsed_dps = time.perf_counter() - t0
        rmse_dps = float(np.sqrt(np.mean((pred_dps[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        all_rmse["DPS"].append(rmse_dps)
        all_times["DPS"].append(elapsed_dps)
        preds.append(pred_dps); rmses.append(rmse_dps); times.append(elapsed_dps)
        print(f"  DPS         : RMSE={rmse_dps:.4f}  t={elapsed_dps:.1f}s")

        # ── RePaint + DPS
        t0 = time.perf_counter()
        pred_rdps = repaint_dps_infer(model, diffusion, x0_obs, path_mask, land_mask_np,
                                      r=10, device=device, stride=stride,
                                      step_size=args.dps_step)
        elapsed_rdps = time.perf_counter() - t0
        rmse_rdps = float(np.sqrt(np.mean((pred_rdps[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        all_rmse["RePaint+DPS"].append(rmse_rdps)
        all_times["RePaint+DPS"].append(elapsed_rdps)
        preds.append(pred_rdps); rmses.append(rmse_rdps); times.append(elapsed_rdps)
        print(f"  RePaint+DPS : RMSE={rmse_rdps:.4f}  t={elapsed_rdps:.1f}s")

        # ── Comparison image for first seed only
        if run_i == 0:
            img_path = os.path.join(args.out_dir, "comparison_seed0.png")
            save_comparison_image(true_np, land_mask_np, path_mask,
                                  preds, methods, rmses, times,
                                  T, stride, seed, img_path)

    # ── Write CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "test_idx",
                         "repaint_r10_rmse", "repaint_r10_time",
                         "repaint_r1_rmse",  "repaint_r1_time",
                         "dps_rmse",         "dps_time",
                         "repaint_dps_rmse", "repaint_dps_time"])
        n_test = len(test_ds)
        for i, seed in enumerate(seeds):
            writer.writerow([
                seed, seed % n_test,
                f"{all_rmse['RePaint-r10'][i]:.6f}", f"{all_times['RePaint-r10'][i]:.3f}",
                f"{all_rmse['RePaint-r1'][i]:.6f}",  f"{all_times['RePaint-r1'][i]:.3f}",
                f"{all_rmse['DPS'][i]:.6f}",          f"{all_times['DPS'][i]:.3f}",
                f"{all_rmse['RePaint+DPS'][i]:.6f}",  f"{all_times['RePaint+DPS'][i]:.3f}",
            ])
    print(f"\nCSV saved : {csv_path}")

    # ── Write human-readable summary
    txt_path = os.path.join(args.out_dir, "summary.txt")
    lines = []
    lines.append(f"Method Comparison  —  T={args.T}  stride={stride}")
    lines.append(f"Checkpoint : {args.checkpoint}")
    lines.append(f"Schedule   : {schedule}   noise_std={noise_std:.5f}")
    lines.append(f"N seeds    : {len(seeds)}")
    lines.append(f"Path steps : {args.path_steps}")
    lines.append("")
    lines.append(f"{'Method':<14} {'Mean RMSE':>10} {'Std RMSE':>10} {'Min':>8} {'Max':>8} {'Mean Time(s)':>13}")
    lines.append("-" * 68)
    for m in methods:
        rs = all_rmse[m]; ts = all_times[m]
        lines.append(f"{m:<14} {np.mean(rs):>10.4f} {np.std(rs):>10.4f} "
                     f"{np.min(rs):>8.4f} {np.max(rs):>8.4f} {np.mean(ts):>13.2f}")
    lines.append("")
    lines.append("Per-seed breakdown:")
    lines.append(f"{'Seed':>6}  {'idx':>5}  {'RPr10-RMSE':>11}  {'RPr10-t':>8}  "
                 f"{'RPr1-RMSE':>10}  {'RPr1-t':>7}  {'DPS-RMSE':>9}  {'DPS-t':>6}  "
                 f"{'RD-RMSE':>9}  {'RD-t':>6}")
    lines.append("-" * 96)
    for i, seed in enumerate(seeds):
        lines.append(
            f"{seed:>6}  {seed % n_test:>5}  "
            f"{all_rmse['RePaint-r10'][i]:>11.4f}  {all_times['RePaint-r10'][i]:>8.2f}  "
            f"{all_rmse['RePaint-r1'][i]:>10.4f}  {all_times['RePaint-r1'][i]:>7.2f}  "
            f"{all_rmse['DPS'][i]:>9.4f}  {all_times['DPS'][i]:>6.2f}  "
            f"{all_rmse['RePaint+DPS'][i]:>9.4f}  {all_times['RePaint+DPS'][i]:>6.2f}"
        )
    report = "\n".join(lines)
    print("\n" + report)
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print(f"Summary saved: {txt_path}")


if __name__ == "__main__":
    main()
