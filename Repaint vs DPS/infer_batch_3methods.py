"""
infer_batch_3methods.py
=======================
For N seeds using biased_walk_path (400-step random walk), runs:
  - RePaint r=10
  - RePaint r=1
  - DPS z=0.04

Saves one 2-row × 4-col comparison PNG per seed:
  Row 0: Ground Truth | RePaint r=10 | RePaint r=1 | DPS z=0.04  (quiver + speed)
  Row 1: Robot Path   | Error r=10   | Error r=1   | Error DPS   (path / hot_r)

Style matches visualize_infer.py (speed background + black arrows).

Usage:
    python infer_batch_3methods.py \\
        --pickle /root/ocean_ddpm/data.pickle \\
        --checkpoint /root/Repaint_vs_DPS/checkpoints_linear/best_model_linear.pt \\
        --out_dir /root/Repaint_vs_DPS/results/batch_3methods \\
        --n_seeds 10 --T 1000 --stride 10 --path_steps 400
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path


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


# ── plotting ──────────────────────────────────────────────────────────────────

def _ext(H, W):
    return [-0.5, W - 0.5, -0.5, H - 0.5]


def plot_quiver(ax, field, land_mask, vmax_spd, title, step=2):
    H, W  = land_mask.shape
    u, v  = field[0].T, field[1].T   # transpose: X=original-col, Y=original-row
    lm    = land_mask.T
    speed = np.ma.masked_where(lm, np.sqrt(u**2 + v**2))

    im = ax.imshow(speed, origin="lower", cmap="cool", vmin=0, vmax=vmax_spd,
                   extent=_ext(W, H), aspect="auto", zorder=0)
    ax.imshow(lm, origin="lower",
              cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
              extent=_ext(W, H), aspect="auto", zorder=1)

    yq, xq = np.mgrid[0:W:step, 0:H:step]
    mask = ~lm[::step, ::step]
    ax.quiver(xq[mask], yq[mask], u[::step, ::step][mask], v[::step, ::step][mask],
              color="black", scale=12, width=0.003, zorder=2)

    ax.set_xlim(-0.5, H - 0.5); ax.set_ylim(-0.5, W - 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(title, fontsize=9)
    return im


def plot_path(ax, land_mask, path_mask, seed, n_pts):
    H, W = land_mask.shape
    ext  = _ext(W, H)
    lm   = land_mask.T
    pm   = path_mask.T.astype(float)

    ax.imshow(lm, origin="lower",
              cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
              extent=ext, aspect="auto", zorder=0)
    ax.imshow(pm, origin="lower", cmap="Reds", alpha=0.8,
              extent=ext, aspect="auto", zorder=1, vmin=0, vmax=1)

    ax.legend(handles=[
        Patch(facecolor="white",   edgecolor="gray", label="Ocean"),
        Patch(facecolor="#d62728",                   label="Path"),
        Patch(facecolor="black",                     label="Land"),
    ], loc="upper right", fontsize=7)

    ax.set_xlim(-0.5, H - 0.5); ax.set_ylim(-0.5, W - 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"Robot Path  ({n_pts} cells, seed={seed})", fontsize=9)


def plot_error(ax, pred, truth, land_mask, vmax_err, title):
    H, W  = land_mask.shape
    err   = np.sqrt((pred[0]-truth[0])**2 + (pred[1]-truth[1])**2)
    ed    = err.T.astype(float)
    ed[land_mask.T] = np.nan

    ax.imshow(np.zeros((W, H, 3)), origin="lower",
              extent=_ext(W, H), aspect="auto")
    im = ax.imshow(ed, origin="lower", cmap="hot_r",
                   norm=mcolors.Normalize(0, vmax_err),
                   extent=_ext(W, H), aspect="auto")

    ax.set_xlim(-0.5, H - 0.5); ax.set_ylim(-0.5, W - 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(title, fontsize=9)
    return im


def save_comparison(true_np, pred_r10, pred_r1, pred_dps,
                    land_mask, path_mask, seed, sample_idx,
                    rmse_r10, rmse_r1, rmse_dps,
                    ckpt_name, T, stride, out_path):

    ocean = ~land_mask
    spd_true = np.sqrt(true_np[0]**2 + true_np[1]**2)
    vmax_spd = float(np.percentile(spd_true[ocean], 98))

    errs = np.concatenate([
        np.sqrt((pred_r10[0]-true_np[0])**2+(pred_r10[1]-true_np[1])**2)[ocean],
        np.sqrt((pred_r1 [0]-true_np[0])**2+(pred_r1 [1]-true_np[1])**2)[ocean],
        np.sqrt((pred_dps[0]-true_np[0])**2+(pred_dps[1]-true_np[1])**2)[ocean],
    ])
    vmax_err = float(np.percentile(errs, 98))

    fig, axes = plt.subplots(2, 4, figsize=(5.5 * 4, 5.0 * 2),
                             constrained_layout=True)
    fig.suptitle(
        f"Test sample {sample_idx}  —  seed={seed}  |  {ckpt_name}"
        f"  (T={T}/stride={stride}, biased_walk_path)",
        fontsize=12, fontweight="bold"
    )

    im_spd = plot_quiver(axes[0, 0], true_np,  land_mask, vmax_spd, "Ground Truth")
    plot_quiver(axes[0, 1], pred_r10, land_mask, vmax_spd,
                f"RePaint r=10\nRMSE={rmse_r10:.4f}")
    plot_quiver(axes[0, 2], pred_r1,  land_mask, vmax_spd,
                f"RePaint r=1\nRMSE={rmse_r1:.4f}")
    plot_quiver(axes[0, 3], pred_dps, land_mask, vmax_spd,
                f"DPS z=0.04\nRMSE={rmse_dps:.4f}")
    fig.colorbar(im_spd, ax=axes[0, :], location="right",
                 shrink=0.6, label="Speed", pad=0.01)

    plot_path(axes[1, 0], land_mask, path_mask, seed, int(path_mask.sum()))
    im_err = plot_error(axes[1, 1], pred_r10, true_np, land_mask, vmax_err,
                        f"Error  (RMSE={rmse_r10:.4f})")
    plot_error(axes[1, 2], pred_r1,  true_np, land_mask, vmax_err,
               f"Error  (RMSE={rmse_r1:.4f})")
    plot_error(axes[1, 3], pred_dps, true_np, land_mask, vmax_err,
               f"Error  (RMSE={rmse_dps:.4f})")
    fig.colorbar(im_err, ax=axes[1, 1:], location="right",
                 shrink=0.6, label="|error| speed", pad=0.01)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--out_dir",     default="batch_3methods")
    p.add_argument("--n_seeds",     type=int, default=10)
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--stride",      type=int, default=10)
    p.add_argument("--path_steps",  type=int, default=400)
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

    seeds = list(range(0, args.n_seeds * 7, 7))[:args.n_seeds]
    print(f"Seeds: {seeds}\n")

    for seed in seeds:
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]
        true_np    = x0_true.numpy()
        path_mask  = biased_walk_path(land_mask, n_steps=args.path_steps, seed=seed)
        x0_obs     = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"Seed={seed}  sample={sample_idx}  path_cells={int(path_mask.sum())}")

        print("  r=10 ...", end="", flush=True)
        pred_r10  = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask,
                                  r=10, device=device, stride=args.stride)
        rmse_r10  = float(np.sqrt(np.mean(
            (pred_r10[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        print(f"  RMSE={rmse_r10:.4f}")

        print("  r=1  ...", end="", flush=True)
        pred_r1   = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask,
                                  r=1,  device=device, stride=args.stride)
        rmse_r1   = float(np.sqrt(np.mean(
            (pred_r1[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        print(f"  RMSE={rmse_r1:.4f}")

        print("  DPS  ...", end="", flush=True)
        pred_dps  = dps_infer(model, diffusion, x0_obs, path_mask, land_mask,
                               device=device, stride=args.stride, step_size=0.04)
        rmse_dps  = float(np.sqrt(np.mean(
            (pred_dps[:, ocean_mask] - true_np[:, ocean_mask])**2)))
        print(f"  RMSE={rmse_dps:.4f}")

        out_path = os.path.join(args.out_dir, f"seed{seed:03d}_comparison.png")
        save_comparison(
            true_np, pred_r10, pred_r1, pred_dps,
            land_mask, path_mask, seed, sample_idx,
            rmse_r10, rmse_r1, rmse_dps,
            ckpt_name, args.T, args.stride, out_path
        )
        print(f"  Saved: {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
