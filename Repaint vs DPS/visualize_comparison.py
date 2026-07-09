"""
visualize_comparison.py
========================
Produces a 2-row × 4-col comparison image for one seed:
  Row 0 (quiver): Ground Truth | RePaint r=10 | RePaint r=1 | DPS z=0.04
  Row 1 (error) : Robot Path   | Error r=10   | Error r=1   | Error DPS

Style matches result_01_all_strides.png:
  - Quiver arrows colored by speed (cool colormap: cyan→magenta)
  - Black land, white ocean background
  - Per-method absolute-error speed map (hot_r colormap)
  - Robot path panel (beige ocean / dark-red path / black land)

Usage:
    python visualize_comparison.py \\
        --pickle /root/ocean_ddpm/data.pickle \\
        --checkpoint /root/Repaint_vs_DPS/checkpoints_linear/best_model_linear.pt \\
        --seed 63 --T 1000 --stride 10 \\
        --out comparison_seed63.png
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


# ── inference functions (copied from run_12methods.py) ────────────────────────

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


# ── plotting helpers ──────────────────────────────────────────────────────────


def rmse_ocean(pred, truth, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - truth[:, ocean_mask]) ** 2)))


def _extent(H, W):
    return [-0.5, W - 0.5, -0.5, H - 0.5]


def plot_quiver(ax, field, land_mask, vmax_spd, title, step=2):
    """Speed background (cool cmap) + black arrows. Returns mappable for shared colorbar."""
    H, W = land_mask.shape
    u  = field[0].T   # (W, H)
    v  = field[1].T
    lm = land_mask.T
    speed = np.ma.masked_where(lm, np.sqrt(u ** 2 + v ** 2))

    im = ax.imshow(
        speed, origin="lower", cmap="cool", vmin=0, vmax=vmax_spd,
        extent=_extent(W, H), aspect="auto", zorder=0,
    )
    ax.imshow(
        lm, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=_extent(W, H), aspect="auto", zorder=1,
    )

    yq, xq = np.mgrid[0:W:step, 0:H:step]
    uq = u[::step, ::step]
    vq = v[::step, ::step]
    mask = ~lm[::step, ::step]
    ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask],
        color="black", scale=12, width=0.003, zorder=2,
    )

    ax.set_xlim(-0.5, H - 0.5)
    ax.set_ylim(-0.5, W - 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, fontsize=9)
    return im


def plot_path(ax, land_mask, path_mask, seed, n_path_pts):
    """Robot path panel matching visualize_infer.py: white ocean, red path, black land."""
    H, W = land_mask.shape
    lm   = land_mask.T          # (W, H)
    ext  = _extent(W, H)

    # White ocean, black land
    ax.imshow(
        lm, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=ext, aspect="auto", zorder=0,
    )
    # Red path overlay
    path_disp = path_mask.T.astype(float)   # (W, H)
    ax.imshow(
        path_disp, origin="lower", cmap="Reds", alpha=0.8,
        extent=ext, aspect="auto", zorder=1, vmin=0, vmax=1,
    )

    ocean_p = Patch(facecolor="white",   edgecolor="gray", label="Ocean")
    path_p  = Patch(facecolor="#d62728",                   label="Path")
    land_p  = Patch(facecolor="black",                     label="Land")
    ax.legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=7)

    ax.set_xlim(-0.5, H - 0.5)
    ax.set_ylim(-0.5, W - 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Robot Path ({n_path_pts} cells, seed={seed})", fontsize=9)


def plot_error(ax, pred, truth, land_mask, vmax_err, title):
    """Absolute error speed map; black land; hot_r colormap. Returns mappable."""
    H, W   = land_mask.shape
    ocean  = ~land_mask

    err      = np.sqrt((pred[0] - truth[0]) ** 2 + (pred[1] - truth[1]) ** 2)
    err_disp = err.T.astype(float)
    lm       = land_mask.T
    err_disp[lm] = np.nan

    bg = np.zeros((W, H, 3))
    ax.imshow(bg, origin="lower", interpolation="nearest",
              extent=_extent(W, H), aspect="auto")

    norm = mcolors.Normalize(vmin=0, vmax=vmax_err)
    im = ax.imshow(err_disp, origin="lower", cmap="hot_r", norm=norm,
                   interpolation="nearest",
                   extent=_extent(W, H), aspect="auto")

    ax.set_xlim(-0.5, H - 0.5)
    ax.set_ylim(-0.5, W - 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, fontsize=9)
    return im


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed",       type=int, default=63)
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--stride",     type=int, default=10)
    p.add_argument("--path_steps", type=int, default=400)
    p.add_argument("--out",        default="comparison.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  seed={args.seed}")

    # ── load data ──
    test_ds    = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = test_ds.land_mask.numpy()    # (H, W) bool
    ocean_mask = ~land_mask
    n_test     = len(test_ds)

    sample_idx = args.seed % n_test
    x0_true    = test_ds[sample_idx]          # (2, H, W)
    true_np    = x0_true.numpy()

    path_mask = biased_walk_path(land_mask, n_steps=args.path_steps,
                                 seed=args.seed)
    x0_obs = x0_true.clone()
    x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

    # ── load model ──
    ckpt      = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=False)
    H, W      = land_mask.shape
    model     = Repaint(in_ch=2, base_ch=64, time_dim=256).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    schedule  = ckpt.get("schedule", "linear")
    noise_std = ckpt.get("noise_std", 1.0)
    diffusion = DDPM(T=args.T, beta_schedule=schedule,
                     noise_std=noise_std, device=device)

    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss','?')}  schedule={schedule}")

    # ── run inference ──
    print("Running RePaint r=10 ...")
    pred_r10 = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask,
                             r=10, device=device, stride=args.stride)
    rmse_r10 = rmse_ocean(pred_r10, true_np, ocean_mask)
    print(f"  RMSE={rmse_r10:.4f}")

    print("Running RePaint r=1 ...")
    pred_r1  = repaint_infer(model, diffusion, x0_obs, path_mask, land_mask,
                             r=1,  device=device, stride=args.stride)
    rmse_r1  = rmse_ocean(pred_r1, true_np, ocean_mask)
    print(f"  RMSE={rmse_r1:.4f}")

    print("Running DPS z=0.04 ...")
    pred_dps = dps_infer(model, diffusion, x0_obs, path_mask, land_mask,
                         device=device, stride=args.stride, step_size=0.04)
    rmse_dps = rmse_ocean(pred_dps, true_np, ocean_mask)
    print(f"  RMSE={rmse_dps:.4f}")

    # ── colour scales ──
    spd_true = np.sqrt(true_np[0] ** 2 + true_np[1] ** 2)
    vmax_spd = float(np.percentile(spd_true[ocean_mask], 98))

    all_errs = np.concatenate([
        np.sqrt((pred_r10[0]-true_np[0])**2 + (pred_r10[1]-true_np[1])**2)[ocean_mask],
        np.sqrt((pred_r1 [0]-true_np[0])**2 + (pred_r1 [1]-true_np[1])**2)[ocean_mask],
        np.sqrt((pred_dps[0]-true_np[0])**2 + (pred_dps[1]-true_np[1])**2)[ocean_mask],
    ])
    vmax_err = float(np.percentile(all_errs, 98))

    # ── figure: 2 rows × 4 cols ──
    # Row 0: Ground Truth | RePaint r=10 | RePaint r=1 | DPS z=0.04  (quiver)
    # Row 1: Robot Path   | Error r=10   | Error r=1   | Error DPS   (path/error)
    H, W      = land_mask.shape
    ckpt_name = os.path.basename(args.checkpoint)
    fig, axes = plt.subplots(2, 4, figsize=(5.5 * 4, 5.0 * 2),
                             constrained_layout=True)

    fig.suptitle(
        f"Test sample {sample_idx}  —  seed={args.seed}"
        f"  checkpoint={ckpt_name}",
        fontsize=13, fontweight="bold"
    )

    # Row 0: quiver panels
    im_spd = plot_quiver(axes[0, 0], true_np,  land_mask, vmax_spd, "Ground Truth")
    plot_quiver(axes[0, 1], pred_r10, land_mask, vmax_spd,
                f"RePaint r=10\nRMSE={rmse_r10:.4f}")
    plot_quiver(axes[0, 2], pred_r1,  land_mask, vmax_spd,
                f"RePaint r=1\nRMSE={rmse_r1:.4f}")
    plot_quiver(axes[0, 3], pred_dps, land_mask, vmax_spd,
                f"DPS z=0.04\nRMSE={rmse_dps:.4f}")
    fig.colorbar(im_spd, ax=axes[0, :], location="right", shrink=0.6,
                 label="Speed", pad=0.01)

    # Row 1: path + error panels
    plot_path(axes[1, 0], land_mask, path_mask, args.seed,
              int(path_mask.sum()))
    im_err = plot_error(axes[1, 1], pred_r10, true_np, land_mask, vmax_err,
                        f"Error  (RMSE={rmse_r10:.4f})")
    plot_error(axes[1, 2], pred_r1,  true_np, land_mask, vmax_err,
               f"Error  (RMSE={rmse_r1:.4f})")
    plot_error(axes[1, 3], pred_dps, true_np, land_mask, vmax_err,
               f"Error  (RMSE={rmse_dps:.4f})")
    fig.colorbar(im_err, ax=axes[1, 1:], location="right", shrink=0.6,
                 label="|error| speed", pad=0.01)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
