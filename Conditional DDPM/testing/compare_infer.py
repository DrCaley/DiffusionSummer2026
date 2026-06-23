"""
Side-by-side batch inference: DDPM-eps (RePaint) vs Voronoi-conditioned DDPM.

For each of 10 random validation samples this script produces a 2×3 figure:
  Row 1 (top):    Ground truth  |  Robot path  |  Voronoi tessellation map
  Row 2 (bottom): DDPM-eps pred |  Voronoi-cond pred  |  Error comparison

Usage (run from workspace root):
    python3.12 "Conditional DDPM/compare_infer.py"
    python3.12 "Conditional DDPM/compare_infer.py" --n_samples 10 --seed 0 \\
        --eps_ckpt  DDPM/checkpoints/best_model.pt \\
        --cond_ckpt "Conditional DDPM/checkpoints_voronoi/best_cond_ddpm_voronoi_cosine.pt"
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — works from any cwd
# ---------------------------------------------------------------------------
_here      = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_here, ".."))
_ddpm_dir  = os.path.join(_repo_root, "DDPM")
_voronoi_model_dir = os.path.join(_repo_root, "Voronoi", "model")

for p in (_here, _repo_root, _ddpm_dir, _voronoi_model_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset        import OceanCurrentDataset
from paths          import biased_walk_path
from voronoi_model  import VoronoiLayer
from cond_model     import CondUNet
from cond_diffusion import CondDDPM

# DDPM eps imports (from DDPM/ sub-tree)
_ddpm_model_dir = os.path.join(_ddpm_dir, "model")
sys.path.insert(0, _ddpm_model_dir)
from diffusion import DDPM
from model     import UNet

_repaint_dir = os.path.join(_ddpm_dir, "testing", "repaint")
sys.path.insert(0, _repaint_dir)
from repaint_infer import repaint


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_LAND_BW    = plt.matplotlib.colors.ListedColormap(["white", "black"])
_LAND_ALPHA = plt.matplotlib.colors.ListedColormap(["none",  "black"])
_EXT        = None   # set per-sample once H, W are known


def _extent(H, W):
    return [-0.5, W - 0.5, -0.5, H - 0.5]


def plot_field(ax, u, v, land_mask, title, cmap="cool", step=2, clim=None):
    H, W = u.shape
    ext  = _extent(H, W)
    ax.imshow(land_mask, origin="lower", cmap=_LAND_BW, extent=ext, aspect="auto", zorder=0)
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    mq     = np.sqrt(uq**2 + vq**2)
    mask   = ~np.isnan(uq) & ~land_mask[::step, ::step]
    if mask.any():
        if clim is None:
            clim = (0, np.nanpercentile(mq[mask], 98))
        q = ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
                      cmap=cmap, clim=clim, scale=12, width=0.003, zorder=2)
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def plot_path(ax, land_mask, path_mask, title):
    H, W = land_mask.shape
    ext  = _extent(H, W)
    ax.imshow(land_mask, origin="lower", cmap=_LAND_BW, extent=ext, aspect="auto", zorder=0)
    disp = np.zeros_like(land_mask, dtype=float)
    disp[path_mask] = 1.0
    ax.imshow(disp, origin="lower", cmap="Reds", alpha=0.8, extent=ext,
              aspect="auto", zorder=1, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="white",   edgecolor="gray", label="Ocean"),
            mpatches.Patch(facecolor="#d62728",                   label="Path"),
            mpatches.Patch(facecolor="black",                     label="Land"),
        ],
        loc="upper right", fontsize=7,
    )





def plot_error_pair(ax, err_eps, err_cond, land_mask):
    """Overlay eps error (red) vs cond error (blue) as transparent heat-maps."""
    H, W = land_mask.shape
    ext  = _extent(H, W)
    ax.imshow(land_mask, origin="lower", cmap=_LAND_BW, extent=ext, aspect="auto", zorder=0)
    vmax = np.nanpercentile(np.concatenate([err_eps[~land_mask], err_cond[~land_mask]]), 95)
    err_eps_ma  = np.ma.masked_where(land_mask, err_eps)
    err_cond_ma = np.ma.masked_where(land_mask, err_cond)
    ax.imshow(err_eps_ma,  origin="lower", cmap="Reds",  alpha=0.55,
              extent=ext, aspect="auto", vmin=0, vmax=vmax, zorder=1)
    ax.imshow(err_cond_ma, origin="lower", cmap="Blues", alpha=0.55,
              extent=ext, aspect="auto", vmin=0, vmax=vmax, zorder=2)
    ax.imshow(land_mask, origin="lower", cmap=_LAND_ALPHA, extent=ext,
              aspect="auto", zorder=3)
    ax.set_title("Error: eps (red) vs voronoi (blue)", fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--eps_ckpt",   default=None,
                   help="DDPM-eps checkpoint. Default: DDPM/checkpoints/best_model.pt")
    p.add_argument("--cond_ckpt",  default=None,
                   help="Voronoi-cond checkpoint. Default: Conditional DDPM/checkpoints_voronoi/best_cond_ddpm_voronoi_cosine.pt")
    p.add_argument("--n_samples",  type=int,   default=10)
    p.add_argument("--path_steps", type=int,   default=150)
    p.add_argument("--resample",   type=int,   default=10,
                   help="RePaint r parameter for eps model")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--out_dir",    default=None,
                   help="Output directory. Default: Conditional DDPM/results_compare")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.eps_ckpt is None:
        args.eps_ckpt = os.path.join(_ddpm_dir, "checkpoints", "best_model.pt")
    if args.cond_ckpt is None:
        args.cond_ckpt = os.path.join(
            _here, "checkpoints_voronoi", "best_cond_ddpm_voronoi_cosine.pt"
        )
    if args.out_dir is None:
        args.out_dir = os.path.join(_here, "results_compare")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device      : {device}")
    print(f"eps  ckpt   : {args.eps_ckpt}")
    print(f"cond ckpt   : {args.cond_ckpt}")
    print(f"Output dir  : {args.out_dir}")

    # ---- Data ----------------------------------------------------------------
    val_ds       = OceanCurrentDataset(args.pickle, split=1)
    land_mask_np = val_ds.land_mask.numpy()   # (H, W)
    H, W         = land_mask_np.shape
    rng          = np.random.default_rng(args.seed)
    sample_idxs  = rng.choice(len(val_ds), size=args.n_samples, replace=False).tolist()
    print(f"Val samples : {sample_idxs}")

    # ---- Voronoi layer -------------------------------------------------------
    voronoi_layer = VoronoiLayer(H=H, W=W, n_sensors=args.path_steps).to(device)

    # ---- Load eps model ------------------------------------------------------
    eps_ckpt = torch.load(args.eps_ckpt, map_location=device, weights_only=False)
    eps_args = eps_ckpt.get("args", {})
    eps_model = UNet(
        in_ch    = 2,
        base_ch  = eps_args.get("base_ch",  64),
        time_dim = eps_args.get("time_dim", 256),
    ).to(device)
    eps_model.load_state_dict(eps_ckpt["model"])
    eps_model.eval()
    eps_diffusion = DDPM(
        T             = eps_args.get("T",        1000),
        beta_schedule = eps_args.get("schedule", "cosine"),
        device        = device,
    )
    print(f"eps  model  : epoch={eps_ckpt.get('epoch','?')}  val_loss={eps_ckpt.get('val_loss',float('nan')):.5f}")

    # ---- Load voronoi-cond model ---------------------------------------------
    cond_ckpt  = torch.load(args.cond_ckpt, map_location=device, weights_only=False)
    cond_args  = cond_ckpt.get("args", {})
    cond_model = CondUNet(
        in_ch      = 2,
        cond_in_ch = 3,
        base_ch    = cond_args.get("base_ch",  64),
        time_dim   = cond_args.get("time_dim", 256),
        cond_dim   = cond_args.get("cond_dim", 256),
    ).to(device)
    cond_model.load_state_dict(cond_ckpt["model"])
    cond_model.eval()
    cond_diffusion = CondDDPM(
        T             = cond_args.get("T",        1000),
        beta_schedule = cond_args.get("schedule", "cosine"),
        device        = device,
    )
    print(f"cond model  : epoch={cond_ckpt.get('epoch','?')}  val_loss={cond_ckpt.get('val_loss',float('nan')):.5f}")

    # ---- Inference loop ------------------------------------------------------
    eps_rmses, cond_rmses = [], []

    for run, sample_idx in enumerate(sample_idxs, start=1):
        seed   = args.seed + run
        x0_true = val_ds[sample_idx].to(device)   # (2, H, W)

        # Build robot path
        path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)
        rows, cols = np.where(path_mask)

        # ── Voronoi tessellation ──────────────────────────────────────────────
        rows_n = torch.tensor(rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
        cols_n = torch.tensor(cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
        sensor_pos  = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)   # (1, K, 2)
        K = len(rows)
        flat_idx    = torch.tensor(rows * W + cols, dtype=torch.long, device=device)
        flat_idx    = flat_idx.unsqueeze(0).unsqueeze(0).expand(1, 2, K)
        sensor_vals = torch.gather(
            x0_true.unsqueeze(0).reshape(1, 2, H * W), 2, flat_idx
        )   # (1, 2, K)
        with torch.no_grad():
            voronoi_grid = voronoi_layer.tessellate(sensor_vals, sensor_pos)   # (1, 3, H, W)
        voronoi_np = voronoi_grid[0].cpu().numpy()   # (3, H, W)

        # ── eps RePaint ───────────────────────────────────────────────────────
        x0_known = x0_true.clone()
        ocean_mask = ~torch.from_numpy(land_mask_np).to(device)
        x0_known[:, ~torch.from_numpy(path_mask).to(device)] = 0.0
        with torch.no_grad():
            eps_pred = repaint(
                eps_model, eps_diffusion, x0_known,
                path_mask, land_mask_np,
                r=args.resample, device=device,
            )   # (2, H, W) cpu tensor

        # ── Voronoi-cond DDPM ─────────────────────────────────────────────────
        with torch.no_grad():
            cond_pred = cond_diffusion.sample(
                cond_model, voronoi_grid, shape=(1, 2, H, W)
            )[0]   # (2, H, W) on device

        # ── Metrics ──────────────────────────────────────────────────────────
        u_true = x0_true[0].cpu().numpy()
        v_true = x0_true[1].cpu().numpy()
        u_eps  = eps_pred[0].numpy()
        v_eps  = eps_pred[1].numpy()
        u_cond = cond_pred[0].cpu().numpy()
        v_cond = cond_pred[1].cpu().numpy()

        ocean = ~land_mask_np
        speed_true = np.sqrt(u_true**2 + v_true**2)
        shared_clim = (0, float(np.nanpercentile(speed_true[ocean], 98)))
        err_eps  = np.sqrt((u_eps  - u_true)**2 + (v_eps  - v_true)**2)
        err_cond = np.sqrt((u_cond - u_true)**2 + (v_cond - v_true)**2)
        rmse_eps  = float(np.sqrt(np.mean(err_eps [ocean]**2)))
        rmse_cond = float(np.sqrt(np.mean(err_cond[ocean]**2)))
        eps_rmses.append(rmse_eps)
        cond_rmses.append(rmse_cond)
        print(f"  [{run:2d}/{args.n_samples}] sample={sample_idx}  "
              f"eps RMSE={rmse_eps:.4f}  cond RMSE={rmse_cond:.4f}")

        # ── Plot (2 rows × 3 cols) ────────────────────────────────────────────
        # Transpose for display: data is (H, W) but display uses (W, H) convention
        T_ = lambda a: a.T
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        fig.suptitle(
            f"DDPM-eps vs Voronoi-cond  |  val sample {sample_idx}  |  seed {seed}\n"
            f"eps RMSE={rmse_eps:.4f}   cond RMSE={rmse_cond:.4f}",
            fontsize=13,
        )

        # Row 1
        plot_field(axes[0, 0], T_(u_true), T_(v_true), T_(land_mask_np), "Ground Truth",
                   clim=shared_clim)
        plot_path (axes[0, 1], T_(land_mask_np), T_(path_mask),
                   f"Robot Path ({path_mask.sum()} cells)")
        plot_field(axes[0, 2], T_(voronoi_np[0]), T_(voronoi_np[1]), T_(land_mask_np),
                   "Voronoi Tessellation", clim=shared_clim)

        # Row 2
        plot_field(axes[1, 0], T_(u_eps),  T_(v_eps),  T_(land_mask_np),
                   f"DDPM-eps (RePaint)  RMSE={rmse_eps:.4f}", clim=shared_clim)
        plot_field(axes[1, 1], T_(u_cond), T_(v_cond), T_(land_mask_np),
                   f"Voronoi-cond DDPM  RMSE={rmse_cond:.4f}", clim=shared_clim)
        plot_error_pair(axes[1, 2], T_(err_eps), T_(err_cond), T_(land_mask_np))

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"compare_{run:02d}_idx{sample_idx}.png")
        plt.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"           saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*56}")
    print(f"  eps  Mean RMSE : {np.mean(eps_rmses):.4f}  ±{np.std(eps_rmses):.4f}")
    print(f"  cond Mean RMSE : {np.mean(cond_rmses):.4f}  ±{np.std(cond_rmses):.4f}")
    print(f"{'─'*56}")

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(1, len(eps_rmses) + 1)
    ax.bar(xs - 0.2, eps_rmses,  0.4, label=f"DDPM-eps  (μ={np.mean(eps_rmses):.4f})",  color="steelblue")
    ax.bar(xs + 0.2, cond_rmses, 0.4, label=f"Voronoi-cond (μ={np.mean(cond_rmses):.4f})", color="darkorange")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{i}" for i in sample_idxs], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Val sample"); ax.set_ylabel("RMSE")
    ax.set_title("DDPM-eps vs Voronoi-cond — per-sample RMSE")
    ax.legend()
    plt.tight_layout()
    summary_path = os.path.join(args.out_dir, "rmse_summary.png")
    plt.savefig(summary_path, dpi=140)
    plt.close(fig)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
