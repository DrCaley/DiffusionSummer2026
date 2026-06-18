"""
Batch inference for the FiLM-conditioned DDPM on 10 random validation samples.

Usage (run from workspace root):
    py "Conditional DDPM/infer.py" --cond voronoi
    py "Conditional DDPM/infer.py" --cond path
    py "Conditional DDPM/infer.py" --cond both
    py "Conditional DDPM/infer.py" --cond voronoi --n_samples 10 --seed 0
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_here      = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_here, ".."))
_voronoi_model_dir = os.path.join(_repo_root, "Voronoi", "model")

sys.path.insert(0, _here)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _voronoi_model_dir)

from dataset        import OceanCurrentDataset
from paths          import biased_walk_path, basic_robot_path
from voronoi_model  import VoronoiLayer
from cond_model     import CondUNet
from cond_diffusion import CondDDPM

COND_MODES = {"voronoi": 3, "path": 1, "path_field": 3, "both": 4}


def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool"):
    H, W = u.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u[::step, ::step]
    vq = v[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    mask = ~np.isnan(uq) & ~land_mask[::step, ::step]
    if mask.any():
        q = ax.quiver(
            xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
            cmap=cmap, clim=(0, np.nanpercentile(mq[mask], 98) if mask.any() else 1),
            scale=12, width=0.003, zorder=2,
        )
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cond",       required=True, choices=list(COND_MODES))
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt checkpoint. Defaults to "
                        "'Conditional DDPM/checkpoints_{cond}/best_cond_ddpm_{cond}_cosine.pt'")
    p.add_argument("--n_samples",  type=int, default=10)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--seed",       type=int, default=0,
                   help="Base seed; each sample increments by 1.")
    p.add_argument("--out_dir",    default=None,
                   help="Output directory. Defaults to 'Conditional DDPM/results_{cond}'.")
    p.add_argument("--resample",   type=int, default=10,
                   help="RePaint resampling iterations per timestep for path/both modes (default: 10).")
    p.add_argument("--no_repaint", action="store_true",
                   help="Use pure conditional sampling (diffusion.sample) instead of RePaint.")
    p.add_argument("--path_fn",    default="biased_walk", choices=["biased_walk", "basic_robot"],
                   help="Path generation function used at inference time.")
    p.add_argument("--segment_len", type=int, default=10,
                   help="Segment length for basic_robot path (default: 10).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build conditioning tensor for a single sample
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_cond_single(x0, land_mask_np, voronoi_layer, cond_mode, path_fn, n_steps, segment_len, seed, device):
    """Return (1, cond_in_ch, H, W) conditioning map for one sample."""
    _, H, W = x0.shape
    C = x0.shape[0]

    if path_fn == "basic_robot":
        path_mask = basic_robot_path(land_mask_np, segment_len=segment_len, seed=seed)
    else:
        path_mask = biased_walk_path(land_mask_np, n_steps=n_steps, seed=seed)
    rows, cols = np.where(path_mask)
    K = len(rows)

    if cond_mode in ("voronoi", "both"):
        rows_n = torch.tensor(rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
        cols_n = torch.tensor(cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
        sensor_pos = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)  # (1, K, 2)

        flat_idx    = torch.tensor(rows * W + cols, dtype=torch.long, device=device)
        flat_idx    = flat_idx.unsqueeze(0).unsqueeze(0).expand(1, C, K)
        sensor_vals = torch.gather(
            x0.unsqueeze(0).reshape(1, C, H * W), 2, flat_idx
        )  # (1, C, K)
        voronoi_grid = voronoi_layer.tessellate(sensor_vals, sensor_pos)  # (1, 3, H, W)

    if cond_mode == "voronoi":
        return voronoi_grid, path_mask
    elif cond_mode == "path":
        path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)
        return path_ch.unsqueeze(0).unsqueeze(0), path_mask   # (1, 1, H, W)
    elif cond_mode == "path_field":
        path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)  # (H, W)
        u_path  = x0[0] * path_ch   # ground-truth u at path cells, 0 elsewhere
        v_path  = x0[1] * path_ch   # ground-truth v at path cells, 0 elsewhere
        cond = torch.stack([u_path, v_path, path_ch], dim=0).unsqueeze(0)  # (1, 3, H, W)
        return cond, path_mask
    else:  # both
        path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)
        return torch.cat([voronoi_grid, path_ch.unsqueeze(0).unsqueeze(0)], dim=1), path_mask


# ---------------------------------------------------------------------------
# Single-sample plot
# ---------------------------------------------------------------------------

def save_sample_plot(
    u_true, v_true, u_pred, v_pred,
    land_mask, path_mask, cond_mode,
    rmse, sample_idx, seed, out_path,
):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f"Cond-DDPM ({cond_mode})  |  val sample {sample_idx}  |  seed {seed}\n"
        f"RMSE={rmse:.4f}",
        fontsize=13,
    )
    axes = axes.flatten()

    # 1 — Ground truth
    plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth")

    # 2 — Sensor path / Voronoi extent
    axes[1].imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_mask.shape[1]-0.5, -0.5, land_mask.shape[0]-0.5],
        aspect="auto", zorder=0,
    )
    path_disp = np.zeros_like(land_mask, dtype=float)
    path_disp[path_mask] = 1.0
    axes[1].imshow(
        path_disp, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, land_mask.shape[1]-0.5, -0.5, land_mask.shape[0]-0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    axes[1].set_title(f"Robot Path ({path_mask.sum()} cells)", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    axes[1].legend(
        handles=[
            mpatches.Patch(facecolor="white",   edgecolor="gray", label="Ocean"),
            mpatches.Patch(facecolor="#d62728",                   label="Path"),
            mpatches.Patch(facecolor="black",                     label="Land"),
        ],
        loc="upper right", fontsize=8,
    )

    # 3 — Prediction
    plot_field(axes[2], u_pred, v_pred, land_mask, f"Prediction ({cond_mode} cond.)", cmap="cool")

    # 4 — Error magnitude
    err = np.sqrt((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)
    err[land_mask] = np.nan
    err_ma = np.ma.masked_where(land_mask, err)
    im = axes[3].imshow(
        err_ma, origin="lower", cmap="hot_r", aspect="auto",
        extent=[-0.5, land_mask.shape[1]-0.5, -0.5, land_mask.shape[0]-0.5],
    )
    axes[3].imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=[-0.5, land_mask.shape[1]-0.5, -0.5, land_mask.shape[0]-0.5],
        aspect="auto", zorder=1,
    )
    plt.colorbar(im, ax=axes[3], label="Speed error", shrink=0.7)
    axes[3].set_title(f"Error  (RMSE={rmse:.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond_in_ch = COND_MODES[args.cond]

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            _here, f"checkpoints_{args.cond}",
            f"best_cond_ddpm_{args.cond}_cosine.pt",
        )
    if args.out_dir is None:
        suffix = "_no_repaint" if args.no_repaint else ""
        args.out_dir = os.path.join(_here, f"results_{args.cond}{suffix}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Cond mode  : {args.cond}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output dir : {args.out_dir}")

    # ---- Data ----------------------------------------------------------------
    val_ds       = OceanCurrentDataset(args.pickle, split=1)
    land_mask_np = val_ds.land_mask.numpy()        # (H, W)
    H, W         = land_mask_np.shape
    rng          = np.random.default_rng(args.seed)
    sample_idxs  = rng.choice(len(val_ds), size=args.n_samples, replace=False).tolist()
    print(f"Val set size: {len(val_ds)}  |  samples: {sample_idxs}")

    # ---- Voronoi layer -------------------------------------------------------
    voronoi_layer = None
    if args.cond in ("voronoi", "both"):
        voronoi_layer = VoronoiLayer(H=H, W=W, n_sensors=args.path_steps).to(device)

    # ---- Load model ----------------------------------------------------------
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = CondUNet(
        in_ch      = 2,
        cond_in_ch = cond_in_ch,
        base_ch    = ckpt_args.get("base_ch",  64),
        time_dim   = ckpt_args.get("time_dim", 256),
        cond_dim   = ckpt_args.get("cond_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?'):.5f}")

    diffusion = CondDDPM(
        T             = ckpt_args.get("T",        1000),
        beta_schedule = ckpt_args.get("schedule", "cosine"),
        device        = device,
    )

    # ---- Inference loop ------------------------------------------------------
    rmses = []

    for run, sample_idx in enumerate(sample_idxs, start=1):
        seed = args.seed + run
        x0_true = val_ds[sample_idx].to(device)   # (2, H, W)

        cond, path_mask = make_cond_single(
            x0_true, land_mask_np, voronoi_layer,
            args.cond, args.path_fn, args.path_steps, args.segment_len, seed, device,
        )

        if args.no_repaint:
            x0_pred = diffusion.sample(model, cond)[0]  # (2, H, W)
        else:
            # Inference: RePaint — anchors true u/v at path cells
            path_t       = torch.from_numpy(path_mask.astype(np.float32)).to(device)
            x0_known     = x0_true * path_t.unsqueeze(0)          # (2, H, W)
            x0_known_b   = x0_known.unsqueeze(0)                  # (1, 2, H, W)
            path_mask_t  = path_t[None, None]                     # (1, 1, H, W)
            ocean_mask_t = torch.from_numpy(
                (~land_mask_np).astype(np.float32)
            ).to(device)[None, None]                              # (1, 1, H, W)
            x0_pred = diffusion.repaint(
                model, cond, x0_known_b, path_mask_t, ocean_mask_t,
                r=args.resample,
            )[0]  # (2, H, W)

        # Metrics on ocean pixels only
        u_true_np = x0_true[0].cpu().numpy()
        v_true_np = x0_true[1].cpu().numpy()
        u_pred_np = x0_pred[0].cpu().numpy()
        v_pred_np = x0_pred[1].cpu().numpy()

        err = np.sqrt((u_pred_np - u_true_np)**2 + (v_pred_np - v_true_np)**2)
        ocean_err = err[~land_mask_np]
        rmse = float(np.sqrt(np.mean(ocean_err**2)))
        rmses.append(rmse)
        print(f"  [{run:2d}/10]  sample={sample_idx}  seed={seed}  RMSE={rmse:.4f}")

        # Plot — transpose to (W, H) for display (matching existing style)
        save_sample_plot(
            u_true_np.T, v_true_np.T,
            u_pred_np.T, v_pred_np.T,
            land_mask_np.T, path_mask.T,
            args.cond, rmse,
            sample_idx, seed,
            os.path.join(args.out_dir, f"sample_{run:02d}_idx{sample_idx}.png"),
        )

    # ---- Summary -------------------------------------------------------------
    print(f"\n{'─'*48}")
    print(f"  Mean RMSE : {np.mean(rmses):.4f}  ±{np.std(rmses):.4f}")
    print(f"{'─'*48}")

    # Summary figure — bar chart of per-sample RMSE
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(1, len(rmses)+1)
    ax.bar(xs, rmses, color="steelblue", label="RMSE")
    ax.axhline(np.mean(rmses), color="red", linestyle="--", label=f"Mean={np.mean(rmses):.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{i}" for i in sample_idxs], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Val sample")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Cond-DDPM ({args.cond}) — per-sample RMSE on 10 val samples")
    ax.legend()
    plt.tight_layout()
    summary_path = os.path.join(args.out_dir, "rmse_summary.png")
    plt.savefig(summary_path, dpi=140)
    plt.close(fig)
    print(f"\nPlots saved to: {args.out_dir}")
    print(f"Summary bar chart: {summary_path}")


if __name__ == "__main__":
    main()
