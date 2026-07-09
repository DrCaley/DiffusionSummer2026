"""
batch_white_voronoi_eddy.py  –  White vs Voronoi vs Eddy-Aware RePaint (r=1).

Runs RePaint for white_noise, voronoi, and eddy_aware models across N seeds
and saves a 1×4 composite PNG (GT | White | Voronoi | Eddy) plus summary.txt.

Usage:
    python "Colored Noise Test/batch_white_voronoi_eddy.py" \\
        --pickle /root/model_pink_noise/data.pickle \\
        --out_dir "Colored Noise Test/outputs/white_voronoi_eddy_r1"
"""

import argparse
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "utils"))
sys.path.insert(0, os.path.join(_HERE, "white_noise"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset       import OceanCurrentDataset
from repaint_infer import biased_walk_path
from repaint_model import Repaint

import inspect

# Eddy-aware and Voronoi models live outside Colored Noise Test/
_EDDY_DIR    = os.path.join(_ROOT, "Eddy Aware Noise")
_VORONOI_DIR = os.path.join(_ROOT, "Voronoi Noise")


# ── Custom repaint ────────────────────────────────────────────────────────────

@torch.no_grad()
def repaint(
    model, diffusion, x0_known, path_mask, land_mask,
    r=1, device="cpu", stride=1,
):
    model.eval()
    H, W = x0_known.shape[1:]

    x0_known  = x0_known.unsqueeze(0).to(device)
    known_t   = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t    = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t   = 1.0 - land_t

    dummy = torch.zeros(1, 2, H, W, device=device)
    mn_sig = inspect.signature(diffusion._make_noise)
    if len(mn_sig.parameters) >= 2:
        xt = diffusion._make_noise(dummy, diffusion.T - 1)
    else:
        xt = diffusion._make_noise(dummy)
    xt = xt * ocean_t

    T         = diffusion.T
    timesteps = list(range(0, T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        for j in range(r):
            xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)

            t_prev_tensor = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_known, _   = diffusion.q_sample(x0_known, t_prev_tensor)

            xt_merged = known_t * xt_known + (1.0 - known_t) * xt_unknown
            xt_merged = xt_merged * ocean_t

            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int)
                xt = xt * ocean_t
            else:
                xt = xt_merged

    return xt.squeeze(0).cpu()


# ── Model loading ─────────────────────────────────────────────────────────────

# Each entry: (key, display_label, checkpoint_dir, diffusion_dir)
MODELS = [
    ("white",   "White",   os.path.join(_HERE, "white_noise"),   os.path.join(_HERE, "white_noise")),
    ("voronoi", "Voronoi", _VORONOI_DIR,                          _VORONOI_DIR),
    ("eddy",    "Eddy",    _EDDY_DIR,                             _EDDY_DIR),
]


def load_model_and_diffusion(ckpt_dir: str, diff_dir: str, device: str,
                              land_mask_np: np.ndarray | None = None):
    """Load a Repaint model + matching DDPM.  Handles white/eddy/voronoi constructors."""
    ckpt_path = os.path.join(ckpt_dir, "checkpoints", "best_model.pt")
    diff_path = os.path.join(diff_dir, "diffusion.py")

    mod_name = os.path.basename(diff_dir).replace(" ", "_")
    spec     = importlib.util.spec_from_file_location(f"diff_{mod_name}", diff_path)
    diff_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diff_mod)

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    noise_std = ckpt.get("noise_std", 1.0)

    model = Repaint(in_ch=2, base_ch=64, time_dim=256).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Determine DDPM constructor signature
    import inspect as _ins
    sig = _ins.signature(diff_mod.DDPM.__init__)
    kwargs = dict(T=1000, device=device, noise_std=noise_std)

    if "sigma_map" in sig.parameters:
        # Eddy-aware: load sigma_map from checkpoint
        sigma_map = ckpt.get("sigma_map")
        if sigma_map is None:
            sigma_path = os.path.join(ckpt_dir, "checkpoints", "sigma_map.pt")
            sigma_map = torch.load(sigma_path, map_location=device, weights_only=False)
        kwargs["sigma_map"] = sigma_map.to(device)

    if "land_mask_np" in sig.parameters:
        kwargs["land_mask_np"] = land_mask_np

    if "n_seeds" in sig.parameters:
        kwargs["n_seeds"] = ckpt.get("n_seeds", 50)

    diffusion = diff_mod.DDPM(**kwargs)
    return model, diffusion, ckpt


def rmse_val(pred: np.ndarray, true: np.ndarray, ocean_mask: np.ndarray) -> float:
    diff = (pred - true)[:, ocean_mask]
    return float(np.sqrt((diff ** 2).mean()))


def plot_field(ax, u, v, land_mask, title, vmax=None):
    land_mask = np.rot90(land_mask, k=3)
    u_r       = np.rot90(u, k=3)
    v_r       = np.rot90(v, k=3)
    u         =  v_r
    v         = -u_r
    H, W      = land_mask.shape
    step = 2
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    mq     = np.sqrt(uq**2 + vq**2)
    mask   = ~land_mask[::step, ::step]
    if vmax is None:
        vmax = float(np.nanpercentile(mq[mask], 98)) if mask.any() else 1.0
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
        cmap="cool", clim=(0, vmax), scale=12, width=0.003, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    return vmax


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",    required=True)
    p.add_argument("--out_dir",   default=None)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--r",         type=int, default=1)
    p.add_argument("--stride",    type=int, default=10)
    p.add_argument("--n_steps",   type=int, default=150)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device    : {device}")
    print(f"r         : {args.r}")
    print(f"n_samples : {args.n_samples}")

    out_dir = args.out_dir or os.path.join(_HERE, "outputs", "white_voronoi_eddy_r1")
    os.makedirs(out_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds         = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = ds.land_mask.numpy()
    ocean_mask = ~land_mask

    # ── Pre-load models ───────────────────────────────────────────────────────
    print("Loading models...")
    loaded = {}
    for key, label, ckpt_dir, diff_dir in MODELS:
        print(f"  {label:10s} ...")
        model, diffusion, ckpt = load_model_and_diffusion(
            ckpt_dir, diff_dir, device, land_mask_np=land_mask
        )
        loaded[key] = (model, diffusion, label,
                       ckpt.get("epoch", "?"),
                       ckpt.get("val_loss", float("nan")))
        print(f"    epoch={loaded[key][3]}  val={loaded[key][4]:.5f}")

    all_rmses = {key: [] for key, *_ in MODELS}

    # ── Per-sample loop ───────────────────────────────────────────────────────
    for i in range(args.n_samples):
        sample_seed = args.seed + i
        x0_norm    = ds[i]
        x0_np      = x0_norm.numpy()

        path_mask = biased_walk_path(land_mask, n_steps=args.n_steps, seed=sample_seed)
        x0_known  = x0_norm.clone()
        x0_known[:, ~path_mask] = 0.0

        preds = {}
        rmses = {}
        for key, label, ckpt_dir, diff_dir in MODELS:
            model, diffusion = loaded[key][0], loaded[key][1]
            pred = repaint(
                model, diffusion, x0_known,
                path_mask=path_mask,
                land_mask=land_mask,
                r=args.r,
                device=device,
                stride=args.stride,
            )
            pred_np = pred.numpy()
            pred_np[:, land_mask] = 0.0
            preds[key] = pred_np
            rmses[key] = rmse_val(pred_np, x0_np, ocean_mask)
            all_rmses[key].append(rmses[key])

        rmse_str = "  ".join(f"{label[:5].lower()}={rmses[k]:.4f}"
                             for k, label, *_ in MODELS)
        print(f"  sample {i:2d}: {rmse_str}")

        # ── Figure: 1 row × 4 cols (GT | White | Voronoi | Eddy) ─────────────
        fig, axes = plt.subplots(1, 4, figsize=(28, 7))
        fig.suptitle(
            f"White vs Voronoi vs Eddy — sample {i}, seed {sample_seed}\n"
            f"stride={args.stride}  r={args.r}  path_steps={args.n_steps}",
            fontsize=12,
        )

        gt_speed = np.sqrt(x0_np[0]**2 + x0_np[1]**2)
        vmax     = float(np.nanpercentile(gt_speed[ocean_mask], 98))

        # Ground truth + path overlay
        plot_field(axes[0], x0_np[0], x0_np[1], land_mask, "Ground Truth", vmax=vmax)
        path_rot     = np.rot90(path_mask, k=3)
        path_overlay = np.ma.masked_where(~path_rot, np.ones(path_rot.shape, dtype=float))
        axes[0].imshow(
            path_overlay, origin="lower", cmap="autumn", alpha=0.45,
            extent=[-0.5, land_mask.shape[0] - 0.5, -0.5, land_mask.shape[1] - 0.5],
            zorder=1,
        )

        # Model panels
        for ax, (key, label, *_) in zip(axes[1:], MODELS):
            plot_field(ax, preds[key][0], preds[key][1],
                       land_mask, f"{label}\nRMSE={rmses[key]:.5f}", vmax=vmax)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"result_{i+1:02d}.png")
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"    saved {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = [
        "White vs Voronoi vs Eddy-Aware RePaint comparison",
        f"N samples       : {args.n_samples}",
        f"resample r      : {args.r}",
        f"stride          : {args.stride}",
        f"path_steps      : {args.n_steps}",
        "",
        f"{'Model':<12}  epoch  val_loss   Mean RMSE      Std      Min      Max",
        "-" * 68,
    ]
    for key, label, *_ in MODELS:
        _, _, _, ep, val = loaded[key]
        rs = all_rmses[key]
        lines.append(
            f"{label:<12}  {str(ep):>5}  {val:.5f}   "
            f"{np.mean(rs):.5f}  {np.std(rs):.5f}  {np.min(rs):.5f}  {np.max(rs):.5f}"
        )

    lines += ["", "Per-sample RMSE:"]
    header = f"  {'sample':>6}  " + "  ".join(f"{label[:8]:>8}" for _, label, *_ in MODELS)
    lines.append(header)
    for i in range(args.n_samples):
        row_vals = "  ".join(f"{all_rmses[k][i]:>8.5f}" for k, *_ in MODELS)
        lines.append(f"  {i:>6}  {row_vals}")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved: {summary_path}")
    print("\n" + "\n".join(lines[5:]))


if __name__ == "__main__":
    main()
