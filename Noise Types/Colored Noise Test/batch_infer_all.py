"""
batch_infer_all.py  –  Batch RePaint inference for all 6 noise models.

For each of N_SAMPLES test samples the script runs RePaint with all 6 models
and saves a 2×4 composite PNG plus a summary.txt.

Layout per image (2 rows × 4 columns):
  Row 0: Ground Truth | White       | Pink       | Red
  Row 1: RMSE chart   | Pink (full) | Red (full) | Annealed

Usage:
    python "Colored Noise Test/batch_infer_all.py" \\
        --pickle  /root/model_pink_noise/data.pickle \\
        --ckpt    best          # or epoch100
        --out_dir "Colored Noise Test/outputs/batch_best"
"""

import argparse
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "utils"))
sys.path.insert(0, os.path.join(_HERE, "white_noise"))   # repaint_infer lives here

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset       import OceanCurrentDataset
from repaint_infer import biased_walk_path
from repaint_model import Repaint

# ── Custom repaint with diffusion-appropriate initialization ─────────────────

import inspect

@torch.no_grad()
def repaint(
    model, diffusion, x0_known, path_mask, land_mask,
    r=10, device="cpu", stride=1,
):
    """
    RePaint inference.  Identical to repaint_infer.repaint but initializes
    x_T using the diffusion's own noise generator so that annealed (and
    other colored-noise) models start from the correct prior at t=T.
    """
    model.eval()
    H, W = x0_known.shape[1:]

    x0_known  = x0_known.unsqueeze(0).to(device)
    known_t   = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t    = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t   = 1.0 - land_t

    # Initialise x_T with the diffusion's prior noise.
    # Annealed diffusion: _make_noise(x, t_int) — use t=T-1 (reddest prior).
    # All others:         _make_noise(x)        — uniform colored prior.
    dummy = torch.zeros(1, 2, H, W, device=device)
    mn_sig = inspect.signature(diffusion._make_noise)
    if len(mn_sig.parameters) >= 2:           # annealed: takes t_int
        xt = diffusion._make_noise(dummy, diffusion.T - 1)
    else:                                      # white / pink / red (full)
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


# ── Model catalogue ──────────────────────────────────────────────────────────

MODELS = [
    # (subdir,             display_label,    row, col)
    ("white_noise",        "White",          0,   1),
    ("pink_noise",         "Pink",           0,   2),
    ("red_noise",          "Red",            0,   3),
    ("pink_noise_full",    "Pink (full)",    1,   1),
    ("red_noise_full",     "Red (full)",     1,   2),
    ("annealed_noise",     "Annealed",       1,   3),
]

CKPT_NAMES = {
    "best":       "best_model.pt",
    "epoch100":   "ckpt_epoch0100.pt",
    "best_by_100": None,   # resolved per-model (see resolve_ckpt_file)
}


def resolve_ckpt_file(subdir: str, ckpt_mode: str) -> str:
    """
    For 'best_by_100': use best_model.pt if its saved epoch <= 100,
    otherwise fall back to ckpt_epoch0100.pt.
    For all other modes use the static CKPT_NAMES mapping.
    """
    if ckpt_mode != "best_by_100":
        return CKPT_NAMES[ckpt_mode]
    best_path = os.path.join(_HERE, subdir, "checkpoints", "best_model.pt")
    try:
        meta = torch.load(best_path, map_location="cpu", weights_only=False)
        if meta.get("epoch", 9999) <= 100:
            return "best_model.pt"
    except Exception:
        pass
    return "ckpt_epoch0100.pt"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model_and_diffusion(subdir: str, ckpt_file: str, device: str):
    ckpt_path = os.path.join(_HERE, subdir, "checkpoints", ckpt_file)
    diff_path = os.path.join(_HERE, subdir, "diffusion.py")

    spec     = importlib.util.spec_from_file_location(f"diff_{subdir}", diff_path)
    diff_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diff_mod)

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    noise_std = ckpt.get("noise_std", 1.0)

    model = Repaint(in_ch=2, base_ch=64, time_dim=256).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = diff_mod.DDPM(T=1000, device=device, noise_std=noise_std)
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
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    return vmax


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",    required=True)
    p.add_argument("--ckpt",      default="best", choices=list(CKPT_NAMES.keys()),
                   help="Which checkpoint to use: 'best', 'epoch100', or 'best_by_100'")
    p.add_argument("--out_dir",   default=None)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--r",         type=int, default=10)
    p.add_argument("--stride",    type=int, default=10)
    p.add_argument("--n_steps",   type=int, default=150)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Ckpt   : {args.ckpt}")

    out_dir = args.out_dir or os.path.join(_HERE, "outputs", f"batch_{args.ckpt}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds         = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = ds.land_mask.numpy()
    ocean_mask = ~land_mask

    # ── Pre-load all models ───────────────────────────────────────────────────
    print("Loading models...")
    loaded = {}          # subdir -> (model, diffusion, label, epoch, val_loss, ckpt_file_used)
    for subdir, label, _, _ in MODELS:
        ckpt_file = resolve_ckpt_file(subdir, args.ckpt)
        print(f"  {label:15s} ({subdir}) ... [{ckpt_file}]")
        model, diffusion, ckpt = load_model_and_diffusion(subdir, ckpt_file, device)
        loaded[subdir] = (model, diffusion, label,
                          ckpt.get("epoch", "?"),
                          ckpt.get("val_loss", float("nan")),
                          ckpt_file)
        print(f"    epoch={loaded[subdir][3]}  val={loaded[subdir][4]:.5f}")

    # ── Summary bookkeeping ───────────────────────────────────────────────────
    all_rmses = {s: [] for s, *_ in MODELS}

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
        for subdir, label, _, _ in MODELS:
            model, diffusion = loaded[subdir][0], loaded[subdir][1]
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
            preds[subdir] = pred_np
            rmses[subdir] = rmse_val(pred_np, x0_np, ocean_mask)
            all_rmses[subdir].append(rmses[subdir])

        rmse_str = "  ".join(f"{s[:3]}={rmses[s]:.4f}" for s, *_ in MODELS)
        print(f"  sample {i:2d}: {rmse_str}")

        # ── Figure: 2 rows × 4 cols ───────────────────────────────────────────
        fig, axes = plt.subplots(2, 4, figsize=(28, 14))
        fig.suptitle(
            f"All-model comparison — sample {i}, seed {sample_seed}\n"
            f"ckpt={args.ckpt}  stride={args.stride}  r={args.r}  path_steps={args.n_steps}",
            fontsize=12,
        )

        gt_speed = np.sqrt(x0_np[0]**2 + x0_np[1]**2)
        vmax     = float(np.nanpercentile(gt_speed[ocean_mask], 98))

        # Ground truth (row 0, col 0)
        plot_field(axes[0, 0], x0_np[0], x0_np[1], land_mask, "Ground Truth", vmax=vmax)
        path_rot     = np.rot90(path_mask, k=3)
        path_overlay = np.ma.masked_where(~path_rot, np.ones(path_rot.shape, dtype=float))
        axes[0, 0].imshow(
            path_overlay, origin="lower", cmap="autumn", alpha=0.45,
            extent=[-0.5, land_mask.shape[0] - 0.5, -0.5, land_mask.shape[1] - 0.5],
            zorder=1,
        )

        # Model panels
        for subdir, label, row, col in MODELS:
            r_val = rmses[subdir]
            plot_field(axes[row, col], preds[subdir][0], preds[subdir][1],
                       land_mask, f"{label}\nRMSE={r_val:.5f}", vmax=vmax)

        # RMSE bar chart (row 1, col 0)
        ax_bar  = axes[1, 0]
        b_labels = [label for _, label, _, _ in MODELS]
        b_vals   = [rmses[s] for s, *_ in MODELS]
        colors   = ["#4c9be8", "#e87d4c", "#c44ce8", "#4ce8a0", "#e84c6e", "#e8c94c"]
        bars = ax_bar.bar(b_labels, b_vals, color=colors, edgecolor="black", width=0.6)
        for bar, val in zip(bars, b_vals):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(b_vals) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8,
            )
        ax_bar.set_ylabel("RMSE (normalised)")
        ax_bar.set_title("RMSE comparison")
        ax_bar.set_ylim(0, max(b_vals) * 1.2)
        ax_bar.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax_bar.set_axisbelow(True)
        ax_bar.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"result_{i+1:02d}_all_models.png")
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"    saved {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    ckpt_desc = "  ".join(
        f"{label}={loaded[s][5]}" for s, label, _, _ in MODELS
    )
    lines = [
        f"All-model RePaint batch inference",
        f"Checkpoint type : {args.ckpt}",
        f"Checkpoints     : {ckpt_desc}",
        f"N samples       : {args.n_samples}",
        f"resample r      : {args.r}",
        f"stride          : {args.stride}",
        f"path_steps      : {args.n_steps}",
        "",
        f"{'Model':<16}  epoch  val_loss   Mean RMSE      Std      Min      Max",
        "-" * 70,
    ]
    for subdir, label, _, _ in MODELS:
        _, _, _, ep, val, _ = loaded[subdir]
        rs  = all_rmses[subdir]
        lines.append(
            f"{label:<16}  {str(ep):>5}  {val:.5f}   "
            f"{np.mean(rs):.5f}  {np.std(rs):.5f}  {np.min(rs):.5f}  {np.max(rs):.5f}"
        )

    lines += ["", "Per-sample RMSE:"]
    header = f"  {'sample':>6}  " + "  ".join(f"{label[:8]:>8}" for _, label, _, _ in MODELS)
    lines.append(header)
    for i in range(args.n_samples):
        row_vals = "  ".join(f"{all_rmses[s][i]:>8.5f}" for s, *_ in MODELS)
        lines.append(f"  {i:>6}  {row_vals}")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written to {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
