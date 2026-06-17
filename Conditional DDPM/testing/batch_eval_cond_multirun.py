"""
Multi-run evaluation of eps + 4 conditioned DDPM models (path, voronoi, both, path_field).

For each of N randomly chosen val samples:
  - Generate one robot path (biased walk, deterministic per sample)
  - Run each model K times with different noise seeds
  - Record RMSE per run
  - Save a 3×4 figure (all 12 panels used):
      Row 0: Ground Truth | Robot Path | eps avg-quiver       | eps std-heatmap
      Row 1: path-cond avg | path-cond std | voronoi-cond avg | voronoi-cond std
      Row 2: both-cond avg | both-cond std | path_field avg   | path_field std

A summary.txt is written with per-model mean ± std RMSE over all sample×run
combinations.  A versioned results/eval_cond_multirun/run_vN/ directory is
created each invocation.

Usage (from workspace root):
    python3 "Conditional DDPM/batch_eval_cond_multirun.py"
    python3 "Conditional DDPM/batch_eval_cond_multirun.py" \\
        --n_samples 10 --n_runs 10 --resample 10 --seed 42 \\
        --eps_ckpt "Model Parameters/Loss Function/models/loss_comparison_800/model_ddpm_eps_gaussian_cosine.pt"
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.normpath(os.path.join(_HERE, ".."))
_DDPM_ROOT = os.path.join(_ROOT, "DDPM")

for _p in (
    _HERE, _ROOT, _DDPM_ROOT,
    os.path.join(_DDPM_ROOT, "model"),
    os.path.join(_DDPM_ROOT, "testing", "repaint"),
    os.path.join(_ROOT, "Voronoi", "model"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset        import OceanCurrentDataset
from paths          import biased_walk_path
from voronoi_model  import VoronoiLayer
from cond_model     import CondUNet
from cond_diffusion import CondDDPM
from diffusion      import DDPM
from model          import UNet
from repaint_infer  import repaint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_LABELS = ["eps", "path", "voronoi", "both", "path_field"]

# Cond-channel count per conditioning type
_COND_IN_CH = {"path": 1, "voronoi": 3, "both": 4, "path_field": 3}

# 3×4 = 12 panels (all used):
#  0=GT  1=Path  2=eps-avg  3=eps-std
#  4=path-avg 5=path-std  6=vor-avg 7=vor-std
#  8=both-avg 9=both-std 10=path_field-avg 11=path_field-std

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _next_versioned_dir(base: str, prefix: str) -> str:
    v = 1
    while True:
        c = os.path.join(base, f"{prefix}_v{v}")
        if not os.path.exists(c):
            return c
        v += 1


def _next_versioned_file(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    v = 2
    while True:
        c = f"{root}_v{v}{ext}"
        if not os.path.exists(c):
            return c
        v += 1


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
_BW_CMAP   = plt.matplotlib.colors.ListedColormap(["white", "black"])
_NONE_CMAP = plt.matplotlib.colors.ListedColormap(["none",  "black"])


def _ext(W, H):
    return [-0.5, H - 0.5, -0.5, W - 0.5]


def _plot_quiver(ax, u, v, land_d, title, clim=None, step=2, cmap="cool"):
    """u, v already transposed to display coords (W, H)."""
    W, H = u.shape
    ext = _ext(W, H)
    ax.imshow(land_d, origin="lower", cmap=_BW_CMAP, extent=ext, aspect="auto", zorder=0)
    yq, xq = np.mgrid[0:W:step, 0:H:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    mq     = np.sqrt(uq**2 + vq**2)
    mask   = ~np.isnan(uq) & ~land_d[::step, ::step]
    if mask.any():
        vmax = clim if clim is not None else (float(np.nanpercentile(mq[mask], 98)) or 1.0)
        q = ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
                      cmap=cmap, clim=(0, vmax), scale=12, width=0.003, zorder=2)
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.65)
    ax.set_title(title, fontsize=6, pad=3)
    ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6)
    ax.tick_params(labelsize=5)


def _plot_heatmap(ax, std_spd, land_d, title):
    """std_spd already transposed; NaN on land."""
    W, H = land_d.shape
    ext  = _ext(W, H)
    disp = std_spd.copy()
    disp[land_d] = np.nan
    ax.imshow(land_d, origin="lower", cmap=_BW_CMAP, extent=ext, aspect="auto", zorder=0)
    vmax = float(np.nanpercentile(disp, 99)) if not np.all(np.isnan(disp)) else 1.0
    im = ax.imshow(disp, origin="lower", cmap="hot_r", extent=ext,
                   aspect="auto", zorder=1, vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Speed std", shrink=0.65)
    ax.set_title(title, fontsize=6, pad=3)
    ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6)
    ax.tick_params(labelsize=5)


def _save_sample_figure(
    sample_idx, path_seed, n_runs,
    x0_np,          # (2, H, W)
    path_mask,       # (H, W) bool
    land_mask_np,    # (H, W) bool
    avg_preds,       # label -> (2, H, W) mean
    std_speeds,      # label -> (H, W) pixel std of speed
    mean_rmses,      # label -> float
    std_rmses,       # label -> float
    active_labels,
    out_path,
):
    T_ = lambda a: a.T   # (H,W) -> (W,H) for imshow origin='lower'

    land_d = T_(land_mask_np)
    path_d = T_(path_mask)

    speed_gt = np.sqrt(x0_np[0]**2 + x0_np[1]**2)
    speed_gt[land_mask_np] = np.nan
    clim = float(np.nanpercentile(speed_gt[~land_mask_np], 98)) if (~land_mask_np).any() else 1.0

    fig, axes = plt.subplots(3, 4, figsize=(4*6, 3*5), constrained_layout=True)
    ax = axes.flatten()

    # Panel 0: Ground Truth
    _plot_quiver(ax[0], T_(x0_np[0]), T_(x0_np[1]), land_d,
                 f"Ground Truth  (val {sample_idx})", clim=clim)

    # Panel 1: Robot Path
    W_d, H_d = land_d.shape
    ext = _ext(W_d, H_d)
    ax[1].imshow(land_d, origin="lower", cmap=_BW_CMAP, extent=ext, aspect="auto", zorder=0)
    path_disp = np.full(land_d.shape, np.nan)
    path_disp[path_d] = 1.0
    ax[1].imshow(path_disp, origin="lower", cmap="Reds",
                 extent=ext, aspect="auto", zorder=1, vmin=0, vmax=1)
    ax[1].set_title(f"Robot Path  ({int(path_mask.sum())} cells  seed={path_seed})",
                    fontsize=6, pad=3)
    ax[1].set_xlabel("X", fontsize=6); ax[1].set_ylabel("Y", fontsize=6)
    ax[1].tick_params(labelsize=5)
    ax[1].legend(handles=[
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
        mpatches.Patch(facecolor="#d62728", label="Path"),
        mpatches.Patch(facecolor="black", label="Land"),
    ], loc="upper right", fontsize=5)

    # Panels 2-11: pairs (avg quiver, std heatmap) per model
    label_order = ["eps", "path", "voronoi", "both", "path_field"]
    for mi, label in enumerate(label_order):
        if label not in active_labels:
            continue
        avg_i = 2 + mi * 2
        hm_i  = avg_i + 1
        if avg_i >= 12:
            break

        avg_pred = avg_preds[label]   # (2, H, W)
        std_spd  = std_speeds[label]  # (H, W)
        mu_r = mean_rmses[label]
        sd_r = std_rmses[label]

        _plot_quiver(ax[avg_i], T_(avg_pred[0]), T_(avg_pred[1]), land_d,
                     f"{label}  (avg {n_runs} runs)\nRMSE={mu_r:.4f} ± {sd_r:.4f}",
                     clim=clim)
        _plot_heatmap(ax[hm_i], T_(std_spd), land_d, f"{label}  run variance")

    # blank unused panels
    for i in range(2 + len(label_order) * 2, 12):
        ax[i].axis("off")

    fig.suptitle(
        f"Val sample {sample_idx}  —  eps + 4 conditioned models  "
        f"({n_runs} runs, same robot path)",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-run eval: eps + path/voronoi/both conditioned models."
    )
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument(
        "--eps_ckpt", default=None,
        help="Path to eps .pt checkpoint. Defaults to "
             "Model Parameters/Loss Function/all_models/model_ddpm_eps_gaussian_cosine.pt",
    )
    p.add_argument("--cond_dir", default=None,
                   help="Dir containing checkpoints_path/, checkpoints_voronoi/, "
                        "checkpoints_both/. Defaults to <this dir>.")
    p.add_argument("--n_samples",   type=int, default=50)
    p.add_argument("--n_runs",      type=int, default=10)
    p.add_argument("--path_steps",  type=int, default=150)
    p.add_argument("--resample",    type=int, default=10)
    p.add_argument("--seed",        type=int, default=42,
                   help="Master RNG seed for sample selection and path seeds.")
    p.add_argument(
        "--out_dir", default=None,
        help="Parent dir for output. A versioned sub-dir is created automatically. "
             "Defaults to <this dir>/results/eval_cond_multirun.",
    )
    p.add_argument(
        "--labels", nargs="+", default=MODEL_LABELS, choices=MODEL_LABELS,
        help="Subset of models to evaluate (default: all 4).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Resolve directories ────────────────────────────────────────────────
    cond_dir = args.cond_dir or _HERE

    parent_out = args.out_dir or os.path.join(_HERE, "results", "eval_cond_multirun")
    os.makedirs(parent_out, exist_ok=True)
    out_dir = _next_versioned_dir(parent_out, "run")
    os.makedirs(out_dir)
    print(f"Output directory: {out_dir}\n")

    # ── Resolve data.pickle ────────────────────────────────────────────────
    pickle_path = args.pickle
    if not os.path.isabs(pickle_path) and not os.path.exists(pickle_path):
        pickle_path = os.path.join(_ROOT, pickle_path)

    val_ds       = OceanCurrentDataset(pickle_path, split=1)
    land_mask_np = val_ds.land_mask.numpy()
    H, W         = land_mask_np.shape

    n_val = len(val_ds)
    rng   = np.random.default_rng(args.seed)
    sample_indices = sorted(
        rng.choice(n_val, size=min(args.n_samples, n_val), replace=False).tolist()
    )
    path_seeds = {idx: int(rng.integers(0, 99999)) for idx in sample_indices}
    run_seeds_for = {
        idx: [int(rng.integers(0, 999999)) for _ in range(args.n_runs)]
        for idx in sample_indices
    }

    print(f"Val set size: {n_val}")
    print(f"Seed        : {args.seed}")
    print(f"Samples     : {sample_indices}\n")

    # ── Load eps model ─────────────────────────────────────────────────────
    _eps_candidates = [
        os.path.join(_ROOT, "Model Parameters", "Loss Function",
                     "models", "loss_comparison_800", "model_ddpm_eps_gaussian_cosine.pt"),
        os.path.join(_ROOT, "Model Parameters", "Loss Function",
                     "models", "loss_comparison_400", "model_ddpm_eps_gaussian_cosine.pt"),
        os.path.join(_ROOT, "Model Parameters", "Loss Function",
                     "all_models", "model_ddpm_eps_gaussian_cosine.pt"),
        os.path.join(_ROOT, "DDPM", "models", "best_model.pt"),
    ]
    eps_ckpt_default = next((p for p in _eps_candidates if os.path.exists(p)), _eps_candidates[0])
    eps_ckpt_path = args.eps_ckpt or eps_ckpt_default
    if not os.path.isabs(eps_ckpt_path) and not os.path.exists(eps_ckpt_path):
        eps_ckpt_path = os.path.join(_ROOT, eps_ckpt_path)

    active_labels = []
    eps_model = None
    eps_diff  = None
    if "eps" in args.labels:
        if os.path.exists(eps_ckpt_path):
            ck = torch.load(eps_ckpt_path, map_location=device, weights_only=False)
            ca = ck.get("args", {})
            eps_model = UNet(in_ch=2, base_ch=ca.get("base_ch", 64),
                             time_dim=ca.get("time_dim", 256)).to(device)
            eps_model.load_state_dict(ck["model"])
            eps_model.eval()
            eps_diff = DDPM(T=ca.get("T", 1000),
                            beta_schedule=ca.get("schedule", "cosine"),
                            device=device)
            active_labels.append("eps")
            print(f"  Loaded 'eps'  epoch={ck.get('epoch','?')}  "
                  f"val_loss={ck.get('val_loss', float('nan')):.5f}")
        else:
            print(f"  [SKIP] eps: not found at {eps_ckpt_path}")

    # ── Load conditioned models ────────────────────────────────────────────
    cond_models = {}
    cond_diffs  = {}
    for cond_name in ["path", "voronoi", "both", "path_field"]:
        if cond_name not in args.labels:
            continue
        ck_path = os.path.join(
            cond_dir, f"checkpoints_{cond_name}",
            f"best_cond_ddpm_{cond_name}_cosine.pt",
        )
        # Fallback: unified models/ subdirectory (server layout)
        if not os.path.exists(ck_path):
            ck_path = os.path.join(
                cond_dir, "models",
                f"best_cond_ddpm_{cond_name}_cosine.pt",
            )
        if not os.path.exists(ck_path):
            print(f"  [SKIP] {cond_name}: not found at {ck_path}")
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        ca = ck.get("args", {})
        cond_in_ch = _COND_IN_CH[cond_name]
        m = CondUNet(
            in_ch=2, cond_in_ch=cond_in_ch,
            base_ch=ca.get("base_ch", 64),
            time_dim=ca.get("time_dim", 256),
            cond_dim=ca.get("cond_dim", 256),
        ).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        cond_models[cond_name] = m
        cond_diffs[cond_name]  = CondDDPM(
            T=ca.get("T", 1000),
            beta_schedule=ca.get("schedule", "cosine"),
            device=device,
        )
        active_labels.append(cond_name)
        print(f"  Loaded '{cond_name}'  epoch={ck.get('epoch','?')}  "
              f"val_loss={ck.get('val_loss', float('nan')):.5f}")

    print(f"\n{len(active_labels)} models loaded: {active_labels}\n")

    voronoi_layer = VoronoiLayer(H=H, W=W, n_sensors=args.path_steps).to(device)

    # ── Accumulators ──────────────────────────────────────────────────────
    # all_rmses[label] = flat list of RMSE values (n_samples × n_runs)
    all_rmses = {lbl: [] for lbl in active_labels}

    total = len(sample_indices)
    for si, sample_idx in enumerate(sample_indices):
        path_seed = path_seeds[sample_idx]
        run_seeds = run_seeds_for[sample_idx]

        print(f"[{si+1:2d}/{total}] Val sample {sample_idx:4d}  path_seed={path_seed}")

        x0_true   = val_ds[sample_idx].to(device)          # (2, H, W)
        path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=path_seed)
        rows, cols = np.where(path_mask)
        path_t    = torch.from_numpy(path_mask).to(device)

        # Build voronoi conditioning once per sample (same path → same tessellation)
        rows_n = torch.tensor(rows, dtype=torch.float32, device=device) / (H-1) * 2 - 1
        cols_n = torch.tensor(cols, dtype=torch.float32, device=device) / (W-1) * 2 - 1
        sensor_pos  = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)
        flat_idx    = torch.tensor(rows * W + cols, dtype=torch.long, device=device)
        flat_idx_b  = flat_idx.unsqueeze(0).unsqueeze(0).expand(1, 2, len(rows))
        sensor_vals = torch.gather(
            x0_true.unsqueeze(0).reshape(1, 2, H*W), 2, flat_idx_b
        )
        with torch.no_grad():
            voronoi_grid = voronoi_layer.tessellate(sensor_vals, sensor_pos)  # (1,3,H,W)

        # path conditioning tensor (fixed for all runs)
        path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)
        path_cond = path_ch[None, None]   # (1,1,H,W)

        # both conditioning: voronoi (3ch) + path (1ch)
        both_cond = torch.cat([voronoi_grid, path_cond], dim=1)  # (1,4,H,W)

        # path_field conditioning: [u_path, v_path, path_mask] (3ch)
        path_field_cond = torch.stack([
            x0_true[0] * path_ch,   # u at path cells, 0 elsewhere
            x0_true[1] * path_ch,   # v at path cells, 0 elsewhere
            path_ch,
        ], dim=0).unsqueeze(0)  # (1,3,H,W)

        # RePaint shared tensors
        x0_obs = x0_true.clone()
        x0_obs[:, ~path_t] = 0.0
        x0_known_b   = x0_obs.unsqueeze(0)              # (1,2,H,W)
        path_mask_t  = path_t[None, None].float()        # (1,1,H,W)
        ocean_mask_t = torch.from_numpy(
            (~land_mask_np).astype(np.float32)
        ).to(device)[None, None]                         # (1,1,H,W)

        # per-label run accumulator
        run_preds   = {lbl: [] for lbl in active_labels}  # list of (2,H,W) np arrays
        run_rmses   = {lbl: [] for lbl in active_labels}

        for run_i, rseed in enumerate(run_seeds):
            for label in active_labels:
                model_seed = rseed + abs(hash(label)) % 100000
                torch.manual_seed(model_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(model_seed)

                with torch.no_grad():
                    if label == "eps":
                        pred = repaint(
                            eps_model, eps_diff, x0_obs,
                            path_mask, land_mask_np,
                            r=args.resample, device=device,
                        )   # (2,H,W) CPU
                        pred_np = pred.numpy()
                    else:
                        cond_map = {"path": path_cond,
                                    "voronoi": voronoi_grid,
                                    "both": both_cond,
                                    "path_field": path_field_cond}
                        pred = cond_diffs[label].repaint(
                            cond_models[label],
                            cond_map[label],
                            x0_known_b,
                            path_mask_t,
                            ocean_mask_t,
                            r=args.resample,
                        )[0]   # (2,H,W) CPU
                        pred_np = pred.cpu().numpy()

                rmse = float(np.sqrt(np.mean(
                    (pred_np[0] - x0_true[0].cpu().numpy())[~land_mask_np]**2 +
                    (pred_np[1] - x0_true[1].cpu().numpy())[~land_mask_np]**2
                )))
                run_preds[label].append(pred_np)
                run_rmses[label].append(rmse)
                all_rmses[label].append(rmse)

            preview = "  ".join(
                f"{lbl}={run_rmses[lbl][-1]:.4f}" for lbl in active_labels
            )
            print(f"    run {run_i+1:2d}/{args.n_runs}  seed={rseed}  {preview}")

        # ── Per-model averages ─────────────────────────────────────────────
        avg_preds  = {}
        std_speeds = {}
        mean_rmses = {}
        std_rmses_s = {}
        for label in active_labels:
            preds_np = np.stack(run_preds[label], axis=0)  # (R, 2, H, W)
            avg_preds[label]   = preds_np.mean(axis=0)
            speeds = np.sqrt(preds_np[:, 0]**2 + preds_np[:, 1]**2)  # (R, H, W)
            std_s  = speeds.std(axis=0)
            std_s[land_mask_np] = np.nan
            std_speeds[label]   = std_s
            mean_rmses[label]   = float(np.mean(run_rmses[label]))
            std_rmses_s[label]  = float(np.std(run_rmses[label]))

        summary_line = "  ".join(
            f"{lbl} {mean_rmses[lbl]:.4f}±{std_rmses_s[lbl]:.4f}"
            for lbl in active_labels
        )
        print(f"    SAMPLE SUMMARY → {summary_line}")

        # ── Save 3×4 figure ────────────────────────────────────────────────
        img_path = _next_versioned_file(
            os.path.join(out_dir, f"sample_{sample_idx:04d}.png")
        )
        _save_sample_figure(
            sample_idx, path_seed, args.n_runs,
            x0_true.cpu().numpy(),
            path_mask, land_mask_np,
            avg_preds, std_speeds,
            mean_rmses, std_rmses_s,
            active_labels,
            img_path,
        )
        print(f"    → {img_path}\n")

    # ── Summary report ─────────────────────────────────────────────────────
    sep  = "=" * 80
    dsep = "-" * 80
    lines = [sep,
             "CONDITIONED DDPM MULTI-RUN EVALUATION SUMMARY".center(80),
             f"n_samples={args.n_samples}  n_runs={args.n_runs}  "
             f"path_steps={args.path_steps}  resample={args.resample}  "
             f"seed={args.seed}".center(80),
             sep]

    col_w = 14
    header = f"{'model':<16}" + f"{'mean_rmse':>{col_w}}" + f"{'std_rmse':>{col_w}}" + \
             f"{'min_rmse':>{col_w}}" + f"{'max_rmse':>{col_w}}"
    lines += [header, dsep]

    for label in active_labels:
        vals = all_rmses[label]
        lines.append(
            f"{label:<16}"
            f"{np.mean(vals):>{col_w}.6f}"
            f"{np.std(vals):>{col_w}.6f}"
            f"{np.min(vals):>{col_w}.6f}"
            f"{np.max(vals):>{col_w}.6f}"
        )

    lines += [sep, ""]

    # Full detail per sample
    lines += ["PER-SAMPLE MEANS  (averaged over runs)", dsep]
    lines.append(f"{'sample':<8}" + "".join(f"  {lbl:>12}" for lbl in active_labels))
    lines.append("-" * (8 + 14 * len(active_labels)))
    for idx in sample_indices:
        row_rmses = {
            lbl: float(np.mean([
                all_rmses[lbl][si * args.n_runs + ri]
                for ri in range(args.n_runs)
            ]))
            for si, s in enumerate(sample_indices) if s == idx
            for lbl in active_labels
        }
        lines.append(
            f"{idx:<8}" + "".join(f"  {row_rmses[lbl]:>12.6f}" for lbl in active_labels)
        )

    report_path = _next_versioned_file(os.path.join(out_dir, "summary.txt"))
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # Print summary to stdout
    print("\n" + "\n".join(lines[:lines.index(dsep) + 4]))
    for label in active_labels:
        vals = all_rmses[label]
        print(f"  {label:<16}  {np.mean(vals):.6f} ± {np.std(vals):.6f}"
              f"  (min={np.min(vals):.4f}  max={np.max(vals):.4f})")
    print(sep)
    print(f"\nImages  : {out_dir}")
    print(f"Report  : {report_path}")

    # ── Summary bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    xs     = np.arange(len(sample_indices))
    width  = 0.8 / max(len(active_labels), 1)
    colors = {"eps": "steelblue", "path": "darkorange",
               "voronoi": "seagreen", "both": "mediumpurple",
               "path_field": "crimson"}
    for i, label in enumerate(active_labels):
        per_sample_means = [
            float(np.mean(all_rmses[label][si*args.n_runs:(si+1)*args.n_runs]))
            for si in range(len(sample_indices))
        ]
        per_sample_stds = [
            float(np.std(all_rmses[label][si*args.n_runs:(si+1)*args.n_runs]))
            for si in range(len(sample_indices))
        ]
        offset = (i - len(active_labels)/2 + 0.5) * width
        ax.bar(xs + offset, per_sample_means, width, yerr=per_sample_stds,
               label=f"{label} (μ={np.mean(all_rmses[label]):.4f})",
               color=colors.get(label, "gray"), capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in sample_indices], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Val sample index")
    ax.set_ylabel("RMSE")
    ax.set_title(f"eps vs path vs voronoi vs both vs path_field — per-sample RMSE "
                 f"(mean ± std over {args.n_runs} runs)")
    ax.legend()
    plt.tight_layout()
    bar_path = _next_versioned_file(os.path.join(out_dir, "rmse_summary_bar.png"))
    fig.savefig(bar_path, dpi=130)
    plt.close(fig)
    print(f"Bar chart: {bar_path}")


if __name__ == "__main__":
    main()
