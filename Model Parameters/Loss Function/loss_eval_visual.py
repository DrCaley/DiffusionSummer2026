"""
Evaluate all 7 loss-function models on 50 random validation samples.
For each sample, runs each model 10 times with different diffusion seeds
but the same robot path, then:

  1. Saves a 4×4 image per sample:
       Col 0: Ground truth | Robot path
       Col 1: model_eps avg   | model_eps variance heatmap
       Col 2: model_curl_div avg | heatmap
       Col 3: model_spectral avg | heatmap
       Col 4+: ... (only 7 models fit in a 4×4 = 16 panels grid)

     Actual layout (16 panels, 2 per model + 2 fixed):
       [0]  Ground truth        [1]  Robot path
       [2]  eps avg             [3]  eps heatmap
       [4]  curl_div avg        [5]  curl_div heatmap
       [6]  spectral avg        [7]  spectral heatmap
       [8]  okubo_weiss avg     [9]  okubo_weiss heatmap
       [10] wasserstein avg     [11] wasserstein heatmap
       [12] stream_fn avg       [13] stream_fn heatmap
       [14] strain_rate avg     [15] strain_rate heatmap

  2. Writes a text report with:
       - Summary table: mean of each evaluation metric for each model
         (averaged over 50 samples × 10 runs)
       - Full detail table at the end (one row per model × sample × run)

Outputs:
    results/eval_visual_loss/sample_{idx:03d}.png  (50 images)
    results/loss_eval_visual_summary.txt

Usage (from workspace root or Loss Function/):
    python3 "Model Parameters/Loss Function/loss_eval_visual.py"
    python3 "Model Parameters/Loss Function/loss_eval_visual.py" \\
        --n_samples 50 --n_runs 10 --path_steps 150 --resample 10
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_MODEL_PARAMS = os.path.dirname(_HERE)
_ROOT         = os.path.dirname(_MODEL_PARAMS)

sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# Also search NoiseSchedule/ for repaint_infer if not at root
_NS_DIR = os.path.join(_MODEL_PARAMS, "NoiseSchedule")
for _d in (_NS_DIR,):
    if os.path.isfile(os.path.join(_d, "repaint_infer.py")) and _d not in sys.path:
        sys.path.insert(0, _d)

from dataset        import OceanCurrentDataset
from loss_functions import (
    curl_div_loss, spectral_loss, okubo_weiss_loss,
    wasserstein_loss, stream_function_loss, strain_rate_loss,
    LOSS_MODES,
)

# Import UNet + DDPM from root (loss-function training used them)
try:
    from model    import UNet
    from diffusion import DDPM
except ImportError:
    from DDPM.model.model     import UNet       # fallback
    from DDPM.model.diffusion import DDPM

try:
    from repaint_infer import biased_walk_path, repaint
except ImportError:
    from NoiseSchedule.repaint_infer import biased_walk_path, repaint

try:
    from geomloss import SamplesLoss
    _sinkhorn      = SamplesLoss("sinkhorn", p=2, blur=0.05)
    _HAS_GEOMLOSS  = True
except ImportError:
    _HAS_GEOMLOSS  = False
    print("  [warn] geomloss not installed — wasserstein metric will be NaN")

# The 7 model labels (same order as LOSS_MODES minus eps is the model name,
# but for the *model checkpoint* names)
MODEL_LABELS = [
    "eps",
    "curl_div",
    "spectral",
    "okubo_weiss",
    "wasserstein",
    "stream_function",
    "strain_rate",
]

METRIC_LABELS = {
    "eps":             "RMSE (field)",
    "curl_div":        "Curl/Div RMSE",
    "spectral":        "Spectral RMSE",
    "okubo_weiss":     "Okubo-Weiss RMSE",
    "wasserstein":     "Wasserstein",
    "stream_function": "Stream fn. RMSE",
    "strain_rate":     "Strain rate RMSE",
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--ckpt_dir",   default=None,
                   help="Dir with the 7 .pt files. "
                        "Defaults to <this_script_dir>/results/")
    p.add_argument("--n_samples",  type=int, default=50,
                   help="Number of random validation samples to evaluate.")
    p.add_argument("--n_runs",     type=int, default=10,
                   help="Inference runs per sample per model.")
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--resample",   type=int, default=10)
    p.add_argument("--seed",       type=int, default=42,
                   help="RNG seed for choosing validation samples.")
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--out_dir",    default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper
# ---------------------------------------------------------------------------

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
            cmap=cmap,
            clim=(0, np.nanpercentile(mq[mask], 98) or 1),
            scale=12, width=0.003, zorder=2,
        )
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.65)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# Variance heatmap helper
# ---------------------------------------------------------------------------

def plot_variance(ax, runs_np, land_mask, title):
    """
    Display a heatmap of the per-pixel std-dev of speed across runs.
    runs_np: (n_runs, 2, H, W) numpy array in original (H,W) space,
             already transposed to display coords (W, H).
    """
    H, W = runs_np.shape[2], runs_np.shape[3]
    speeds = np.sqrt(runs_np[:, 0] ** 2 + runs_np[:, 1] ** 2)  # (n_runs, H, W)
    std_map = np.std(speeds, axis=0)                             # (H, W)
    std_map[land_mask] = np.nan

    std_masked = np.ma.masked_where(land_mask, std_map)
    vmax = np.nanpercentile(std_map[~land_mask], 98) if (~land_mask).any() else 1.0

    im = ax.imshow(
        std_masked, origin="lower", cmap="hot_r", aspect="auto",
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], vmin=0, vmax=vmax,
    )
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=1,
    )
    plt.colorbar(im, ax=ax, label="Std dev", shrink=0.65)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# Compute metrics for one prediction
# ---------------------------------------------------------------------------

def compute_metrics(pred_dev, true_dev, ocean, device):
    metrics = {}
    metrics["eps"] = F.mse_loss(pred_dev * ocean, true_dev * ocean).sqrt().item()
    metrics["curl_div"]        = curl_div_loss(pred_dev, true_dev, ocean).item()
    metrics["spectral"]        = spectral_loss(pred_dev, true_dev, ocean).item()
    metrics["okubo_weiss"]     = okubo_weiss_loss(pred_dev, true_dev, ocean).item()
    metrics["stream_function"] = stream_function_loss(pred_dev, true_dev, ocean).item()
    metrics["strain_rate"]     = strain_rate_loss(pred_dev, true_dev, ocean).item()
    if _HAS_GEOMLOSS:
        metrics["wasserstein"] = wasserstein_loss(pred_dev, true_dev, ocean, _sinkhorn).item()
    else:
        metrics["wasserstein"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Save per-sample 4×4 image
# ---------------------------------------------------------------------------

def save_sample_image(
    sample_idx,
    x0_true_np,       # (2, H, W)
    path_mask,        # (H, W) bool
    land_mask_np,     # (H, W) bool
    model_avgs,       # dict label -> (2, H, W) mean prediction
    model_runs,       # dict label -> (n_runs, 2, H, W) all predictions
    model_mean_metrics,  # dict label -> dict metric -> mean_value
    out_path,
):
    # Transpose everything to display coords (W, H)
    land_d = land_mask_np.T
    path_d = path_mask.T

    n_cols = 4
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4),
                             constrained_layout=True)
    axes = axes.flatten()

    # Panel 0: ground truth
    plot_field(axes[0], x0_true_np[0].T, x0_true_np[1].T, land_d, "Ground Truth")

    # Panel 1: robot path
    axes[1].imshow(
        land_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    path_display = np.zeros_like(land_d, dtype=float)
    path_display[path_d] = 1.0
    axes[1].imshow(
        path_display, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    axes[1].set_title(f"Robot Path ({int(path_mask.sum())} cells)", fontsize=8)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    axes[1].legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
            mpatches.Patch(facecolor="#d62728", label="Path"),
            mpatches.Patch(facecolor="black",   label="Land"),
        ],
        loc="upper right", fontsize=7,
    )

    # Panels 2–15: pairs (avg, heatmap) for each model
    for mi, label in enumerate([l for l in MODEL_LABELS if l in model_avgs]):
        avg_np  = model_avgs[label]    # (2, H, W)
        runs_np = model_runs[label]    # (n_runs, 2, H, W)
        m_vals  = model_mean_metrics[label]  # dict metric -> float

        avg_slot  = 2 + mi * 2
        heat_slot = 3 + mi * 2

        # Build title with mean eps for this model on this sample
        eps_val = m_vals.get("eps", float("nan"))
        avg_title = f"Model: {label}\neps={eps_val:.4f}"

        # Transpose runs for heatmap: (n_runs, 2, W, H)
        runs_d = np.stack([r.T for r in runs_np[:, 0]], axis=0)  # (n_runs, W, H)
        runs_d_v = np.stack([r.T for r in runs_np[:, 1]], axis=0)
        runs_d_stacked = np.stack([
            np.stack([runs_d[i], runs_d_v[i]], axis=0) for i in range(len(runs_np))
        ])  # (n_runs, 2, W, H)

        plot_field(axes[avg_slot], avg_np[0].T, avg_np[1].T, land_d,
                   avg_title, cmap="cool")
        plot_variance(axes[heat_slot], runs_d_stacked, land_d,
                      f"Model: {label}\nVariance across {len(runs_np)} runs")

    fig.suptitle(
        f"Val sample {sample_idx} — 7 models × {len(list(model_runs.values())[0])} runs",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Paths
    if args.ckpt_dir is None:
        # Models are in 'loss_comparison' sibling of this folder
        ckpt_dir = os.path.join(_MODEL_PARAMS, "loss_comparison")
    else:
        ckpt_dir = args.ckpt_dir

    if args.out_dir is None:
        out_dir = os.path.join(_HERE, "results", "eval_visual_loss")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    pickle_path = args.pickle
    if not os.path.isabs(pickle_path) and not os.path.exists(pickle_path):
        pickle_path = os.path.join(_ROOT, pickle_path)

    val_ds       = OceanCurrentDataset(pickle_path, split=1)
    land_mask_np = val_ds.land_mask.numpy()
    land_mask_t  = val_ds.land_mask.to(device)
    ocean        = (~land_mask_t).float()[None, None]   # (1,1,H,W)

    n_val = len(val_ds)
    rng   = np.random.default_rng(args.seed)
    sample_indices = sorted(rng.choice(n_val, size=min(args.n_samples, n_val), replace=False).tolist())
    print(f"Evaluating {len(sample_indices)} val samples: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}\n")

    # Path seed per sample: deterministic from sample index
    path_seeds = {idx: int(idx * 7 + 1) for idx in sample_indices}

    # -------------------------------------------------------------------------
    # Load all 7 models once
    # -------------------------------------------------------------------------
    models_loaded = {}
    diffusions    = {}
    for label in MODEL_LABELS:
        ckpt_path = os.path.join(ckpt_dir, f"model_ddpm_{label}_gaussian_cosine.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {label}: not found at {ckpt_path}")
            continue
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        base_ch   = ckpt_args.get("base_ch",  args.base_ch)
        time_dim  = ckpt_args.get("time_dim", args.time_dim)
        T         = ckpt_args.get("T",        args.T)
        schedule  = ckpt_args.get("schedule", "cosine")
        net = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        models_loaded[label] = net
        diffusions[label]    = DDPM(T=T, beta_schedule=schedule, device=device)
        print(f"  Loaded model '{label}'  (epoch {ckpt.get('epoch','?')}, T={T}, sched={schedule})")
    print()

    active_labels = [l for l in MODEL_LABELS if l in models_loaded]

    # -------------------------------------------------------------------------
    # Storage: all_detail_rows for the report
    # all_detail_rows: list of dicts {model, sample, run, seed, metric...}
    # -------------------------------------------------------------------------
    all_detail_rows = []

    # summary accumulators: model -> metric -> list of values
    summary_acc = {label: {m: [] for m in LOSS_MODES} for label in active_labels}

    # -------------------------------------------------------------------------
    # Main loop: samples
    # -------------------------------------------------------------------------
    for si, sample_idx in enumerate(sample_indices):
        print(f"[{si+1:3d}/{len(sample_indices)}] Val sample {sample_idx}")

        x0_true     = val_ds[sample_idx]                         # (2, H, W) CPU
        path_seed   = path_seeds[sample_idx]
        path_mask   = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=path_seed)

        x0_observed = x0_true.clone()
        path_t      = torch.from_numpy(path_mask)
        x0_observed[:, ~path_t] = 0.0

        true_dev = x0_true.unsqueeze(0).to(device)               # (1,2,H,W)

        # model_avgs[label] = (2,H,W) mean over n_runs
        # model_runs[label] = (n_runs, 2, H, W)
        model_avgs         = {}
        model_runs_all     = {}
        model_mean_metrics = {}   # label -> {metric -> mean over runs}

        for label in active_labels:
            net       = models_loaded[label]
            diffusion = diffusions[label]

            run_preds = []
            run_metrics_list = []

            for run_i in range(args.n_runs):
                # Different diffusion seed each run, same path
                torch.manual_seed(run_i * 17 + 3)
                if device == "cuda":
                    torch.cuda.manual_seed_all(run_i * 17 + 3)

                x0_pred = repaint(
                    net, diffusion, x0_observed,
                    path_mask, land_mask_np,
                    r=args.resample, device=device,
                )   # (2,H,W) CPU

                pred_dev = x0_pred.unsqueeze(0).to(device)
                m = compute_metrics(pred_dev, true_dev, ocean, device)

                run_preds.append(x0_pred.numpy())     # (2,H,W)
                run_metrics_list.append(m)

                for metric_key, val in m.items():
                    summary_acc[label][metric_key].append(val)

                all_detail_rows.append({
                    "model":  label,
                    "sample": sample_idx,
                    "run":    run_i + 1,
                    "seed":   run_i * 17 + 3,
                    **m,
                })

            # Compute mean prediction and mean metrics over runs
            runs_arr = np.stack(run_preds, axis=0)   # (n_runs, 2, H, W)
            avg_pred = runs_arr.mean(axis=0)          # (2, H, W)
            mean_m   = {k: float(np.nanmean([r[k] for r in run_metrics_list]))
                        for k in LOSS_MODES}

            model_avgs[label]         = avg_pred
            model_runs_all[label]     = runs_arr
            model_mean_metrics[label] = mean_m

            eps_mean = mean_m["eps"]
            print(f"    {label:<16}  eps_mean={eps_mean:.5f}")

        # Save 4×4 image for this sample
        img_path = os.path.join(out_dir, f"sample_{sample_idx:03d}.png")
        save_sample_image(
            sample_idx,
            x0_true.numpy(),
            path_mask,
            land_mask_np,
            model_avgs,
            model_runs_all,
            model_mean_metrics,
            img_path,
        )
        print(f"    -> {img_path}\n")

    # -------------------------------------------------------------------------
    # Build text report
    # -------------------------------------------------------------------------
    report_path = os.path.join(_HERE, "results", "loss_eval_visual_summary.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    col_w   = 16
    metrics = list(LOSS_MODES)
    n_m     = len(metrics)

    divider = "=" * (18 + col_w * n_m)
    lines   = []

    # ---- Summary table ----
    lines.append(divider)
    lines.append(f"{'LOSS MODEL EVALUATION SUMMARY':^{len(divider)}}")
    lines.append(f"n_samples={len(sample_indices)}  n_runs_per_sample={args.n_runs}  "
                 f"path_steps={args.path_steps}  resample={args.resample}")
    lines.append(divider)
    header = f"{'model':<18}" + "".join(f"{m:>{col_w}}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))
    for label in active_labels:
        mean_line = f"{label:<18}"
        std_line  = f"{'':18}"
        for m in metrics:
            vals = [v for v in summary_acc[label][m] if not np.isnan(v)]
            mv = np.mean(vals) if vals else float("nan")
            sv = np.std(vals)  if vals else float("nan")
            mean_line += f"{mv:>{col_w}.6f}" if not np.isnan(mv) else f"{'NaN':>{col_w}}"
            std_line  += f"{'(±'+f'{sv:.5f}'+')':>{col_w}}" if not np.isnan(sv) else f"{'':>{col_w}}"
        lines.append(mean_line)
        lines.append(std_line)
        lines.append("")
    lines.append(divider)

    # ---- Full detail table ----
    lines.append("")
    lines.append("FULL DETAIL  (model × sample × run)")
    lines.append(divider)
    det_header = f"{'model':<18}{'sample':>8}{'run':>5}{'seed':>7}" + \
                 "".join(f"{m:>{col_w}}" for m in metrics)
    lines.append(det_header)
    lines.append("-" * len(det_header))
    for row in all_detail_rows:
        det_line = f"{row['model']:<18}{row['sample']:>8}{row['run']:>5}{row['seed']:>7}"
        for m in metrics:
            v = row.get(m, float("nan"))
            det_line += f"{v:>{col_w}.6f}" if not np.isnan(v) else f"{'NaN':>{col_w}}"
        lines.append(det_line)
    lines.append(divider)

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report + "\n")

    print(f"Report -> {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
