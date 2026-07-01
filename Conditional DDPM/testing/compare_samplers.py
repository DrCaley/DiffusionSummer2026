"""
Compare posterior-sampling procedures for the CONDITIONAL stream-function DDPM.

Runs each selected sampler (vanilla ensemble / particle filter / DPS) on the
same held-out frames from the SAME fixed conditioning, then reports an honest
scorecard that scores the three things we actually care about:

  * ACCURACY WHERE INFORMED   — obs_rmse   : error at the observed path cells
  * OVERALL PLAUSIBILITY      — mean_rmse  : ensemble-mean error vs ground truth
                                best_rmse  : does ANY draw land close? (good guess)
  * DIVERSITY (the north star)— spread_near / spread_far : ensemble disagreement
                                near vs far from the path (far should stay high)
  * CALIBRATION               — coverage   : fraction of cells where the truth
                                falls within the ensemble's min–max envelope
  * SANITY                    — max|div|   : divergence (≈0 by construction)

Lower is better for the RMSE/div columns; for spread/coverage there is no single
"better" — we want spread to stay HIGH in the far field (diverse guesses) while
obs_rmse stays LOW (faithful where measured).  A method that drives every number
to zero is just collapsing to one blurry mean — exactly what we want to avoid.

Usage (from workspace root, once a checkpoint exists):
    python "Conditional DDPM/testing/compare_samplers.py" \
        --checkpoint "Conditional DDPM/checkpoints_cond/best_streamfncond_minsnr5_ang1_lags13-25_div_free_cosine.pt" \
        --pickle Datasets/data_divfree_chrono.pickle \
        --split 1 --n_samples 4 --random \
        --methods vanilla,particle,dps \
        --n_ensemble 8 --inference_steps 100 \
        --out_dir "Conditional DDPM/results/sampler_compare"
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- path shim --------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", ".."))
for _p in [_here, _root, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), os.path.join(_here, "..", "model")]:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from diffusion    import DDPM                              # noqa: E402
from model        import StreamFunctionUNet                # noqa: E402
from cond_dataset import ConditionalOceanDataset, cond_channels  # noqa: E402

from samplers  import SAMPLERS                             # noqa: E402
from infer_cond import (                                   # noqa: E402
    build_cond, distance_to_path, plot_field, plot_path,
)


# ===========================================================================
# Metrics
# ===========================================================================

def _div_free_check(field, ocean):
    """Mean |divergence| over ocean (finite difference); ~0 by construction."""
    u, v = field[0], field[1]
    div = np.gradient(u, axis=0) + np.gradient(v, axis=1)
    return float(np.mean(np.abs(div[ocean])))


def score_ensemble(members, true_np, path_mask, ocean, dist):
    """Compute the honest scorecard for one method on one sample.

    members: list of (2, H, W) numpy fields.
    Returns a dict of scalar metrics.
    """
    M     = np.stack(members, axis=0)                      # (M, 2, H, W)
    mean  = M.mean(axis=0)                                 # (2, H, W)

    pm    = path_mask & ocean
    far   = (dist > 2.0) & ocean
    near  = (dist <= 2.0) & ocean

    def rmse(a, b, msk):
        d = (a[:, msk] - b[:, msk])
        return float(np.sqrt(np.mean(d ** 2))) if msk.any() else float("nan")

    obs_rmse  = rmse(mean, true_np, pm)
    mean_rmse = rmse(mean, true_np, ocean)
    best_rmse = min(rmse(m, true_np, ocean) for m in members)

    # Ensemble spread: per-cell std-vector magnitude, averaged near / far.
    std    = M.std(axis=0)                                 # (2, H, W)
    smag   = np.sqrt(std[0] ** 2 + std[1] ** 2)            # (H, W)
    sp_near = float(np.mean(smag[near])) if near.any() else float("nan")
    sp_far  = float(np.mean(smag[far]))  if far.any()  else float("nan")

    # Calibration: truth inside the ensemble min–max envelope (per component).
    lo, hi = M.min(axis=0), M.max(axis=0)                  # (2, H, W)
    inside = (true_np >= lo) & (true_np <= hi)             # (2, H, W)
    coverage = float(inside[:, ocean].mean())

    return {
        "obs_rmse":    obs_rmse,
        "mean_rmse":   mean_rmse,
        "best_rmse":   best_rmse,
        "spread_near": sp_near,
        "spread_far":  sp_far,
        "coverage":    coverage,
        "max_div":     _div_free_check(mean, ocean),
    }


_COLS = [
    ("obs_rmse",    "obs_rmse",   "↓ accuracy@obs"),
    ("mean_rmse",   "mean_rmse",  "↓ field error"),
    ("best_rmse",   "best_rmse",  "↓ best draw"),
    ("spread_near", "spr_near",   "  spread near"),
    ("spread_far",  "spr_far",    "↑ spread far"),
    ("coverage",    "coverage",   "↑ calib"),
    ("max_div",     "max|div|",   "≈0 sanity"),
]


def print_table(title, rows):
    """rows: dict method -> metrics dict."""
    header = f"{'method':10s}" + "".join(f"{h:>12s}" for _, h, _ in _COLS)
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for method, mt in rows.items():
        line = f"{method:10s}" + "".join(f"{mt[k]:12.4f}" for k, _, _ in _COLS)
        print(line)
    print("legend: " + "  ".join(f"{h}={d.strip()}" for _, h, d in _COLS))


# ===========================================================================
# Rendering
# ===========================================================================

def render_comparison(out_path, idx, true_np, path_mask, land_np,
                      per_method, vmax):
    """One row per method: ensemble mean | member 0 | spread heatmap.

    Top row shows ground truth + observations for reference.
    """
    methods = list(per_method.keys())
    nrow = 1 + len(methods)
    land_d = land_np.T

    fig, axes = plt.subplots(nrow, 3, figsize=(16, 4.6 * nrow), dpi=90)
    if nrow == 1:
        axes = axes[None, :]

    plot_field(axes[0, 0], true_np[0].T, true_np[1].T, land_d, "Ground truth", vmax=vmax)
    plot_path(axes[0, 1], path_mask.T, land_d,
              f"Observations ({int(path_mask.sum())} cells)")
    axes[0, 2].axis("off")

    for r, method in enumerate(methods, start=1):
        members = per_method[method]
        M    = np.stack(members, 0)
        mean = M.mean(0)
        std  = M.std(0)
        smag = np.sqrt(std[0] ** 2 + std[1] ** 2)
        smag_d = smag.T.copy()
        smag_d[land_d] = np.nan

        plot_field(axes[r, 0], mean[0].T, mean[1].T, land_d,
                   f"{method}: ensemble mean", vmax=vmax)
        plot_field(axes[r, 1], members[0][0].T, members[0][1].T, land_d,
                   f"{method}: member 0", vmax=vmax)
        im = axes[r, 2].imshow(smag_d, origin="lower", cmap="magma", aspect="auto")
        axes[r, 2].imshow(
            land_d, origin="lower",
            cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]), aspect="auto")
        axes[r, 2].set_title(f"{method}: ensemble spread", fontsize=11)
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Sampler comparison — sample {idx}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Args / main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compare conditional DDPM samplers.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pickle", default="Datasets/data_divfree_chrono.pickle")
    p.add_argument("--split", type=int, default=1, help="0=train,1=val,2=test")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--random", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--methods", default="vanilla,particle,dps",
                   help="Comma list from: vanilla, particle, dps")
    p.add_argument("--n_ensemble", type=int, default=8)
    p.add_argument("--inference_steps", type=int, default=100)
    p.add_argument("--path_steps", type=int, default=160)
    p.add_argument("--obs_sigma", type=float, default=0.1,
                   help="Observation noise (normalized units) for particle/dps.")
    p.add_argument("--ess_frac", type=float, default=0.5,
                   help="Resample when ESS < ess_frac * N (particle).")
    p.add_argument("--dps_zeta", type=float, default=0.05,
                   help="DPS guidance step size (field units).")
    p.add_argument("--out_dir", default="Conditional DDPM/results/sampler_compare")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in SAMPLERS:
            raise SystemExit(f"Unknown method {m!r}; choose from {list(SAMPLERS)}")
    print(f"Device     : {device}")
    print(f"Methods    : {methods}")
    print(f"Checkpoint : {args.checkpoint}")

    # ---- Checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type")
    if pred_type not in ("x0_streamfn_cond", "v_streamfn_cond"):
        raise ValueError(
            f"Expected pred_type 'x0_streamfn_cond' or 'v_streamfn_cond', got "
            f"{pred_type!r}.")
    ca         = ckpt.get("args", {})
    base_ch    = ca.get("base_ch", 64)
    time_dim   = ca.get("time_dim", 256)
    T          = ca.get("T", 1000)
    noise_type = ca.get("noise_type", "div_free")
    schedule   = ca.get("schedule", "cosine")
    lags       = tuple(ckpt.get("lags", ca.get("lags", (13, 25))))
    cond_ch    = ckpt.get("cond_ch", cond_channels(lags))
    data_mean  = ckpt.get("data_mean", 0.0)
    data_std   = ckpt.get("data_std", None)
    spectral_filter = ckpt.get("spectral_filter", None)
    print(f"Model      : epoch {ckpt.get('epoch','?')}  val={ckpt.get('val_loss', float('nan')):.5f}  "
          f"lags={lags}  cond_ch={cond_ch}  pred={pred_type}")

    # ---- Data (same normalization the model trained with) ----
    ds = ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=data_mean, data_std=data_std,
        path_steps=args.path_steps, deterministic=True)
    land_np = ds.land_mask.cpu().numpy().astype(bool)
    ocean   = ~land_np

    # ---- Model + diffusion ----
    stream_model = StreamFunctionUNet(
        in_ch=2, base_ch=base_ch, time_dim=time_dim, cond_ch=cond_ch).to(device)
    stream_model.load_state_dict(ckpt["model"])
    stream_model.eval()
    diffusion = DDPM(T=T, beta_schedule=schedule, device=device,
                     noise_type=noise_type, spectral_filter=spectral_filter)

    # ---- Per-method sampler kwargs ----
    kw = {
        "vanilla":  dict(),
        "particle": dict(obs_sigma=args.obs_sigma, ess_frac=args.ess_frac),
        "dps":      dict(obs_sigma=args.obs_sigma, zeta=args.dps_zeta),
    }

    # ---- Sample indices ----
    rng = np.random.default_rng(args.seed)
    indices = (rng.integers(0, len(ds), size=args.n_samples).tolist()
               if args.random else list(range(min(args.n_samples, len(ds)))))

    # ---- Run ----
    agg = {m: [] for m in methods}
    for s_i, idx in enumerate(indices):
        seed = args.seed + idx
        b = build_cond(ds, idx, args.path_steps, seed)
        true_np   = b["target"].cpu().numpy()
        path_mask = b["path_mask"]
        dist      = distance_to_path(path_mask, ocean)
        cov_pct   = 100.0 * path_mask.sum() / ocean.sum()
        spd = np.sqrt(true_np[0] ** 2 + true_np[1] ** 2); spd[land_np] = np.nan
        vmax = float(np.nanpercentile(spd, 98)) or 1.0

        print(f"\n[{s_i+1}/{len(indices)}] sample {idx}  obs={path_mask.sum()} "
              f"({cov_pct:.1f}%)  drawing {args.n_ensemble} members per method ...")

        per_method, rows = {}, {}
        for m in methods:
            members, aux = SAMPLERS[m](
                stream_model, diffusion, b["cond"], land_np,
                n_members=args.n_ensemble, inference_steps=args.inference_steps,
                device=device, seed=seed, pred_type=pred_type, **kw[m])
            per_method[m] = members
            sc = score_ensemble(members, true_np, path_mask, ocean, dist)
            rows[m] = sc
            agg[m].append(sc)
            extra = f"  (resamples={aux['resamples']})" if "resamples" in aux else ""
            print(f"   {m:10s} obs_rmse={sc['obs_rmse']:.4f}  mean_rmse={sc['mean_rmse']:.4f}  "
                  f"best={sc['best_rmse']:.4f}  spr_far={sc['spread_far']:.4f}  "
                  f"cover={sc['coverage']:.3f}{extra}")

        print_table(f"sample {idx} scorecard", rows)
        render_comparison(
            os.path.join(args.out_dir, f"sample{idx:04d}_compare.png"),
            idx, true_np, path_mask, land_np, per_method, vmax)

    # ---- Aggregate ----
    avg = {m: {k: float(np.nanmean([s[k] for s in agg[m]])) for k, _, _ in _COLS}
           for m in methods}
    print_table(f"AGGREGATE over {len(indices)} samples", avg)
    print(f"\nDone. Comparison figures in {args.out_dir}")


if __name__ == "__main__":
    main()
