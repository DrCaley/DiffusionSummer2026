"""
2×3 visualisation comparing RePaint vs PPR on a single test sample.

Layout
------
Row 0 (quiver plots): Ground truth  |  RePaint reconstruction  |  PPR reconstruction
Row 1 (divergence):   GT |div|       |  RePaint |div|           |  PPR |div|

The divergence heatmaps are the headline diagnostic: RePaint will show a visible
seam along the robot-path boundary; PPR should be nearly uniform near-zero.

Usage (from workspace root):
    python DDPM/testing/visualize_ppr.py --checkpoint best_ddpm_eps_div_free_cosine.pt
    python DDPM/testing/visualize_ppr.py --sample 3 --seed 22 --out ppr_sample3.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HYPERPARAMETERS  (edit defaults here, or override on the command line)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --path_steps  150   Robot path length (number of visited ocean cells)
  --seed        42    Random seed for the robot path

  RePaint
  --resample    3     Resampling iterations per timestep  (r in the RePaint paper)
                      Higher = more self-consistent but slower.  10 = paper default.

  PPR
  --ppr_resample 1    Resampling iterations per timestep for PPR.
                      PPR achieves consistency via joint projection, so r=1 is
                      usually sufficient (and faster).
  --proj_iter   20    POCS iterations inside joint_project.
                      More iterations → tighter divergence-free + obs consistency.

  Model / schedule
  --T           1000  Diffusion timesteps (must match training)
  --base_ch     64    UNet base channels  (must match training)
  --time_dim    256   Time embedding dim  (must match training)

  Device
  --device      auto  Force device: cuda / mps / cpu.  Auto-detects if omitted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

_here  = os.path.dirname(os.path.abspath(__file__))
_root  = os.path.join(_here, "..", "..")   # workspace root
_model = os.path.join(_here, "..", "model")
_repaint = os.path.join(_here, "repaint")
_ppr   = os.path.join(_here, "ppr")
for _p in [_root, os.path.join(_root, "utils"), _model, _repaint, _ppr]:
    sys.path.insert(0, _p)

from dataset              import OceanCurrentDataset
from diffusion            import DDPM
from model                import UNet
from plot_utils           import plot_field
from divfree_projection   import divergence as compute_divergence
from repaint_infer        import biased_walk_path, repaint
from ppr_infer            import ppr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _div_map(x_np: np.ndarray, land_mask_np: np.ndarray) -> np.ndarray:
    """Absolute divergence at ocean cells, NaN at land.  Shape (H, W)."""
    ocean_mask_t = torch.from_numpy(~land_mask_np)
    div = compute_divergence(
        torch.from_numpy(x_np).unsqueeze(0),   # (1, 2, H, W)
        ocean_mask_t,
    )[0].numpy()                                # (H, W)
    out = np.abs(div)
    out[land_mask_np] = np.nan
    return out


def _div_panel(ax, div_map: np.ndarray, land_mask: np.ndarray,
               title: str, vmax: float | None = None):
    """Plot a divergence heatmap on ax."""
    masked = np.ma.masked_where(land_mask, div_map)
    im = ax.imshow(
        masked, origin="lower", cmap="inferno", aspect="auto",
        vmin=0, vmax=vmax,
        extent=[-0.5, land_mask.shape[1] - 0.5,
                -0.5, land_mask.shape[0] - 0.5],
    )
    # Land overlay
    ax.imshow(
        land_mask, origin="lower",
        cmap=mcolors.ListedColormap(["none", "black"]),
        extent=[-0.5, land_mask.shape[1] - 0.5,
                -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=1,
    )
    plt.colorbar(im, ax=ax, label="|div|", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--checkpoint", default="DDPM/checkpoints/best_model.pt")
    p.add_argument("--sample",     type=int,   default=0)
    p.add_argument("--path_steps",      type=int,   default=150)
    p.add_argument("--resample",        type=int,   default=3,   help="RePaint resamples per step")
    p.add_argument("--ppr_resample",    type=int,   default=1,   help="PPR resamples per step (1 = single pass)")
    p.add_argument("--proj_iter",       type=int,   default=20)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--T",               type=int,   default=1000)
    p.add_argument("--base_ch",         type=int,   default=64)
    p.add_argument("--time_dim",        type=int,   default=256)
    p.add_argument("--inference_steps", type=int,   default=100,
                   help="Denoising steps at inference (default 100). "
                        "Must divide T evenly. E.g. 100 visits every 10th step of T=1000.")
    p.add_argument("--device",          default=None, help="cuda / mps / cpu (auto-detect if omitted)")
    p.add_argument("--out",             default="DDPM/best_model_results/ppr_comparison.png")
    p.add_argument("--no_scale_uncond",  action="store_true",
                   help="Skip amplitude-normalizing uncond sample to GT scale")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- Data ----
    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    land_mask_np = test_ds.land_mask.numpy()       # (H, W) bool
    x0_true      = test_ds[args.sample]            # (2, H, W)

    # ---- Robot path ----
    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=args.seed)
    print(f"Path: {path_mask.sum()} cells  "
          f"({100*path_mask.sum()/(~land_mask_np).sum():.1f}% of ocean)")

    x0_observed = x0_true.clone()
    x0_observed[:, ~torch.from_numpy(path_mask)] = 0.0

    # ---- Model ----
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T          = ckpt_args.get("T",          args.T)
    noise_type       = ckpt_args.get("noise_type", "gaussian")
    spectral_filter  = ckpt.get("spectral_filter", None)
    data_mean        = ckpt.get("data_mean", None)
    data_std         = ckpt.get("data_std",  None)
    _normalize       = data_mean is not None

    model = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sf_str   = "yes" if spectral_filter is not None else "no"
    norm_str = f"mean={data_mean:.4f} std={data_std:.4f}" if _normalize else "no"
    print(f"Checkpoint: epoch {ckpt.get('epoch', '?')}, T={T}, noise={noise_type}, "
          f"spectral_filter={sf_str}, normalize={norm_str}")

    diffusion = DDPM(T=T, beta_schedule="cosine", device=device, noise_type=noise_type,
                     spectral_filter=spectral_filter)

    # Normalize observations to match training-time data scale
    x0_obs_for_infer = ((x0_observed - data_mean) / data_std) if _normalize else x0_observed

    # ---- Summarise run config ----
    print()
    print("┌─────────────────────────────────────────┐")
    print("│            Run configuration            │")
    print("├─────────────────────────────────────────┤")
    print(f"│  Sample index   : {args.sample:<22}│")
    print(f"│  Path steps     : {args.path_steps:<22}│")
    print(f"│  Diffusion T    : {T:<22}│")
    print(f"│  RePaint r      : {args.resample:<22}│")
    print(f"│  PPR r          : {args.ppr_resample:<22}│")
    print(f"│  POCS iters     : {args.proj_iter:<22}│")
    print(f"│  Inference steps: {args.inference_steps:<22}│")
    print(f"│  Device         : {device:<22}│")
    print("└─────────────────────────────────────────┘")
    print()

    # ---- Unconditioned DDPM sample ----
    print(f"[1/3] Running unconditioned DDPM sample (inference_steps={args.inference_steps}) …")
    t0 = time.time()
    H, W = x0_true.shape[1:]
    xt = diffusion._sample_noise(torch.zeros(1, 2, H, W, device=device))
    schedule = diffusion.build_inference_schedule(args.inference_steps)
    for t_int, t_prev_int in schedule:
        xt = diffusion.p_sample_step(model, xt, t_int, t_prev_int)
    x0_uncond_norm = xt.squeeze(0).cpu()
    x0_uncond = (x0_uncond_norm * data_std + data_mean) if _normalize else x0_uncond_norm
    # zero land cells
    ocean_t = torch.from_numpy(~land_mask_np)
    x0_uncond[:, ~ocean_t] = 0.0
    print(f"      done in {time.time()-t0:.1f}s")

    # ---- RePaint ----
    print(f"[2/3] Running RePaint  (T={T}, r={args.resample}) …")
    t0 = time.time()
    x0_repaint_norm = repaint(
        model, diffusion, x0_obs_for_infer, path_mask, land_mask_np,
        r=args.resample, device=device, inference_steps=args.inference_steps,
    )
    x0_repaint = (x0_repaint_norm * data_std + data_mean) if _normalize else x0_repaint_norm
    repaint_nan = x0_repaint.isnan().any().item()
    print(f"      done in {time.time()-t0:.1f}s  {'⚠ NaN detected!' if repaint_nan else 'OK'}")

    # ---- PPR ----
    print(f"[3/3] Running PPR      (T={T}, r={args.ppr_resample}, proj_iter={args.proj_iter}) …")
    t0 = time.time()
    x0_ppr_norm = ppr(
        model, diffusion, x0_obs_for_infer, path_mask, land_mask_np,
        r=args.ppr_resample, proj_iter=args.proj_iter, device=device,
        inference_steps=args.inference_steps,
    )
    x0_ppr = (x0_ppr_norm * data_std + data_mean) if _normalize else x0_ppr_norm
    print(f"      done in {time.time()-t0:.1f}s")

    # ---- Metrics ----
    ocean = ~land_mask_np
    print()
    print("┌──────────────┬───────────┬───────────────┐")
    print("│  Method      │   RMSE    │  mean |div|   │")
    print("├──────────────┼───────────┼───────────────┤")
    for name, pred in [("Uncond", x0_uncond), ("RePaint", x0_repaint), ("PPR", x0_ppr)]:
        u_p, v_p = pred[0].numpy(), pred[1].numpy()
        u_t, v_t = x0_true[0].numpy(), x0_true[1].numpy()
        rmse     = float(np.sqrt(((u_p-u_t)**2 + (v_p-v_t)**2)[ocean].mean()))
        div      = np.abs(_div_map(pred.numpy(), land_mask_np))
        mean_div = float(np.nanmean(div))
        print(f"│  {name:<12}│  {rmse:.5f}  │  {mean_div:.7f}  │")
    print("└──────────────┴───────────┴───────────────┘")
    print()

    # ---- Transpose for display: (H=94,W=44) → (44,94) ----
    def _T(arr): return arr.T
    land_d  = _T(land_mask_np)
    path_d  = _T(path_mask)

    # Ground truth
    u_true_d = _T(x0_true[0].numpy())
    v_true_d = _T(x0_true[1].numpy())
    div_true = _T(_div_map(x0_true.numpy(), land_mask_np))

    # RePaint
    u_rp_d = _T(x0_repaint[0].numpy())
    v_rp_d = _T(x0_repaint[1].numpy())
    div_rp  = _T(_div_map(x0_repaint.numpy(), land_mask_np))

    # PPR
    u_ppr_d = _T(x0_ppr[0].numpy())
    v_ppr_d = _T(x0_ppr[1].numpy())
    div_ppr  = _T(_div_map(x0_ppr.numpy(), land_mask_np))

    # Unconditioned — amplitude-scale to GT std for direction comparison
    _oc = ~land_mask_np
    _gt_std  = float(x0_true[:, _oc].std())
    _unc_std = float(x0_uncond[:, _oc].std())
    _unc_scale = (_gt_std / _unc_std) if (not args.no_scale_uncond and _unc_std > 1e-8) else 1.0
    x0_uncond_disp = x0_uncond * _unc_scale  # scaled for quiver only
    u_unc_d = _T(x0_uncond_disp[0].numpy())
    v_unc_d = _T(x0_uncond_disp[1].numpy())
    div_unc  = _T(_div_map(x0_uncond.numpy(), land_mask_np))

    # Shared divergence colorscale
    vmax = float(np.nanpercentile(div_rp, 99))

    # ---- 2×4 Figure ----
    fig, axes = plt.subplots(2, 4, figsize=(26, 10))

    # Row 0: quiver plots
    plot_field(axes[0, 0], u_true_d, v_true_d, land_d, "Ground Truth")
    _scale_str = f" (×{_unc_scale:.2f} to GT amp)" if _unc_scale != 1.0 else ""
    plot_field(axes[0, 1], u_unc_d,  v_unc_d,  land_d, f"DDPM (uncond{_scale_str})")
    plot_field(axes[0, 2], u_rp_d,   v_rp_d,   land_d, "RePaint (hard snap)")
    plot_field(axes[0, 3], u_ppr_d,  v_ppr_d,  land_d, "PPR (joint projection)")

    # Mark robot path on RePaint panel
    path_disp = np.zeros_like(land_d, dtype=float)
    path_disp[path_d] = 1.0
    axes[0, 2].imshow(
        path_disp, origin="lower", cmap="Reds", alpha=0.5,
        extent=[-0.5, land_d.shape[1]-0.5, -0.5, land_d.shape[0]-0.5],
        aspect="auto", zorder=2,
    )

    # Row 1: divergence heatmaps
    _div_panel(axes[1, 0], div_true, land_d, "|divergence| — Ground Truth", vmax=vmax)
    _div_panel(axes[1, 1], div_unc,  land_d, "|divergence| — Unconditioned", vmax=vmax)
    _div_panel(axes[1, 2], div_rp,   land_d, "|divergence| — RePaint",      vmax=vmax)
    _div_panel(axes[1, 3], div_ppr,  land_d, "|divergence| — PPR",          vmax=vmax)

    fig.suptitle(
        f"Val sample {args.sample}, seed {args.seed}  |  "
        f"T={T}, r={args.resample}, proj_iter={args.proj_iter}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
