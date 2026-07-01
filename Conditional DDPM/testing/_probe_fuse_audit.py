"""
PHASE 1 AUDIT — is the two-head fusion really that good, or are we fooling
ourselves?  Three honesty checks the quick probe did NOT do:

  1. SHARED colour scale.  IC.plot_field auto-scales every panel to its own 98th
     percentile, so a globally wrong magnitude still LOOKS identical to truth.
     Here every field panel uses ONE vmax (truth's), and we add an explicit
     per-cell vector-ERROR map, so magnitude/direction errors are visible.

  2. HONEST whole-ocean numbers on RANDOM frames (not the 3 hand-picked ones):
     vector RMSE%, cosine, speed ratio over ALL ocean cells, diffusion-only vs
     fused, plus the known/unobserved split.

  3. LEAKAGE check.  The fused field must only use ground truth on the robot
     path.  We re-predict the UNet speed with the path ERASED (empty obs) and
     fuse that; if the empty-obs fused field is nearly as accurate as the real
     one, the apparent skill is climatology, not observation use — and if it
     still looked like truth that would expose a leak.  We report, per frame,
     corr(UNet speed, truth speed) for real-obs vs empty-obs, and the empty-obs
     fused RMSE.

Run:
  .venv/bin/python "Conditional DDPM/testing/_probe_fuse_audit.py" \
      --checkpoint Models/StreamFn_Cond_x0_mag_spread.pt \
      --mag_checkpoint Models/Magnitude_UNet_New.pt \
      --pickle Datasets/pickles/data_divfree_chrono.pickle \
      --split 2 --n_frames 8 --seed 3 --n_model 24 --path_steps 90 \
      --out_dir "Conditional DDPM/results/cond_fields_audit"
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

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", ".."))
for _p in (_here, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), _root):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import infer_cond as IC                       # noqa: E402
from _probe_calib_mag import (                # noqa: E402
    load_magnitude_model, predict_speed_norm, apply_unet_magnitude, pcorr, EPS,
)
from _probe_fuse_fields import cell_metrics    # noqa: E402


def whole_metrics(field, true, ocean):
    """(speed_ratio, cosine, vec_rmse%) over all ocean cells (single field)."""
    sr, ang, rmse = cell_metrics(field, true, ocean)
    cos = float(np.cos(np.radians(ang)))
    return sr, cos, rmse


def render_audit(out_path, label, true_np, obs_path, raw0, fused0, fused_mean,
                 unet_speed_phys, spread, land_np, data_std, cov_pct, txt):
    """2x4 panel, SHARED magnitude scale across all field panels + error map."""
    land_d = land_np.T
    s = data_std
    # Shared colour scale = truth's 98th-percentile speed (physical units).
    ocean_d = ~land_d
    tspd = np.sqrt((true_np[0] * s) ** 2 + (true_np[1] * s) ** 2).T
    vmax = float(np.nanpercentile(tspd[ocean_d], 98)) if ocean_d.any() else 1.0

    fig, axes = plt.subplots(2, 4, figsize=(26, 11), dpi=90)
    ax = axes.flatten()
    IC.plot_field(ax[0], true_np[0].T * s, true_np[1].T * s, land_d,
                  "Ground truth", vmax=vmax)
    IC.plot_field(ax[1], raw0[0].T * s, raw0[1].T * s, land_d,
                  "Diffusion-only draw", vmax=vmax)
    IC.plot_field(ax[2], fused0[0].T * s, fused0[1].T * s, land_d,
                  "Fused draw (member 0)", vmax=vmax)
    IC.plot_field(ax[3], fused_mean[0].T * s, fused_mean[1].T * s, land_d,
                  "Fused ensemble mean", vmax=vmax)
    IC.plot_path(ax[4], obs_path.T, land_d,
                 f"Robot obs ({int(obs_path.sum())} cells, {cov_pct:.1f}%)")

    # UNet speed map (shared scale)
    usp = unet_speed_phys.T.copy(); usp[land_d] = np.nan
    im5 = ax[5].imshow(usp, origin="lower", cmap="cool", vmin=0.0, vmax=vmax,
                       extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                       aspect="auto")
    ax[5].imshow(land_d, origin="lower",
                 cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
                 extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                 aspect="auto", zorder=2)
    plt.colorbar(im5, ax=ax[5], label="speed", shrink=0.7)
    ax[5].set_title("UNet speed (fusion magnitude)", fontsize=11)

    # Per-cell vector error |fused_mean - truth| (shared scale)
    err = np.sqrt(((fused_mean[0] - true_np[0]) * s) ** 2
                  + ((fused_mean[1] - true_np[1]) * s) ** 2).T
    err[land_d] = np.nan
    im6 = ax[6].imshow(err, origin="lower", cmap="inferno", vmin=0.0, vmax=vmax,
                       extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                       aspect="auto")
    ax[6].imshow(land_d, origin="lower",
                 cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
                 extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                 aspect="auto", zorder=2)
    plt.colorbar(im6, ax=ax[6], label="|error|", shrink=0.7)
    ax[6].set_title("Vector error |fused mean - truth|  (SAME scale)", fontsize=11)

    sp = spread.T.copy()
    im7 = ax[7].imshow(sp, origin="lower", cmap="magma", vmin=0.0, vmax=1.0,
                       extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                       aspect="auto")
    ax[7].imshow(land_d, origin="lower",
                 cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
                 extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                 aspect="auto", zorder=2)
    plt.colorbar(im7, ax=ax[7], label="1 - R", shrink=0.7)
    ax[7].set_title("Directional spread (uncertainty)", fontsize=11)

    for a in ax:
        a.set_xlabel("X"); a.set_ylabel("Y")
    plt.suptitle(f"Fusion AUDIT (shared scale)  —  {label}\n{txt}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--mag_checkpoint", default="Models/Magnitude_UNet_New.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--frames", default="")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--n_model", type=int, default=24)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--out_dir", default="Conditional DDPM/results/cond_fields_audit")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)
    print(f"model: {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch')} "
          f"pred={pred_type}  n_model={args.n_model}  device={device}")

    ds = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=ckpt.get("data_mean", 0.0), data_std=data_std,
        path_steps=args.path_steps, deterministic=True)
    land_np = ds.land_mask.cpu().numpy().astype(bool); ocean_np = ~land_np

    model = IC.StreamFunctionUNet(in_ch=2, base_ch=ca.get("base_ch", 64),
        time_dim=ca.get("time_dim", 256), cond_ch=cond_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = IC.DDPM(T=ca.get("T", 1000), beta_schedule=ca.get("schedule", "cosine"),
        device=device, noise_type=ca.get("noise_type", "div_free"),
        spectral_filter=ckpt.get("spectral_filter", None))

    mag_net, sm, ss = load_magnitude_model(args.mag_checkpoint, device)
    print(f"  magnitude UNet: speed_mean={sm:.4f} speed_std={ss:.4f}")

    if args.frames.strip():
        idxs = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        rng = np.random.default_rng(args.seed)
        idxs = sorted(rng.choice(len(ds.valid), size=max(1, args.n_frames),
                                 replace=False).tolist())
    print(f"frames (split indices): {idxs}\n")

    print(f"{'frame':>6} {'%kn':>5} | {'WHOLE-OCEAN vec rmse%':>22} | "
          f"{'cosine':>14} | {'speedR':>14} | {'LEAK corr(unet,truth)':>22} "
          f"{'emptyRMSE%':>10}")
    print(f"{'':>6} {'':>5} | {'diff -> fused':>22} | {'diff->fused':>14} | "
          f"{'diff->fused':>14} | {'real / empty':>22}")

    agg = []
    for src_idx in idxs:
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
        src = b["target"].cpu().numpy()
        pm = b["path_mask"]
        pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
        pm_ocean = pm & ocean_np
        unobs = ocean_np & ~pm_ocean
        src_f = int(ds.valid[src_idx])
        cov = 100.0 * pm_ocean.sum() / ocean_np.sum()

        sargs = argparse.Namespace(pred_type=pred_type,
            inference_steps=args.inference_steps, capture_every=10 ** 9,
            n_ensemble=args.n_model)
        _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                          sargs, device, base_seed=src_idx)

        spd_phys = np.sqrt((src ** 2).sum(axis=0)) * data_std
        speed_norm = predict_speed_norm(mag_net, sm, ss, spd_phys, pm,
                                        land_np, data_std, device,
                                        cond=b["cond"])
        fused = apply_unet_magnitude(members, speed_norm, ocean_np)
        fused_mean = np.mean(fused, axis=0).astype(np.float32)
        diff_mean = np.mean(members, axis=0).astype(np.float32)

        # ---- LEAKAGE: re-predict speed with the path ERASED ----
        # For the conditioned UNet the path lives inside `cond`, so erasing it
        # means rebuilding cond with an empty observation set (priors + geometry
        # only); for the 3-channel UNet the empty path_mask alone erases it.
        empty_pm = np.zeros_like(pm)
        empty_obs = IC.observation_channels(b["target"], empty_pm)
        empty_cond = IC.assemble_cond(empty_obs, b["priors"], ds.geom)
        speed_empty = predict_speed_norm(mag_net, sm, ss, spd_phys, empty_pm,
                                         land_np, data_std, device,
                                         cond=empty_cond)
        fused_empty = apply_unet_magnitude(members, speed_empty, ocean_np)
        fused_empty_mean = np.mean(fused_empty, axis=0).astype(np.float32)

        true_spd = np.sqrt((src ** 2).sum(axis=0))
        corr_real = pcorr((speed_norm)[ocean_np], true_spd[ocean_np])
        corr_empty = pcorr((speed_empty)[ocean_np], true_spd[ocean_np])

        # ---- honest whole-ocean numbers (ensemble-mean field) ----
        d_sr, d_cos, d_rmse = whole_metrics(diff_mean, src, ocean_np)
        f_sr, f_cos, f_rmse = whole_metrics(fused_mean, src, ocean_np)
        e_sr, e_cos, e_rmse = whole_metrics(fused_empty_mean, src, ocean_np)
        # known/unobserved fused split (ensemble-mean)
        ku = cell_metrics(fused_mean, src, pm_ocean)
        uu = cell_metrics(fused_mean, src, unobs)

        print(f"{src_f:>6} {cov:>4.1f}% | "
              f"{d_rmse:>9.0f} -> {f_rmse:>9.0f} | "
              f"{d_cos:>6.2f}->{f_cos:>5.2f} | "
              f"{d_sr:>6.2f}->{f_sr:>5.2f} | "
              f"{corr_real:>+9.3f} /{corr_empty:>+8.3f} {e_rmse:>10.0f}")
        agg.append((d_rmse, f_rmse, e_rmse, d_cos, f_cos, d_sr, f_sr,
                    corr_real, corr_empty, ku[2], uu[2], ku[1], uu[1]))

        unet_speed_phys = speed_norm * data_std
        spread = IC.directional_spread(members, ocean_np)
        txt = (f"frame {src_f}  cov {cov:.1f}%  |  whole-ocean vec RMSE "
               f"{d_rmse:.0f}%->{f_rmse:.0f}%  cos {d_cos:.2f}->{f_cos:.2f}  |  "
               f"known RMSE {ku[2]:.0f}% / unobs RMSE {uu[2]:.0f}%  |  "
               f"LEAK corr(unet,truth) real {corr_real:+.2f} vs empty {corr_empty:+.2f}  "
               f"(empty fused RMSE {e_rmse:.0f}%)")
        render_audit(os.path.join(args.out_dir, f"audit_frame{src_f}.png"),
                     f"frame {src_f}", src, pm, members[0], fused[0], fused_mean,
                     unet_speed_phys, spread, land_np, data_std, cov, txt)

    a = np.array(agg, dtype=np.float64)
    m = np.nanmean(a, axis=0)
    print(f"\n  N={len(agg)} frames  MEAN (ensemble-mean field, whole ocean)")
    print(f"  vec RMSE%   diffusion {m[0]:.0f}  ->  fused {m[1]:.0f}   "
          f"(empty-obs fused {m[2]:.0f})")
    print(f"  cosine      diffusion {m[3]:.2f}  ->  fused {m[4]:.2f}")
    print(f"  speed ratio diffusion {m[5]:.2f}  ->  fused {m[6]:.2f}")
    print(f"  fused split  known RMSE {m[9]:.0f}%  unobs RMSE {m[10]:.0f}%  | "
          f"known ang {m[11]:.0f}deg  unobs ang {m[12]:.0f}deg")
    print(f"  LEAK  corr(unet speed, truth)  real {m[7]:+.2f}  vs  empty-obs {m[8]:+.2f}")
    print(f"        => obs lift = {m[7] - m[8]:+.2f}; empty-obs fused RMSE {m[2]:.0f}% "
          f"(should be MUCH worse than fused {m[1]:.0f}% if obs truly drive skill)")
    print(f"\n  panels saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
