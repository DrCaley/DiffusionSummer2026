"""
Coverage sweep: how do all metrics change as path_steps varies?

For each path_steps value runs n_frames test frames and reports:
  Calibration:  r_angle, r_magnitude, r_overall  (spread vs empirical neighbours)
  Accuracy:     r_ang_acc, r_mag_acc, r_vec_acc, MSE (m/s)^2, angle_err (deg)  on ensemble mean
  Coverage:     mean % ocean cells observed

Also saves a summary visual: one fixed frame shown at each path_steps value
(ensemble mean field + directional spread map side by side).

Usage (from /workspace/DiffusionSummer2026):
  python "Conditional DDPM/testing/_probe_coverage_sweep.py" \
    --hetero_checkpoint "Magnitude/checkpoints_cond_mag_hetero_v2/best_cond_magnitude_hetero.pt" \
    --n_frames 8 --n_draws 4 --inference_steps 50 --seed 0
"""
import argparse, os, sys, time
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

import infer_cond as IC
from _probe_calib_mag import (
    pcorr, directional_spread, vector_spread, magnitude_spread,
    load_magnitude_model, predict_speed_norm,
    helmholtz_project, EPS,
)
from _probe_multidraw import (
    load_hetero_magnitude_model, predict_speed_mean_sigma, coupled_magnitude,
)

SWEEP_STEPS = [10, 30, 60, 90, 150, 250, 400]


def angle_err_deg(pred, true, mask):
    """Mean angle error in degrees over masked cells (ensemble mean vs truth)."""
    pu, pv = pred[0][mask], pred[1][mask]
    tu, tv = true[0][mask], true[1][mask]
    pm = np.sqrt(pu**2 + pv**2) + EPS
    tm = np.sqrt(tu**2 + tv**2) + EPS
    cos = np.clip((pu/pm)*(tu/tm) + (pv/pm)*(tv/tm), -1, 1)
    return float(np.degrees(np.arccos(cos)).mean())


def vec_rmse_pct(members, true, mask):
    tu, tv = true[0][mask], true[1][mask]
    trms = np.sqrt((tu**2 + tv**2).mean()) + EPS
    vals = [np.sqrt(((m[0][mask]-tu)**2 + (m[1][mask]-tv)**2).mean()) / trms
            for m in members]
    return float(100.0 * np.mean(vals))


def acc_corr(pred_mean, true, mask):
    """r for speed and direction separately on the ensemble mean."""
    pu, pv = pred_mean[0][mask], pred_mean[1][mask]
    tu, tv = true[0][mask], true[1][mask]
    pm = np.sqrt(pu**2 + pv**2); tm = np.sqrt(tu**2 + tv**2)
    r_mag = float(np.corrcoef(pm, tm)[0, 1])
    # directional: cos of angle between unit vectors
    pum = pu/(pm+EPS); pvm = pv/(pm+EPS)
    tum = tu/(tm+EPS); tvm = tv/(tm+EPS)
    r_ang = float(np.corrcoef(pum*tum + pvm*tvm,
                               np.ones_like(tum))[0, 1])  # fallback
    # use vector corr as r_vec
    r_vec = float(np.corrcoef(np.sqrt(pu**2+pv**2), np.sqrt(tu**2+tv**2))[0,1])
    # direction corr: correlate unit-dot with 1 is meaningless; use angle-cos directly
    cos_sim = np.clip(pum*tum + pvm*tvm, -1, 1)
    r_ang = float(np.mean(cos_sim))         # mean cos similarity (not Pearson, but interpretable)
    return r_ang, r_mag, r_vec


def quiver_ax(ax, field, land, title, vmax, step=2):
    H, W = land.shape
    xx, yy = np.meshgrid(np.arange(W)[::step], np.arange(H)[::step])
    u = field[0][::step, ::step]; v = field[1][::step, ::step]
    spd = np.sqrt(u**2 + v**2)
    ax.quiver(xx, yy, u, v, spd, cmap="cool", scale=vmax*30,
              width=0.003, clim=(0, vmax))
    ax.imshow(land.T, origin="lower",
              cmap=mcolors.ListedColormap(["none","black"]),
              extent=[-0.5,W-0.5,-0.5,H-0.5], aspect="auto")
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def spread_ax(ax, spread, land, title):
    H, W = land.shape
    im = ax.imshow(spread.T, origin="lower", cmap="hot_r", vmin=0, vmax=1,
                   extent=[-0.5,W-0.5,-0.5,H-0.5], aspect="auto")
    ax.imshow(land.T, origin="lower",
              cmap=mcolors.ListedColormap(["none","black"]),
              extent=[-0.5,W-0.5,-0.5,H-0.5], aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",        default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--mag_checkpoint",    default="Models/Cond_Magnitude_UNet.pt")
    ap.add_argument("--hetero_checkpoint", default="Magnitude/checkpoints_cond_mag_hetero_v2/best_cond_magnitude_hetero.pt")
    ap.add_argument("--pickle",            default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split",     type=int, default=2)
    ap.add_argument("--n_frames",  type=int, default=8)
    ap.add_argument("--n_draws",   type=int, default=4)
    ap.add_argument("--n_emp",     type=int, default=30)
    ap.add_argument("--inference_steps", type=int, default=50)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--guard",     type=int, default=48)
    ap.add_argument("--min_sep",   type=int, default=12)
    ap.add_argument("--prior_weight", type=float, default=1.0)
    ap.add_argument("--out_dir",   default="Conditional DDPM/results/coverage_sweep")
    ap.add_argument("--vis_frame_idx", type=int, default=5,
                    help="which of the n_frames frames to use for the visual")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  n_frames={args.n_frames}  n_draws={args.n_draws}  "
          f"inference_steps={args.inference_steps}")

    # ── load models ──────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)

    # build dataset with the LARGEST path_steps first so valid frames are consistent
    ds_base = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=ckpt.get("data_mean", 0.0), data_std=data_std,
        path_steps=90, deterministic=True)
    ds_base.cond_ch = cond_ch  # propagate for legacy obs detection in build_cond
    land_np = ds_base.land_mask.cpu().numpy().astype(bool)
    ocean_np = ~land_np; n_ocean = int(ocean_np.sum())
    fields_all = ds_base.fields.cpu().numpy(); N = fields_all.shape[0]
    max_lag = max(lags)

    model = IC.StreamFunctionUNet(in_ch=2, base_ch=ca.get("base_ch", 64),
        time_dim=ca.get("time_dim", 256), cond_ch=cond_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = IC.DDPM(T=ca.get("T", 1000), beta_schedule=ca.get("schedule", "cosine"),
        device=device, noise_type=ca.get("noise_type", "div_free"),
        spectral_filter=ckpt.get("spectral_filter", None))
    mag_net, sm, ss = load_magnitude_model(args.mag_checkpoint, device)
    het_net, hsm, hss, het_clip = load_hetero_magnitude_model(
        args.hetero_checkpoint, device)

    sargs_tmpl = argparse.Namespace(pred_type=pred_type,
        inference_steps=args.inference_steps, capture_every=10**9,
        n_ensemble=args.n_draws)

    # fixed frame indices across all path_steps values
    rng = np.random.default_rng(args.seed)
    frame_idxs = rng.choice(len(ds_base.valid),
                             size=min(args.n_frames, len(ds_base.valid)),
                             replace=False).tolist()
    vis_src_idx = int(frame_idxs[min(args.vis_frame_idx, len(frame_idxs)-1)])
    vis_frame_id = int(ds_base.valid[vis_src_idx])
    print(f"visual frame: {vis_frame_id}  (split idx {vis_src_idx})")

    results = {}   # path_steps → dict of mean metrics
    vis_data = {}  # path_steps → (mean_field, spread_map, cov)

    t0 = time.time()

    for ps in SWEEP_STEPS:
        print(f"\n── path_steps={ps} ──")

        # rebuild dataset with this path_steps
        ds = IC.ConditionalOceanDataset(
            args.pickle, split=args.split, lags=lags,
            data_mean=ckpt.get("data_mean", 0.0), data_std=data_std,
            path_steps=ps, deterministic=True)
        ds.cond_ch = cond_ch  # propagate for legacy obs detection in build_cond

        r_ang_cal, r_mag_cal, r_vec_cal = [], [], []
        r_ang_acc, r_mag_acc, r_vec_acc = [], [], []
        rmse_list, mse_list, ang_err_list, cov_list = [], [], [], []

        for src_idx in frame_idxs:
            src_idx = int(src_idx)
            if src_idx >= len(ds.valid):
                continue

            b = IC.build_cond(ds, src_idx, ps, seed=src_idx)
            true_np = b["target"].cpu().numpy()
            pm = b["path_mask"]
            pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
            pm_ocean = pm & ocean_np
            cov = 100.0 * pm_ocean.sum() / ocean_np.sum()
            src_f = int(ds.valid[src_idx])

            # empirical neighbours
            obs_src = true_np[:, pm_ocean]
            obs_all = fields_all[:, :, pm_ocean]
            dist = ((obs_all - obs_src[None])**2).sum(axis=(1,2)) / max(pm_ocean.sum(), 1) / 2
            src_priors = np.concatenate([fields_all[src_f - L] for L in lags], axis=0)
            src_p_ocean = src_priors[:, ocean_np]
            prior_dist = np.full(N, np.inf)
            f_idx = np.arange(max_lag, N)
            acc = np.zeros(len(f_idx)); c = 0
            for li, L in enumerate(lags):
                cand = fields_all[f_idx - L][:, :, ocean_np]
                ref  = src_p_ocean[2*li:2*li+2]
                acc += ((cand - ref[None])**2).sum(axis=(1,2)); c += 2
            prior_dist[f_idx] = acc / (c * n_ocean)
            dist = dist + args.prior_weight * prior_dist
            order = np.argsort(dist); picks = []
            for f in order:
                f = int(f)
                if not np.isfinite(dist[f]) or abs(f - src_f) <= args.guard:
                    continue
                if any(abs(f - p) < args.min_sep for p in picks):
                    continue
                picks.append(f)
                if len(picks) == args.n_emp - 1:
                    break
            empirical = [true_np] + [fields_all[f] for f in picks]

            # model ensemble
            _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                              sargs_tmpl, device, base_seed=src_idx)
            # fuse + reproject
            mu_n, sig_n = predict_speed_mean_sigma(
                het_net, hsm, hss, land_np, data_std, device, b["cond"], het_clip)
            draws = coupled_magnitude(members, mu_n, sig_n, ocean_np)
            draws = [helmholtz_project(d, ocean_np) for d in draws]

            # calibration spreads
            mod_ang = directional_spread(draws,     ocean_np)
            emp_ang = directional_spread(empirical, ocean_np)
            mod_mag = magnitude_spread(draws,       ocean_np)
            emp_mag = magnitude_spread(empirical,   ocean_np)
            mod_vec = vector_spread(draws,          ocean_np)
            emp_vec = vector_spread(empirical,      ocean_np)
            r_ang_cal.append(pcorr(mod_ang[ocean_np], emp_ang[ocean_np]))
            r_mag_cal.append(pcorr(mod_mag[ocean_np], emp_mag[ocean_np]))
            r_vec_cal.append(pcorr(mod_vec[ocean_np], emp_vec[ocean_np]))

            # accuracy on ensemble mean
            mean_field = np.mean(draws, axis=0)
            ra, rm, rv = acc_corr(mean_field, true_np, ocean_np)
            r_ang_acc.append(ra); r_mag_acc.append(rm); r_vec_acc.append(rv)
            rmse_list.append(vec_rmse_pct(draws, true_np, ocean_np))
            mse = float(((mean_field - true_np)**2)[:, ocean_np].mean()) * data_std**2
            mse_list.append(mse)
            ang_err_list.append(angle_err_deg(mean_field, true_np, ocean_np))
            cov_list.append(cov)

            # store visual data for this path_steps
            if src_idx == vis_src_idx and ps not in vis_data:
                spr = directional_spread(draws, ocean_np)
                vis_data[ps] = (mean_field.copy(), spr.copy(), cov)

        results[ps] = dict(
            cov       = float(np.mean(cov_list)),
            r_ang_cal = float(np.mean(r_ang_cal)),
            r_mag_cal = float(np.mean(r_mag_cal)),
            r_vec_cal = float(np.mean(r_vec_cal)),
            rmse_pct  = float(np.mean(rmse_list)),
            mse_m2    = float(np.mean(mse_list)),
            ang_err   = float(np.mean(ang_err_list)),
        )
        r = results[ps]
        elapsed = time.time() - t0
        print(f"  cov={r['cov']:.1f}%  r_ang={r['r_ang_cal']:.3f}  "
              f"r_mag={r['r_mag_cal']:.3f}  r_vec={r['r_vec_cal']:.3f}  "
              f"rmse%={r['rmse_pct']:.1f}  mse={r['mse_m2']:.5f}  "
              f"ang={r['ang_err']:.1f}°  [{elapsed:.0f}s elapsed]")

    # ── print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"COVERAGE SWEEP  ({args.n_frames} frames × {args.n_draws} draws, "
          f"{args.inference_steps} steps, fuse=coupled+reproj)")
    print(f"{'='*90}")
    print(f"{'path_steps':>12} {'cov%':>6} {'r_ang':>7} {'r_mag':>7} {'r_vec':>7} "
          f"{'rmse%':>7} {'mse(m/s)²':>10} {'angle°':>8}")
    for ps in SWEEP_STEPS:
        if ps not in results:
            continue
        r = results[ps]
        print(f"{ps:>12} {r['cov']:>6.1f} {r['r_ang_cal']:>7.3f} "
              f"{r['r_mag_cal']:>7.3f} {r['r_vec_cal']:>7.3f} "
              f"{r['rmse_pct']:>7.1f} {r['mse_m2']:>10.5f} {r['ang_err']:>8.1f}")

    # ── visual: one frame across all path_steps ───────────────────────────────
    ps_vis = [ps for ps in SWEEP_STEPS if ps in vis_data]
    n = len(ps_vis)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 9), dpi=90)

    # get shared vmax from ground truth
    gt = ds_base.fields[vis_frame_id].numpy()
    vmax = float(np.percentile(np.sqrt((gt**2).sum(axis=0))[ocean_np], 98))

    for col, ps in enumerate(ps_vis):
        mean_f, spr, cov = vis_data[ps]
        r = results[ps]
        quiver_ax(axes[0, col], mean_f, land_np,
                  f"path={ps}  cov={cov:.1f}%\nrmse%={r['rmse_pct']:.0f}  ang={r['ang_err']:.1f}°",
                  vmax)
        spr_plot = spr.copy(); spr_plot[land_np] = np.nan
        spread_ax(axes[1, col], spr_plot, land_np,
                  f"Dir spread\nr_ang={r['r_ang_cal']:.3f}  r_mag={r['r_mag_cal']:.3f}")

    fig.suptitle(f"Coverage sweep — frame {vis_frame_id}  |  coupled+reproj pipeline",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out_vis = os.path.join(args.out_dir, f"coverage_sweep_frame{vis_frame_id}.png")
    plt.savefig(out_vis, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisual saved: {out_vis}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
