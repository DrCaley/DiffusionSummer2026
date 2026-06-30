"""
PHASE 1 — two-head fusion probe (temporary).

Tests the decomposition idea WITHOUT any retraining:
    fused field  =  unit_direction( diffusion member )  x  speed( Magnitude-UNet )

The diffusion model owns DIRECTION + ensemble diversity (so the scale-invariant
r_dir uncertainty metric is untouched); the deterministic UNet owns per-cell
SPEED (so the collapsed magnitudes are restored).

For each frame it reports, BEFORE vs AFTER fusion:
  * known-cell (robot path) fidelity   : speed ratio, angle error, vector RMSE
  * unobserved-cell fidelity           : same three
  * r_dir  (direction-only spread corr)  -- must be identical raw vs fused
  * r_vec  (speed+angle spread corr)     -- the combined uncertainty map vs the
                                            true total-velocity dispersion

and renders a 2x3 field panel per frame:
    truth | robot obs (path) | diffusion-only draw
    fused draw | fused ensemble-mean | directional-spread uncertainty

Run locally (MPS) or on a CUDA box; does not modify any shared file:
  .venv/bin/python "Conditional DDPM/testing/_probe_fuse_fields.py" \
      --checkpoint Models/StreamFn_Cond_x0_mag_spread.pt \
      --mag_checkpoint Models/Magnitude_UNet_New.pt \
      --pickle Datasets/pickles/data_divfree_chrono.pickle \
      --split 2 --frames 1050,1149,1587 --seed 7 \
      --n_model 24 --inference_steps 100 --path_steps 90 \
      --out_dir "Conditional DDPM/results/cond_fields_fused"
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
    load_magnitude_model, predict_speed_norm, apply_unet_magnitude,
    vector_spread, pcorr, EPS,
)


def fuse_members(members, speed_norm, ocean_np):
    """unit_dir(member) x speed_norm for every member -> fused fields list."""
    return apply_unet_magnitude(members, speed_norm, ocean_np)


def cell_metrics(pred, true, mask):
    """
    pred,true : (2,H,W) normalized fields.  mask : (H,W) bool cells to score.
    Returns (speed_ratio, angle_deg, vec_rmse_pct).
      speed_ratio  = mean ||pred|| / mean ||true||      (1.0 = perfect magnitude)
      angle_deg    = mean per-cell direction error
      vec_rmse_pct = rms||pred-true|| / rms||true|| * 100
    """
    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    pu, pv = pred[0][mask], pred[1][mask]
    tu, tv = true[0][mask], true[1][mask]
    ps = np.sqrt(pu ** 2 + pv ** 2)
    ts = np.sqrt(tu ** 2 + tv ** 2)
    speed_ratio = float(ps.mean() / (ts.mean() + EPS))
    cos = np.clip((pu * tu + pv * tv) / (ps * ts + EPS), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos)).mean())
    vec_rmse = np.sqrt(((pu - tu) ** 2 + (pv - tv) ** 2).mean())
    true_rms = np.sqrt((tu ** 2 + tv ** 2).mean())
    return speed_ratio, angle_deg, float(100.0 * vec_rmse / (true_rms + EPS))


def avg_member_metrics(members, true, mask):
    """Average cell_metrics over the ensemble (honest per-draw, not mean field)."""
    rs = np.array([cell_metrics(m, true, mask) for m in members], dtype=np.float64)
    return tuple(np.nanmean(rs, axis=0))


def render_panel(out_path, label, true_np, obs_path, raw0, fused0, fused_mean,
                 spread, land_np, data_std, cov_pct, txt):
    """2x3 physical-unit field panel."""
    land_d = land_np.T
    s = data_std
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=90)
    ax = axes.flatten()
    IC.plot_field(ax[0], true_np[0].T * s, true_np[1].T * s, land_d, "Ground truth")
    IC.plot_path(ax[1], obs_path.T, land_d,
                 f"Robot observations  ({int(obs_path.sum())} cells, {cov_pct:.1f}%)")
    IC.plot_field(ax[2], raw0[0].T * s, raw0[1].T * s, land_d,
                  "Diffusion-only draw  (collapsed magnitude)")
    IC.plot_field(ax[3], fused0[0].T * s, fused0[1].T * s, land_d,
                  "Fused draw  (UNet speed x diffusion direction)")
    IC.plot_field(ax[4], fused_mean[0].T * s, fused_mean[1].T * s, land_d,
                  "Fused ensemble mean")
    sp = spread.T.copy()
    im = ax[5].imshow(sp, origin="lower", cmap="magma", vmin=0.0, vmax=1.0,
                      extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                      aspect="auto")
    ax[5].imshow(land_d, origin="lower",
                 cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
                 extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                 aspect="auto", zorder=2)
    plt.colorbar(im, ax=ax[5], label="1 - R", shrink=0.7)
    ax[5].set_title("Directional spread (uncertainty)", fontsize=11)
    ax[5].set_xlabel("X"); ax[5].set_ylabel("Y")
    plt.suptitle(f"Two-head fusion  —  {label}\n{txt}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--mag_checkpoint", default="Models/Magnitude_UNet_New.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--frames", default="",
                    help="comma-sep SPLIT INDICES; empty => random --n_frames")
    ap.add_argument("--n_frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--n_model", type=int, default=24)
    ap.add_argument("--n_emp", type=int, default=80)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--guard", type=int, default=48)
    ap.add_argument("--min_sep", type=int, default=12)
    ap.add_argument("--out_dir", default="Conditional DDPM/results/cond_fields_fused")
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
    n_ocean = max(int(ocean_np.sum()), 1)

    model = IC.StreamFunctionUNet(in_ch=2, base_ch=ca.get("base_ch", 64),
        time_dim=ca.get("time_dim", 256), cond_ch=cond_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = IC.DDPM(T=ca.get("T", 1000), beta_schedule=ca.get("schedule", "cosine"),
        device=device, noise_type=ca.get("noise_type", "div_free"),
        spectral_filter=ckpt.get("spectral_filter", None))

    mag_net, sm, ss = load_magnitude_model(args.mag_checkpoint, device)
    print(f"  magnitude UNet: speed_mean={sm:.4f} speed_std={ss:.4f}")

    fields = ds.fields.cpu().numpy(); N = fields.shape[0]

    if args.frames.strip():
        idxs = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        rng = np.random.default_rng(args.seed)
        idxs = sorted(rng.choice(len(ds.valid), size=max(1, args.n_frames),
                                 replace=False).tolist())
    print(f"frames (split indices): {idxs}\n")

    hdr = (f"{'frame':>6} {'%kn':>5} | {'KNOWN spd':>9} {'ang':>5} {'rmse%':>6}"
           f" -> {'spd':>5} {'ang':>5} {'rmse%':>6} | {'UNOBS spd':>9} {'ang':>5}"
           f" {'rmse%':>6} -> {'spd':>5} {'ang':>5} {'rmse%':>6} |"
           f" {'r_dir':>6} {'r_vraw':>6} {'r_vfus':>6}")
    print(hdr)

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

        # ---- model ensemble ----
        sargs = argparse.Namespace(pred_type=pred_type,
            inference_steps=args.inference_steps, capture_every=10 ** 9,
            n_ensemble=args.n_model)
        _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                          sargs, device, base_seed=src_idx)

        # ---- UNet speed (normalized units), fuse ----
        spd_phys = np.sqrt((src ** 2).sum(axis=0)) * data_std
        speed_norm = predict_speed_norm(mag_net, sm, ss, spd_phys, pm,
                                        land_np, data_std, device,
                                        cond=b["cond"])
        fused = fuse_members(members, speed_norm, ocean_np)

        # ---- field fidelity (avg over members) ----
        k_raw = avg_member_metrics(members, src, pm_ocean)
        k_fus = avg_member_metrics(fused, src, pm_ocean)
        u_raw = avg_member_metrics(members, src, unobs)
        u_fus = avg_member_metrics(fused, src, unobs)
        a_raw = avg_member_metrics(members, src, ocean_np)
        a_fus = avg_member_metrics(fused, src, ocean_np)

        # ---- empirical neighbour posterior (path + priors) ----
        obs_src = src[:, pm_ocean]; obs_all = fields[:, :, pm_ocean]
        npath = max(int(pm_ocean.sum()), 1)
        dist = ((obs_all - obs_src[None]) ** 2).sum(axis=(1, 2)) / (2 * npath)
        src_priors = np.concatenate([fields[src_f - L] for L in lags], axis=0)
        src_p_ocean = src_priors[:, ocean_np]; max_lag = max(lags)
        prior_dist = np.full(N, np.inf, dtype=np.float64)
        f_idx = np.arange(max_lag, N)
        acc = np.zeros(f_idx.shape[0], dtype=np.float64); c = 0
        for li, L in enumerate(lags):
            cand = fields[f_idx - L][:, :, ocean_np]
            ref = src_p_ocean[2 * li:2 * li + 2]
            acc += ((cand - ref[None]) ** 2).sum(axis=(1, 2)); c += 2
        prior_dist[f_idx] = acc / (c * n_ocean)
        dist = dist + prior_dist
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
        empirical = [src] + [fields[f] for f in picks]

        # ---- uncertainty correlations ----
        emp_dir = IC.directional_spread(empirical, ocean_np)
        mod_dir = IC.directional_spread(members, ocean_np)
        emp_vec = vector_spread(empirical, ocean_np, "abs")
        mod_vraw = vector_spread(members, ocean_np, "abs")
        mod_vfus = vector_spread(fused, ocean_np, "abs")
        vd = ocean_np & np.isfinite(emp_dir) & np.isfinite(mod_dir)
        vv = ocean_np & np.isfinite(emp_vec) & np.isfinite(mod_vfus)
        r_dir = pcorr(emp_dir[vd], mod_dir[vd])
        r_vraw = pcorr(emp_vec[vv], mod_vraw[vv])
        r_vfus = pcorr(emp_vec[vv], mod_vfus[vv])

        print(f"{src_f:>6} {cov:>4.1f}% | "
              f"{k_raw[0]:>9.2f} {k_raw[1]:>5.1f} {k_raw[2]:>6.0f} -> "
              f"{k_fus[0]:>5.2f} {k_fus[1]:>5.1f} {k_fus[2]:>6.0f} | "
              f"{u_raw[0]:>9.2f} {u_raw[1]:>5.1f} {u_raw[2]:>6.0f} -> "
              f"{u_fus[0]:>5.2f} {u_fus[1]:>5.1f} {u_fus[2]:>6.0f} | "
              f"{r_dir:>+6.3f} {r_vraw:>+6.3f} {r_vfus:>+6.3f}"
              f"  || ALL rmse {a_raw[2]:>3.0f}->{a_fus[2]:>3.0f}%")
        agg.append((k_raw, k_fus, u_raw, u_fus, r_dir, r_vraw, r_vfus, a_raw, a_fus))

        # ---- render ----
        fused_mean = np.mean(fused, axis=0).astype(np.float32)
        txt = (f"split idx {src_idx} (frame {src_f})  cov {cov:.1f}%  |  "
               f"known mag {k_raw[0]:.2f}->{k_fus[0]:.2f}x  ang {k_fus[1]:.0f}deg  |  "
               f"r_dir {r_dir:+.2f}  r_vec {r_vraw:+.2f}->{r_vfus:+.2f}")
        render_panel(os.path.join(args.out_dir, f"fused_frame{src_f}.png"),
                     f"frame {src_f}", src, pm, members[0], fused[0], fused_mean,
                     mod_dir, land_np, data_std, cov, txt)

    a = np.array([[*g[0], *g[1], *g[2], *g[3], g[4], g[5], g[6], *g[7], *g[8]]
                   for g in agg], dtype=np.float64)
    m = np.nanmean(a, axis=0)
    print(f"\n  N={len(agg)} frames  MEAN")
    print(f"  KNOWN  speed {m[0]:.2f}->{m[3]:.2f}x  angle {m[1]:.1f}->{m[4]:.1f}deg"
          f"  rmse {m[2]:.0f}->{m[5]:.0f}%")
    print(f"  UNOBS  speed {m[6]:.2f}->{m[9]:.2f}x  angle {m[7]:.1f}->{m[10]:.1f}deg"
          f"  rmse {m[8]:.0f}->{m[11]:.0f}%")
    print(f"  r_dir(dir-only) {m[12]:+.3f}   r_vec raw {m[13]:+.3f} -> fused {m[14]:+.3f}")
    print(f"  ALL-OCEAN  rmse {m[17]:.0f}->{m[20]:.0f}%   speed {m[15]:.2f}->{m[18]:.2f}x"
          f"  angle {m[16]:.1f}deg")
    print(f"\n  panels saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
