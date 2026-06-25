"""
Calibration-ceiling diagnostic (temporary).

Answers the question grinding losses cannot: is r_dir low because the MODEL's
uncertainty pattern is wrong, or because we estimate each spread map from only
~50 Monte-Carlo samples (regression dilution)?

For each frame it reports:
  r_dir            Pearson(emp_spread, mod_spread)            (the headline)
  rho_dir          Spearman(emp_spread, mod_spread)          (rank, tail-robust)
  r_partial        Pearson with distance-to-path regressed   (non-trivial calib)
                   out of BOTH maps  -> "do equidistant cells rank right?"
  r_mm             split-half model reliability  (two disjoint halves of the
                   model draws, correlated)  -> model-side measurement ceiling
  r_ee             split-half empirical reliability                -> data-side ceiling
  r_corr           r_dir / sqrt(r_mm * r_ee)  attenuation-corrected calibration

Interpretation:
  * If r_mm, r_ee are themselves low (~0.5), the ceiling is SAMPLING NOISE: the
    cure is more draws / more frames, not retraining.  r_corr is the real signal.
  * If r_mm, r_ee are high (~0.9) but r_dir stays ~0.45, the model's uncertainty
    PATTERN is genuinely wrong: that needs a calibration-aware objective.

Empirical matching replicates uncertainty_validation.py (match_priors) exactly.
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", ".."))
for _p in (_here, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), _root):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import infer_cond as IC  # noqa: E402


def pcorr(a, b, eps=1e-12):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def partial_pcorr(a, b, z, eps=1e-12):
    """Pearson(a, b) with covariate z linearly regressed out of both."""
    def resid(y):
        Z = np.stack([np.ones_like(z), z], axis=1)        # (n, 2)
        coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return y - Z @ coef
    return pcorr(resid(a), resid(b), eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--frames", default="1500,3000,11865")
    ap.add_argument("--n_frames", type=int, default=0,
                    help="if >0, randomly sample this many valid frames "
                         "(overrides --frames) to measure the r_dir distribution")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for --n_frames random frame sampling")
    ap.add_argument("--conc_gate", type=float, default=0.0,
                    help="emp_conc threshold for the 'structured' subset; "
                         "0 = use the median emp_conc across sampled frames")
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--n_model", type=int, default=120,
                    help="model draws (split in half for reliability)")
    ap.add_argument("--n_emp", type=int, default=40,
                    help="empirical neighbours (split in half for reliability)")
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--guard", type=int, default=48)
    ap.add_argument("--min_sep", type=int, default=12)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = ckpt.get("data_std")
    print(f"model: {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch')} "
          f"pred={pred_type}  n_model={args.n_model}  n_emp={args.n_emp}")

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

    fields = ds.fields.cpu().numpy(); N = fields.shape[0]

    def spread_dir(members):
        return IC.directional_spread(members, ocean_np)

    @torch.no_grad()
    def eval_frame(src_idx):
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
        src = b["target"].cpu().numpy()
        pm = b["path_mask"]
        pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
        pm_ocean = pm & ocean_np
        src_f = int(ds.valid[src_idx])
        cov = 100.0 * pm_ocean.sum() / ocean_np.sum()

        # ---- empirical matching (path + priors), exactly as the validated tool ----
        obs_src = src[:, pm_ocean]
        obs_all = fields[:, :, pm_ocean]
        npath = max(int(pm_ocean.sum()), 1)
        dist = ((obs_all - obs_src[None]) ** 2).sum(axis=(1, 2)) / (2 * npath)
        src_priors = np.concatenate([fields[src_f - L] for L in lags], axis=0)
        src_p_ocean = src_priors[:, ocean_np]
        max_lag = max(lags)
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
            if not np.isfinite(dist[f]):
                continue
            if abs(f - src_f) <= args.guard:
                continue
            if any(abs(f - p) < args.min_sep for p in picks):
                continue
            picks.append(f)
            if len(picks) == args.n_emp - 1:
                break
        empirical = [src] + [fields[f] for f in picks]

        # ---- model ensemble (large) ----
        sargs = argparse.Namespace(pred_type=pred_type,
            inference_steps=args.inference_steps, capture_every=10 ** 9,
            n_ensemble=args.n_model)
        _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                          sargs, device, base_seed=src_idx)

        # ---- spreads + headline metrics ----
        emp_dir = spread_dir(empirical); mod_dir = spread_dir(members)
        valid = ocean_np & np.isfinite(emp_dir) & np.isfinite(mod_dir)
        ev, mv = emp_dir[valid], mod_dir[valid]
        r_dir = pcorr(ev, mv)
        rho_dir = float(spearmanr(ev, mv).correlation)

        # empirical concentration: spatial coefficient of variation of the TRUE
        # uncertainty map.  Low emp_conc => uncertainty is ~uniform => no "where to
        # measure" signal exists => r_dir is ill-conditioned (a metric artifact).
        emp_conc = float(ev.std() / (abs(ev.mean()) + 1e-9))

        # distance-to-path, partialled out of both
        dpath = ndimage.distance_transform_edt(~pm).astype(np.float32)
        r_partial = partial_pcorr(ev, mv, dpath[valid])

        # ---- split-half reliability ----
        rng = np.random.default_rng(src_idx)
        mi = rng.permutation(len(members))
        ma = [members[k] for k in mi[:len(members) // 2]]
        mb = [members[k] for k in mi[len(members) // 2:]]
        r_mm = pcorr(spread_dir(ma)[valid], spread_dir(mb)[valid])
        ei = rng.permutation(len(empirical))
        ea = [empirical[k] for k in ei[:len(empirical) // 2]]
        eb = [empirical[k] for k in ei[len(empirical) // 2:]]
        r_ee = pcorr(spread_dir(ea)[valid], spread_dir(eb)[valid])

        denom = np.sqrt(max(r_mm, 1e-6) * max(r_ee, 1e-6))
        r_corr = r_dir / denom if denom > 0 else float("nan")
        return src_f, cov, r_dir, rho_dir, r_partial, r_mm, r_ee, r_corr, emp_conc

    if args.n_frames > 0:
        rng0 = np.random.default_rng(args.seed)
        n_valid = len(ds.valid)
        k = min(args.n_frames, n_valid)
        idxs = sorted(int(x) for x in
                      rng0.choice(n_valid, size=k, replace=False))
        print(f"random sweep: {k} frames, seed={args.seed}, n_valid={n_valid}")
    else:
        idxs = [int(x) for x in args.frames.split(",")]
    print(f"\n{'frame':>6} {'%kn':>5} {'r_dir':>7} {'rho':>7} {'r_part':>7} "
          f"{'r_mm':>6} {'r_ee':>6} {'r_corr':>7} {'e_conc':>7}")
    rows = []
    for ix in idxs:
        row = eval_frame(ix)
        rows.append(row)
        sf, cov, rd, rho, rp, rmm, ree, rc, ec = row
        print(f"{sf:>6} {cov:>4.1f}% {rd:>+7.3f} {rho:>+7.3f} {rp:>+7.3f} "
              f"{rmm:>6.3f} {ree:>6.3f} {rc:>+7.3f} {ec:>7.3f}")
    arr = np.array([[r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in rows])
    m = arr.mean(axis=0); sd = arr.std(axis=0)
    rd_col = arr[:, 0]; conc_col = arr[:, 6]
    # emp_conc-weighted r_dir: down-weights ill-conditioned uniform frames
    w = np.clip(conc_col, 0.0, None)
    wmean = float((w * rd_col).sum() / (w.sum() + 1e-12))
    # structured subset: frames where a real spatial signal exists
    gate = args.conc_gate if args.conc_gate > 0 else float(np.median(conc_col))
    smask = conc_col >= gate
    smean = float(rd_col[smask].mean()) if smask.any() else float("nan")
    sstd = float(rd_col[smask].std()) if smask.any() else float("nan")
    print(f"\n  N={len(rows)} frames")
    print(f"  unweighted   : r_dir={m[0]:+.3f} ± {sd[0]:.3f}  rho={m[1]:+.3f}  "
          f"r_partial={m[2]:+.3f}")
    print(f"  reliability  : r_mm={m[3]:.3f}  r_ee={m[4]:.3f}  r_corr={m[5]:+.3f}")
    print(f"  emp_conc-wtd : r_dir={wmean:+.3f}   (down-weights uniform-uncertainty frames)")
    print(f"  structured   : r_dir={smean:+.3f} ± {sstd:.3f}  "
          f"(emp_conc>={gate:.2f}, n={int(smask.sum())}/{len(rows)})")
    print("\ninterpretation:")
    print("  r_mm/r_ee near 1.0  -> measurement is clean; r_dir is the real story")
    print("  r_mm/r_ee low       -> sampling noise caps r_dir; r_corr is true calib")
    print("  r_partial << r_dir  -> most of r_dir is just the trivial near/far gradient")
    print("  structured >> unweighted -> low mean is driven by uniform frames (artifact)")


if __name__ == "__main__":
    main()
