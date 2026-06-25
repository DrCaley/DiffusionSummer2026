"""Uncertainty-calibration validation for the conditional stream-function model.

The active-sensing premise: the diffusion ensemble's per-cell disagreement is a
usable UNCERTAINTY map telling the robot where to measure next.  This only holds
if the model's uncertainty matches the TRUE data-manifold uncertainty given the
observations.  This script tests that empirically.

Procedure
---------
1. Pick a source frame and reveal its field on a short robot path (the "known
   area").
2. EMPIRICAL posterior: search the dataset for the 9 OTHER fields whose values
   on the known cells best match the source (temporally de-duplicated so they
   are genuinely distinct fields, not adjacent-hour near-copies).  Those 9 + the
   source = 10 data-consistent fields.
3. MODEL posterior: condition the model on the same observations (+ priors +
   geometry) and draw 10 ensemble members.
4. Build a directional-uncertainty heatmap (1 - R) for each set and compare
   them (spatial Pearson correlation + side-by-side render).  If the spatial
   patterns agree, the model's uncertainty is calibrated to the real data.
5. Also report which of the 10 model draws is closest to the source ground
   truth (angle + RMSE) and render it.

Best model by default: Models/StreamFn_Cond_x0_mag.pt (x0 + magnitude loss,
fixed div-free noise) — the current best single model; plain ancestral DDPM.
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

import infer_cond as IC  # noqa: E402


def pcorr(a, b, eps=1e-12):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def vector_std_map(members, ocean_np):
    """Per-cell vector spread sqrt(mean_k ||m_k - mean||^2)  -> (H, W), NaN land."""
    arr = np.stack(members, axis=0)                  # (K, 2, H, W)
    mean = arr.mean(axis=0, keepdims=True)
    dev2 = ((arr - mean) ** 2).sum(axis=1)           # (K, H, W)
    out = np.sqrt(dev2.mean(axis=0)).astype(np.float32)
    out[~ocean_np] = np.nan
    return out


def heatmap(ax, m, land_d, title, vmax=None):
    sp = m.T.copy()
    vmax = vmax if vmax is not None else np.nanpercentile(sp, 99)
    im = ax.imshow(sp, origin="lower", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-6),
                   extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                   aspect="auto")
    ax.imshow(land_d, origin="lower",
              cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
              extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
              aspect="auto", zorder=2)
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=0, help="0=train")
    ap.add_argument("--src_idx", type=int, default=500, help="index into split")
    ap.add_argument("--frames", default=None,
                    help="Comma-separated list of src indices for a NUMBERS-ONLY "
                         "multi-frame sweep (no figures); prints r per frame + "
                         "mean/std summary.  Overrides --src_idx when set.")
    ap.add_argument("--path_steps", type=int, default=90, help="short known path")
    ap.add_argument("--n", type=int, default=10, help="ensemble / neighbour count")
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--guard", type=int, default=48,
                    help="exclude frames within +/- guard of source (anti-dup)")
    ap.add_argument("--min_sep", type=int, default=12,
                    help="min temporal separation between chosen neighbours")
    ap.add_argument("--match_priors", action="store_true",
                    help="Make the empirical search condition on the SAME info "
                         "the model sees: rank candidate frames by path-cell "
                         "match AND temporal-prior (lag) match, each per-cell "
                         "normalized.  This is the deployment-faithful test "
                         "(both posteriors condition on path + recent history).")
    ap.add_argument("--out_dir", default="Conditional DDPM/results/uncertainty_validation")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = ckpt.get("data_std"); phys = float(data_std) if data_std else 1.0
    print(f"model: {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch')} "
          f"pred={pred_type}")

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

    fields = ds.fields.cpu().numpy()                  # (N, 2, H, W)
    N = fields.shape[0]
    n_ocean = max(int(ocean_np.sum()), 1)

    def eval_frame(src_idx, make_fig):
        """Compute empirical-vs-model uncertainty correlation for one frame."""
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
        src = b["target"].cpu().numpy()                   # (2, H, W) normalized
        pm = b["path_mask"]
        pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
        pm_ocean = pm & ocean_np
        src_f = int(ds.valid[src_idx])
        cov = 100.0 * pm_ocean.sum() / ocean_np.sum()

        # ---- EMPIRICAL posterior: distinct data fields matching the known area ----
        obs_src = src[:, pm_ocean]                         # (2, npath)
        obs_all = fields[:, :, pm_ocean]                  # (N, 2, npath)
        npath = max(int(pm_ocean.sum()), 1)
        dist = ((obs_all - obs_src[None]) ** 2).sum(axis=(1, 2)) / (2 * npath)   # (N,)

        if args.match_priors:
            src_priors = np.concatenate([fields[src_f - L] for L in lags], axis=0)
            src_p_ocean = src_priors[:, ocean_np]
            max_lag = max(lags)
            prior_dist = np.full(N, np.inf, dtype=np.float64)
            f_idx = np.arange(max_lag, N)
            acc = np.zeros(f_idx.shape[0], dtype=np.float64)
            c = 0
            for li, L in enumerate(lags):
                cand = fields[f_idx - L][:, :, ocean_np]
                ref = src_p_ocean[2 * li:2 * li + 2]
                acc += ((cand - ref[None]) ** 2).sum(axis=(1, 2))
                c += 2
            prior_dist[f_idx] = acc / (c * n_ocean)
            dist = dist + prior_dist

        order = np.argsort(dist)
        picks = []
        for f in order:
            f = int(f)
            if not np.isfinite(dist[f]):
                continue
            if abs(f - src_f) <= args.guard:
                continue
            if any(abs(f - p) < args.min_sep for p in picks):
                continue
            picks.append(f)
            if len(picks) == args.n - 1:
                break
        empirical = [src] + [fields[f] for f in picks]

        # ---- MODEL posterior: n ensemble members on the same conditioning ----
        sargs = argparse.Namespace(pred_type=pred_type,
            inference_steps=args.inference_steps, capture_every=10 ** 9,
            n_ensemble=args.n)
        _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                          sargs, device, base_seed=src_idx)

        emp_dir = IC.directional_spread(empirical, ocean_np)
        mod_dir = IC.directional_spread(members, ocean_np)
        emp_std = vector_std_map(empirical, ocean_np)
        mod_std = vector_std_map(members, ocean_np)
        valid = ocean_np & np.isfinite(emp_dir) & np.isfinite(mod_dir)
        r_dir = pcorr(emp_dir[valid], mod_dir[valid])
        r_std = pcorr(emp_std[valid], mod_std[valid])

        errs = [float(np.nanmean(IC.angle_error_deg(m, src, ocean_np)[ocean_np]))
                for m in members]
        rmses = [float(np.sqrt(np.mean((m[:, ocean_np] - src[:, ocean_np]) ** 2)))
                 for m in members]
        best_k = int(np.argmin(errs))

        if not make_fig:
            return src_f, cov, r_dir, r_std, errs[best_k]

        # ---- render (single-frame mode) ----
        land_d = land_np.T
        fig, ax = plt.subplots(2, 3, figsize=(20, 11), dpi=95)
        a = ax.flatten()
        IC.plot_field(a[0], (src[0] * phys).T, (src[1] * phys).T, land_d,
                      f"Ground truth  (frame {src_f})")
        IC.plot_path(a[1], pm.T, land_d, f"Known area  ({int(pm_ocean.sum())} cells, {cov:.1f}%)")
        bp = members[best_k]
        IC.plot_field(a[2], (bp[0] * phys).T, (bp[1] * phys).T, land_d,
                      f"Closest model draw (#{best_k})  angle={errs[best_k]:.1f}deg")
        vmax = max(np.nanpercentile(emp_dir[valid], 99), np.nanpercentile(mod_dir[valid], 99))
        heatmap(a[3], emp_dir, land_d, f"Empirical uncertainty  ({args.n} data fields)", vmax)
        heatmap(a[4], mod_dir, land_d, f"Model uncertainty  ({args.n} diffusion draws)", vmax)
        a[5].scatter(emp_dir[valid], mod_dir[valid], s=4, alpha=0.3, c="tab:blue")
        lim = max(emp_dir[valid].max(), mod_dir[valid].max())
        a[5].plot([0, lim], [0, lim], "k--", lw=1)
        a[5].set_xlabel("empirical spread"); a[5].set_ylabel("model spread")
        a[5].set_title(f"Per-cell agreement   r(dir)={r_dir:+.2f}  r(std)={r_std:+.2f}",
                       fontsize=11)
        a[5].set_aspect("equal", "box")
        plt.suptitle(
            f"Uncertainty calibration — does model spread match the data manifold?\n"
            f"frame {src_f}, {cov:.1f}% known   |   directional-spread correlation = {r_dir:+.2f}",
            fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"uncert_frame{src_f}.png")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"saved -> {out}")
        return src_f, cov, r_dir, r_std, errs[best_k]

    if args.frames:
        idxs = [int(x) for x in args.frames.split(",")]
        print(f"\nmulti-frame sweep ({len(idxs)} frames, n={args.n}, "
              f"match_priors={args.match_priors}):")
        print(f"  {'frame':>6} | {'%known':>6} | {'r_dir':>6} | {'r_std':>6} | {'best°':>6}")
        rds, rss = [], []
        for ix in idxs:
            src_f, cov, r_dir, r_std, best_ang = eval_frame(ix, make_fig=False)
            rds.append(r_dir); rss.append(r_std)
            print(f"  {src_f:>6} | {cov:>5.1f}% | {r_dir:>+.3f} | {r_std:>+.3f} | {best_ang:>5.1f}")
        rds = np.array(rds); rss = np.array(rss)
        print(f"\n  r_dir: mean={rds.mean():+.3f}  std={rds.std():.3f}  "
              f"min={rds.min():+.3f}  max={rds.max():+.3f}")
        print(f"  r_std: mean={rss.mean():+.3f}  std={rss.std():.3f}")
    else:
        src_f, cov, r_dir, r_std, best_ang = eval_frame(args.src_idx, make_fig=True)
        print(f"\nframe {src_f}  ({cov:.1f}% known):  r_dir={r_dir:+.3f}  "
              f"r_std={r_std:+.3f}  closest_draw_angle={best_ang:.1f}deg")



if __name__ == "__main__":
    main()
