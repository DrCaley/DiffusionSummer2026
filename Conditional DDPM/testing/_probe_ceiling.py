"""
Feasibility / ceiling probe for posterior-matching (DATA ONLY, no model).

Question it answers: if we were to TRAIN the model to match the empirical
posterior's directional-spread map, what is the maximum r_dir we could ever
hope to reach, and how clean is that training target?

A model can never correlate with the empirical uncertainty map better than that
map correlates with ITSELF (split-half reliability).  So the achievable r_dir
ceiling is governed by the target's reliability, which we measure here as a
function of the neighbour count n_emp.

For each frame and each n_emp:
  * build the empirical posterior (path + temporal-prior matching, exactly as
    uncertainty_validation.py / _probe_calib_diag.py)
  * split the n_emp neighbours into two disjoint halves
  * compute the directional-spread map of each half
  * r_half = Pearson(halfA, halfB)
  * Spearman-Brown full reliability  r_full = 2*r_half / (1 + r_half)
    (r_full ~ the ceiling for a perfect model measured against this target)

NO model is loaded and NO diffusion sampling is run -> CPU-fast, no GPU needed.
"""
import argparse
import os
import sys

import numpy as np
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_emp_list", default="20,40,80,160",
                    help="neighbour counts to test target reliability at")
    ap.add_argument("--lags", default="13,25")
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--guard", type=int, default=48)
    ap.add_argument("--min_sep", type=int, default=12)
    args = ap.parse_args()

    lags = tuple(int(x) for x in args.lags.split(","))
    n_emp_list = sorted(int(x) for x in args.n_emp_list.split(","))
    n_emp_max = max(n_emp_list)

    ds = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=0.0, data_std=None,
        path_steps=args.path_steps, deterministic=True)
    land_np = ds.land_mask.cpu().numpy().astype(bool); ocean_np = ~land_np
    n_ocean = max(int(ocean_np.sum()), 1)
    fields = ds.fields.cpu().numpy(); N = fields.shape[0]

    def neighbours(src_idx, k):
        """Return [src] + up to k nearest dataset frames (path + prior match)."""
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
        src = b["target"].cpu().numpy()
        pm = b["path_mask"]
        pm = (pm.cpu().numpy() if hasattr(pm, "cpu") else np.asarray(pm)).astype(bool)
        pm_ocean = pm & ocean_np
        src_f = int(ds.valid[src_idx])
        cov = 100.0 * pm_ocean.sum() / ocean_np.sum()

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
            if len(picks) == k:
                break
        return src_f, cov, src, [fields[f] for f in picks]

    rng = np.random.default_rng(args.seed)
    idxs = sorted(int(x) for x in rng.choice(len(ds.valid),
                  size=min(args.n_frames, len(ds.valid)), replace=False))
    print(f"ceiling probe (DATA ONLY, no model): {len(idxs)} frames, seed={args.seed}")
    print(f"n_emp tested: {n_emp_list}   lags={lags}   path_steps={args.path_steps}\n")

    # r_full[n_emp] = list of per-frame Spearman-Brown reliabilities
    rfull = {k: [] for k in n_emp_list}
    rhalf = {k: [] for k in n_emp_list}
    for ix in idxs:
        src_f, cov, src, nbrs = neighbours(ix, n_emp_max)
        if len(nbrs) < n_emp_list[0]:
            continue
        for k in n_emp_list:
            if len(nbrs) < k:
                continue
            sub = nbrs[:k]
            half = k // 2
            ea = [src] + sub[:half]
            eb = [src] + sub[half:2 * half]
            da = IC.directional_spread(ea, ocean_np)
            db = IC.directional_spread(eb, ocean_np)
            v = ocean_np & np.isfinite(da) & np.isfinite(db)
            rh = pcorr(da[v], db[v])
            rh = max(rh, 0.0)
            rf = 2 * rh / (1 + rh) if (1 + rh) > 0 else 0.0
            rhalf[k].append(rh); rfull[k].append(rf)

    print(f"{'n_emp':>6} {'r_half':>8} {'r_full(SB)':>11} {'std':>7} {'n':>4}")
    print("  (r_full = ceiling on r_dir a PERFECT model could reach vs this target)")
    for k in n_emp_list:
        if not rfull[k]:
            continue
        rh = np.array(rhalf[k]); rf = np.array(rfull[k])
        print(f"{k:>6} {rh.mean():>+8.3f} {rf.mean():>+11.3f} "
              f"{rf.std():>7.3f} {len(rf):>4}")

    # headline
    kbig = n_emp_list[-1]
    if rfull[kbig]:
        ceil = float(np.mean(rfull[kbig]))
        print(f"\nCEILING ESTIMATE (n_emp={kbig}): r_dir <= {ceil:+.3f}")
        print(f"CURRENT model true calibration (from sweep r_corr): ~+0.50")
        print(f"=> learnable headroom: ~{ceil - 0.50:+.3f}  "
              f"({'WORTH training' if ceil - 0.50 > 0.15 else 'MARGINAL'})")
        print("\ninterpretation:")
        print("  * r_full rising with n_emp -> target sharpens with more neighbours")
        print("  * r_full plateauing high (>0.8) -> target is CLEAN, ceiling is real")
        print("  * r_full low/flat (<0.6) -> target too noisy to train on; STOP")


if __name__ == "__main__":
    main()
