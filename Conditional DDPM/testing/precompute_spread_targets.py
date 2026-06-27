"""
Precompute the empirical directional-spread TARGET map for every frame of a
split, for the directional-spread-matching training loss
(``DDPM/model/diffusion.py::training_loss_streamfn_spread``).

For each frame the target is the per-cell directional spread (1 − |mean unit
vector|) of an empirical posterior ensemble: the frame itself plus its ``n_emp−1``
nearest neighbours under the SAME path + prior matching used by the calibration
probes (``_probe_calib_diag.py``).  This is exactly the empirical map r_dir is
correlated against at evaluation time, so training to match it optimises r_dir
directly.

Alignment guarantee
-------------------
The observation path for frame ``idx`` is built with ``build_cond(ds, idx,
path_steps, seed=idx)`` — byte-for-byte identical to what
``ConditionalOceanDataset(..., deterministic=True, path_steps=<this int>)``
produces for ``ds[idx]``.  So the path the trainer reveals to the model and the
path this target was matched on are the same.  ``--path_steps`` MUST be a fixed
int and MUST equal the trainer's ``--path_steps``; ``--lags`` must match too.

Output: a ``.npy`` of shape ``(len(split), H, W)`` float32, land cells = 0, row
``i`` ↔ dataset index ``i``.  Saved alongside a ``.json`` sidecar recording the
settings so training can sanity-check them.

Run from the workspace root, e.g.:
    python "Conditional DDPM/testing/precompute_spread_targets.py" \
        --pickle Datasets/data_divfree_chrono.pickle --split 0 \
        --lags 13,25 --path_steps 90 --n_emp 80 \
        --out "Conditional DDPM/spread_targets_train.npy"
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", ".."))
for _p in (_here, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), _root):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import infer_cond as IC  # noqa: E402


def _parse_lags(spec):
    return tuple(int(s) for s in spec.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="Datasets/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--lags", type=_parse_lags, default=(13, 25))
    ap.add_argument("--data_mean", type=float, default=0.0)
    ap.add_argument("--path_steps", type=int, default=90,
                    help="FIXED robot-path length; must equal the trainer's "
                         "--path_steps (and be a single int).")
    ap.add_argument("--n_emp", type=int, default=80,
                    help="empirical neighbours (incl. the frame itself); 80 is "
                         "the reliability sweet spot from the ceiling probe.")
    ap.add_argument("--guard", type=int, default=48,
                    help="exclude neighbours within +-guard frames of the source "
                         "(temporal-autocorrelation leakage band).")
    ap.add_argument("--min_sep", type=int, default=12,
                    help="minimum frame separation between chosen neighbours.")
    ap.add_argument("--out", required=True,
                    help="output .npy path; a .json sidecar is written next to it.")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE-TEST ONLY: process just the first N frames (rest "
                         "stay 0).  0 = all frames (use this for real targets).")
    args = ap.parse_args()

    _, data_std = IC.ConditionalOceanDataset.compute_stats(args.pickle, split=0)
    ds = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=args.lags,
        data_mean=args.data_mean, data_std=data_std,
        path_steps=args.path_steps, deterministic=True)

    land_np  = ds.land_mask.cpu().numpy().astype(bool)
    ocean_np = ~land_np
    n_ocean  = max(int(ocean_np.sum()), 1)
    H, W     = land_np.shape
    fields   = ds.fields.cpu().numpy()          # (N, 2, H, W) normalized, land=0
    N        = fields.shape[0]
    n_split  = len(ds.valid)
    max_lag  = max(args.lags)

    print(f"split={args.split}  frames={n_split}  N={N}  grid={H}x{W}  "
          f"ocean={n_ocean}  lags={args.lags}  path_steps={args.path_steps}  "
          f"n_emp={args.n_emp}")

    # Candidate prior arrays do NOT depend on the source frame — hoist them out
    # of the per-frame loop.  Expanding ||cand - ref||^2 = ||cand||^2 - 2 cand.ref
    # + ||ref||^2 turns the per-frame distance into a BLAS matrix-vector product
    # (cand_flat @ ref) plus precomputed constants, ~10x faster than broadcasting
    # a (M, 2, n_ocean) temporary every frame.  cand_flat[li][m] is the ocean
    # vector of frame (max_lag + m) - L for lag li.
    f_idx    = np.arange(max_lag, N)
    cand_flat, cand_sq = [], []
    for L in args.lags:
        cf = fields[f_idx - L][:, :, ocean_np].reshape(f_idx.shape[0], -1)  # (M, 2*n_ocean)
        cand_flat.append(cf.astype(np.float64))
        cand_sq.append((cf.astype(np.float64) ** 2).sum(axis=1))           # (M,)
    c_lags = 2 * len(args.lags)

    # Same expansion for the PATH distance.  The path cells differ per frame, but
    # ||field_n||^2 over those cells can be read from a precomputed per-frame,
    # per-cell square table, and the cross term is one gemv — so no (N, 2, npath)
    # temporary is built per frame.  fields_g/sq_g are (N, 2, H*W) views/tables.
    fields_g = fields.reshape(N, 2, H * W).astype(np.float64)               # (N,2,HW)
    fields_sq_g = fields_g ** 2                                             # (N,2,HW)

    targets = np.zeros((n_split, H, W), dtype=np.float32)
    t0 = time.time()

    n_proc = n_split if args.limit <= 0 else min(args.limit, n_split)
    if n_proc < n_split:
        print(f"** SMOKE MODE: processing only {n_proc}/{n_split} frames **")
    for i in range(n_proc):
        b  = IC.build_cond(ds, i, args.path_steps, seed=i)
        src = b["target"].cpu().numpy()                              # (2, H, W)
        pm  = b["path_mask"]
        pm  = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
        pm_ocean = pm & ocean_np
        src_f = int(ds.valid[i])

        # ---- path distance over all N frames, via the ||a-b||^2 expansion ----
        path_lin = np.flatnonzero(pm_ocean.reshape(-1))             # (npath,)
        npath    = max(path_lin.size, 1)
        src_p    = src.reshape(2, H * W)[:, path_lin].reshape(-1).astype(np.float64)  # (2*npath,)
        Fp       = fields_g[:, :, path_lin].reshape(N, -1)         # (N, 2*npath)
        term_sq  = fields_sq_g[:, :, path_lin].reshape(N, -1).sum(axis=1)  # (N,)
        dist = (term_sq - 2.0 * (Fp @ src_p) + (src_p ** 2).sum()) / (2 * npath)

        # ---- prior distance (only defined for f >= max_lag), via gemv ----
        src_priors  = np.concatenate([fields[src_f - L] for L in args.lags], axis=0)
        src_p_ocean = src_priors[:, ocean_np]                       # (2*nlags, n_ocean)
        prior_dist = np.full(N, np.inf, dtype=np.float64)
        acc = np.zeros(f_idx.shape[0], dtype=np.float64)
        for li in range(len(args.lags)):
            ref = src_p_ocean[2 * li:2 * li + 2].reshape(-1).astype(np.float64)  # (2*n_ocean,)
            cross = cand_flat[li] @ ref                            # (M,)
            acc += cand_sq[li] - 2.0 * cross + (ref ** 2).sum()
        prior_dist[f_idx] = acc / (c_lags * n_ocean)
        dist = dist + prior_dist

        # ---- pick neighbours (guard + min separation), exactly as the probe ----
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
        emp_spread = IC.directional_spread(empirical, ocean_np)     # (H, W), NaN land
        targets[i] = np.nan_to_num(emp_spread, nan=0.0).astype(np.float32)

        if (i + 1) % 200 == 0 or i + 1 == n_proc:
            el = time.time() - t0
            eta = el / (i + 1) * (n_proc - i - 1)
            print(f"  {i+1:6d}/{n_proc}  neigh={len(picks)}  "
                  f"{el:6.1f}s elapsed  ~{eta:6.1f}s left", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.save(args.out, targets)
    sidecar = {
        "pickle": args.pickle, "split": args.split, "lags": list(args.lags),
        "data_mean": args.data_mean, "data_std": float(data_std),
        "path_steps": args.path_steps, "n_emp": args.n_emp,
        "guard": args.guard, "min_sep": args.min_sep,
        "n_frames": n_split, "grid": [H, W],
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\nsaved {targets.shape} -> {args.out}")
    print(f"target stats: mean={targets[:, ocean_np].mean():.4f}  "
          f"std={targets[:, ocean_np].std():.4f}  "
          f"min={targets[:, ocean_np].min():.4f}  max={targets[:, ocean_np].max():.4f}")


if __name__ == "__main__":
    main()
