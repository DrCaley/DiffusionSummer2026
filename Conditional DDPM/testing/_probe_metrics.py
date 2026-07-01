"""
FULL-PIPELINE METRICS BENCHMARK — fused (diffusion direction x UNet speed) vs
the diffusion-only baseline, on N random frames.  Reports exactly the five
quantities asked for, each as a spatial mean over ocean cells then averaged
over frames, with a known / unobserved split:

  1. ANGLE correlation   r_ang  = pcorr(cos(theta_pred,theta_true) style) ->
        we use spatial Pearson corr between predicted & true unit-vector
        components stacked [uh; vh] (a proper directional correlation).
  2. MAGNITUDE correlation r_mag = pcorr(||pred||, ||true||) over cells.
  3. VECTOR correlation   r_vec = pcorr([u_pred;v_pred], [u_true;v_true]) ->
        angle AND magnitude together (the honest "did we get the field" number).
  4. MSE                  vector mean-squared error in PHYSICAL units (m/s)^2.
  5. ANGLE loss           mean per-cell angular error in degrees.

Honesty guards already validated in the audit: fused field uses truth ONLY on
the robot path; no leak (proven by empty-obs control). Numbers here are the
ensemble-MEAN field unless --per_member is set.

Run:
  .venv/bin/python "Conditional DDPM/testing/_probe_metrics.py" \
      --checkpoint Models/StreamFn_Cond_x0_mag_spread.pt \
      --mag_checkpoint Models/Magnitude_UNet_New.pt \
      --pickle Datasets/pickles/data_divfree_chrono.pickle \
      --split 2 --n_frames 20 --seed 3 --n_model 24 --path_steps 90
"""
import argparse
import os
import sys

import numpy as np
import torch

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


def field_metrics(pred, true, mask, data_std):
    """All five metrics over `mask` cells. pred,true=(2,H,W) normalized units."""
    if mask.sum() == 0:
        return dict(r_ang=np.nan, r_mag=np.nan, r_vec=np.nan,
                    mse=np.nan, ang=np.nan)
    pu, pv = pred[0][mask], pred[1][mask]
    tu, tv = true[0][mask], true[1][mask]
    ps = np.sqrt(pu ** 2 + pv ** 2); ts = np.sqrt(tu ** 2 + tv ** 2)
    # unit components (direction only)
    puh, pvh = pu / (ps + EPS), pv / (ps + EPS)
    tuh, tvh = tu / (ts + EPS), tv / (ts + EPS)
    # 1. angle correlation: spatial corr of stacked unit components
    r_ang = pcorr(np.concatenate([puh, pvh]), np.concatenate([tuh, tvh]))
    # 2. magnitude correlation
    r_mag = pcorr(ps, ts)
    # 3. vector correlation: stacked raw components (angle + magnitude)
    r_vec = pcorr(np.concatenate([pu, pv]), np.concatenate([tu, tv]))
    # 4. MSE in physical units
    s2 = data_std ** 2
    mse = float((((pu - tu) ** 2 + (pv - tv) ** 2) * s2).mean())
    # 5. mean angle error (deg)
    cos = np.clip(puh * tuh + pvh * tvh, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cos)).mean())
    return dict(r_ang=r_ang, r_mag=r_mag, r_vec=r_vec, mse=mse, ang=ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--mag_checkpoint", default="Models/Magnitude_UNet_New.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--frames", default="")
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--n_model", type=int, default=24)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--per_member", action="store_true",
                    help="average metrics over individual draws (default: mean field)")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)
    print(f"model: {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch')} "
          f"pred={pred_type}  n_model={args.n_model}  device={device}  "
          f"field={'per-member' if args.per_member else 'ensemble-mean'}")

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

    if args.frames.strip():
        idxs = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        rng = np.random.default_rng(args.seed)
        idxs = sorted(rng.choice(len(ds.valid), size=max(1, args.n_frames),
                                 replace=False).tolist())
    print(f"frames (split indices, n={len(idxs)}): {idxs}\n")

    keys = ["r_ang", "r_mag", "r_vec", "mse", "ang"]
    regions = ["whole", "known", "unobs"]
    acc = {(r, m): {k: [] for k in keys} for r in regions for m in ("diff", "fused")}

    for src_idx in idxs:
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
        src = b["target"].cpu().numpy()
        pm = b["path_mask"]
        pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
        pm_ocean = pm & ocean_np
        unobs = ocean_np & ~pm_ocean

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

        masks = dict(whole=ocean_np, known=pm_ocean, unobs=unobs)
        for r, mask in masks.items():
            if args.per_member:
                dvals = [field_metrics(m, src, mask, data_std) for m in members]
                fvals = [field_metrics(f, src, mask, data_std) for f in fused]
                dM = {k: np.nanmean([d[k] for d in dvals]) for k in keys}
                fM = {k: np.nanmean([f[k] for f in fvals]) for k in keys}
            else:
                dM = field_metrics(np.mean(members, axis=0), src, mask, data_std)
                fM = field_metrics(np.mean(fused, axis=0), src, mask, data_std)
            for k in keys:
                acc[(r, "diff")][k].append(dM[k])
                acc[(r, "fused")][k].append(fM[k])

    # ---- report ----
    def mean(r, m, k):
        return float(np.nanmean(acc[(r, m)][k]))

    hdr = (f"\n{'region':>7} {'model':>6} | {'r_angle':>8} {'r_mag':>7} "
           f"{'r_vec':>7} | {'MSE(m/s)^2':>11} {'angle(deg)':>10}")
    print(hdr); print("-" * len(hdr))
    for r in regions:
        for m in ("diff", "fused"):
            print(f"{r:>7} {m:>6} | "
                  f"{mean(r, m, 'r_ang'):>8.3f} {mean(r, m, 'r_mag'):>7.3f} "
                  f"{mean(r, m, 'r_vec'):>7.3f} | "
                  f"{mean(r, m, 'mse'):>11.5f} {mean(r, m, 'ang'):>10.1f}")
        # delta
        print(f"{'':>7} {'Δ':>6} | "
              f"{mean(r,'fused','r_ang')-mean(r,'diff','r_ang'):>+8.3f} "
              f"{mean(r,'fused','r_mag')-mean(r,'diff','r_mag'):>+7.3f} "
              f"{mean(r,'fused','r_vec')-mean(r,'diff','r_vec'):>+7.3f} | "
              f"{mean(r,'fused','mse')-mean(r,'diff','mse'):>+11.5f} "
              f"{mean(r,'fused','ang')-mean(r,'diff','ang'):>+10.1f}")
        print()

    print("notes: r_angle uses unit-vector components (direction only); fusion "
          "CANNOT change it -> diff==fused expected (scale-invariance check).")
    print("       r_mag/r_vec/MSE are where fusion should help; angle(deg) ~ diff.")


if __name__ == "__main__":
    main()
