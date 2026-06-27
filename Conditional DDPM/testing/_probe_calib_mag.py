"""
Magnitude-aware calibration probe (temporary).

Extends _probe_calib_diag.py with an OPTIONAL magnitude component in the
uncertainty (spread) map, while keeping the direction-only r_dir behaviour
byte-for-byte when magnitude is off.

Spread modes (--spread_mode):
  dir   directional spread  1 - |mean unit vector|        (== current r_dir)
  vec   full-vector RMS dispersion across the ensemble     (includes magnitude)
            disp(x) = sqrt( mean_k || v_k(x) - mean_v(x) ||^2 )
            --mag_norm abs : raw physical dispersion (upweights fast flow)
            --mag_norm cov : disp / mean_k||v_k||  (coefficient of variation)

Magnitude source for the MODEL ensemble in vec mode (--mag_source):
  model  use each diffusion member's own (collapsed) magnitudes
  unet   replace every member's per-cell speed with the Magnitude-UNet
         prediction (direction preserved) -> "fix the magnitudes first, then
         correlate".  Requires --mag_checkpoint.  Because the UNet speed is the
         same for all members, this becomes a SPEED-WEIGHTED directional spread.

The EMPIRICAL target always uses the real neighbour-field magnitudes (the honest
ground-truth dispersion), so vec mode asks: does our (optionally magnitude-fixed)
model dispersion correlate with the true total-velocity dispersion?

All reliability diagnostics (r_mm, r_ee, r_partial, emp_conc) use the SELECTED
spread function so the numbers stay self-consistent.

Run locally (MPS), does not touch the servers:
  .venv/bin/python "Conditional DDPM/testing/_probe_calib_mag.py" \
      --checkpoint Models/StreamFn_Cond_x0_mag.pt \
      --pickle Datasets/pickles/data_divfree_chrono.pickle \
      --split 2 --n_frames 16 --seed 0 --n_model 40 --n_emp 80 \
      --spread_mode vec --mag_source unet \
      --mag_checkpoint Models/Magnitude_UNet_New.pt
"""
import argparse
import importlib.util
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

EPS = 1e-8


def pcorr(a, b, eps=1e-12):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def partial_pcorr(a, b, z, eps=1e-12):
    """Pearson(a, b) with covariate z linearly regressed out of both."""
    def resid(y):
        Z = np.stack([np.ones_like(z), z], axis=1)
        coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return y - Z @ coef
    return pcorr(resid(a), resid(b), eps)


# ---------------------------------------------------------------------------
# Spread functions
# ---------------------------------------------------------------------------

def directional_spread(members, ocean_np):
    """1 - |mean unit vector|.  Magnitude-invariant.  NaN at land."""
    return IC.directional_spread(members, ocean_np)


def vector_spread(members, ocean_np, mag_norm="abs"):
    """
    Full-vector RMS dispersion across the ensemble (magnitude-aware).
      disp(x) = sqrt( mean_k || v_k(x) - mean_v(x) ||^2 )
    mag_norm='cov' divides by mean_k||v_k|| (coefficient of variation).
    NaN at land.
    """
    arr = np.stack(members, axis=0).astype(np.float64)        # (K, 2, H, W)
    mean = arr.mean(axis=0)                                    # (2, H, W)
    dev = arr - mean[None]                                     # (K, 2, H, W)
    disp = np.sqrt((dev ** 2).sum(axis=1).mean(axis=0))       # (H, W)
    if mag_norm == "cov":
        mean_mag = np.sqrt((arr ** 2).sum(axis=1)).mean(axis=0)
        disp = disp / (mean_mag + EPS)
    out = disp.astype(np.float32)
    out[~ocean_np] = np.nan
    return out


def magnitude_spread(members, ocean_np, mag_norm="abs"):
    """
    SPEED-ONLY dispersion across the ensemble (direction discarded).
      speed_k(x) = ||v_k(x)||;  disp(x) = std_k speed_k(x)
    Isolates how uncertain the SPEED is, independent of directional ambiguity.
    mag_norm='cov' divides by mean_k speed_k (coefficient of variation).
    NaN at land.
    """
    arr = np.stack(members, axis=0).astype(np.float64)        # (K, 2, H, W)
    speed = np.sqrt((arr ** 2).sum(axis=1))                    # (K, H, W)
    disp = speed.std(axis=0)                                   # (H, W)
    if mag_norm == "cov":
        disp = disp / (speed.mean(axis=0) + EPS)
    out = disp.astype(np.float32)
    out[~ocean_np] = np.nan
    return out


def apply_unet_magnitude(members, speed_norm, ocean_np):
    """
    Replace every member's per-cell speed with `speed_norm` (H, W, normalized
    units) while keeping its direction.  Land cells -> 0.
    """
    fixed = []
    for m in members:
        u, v = m[0], m[1]
        mag = np.sqrt(u ** 2 + v ** 2) + EPS
        uh, vh = u / mag, v / mag
        fu = (uh * speed_norm).astype(np.float32)
        fv = (vh * speed_norm).astype(np.float32)
        fu[~ocean_np] = 0.0; fv[~ocean_np] = 0.0
        fixed.append(np.stack([fu, fv], axis=0))
    return fixed


# ---------------------------------------------------------------------------
# Magnitude UNet
# ---------------------------------------------------------------------------

def load_magnitude_model(checkpoint, device):
    """
    Load a MagnitudeUNet speed regressor -> (net, speed_mean, speed_std).

    The input-channel count is read from the checkpoint weights, so this loads
    both the original 3-channel UNet ([obs_speed, path_mask, land]) and the
    conditioned UNet (the full ``cond_channels(lags)`` stack).  ``net.in_ch`` is
    set so ``predict_speed_norm`` knows which input to build.
    """
    mag_model_path = os.path.join(_root, "Magnitude", "model.py")
    spec = importlib.util.spec_from_file_location("mag_model", mag_model_path)
    mag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mag)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    base_ch = ckpt.get("args", {}).get("base_ch", 64)
    in_ch = int(ckpt["model"]["enc0.conv1.weight"].shape[1])  # weights are truth
    net = mag.MagnitudeUNet(in_ch=in_ch, base_ch=base_ch).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    net.in_ch = in_ch
    return net, float(ckpt["speed_mean"]), float(ckpt["speed_std"])


@torch.no_grad()
def predict_speed_norm(mag_net, speed_mean, speed_std, spd_phys,
                       path_mask, land_mask, data_std, device, cond=None):
    """
    Predict the dense speed field and return it in the conditional model's
    std-normalized units (so it is consistent with ds.fields).

    Two input conventions, selected by ``mag_net.in_ch``:
      * 3-channel UNet  -> built here from [obs_speed, path_mask, land].
      * conditioned UNet -> uses ``cond`` (the SAME (C, H, W) stack the diffusion
        model consumes, with the robot path already baked in); ``cond`` required.

    spd_phys : (H, W) TRUE speed in original physical units (norm * data_std),
               used only for the 3-channel path observations.
    """
    in_ch = int(getattr(mag_net, "in_ch", 3))
    if in_ch <= 3:
        obs = np.zeros_like(spd_phys)
        obs[path_mask] = spd_phys[path_mask] / speed_std
        inp = np.stack([obs,
                        path_mask.astype(np.float32),
                        land_mask.astype(np.float32)], axis=0)[None]
        inp = torch.from_numpy(inp).to(device)
    else:
        if cond is None:
            raise ValueError(
                f"conditioned magnitude UNet (in_ch={in_ch}) requires `cond`")
        c = cond if torch.is_tensor(cond) else torch.from_numpy(np.asarray(cond))
        if c.shape[0] != in_ch:
            raise ValueError(
                f"cond has {c.shape[0]} channels but UNet expects {in_ch}")
        inp = c.unsqueeze(0).to(device).float()
    pred = mag_net(inp)[0, 0].cpu().numpy()
    pred = np.clip(pred * speed_std + speed_mean, 0.0, None)   # physical units
    pred[land_mask] = 0.0
    return (pred / data_std).astype(np.float32)               # -> normalized units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--frames", default="1500,3000,11865")
    ap.add_argument("--n_frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conc_gate", type=float, default=0.0)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--n_model", type=int, default=40)
    ap.add_argument("--n_emp", type=int, default=80)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--guard", type=int, default=48)
    ap.add_argument("--min_sep", type=int, default=12)
    # magnitude knobs
    ap.add_argument("--spread_mode", choices=["dir", "vec", "mag"], default="dir",
                    help="dir = direction-only (current r_dir); vec = magnitude-aware "
                         "full vector; mag = speed-only (direction discarded)")
    ap.add_argument("--mag_norm", choices=["abs", "cov"], default="abs",
                    help="vec mode: raw dispersion (abs) or coefficient of variation (cov)")
    ap.add_argument("--mag_source", choices=["model", "unet"], default="model",
                    help="vec mode: members' own magnitudes, or UNet-fixed magnitudes")
    ap.add_argument("--mag_checkpoint", default="Models/Magnitude_UNet_New.pt")
    args = ap.parse_args()

    use_unet = (args.spread_mode == "vec" and args.mag_source == "unet")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)
    print(f"model: {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch')} "
          f"pred={pred_type}  n_model={args.n_model}  n_emp={args.n_emp}")
    print(f"spread_mode={args.spread_mode}"
          + (f"  mag_norm={args.mag_norm}  mag_source={args.mag_source}"
             if args.spread_mode == "vec" else "")
          + (f"  mag_ckpt={os.path.basename(args.mag_checkpoint)}" if use_unet else ""))

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

    mag_net = sm = ss = None
    if use_unet:
        mag_net, sm, ss = load_magnitude_model(args.mag_checkpoint, device)
        print(f"  magnitude UNet loaded: speed_mean={sm:.4f} speed_std={ss:.4f}")

    fields = ds.fields.cpu().numpy(); N = fields.shape[0]

    def spread_fn(members, speed_norm=None):
        """Selected spread map.  speed_norm (H,W) applies UNet magnitudes (model side)."""
        if args.spread_mode == "dir":
            return directional_spread(members, ocean_np)
        mem = members
        if speed_norm is not None:
            mem = apply_unet_magnitude(members, speed_norm, ocean_np)
        if args.spread_mode == "mag":
            return magnitude_spread(mem, ocean_np, mag_norm=args.mag_norm)
        return vector_spread(mem, ocean_np, mag_norm=args.mag_norm)

    @torch.no_grad()
    def eval_frame(src_idx):
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
        src = b["target"].cpu().numpy()
        pm = b["path_mask"]
        pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
        pm_ocean = pm & ocean_np
        src_f = int(ds.valid[src_idx])
        cov = 100.0 * pm_ocean.sum() / ocean_np.sum()

        # ---- empirical matching (path + priors), identical to validated tool ----
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

        # ---- UNet speed (model side only), in normalized units ----
        speed_norm = None
        if use_unet:
            spd_phys = np.sqrt((src ** 2).sum(axis=0)) * data_std   # true speed, phys
            speed_norm = predict_speed_norm(mag_net, sm, ss, spd_phys, pm,
                                            land_np, data_std, device,
                                            cond=b["cond"])

        # ---- spreads + headline metrics ----
        emp_s = spread_fn(empirical)                 # empirical: real magnitudes
        mod_s = spread_fn(members, speed_norm)        # model: optional UNet fix
        valid = ocean_np & np.isfinite(emp_s) & np.isfinite(mod_s)
        ev, mv = emp_s[valid], mod_s[valid]
        r_dir = pcorr(ev, mv)
        rho_dir = float(spearmanr(ev, mv).correlation)
        emp_conc = float(ev.std() / (abs(ev.mean()) + 1e-9))

        dpath = ndimage.distance_transform_edt(~pm).astype(np.float32)
        r_partial = partial_pcorr(ev, mv, dpath[valid])

        # ---- split-half reliability (selected spread) ----
        rng = np.random.default_rng(src_idx)
        mi = rng.permutation(len(members))
        ma = [members[k] for k in mi[:len(members) // 2]]
        mb = [members[k] for k in mi[len(members) // 2:]]
        r_mm = pcorr(spread_fn(ma, speed_norm)[valid], spread_fn(mb, speed_norm)[valid])
        ei = rng.permutation(len(empirical))
        ea = [empirical[k] for k in ei[:len(empirical) // 2]]
        eb = [empirical[k] for k in ei[len(empirical) // 2:]]
        r_ee = pcorr(spread_fn(ea)[valid], spread_fn(eb)[valid])

        denom = np.sqrt(max(r_mm, 1e-6) * max(r_ee, 1e-6))
        r_corr = r_dir / denom if denom > 0 else float("nan")
        return src_f, cov, r_dir, rho_dir, r_partial, r_mm, r_ee, r_corr, emp_conc

    if args.n_frames > 0:
        rng0 = np.random.default_rng(args.seed)
        n_valid = len(ds.valid)
        k = min(args.n_frames, n_valid)
        idxs = sorted(int(x) for x in rng0.choice(n_valid, size=k, replace=False))
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
    w = np.clip(conc_col, 0.0, None)
    wmean = float((w * rd_col).sum() / (w.sum() + 1e-12))
    gate = args.conc_gate if args.conc_gate > 0 else float(np.median(conc_col))
    smask = conc_col >= gate
    smean = float(rd_col[smask].mean()) if smask.any() else float("nan")
    sstd = float(rd_col[smask].std()) if smask.any() else float("nan")
    tag = ("direction-only r_dir" if args.spread_mode == "dir"
           else f"speed-only r_mag ({args.mag_norm}, mag={args.mag_source})"
           if args.spread_mode == "mag"
           else f"vector r_dir ({args.mag_norm}, mag={args.mag_source})")
    print(f"\n  N={len(rows)} frames   [{tag}]")
    print(f"  unweighted   : r_dir={m[0]:+.3f} ± {sd[0]:.3f}  rho={m[1]:+.3f}  "
          f"r_partial={m[2]:+.3f}")
    print(f"  reliability  : r_mm={m[3]:.3f}  r_ee={m[4]:.3f}  r_corr={m[5]:+.3f}")
    print(f"  emp_conc-wtd : r_dir={wmean:+.3f}")
    print(f"  structured   : r_dir={smean:+.3f} ± {sstd:.3f}  "
          f"(emp_conc>={gate:.2f}, n={int(smask.sum())}/{len(rows)})")


if __name__ == "__main__":
    main()
