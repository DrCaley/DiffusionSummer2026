"""
One-pass uncertainty-map calibration: r_angle, r_magnitude, r_overall.

Computes ALL THREE spread correlations from a SINGLE model ensemble per frame
(the expensive diffusion draws are shared), instead of running the model three
times.  Each correlation is the Pearson r between the MODEL ensemble's spread
map and the EMPIRICAL neighbour-posterior spread map over ocean cells:

  r_angle     direction-only spread   (1 - |mean unit vector|)     magnitude-free
  r_magnitude speed-only spread        (std of per-cell speed)       UNet-fixed model
  r_overall   full-vector spread       (RMS vector dispersion)       UNet-fixed model

The empirical target always uses the real neighbour-field magnitudes.  On the
model side, r_magnitude and r_overall use the Magnitude-UNet-restored speeds
(our actual fused pipeline); r_angle is magnitude-invariant so the UNet is moot.

Reuses the validated helpers + empirical-matching logic from _probe_calib_mag.

  .venv/bin/python "Conditional DDPM/testing/_probe_calib_all.py" \
      --checkpoint Models/StreamFn_Cond_x0_mag_spread.pt \
      --mag_checkpoint Models/Cond_Magnitude_UNet.pt \
      --pickle Datasets/pickles/data_divfree_chrono.pickle \
      --split 2 --n_frames 20 --seed 0 --n_model 10 --n_emp 20
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
from _probe_calib_mag import (  # noqa: E402
    pcorr, directional_spread, vector_spread, magnitude_spread,
    apply_unet_magnitude, load_magnitude_model, predict_speed_norm,
    helmholtz_project, EPS,
)


def load_hetero_magnitude_model(checkpoint, device):
    """
    Load a HeteroMagnitudeUNet (mean + log-variance heads) ->
    (net, speed_mean, speed_std, (logvar_min, logvar_max)).
    """
    import importlib.util
    mag_model_path = os.path.join(_root, "Magnitude", "model.py")
    spec = importlib.util.spec_from_file_location("mag_model_h", mag_model_path)
    mag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mag)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    base_ch = ckpt.get("args", {}).get("base_ch", 64)
    in_ch = int(ckpt["model"]["enc0.conv1.weight"].shape[1])
    sd = ckpt["model"]
    head_hidden = (int(sd["logvar_head.0.weight"].shape[0])
                   if any(k.startswith("logvar_head") for k in sd) else 0)
    net = mag.HeteroMagnitudeUNet(in_ch=in_ch, base_ch=base_ch,
                                  head_hidden=head_hidden).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval(); net.in_ch = in_ch
    clip = tuple(ckpt.get("logvar_clip", (-8.0, 4.0)))
    return net, float(ckpt["speed_mean"]), float(ckpt["speed_std"]), clip


@torch.no_grad()
def predict_speed_mean_sigma(net, speed_mean, speed_std, land_mask, data_std,
                             device, cond, logvar_clip):
    """
    Run the hetero UNet on the conditioning stack and return per-cell
    (mu_norm, sigma_norm) speed in the diffusion model's std-normalized units.
    sigma is propagated from standardized to normalized units via speed_std.
    """
    c = cond if torch.is_tensor(cond) else torch.from_numpy(np.asarray(cond))
    inp = c.unsqueeze(0).to(device).float()
    mean, logvar = net(inp)
    mu = mean[0, 0].cpu().numpy()
    lv = logvar[0, 0].clamp(*logvar_clip).cpu().numpy()
    sigma_std = np.exp(0.5 * lv)                               # standardized units
    mu_phys = np.clip(mu * speed_std + speed_mean, 0.0, None)  # physical units
    sigma_phys = sigma_std * speed_std                         # physical units
    mu_phys[land_mask] = 0.0; sigma_phys[land_mask] = 0.0
    return (mu_phys / data_std).astype(np.float32), (sigma_phys / data_std).astype(np.float32)


def hetero_magnitude(members, speed_mu, speed_sigma, ocean_np, base_seed):
    """
    Per-draw STOCHASTIC speed: each member keeps its diffusion DIRECTION but
    gets a speed sampled from N(mu(x), sigma(x)^2), clipped at 0.  Different seed
    per draw -> genuine, calibrated magnitude diversity (the point of the test).
    """
    out = []
    for k, m in enumerate(members):
        rng = np.random.default_rng(base_seed * 100003 + k)
        eps = rng.standard_normal(speed_mu.shape).astype(np.float32)
        spd = np.clip(speed_mu + speed_sigma * eps, 0.0, None)
        u, v = m[0], m[1]
        mag = np.sqrt(u ** 2 + v ** 2) + EPS
        fu = (u / mag * spd).astype(np.float32)
        fv = (v / mag * spd).astype(np.float32)
        fu[~ocean_np] = 0.0; fv[~ocean_np] = 0.0
        out.append(np.stack([fu, fv], axis=0))
    return out


def coupled_magnitude(members, speed_mu, speed_sigma, ocean_np):
    """
    DIRECTION-COUPLED, spatially-coherent magnitude calibration (no white noise).

    Instead of sampling each cell's speed from an independent N(mu,sigma^2), we
    reuse the diffusion draw's OWN magnitude anomaly as the per-draw variation --
    it is already spatially smooth and consistent with the direction we keep --
    and only rescale it so the ensemble has the calibrated per-cell mean mu(x)
    and std sigma(x) from the hetero UNet:

        m_k(x)  = ||member_k(x)||                 (native diffusion magnitude)
        mbar(x) = mean_k m_k(x);  s(x) = std_k m_k(x)
        z_k(x)  = (m_k(x) - mbar(x)) / s(x)       (coherent, zero-mean/unit-std)
        speed_k(x) = clip( mu(x) + sigma(x) * z_k(x), 0 )

    By construction the rendered ensemble has per-cell mean ~ mu and std ~ sigma
    (the calibration), but the SHAPE of the variation is the diffusion field's --
    so speed stays coupled to angle and the field stays physically coherent.
    """
    arr = np.stack(members, axis=0).astype(np.float64)        # (K, 2, H, W)
    mag = np.sqrt((arr ** 2).sum(axis=1))                     # (K, H, W)
    mbar = mag.mean(axis=0)                                   # (H, W)
    s = mag.std(axis=0)                                       # (H, W)
    z = (mag - mbar[None]) / (s[None] + EPS)                  # (K, H, W)
    out = []
    for k, m in enumerate(members):
        spd = np.clip(speed_mu + speed_sigma * z[k], 0.0, None)
        u, v = m[0], m[1]
        d = np.sqrt(u ** 2 + v ** 2) + EPS
        fu = (u / d * spd).astype(np.float32)
        fv = (v / d * spd).astype(np.float32)
        fu[~ocean_np] = 0.0; fv[~ocean_np] = 0.0
        out.append(np.stack([fu, fv], axis=0))
    return out


def reinject_magnitude(members, speed_norm, ocean_np):
    """
    UNet mean speed x per-draw RELATIVE magnitude.  Preserves the calibrated
    ensemble-mean speed (so accuracy ~= replace) while keeping each draw's own
    magnitude variation (so magnitude uncertainty is not flattened):

      sbar(x)    = mean_k ||member_k(x)||
      fused_k(x) = unit_dir(member_k) * speed_norm(x) * ||member_k(x)|| / sbar(x)
    """
    arr = np.stack(members, axis=0).astype(np.float64)
    sbar = np.sqrt((arr ** 2).sum(axis=1)).mean(axis=0)        # (H, W)
    out = []
    for m in members:
        u, v = m[0].astype(np.float64), m[1].astype(np.float64)
        spd = np.sqrt(u ** 2 + v ** 2)
        fu = (u / (spd + EPS) * speed_norm * spd / (sbar + EPS)).astype(np.float32)
        fv = (v / (spd + EPS) * speed_norm * spd / (sbar + EPS)).astype(np.float32)
        fu[~ocean_np] = 0.0; fv[~ocean_np] = 0.0
        out.append(np.stack([fu, fv], axis=0))
    return out


def vec_rmse_pct(members, true, mask):
    """Mean over draws of ALL-OCEAN vector RMSE as % of the true RMS speed."""
    tu, tv = true[0][mask], true[1][mask]
    trms = np.sqrt((tu ** 2 + tv ** 2).mean()) + EPS
    vals = []
    for m in members:
        du = m[0][mask] - tu; dv = m[1][mask] - tv
        vals.append(np.sqrt((du ** 2 + dv ** 2).mean()) / trms)
    return float(100.0 * np.mean(vals))


def render_calib_maps(maps, land_np, src_f, cov, fuse_mode, out_dir):
    """
    3x2 panel of uncertainty (spread) maps: rows = angle / magnitude / overall,
    columns = EMPIRICAL (neighbour-posterior) vs MODEL (ensemble).  Each row uses
    a SHARED colour scale across its two maps so empirical and model are directly
    comparable; the per-cell Pearson r is shown in the row label.
    """
    land_d = land_np.T
    land_cmap = mcolors.ListedColormap([(0, 0, 0, 0), "black"])
    rows = [("angle", "1 - |mean unit vec|"),
            ("magnitude", "std speed"),
            ("overall", "RMS vector disp")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), dpi=95)
    H, W = land_d.shape
    ext = [-0.5, W - 0.5, -0.5, H - 0.5]
    for i, (key, unit) in enumerate(rows):
        emp, mod, r = maps[key]
        emp_d, mod_d = emp.T, mod.T
        finite = np.isfinite(emp_d) & np.isfinite(mod_d)
        vmax = float(np.nanpercentile(np.concatenate(
            [emp_d[finite], mod_d[finite]]), 98)) if finite.any() else 1.0
        vmax = max(vmax, 1e-6)
        for j, (data, who) in enumerate(((emp_d, "Empirical"), (mod_d, "Model"))):
            ax = axes[i, j]
            im = ax.imshow(data, origin="lower", cmap="magma", vmin=0.0, vmax=vmax,
                           extent=ext, aspect="auto")
            ax.imshow(land_d, origin="lower", cmap=land_cmap, extent=ext,
                      aspect="auto", zorder=2)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{who} — r_{key}" + (f"  (r={r:+.3f})" if j == 1 else ""),
                         fontsize=11)
            ax.set_xlabel("X"); ax.set_ylabel("Y")
        axes[i, 0].set_ylabel(f"r_{key}\n{unit}\n\nY", fontsize=10)
    plt.suptitle(
        f"Uncertainty maps — empirical vs model  (frame {src_f}, cov {cov:.1f}%, "
        f"fuse={fuse_mode})\nrows: angle / magnitude / overall; shared scale per row",
        fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, f"calib_maps_frame{src_f}_{fuse_mode}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--mag_checkpoint", default="Models/Cond_Magnitude_UNet.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--n_model", type=int, default=10)
    ap.add_argument("--n_emp", type=int, default=20)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--guard", type=int, default=48)
    ap.add_argument("--min_sep", type=int, default=12)
    ap.add_argument("--mag_norm", choices=["abs", "cov"], default="abs")
    ap.add_argument("--fuse_mode",
                    choices=["replace", "reinject", "none", "hetero", "coupled"],
                    default="replace",
                    help="magnitude source for r_magnitude/r_overall + accuracy: "
                         "replace=deterministic UNet speed (no mag diversity); "
                         "reinject=UNet mean x per-draw relative magnitude; "
                         "none=raw diffusion magnitude; "
                         "hetero=sample per-draw speed from N(mu,sigma^2) of a "
                         "HeteroMagnitudeUNet (--hetero_checkpoint); "
                         "coupled=hetero mu/sigma calibration applied to the "
                         "diffusion's OWN magnitude anomaly (no white noise, "
                         "angle-coupled, spatially coherent)")
    ap.add_argument("--hetero_checkpoint",
                    default="Magnitude/checkpoints_cond_mag_hetero/best_cond_magnitude_hetero.pt",
                    help="HeteroMagnitudeUNet checkpoint (used when fuse_mode=hetero)")
    ap.add_argument("--prior_weight", type=float, default=1.0,
                    help="relative weight of the temporal-prior distance vs the "
                         "path-observation distance in empirical neighbour matching")
    ap.add_argument("--render_frame", type=int, default=-1,
                    help="frame number (value in ds.valid) or split index to render "
                         "the 6 uncertainty maps for; -1 disables rendering")
    ap.add_argument("--out_dir", default="Conditional DDPM/results/cond_calib_maps")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)
    print(f"model: {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch')} "
          f"pred={pred_type}  device={device}")
    print("hyperparameters: " + "  ".join(
        f"{k}={v}" for k, v in (
            ("split", args.split), ("n_frames", args.n_frames), ("seed", args.seed),
            ("path_steps", args.path_steps), ("n_model", args.n_model),
            ("n_emp", args.n_emp), ("inference_steps", args.inference_steps),
            ("guard", args.guard), ("min_sep", args.min_sep),
            ("mag_norm", args.mag_norm), ("prior_weight", args.prior_weight),
            ("fuse_mode", args.fuse_mode),
        )))

    ds = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=ckpt.get("data_mean", 0.0), data_std=data_std,
        path_steps=args.path_steps, deterministic=True)
    ds.cond_ch = cond_ch  # propagate to build_cond for legacy obs detection
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

    het_net = het_clip = None
    hsm = hss = None
    if args.fuse_mode in ("hetero", "coupled"):
        het_net, hsm, hss, het_clip = load_hetero_magnitude_model(
            args.hetero_checkpoint, device)
        print(f"  hetero UNet: {os.path.basename(args.hetero_checkpoint)} "
              f"speed_mean={hsm:.4f} speed_std={hss:.4f} logvar_clip={het_clip}")

    fields = ds.fields.cpu().numpy(); N = fields.shape[0]

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
        empirical = [src] + [fields[f] for f in picks]

        # ---- model ensemble (shared across all three spreads) ----
        sargs = argparse.Namespace(pred_type=pred_type,
            inference_steps=args.inference_steps, capture_every=10 ** 9,
            n_ensemble=args.n_model)
        _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                          sargs, device, base_seed=src_idx)

        # ---- UNet-restored speeds for the model side ----
        spd_phys = np.sqrt((src ** 2).sum(axis=0)) * data_std
        if args.fuse_mode == "replace":
            speed_norm = predict_speed_norm(mag_net, sm, ss, spd_phys, pm,
                                            land_np, data_std, device, cond=b["cond"])
            members_fix = apply_unet_magnitude(members, speed_norm, ocean_np)
        elif args.fuse_mode == "reinject":
            speed_norm = predict_speed_norm(mag_net, sm, ss, spd_phys, pm,
                                            land_np, data_std, device, cond=b["cond"])
            members_fix = reinject_magnitude(members, speed_norm, ocean_np)
        elif args.fuse_mode == "hetero":
            mu_n, sig_n = predict_speed_mean_sigma(
                het_net, hsm, hss, land_np, data_std, device, b["cond"], het_clip)
            members_fix = hetero_magnitude(members, mu_n, sig_n, ocean_np, src_idx)
        elif args.fuse_mode == "coupled":
            mu_n, sig_n = predict_speed_mean_sigma(
                het_net, hsm, hss, land_np, data_std, device, b["cond"], het_clip)
            members_fix = coupled_magnitude(members, mu_n, sig_n, ocean_np)
            members_fix = [helmholtz_project(d, ocean_np) for d in members_fix]
        else:  # none
            members_fix = [m.astype(np.float32) for m in members]

        # ---- accuracy: ALL-OCEAN vector RMSE% of the rendered (fused) draws ----
        rmse = vec_rmse_pct(members_fix, src, ocean_np)

        # ---- three spreads: empirical (real mag) vs model ----
        def corr(emp_s, mod_s):
            v = ocean_np & np.isfinite(emp_s) & np.isfinite(mod_s)
            return pcorr(emp_s[v], mod_s[v])

        ang_emp = directional_spread(empirical, ocean_np)
        ang_mod = directional_spread(members, ocean_np)
        mag_emp = magnitude_spread(empirical, ocean_np, args.mag_norm)
        mag_mod = magnitude_spread(members_fix, ocean_np, args.mag_norm)
        all_emp = vector_spread(empirical, ocean_np, args.mag_norm)
        all_mod = vector_spread(members_fix, ocean_np, args.mag_norm)
        r_ang = corr(ang_emp, ang_mod)
        r_mag = corr(mag_emp, mag_mod)
        r_all = corr(all_emp, all_mod)
        maps = {
            "angle":     (ang_emp, ang_mod, r_ang),
            "magnitude": (mag_emp, mag_mod, r_mag),
            "overall":   (all_emp, all_mod, r_all),
        }
        return src_f, cov, r_ang, r_mag, r_all, rmse, maps

    # ---- single-frame render of the six uncertainty maps ----
    if args.render_frame >= 0:
        hits = np.where(np.asarray(ds.valid) == args.render_frame)[0]
        rix = int(hits[0]) if len(hits) else int(args.render_frame)
        sf, cov, ra, rm, ro, rmse, maps = eval_frame(rix)
        os.makedirs(args.out_dir, exist_ok=True)
        out = render_calib_maps(maps, land_np, sf, cov, args.fuse_mode, args.out_dir)
        print(f"frame {sf}  cov {cov:.1f}%  r_angle {ra:+.3f}  r_magnitude {rm:+.3f}  "
              f"r_overall {ro:+.3f}  rmse {rmse:.1f}%")
        print(f"saved: {out}")
        return

    rng0 = np.random.default_rng(args.seed)
    n_valid = len(ds.valid)
    k = min(args.n_frames, n_valid)
    idxs = sorted(int(x) for x in rng0.choice(n_valid, size=k, replace=False))
    print(f"random sweep: {k} frames, seed={args.seed}, n_valid={n_valid}\n")

    print(f"{'frame':>6} {'%kn':>5} {'r_angle':>8} {'r_magn':>8} {'r_overall':>10} {'rmse%':>7}")
    rows = []
    for ix in idxs:
        sf, cov, ra, rm, ro, rmse, _ = eval_frame(ix)
        rows.append((ra, rm, ro, rmse))
        print(f"{sf:>6} {cov:>4.1f}% {ra:>+8.3f} {rm:>+8.3f} {ro:>+10.3f} {rmse:>6.1f}%")

    arr = np.array(rows, dtype=np.float64)
    m = arr.mean(axis=0); sd = arr.std(axis=0)
    print(f"\n  N={len(rows)} frames  MEAN  (fuse_mode={args.fuse_mode})")
    print(f"  r_angle    = {m[0]:+.3f} ± {sd[0]:.3f}")
    print(f"  r_magnitude= {m[1]:+.3f} ± {sd[1]:.3f}")
    print(f"  r_overall  = {m[2]:+.3f} ± {sd[2]:.3f}")
    print(f"  ALL-OCEAN rmse = {m[3]:.1f}% ± {sd[3]:.1f}   (accuracy; lower=better)")


if __name__ == "__main__":
    main()
