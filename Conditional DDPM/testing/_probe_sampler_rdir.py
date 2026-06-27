"""
Sampler r_dir comparison (temporary).

Question it answers: does harder observation-enforcement at sampling time
(DPS gradient guidance / SMC particle filter) sharpen the model's directional-
spread PATTERN enough to raise r_dir, WITHOUT any retraining?

For each frame we build the empirical posterior's directional-spread map ONCE
(path + temporal-prior neighbour matching, identical to _probe_calib_diag.py /
uncertainty_validation.py), then for EACH sampler we draw an ensemble from the
SAME conditioning, compute the model directional-spread map, and correlate it
with that single empirical target.  Because the target is shared across methods,
the only thing that changes between columns is the sampler -> an apples-to-apples
r_dir comparison.

r_dir here is the EXACT metric the project reports: spatial Pearson of the
per-cell directional spread (1 - |mean unit vector|) over ocean cells.

Reports, per sampler: mean r_dir (all frames) and the structured subset
(emp_conc >= median), where a real spatial uncertainty signal exists.
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", ".."))
for _p in (_here, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), _root):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import infer_cond as IC          # noqa: E402
from samplers import SAMPLERS    # noqa: E402


def pcorr(a, b, eps=1e-12):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--methods", default="vanilla,dps,particle")
    ap.add_argument("--n_model", type=int, default=40,
                    help="ensemble members drawn per sampler per frame")
    ap.add_argument("--n_emp", type=int, default=80,
                    help="empirical neighbours (ceiling-probe sweet spot)")
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--guard", type=int, default=48)
    ap.add_argument("--min_sep", type=int, default=12)
    ap.add_argument("--obs_sigma", type=float, default=0.1)
    ap.add_argument("--dps_zeta", type=float, default=0.05)
    ap.add_argument("--ess_frac", type=float, default=0.5)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in SAMPLERS:
            raise SystemExit(f"unknown method {m!r}; choose from {list(SAMPLERS)}")
    kw = {
        "vanilla":  dict(),
        "particle": dict(obs_sigma=args.obs_sigma, ess_frac=args.ess_frac),
        "dps":      dict(obs_sigma=args.obs_sigma, zeta=args.dps_zeta),
    }

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = ckpt.get("data_std")
    print(f"device={device}  model={os.path.basename(args.checkpoint)} "
          f"ep={ckpt.get('epoch')} pred={pred_type}")
    print(f"methods={methods}  n_model={args.n_model}  n_emp={args.n_emp}  "
          f"steps={args.inference_steps}")

    ds = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=ckpt.get("data_mean", 0.0), data_std=data_std,
        path_steps=args.path_steps, deterministic=True)
    land_np = ds.land_mask.cpu().numpy().astype(bool); ocean_np = ~land_np
    n_ocean = max(int(ocean_np.sum()), 1)

    model = IC.StreamFunctionUNet(in_ch=2, base_ch=ca.get("base_ch", 64),
        time_dim=ca.get("time_dim", 256), cond_ch=cond_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = IC.DDPM(T=ca.get("T", 1000),
        beta_schedule=ca.get("schedule", "cosine"), device=device,
        noise_type=ca.get("noise_type", "div_free"),
        spectral_filter=ckpt.get("spectral_filter", None))

    fields = ds.fields.cpu().numpy(); N = fields.shape[0]

    def spread_dir(members):
        return IC.directional_spread(members, ocean_np)

    def empirical_spread(src_idx):
        """Directional-spread map of the empirical posterior + emp_conc + valid mask."""
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
        src = b["target"].cpu().numpy()
        pm = b["path_mask"]
        pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
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
            if len(picks) == args.n_emp - 1:
                break
        empirical = [src] + [fields[f] for f in picks]
        emp_dir = spread_dir(empirical)
        return b, src_f, cov, emp_dir

    rng = np.random.default_rng(args.seed)
    idxs = sorted(int(x) for x in rng.choice(
        len(ds.valid), size=min(args.n_frames, len(ds.valid)), replace=False))

    # results[method] -> list of (r_dir, rho, emp_conc) per frame
    results = {m: [] for m in methods}
    concs = []

    hdr = f"{'frame':>6} {'%kn':>5} {'e_conc':>7}" + "".join(
        f" {m[:7]:>8}" for m in methods)
    print("\n" + hdr)
    for src_idx in idxs:
        b, src_f, cov, emp_dir = empirical_spread(src_idx)
        for m in methods:
            members, _ = SAMPLERS[m](
                model, diffusion, b["cond"], land_np,
                n_members=args.n_model, inference_steps=args.inference_steps,
                device=device, seed=src_idx, pred_type=pred_type, **kw[m])
            mod_dir = spread_dir(members)
            valid = ocean_np & np.isfinite(emp_dir) & np.isfinite(mod_dir)
            ev, mv = emp_dir[valid], mod_dir[valid]
            r_dir = pcorr(ev, mv)
            rho = float(spearmanr(ev, mv).correlation)
            results[m].append((r_dir, rho))
        # emp_conc is model-independent (computed from the shared empirical map)
        vmask = ocean_np & np.isfinite(emp_dir)
        ev = emp_dir[vmask]
        emp_conc = float(ev.std() / (abs(ev.mean()) + 1e-9))
        concs.append(emp_conc)
        row = f"{src_f:>6} {cov:>4.1f}% {emp_conc:>7.3f}" + "".join(
            f" {results[m][-1][0]:>+8.3f}" for m in methods)
        print(row)

    concs = np.array(concs)
    gate = float(np.median(concs))
    smask = concs >= gate
    print(f"\n  N={len(idxs)} frames   structured gate emp_conc>={gate:.2f} "
          f"(n={int(smask.sum())})")
    print(f"\n  {'method':10s} {'r_dir(all)':>12} {'rho(all)':>10} "
          f"{'r_dir(struct)':>14}")
    print("  " + "-" * 48)
    for m in methods:
        arr = np.array(results[m])             # (N, 2)
        rd, rho = arr[:, 0], arr[:, 1]
        sd = rd[smask]
        print(f"  {m:10s} {rd.mean():>+8.3f}±{rd.std():.3f} {rho.mean():>+10.3f} "
              f"{sd.mean():>+8.3f}±{sd.std():.3f}")
    print("\ninterpretation:")
    print("  r_dir up vs vanilla -> harder obs-enforcement sharpens the spread")
    print("     pattern; a free quick win, no retraining needed.")
    print("  r_dir flat/down     -> obs-enforcement does not help the PATTERN;")
    print("     the spread ceiling is set by training, not sampling -> Lever 1.")


if __name__ == "__main__":
    main()
