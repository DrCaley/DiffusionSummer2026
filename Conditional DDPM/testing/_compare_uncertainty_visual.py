"""Side-by-side uncertainty comparison: EMPIRICAL vs BASELINE vs FINE-TUNED.

Renders, for a small set of structured TEST frames, the empirical (data-manifold)
directional-uncertainty heatmap next to the BASELINE model's uncertainty and the
spread-loss FINE-TUNED model's uncertainty, so the calibration improvement is
visible directly.  r_dir (spatial Pearson of model-vs-empirical spread) annotated
on each model panel.

Frame selection mirrors _probe_calib_diag's random sweep (same --seed / --n_frames
draw over the split), then keeps the top --render frames by empirical concentration
(e_conc = spatial CoV of the TRUE uncertainty map) so the figures show frames where
a real "where to measure" signal exists.  Empirical matching is byte-for-byte the
probe's (path + prior distance, guard / min_sep), so these pictures correspond to
the numbers the probe reports.
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


def heatmap(ax, m, land_d, title, vmax):
    sp = m.T.copy()
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


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ca = ckpt.get("args", {}); cond_ch = ckpt.get("cond_ch", 10)
    model = IC.StreamFunctionUNet(in_ch=2, base_ch=ca.get("base_ch", 64),
        time_dim=ca.get("time_dim", 256), cond_ch=cond_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = IC.DDPM(T=ca.get("T", 1000), beta_schedule=ca.get("schedule", "cosine"),
        device=device, noise_type=ca.get("noise_type", "div_free"),
        spectral_filter=ckpt.get("spectral_filter", None))
    return model, diffusion, ckpt


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="Models/StreamFn_Cond_x0_mag.pt")
    ap.add_argument("--finetuned", default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=2, help="2 = TEST")
    ap.add_argument("--n_frames", type=int, default=20,
                    help="size of the random draw to pick structured frames from")
    ap.add_argument("--render", type=int, default=3, help="how many figures to save")
    ap.add_argument("--select", choices=["structured", "random"], default="structured",
                    help="structured = top --render frames by empirical concentration "
                         "(clearest signal, but cherry-picked); random = first --render "
                         "frames of the unbiased draw (representative, matches the "
                         "honest numbers).")
    ap.add_argument("--render_seed", type=int, default=0,
                    help="seed for choosing WHICH frames to render when --select random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--n_model", type=int, default=40)
    ap.add_argument("--n_emp", type=int, default=80)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--guard", type=int, default=48)
    ap.add_argument("--min_sep", type=int, default=12)
    ap.add_argument("--out_dir", default="Conditional DDPM/results/uncertainty_compare")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {device}")

    base_model, base_diff, base_ck = load_model(args.baseline, device)
    ft_model, ft_diff, ft_ck = load_model(args.finetuned, device)
    base_pred = base_ck.get("pred_type"); ft_pred = ft_ck.get("pred_type")
    lags = tuple(base_ck.get("lags", (13, 25)))
    data_std = base_ck.get("data_std"); phys = float(data_std) if data_std else 1.0
    print(f"baseline : {os.path.basename(args.baseline)} ep={base_ck.get('epoch')}")
    print(f"finetuned: {os.path.basename(args.finetuned)} ep={ft_ck.get('epoch')}")

    ds = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=base_ck.get("data_mean", 0.0), data_std=data_std,
        path_steps=args.path_steps, deterministic=True)
    land_np = ds.land_mask.cpu().numpy().astype(bool); ocean_np = ~land_np
    n_ocean = max(int(ocean_np.sum()), 1)
    fields = ds.fields.cpu().numpy(); N = fields.shape[0]

    def empirical_set(src_idx):
        """Return (src, path_mask_ocean, src_f, cov, empirical_fields) — probe-exact."""
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
        return b, src, pm, pm_ocean, src_f, cov, empirical

    # ---- frame selection: same draw as the probe, rank by empirical concentration ----
    rng0 = np.random.default_rng(args.seed)
    n_valid = len(ds.valid)
    k = min(args.n_frames, n_valid)
    idxs = sorted(int(x) for x in rng0.choice(n_valid, size=k, replace=False))
    print(f"\nscoring {k} frames (seed={args.seed}) by empirical concentration ...")
    scored = []
    for ix in idxs:
        b, src, pm, pm_ocean, src_f, cov, empirical = empirical_set(ix)
        emp_dir = IC.directional_spread(empirical, ocean_np)
        valid = ocean_np & np.isfinite(emp_dir)
        ev = emp_dir[valid]
        e_conc = float(ev.std() / (abs(ev.mean()) + 1e-9))
        scored.append((e_conc, ix, src_f))
        print(f"  idx {ix:>4}  frame {src_f:>5}  e_conc={e_conc:.3f}")
    if args.select == "structured":
        scored.sort(reverse=True)
        chosen = scored[:args.render]
        print("\n[structured] rendering top-e_conc frames: "
              + ", ".join(f"{s[2]}(e_conc={s[0]:.2f})" for s in chosen))
    else:
        rngr = np.random.default_rng(args.render_seed)
        pick = sorted(rngr.choice(len(scored), size=min(args.render, len(scored)),
                                  replace=False))
        chosen = [scored[i] for i in pick]
        print("\n[random] rendering random frames: "
              + ", ".join(f"{s[2]}(e_conc={s[0]:.2f})" for s in chosen))

    for e_conc, ix, _f in chosen:
        b, src, pm, pm_ocean, src_f, cov, empirical = empirical_set(ix)
        sargs_base = argparse.Namespace(pred_type=base_pred,
            inference_steps=args.inference_steps, capture_every=10 ** 9,
            n_ensemble=args.n_model)
        sargs_ft = argparse.Namespace(pred_type=ft_pred,
            inference_steps=args.inference_steps, capture_every=10 ** 9,
            n_ensemble=args.n_model)
        _, _, base_members = IC.ensemble_infer(base_model, base_diff, b["cond"],
                                               land_np, sargs_base, device, base_seed=ix)
        _, _, ft_members = IC.ensemble_infer(ft_model, ft_diff, b["cond"],
                                             land_np, sargs_ft, device, base_seed=ix)

        emp_dir = IC.directional_spread(empirical, ocean_np)
        base_dir = IC.directional_spread(base_members, ocean_np)
        ft_dir = IC.directional_spread(ft_members, ocean_np)
        valid = (ocean_np & np.isfinite(emp_dir) & np.isfinite(base_dir)
                 & np.isfinite(ft_dir))
        r_base = pcorr(emp_dir[valid], base_dir[valid])
        r_ft = pcorr(emp_dir[valid], ft_dir[valid])

        land_d = land_np.T
        vmax = max(np.nanpercentile(emp_dir[valid], 99),
                   np.nanpercentile(base_dir[valid], 99),
                   np.nanpercentile(ft_dir[valid], 99))
        fig, ax = plt.subplots(1, 4, figsize=(26, 6), dpi=95)
        IC.plot_field(ax[0], (src[0] * phys).T, (src[1] * phys).T, land_d,
                      f"Ground truth + known path\nframe {src_f}, {cov:.1f}% known")
        IC.plot_path(ax[0], pm.T, land_d, "")
        heatmap(ax[1], emp_dir, land_d,
                f"EMPIRICAL uncertainty\n(truth, {args.n_emp} data fields)", vmax)
        heatmap(ax[2], base_dir, land_d,
                f"BASELINE model\nr_dir = {r_base:+.3f}", vmax)
        heatmap(ax[3], ft_dir, land_d,
                f"FINE-TUNED (spread loss)\nr_dir = {r_ft:+.3f}", vmax)
        plt.suptitle(
            f"Uncertainty calibration on TEST frame {src_f}  "
            f"(e_conc={e_conc:.2f}, {args.select})   —   baseline r_dir={r_base:+.2f}  →  "
            f"fine-tuned r_dir={r_ft:+.2f}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(args.out_dir, f"compare_{args.select}_frame{src_f}.png")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"saved -> {out}   base r_dir={r_base:+.3f}  ft r_dir={r_ft:+.3f}")


if __name__ == "__main__":
    main()
