"""
DIRECTIONAL-VARIANCE VISUALIZER — make the surviving (angular) ensemble variance
VISIBLE.

In the fused pipeline every draw shares one UNet speed field, so a normal quiver
multidraw looks near-identical: the only thing that changes between draws is the
per-cell flow ANGLE, and that wobble is hidden under the dominant (shared) speed.

This script strips speed away and plots the angle directly.  For one frame it
draws a grid:

    row 0:  [ ground truth | ensemble-mean direction | directional spread 1-R ]
    rows 1+: per-draw SIGNED angular deviation from the ensemble-mean direction
             (diverging colormap, degrees), unit-vector quiver overlaid.

Where the path/priors pin the flow the deviation map is ~0 (white); where the
field is genuinely uncertain the draws fan out into strong +/- colour -- that
colour IS the variance the fusion keeps and r_dir measures.

Run:
  .venv/bin/python "Conditional DDPM/testing/_probe_dirvar.py" \
      --checkpoint Models/StreamFn_Cond_x0_mag_spread.pt \
      --mag_checkpoint Models/Cond_Magnitude_UNet.pt \
      --pickle Datasets/pickles/data_divfree_chrono.pickle \
      --split 2 --frame 4476 --n_draws 6 --path_steps 90 \
      --out_dir "Conditional DDPM/results/cond_dirvar"
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


def mean_direction(members, ocean_np, eps=1e-8):
    """Per-cell ensemble-mean unit direction -> (uh, vh) and resultant R."""
    us, vs = [], []
    for m in members:
        uh, vh, _ = IC.unit_normalize(m, ocean_np, eps)
        us.append(uh); vs.append(vh)
    mu = np.mean(us, axis=0); mv = np.mean(vs, axis=0)
    R = np.sqrt(mu ** 2 + mv ** 2)
    safe = R > eps
    du = np.zeros_like(mu); dv = np.zeros_like(mv)
    du[safe] = mu[safe] / R[safe]; dv[safe] = mv[safe] / R[safe]
    return du, dv, R


def signed_angle_dev(member, mean_u, mean_v, ocean_np, eps=1e-8):
    """Signed angle (deg, [-180,180]) of each cell's direction relative to the
    ensemble-mean direction; +ve = counter-clockwise of the mean."""
    u, v, _ = IC.unit_normalize(member, ocean_np, eps)
    # cross gives sin (sign), dot gives cos -> atan2 = signed angle between
    cross = mean_u * v - mean_v * u
    dot = mean_u * u + mean_v * v
    ang = np.degrees(np.arctan2(cross, dot)).astype(np.float32)
    ang[~ocean_np] = np.nan
    return ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--mag_checkpoint", default="Models/Cond_Magnitude_UNet.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--frame", type=int, default=-1,
                    help="FRAME number (value in ds.valid); -1 picks random via --seed")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n_draws", type=int, default=6)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--amax", type=float, default=90.0,
                    help="deviation colour scale +/- degrees")
    ap.add_argument("--step", type=int, default=2, help="quiver subsample")
    ap.add_argument("--out_dir", default="Conditional DDPM/results/cond_dirvar")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)
    print(f"model: {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch')} "
          f"n_draws={args.n_draws} device={device}")

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

    if args.frame >= 0:
        hits = np.where(np.asarray(ds.valid) == args.frame)[0]
        src_idx = int(hits[0]) if len(hits) else int(args.frame)
    else:
        rng = np.random.default_rng(args.seed)
        src_idx = int(rng.integers(0, len(ds.valid)))
    src_f = int(ds.valid[src_idx])

    b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
    src = b["target"].cpu().numpy()
    pm = b["path_mask"]
    pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
    pm_ocean = pm & ocean_np
    cov = 100.0 * pm_ocean.sum() / ocean_np.sum()

    sargs = argparse.Namespace(pred_type=pred_type,
        inference_steps=args.inference_steps, capture_every=10 ** 9,
        n_ensemble=args.n_draws)
    _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                      sargs, device, base_seed=src_idx)

    mean_u, mean_v, R = mean_direction(members, ocean_np)
    spread = (1.0 - R); spread[~ocean_np] = np.nan
    devs = [signed_angle_dev(m, mean_u, mean_v, ocean_np) for m in members]
    rms_dev = float(np.sqrt(np.nanmean(np.stack(devs) ** 2)))

    # ---- render ----
    land_d = land_np.T
    H, W = land_d.shape
    extent = [-0.5, W - 0.5, -0.5, H - 0.5]
    land_cmap = mcolors.ListedColormap([(0, 0, 0, 0), "black"])

    n = args.n_draws
    ncol = max(3, int(np.ceil(n / 2)))
    fig, axes = plt.subplots(3, ncol, figsize=(6.2 * ncol, 16), dpi=90)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")

    s = data_std
    tspd = np.sqrt((src[0] * s) ** 2 + (src[1] * s) ** 2).T
    vmax = float(np.nanpercentile(tspd[~land_d], 98)) if (~land_d).any() else 1.0

    # row 0, col 0: ground truth field
    ax[0].axis("on")
    IC.plot_field(ax[0], src[0].T * s, src[1].T * s, land_d,
                  "Ground truth", vmax=vmax)

    # row 0, col 1: ensemble-mean DIRECTION (unit quiver)
    ax[1].axis("on")
    muT = mean_u.T; mvT = mean_v.T
    ax[1].imshow(land_d, origin="lower", cmap=mcolors.ListedColormap(["white", "0.8"]),
                 extent=extent, aspect="auto", zorder=0)
    yy, xx = np.mgrid[0:H:args.step, 0:W:args.step]
    ax[1].quiver(xx, yy, muT[::args.step, ::args.step], mvT[::args.step, ::args.step],
                 scale=30, width=0.003, color="tab:blue")
    ax[1].imshow(land_d, origin="lower", cmap=land_cmap, extent=extent,
                 aspect="auto", zorder=2)
    ax[1].set_title("Ensemble-mean direction", fontsize=11)
    ax[1].set_xlim(-0.5, W - 0.5); ax[1].set_ylim(-0.5, H - 0.5)
    ax[1].set_xlabel("X"); ax[1].set_ylabel("Y")

    # row 0, col 2: directional spread 1-R
    ax[2].axis("on")
    sp = spread.T.copy()
    im = ax[2].imshow(sp, origin="lower", cmap="magma", vmin=0.0, vmax=1.0,
                      extent=extent, aspect="auto")
    ax[2].imshow(land_d, origin="lower", cmap=land_cmap, extent=extent,
                 aspect="auto", zorder=2)
    plt.colorbar(im, ax=ax[2], label="1 - R", shrink=0.7)
    ax[2].set_title("Directional spread (uncertainty)", fontsize=11)
    ax[2].set_xlabel("X"); ax[2].set_ylabel("Y")

    # rows 1-2: per-draw signed angular deviation from the mean direction
    for k in range(n):
        a = ax[ncol + k]
        a.axis("on")
        dv = devs[k].T
        im = a.imshow(dv, origin="lower", cmap="RdBu_r",
                      vmin=-args.amax, vmax=args.amax, extent=extent, aspect="auto")
        # unit-vector quiver of this draw on top
        uh, vh, _ = IC.unit_normalize(members[k], ocean_np)
        a.quiver(xx, yy, uh.T[::args.step, ::args.step], vh.T[::args.step, ::args.step],
                 scale=34, width=0.0025, color="0.15", alpha=0.6)
        a.imshow(land_d, origin="lower", cmap=land_cmap, extent=extent,
                 aspect="auto", zorder=2)
        plt.colorbar(im, ax=a, label="deg from mean", shrink=0.7)
        rms_k = float(np.sqrt(np.nanmean(devs[k] ** 2)))
        a.set_title(f"Draw {k + 1}: angular deviation  (rms {rms_k:.1f}deg)",
                    fontsize=11)
        a.set_xlabel("X"); a.set_ylabel("Y")

    plt.suptitle(
        f"Directional variance that the fusion KEEPS — frame {src_f}, "
        f"coverage {cov:.1f}%, {n} draws\n"
        f"speed removed; colour = signed angle of each draw vs ensemble-mean "
        f"direction (overall rms {rms_dev:.1f}deg). White = draws agree; "
        f"strong colour = genuine uncertainty.",
        fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.out_dir, f"dirvar_frame{src_f}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"frame {src_f}  coverage {cov:.1f}%  draws {n}  rms_dev {rms_dev:.1f}deg")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
