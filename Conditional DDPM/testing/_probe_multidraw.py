"""
MULTI-DRAW INFERENCE — show the diffusion's non-determinism: many plausible
FUSED fields drawn from the SAME conditioning (same robot path, same priors),
using the best pipeline (fine-tuned diffusion DIRECTION x conditioned-UNet SPEED).

Each draw shares the conditioning but uses a different noise seed, so they agree
where the path/priors constrain the flow and diverge where the field is genuinely
uncertain — that divergence IS the active-sensing signal.

Renders, for ONE frame, a grid:
    [ ground truth | ensemble mean | directional spread ]
    [ draw 1 | draw 2 | ... | draw N ]
all FUSED and on a SHARED colour scale (truth's 98th pctile) so magnitudes are
comparable across panels.

Run:
  .venv/bin/python "Conditional DDPM/testing/_probe_multidraw.py" \
      --checkpoint Models/StreamFn_Cond_x0_mag_spread.pt \
      --mag_checkpoint Models/Cond_Magnitude_UNet.pt \
      --pickle Datasets/pickles/data_divfree_chrono.pickle \
      --split 2 --frame 4476 --n_draws 6 --path_steps 90 \
      --out_dir "Conditional DDPM/results/cond_multidraw"
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
from _probe_calib_mag import (                # noqa: E402
    load_magnitude_model, predict_speed_norm, apply_unet_magnitude,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--mag_checkpoint", default="Models/Cond_Magnitude_UNet.pt")
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--frame", type=int, default=-1,
                    help="split index; -1 picks a random frame via --seed")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n_draws", type=int, default=6)
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--out_dir", default="Conditional DDPM/results/cond_multidraw")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)
    print(f"model: {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch')} "
          f"mag={os.path.basename(args.mag_checkpoint)} n_draws={args.n_draws} "
          f"device={device}")

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

    if args.frame >= 0:
        # --frame is a FRAME number (value in ds.valid); fall back to treating it
        # as a split index if it isn't a valid frame.
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

    spd_phys = np.sqrt((src ** 2).sum(axis=0)) * data_std
    speed_norm = predict_speed_norm(mag_net, sm, ss, spd_phys, pm,
                                    land_np, data_std, device, cond=b["cond"])
    fused = apply_unet_magnitude(members, speed_norm, ocean_np)
    fused_mean = np.mean(fused, axis=0).astype(np.float32)
    spread = IC.directional_spread(members, ocean_np)

    # ---- render ----
    s = data_std; land_d = land_np.T; ocean_d = ~land_d
    tspd = np.sqrt((src[0] * s) ** 2 + (src[1] * s) ** 2).T
    vmax = float(np.nanpercentile(tspd[ocean_d], 98)) if ocean_d.any() else 1.0

    n = args.n_draws
    ncol = max(3, int(np.ceil(n / 2)))
    fig, axes = plt.subplots(3, ncol, figsize=(6.2 * ncol, 16), dpi=90)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")

    # row 0: truth | mean | spread
    ax[0].axis("on")
    IC.plot_field(ax[0], src[0].T * s, src[1].T * s, land_d,
                  "Ground truth", vmax=vmax)
    ax[1].axis("on")
    IC.plot_field(ax[1], fused_mean[0].T * s, fused_mean[1].T * s, land_d,
                  "Fused ensemble mean", vmax=vmax)
    ax[2].axis("on")
    sp = spread.T.copy()
    im = ax[2].imshow(sp, origin="lower", cmap="magma", vmin=0.0, vmax=1.0,
                      extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                      aspect="auto")
    ax[2].imshow(land_d, origin="lower",
                 cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
                 extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                 aspect="auto", zorder=2)
    plt.colorbar(im, ax=ax[2], label="1 - R", shrink=0.7)
    ax[2].set_title("Directional spread (uncertainty)", fontsize=11)
    ax[2].set_xlabel("X"); ax[2].set_ylabel("Y")

    # rows 1-2: the individual fused draws
    for k in range(n):
        a = ax[ncol + k]
        a.axis("on")
        IC.plot_field(a, fused[k][0].T * s, fused[k][1].T * s, land_d,
                      f"Plausible draw {k + 1}", vmax=vmax)

    plt.suptitle(
        f"Best pipeline — {n} plausible FUSED fields from the SAME conditioning  "
        f"(frame {src_f}, coverage {cov:.1f}%)\n"
        f"diffusion direction x conditioned-UNet speed; shared colour scale",
        fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, f"multidraw_frame{src_f}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"frame {src_f}  coverage {cov:.1f}%  draws {n}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
