"""
Inference + visualisation for the CONDITIONAL stream-function divergence-free DDPM.

Counterpart of the unconditional `direction_magnitude_display.py`, written for the
model trained by `train_streamfn_cond.py` (pred_type = "x0_streamfn_cond").  The
model ingests, alongside the noisy field x_t, a 10-channel conditioning stack
(soft robot-path observations, 13 h + 25 h temporal-prior fields, and static
geometry) and emits a scalar stream function whose curl is an EXACTLY
divergence-free (u, v) field.

Because the observations enter as INPUT CHANNELS (not a hard inpainting
constraint), sampling is plain ancestral DDPM with the conditioning threaded
through every model call — handled cleanly by wrapping the stream model in
`EpsFromStreamFn(..., cond=cond)` and reusing the standard `p_sample_step`
reverse loop.  Diffusion non-determinism then yields a DIVERSE ENSEMBLE of
plausible fields from the SAME fixed constraints — the project north star.

For each requested validation sample this produces (in the established visual
style — quiver panels coloured by speed, land in black):

  (A) a static summary PNG — 2x3 panels:
        1. Ground truth
        2. Robot observations (arrows on the path) + temporal prior underlay
        3. Temporal prior (prev 13 h)
        4. Conditional prediction (ensemble mean)
        5. Ensemble member 0 (one plausible draw)
        6. Directional spread (ensemble disagreement = uncertainty)

  (B) a denoising GIF — 1x2 panels animated over the reverse process:
        [ noisy field x_t | model x_hat_0 ]   (so you watch the field emerge)

  (C) an ensemble-diversity PNG — the N individual draws side by side, showing
      how the guesses agree where informed and diverge where unconstrained.

Usage (from workspace root, after a checkpoint exists):
    python DDPM/testing/cond_infer_display.py \
        --checkpoint DDPM/model/checkpoints_streamfn_cond/best_streamfncond_minsnr5_ang1_lags13-25_div_free_cosine.pt \
        --pickle     Datasets/data_divfree.pickle \
        --n_samples  4 --random --seed 1234 \
        --n_ensemble 6 --inference_steps 100 \
        --out_dir    DDPM/best_model_results/cond_streamfn
"""

import argparse
import os
import sys
from io import BytesIO

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — works from the workspace root or the flat server layout.
# ---------------------------------------------------------------------------
_here  = os.path.dirname(os.path.abspath(__file__))
_root  = os.path.normpath(os.path.join(_here, "..", ".."))
_model = os.path.join(_here, "..", "model")
for _p in [_root, os.path.join(_root, "utils"), _model, os.path.join(_root, "DDPM", "model")]:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from diffusion    import DDPM, EpsFromStreamFn
from model        import StreamFunctionUNet
from cond_dataset import (
    ConditionalOceanDataset,
    observation_channels,
    assemble_cond,
    cond_channels,
)
from paths import biased_walk_path


# ===========================================================================
# Vector / plotting helpers  (consistent with plot_utils + the angle display)
# ===========================================================================

def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool", vmax=None):
    """Quiver plot of a (H, W) field; land black, arrows coloured by speed."""
    H, W = u.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=mcolors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u[::step, ::step]
    vq = v[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    mask = ~np.isnan(uq) & ~land_mask[::step, ::step]
    clim_max = vmax if vmax is not None else (
        np.nanpercentile(mq[mask], 98) if mask.any() else 1.0)
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
        cmap=cmap, clim=(0, clim_max), scale=12, width=0.003, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def plot_path(ax, path_mask, land_mask, title):
    """Show the robot path cells over the land mask."""
    H, W = land_mask.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=mcolors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    rows, cols = np.where(path_mask)
    ax.scatter(cols, rows, s=8, c="tab:red", zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.5, H - 0.5)


def unit_normalize(field_np, ocean_np, eps=1e-8):
    """Unit-normalize each vector of a (2, H, W) field; land/near-zero -> 0."""
    u, v = field_np[0], field_np[1]
    mag  = np.sqrt(u ** 2 + v ** 2)
    safe = mag > eps
    uh = np.zeros_like(u); vh = np.zeros_like(v)
    uh[safe] = u[safe] / mag[safe]
    vh[safe] = v[safe] / mag[safe]
    uh[~ocean_np] = 0.0; vh[~ocean_np] = 0.0
    return uh, vh, mag


def angle_error_deg(pred_np, true_np, ocean_np, eps=1e-8):
    """Per-cell angular error in degrees [0,180]; NaN at land / near-zero."""
    up, vp = pred_np[0], pred_np[1]
    ut, vt = true_np[0], true_np[1]
    dot = up * ut + vp * vt
    mp  = np.sqrt(up ** 2 + vp ** 2)
    mt  = np.sqrt(ut ** 2 + vt ** 2)
    cos = np.clip(dot / (mp * mt + eps), -1.0, 1.0)
    err = np.degrees(np.arccos(cos))
    valid = ocean_np & (mp > eps) & (mt > eps)
    out = np.full(err.shape, np.nan, dtype=np.float32)
    out[valid] = err[valid]
    return out


def directional_spread(members, ocean_np, eps=1e-8):
    """Per-cell circular spread (1 - resultant length) across the ensemble."""
    us, vs = [], []
    for m in members:
        uh, vh, _ = unit_normalize(m, ocean_np, eps)
        us.append(uh); vs.append(vh)
    mean_u = np.mean(us, axis=0); mean_v = np.mean(vs, axis=0)
    R = np.sqrt(mean_u ** 2 + mean_v ** 2)
    spread = 1.0 - R
    spread[~ocean_np] = np.nan
    return spread.astype(np.float32)


# ===========================================================================
# Distance-to-path stratification (near-field vs far-field error)
# ===========================================================================

_DIST_BANDS = [
    (0.0,  2.0,    "near  (0-2)"),
    (2.0,  5.0,    "mid   (2-5)"),
    (5.0,  10.0,   "far   (5-10)"),
    (10.0, np.inf, "deep  (10+)"),
]


def distance_to_path(path_mask, ocean_np):
    """Euclidean distance (cells) from each ocean cell to nearest path cell."""
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(~path_mask).astype(np.float32)
    dist[~ocean_np] = np.nan
    return dist


def stratified_rows(err_vals, dist_vals):
    """Bin per-cell angular error by distance-to-path band."""
    rows = []
    for lo, hi, label in _DIST_BANDS:
        m = (dist_vals >= lo) & (dist_vals < hi)
        if m.any():
            e = err_vals[m]
            rows.append((label, int(e.size), float(np.mean(e)),
                         float(np.mean(np.cos(np.radians(e))))))
        else:
            rows.append((label, 0, float("nan"), float("nan")))
    return rows


# ===========================================================================
# Conditioning construction  (EXACTLY matches ConditionalOceanDataset.__getitem__)
# ===========================================================================

def build_cond(ds, idx, path_steps, seed):
    """
    Build the conditioning tensor for one validation frame, returning everything
    needed for both sampling and visualisation.

    Uses the SAME helpers as training (observation_channels / assemble_cond /
    ds.geom), so the conditioning is guaranteed identical to what the model saw.

    Returns dict with:
        target    (2, H, W) normalized ground-truth field
        priors    (2*nlags, H, W) temporal-prior fields
        cond      (C, H, W) assembled conditioning
        path_mask (H, W) bool robot path
    """
    f = int(ds.valid[idx])
    target = ds.fields[f]                                         # (2, H, W)
    priors = torch.cat([ds.fields[f - L] for L in ds.lags], dim=0)
    path_mask = biased_walk_path(ds._land_np, n_steps=path_steps,
                                 seed=seed, straight_bias=ds.straight_bias)
    obs  = observation_channels(target, path_mask)               # (3, H, W)
    cond = assemble_cond(obs, priors, ds.geom)                   # (C, H, W)
    return {"target": target, "priors": priors, "cond": cond, "path_mask": path_mask}


# ===========================================================================
# Conditional ancestral sampling — captures the denoising trajectory
# ===========================================================================

@torch.no_grad()
def sample_capture(stream_model, diffusion, cond, land_np, args, device, seed):
    """
    Draw one conditional posterior sample, capturing (t, x_t, x_hat_0) frames.

    The conditioning is baked into an EpsFromStreamFn adapter so the standard
    `p_sample_step` reverse loop runs unchanged.  x_hat_0 is read directly from
    the (x0-prediction) stream model — it is divergence-free by construction.

    Returns (final_pred_np (2,H,W), frames list of (t_int, xt_np, x0hat_np)).
    """
    H, W = land_np.shape
    ocean_f = torch.from_numpy(~land_np).float().to(device)[None, None]
    cond_b  = cond.unsqueeze(0).to(device)                       # (1, C, H, W)

    eps_model = EpsFromStreamFn(stream_model, diffusion, cond=cond_b).to(device)

    torch.manual_seed(seed)
    xt = diffusion._sample_noise(torch.empty(1, 2, H, W, device=device))
    xt = xt * diffusion.noise_scale * ocean_f

    schedule = diffusion.build_inference_schedule(args.inference_steps)
    frames   = []

    def x0hat_np(xt_, t_):
        t_t = torch.full((1,), max(t_, 0), device=device, dtype=torch.long)
        x0  = stream_model(xt_, t_t, cond_b)
        return (x0 * ocean_f).squeeze(0).cpu().numpy()

    n = len(schedule)
    for step_i, (t_int, t_prev_int) in enumerate(schedule):
        xt = diffusion.p_sample_step(eps_model, xt, t_int, t_prev_int) * ocean_f
        if step_i == 0 or t_prev_int < 0 or step_i % max(1, args.capture_every) == 0:
            frames.append((t_int, (xt * ocean_f).squeeze(0).cpu().numpy(),
                           x0hat_np(xt, max(t_prev_int, 0))))

    final_pred = frames[-1][2]
    return final_pred, frames


@torch.no_grad()
def ensemble_infer(stream_model, diffusion, cond, land_np, args, device, base_seed=0):
    """Draw args.n_ensemble diverse conditional samples from the SAME cond."""
    members, frames0 = [], None
    for k in range(max(1, args.n_ensemble)):
        seed = (base_seed + 1) * 100003 + k
        pred_k, frames_k = sample_capture(
            stream_model, diffusion, cond, land_np, args, device, seed)
        members.append(pred_k)
        if k == 0:
            frames0 = frames_k
    mean_pred = np.mean(members, axis=0).astype(np.float32)
    return mean_pred, frames0, members


# ===========================================================================
# Rendering
# ===========================================================================

def render_summary(out_path, idx, seed, true_np, prior_np, mean_np, member0_np,
                   spread, path_mask, land_np, vmax, cov_pct, stats_txt):
    """2x3 static summary panel."""
    ocean_np = ~land_np
    land_d = land_np.T
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=90)
    ax = axes.flatten()

    plot_field(ax[0], true_np[0].T, true_np[1].T, land_d, "Ground truth", vmax=vmax)
    plot_path(ax[1], path_mask.T, land_d,
              f"Robot observations  ({path_mask.sum()} cells, {cov_pct:.1f}%)")
    plot_field(ax[2], prior_np[0].T, prior_np[1].T, land_d,
               "Temporal prior  (prev 13 h)", vmax=vmax)
    plot_field(ax[3], mean_np[0].T, mean_np[1].T, land_d,
               "Conditional prediction  (ensemble mean)", vmax=vmax)
    plot_field(ax[4], member0_np[0].T, member0_np[1].T, land_d,
               "One plausible draw  (member 0)", vmax=vmax)

    sp = spread.T.copy()
    im = ax[5].imshow(sp, origin="lower", cmap="magma", vmin=0.0, vmax=1.0,
                      extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                      aspect="auto")
    ax[5].imshow(land_d, origin="lower",
                 cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
                 extent=[-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5],
                 aspect="auto", zorder=2)
    plt.colorbar(im, ax=ax[5], label="1 - R  (direction disagreement)", shrink=0.7)
    ax[5].set_title("Ensemble directional spread  (uncertainty)", fontsize=11)
    ax[5].set_xlabel("X"); ax[5].set_ylabel("Y")

    plt.suptitle(
        f"Conditional Stream-fn DDPM  —  val sample {idx}  (seed {seed})\n{stats_txt}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_ensemble(out_path, idx, members, land_np, vmax):
    """Grid of the individual ensemble draws."""
    land_d = land_np.T
    n = len(members)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.0 * rows), dpi=90)
    axes = np.atleast_1d(axes).flatten()
    for k, m in enumerate(members):
        plot_field(axes[k], m[0].T, m[1].T, land_d, f"Draw {k}", vmax=vmax)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    plt.suptitle(f"Conditional ensemble — {n} diverse draws — val sample {idx}",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_gif(out_path, idx, frames, true_np, land_np, fps):
    """1x2 denoising GIF: [noisy x_t | model x_hat_0], unit-normalized."""
    ocean_np = ~land_np
    land_d = land_np.T
    ut, vt, _ = unit_normalize(true_np, ocean_np)
    pil_frames = []
    T = 1000
    for (t_int, xt_np, x0_np) in frames:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=80)
        uxt, vxt, _ = unit_normalize(xt_np, ocean_np)
        ux0, vx0, _ = unit_normalize(x0_np, ocean_np)
        plot_field(axes[0], uxt.T, vxt.T, land_d,
                   f"Noisy field  $x_t$   (t = {t_int})", vmax=1.0)
        plot_field(axes[1], ux0.T, vx0.T, land_d,
                   r"Model $\hat{x}_0$" + f"   (t = {t_int})", vmax=1.0)
        plt.suptitle(f"Conditional denoising — val sample {idx}   (t = {t_int})",
                     fontsize=13)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).convert("RGB"); img.load()
        pil_frames.append(img)
        plt.close(fig)
    if not pil_frames:
        return
    dur = max(1, int(1000 / fps))
    durations = [dur] * (len(pil_frames) - 1) + [1500]
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=durations, loop=0, optimize=False)


# ===========================================================================
# Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Conditional stream-function DDPM inference + visualisation.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pickle", default="Datasets/data_divfree.pickle")
    p.add_argument("--split", type=int, default=1, help="0=train,1=val,2=test")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--random", action="store_true",
                   help="Pick random sample indices (else 0..n_samples-1).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n_ensemble", type=int, default=6,
                   help="Number of diverse posterior draws per sample.")
    p.add_argument("--inference_steps", type=int, default=100)
    p.add_argument("--capture_every", type=int, default=5,
                   help="Capture a GIF frame every N reverse steps.")
    p.add_argument("--path_steps", type=int, default=160,
                   help="Robot-path length (fixed at inference).")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--no_gif", action="store_true", help="Skip the denoising GIF.")
    p.add_argument("--out_dir", default="DDPM/best_model_results/cond_streamfn")
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    # ---- Checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("pred_type") != "x0_streamfn_cond":
        raise ValueError(
            f"Expected pred_type 'x0_streamfn_cond', got {ckpt.get('pred_type')!r}. "
            "Use direction_magnitude_display.py for unconditional models.")
    ca        = ckpt.get("args", {})
    base_ch   = ca.get("base_ch", 64)
    time_dim  = ca.get("time_dim", 256)
    T         = ca.get("T", 1000)
    noise_type = ca.get("noise_type", "div_free")
    schedule  = ca.get("schedule", "cosine")
    lags      = tuple(ckpt.get("lags", ca.get("lags", (13, 25))))
    cond_ch   = ckpt.get("cond_ch", cond_channels(lags))
    data_mean = ckpt.get("data_mean", 0.0)
    data_std  = ckpt.get("data_std", None)
    spectral_filter = ckpt.get("spectral_filter", None)
    print(f"Model      : epoch {ckpt.get('epoch','?')}  val={ckpt.get('val_loss', float('nan')):.5f}  "
          f"lags={lags}  cond_ch={cond_ch}  noise={noise_type}")

    # ---- Data (same normalization the model trained with) ----
    ds = ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=data_mean, data_std=data_std,
        path_steps=args.path_steps, deterministic=True,
    )
    land_np  = ds.land_mask.cpu().numpy().astype(bool)
    ocean_np = ~land_np
    n_ocean  = int(ocean_np.sum())
    print(f"Data       : split {args.split}  samples={len(ds)}  ocean={n_ocean}")

    # ---- Model + diffusion ----
    stream_model = StreamFunctionUNet(
        in_ch=2, base_ch=base_ch, time_dim=time_dim, cond_ch=cond_ch).to(device)
    stream_model.load_state_dict(ckpt["model"])
    stream_model.eval()
    diffusion = DDPM(T=T, beta_schedule=schedule, device=device,
                     noise_type=noise_type, spectral_filter=spectral_filter)

    # ---- Sample indices ----
    rng = np.random.default_rng(args.seed)
    if args.random:
        indices = rng.integers(0, len(ds), size=args.n_samples).tolist()
    else:
        indices = list(range(min(args.n_samples, len(ds))))

    for s_i, idx in enumerate(indices):
        seed = args.seed + idx
        b = build_cond(ds, idx, args.path_steps, seed)
        true_np  = b["target"].cpu().numpy()
        prior_np = b["priors"][:2].cpu().numpy()                 # prev-13 (u,v)
        path_mask = b["path_mask"]
        cov_pct = 100.0 * path_mask.sum() / n_ocean

        print(f"\n[{s_i+1}/{len(indices)}] sample {idx}  path={path_mask.sum()}/{n_ocean} "
              f"({cov_pct:.1f}%)  drawing {args.n_ensemble} members ...")

        mean_np, frames0, members = ensemble_infer(
            stream_model, diffusion, b["cond"], land_np, args, device, base_seed=idx)

        # ---- Metrics (ensemble mean + member 0), stratified by distance ----
        dist = distance_to_path(path_mask, ocean_np)
        err_mean = angle_error_deg(mean_np, true_np, ocean_np)
        ev = err_mean[ocean_np & ~np.isnan(err_mean)]
        dv = dist[ocean_np & ~np.isnan(err_mean)]
        rmse_mean = float(np.sqrt(np.mean((mean_np[:, ocean_np] - true_np[:, ocean_np]) ** 2)))
        unobs = dv > 2.0
        uo = ev[unobs]
        print(f"   RMSE(mean)={rmse_mean:.4f}  angle(all)={np.mean(ev):.1f}deg  "
              f"unobserved(>2)={np.mean(uo):.1f}deg (n={uo.size})")
        for label, n_c, m_e, m_c in stratified_rows(ev, dv):
            print(f"     {label:12s} n={n_c:5d}  angle={m_e:6.1f}deg  cos={m_c:+.3f}")
        stats_txt = (f"RMSE(mean)={rmse_mean:.4f}   angle(all)={np.mean(ev):.1f}deg   "
                     f"unobserved={np.mean(uo):.1f}deg   coverage={cov_pct:.1f}%")

        # ---- Common colour scale from ground truth ----
        spd = np.sqrt(true_np[0] ** 2 + true_np[1] ** 2)
        spd[land_np] = np.nan
        vmax = float(np.nanpercentile(spd, 98)) or 1.0

        spread = directional_spread(members, ocean_np)

        base = f"sample{idx:04d}"
        render_summary(
            os.path.join(args.out_dir, f"{base}_summary.png"),
            idx, seed, true_np, prior_np, mean_np, members[0],
            spread, path_mask, land_np, vmax, cov_pct, stats_txt)
        render_ensemble(
            os.path.join(args.out_dir, f"{base}_ensemble.png"),
            idx, members, land_np, vmax)
        if not args.no_gif:
            render_gif(
                os.path.join(args.out_dir, f"{base}_denoise.gif"),
                idx, frames0, true_np, land_np, args.fps)
        print(f"   saved -> {args.out_dir}/{base}_*.png/gif")

    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
