"""
Compare coupled-fused fields BEFORE and AFTER Helmholtz re-projection.

Re-projection removes the divergent component via FFT:
  phi_k = -div_k / (kx^2 + ky^2)   (Poisson solve, k=0 -> 0)
  u_df  = u - grad(phi)              (subtract irrotational part)

Land cells are zeroed before projection and masked back after.

Outputs per run:
  1. Divergence stats table: raw / fused / reprojected
  2. Multidraw comparison PNG (4 frames): fused vs reprojected
  3. Calibration metrics table: r_angle, r_mag, r_overall, RMSE%
     for coupled vs coupled+reprojected

Usage (run from /workspace/DiffusionSummer2026):
  python "Conditional DDPM/testing/_probe_reproject.py" \
      --hetero_checkpoint "Magnitude/checkpoints_cond_mag_hetero_v2/best_cond_magnitude_hetero.pt" \
      --n_frames 20 --n_draws 6 --n_vis 4 --seed 0
"""
import argparse, os, sys
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

import infer_cond as IC
from _probe_calib_mag import (
    pcorr, directional_spread, vector_spread, magnitude_spread,
    load_magnitude_model, EPS,
)
from _probe_multidraw import (
    load_hetero_magnitude_model, predict_speed_mean_sigma, coupled_magnitude,
)


# ── Helmholtz projection ────────────────────────────────────────────────────

def _fft_project_once(ux, uy, kx, ky, k2, H, W):
    """One FFT Helmholtz step: removes divergent component from full grid."""
    Ux = np.fft.rfft2(ux)
    Uy = np.fft.rfft2(uy)
    Div = 1j * kx * Ux + 1j * ky * Uy
    Phi = -Div / k2
    Phi[0, 0] = 0.0
    ux_df = np.fft.irfft2(Ux - 1j * kx * Phi, s=(H, W))
    uy_df = np.fft.irfft2(Uy - 1j * ky * Phi, s=(H, W))
    return ux_df, uy_df


def helmholtz_project(field, ocean_mask, max_iters=5, tol=1e-4):
    """
    Iterative Helmholtz projection on a masked domain.

    Single-pass FFT projection assumes periodic full rectangle, so re-zeroing
    land after each pass introduces new divergence at coastlines. Iterating
    converges because each pass removes the divergence the previous re-masking
    introduced: project → re-mask → project → re-mask → ...

    Stops when mean|div| over ocean interior changes by less than `tol`
    relative to the previous iteration, or after `max_iters` passes.
    """
    ux = field[0].copy().astype(np.float64)
    uy = field[1].copy().astype(np.float64)
    land = ~ocean_mask

    H, W = ux.shape
    kx = np.fft.fftfreq(H, d=1.0 / (2 * np.pi))[:, None]
    ky = np.fft.rfftfreq(W, d=1.0 / (2 * np.pi))[None, :]
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0

    # interior mask for convergence check (avoid one-sided boundary artefacts)
    interior = np.zeros((H, W), dtype=bool)
    interior[1:-1, 1:-1] = True
    check = interior & ocean_mask

    prev_mean_div = np.inf
    for it in range(max_iters):
        ux[land] = 0.0; uy[land] = 0.0
        ux, uy = _fft_project_once(ux, uy, kx, ky, k2, H, W)
        ux[land] = 0.0; uy[land] = 0.0

        div = np.abs(_div_np(ux, uy))
        mean_div = float(div[check].mean())
        if abs(prev_mean_div - mean_div) / (prev_mean_div + 1e-12) < tol:
            break
        prev_mean_div = mean_div

    return np.stack([ux.astype(np.float32), uy.astype(np.float32)], axis=0)


def _div_np(ux, uy):
    """Central-diff divergence on numpy arrays (float64)."""
    dux = np.zeros_like(ux)
    dux[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / 2.0
    dux[:, 0]    = ux[:, 1]  - ux[:, 0]
    dux[:, -1]   = ux[:, -1] - ux[:, -2]
    duy = np.zeros_like(uy)
    duy[1:-1, :] = (uy[2:, :] - uy[:-2, :]) / 2.0
    duy[0, :]    = uy[1, :]  - uy[0, :]
    duy[-1, :]   = uy[-1, :] - uy[-2, :]
    return dux + duy


# ── Divergence (central diff) ───────────────────────────────────────────────

def divergence(field):
    return _div_np(field[0].astype(np.float64), field[1].astype(np.float64))


def div_stats(arr):
    return dict(mean=arr.mean(), p95=np.percentile(arr, 95),
                p99=np.percentile(arr, 99), mx=arr.max())


# ── Visuals ─────────────────────────────────────────────────────────────────

def _dir_spread_map(draws, ocean):
    """Per-cell directional spread: 1 - |mean unit vector| over draws."""
    units = []
    for m in draws:
        spd = np.sqrt(m[0]**2 + m[1]**2) + EPS
        units.append(np.stack([m[0]/spd, m[1]/spd], axis=0))
    mu = np.mean(units, axis=0)
    spread = 1.0 - np.sqrt(mu[0]**2 + mu[1]**2)
    spread[~ocean] = np.nan
    return spread


def _add_quiver(ax, field, land, title, vmax, step=2):
    H, W = land.shape
    xx, yy = np.meshgrid(np.arange(W)[::step], np.arange(H)[::step])
    u = field[0][::step, ::step]; v = field[1][::step, ::step]
    spd = np.sqrt(u**2 + v**2)
    ax.quiver(xx, yy, u, v, spd, cmap="cool", scale=vmax*30,
              width=0.003, clim=(0, vmax))
    ax.imshow(land.T, origin="lower",
              cmap=mcolors.ListedColormap(["none", "black"]),
              extent=[-0.5, W-0.5, -0.5, H-0.5], aspect="auto")
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def _add_spread(ax, spread, land, title):
    H, W = land.shape
    im = ax.imshow(spread.T, origin="lower", cmap="hot_r", vmin=0, vmax=1,
                   extent=[-0.5, W-0.5, -0.5, H-0.5], aspect="auto")
    ax.imshow(land.T, origin="lower",
              cmap=mcolors.ListedColormap(["none", "black"]),
              extent=[-0.5, W-0.5, -0.5, H-0.5], aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def render_multidraw_pair(out_path, frame_idx, true_np, draws_fused, draws_reproj,
                          land_np, ocean_np, fuse_div, reproj_div, cov):
    """
    Two side-by-side 3×(n_draws/2+1) grids: left=coupled, right=reprojected.
    Top row of each: ground truth | ensemble mean | directional spread.
    Remaining rows: all 6 individual draws (3 per col pair).
    """
    n = len(draws_fused)   # should be 6
    spd_true = np.sqrt((true_np**2).sum(axis=0))
    vmax = float(np.percentile(spd_true[~land_np], 98))

    mean_f = np.mean(draws_fused,  axis=0)
    mean_r = np.mean(draws_reproj, axis=0)
    spr_f  = _dir_spread_map(draws_fused,  ocean_np)
    spr_r  = _dir_spread_map(draws_reproj, ocean_np)
    disp_f = float(np.nanmean(spr_f))
    disp_r = float(np.nanmean(spr_r))

    n_draw_rows = n          # one draw per row, two panels per row (fused | reproj)
    n_rows = 1 + n_draw_rows # header + draws
    # cols: truth | fused_mean | fused_spread | reproj_mean | reproj_spread
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, 4 * n_rows), dpi=85)

    # header row
    _add_quiver(axes[0, 0], true_np, land_np, "Ground truth", vmax)
    _add_quiver(axes[0, 1], mean_f,  land_np,
                f"Coupled mean  |div|={fuse_div:.4f}", vmax)
    _add_spread(axes[0, 2], spr_f, land_np,
                f"Coupled spread  disp={disp_f:.1%}")
    _add_quiver(axes[0, 3], mean_r, land_np,
                f"Reproj mean  |div|={reproj_div:.4f}", vmax)
    _add_spread(axes[0, 4], spr_r, land_np,
                f"Reproj spread  disp={disp_r:.1%}")

    # draw rows
    for k in range(n):
        axes[k+1, 0].axis("off")
        _add_quiver(axes[k+1, 1], draws_fused[k],  land_np, f"Coupled draw {k+1}", vmax)
        axes[k+1, 2].axis("off")
        _add_quiver(axes[k+1, 3], draws_reproj[k], land_np, f"Reproj draw {k+1}",  vmax)
        axes[k+1, 4].axis("off")

    fig.suptitle(
        f"Coupled vs Helmholtz-reprojected (5 iters)  —  "
        f"frame {frame_idx}  |  coverage {cov:.1f}%",
        fontsize=10, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── Metrics ─────────────────────────────────────────────────────────────────

def vec_rmse_pct(members, true, mask):
    tu, tv = true[0][mask], true[1][mask]
    trms = np.sqrt((tu**2 + tv**2).mean()) + EPS
    vals = [np.sqrt(((m[0][mask]-tu)**2 + (m[1][mask]-tv)**2).mean()) / trms
            for m in members]
    return float(100.0 * np.mean(vals))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",        default="Models/StreamFn_Cond_x0_mag_spread.pt")
    ap.add_argument("--hetero_checkpoint", default="Magnitude/checkpoints_cond_mag_hetero_v2/best_cond_magnitude_hetero.pt")
    ap.add_argument("--pickle",            default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--split",     type=int, default=2)
    ap.add_argument("--n_frames",  type=int, default=20)
    ap.add_argument("--n_draws",   type=int, default=6)
    ap.add_argument("--n_vis",     type=int, default=3,  help="frames to render as PNG")
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--path_steps",      type=int, default=90)
    ap.add_argument("--inference_steps", type=int, default=100)
    ap.add_argument("--out_dir",   default="Conditional DDPM/results/cond_reproject")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device}  n_frames={args.n_frames}  n_draws={args.n_draws}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type"); ca = ckpt.get("args", {})
    lags = tuple(ckpt.get("lags", (13, 25))); cond_ch = ckpt.get("cond_ch", 10)
    data_std = float(ckpt.get("data_std") or 1.0)

    ds = IC.ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=ckpt.get("data_mean", 0.0), data_std=data_std,
        path_steps=args.path_steps, deterministic=True)
    land_np  = ds.land_mask.cpu().numpy().astype(bool)
    ocean_np = ~land_np

    model = IC.StreamFunctionUNet(in_ch=2, base_ch=ca.get("base_ch", 64),
        time_dim=ca.get("time_dim", 256), cond_ch=cond_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = IC.DDPM(T=ca.get("T", 1000), beta_schedule=ca.get("schedule", "cosine"),
        device=device, noise_type=ca.get("noise_type", "div_free"),
        spectral_filter=ckpt.get("spectral_filter", None))
    het_net, hsm, hss, het_clip = load_hetero_magnitude_model(
        args.hetero_checkpoint, device)

    sargs = argparse.Namespace(pred_type=pred_type,
        inference_steps=args.inference_steps, capture_every=10**9,
        n_ensemble=args.n_draws)

    interior = np.zeros(ocean_np.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    int_ocean = interior & ocean_np

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds.valid), size=min(args.n_frames, len(ds.valid)), replace=False)

    # accumulators: divergence
    raw_d, fused_d, reproj_d = [], [], []
    # accumulators: metrics
    m_fused  = dict(r_ang=[], r_mag=[], r_vec=[], rmse=[])
    m_reproj = dict(r_ang=[], r_mag=[], r_vec=[], rmse=[])

    vis_done = 0

    for i, src_idx in enumerate(indices):
        src_idx = int(src_idx)
        b = IC.build_cond(ds, src_idx, args.path_steps, seed=src_idx)
        true_np = b["target"].cpu().numpy()
        pm = b["path_mask"]
        pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
        cov = 100.0 * (pm & ocean_np).sum() / ocean_np.sum()

        _, _, members = IC.ensemble_infer(model, diffusion, b["cond"], land_np,
                                          sargs, device, base_seed=src_idx)

        mu_n, sig_n = predict_speed_mean_sigma(
            het_net, hsm, hss, land_np, data_std, device, b["cond"], het_clip)
        draws_fused  = coupled_magnitude(members, mu_n, sig_n, ocean_np)
        draws_reproj = [helmholtz_project(f, ocean_np) for f in draws_fused]

        # divergence
        for raw, fused, reproj in zip(members, draws_fused, draws_reproj):
            raw_np = raw if isinstance(raw, np.ndarray) else raw.cpu().numpy()
            raw_d.append(   np.abs(divergence(raw_np)  [int_ocean]))
            fused_d.append( np.abs(divergence(fused)   [int_ocean]))
            reproj_d.append(np.abs(divergence(reproj)  [int_ocean]))

        # calibration metrics (spread correlations)
        ang_f  = directional_spread(draws_fused,  ocean_np)
        ang_r  = directional_spread(draws_reproj, ocean_np)
        mag_f  = magnitude_spread(draws_fused,  ocean_np)
        mag_r  = magnitude_spread(draws_reproj, ocean_np)
        vec_f  = vector_spread(draws_fused,  ocean_np)
        vec_r  = vector_spread(draws_reproj, ocean_np)

        # we don't have empirical neighbours here, so compare spread self-consistency
        # (ang spread preserved, mag/vec spread after reproj)
        # Use ensemble mean as proxy target for RMSE
        for draws, acc in [(draws_fused, m_fused), (draws_reproj, m_reproj)]:
            ang = directional_spread(draws, ocean_np)
            mag = magnitude_spread(draws, ocean_np)
            vec = vector_spread(draws, ocean_np)
            acc["r_ang"].append(float(np.mean(ang[ocean_np])))
            acc["r_mag"].append(float(np.mean(mag[ocean_np])))
            acc["r_vec"].append(float(np.mean(vec[ocean_np])))
            acc["rmse"].append(vec_rmse_pct(draws, true_np, ocean_np))

        # visuals for first n_vis frames
        if vis_done < args.n_vis:
            src_f = int(ds.valid[src_idx])
            fmean_div = float(np.abs(divergence(np.mean(draws_fused,  axis=0))[int_ocean]).mean())
            rmean_div = float(np.abs(divergence(np.mean(draws_reproj, axis=0))[int_ocean]).mean())
            out_path = os.path.join(args.out_dir, f"reproject_frame{src_f}.png")
            render_multidraw_pair(out_path, src_f, true_np,
                                  draws_fused, draws_reproj, land_np, ocean_np,
                                  fmean_div, rmean_div, cov)
            vis_done += 1

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(indices)} frames done")

    # ── Results ──────────────────────────────────────────────────────────────
    raw_all    = np.concatenate(raw_d)
    fused_all  = np.concatenate(fused_d)
    reproj_all = np.concatenate(reproj_d)

    def pstats(arr, label):
        print(f"  {label:35s}  mean={arr.mean():.5f}  p95={np.percentile(arr,95):.5f}  max={arr.max():.5f}")

    print(f"\n{'='*70}")
    print(f"DIVERGENCE  ({len(indices)} frames × {args.n_draws} draws)")
    print(f"{'='*70}")
    pstats(raw_all,    "Raw diffusion  (stream-fn floor)")
    pstats(fused_all,  "Coupled-fused")
    pstats(reproj_all, "Coupled-fused + Helmholtz reproj")
    print(f"  fusion  → {fused_all.mean()/raw_all.mean():.2f}× raw")
    print(f"  reproj  → {reproj_all.mean()/raw_all.mean():.2f}× raw  "
          f"({reproj_all.mean()/fused_all.mean()*100:.0f}% of fused)")

    def mstats(d, label):
        print(f"  {label:35s}  "
              f"ang_spread={np.mean(d['r_ang']):.4f}  "
              f"mag_spread={np.mean(d['r_mag']):.4f}  "
              f"vec_spread={np.mean(d['r_vec']):.4f}  "
              f"RMSE%={np.mean(d['rmse']):.1f}")

    print(f"\n{'='*70}")
    print(f"SPREAD & ACCURACY  (mean over {len(indices)} frames)")
    print(f"{'='*70}")
    mstats(m_fused,  "Coupled-fused")
    mstats(m_reproj, "Coupled-fused + Helmholtz reproj")
    print(f"\nDone. Visuals saved to {args.out_dir}")


if __name__ == "__main__":
    main()
