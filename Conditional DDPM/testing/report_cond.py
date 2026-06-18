"""
Report + GIFs for the FiLM-conditioned (voronoi) DDPM on random val samples.

For each of N random validation samples it:
  - builds the robot path + voronoi conditioning map (from the NORMALIZED field,
    matching training),
  - runs conditional reverse diffusion (pure conditional by default, or anchored
    RePaint with --repaint),
  - denormalizes the prediction back to physical units,
  - reports the same honest metric suite as DDPM/testing/ppr_batch_infer.py:
        RMSE, mean |divergence|, AnomRat, ACC, KE-spectral low/high error,
  - and renders a 2x2 denoising GIF (GT | noisy xt | voronoi cond | x-hat_0).

Climatology (train-mean field, physical units) is reported as the skill floor.

Usage (from workspace root):
    python "Conditional DDPM/testing/report_cond.py" \
        --checkpoint Models/Cond_Div_Free_DDPM.pt \
        --pickle Datasets/data_divfree.pickle \
        --n_samples 3 --seed 1234
    # anchored RePaint variant (slower):
    python "Conditional DDPM/testing/report_cond.py" --repaint --resample 10
"""

import argparse
import os
import sys
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — works from the local nested layout or a flat server checkout.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
for _up in range(5):
    _cand = os.path.normpath(os.path.join(_HERE, *([".."] * _up)))
    if os.path.isdir(os.path.join(_cand, "utils")) and os.path.isdir(os.path.join(_cand, "DDPM")):
        _ROOT = _cand
        break

for _p in (
    _HERE,                                              # cond_model, cond_diffusion
    os.path.join(_ROOT, "utils"),                       # dataset, paths
    _ROOT,                                              # flat-layout fallback
    os.path.join(_ROOT, "Voronoi", "model"),            # voronoi_model
    os.path.join(_ROOT, "DDPM", "model"),               # divfree_projection
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset            import OceanCurrentDataset
from paths              import biased_walk_path
from voronoi_model      import VoronoiLayer
from cond_model         import CondUNet
from cond_diffusion     import CondDDPM
from divfree_projection import divergence as compute_divergence

COND_MODES = {"voronoi": 3, "path": 1, "path_field": 3, "both": 4}


# ---------------------------------------------------------------------------
# Plot helper (quiver)
# ---------------------------------------------------------------------------

def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool", vmax=None):
    H, W = u.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    mq     = np.sqrt(uq ** 2 + vq ** 2)
    mask   = ~np.isnan(uq) & ~land_mask[::step, ::step]
    clim_max = vmax if vmax is not None else (float(np.nanpercentile(mq[mask], 98)) if mask.any() else 1.0)
    if mask.any():
        q = ax.quiver(
            xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
            cmap=cmap, clim=(0, clim_max), scale=12, width=0.003, zorder=2,
        )
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def _denorm(arr_np, data_mean, data_std, land_mask_np):
    """Normalized (C,H,W) numpy -> physical units, land zeroed."""
    if data_mean is None:
        out = arr_np.copy()
    else:
        out = arr_np * data_std + data_mean
    out[:, land_mask_np] = 0.0
    return out


def plot_path(ax, path_mask_d, land_mask_d, path_cells, seed):
    """Static robot-path panel (same style as GIF making/repaint_gif.py)."""
    ax.imshow(
        land_mask_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_mask_d.shape[1] - 0.5, -0.5, land_mask_d.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    PATH_COLOR = (0.84, 0.10, 0.11, 1.0)
    path_rgba = np.zeros((*land_mask_d.shape, 4), dtype=float)
    path_rgba[path_mask_d] = PATH_COLOR
    ax.imshow(
        path_rgba, origin="lower",
        extent=[-0.5, land_mask_d.shape[1] - 0.5, -0.5, land_mask_d.shape[0] - 0.5],
        aspect="auto", zorder=1, interpolation="nearest",
    )
    ax.set_title(f"Robot Path ({path_cells} cells, seed={seed})", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(handles=[
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
        mpatches.Patch(facecolor=PATH_COLOR[:3],            label="Path"),
        mpatches.Patch(facecolor="black",                   label="Land"),
    ], loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# Voronoi conditioning (built from the NORMALIZED field, matching training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_voronoi_cond(x0_norm, land_mask_np, voronoi_layer, n_steps, seed, device):
    """Return ((1,3,H,W) voronoi cond, path_mask) for one normalized sample."""
    C, H, W = x0_norm.shape
    path_mask = biased_walk_path(land_mask_np, n_steps=n_steps, seed=seed)
    rows, cols = np.where(path_mask)
    K = len(rows)

    rows_n = torch.tensor(rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
    cols_n = torch.tensor(cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
    sensor_pos = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)          # (1, K, 2)

    flat_idx    = torch.tensor(rows * W + cols, dtype=torch.long, device=device)
    flat_idx    = flat_idx.unsqueeze(0).unsqueeze(0).expand(1, C, K)
    sensor_vals = torch.gather(x0_norm.unsqueeze(0).reshape(1, C, H * W), 2, flat_idx)
    voronoi_grid = voronoi_layer.tessellate(sensor_vals, sensor_pos)        # (1, 3, H, W)
    return voronoi_grid, path_mask


# ---------------------------------------------------------------------------
# Traced reverse diffusion — captures x-hat_0 frames for the GIF
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_traced(diffusion, model, cond, ocean_mask_t, n_frames, device,
                  n_steps=None, repaint=False, x0_known=None, path_mask_t=None, r=10):
    """
    Reverse diffusion with optional strided (respaced) schedule.

    n_steps = None        -> full T-step ancestral chain.
    n_steps = K (< T)     -> K evenly-spaced timesteps (respaced DDPM posterior).

    Returns (x0_pred_norm (2,H,W), frames) where frames is a list of
    (t_int, xt_norm, x0hat_norm) captured at ~n_frames evenly spaced schedule
    positions, in playback order (noisy -> clean).
    """
    T  = diffusion.T
    ns = diffusion.noise_scale
    B  = 1
    H, W = ocean_mask_t.shape[-2:]
    om = ocean_mask_t  # (1,1,H,W)

    # Build the (t, t_prev) schedule, descending. t_prev of the last step is 0.
    if n_steps is None or n_steps >= T:
        ts = list(range(T - 1, -1, -1))
    else:
        ts = sorted(set(np.linspace(0, T - 1, num=n_steps, dtype=int).tolist()), reverse=True)
    schedule = [(ts[i], ts[i + 1] if i + 1 < len(ts) else 0) for i in range(len(ts))]

    # Which schedule positions to snapshot for the GIF.
    n_cap   = min(n_frames, len(schedule))
    cap_pos = set(np.linspace(0, len(schedule) - 1, num=n_cap, dtype=int).tolist())
    cap_pos.add(len(schedule) - 1)

    if repaint:
        xt = diffusion._sample_noise(torch.empty(B, 2, H, W, device=device)) * ns * om
    else:
        xt = diffusion._sample_noise(torch.empty(B, 2, H, W, device=device)) * ns

    frames = []
    for pos, (t_int, s_int) in enumerate(schedule):
        x0hat = None
        for j in range(r if repaint else 1):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            pred_noise = model(xt, t, cond)

            ab_t = diffusion.alpha_bar[t_int]
            ab_s = diffusion.alpha_bar[s_int] if t_int > 0 else torch.ones((), device=device)

            x0hat = ((xt - (1.0 - ab_t).sqrt() * pred_noise) / ab_t.sqrt()).clamp(-3.0 * ns, 3.0 * ns)

            if t_int == 0:
                xt_model = x0hat
            else:
                # Respaced DDPM posterior q(x_s | x_t, x0).
                alpha = ab_t / ab_s
                beta  = 1.0 - alpha
                mean  = ((ab_s.sqrt() * beta / (1.0 - ab_t)) * x0hat +
                         (alpha.sqrt() * (1.0 - ab_s) / (1.0 - ab_t)) * xt)
                var   = beta * (1.0 - ab_s) / (1.0 - ab_t)
                xt_model = mean + var.sqrt() * ns * diffusion._sample_noise(xt)

            if repaint:
                s_q = torch.full((B,), max(s_int, 0), device=device, dtype=torch.long)
                xt_known, _ = diffusion.q_sample(x0_known, s_q)
                xt = path_mask_t * xt_known + (1.0 - path_mask_t) * xt_model
                xt = xt * om
                if j < r - 1 and t_int > 0:
                    # Re-noise back from s to t for the next resample iteration.
                    ab_t2 = diffusion.alpha_bar[t_int]
                    ab_s2 = diffusion.alpha_bar[s_int]
                    a = ab_t2 / ab_s2
                    xt = a.sqrt() * xt + (1.0 - a).sqrt() * ns * diffusion._sample_noise(xt)
                    xt = xt * om
            else:
                xt = xt_model

        if pos in cap_pos:
            frames.append((t_int, xt[0].detach().cpu().numpy(), x0hat[0].detach().cpu().numpy()))

    x0_pred_norm = xt[0]
    return x0_pred_norm, frames


# ---------------------------------------------------------------------------
# GIF frame rendering
# ---------------------------------------------------------------------------

def render_frame(t_int, T, gt_phys, cond_speed_uv, xt_phys, x0hat_phys,
                 pred_phys, path_mask_d, land_d, vmax, sample_idx, step_done,
                 path_cells, seed):
    """
    2x3 layout. Static panels (same every frame): Ground Truth, Robot Path,
    Final Prediction, Voronoi conditioning. Animated panels: noisy x_t and the
    model's running x-hat_0 estimate.

      [0] Ground Truth        [1] Robot Path          [2] Final Prediction (static)
      [3] Voronoi cond.       [4] Noisy field x_t     [5] Model x-hat_0 estimate
    """
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    axes = axes.flatten()
    plot_field(axes[0], gt_phys[0], gt_phys[1], land_d, "Ground Truth (target)", vmax=vmax)
    plot_path(axes[1], path_mask_d, land_d, path_cells, seed)
    plot_field(axes[2], pred_phys[0], pred_phys[1], land_d,
               "Final Prediction (t=0)", vmax=vmax)
    plot_field(axes[3], cond_speed_uv[0], cond_speed_uv[1], land_d,
               "Voronoi conditioning", vmax=vmax)
    plot_field(axes[4], xt_phys[0], xt_phys[1], land_d,
               f"Noisy field $x_t$   (t = {t_int})", vmax=vmax)
    plot_field(axes[5], x0hat_phys[0], x0hat_phys[1], land_d,
               r"Model $\hat{x}_0$ estimate" + f"   (t = {t_int})", vmax=vmax)
    pct = 100.0 * step_done / T
    plt.suptitle(
        f"Conditional DDPM denoising  —  val sample {sample_idx}"
        f"   |   step {step_done}/{T}  ({pct:.0f}%)",
        fontsize=13,
    )
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.load()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Metrics (mirrors DDPM/testing/ppr_batch_infer.py)
# ---------------------------------------------------------------------------

def _ke_spectrum_error(pred_np, true_np, land_mask_np):
    def _ke_spec(uv):
        ocean_f = (~land_mask_np).astype(np.float32)
        u, v = uv[0] * ocean_f, uv[1] * ocean_f
        H, W = u.shape
        fu, fv = np.fft.rfft2(u), np.fft.rfft2(v)
        ke = (np.abs(fu) ** 2 + np.abs(fv) ** 2) / 2.0
        kx = np.fft.fftfreq(H)[:, None]; ky = np.fft.rfftfreq(W)[None, :]
        k  = np.sqrt(kx ** 2 + ky ** 2)
        N  = min(H, W) // 2
        bins = np.linspace(0, k.max(), N + 1)
        spec = np.zeros(N)
        for i in range(N):
            m = (k >= bins[i]) & (k < bins[i + 1])
            if m.any():
                spec[i] = ke[m].mean()
        return spec
    diff = np.abs(_ke_spec(pred_np) - _ke_spec(true_np))
    mid  = len(diff) // 2
    return float(diff[:mid].mean()), float(diff[mid:].mean())


def compute_metrics(x0_pred_phys, x0_true_phys, land_mask_np, clim):
    ocean = ~land_mask_np
    err_sq = ((x0_pred_phys - x0_true_phys) ** 2).sum(0)
    rmse   = float(np.sqrt(err_sq[ocean].mean()))

    ocean_mask_t = torch.from_numpy(ocean)
    div  = compute_divergence(torch.from_numpy(x0_pred_phys).unsqueeze(0), ocean_mask_t)
    mean_div = float(div[0][ocean_mask_t].abs().mean().item())

    pa = (x0_pred_phys - clim)[:, ocean].reshape(-1)
    ta = (x0_true_phys - clim)[:, ocean].reshape(-1)
    anom_ratio = float(np.sqrt((pa ** 2).mean()) / (np.sqrt((ta ** 2).mean()) + 1e-12))
    denom = np.sqrt((pa ** 2).sum()) * np.sqrt((ta ** 2).sum()) + 1e-12
    acc   = float((pa * ta).sum() / denom)

    low_err, high_err = _ke_spectrum_error(x0_pred_phys, x0_true_phys, land_mask_np)
    return rmse, mean_div, anom_ratio, acc, low_err, high_err


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Conditional DDPM report + GIFs")
    p.add_argument("--checkpoint", default="Models/Cond_Div_Free_DDPM.pt")
    p.add_argument("--pickle",     default="Datasets/data_divfree.pickle")
    p.add_argument("--n_samples",  type=int, default=3)
    p.add_argument("--seed",       type=int, default=1234)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--inference_steps", type=int, default=None,
                   help="Denoising steps at inference (respaced schedule). "
                        "Default: full T steps.")
    p.add_argument("--repaint",    action="store_true",
                   help="Anchor observed path cells via RePaint (slower). "
                        "Default: pure conditional sampling.")
    p.add_argument("--resample",   type=int, default=10, help="RePaint resamples per step")
    p.add_argument("--gif_frames", type=int, default=50, help="Number of GIF frames to capture")
    p.add_argument("--n_gifs",     type=int, default=None,
                   help="Render GIFs only for the first N samples (default: all).")
    p.add_argument("--fps",        type=int, default=12)
    p.add_argument("--out_dir",    default="Conditional DDPM/results_report")
    p.add_argument("--device",     default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Checkpoint ----
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    cond_mode = ckpt_args.get("cond", "voronoi")
    cond_in_ch = COND_MODES[cond_mode]
    data_mean  = ckpt.get("data_mean", None)
    data_std   = ckpt.get("data_std",  None)
    spectral_filter = ckpt.get("spectral_filter", None)
    noise_type = ckpt_args.get("noise_type", "gaussian")
    T          = ckpt_args.get("T", 1000)

    model = CondUNet(
        in_ch=2, cond_in_ch=cond_in_ch,
        base_ch=ckpt_args.get("base_ch", 64),
        time_dim=ckpt_args.get("time_dim", 256),
        cond_dim=ckpt_args.get("cond_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = CondDDPM(
        T=T, beta_schedule=ckpt_args.get("schedule", "cosine"), device=device,
        noise_type=noise_type, spectral_filter=spectral_filter,
    )

    norm_str = f"mean={data_mean:.4f} std={data_std:.4f}" if data_mean is not None else "no"
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss',float('nan')):.5f}  "
          f"cond={cond_mode}  noise={noise_type}  normalize={norm_str}")
    print(f"Mode       : {'RePaint (anchored, r=%d)' % args.resample if args.repaint else 'pure conditional'}")

    # ---- Data (physical units; normalize per-sample inside the loop) ----
    val_ds       = OceanCurrentDataset(args.pickle, split=1)
    train_ds     = OceanCurrentDataset(args.pickle, split=0)
    land_mask_np = val_ds.land_mask.numpy()
    H, W         = land_mask_np.shape
    ocean        = ~land_mask_np
    land_d       = land_mask_np.T

    clim = train_ds.data.mean(dim=0).numpy()      # (2, H, W) physical
    clim[:, ~ocean] = 0.0

    rng         = np.random.default_rng(args.seed)
    sample_idxs = rng.choice(len(val_ds), size=min(args.n_samples, len(val_ds)),
                             replace=False).tolist()
    print(f"Val set size: {len(val_ds)}  |  samples: {sample_idxs}\n")

    voronoi_layer = VoronoiLayer(H=H, W=W, n_sensors=args.path_steps).to(device)

    header = (f"{'Run':>3}  {'Sample':>6}  {'Cells':>5}  {'RMSE':>8}  {'|div|':>9}  "
              f"{'AnomRat':>8}  {'ACC':>7}  {'SpecLo':>8}  {'SpecHi':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    clim_rmses = []
    for run, sample_idx in enumerate(sample_idxs, start=1):
        seed = run * 7 + 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        x0_true_phys = val_ds[sample_idx]                          # (2,H,W) physical
        x0_true_norm = (x0_true_phys - data_mean) / data_std if data_mean is not None else x0_true_phys
        x0_true_norm = x0_true_norm.to(device)

        cond, path_mask = make_voronoi_cond(
            x0_true_norm, land_mask_np, voronoi_layer, args.path_steps, seed, device)

        ocean_mask_t = torch.from_numpy(ocean.astype(np.float32)).to(device)[None, None]
        if args.repaint:
            path_t      = torch.from_numpy(path_mask.astype(np.float32)).to(device)
            x0_known    = (x0_true_norm * path_t.unsqueeze(0)).unsqueeze(0)
            path_mask_t = path_t[None, None]
            x0_pred_norm, frames = sample_traced(
                diffusion, model, cond, ocean_mask_t, args.gif_frames, device,
                n_steps=args.inference_steps, repaint=True,
                x0_known=x0_known, path_mask_t=path_mask_t, r=args.resample)
        else:
            x0_pred_norm, frames = sample_traced(
                diffusion, model, cond, ocean_mask_t, args.gif_frames, device,
                n_steps=args.inference_steps)

        x0_pred_phys = _denorm(x0_pred_norm.detach().cpu().numpy(), data_mean, data_std, land_mask_np)
        x0_true_np   = x0_true_phys.numpy()

        rmse, mean_div, anom_ratio, acc, low_err, high_err = compute_metrics(
            x0_pred_phys, x0_true_np, land_mask_np, clim)
        rows.append((rmse, mean_div, anom_ratio, acc, low_err, high_err))
        clim_rmses.append(float(np.sqrt((((clim - x0_true_np) ** 2).sum(0))[ocean].mean())))

        print(f"{run:>3}  {sample_idx:>6}  {int(path_mask.sum()):>5}  {rmse:>8.4f}  "
              f"{mean_div:>9.6f}  {anom_ratio:>8.3f}  {acc:>7.3f}  {low_err:>8.4f}  {high_err:>8.4f}")

        # ---- Build GIF (only for the first --n_gifs samples) ----
        if args.n_gifs is not None and run > args.n_gifs:
            continue
        cond_phys = _denorm(cond[0].detach().cpu().numpy()[:2], data_mean, data_std, land_mask_np)
        speed  = np.sqrt(x0_true_np[0] ** 2 + x0_true_np[1] ** 2); speed[land_mask_np] = np.nan
        vmax   = float(np.nanpercentile(speed, 98)) or 1.0
        path_mask_d = path_mask.T
        path_cells  = int(path_mask.sum())

        pil_frames = []
        for (t_int, xt_norm, x0hat_norm) in frames:
            xt_phys    = _denorm(xt_norm, data_mean, data_std, land_mask_np)
            x0hat_phys = _denorm(x0hat_norm, data_mean, data_std, land_mask_np)
            pil_frames.append(render_frame(
                t_int, T,
                (x0_true_np[0].T, x0_true_np[1].T),
                (cond_phys[0].T, cond_phys[1].T),
                (xt_phys[0].T, xt_phys[1].T),
                (x0hat_phys[0].T, x0hat_phys[1].T),
                (x0_pred_phys[0].T, x0_pred_phys[1].T),
                path_mask_d, land_d, vmax, sample_idx, T - t_int,
                path_cells, seed))

        gif_path = os.path.join(args.out_dir, f"cond_denoise_val{sample_idx}.gif")
        dur = max(1, int(1000 / args.fps))
        durations = [dur] * (len(pil_frames) - 1) + [1200]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           duration=durations, loop=0, optimize=False)
        print(f"      GIF -> {gif_path}  ({len(pil_frames)} frames)")

    # ---- Summary ----
    arr = np.array(rows)
    clim_m = float(np.mean(clim_rmses))
    print("\n" + "=" * (len(header) + 2))
    print("  SUMMARY")
    print("=" * (len(header) + 2))
    print(f"  {'Method':<14}{'RMSE':>9}{'Std':>9}{'|div|':>11}{'AnomRat':>10}{'ACC':>9}{'SpecLo':>11}{'SpecHi':>10}")
    print(f"  {'-'*84}")
    m = arr.mean(0)
    print(f"  {'cond-'+cond_mode:<14}{m[0]:>9.4f}{arr[:,0].std():>9.4f}{m[1]:>11.6f}"
          f"{m[2]:>10.3f}{m[3]:>9.3f}{m[4]:>11.4f}{m[5]:>10.4f}")
    print(f"  {'climatology':<14}{clim_m:>9.4f}{'-':>9}{'-':>11}{'0.000':>10}{'0.000':>9}{'-':>11}{'-':>10}")
    print(f"\n  AnomRat = anomaly-RMS / true-anomaly-RMS (~1 = right fluctuation energy).")
    print(f"  ACC     = anomaly pattern correlation vs GT (1 = perfect, 0 = climatology, <0 = worse).")
    print(f"  REAL skill needs RMSE < climatology ({clim_m:.4f}) AND ACC > 0.")
    print(f"\n  Results + GIFs in: {args.out_dir}")


if __name__ == "__main__":
    main()
