"""
Direction × Magnitude display for the angle-decomposition pipeline.

The angle-loss DDPM only models *direction* (its magnitudes are meaningless), so
the final usable field is assembled as

        final field = unit_direction(DDPM)  ×  magnitude(UNet speed model)

This script renders that story for one or more validation samples.  For each
sample it produces:

  (A) a static summary PNG — a 2×3 panel:
        1. Ground-truth field            (arrows coloured by speed)
        2. Ground-truth direction        (unit arrows, cyclic colour = compass)
        3. Robot path
        4. DDPM final prediction         (raw model output — magnitudes ignored)
        5. DDPM prediction direction     (unit arrows — compare directly to #2)
        6. Fused field: DDPM direction × UNet magnitude
           (skipped with a note if no magnitude checkpoint is supplied)

  (B) a denoising GIF — a 1×2 panel animated over the reverse process:
        [ current field x_t (unit-normalized) | model x̂₀ (unit-normalized) ]
        both normalized the same way so you watch direction emerge from noise.

Works for both PPR and RePaint inference via --method.  The GIF's final frame is
the exact field shown in summary panels 4/5, so the animation and the statics
are guaranteed consistent.

Usage (from workspace root):
    python DDPM/testing/direction_magnitude_display.py \
        --checkpoint     checkpoints_angle/best_ddpm_angle_div_free_cosine.pt \
        --pickle         Datasets/data.pickle \
        --method         repaint \
        --n_samples      5 --random --seed 1234 \
        --inference_steps 100 --resample 10 \
        --mag_checkpoint Magnitude/checkpoints/best_magnitude_unet.pt \
        --out_dir        DDPM/best_model_results/dir_mag
"""

import argparse
import importlib.util
import os
import sys
from io import BytesIO

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image

_here    = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.normpath(os.path.join(_here, "..", ".."))
_model   = os.path.join(_here, "..", "model")
_repaint = os.path.join(_here, "repaint")
_ppr     = os.path.join(_here, "ppr")
for _p in [_root, os.path.join(_root, "utils"), _model, _repaint, _ppr]:
    sys.path.insert(0, _p)

from dataset            import OceanCurrentDataset
from diffusion          import DDPM, EpsFromStreamFn
from model              import UNet, StreamFunctionUNet
from divfree_projection import joint_project
from repaint_infer      import biased_walk_path


# ===========================================================================
# Vector helpers
# ===========================================================================

def unit_normalize(field_np: np.ndarray, ocean_np: np.ndarray, eps: float = 1e-8):
    """Unit-normalize every vector of a (2, H, W) field; land/near-zero -> 0."""
    u, v  = field_np[0], field_np[1]
    mag   = np.sqrt(u ** 2 + v ** 2)
    safe  = mag > eps
    u_hat = np.zeros_like(u)
    v_hat = np.zeros_like(v)
    u_hat[safe] = u[safe] / mag[safe]
    v_hat[safe] = v[safe] / mag[safe]
    u_hat[~ocean_np] = 0.0
    v_hat[~ocean_np] = 0.0
    return u_hat, v_hat, mag


def angle_error_deg(pred_np: np.ndarray, true_np: np.ndarray,
                    ocean_np: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-cell angular error in degrees [0,180]; NaN at land / near-zero cells."""
    up, vp = pred_np[0], pred_np[1]
    ut, vt = true_np[0], true_np[1]
    dot    = up * ut + vp * vt
    mp     = np.sqrt(up ** 2 + vp ** 2)
    mt     = np.sqrt(ut ** 2 + vt ** 2)
    cos    = np.clip(dot / (mp * mt + eps), -1.0, 1.0)
    err    = np.degrees(np.arccos(cos))
    valid  = ocean_np & (mp > eps) & (mt > eps)
    out    = np.full(err.shape, np.nan, dtype=np.float32)
    out[valid] = err[valid]
    return out


def directional_spread(members: list[np.ndarray], ocean_np: np.ndarray,
                       eps: float = 1e-8) -> np.ndarray:
    """
    Per-cell directional disagreement across an ensemble of (2, H, W) fields.

    Each member's vector is unit-normalized, then the circular spread is
    1 − R where R = |mean unit vector| ∈ [0, 1] is the resultant length.
    R = 1 → all members point the same way (confident); R ≈ 0 → directions
    scattered (uncertain).  Magnitude-independent, matching the angle metric.

    Returns (H, W) float in [0, 1]; NaN at land.
    """
    us, vs = [], []
    for m in members:
        uh, vh, _ = unit_normalize(m, ocean_np, eps)
        us.append(uh); vs.append(vh)
    mean_u = np.mean(us, axis=0)
    mean_v = np.mean(vs, axis=0)
    R = np.sqrt(mean_u ** 2 + mean_v ** 2)          # resultant length [0,1]
    spread = 1.0 - R
    spread[~ocean_np] = np.nan
    return spread.astype(np.float32)


@torch.no_grad()
def ensemble_infer(model, diffusion, x0_obs, path_mask, land_np, args, device,
                   base_seed=0):
    """
    Draw args.n_ensemble independent posterior samples and average them.

    Returns (mean_pred, frames, members):
      mean_pred : (2, H, W) posterior-mean field (average of the members, in the
                  model's normalized space — averaging is linear so a stream-
                  function model's div-free property is preserved).
      frames    : denoising trajectory of the FIRST member (for the GIF).
      members   : list of the individual member predictions.
    Each member is seeded deterministically from base_seed so runs are
    reproducible and members are independent across samples.
    """
    members, frames0 = [], None
    for k in range(max(1, args.n_ensemble)):
        torch.manual_seed((base_seed + 1) * 100003 + k)
        pred_k, frames_k = infer_capture(
            model, diffusion, x0_obs, path_mask, land_np, args, device)
        members.append(pred_k)
        if k == 0:
            frames0 = frames_k
    mean_pred = np.mean(members, axis=0).astype(np.float32)
    return mean_pred, frames0, members


# ===========================================================================
# Distance-to-path stratification (Option D): near-field vs far-field error
# ===========================================================================

_DIST_BANDS = [
    (0.0,  2.0,      "near  (0-2)"),
    (2.0,  5.0,      "mid   (2-5)"),
    (5.0,  10.0,     "far   (5-10)"),
    (10.0, np.inf,   "deep  (10+)"),
]


def distance_to_path(path_mask: np.ndarray, ocean_np: np.ndarray) -> np.ndarray:
    """Euclidean distance (in grid cells) from each ocean cell to the nearest
    observed path cell.  Land cells are NaN."""
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(~path_mask).astype(np.float32)
    dist[~ocean_np] = np.nan
    return dist


def stratified_rows(err_vals: np.ndarray, dist_vals: np.ndarray):
    """Bin per-cell angular error by distance-to-path band.  Returns a list of
    (label, n_cells, mean_err_deg, mean_cos)."""
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
# Magnitude (speed) UNet — loaded from file to avoid the `model` name clash
# ===========================================================================

def load_magnitude_model(checkpoint: str, device: str):
    """
    Load the Magnitude UNet speed regressor from its checkpoint.  Returns
    (model, speed_mean, speed_std) or None if the checkpoint does not exist.
    """
    if not checkpoint or not os.path.isfile(checkpoint):
        return None
    mag_model_path = os.path.join(_root, "Magnitude", "model.py")
    spec   = importlib.util.spec_from_file_location("mag_model", mag_model_path)
    mag    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mag)
    ckpt    = torch.load(checkpoint, map_location=device, weights_only=False)
    base_ch = ckpt.get("args", {}).get("base_ch", 64)
    net     = mag.MagnitudeUNet(in_ch=3, base_ch=base_ch).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, float(ckpt["speed_mean"]), float(ckpt["speed_std"])


@torch.no_grad()
def predict_magnitude(mag_net, speed_mean, speed_std,
                      spd_true, path_mask, land_mask, device):
    """Predict the dense speed field (H, W, phys units) from observed speeds only."""
    obs = np.zeros_like(spd_true)
    obs[path_mask] = spd_true[path_mask] / speed_std
    inp = np.stack([obs,
                    path_mask.astype(np.float32),
                    land_mask.astype(np.float32)], axis=0)[None]
    pred = mag_net(torch.from_numpy(inp).to(device))[0, 0].cpu().numpy()
    pred = np.clip(pred * speed_std + speed_mean, 0.0, None)
    pred[land_mask] = 0.0
    return pred.astype(np.float32)


# ===========================================================================
# Frame-capturing inference  (RePaint / PPR) — yields the denoising trajectory
# ===========================================================================

@torch.no_grad()
def infer_capture(model, diffusion, x0_known, path_mask, land_mask, args, device):
    """
    Run the chosen inference method and capture (t, x_t, x̂₀) along the reverse
    process.  Returns (final_pred_np, frames) where frames is a list of
    (t_int, xt_np, x0hat_np).  The final frame's x̂₀ == final_pred_np.
    """
    H, W = x0_known.shape[1:]
    x0_known_t = x0_known.unsqueeze(0).to(device)              # (1, 2, H, W)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t     = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_f    = 1.0 - land_t
    obs_mask   = torch.from_numpy(path_mask).to(device)
    ocean_mask = torch.from_numpy(~land_mask).to(device)

    if diffusion.noise_type == "gaussian":
        clamp = 3.0 * diffusion.noise_scale
    else:
        clamp = 3.0 * diffusion.noise_scale

    xt = diffusion._sample_noise(torch.empty(1, 2, H, W, device=device)) * diffusion.noise_scale
    xt = xt * ocean_f

    schedule = diffusion.build_inference_schedule(args.inference_steps)
    n_sched  = len(schedule)
    frames   = []

    def x0hat_from(xt_, t_):
        t_tensor = torch.full((1,), max(t_, 0), device=device, dtype=torch.long)
        eps_hat  = model(xt_, t_tensor)
        ab       = diffusion.alpha_bar[max(t_, 0)]
        return (xt_ - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt().clamp(min=1e-8)

    for step_i, (t_int, t_prev_int) in enumerate(schedule):
        last_x0hat = None
        for j in range(args.resample):
            if args.method == "repaint":
                # --- RePaint: hard-snap observations into the noisy field ---
                xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)
                t_prev_q   = max(t_prev_int, 0)
                t_prev_ten = torch.full((1,), t_prev_q, device=device, dtype=torch.long)
                xt_known, _ = diffusion.q_sample(x0_known_t, t_prev_ten)
                xt_merged  = known_t * xt_known + (1.0 - known_t) * xt_unknown
                xt_merged  = torch.nan_to_num(xt_merged * ocean_f, nan=0.0,
                                              posinf=clamp, neginf=-clamp)
                if j < args.resample - 1 and t_prev_int >= 0:
                    xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int) * ocean_f
                else:
                    xt = xt_merged
            elif args.method == "ppr":
                # --- PPR: project the clean Tweedie estimate, then renoise ---
                x0_hat = x0hat_from(xt, t_int).clamp(-clamp, clamp)
                if args.projector == "snap_x0":
                    x0_hat = x0_hat.clone()
                    x0_hat[:, :, obs_mask] = x0_known_t[:, :, obs_mask]
                else:
                    x0_hat = joint_project(x0_hat, ocean_mask, obs_mask,
                                           x0_known_t, n_iter=args.proj_iter,
                                           projector=args.projector)
                x0_hat = x0_hat * ocean_f
                last_x0hat = x0_hat
                if t_prev_int < 0:
                    xt = x0_hat
                else:
                    ab      = diffusion.alpha_bar[t_int]
                    ab_prev = diffusion.alpha_bar[t_prev_int]
                    beta_eff = 1.0 - ab / ab_prev
                    var      = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
                    coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
                    coef2 = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
                    xt = coef1 * x0_hat + coef2 * xt + var.sqrt() * diffusion._sample_noise(xt)
                xt = xt * ocean_f
                if j < args.resample - 1 and t_prev_int >= 0:
                    xt = diffusion.q_sample_from_prev(xt, t_int, t_prev_int) * ocean_f
            else:
                # --- DPS: Diffusion Posterior Sampling (Chung et al., ICLR 2023) ---
                # Unconditional ancestral step + gradient of the measurement
                # likelihood ‖obs − x̂₀‖² back-propagated through the network.
                # Softer than RePaint's hard snap: guides rather than overwrites.
                with torch.enable_grad():
                    xt_g     = xt.detach().requires_grad_(True)
                    t_tensor = torch.full((1,), t_int, device=device, dtype=torch.long)
                    eps_hat  = model(xt_g, t_tensor)
                    ab       = diffusion.alpha_bar[t_int]
                    x0_g     = (xt_g - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt().clamp(min=1e-8)
                    resid    = (x0_g - x0_known_t) * known_t * ocean_f
                    loss     = (resid ** 2).sum()
                    grad     = torch.autograd.grad(loss, xt_g)[0]
                x0_hat     = x0_g.detach().clamp(-clamp, clamp) * ocean_f
                last_x0hat = x0_hat
                if t_prev_int < 0:
                    xt = x0_hat
                else:
                    ab_prev  = diffusion.alpha_bar[t_prev_int]
                    beta_eff = 1.0 - ab / ab_prev
                    var      = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
                    coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
                    coef2 = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
                    mean  = coef1 * x0_hat + coef2 * xt.detach()
                    xt    = mean + var.sqrt() * diffusion._sample_noise(xt)
                    zeta  = args.dps_scale / (resid.detach().norm() + 1e-8)
                    xt    = xt - zeta * grad
                xt = (xt * ocean_f).detach()
                break  # DPS does not use RePaint-style resampling

        capture = (step_i == 0 or step_i == n_sched - 1
                   or step_i % args.capture_every == 0)
        if capture:
            if args.method == "repaint":
                x0hat_np = x0hat_from(xt, t_prev_int).clamp(-clamp, clamp)
                x0hat_np = (x0hat_np * ocean_f).squeeze(0).cpu().numpy()
            else:
                x0hat_np = last_x0hat.squeeze(0).cpu().numpy()
            frames.append((t_int, (xt * ocean_f).squeeze(0).cpu().numpy(), x0hat_np))

    # Final prediction: the model's clean estimate at the end of the chain.
    final_pred = frames[-1][2] if frames else (xt * ocean_f).squeeze(0).cpu().numpy()
    return final_pred, frames


# ===========================================================================
# Plot primitives  (no transpose — (H, W) with origin="lower")
# ===========================================================================

_LAND_BW   = mcolors.ListedColormap(["white", "black"])
_LAND_OVER = mcolors.ListedColormap(["none", "black"])
_PATH_RGBA = (0.84, 0.10, 0.11, 1.0)


def _speed_quiver(ax, u, v, land_np, path_np, title, step=2, vmax=None):
    """Arrows coloured by speed magnitude."""
    H, W = u.shape
    ax.imshow(land_np, origin="lower", cmap=_LAND_BW,
              extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0)
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    mq     = np.sqrt(uq ** 2 + vq ** 2)
    land_q = land_np[::step, ::step]
    mask   = (~np.isnan(uq)) & (~land_q) & (mq > 1e-9)
    clim   = vmax if vmax is not None else (np.nanpercentile(mq[mask], 98) if mask.any() else 1.0)
    q = ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
                  cmap="cool", clim=(0, clim), scale=12, width=0.003, zorder=2)
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    if path_np is not None:
        py, px = np.where(path_np)
        ax.scatter(px, py, s=5, c="red", marker="s", linewidths=0, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def _direction_quiver(ax, u_hat, v_hat, land_np, path_np, title, step=2):
    """Unit arrows coloured by compass direction (cyclic colormap)."""
    H, W = u_hat.shape
    ax.imshow(land_np, origin="lower", cmap=_LAND_BW,
              extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0)
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u_hat[::step, ::step], v_hat[::step, ::step]
    mq     = np.sqrt(uq ** 2 + vq ** 2)
    land_q = land_np[::step, ::step]
    mask   = (mq > 1e-6) & (~land_q)
    ang    = np.arctan2(vq, uq) % (2 * np.pi)
    q = ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask], ang[mask],
                  cmap="twilight", clim=(0, 2 * np.pi),
                  scale=30, width=0.004, pivot="mid", zorder=2)
    cb = plt.colorbar(q, ax=ax, label="Direction", shrink=0.7)
    cb.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cb.set_ticklabels(["E", "N", "W", "S", "E"])
    if path_np is not None:
        py, px = np.where(path_np)
        ax.scatter(px, py, s=5, c="red", marker="s", linewidths=0, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def _path_panel(ax, land_np, path_np, path_cells, seed):
    H, W = land_np.shape
    ext  = [-0.5, W - 0.5, -0.5, H - 0.5]
    ax.imshow(land_np, origin="lower", cmap=_LAND_BW, extent=ext, aspect="auto", zorder=0)
    rgba = np.zeros((H, W, 4))
    rgba[path_np] = _PATH_RGBA
    ax.imshow(rgba, origin="lower", extent=ext, aspect="auto",
              zorder=1, interpolation="nearest")
    ax.set_title(f"Robot path ({path_cells} cells, seed={seed})", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(handles=[
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
        mpatches.Patch(facecolor=_PATH_RGBA[:3], label="Path"),
        mpatches.Patch(facecolor="black", label="Land"),
    ], loc="upper right", fontsize=8)


# ===========================================================================
# Static summary figure
# ===========================================================================

def render_summary(sample_idx, seed, path_cells, args,
                   true_np, pred_np, fused_np,
                   land_np, path_mask, metrics):
    ocean_np = ~land_np
    vmax = float(np.nanpercentile(np.sqrt(true_np[0] ** 2 + true_np[1] ** 2)[ocean_np], 98))

    ut_hat, vt_hat, _ = unit_normalize(true_np, ocean_np)
    up_hat, vp_hat, _ = unit_normalize(pred_np, ocean_np)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    _speed_quiver(axes[0, 0], true_np[0], true_np[1], land_np, path_mask,
                  "1. Ground truth", vmax=vmax)
    _direction_quiver(axes[0, 1], ut_hat, vt_hat, land_np, None,
                      "2. Ground-truth direction")
    _path_panel(axes[0, 2], land_np, path_mask, path_cells, seed)
    _speed_quiver(axes[1, 0], pred_np[0], pred_np[1], land_np, path_mask,
                  f"4. DDPM prediction ({args.method.upper()})\n[magnitudes not meaningful]")
    _direction_quiver(axes[1, 1], up_hat, vp_hat, land_np, path_mask,
                      f"5. DDPM direction\nmean={metrics['mean_err']:.1f}°  cos={metrics['cos']:.3f}")

    if fused_np is not None:
        _speed_quiver(axes[1, 2], fused_np[0], fused_np[1], land_np, path_mask,
                      f"6. Fused: DDPM dir × UNet mag\nspeed RMSE={metrics['mag_rmse']:.4f}",
                      vmax=vmax)
    else:
        axes[1, 2].axis("off")
        axes[1, 2].text(0.5, 0.5,
                        "6. Fused field\n(magnitude model not\nprovided — pass\n--mag_checkpoint)",
                        ha="center", va="center", fontsize=13, color="gray")

    fig.suptitle(
        f"Direction × Magnitude  |  sample {sample_idx}  |  method={args.method}  |  "
        f"path={path_cells} cells", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, f"summary_{args.method}_val{sample_idx}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ===========================================================================
# Denoising GIF  (1×2: x_t direction | x̂₀ direction, both unit-normalized)
# ===========================================================================

def render_gif(sample_idx, args, frames, land_np):
    ocean_np = ~land_np
    n_sched  = len(frames)
    images   = []
    for k, (t_int, xt_np, x0hat_np) in enumerate(frames):
        uxt, vxt, _   = unit_normalize(xt_np, ocean_np)
        ux0, vx0, _   = unit_normalize(x0hat_np, ocean_np)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=80)
        _direction_quiver(axes[0], uxt, vxt, land_np, None,
                          f"Current field $x_t$ direction   (t={t_int})")
        _direction_quiver(axes[1], ux0, vx0, land_np, None,
                          r"Model $\hat{x}_0$ direction" + f"   (t={t_int})")
        pct = 100.0 * (k + 1) / n_sched
        fig.suptitle(f"Denoising — sample {sample_idx} — {args.method.upper()} — "
                     f"step {k + 1}/{n_sched} ({pct:.0f}%)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.load()
        images.append(img)

    # Hold the final frame a little longer.
    images.extend([images[-1]] * max(0, args.fps))
    out = os.path.join(args.out_dir, f"denoise_{args.method}_val{sample_idx}.gif")
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=int(1000 / args.fps), loop=0)
    return out


# ===========================================================================
# Args / main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Direction × Magnitude display: static summary + denoising GIF.")
    p.add_argument("--pickle",     default="Datasets/data.pickle")
    p.add_argument("--checkpoint", default="checkpoints_angle/best_ddpm_angle_div_free_cosine.pt")
    p.add_argument("--mag_checkpoint", default=None,
                   help="Magnitude UNet checkpoint (optional; enables fused panel).")
    p.add_argument("--method",     default="repaint", choices=["ppr", "repaint", "dps"])
    p.add_argument("--dps_scale",  type=float, default=1.0,
                   help="DPS guidance step size ζ' (used when --method dps).")
    p.add_argument("--n_samples",  type=int, default=5)
    p.add_argument("--n_ensemble", type=int, default=1,
                   help="Draw this many posterior samples per field and average "
                        "them (the posterior-mean estimator). >1 also reports a "
                        "per-cell directional-uncertainty map. Default 1 (off).")
    p.add_argument("--random",     action="store_true")
    p.add_argument("--seed",       type=int, default=1234)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--inference_steps", type=int, default=100)
    p.add_argument("--resample",   type=int, default=10)
    p.add_argument("--proj_iter",  type=int, default=20)
    p.add_argument("--projector",  default="snap_x0", choices=["pocs", "snap_x0"])
    p.add_argument("--capture_every", type=int, default=5,
                   help="Capture a GIF frame every N reverse steps.")
    p.add_argument("--fps",        type=int, default=8)
    p.add_argument("--step",       type=int, default=2, help="quiver subsample step")
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--no_gif",     action="store_true", help="Skip GIF rendering.")
    p.add_argument("--device",     default=None)
    p.add_argument("--out_dir",    default="DDPM/best_model_results/dir_mag")
    return p.parse_args()


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args   = parse_args()
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device : {device}")

    # ---- Angle (direction) DDPM ----
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)
    noise_type      = ckpt_args.get("noise_type", "gaussian")
    spectral_filter = ckpt.get("spectral_filter", None)
    data_mean = ckpt.get("data_mean", None)
    data_std  = ckpt.get("data_std", None)

    # Load val data in the SAME normalization the model was trained with, so the
    # network sees inputs at the scale it expects.  Critical for std-only models
    # (data_mean=0, data_std<1): feeding raw-scale data would recreate the SNR
    # mismatch.  Un-normalized models store data_mean=None -> no normalization.
    val_ds  = OceanCurrentDataset(args.pickle, split=1,
                                  data_mean=data_mean, data_std=data_std)
    land_np = val_ds.land_mask.numpy().astype(bool)
    if data_mean is not None and data_std is not None:
        print(f"Normalization : data_mean={data_mean}  data_std={data_std}")
    else:
        print("Normalization : none (raw scale)")

    diffusion = DDPM(T=T, beta_schedule="cosine", device=device,
                     noise_type=noise_type, spectral_filter=spectral_filter)

    pred_type = ckpt.get("pred_type", "eps")
    if pred_type == "x0_streamfn":
        # Divergence-free stream-function model: predicts x̂₀ directly via the
        # curl of a scalar stream function.  Wrap it in EpsFromStreamFn so every
        # downstream sampler (RePaint / PPR / DPS) — all of which expect an
        # eps-predicting network — works unchanged while the recovered x̂₀ stays
        # divergence-free by construction.
        stream_model = StreamFunctionUNet(
            in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
        stream_model.load_state_dict(ckpt["model"])
        stream_model.eval()
        model = EpsFromStreamFn(stream_model, diffusion).to(device)
        model.eval()
        print(f"Stream-fn model : epoch {ckpt.get('epoch', '?')}  T={T}  "
              f"noise={noise_type}  (divergence-free x0)")
    else:
        model = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Angle model : epoch {ckpt.get('epoch', '?')}  T={T}  noise={noise_type}")

    # ---- Magnitude UNet (optional) ----
    mag = load_magnitude_model(args.mag_checkpoint, device)
    if mag is None:
        print("Magnitude model: none (fused panel disabled)")
    else:
        print(f"Magnitude model: loaded ({args.mag_checkpoint})")

    # ---- Sample indices ----
    if args.random:
        rng  = np.random.default_rng(args.seed)
        idxs = rng.choice(len(val_ds), size=min(args.n_samples, len(val_ds)),
                          replace=False).tolist()
    else:
        idxs = [i % len(val_ds) for i in range(args.n_samples)]
    print(f"Val samples : {idxs}\n")

    angle_errs, cos_sims, mag_rmses = [], [], []
    all_err, all_dist = [], []
    member_errs, spread_near, spread_far = [], [], []
    for i, sidx in enumerate(idxs):
        seed      = i * 7 + 1
        x0_true   = val_ds[sidx]
        path_mask = biased_walk_path(land_np, n_steps=args.path_steps, seed=seed)
        path_mask &= ~land_np
        x0_obs    = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        final_pred, frames, members = ensemble_infer(
            model, diffusion, x0_obs, path_mask, land_np, args, device,
            base_seed=seed)

        true_np = x0_true.numpy()
        pred_np = final_pred
        ocean_np = ~land_np

        # Back to physical units for metrics/plots.  Angle is unchanged by this
        # affine inverse (mean=0 for std-only -> pure rescale, no rotation); for
        # un-normalized models data_mean is None and this is a no-op.
        if data_mean is not None and data_std is not None:
            true_np = true_np * float(data_std) + float(data_mean)
            pred_np = pred_np * float(data_std) + float(data_mean)

        err_map  = angle_error_deg(pred_np, true_np, ocean_np)
        valid    = ~np.isnan(err_map)
        mean_err = float(np.nanmean(err_map))
        cos      = float(np.mean(np.cos(np.radians(err_map[valid])))) if valid.any() else float("nan")
        metrics  = {"mean_err": mean_err, "cos": cos, "mag_rmse": float("nan")}

        # ---- Option D: stratify error by distance-to-path ----
        dist_map  = distance_to_path(path_mask, ocean_np)
        all_err.append(err_map[valid])
        all_dist.append(dist_map[valid])
        near_sel  = valid & (dist_map <= 2.0)
        near_mean = float(np.mean(err_map[near_sel])) if near_sel.any() else float("nan")

        # ---- Ensemble: per-member error + directional-uncertainty map ----
        ens_str = ""
        if args.n_ensemble > 1:
            def _unnorm(a):
                if data_mean is not None and data_std is not None:
                    return a * float(data_std) + float(data_mean)
                return a
            m_errs = [float(np.nanmean(angle_error_deg(_unnorm(m), true_np, ocean_np)))
                      for m in members]
            memb_mean = float(np.mean(m_errs))
            member_errs.append(memb_mean)
            spread_map = directional_spread([_unnorm(m) for m in members], ocean_np)
            far_sel = valid & (dist_map > 10.0)
            sn = float(np.nanmean(spread_map[near_sel])) if near_sel.any() else float("nan")
            sf = float(np.nanmean(spread_map[far_sel]))  if far_sel.any()  else float("nan")
            spread_near.append(sn); spread_far.append(sf)
            ens_str = (f"  [ens{args.n_ensemble}: members={memb_mean:.1f}° "
                       f"mean={mean_err:.1f}° | spread near={sn:.2f} far={sf:.2f}]")

        # ---- Fuse: DDPM direction × UNet magnitude ----
        fused_np = None
        if mag is not None:
            mag_net, smean, sstd = mag
            spd_true = np.sqrt(true_np[0] ** 2 + true_np[1] ** 2).astype(np.float32)
            spd_true[land_np] = 0.0
            spd_pred = predict_magnitude(mag_net, smean, sstd,
                                         spd_true, path_mask, land_np, device)
            up_hat, vp_hat, _ = unit_normalize(pred_np, ocean_np)
            fused_np = np.stack([up_hat * spd_pred, vp_hat * spd_pred], axis=0)
            ocean_err = (spd_pred - spd_true)[ocean_np]
            metrics["mag_rmse"] = float(np.sqrt(np.mean(ocean_err ** 2)))
            mag_rmses.append(metrics["mag_rmse"])

        path_cells = int(path_mask.sum())
        summary_path = render_summary(sidx, seed, path_cells, args,
                                      true_np, pred_np, fused_np,
                                      land_np, path_mask, metrics)
        gif_path = None
        if not args.no_gif:
            gif_path = render_gif(sidx, args, frames, land_np)

        angle_errs.append(mean_err); cos_sims.append(cos)
        extra = f"  magRMSE={metrics['mag_rmse']:.4f}" if mag is not None else ""
        print(f"[{i+1}/{len(idxs)}] sample {sidx}: mean_err={mean_err:.1f}°  "
              f"near(≤2)={near_mean:.1f}°  cos={cos:.3f}{extra}{ens_str}")
        print(f"         summary: {summary_path}")
        if gif_path:
            print(f"         gif    : {gif_path}")

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY ({len(idxs)} samples, method={args.method})")
    print(f"{'=' * 60}")
    print(f"  Mean angular error : {np.mean(angle_errs):.1f}°  (± {np.std(angle_errs):.1f})")
    print(f"  Mean cosine sim    : {np.mean(cos_sims):.3f}")
    if mag_rmses:
        print(f"  Mean magnitude RMSE: {np.mean(mag_rmses):.4f}")
    if args.n_ensemble > 1 and member_errs:
        print(f"  Ensemble ({args.n_ensemble} draws):")
        print(f"    Single-member error : {np.mean(member_errs):.1f}°  "
              f"(avg over members)")
        print(f"    Posterior-mean error: {np.mean(angle_errs):.1f}°  "
              f"(Δ {np.mean(member_errs) - np.mean(angle_errs):+.1f}° vs members)")
        print(f"    Directional spread  : near(≤2)={np.nanmean(spread_near):.2f}  "
              f"far(>10)={np.nanmean(spread_far):.2f}   (0=agree, 1=scattered)")

    # ---- Option D: aggregate stratified table over all cells of all samples ----
    if all_err:
        ev = np.concatenate(all_err); dv = np.concatenate(all_dist)
        print(f"\n  Stratified by distance-to-path (all {ev.size} scored cells):")
        print(f"    {'band':<14}{'cells':>8}{'mean err':>11}{'cos':>9}")
        for label, n, me, c in stratified_rows(ev, dv):
            if n:
                print(f"    {label:<14}{n:>8}{me:>10.1f}°{c:>9.3f}")
            else:
                print(f"    {label:<14}{n:>8}{'n/a':>11}{'n/a':>9}")

        # Coverage-fair headline: exclude observed cells (dist<=2) so the
        # overall number is not skewed downward by simply observing more cells.
        unobs = dv > 2.0
        if unobs.any():
            uo = ev[unobs]
            print(f"\n  Unobserved-only (dist>2): {np.mean(uo):.1f}°  "
                  f"cos={np.mean(np.cos(np.radians(uo))):.3f}  "
                  f"({uo.size} cells, excludes near band)")

    print(f"\n  Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
