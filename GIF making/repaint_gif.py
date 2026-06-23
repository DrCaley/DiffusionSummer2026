"""
Visualise the RePaint denoising process as a GIF.

2x2 panel layout:
  Top-left     : Ground truth            (static)
  Top-right    : Noisy field xt          (shows denoising progress)
  Bottom-left  : Robot-path observation  (static)
  Bottom-right : Model x0-hat estimate   (model's current clean-field guess)

Supports all noise schedules trained in NoiseSchedule/:
    linear, cosine, cosine_s02, cosine_s05, cosine_s10,
    quadratic, sigmoid, geometric

Usage (run from workspace root):
    python3 NoiseSchedule/repaint_gif.py --schedule cosine_s10
    python3 NoiseSchedule/repaint_gif.py --schedule linear --fps 12 --capture_every 20
    # Run all schedules:
    for s in linear cosine cosine_s02 cosine_s05 cosine_s10 quadratic sigmoid geometric; do
        python3 NoiseSchedule/repaint_gif.py --schedule $s
    done
"""

import argparse
import os
import sys
from io import BytesIO

_here = os.path.dirname(os.path.abspath(__file__))
# Locate the repo root by walking up until we find a directory that contains
# both utils/ (dataset.py, paths.py) and DDPM/.  Works whether this script sits
# in GIF making/ (local layout) or at the flat repo root (server layout).
_repo_root = _here
for _up in range(4):
    _cand = os.path.normpath(os.path.join(_here, *(['..'] * _up)))
    if os.path.isdir(os.path.join(_cand, "utils")) and os.path.isdir(os.path.join(_cand, "DDPM")):
        _repo_root = _cand
        break

# Candidate module directories for both local and server layouts.
for _p in (
    os.path.join(_repo_root, "utils"),                              # dataset, paths (local)
    _repo_root,                                                     # dataset, paths (server flat)
    os.path.join(_repo_root, "Model Parameters", "NoiseSchedule"),  # repaint_model
    os.path.join(_repo_root, "DDPM", "model"),                      # diffusion (local)
    os.path.join(_repo_root, "DDPM"),                              # diffusion (server)
):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from paths import biased_walk_path
from divfree_projection import joint_project, divergence as compute_divergence

ALL_SCHEDULES = [
    "linear", "cosine", "cosine_s0001", "cosine_s02", "cosine_s10",
    "quadratic", "sigmoid", "geometric",
]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render RePaint denoising as a GIF for any noise schedule."
    )
    p.add_argument("--pickle",        default="data.pickle")
    p.add_argument("--schedule",      default="cosine", choices=ALL_SCHEDULES)
    p.add_argument("--checkpoint",    default=None,
                   help="Path to checkpoint. "
                        "Defaults to checkpoints_repaint_{schedule}/best_model_{schedule}.pt")
    p.add_argument("--sample",        type=int, default=0,
                   help="Dataset index to visualise.")
    p.add_argument("--split",         type=int, default=2,
                   help="Dataset split: 0=train, 1=val, 2=test (default: 2)")
    p.add_argument("--path_steps",    type=int, default=150)
    p.add_argument("--resample",      type=int, default=10,
                   help="RePaint r parameter")
    p.add_argument("--inference_steps", type=int, default=1000,
                   help="Denoising steps (must divide T). 1000 = full schedule.")
    p.add_argument("--final_project", type=int, default=0,
                   help="POCS iters applied ONCE to the final prediction (0=off). "
                        "Cleans divergence post-hoc with no per-step energy drift; "
                        "appends one cleaned frame to the end of the GIF.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--capture_every", type=int, default=20,
                   help="Capture a frame every N reverse timesteps "
                        "(default 20 -> ~52 frames for T=1000).")
    p.add_argument("--fps",           type=int, default=10,
                   help="Playback speed of the output GIF.")
    p.add_argument("--T",             type=int, default=1000)
    p.add_argument("--base_ch",       type=int, default=64)
    p.add_argument("--time_dim",      type=int, default=256)
    p.add_argument("--out",           default=None,
                   help="Output .gif path. "
                        "Defaults to model_{schedule}_results/repaint_process.gif")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper  (identical to batch_repaint.py)
# ---------------------------------------------------------------------------

def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool", vmax=None):
    H, W = u.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u[::step, ::step]
    vq = v[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    mask = ~np.isnan(uq) & ~land_mask[::step, ::step]
    clim_max = vmax if vmax is not None else (np.nanpercentile(mq[mask], 98) if mask.any() else 1)
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
        cmap=cmap, clim=(0, clim_max),
        scale=12, width=0.003, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


# ---------------------------------------------------------------------------
# Modified RePaint loop that yields intermediate frames
# ---------------------------------------------------------------------------

@torch.no_grad()
def repaint_frames(
    model, diffusion,
    x0_known, path_mask, land_mask,
    r=10, device="cpu", capture_every=20, inference_steps=None,
):
    """
    Identical control flow to repaint_infer.repaint(), but yields
    (t_int, xt_np, x0hat_np) at captured timesteps so the caller can
    build animation frames without a second forward pass.

    x0hat is estimated as: x0_hat = (xt - sqrt(1 - abar_t) * eps_pred) / sqrt(abar_t)
    """
    H, W = x0_known.shape[1:]
    x0_known = x0_known.unsqueeze(0).to(device)   # (1, 2, H, W)

    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t  = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t = 1.0 - land_t

    clamp_val = 3.0 * diffusion.noise_scale
    xt = diffusion._sample_noise(torch.empty(1, 2, H, W, device=device)) * diffusion.noise_scale
    xt = torch.clamp(xt, -clamp_val, clamp_val) * ocean_t

    def get_x0hat(xt_, t_):
        t_tensor   = torch.full((1,), t_, device=device, dtype=torch.long)
        pred_noise = model(xt_, t_tensor)
        ab         = diffusion.alpha_bar[max(t_, 0)]
        x0hat      = (xt_ - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        return x0hat.clamp(-clamp_val, clamp_val).squeeze(0).cpu().numpy()

    n_steps  = inference_steps if inference_steps is not None else diffusion.T
    schedule = diffusion.build_inference_schedule(n_steps)   # [(t, t_prev), ...]
    n_sched  = len(schedule)

    for step_i, (t_int, t_prev_int) in enumerate(schedule):
        for j in range(r):
            # --- Step 1: model reverse step for unknown pixels (proper posterior) ---
            xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)

            # --- Step 2: forward-diffuse x0_known to timestep t_prev (or 0) ---
            t_prev_q      = max(t_prev_int, 0)
            t_prev_tensor = torch.full((1,), t_prev_q, device=device, dtype=torch.long)
            xt_known, _   = diffusion.q_sample(x0_known, t_prev_tensor)

            # --- Step 3: merge known / unknown ---
            xt_merged = known_t * xt_known + (1.0 - known_t) * xt_unknown
            xt_merged = xt_merged * ocean_t
            xt_merged = torch.nan_to_num(xt_merged, nan=0.0, posinf=1.0, neginf=-1.0)

            # --- Step 4: resample (go forward one step) if not last iteration ---
            if j < r - 1 and t_prev_int >= 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int) * ocean_t
            else:
                xt = xt_merged

        # Capture by schedule position so it works for any (subsampled) schedule:
        # always the first and final step, plus every `capture_every` steps.
        if step_i == 0 or step_i == n_sched - 1 or step_i % capture_every == 0:
            yield t_int, xt.squeeze(0).cpu().numpy(), get_x0hat(xt, t_int)


# ---------------------------------------------------------------------------
# Render one 2x2 figure -> PIL Image
# ---------------------------------------------------------------------------

def make_frame(t_int, T, schedule, idx,
               u_true, v_true,
               u_xt, v_xt,
               u_x0hat, v_x0hat,
               path_mask_d, land_mask_d, path_cells, seed, vmax=None):
    """
    All u/v arrays and masks are already transposed (.T) for display.
    Layout:
      [0] Ground truth  |  [1] Noisy field xt   (current denoising state)
      [2] Input path    |  [3] Model x0-hat      (current clean prediction)
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=80)
    axes = axes.flatten()

    # 0. Ground truth
    plot_field(axes[0], u_true, v_true, land_mask_d, "Ground Truth", vmax=vmax)

    # 1. Noisy field xt
    plot_field(axes[1], u_xt, v_xt, land_mask_d,
               f"Noisy field  $x_t$   (t = {t_int})", vmax=vmax)

    # 2. Robot path input  (same style as batch_repaint.py)
    axes[2].imshow(
        land_mask_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_mask_d.shape[1] - 0.5,
                -0.5, land_mask_d.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    # Use a solid RGBA overlay so the legend colour exactly matches the display
    PATH_COLOR = (0.84, 0.10, 0.11, 1.0)   # consistent red across all frames
    path_rgba = np.zeros((*land_mask_d.shape, 4), dtype=float)
    path_rgba[path_mask_d] = PATH_COLOR
    axes[2].imshow(
        path_rgba, origin="lower",
        extent=[-0.5, land_mask_d.shape[1] - 0.5,
                -0.5, land_mask_d.shape[0] - 0.5],
        aspect="auto", zorder=1, interpolation="nearest",
    )
    axes[2].set_title(f"Input — Robot Path ({path_cells} cells, seed={seed})", fontsize=11)
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    ocean_p = mpatches.Patch(facecolor="white",                    edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor=PATH_COLOR[:3],                               label="Path")
    land_p  = mpatches.Patch(facecolor="black",                                      label="Land")
    axes[2].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

    # 3. Model x0-hat estimate
    plot_field(axes[3], u_x0hat, v_x0hat, land_mask_d,
               r"Model $\hat{x}_0$ estimate" + f"   (t = {t_int})", vmax=vmax)

    pct = 100.0 * (T - t_int) / T
    plt.suptitle(
        f"RePaint Denoising  —  schedule={schedule}  —  test sample {idx}"
        f"   |   step {T - t_int}/{T}  ({pct:.0f}%)",
        fontsize=13,
    )
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.load()   # force decode before buf is GC'd
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            script_dir, "DDPM", "models",
            "model_ddpm_eps_gaussian_cosine_ns0p12.pt",
        )

    out_dir = os.path.join(script_dir, f"model_{args.schedule}_results")
    if args.out is None:
        args.out = os.path.join(out_dir, "repaint_process.gif")
    os.makedirs(out_dir, exist_ok=True)

    n_frames_approx = 1000 // args.capture_every + 2
    print(f"Device        : {device}")
    print(f"Schedule      : {args.schedule}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Output        : {args.out}")
    print(f"capture_every : {args.capture_every}  ->  ~{n_frames_approx} frames")
    print(f"FPS           : {args.fps}")

    # ---- Model ---------------------------------------------------------------
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)
    schedule  = ckpt_args.get("schedule", args.schedule)

    # Normalization metadata (colored model trained with --normalize)
    data_mean = ckpt.get("data_mean", None)
    data_std  = ckpt.get("data_std",  None)

    # ---- Data ----------------------------------------------------------------
    test_ds      = OceanCurrentDataset(args.pickle, split=args.split,
                                       data_mean=data_mean, data_std=data_std)
    land_mask_np = test_ds.land_mask.numpy()
    if data_mean is not None:
        print(f"Normalized    : mean={data_mean:.5f}  std={data_std:.5f}")

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded        : epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}")

    noise_std = ckpt_args.get("noise_scale", ckpt.get("noise_std", None))
    if noise_std is None:
        train_ds  = OceanCurrentDataset(args.pickle, split=0,
                                        data_mean=data_mean, data_std=data_std)
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())
        print(f"noise_std     : {noise_std:.5f}  (computed from training data)")
    else:
        print(f"noise_std     : {noise_std:.5f}  (from checkpoint)")

    # Divergence-free colored noise (matches the new pipeline) when the
    # checkpoint was trained that way; otherwise fall back to gaussian.
    noise_type      = ckpt_args.get("noise_type", "gaussian")
    spectral_filter = ckpt.get("spectral_filter", None)
    print(f"noise_type    : {noise_type}")

    diffusion = DDPM(T=T, beta_schedule=schedule, device=device,
                     noise_type=noise_type, spectral_filter=spectral_filter,
                     noise_scale=noise_std)

    # ---- Sample & path -------------------------------------------------------
    idx       = args.sample % len(test_ds)
    x0_true   = test_ds[idx]
    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=args.seed)

    x0_obs = x0_true.clone()
    x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

    true_np = x0_true.numpy()
    n_ocean = int((~test_ds.land_mask).sum().item())

    # ---- Un-normalize for display (physical units) --------------------------
    # The colored model trains on normalized data (mean/std stored in the
    # checkpoint), so its fields live in normalized units (~10x physical). We
    # convert back to physical units for display so vector magnitudes match the
    # un-normalized (old-pipeline) GIFs instead of looking "insanely large".
    # Old/unnormalized checkpoints (data_mean is None) are displayed as-is.
    def _denorm(arr):
        if data_mean is None:
            return arr
        out = arr * data_std + data_mean
        out[:, land_mask_np] = 0.0   # re-zero land (mean offset must not leak in)
        return out

    true_np = _denorm(true_np)

    # Fixed colour scale from ground truth — keeps all frames comparable
    true_speed = np.sqrt(true_np[0] ** 2 + true_np[1] ** 2)
    true_speed[land_mask_np] = np.nan
    vmax = float(np.nanpercentile(true_speed, 98)) or 1.0

    # Pre-transpose for display (matches batch_repaint.py convention)
    u_true_d    = true_np[0].T
    v_true_d    = true_np[1].T
    land_mask_d = land_mask_np.T
    path_mask_d = path_mask.T

    print(f"\nTest sample   : {idx}  |  path covers "
          f"{path_mask.sum()} / {n_ocean} ocean cells "
          f"({100 * path_mask.sum() / n_ocean:.1f}%)")
    print("Running RePaint inference and collecting frames ...")

    # ---- Inference + frame collection ----------------------------------------
    pil_frames = []
    final_pred_np = None
    final_raw_np  = None

    for t_int, xt_np, x0hat_np in repaint_frames(
        model, diffusion, x0_obs, path_mask, land_mask_np,
        r=args.resample, device=device, capture_every=args.capture_every,
        inference_steps=args.inference_steps,
    ):
        final_raw_np = xt_np   # raw (normalized) field, before display denorm
        xt_np    = _denorm(xt_np)
        x0hat_np = _denorm(x0hat_np)
        pil_frames.append(make_frame(
            t_int, T, schedule, idx,
            u_true_d,      v_true_d,
            xt_np[0].T,    xt_np[1].T,
            x0hat_np[0].T, x0hat_np[1].T,
            path_mask_d, land_mask_d, path_mask.sum(), args.seed, vmax=vmax,
        ))
        final_pred_np = xt_np   # last yielded (t=0) is the final prediction

        if len(pil_frames) % 10 == 0 or t_int == 0:
            print(f"  t={t_int:4d}  frame {len(pil_frames)}")

    # ---- Optional post-hoc divergence-free projection ------------------------
    # One joint POCS projection on the final field cleans the RePaint seam
    # (|div| -> ~0) without the per-step energy drift of in-loop PPR.
    ocean = ~land_mask_np
    ocean_mask_t = torch.from_numpy(ocean)
    if args.final_project > 0 and final_raw_np is not None:
        div_before = float(compute_divergence(
            torch.from_numpy(_denorm(final_raw_np)).unsqueeze(0).float(),
            ocean_mask_t)[0][ocean_mask_t].abs().mean())
        x_proj = joint_project(
            torch.from_numpy(final_raw_np).unsqueeze(0).float(),
            ocean_mask_t, torch.from_numpy(path_mask),
            x0_obs.unsqueeze(0).float(), n_iter=args.final_project,
        ).squeeze(0).numpy()
        proj_disp = _denorm(x_proj)
        div_after = float(compute_divergence(
            torch.from_numpy(proj_disp).unsqueeze(0).float(),
            ocean_mask_t)[0][ocean_mask_t].abs().mean())
        # Append one final cleaned frame (projected field in both xt + x0-hat panels)
        pil_frames.append(make_frame(
            0, T, schedule, idx,
            u_true_d,         v_true_d,
            proj_disp[0].T,   proj_disp[1].T,
            proj_disp[0].T,   proj_disp[1].T,
            path_mask_d, land_mask_d, path_mask.sum(), args.seed, vmax=vmax,
        ))
        final_pred_np = proj_disp
        print(f"Final projection ({args.final_project} POCS iters): "
              f"|div| {div_before:.6f} -> {div_after:.6f}")

    # ---- RMSE over ocean cells (both channels) --------------------------------
    diff  = final_pred_np[:, ocean] - true_np[:, ocean]
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    print(f"\nRMSE          : {rmse:.6f}  (sample {idx}, {ocean.sum()} ocean cells)")

    # Frames are already in playback order: t=T-1 (pure noise) -> t=0 (clean)
    print(f"Total frames  : {len(pil_frames)}")

    # ---- Save GIF ------------------------------------------------------------
    print("Saving GIF ...")
    duration_ms = max(1, int(1000 / args.fps))
    # Pause on the final frame for 1 second
    durations = [duration_ms] * (len(pil_frames) - 1) + [1000]
    pil_frames[0].save(
        args.out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
