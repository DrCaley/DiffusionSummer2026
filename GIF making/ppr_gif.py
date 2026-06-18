"""
Visualise the Predict-Project-Renoise (PPR) denoising process as a GIF.

Mirror of repaint_gif.py but for the PPR sampler (DDPM/testing/ppr/ppr_infer.py).
Same 2x2 panel layout, same physical-unit display, same step-skipping (capture_every):

  Top-left     : Ground truth                (static)
  Top-right    : Noisy field xt              (shows denoising progress)
  Bottom-left  : Robot-path observation      (static)
  Bottom-right : Projected x0-hat estimate   (model's current clean field,
                                              after the PPR data-consistency
                                              projection / obs snap)

Fields are displayed in *physical* units: if the checkpoint stores
data_mean/data_std (colored model trained with --normalize), the normalized
fields are converted back to physical units so the vector magnitudes match the
un-normalized (old-pipeline) GIFs.

Usage (run from workspace root):
    python "GIF making/ppr_gif.py" \
        --checkpoint Models/Div_Free_DDPM_Colored.pt \
        --pickle Datasets/data_divfree.pickle \
        --projector snap_x0 --out "GIF making/inference_gifs/ppr_colored_sample0.gif"
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
    os.path.join(_repo_root, "utils"),                  # dataset, paths, plot_utils
    _repo_root,                                         # dataset, paths (server flat)
    os.path.join(_repo_root, "DDPM", "model"),          # diffusion, model, divfree_projection
    os.path.join(_repo_root, "DDPM"),                  # diffusion (server)
    os.path.join(_repo_root, "DDPM", "testing", "ppr"), # ppr_infer
):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset            import OceanCurrentDataset
from diffusion          import DDPM
from model              import UNet
from divfree_projection import joint_project
from paths              import biased_walk_path

ALL_SCHEDULES = [
    "linear", "cosine", "cosine_s0001", "cosine_s02", "cosine_s10",
    "quadratic", "sigmoid", "geometric",
]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render PPR denoising as a GIF (physical-unit display)."
    )
    p.add_argument("--pickle",        default="data.pickle")
    p.add_argument("--schedule",      default="cosine", choices=ALL_SCHEDULES)
    p.add_argument("--checkpoint",    default=None)
    p.add_argument("--sample",        type=int, default=0)
    p.add_argument("--split",         type=int, default=2,
                   help="Dataset split: 0=train, 1=val, 2=test (default: 2)")
    p.add_argument("--path_steps",    type=int, default=150)
    p.add_argument("--resample",      type=int, default=1,
                   help="PPR resampling iterations per timestep (r). "
                        "PPR reaches consistency via projection, so r=1 usually "
                        "suffices and is fast.")
    p.add_argument("--proj_iter",     type=int, default=20,
                   help="POCS iterations inside joint_project (pocs projector only).")
    p.add_argument("--projector",     default="snap_x0", choices=["pocs", "snap_x0"],
                   help="Data-consistency mode: 'snap_x0' (projection-free obs "
                        "snap on x0-hat, relies on the model's div-free prior) "
                        "or 'pocs' (joint div-free + obs POCS projection).")
    p.add_argument("--inference_steps", type=int, default=1000,
                   help="Denoising steps (must divide T). 1000 = full schedule.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--capture_every", type=int, default=25,
                   help="Capture a frame every N reverse timesteps.")
    p.add_argument("--fps",           type=int, default=10)
    p.add_argument("--T",             type=int, default=1000)
    p.add_argument("--base_ch",       type=int, default=64)
    p.add_argument("--time_dim",      type=int, default=256)
    p.add_argument("--out",           default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper  (identical to repaint_gif.py)
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
# PPR loop that yields intermediate frames
# ---------------------------------------------------------------------------

@torch.no_grad()
def ppr_frames(
    model, diffusion,
    x0_known, path_mask, land_mask,
    r=1, proj_iter=20, projector="snap_x0",
    inference_steps=None, data_mean=None, data_std=None,
    device="cpu", capture_every=25,
):
    """
    Identical control flow to ppr_infer.ppr(), but yields
    (t_int, xt_np, x0hat_np) at captured timesteps so the caller can build
    animation frames without a second pass.  x0hat is the *projected* clean
    estimate (after the PPR obs snap / joint projection).
    """
    H, W = x0_known.shape[1:]

    # Tweedie clamp bounds (normalized-space if the model was trained normalized)
    if data_mean is not None and data_std is not None:
        clamp_lo = (-1.0 - data_mean) / data_std
        clamp_hi = ( 1.0 - data_mean) / data_std
    else:
        clamp_lo, clamp_hi = -1.0, 1.0

    x0_known_t = x0_known.unsqueeze(0).to(device)          # (1, 2, H, W)
    obs_mask   = torch.from_numpy(path_mask).to(device)    # (H, W) bool
    ocean_mask = torch.from_numpy(~land_mask).to(device)   # (H, W) bool
    ocean_f    = ocean_mask.float()[None, None]            # (1, 1, H, W)

    xt = diffusion._sample_noise(torch.empty(1, 2, H, W, device=device))
    xt = xt * ocean_f

    n_steps  = inference_steps if inference_steps is not None else diffusion.T
    schedule = diffusion.build_inference_schedule(n_steps)
    n_sched  = len(schedule)

    for step_i, (t_int, t_prev_int) in enumerate(schedule):
        x0_hat = None
        for j in range(r):
            t_tensor = torch.full((1,), t_int, device=device, dtype=torch.long)
            eps_hat  = model(xt, t_tensor)

            ab     = diffusion.alpha_bar[t_int]
            x0_hat = (xt - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt().clamp(min=1e-8)
            x0_hat = x0_hat.clamp(clamp_lo, clamp_hi)

            if projector == "snap_x0":
                x0_hat = x0_hat.clone()
                x0_hat[:, :, obs_mask] = x0_known_t[:, :, obs_mask]
            else:
                x0_hat = joint_project(
                    x0_hat, ocean_mask, obs_mask, x0_known_t,
                    n_iter=proj_iter, projector=projector,
                )
            x0_hat = x0_hat * ocean_f

            if t_prev_int < 0:
                xt = x0_hat
            else:
                ab_prev  = diffusion.alpha_bar[t_prev_int]
                beta_eff = 1.0 - ab / ab_prev
                var      = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
                coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
                coef2 = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
                mean  = coef1 * x0_hat + coef2 * xt
                xt = mean + var.sqrt() * diffusion._sample_noise(xt)

            xt = xt * ocean_f

            if j < r - 1 and t_prev_int >= 0:
                xt = diffusion.q_sample_from_prev(xt, t_int, t_prev_int)
                xt = xt * ocean_f

        # Capture by schedule position so it works for any (subsampled) schedule:
        # always the first and final step, plus every `capture_every` steps.
        if step_i == 0 or step_i == n_sched - 1 or step_i % capture_every == 0:
            yield t_int, xt.squeeze(0).cpu().numpy(), x0_hat.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Render one 2x2 figure -> PIL Image
# ---------------------------------------------------------------------------

def make_frame(t_int, T, schedule, idx,
               u_true, v_true,
               u_xt, v_xt,
               u_x0hat, v_x0hat,
               path_mask_d, land_mask_d, path_cells, seed, vmax=None):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=80)
    axes = axes.flatten()

    plot_field(axes[0], u_true, v_true, land_mask_d, "Ground Truth", vmax=vmax)
    plot_field(axes[1], u_xt, v_xt, land_mask_d,
               f"Noisy field  $x_t$   (t = {t_int})", vmax=vmax)

    axes[2].imshow(
        land_mask_d, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_mask_d.shape[1] - 0.5,
                -0.5, land_mask_d.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    PATH_COLOR = (0.84, 0.10, 0.11, 1.0)
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
    ocean_p = mpatches.Patch(facecolor="white",       edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor=PATH_COLOR[:3],                  label="Path")
    land_p  = mpatches.Patch(facecolor="black",                        label="Land")
    axes[2].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

    plot_field(axes[3], u_x0hat, v_x0hat, land_mask_d,
               r"Projected $\hat{x}_0$ estimate" + f"   (t = {t_int})", vmax=vmax)

    pct = 100.0 * (T - t_int) / T
    plt.suptitle(
        f"PPR Denoising  —  schedule={schedule}  —  test sample {idx}"
        f"   |   step {T - t_int}/{T}  ({pct:.0f}%)",
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
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = os.path.join(_here, "inference_gifs")
    if args.out is None:
        args.out = os.path.join(out_dir, f"ppr_sample{args.sample}.gif")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_frames_approx = args.inference_steps // args.capture_every + 2
    print(f"Device        : {device}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Projector     : {args.projector}")
    print(f"Output        : {args.out}")
    print(f"capture_every : {args.capture_every}  ->  ~{n_frames_approx} frames")

    # ---- Model ---------------------------------------------------------------
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)
    schedule  = ckpt_args.get("schedule", args.schedule)

    data_mean = ckpt.get("data_mean", None)
    data_std  = ckpt.get("data_std",  None)

    test_ds      = OceanCurrentDataset(args.pickle, split=args.split,
                                       data_mean=data_mean, data_std=data_std)
    land_mask_np = test_ds.land_mask.numpy()
    if data_mean is not None:
        print(f"Normalized    : mean={data_mean:.5f}  std={data_std:.5f}")

    model = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded        : epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}")

    noise_type      = ckpt_args.get("noise_type", "gaussian")
    spectral_filter = ckpt.get("spectral_filter", None)
    print(f"noise_type    : {noise_type}")

    diffusion = DDPM(T=T, beta_schedule=schedule, device=device,
                     noise_type=noise_type, spectral_filter=spectral_filter)

    # ---- Sample & path -------------------------------------------------------
    idx       = args.sample % len(test_ds)
    x0_true   = test_ds[idx]
    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=args.seed)

    x0_obs = x0_true.clone()
    x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

    true_np = x0_true.numpy()
    n_ocean = int((~test_ds.land_mask).sum().item())

    # ---- Un-normalize for display (physical units) --------------------------
    def _denorm(arr):
        if data_mean is None:
            return arr
        out = arr * data_std + data_mean
        out[:, land_mask_np] = 0.0
        return out

    true_np = _denorm(true_np)

    true_speed = np.sqrt(true_np[0] ** 2 + true_np[1] ** 2)
    true_speed[land_mask_np] = np.nan
    vmax = float(np.nanpercentile(true_speed, 98)) or 1.0

    u_true_d    = true_np[0].T
    v_true_d    = true_np[1].T
    land_mask_d = land_mask_np.T
    path_mask_d = path_mask.T

    print(f"\nTest sample   : {idx}  |  path covers "
          f"{path_mask.sum()} / {n_ocean} ocean cells "
          f"({100 * path_mask.sum() / n_ocean:.1f}%)")
    print("Running PPR inference and collecting frames ...")

    # ---- Inference + frame collection ----------------------------------------
    pil_frames = []
    final_pred_np = None

    for t_int, xt_np, x0hat_np in ppr_frames(
        model, diffusion, x0_obs, path_mask, land_mask_np,
        r=args.resample, proj_iter=args.proj_iter, projector=args.projector,
        inference_steps=args.inference_steps,
        data_mean=data_mean, data_std=data_std,
        device=device, capture_every=args.capture_every,
    ):
        xt_np    = _denorm(xt_np)
        x0hat_np = _denorm(x0hat_np)
        pil_frames.append(make_frame(
            t_int, T, schedule, idx,
            u_true_d,      v_true_d,
            xt_np[0].T,    xt_np[1].T,
            x0hat_np[0].T, x0hat_np[1].T,
            path_mask_d, land_mask_d, path_mask.sum(), args.seed, vmax=vmax,
        ))
        final_pred_np = xt_np

        if len(pil_frames) % 10 == 0 or t_int == 0:
            print(f"  t={t_int:4d}  frame {len(pil_frames)}")

    # ---- RMSE over ocean cells (physical units) ------------------------------
    ocean = ~land_mask_np
    diff  = final_pred_np[:, ocean] - true_np[:, ocean]
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    print(f"\nRMSE          : {rmse:.6f}  (sample {idx}, {ocean.sum()} ocean cells)")
    print(f"Total frames  : {len(pil_frames)}")

    # ---- Save GIF ------------------------------------------------------------
    print("Saving GIF ...")
    duration_ms = max(1, int(1000 / args.fps))
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
