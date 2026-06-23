"""
Visualise the RePaint denoising process for the FiLM-conditioned DDPM as a GIF.

2x2 panel layout:
  Top-left     : Ground truth            (static)
  Top-right    : Noisy field xt          (shows denoising progress)
  Bottom-left  : Voronoi input           (static)
  Bottom-right : Model x0-hat estimate   (model's current clean-field guess)

Usage (run from workspace root):
    python repaint_gif_cond.py
    python repaint_gif_cond.py --sample 5 --seed 5 --fps 12
    python repaint_gif_cond.py --checkpoint "Conditional DDPM/checkpoints_voronoi_ns012/best_cond_ddpm_voronoi_cosine.pt"
"""

import argparse
import os
import sys
from io import BytesIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_root              = os.path.dirname(os.path.abspath(__file__))
_cond_dir          = os.path.join(_root, "Conditional DDPM")
_voronoi_model_dir = os.path.join(_root, "Voronoi", "model")

sys.path.insert(0, _voronoi_model_dir)   # voronoi_model.py
sys.path.insert(0, _cond_dir)            # cond_model.py, cond_diffusion.py
sys.path.insert(0, _root)               # dataset.py, paths.py

from dataset        import OceanCurrentDataset
from paths          import biased_walk_path
from voronoi_model  import VoronoiLayer
from cond_model     import CondUNet
from cond_diffusion import CondDDPM


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render conditional RePaint denoising as a GIF."
    )
    p.add_argument("--pickle",        default="data.pickle")
    p.add_argument("--checkpoint",    default=None,
                   help="Path to checkpoint. Defaults to "
                        "Conditional DDPM/checkpoints_voronoi_ns012/best_cond_ddpm_voronoi_cosine.pt")
    p.add_argument("--split",         type=int, default=1,
                   help="Dataset split: 0=train, 1=val, 2=test (default: 1)")
    p.add_argument("--sample",        type=int, default=0,
                   help="Dataset index to visualise.")
    p.add_argument("--seed",          type=int, default=None,
                   help="Path seed. Defaults to sample index.")
    p.add_argument("--path_steps",    type=int, default=150)
    p.add_argument("--resample",      type=int, default=10,
                   help="RePaint r parameter")
    p.add_argument("--capture_every", type=int, default=20)
    p.add_argument("--fps",           type=int, default=10)
    p.add_argument("--T",             type=int, default=1000)
    p.add_argument("--base_ch",       type=int, default=64)
    p.add_argument("--time_dim",      type=int, default=256)
    p.add_argument("--cond_dim",      type=int, default=256)
    p.add_argument("--out",           default=None,
                   help="Output .gif path. Defaults to repaint_gif_cond_results/repaint_val_sample{idx}.gif")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quiver helper
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
# Conditioning helper  (identical logic to infer.py:make_cond_single)
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_voronoi_cond(x0, land_mask_np, voronoi_layer, path_mask, device):
    """Build (1, 3, H, W) voronoi conditioning for one sample."""
    C, H, W = x0.shape
    rows, cols = np.where(path_mask)

    rows_n = torch.tensor(rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
    cols_n = torch.tensor(cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
    sensor_pos = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)  # (1, K, 2)

    flat_idx    = torch.tensor(rows * W + cols, dtype=torch.long, device=device)
    flat_idx    = flat_idx.unsqueeze(0).unsqueeze(0).expand(1, C, len(rows))
    sensor_vals = torch.gather(
        x0.unsqueeze(0).reshape(1, C, H * W), 2, flat_idx
    )  # (1, C, K)

    return voronoi_layer.tessellate(sensor_vals, sensor_pos)  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Conditional RePaint loop that yields intermediate frames
# ---------------------------------------------------------------------------

@torch.no_grad()
def repaint_frames_cond(
    model, diffusion, voronoi_layer,
    x0_known, path_mask, land_mask, cond,
    r=10, device="cpu", capture_every=20,
):
    """
    Conditional RePaint — same control flow as repaint_frames() but passes
    `cond` to every model call.

    Yields (t_int, xt_np, x0hat_np) at captured timesteps.
    """
    H, W = x0_known.shape[1:]
    x0_known = x0_known.unsqueeze(0).to(device)   # (1, 2, H, W)

    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t  = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_scale * ocean_t
    T  = diffusion.T

    clamp_val = 3.0 * diffusion.noise_scale

    def get_x0hat(xt_, t_):
        t_tensor   = torch.full((1,), t_, device=device, dtype=torch.long)
        pred_noise = model(xt_, t_tensor, cond)
        ab         = diffusion.alpha_bar[max(t_, 0)]
        x0hat      = (xt_ - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        return x0hat.clamp(-clamp_val, clamp_val).squeeze(0).cpu().numpy()

    for t_int in reversed(range(T)):
        for j in range(r):
            xt_unknown = diffusion.p_sample_step(model, xt, t_int, cond)

            t_prev   = max(t_int - 1, 0)
            t_prev_t = torch.full((1,), t_prev, device=device, dtype=torch.long)
            xt_known, _ = diffusion.q_sample(x0_known, t_prev_t)

            xt_merged = known_t * xt_known + (1.0 - known_t) * xt_unknown
            xt_merged = xt_merged * ocean_t

            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int) * ocean_t
            else:
                xt = xt_merged

        if t_int == T - 1 or t_int == 0 or t_int % capture_every == 0:
            yield t_int, xt.squeeze(0).cpu().numpy(), get_x0hat(xt, t_int)


# ---------------------------------------------------------------------------
# Render one 2x2 figure -> PIL Image
# ---------------------------------------------------------------------------

def make_frame(t_int, T, idx,
               u_true, v_true,
               u_xt, v_xt,
               u_x0hat, v_x0hat,
               voronoi_np, land_mask_d, path_cells, seed, vmax=None):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=80)
    axes = axes.flatten()

    plot_field(axes[0], u_true, v_true, land_mask_d, "Ground Truth", vmax=vmax)
    plot_field(axes[1], u_xt,  v_xt,  land_mask_d,
               f"Noisy field  $x_t$   (t = {t_int})", vmax=vmax)
    plot_field(axes[2], voronoi_np[0].T, voronoi_np[1].T, land_mask_d,
               f"Voronoi input  ({path_cells} sensors, seed={seed})", vmax=vmax)
    plot_field(axes[3], u_x0hat, v_x0hat, land_mask_d,
               r"Model $\hat{x}_0$ estimate" + f"   (t = {t_int})", vmax=vmax)

    pct = 100.0 * (T - t_int) / T
    plt.suptitle(
        f"Cond-DDPM RePaint  —  voronoi  —  val sample {idx}"
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

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            _cond_dir, "checkpoints_voronoi_ns012",
            "best_cond_ddpm_voronoi_cosine.pt",
        )

    out_dir = os.path.join(_root, "repaint_gif_cond_results")
    os.makedirs(out_dir, exist_ok=True)
    if args.out is None:
        args.out = os.path.join(out_dir, f"repaint_val_sample{args.sample}.gif")

    seed = args.seed if args.seed is not None else args.sample

    n_frames_approx = 1000 // args.capture_every + 2
    print(f"Device        : {device}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Output        : {args.out}")
    print(f"capture_every : {args.capture_every}  ->  ~{n_frames_approx} frames")

    # ---- Data ----------------------------------------------------------------
    ds           = OceanCurrentDataset(args.pickle, split=args.split)
    land_mask_np = ds.land_mask.numpy()
    H, W         = land_mask_np.shape

    # ---- Model ---------------------------------------------------------------
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    cond_dim  = ckpt_args.get("cond_dim", args.cond_dim)
    T         = ckpt_args.get("T",        args.T)
    schedule  = ckpt_args.get("schedule", "cosine")
    noise_scale = ckpt_args.get("noise_scale", 1.0)

    model = CondUNet(in_ch=2, cond_in_ch=3, base_ch=base_ch,
                     time_dim=time_dim, cond_dim=cond_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded        : epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}, "
          f"noise_scale={noise_scale}")

    diffusion = CondDDPM(T=T, beta_schedule=schedule, device=device,
                         noise_scale=noise_scale)

    voronoi_layer = VoronoiLayer(H=H, W=W, n_sensors=args.path_steps).to(device)

    # ---- Sample & path -------------------------------------------------------
    idx       = args.sample % len(ds)
    x0_true   = ds[idx].to(device)
    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

    cond = make_voronoi_cond(x0_true, land_mask_np, voronoi_layer, path_mask, device)

    x0_obs = x0_true.clone()
    x0_obs[:, ~torch.from_numpy(path_mask).to(device)] = 0.0

    true_np  = x0_true.cpu().numpy()
    n_ocean  = int((~ds.land_mask).sum().item())

    true_speed = np.sqrt(true_np[0] ** 2 + true_np[1] ** 2)
    true_speed[land_mask_np] = np.nan
    vmax = float(np.nanpercentile(true_speed, 98)) or 1.0

    u_true_d    = true_np[0].T
    v_true_d    = true_np[1].T
    land_mask_d = land_mask_np.T
    voronoi_np  = cond.squeeze(0).cpu().numpy()   # (3, H, W)

    print(f"\nVal sample    : {idx}  |  path covers "
          f"{path_mask.sum()} / {n_ocean} ocean cells "
          f"({100 * path_mask.sum() / n_ocean:.1f}%)")
    print("Running conditional RePaint and collecting frames ...")

    # ---- Inference + frame collection ----------------------------------------
    pil_frames    = []
    final_pred_np = None

    for t_int, xt_np, x0hat_np in repaint_frames_cond(
        model, diffusion, voronoi_layer,
        x0_obs, path_mask, land_mask_np, cond,
        r=args.resample, device=device, capture_every=args.capture_every,
    ):
        pil_frames.append(make_frame(
            t_int, T, idx,
            u_true_d,      v_true_d,
            xt_np[0].T,    xt_np[1].T,
            x0hat_np[0].T, x0hat_np[1].T,
            voronoi_np, land_mask_d, path_mask.sum(), seed, vmax=vmax,
        ))
        final_pred_np = xt_np

        if len(pil_frames) % 10 == 0 or t_int == 0:
            print(f"  t={t_int:4d}  frame {len(pil_frames)}")

    # ---- RMSE ----------------------------------------------------------------
    ocean = ~land_mask_np
    diff  = final_pred_np[:, ocean] - true_np[:, ocean]
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    print(f"\nRMSE          : {rmse:.6f}  (sample {idx}, {ocean.sum()} ocean cells)")
    print(f"Total frames  : {len(pil_frames)}")

    # ---- Save GIF ------------------------------------------------------------
    print("Saving GIF ...")
    duration_ms = max(1, int(1000 / args.fps))
    durations   = [duration_ms] * (len(pil_frames) - 1) + [1000]
    pil_frames[0].save(
        args.out,
        save_all=True, append_images=pil_frames[1:],
        duration=durations, loop=0, optimize=False,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
