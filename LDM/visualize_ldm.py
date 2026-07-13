"""
visualize_ldm.py  —  Inference visualizations for the Latent DPS model.
Style: visualize_infer.py  (speed imshow + black quiver, hot_r error, Reds path)
Layout per sample: 2 rows × 3 cols
  Row 0: Ground Truth | Latent DPS prediction | AE prediction (for comparison)
  Row 1: Robot Path   | Latent DPS error      | AE error
"""
import argparse
import os
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from diffusion import DDPM
from vae_model import OceanVAE
from latent_unet import LatentUNet
from ae_model import RepaintAutoencoder


# ── path generator ─────────────────────────────────────────────────────────

def biased_walk_path(land_mask, n_steps=150, seed=None, straight_bias=0.75):
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape
    ocean_cells = list(zip(*np.where(~land_mask)))
    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c  = int(start[0]), int(start[1])
    path_mask = np.zeros((H, W), dtype=bool); path_mask[r, c] = True
    all_dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    cur_dir  = all_dirs[rng.integers(4)]
    vc = np.zeros((H, W), dtype=np.float32); vc[r, c] = 1.0
    for _ in range(n_steps - 1):
        valid = [(dr,dc) for dr,dc in all_dirs
                 if 0<=r+dr<H and 0<=c+dc<W and not land_mask[r+dr,c+dc]]
        if not valid: break
        side = (1.0 - straight_bias) / 2.0
        weights = []
        for dr, dc in valid:
            dot = dr*cur_dir[0] + dc*cur_dir[1]
            w = straight_bias if dot==1 else (side if dot==0 else side*0.05)
            weights.append(w / (1.0 + vc[r+dr, c+dc]))
        weights = np.array(weights); weights /= weights.sum()
        idx = rng.choice(len(valid), p=weights)
        dr, dc = valid[idx]; r, c = r+dr, c+dc
        cur_dir = (dr, dc); vc[r, c] += 1.0; path_mask[r, c] = True
    return path_mask


# ── Latent DPS inference ────────────────────────────────────────────────────

def latent_dps_infer(vae, latent_model, diffusion,
                     x0_obs_np, path_mask, land_mask,
                     device, zeta=0.04):
    H, W = land_mask.shape
    x0_obs  = torch.from_numpy(x0_obs_np).unsqueeze(0).to(device)
    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    C, Hl, Wl = vae.latent_shape
    zt = torch.randn(1, C, Hl, Wl, device=device) * diffusion.noise_std
    vae.eval(); latent_model.eval()

    for t_int in reversed(range(diffusion.T)):
        t_prev = max(t_int - 1, 0)
        zt_in = zt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        eps_pred = latent_model(zt_in, t_vec)
        ab       = diffusion.alpha_bar[t_int]
        z0_hat   = (zt_in - (1.0 - ab).sqrt() * eps_pred) / ab.sqrt()
        z0_hat   = z0_hat.clamp(-5.0, 5.0)

        x0_hat = vae.decode(z0_hat, orig_H=H, orig_W=W) * ocean_t
        residual = known_t * (x0_hat - x0_obs)
        norm_sq  = (residual ** 2).sum()
        grad = torch.autograd.grad(norm_sq, zt_in)[0]

        with torch.no_grad():
            zt_next  = diffusion.p_sample_step(latent_model, zt_in.detach(), t_int, t_prev)
            norm     = norm_sq.sqrt().item() + 1e-8
            zt_next  = zt_next - (zeta / norm) * grad.detach()

        zt = zt_next

    with torch.no_grad():
        x0_final = vae.decode(zt, orig_H=H, orig_W=W) * ocean_t
    return x0_final.squeeze(0).cpu().numpy()


# ── AE inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def ae_infer(ae_model, x_obs_np, path_mask, device):
    mask_ch = path_mask.astype(np.float32)[None]
    ae_inp  = torch.from_numpy(np.concatenate([x_obs_np, mask_ch], axis=0)).unsqueeze(0).to(device)
    ae_model.eval()
    return ae_model(ae_inp).squeeze(0).cpu().numpy()


# ── drawing helpers ─────────────────────────────────────────────────────────

def plot_field(ax, field, land_mask, title, vmax=None, step=2, add_cbar=True):
    u, v = field[0].T, field[1].T
    lm   = land_mask.T
    H, W = u.shape
    speed = np.ma.masked_where(lm, np.sqrt(u**2 + v**2))
    if vmax is None:
        vmax = float(np.nanpercentile(speed.compressed(), 98)) if speed.count() else 1.0
    vmax = max(vmax, 1e-6)
    ext = [-0.5, W-0.5, -0.5, H-0.5]
    im = ax.imshow(speed, origin="lower", cmap="cool", vmin=0, vmax=vmax,
                   extent=ext, aspect="auto", zorder=0)
    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["none", "black"]),
              extent=ext, aspect="auto", zorder=1)
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq  = u[::step, ::step], v[::step, ::step]
    mask    = ~lm[::step, ::step]
    ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask],
              color="black", scale=12, width=0.003, zorder=2)
    if add_cbar:
        plt.colorbar(im, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=10); ax.set_xlabel("X"); ax.set_ylabel("Y")
    return vmax


def plot_path(ax, land_mask, path_mask, title):
    lm, pm = land_mask.T, path_mask.T
    H, W   = lm.shape
    ext    = [-0.5, W-0.5, -0.5, H-0.5]
    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["white", "black"]),
              extent=ext, aspect="auto", zorder=0)
    pd = np.zeros((H, W), dtype=float); pd[pm] = 1.0
    ax.imshow(pd, origin="lower", cmap="Reds", alpha=0.8,
              extent=ext, aspect="auto", zorder=1, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728", label="Path")
    land_p  = mpatches.Patch(facecolor="black",   label="Land")
    ax.legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=7)


def plot_error(ax, pred, true, land_mask, title, add_cbar=True):
    lm  = land_mask.T
    err = np.sqrt((pred[0]-true[0])**2 + (pred[1]-true[1])**2).T
    H, W = lm.shape
    ext  = [-0.5, W-0.5, -0.5, H-0.5]
    em   = np.ma.masked_where(lm, err)
    im   = ax.imshow(em, origin="lower", cmap="hot_r", aspect="auto", extent=ext, zorder=0)
    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["none", "black"]),
              extent=ext, aspect="auto", zorder=1)
    if add_cbar:
        plt.colorbar(im, ax=ax, label="|error| speed", shrink=0.7)
    ax.set_title(title, fontsize=10); ax.set_xlabel("X"); ax.set_ylabel("Y")


def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask])**2)))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--vae_ckpt",    default="/root/ldm/checkpoints_vae/best_vae.pt")
    p.add_argument("--ldm_ckpt",    default="/root/ldm/checkpoints_ldm/best_latent_ddpm.pt")
    p.add_argument("--ae_ckpt",     default="/root/autoencoder_train/checkpoints/best_model_autoencoder.pt")
    p.add_argument("--sample_idxs", default="3,7,15,22,30")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--path_steps",  type=int, default=150)
    p.add_argument("--zeta",        type=float, default=0.04)
    p.add_argument("--out_dir",     default="/root/ldm/inference_images")
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sample_idxs = [int(x) for x in args.sample_idxs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    def load_s(split_name):
        IDX = {"train": 0, "test": 2}
        key = split_name if (isinstance(data, dict) and split_name in data) else IDX[split_name]
        arr = np.asarray(data[key], dtype=np.float32)
        return np.nan_to_num(np.transpose(arr, (3,2,0,1)).astype(np.float32)), np.isnan(arr[:,:,0,0])

    _, train_land = load_s("train")
    test_fields, land_mask = load_s("test")
    ocean_mask = ~land_mask

    # Load models
    print("Loading VAE...")
    vae_ck = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_a  = vae_ck.get("args", {})
    vae    = OceanVAE(c_lat=vae_a.get("c_lat", 4), base_ch=vae_a.get("base_ch", 32)).to(device)
    vae.load_state_dict(vae_ck["model"]); vae.eval()

    print("Loading latent DDPM...")
    ldm_ck = torch.load(args.ldm_ckpt, map_location=device, weights_only=False)
    ldm_a  = ldm_ck.get("args", {})
    c_lat  = ldm_ck.get("c_lat", vae.c_lat)
    lat_m  = LatentUNet(in_ch=c_lat, base_ch=ldm_a.get("base_ch", 64), time_dim=ldm_a.get("time_dim", 256)).to(device)
    lat_m.load_state_dict(ldm_ck["model"]); lat_m.eval()
    diff   = DDPM(T=ldm_a.get("T", 1000), beta_schedule=ldm_a.get("schedule", "linear"),
                  device=device, noise_std=ldm_ck.get("noise_std", 1.0))

    print("Loading AE...")
    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae_m  = RepaintAutoencoder(in_ch=3, out_ch=2, base_ch=ae_ck.get("args", {}).get("base_ch", 64)).to(device)
    ae_m.load_state_dict(ae_ck["model"]); ae_m.eval()

    for idx in sample_idxs:
        print(f"\nSample {idx}...")
        true      = test_fields[idx]
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps, seed=args.seed + idx)
        x_obs     = true.copy(); x_obs[:, ~path_mask] = 0.0
        pct_obs   = 100.0 * path_mask.sum() / ocean_mask.sum()

        print("  Running Latent DPS...")
        pred_ldm = latent_dps_infer(vae, lat_m, diff, x_obs, path_mask, land_mask,
                                     device=device, zeta=args.zeta)
        print("  Running AE...")
        pred_ae  = ae_infer(ae_m, x_obs, path_mask, device)

        rmse_ldm = rmse_ocean(pred_ldm, true, ocean_mask)
        rmse_ae  = rmse_ocean(pred_ae,  true, ocean_mask)
        print(f"  Latent DPS RMSE={rmse_ldm:.4f}  AE RMSE={rmse_ae:.4f}")

        # Shared vmax
        spd   = np.sqrt(true[0]**2 + true[1]**2)
        vmax  = max(float(np.nanpercentile(spd[ocean_mask], 98)), 1e-6)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Latent DPS vs AE  —  sample_idx={idx}  seed={args.seed}  "
            f"({pct_obs:.1f}% ocean observed)",
            fontsize=13, fontweight="bold"
        )

        plot_field(axes[0,0], true,     land_mask, "Ground Truth",                vmax=vmax)
        plot_field(axes[0,1], pred_ldm, land_mask, f"Latent DPS\nRMSE={rmse_ldm:.4f}", vmax=vmax)
        plot_field(axes[0,2], pred_ae,  land_mask, f"Autoencoder\nRMSE={rmse_ae:.4f}",  vmax=vmax)

        plot_path(axes[1,0], land_mask, path_mask,
                  f"Robot Path ({int(path_mask.sum())} cells, seed={args.seed+idx})")
        plot_error(axes[1,1], pred_ldm, true, land_mask, f"Error — Latent DPS\nRMSE={rmse_ldm:.4f}")
        plot_error(axes[1,2], pred_ae,  true, land_mask, f"Error — AE\nRMSE={rmse_ae:.4f}")

        plt.tight_layout()
        out = os.path.join(args.out_dir, f"ldm_sample_{idx:03d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
