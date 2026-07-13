"""
visualize_base_physics.py
Generates comparison images for base + physics models only.
Layout (per image): 2 rows × 3 cols
  Row 0: Ground Truth | Baseline (RePaint r=1) | Physics (RePaint r=1)
  Row 1: Robot Path   | Baseline error         | Physics error
Style: visualize_infer.py  (speed imshow + black quiver, hot_r error, Reds path)
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
from repaint_model import Repaint


# ── path generator ────────────────────────────────────────────────────────────

def biased_walk_path(land_mask, n_steps=150, seed=None, straight_bias=0.75):
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape
    ocean_cells = list(zip(*np.where(~land_mask)))
    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c = int(start[0]), int(start[1])
    path_mask = np.zeros((H, W), dtype=bool)
    path_mask[r, c] = True
    all_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cur_dir = all_dirs[rng.integers(4)]
    visit_count = np.zeros((H, W), dtype=np.float32)
    visit_count[r, c] = 1.0
    for _ in range(n_steps - 1):
        valid = [(dr, dc) for dr, dc in all_dirs
                 if 0 <= r+dr < H and 0 <= c+dc < W and not land_mask[r+dr, c+dc]]
        if not valid:
            break
        side = (1.0 - straight_bias) / 2.0
        weights = []
        for dr, dc in valid:
            dot = dr * cur_dir[0] + dc * cur_dir[1]
            w = straight_bias if dot == 1 else (side if dot == 0 else side * 0.05)
            weights.append(w / (1.0 + visit_count[r+dr, c+dc]))
        weights = np.array(weights); weights /= weights.sum()
        idx = rng.choice(len(valid), p=weights)
        dr, dc = valid[idx]; r, c = r+dr, c+dc
        cur_dir = (dr, dc); visit_count[r, c] += 1.0; path_mask[r, c] = True
    return path_mask


# ── RePaint r=1 inference ─────────────────────────────────────────────────────

@torch.no_grad()
def repaint_r1(model, diffusion, x0_known, path_mask, land_mask, device):
    model.eval()
    H, W = x0_known.shape[1:]
    x0_t  = x0_known.unsqueeze(0).to(device)
    known = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]
    xt    = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean
    for t_int in reversed(range(diffusion.T)):
        t_prev = max(t_int - 1, 0)
        xt_u   = diffusion.p_sample_step(model, xt, t_int, t_prev)
        tp     = torch.full((1,), t_prev, device=device, dtype=torch.long)
        xt_k, _= diffusion.q_sample(x0_t, tp)
        xt     = known * xt_k + (1.0 - known) * xt_u
        xt     = xt * ocean
    return xt.squeeze(0).cpu().numpy()


def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask]) ** 2)))


# ── drawing helpers (visualize_infer.py style) ────────────────────────────────

def plot_field(ax, field, land_mask, title, vmax=None, step=2, add_cbar=True):
    u, v = field[0].T, field[1].T
    lm   = land_mask.T
    H, W = u.shape
    speed = np.ma.masked_where(lm, np.sqrt(u**2 + v**2))
    if vmax is None:
        vmax = float(np.nanpercentile(speed.compressed(), 98)) if speed.count() else 1.0
    vmax = max(vmax, 1e-6)
    ext = [-0.5, W-0.5, -0.5, H-0.5]
    im = ax.imshow(speed, origin="lower", cmap="cool",
                   vmin=0, vmax=vmax, extent=ext, aspect="auto", zorder=0)
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
    ax.set_title(title, fontsize=11); ax.set_xlabel("X"); ax.set_ylabel("Y")
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
    ax.set_title(title, fontsize=11); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728", label="Path")
    land_p  = mpatches.Patch(facecolor="black",   label="Land")
    ax.legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)


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
    ax.set_title(title, fontsize=11); ax.set_xlabel("X"); ax.set_ylabel("Y")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",       default="/root/ocean_ddpm/data.pickle")
    p.add_argument("--base_ckpt",    default="/root/autoencoder_train/checkpoints_linear/best_model_linear.pt")
    p.add_argument("--physics_ckpt", default="/root/autoencoder_train/checkpoints_physics/best_model_physics.pt")
    p.add_argument("--sample_idxs",  default="3,7,15", help="Comma-separated test sample indices")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--path_steps",   type=int, default=150)
    p.add_argument("--out_dir",      default="/root/autoencoder_train/inference_results/base_physics")
    p.add_argument("--device",       default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sample_idxs = [int(x) for x in args.sample_idxs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    SPLIT_IDX = {"train": 0, "test": 2}
    def load(split):
        key = split if (isinstance(data, dict) and split in data) else SPLIT_IDX[split]
        arr = np.asarray(data[key], dtype=np.float32)
        return np.nan_to_num(np.transpose(arr, (3,2,0,1)).astype(np.float32)), np.isnan(arr[:,:,0,0])

    train_fields, _ = load("train")
    test_fields, land_mask = load("test")
    ocean_mask = ~land_mask

    def load_model(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        ca = ck.get("args", {})
        ns = ck.get("noise_std") or float(train_fields[:, :, ocean_mask].std())
        m  = Repaint(in_ch=2, base_ch=ca.get("base_ch", 64),
                     time_dim=ca.get("time_dim", 256)).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        d  = DDPM(T=ca.get("T", 1000), beta_schedule=ck.get("schedule", "linear"),
                  device=device, noise_std=ns)
        return m, d

    print("Loading base model...")
    base_m, base_d = load_model(args.base_ckpt)
    print("Loading physics model...")
    phys_m, phys_d = load_model(args.physics_ckpt)

    for idx in sample_idxs:
        print(f"\nSample {idx}...")
        true      = test_fields[idx]
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps, seed=args.seed + idx)
        x_obs     = true.copy(); x_obs[:, ~path_mask] = 0.0
        pct_obs   = 100.0 * path_mask.sum() / ocean_mask.sum()

        print(f"  Running base RePaint r=1...")
        pred_base = repaint_r1(base_m, base_d, torch.from_numpy(x_obs), path_mask, land_mask, device)
        print(f"  Running physics RePaint r=1...")
        pred_phys = repaint_r1(phys_m, phys_d, torch.from_numpy(x_obs), path_mask, land_mask, device)

        rmse_base = rmse_ocean(pred_base, true, ocean_mask)
        rmse_phys = rmse_ocean(pred_phys, true, ocean_mask)
        print(f"  Base RMSE={rmse_base:.4f}  Physics RMSE={rmse_phys:.4f}")

        # Shared vmax
        spd = np.sqrt(true[0]**2 + true[1]**2)
        vmax = max(float(np.nanpercentile(spd[ocean_mask], 98)), 1e-6)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Base vs Physics  —  sample_idx={idx}  seed={args.seed}  "
            f"({pct_obs:.1f}% ocean observed)",
            fontsize=13, fontweight="bold"
        )

        plot_field(axes[0,0], true,      land_mask, "Ground Truth",            vmax=vmax)
        plot_field(axes[0,1], pred_base, land_mask, f"Baseline (RePaint r=1)\nRMSE={rmse_base:.4f}", vmax=vmax)
        plot_field(axes[0,2], pred_phys, land_mask, f"Physics (RePaint r=1)\nRMSE={rmse_phys:.4f}",  vmax=vmax)

        plot_path(axes[1,0], land_mask, path_mask,
                  f"Robot Path ({int(path_mask.sum())} cells, seed={args.seed+idx})")
        plot_error(axes[1,1], pred_base, true, land_mask, f"Error — Baseline\nRMSE={rmse_base:.4f}")
        plot_error(axes[1,2], pred_phys, true, land_mask, f"Error — Physics\nRMSE={rmse_phys:.4f}")

        plt.tight_layout()
        out = os.path.join(args.out_dir, f"sample_{idx:03d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
