"""
Visualize one sample comparing all 4 models: baseline, physics, autoencoder, atmodist.
Style matches Visualization/visualize_infer.py:
  - Speed as imshow background (cool colormap)
  - Black quiver arrows overlaid
  - Black land overlay
  - Robot path: white ocean + Reds overlay
  - Error: hot_r heatmap + black land overlay
  - origin="lower", transpose (94,44)->(44,94)

Layout: 2 rows × 5 cols
  Row 0: Ground Truth | Base | Physics | Autoencoder | AtmoDist
  Row 1: Robot Path   | Base err | Physics err | AE err | AtmoDist err
"""
import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from diffusion import DDPM
from repaint_model import Repaint
from ae_model import RepaintAutoencoder
from atmodist_model import AtmoDistEncoder


# ── helpers (same as batch script) ──────────────────────────────────────────

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


@torch.no_grad()
def repaint_infer_r1(model, diffusion, x0_known, path_mask, land_mask, device):
    model.eval()
    H, W = x0_known.shape[1:]
    x0_known = x0_known.unsqueeze(0).to(device)
    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]
    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    for t_int in reversed(range(diffusion.T)):
        t_prev_int = max(t_int - 1, 0)
        xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)
        t_prev_tensor = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
        xt_known, _ = diffusion.q_sample(x0_known, t_prev_tensor)
        xt = known_t * xt_known + (1.0 - known_t) * xt_unknown
        xt = xt * ocean_t
    return xt.squeeze(0).cpu().numpy()


def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask]) ** 2)))


def load_split(data, split_name):
    SPLIT_IDX = {"train": 0, "val": 1, "test": 2}
    key = split_name if (isinstance(data, dict) and split_name in data) else SPLIT_IDX[split_name]
    arr = np.asarray(data[key], dtype=np.float32)
    fields = np.transpose(arr, (3, 2, 0, 1)).astype(np.float32)
    land_mask = np.isnan(arr[:, :, 0, 0])
    return np.nan_to_num(fields, nan=0.0), land_mask


# ── drawing helpers (visualize_infer.py style) ──────────────────────────────

def plot_field(ax, field, land_mask, title, step=2, vmax=None, add_cbar=True):
    """Speed imshow (cool) + black quiver + black land overlay. origin='lower'."""
    # Transpose (94,44)->(44,94) so X=0..93, Y=0..43, land at top (high Y)
    u = field[0].T
    v = field[1].T
    lm = land_mask.T
    H, W = u.shape

    speed = np.ma.masked_where(lm, np.sqrt(u**2 + v**2))
    if vmax is None:
        vmax = float(np.nanpercentile(speed.compressed(), 98)) if speed.count() else 1.0
    vmax = max(vmax, 1e-6)

    extent = [-0.5, W - 0.5, -0.5, H - 0.5]

    im = ax.imshow(speed, origin="lower", cmap="cool",
                   vmin=0, vmax=vmax, extent=extent, aspect="auto", zorder=0)
    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["none", "black"]),
              extent=extent, aspect="auto", zorder=1)

    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u[::step, ::step]; vq = v[::step, ::step]
    mask = ~lm[::step, ::step]
    ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask],
              color="black", scale=12, width=0.003, zorder=2)

    if add_cbar:
        plt.colorbar(im, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    return vmax


def plot_path(ax, land_mask, path_mask, title):
    """White ocean + Reds path overlay + black land overlay."""
    lm = land_mask.T
    pm = path_mask.T
    H, W = lm.shape
    extent = [-0.5, W - 0.5, -0.5, H - 0.5]

    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["white", "black"]),
              extent=extent, aspect="auto", zorder=0)

    path_display = np.zeros((H, W), dtype=float)
    path_display[pm] = 1.0
    ax.imshow(path_display, origin="lower", cmap="Reds", alpha=0.8,
              extent=extent, aspect="auto", zorder=1, vmin=0, vmax=1)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728",                 label="Path")
    land_p  = mpatches.Patch(facecolor="black",                   label="Land")
    ax.legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=7)


def plot_error(ax, pred, true, land_mask, title, add_cbar=True):
    """Per-cell error magnitude heatmap (hot_r) + black land."""
    lm = land_mask.T
    err = np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2).T
    H, W = lm.shape
    extent = [-0.5, W - 0.5, -0.5, H - 0.5]

    err_plot = np.ma.masked_where(lm, err)
    im = ax.imshow(err_plot, origin="lower", cmap="hot_r",
                   aspect="auto", extent=extent, zorder=0)
    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["none", "black"]),
              extent=extent, aspect="auto", zorder=1)

    if add_cbar:
        plt.colorbar(im, ax=ax, label="|error| speed", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",       default="/root/ocean_ddpm/data.pickle")
    p.add_argument("--base_ckpt",    default="/root/autoencoder_train/checkpoints_linear/best_model_linear.pt")
    p.add_argument("--physics_ckpt", default="/root/autoencoder_train/checkpoints_physics/best_model_physics.pt")
    p.add_argument("--ae_ckpt",      default="/root/autoencoder_train/checkpoints/best_model_autoencoder.pt")
    p.add_argument("--atmodist_ckpt",default="/root/autoencoder_train/checkpoints_atmodist/best_model_atmodist.pt")
    p.add_argument("--sample_idx",   type=int, default=0)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--path_steps",   type=int, default=150)
    p.add_argument("--out",          default="/root/autoencoder_train/inference_results/comparison.png")
    p.add_argument("--device",       default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    train_fields, train_land = load_split(data, "train")
    test_fields,  test_land  = load_split(data, "test")
    land_mask  = test_land
    ocean_mask = ~land_mask

    true = test_fields[args.sample_idx]
    path_mask = biased_walk_path(land_mask, n_steps=args.path_steps,
                                  seed=args.seed + args.sample_idx)
    x_obs = true.copy(); x_obs[:, ~path_mask] = 0.0

    # ── load models ──────────────────────────────────────────────────────────
    def load_repaint(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        ca = ck.get("args", {})
        mdl = Repaint(in_ch=2, base_ch=ca.get("base_ch", 64),
                      time_dim=ca.get("time_dim", 256)).to(device)
        mdl.load_state_dict(ck["model"]); mdl.eval()
        ns = ck.get("noise_std") or float(train_fields[:, :, ocean_mask].std())
        diff = DDPM(T=ca.get("T", 1000), beta_schedule=ck.get("schedule", "linear"),
                    device=device, noise_std=ns)
        return mdl, diff

    base_m,  base_d  = load_repaint(args.base_ckpt)
    phys_m,  phys_d  = load_repaint(args.physics_ckpt)

    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae_m  = RepaintAutoencoder(in_ch=3, out_ch=2,
                               base_ch=ae_ck.get("args", {}).get("base_ch", 64)).to(device)
    ae_m.load_state_dict(ae_ck["model"]); ae_m.eval()

    atm_ck = torch.load(args.atmodist_ckpt, map_location=device, weights_only=False)
    atm_ca = atm_ck.get("args", {})
    atm_m  = AtmoDistEncoder(in_ch=2, base_ch=atm_ca.get("base_ch", 64),
                              emb_dim=atm_ca.get("emb_dim", 256), n_classes=6).to(device)
    atm_m.load_state_dict(atm_ck["model"]); atm_m.eval()

    # precompute train embeddings
    print("Computing train embeddings...")
    with torch.no_grad():
        train_emb = torch.cat([
            torch.nn.functional.normalize(
                atm_m.encode(torch.from_numpy(train_fields[i:i+256]).to(device)), dim=1)
            for i in range(0, train_fields.shape[0], 256)
        ], dim=0)

    # ── run inference ─────────────────────────────────────────────────────────
    print("Running base RePaint r=1...")
    pred_base = repaint_infer_r1(base_m, base_d, torch.from_numpy(x_obs),
                                  path_mask, land_mask, device)

    print("Running physics RePaint r=1...")
    pred_phys = repaint_infer_r1(phys_m, phys_d, torch.from_numpy(x_obs),
                                  path_mask, land_mask, device)

    print("Running autoencoder...")
    ae_inp = np.concatenate([x_obs, path_mask.astype(np.float32)[None]], axis=0)
    with torch.no_grad():
        pred_ae = ae_m(torch.from_numpy(ae_inp).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    print("Running AtmoDist retrieval...")
    with torch.no_grad():
        obs_t = torch.from_numpy(x_obs).unsqueeze(0).to(device)
        e = torch.nn.functional.normalize(atm_m.encode(obs_t), dim=1)
        nn_idx = int(torch.argmax(torch.matmul(train_emb, e.T).squeeze()).item())
    pred_atm = train_fields[nn_idx]

    # ── compute RMSEs ─────────────────────────────────────────────────────────
    rmse_base = rmse_ocean(pred_base, true, ocean_mask)
    rmse_phys = rmse_ocean(pred_phys, true, ocean_mask)
    rmse_ae   = rmse_ocean(pred_ae,   true, ocean_mask)
    rmse_atm  = rmse_ocean(pred_atm,  true, ocean_mask)
    pct_obs   = 100.0 * path_mask.sum() / ocean_mask.sum()

    print(f"Base RMSE:    {rmse_base:.4f}")
    print(f"Physics RMSE: {rmse_phys:.4f}")
    print(f"AE RMSE:      {rmse_ae:.4f}")
    print(f"AtmoDist RMSE:{rmse_atm:.4f}")
    print(f"Observed:     {pct_obs:.1f}% of ocean cells")

    # ── plot ──────────────────────────────────────────────────────────────────
    # Shared vmax across all speed panels (98th pct of true field)
    speed_true = np.sqrt(true[0]**2 + true[1]**2)
    ocean_mask_arr = ~land_mask
    vmax = float(np.nanpercentile(speed_true[ocean_mask_arr], 98)) if ocean_mask_arr.any() else 1.0
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 5, figsize=(24, 9))
    fig.suptitle(
        f"Model Comparison  —  sample_idx={args.sample_idx}  seed={args.seed}  "
        f"({pct_obs:.1f}% ocean observed)",
        fontsize=13, fontweight="bold"
    )

    # Row 0: speed fields
    plot_field(axes[0, 0], true,      land_mask, "Ground Truth",            vmax=vmax, add_cbar=True)
    plot_field(axes[0, 1], pred_base, land_mask, f"Baseline (RePaint r=1)\nRMSE={rmse_base:.4f}", vmax=vmax, add_cbar=True)
    plot_field(axes[0, 2], pred_phys, land_mask, f"Physics (RePaint r=1)\nRMSE={rmse_phys:.4f}",  vmax=vmax, add_cbar=True)
    plot_field(axes[0, 3], pred_ae,   land_mask, f"Autoencoder\nRMSE={rmse_ae:.4f}",               vmax=vmax, add_cbar=True)
    plot_field(axes[0, 4], pred_atm,  land_mask, f"AtmoDist Retrieval\nRMSE={rmse_atm:.4f}",       vmax=vmax, add_cbar=True)

    # Row 1: path + error maps
    plot_path(axes[1, 0], land_mask, path_mask,
              f"Robot Path\n({int(path_mask.sum())} cells, seed={args.seed})")
    plot_error(axes[1, 1], pred_base, true, land_mask, f"Error — Baseline\nRMSE={rmse_base:.4f}",   add_cbar=True)
    plot_error(axes[1, 2], pred_phys, true, land_mask, f"Error — Physics\nRMSE={rmse_phys:.4f}",    add_cbar=True)
    plot_error(axes[1, 3], pred_ae,   true, land_mask, f"Error — Autoencoder\nRMSE={rmse_ae:.4f}",  add_cbar=True)
    plot_error(axes[1, 4], pred_atm,  true, land_mask, f"Error — AtmoDist\nRMSE={rmse_atm:.4f}",   add_cbar=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
