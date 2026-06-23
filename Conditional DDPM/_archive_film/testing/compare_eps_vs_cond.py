"""
Fair head-to-head: eps (RePaint) vs conditioned models on the SAME 10 val
samples and path seeds that were used in the conditional-model infer runs.

Samples and seeds are taken from the infer.py output with --seed 0:
  samples = [1663, 1598, 1246, 1000, 528, 80, 32, 603, 344, 147]
  seeds   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  (seed + run index)

Usage (run from workspace root):
    python3 "Conditional DDPM/compare_eps_vs_cond.py"
    python3 "Conditional DDPM/compare_eps_vs_cond.py" --resample 10
"""

import argparse, os, sys
import numpy as np
import torch
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_here      = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_here, ".."))
# On the server the DDPM tree is flat (no model/ or testing/repaint/ subdirs);
# locally it has subdirs — support both.
_ddpm_root      = os.path.join(_repo_root, "DDPM")
_ddpm_model_dir = os.path.join(_ddpm_root, "model")      # local
_repaint_dir    = os.path.join(_ddpm_root, "testing", "repaint")  # local
_voronoi_dir    = os.path.join(_repo_root, "Voronoi", "model")

for p in (_here, _repo_root, _ddpm_root, _ddpm_model_dir, _repaint_dir, _voronoi_dir):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from dataset        import OceanCurrentDataset
from paths          import biased_walk_path
from voronoi_model  import VoronoiLayer
from cond_model     import CondUNet
from cond_diffusion import CondDDPM
from diffusion      import DDPM
from model          import UNet
from repaint_infer  import repaint

SAMPLES = [1663, 1598, 1246, 1000, 528, 80, 32, 603, 344, 147]
SEEDS   = list(range(1, 11))   # seeds 1-10, matching infer.py with --seed 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--eps_ckpt",    default="checkpoints/best_model.pt")
    p.add_argument("--path_steps",  type=int, default=150)
    p.add_argument("--resample",    type=int, default=10)
    p.add_argument("--seed",        type=int, default=0,
                   help="Val sample selection seed (matches infer.py default).")
    p.add_argument("--out_dir",     default=None,
                   help="Output dir for plots. Defaults to 'Conditional DDPM/results_compare_grid'.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_LAND_BW    = plt.matplotlib.colors.ListedColormap(["white", "black"])
_LAND_ALPHA = plt.matplotlib.colors.ListedColormap(["none",  "black"])


def _ext(H, W):
    return [-0.5, W - 0.5, -0.5, H - 0.5]


def plot_field(ax, u, v, land_mask, title, clim=None, step=2, cmap="cool"):
    H, W = u.shape
    ext  = _ext(H, W)
    ax.imshow(land_mask, origin="lower", cmap=_LAND_BW, extent=ext, aspect="auto", zorder=0)
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    mq     = np.sqrt(uq**2 + vq**2)
    mask   = ~np.isnan(uq) & ~land_mask[::step, ::step]
    if mask.any():
        if clim is None:
            clim = (0, float(np.nanpercentile(mq[mask], 98)) or 1.0)
        q = ax.quiver(xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
                      cmap=cmap, clim=clim, scale=12, width=0.003, zorder=2)
        plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def plot_path(ax, land_mask, path_mask, title):
    H, W = land_mask.shape
    ext  = _ext(H, W)
    ax.imshow(land_mask, origin="lower", cmap=_LAND_BW, extent=ext, aspect="auto", zorder=0)
    disp = np.full(land_mask.shape, np.nan)
    disp[path_mask] = 1.0
    ax.imshow(disp, origin="lower", cmap="Reds", extent=ext,
              aspect="auto", zorder=1, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(handles=[
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean"),
        mpatches.Patch(facecolor="#d62728", label="Path"),
        mpatches.Patch(facecolor="black",   label="Land"),
    ], loc="upper right", fontsize=7)


def plot_error(ax, err, land_mask, title):
    H, W    = err.shape
    ext     = _ext(H, W)
    err_ma  = np.ma.masked_where(land_mask, err)
    vmax    = float(np.nanpercentile(err[~land_mask], 98)) if (~land_mask).any() else 1.0
    im = ax.imshow(err_ma, origin="lower", cmap="hot_r", aspect="auto",
                   extent=ext, vmin=0, vmax=vmax)
    ax.imshow(land_mask, origin="lower", cmap=_LAND_ALPHA, extent=ext,
              aspect="auto", zorder=1)
    plt.colorbar(im, ax=ax, label="Speed error", shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def save_grid_plot(
    sample_idx, seed,
    u_true, v_true,
    eps_pred, path_pred, voronoi_pred,
    path_mask, voronoi_np,
    land_mask_np,
    rmse_eps, rmse_path, rmse_voronoi,
    out_path,
):
    """
    3×3 grid:
      Row 0: Ground truth      | eps prediction       | eps RMSE heatmap
      Row 1: Robot path        | path-cond prediction | path-cond RMSE heatmap
      Row 2: Voronoi tessellat | voronoi-cond pred    | voronoi-cond RMSE heatmap
    """
    T_ = lambda a: a.T   # (H,W) → (W,H) for display

    land_d = T_(land_mask_np)
    path_d = T_(path_mask)

    u_t, v_t = T_(u_true), T_(v_true)
    speed_gt = np.sqrt(u_t**2 + v_t**2)
    speed_gt[land_d] = np.nan
    clim = (0, float(np.nanpercentile(speed_gt[~land_d], 98)) if (~land_d).any() else 1.0)

    # Error maps
    err_eps  = np.sqrt((T_(eps_pred[0])     - u_t)**2 + (T_(eps_pred[1])     - v_t)**2)
    err_path = np.sqrt((T_(path_pred[0])    - u_t)**2 + (T_(path_pred[1])    - v_t)**2)
    err_vor  = np.sqrt((T_(voronoi_pred[0]) - u_t)**2 + (T_(voronoi_pred[1]) - v_t)**2)
    for e in (err_eps, err_path, err_vor):
        e[land_d] = np.nan

    fig, axes = plt.subplots(3, 3, figsize=(22, 15))
    fig.suptitle(
        f"Val sample {sample_idx}  |  seed {seed}\n"
        f"eps RMSE={rmse_eps:.4f}   path-cond RMSE={rmse_path:.4f}   voronoi-cond RMSE={rmse_voronoi:.4f}",
        fontsize=12,
    )

    # Row 0
    plot_field(axes[0, 0], u_t, v_t, land_d, "Ground Truth", clim=clim)
    plot_field(axes[0, 1], T_(eps_pred[0]), T_(eps_pred[1]), land_d,
               f"DDPM-eps (RePaint)  RMSE={rmse_eps:.4f}", clim=clim)
    plot_error(axes[0, 2], err_eps, land_d, "eps — error heatmap")

    # Row 1
    plot_path(axes[1, 0], land_d, path_d, f"Robot Path ({int(path_mask.sum())} cells)")
    plot_field(axes[1, 1], T_(path_pred[0]), T_(path_pred[1]), land_d,
               f"path-cond + RePaint  RMSE={rmse_path:.4f}", clim=clim)
    plot_error(axes[1, 2], err_path, land_d, "path-cond — error heatmap")

    # Row 2
    plot_field(axes[2, 0], T_(voronoi_np[0]), T_(voronoi_np[1]), land_d,
               "Voronoi Tessellation", clim=clim)
    plot_field(axes[2, 1], T_(voronoi_pred[0]), T_(voronoi_pred[1]), land_d,
               f"voronoi-cond + RePaint  RMSE={rmse_voronoi:.4f}", clim=clim)
    plot_error(axes[2, 2], err_vor, land_d, "voronoi-cond — error heatmap")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.out_dir is None:
        args.out_dir = os.path.join(_here, "results_compare_grid")
    os.makedirs(args.out_dir, exist_ok=True)

    val_ds       = OceanCurrentDataset(args.pickle, split=1)
    land_mask_np = val_ds.land_mask.numpy()
    H, W         = land_mask_np.shape

    # ── Load eps model ──────────────────────────────────────────────────────
    eps_ck = torch.load(args.eps_ckpt, map_location=device, weights_only=False)
    ea     = eps_ck.get("args", {})
    eps_model = UNet(in_ch=2, base_ch=ea.get("base_ch", 64),
                     time_dim=ea.get("time_dim", 256)).to(device)
    eps_model.load_state_dict(eps_ck["model"])
    eps_model.eval()
    eps_diff = DDPM(T=ea.get("T", 1000), beta_schedule=ea.get("schedule","cosine"),
                    device=device)
    print(f"eps  epoch={eps_ck.get('epoch','?')}  val_loss={eps_ck.get('val_loss',float('nan')):.5f}")

    # ── Load conditioned models ─────────────────────────────────────────────
    cond_models = {}
    cond_diffs  = {}
    for cond_name, cond_in_ch in [("voronoi", 3), ("path", 1)]:
        ck_path = os.path.join(_here, f"checkpoints_{cond_name}",
                               f"best_cond_ddpm_{cond_name}_cosine.pt")
        if not os.path.exists(ck_path):
            print(f"  SKIP {cond_name}: checkpoint not found at {ck_path}")
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        ca = ck.get("args", {})
        m  = CondUNet(in_ch=2, cond_in_ch=cond_in_ch,
                      base_ch=ca.get("base_ch",64),
                      time_dim=ca.get("time_dim",256),
                      cond_dim=ca.get("cond_dim",256)).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        cond_models[cond_name] = m
        cond_diffs[cond_name]  = CondDDPM(T=ca.get("T",1000),
                                           beta_schedule=ca.get("schedule","cosine"),
                                           device=device)
        print(f"{cond_name}  epoch={ck.get('epoch','?')}  val_loss={ck.get('val_loss',float('nan')):.5f}")

    voronoi_layer = VoronoiLayer(H=H, W=W, n_sensors=args.path_steps).to(device)

    # ── Evaluate ────────────────────────────────────────────────────────────
    results = {name: [] for name in ["eps"] + list(cond_models.keys())}

    print(f"\n{'Sample':>8}  {'Seed':>4}  {'eps':>8}  " +
          "  ".join(f"{n:>10}" for n in cond_models))
    print("-" * (28 + 12 * len(cond_models)))

    cond_preds_all   = {n: [] for n in list(cond_models.keys())}
    voronoi_grids_all = []

    for run_i, (sample_idx, seed) in enumerate(zip(SAMPLES, SEEDS)):
        x0_true = val_ds[sample_idx].to(device)   # (2, H, W)

        # Build robot path
        path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)
        rows, cols = np.where(path_mask)

        # ── Voronoi tessellation (needed for both voronoi-cond and row-2 plot) ──
        rows_n = torch.tensor(rows, dtype=torch.float32, device=device) / (H-1) * 2 - 1
        cols_n = torch.tensor(cols, dtype=torch.float32, device=device) / (W-1) * 2 - 1
        sensor_pos  = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)
        flat_idx    = torch.tensor(rows * W + cols, dtype=torch.long, device=device)
        flat_idx_b  = flat_idx.unsqueeze(0).unsqueeze(0).expand(1, 2, len(rows))
        sensor_vals = torch.gather(
            x0_true.unsqueeze(0).reshape(1, 2, H*W), 2, flat_idx_b
        )
        with torch.no_grad():
            voronoi_grid = voronoi_layer.tessellate(sensor_vals, sensor_pos)  # (1,3,H,W)
        voronoi_np = voronoi_grid[0].cpu().numpy()   # (3, H, W)
        voronoi_grids_all.append(voronoi_np)

        # ── eps RePaint ───────────────────────────────────────────────────────
        path_t  = torch.from_numpy(path_mask).to(device)
        x0_obs  = x0_true.clone()
        x0_obs[:, ~path_t] = 0.0
        with torch.no_grad():
            eps_pred = repaint(eps_model, eps_diff, x0_obs, path_mask,
                               land_mask_np, r=args.resample, device=device)
        rmse_eps = float(np.sqrt(np.mean(
            (eps_pred[0].numpy() - x0_true[0].cpu().numpy())[~land_mask_np]**2 +
            (eps_pred[1].numpy() - x0_true[1].cpu().numpy())[~land_mask_np]**2
        )))
        results["eps"].append(rmse_eps)

        # ── Conditioned RePaint ───────────────────────────────────────────────
        cond_rmses = {}
        cond_pred_this = {}
        for cond_name, cond_model in cond_models.items():
            cond_diff = cond_diffs[cond_name]

            if cond_name == "voronoi":
                cond = voronoi_grid
            else:  # path
                path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)
                cond = path_ch[None, None]   # (1, 1, H, W)

            x0_known_b  = x0_true.clone().unsqueeze(0)
            x0_known_b[0, :, ~path_t] = 0.0
            path_mask_t  = path_t[None, None].float()
            ocean_mask_t = torch.from_numpy(
                (~land_mask_np).astype(np.float32)).to(device)[None, None]

            with torch.no_grad():
                pred = cond_diff.repaint(
                    cond_model, cond, x0_known_b, path_mask_t, ocean_mask_t,
                    r=args.resample
                )[0]

            pred_np = pred.cpu().numpy()
            rmse = float(np.sqrt(np.mean(
                (pred_np[0] - x0_true[0].cpu().numpy())[~land_mask_np]**2 +
                (pred_np[1] - x0_true[1].cpu().numpy())[~land_mask_np]**2
            )))
            results[cond_name].append(rmse)
            cond_rmses[cond_name] = rmse
            cond_pred_this[cond_name] = pred_np
            cond_preds_all[cond_name].append(pred_np)

        row = f"{sample_idx:>8}  {seed:>4}  {rmse_eps:>8.4f}  " + \
              "  ".join(f"{cond_rmses[n]:>10.4f}" for n in cond_models)
        print(row)

        # ── Save 3×3 plot ────────────────────────────────────────────────────
        if "path" in cond_pred_this and "voronoi" in cond_pred_this:
            out_path = os.path.join(args.out_dir, f"grid_{run_i+1:02d}_idx{sample_idx}.png")
            save_grid_plot(
                sample_idx, seed,
                x0_true[0].cpu().numpy(), x0_true[1].cpu().numpy(),
                eps_pred.numpy(), cond_pred_this["path"], cond_pred_this["voronoi"],
                path_mask, voronoi_np,
                land_mask_np,
                rmse_eps, cond_rmses["path"], cond_rmses["voronoi"],
                out_path,
            )
            print(f"           -> {out_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * (28 + 12 * len(cond_models)))
    all_names = ["eps"] + list(cond_models.keys())
    header = f"{'':>8}  {'':>4}  " + "  ".join(f"{n:>8}" for n in all_names)
    print(header)
    means = [np.mean(results[n]) for n in all_names]
    stds  = [np.std(results[n])  for n in all_names]
    print("  ".join(f"{m:>10.4f}" for m in means) + "  (mean)")
    print("  ".join(f"{s:>10.4f}" for s in stds)  + "  (std)")
    print()
    for i, name in enumerate(all_names):
        print(f"  {name:12s}: {means[i]:.4f} ± {stds[i]:.4f}")

    # ── Summary bar chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    xs     = np.arange(1, len(SAMPLES) + 1)
    width  = 0.25
    colors = {"eps": "steelblue", "path": "darkorange", "voronoi": "seagreen"}
    for i, name in enumerate(all_names):
        ax.bar(xs + (i - 1) * width, results[name], width,
               label=f"{name} (μ={means[i]:.4f})", color=colors.get(name, "gray"))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{i}" for i in SAMPLES], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Val sample"); ax.set_ylabel("RMSE")
    ax.set_title("eps vs path-cond vs voronoi-cond — per-sample RMSE")
    ax.legend()
    plt.tight_layout()
    summary_path = os.path.join(args.out_dir, "rmse_summary.png")
    fig.savefig(summary_path, dpi=130)
    plt.close(fig)
    print(f"\nPlots saved to: {args.out_dir}")
    print(f"Summary bar chart: {summary_path}")


if __name__ == "__main__":
    main()
