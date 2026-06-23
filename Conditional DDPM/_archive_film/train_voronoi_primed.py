"""
Training script for the Voronoi-Primed Conditional DDPM.

Conditioning mode: path_field (hard-coded)
    cond = [u_path, v_path, path_mask]  —  (B, 3, H, W)
    Sparse ground-truth measurements at the robot path cells.

Forward process: starts from the Voronoi tessellation of the same path
    x_t = sqrt(ᾱ_t) · x_voronoi + sqrt(1−ᾱ_t) · ε

Loss: adjusted epsilon prediction that recovers ground-truth x0
    ε_target = (x_t − sqrt(ᾱ_t) · x0_gt) / sqrt(1−ᾱ_t)
    L = MSE(model(x_t, t, cond), ε_target)  on ocean pixels only

Both cond and x_voronoi are derived from the same biased-walk path per sample,
so the model receives consistent spatial information from both sources.

Usage (run from workspace root):
    python "Conditional DDPM/train_voronoi_primed.py"
    python "Conditional DDPM/train_voronoi_primed.py" --epochs 400 --batch 16
    python "Conditional DDPM/train_voronoi_primed.py" --schedule cosine --path_steps 200
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup — allow imports from workspace root and Voronoi/model/
# ---------------------------------------------------------------------------
_here              = os.path.dirname(os.path.abspath(__file__))
_repo_root         = os.path.normpath(os.path.join(_here, ".."))
_voronoi_model_dir = os.path.join(_repo_root, "Voronoi", "model")

sys.path.insert(0, _here)               # cond_model.py, cond_diffusion_vp.py
sys.path.insert(0, _repo_root)          # dataset.py, paths.py
sys.path.insert(0, _voronoi_model_dir)  # voronoi_model.py

from dataset            import OceanCurrentDataset
from paths              import biased_walk_path
from voronoi_model      import VoronoiLayer
from cond_model         import CondUNet
from cond_diffusion_vp  import CondDDPMVP


# ---------------------------------------------------------------------------
# On-the-fly condition + Voronoi generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _make_cond_and_voronoi(
    x0:            torch.Tensor,      # (B, 2, H, W) clean fields
    land_mask_np:  np.ndarray,        # (H, W) bool
    voronoi_layer: VoronoiLayer,
    n_steps:       int,
    epoch:         int,
    batch_idx:     int,
    device:        str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the path_field conditioning tensor and the Voronoi field for one batch.

    Both are derived from the same biased-walk path per sample, ensuring the
    model receives consistent spatial information during training.

    Returns:
        cond:      (B, 3, H, W)  [u_path, v_path, path_mask]
        x_voronoi: (B, 2, H, W)  Voronoi u, v channels (sensor mask stripped)
    """
    B, C, H, W = x0.shape
    cond_list    = []
    voronoi_list = []

    for b in range(B):
        seed      = epoch * 100_000 + batch_idx * 1_000 + b
        path_mask = biased_walk_path(land_mask_np, n_steps=n_steps, seed=seed)
        rows, cols = np.where(path_mask)
        K = len(rows)

        # ---- path_field conditioning: sparse direct observations ----
        path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)  # (H, W)
        u_path  = x0[b, 0] * path_ch
        v_path  = x0[b, 1] * path_ch
        cond_list.append(
            torch.stack([u_path, v_path, path_ch], dim=0).unsqueeze(0)   # (1, 3, H, W)
        )

        # ---- Voronoi tessellation from the same path sensors ----
        rows_n = (
            torch.tensor(rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
        )
        cols_n = (
            torch.tensor(cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
        )
        sensor_pos  = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)   # (1, K, 2)

        flat_idx    = torch.tensor(rows * W + cols, dtype=torch.long, device=device)
        flat_idx    = flat_idx.unsqueeze(0).unsqueeze(0).expand(1, C, K)
        sensor_vals = torch.gather(
            x0[b : b + 1].reshape(1, C, H * W), 2, flat_idx
        )  # (1, C, K)

        voronoi_full = voronoi_layer.tessellate(sensor_vals, sensor_pos)  # (1, C+1, H, W)
        voronoi_uv   = voronoi_full[:, :C, :, :]                          # (1, 2, H, W)
        voronoi_list.append(voronoi_uv)

    cond      = torch.cat(cond_list,    dim=0)   # (B, 3, H, W)
    x_voronoi = torch.cat(voronoi_list, dim=0)   # (B, 2, H, W)
    return cond, x_voronoi


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Voronoi-Primed FiLM-conditioned DDPM (path_field mode)."
    )
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--epochs",      type=int,   default=400)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--time_dim",    type=int,   default=256)
    p.add_argument("--cond_dim",    type=int,   default=256)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--schedule",    default="cosine", choices=["cosine", "linear"])
    p.add_argument("--path_steps",  type=int,   default=150)
    p.add_argument("--save_dir",    default=None,
                   help="Checkpoint directory. Defaults to "
                        "'Conditional DDPM/checkpoints_vp_path_field/'.")
    p.add_argument("--workers",     type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_dir is None:
        args.save_dir = os.path.join(_here, "checkpoints_vp_path_field")

    print(f"Device       : {device}")
    print(f"Cond mode    : path_field  (3 conditioning channels)")
    print(f"Forward proc : Voronoi-primed  (x_t from Voronoi, loss vs x0_gt)")
    print(f"Schedule     : {args.schedule}")
    print(f"Path steps   : {args.path_steps}")
    print(f"Save dir     : {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data ----------------------------------------------------------------
    train_ds     = OceanCurrentDataset(args.pickle, split=0)
    val_ds       = OceanCurrentDataset(args.pickle, split=1)
    land_mask    = train_ds.land_mask.to(device)
    land_mask_np = train_ds.land_mask.numpy()
    H, W = train_ds.data.shape[2], train_ds.data.shape[3]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Voronoi tessellation layer (no learnable weights) -------------------
    voronoi_layer = VoronoiLayer(H=H, W=W, n_sensors=args.path_steps).to(device)

    # ---- Model ---------------------------------------------------------------
    model = CondUNet(
        in_ch      = 2,
        cond_in_ch = 3,       # path_field: [u_path, v_path, path_mask]
        base_ch    = args.base_ch,
        time_dim   = args.time_dim,
        cond_dim   = args.cond_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params : {n_params:,}")

    # ---- Diffusion -----------------------------------------------------------
    diffusion = CondDDPMVP(
        T             = args.T,
        beta_schedule = args.schedule,
        device        = device,
        noise_scale   = args.noise_scale,
    )

    # ---- Optimiser -----------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    run_tag  = f"cond_ddpm_vp_path_field_{args.schedule}"
    best_val = float("inf")

    # ---- Training loop -------------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        # -- Train --
        model.train()
        train_loss = 0.0
        for batch_idx, x0 in enumerate(train_loader):
            x0 = x0.to(device)
            cond, x_voronoi = _make_cond_and_voronoi(
                x0, land_mask_np, voronoi_layer,
                args.path_steps, epoch, batch_idx, device,
            )
            loss = diffusion.training_loss(model, x0, x_voronoi, land_mask, cond)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, x0 in enumerate(val_loader):
                x0 = x0.to(device)
                cond, x_voronoi = _make_cond_and_voronoi(
                    x0, land_mask_np, voronoi_layer,
                    args.path_steps, epoch, batch_idx, device,
                )
                val_loss += diffusion.training_loss(
                    model, x0, x_voronoi, land_mask, cond
                ).item()
        val_loss /= len(val_loader)

        scheduler.step()

        # -- Checkpoint --
        saved_best = False
        if val_loss < best_val:
            best_val   = val_loss
            saved_best = True
            torch.save(
                {
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "val_loss": val_loss,
                    "args":     vars(args),
                },
                os.path.join(args.save_dir, f"best_{run_tag}.pt"),
            )

        if epoch % 10 == 0 or saved_best:
            tag = " *" if saved_best else ""
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={train_loss:.5f} | val={val_loss:.5f}{tag}"
            )

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: {args.save_dir}/best_{run_tag}.pt")


if __name__ == "__main__":
    main()
