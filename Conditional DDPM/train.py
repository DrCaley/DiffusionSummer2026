"""
Training script for FiLM-conditioned DDPM on ocean current fields.

Four conditioning modes (--cond):

    voronoi     (cond_in_ch=3)
        Condition on the raw Voronoi tessellation computed from biased-walk
        sensor positions.  Channels: [u_voronoi, v_voronoi, sensor_mask].
        The model learns to refine the Voronoi reconstruction by denoising
        toward the ground-truth field.

    path        (cond_in_ch=1)
        Condition on the binary path mask only: which grid cells the robot
        visited.  Channel: [path_mask].  The model sees the sampling geometry
        but NOT the measured values — it must infer the full field from
        structural priors and the observation locations.

    path_field  (cond_in_ch=3)
        Condition on the actual vector field values measured along the path.
        Channels: [u_path, v_path, path_mask].  u_path and v_path contain
        the ground-truth u/v components at every visited cell and are 0
        elsewhere.  The model sees sparse direct observations rather than
        an interpolated or mask-only representation.

    both        (cond_in_ch=4)
        Condition on both: [u_voronoi, v_voronoi, sensor_mask, path_mask].
        Maximum information: interpolated values AND explicit path shape.
        The sensor_mask (Voronoi channel 3) and path_mask differ slightly —
        sensor_mask marks the nearest-sensor cell per Voronoi region while
        path_mask marks every cell the robot walked through.

Conditioning is generated on-the-fly each training step using the
biased_walk_path() function from paths.py.  Every sample in a batch gets
a different random path (seed = epoch * 1e5 + batch_idx * 1e3 + sample_idx)
so the model trains over a large distribution of observation geometries.

Usage (run from workspace root):
    python "Conditional DDPM/train.py" --cond voronoi
    python "Conditional DDPM/train.py" --cond path   --epochs 150
    python "Conditional DDPM/train.py" --cond both   --lr 1e-4 --batch 16
    python "Conditional DDPM/train.py" --cond voronoi --path_steps 200
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
_here      = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_here, ".."))
_voronoi_model_dir = os.path.join(_repo_root, "Voronoi", "model")

sys.path.insert(0, _here)            # Conditional DDPM/  (model.py, diffusion.py)
sys.path.insert(0, _repo_root)       # workspace root     (dataset.py, paths.py)
sys.path.insert(0, _voronoi_model_dir)  # Voronoi/model/  (voronoi_model.py)

from dataset        import OceanCurrentDataset
from paths          import biased_walk_path
from voronoi_model  import VoronoiLayer
from cond_model     import CondUNet
from cond_diffusion import CondDDPM


# ---------------------------------------------------------------------------
# Conditioning channel counts
# ---------------------------------------------------------------------------
COND_MODES = {
    "voronoi":    3,   # [u_vor, v_vor, sensor_mask]
    "path":       1,   # [path_mask]
    "path_field": 3,   # [u_path, v_path, path_mask]
    "both":       4,   # [u_vor, v_vor, sensor_mask, path_mask]
}


# ---------------------------------------------------------------------------
# On-the-fly condition generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _make_cond(
    x0:             torch.Tensor,        # (B, 2, H, W) clean field
    land_mask_np:   np.ndarray,          # (H, W) bool
    voronoi_layer:  VoronoiLayer | None, # pre-built; None for path-only
    cond_mode:      str,
    n_steps:        int,
    epoch:          int,
    batch_idx:      int,
    device:         str,
) -> torch.Tensor:
    """
    Build the conditioning tensor for one batch.

    Each sample gets a fresh biased-walk path drawn with a deterministic
    per-sample seed so results are reproducible and diverse across steps.

    Returns:
        cond: (B, cond_in_ch, H, W) on `device`
    """
    B, C, H, W = x0.shape
    cond_list = []

    for b in range(B):
        seed      = epoch * 100_000 + batch_idx * 1_000 + b
        path_mask = biased_walk_path(land_mask_np, n_steps=n_steps, seed=seed)
        rows, cols = np.where(path_mask)
        K = len(rows)

        if cond_mode in ("voronoi", "both"):
            # ---- Voronoi tessellation ----
            rows_n = (
                torch.tensor(rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
            )
            cols_n = (
                torch.tensor(cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
            )
            sensor_pos  = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)  # (1, K, 2)

            flat_idx    = torch.tensor(rows * W + cols, dtype=torch.long, device=device)
            flat_idx    = flat_idx.unsqueeze(0).unsqueeze(0).expand(1, C, K)
            sensor_vals = torch.gather(
                x0[b : b + 1].reshape(1, C, H * W), 2, flat_idx
            )  # (1, C, K)

            voronoi_grid = voronoi_layer.tessellate(sensor_vals, sensor_pos)  # (1, 3, H, W)

        if cond_mode == "voronoi":
            cond_list.append(voronoi_grid)

        elif cond_mode == "path":
            path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)
            cond_list.append(path_ch.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)

        elif cond_mode == "path_field":
            path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)  # (H, W)
            u_path  = x0[b, 0] * path_ch   # ground-truth u at path cells, 0 elsewhere
            v_path  = x0[b, 1] * path_ch   # ground-truth v at path cells, 0 elsewhere
            cond_list.append(
                torch.stack([u_path, v_path, path_ch], dim=0).unsqueeze(0)  # (1, 3, H, W)
            )

        else:  # "both"
            path_ch = torch.from_numpy(path_mask.astype(np.float32)).to(device)
            path_ch = path_ch.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)
            cond_list.append(torch.cat([voronoi_grid, path_ch], dim=1))  # (1, 4, H, W)

    return torch.cat(cond_list, dim=0)   # (B, cond_in_ch, H, W)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a FiLM-conditioned DDPM on ocean current fields."
    )
    p.add_argument("--cond",       required=True, choices=list(COND_MODES),
                   help="Conditioning mode: voronoi | path | both")
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--epochs",     type=int,   default=400)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--time_dim",   type=int,   default=256)
    p.add_argument("--cond_dim",   type=int,   default=256,
                   help="Dimension of the FiLM condition embedding.")
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="Std of forward-process noise (set to ~0.12 to match data scale).")
    p.add_argument("--schedule",   default="cosine", choices=["cosine", "linear"])
    p.add_argument("--path_steps", type=int,   default=150,
                   help="Number of biased-walk steps for sensor path generation.")
    p.add_argument("--save_dir",   default=None,
                   help="Checkpoint directory. Defaults to "
                        "'Conditional DDPM/checkpoints_{cond}/'.")
    p.add_argument("--workers",    type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond_in_ch = COND_MODES[args.cond]

    if args.save_dir is None:
        args.save_dir = os.path.join(_here, f"checkpoints_{args.cond}")

    print(f"Device       : {device}")
    print(f"Cond mode    : {args.cond}  ({cond_in_ch} conditioning channels)")
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
    # Used for voronoi and both modes; None for path-only.
    voronoi_layer: VoronoiLayer | None = None
    if args.cond in ("voronoi", "both"):
        voronoi_layer = VoronoiLayer(H=H, W=W, n_sensors=args.path_steps).to(device)

    # ---- Model ---------------------------------------------------------------
    model = CondUNet(
        in_ch      = 2,
        cond_in_ch = cond_in_ch,
        base_ch    = args.base_ch,
        time_dim   = args.time_dim,
        cond_dim   = args.cond_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params : {n_params:,}")

    # ---- Diffusion -----------------------------------------------------------
    diffusion = CondDDPM(T=args.T, beta_schedule=args.schedule, device=device,
                         noise_scale=args.noise_scale)

    # ---- Optimiser -----------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    run_tag  = f"cond_ddpm_{args.cond}_{args.schedule}"
    best_val = float("inf")

    # ---- Training loop -------------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        # -- Train --
        model.train()
        train_loss = 0.0
        for batch_idx, x0 in enumerate(train_loader):
            x0 = x0.to(device)
            cond = _make_cond(
                x0, land_mask_np, voronoi_layer,
                args.cond, args.path_steps, epoch, batch_idx, device,
            )
            loss = diffusion.training_loss(model, x0, land_mask, cond)
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
                cond = _make_cond(
                    x0, land_mask_np, voronoi_layer,
                    args.cond, args.path_steps, epoch, batch_idx, device,
                )
                loss = diffusion.training_loss(model, x0, land_mask, cond)
                val_loss += loss.item()
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
