"""
Voronoi-primed DDPM training.

Instead of learning to denoise clean ocean fields, this DDPM is trained to
denoise *Voronoi-reconstructed* fields.  At every training step:

  1. Load a batch of clean fields x0 from the dataset.
  2. Run the frozen Voronoi model (walk-path sensors) on x0 -> x_voronoi.
  3. Use x_voronoi as the diffusion target (x0 in the forward process).
  4. Train the DDPM to predict the noise added to x_voronoi.

At inference this model is used with SDEdit initialised from x_voronoi:
the DDPM now knows the distribution of Voronoi fields at every noise level,
so denoising a noisy Voronoi estimate no longer fights against the prior.

The ceiling of this approach is Voronoi performance -- the model can only
learn to reconstruct Voronoi-quality fields, not cleaner ground truth.
For ground-truth quality, see Direction 1 (conditional diffusion).

Usage (run from workspace root):
    python3 NoiseSchedule/train_voronoi_primed.py --schedule quadratic
    python3 NoiseSchedule/train_voronoi_primed.py --schedule quadratic --epochs 100
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Voronoi"))

from dataset        import OceanCurrentDataset
from diffusion      import DDPM
from repaint_model  import Repaint
from voronoi_model  import VoronoiNet
from repaint_infer  import biased_walk_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train DDPM on Voronoi-reconstructed fields."
    )
    p.add_argument("--pickle",       default="data.pickle")
    p.add_argument("--voronoi_ckpt", default="Voronoi/checkpoints_voronoi_walk/best_model_walk.pt",
                   help="Path to frozen Voronoi checkpoint.")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch",        type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--base_ch",      type=int,   default=64)
    p.add_argument("--time_dim",     type=int,   default=256)
    p.add_argument("--T",            type=int,   default=1000)
    p.add_argument("--schedule",     default="quadratic",
                   choices=["linear", "cosine", "quadratic", "sigmoid"])
    p.add_argument("--path_steps",   type=int,   default=150,
                   help="Robot walk length used to generate Voronoi inputs.")
    p.add_argument("--save_dir",     default=None,
                   help="Checkpoint directory. Defaults to "
                        "NoiseSchedule/checkpoints_voronoi_primed_{schedule}.")
    p.add_argument("--workers",      type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Voronoi inference for a single batch  (no grad, frozen model)
# ---------------------------------------------------------------------------

@torch.no_grad()
def voronoi_forward(voronoi_model, x0, land_mask_np, path_steps, epoch, batch_idx, device):
    """
    For each sample in the batch, generate a biased-walk path and run Voronoi.
    Each sample gets a different path (different seed) for training diversity.

    Returns x_voronoi: (B, 2, H, W) Voronoi-reconstructed fields.
    """
    B, C, H, W = x0.shape
    x_voronoi_list = []

    for b in range(B):
        seed = epoch * 100000 + batch_idx * 1000 + b
        path_mask = biased_walk_path(land_mask_np, n_steps=path_steps, seed=seed)
        rows, cols = np.where(path_mask)
        K = len(rows)

        rows_n = torch.tensor(rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
        cols_n = torch.tensor(cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
        sensor_pos  = torch.stack([rows_n, cols_n], dim=1).unsqueeze(0)  # (1, K, 2)

        flat_idx    = torch.tensor(
            rows * W + cols, dtype=torch.long, device=device
        ).unsqueeze(0).unsqueeze(0).expand(1, C, K)
        sensor_vals = torch.gather(x0[b:b+1].reshape(1, C, H * W), 2, flat_idx)  # (1, C, K)

        voronoi_grid = voronoi_model.voronoi.tessellate(sensor_vals, sensor_pos)
        x_voronoi_list.append(voronoi_model.unet(voronoi_grid))             # (1, 2, H, W)

    return torch.cat(x_voronoi_list, dim=0)   # (B, 2, H, W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(__file__)
    if args.save_dir is None:
        args.save_dir = os.path.join(
            script_dir, f"checkpoints_voronoi_primed_{args.schedule}"
        )

    print(f"Device   : {device}")
    print(f"Schedule : {args.schedule}")
    print(f"Save dir : {args.save_dir}")

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

    # ---- Frozen Voronoi model ------------------------------------------------
    vor_ckpt = torch.load(args.voronoi_ckpt, map_location=device, weights_only=False)
    vor_args = vor_ckpt.get("args", {})
    voronoi_model = VoronoiNet(
        H=H, W=W,
        n_sensors=args.path_steps,
        in_ch=2,
        base_ch=vor_args.get("base_ch", 64),
    ).to(device)
    voronoi_model.load_state_dict(vor_ckpt["model"])
    voronoi_model.eval()
    for p in voronoi_model.parameters():
        p.requires_grad_(False)
    print(f"Loaded Voronoi checkpoint (epoch {vor_ckpt.get('epoch', '?')}) — frozen")

    # ---- DDPM model + diffusion ----------------------------------------------
    model     = Repaint(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    diffusion = DDPM(T=args.T, beta_schedule=args.schedule, device=device)
    print(f"DDPM parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- Optimiser -----------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Training loop -------------------------------------------------------
    best_val  = float("inf")
    best_name = f"best_model_vprimed_{args.schedule}.pt"

    for epoch in range(1, args.epochs + 1):

        # -- Train --
        model.train()
        train_loss = 0.0
        for batch_idx, x0 in enumerate(train_loader):
            x0        = x0.to(device)
            x_voronoi = voronoi_forward(
                voronoi_model, x0, land_mask_np,
                args.path_steps, epoch, batch_idx, device,
            )
            loss = diffusion.training_loss(model, x_voronoi, land_mask)
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
                x0        = x0.to(device)
                x_voronoi = voronoi_forward(
                    voronoi_model, x0, land_mask_np,
                    args.path_steps, epoch, batch_idx, device,
                )
                val_loss += diffusion.training_loss(model, x_voronoi, land_mask).item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        # -- Checkpoint --
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_loss": val_loss, "args": vars(args)},
                os.path.join(args.save_dir, best_name),
            )

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                os.path.join(
                    args.save_dir,
                    f"ckpt_vprimed_{args.schedule}_epoch{epoch:04d}.pt",
                ),
            )

    print(f"\nTraining complete.  Best val loss: {best_val:.5f}")
    print(f"Best checkpoint : {args.save_dir}/{best_name}")


if __name__ == "__main__":
    main()
