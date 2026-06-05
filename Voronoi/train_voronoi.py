"""
Training script for VoronoiNet -- sparse-sensor field reconstruction via
Voronoi tessellation-assisted deep learning.

Reference: Fukami et al. (2021), https://arxiv.org/abs/2101.00554

Usage
-----
    py "Voronoi/train_voronoi.py"
    py "Voronoi/train_voronoi.py" --epochs 200 --batch 32 --n_sensors 50
    py "Voronoi/train_voronoi.py" --sensor_mode walk --path_steps 150 --save_dir Voronoi/checkpoints_voronoi_walk
    py "Voronoi/train_voronoi.py" --n_sensors 20 --base_ch 32

Run from the workspace root so that dataset.py is on the Python path.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# dataset.py lives one level above this script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import OceanCurrentDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DDPM"))
from repaint_infer import biased_walk_path

from voronoi_model import VoronoiNet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train VoronoiNet for ocean current field reconstruction."
    )
    p.add_argument("--pickle",    default="data.pickle",
                   help="Path to data.pickle (relative to cwd or absolute).")
    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--base_ch",   type=int,   default=64)
    p.add_argument("--n_sensors", type=int,   default=50,
                   help="Number of sparse sensors sampled per training step (scattered mode).")
    p.add_argument("--sensor_mode", default="scattered", choices=["scattered", "walk"],
                   help="'scattered' = random points; 'walk' = biased robot path.")
    p.add_argument("--path_steps", type=int,   default=150,
                   help="Robot walk length when --sensor_mode walk.")
    p.add_argument("--save_dir",  default="Voronoi/checkpoints_voronoi_scattered")
    p.add_argument("--workers",   type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data ----------------------------------------------------------------
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    H = train_ds.data.shape[2]
    W = train_ds.data.shape[3]

    land_mask  = train_ds.land_mask.to(device)              # (H, W) bool
    ocean_mask = (~land_mask).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model ---------------------------------------------------------------
    model = VoronoiNet(
        H=H, W=W,
        n_sensors=args.n_sensors,
        in_ch=2,
        base_ch=args.base_ch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VoronoiNet parameters : {n_params:,}")
    print(f"Grid size             : {H} x {W}")
    print(f"Sensor mode           : {args.sensor_mode}")
    if args.sensor_mode == "walk":
        print(f"Path steps            : {args.path_steps}")
    else:
        print(f"Sensors per step      : {args.n_sensors}")

    # ---- Optimiser -----------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Training loop -------------------------------------------------------
    best_val  = float("inf")
    best_name = f"best_model_{args.sensor_mode}.pt"
    land_mask_np = train_ds.land_mask.numpy()  # for walk-path generation

    def forward_batch(x0, epoch_idx, batch_idx, split="train"):
        """Run one forward pass using the configured sensor mode."""
        if args.sensor_mode == "walk":
            seed = epoch_idx * (100000 if split == "train" else 99999) + batch_idx
            path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)
            rows, cols = path_mask.nonzero()
            K = len(rows)
            rows_n = torch.tensor(rows, dtype=torch.float32, device=device) / (H - 1) * 2 - 1
            cols_n = torch.tensor(cols, dtype=torch.float32, device=device) / (W - 1) * 2 - 1
            sensor_pos = torch.stack([rows_n, cols_n], dim=1)          # (K, 2)
            B = x0.shape[0]
            flat_idx = torch.tensor(
                rows * W + cols, dtype=torch.long, device=device
            ).unsqueeze(0).unsqueeze(0).expand(B, 2, K)
            sensor_vals = torch.gather(x0.reshape(B, 2, H * W), 2, flat_idx)
            voronoi = model.voronoi.tessellate(
                sensor_vals, sensor_pos.unsqueeze(0).expand(B, K, 2)
            )
            return model.unet(voronoi)
        else:
            return model(x0, n_sensors=args.n_sensors, land_mask=land_mask)

    for epoch in range(1, args.epochs + 1):

        # -- Train --
        model.train()
        train_loss = 0.0
        for batch_idx, x0 in enumerate(train_loader):
            x0   = x0.to(device)
            pred = forward_batch(x0, epoch, batch_idx, split="train")
            loss = F.mse_loss(pred * ocean_mask, x0 * ocean_mask)
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
                x0   = x0.to(device)
                pred = forward_batch(x0, epoch, batch_idx, split="val")
                val_loss += F.mse_loss(pred * ocean_mask, x0 * ocean_mask).item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        # -- Checkpoint --
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "val_loss":  val_loss,
                    "args":      vars(args),
                },
                os.path.join(args.save_dir, best_name),
            )

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                os.path.join(args.save_dir, f"ckpt_{args.sensor_mode}_epoch{epoch:04d}.pt"),
            )

    print(f"Training complete.  Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
