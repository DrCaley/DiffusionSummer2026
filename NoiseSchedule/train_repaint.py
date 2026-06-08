"""
Training script for noise-schedule ablation study.

Trains the Repaint UNet (identical architecture to DDPM/model.py UNet) under
four different noise schedules: linear, cosine, quadratic, sigmoid.

Usage (run from workspace root):
    python3 NoiseSchedule/train_repaint.py --schedule cosine
    python3 NoiseSchedule/train_repaint.py --schedule linear --epochs 100 --batch 32
    python3 NoiseSchedule/train_repaint.py --schedule quadratic
    python3 NoiseSchedule/train_repaint.py --schedule sigmoid
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from dataset        import OceanCurrentDataset
from diffusion      import DDPM
from repaint_model  import Repaint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Repaint UNet for noise-schedule ablation."
    )
    p.add_argument("--pickle",   default="data.pickle")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--base_ch",  type=int,   default=64)
    p.add_argument("--time_dim", type=int,   default=256)
    p.add_argument("--T",        type=int,   default=1000)
    p.add_argument("--schedule", default="cosine",
                   choices=["linear", "cosine", "quadratic", "sigmoid", "geometric"],
                   help="Noise schedule to use for training.")
    p.add_argument("--save_dir", default=None,
                   help="Checkpoint directory. Defaults to "
                        "NoiseSchedule/checkpoints_repaint_{schedule}.")
    p.add_argument("--workers",  type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(__file__)
    if args.save_dir is None:
        args.save_dir = os.path.join(
            script_dir, f"checkpoints_repaint_{args.schedule}"
        )

    print(f"Device   : {device}")
    print(f"Schedule : {args.schedule}")
    print(f"Save dir : {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    land_mask = train_ds.land_mask.to(device)   # (H, W) bool

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model + diffusion ----
    model = Repaint(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    diffusion = DDPM(T=args.T, beta_schedule=args.schedule, device=device)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Training loop ----
    best_val  = float("inf")
    best_name = f"best_model_{args.schedule}.pt"

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        for x0 in train_loader:
            x0 = x0.to(device)
            loss = diffusion.training_loss(model, x0, land_mask)
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
            for x0 in val_loader:
                x0 = x0.to(device)
                val_loss += diffusion.training_loss(model, x0, land_mask).item()
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
                    f"ckpt_{args.schedule}_epoch{epoch:04d}.pt"
                ),
            )

    print(f"\nTraining complete.  Best val loss: {best_val:.5f}")
    print(f"Best checkpoint : {args.save_dir}/{best_name}")


if __name__ == "__main__":
    main()
