"""
train.py – Training script for the Colored Gaussian Noise DDPM.

The key differences from the basic/divergence-free DDPM:
  - Noise is sampled via colored_gaussian_noise() (Fourier-filtered).
  - Everything else (UNet, cosine schedule, MSE loss, RePaint inference)
    is unchanged.

Checkpoint naming convention:
  checkpoints/model_DDPM_MSE_coloredGaussian_cosine.pt  (best val loss)
  checkpoints/epoch_{N}.pt                               (periodic)

Usage
-----
cd DDPM
python train.py --pickle ../data.pickle --epochs 1500 --batch 32
"""

import argparse
import math
import os
import signal
import sys
import time

import torch
import torch.nn as nn

# Allow importing model_parameters from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_dataloaders
from model.unet import UNet, count_parameters
from model.diffusion import q_sample
from model_parameters.noise_types import colored_gaussian_noise
from model_parameters.noise_schedules import cosine_beta_schedule
from model_parameters.loss_functions import masked_mse_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    train_loader, val_loader, _, land_mask = get_dataloaders(
        args.pickle, batch_size=args.batch, num_workers=args.workers
    )
    ocean_mask = (~land_mask).to(device)   # (H, W)  True = ocean
    n_ocean = ocean_mask.sum().item()
    print(f"Ocean pixels: {n_ocean} / {land_mask.numel()}")

    # ---- Noise schedule ----
    T = args.T
    betas, alpha_bar = cosine_beta_schedule(T)
    sqrt_alpha_bar           = alpha_bar.sqrt().to(device)
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt().to(device)

    # ---- Model ----
    model = UNet(
        in_channels=2,
        base_ch=args.base_ch,
        ch_mults=tuple(args.ch_mults),
        time_dim=args.time_dim,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    start_epoch   = 1

    # --- SIGTERM handler: save checkpoint then exit cleanly ---
    def _sigterm_handler(signum, frame):
        print("\n[SIGTERM] Saving checkpoint before exit...", flush=True)
        _save_path = os.path.join(args.ckpt_dir, "model_DDPM_MSE_coloredGaussian_cosine_interrupted.pt")
        torch.save(
            {
                "epoch": _current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": _current_val_loss,
                "args": vars(args),
            },
            _save_path,
        )
        print(f"[SIGTERM] Saved → {_save_path}", flush=True)
        sys.exit(0)

    _current_epoch     = start_epoch
    _current_val_loss  = float("inf")
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ---- Resume from checkpoint ----
    if args.resume and os.path.isfile(args.resume):
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r["model_state_dict"])
        optimizer.load_state_dict(ckpt_r["optimizer_state_dict"])
        start_epoch   = ckpt_r.get("epoch", 0) + 1
        best_val_loss = ckpt_r.get("val_loss", float("inf"))
        prev_val_loss = best_val_loss
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch - 1}  (val_loss={best_val_loss:.5f})")

    log_path = os.path.join(args.ckpt_dir, "training_log.csv")
    log_mode = "a" if (args.resume and os.path.isfile(log_path)) else "w"
    with open(log_path, log_mode) as fh:
        if log_mode == "w":
            fh.write("epoch,train_loss,val_loss,lr,elapsed_s\n")

    best_ckpt_name = "model_DDPM_MSE_coloredGaussian_cosine.pt"
    print(f"\nTraining from epoch {start_epoch} to {args.epochs} …\n")

    for epoch in range(start_epoch, args.epochs + 1):
        _current_epoch = epoch
        t0 = time.time()

        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_count    = 0

        for x0 in train_loader:
            x0 = x0.to(device)
            B  = x0.shape[0]

            t_batch = torch.randint(0, T, (B,), device=device)
            eps     = colored_gaussian_noise(x0.shape, alpha=args.noise_alpha, device=device)
            xt      = q_sample(x0, t_batch, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, eps)

            eps_pred = model(xt, t_batch)
            loss     = masked_mse_loss(eps_pred, eps, ocean_mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * B
            train_count    += B

        train_loss = train_loss_sum / train_count
        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_count    = 0

        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                B  = x0.shape[0]
                t_batch  = torch.randint(0, T, (B,), device=device)
                eps      = colored_gaussian_noise(x0.shape, alpha=args.noise_alpha, device=device)
                xt       = q_sample(x0, t_batch, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, eps)
                eps_pred = model(xt, t_batch)
                loss     = masked_mse_loss(eps_pred, eps, ocean_mask)
                val_loss_sum += loss.item() * B
                val_count    += B

        val_loss = val_loss_sum / val_count
        _current_val_loss = val_loss
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s"
        )

        with open(log_path, "a") as fh:
            fh.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{lr_now:.2e},{elapsed:.1f}\n")

        # ---- Save best ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                os.path.join(args.ckpt_dir, best_ckpt_name),
            )
            print(f"  → best model saved  (val={val_loss:.5f})")

        # ---- Periodic checkpoint (every 100 epochs) ----
        if epoch % args.ckpt_freq == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                os.path.join(args.ckpt_dir, f"epoch_{epoch}.pt"),
            )

        # ---- Convergence check ----
        if epoch > 1 and prev_val_loss > 0:
            rel_change = abs(val_loss - prev_val_loss) / prev_val_loss
            if rel_change < args.convergence_tol and epoch >= args.min_epochs:
                conv_ckpt = os.path.join(args.ckpt_dir, "model_DDPM_MSE_coloredGaussian_cosine_converged.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "args": vars(args),
                    },
                    conv_ckpt,
                )
                print(
                    f"\nConverged at epoch {epoch}: "
                    f"relative val-loss change = {rel_change:.2e} < {args.convergence_tol}"
                )
                print(f"  → convergence checkpoint saved: {conv_ckpt}")
                break

        prev_val_loss = val_loss

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Colored Gaussian Noise DDPM")
    p.add_argument("--pickle",          default="../data.pickle")
    p.add_argument("--epochs",          type=int,   default=1500)
    p.add_argument("--batch",           type=int,   default=32)
    p.add_argument("--T",               type=int,   default=1000)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--base_ch",         type=int,   default=128)
    p.add_argument("--ch_mults",        type=int,   nargs="+", default=[1, 2, 2])
    p.add_argument("--time_dim",        type=int,   default=512)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--noise_alpha",     type=float, default=2.0,
                   help="Spectral exponent for colored noise (0=white, 1=pink, 2=red)")
    p.add_argument("--workers",         type=int,   default=4)
    p.add_argument("--ckpt_dir",        default="checkpoints")
    p.add_argument("--ckpt_freq",       type=int,   default=100,
                   help="Save a periodic checkpoint every N epochs")
    p.add_argument("--convergence_tol", type=float, default=1e-4,
                   help="Relative val-loss change threshold for early stop")
    p.add_argument("--min_epochs",      type=int,   default=100,
                   help="Minimum epochs before convergence check activates")
    p.add_argument("--resume",          default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
