"""
Train the Repaint UNet with M2+S2 tidal phase conditioning.

Conditioning features (4-dim): [sin_M2, cos_M2, sin_S2, cos_S2]
These are precomputed in data_tidal.pickle and injected into the UNet
via a small MLP added to the diffusion timestep embedding.

Usage (from /root/stjohn_tidal/):
    python -u train.py \
        --pickle /root/stjohn_tidal/data_tidal.pickle \
        --schedule linear --epochs 300 --patience 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from dataset       import OceanCurrentDataset
from diffusion     import DDPM, CURL_DIV_WEIGHT
from repaint_model import Repaint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",   default="data_tidal.pickle")
    p.add_argument("--schedule", default="linear",
                   choices=["linear", "cosine", "geometric"])
    p.add_argument("--epochs",   type=int,   default=300)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--base_ch",  type=int,   default=64)
    p.add_argument("--time_dim", type=int,   default=256)
    p.add_argument("--T",        type=int,   default=1000)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--resume",   default=None)
    p.add_argument("--workers",  type=int,   default=0)
    p.add_argument("--patience", type=int,   default=50)
    p.add_argument("--curl_div_weight", type=float, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.save_dir is None:
        args.save_dir = os.path.join(script_dir, f"checkpoints_{args.schedule}")
    os.makedirs(args.save_dir, exist_ok=True)

    curl_div_w = CURL_DIV_WEIGHT if args.curl_div_weight is None else args.curl_div_weight

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    cond_dim  = train_ds.feat_dim
    land_mask = train_ds.land_mask.to(device)

    ocean_pixels = train_ds.data[:, :, ~train_ds.land_mask]
    noise_std    = float(ocean_pixels.std())

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=(device=="cuda"))

    # ── Model ─────────────────────────────────────────────────────────
    model = Repaint(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim,
                    cond_dim=cond_dim).to(device)
    ddpm  = DDPM(T=args.T, beta_schedule=args.schedule, device=device,
                 noise_std=noise_std, curl_div_weight=curl_div_w)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val    = float("inf")
    patience_ct = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("val_loss", best_val)
        print(f"Resumed from epoch {start_epoch-1}, best_val={best_val:.8f}")

    print(f"Device    : {device}")
    print(f"Schedule  : {args.schedule}")
    print(f"cond_dim  : {cond_dim}  {train_ds.feat_names}")
    print(f"noise_std : {noise_std:.5f}")
    print(f"Save dir  : {args.save_dir}")
    print(f"Loss      : eps_mse + {curl_div_w} * curl_div")

    best_ckpt = os.path.join(args.save_dir, f"best_model_{args.schedule}.pt")

    for epoch in range(start_epoch, args.epochs):
        # ── Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x0, cond = batch
            x0, cond = x0.to(device), cond.to(device)
            loss = ddpm.training_loss(model, (x0, cond), land_mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * x0.shape[0]
        train_loss /= len(train_ds)

        # ── Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x0, cond = batch
                x0, cond = x0.to(device), cond.to(device)
                loss = ddpm.training_loss(model, (x0, cond), land_mask)
                val_loss += loss.item() * x0.shape[0]
        val_loss /= len(val_ds)

        improved = val_loss < best_val
        if improved:
            best_val    = val_loss
            patience_ct = 0
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "opt":      opt.state_dict(),
                "val_loss": best_val,
                "schedule": args.schedule,
                "noise_std": noise_std,
                "args": {
                    "base_ch":  args.base_ch,
                    "time_dim": args.time_dim,
                    "T":        args.T,
                    "cond_dim": cond_dim,
                },
            }, best_ckpt)
        else:
            patience_ct += 1

        marker = " *" if improved else ""
        print(f"Epoch {epoch:4d}  train={train_loss:.8f}  val={val_loss:.8f}"
              f"  best={best_val:.8f}  patience={patience_ct}/{args.patience}{marker}",
              flush=True)

        if args.patience > 0 and patience_ct >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nTraining complete. Best val={best_val:.8f}  saved to {best_ckpt}")


if __name__ == "__main__":
    main()
