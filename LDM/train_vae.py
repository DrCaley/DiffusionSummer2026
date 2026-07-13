"""
train_vae.py  —  Train the OceanVAE on ocean current fields.

Loss: ocean-cell MSE reconstruction + β * KL divergence
      (small β keeps latent space structured without collapsing variance)

Usage (on remote):
    cd /root/autoencoder_train
    python3 -u train_vae.py \
        --pickle /root/ocean_ddpm/data_local.pickle \
        --save_dir /root/ldm/checkpoints_vae \
        --epochs 1000 --beta 0.0001
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae_model import OceanVAE
from dataset import OceanCurrentDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",    default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--save_dir",  default="/root/ldm/checkpoints_vae")
    p.add_argument("--epochs",    type=int,   default=1000)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--beta",      type=float, default=0.0001,
                   help="KL weight. Small values (1e-4) keep latent well-structured.")
    p.add_argument("--c_lat",     type=int,   default=4,
                   help="Number of latent channels.")
    p.add_argument("--base_ch",   type=int,   default=32)
    p.add_argument("--patience",  type=int,   default=60)
    p.add_argument("--resume",    default=None)
    return p.parse_args()


def vae_loss(recon, x, mu, logvar, land_mask, beta):
    ocean = (~land_mask).float()[None, None]           # (1,1,H,W)
    n_ocean = ocean.sum().item()

    # Reconstruction loss (MSE on ocean cells only)
    recon_loss = F.mse_loss(recon * ocean, x * ocean) * (ocean.numel() / max(n_ocean, 1))

    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device   : {device}")
    print(f"c_lat    : {args.c_lat}")
    print(f"beta     : {args.beta}")

    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)
    land_mask = train_ds.land_mask.to(device)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=2, pin_memory=True)

    model = OceanVAE(c_lat=args.c_lat, base_ch=args.base_ch).to(device)
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Latent shape: {model.latent_shape}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    start_epoch = 0
    best_val    = float("inf")
    patience_ctr = 0

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        best_val     = ck.get("val_loss", float("inf"))
        start_epoch  = ck.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_dl:
            x = batch.to(device)
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, land_mask, args.beta)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                x = batch.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar, land_mask, args.beta)
                va_loss += loss.item()
        va_loss /= len(val_dl)
        sched.step()

        print(f"Epoch {epoch:4d}/{args.epochs} | train={tr_loss:.6f} | val={va_loss:.6f}")

        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "val_loss": va_loss, "args": vars(args),
        }
        if va_loss < best_val:
            best_val = va_loss; patience_ctr = 0
            torch.save(ckpt, os.path.join(args.save_dir, "best_vae.pt"))
        else:
            patience_ctr += 1
        torch.save(ckpt, os.path.join(args.save_dir, "last_vae.pt"))

        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Checkpoint: {args.save_dir}/best_vae.pt")


if __name__ == "__main__":
    main()
