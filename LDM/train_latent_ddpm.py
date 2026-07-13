"""
train_latent_ddpm.py  —  Train a DDPM on VAE latent codes.

Pipeline:
  1. Load trained VAE (frozen encoder).
  2. For each training field, encode to latent z = mu  (deterministic, no reparameterisation
     during DDPM training — use mode of posterior for stability).
  3. Train a standard epsilon-prediction DDPM on those latent codes.
     Architecture: reuse Repaint UNet with in_ch = c_lat.

Usage (on remote, after VAE is trained):
    python3 -u train_latent_ddpm.py \
        --pickle    /root/ocean_ddpm/data_local.pickle \
        --vae_ckpt  /root/ldm/checkpoints_vae/best_vae.pt \
        --save_dir  /root/ldm/checkpoints_ldm \
        --epochs    1000
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae_model import OceanVAE
from latent_unet import LatentUNet
from diffusion import DDPM
from dataset import OceanCurrentDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",    default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--vae_ckpt",  default="/root/ldm/checkpoints_vae/best_vae.pt")
    p.add_argument("--save_dir",  default="/root/ldm/checkpoints_ldm")
    p.add_argument("--epochs",    type=int,   default=1000)
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--base_ch",   type=int,   default=64)
    p.add_argument("--time_dim",  type=int,   default=256)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--schedule",  default="linear")
    p.add_argument("--patience",  type=int,   default=60)
    p.add_argument("--resume",    default=None)
    return p.parse_args()


@torch.no_grad()
def encode_dataset(vae: OceanVAE, ds, device, batch_size=256):
    """Encode all fields to latent mu (deterministic) — returns (N, C, H_lat, W_lat)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    latents = []
    for batch in loader:
        x = batch.to(device)
        mu, _ = vae.encode(x)
        latents.append(mu.cpu())
    return torch.cat(latents, dim=0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load VAE ──────────────────────────────────────────────────────────
    print("Loading VAE...")
    vae_ck = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_args = vae_ck.get("args", {})
    vae = OceanVAE(c_lat=vae_args.get("c_lat", 4),
                   base_ch=vae_args.get("base_ch", 32)).to(device)
    vae.load_state_dict(vae_ck["model"])
    vae.eval()
    c_lat = vae.c_lat
    print(f"VAE loaded. c_lat={c_lat}, latent shape: {vae.latent_shape}")

    # ── Encode datasets ───────────────────────────────────────────────────
    print("Encoding train/val datasets...")
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    train_latents = encode_dataset(vae, train_ds, device)  # (N_train, c_lat, H, W)
    val_latents   = encode_dataset(vae, val_ds,   device)
    print(f"Train latents: {train_latents.shape}")

    LAT_H, LAT_W = train_latents.shape[2], train_latents.shape[3]
    print(f"Latent spatial: ({LAT_H}, {LAT_W}) — LatentUNet needs divisible by 4")
    assert LAT_H % 4 == 0 and LAT_W % 4 == 0, \
        f"Latent ({LAT_H},{LAT_W}) not divisible by 4; adjust VAE PAD_H/PAD_W"

    # Compute latent std for noise_std (normalise the diffusion noise to the data scale)
    noise_std = float(train_latents.std().item())
    print(f"Latent noise_std: {noise_std:.5f}")

    train_dl = DataLoader(TensorDataset(train_latents), batch_size=args.batch,
                          shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(TensorDataset(val_latents),   batch_size=args.batch,
                          shuffle=False, num_workers=0, pin_memory=True)

    # ── Build latent DDPM ─────────────────────────────────────────────────
    model     = LatentUNet(in_ch=c_lat, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    diffusion = DDPM(T=args.T, beta_schedule=args.schedule,
                     device=device, noise_std=noise_std)
    print(f"Latent UNet params: {sum(p.numel() for p in model.parameters()):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    start_epoch  = 0
    best_val     = float("inf")
    patience_ctr = 0

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        best_val    = ck.get("val_loss", float("inf"))
        start_epoch = ck.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for (z0,) in train_dl:
            z0   = z0.to(device)
            B    = z0.shape[0]
            t    = torch.randint(0, args.T, (B,), device=device)
            zt, noise = diffusion.q_sample(z0, t)
            pred_noise = model(zt, t)
            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for (z0,) in val_dl:
                z0 = z0.to(device)
                B  = z0.shape[0]
                t  = torch.randint(0, args.T, (B,), device=device)
                zt, noise = diffusion.q_sample(z0, t)
                va_loss += F.mse_loss(model(zt, t), noise).item()
        va_loss /= len(val_dl)
        sched.step()

        print(f"Epoch {epoch:4d}/{args.epochs} | train={tr_loss:.6f} | val={va_loss:.6f}")

        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "val_loss": va_loss, "args": vars(args),
            "noise_std": noise_std, "c_lat": c_lat,
            "lat_h": LAT_H, "lat_w": LAT_W,
            "vae_ckpt": args.vae_ckpt,
        }
        if va_loss < best_val:
            best_val = va_loss; patience_ctr = 0
            torch.save(ckpt, os.path.join(args.save_dir, "best_latent_ddpm.pt"))
        else:
            patience_ctr += 1
        torch.save(ckpt, os.path.join(args.save_dir, "last_latent_ddpm.pt"))

        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Checkpoint: {args.save_dir}/best_latent_ddpm.pt")


if __name__ == "__main__":
    main()
