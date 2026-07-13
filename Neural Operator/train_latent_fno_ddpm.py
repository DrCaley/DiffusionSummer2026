"""
train_latent_fno_ddpm.py
==========================
Stage 2 of FNO latent diffusion: train a noise-conditioned FNO (same
FNO2dDDPM architecture as model_fno_ddpm.py, sized for the latent tensor
instead of the pixel field) to denoise latent codes z0 = encoder(x0), with
the encoder frozen from train_fno_autoencoder.py.

Differences from the pixel-space FNO-DDPM training (train_fno_ddpm.py):
  - land_mask is irrelevant in latent space (the encoder's strided convs +
    spectral blocks mix land/ocean geography together, so there's no clean
    per-latent-pixel land/ocean split) — training_loss is called with an
    all-False land_mask at latent resolution, i.e. plain unmasked MSE.
  - curl_div_weight is forced to 0.0: that structural loss assumes exactly
    2 channels (u, v); it's not meaningful on an 8-channel latent.
  - noise_std is computed from the *encoded* latents, not the raw field.

Usage:
    python3 train_latent_fno_ddpm.py --pickle /root/ocean_ddpm/data_local.pickle \\
        --ae_checkpoint /root/NeuralOperator/checkpoints_fno_autoencoder/best_fno_autoencoder.pt \\
        --epochs 300 --schedule linear
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)


def _find_diffusion_dir(explicit=None):
    candidates = [explicit] if explicit else []
    candidates += [
        os.path.join(_SCRIPT_DIR, "..", "Repaint vs DPS"),
        os.path.join(_SCRIPT_DIR, "..", "Repaint_vs_DPS"),
        "/root/Repaint_vs_DPS",
    ]
    for d in candidates:
        if not d:
            continue
        d = os.path.abspath(d)
        if os.path.isfile(os.path.join(d, "diffusion.py")):
            return d
    raise RuntimeError(f"Cannot find diffusion.py — tried: {candidates}")


def parse_args():
    p = argparse.ArgumentParser(description="Train latent FNO-DDPM.")
    p.add_argument("--pickle",        default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--ae_checkpoint", required=True)
    p.add_argument("--diffusion_dir", default=None)
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--width",         type=int,   default=64)
    p.add_argument("--modes1",        type=int,   default=12)
    p.add_argument("--modes2",        type=int,   default=6)
    p.add_argument("--n_layers",      type=int,   default=4)
    p.add_argument("--time_dim",      type=int,   default=256)
    p.add_argument("--T",             type=int,   default=1000)
    p.add_argument("--schedule",      default="linear",
                   choices=["linear", "cosine", "quadratic", "sigmoid", "geometric"])
    p.add_argument("--save_dir",      default=None)
    p.add_argument("--workers",       type=int,   default=0)
    p.add_argument("--ckpt_every",    type=int,   default=10)
    return p.parse_args()


def main():
    args = parse_args()

    diff_dir = _find_diffusion_dir(args.diffusion_dir)
    sys.path.insert(0, diff_dir)
    print(f"Using dataset/diffusion helpers from: {diff_dir}")

    import torch
    from torch.utils.data import DataLoader
    from dataset             import OceanCurrentDataset
    from diffusion           import DDPM
    from model_fno_ddpm      import FNO2dDDPM
    from model_fno_autoencoder import FNOAutoencoder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_dir is None:
        args.save_dir = os.path.join(_SCRIPT_DIR, "checkpoints_latent_fno_ddpm")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Schedule   : {args.schedule}")
    print(f"Save dir   : {args.save_dir}")

    # ---- Load + freeze the autoencoder's encoder ----
    ae_ckpt   = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    ae_args   = ae_ckpt.get("args", {})
    autoencoder = FNOAutoencoder(
        in_ch=2, base=ae_args.get("base", 32), latent_ch=ae_args.get("latent_ch", 8),
        modes1=ae_args.get("modes1", 12), modes2=ae_args.get("modes2", 6),
        n_blocks=ae_args.get("n_blocks", 2),
    ).to(device)
    autoencoder.load_state_dict(ae_ckpt["model"])
    autoencoder.eval()
    for p_ in autoencoder.parameters():
        p_.requires_grad_(False)
    encoder = autoencoder.encoder
    latent_ch = autoencoder.latent_ch
    print(f"Loaded autoencoder from {args.ae_checkpoint} (epoch {ae_ckpt.get('epoch','?')}, "
          f"val_loss={ae_ckpt.get('val_loss', float('nan')):.6f})")

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Encode a sample to get latent spatial shape + noise_std ----
    with torch.no_grad():
        sample_z = encoder(train_ds[0].unsqueeze(0).to(device))
    _, _, latent_H, latent_W = sample_z.shape
    print(f"Latent shape: ({latent_ch}, {latent_H}, {latent_W})")

    # Estimate noise_std from encoded training latents (subsample for speed)
    with torch.no_grad():
        stds = []
        for i, x0 in enumerate(train_loader):
            if i >= 20:
                break
            z0 = encoder(x0.to(device))
            stds.append(z0.std().item())
    noise_std = float(sum(stds) / len(stds))
    print(f"noise_std (encoded latent std, ~20 batches) : {noise_std:.5f}")

    # All-ocean "land mask" at latent resolution — disables ocean masking in
    # training_loss (there's no meaningful land/ocean split in latent space).
    latent_land_mask = torch.zeros(latent_H, latent_W, dtype=torch.bool, device=device)

    # ---- Model + diffusion (curl_div disabled: not meaningful on a
    # latent_ch-channel tensor, that loss assumes exactly u,v channels) ----
    model = FNO2dDDPM(
        in_ch=latent_ch, width=args.width, modes1=args.modes1, modes2=args.modes2,
        time_dim=args.time_dim, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    diffusion = DDPM(T=args.T, beta_schedule=args.schedule, device=device,
                     noise_std=noise_std, curl_div_weight=0.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val  = float("inf")
    best_name = f"best_latent_fno_ddpm_{args.schedule}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x0 in train_loader:
            x0 = x0.to(device)
            with torch.no_grad():
                z0 = encoder(x0)
            loss = diffusion.training_loss(model, z0, latent_land_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                z0 = encoder(x0)
                val_loss += diffusion.training_loss(model, z0, latent_land_mask).item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | val={val_loss:.5f}",
              flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_loss": val_loss, "schedule": args.schedule,
                 "noise_std": noise_std, "latent_ch": latent_ch,
                 "ae_checkpoint": args.ae_checkpoint, "args": vars(args)},
                os.path.join(args.save_dir, best_name),
            )
            print(f"  -> saved new best (val={val_loss:.5f})", flush=True)

        if epoch % args.ckpt_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "schedule": args.schedule, "noise_std": noise_std,
                 "latent_ch": latent_ch, "ae_checkpoint": args.ae_checkpoint,
                 "args": vars(args)},
                os.path.join(args.save_dir, f"ckpt_latent_fno_ddpm_epoch{epoch:04d}.pt"),
            )

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: {args.save_dir}/{best_name}")


if __name__ == "__main__":
    main()
