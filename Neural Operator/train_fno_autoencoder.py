"""
train_fno_autoencoder.py
==========================
Stage 1 of FNO latent diffusion: pretrain the FNOAutoencoder (encoder +
decoder) as a plain reconstruction model — masked MSE on ocean pixels, no
diffusion involved yet. Once this converges, its frozen encoder is used to
map clean fields to latent codes for train_latent_fno_ddpm.py.

Usage:
    python3 train_fno_autoencoder.py --pickle /root/ocean_ddpm/data_local.pickle \\
        --epochs 100 --latent_ch 8
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
        if os.path.isfile(os.path.join(d, "dataset.py")):
            return d
    raise RuntimeError(f"Cannot find dataset.py — tried: {candidates}")


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain the FNO autoencoder.")
    p.add_argument("--pickle",        default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--diffusion_dir", default=None)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--base",          type=int,   default=32)
    p.add_argument("--latent_ch",     type=int,   default=8)
    p.add_argument("--modes1",        type=int,   default=12)
    p.add_argument("--modes2",        type=int,   default=6)
    p.add_argument("--n_blocks",      type=int,   default=2)
    p.add_argument("--save_dir",      default=None)
    p.add_argument("--workers",       type=int,   default=0)
    p.add_argument("--ckpt_every",    type=int,   default=10)
    return p.parse_args()


def main():
    args = parse_args()

    diff_dir = _find_diffusion_dir(args.diffusion_dir)
    sys.path.insert(0, diff_dir)
    print(f"Using dataset helpers from: {diff_dir}")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from dataset import OceanCurrentDataset
    from model_fno_autoencoder import FNOAutoencoder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_dir is None:
        args.save_dir = os.path.join(_SCRIPT_DIR, "checkpoints_fno_autoencoder")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Save dir   : {args.save_dir}")

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    land_mask = train_ds.land_mask.to(device)          # (H, W) bool
    ocean     = (~land_mask).float()[None, None]        # (1, 1, H, W)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model ----
    model = FNOAutoencoder(
        in_ch=2, base=args.base, latent_ch=args.latent_ch,
        modes1=args.modes1, modes2=args.modes2, n_blocks=args.n_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    sample = train_ds[0].unsqueeze(0).to(device)
    with torch.no_grad():
        z_shape = model.encoder(sample).shape
    print(f"Latent shape : {tuple(z_shape[1:])}  "
          f"(field dims={2*94*44}, latent dims={z_shape[1]*z_shape[2]*z_shape[3]})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val  = float("inf")
    best_name = "best_fno_autoencoder.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x0 in train_loader:
            x0 = x0.to(device)
            pred = model(x0)
            loss = F.mse_loss(pred * ocean, x0 * ocean)
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
                pred = model(x0)
                val_loss += F.mse_loss(pred * ocean, x0 * ocean).item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.6f} | val={val_loss:.6f}",
              flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_loss": val_loss, "args": vars(args)},
                os.path.join(args.save_dir, best_name),
            )
            print(f"  -> saved new best (val={val_loss:.6f})", flush=True)

        if epoch % args.ckpt_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                os.path.join(args.save_dir, f"ckpt_fno_ae_epoch{epoch:04d}.pt"),
            )

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    print(f"Best checkpoint: {args.save_dir}/{best_name}")


if __name__ == "__main__":
    main()
