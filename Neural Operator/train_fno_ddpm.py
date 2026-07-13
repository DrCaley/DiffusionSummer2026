"""
train_fno_ddpm.py
==================
Trains the noise-conditioned FNO (FNO-DDPM) as a direct drop-in replacement
for the Repaint UNet epsilon-predictor: FNO(x_t, t) -> eps. Same DDPM
schedule (diffusion.py, including the curl/div structural loss term), same
OceanCurrentDataset, same checkpoint conventions as train_repaint.py — so the
resulting checkpoint loads into infer_batch_3methods.py / repaint_infer.py
by swapping `from repaint_model import Repaint` for
`from model_fno_ddpm import FNO2dDDPM as Repaint`.

Usage (run from the machine with the pickle + diffusion.py/dataset.py, e.g.
/root/Repaint_vs_DPS on the remote, or pass --diffusion_dir explicitly):
    python3 train_fno_ddpm.py --pickle /root/ocean_ddpm/data.pickle \\
        --epochs 150 --schedule linear
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
    p = argparse.ArgumentParser(description="Train FNO-DDPM (noise-conditioned FNO).")
    p.add_argument("--pickle",      default="/root/ocean_ddpm/data.pickle")
    p.add_argument("--diffusion_dir", default=None,
                   help="Dir containing diffusion.py/dataset.py (auto-detected if omitted).")
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--width",       type=int,   default=64)
    p.add_argument("--modes1",      type=int,   default=16)
    p.add_argument("--modes2",      type=int,   default=16)
    p.add_argument("--n_layers",    type=int,   default=4)
    p.add_argument("--time_dim",    type=int,   default=256)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--schedule",    default="linear",
                   choices=["linear", "cosine", "quadratic", "sigmoid", "geometric"])
    p.add_argument("--save_dir",    default=None)
    p.add_argument("--workers",     type=int,   default=0)
    p.add_argument("--ckpt_every",  type=int,   default=10)
    return p.parse_args()


def main():
    args = parse_args()

    diff_dir = _find_diffusion_dir(args.diffusion_dir)
    sys.path.insert(0, diff_dir)
    print(f"Using dataset/diffusion helpers from: {diff_dir}")

    import torch
    from torch.utils.data import DataLoader
    from dataset        import OceanCurrentDataset
    from diffusion      import DDPM
    from model_fno_ddpm import FNO2dDDPM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_dir is None:
        args.save_dir = os.path.join(_SCRIPT_DIR, "checkpoints_fno_ddpm")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Schedule   : {args.schedule}")
    print(f"Save dir   : {args.save_dir}")

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    land_mask = train_ds.land_mask.to(device)   # (H, W) bool

    ocean_pixels = train_ds.data[:, :, ~train_ds.land_mask]
    noise_std    = float(ocean_pixels.std())
    print(f"noise_std (train ocean std) : {noise_std:.5f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model + diffusion ----
    model = FNO2dDDPM(
        in_ch=2, width=args.width, modes1=args.modes1, modes2=args.modes2,
        time_dim=args.time_dim, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    diffusion = DDPM(T=args.T, beta_schedule=args.schedule, device=device,
                     noise_std=noise_std)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val  = float("inf")
    best_name = f"best_fno_ddpm_{args.schedule}.pt"

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

        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | val={val_loss:.5f}",
              flush=True)

        # -- Checkpoint --
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_loss": val_loss, "schedule": args.schedule,
                 "noise_std": noise_std, "args": vars(args)},
                os.path.join(args.save_dir, best_name),
            )
            print(f"  -> saved new best ({best_name}, val={val_loss:.5f})", flush=True)

        if epoch % args.ckpt_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "schedule": args.schedule, "noise_std": noise_std,
                 "args": vars(args)},
                os.path.join(args.save_dir, f"ckpt_fno_ddpm_epoch{epoch:04d}.pt"),
            )

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: {args.save_dir}/{best_name}")


if __name__ == "__main__":
    main()
