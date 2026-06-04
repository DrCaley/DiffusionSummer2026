"""
Training script for the topology-regularised DDPM on ocean current fields.

Identical to Basic DDPM/train.py except it uses TopoDDPM, which adds a
curl+divergence regularisation term to the noise-prediction MSE loss.
Both terms are logged separately at each epoch.

Usage:
    py train.py
    py train.py --epochs 200 --batch 64 --base_ch 64 --lr 2e-4 --topo_weight 0.1
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — allows running from any working directory.
# dataset.py lives at the workspace root; model.py lives in 'Basic DDPM/'.
# We append (not insert) so that 'Topo DDPM/' (added by Python automatically
# as the script directory) stays first — keeping our diffusion.py, not the
# base one, as the 'diffusion' module.
# ---------------------------------------------------------------------------
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_HERE)
_BASIC = os.path.join(_ROOT, "Basic DDPM")
sys.path.append(_ROOT)    # for dataset.py
sys.path.append(_BASIC)   # for model.py

import torch
from torch.utils.data import DataLoader

from dataset   import OceanCurrentDataset
from model     import UNet
from diffusion import TopoDDPM


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--time_dim",    type=int,   default=256)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--noise_type",  default="gaussian", choices=["gaussian"],
                   help="Type of noise used in the forward process (default: gaussian)")
    p.add_argument("--schedule",    default="cosine", choices=["cosine", "linear"],
                   help="Noise schedule (default: cosine)")
    p.add_argument("--topo_weight", type=float, default=0.1,
                   help="Weight λ for the topological (curl+divergence) loss term.")
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--workers",     type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Topological loss weight λ = {args.topo_weight}")

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
    model = UNet(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    diffusion = TopoDDPM(T=args.T, beta_schedule=args.schedule, device=device)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Run tag: model_loss_noise_type_schedule ----
    run_tag = f"ddpm_curl_div_{args.noise_type}_{args.schedule}"

    # ---- Training loop ----
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_total = train_eps = train_topo = 0.0
        for x0 in train_loader:
            x0 = x0.to(device)
            loss, eps_loss, topo_loss = diffusion.training_loss(
                model, x0, land_mask, topo_weight=args.topo_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item()
            train_eps   += eps_loss.item()
            train_topo  += topo_loss.item()
        n = len(train_loader)
        train_total /= n
        train_eps   /= n
        train_topo  /= n

        # -- Validate --
        model.eval()
        val_total = val_eps = val_topo = 0.0
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                loss, eps_loss, topo_loss = diffusion.training_loss(
                    model, x0, land_mask, topo_weight=args.topo_weight,
                )
                val_total += loss.item()
                val_eps   += eps_loss.item()
                val_topo  += topo_loss.item()
        n = len(val_loader)
        val_total /= n
        val_eps   /= n
        val_topo  /= n

        scheduler.step()

        # -- Checkpoint --
        saved_best = False
        if val_total < best_val:
            best_val = val_total
            saved_best = True
            torch.save(
                {
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "val_loss": val_total,
                    "val_eps":  val_eps,
                    "val_topo": val_topo,
                    "args":     vars(args),
                },
                os.path.join(args.save_dir, f"best_{run_tag}.pt"),
            )

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                os.path.join(args.save_dir, f"ckpt_ep{epoch:04d}_{run_tag}.pt"),
            )

        if epoch % 10 == 0 or saved_best:
            tag = " *" if saved_best else ""
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={train_total:.5f} (eps={train_eps:.5f} topo={train_topo:.5f}) | "
                f"val={val_total:.5f}   (eps={val_eps:.5f}   topo={val_topo:.5f}){tag}"
            )

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint saved to: {args.save_dir}/best_{run_tag}.pt")


if __name__ == "__main__":
    main()
