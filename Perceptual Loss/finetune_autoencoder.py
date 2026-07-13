"""
Fine-tune the RepaintAutoencoder (originally trained for sparse-observation
reconstruction, mask_ratio in [0.01, 0.06]) toward the near-fully-observed
regime it will actually see when used as a frozen perceptual-loss encoder
(perceptual_loss.py always passes an all-ones observed-mask channel).

Starts from the pretrained weights in Autoencoder Cascade/best_model_autoencoder.pt
but resets the optimizer/scheduler/epoch count, since the training regime
(mask ratio) is different from the original run.

Usage (run from workspace root):
    python "Perceptual Loss/finetune_autoencoder.py"
    python "Perceptual Loss/finetune_autoencoder.py" --epochs 200 --mask_ratio_min 0.9 --mask_ratio_max 1.0
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ae_model import RepaintAutoencoder
from dataset import OceanCurrentDataset
from loss_functions import curl_div_loss


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune RepaintAutoencoder toward near-fully-observed inputs.")
    p.add_argument("--pickle", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data.pickle"))
    p.add_argument("--init_from", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Autoencoder Cascade", "best_model_autoencoder.pt"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--save_dir", default=None,
                    help="Defaults to Perceptual Loss/checkpoints_autoencoder_finetuned/")
    p.add_argument("--resume", default=None,
                    help="Resume a fine-tuning run in progress (keeps optimizer/scheduler/epoch state).")
    p.add_argument("--mask_ratio_min", type=float, default=0.8,
                    help="Minimum observed fraction of ocean pixels. Pushed high (near full observation) "
                         "so the encoder is tuned for the regime perceptual_loss.py uses.")
    p.add_argument("--mask_ratio_max", type=float, default=1.0)
    p.add_argument("--curl_div_weight", type=float, default=0.002)
    return p.parse_args()


def _extract_x0(batch, device):
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device)
    return batch.to(device)


def random_sparse_mask(land_mask: torch.Tensor, ratio_min: float, ratio_max: float) -> torch.Tensor:
    H, W = land_mask.shape
    ocean = ~land_mask
    keep_ratio = random.uniform(ratio_min, ratio_max)
    rand = torch.rand(H, W, device=land_mask.device)
    mask = (rand < keep_ratio) & ocean
    return mask.float().unsqueeze(0)


def build_input(x0: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    obs = obs_mask.unsqueeze(0)
    x_masked = x0 * obs
    obs_channel = obs.expand(x0.shape[0], 1, x0.shape[2], x0.shape[3])
    return torch.cat([x_masked, obs_channel], dim=1)


def train_step(model, x0, land_mask, optimizer, args):
    model.train()
    obs_mask = random_sparse_mask(land_mask, args.mask_ratio_min, args.mask_ratio_max)
    inp = build_input(x0, obs_mask)
    pred = model(inp)

    ocean = (~land_mask).float()[None, None]
    recon = F.mse_loss(pred * ocean, x0 * ocean)
    cd = curl_div_loss(pred, x0, ocean)
    loss = recon + args.curl_div_weight * cd

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def val_step(model, x0, land_mask, args):
    model.eval()
    obs_mask = random_sparse_mask(land_mask, args.mask_ratio_min, args.mask_ratio_max)
    inp = build_input(x0, obs_mask)
    pred = model(inp)

    ocean = (~land_mask).float()[None, None]
    recon = F.mse_loss(pred * ocean, x0 * ocean)
    cd = curl_div_loss(pred, x0, ocean)
    loss = recon + args.curl_div_weight * cd
    return float(loss.item())


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.save_dir is None:
        args.save_dir = os.path.join(script_dir, "checkpoints_autoencoder_finetuned")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device          : {device}")
    print(f"Save dir        : {args.save_dir}")
    print(f"Mask ratio      : [{args.mask_ratio_min}, {args.mask_ratio_max}]")
    print(f"curl_div_weight : {args.curl_div_weight}")

    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds = OceanCurrentDataset(args.pickle, split=1)
    land_mask = train_ds.land_mask.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = RepaintAutoencoder(in_ch=3, out_ch=2, base_ch=args.base_ch).to(device)
    print(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    start_epoch = 0
    best_val = float("inf")
    patience_ctr = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "val_loss" in ckpt:
            best_val = ckpt["val_loss"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed fine-tuning from epoch {start_epoch}, best_val={best_val:.5f}")
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Initialized from pretrained weights: {args.init_from}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = 0.0
        for batch in train_loader:
            x0 = _extract_x0(batch, device)
            train_loss += train_step(model, x0, land_mask, optimizer, args)
        train_loss /= len(train_loader)

        val_loss = 0.0
        for batch in val_loader:
            x0 = _extract_x0(batch, device)
            val_loss += val_step(model, x0, land_mask, args)
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(ckpt_data, os.path.join(args.save_dir, "best_model_autoencoder_finetuned.pt"))
        else:
            patience_ctr += 1

        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nFine-tuning complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint : {args.save_dir}/best_model_autoencoder_finetuned.pt")


if __name__ == "__main__":
    main()
