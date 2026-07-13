import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ae_model import RepaintAutoencoder
from dataset import OceanCurrentDataset
from loss_functions import curl_div_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train Repaint-based autoencoder for sparse reconstruction.")
    p.add_argument("--pickle", default="data_timecond.pickle")
    p.add_argument("--data_mean", type=float, default=None,
                   help="Optional: data mean to subtract before normalization.")
    p.add_argument("--data_std", type=float, default=None,
                   help="Optional: data std to divide by for normalization.")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--save_dir", default="checkpoints_autoencoder")
    p.add_argument("--resume", default=None)
    p.add_argument("--mask_ratio_min", type=float, default=0.01,
                   help="Minimum observed fraction of ocean pixels for random sparse mask.")
    p.add_argument("--mask_ratio_max", type=float, default=0.06,
                   help="Maximum observed fraction of ocean pixels for random sparse mask.")
    p.add_argument("--curl_div_weight", type=float, default=0.002)
    return p.parse_args()


def _extract_x0(batch, device):
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device)
    return batch.to(device)


def random_sparse_mask(land_mask: torch.Tensor, ratio_min: float, ratio_max: float) -> torch.Tensor:
    """
    Create random sparse observation mask over ocean pixels only.
    Returns mask shape (1, H, W), float in {0,1}.
    """
    H, W = land_mask.shape
    ocean = ~land_mask
    keep_ratio = random.uniform(ratio_min, ratio_max)
    rand = torch.rand(H, W, device=land_mask.device)
    mask = (rand < keep_ratio) & ocean
    return mask.float().unsqueeze(0)


def build_input(x0: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    """
    x0: (B,2,H,W)
    obs_mask: (1,H,W)
    returns (B,3,H,W): masked uv + mask channel
    """
    obs = obs_mask.unsqueeze(0)  # (1,1,H,W)
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

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device          : {device}")
    print(f"Save dir        : {args.save_dir}")
    print(f"Mask ratio      : [{args.mask_ratio_min}, {args.mask_ratio_max}]")
    print(f"curl_div_weight : {args.curl_div_weight}")

    # Normalization is optional. Only apply normalization if BOTH --data_mean
    # and --data_std are provided; otherwise leave the data unnormalized.
    if (args.data_mean is None) ^ (args.data_std is None):
        print("Warning: both --data_mean and --data_std must be provided to enable normalization. Proceeding without normalization.")
        args.data_mean = None
        args.data_std = None

    train_ds = OceanCurrentDataset(args.pickle, split=0, data_mean=args.data_mean, data_std=args.data_std)
    val_ds = OceanCurrentDataset(args.pickle, split=1, data_mean=args.data_mean, data_std=args.data_std)
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
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.5f}")

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
            torch.save(ckpt_data, os.path.join(args.save_dir, "best_model_autoencoder.pt"))
        else:
            patience_ctr += 1

        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint : {args.save_dir}/best_model_autoencoder.pt")


if __name__ == "__main__":
    main()
