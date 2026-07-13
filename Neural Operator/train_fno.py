import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import PickleFieldDataset
from model_fno import get_model
from tqdm import tqdm


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = PickleFieldDataset(args.data_path, split='train', val_fraction=args.val_frac)
    val_ds = PickleFieldDataset(args.data_path, split='val', val_fraction=args.val_frac)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    sample = train_ds[0]
    H, W, C = sample.shape

    model = get_model(in_ch=C, out_ch=C, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    epochs_no_improve = 0

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # batch: (B,H,W,C)
            batch = batch.to(device)
            # create input: here we use identity input (you may replace with path-encoding)
            x_in = batch  # using field as both input and target in an autoencoding setup
            optimizer.zero_grad()
            out = model(x_in)
            loss = criterion(out, batch)
            if not torch.isfinite(loss):
                print(f'  [warn] non-finite loss ({loss.item()}), skipping batch')
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * batch.size(0)
            count += batch.size(0)
            pbar.set_postfix(loss=running / count)

        # validation
        model.eval()
        val_loss = 0.0
        vcount = 0
        with torch.no_grad():
            for vbatch in val_loader:
                vbatch = vbatch.to(device)
                out = model(vbatch)
                l = criterion(out, vbatch)
                val_loss += l.item() * vbatch.size(0)
                vcount += vbatch.size(0)
        val_loss = val_loss / max(1, vcount)

        print(f"Epoch {epoch} validation loss: {val_loss:.6f}")

        # early stopping logic
        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            ckpt = os.path.join(args.out_dir, f"best_fno.pth")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'val_loss': val_loss}, ckpt)
            print(f"Saved best model to {ckpt}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (patience={args.patience}). Best val {best_val:.6f}")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='ocean_ddpm/data.pickle')
    parser.add_argument('--out-dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--val-frac', type=float, default=0.1)
    args = parser.parse_args()
    train(args)
