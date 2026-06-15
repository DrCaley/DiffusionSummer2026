from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.data import OceanCurrentDataset, load_cleaned_pickle, prepare_divergence_free_pickle, split_metadata
from model.diffusion import GaussianDiffusion
from model.networks import UNetModel
from model.pathing import build_inpainting_condition, observed_field, sample_robot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the divergence-free DDPM.")
    parser.add_argument("--pickle", type=Path, default=Path("..") / "data.pickle")
    parser.add_argument("--clean-pickle", type=Path, default=Path("..") / "data_divfree.pickle")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--noise-type", type=str, default="divergence_free")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--negligible-threshold", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints") / "x0_hybrid_inpainting_2026-06-15")
    parser.add_argument("--path-steps", type=int, default=150)
    parser.add_argument("--prediction-type", type=str, default="x0")
    parser.add_argument("--reconstruction-loss-weight", type=float, default=1.0)
    parser.add_argument("--noise-loss-weight", type=float, default=0.25)
    parser.add_argument("--rebuild-clean-data", action="store_true")
    return parser.parse_args()


def _load_data(args: argparse.Namespace):
    if args.rebuild_clean_data or not args.clean_pickle.exists():
        prepare_divergence_free_pickle(args.pickle, args.clean_pickle)
    splits, land_mask = load_cleaned_pickle(args.clean_pickle)
    return splits, land_mask


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    splits, land_mask = _load_data(args)
    metadata = split_metadata(splits)
    print(f"Dataset: {metadata}")

    model_in_channels = 2
    model_condition_channels = 4
    run_args = dict(vars(args))
    run_args["model_in_channels"] = model_in_channels
    run_args["model_condition_channels"] = model_condition_channels

    train_dataset = OceanCurrentDataset(splits[0], land_mask)
    val_dataset = OceanCurrentDataset(splits[1], land_mask)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = UNetModel(condition_channels=model_condition_channels).to(device)
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        noise_type=args.noise_type,
        prediction_type=args.prediction_type,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        noise_loss_weight=args.noise_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = args.run_dir
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / "best_model.pt"
    training_log_path = checkpoints_dir / "training_log.csv"
    journal_path = run_dir / "journal.md"

    with training_log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "timestamp_utc", "train_loss", "val_loss"])

    with journal_path.open("w", encoding="utf-8") as journal_file:
        journal_file.write("# Training Journal\n\n")
        journal_file.write(f"Start: {datetime.now(timezone.utc).isoformat()}\n")
        journal_file.write(f"Run directory: {run_dir}\n")
        journal_file.write(f"Device: {device}\n")
        journal_file.write(f"Timesteps: {args.timesteps}\n")
        journal_file.write(f"Noise type: {args.noise_type}\n")
        journal_file.write(f"Prediction type: {args.prediction_type}\n")
        journal_file.write(f"Reconstruction loss weight: {args.reconstruction_loss_weight}\n")
        journal_file.write(f"Noise loss weight: {args.noise_loss_weight}\n")
        journal_file.write(f"Path steps: {args.path_steps}\n\n")

    best_val = float("inf")
    stagnant_epochs = 0
    previous_val = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch_index, batch in enumerate(progress):
            fields = batch["field"].to(device)
            land = batch["land_mask"].to(device)
            observation_masks = []
            for sample_index in range(fields.shape[0]):
                path_seed = epoch * 100000 + batch_index * args.batch + sample_index
                path = sample_robot_path(land[sample_index].cpu().numpy(), steps=args.path_steps, seed=path_seed)
                observation_masks.append(path.mask)
            observation_mask = torch.stack(observation_masks, dim=0).to(device=device)
            observed = observed_field(fields, observation_mask)
            conditioning = build_inpainting_condition(observed, observation_mask, land)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(
                model,
                fields,
                land,
                observation_mask=observation_mask,
                conditioning=conditioning,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item()) * fields.shape[0]
            train_count += int(fields.shape[0])
            progress.set_postfix(loss=float(loss.item()))

        train_loss = train_total / max(train_count, 1)

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                fields = batch["field"].to(device)
                land = batch["land_mask"].to(device)
                observation_masks = []
                for sample_index in range(fields.shape[0]):
                    path_seed = 10_000_000 + batch_index * args.batch + sample_index
                    path = sample_robot_path(land[sample_index].cpu().numpy(), steps=args.path_steps, seed=path_seed)
                    observation_masks.append(path.mask)
                observation_mask = torch.stack(observation_masks, dim=0).to(device=device)
                observed = observed_field(fields, observation_mask)
                conditioning = build_inpainting_condition(observed, observation_mask, land)
                loss = diffusion.training_loss(
                    model,
                    fields,
                    land,
                    observation_mask=observation_mask,
                    conditioning=conditioning,
                )
                val_total += float(loss.item()) * fields.shape[0]
                val_count += int(fields.shape[0])

        val_loss = val_total / max(val_count, 1)
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        with journal_path.open("a", encoding="utf-8") as journal_file:
            journal_file.write(f"- Epoch {epoch:03d}: train={train_loss:.6f}, val={val_loss:.6f}\n")

        with training_log_path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                epoch,
                datetime.now(timezone.utc).isoformat(),
                f"{train_loss:.10f}",
                f"{val_loss:.10f}",
            ])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": run_args,
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)

        if epoch % args.save_every == 0:
            torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch:04d}.pt")

        if previous_val is not None:
            relative_change = abs(previous_val - val_loss) / max(abs(previous_val), 1e-8)
            stagnant_epochs = stagnant_epochs + 1 if relative_change < args.negligible_threshold else 0
            if stagnant_epochs >= args.patience:
                print(f"Stopping early after {epoch} epochs because validation loss stabilized.")
                break
        previous_val = val_loss

    torch.save({"model_state_dict": model.state_dict(), "args": run_args}, checkpoints_dir / "final_model.pt")
    print(f"Best checkpoint: {best_path}")
    print(f"Training log: {training_log_path}")
    print(f"Journal: {journal_path}")


if __name__ == "__main__":
    main()
