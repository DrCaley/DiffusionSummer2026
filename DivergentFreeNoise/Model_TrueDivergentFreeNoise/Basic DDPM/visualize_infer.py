from __future__ import annotations

import argparse
import os
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model.data import OceanCurrentDataset, field_to_numpy, load_cleaned_pickle, prepare_divergence_free_pickle
from model.diffusion import GaussianDiffusion
from model.metrics import rmse_and_mae
from model.networks import UNetModel
from model.pathing import build_inpainting_condition, observed_field, sample_robot_path
from model.plotting import plot_actual_field, plot_loss_field, plot_prediction_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a DDPM reconstruction.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pickle", type=Path, default=Path("..") / "data_divfree.pickle")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--path_steps", type=int, default=150)
    parser.add_argument("--resample", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("testing") / "results")
    parser.add_argument("--rebuild-clean-data", action="store_true")
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[UNetModel, dict]:
    candidate_paths = [checkpoint_path]
    if checkpoint_path.name == "best_model.pt":
        candidate_paths.append(checkpoint_path.with_name("epoch_0400.pt"))

    last_error: Exception | None = None
    for candidate_path in dict.fromkeys(candidate_paths):
        try:
            original_windows_path = pathlib.WindowsPath
            original_posix_path = pathlib.PosixPath
            try:
                if os.name == "nt":
                    pathlib.PosixPath = pathlib.WindowsPath
                else:
                    pathlib.WindowsPath = pathlib.PosixPath
                checkpoint = torch.load(candidate_path, map_location=device, weights_only=False)
            finally:
                pathlib.WindowsPath = original_windows_path
                pathlib.PosixPath = original_posix_path
            model_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            inferred_condition_channels = 0
            input_weight = state_dict.get("input_conv.weight") if isinstance(state_dict, dict) else None
            if input_weight is not None and hasattr(input_weight, "shape") and len(input_weight.shape) == 4:
                inferred_condition_channels = max(int(input_weight.shape[1]) - 2, 0)
            model = UNetModel(
                in_channels=int(model_args.get("model_in_channels", 2)),
                condition_channels=int(model_args.get("model_condition_channels", inferred_condition_channels)),
            ).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model, checkpoint if isinstance(checkpoint, dict) else {}
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to load checkpoint from {checkpoint_path}") from last_error


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.rebuild_clean_data or not args.pickle.exists():
        source_pickle = args.pickle.with_name("data.pickle")
        prepare_divergence_free_pickle(source_pickle, args.pickle)
    splits, land_mask = load_cleaned_pickle(args.pickle)
    dataset = OceanCurrentDataset(splits[1], land_mask)

    sample_index = int(args.sample) % len(dataset)
    batch = dataset[sample_index]
    ground_truth = batch["field"].unsqueeze(0).to(device)
    land = batch["land_mask"].to(device)

    path = sample_robot_path(land.cpu().numpy(), steps=args.path_steps)
    observation_mask = path.mask.to(device)
    observed = observed_field(ground_truth, observation_mask)

    model, checkpoint = _load_model(args.checkpoint, device)
    conditioning = None if getattr(model, "condition_channels", 0) == 0 else build_inpainting_condition(observed, observation_mask, land)
    checkpoint_args = checkpoint.get("args", {})
    diffusion = GaussianDiffusion(
        timesteps=int(checkpoint_args.get("timesteps", 1000)),
        noise_type=str(checkpoint_args.get("noise_type", "divergence_free")),
        prediction_type=str(checkpoint_args.get("prediction_type", "epsilon")),
    ).to(device)
    predicted = diffusion.repaint(model, observed, observation_mask, land, resample=args.resample, conditioning=conditioning)
    predicted_skip_last = diffusion.repaint(model, observed, observation_mask, land, resample=args.resample, skip_last_step=True, conditioning=conditioning)

    rmse, mae = rmse_and_mae(predicted, ground_truth, land)
    rmse_skip, mae_skip = rmse_and_mae(predicted_skip_last, ground_truth, land)
    error = (predicted - ground_truth).abs().squeeze(0)
    error_skip = (predicted_skip_last - ground_truth).abs().squeeze(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual_path = args.output_dir / f"actual_sample_{sample_index}.png"
    predicted_path = args.output_dir / f"predicted_sample_{sample_index}.png"
    predicted_skip_path = args.output_dir / f"predicted_skip_last_sample_{sample_index}.png"
    loss_path = args.output_dir / f"loss_sample_{sample_index}.png"
    loss_skip_path = args.output_dir / f"loss_skip_last_sample_{sample_index}.png"
    overview_path = args.output_dir / f"overview_sample_{sample_index}.png"
    comparison_path = args.output_dir / f"comparison_sample_{sample_index}.png"

    plot_actual_field(field_to_numpy(ground_truth.squeeze(0)), land.cpu().numpy(), actual_path, title=f"Actual field | RMSE={rmse:.4f} MAE={mae:.4f}")
    plot_prediction_field(field_to_numpy(predicted.squeeze(0)), land.cpu().numpy(), observation_mask.cpu().numpy(), predicted_path, title=f"Predicted field | RMSE={rmse:.4f} MAE={mae:.4f}")
    plot_prediction_field(field_to_numpy(predicted_skip_last.squeeze(0)), land.cpu().numpy(), observation_mask.cpu().numpy(), predicted_skip_path, title=f"Predicted skip-last | RMSE={rmse_skip:.4f} MAE={mae_skip:.4f}")
    plot_loss_field(error.cpu().numpy(), land.cpu().numpy(), loss_path, title=f"Absolute error | RMSE={rmse:.4f} MAE={mae:.4f}")
    plot_loss_field(error_skip.cpu().numpy(), land.cpu().numpy(), loss_skip_path, title=f"Absolute error skip-last | RMSE={rmse_skip:.4f} MAE={mae_skip:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, image_path, title in zip(axes, [actual_path, predicted_path, loss_path], ["Actual", "Predicted", "Loss"]):
        image = plt.imread(image_path)
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Reconstruction overview | RMSE={rmse:.4f} MAE={mae:.4f}")
    fig.tight_layout()
    fig.savefig(overview_path, dpi=200)
    plt.close(fig)

    compare_fig, compare_axes = plt.subplots(2, 2, figsize=(16, 10))
    compare_items = [
        (actual_path, f"Actual | RMSE={rmse:.4f} MAE={mae:.4f}"),
        (predicted_path, f"Full RePaint | RMSE={rmse:.4f} MAE={mae:.4f}"),
        (predicted_skip_path, f"Skip last step | RMSE={rmse_skip:.4f} MAE={mae_skip:.4f}"),
        (loss_skip_path, f"Skip-last absolute error | RMSE={rmse_skip:.4f} MAE={mae_skip:.4f}"),
    ]
    for axis, (image_path, title) in zip(compare_axes.flat, compare_items):
        image = plt.imread(image_path)
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    compare_fig.tight_layout()
    compare_fig.savefig(comparison_path, dpi=200)
    plt.close(compare_fig)

    print(f"Saved figures to {args.output_dir}")
    print(f"RMSE={rmse:.6f} MAE={mae:.6f}")
    print(f"RMSE skip-last={rmse_skip:.6f} MAE skip-last={mae_skip:.6f}")


if __name__ == "__main__":
    main()
