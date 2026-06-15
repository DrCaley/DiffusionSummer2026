from __future__ import annotations

import argparse
import os
import pathlib
from pathlib import Path

import torch
from tqdm import tqdm

from model.data import OceanCurrentDataset, load_cleaned_pickle, prepare_divergence_free_pickle
from model.diffusion import GaussianDiffusion
from model.metrics import batch_metrics
from model.networks import UNetModel
from model.pathing import build_inpainting_condition, observed_field, sample_robot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch DDPM evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pickle", type=Path, default=Path("..") / "data_divfree.pickle")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--path_steps", type=int, default=150)
    parser.add_argument("--resample", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    dataset = OceanCurrentDataset(splits[2], land_mask)
    model, checkpoint = _load_model(args.checkpoint, device)
    checkpoint_args = checkpoint.get("args", {})
    diffusion = GaussianDiffusion(
        timesteps=int(checkpoint_args.get("timesteps", 1000)),
        noise_type=str(checkpoint_args.get("noise_type", "divergence_free")),
        prediction_type=str(checkpoint_args.get("prediction_type", "epsilon")),
    ).to(device)

    n_samples = min(args.n, len(dataset))
    metrics = []
    for index in tqdm(range(n_samples), desc="Batch inference"):
        batch = dataset[index]
        ground_truth = batch["field"].unsqueeze(0).to(device)
        land = batch["land_mask"].to(device)
        path = sample_robot_path(land.cpu().numpy(), steps=args.path_steps, seed=index)
        observation_mask = path.mask.to(device)
        observed = observed_field(ground_truth, observation_mask)
        conditioning = None if getattr(model, "condition_channels", 0) == 0 else build_inpainting_condition(observed, observation_mask, land)
        predicted = diffusion.repaint(model, observed, observation_mask, land, resample=args.resample, conditioning=conditioning)
        metrics.append(batch_metrics(predicted, ground_truth, land))

    rmse_values = [item["rmse"] for item in metrics]
    mae_values = [item["mae"] for item in metrics]
    rmse_std = float(torch.tensor(rmse_values).std(unbiased=False).item()) if len(rmse_values) > 1 else 0.0
    mae_std = float(torch.tensor(mae_values).std(unbiased=False).item()) if len(mae_values) > 1 else 0.0
    print(f"Mean RMSE: {sum(rmse_values) / len(rmse_values):.6f} ± {rmse_std:.6f}")
    print(f"Mean MAE: {sum(mae_values) / len(mae_values):.6f} ± {mae_std:.6f}")


if __name__ == "__main__":
    main()
