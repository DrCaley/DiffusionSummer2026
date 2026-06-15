from __future__ import annotations

import torch


def _ocean_mask(land_mask: torch.Tensor) -> torch.Tensor:
    if land_mask.dtype != torch.bool:
        land_mask = land_mask.bool()
    ocean = (~land_mask).to(dtype=torch.float32)
    if ocean.dim() == 2:
        ocean = ocean.unsqueeze(0).unsqueeze(0)
    elif ocean.dim() == 3:
        ocean = ocean.unsqueeze(1)
    elif ocean.dim() != 4:
        raise ValueError(f"Unsupported land mask shape: {tuple(land_mask.shape)}")
    return ocean


def rmse_and_mae(prediction: torch.Tensor, target: torch.Tensor, land_mask: torch.Tensor) -> tuple[float, float]:
    prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    ocean = _ocean_mask(land_mask).to(device=prediction.device, dtype=prediction.dtype)
    diff = (prediction - target) * ocean
    mse = (diff.square().sum() / ocean.sum().clamp_min(1.0)).item()
    mae = (diff.abs().sum() / ocean.sum().clamp_min(1.0)).item()
    return float(mse**0.5), float(mae)


def masked_divergence(field: torch.Tensor) -> torch.Tensor:
    if field.dim() != 4 or field.shape[1] != 2:
        raise ValueError("Expected field shape (B, 2, H, W)")
    u = field[:, 0]
    v = field[:, 1]
    divergence = torch.zeros_like(u)
    divergence[:, :, 1:-1] += 0.5 * (u[:, :, 2:] - u[:, :, :-2])
    divergence[:, 1:-1, :] += 0.5 * (v[:, 2:, :] - v[:, :-2, :])
    return divergence


def batch_metrics(prediction: torch.Tensor, target: torch.Tensor, land_mask: torch.Tensor) -> dict[str, float]:
    rmse, mae = rmse_and_mae(prediction, target, land_mask)
    return {"rmse": rmse, "mae": mae}
