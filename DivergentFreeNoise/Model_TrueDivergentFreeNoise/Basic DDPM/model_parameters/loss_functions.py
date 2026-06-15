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


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, land_mask: torch.Tensor) -> torch.Tensor:
    ocean = _ocean_mask(land_mask).to(device=prediction.device, dtype=prediction.dtype)
    squared_error = (prediction - target) ** 2 * ocean
    denominator = ocean.sum().clamp_min(1.0)
    return squared_error.sum() / denominator


def masked_mse_with_mask(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask != 0
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    squared_error = (prediction - target) ** 2 * mask
    denominator = mask.sum().clamp_min(1.0)
    return squared_error.sum() / denominator


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, land_mask: torch.Tensor) -> torch.Tensor:
    ocean = _ocean_mask(land_mask).to(device=prediction.device, dtype=prediction.dtype)
    absolute_error = (prediction - target).abs() * ocean
    denominator = ocean.sum().clamp_min(1.0)
    return absolute_error.sum() / denominator
