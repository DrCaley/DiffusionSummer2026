"""
loss_functions.py – Loss functions for DDPM training on ocean current fields.

Exports
-------
masked_mse_loss(pred, target, ocean_mask) -> scalar Tensor
    Mean squared error restricted to ocean (non-land) pixels.
"""

import torch


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ocean_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over ocean pixels only.

    Parameters
    ----------
    pred       : FloatTensor (B, C, H, W) – model output (predicted noise).
    target     : FloatTensor (B, C, H, W) – ground-truth noise.
    ocean_mask : BoolTensor (H, W)        – True = ocean pixel to include.

    Returns
    -------
    loss : scalar FloatTensor
    """
    diff = (pred - target) ** 2      # (B, C, H, W)
    # ocean_mask broadcasts over B and C dimensions
    return diff[:, :, ocean_mask].mean()
