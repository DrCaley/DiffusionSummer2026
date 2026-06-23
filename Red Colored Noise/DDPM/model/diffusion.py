"""
diffusion.py – DDPM forward-process utilities.

Used by train.py, repaint_infer.py, and dps_infer.py.
"""

import torch


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Sample x_t from x_0 using the closed-form forward process.

    x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * noise

    Parameters
    ----------
    x0                       : (B, C, H, W)
    t                        : LongTensor (B,)
    sqrt_alpha_bar           : FloatTensor (T,)
    sqrt_one_minus_alpha_bar : FloatTensor (T,)
    noise                    : (B, C, H, W)  – the epsilon added

    Returns
    -------
    xt : FloatTensor (B, C, H, W)
    """
    sb  = sqrt_alpha_bar[t][:, None, None, None]
    smb = sqrt_one_minus_alpha_bar[t][:, None, None, None]
    return sb * x0 + smb * noise


def tweedie_x0(
    xt: torch.Tensor,
    t: int,
    eps_pred: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """Tweedie posterior mean estimate of x_0 given x_t and predicted noise.

    x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

    Parameters
    ----------
    xt                       : (B, C, H, W)
    t                        : int  – current timestep index
    eps_pred                 : (B, C, H, W)  – model's noise prediction
    sqrt_alpha_bar           : FloatTensor (T,)
    sqrt_one_minus_alpha_bar : FloatTensor (T,)

    Returns
    -------
    x0_hat : FloatTensor (B, C, H, W)
    """
    sab  = sqrt_alpha_bar[t]
    smab = sqrt_one_minus_alpha_bar[t]
    return (xt - smab * eps_pred) / sab.clamp(min=1e-8)
