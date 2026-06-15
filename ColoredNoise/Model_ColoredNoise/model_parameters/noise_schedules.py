"""
noise_schedules.py – Noise schedules for DDPM training.

Exports
-------
cosine_beta_schedule(T, s) -> (betas, alpha_bar)
    Cosine schedule from Nichol & Dhariwal (2021), "Improved DDPMs".
    Returns float32 tensors of length T.
"""

import math
import torch


def cosine_beta_schedule(T: int, s: float = 0.008):
    """Cosine noise schedule.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    s : float
        Small offset to prevent beta_0 from being too small (default 0.008).

    Returns
    -------
    betas     : FloatTensor of shape (T,)  – per-step noise variances.
    alpha_bar : FloatTensor of shape (T,)  – cumulative product of (1 - beta_t).
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = torch.clamp(1.0 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)
    return betas.float(), alpha_bar[1:].float()
