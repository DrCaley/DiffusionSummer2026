from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64) / timesteps
    alphas_cumprod = np.cos(((x + s) / (1.0 + s)) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999)


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor


def make_schedule(timesteps: int, device: torch.device | None = None) -> DiffusionSchedule:
    betas = torch.from_numpy(cosine_beta_schedule(timesteps)).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    if device is not None:
        betas = betas.to(device)
        alphas = alphas.to(device)
        alphas_cumprod = alphas_cumprod.to(device)
        alphas_cumprod_prev = alphas_cumprod_prev.to(device)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
        posterior_variance = posterior_variance.to(device)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        posterior_variance=posterior_variance,
    )
