from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _project_divergence_free_numpy(field: np.ndarray) -> np.ndarray:
    if field.ndim != 3 or field.shape[0] != 2:
        raise ValueError(f"Expected field shape (2, H, W), got {field.shape}")

    _, height, width = field.shape
    ky, kx = np.meshgrid(2.0 * np.pi * np.fft.fftfreq(height), 2.0 * np.pi * np.fft.fftfreq(width), indexing="ij")

    u_hat = np.fft.fft2(field[0])
    v_hat = np.fft.fft2(field[1])
    dot = kx * u_hat + ky * v_hat
    k2 = kx**2 + ky**2

    factor = np.zeros_like(dot)
    valid = k2 > 0.0
    factor[valid] = dot[valid] / k2[valid]

    u_hat = u_hat - kx * factor
    v_hat = v_hat - ky * factor
    projected = np.stack([np.fft.ifft2(u_hat).real, np.fft.ifft2(v_hat).real], axis=0)
    return projected.astype(np.float32, copy=False)


def project_divergence_free_torch(field: torch.Tensor) -> torch.Tensor:
    added_batch = field.dim() == 3
    if added_batch:
        field = field.unsqueeze(0)
    if field.dim() != 4 or field.shape[1] != 2:
        raise ValueError(f"Expected field shape (B, 2, H, W) or (2, H, W), got {tuple(field.shape)}")

    device = field.device
    dtype = field.dtype
    height, width = field.shape[-2:]
    ky, kx = torch.meshgrid(
        2.0 * torch.pi * torch.fft.fftfreq(height, device=device),
        2.0 * torch.pi * torch.fft.fftfreq(width, device=device),
        indexing="ij",
    )
    ky = ky.unsqueeze(0)
    kx = kx.unsqueeze(0)

    u_hat = torch.fft.fft2(field[:, 0])
    v_hat = torch.fft.fft2(field[:, 1])
    dot = kx * u_hat + ky * v_hat
    k2 = kx.square() + ky.square()

    factor = torch.where(k2 > 0, dot / k2, torch.zeros_like(dot))
    u_hat = u_hat - kx * factor
    v_hat = v_hat - ky * factor
    projected = torch.stack([torch.fft.ifft2(u_hat).real, torch.fft.ifft2(v_hat).real], dim=1)
    projected = projected.to(dtype=dtype)
    return projected.squeeze(0) if added_batch else projected


@dataclass(frozen=True)
class GaussianNoise:
    def sample(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype)


@dataclass(frozen=True)
class DivergenceFreeGaussianNoise:
    def sample(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        noise = torch.randn(shape, device=device, dtype=dtype)
        return project_divergence_free_torch(noise)


def build_noise_sampler(kind: str) -> GaussianNoise | DivergenceFreeGaussianNoise:
    normalized = kind.strip().lower().replace("-", "_")
    if normalized in {"divfree", "divergence_free", "divergencefree", "solenoidal"}:
        return DivergenceFreeGaussianNoise()
    return GaussianNoise()


def project_divergence_free_sample(field: np.ndarray) -> np.ndarray:
    return _project_divergence_free_numpy(field)
