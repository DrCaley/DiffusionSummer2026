"""
repaint_infer.py – RePaint inpainting inference for the Colored-Noise DDPM.

Algorithm: Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion
Probabilistic Models", CVPR 2022.  https://arxiv.org/pdf/2201.09865

At each reverse step t:
  1. Re-noise the known observations to step t: x_t^{known}
  2. Run one DDPM reverse step on the current x_t: x_t^{unknown}
  3. Blend: x_t = x_t^{known} * mask + x_t^{unknown} * (1 - mask)
  4. For resampling (r > 1 inner iterations):
       re-noise x_t back to step t and repeat steps 1-3.

This module is importable.  Use repaint_sample() in your scripts.
"""

import math
import sys
import os

import torch

# Allow sibling imports when run from DDPM/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_parameters.noise_types import colored_gaussian_noise
from model_parameters.noise_schedules import cosine_beta_schedule


@torch.no_grad()
def repaint_sample(
    model: torch.nn.Module,
    x0_known: torch.Tensor,
    known_mask: torch.Tensor,
    T: int,
    device: torch.device,
    resample: int = 10,
    noise_alpha: float = 2.0,
) -> torch.Tensor:
    """Run RePaint reverse diffusion to reconstruct an inpainted field.

    Parameters
    ----------
    model       : trained UNet (eval mode)
    x0_known    : (1, 2, H, W)  – full ground-truth; observed at known_mask pixels
    known_mask  : (H, W) BoolTensor  – True = observed (robot path)
    T           : int  – number of diffusion timesteps
    device      : torch.device
    resample    : int  – inner resampling iterations per timestep (r in paper)
    noise_alpha : float  – spectral exponent matching the training noise

    Returns
    -------
    x0_pred : (1, 2, H, W) FloatTensor  – reconstructed field
    """
    betas, alpha_bar = cosine_beta_schedule(T)
    alpha     = 1.0 - betas
    alpha     = alpha.to(device)
    betas     = betas.to(device)
    alpha_bar = alpha_bar.to(device)

    sqrt_alpha_bar   = alpha_bar.sqrt()
    sqrt_1m_ab       = (1.0 - alpha_bar).sqrt()
    sqrt_recip_alpha = (1.0 / alpha.sqrt())
    sqrt_recip_1m_ab = (1.0 / (1.0 - alpha_bar).sqrt())

    km = known_mask.to(device)              # (H, W)
    x0_known = x0_known.to(device)

    # Start from pure colored noise
    xt = colored_gaussian_noise(x0_known.shape, alpha=noise_alpha, device=device)

    for t_int in reversed(range(T)):
        t_tensor = torch.full((1,), t_int, dtype=torch.long, device=device)

        for inner in range(resample):
            # --- known part: re-noise x_0 to timestep t ---
            if t_int > 0:
                eps_known = colored_gaussian_noise(x0_known.shape, alpha=noise_alpha, device=device)
                xt_known  = sqrt_alpha_bar[t_int] * x0_known + sqrt_1m_ab[t_int] * eps_known
            else:
                xt_known = x0_known

            # --- unknown part: one DDPM reverse step ---
            eps_pred  = model(xt, t_tensor)
            coef1     = sqrt_recip_alpha[t_int]
            coef2     = betas[t_int] * sqrt_recip_1m_ab[t_int]
            mu_theta  = coef1 * (xt - coef2 * eps_pred)

            if t_int > 0:
                z          = colored_gaussian_noise(x0_known.shape, alpha=noise_alpha, device=device)
                xt_unknown = mu_theta + betas[t_int].sqrt() * z
            else:
                xt_unknown = mu_theta

            # --- blend ---
            xt = xt_known * km[None, None] + xt_unknown * (~km[None, None])

            # --- resample: re-noise for next inner iteration ---
            if inner < resample - 1 and t_int > 0:
                eps_r = colored_gaussian_noise(x0_known.shape, alpha=noise_alpha, device=device)
                xt    = alpha[t_int].sqrt() * xt + betas[t_int].sqrt() * eps_r

    return xt
