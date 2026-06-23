"""
dps_infer.py – Diffusion Posterior Sampling (DPS) inference for the Colored-Noise DDPM.

Algorithm: Chung et al., "Diffusion Posterior Sampling for General Noisy
Inverse Problems", ICLR 2023.  https://arxiv.org/abs/2209.14687

At each reverse step t:
  1. Run the standard DDPM reverse step to get the mean mu_theta(x_t, t).
  2. Compute the Tweedie estimate x̂_0(x_t) = (x_t - sqrt(1-ᾱ_t)*ε_θ) / sqrt(ᾱ_t).
  3. Evaluate the measurement residual: r = y - A(x̂_0), where A(·) masks
     the field to the robot-path pixels.
  4. Apply a gradient correction step:
       x_{t-1} = mu_theta + sigma_t * z  -  zeta_t * ∇_{x_t} ||r||^2

The gradient is computed via autograd through the Tweedie estimate.
Step size zeta_t = step_size / ||r|| (adaptive normalisation).

This module is importable.  Use dps_sample() in your scripts.
"""

import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_parameters.noise_types import colored_gaussian_noise
from model_parameters.noise_schedules import cosine_beta_schedule


def dps_sample(
    model: torch.nn.Module,
    x0_known: torch.Tensor,
    known_mask: torch.Tensor,
    T: int,
    device: torch.device,
    step_size: float = 1.0,
    noise_alpha: float = 2.0,
) -> torch.Tensor:
    """Run DPS reverse diffusion to reconstruct the inpainted field.

    Parameters
    ----------
    model       : trained UNet (eval mode)
    x0_known    : (1, 2, H, W)  – ground truth; observed at known_mask pixels
    known_mask  : (H, W) BoolTensor  – True = observed (robot path)
    T           : int  – number of diffusion timesteps
    device      : torch.device
    step_size   : float  – DPS guidance strength (zeta scale factor)
    noise_alpha : float  – spectral exponent matching training noise

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

    km       = known_mask.to(device)          # (H, W)
    y_obs    = x0_known.to(device)            # (1, 2, H, W) observations

    # Observed values only at robot-path pixels
    y = y_obs[:, :, km]                       # (1, 2, N_obs)

    # Start from pure colored noise
    xt = colored_gaussian_noise(x0_known.shape, alpha=noise_alpha, device=device)

    for t_int in reversed(range(T)):
        xt = xt.detach().requires_grad_(True)

        t_tensor = torch.full((1,), t_int, dtype=torch.long, device=device)

        # Epsilon prediction (with grad)
        eps_pred = model(xt, t_tensor)

        # Tweedie estimate of x_0
        x0_hat = (xt - sqrt_1m_ab[t_int] * eps_pred) / sqrt_alpha_bar[t_int].clamp(min=1e-8)

        # Measurement residual at observed pixels
        y_hat  = x0_hat[:, :, km]             # (1, 2, N_obs)
        resid  = y - y_hat                    # (1, 2, N_obs)
        norm_r = resid.norm()

        # Gradient of ||y - A(x̂_0)||^2 w.r.t. x_t
        loss_dps = (resid ** 2).sum()
        grad     = torch.autograd.grad(loss_dps, xt)[0]   # (1, 2, H, W)

        with torch.no_grad():
            # Standard DDPM reverse mean
            coef1    = sqrt_recip_alpha[t_int]
            coef2    = betas[t_int] * sqrt_recip_1m_ab[t_int]
            mu_theta = coef1 * (xt - coef2 * eps_pred)

            # Adaptive step size: zeta_t = step_size / ||r||
            zeta = step_size / (norm_r.item() + 1e-8)

            if t_int > 0:
                z  = colored_gaussian_noise(x0_known.shape, alpha=noise_alpha, device=device)
                xt = mu_theta + betas[t_int].sqrt() * z - zeta * grad
            else:
                xt = mu_theta - zeta * grad

    return xt.detach()
