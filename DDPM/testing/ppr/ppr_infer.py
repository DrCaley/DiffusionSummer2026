"""
Predict-Project-Renoise (PPR) inference for divergence-free ocean current reconstruction.

Replaces the hard observation snap in RePaint with a joint projection applied to
the *clean* Tweedie estimate at every reverse step:

    x̂₀ = (xₜ − √(1−ᾱₜ)·ε̂) / √ᾱₜ          (Tweedie denoised estimate)
    x̂₀ = joint_project(x̂₀, ...)             ← divergence-free + data-consistent
    x_{t-1} = DDPM_posterior(xₜ, x̂₀, t)    (re-noises from the projected clean field)

This eliminates the divergence seam produced by RePaint's hard pixel-snapping:
RePaint inserts raw observed values into the *noisy* field at every step, creating
a sharp discontinuity at the observed/unobserved boundary that manifests as
spurious divergence in the final reconstruction. PPR avoids this entirely.
"""

import numpy as np
import torch
from tqdm import tqdm

# These are importable via PYTHONPATH set in .env:
#   DDPM/model/   → diffusion, divfree_projection
#   utils/        → paths
from divfree_projection import joint_project
from paths import biased_walk_path, random_walk_path  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# PPR inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def ppr(
    model:            torch.nn.Module,
    diffusion,                       # DDPM instance (from DDPM/model/diffusion.py)
    x0_known:         torch.Tensor,  # (2, H, W)  true u/v at path cells, 0 elsewhere
    path_mask:        np.ndarray,    # (H, W) bool, True = observed robot-path cell
    land_mask:        np.ndarray,    # (H, W) bool, True = land cell
    r:                int   = 10,    # RePaint resampling iterations per timestep
    proj_iter:        int   = 20,    # POCS iterations inside joint_project
    device:           str   = "cpu",
    projector:        str   = "pocs",
    inference_steps:  int | None = None,  # None = use all T steps
) -> torch.Tensor:
    """
    Run PPR to reconstruct the full current field from sparse path observations.

    Args:
        model:           trained unconditional UNet
        diffusion:       DDPM instance (carries schedule, noise type, alpha_bar etc.)
        x0_known:        (2, H, W) — true u/v at robot path cells, 0 elsewhere
        path_mask:       (H, W) bool — True at cells the robot visited
        land_mask:       (H, W) bool — True at land cells
        r:               RePaint resampling count (r=1 = no resampling)
        proj_iter:       POCS iteration count for joint_project
        device:          torch device string
        projector:       "pocs" (only option for now)
        inference_steps: number of denoising steps (default: full T).
                         E.g. 100 with T=1000 visits t=999,989,...,9 (every 10th).

    Returns:
        x0_pred: (2, H, W) reconstructed vector field (land pixels = 0,
                 approximately divergence-free, matches observations at path cells)
    """
    model.eval()
    H, W = x0_known.shape[1:]

    # Move observations to device
    x0_known_t = x0_known.unsqueeze(0).to(device)        # (1, 2, H, W)

    # Masks on device
    obs_mask   = torch.from_numpy(path_mask).to(device)   # (H, W) bool
    ocean_mask = torch.from_numpy(~land_mask).to(device)  # (H, W) bool
    ocean_f    = ocean_mask.float()[None, None]            # (1, 1, H, W)

    # --- Start from noise (type follows diffusion.noise_type) ---
    xt = diffusion._sample_noise(torch.empty(1, 2, H, W, device=device))
    xt = xt * ocean_f

    n_steps  = inference_steps if inference_steps is not None else diffusion.T
    schedule = diffusion.build_inference_schedule(n_steps)  # [(t, t_prev), ...]

    for t_int, t_prev_int in tqdm(schedule, total=len(schedule), desc="PPR", leave=False):
        for j in range(r):

            # ---- 1. Model predicts noise ε̂ ----
            t_tensor = torch.full((1,), t_int, device=device, dtype=torch.long)
            eps_hat  = model(xt, t_tensor)

            # ---- 2. Tweedie: recover clean estimate x̂₀ ----
            ab     = diffusion.alpha_bar[t_int]
            x0_hat = (xt - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt().clamp(min=1e-8)
            x0_hat = x0_hat.clamp(-1.0, 1.0)

            # ---- 3. Joint projection onto {div-free ∩ matches observations} ----
            x0_hat = joint_project(
                x0_hat, ocean_mask, obs_mask, x0_known_t,
                n_iter=proj_iter, projector=projector,
            )
            x0_hat = x0_hat * ocean_f

            # ---- 4. DDPM posterior: x_{t_prev} from projected x̂₀ ----
            if t_prev_int < 0:
                xt = x0_hat
            else:
                ab_prev  = diffusion.alpha_bar[t_prev_int]
                beta_eff = 1.0 - ab / ab_prev   # always positive: ab_prev > ab
                var      = (1.0 - ab_prev) / (1.0 - ab) * beta_eff

                coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
                coef2 = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
                mean  = coef1 * x0_hat + coef2 * xt

                xt = mean + var.sqrt() * diffusion._sample_noise(xt)

            xt = xt * ocean_f

            # ---- 5. Resample (RePaint: go forward and repeat) ----
            if j < r - 1 and t_prev_int >= 0:
                xt = diffusion.q_sample_from_prev(xt, t_int, t_prev_int)
                xt = xt * ocean_f

    return xt.squeeze(0).cpu()   # (2, H, W)
