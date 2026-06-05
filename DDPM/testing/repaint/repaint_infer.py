"""
RePaint-style inference for ocean current inpainting.

At inference time, we know the u/v values only along the robot's path
(a random walk on ocean grid cells).  The DDPM fills in everything else.

RePaint algorithm (per timestep t, repeated r times):
  1. Reverse step  → x_{t-1} from x_t via the model
  2. Merge         → known path cells get q(x_{t-1} | x_0_known),
                     unknown cells keep the model's prediction
  3. Resample      → if not the last iteration, go forward one step
                     x_t = q(x_t | x_{t-1}) and repeat
  4. Advance       → move to t-1 with the merged x_{t-1}
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import random_walk_path, biased_walk_path  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# RePaint inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def repaint(
    model:      torch.nn.Module,
    diffusion,                        # DDPM instance
    x0_known:   torch.Tensor,         # (2, H, W) observed field (0 outside path)
    path_mask:  np.ndarray,           # (H, W) bool, True = known path cell
    land_mask:  np.ndarray,           # (H, W) bool, True = land
    r:          int = 10,             # resampling iterations per timestep
    device:     str = "cpu",
) -> torch.Tensor:
    """
    Run RePaint to reconstruct the full current field from sparse path observations.

    Args:
        model:     trained unconditional UNet
        diffusion: DDPM instance
        x0_known:  (2, H, W) tensor — true u/v at path cells, 0 elsewhere
        path_mask: (H, W) bool — True at cells the robot visited
        land_mask: (H, W) bool — True at land cells
        r:         RePaint resampling count (r=1 = no resampling, r=10 = paper default)
        device:    torch device string

    Returns:
        x0_pred: (2, H, W) reconstructed vector field (land pixels = 0)
    """
    model.eval()
    H, W = x0_known.shape[1:]

    x0_known = x0_known.unsqueeze(0).to(device)      # (1, 2, H, W)

    # Masks as (1, 1, H, W) float for broadcasting
    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t  = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t = 1.0 - land_t

    # Start from pure noise
    xt = torch.randn(1, 2, H, W, device=device)
    xt = xt * ocean_t  # land stays 0

    T = diffusion.T

    for t_int in reversed(range(T)):
        for j in range(r):
            # --- Step 1: model reverse step for unknown pixels ---
            xt_unknown = diffusion.p_sample_step(model, xt, t_int)

            # --- Step 2: forward-diffuse x0_known to timestep (t-1 or 0) ---
            t_prev = max(t_int - 1, 0)
            t_prev_tensor = torch.full((1,), t_prev, device=device, dtype=torch.long)
            xt_known, _ = diffusion.q_sample(x0_known, t_prev_tensor)

            # --- Step 3: merge ---
            xt_merged = known_t * xt_known + (1.0 - known_t) * xt_unknown
            xt_merged = xt_merged * ocean_t   # keep land at 0

            # --- Step 4: resample (go forward one step) if not last iteration ---
            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int)
                xt = xt * ocean_t
            else:
                xt = xt_merged

    return xt.squeeze(0).cpu()   # (2, H, W)
