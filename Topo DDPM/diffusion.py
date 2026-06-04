"""
Topological DDPM — extends the base DDPM with a combined
epsilon-MSE + topological (curl + divergence) loss.
"""

import importlib.util
import os

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Load the base DDPM from the sibling 'Basic DDPM' folder.
# Using importlib avoids sys.path mutation and any module-name collisions
# (both folders have a 'diffusion.py', so a simple sys.path insert would
# cause the wrong one to be found on the second import).
# ---------------------------------------------------------------------------
_base_path = os.path.join(os.path.dirname(__file__), "..", "Basic DDPM", "diffusion.py")
_spec = importlib.util.spec_from_file_location(
    "basic_ddpm_diffusion", os.path.abspath(_base_path)
)
_base_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_mod)

DDPM = _base_mod.DDPM   # re-export so train.py can do: from diffusion import TopoDDPM


class TopoDDPM(DDPM):
    """
    DDPM subclass that adds a topological regularisation term to the loss.

    In addition to the standard epsilon-MSE loss, it reconstructs the
    denoised field x̂₀ from the model prediction and penalises differences
    in the curl (vorticity) and divergence between prediction and ground truth.

        Loss = MSE(ε̂, ε)  +  λ · MSE(topo(x̂₀), topo(x₀))

    where topo(·) = [curl, divergence] computed via central-difference
    finite differences.  Both terms are masked to ocean pixels only.
    """

    def _topology_features(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute curl and divergence of a (B, 2, H, W) vector field
        using central-difference convolution kernels.

        Args:
            field: (B, 2, H, W) — channel 0 = u (zonal), channel 1 = v (meridional)

        Returns:
            (B, 2, H, W) — channel 0 = curl (vorticity), channel 1 = divergence
        """
        u = field[:, 0:1]   # (B, 1, H, W)
        v = field[:, 1:2]

        # Central-difference kernels: [-1, 0, 1] / 2
        kx = torch.tensor(
            [[[[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]]]],
            device=field.device,
        ) / 2.0

        ky = torch.tensor(
            [[[[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]]]],
            device=field.device,
        ) / 2.0

        du_dx = F.conv2d(u, kx, padding=1)
        du_dy = F.conv2d(u, ky, padding=1)
        dv_dx = F.conv2d(v, kx, padding=1)
        dv_dy = F.conv2d(v, ky, padding=1)

        curl = dv_dx - du_dy   # vorticity: dv/dx - du/dy
        div  = du_dx + dv_dy   # divergence: du/dx + dv/dy

        return torch.cat([curl, div], dim=1)   # (B, 2, H, W)

    def training_loss(
        self,
        model:       torch.nn.Module,
        x0:          torch.Tensor,
        land_mask:   torch.Tensor,
        topo_weight: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combined epsilon-MSE + topological loss, masked to ocean pixels.

        Args:
            model:       UNet that predicts noise
            x0:          (B, 2, H, W) clean fields
            land_mask:   (H, W) bool, True = land (excluded from loss)
            topo_weight: λ — weight of the topological term

        Returns:
            (total_loss, eps_loss, topo_loss) — all three for logging
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)

        ocean = (~land_mask).float()[None, None]   # (1, 1, H, W)

        # --- Epsilon (noise-prediction) MSE — identical to base DDPM ---
        eps_loss = F.mse_loss(pred_noise * ocean, noise * ocean)

        # --- Reconstruct x̂₀ from xt and the predicted noise ---
        sqrt_ab  = self.sqrt_ab[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_mab[t][:, None, None, None]
        x0_pred  = (xt - sqrt_mab * pred_noise) / sqrt_ab.clamp(min=1e-8)

        # --- Topological features on ocean pixels only ---
        topo_pred = self._topology_features(x0_pred) * ocean
        topo_true = self._topology_features(x0)       * ocean
        topo_loss = F.mse_loss(topo_pred, topo_true)

        total = eps_loss + topo_weight * topo_loss
        return total, eps_loss, topo_loss
