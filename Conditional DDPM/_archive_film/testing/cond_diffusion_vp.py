"""
CondDDPMVP — Voronoi-Primed Conditional DDPM.

The forward process starts from the Voronoi tessellation of the robot path
(x_voronoi) rather than the ground-truth field (x0_gt).  The model is trained
to predict an adjusted epsilon that, when applied through the standard reverse
formula, recovers x0_gt rather than x_voronoi.

Forward process:
    x_t = sqrt(ᾱ_t) · x_voronoi + sqrt(1−ᾱ_t) · ε

Adjusted epsilon target:
    ε_target = (x_t − sqrt(ᾱ_t) · x0_gt) / sqrt(1−ᾱ_t)
             = ε + [sqrt(ᾱ_t) / sqrt(1−ᾱ_t)] · (x_voronoi − x0_gt)

At large t (high noise) the SNR term → 0, so ε_target ≈ ε.
At small t (low noise) the correction dominates, teaching the model to undo
the Voronoi residual in the fine-detail pass.

Reverse step p_sample_step is UNCHANGED from CondDDPM — only the learned
score function differs because of the modified training objective.

RePaint initialises x_T from a noised Voronoi field instead of pure Gaussian
noise, consistent with the forward distribution seen during training.
"""

import torch
import torch.nn.functional as F

from cond_diffusion import CondDDPM


class CondDDPMVP(CondDDPM):
    """
    Voronoi-Primed Conditional DDPM.

    Inherits all noise-schedule utilities, q_sample, q_sample_from_prev,
    p_sample_step, and sample from CondDDPM.  Overrides training_loss and
    repaint to use the Voronoi-primed forward process.
    """

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model:      "torch.nn.Module",
        x0_gt:      torch.Tensor,
        x_voronoi:  torch.Tensor,
        land_mask:  torch.Tensor,
        cond:       torch.Tensor,
    ) -> torch.Tensor:
        """
        Voronoi-primed epsilon-prediction loss, ocean pixels only.

        Args:
            model:      CondUNet — forward(xt, t, cond) → predicted noise
            x0_gt:      (B, 2, H, W) ground-truth velocity fields
            x_voronoi:  (B, 2, H, W) Voronoi tessellation (u, v channels only,
                        sensor-mask channel stripped)
            land_mask:  (H, W) bool, True = land (excluded from loss)
            cond:       (B, 3, H, W) path_field conditioning
                        [u_path, v_path, path_mask]

        Returns:
            loss: scalar tensor
        """
        B = x0_gt.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)

        noise    = torch.randn_like(x_voronoi) * self.noise_scale
        sqrt_ab  = self.sqrt_ab[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_mab[t][:, None, None, None]

        # Forward from Voronoi
        x_t = sqrt_ab * x_voronoi + sqrt_mab * noise

        # Adjusted target: epsilon that recovers x0_gt via the reverse formula
        eps_target = (x_t - sqrt_ab * x0_gt) / sqrt_mab.clamp(min=1e-8)

        pred_noise = model(x_t, t, cond)

        ocean = (~land_mask).float()[None, None]   # (1, 1, H, W)
        return F.mse_loss(pred_noise * ocean, eps_target * ocean)

    # ------------------------------------------------------------------
    # RePaint inference — initialise from noised Voronoi
    # ------------------------------------------------------------------

    @torch.no_grad()
    def repaint(
        self,
        model:      "torch.nn.Module",
        cond:       torch.Tensor,       # (1, 3, H, W)
        x0_known:   torch.Tensor,       # (1, 2, H, W) true u/v at path cells, 0 elsewhere
        path_mask:  torch.Tensor,       # (1, 1, H, W) float, 1 = known cell
        ocean_mask: torch.Tensor,       # (1, 1, H, W) float, 1 = ocean cell
        x_voronoi:  torch.Tensor,       # (1, 2, H, W) Voronoi u, v channels
        r:          int = 10,
    ) -> torch.Tensor:
        """
        RePaint reverse diffusion, initialised from a noised Voronoi field.

        The only difference from CondDDPM.repaint is the initialisation:
        instead of x_T ~ N(0, noise_scale² I), we draw:
            x_T = sqrt(ᾱ_{T-1}) · x_voronoi + sqrt(1−ᾱ_{T-1}) · ε

        This is consistent with the forward distribution used during training.
        All subsequent merge / resample steps are identical to the base class.

        Args:
            model:      CondUNet
            cond:       (1, 3, H, W) path_field conditioning
            x0_known:   (1, 2, H, W) ground-truth u/v at path cells, 0 elsewhere
            path_mask:  (1, 1, H, W) float — 1 at known cells
            ocean_mask: (1, 1, H, W) float — 1 at ocean cells
            x_voronoi:  (1, 2, H, W) Voronoi tessellation (u, v only)
            r:          resampling iterations per timestep

        Returns:
            x0_pred: (1, 2, H, W)
        """
        B, _, H, W = x0_known.shape

        # Initialise x_T from noised Voronoi at the highest noise level (t = T-1)
        t_start    = torch.full((B,), self.T - 1, device=self.device, dtype=torch.long)
        noise_init = torch.randn(B, 2, H, W, device=self.device) * self.noise_scale
        sqrt_ab_T  = self.sqrt_ab[t_start][:, None, None, None]
        sqrt_mab_T = self.sqrt_one_mab[t_start][:, None, None, None]
        xt = sqrt_ab_T * x_voronoi + sqrt_mab_T * noise_init
        xt = xt * ocean_mask

        for t_int in reversed(range(self.T)):
            for j in range(r):
                # Step 1: model reverse step
                xt_model = self.p_sample_step(model, xt, t_int, cond)

                # Step 2: forward-diffuse x0_known (ground truth) to t-1
                t_prev   = max(t_int - 1, 0)
                t_prev_t = torch.full((B,), t_prev, device=self.device, dtype=torch.long)
                xt_known, _ = self.q_sample(x0_known, t_prev_t)

                # Step 3: merge known / unknown
                xt = path_mask * xt_known + (1.0 - path_mask) * xt_model
                xt = xt * ocean_mask

                # Step 4: resample forward if not last iteration
                if j < r - 1 and t_int > 0:
                    xt = self.q_sample_from_prev(xt, t_int)
                    xt = xt * ocean_mask

        return xt
