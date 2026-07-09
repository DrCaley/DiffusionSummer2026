"""
DDPM diffusion utilities for the Voronoi-correlated noise experiment.

Noise schedule : linear (beta_1=1e-4 → beta_T=0.02)
Noise type     : Voronoi-correlated — spatially piecewise-constant Gaussian.

How it works
------------
At each noise injection, N_SEEDS random seed points are scattered across
ocean grid cells.  Every ocean cell is assigned to its nearest seed point
(nearest-neighbour / Voronoi tessellation).  A single scalar noise value
is drawn from N(0, noise_std²) for each Voronoi cell, and that value is
broadcast to every grid cell in the region.  The result is noise that is:
  - Locally constant within a Voronoi cell  (spatially correlated at
    the scale of typical cell size ~H*W/N_SEEDS grid cells)
  - Uncorrelated across different Voronoi cells
  - Sharp-edged at Voronoi boundaries (unlike smooth spectral filters)

This mimics the patchiness of ocean mesoscale features — eddy-scale blobs
of correlated anomaly with sharp fronts between patches.

The seed count N_SEEDS controls the spatial scale:
  - Few seeds  → large cells → long-range correlation (red-like)
  - Many seeds → small cells → approaches white noise

Default N_SEEDS = 50  gives cells ~83 grid points each on a 94×44 grid.

Loss : eps-MSE  +  CURL_DIV_WEIGHT * curl_div_loss
"""

import torch
import torch.nn.functional as F
import numpy as np

from loss_functions import curl_div_loss

# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
CURL_DIV_WEIGHT = 0.002
NOISE_TYPE      = "voronoi"

# Number of Voronoi seed points per noise sample.
# Larger → smaller cells → less spatial correlation.
N_SEEDS = 50


# ---------------------------------------------------------------------------
# Voronoi cell assignment  (precomputed once per grid shape)
# ---------------------------------------------------------------------------

def _voronoi_label_map(H: int, W: int, n_seeds: int,
                       land_mask: np.ndarray | None,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Return an (H, W) int32 array where each ocean cell holds the index
    [0, n_seeds) of its nearest seed point.  Land cells are labelled -1.

    Seed points are sampled uniformly from ocean cells.
    """
    if land_mask is not None:
        ocean_cells = np.argwhere(~land_mask)   # (N_ocean, 2)
    else:
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ocean_cells = np.stack([ys.ravel(), xs.ravel()], axis=1)

    n_seeds = min(n_seeds, len(ocean_cells))
    idx     = rng.choice(len(ocean_cells), size=n_seeds, replace=False)
    seeds   = ocean_cells[idx]   # (n_seeds, 2)  — (row, col) pairs

    # Build full grid coordinates
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid_rc    = np.stack([rows.ravel(), cols.ravel()], axis=1)  # (H*W, 2)

    # Nearest-neighbour assignment (L2 distance)
    # seeds: (n_seeds, 2), grid_rc: (H*W, 2)
    diff  = grid_rc[:, None, :] - seeds[None, :, :]   # (H*W, n_seeds, 2)
    dist2 = (diff ** 2).sum(axis=2)                    # (H*W, n_seeds)
    labels = dist2.argmin(axis=1).reshape(H, W).astype(np.int32)

    if land_mask is not None:
        labels[land_mask] = -1

    return labels


# ---------------------------------------------------------------------------
# DDPM — linear schedule + Voronoi-correlated noise
# ---------------------------------------------------------------------------

class DDPM:
    """
    DDPM with a linear beta schedule and Voronoi-correlated noise.

    At every noise draw, N_SEEDS Voronoi cells are generated and each
    receives a single independent Gaussian draw broadcast to all its cells.
    This produces spatially correlated noise with mesoscale patch structure
    while remaining statistically zero-mean and normalised to noise_std.

    Args:
        land_mask_np : optional (H, W) bool numpy array (True = land).
                       If supplied, seed points and cell assignments avoid land.
        n_seeds      : number of Voronoi seeds (controls patch size).
    """

    def __init__(self, T: int = 1000, device: str = "cpu",
                 noise_std: float = 1.0,
                 curl_div_weight: float = CURL_DIV_WEIGHT,
                 land_mask_np: np.ndarray | None = None,
                 n_seeds: int = N_SEEDS):
        self.T               = T
        self.device          = device
        self.noise_std       = noise_std
        self.curl_div_weight = curl_div_weight
        self.land_mask_np    = land_mask_np
        self.n_seeds         = n_seeds
        self._rng            = np.random.default_rng(0)

        betas             = torch.linspace(1e-4, 0.02, T)
        self.betas        = betas.to(device)
        alphas            = 1.0 - self.betas
        self.alphas       = alphas
        self.alpha_bar    = torch.cumprod(alphas, dim=0)
        self.alpha_bar_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bar[:-1]]
        )
        self.sqrt_ab      = self.alpha_bar.sqrt()
        self.sqrt_one_mab = (1.0 - self.alpha_bar).sqrt()

        # Cache grid shape after first call
        self._H = None
        self._W = None

    # ------------------------------------------------------------------
    # Noise generation
    # ------------------------------------------------------------------

    def _make_noise(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Voronoi-correlated noise.

        For each sample in the batch, independently:
          1. Generate N_SEEDS Voronoi cells on the (H, W) grid.
          2. Draw one N(0,1) scalar per cell.
          3. Map each grid cell to its cell's scalar value.
          4. Normalise batch-wide to noise_std.
        """
        B, C, H, W = x0.shape
        device = x0.device

        # Build Voronoi label map once per unique grid shape
        noise_np = np.zeros((B, C, H, W), dtype=np.float32)

        for b in range(B):
            labels = _voronoi_label_map(
                H, W, self.n_seeds, self.land_mask_np, self._rng
            )
            n_cells = labels.max() + 1
            # One scalar per channel per cell
            cell_vals = self._rng.standard_normal((C, n_cells)).astype(np.float32)

            for c in range(C):
                cell_map = np.where(labels >= 0, cell_vals[c, labels], 0.0)
                noise_np[b, c] = cell_map

        noise = torch.from_numpy(noise_np).to(device)

        # Normalise to noise_std
        std = noise.std() + 1e-8
        return noise / std * self.noise_std

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε."""
        if noise is None:
            noise = self._make_noise(x0)
        sqrt_ab  = self.sqrt_ab[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_mab[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_mab * noise, noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model:     torch.nn.Module,
        x0:        torch.Tensor,
        land_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)

        ocean    = (~land_mask).float()[None, None]
        eps_loss = F.mse_loss(pred_noise * ocean, noise * ocean)

        if self.curl_div_weight == 0.0:
            return eps_loss

        ab     = self.alpha_bar[t][:, None, None, None]
        x0_hat = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        cd_loss = curl_div_loss(x0_hat, x0, ocean)
        return eps_loss + self.curl_div_weight * cd_loss

    # ------------------------------------------------------------------
    # Single reverse step  p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(
        self,
        model:      torch.nn.Module,
        xt:         torch.Tensor,
        t_int:      int,
        t_prev_int: int | None = None,
    ) -> torch.Tensor:
        if t_prev_int is None:
            t_prev_int = max(t_int - 1, 0)

        B = xt.shape[0]
        t = torch.full((B,), t_int, device=self.device, dtype=torch.long)

        pred_noise = model(xt, t)

        ab      = self.alpha_bar[t_int]
        ab_prev = (
            self.alpha_bar[t_prev_int]
            if t_prev_int > 0
            else torch.tensor(1.0, device=self.device)
        )

        alpha_eff = ab / ab_prev
        beta_eff  = 1.0 - alpha_eff

        x0_pred = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1.5, 1.5)

        if t_int == 0:
            return x0_pred

        coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
        coef2 = alpha_eff.sqrt() * (1.0 - ab_prev) / (1.0 - ab)
        mean  = coef1 * x0_pred + coef2 * xt

        var = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
        # Reverse-step noise: use Voronoi-correlated noise for consistency
        step_noise = self._make_noise(xt)
        return mean + var.sqrt() * step_noise

    # ------------------------------------------------------------------
    # One forward step  q(x_t | x_{t-1})  — used by RePaint resampling
    # ------------------------------------------------------------------

    def q_sample_from_prev(
        self, x_prev: torch.Tensor, t_int: int, t_prev_int: int = -1
    ) -> torch.Tensor:
        if t_prev_int < 0:
            t_prev_int = max(t_int - 1, 0)
        ab      = self.alpha_bar[t_int]
        ab_prev = (
            self.alpha_bar[t_prev_int]
            if t_prev_int > 0
            else torch.tensor(1.0, device=self.device)
        )
        alpha_eff  = ab / ab_prev
        step_noise = self._make_noise(x_prev)
        return alpha_eff.sqrt() * x_prev + (1.0 - alpha_eff).sqrt() * step_noise
