"""
Gaussian Process inpainting for ocean current vector fields.

Given sparse robot path observations, two independent GPs are fitted —
one for the u (east-west) component and one for v (north-south) —
using normalised (row, col) grid coordinates as inputs, then the full
ocean field is predicted from them.

Kernel:  Matérn ν = 5/2  (C²-smooth, physically appropriate for currents)
         + WhiteKernel   (absorbs small observation noise)

Hyperparameters (length scale, noise level) are optimised per snapshot via
log-marginal-likelihood maximisation using scikit-learn's built-in L-BFGS-B
solver.  Because n_obs ≪ n_pred (≈ 150–300 vs ≈ 3900), the full GP is used
without any approximation; typical wall-clock time is under a second per run.

Path generators (random_walk_path, biased_walk_path) live in paths.py at the
repo root and are shared with the DDPM approach.
"""

import os
import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import biased_walk_path, random_walk_path  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# GP reconstruction
# ---------------------------------------------------------------------------

def gp_reconstruct(
    x0_true:      np.ndarray,          # (2, H, W) float32 — u and v channels
    path_mask:    np.ndarray,          # (H, W) bool — observed cells
    land_mask:    np.ndarray,          # (H, W) bool — land cells (excluded)
    length_scale: float = 0.15,        # initial kernel length scale in [0, 1] coords
    noise_level:  float = 1e-4,        # initial white noise variance
    n_restarts:   int   = 2,           # L-BFGS-B restarts for hyperparameter fitting
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the full current field from sparse path observations using GPs.

    Training inputs:  (row, col) coordinates of path cells, normalised to [0, 1].
    Training targets: u (or v) values at those cells.
    Prediction:       all ocean cells (land excluded).

    Args:
        x0_true:      (2, H, W) ground-truth field (channel 0 = u, channel 1 = v).
                      Only the values at path_mask cells are used during fitting.
        path_mask:    (H, W) bool — cells the robot visited (observed cells).
        land_mask:    (H, W) bool — land cells; these are skipped entirely.
        length_scale: initial GP length scale expressed as a fraction of the
                      normalised domain [0, 1].  0.15 ≈ 15% of the domain extent.
        noise_level:  initial white-noise variance (regularises the GP and captures
                      any small measurement error).
        n_restarts:   number of random restarts when optimising hyperparameters.

    Returns:
        u_pred: (H, W) predicted u field  (land pixels = 0)
        v_pred: (H, W) predicted v field  (land pixels = 0)
        u_std:  (H, W) posterior std for u (land pixels = 0)
        v_std:  (H, W) posterior std for v (land pixels = 0)
    """
    H, W = x0_true.shape[1:]

    # Normalised (row, col) coordinates — keeps the length scale
    # hyperparameter in a sensible [0, 1] domain regardless of grid size.
    rows, cols = np.mgrid[0:H, 0:W]
    row_n = rows / (H - 1)
    col_n = cols / (W - 1)

    # ---- Training set: path cells that are not land ----
    obs_idx = path_mask & ~land_mask
    X_obs   = np.stack([row_n[obs_idx], col_n[obs_idx]], axis=1)  # (n_obs, 2)
    y_u     = x0_true[0][obs_idx].astype(np.float64)
    y_v     = x0_true[1][obs_idx].astype(np.float64)

    # ---- Prediction set: all ocean cells ----
    ocean_idx = ~land_mask
    X_pred    = np.stack([row_n[ocean_idx], col_n[ocean_idx]], axis=1)  # (n_ocean, 2)

    # ---- Kernel ----
    # Matérn ν=2.5: C²-smooth, good compromise between RBF (oversmooth)
    # and ν=0.5 (too rough).  WhiteKernel absorbs noise.
    kernel = (
        Matern(
            length_scale=length_scale,
            length_scale_bounds=(1e-3, 5.0),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=noise_level,
            noise_level_bounds=(1e-7, 1e-1),
        )
    )

    # normalize_y=True centres the target, improving numerical stability
    # and making the prior mean equal to the sample mean rather than zero.
    gp_u = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
    )
    gp_v = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
    )

    gp_u.fit(X_obs, y_u)
    gp_v.fit(X_obs, y_v)

    u_ocean, u_ocean_std = gp_u.predict(X_pred, return_std=True)
    v_ocean, v_ocean_std = gp_v.predict(X_pred, return_std=True)

    # ---- Assemble full (H, W) arrays (land stays 0) ----
    u_pred = np.zeros((H, W), dtype=np.float32)
    v_pred = np.zeros((H, W), dtype=np.float32)
    u_std  = np.zeros((H, W), dtype=np.float32)
    v_std  = np.zeros((H, W), dtype=np.float32)

    u_pred[ocean_idx] = u_ocean.astype(np.float32)
    v_pred[ocean_idx] = v_ocean.astype(np.float32)
    u_std[ocean_idx]  = u_ocean_std.astype(np.float32)
    v_std[ocean_idx]  = v_ocean_std.astype(np.float32)

    return u_pred, v_pred, u_std, v_std
