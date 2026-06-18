"""
Gaussian Process inpainting for ocean current *magnitude* (speed) fields.

This is the magnitude half of a direction/magnitude decomposition:

    final field  =  direction (from the angle-loss DDPM)  ×  speed (this GP)

Given sparse robot-path observations, a single GP is fitted on the scalar
speed |v| = sqrt(u^2 + v^2) at the observed cells, using normalised (row, col)
grid coordinates as inputs, then the full ocean speed field is predicted.

Unlike gp_infer.py (which fits two vector GPs, one per component), this module
models ONLY magnitude.  It deliberately ignores the DDPM's predicted field:
under a pure angle loss the DDPM's magnitudes are meaningless, so the speed is
reconstructed independently from the known speeds alone.

Kernel:  Matérn ν = 5/2  (C²-smooth, physically appropriate for currents)
         + WhiteKernel   (absorbs small observation noise)

Speed is non-negative.  Two target spaces are supported:
    * "linear" (default): fit |v| directly; negative posterior means are
      clamped to 0 (rare for smooth fields).
    * "log": fit log(|v| + eps); the exponential of the posterior mean is
      guaranteed positive and better handles the wide dynamic range of speeds.
"""

import os
import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "utils"))
from paths import biased_walk_path, random_walk_path  # noqa: F401  (re-exported)


def speed_field(x0: np.ndarray) -> np.ndarray:
    """Per-cell speed |v| = sqrt(u^2 + v^2) for a (2, H, W) field."""
    return np.sqrt(x0[0] ** 2 + x0[1] ** 2)


def climatology_speed(dataset, land_mask: np.ndarray) -> np.ndarray:
    """
    Mean speed field over an entire dataset split (the "climatology").

    Averages |v| at every cell across all snapshots, giving a realistic
    per-cell prior for the typical current speed.  Land cells are set to 0.

    Args:
        dataset:   an OceanCurrentDataset (or any indexable yielding (2, H, W)
                   torch tensors / arrays).
        land_mask: (H, W) bool — land cells (set to 0 in the output).

    Returns:
        (H, W) float32 mean-speed field.
    """
    acc = None
    for i in range(len(dataset)):
        x0 = dataset[i]
        x0 = x0.numpy() if hasattr(x0, "numpy") else np.asarray(x0)
        sp = speed_field(x0)
        acc = sp if acc is None else acc + sp
    mean_sp = (acc / len(dataset)).astype(np.float32)
    mean_sp[land_mask] = 0.0
    return mean_sp


# ---------------------------------------------------------------------------
# GP magnitude reconstruction
# ---------------------------------------------------------------------------

def gp_reconstruct_magnitude(
    x0_true:      np.ndarray,          # (2, H, W) float32 — u and v channels
    path_mask:    np.ndarray,          # (H, W) bool — observed cells
    land_mask:    np.ndarray,          # (H, W) bool — land cells (excluded)
    length_scale: float = 0.15,        # initial kernel length scale in [0, 1] coords
    noise_level:  float = 1e-4,        # initial white noise variance
    n_restarts:   int   = 2,           # L-BFGS-B restarts for hyperparameter fitting
    target_space: str   = "linear",    # "linear" | "log"
    log_eps:      float = 1e-3,        # offset for log-space fitting
    climatology:  np.ndarray | None = None,  # (H, W) mean-speed prior, optional
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the full speed (magnitude) field from sparse path observations.

    Training inputs:  (row, col) coordinates of path cells, normalised to [0, 1].
    Training targets: speed |v| (or log|v|) at those cells.
    Prediction:       all ocean cells (land excluded).

    Args:
        x0_true:      (2, H, W) ground-truth field (channel 0 = u, channel 1 = v).
                      Only the speeds at path_mask cells are used during fitting.
        path_mask:    (H, W) bool — cells the robot visited (observed cells).
        land_mask:    (H, W) bool — land cells; these are skipped entirely.
        length_scale: initial GP length scale as a fraction of the normalised
                      domain [0, 1].  0.15 ≈ 15% of the domain extent.
        noise_level:  initial white-noise variance.
        n_restarts:   number of random restarts when optimising hyperparameters.
        target_space: "linear" fits |v| directly (clamped ≥ 0);
                      "log" fits log(|v| + log_eps) and exponentiates back.
        log_eps:      small offset used only when target_space == "log".
        climatology:  optional (H, W) mean-speed field.  When provided, the GP
                      models the RESIDUAL speed (observed − climatology) and the
                      climatology is added back after prediction.  Unobserved
                      regions then fall back to the realistic per-cell average
                      instead of a flat global constant.  Only valid with
                      target_space == "linear" (residuals can be negative).

    Returns:
        mag_pred: (H, W) predicted speed field        (land pixels = 0)
        mag_std:  (H, W) posterior std of the speed   (land pixels = 0)
        mag_true: (H, W) ground-truth speed field     (land pixels = 0)
    """
    if target_space not in ("linear", "log"):
        raise ValueError(f"target_space must be 'linear' or 'log', got {target_space!r}")
    if climatology is not None and target_space == "log":
        raise ValueError("climatology prior is only supported with target_space='linear'")

    H, W = x0_true.shape[1:]

    # Ground-truth speed everywhere (used for both fitting and evaluation).
    mag_full = speed_field(x0_true).astype(np.float64)

    # Normalised (row, col) coordinates — keeps the length-scale hyperparameter
    # in a sensible [0, 1] domain regardless of grid size.
    rows, cols = np.mgrid[0:H, 0:W]
    row_n = rows / (H - 1)
    col_n = cols / (W - 1)

    # ---- Training set: path cells that are not land ----
    obs_idx = path_mask & ~land_mask
    X_obs   = np.stack([row_n[obs_idx], col_n[obs_idx]], axis=1)  # (n_obs, 2)
    y_mag   = mag_full[obs_idx]

    # Optional climatology prior: model the residual from the mean-speed field
    # so unobserved regions fall back to climatology rather than a flat constant.
    if climatology is not None:
        clim = np.asarray(climatology, dtype=np.float64)
        y_mag = y_mag - clim[obs_idx]

    if target_space == "log":
        y_train = np.log(y_mag + log_eps)
    else:
        y_train = y_mag

    # ---- Prediction set: all ocean cells ----
    ocean_idx = ~land_mask
    X_pred    = np.stack([row_n[ocean_idx], col_n[ocean_idx]], axis=1)  # (n_ocean, 2)

    # ---- Kernel: Matérn ν=2.5 (C²-smooth) + white noise ----
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

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
    )
    gp.fit(X_obs, y_train)

    pred_ocean, std_ocean = gp.predict(X_pred, return_std=True)

    if target_space == "log":
        # Exponentiate back to speed; std is in log space so we approximate the
        # linear-space std via the delta method: std_lin ≈ exp(mean) * std_log.
        mean_lin = np.exp(pred_ocean) - log_eps
        std_lin  = np.exp(pred_ocean) * std_ocean
        pred_ocean, std_ocean = mean_lin, std_lin

    # Add the climatology prior back to the predicted residual.
    if climatology is not None:
        clim = np.asarray(climatology, dtype=np.float64)
        pred_ocean = pred_ocean + clim[ocean_idx]

    # Speed is non-negative.
    pred_ocean = np.clip(pred_ocean, 0.0, None)

    # ---- Assemble full (H, W) arrays (land stays 0) ----
    mag_pred = np.zeros((H, W), dtype=np.float32)
    mag_std  = np.zeros((H, W), dtype=np.float32)
    mag_true = np.zeros((H, W), dtype=np.float32)

    mag_pred[ocean_idx] = pred_ocean.astype(np.float32)
    mag_std[ocean_idx]  = std_ocean.astype(np.float32)
    mag_true[ocean_idx] = mag_full[ocean_idx].astype(np.float32)

    return mag_pred, mag_std, mag_true
