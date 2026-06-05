"""
Multi-Loss DDPM — thin wrapper around the unified DDPM in DDPM/model/diffusion.py.

All loss logic (curl_div, spectral, okubo_weiss, wasserstein) now lives in
Model Parameters/loss_functions.py and is used directly by the base DDPM class.

This file exists for backward compatibility.  MultiLossDDPM is an alias for DDPM.
"""

import importlib.util
import os

# Load the unified DDPM from DDPM/model/diffusion.py
_base_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "DDPM", "model", "diffusion.py"
)
_spec = importlib.util.spec_from_file_location(
    "ddpm_diffusion", os.path.abspath(_base_path)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DDPM         = _mod.DDPM
LOSS_MODES   = _mod.LOSS_MODES
DEFAULT_WEIGHTS = _mod.DEFAULT_WEIGHTS

# Backward-compatible alias
MultiLossDDPM = DDPM
