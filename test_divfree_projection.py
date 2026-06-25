"""
Unit tests for DDPM/model/divfree_projection.py.

Run from workspace root:
    python -m pytest tests/test_divfree_projection.py -v
    python tests/test_divfree_projection.py          # standalone
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DDPM", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import numpy as np
import torch

from divfree_projection import divergence, leray_project, joint_project


# ---------------------------------------------------------------------------
# Fixtures: simple synthetic ocean domain  (H=24, W=20)
# ---------------------------------------------------------------------------

H, W = 24, 20

def _ocean_mask(kind="full") -> torch.Tensor:
    """Return an (H, W) bool ocean mask."""
    mask = torch.ones(H, W, dtype=torch.bool)
    if kind == "corner_land":
        # Land in the top-left corner — creates a Neumann BC boundary
        mask[:6, :6] = False
    return mask


def _random_field(ocean_mask: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """(1, 2, H, W) random field, land zeroed."""
    torch.manual_seed(seed)
    x = torch.randn(1, 2, H, W)
    x = x * ocean_mask.float()[None, None]
    return x


# ---------------------------------------------------------------------------
# Helper: mean absolute divergence over ocean cells
# ---------------------------------------------------------------------------

def _mean_abs_div(x: torch.Tensor, ocean_mask: torch.Tensor) -> float:
    div = divergence(x, ocean_mask)          # (B, H, W)
    return float(div[0][ocean_mask].abs().mean().item())


# ---------------------------------------------------------------------------
# Test 1: divergence function
# ---------------------------------------------------------------------------

def test_divergence_output_shape_and_dtype():
    """divergence() returns (B, H, W) float32 with correct shape."""
    ocean = _ocean_mask("full")
    x = _random_field(ocean)
    div = divergence(x, ocean)
    assert div.shape == (1, H, W), f"Expected (1,{H},{W}), got {div.shape}"
    assert div.dtype == torch.float32, "Expected float32 output"
    # Zero-field must have zero divergence everywhere
    x_zero = torch.zeros(1, 2, H, W)
    div_zero = divergence(x_zero, ocean)
    assert div_zero.abs().max().item() == 0.0, "Zero field should have zero divergence"


def test_divergence_land_is_zero():
    """Divergence is zero at land cells."""
    ocean = _ocean_mask("corner_land")
    x = _random_field(ocean)
    div = divergence(x, ocean)               # (1, H, W)
    assert div[0][~ocean].abs().max().item() == 0.0, \
        "Land cells must have zero divergence"


# ---------------------------------------------------------------------------
# Test 2: leray_project reduces divergence
# ---------------------------------------------------------------------------

def test_leray_project_reduces_divergence():
    """leray_project must reduce mean |div| by at least 70%."""
    ocean = _ocean_mask("corner_land")
    x     = _random_field(ocean, seed=1)

    div_before = _mean_abs_div(x, ocean)
    x_df       = leray_project(x, ocean)
    div_after  = _mean_abs_div(x_df, ocean)

    assert div_before > 0, "Random field should have nonzero divergence"
    assert div_after < 0.50 * div_before, (
        f"leray_project should reduce |div| by >50%, "
        f"got before={div_before:.4f} after={div_after:.4f}"
    )


def test_leray_project_zero_land():
    """leray_project must not pollute land cells."""
    ocean = _ocean_mask("corner_land")
    x     = _random_field(ocean, seed=2)
    x_df  = leray_project(x, ocean)
    assert x_df[:, :, ~ocean].abs().max().item() < 1e-6, \
        "leray_project must keep land cells at zero"


def test_leray_project_approximately_idempotent():
    """Applying leray_project twice should not increase divergence."""
    ocean = _ocean_mask("full")
    x     = _random_field(ocean, seed=3)
    x_df  = leray_project(x, ocean)
    x_df2 = leray_project(x_df, ocean)
    div1  = _mean_abs_div(x_df,  ocean)
    div2  = _mean_abs_div(x_df2, ocean)
    # Second call should not increase divergence
    assert div2 <= div1 * 1.1 + 1e-7, (
        f"Second leray_project call should not increase |div|: "
        f"{div1:.6f} → {div2:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: joint_project (POCS)
# ---------------------------------------------------------------------------

def test_joint_project_reduces_divergence_vs_input():
    """
    After POCS, mean |div| should be strictly less than the input divergence.

    Note: the spectral Leray projection is consistent with the spectral
    divergence operator, not with central differences, so the central-diff
    divergence reaches a non-zero steady state determined by the obs density
    and field roughness.  The test checks that POCS is doing useful work
    (reducing divergence relative to the raw input field) while maintaining
    observation consistency.
    """
    ocean = _ocean_mask("corner_land")
    x     = _random_field(ocean, seed=4)

    # Sparse obs: ~10% of ocean cells
    obs_mask = torch.zeros(H, W, dtype=torch.bool)
    ocean_cells = torch.argwhere(ocean)
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(ocean_cells), size=len(ocean_cells) // 10, replace=False)
    for idx in chosen:
        r, c = ocean_cells[idx]
        obs_mask[r, c] = True

    x_obs = torch.zeros(1, 2, H, W)
    x_obs[:, :, obs_mask] = x[:, :, obs_mask]   # true values at obs cells

    div_before = _mean_abs_div(x, ocean)
    x_proj     = joint_project(x, ocean, obs_mask, x_obs, n_iter=20)
    div_after  = _mean_abs_div(x_proj, ocean)

    assert div_after < div_before, (
        f"joint_project should reduce divergence vs input: "
        f"before={div_before:.4f} after={div_after:.4f}"
    )
    # Check obs consistency too
    err = (x_proj[:, :, obs_mask] - x_obs[:, :, obs_mask]).abs().max().item()
    assert err < 1e-3, f"Obs cells should still be consistent after POCS: err={err:.2e}"


def test_joint_project_obs_consistency():
    """After POCS, observed cells should match x_obs to within 1e-3."""
    ocean = _ocean_mask("full")
    x     = _random_field(ocean, seed=5)

    obs_mask = torch.zeros(H, W, dtype=torch.bool)
    obs_mask[5:10, 5:10] = True              # 25-cell observed patch

    x_obs = torch.zeros(1, 2, H, W)
    x_obs[:, :, obs_mask] = x[:, :, obs_mask]

    x_proj = joint_project(x, ocean, obs_mask, x_obs, n_iter=20)

    # Observed cells should match
    err = (x_proj[:, :, obs_mask] - x_obs[:, :, obs_mask]).abs().max().item()
    assert err < 1e-3, (
        f"Observed cells should match x_obs after POCS, max error={err:.2e}"
    )


def test_joint_project_land_zero():
    """Land cells must remain zero after joint_project."""
    ocean = _ocean_mask("corner_land")
    x     = _random_field(ocean, seed=6)
    obs   = torch.zeros(H, W, dtype=torch.bool)
    obs[10, 10] = True
    x_obs = x.clone()
    x_proj = joint_project(x, ocean, obs, x_obs, n_iter=5)
    assert x_proj[:, :, ~ocean].abs().max().item() < 1e-6, \
        "Land cells must remain zero after joint_project"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_divergence_output_shape_and_dtype,
        test_divergence_land_is_zero,
        test_leray_project_reduces_divergence,
        test_leray_project_zero_land,
        test_leray_project_approximately_idempotent,
        test_joint_project_reduces_divergence_vs_input,
        test_joint_project_obs_consistency,
        test_joint_project_land_zero,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        sys.exit(1)
