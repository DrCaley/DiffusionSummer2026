from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RobotPath:
    coordinates: list[tuple[int, int]]
    mask: torch.Tensor


def _valid_neighbors(position: tuple[int, int], ocean_mask: np.ndarray) -> list[tuple[int, int]]:
    row, col = position
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    valid = []
    height, width = ocean_mask.shape
    for next_row, next_col in candidates:
        if 0 <= next_row < height and 0 <= next_col < width and ocean_mask[next_row, next_col]:
            valid.append((next_row, next_col))
    return valid


def sample_robot_path(land_mask: np.ndarray, steps: int = 150, seed: int | None = None) -> RobotPath:
    rng = np.random.default_rng(seed)
    ocean_mask = ~land_mask.astype(bool)
    ocean_cells = np.argwhere(ocean_mask)
    if ocean_cells.size == 0:
        raise ValueError("No ocean cells found")

    start_index = int(rng.integers(len(ocean_cells)))
    current = tuple(int(v) for v in ocean_cells[start_index])
    coordinates = [current]
    momentum = (0, 0)

    for _ in range(max(1, steps - 1)):
        neighbors = _valid_neighbors(current, ocean_mask)
        if not neighbors:
            current = tuple(int(v) for v in ocean_cells[int(rng.integers(len(ocean_cells)))])
            coordinates.append(current)
            momentum = (0, 0)
            continue

        weights = []
        for neighbor in neighbors:
            delta = (neighbor[0] - current[0], neighbor[1] - current[1])
            score = 1.0
            if delta == momentum:
                score = 2.5
            elif momentum != (0, 0) and delta == (-momentum[0], -momentum[1]):
                score = 0.5
            weights.append(score)

        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        choice = neighbors[int(rng.choice(len(neighbors), p=probabilities))]
        momentum = (choice[0] - current[0], choice[1] - current[1])
        current = choice
        coordinates.append(current)

    mask = torch.zeros_like(torch.from_numpy(land_mask), dtype=torch.bool)
    for row, col in coordinates:
        mask[row, col] = True
    return RobotPath(coordinates=coordinates, mask=mask)


def build_observation_mask(path: RobotPath | torch.Tensor) -> torch.Tensor:
    if isinstance(path, RobotPath):
        return path.mask
    return path


def observed_field(field: torch.Tensor, observation_mask: torch.Tensor) -> torch.Tensor:
    mask = observation_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return field * mask.to(device=field.device, dtype=field.dtype)


def _expand_condition_mask(mask: torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1, -1)
    return mask.to(device=device, dtype=dtype)


def build_inpainting_condition(
    observed: torch.Tensor,
    observation_mask: torch.Tensor,
    land_mask: torch.Tensor,
) -> torch.Tensor:
    if observed.dim() != 4 or observed.shape[1] != 2:
        raise ValueError(f"Expected observed field shape (B, 2, H, W), got {tuple(observed.shape)}")

    batch = observed.shape[0]
    device = observed.device
    dtype = observed.dtype
    obs_mask = _expand_condition_mask(observation_mask, batch, device, dtype)
    land = _expand_condition_mask(land_mask, batch, device, dtype)
    return torch.cat([observed, obs_mask, land], dim=1)
