"""
Robot path generators — shared between the DDPM and GP inpainting approaches.
"""

import numpy as np


def random_walk_path(
    land_mask: np.ndarray,
    n_steps:   int = 150,
    seed:      int | None = None,
) -> np.ndarray:
    """
    Simulate a slow robot doing a random walk on ocean grid cells.

    Args:
        land_mask: (H, W) bool, True = land (robot cannot enter)
        n_steps:   number of steps (robot can revisit cells)
        seed:      optional RNG seed

    Returns:
        path_mask: (H, W) bool, True = cells the robot visited
    """
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape

    ocean_cells = list(zip(*np.where(~land_mask)))
    if not ocean_cells:
        raise ValueError("No ocean cells found in land_mask.")

    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c = int(start[0]), int(start[1])

    path_mask = np.zeros((H, W), dtype=bool)
    path_mask[r, c] = True

    for _ in range(n_steps - 1):
        neighbors = [
            (r + dr, c + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dr < H and 0 <= c + dc < W and not land_mask[r + dr, c + dc]
        ]
        if neighbors:
            r, c = neighbors[rng.integers(len(neighbors))]
            path_mask[r, c] = True

    return path_mask


def biased_walk_path(
    land_mask:     np.ndarray,
    n_steps:       int = 150,
    seed:          int | None = None,
    straight_bias: float = 0.75,
) -> np.ndarray:
    """
    Random walk with directional persistence: the robot strongly prefers
    to continue in its current direction, producing a roughly straight path
    that still meanders and automatically navigates around land.

    Args:
        land_mask:     (H, W) bool, True = land
        n_steps:       number of steps
        seed:          optional RNG seed
        straight_bias: weight given to continuing straight (0–1).
                       0.75 → ~75% chance to continue, ~12.5% each side-step,
                       very small chance to reverse.

    Returns:
        path_mask: (H, W) bool, True = cells the robot visited
    """
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape

    ocean_cells = list(zip(*np.where(~land_mask)))
    if not ocean_cells:
        raise ValueError("No ocean cells found in land_mask.")

    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c = int(start[0]), int(start[1])

    path_mask = np.zeros((H, W), dtype=bool)
    path_mask[r, c] = True

    all_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cur_dir  = all_dirs[rng.integers(4)]

    # Visit-count grid for exploration bonus (unvisited neighbours preferred)
    visit_count = np.zeros((H, W), dtype=np.float32)
    visit_count[r, c] = 1.0

    for _ in range(n_steps - 1):
        valid = [
            (dr, dc)
            for dr, dc in all_dirs
            if 0 <= r + dr < H and 0 <= c + dc < W and not land_mask[r + dr, c + dc]
        ]
        if not valid:
            break

        side = (1.0 - straight_bias) / 2.0
        weights = []
        for dr, dc in valid:
            dot = dr * cur_dir[0] + dc * cur_dir[1]
            if dot == 1:    # straight ahead
                w = straight_bias
            elif dot == 0:  # perpendicular
                w = side
            else:           # reverse — strongly discouraged
                w = side * 0.001

            # Exploration bonus: scale down weight if neighbour already visited
            nr, nc = r + dr, c + dc
            novelty = 1.0 / (1.0 + visit_count[nr, nc])
            weights.append(w * novelty)

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        idx = rng.choice(len(valid), p=weights)
        dr, dc = valid[idx]
        r, c = r + dr, c + dc
        cur_dir = (dr, dc)
        visit_count[r, c] += 1.0
        path_mask[r, c] = True

    return path_mask


def basic_robot_path(
    land_mask:    np.ndarray,
    segment_len:  int = 10,
    seed:         int | None = None,
) -> np.ndarray:
    """
    Two-segment straight-line path with a single random turn in the middle.

    The robot travels straight for `segment_len` steps, makes one random
    heading change (90 deg right, 45 deg right, straight, 45 deg left, or
    90 deg left), then travels straight for another `segment_len` steps.
    Total path length is at most 2 * segment_len + 1 cells.

    Uses 8-directional movement so 45 deg turns are exact on the grid.
    Steps that would enter land or leave the grid are skipped silently.

    Args:
        land_mask:   (H, W) bool, True = land
        segment_len: steps per straight segment (default 10)
        seed:        optional RNG seed

    Returns:
        path_mask: (H, W) bool, True = cells the robot visited
    """
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape

    # 8 directions clockwise: N, NE, E, SE, S, SW, W, NW
    ALL_DIRS = [
        (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
        ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
    ]

    ocean_cells = list(zip(*np.where(~land_mask)))
    if not ocean_cells:
        raise ValueError("No ocean cells found in land_mask.")

    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c  = int(start[0]), int(start[1])

    path_mask = np.zeros((H, W), dtype=bool)
    path_mask[r, c] = True

    target = 2 * segment_len + 1  # start cell + two full segments

    for _ in range(10_000):
        r, c    = int(start[0]), int(start[1])
        pm      = np.zeros((H, W), dtype=bool)
        pm[r, c] = True
        dir_idx = int(rng.integers(8))

        def walk(n, direction):
            nonlocal r, c
            for _ in range(n):
                dr, dc = ALL_DIRS[direction]
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not land_mask[nr, nc]:
                    r, c = nr, nc
                    pm[r, c] = True
                else:
                    return False  # blocked
            return True

        seg1_ok = walk(segment_len, dir_idx)

        offset  = int(rng.choice([-2, -1, 0, 1, 2]))
        dir_idx = (dir_idx + offset) % 8

        seg2_ok = walk(segment_len, dir_idx)

        if seg1_ok and seg2_ok and pm.sum() == target:
            return pm

        # Try a fresh start position and heading next attempt
        start = ocean_cells[rng.integers(len(ocean_cells))]

    # Fallback: return whatever the last attempt produced
    return pm
