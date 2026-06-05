"""
Voronoi Tessellation-assisted Deep Learning for global field reconstruction.

Reference
---------
Fukami, K., Maulik, R., Ramachandra, N., Fukagata, K., & Taira, K. (2021).
"Global field reconstruction from sparse sensors with Voronoi tessellation-
assisted deep learning." Nature Machine Intelligence, 3, 945-956.
https://arxiv.org/abs/2101.00554

Architecture
------------
1.  VoronoiLayer  -- maps K sparse sensor (position, value) pairs onto a
                     structured (C+1, H, W) grid via nearest-neighbour
                     (Voronoi) tessellation.  The extra channel is a binary
                     mask that marks which grid cells contain a sensor.
2.  VoronoiUNet   -- encoder-decoder U-Net that maps (C+1, H, W) -> (C, H, W).
                     No diffusion / time-step conditioning.
3.  VoronoiNet    -- convenience wrapper that combines both stages into a
                     single nn.Module.

Usage (training on the ocean current dataset)
---------------------------------------------
::
    model = VoronoiNet(H=94, W=44, n_sensors=50, in_ch=2).to(device)
    pred  = model(x0, land_mask=land_mask)    # x0 : (B, 2, 94, 44)
    loss  = F.mse_loss(pred * ocean_mask, x0 * ocean_mask)

Usage (inference from external sensor readings)
-----------------------------------------------
::
    voronoi = model.voronoi.tessellate(sensor_values, sensor_positions)
    pred    = model.unet(voronoi)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_groups(channels: int) -> int:
    """Return the largest divisor of *channels* that is <= 32."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Residual block  (no time-step conditioning)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Pre-activation residual block with GroupNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch),  in_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch,  out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Voronoi tessellation layer
# ---------------------------------------------------------------------------

class VoronoiLayer(nn.Module):
    """
    Converts K sparse sensor readings into a structured-grid field via
    Voronoi (nearest-neighbour) tessellation.

    For each grid point the layer assigns the value of the closest sensor,
    which is equivalent to the Voronoi diagram of the sensor positions.
    An additional binary channel marks which grid cells are closest to a
    sensor (i.e. the Voronoi generator cells).

    Parameters
    ----------
    H, W      : spatial grid dimensions (height, width).
    n_sensors : default number of sensors used when none is supplied to
                forward().
    """

    def __init__(self, H: int, W: int, n_sensors: int = 50):
        super().__init__()
        self.H, self.W = H, W
        self.n_sensors = n_sensors

        # Precompute normalised (row, col) coordinate for every grid cell.
        # Stored as a buffer so it moves with the model to the right device.
        ys = torch.linspace(-1.0, 1.0, H)
        xs = torch.linspace(-1.0, 1.0, W)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")          # (H, W) each
        grid = torch.stack([gy.flatten(), gx.flatten()], dim=1)  # (H*W, 2)
        self.register_buffer("grid_coords", grid)

    # ------------------------------------------------------------------
    # Core: tessellate from caller-supplied sensor positions / values
    # ------------------------------------------------------------------

    def tessellate(
        self,
        sensor_values: torch.Tensor,     # (B, C, K)
        sensor_positions: torch.Tensor,  # (B, K, 2)  normalised to [-1, 1]
    ) -> torch.Tensor:
        """
        Return Voronoi-tessellated field of shape (B, C+1, H, W).

        The first C channels hold the value of the nearest sensor at each
        grid point.  The last channel is a binary mask: 1 at the grid cell
        whose centre is closest to each sensor, 0 elsewhere.
        """
        B, C, K = sensor_values.shape
        HW = self.H * self.W

        # Broadcast grid to batch dimension: (B, HW, 2)
        g = self.grid_coords.unsqueeze(0).expand(B, HW, 2)

        # Batched pairwise L2 distance: (B, HW, K)
        dist = torch.cdist(g.float(), sensor_positions.float())

        # --- Voronoi field: for each grid point, gather value of nearest sensor ---
        nn_idx = dist.argmin(dim=2)                             # (B, HW)
        idx_c  = nn_idx.unsqueeze(1).expand(B, C, HW).contiguous()
        voronoi = torch.gather(sensor_values, 2, idx_c)         # (B, C, HW)
        voronoi = voronoi.view(B, C, self.H, self.W)

        # --- Sensor mask: mark the grid cell closest to each sensor ---
        # dist permuted to (B, K, HW) -> argmin over HW gives 1 cell per sensor
        sensor_cell = dist.permute(0, 2, 1).argmin(dim=2)       # (B, K)
        mask = torch.zeros(B, HW, device=sensor_values.device, dtype=sensor_values.dtype)
        mask.scatter_(1, sensor_cell, 1.0)
        mask = mask.view(B, 1, self.H, self.W)

        return torch.cat([voronoi, mask], dim=1)                 # (B, C+1, H, W)

    # ------------------------------------------------------------------
    # Convenience: sample sensors from a ground-truth field
    # ------------------------------------------------------------------

    def sample_and_tessellate(
        self,
        x: torch.Tensor,                             # (B, C, H, W)
        n_sensors: Optional[int] = None,
        avoid_land: Optional[torch.Tensor] = None,   # (H, W) bool, True = land
    ) -> torch.Tensor:
        """
        Randomly sample *n_sensors* pixels from *x* (avoiding land if a mask
        is supplied), run Voronoi tessellation, and return (B, C+1, H, W).

        Sensors are sampled **independently per batch element** to maximise
        training diversity.
        """
        B, C, H, W = x.shape
        K   = n_sensors if n_sensors is not None else self.n_sensors
        dev = x.device

        if avoid_land is not None:
            pool = (~avoid_land).flatten().nonzero(as_tuple=True)[0]  # (N_ocean,)
            if len(pool) < K:
                raise ValueError(
                    f"Only {len(pool)} non-land pixels available but {K} sensors requested."
                )
            pos_flat = torch.stack(
                [pool[torch.randperm(len(pool), device=dev)[:K]] for _ in range(B)]
            )  # (B, K)
        else:
            pos_flat = torch.stack(
                [torch.randperm(H * W, device=dev)[:K] for _ in range(B)]
            )  # (B, K)

        # Flat index -> normalised (row, col) in [-1, 1]
        rows_n = (pos_flat // W).float() / (H - 1) * 2.0 - 1.0
        cols_n = (pos_flat  % W).float() / (W - 1) * 2.0 - 1.0
        sensor_positions = torch.stack([rows_n, cols_n], dim=2)  # (B, K, 2)

        # Extract ground-truth values at sampled positions
        x_flat        = x.reshape(B, C, H * W)
        idx_c         = pos_flat.unsqueeze(1).expand(B, C, K).contiguous()
        sensor_values = torch.gather(x_flat, 2, idx_c)           # (B, C, K)

        return self.tessellate(sensor_values, sensor_positions)

    def forward(
        self,
        x: torch.Tensor,
        n_sensors: Optional[int] = None,
        avoid_land: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.sample_and_tessellate(x, n_sensors, avoid_land)


# ---------------------------------------------------------------------------
# Voronoi-assisted U-Net  (no time conditioning)
# ---------------------------------------------------------------------------

class VoronoiUNet(nn.Module):
    """
    Encoder-decoder U-Net that reconstructs a full field from a
    Voronoi-tessellated sparse-sensor input.

    Mirrors the spatial structure of the existing UNet in *model.py*:
    the input is padded from (94, 44) to (96, 48) so that five successive
    factor-of-2 downsampling steps reach a (3, 6) bottleneck cleanly.

    Parameters
    ----------
    in_ch   : number of physical field channels C (default 2 for u, v).
    base_ch : base channel width for the U-Net.

    Input  shape : (B, C+1, H, W)  -- C physical channels + 1 sensor mask
    Output shape : (B, C,   H, W)  -- reconstructed full field
    """

    # (left, right, top, bottom)  ->  W: 44 -> 48,  H: 94 -> 96
    _PAD = (2, 2, 1, 1)

    def __init__(self, in_ch: int = 2, base_ch: int = 64):
        super().__init__()
        c   = base_ch
        cin = in_ch + 1   # C physical channels + 1 sensor-position mask

        # Encoder
        self.enc0 = ResBlock(cin,     c   )   # 96 x 48
        self.enc1 = ResBlock(c,       c*2 )   # 48 x 24
        self.enc2 = ResBlock(c*2,     c*4 )   # 24 x 12
        self.enc3 = ResBlock(c*4,     c*8 )   # 12 x  6

        # Bottleneck
        self.mid  = ResBlock(c*8,     c*8 )   #  6 x  3

        # Decoder (skip connections double the input channels)
        self.dec3 = ResBlock(c*8+c*8, c*4 )   # 12 x  6
        self.dec2 = ResBlock(c*4+c*4, c*2 )   # 24 x 12
        self.dec1 = ResBlock(c*2+c*2, c   )   # 48 x 24
        self.dec0 = ResBlock(c   +c,  c   )   # 96 x 48

        self.out_conv = nn.Conv2d(c, in_ch, 1)

        self.down = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, C+1, H, W)  Voronoi-tessellated field + sensor mask.

        Returns
        -------
        (B, C, H, W)  reconstructed full field.
        """
        x  = F.pad(x, self._PAD)               # -> 96 x 48

        # Encoder
        e0 = self.enc0(x)                       # 96 x 48
        e1 = self.enc1(self.down(e0))           # 48 x 24
        e2 = self.enc2(self.down(e1))           # 24 x 12
        e3 = self.enc3(self.down(e2))           # 12 x  6

        # Bottleneck
        h  = self.mid(self.down(e3))            #  6 x  3

        # Decoder with skip connections
        h = self.dec3(torch.cat([self.up(h),  e3], dim=1))  # 12 x  6
        h = self.dec2(torch.cat([self.up(h),  e2], dim=1))  # 24 x 12
        h = self.dec1(torch.cat([self.up(h),  e1], dim=1))  # 48 x 24
        h = self.dec0(torch.cat([self.up(h),  e0], dim=1))  # 96 x 48

        h = self.out_conv(h)
        return h[:, :, 1:-1, 2:-2]             # unpad -> 94 x 44


# ---------------------------------------------------------------------------
# End-to-end VoronoiNet
# ---------------------------------------------------------------------------

class VoronoiNet(nn.Module):
    """
    End-to-end Voronoi tessellation-assisted field reconstruction network.

    Combines *VoronoiLayer* (sparse -> structured grid) and *VoronoiUNet*
    (structured grid -> full field reconstruction).

    Parameters
    ----------
    H, W      : spatial grid dimensions (94, 44 for the ocean dataset).
    n_sensors : default number of sensors sampled per training step.
    in_ch     : number of physical channels (2 for u, v ocean currents).
    base_ch   : base channel width of the internal U-Net.
    """

    def __init__(
        self,
        H:         int = 94,
        W:         int = 44,
        n_sensors: int = 50,
        in_ch:     int = 2,
        base_ch:   int = 64,
    ):
        super().__init__()
        self.voronoi = VoronoiLayer(H, W, n_sensors)
        self.unet    = VoronoiUNet(in_ch=in_ch, base_ch=base_ch)

    def forward(
        self,
        x:         torch.Tensor,                       # (B, C, H, W)
        n_sensors: Optional[int]          = None,
        land_mask: Optional[torch.Tensor] = None,      # (H, W) bool
    ) -> torch.Tensor:
        """
        Randomly sample sensors from *x*, tessellate, and reconstruct.

        Args
        ----
        x         : ground-truth field (B, C, H, W).
        n_sensors : override the default sensor count.
        land_mask : (H, W) bool mask (True = land); sensors are only
                    drawn from ocean pixels when this is provided.

        Returns
        -------
        (B, C, H, W) reconstructed field.
        """
        v = self.voronoi.sample_and_tessellate(x, n_sensors, avoid_land=land_mask)
        return self.unet(v)
