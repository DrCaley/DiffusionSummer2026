from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm_groups(channels: int) -> int:
    groups = min(32, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        device = timesteps.device
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=device).float() / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.act(self.norm1(x)))
        time_term = self.time_proj(self.act(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        x = x + time_term
        x = self.conv2(self.dropout(self.act(self.norm2(x))))
        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        condition_channels: int = 0,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        time_embedding_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim * 4),
        )
        time_channels = time_embedding_dim * 4

        self.input_conv = nn.Conv2d(in_channels + condition_channels, base_channels, kernel_size=3, padding=1)

        down_stages = []
        current_channels = base_channels
        for multiplier in channel_mults:
            out_channels = base_channels * multiplier
            blocks = nn.ModuleList(
                [ResBlock(current_channels if block_index == 0 else out_channels, out_channels, time_channels, dropout) for block_index in range(num_res_blocks)]
            )
            down_stages.append(nn.ModuleDict({"blocks": blocks, "downsample": Downsample(out_channels)}))
            current_channels = out_channels
        self.down_stages = nn.ModuleList(down_stages)

        self.middle_block1 = ResBlock(current_channels, current_channels, time_channels, dropout)
        self.middle_block2 = ResBlock(current_channels, current_channels, time_channels, dropout)

        up_stages = []
        reversed_mults = list(channel_mults)[::-1]
        for index, multiplier in enumerate(reversed_mults):
            skip_channels = base_channels * multiplier
            out_channels = base_channels * reversed_mults[index + 1] if index + 1 < len(reversed_mults) else base_channels
            blocks = nn.ModuleList([ResBlock(current_channels + skip_channels, out_channels, time_channels, dropout), ResBlock(out_channels, out_channels, time_channels, dropout)])
            up_stages.append(nn.ModuleDict({"upsample": Upsample(current_channels), "blocks": blocks}))
            current_channels = out_channels
        self.up_stages = nn.ModuleList(up_stages)

        self.output_norm = nn.GroupNorm(_group_norm_groups(current_channels), current_channels)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        height, width = x.shape[-2:]
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return padded, (pad_left, pad_right, pad_top, pad_bottom)

    @staticmethod
    def _crop_to_original(x: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
        pad_left, pad_right, pad_top, pad_bottom = padding
        return x[..., pad_top : x.shape[-2] - pad_bottom, pad_left : x.shape[-1] - pad_right]

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        original_shape = x.shape[-2:]
        x, padding = self._pad_to_multiple(x, 16)
        time_embedding = self.time_embedding(timesteps)

        if self.condition_channels > 0:
            if conditioning is None:
                conditioning = torch.zeros(
                    x.shape[0],
                    self.condition_channels,
                    x.shape[-2],
                    x.shape[-1],
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                if conditioning.shape[0] != x.shape[0]:
                    raise ValueError(f"Expected conditioning batch {x.shape[0]}, got {conditioning.shape[0]}")
                conditioning = conditioning.to(device=x.device, dtype=x.dtype)
                pad_left, pad_right, pad_top, pad_bottom = padding
                if any((pad_left, pad_right, pad_top, pad_bottom)):
                    conditioning = F.pad(conditioning, (pad_left, pad_right, pad_top, pad_bottom))
            x = torch.cat([x, conditioning], dim=1)
        elif conditioning is not None:
            raise ValueError("Conditioning was provided to an unconditional UNetModel")

        x = self.input_conv(x)
        skips: list[torch.Tensor] = []

        for stage in self.down_stages:
            for block in stage["blocks"]:
                x = block(x, time_embedding)
            skips.append(x)
            x = stage["downsample"](x)

        x = self.middle_block1(x, time_embedding)
        x = self.middle_block2(x, time_embedding)

        for stage in self.up_stages:
            x = stage["upsample"](x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in stage["blocks"]:
                x = block(x, time_embedding)

        x = self.output_conv(self.output_act(self.output_norm(x)))
        x = self._crop_to_original(x, padding)
        if x.shape[-2:] != original_shape:
            x = F.interpolate(x, size=original_shape, mode="bilinear", align_corners=False)
        return x
