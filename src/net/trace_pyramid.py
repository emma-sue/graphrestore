"""Low-level state-change trace pyramid for the program planner."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
        )
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class DepthwiseDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=2,
            padding=1,
            groups=in_channels,
        )
        self.project = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.project(self.depthwise(x))))


@dataclass(frozen=True)
class TraceFeatures:
    half: torch.Tensor
    quarter: torch.Tensor
    eighth: torch.Tensor


class TracePyramid(nn.Module):
    """Encode ``concat(x0, xt, xt-x0, abs(xt-x0))`` at three scales."""

    def __init__(
        self,
        in_channels: int = 12,
        channels: tuple[int, int, int] = (48, 96, 192),
    ):
        super().__init__()
        if in_channels != 12:
            raise ValueError("the GraphRestore trace input is contract-bound to 12 channels")
        if len(channels) != 3 or any(channel <= 0 for channel in channels):
            raise ValueError("trace channels must contain three positive values")
        # Avoid names such as ``half`` which collide with ``nn.Module.half``.
        self.half_stage = ConvNormAct(in_channels, channels[0], stride=2)
        self.quarter_stage = DepthwiseDownsample(channels[0], channels[1])
        self.eighth_stage = DepthwiseDownsample(channels[1], channels[2])

    def forward(self, x0: torch.Tensor, xt: torch.Tensor) -> TraceFeatures:
        if x0.shape != xt.shape or x0.ndim != 4 or x0.shape[1] != 3:
            raise ValueError("x0 and xt must be same-shaped RGB BCHW tensors")
        difference = xt - x0
        trace = torch.cat((x0, xt, difference, difference.abs()), dim=1)
        half = self.half_stage(trace)
        quarter = self.quarter_stage(half)
        eighth = self.eighth_stage(quarter)
        return TraceFeatures(half=half, quarter=quarter, eighth=eighth)
