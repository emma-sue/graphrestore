"""Named latent skill adapters and cooperative correction blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


SKILLS: tuple[str, ...] = (
    "noise",
    "motion_blur",
    "defocus_blur",
    "jpeg_artifact",
    "rain",
    "haze",
    "low_light",
    "low_resolution",
)
SKILL_TO_INDEX = {name: index for index, name in enumerate(SKILLS)}


def skill_indices(skills: Sequence[str | int]) -> tuple[int, ...]:
    """Resolve skill names/indices while rejecting duplicates and drift."""

    resolved: list[int] = []
    for skill in skills:
        if isinstance(skill, str):
            if skill not in SKILL_TO_INDEX:
                raise KeyError(f"unknown skill: {skill}")
            index = SKILL_TO_INDEX[skill]
        else:
            index = int(skill)
            if not 0 <= index < len(SKILLS):
                raise IndexError(f"skill index out of range: {index}")
        if index in resolved:
            raise ValueError(f"duplicate skill: {SKILLS[index]}")
        resolved.append(index)
    return tuple(resolved)


class SkillAdapter(nn.Module):
    """Conv1x1 -> GELU -> DWConv3x3 -> GELU -> zero-init Conv1x1."""

    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        if channels <= 0 or bottleneck <= 0:
            raise ValueError("channels and bottleneck must be positive")
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.depthwise = nn.Conv2d(
            bottleneck,
            bottleneck,
            3,
            padding=1,
            groups=bottleneck,
        )
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.down(x))
        x = F.gelu(self.depthwise(x))
        return self.up(x)


class CooperativeMixer(nn.Module):
    """Zero-initialized multi-skill correction applied after the direct sum."""

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
        )
        self.project = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(F.gelu(self.depthwise(x)))
