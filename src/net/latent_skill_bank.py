"""Spatially guarded latent skill bank inserted after decoder blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .mio_stagea import RestorationDecoder
from .restormer_blocks import BlockStack
from .skill_adapter import CooperativeMixer, SKILLS, SkillAdapter


DEFAULT_BOTTLENECKS = {
    "level3": 24,
    "level2": 16,
    "level1": 12,
    "refinement": 12,
}


class LatentSkillBank(nn.Module):
    """Eight independent adapters per decoder block plus cooperative mixers."""

    def __init__(
        self,
        *,
        level_channels: Mapping[str, int],
        level_blocks: Mapping[str, int],
        bottlenecks: Mapping[str, int] | None = None,
        skills: Sequence[str] = SKILLS,
    ):
        super().__init__()
        self.skills = tuple(skills)
        if self.skills != SKILLS:
            raise ValueError("GraphRestore skill identity/order is contract-bound")
        bottleneck_map = dict(DEFAULT_BOTTLENECKS)
        if bottlenecks is not None:
            bottleneck_map.update({key: int(value) for key, value in bottlenecks.items()})

        expected_levels = ("level3", "level2", "level1", "refinement")
        if set(level_channels) != set(expected_levels):
            raise ValueError("level_channels must define all four decoder levels")
        if set(level_blocks) != set(expected_levels):
            raise ValueError("level_blocks must define all four decoder levels")

        self.adapters = nn.ModuleDict()
        self.mixers = nn.ModuleDict()
        for level in expected_levels:
            channels = int(level_channels[level])
            count = int(level_blocks[level])
            bottleneck = int(bottleneck_map[level])
            if count < 0:
                raise ValueError(f"negative block count for {level}")
            self.adapters[level] = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            skill: SkillAdapter(channels, bottleneck)
                            for skill in self.skills
                        }
                    )
                    for _ in range(count)
                ]
            )
            self.mixers[level] = nn.ModuleList(
                [CooperativeMixer(channels) for _ in range(count)]
            )

    @property
    def num_skills(self) -> int:
        return len(self.skills)

    def _validate_control(
        self,
        hidden: torch.Tensor,
        guards: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if guards.ndim != 4 or guards.shape[1] != self.num_skills:
            raise ValueError(
                f"guards must be [B,{self.num_skills},H,W], got {tuple(guards.shape)}"
            )
        if active_mask.ndim != 2 or active_mask.shape[1] != self.num_skills:
            raise ValueError(
                f"active_mask must be [B,{self.num_skills}], got "
                f"{tuple(active_mask.shape)}"
            )
        if guards.shape[0] != hidden.shape[0] or active_mask.shape[0] != hidden.shape[0]:
            raise ValueError("hidden/guards/active_mask batch sizes differ")
        resized = F.interpolate(
            guards.to(device=hidden.device, dtype=hidden.dtype),
            size=hidden.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mask = active_mask.to(device=hidden.device, dtype=hidden.dtype)
        return resized, mask

    def apply_after_block(
        self,
        level: str,
        block_index: int,
        hidden: torch.Tensor,
        *,
        guards: torch.Tensor,
        active_mask: torch.Tensor,
        active_skill_indices: Sequence[int] | None = None,
        has_multi_skill_sample: bool | None = None,
    ) -> torch.Tensor:
        """Apply direct guarded skills, then a zero-init cooperative correction."""

        if level not in self.adapters:
            raise KeyError(f"unknown decoder level: {level}")
        if not 0 <= block_index < len(self.adapters[level]):
            raise IndexError(f"block index out of range for {level}: {block_index}")
        resized, mask = self._validate_control(hidden, guards, active_mask)

        skill_sum = torch.zeros_like(hidden)
        block_adapters = self.adapters[level][block_index]
        if active_skill_indices is None:
            active_skill_indices = tuple(range(self.num_skills))
        for skill_index in active_skill_indices:
            skill = self.skills[skill_index]
            sample_weight = mask[:, skill_index]
            weight = resized[:, skill_index : skill_index + 1]
            weight = weight * sample_weight[:, None, None, None]
            skill_sum = skill_sum + weight * block_adapters[skill](hidden)

        active_count = (mask > 0).sum(dim=1).to(dtype=hidden.dtype)
        normalization = active_count.clamp_min(1).sqrt()[:, None, None, None]
        skill_sum = skill_sum / normalization
        updated = hidden + skill_sum

        multi_mask = (active_count > 1).to(dtype=hidden.dtype)[:, None, None, None]
        if has_multi_skill_sample is None:
            has_multi_skill_sample = bool(torch.any(multi_mask > 0).item())
        if has_multi_skill_sample:
            correction = self.mixers[level][block_index](skill_sum)
            updated = updated + multi_mask * correction
        return updated


class GuardedRestorationDecoder(RestorationDecoder):
    """Restoration decoder with adapters after every original decoder block.

    All inherited backbone parameters retain the exact ``decoder.*`` key layout;
    additions are confined to ``decoder.skill_bank.*``.
    """

    def __init__(
        self,
        dim: int = 48,
        blocks: Sequence[int] = (6, 6, 4),
        refinement: int = 4,
        heads: Sequence[int] = (1, 2, 4),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        gradient_checkpointing: bool = False,
        skill_bottlenecks: Mapping[str, int] | None = None,
    ):
        super().__init__(
            dim,
            blocks,
            refinement,
            heads,
            expansion,
            bias,
            norm_type,
            gradient_checkpointing,
        )
        self.skill_bank = LatentSkillBank(
            level_channels={
                "level3": dim * 4,
                "level2": dim * 2,
                "level1": dim,
                "refinement": dim,
            },
            level_blocks={
                "level3": int(blocks[0]),
                "level2": int(blocks[1]),
                "level1": int(blocks[2]),
                "refinement": int(refinement),
            },
            bottlenecks=skill_bottlenecks,
        )

    def _run_guarded_stack(
        self,
        stack: BlockStack,
        hidden: torch.Tensor,
        level: str,
        guards: torch.Tensor,
        active_mask: torch.Tensor,
        active_skill_indices: Sequence[int],
        has_multi_skill_sample: bool,
    ) -> torch.Tensor:
        for block_index in range(len(stack)):
            hidden = stack.run_block(block_index, hidden)
            hidden = self.skill_bank.apply_after_block(
                level,
                block_index,
                hidden,
                guards=guards,
                active_mask=active_mask,
                active_skill_indices=active_skill_indices,
                has_multi_skill_sample=has_multi_skill_sample,
            )
        return hidden

    def forward(
        self,
        features: Sequence[torch.Tensor],
        *,
        guards: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        first, second, third, fourth = features
        active_skill_indices = tuple(
            torch.nonzero(
                active_mask.to(dtype=torch.bool).any(dim=0),
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        has_multi_skill_sample = bool(
            torch.any(active_mask.to(dtype=torch.bool).sum(dim=1) > 1).item()
        )
        third_out = self.fuse3(torch.cat((self.up43(fourth), third), dim=1))
        third_out = self._run_guarded_stack(
            self.level3,
            third_out,
            "level3",
            guards,
            active_mask,
            active_skill_indices,
            has_multi_skill_sample,
        )
        second_out = self.fuse2(torch.cat((self.up32(third_out), second), dim=1))
        second_out = self._run_guarded_stack(
            self.level2,
            second_out,
            "level2",
            guards,
            active_mask,
            active_skill_indices,
            has_multi_skill_sample,
        )
        first_out = self.fuse1(torch.cat((self.up21(second_out), first), dim=1))
        first_out = self._run_guarded_stack(
            self.level1,
            first_out,
            "level1",
            guards,
            active_mask,
            active_skill_indices,
            has_multi_skill_sample,
        )
        first_out = self._run_guarded_stack(
            self.refinement,
            first_out,
            "refinement",
            guards,
            active_mask,
            active_skill_indices,
            has_multi_skill_sample,
        )
        return self.head(first_out)
