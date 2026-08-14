"""Guarded cooperative execution of one compiled program level."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from .latent_skill_bank import GuardedRestorationDecoder
from .skill_adapter import SKILLS


def _validate_controls(
    current: torch.Tensor,
    guards: torch.Tensor,
    active_mask: torch.Tensor,
) -> None:
    if current.ndim != 4 or current.shape[1] != 3:
        raise ValueError("current image must be RGB BCHW")
    if guards.ndim != 4 or guards.shape[:2] != (current.shape[0], len(SKILLS)):
        raise ValueError("guards must be [B,8,H,W]")
    if active_mask.shape != (current.shape[0], len(SKILLS)):
        raise ValueError("active_mask must be [B,8]")


def soft_union_guard(
    guards: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Compute ``1-prod(1-g_k)`` over active skills only."""

    if guards.ndim != 4 or active_mask.ndim != 2:
        raise ValueError("guards/active_mask ranks must be 4/2")
    resized = torch.nn.functional.interpolate(
        guards,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    selected = resized * active_mask.to(
        device=resized.device,
        dtype=resized.dtype,
    )[:, :, None, None]
    selected = selected.clamp(0.0, 1.0)
    return 1.0 - torch.prod(1.0 - selected, dim=1, keepdim=True)


@dataclass(frozen=True)
class ExecutionResult:
    next_image: torch.Tensor
    delta: torch.Tensor
    union_guard: torch.Tensor
    execution_guards: torch.Tensor
    active_mask: torch.Tensor
    forced_presence_mask: torch.Tensor
    residual_norm: torch.Tensor
    identity_mask: torch.Tensor


class CooperativeExecutor(nn.Module):
    """Execute one active DAG level through one guarded decoder pass.

    This module owns no backbone parameters.  In forced counterfactual episodes,
    ``forced_presence_mask`` makes the named skill active; the caller must pass
    guards computed with the execution presence override while retaining the
    original absent-skill planner target.
    """

    def forward(
        self,
        current: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
        decoder: GuardedRestorationDecoder,
        *,
        guards: torch.Tensor,
        active_mask: torch.Tensor,
        forced_presence_mask: torch.Tensor | None = None,
    ) -> ExecutionResult:
        _validate_controls(current, guards, active_mask)
        active = active_mask.to(device=current.device, dtype=torch.bool)
        if forced_presence_mask is None:
            forced = torch.zeros_like(active)
        else:
            if forced_presence_mask.shape != active.shape:
                raise ValueError("forced_presence_mask must be [B,8]")
            forced = forced_presence_mask.to(device=current.device, dtype=torch.bool)
            active = active | forced

        delta = decoder(
            encoder_features,
            guards=guards,
            active_mask=active,
        )
        if delta.shape != current.shape:
            raise RuntimeError(
                f"decoder delta shape {tuple(delta.shape)} != image {tuple(current.shape)}"
            )
        union = soft_union_guard(
            guards.to(device=current.device, dtype=current.dtype),
            active,
            output_size=current.shape[-2:],
        )
        next_image = current + union * delta
        residual = union * delta
        residual_norm = residual.square().mean(dim=(1, 2, 3)).sqrt()
        identity_mask = union.flatten(1).amax(dim=1) == 0
        return ExecutionResult(
            next_image=next_image,
            delta=delta,
            union_guard=union,
            execution_guards=guards,
            active_mask=active,
            forced_presence_mask=forced,
            residual_norm=residual_norm,
            identity_mask=identity_mask,
        )
