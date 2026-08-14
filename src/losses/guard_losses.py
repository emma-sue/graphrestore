"""Continuous spatial/global/absent guard supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class GuardLossBreakdown:
    total: Tensor
    dense: Tensor
    global_mean: Tensor
    absent: Tensor


def _masked_smooth_l1(values: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.to(dtype=torch.bool)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    if not expanded.any():
        return values.new_zeros(())
    return F.smooth_l1_loss(values[expanded], targets[expanded])


def guard_supervision_loss(
    guard_logits: Tensor,
    dense_targets: Tensor,
    global_severity_targets: Tensor,
    *,
    dense_skill_mask: Tensor,
    global_skill_mask: Tensor,
    absent_skill_mask: Tensor,
) -> GuardLossBreakdown:
    if guard_logits.ndim != 4:
        raise ValueError("guard_logits must be BxKxHxW")
    predicted = guard_logits.sigmoid()
    if dense_targets.shape != predicted.shape:
        raise ValueError("dense guard targets must match logits")
    if global_severity_targets.shape != predicted.shape[:2]:
        raise ValueError("global severity targets must be BxK")
    dense = _masked_smooth_l1(predicted, dense_targets, dense_skill_mask)
    global_mean = _masked_smooth_l1(
        predicted.mean(dim=(-2, -1)), global_severity_targets, global_skill_mask
    )
    absent = _masked_smooth_l1(predicted, torch.zeros_like(predicted), absent_skill_mask)
    total = dense + 0.5 * global_mean + 0.5 * absent
    return GuardLossBreakdown(total, dense, global_mean, absent)
