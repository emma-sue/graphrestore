"""Planner supervision with three-class ambiguous partial labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from src.training.relation_supervision import (
    AMBIGUOUS_SERIAL_MASS_WEIGHT,
    ambiguous_relation_partial_label_loss,
)

from .cycle_consistency import cycle_consistency_loss
from .guard_losses import GuardLossBreakdown


def focal_binary_cross_entropy(logits: Tensor, targets: Tensor, gamma: float = 2.0) -> Tensor:
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    return ((1.0 - p_t).pow(gamma) * bce).mean()


def _relation_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor | None,
    ambiguous_mask: Tensor | None = None,
) -> Tensor:
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("relation logits must be BxPx3")
    if labels.shape != logits.shape[:2]:
        raise ValueError("relation labels must be BxP")
    if labels.device != logits.device:
        raise ValueError("relation labels and logits must share a device")

    valid = labels.ge(0) & labels.lt(3)
    if ambiguous_mask is None:
        ambiguous = labels.eq(-1)
    else:
        if ambiguous_mask.shape != labels.shape:
            raise ValueError("relation ambiguous_mask must be BxP")
        if ambiguous_mask.device != logits.device:
            raise ValueError("relation ambiguous_mask and logits must share a device")
        ambiguous = ambiguous_mask.to(dtype=torch.bool)
        if bool((ambiguous & ~labels.eq(-1)).any()):
            raise ValueError("ambiguous relation entries must use target label -1")

    supervised = valid | ambiguous
    if weights is not None:
        if weights.shape != labels.shape:
            raise ValueError("relation weights must be BxP")
        if weights.device != logits.device:
            raise ValueError("relation weights and logits must share a device")
        selected_weights = weights.to(dtype=logits.dtype)
        if not bool(torch.isfinite(selected_weights).all()) or bool(
            selected_weights.lt(0).any()
        ):
            raise ValueError("relation weights must be finite and non-negative")

        # Zero is a supervision mask for non-relation episodes. Positive values
        # must equal the final-spec fixed weights, but are not multiplied again.
        positive = selected_weights.gt(0)
        non_ambiguous_positive = valid & positive
        ambiguous_positive = ambiguous & positive
        expected_non_ambiguous = torch.ones_like(selected_weights)
        expected_ambiguous = torch.full_like(
            selected_weights, AMBIGUOUS_SERIAL_MASS_WEIGHT
        )
        if bool(
            (
                non_ambiguous_positive
                & ~torch.isclose(
                    selected_weights,
                    expected_non_ambiguous,
                    rtol=0.0,
                    atol=1e-6,
                )
            ).any()
        ):
            raise ValueError("non-ambiguous relation weights must be 1.0 or 0")
        if bool(
            (
                ambiguous_positive
                & ~torch.isclose(
                    selected_weights,
                    expected_ambiguous,
                    rtol=0.0,
                    atol=1e-6,
                )
            ).any()
        ):
            raise ValueError("ambiguous relation weights must be 0.25 or 0")
        if bool((positive & ~supervised).any()):
            raise ValueError("positive relation weight requires a supervised target")
        supervised &= positive
        ambiguous &= positive

    if not bool(supervised.any()):
        return logits.sum() * 0.0
    return ambiguous_relation_partial_label_loss(
        logits[supervised],
        labels[supervised],
        ambiguous[supervised],
    )


@dataclass(frozen=True)
class PlannerLossBreakdown:
    total: Tensor
    presence: Tensor
    guard: Tensor
    relation: Tensor
    stop: Tensor
    cycle: Tensor


def planner_loss(
    *,
    presence_logits: Tensor,
    presence_targets: Tensor,
    relation_logits: Tensor,
    relation_targets: Tensor,
    stop_logits: Tensor,
    stop_targets: Tensor,
    guard: GuardLossBreakdown,
    relation_weights: Tensor | None = None,
    relation_ambiguous_mask: Tensor | None = None,
) -> PlannerLossBreakdown:
    presence = focal_binary_cross_entropy(presence_logits, presence_targets, gamma=2.0)
    relation = _relation_cross_entropy(
        relation_logits,
        relation_targets,
        relation_weights,
        relation_ambiguous_mask,
    )
    stop = F.binary_cross_entropy_with_logits(stop_logits, stop_targets.to(stop_logits))
    cycle = cycle_consistency_loss(relation_logits)
    total = presence + 0.5 * guard.total + relation + 0.25 * stop + 0.01 * cycle
    return PlannerLossBreakdown(total, presence, guard.total, relation, stop, cycle)
