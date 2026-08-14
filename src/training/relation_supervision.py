"""Three-class relation supervision with stable ambiguous partial labels."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

SERIAL_CLASS_INDICES = (0, 1)
PARALLEL_CLASS_INDEX = 2
RELATION_CLASS_COUNT = 3
AMBIGUOUS_SERIAL_MASS_WEIGHT = 0.25


def _validate_inputs(
    logits: Tensor,
    targets: Tensor,
    ambiguous_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    if logits.ndim < 2 or logits.shape[-1] != RELATION_CLASS_COUNT:
        raise ValueError(
            f"relation logits must end in exactly 3 classes, got {tuple(logits.shape)}"
        )
    expected_shape = logits.shape[:-1]
    if targets.shape != expected_shape:
        raise ValueError(
            f"target shape {tuple(targets.shape)} != logits batch shape {tuple(expected_shape)}"
        )
    if ambiguous_mask.shape != expected_shape:
        raise ValueError(
            "ambiguous_mask shape "
            f"{tuple(ambiguous_mask.shape)} != logits batch shape {tuple(expected_shape)}"
        )
    if targets.device != logits.device or ambiguous_mask.device != logits.device:
        raise ValueError("logits, targets, and ambiguous_mask must share a device")
    if targets.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"relation targets must be integer typed, got {targets.dtype}")
    if targets.numel() == 0:
        raise ValueError("relation supervision batch must not be empty")

    ambiguous = ambiguous_mask.to(dtype=torch.bool)
    non_ambiguous_targets = targets[~ambiguous]
    if non_ambiguous_targets.numel() > 0:
        minimum = int(non_ambiguous_targets.min().item())
        maximum = int(non_ambiguous_targets.max().item())
        if minimum < 0 or maximum >= RELATION_CLASS_COUNT:
            raise ValueError(
                "non-ambiguous relation targets must be in [0, 2], "
                f"observed [{minimum}, {maximum}]"
            )
    return targets.to(dtype=torch.long), ambiguous


def ambiguous_relation_partial_label_loss(
    logits: Tensor,
    targets: Tensor,
    ambiguous_mask: Tensor,
) -> Tensor:
    """Apply the final-spec loss without double weighting ambiguous rows.

    Non-ambiguous rows use one-hot CE with weight 1. Ambiguous rows supervise
    only the combined mass of the two serial classes. The exact loss is::

        (sum(nonamb_CE) + 0.25*sum(-log(serial_mass)))
        --------------------------------------------------
                    n_nonamb + 0.25*n_amb
    """

    targets, ambiguous = _validate_inputs(logits, targets, ambiguous_mask)
    non_ambiguous = ~ambiguous
    # The adjudication explicitly requires FP32 log_softmax + logsumexp even
    # when the planner runs under BF16 autocast.
    log_probabilities = F.log_softmax(logits.float(), dim=-1)

    non_ambiguous_loss = log_probabilities.new_zeros(())
    if bool(non_ambiguous.any()):
        selected_log_probability = log_probabilities[non_ambiguous].gather(
            dim=-1,
            index=targets[non_ambiguous].unsqueeze(-1),
        )
        non_ambiguous_loss = -selected_log_probability.sum()

    ambiguous_serial_loss = log_probabilities.new_zeros(())
    if bool(ambiguous.any()):
        serial_log_mass = torch.logsumexp(
            log_probabilities[ambiguous][..., list(SERIAL_CLASS_INDICES)],
            dim=-1,
        )
        ambiguous_serial_loss = -serial_log_mass.sum()

    n_non_ambiguous = int(non_ambiguous.sum().item())
    n_ambiguous = int(ambiguous.sum().item())
    denominator = n_non_ambiguous + AMBIGUOUS_SERIAL_MASS_WEIGHT * n_ambiguous
    numerator = (
        non_ambiguous_loss
        + AMBIGUOUS_SERIAL_MASS_WEIGHT * ambiguous_serial_loss
    )
    return numerator / denominator


@torch.no_grad()
def non_ambiguous_relation_metrics(
    logits: Tensor,
    targets: Tensor,
    ambiguous_mask: Tensor,
    *,
    pair_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Exclude ambiguous rows from relation and parallel metrics.

    The explicit zero-valued audit fields are also consumed by Stage2 prior and
    majority aggregation tests so ambiguous samples cannot leak into them.
    """

    targets, ambiguous = _validate_inputs(logits, targets, ambiguous_mask)
    non_ambiguous = ~ambiguous
    predictions = logits.argmax(dim=-1)
    included_predictions = predictions[non_ambiguous]
    included_targets = targets[non_ambiguous]

    n_total = targets.numel()
    n_ambiguous = int(ambiguous.sum().item())
    n_non_ambiguous = int(non_ambiguous.sum().item())
    correct = int((included_predictions == included_targets).sum().item())
    true_parallel = included_targets == PARALLEL_CLASS_INDEX
    predicted_parallel = included_predictions == PARALLEL_CLASS_INDEX
    parallel_true_positive = int((true_parallel & predicted_parallel).sum().item())
    predicted_parallel_count = int(predicted_parallel.sum().item())
    true_parallel_count = int(true_parallel.sum().item())

    if pair_ids is not None and len(pair_ids) != n_total:
        raise ValueError(
            f"pair_ids length {len(pair_ids)} != flattened target count {n_total}"
        )
    flattened_targets = targets.reshape(-1)
    flattened_ambiguous = ambiguous.reshape(-1)
    flattened_pair_ids = (
        tuple(pair_ids)
        if pair_ids is not None
        else tuple("__all__" for _ in range(n_total))
    )
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in flattened_pair_ids):
        raise ValueError("pair_ids must contain non-empty strings")
    pair_counts: dict[str, Counter[int]] = {}
    for index, pair_id in enumerate(flattened_pair_ids):
        if bool(flattened_ambiguous[index]):
            continue
        pair_counts.setdefault(pair_id, Counter())[int(flattened_targets[index])] += 1
    pair_prior_non_ambiguous = {
        pair_id: {
            "i_before_j": counts.get(0, 0),
            "j_before_i": counts.get(1, 0),
            "parallel": counts.get(2, 0),
            "n_non_ambiguous": sum(counts.values()),
        }
        for pair_id, counts in sorted(pair_counts.items())
    }
    majority_label_share_non_ambiguous = {
        pair_id: max(counts.values()) / sum(counts.values())
        for pair_id, counts in sorted(pair_counts.items())
    }

    return {
        "n_total": int(n_total),
        "n_non_ambiguous": n_non_ambiguous,
        "n_ambiguous": n_ambiguous,
        "ambiguous_fraction": n_ambiguous / n_total,
        "relation_correct_non_ambiguous": correct,
        "relation_accuracy_non_ambiguous": (
            correct / n_non_ambiguous if n_non_ambiguous else math.nan
        ),
        "parallel_true_positive_non_ambiguous": parallel_true_positive,
        "parallel_predicted_non_ambiguous": predicted_parallel_count,
        "parallel_target_non_ambiguous": true_parallel_count,
        "parallel_precision_non_ambiguous": (
            parallel_true_positive / predicted_parallel_count
            if predicted_parallel_count
            else math.nan
        ),
        "parallel_recall_non_ambiguous": (
            parallel_true_positive / true_parallel_count
            if true_parallel_count
            else math.nan
        ),
        "parallel_fraction_nonambiguous": (
            true_parallel_count / n_non_ambiguous
            if n_non_ambiguous
            else math.nan
        ),
        "pair_prior_non_ambiguous": pair_prior_non_ambiguous,
        "majority_label_share_non_ambiguous": (
            majority_label_share_non_ambiguous
        ),
        "ambiguous_in_relation_metrics": 0,
        "ambiguous_in_pair_prior": 0,
        "ambiguous_in_majority_label_share": 0,
        "ambiguous_in_parallel_fraction_nonambiguous": 0,
    }
