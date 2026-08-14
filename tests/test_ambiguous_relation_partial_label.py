from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.losses.guard_losses import GuardLossBreakdown
from src.losses.planner_losses import planner_loss
from src.training.relation_supervision import (
    AMBIGUOUS_SERIAL_MASS_WEIGHT,
    RELATION_CLASS_COUNT,
    ambiguous_relation_partial_label_loss,
    non_ambiguous_relation_metrics,
)


def test_ambiguous_relation_partial_label() -> None:
    logits = torch.tensor(
        [
            [2.0, -0.5, 0.25],
            [-0.2, 1.1, 0.4],
            [0.3, -0.1, 1.5],
            [0.8, 0.2, 1.9],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([0, -1, 2, -1], dtype=torch.long)
    ambiguous = torch.tensor([False, True, False, True])

    assert logits.shape[-1] == RELATION_CLASS_COUNT == 3
    loss = ambiguous_relation_partial_label_loss(logits, targets, ambiguous)

    log_p = F.log_softmax(logits.float(), dim=-1)
    expected_non_ambiguous = -log_p[[0, 2], [0, 2]].sum()
    expected_ambiguous = -torch.logsumexp(log_p[[1, 3], :2], dim=-1).sum()
    expected = (
        expected_non_ambiguous
        + AMBIGUOUS_SERIAL_MASS_WEIGHT * expected_ambiguous
    ) / (2 + AMBIGUOUS_SERIAL_MASS_WEIGHT * 2)
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()

    pair_ids = ["blur_rain", "blur_rain", "noise_haze", "noise_haze"]
    metrics = non_ambiguous_relation_metrics(
        logits.detach(), targets, ambiguous, pair_ids=pair_ids
    )
    assert metrics["n_total"] == 4
    assert metrics["n_non_ambiguous"] == 2
    assert metrics["n_ambiguous"] == 2
    assert metrics["ambiguous_fraction"] == pytest.approx(0.5)
    assert metrics["relation_accuracy_non_ambiguous"] == pytest.approx(1.0)
    assert metrics["ambiguous_in_relation_metrics"] == 0
    assert metrics["ambiguous_in_pair_prior"] == 0
    assert metrics["ambiguous_in_majority_label_share"] == 0
    assert metrics["ambiguous_in_parallel_fraction_nonambiguous"] == 0
    assert metrics["parallel_fraction_nonambiguous"] == pytest.approx(0.5)
    assert metrics["pair_prior_non_ambiguous"] == {
        "blur_rain": {
            "i_before_j": 1,
            "j_before_i": 0,
            "parallel": 0,
            "n_non_ambiguous": 1,
        },
        "noise_haze": {
            "i_before_j": 0,
            "j_before_i": 0,
            "parallel": 1,
            "n_non_ambiguous": 1,
        },
    }
    assert metrics["majority_label_share_non_ambiguous"] == {
        "blur_rain": 1.0,
        "noise_haze": 1.0,
    }

    changed_ambiguous_logits = logits.detach().clone()
    changed_ambiguous_logits[ambiguous] = torch.tensor(
        [[-100.0, -100.0, 100.0], [100.0, -100.0, -100.0]],
        dtype=changed_ambiguous_logits.dtype,
    )
    changed_metrics = non_ambiguous_relation_metrics(
        changed_ambiguous_logits, targets, ambiguous, pair_ids=pair_ids
    )
    assert changed_metrics == metrics

    ambiguous_target = torch.tensor([-1], dtype=torch.long)
    ambiguous_mask = torch.tensor([True])
    base = torch.tensor([[0.2, -0.7, 1.0]], dtype=torch.float64)
    more_serial_mass = torch.tensor([[1.2, -0.7, 1.0]], dtype=torch.float64)
    assert ambiguous_relation_partial_label_loss(
        more_serial_mass, ambiguous_target, ambiguous_mask
    ) < ambiguous_relation_partial_label_loss(base, ambiguous_target, ambiguous_mask)

    swapped = base[:, [1, 0, 2]]
    torch.testing.assert_close(
        ambiguous_relation_partial_label_loss(base, ambiguous_target, ambiguous_mask),
        ambiguous_relation_partial_label_loss(
            swapped, ambiguous_target, ambiguous_mask
        ),
    )

    exchanged_targets = torch.where(
        targets == 0,
        torch.ones_like(targets),
        torch.where(targets == 1, torch.zeros_like(targets), targets),
    )
    torch.testing.assert_close(
        ambiguous_relation_partial_label_loss(logits.detach(), targets, ambiguous),
        ambiguous_relation_partial_label_loss(
            logits.detach()[:, [1, 0, 2]], exchanged_targets, ambiguous
        ),
    )


def test_partial_label_rejects_non_three_class_head() -> None:
    with pytest.raises(ValueError, match="exactly 3 classes"):
        ambiguous_relation_partial_label_loss(
            torch.zeros(2, 4),
            torch.tensor([0, 1]),
            torch.tensor([False, True]),
        )


def test_ambiguous_partial_label_forces_fp32_math_from_bfloat16() -> None:
    logits = torch.tensor(
        [[0.75, -0.5, 1.25], [-0.25, 1.5, 0.1]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    targets = torch.tensor([-1, 1], dtype=torch.long)
    ambiguous = torch.tensor([True, False])
    loss = ambiguous_relation_partial_label_loss(logits, targets, ambiguous)
    assert loss.dtype == torch.float32
    stable_log_p = F.log_softmax(logits.float(), dim=-1)
    expected = (
        -AMBIGUOUS_SERIAL_MASS_WEIGHT
        * torch.logsumexp(stable_log_p[0, :2], dim=-1)
        - stable_log_p[1, 1]
    ) / (1.0 + AMBIGUOUS_SERIAL_MASS_WEIGHT)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_planner_loss_relation_uses_partial_label_formula_exactly() -> None:
    relation_logits = torch.zeros(
        1, 28, 3, dtype=torch.float64, requires_grad=True
    )
    selected_logits = torch.tensor(
        [
            [1.4, -0.2, 0.1],
            [-0.3, 0.9, 1.1],
            [0.2, -0.5, 1.3],
            [1.2, -0.8, 0.7],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        relation_logits[0, :4].copy_(selected_logits)
    relation_targets = torch.full((1, 28), -2, dtype=torch.long)
    relation_targets[0, :4] = torch.tensor([0, -1, 2, -1])
    ambiguous_mask = torch.zeros((1, 28), dtype=torch.bool)
    ambiguous_mask[0, [1, 3]] = True
    relation_weights = torch.zeros((1, 28), dtype=torch.float64)
    relation_weights[0, :4] = torch.tensor([1.0, 0.25, 1.0, 0.25])

    scalar_zero = relation_logits.new_zeros(())
    guard = GuardLossBreakdown(
        total=scalar_zero,
        dense=scalar_zero,
        global_mean=scalar_zero,
        absent=scalar_zero,
    )
    breakdown = planner_loss(
        presence_logits=torch.zeros(1, 8, dtype=torch.float64),
        presence_targets=torch.zeros(1, 8, dtype=torch.float64),
        relation_logits=relation_logits,
        relation_targets=relation_targets,
        relation_weights=relation_weights,
        relation_ambiguous_mask=ambiguous_mask,
        stop_logits=torch.zeros(1, 1, dtype=torch.float64),
        stop_targets=torch.zeros(1, 1, dtype=torch.float64),
        guard=guard,
    )

    log_p = F.log_softmax(selected_logits.float(), dim=-1)
    expected = (
        -log_p[0, 0]
        - log_p[2, 2]
        + 0.25
        * (
            -torch.logsumexp(log_p[1, :2], dim=-1)
            - torch.logsumexp(log_p[3, :2], dim=-1)
        )
    ) / (2 + 0.25 * 2)
    torch.testing.assert_close(breakdown.relation, expected)

    breakdown.relation.backward()
    assert relation_logits.grad is not None
    assert torch.isfinite(relation_logits.grad).all()

    inferred = planner_loss(
        presence_logits=torch.zeros(1, 8, dtype=torch.float64),
        presence_targets=torch.zeros(1, 8, dtype=torch.float64),
        relation_logits=relation_logits.detach(),
        relation_targets=relation_targets,
        relation_weights=relation_weights,
        stop_logits=torch.zeros(1, 1, dtype=torch.float64),
        stop_targets=torch.zeros(1, 1, dtype=torch.float64),
        guard=guard,
    )
    torch.testing.assert_close(inferred.relation, expected)
