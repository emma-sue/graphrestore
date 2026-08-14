from __future__ import annotations

import torch

from src.losses.cycle_consistency import cycle_consistency_loss
from src.losses.restoration import restoration_loss


def test_restoration_loss_has_zero_step_term_without_intermediate() -> None:
    prediction = torch.rand(1, 3, 16, 16, requires_grad=True)
    target = torch.rand_like(prediction)
    result = restoration_loss(prediction, target, intermediate=None, lambda_ssim=0.05)
    assert result.step.item() == 0.0
    result.total.backward()
    assert prediction.grad is not None


def test_cycle_loss_is_small_for_consistent_total_order() -> None:
    logits = torch.full((1, 28, 3), -12.0)
    logits[..., 0] = 12.0  # lower skill index before higher skill index.
    assert cycle_consistency_loss(logits).item() < 1e-8
