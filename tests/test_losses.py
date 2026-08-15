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


def test_stage0_restoration_ssim_loss_stays_fp32_under_bf16_autocast() -> None:
    prediction = torch.full(
        (1, 3, 16, 16), 0.5, dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.full(prediction.shape, 0.5, dtype=torch.float32)
    target[..., ::2, ::2] += 0.01
    reference = restoration_loss(
        prediction.detach().float(), target, intermediate=None, lambda_ssim=0.05
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = restoration_loss(
            prediction, target, intermediate=None, lambda_ssim=0.05
        )

    torch.testing.assert_close(result.ssim, reference.ssim, rtol=0.0, atol=0.0)
    assert result.ssim.dtype == torch.float32
    assert result.total.dtype == torch.float32
    assert bool(torch.isfinite(result.total))
    assert float(result.ssim) >= 0.0
    result.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_cycle_loss_is_small_for_consistent_total_order() -> None:
    logits = torch.full((1, 28, 3), -12.0)
    logits[..., 0] = 12.0  # lower skill index before higher skill index.
    assert cycle_consistency_loss(logits).item() < 1e-8
