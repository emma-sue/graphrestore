"""Image fidelity objectives locked by V7.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.metrics.agenticir_official import train_ssim_y


def charbonnier(prediction: Tensor, target: Tensor, eps_squared: float = 1.0e-6) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Charbonnier inputs must have identical shapes")
    return torch.sqrt((prediction - target).square() + eps_squared).mean()


@dataclass(frozen=True)
class RestorationLossBreakdown:
    total: Tensor
    final: Tensor
    step: Tensor
    ssim: Tensor
    noop_pixel: Tensor
    noop_ssim: Tensor


def restoration_loss(
    final: Tensor,
    target: Tensor,
    *,
    intermediate: list[tuple[Tensor, Tensor]] | None = None,
    episode_type: str = "restoration",
    input_image: Tensor | None = None,
    lambda_ssim: float = 0.05,
    step_weight: float = 0.30,
) -> RestorationLossBreakdown:
    zero = final.new_zeros(())
    if episode_type in {"clean_misuse", "wrong_skill"}:
        if input_image is None:
            raise ValueError("counterfactual episodes require input_image")
        noop_pixel = charbonnier(final, input_image)
        noop_ssim = 1.0 - train_ssim_y(final, input_image).mean()
        total = noop_pixel + 0.05 * noop_ssim
        return RestorationLossBreakdown(total, zero, zero, zero, noop_pixel, noop_ssim)

    final_pixel = charbonnier(final, target)
    ssim_term = 1.0 - train_ssim_y(final, target).mean()
    if intermediate:
        step_term = torch.stack([charbonnier(pred, step_target) for pred, step_target in intermediate]).mean()
    else:
        step_term = zero
    total = final_pixel + step_weight * step_term + float(lambda_ssim) * ssim_term
    return RestorationLossBreakdown(total, final_pixel, step_term, ssim_term, zero, zero)
