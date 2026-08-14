"""Deterministic restoration-first checkpoint ordering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationScore:
    group_a_psnr: float
    group_a_ssim: float
    single_psnr: float
    single_ssim: float = float("-inf")
    step: int = 0


def is_better_checkpoint(candidate: ValidationScore, incumbent: ValidationScore | None) -> bool:
    if incumbent is None:
        return True
    difference = candidate.group_a_psnr - incumbent.group_a_psnr
    # Decimal validation summaries such as 30.03-30.01 can land one ULP below
    # 0.02; keep the contract's strict "gap < 0.02" boundary deterministic.
    if abs(difference) + 1.0e-12 >= 0.02:
        return difference > 0
    if candidate.group_a_ssim != incumbent.group_a_ssim:
        return candidate.group_a_ssim > incumbent.group_a_ssim
    if candidate.single_psnr != incumbent.single_psnr:
        return candidate.single_psnr > incumbent.single_psnr
    if candidate.single_ssim != incumbent.single_ssim:
        return candidate.single_ssim > incumbent.single_ssim
    return candidate.step < incumbent.step
