"""Layer-wise AdamW groups and warmup-cosine schedule."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .checkpointing import unwrap_model


def _is_norm_or_bias(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith(".bias") or ".norm" in name.lower()


def parameter_groups(
    model: nn.Module,
    prefix_to_lr: Sequence[tuple[tuple[str, ...], float]],
    *,
    weight_decay: float,
    weight_decay_norm_bias: float = 0.0,
) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    assigned: set[int] = set()
    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [lr for prefixes, lr in prefix_to_lr if name.startswith(prefixes)]
        if len(matches) != 1:
            raise ValueError(f"parameter {name!r} matched {len(matches)} LR groups")
        decay = weight_decay_norm_bias if _is_norm_or_bias(name, parameter) else weight_decay
        grouped.setdefault((float(matches[0]), float(decay)), []).append(parameter)
        if id(parameter) in assigned:
            raise RuntimeError(f"parameter {name!r} assigned more than once")
        assigned.add(id(parameter))
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if assigned != expected:
        raise RuntimeError("optimizer grouping did not cover every trainable parameter exactly once")
    return [
        {"params": parameters, "lr": lr, "initial_lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in grouped.items()
    ]


def build_adamw(
    groups: list[dict[str, object]],
    *,
    betas: tuple[float, float] = (0.9, 0.999),
    fused_if_supported: bool = True,
) -> torch.optim.AdamW:
    kwargs: dict[str, object] = {"betas": betas}
    if fused_if_supported and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
        last_epoch: int = -1,
    ):
        if not 0 <= warmup_steps < max_steps:
            raise ValueError("require 0 <= warmup_steps < max_steps")
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        step = max(0, self.last_epoch)
        values = []
        for base_lr in self.base_lrs:
            floor = min(self.min_lr, base_lr)
            if self.warmup_steps and step < self.warmup_steps:
                scale = float(step + 1) / float(self.warmup_steps)
                values.append(base_lr * scale)
                continue
            progress = min(
                1.0,
                max(0.0, (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)),
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            values.append(floor + (base_lr - floor) * cosine)
        return values


def set_stage0_trainability(model: nn.Module, step: int) -> None:
    """Apply the V7.1 Stage0 freeze boundary without rebuilding the optimizer."""

    for name, parameter in unwrap_model(model).named_parameters():
        frozen_early = name.startswith(
            (
                "encoder.patch.",
                "encoder.level1.",
                "encoder.down12.",
                "encoder.level2.",
            )
        ) and step < 2000
        parameter.requires_grad_(not frozen_early)
