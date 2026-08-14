"""Small, auditable runtime helpers for single-GPU GraphRestore training."""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, ContextManager, Iterable

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class MicroBatchTrial:
    """One pre-step0 memory/throughput observation."""

    micro_batch: int
    crop_size: int
    gradient_checkpointing: bool
    consecutive_optimizer_steps: int
    consecutive_forward_backward: int
    images_per_second: float
    peak_reserved_bytes: int
    total_memory_bytes: int
    peak_reserved_fraction: float
    finite: bool
    oom: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MicroBatchSelection:
    """Frozen Stage0 runtime shape selected before formal optimizer step zero."""

    micro_batch: int
    effective_batch: int
    accumulation_steps: int
    crop_size: int
    gradient_checkpointing: bool
    trial: MicroBatchTrial
    trials: tuple[MicroBatchTrial, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["trial"] = self.trial.to_dict()
        value["trials"] = [trial.to_dict() for trial in self.trials]
        return value


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the Stage0 process."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(*, tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    """Apply the locked V7.1 CUDA math settings without changing layout."""

    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def autocast_context(device: torch.device) -> ContextManager[object]:
    """Use BF16 only on CUDA; CPU tests remain ordinary FP32."""

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def move_training_batch(
    batch: dict[str, object], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Move only Stage0 tensors, leaving metadata and unused subset targets on CPU."""

    input_image = batch.get("input")
    target = batch.get("target")
    if not isinstance(input_image, Tensor) or not isinstance(target, Tensor):
        raise TypeError("Stage0 batch must contain tensor input and target")
    non_blocking = device.type == "cuda"
    return (
        input_image.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        target.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
    )


def select_micro_batch(
    trials: Iterable[MicroBatchTrial],
    *,
    effective_batch: int = 8,
    maximum_peak_fraction: float = 0.90,
) -> MicroBatchSelection:
    """Choose the fastest valid candidate; fail closed if none leaves 10% headroom."""

    observed = tuple(trials)
    if not observed:
        raise ValueError("at least one micro-batch trial is required")
    if effective_batch <= 0:
        raise ValueError("effective_batch must be positive")
    valid = [
        trial
        for trial in observed
        if not trial.oom
        and trial.finite
        and trial.consecutive_forward_backward >= 10
        and trial.peak_reserved_fraction <= maximum_peak_fraction
        and trial.micro_batch > 0
        and effective_batch % trial.micro_batch == 0
    ]
    if not valid:
        details = "; ".join(
            f"micro={trial.micro_batch}, oom={trial.oom}, finite={trial.finite}, "
            f"peak={trial.peak_reserved_fraction:.4f}, error={trial.error}"
            for trial in observed
        )
        raise RuntimeError(
            "no Stage0 candidate satisfied finite 10-pass execution and the "
            f"{maximum_peak_fraction:.0%} VRAM ceiling: {details}"
        )
    # Throughput is the primary rule.  Larger micro-batch is a deterministic tie-break.
    chosen = max(valid, key=lambda item: (item.images_per_second, item.micro_batch))
    return MicroBatchSelection(
        micro_batch=chosen.micro_batch,
        effective_batch=effective_batch,
        accumulation_steps=effective_batch // chosen.micro_batch,
        crop_size=chosen.crop_size,
        gradient_checkpointing=chosen.gradient_checkpointing,
        trial=chosen,
        trials=observed,
    )


def run_micro_batch_trials(
    candidates: Iterable[int],
    trial: Callable[[int], MicroBatchTrial],
    *,
    effective_batch: int = 8,
    maximum_peak_fraction: float = 0.90,
) -> MicroBatchSelection:
    """Evaluate every locked candidate before selecting by measured throughput."""

    ordered = tuple(int(value) for value in candidates)
    if ordered != (8, 4, 2, 1):
        raise ValueError("Stage0 micro-batch candidates must be exactly (8,4,2,1)")
    return select_micro_batch(
        (trial(value) for value in ordered),
        effective_batch=effective_batch,
        maximum_peak_fraction=maximum_peak_fraction,
    )
