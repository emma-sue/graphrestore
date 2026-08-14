"""Reproducible training infrastructure shared by Stage0--4."""

from .checkpointing import (
    CheckpointProvenanceError,
    atomic_torch_save,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    unwrap_model,
)
from .ema import ExponentialMovingAverage
from .optimization import WarmupCosineScheduler, build_adamw, set_stage0_trainability
from .selection import ValidationScore, is_better_checkpoint

__all__ = [
    "CheckpointProvenanceError",
    "ExponentialMovingAverage",
    "ValidationScore",
    "WarmupCosineScheduler",
    "atomic_torch_save",
    "build_adamw",
    "capture_rng_state",
    "is_better_checkpoint",
    "load_checkpoint",
    "restore_rng_state",
    "set_stage0_trainability",
    "unwrap_model",
]
