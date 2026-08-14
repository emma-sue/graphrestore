"""Atomic, provenance-checked GraphRestore checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


class CheckpointProvenanceError(RuntimeError):
    pass


def unwrap_model(model: nn.Module) -> nn.Module:
    """Remove torch.compile's wrapper without changing normal modules."""

    return getattr(model, "_orig_mod", model)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def atomic_torch_save(payload: Mapping[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def checkpoint_payload(
    *,
    stage: str,
    step: int,
    model: nn.Module,
    ema_state: Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    sampler_state: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": stage,
        "step": int(step),
        "model": unwrap_model(model).state_dict(),
        "ema": dict(ema_state) if ema_state is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_states": capture_rng_state(),
        "sampler_state": dict(sampler_state) if sampler_state is not None else None,
        "provenance": dict(provenance),
        "metrics": dict(metrics or {}),
    }


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, path))
        else:
            flattened[path] = value
    return flattened


def verify_provenance(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    actual_flat = _flatten_mapping(actual)
    expected_flat = _flatten_mapping(expected)
    mismatches = {
        key: {"expected": value, "actual": actual_flat.get(key, "<missing>")}
        for key, value in expected_flat.items()
        if actual_flat.get(key, object()) != value
    }
    if mismatches:
        raise CheckpointProvenanceError(f"checkpoint provenance mismatch: {mismatches}")


def load_checkpoint(
    checkpoint: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
    require_resumable: bool | None = None,
    expected_model_role: str | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != "graphrestore-checkpoint-v1":
        raise CheckpointProvenanceError("unsupported or missing checkpoint schema")
    if expected_provenance is not None:
        verify_provenance(payload.get("provenance", {}), expected_provenance)
    # Resume eligibility must be checked before mutating the model, optimizer,
    # scheduler, or RNG state.  Selection checkpoints intentionally pair EMA
    # model weights with raw optimizer moments and are therefore not exact
    # training snapshots.
    if require_resumable is not None and payload.get("resumable") is not require_resumable:
        raise CheckpointProvenanceError(
            "checkpoint resumable flag mismatch: "
            f"expected {require_resumable}, got {payload.get('resumable', '<missing>')}"
        )
    if expected_model_role is not None and payload.get("model_role") != expected_model_role:
        raise CheckpointProvenanceError(
            "checkpoint model role mismatch: "
            f"expected {expected_model_role!r}, got {payload.get('model_role', '<missing>')!r}"
        )
    if model is not None:
        unwrap_model(model).load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng_states"])
    return payload
