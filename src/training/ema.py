"""FP32 exponential moving average with auditable state."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn

from .checkpointing import unwrap_model


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.num_updates = 0
        source = unwrap_model(model).state_dict()
        self.shadow = {
            name: value.detach().clone().float() if value.is_floating_point() else value.detach().clone()
            for name, value in source.items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = unwrap_model(model).state_dict()
        if source.keys() != self.shadow.keys():
            raise RuntimeError("EMA/model state keys drifted")
        self.num_updates += 1
        for name, value in source.items():
            target = self.shadow[name]
            if value.is_floating_point():
                target.mul_(self.decay).add_(value.detach().to(target), alpha=1.0 - self.decay)
            else:
                target.copy_(value.detach().to(target))

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        loaded = state["shadow"]
        if not isinstance(loaded, Mapping):
            raise TypeError("EMA shadow must be a mapping")
        if loaded.keys() != self.shadow.keys():
            raise RuntimeError("EMA state keys drifted")
        # Checkpoints are loaded on CPU for portability.  Preserve the current
        # EMA tensor placement so a resumed CUDA run does not copy the entire
        # model GPU->CPU during every update.
        restored: dict[str, Tensor] = {}
        for name, value in loaded.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"EMA shadow entry {name!r} is not a tensor")
            reference = self.shadow[name]
            restored[name] = value.detach().to(
                device=reference.device,
                dtype=reference.dtype,
            ).clone()
        self.shadow = restored

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        destination = unwrap_model(model).state_dict()
        if destination.keys() != self.shadow.keys():
            raise RuntimeError("EMA/model state keys drifted")
        for name, value in destination.items():
            value.copy_(self.shadow[name].to(device=value.device, dtype=value.dtype))

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        destination = unwrap_model(model)
        backup = {name: value.detach().clone() for name, value in destination.state_dict().items()}
        self.copy_to(destination)
        try:
            yield
        finally:
            destination.load_state_dict(backup, strict=True)
