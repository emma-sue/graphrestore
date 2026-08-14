"""Checkpoint-compatible prompt-free Restormer host for MiO-StageA.

The module hierarchy intentionally matches the selected ProVIR Stage-A host:
all 495 parent tensors retain their original ``encoder.*`` / ``decoder.*``
keys.  New GraphRestore modules must be added under separately whitelisted
prefixes rather than silently remapping this backbone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .restormer_blocks import (
    Downsample,
    OverlapPatchEmbed,
    Upsample,
    crop_to_shape,
    make_blocks,
    pad_to_multiple,
)


class SharedEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        blocks: Sequence[int] = (4, 6, 6, 8),
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if len(blocks) != 4 or len(heads) != 4:
            raise ValueError("SharedEncoder requires four block and head counts")
        widths = [dim * 2**level for level in range(4)]
        self.patch = OverlapPatchEmbed(3, dim, bias)
        self.level1 = make_blocks(
            widths[0], blocks[0], heads[0], expansion, bias, norm_type,
            gradient_checkpointing,
        )
        self.down12 = Downsample(widths[0])
        self.level2 = make_blocks(
            widths[1], blocks[1], heads[1], expansion, bias, norm_type,
            gradient_checkpointing,
        )
        self.down23 = Downsample(widths[1])
        self.level3 = make_blocks(
            widths[2], blocks[2], heads[2], expansion, bias, norm_type,
            gradient_checkpointing,
        )
        self.down34 = Downsample(widths[2])
        self.level4 = make_blocks(
            widths[3], blocks[3], heads[3], expansion, bias, norm_type,
            gradient_checkpointing,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        first = self.level1(self.patch(x))
        second = self.level2(self.down12(first))
        third = self.level3(self.down23(second))
        fourth = self.level4(self.down34(third))
        return first, second, third, fourth


class RestorationDecoder(nn.Module):
    """Decode encoder features into an RGB residual, never a restored image."""

    def __init__(
        self,
        dim: int = 48,
        blocks: Sequence[int] = (6, 6, 4),
        refinement: int = 4,
        heads: Sequence[int] = (1, 2, 4),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if len(blocks) != 3 or len(heads) != 3:
            raise ValueError("RestorationDecoder requires three block and head counts")
        first, second, third, fourth = [dim * 2**level for level in range(4)]
        self.up43 = Upsample(fourth)
        self.fuse3 = nn.Conv2d(third * 2, third, 1, bias=bias)
        self.level3 = make_blocks(
            third, blocks[0], heads[2], expansion, bias, norm_type,
            gradient_checkpointing,
        )
        self.up32 = Upsample(third)
        self.fuse2 = nn.Conv2d(second * 2, second, 1, bias=bias)
        self.level2 = make_blocks(
            second, blocks[1], heads[1], expansion, bias, norm_type,
            gradient_checkpointing,
        )
        self.up21 = Upsample(second)
        self.fuse1 = nn.Conv2d(first * 2, first, 1, bias=bias)
        self.level1 = make_blocks(
            first, blocks[2], heads[0], expansion, bias, norm_type,
            gradient_checkpointing,
        )
        self.refinement = make_blocks(
            first, refinement, heads[0], expansion, bias, norm_type,
            gradient_checkpointing,
        )
        self.head = nn.Conv2d(first, 3, 3, padding=1, bias=bias)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        first, second, third, fourth = features
        third_out = self.level3(
            self.fuse3(torch.cat((self.up43(fourth), third), dim=1))
        )
        second_out = self.level2(
            self.fuse2(torch.cat((self.up32(third_out), second), dim=1))
        )
        first_out = self.level1(
            self.fuse1(torch.cat((self.up21(second_out), first), dim=1))
        )
        return self.head(self.refinement(first_out))


class MiOStageA(nn.Module):
    """Pure Stage-A host: ``output = input + decoder_delta``."""

    def __init__(
        self,
        dim: int = 48,
        encoder_blocks: Sequence[int] = (4, 6, 6, 8),
        decoder_blocks: Sequence[int] = (6, 6, 4),
        refinement: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = SharedEncoder(
            dim,
            encoder_blocks,
            heads,
            expansion,
            bias,
            norm_type,
            gradient_checkpointing,
        )
        self.decoder = RestorationDecoder(
            dim,
            decoder_blocks,
            refinement,
            heads[:3],
            expansion,
            bias,
            norm_type,
            gradient_checkpointing,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.encoder(x)

    def decode_delta(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded, original_shape = pad_to_multiple(x, 8)
        restored = padded + self.decode_delta(self.encode(padded))
        return crop_to_shape(restored, original_shape)


# Compatibility name for scripts which refer to the audited ProVIR class name.
CleanRestormerAiO = MiOStageA


class BackboneLoadError(RuntimeError):
    """The parent state dict does not exactly match the registered backbone."""


@dataclass(frozen=True)
class BackboneLoadReport:
    source_tensor_count: int
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    allowed_missing_prefixes: tuple[str, ...]

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_keys)


DEFAULT_EXPANDED_MISSING_PREFIXES = (
    "decoder.skill_bank.",
    "planner.",
    "presence_thresholds",
)


def _load_payload(
    checkpoint_or_payload: str | Path | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(checkpoint_or_payload, (str, Path)):
        payload = torch.load(
            Path(checkpoint_or_payload),
            map_location="cpu",
            weights_only=False,
        )
    else:
        payload = checkpoint_or_payload
    if not isinstance(payload, Mapping):
        raise BackboneLoadError("parent checkpoint payload must be a mapping")
    return payload


def extract_parent_model_state(
    checkpoint_or_payload: str | Path | Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Extract the contract-bound ``payload['model']`` tensor mapping."""

    payload = _load_payload(checkpoint_or_payload)
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise BackboneLoadError("parent checkpoint has no non-empty 'model' mapping")
    non_tensors = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise BackboneLoadError(
            f"parent model contains non-tensor values: {non_tensors[:8]}"
        )
    return state  # type: ignore[return-value]


def load_parent_backbone(
    model: nn.Module,
    checkpoint_or_payload: str | Path | Mapping[str, Any],
    *,
    reference_model: MiOStageA | None = None,
    allowed_missing_prefixes: Sequence[str] = (),
) -> BackboneLoadReport:
    """Audit and load the selected Stage-A backbone without silent partial load.

    The source must first match a pure :class:`MiOStageA` exactly.  It must then
    match every corresponding key and shape in ``model``.  The only target keys
    allowed to remain initialized are explicit new-module prefixes.
    """

    source = extract_parent_model_state(checkpoint_or_payload)
    if reference_model is None:
        reference_model = MiOStageA()
    reference = reference_model.state_dict()

    source_keys = set(source)
    reference_keys = set(reference)
    source_unexpected = sorted(source_keys - reference_keys)
    source_missing = sorted(reference_keys - source_keys)
    source_shape = sorted(
        key
        for key in source_keys & reference_keys
        if tuple(source[key].shape) != tuple(reference[key].shape)
    )
    if source_unexpected or source_missing or source_shape:
        raise BackboneLoadError(
            "parent is not the registered pure host: "
            f"missing={source_missing[:8]}, unexpected={source_unexpected[:8]}, "
            f"shape_mismatch={source_shape[:8]}"
        )

    target = model.state_dict()
    target_keys = set(target)
    unexpected = sorted(source_keys - target_keys)
    shape_mismatches = sorted(
        key
        for key in source_keys & target_keys
        if tuple(source[key].shape) != tuple(target[key].shape)
    )
    missing = sorted(target_keys - source_keys)
    prefixes = tuple(str(prefix) for prefix in allowed_missing_prefixes)
    invalid_missing = [
        key
        for key in missing
        if not any(
            key == prefix or (prefix.endswith(".") and key.startswith(prefix))
            for prefix in prefixes
        )
    ]
    if unexpected or shape_mismatches or invalid_missing:
        raise BackboneLoadError(
            "expanded backbone load audit failed: "
            f"invalid_missing={invalid_missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatch={shape_mismatches[:8]}"
        )

    if missing:
        incompatible = model.load_state_dict(source, strict=False)
        if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
            raise BackboneLoadError(
                "PyTorch load result differed from the pre-load key audit"
            )
    else:
        model.load_state_dict(source, strict=True)

    return BackboneLoadReport(
        source_tensor_count=len(source),
        loaded_keys=tuple(sorted(source_keys)),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
        shape_mismatches=tuple(shape_mismatches),
        allowed_missing_prefixes=prefixes,
    )
