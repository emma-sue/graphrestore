"""Deterministic subset targets and local-skill guard supervision."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .agenticir_degradations import (
    AgenticIRAdapterError,
    AgenticIRDegradationAdapter,
    AppliedSequence,
    OperatorTrace,
)
from .manifests import SKILLS, SKILL_TO_ID, PrimaryRecipe
from .scale_canonicalizer import (
    MioIRScaleCanonicalizer,
    bgr_uint8_to_rgb_float,
)

DENSE_GUARD_SKILLS = frozenset({"rain", "haze", "low_light"})
GLOBAL_GUARD_SKILLS = frozenset(set(SKILLS) - DENSE_GUARD_SKILLS)


@dataclass
class SubsetTargets:
    """Full-resolution tensors synthesized from one frozen recipe."""

    input_rgb: torch.Tensor
    gt_clean_rgb: torch.Tensor
    target_after_i_rgb: torch.Tensor
    target_after_j_rgb: torch.Tensor
    only_i_rgb: torch.Tensor
    only_j_rgb: torch.Tensor
    guard_targets: torch.Tensor
    global_severity_targets: torch.Tensor
    presence_target: torch.Tensor
    dense_guard_mask: torch.Tensor
    global_guard_mask: torch.Tensor
    traces: tuple[OperatorTrace, ...]


def _canonical_rgb(
    applied: AppliedSequence,
    canonicalizer: MioIRScaleCanonicalizer,
) -> torch.Tensor:
    if applied.contains_low_resolution:
        return canonicalizer.canonicalize_native_lq(
            applied.output_bgr_uint8, scale=4
        )
    return bgr_uint8_to_rgb_float(applied.output_bgr_uint8)


def _low_light_guard(before_bgr: np.ndarray, after_bgr: np.ndarray) -> np.ndarray:
    before_y = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(
        np.float32
    ) / 255.0
    after_y = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(
        np.float32
    ) / 255.0
    return np.clip(1.0 - after_y / (before_y + 1e-6), 0.0, 1.0)


def guard_from_trace(trace: OperatorTrace, height: int, width: int) -> torch.Tensor:
    """Construct the contract-defined full-resolution guard for one trace."""

    if trace.skill_name == "rain":
        if trace.before_bgr_uint8 is None or trace.after_bgr_uint8 is None:
            raise AgenticIRAdapterError("rain trace lacks before/after images")
        guard = np.abs(
            trace.after_bgr_uint8.astype(np.float32)
            - trace.before_bgr_uint8.astype(np.float32)
        ).mean(axis=2) / 255.0
    elif trace.skill_name == "haze":
        if trace.transmission is None:
            raise AgenticIRAdapterError("haze trace lacks transmission")
        guard = 1.0 - trace.transmission
    elif trace.skill_name == "low_light":
        if trace.before_bgr_uint8 is None or trace.after_bgr_uint8 is None:
            raise AgenticIRAdapterError("low-light trace lacks before/after images")
        guard = _low_light_guard(
            trace.before_bgr_uint8, trace.after_bgr_uint8
        )
    else:
        guard = np.full(
            (height, width), trace.global_severity, dtype=np.float32
        )
    if guard.shape != (height, width):
        raise AgenticIRAdapterError(
            f"guard shape {guard.shape} does not match clean {(height, width)}"
        )
    return torch.from_numpy(np.ascontiguousarray(guard)).float().clamp_(0.0, 1.0)


def build_guard_targets(
    traces: tuple[OperatorTrace, ...],
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return guards, global severities, and actual-presence targets."""

    guards = torch.zeros((len(SKILLS), height, width), dtype=torch.float32)
    severities = torch.zeros(len(SKILLS), dtype=torch.float32)
    presence = torch.zeros(len(SKILLS), dtype=torch.float32)
    seen: set[int] = set()
    for trace in traces:
        skill_id = SKILL_TO_ID[trace.skill_name]
        if skill_id in seen:
            raise AgenticIRAdapterError(
                f"duplicate skill in one recipe: {trace.skill_name}"
            )
        seen.add(skill_id)
        guard = guard_from_trace(trace, height, width)
        guards[skill_id] = guard
        presence[skill_id] = 1.0
        if trace.skill_name in DENSE_GUARD_SKILLS:
            severities[skill_id] = guard.mean()
        else:
            severities[skill_id] = trace.global_severity
    return guards, severities, presence


def synthesize_subset_targets(
    clean_bgr_uint8: np.ndarray,
    recipe: PrimaryRecipe,
    adapter: AgenticIRDegradationAdapter,
    canonicalizer: MioIRScaleCanonicalizer,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
) -> SubsetTargets:
    """Synthesize ``x_both``, remaining-only targets, clean, and guards.

    For a pair ``(i, j)``, ``target_after_i`` is ``only_j`` and
    ``target_after_j`` is ``only_i``.  All variants reuse the exact recorded
    per-operator seed and official parameter order.
    """

    if (
        not isinstance(clean_bgr_uint8, np.ndarray)
        or clean_bgr_uint8.dtype != np.uint8
        or clean_bgr_uint8.ndim != 3
        or clean_bgr_uint8.shape[2] != 3
    ):
        raise TypeError("clean_bgr_uint8 must be a BGR uint8 HWC image")
    clean = np.ascontiguousarray(clean_bgr_uint8)
    if crop_box is None:
        clean_episode = clean

        def apply(parameters, capture_traces):
            return adapter.apply_sequence(
                clean,
                parameters,
                clean_id=recipe.clean_id,
                capture_traces=capture_traces,
            )

    else:
        top, left, crop_height, crop_width = crop_box
        clean_episode = clean[
            top : top + crop_height, left : left + crop_width
        ].copy()

        def apply(parameters, capture_traces):
            return adapter.apply_sequence_crop(
                clean,
                parameters,
                clean_id=recipe.clean_id,
                crop_box=crop_box,
                capture_traces=capture_traces,
            )

    height, width = clean_episode.shape[:2]
    both = apply(recipe.operator_params, True)
    input_rgb = _canonical_rgb(both, canonicalizer)
    gt_clean_rgb = bgr_uint8_to_rgb_float(clean_episode)

    if recipe.is_pair:
        only_i_applied = apply(recipe.operator_params[:1], False)
        only_i_rgb = _canonical_rgb(only_i_applied, canonicalizer)
        only_j_applied = apply(recipe.operator_params[1:], False)
        only_j_rgb = _canonical_rgb(only_j_applied, canonicalizer)
        target_after_i = only_j_rgb
        target_after_j = only_i_rgb
    else:
        # For a single recipe x_i is already the only-i state; avoid replaying
        # an expensive full-resolution official operator a second time.
        only_i_rgb = input_rgb
        only_j_rgb = gt_clean_rgb
        target_after_i = gt_clean_rgb
        target_after_j = gt_clean_rgb

    expected_shape = tuple(gt_clean_rgb.shape)
    candidates = {
        "input": input_rgb,
        "only_i": only_i_rgb,
        "only_j": only_j_rgb,
        "target_after_i": target_after_i,
        "target_after_j": target_after_j,
    }
    for name, tensor in candidates.items():
        if tuple(tensor.shape) != expected_shape:
            raise AgenticIRAdapterError(
                f"{recipe.sample_id}: {name} shape {tuple(tensor.shape)} "
                f"does not match clean {expected_shape}"
            )
    guards, severities, presence = build_guard_targets(
        both.traces, height=height, width=width
    )
    dense_mask = torch.tensor(
        [name in DENSE_GUARD_SKILLS for name in SKILLS], dtype=torch.bool
    )
    global_mask = ~dense_mask
    return SubsetTargets(
        input_rgb=input_rgb,
        gt_clean_rgb=gt_clean_rgb,
        target_after_i_rgb=target_after_i,
        target_after_j_rgb=target_after_j,
        only_i_rgb=only_i_rgb,
        only_j_rgb=only_j_rgb,
        guard_targets=guards,
        global_severity_targets=severities,
        presence_target=presence,
        dense_guard_mask=dense_mask,
        global_guard_mask=global_mask,
        traces=both.traces,
    )
