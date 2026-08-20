"""Fail-closed, exactly resumable formal MiO100 evaluation.

The module deliberately separates authorization and CPU evidence checks from
CUDA model construction.  A caller must validate an immutable one-shot
authorization before this module will inspect a formal manifest or allocate a
model.  Per-image inference is committed as an atomic directory containing a
lossless prediction and a hash-bound receipt.  The AgenticIR-compatible output
tree is made from hard links to those committed predictions, so interruption
between inference and publication never requires an optimizer/model rerun.

Except for the frozen ``contains_low_resolution`` metadata needed to select the
pre-registered input-scale canonicalization, ground-truth degradation labels
are used only after autonomous GraphRestore inference, for output routing and
aggregation.  They are never supplied to the model, planner, graph compiler,
or presence thresholds.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from src.data.scale_canonicalizer import (
    bgr_uint8_to_rgb_float,
    load_agenticir_online_canonical_input,
)
from src.evaluation.formal_inventory import (
    FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    FORMAL_AUTHORIZATION_PROTOCOL_SHA256,
    FORMAL_DATA_INVENTORY_PATH,
    REQUIRED_AUTHORIZATION_BINDINGS as INVENTORY_AUTHORIZATION_BINDINGS,
    FormalDataInventory,
    FormalInventoryError,
    InventoryFileIdentity,
    authorization_binding_paths,
    load_formal_data_inventory as load_strict_formal_data_inventory,
    stream_file_identity,
)
from src.metrics.agenticir_official import (
    OFFICIAL_GROUPS,
    aggregate_official_records,
    official_psnr_ssim,
)
from src.net import GraphRestore, GraphRestoreOutput, SKILLS
from src.utils.hashing import is_sha256, sha256_file, sha256_json
from src.utils.io import fsync_directory, iter_jsonl, load_json, load_yaml, utc_now_iso


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
AUTHORIZATION_SCHEMA = "graphrestore-formal-mio100-approval-v1"
RUN_CONTRACT_SCHEMA = "graphrestore-formal-mio100-run-contract-v1"
RECEIPT_SCHEMA = "graphrestore-formal-mio100-image-receipt-v1"
SHARD_SCHEMA = "graphrestore-formal-mio100-shard-v1"
SUMMARY_SCHEMA = "graphrestore-formal-mio100-summary-v1"
COMPLETE_SCHEMA = "graphrestore-formal-mio100-complete-v1"
TABLE1_INPUT_SCHEMA = "graphrestore.agenticir_table1_input.v1"

FORMAL_MANIFEST_FILENAME = "mio100_test_1440_agenticir_online_canonical.jsonl"
FORMAL_MANIFEST_SHA256 = (
    "83fb90dfa121681123f55e73df32eb6c1bc37e685c0e27ae07ad7e59a687a7f5"
)
MIOIR_MATLAB_FUNCTIONS_SHA256 = (
    "29a3a3d209ce15724202bfb01415e5d4e574e7b853090551a7938c7b78ec4975"
)
FORMAL_METHOD_NAME = "graphrestore_v7_1_stage4_step040000"
FORMAL_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/aaa/graphrestore/artifacts/formal_mio100/"
    "graphrestore_v7_1_stage4_step040000"
)
FORMAL_ROW_COUNT = 1_440
FORMAL_GROUP_COUNTS: Mapping[str, int] = {"A": 640, "B": 400, "C": 400}
FORMAL_COMBINATION_COUNTS: Mapping[str, int] = {
    combination: (80 if group == "A" else 100)
    for group, combinations in OFFICIAL_GROUPS.items()
    for combination in combinations
}
FORMAL_SHARD_COUNT = 1
MAXIMUM_VRAM_RESERVED_FRACTION = 0.90

REQUIRED_AUTHORIZATION_BINDINGS = INVENTORY_AUTHORIZATION_BINDINGS

_AUTHORIZATION_KEYS = {
    "schema_version",
    "kind",
    "protocol_id",
    "approved",
    "formal_mio100_authorized",
    "one_shot",
    "inference_only",
    "authorized_groups",
    "manifest_row_count",
    "method_name",
    "shard_count",
    "output_root",
    "approved_utc",
    "restrictions",
    "bindings",
}
_RESTRICTIONS = {
    "task_label_routing": False,
    "tta": False,
    "model_soup": False,
    "threshold_tuning": False,
    "result_driven_rerun": False,
    "overwrite": False,
}
_SAFE_METHOD_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.png$")
_DIAGNOSTIC_KEYS = {
    "program_levels",
    "parallel_levels",
    "active_skill_calls",
    "reentry_requests",
    "unexpected_activations",
    "precycle_graphs",
    "dropped_edges",
}
_RECEIPT_KEYS = {
    "schema_version",
    "contract_sha256",
    "created_utc",
    "index",
    "sample_id",
    "row_sha256",
    "clean_id",
    "group",
    "combination",
    "native_lq_path",
    "native_lq_sha256",
    "target_png",
    "target_sha256",
    "prediction_png",
    "prediction_sha256",
    "psnr",
    "ssim",
    "latency_ms",
    "peak_reserved_fraction",
    "diagnostics",
}


class MiO100EvaluationError(RuntimeError):
    """A frozen formal-evaluation invariant was violated."""


@dataclass(frozen=True)
class ArtifactBinding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class FormalAuthorization:
    path: Path
    sha256: str
    approved_utc: str
    output_root: Path
    method_name: str
    shard_count: int
    bindings: Mapping[str, ArtifactBinding]


@dataclass(frozen=True)
class MiO100Record:
    index: int
    sample_id: str
    clean_id: str
    group: str
    degradations: tuple[str, ...]
    combination: str
    native_lq_path: Path
    target_path: Path
    contains_low_resolution: bool
    row: Mapping[str, Any]
    row_sha256: str
    expected_native_sha256: str | None = None
    expected_target_sha256: str | None = None
    native_file_identity: InventoryFileIdentity | None = None
    target_file_identity: InventoryFileIdentity | None = None

    @property
    def depth_dir(self) -> str:
        return "d2" if len(self.degradations) == 2 else "d3"

    @property
    def output_filename(self) -> str:
        return self.target_path.name

    @property
    def record_key(self) -> str:
        short = hashlib.sha256(self.sample_id.encode("utf-8")).hexdigest()[:16]
        return f"{self.index:06d}-{short}"


@dataclass(frozen=True)
class Stage4Checkpoint:
    path: Path
    sha256: str
    model_state: Mapping[str, Tensor]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    prediction: Tensor
    diagnostics: Mapping[str, int | float]
    latency_ms: float


@dataclass(frozen=True)
class EvaluationRun:
    root: Path
    method_name: str
    contract_path: Path
    contract_sha256: str
    contract: Mapping[str, Any]


@dataclass(frozen=True)
class FormalEvaluatorCompletion:
    """Fully cross-validated, JSON-stable formal evaluator evidence."""

    complete_path: Path
    complete_sha256: str
    authorization: ArtifactBinding
    run_contract: ArtifactBinding
    summary: ArtifactBinding
    per_image: ArtifactBinding
    table1_input: ArtifactBinding
    checkpoint: ArtifactBinding
    manifest: ArtifactBinding
    formal_data_inventory: ArtifactBinding
    predictions_digest: str
    evidence: Mapping[str, Any]


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MiO100EvaluationError(f"{field} must be a mapping")
    return value


def _canonical_regular_file(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise MiO100EvaluationError(f"{field} must be an absolute path")
    if candidate.is_symlink():
        raise MiO100EvaluationError(f"{field} must not be a symlink: {candidate}")
    try:
        canonical = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MiO100EvaluationError(f"missing {field}: {candidate}") from exc
    if canonical != candidate or not canonical.is_file():
        raise MiO100EvaluationError(
            f"{field} must be a canonical regular file: {candidate}"
        )
    return canonical


def _hash_stable_file(path: Path, *, field: str) -> str:
    before = sha256_file(path)
    after = sha256_file(path)
    if before != after:
        raise MiO100EvaluationError(f"{field} changed while hashing: {path}")
    return before


def _require_read_only(path: Path, *, field: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise MiO100EvaluationError(f"{field} must be immutable/read-only: {path}")


def _validate_utc(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z") or "T" not in value:
        raise MiO100EvaluationError(f"{field} must be an RFC3339 UTC timestamp")
    return value


def validate_formal_authorization(
    path: str | Path,
    *,
    expected_bindings: Mapping[str, str | Path] | None = None,
    expected_output_root: str | Path = FORMAL_OUTPUT_ROOT,
    expected_method_name: str = FORMAL_METHOD_NAME,
    expected_shard_count: int = FORMAL_SHARD_COUNT,
) -> FormalAuthorization:
    """Validate the independent immutable authorization before any data/CUDA use.

    ``expected_bindings`` is injectable so the outer approval publisher can
    additionally pin canonical paths.  Regardless of injection, every required
    binding key must exist, point to a canonical regular file, and hash exactly.
    """

    authorization_path = _canonical_regular_file(path, field="formal authorization")
    _require_read_only(authorization_path, field="formal authorization")
    payload = _mapping(load_json(authorization_path), "formal authorization")
    if set(payload) != _AUTHORIZATION_KEYS:
        raise MiO100EvaluationError(
            "formal authorization fields drifted: "
            f"expected={sorted(_AUTHORIZATION_KEYS)}, actual={sorted(payload)}"
        )
    expected_scalars: Mapping[str, object] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "kind": "formal_mio100_approval",
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "formal_mio100_authorized": True,
        "one_shot": True,
        "inference_only": True,
        "authorized_groups": ["A", "B", "C"],
        "manifest_row_count": FORMAL_ROW_COUNT,
        "method_name": expected_method_name,
        "shard_count": expected_shard_count,
        "output_root": str(Path(expected_output_root).resolve(strict=False)),
        "restrictions": dict(_RESTRICTIONS),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected_scalars.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise MiO100EvaluationError(f"formal authorization scope drifted: {mismatches}")
    approved_utc = _validate_utc(payload.get("approved_utc"), field="approved_utc")
    raw_bindings = _mapping(payload.get("bindings"), "authorization bindings")
    if set(raw_bindings) != set(REQUIRED_AUTHORIZATION_BINDINGS):
        raise MiO100EvaluationError(
            "formal authorization binding keys drifted: "
            f"expected={sorted(REQUIRED_AUTHORIZATION_BINDINGS)}, "
            f"actual={sorted(raw_bindings)}"
        )

    expected_paths = {
        name: Path(value).resolve(strict=False)
        for name, value in (expected_bindings or {}).items()
    }
    unknown_expected = set(expected_paths) - set(REQUIRED_AUTHORIZATION_BINDINGS)
    if unknown_expected:
        raise MiO100EvaluationError(
            f"unknown injected authorization bindings: {sorted(unknown_expected)}"
        )
    bindings: dict[str, ArtifactBinding] = {}
    for name in REQUIRED_AUTHORIZATION_BINDINGS:
        raw = _mapping(raw_bindings[name], f"authorization binding {name}")
        if set(raw) != {"path", "sha256"}:
            raise MiO100EvaluationError(
                f"authorization binding {name} must contain only path/sha256"
            )
        raw_path = raw.get("path")
        digest = raw.get("sha256")
        if not isinstance(raw_path, str) or not is_sha256(digest):
            raise MiO100EvaluationError(f"authorization binding {name} is malformed")
        bound_path = _canonical_regular_file(raw_path, field=f"binding {name}")
        if name in expected_paths and bound_path != expected_paths[name]:
            raise MiO100EvaluationError(
                f"authorization binding {name} path drifted: "
                f"{bound_path} != {expected_paths[name]}"
            )
        actual = _hash_stable_file(bound_path, field=f"binding {name}")
        if actual != digest:
            raise MiO100EvaluationError(
                f"authorization binding {name} hash drifted: {actual} != {digest}"
            )
        if name in {
            "formal_data_inventory",
            "formal_authorization_protocol",
            "metric_weight_inventory",
        }:
            _require_read_only(bound_path, field=f"binding {name}")
        bindings[name] = ArtifactBinding(bound_path, digest)

    return FormalAuthorization(
        path=authorization_path,
        sha256=_hash_stable_file(authorization_path, field="formal authorization"),
        approved_utc=approved_utc,
        output_root=Path(str(payload["output_root"])),
        method_name=str(payload["method_name"]),
        shard_count=int(payload["shard_count"]),
        bindings=bindings,
    )


def validate_stage4_completion(
    path: str | Path,
    *,
    checkpoint_sha256: str,
    authorization: FormalAuthorization | None = None,
) -> Mapping[str, Any]:
    complete_path = _canonical_regular_file(path, field="Stage4 completion")
    payload = _mapping(load_json(complete_path), "Stage4 completion")
    expected: Mapping[str, object] = {
        "schema_version": "graphrestore-stage4-runtime-v1",
        "protocol_id": PROTOCOL_ID,
        "step": 40_000,
        "formal_mio100_started": False,
        "waiting_for": "new_user_authorization_for_formal_mio100",
        "best_ema_sha256": checkpoint_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise MiO100EvaluationError(f"Stage4 completion gate failed: {mismatches}")
    best_score = _mapping(payload.get("best_score"), "Stage4 completion best score")
    latest_score = _mapping(
        payload.get("latest_score"), "Stage4 completion latest score"
    )
    if best_score.get("step") != 40_000 or latest_score.get("step") != 40_000:
        raise MiO100EvaluationError(
            "formal MiO100 requires the selected and latest Stage4 step 40000"
        )
    for key in (
        "maximum_train_peak_reserved_fraction",
        "maximum_validation_peak_reserved_fraction",
    ):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < MAXIMUM_VRAM_RESERVED_FRACTION
        ):
            raise MiO100EvaluationError(f"Stage4 completion has invalid {key}")
    if payload.get("diagnostics_selected_best_ema_sha256") != checkpoint_sha256:
        raise MiO100EvaluationError(
            "Stage4 final diagnostics did not use the selected step-40000 EMA"
        )
    if authorization is not None:
        field_bindings = {
            "stage4_checkpoint": ("best_ema_path", "best_ema_sha256"),
            "stage4_validation": ("validation", "validation_sha256"),
            "stage4_calibration_history": (
                "calibration_history",
                "calibration_history_sha256",
            ),
            "stage4_report": ("report", "report_sha256"),
            "stage4_diagnostics_json": (
                "diagnostics_json",
                "diagnostics_json_sha256",
            ),
            "stage4_diagnostics_report": (
                "diagnostics_report",
                "diagnostics_report_sha256",
            ),
        }
        for binding_name, (path_field, sha_field) in field_bindings.items():
            binding = authorization.bindings[binding_name]
            if (
                payload.get(path_field) != str(binding.path)
                or payload.get(sha_field) != binding.sha256
            ):
                raise MiO100EvaluationError(
                    f"Stage4 completion/{binding_name} binding drifted"
                )
        diagnostics = _mapping(
            load_json(authorization.bindings["stage4_diagnostics_json"].path),
            "Stage4 zero-training diagnostics",
        )
        compiler_modes = _mapping(
            diagnostics.get("compiler_modes"), "Stage4 compiler diagnostics"
        )
        guard_modes = _mapping(
            diagnostics.get("guard_modes"), "Stage4 guard diagnostics"
        )
        if (
            diagnostics.get("schema_version")
            != "graphrestore-stage4-zero-training-diagnostics-v1"
            or diagnostics.get("protocol_id") != PROTOCOL_ID
            or diagnostics.get("selected_best_ema_path")
            != str(authorization.bindings["stage4_checkpoint"].path)
            or diagnostics.get("selected_best_ema_sha256") != checkpoint_sha256
            or diagnostics.get("optimizer_updates") != 0
            or diagnostics.get("model_ema_rng_unchanged") is not True
            or set(compiler_modes)
            != {"full_partial_order", "forced_total_order", "parallel_only"}
            or set(guard_modes) != {"predicted_spatial", "global_mean", "all_one"}
        ):
            raise MiO100EvaluationError(
                "Stage4 six-mode zero-training diagnostic gate failed"
            )
        for mode_name, raw_mode in (*compiler_modes.items(), *guard_modes.items()):
            mode = _mapping(raw_mode, f"Stage4 diagnostic mode {mode_name}")
            peak = mode.get("peak_reserved_fraction")
            if (
                mode.get("image_count") != 1_600
                or isinstance(peak, bool)
                or not isinstance(peak, (int, float))
                or not math.isfinite(float(peak))
                or not 0.0 <= float(peak) < MAXIMUM_VRAM_RESERVED_FRACTION
            ):
                raise MiO100EvaluationError(
                    f"Stage4 diagnostic mode {mode_name} evidence is invalid"
                )
    return payload


def validate_protocol_bindings(authorization: FormalAuthorization) -> None:
    """Cross-check bound inventory/parity semantics, not only their byte hashes."""

    protocol_binding = authorization.bindings["formal_authorization_protocol"]
    if (
        protocol_binding.path != FORMAL_AUTHORIZATION_PROTOCOL_PATH
        or protocol_binding.sha256 != FORMAL_AUTHORIZATION_PROTOCOL_SHA256
    ):
        raise MiO100EvaluationError("formal authorization protocol binding drifted")
    inventory_binding = authorization.bindings["manifest_inventory"]
    inventory = _mapping(load_json(inventory_binding.path), "manifest inventory")
    if inventory.get("schema_version") != (
        "graphrestore.agenticir_online_canonical.inventory.v1"
    ):
        raise MiO100EvaluationError("online-canonical manifest inventory drifted")
    canonicalizer = _mapping(
        inventory.get("canonicalizer"), "manifest inventory canonicalizer"
    )
    mioir_binding = authorization.bindings["mioir_matlab_functions"]
    if canonicalizer != {
        "implementation": str(mioir_binding.path),
        "operation": "native BGR uint8 -> RGB float -> MiOIR imresize x4 -> clamp",
        "requantized_after_resize": "false",
        "sha256": mioir_binding.sha256,
    }:
        raise MiO100EvaluationError("manifest inventory canonicalizer binding drifted")
    manifests = _mapping(inventory.get("manifests"), "manifest inventory manifests")
    formal_binding = authorization.bindings["formal_manifest"]
    formal_entry = _mapping(
        manifests.get(FORMAL_MANIFEST_FILENAME), "formal manifest inventory entry"
    )
    if (
        formal_entry.get("path") != str(formal_binding.path)
        or formal_entry.get("sha256") != formal_binding.sha256
        or formal_entry.get("rows") != FORMAL_ROW_COUNT
    ):
        raise MiO100EvaluationError("formal manifest inventory entry drifted")

    parity = _mapping(
        load_json(authorization.bindings["metric_parity_summary"].path),
        "metric parity summary",
    )
    facts = _mapping(parity.get("facts"), "metric parity facts")
    versions = _mapping(facts.get("versions"), "metric parity versions")
    reference = _mapping(
        versions.get("reference_environment"), "metric parity reference environment"
    )
    if (
        parity.get("protocol") != "graphrestore-v7.1-agenticir-metric-parity"
        or parity.get("passed") is not True
        or parity.get("failure_count") != 0
        or facts.get("canonical_float_exact") is not True
        or facts.get("canonical_uint8_exact") is not True
        or facts.get("max_psnr_abs_diff") != 0.0
        or isinstance(facts.get("max_ssim_abs_diff"), bool)
        or not isinstance(facts.get("max_ssim_abs_diff"), (int, float))
        or not math.isfinite(float(facts["max_ssim_abs_diff"]))
        or float(facts["max_ssim_abs_diff"]) > 4.0e-7
        or versions.get("agenticir_scorer_sha256")
        != authorization.bindings["agenticir_scorer"].sha256
        or reference.get("pyiqa") != "0.1.10"
    ):
        raise MiO100EvaluationError("metric parity evidence drifted")


def _canonical_row_bytes(row: Mapping[str, Any]) -> bytes:
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load_formal_manifest(
    path: str | Path,
    *,
    expected_sha256: str = FORMAL_MANIFEST_SHA256,
    expected_group_counts: Mapping[str, int] = FORMAL_GROUP_COUNTS,
    expected_combination_counts: Mapping[str, int] = FORMAL_COMBINATION_COUNTS,
    required_filename: str = FORMAL_MANIFEST_FILENAME,
) -> tuple[MiO100Record, ...]:
    manifest_path = _canonical_regular_file(path, field="formal MiO100 manifest")
    if manifest_path.name != required_filename:
        raise MiO100EvaluationError(
            "only the full online-canonical MiO100 manifest is accepted: "
            f"{required_filename}"
        )
    if not is_sha256(expected_sha256):
        raise MiO100EvaluationError("expected manifest SHA256 is malformed")
    actual_sha = _hash_stable_file(manifest_path, field="formal manifest")
    if actual_sha != expected_sha256:
        raise MiO100EvaluationError(
            f"formal manifest hash drifted: {actual_sha} != {expected_sha256}"
        )

    records: list[MiO100Record] = []
    sample_ids: set[str] = set()
    output_keys: set[tuple[str, str]] = set()
    group_counts: dict[str, int] = {}
    combination_counts: dict[str, int] = {}
    official_group_for = {
        combination: group
        for group, combinations in OFFICIAL_GROUPS.items()
        for combination in combinations
    }
    for index, (line_number, row) in enumerate(iter_jsonl(manifest_path)):
        context = f"{manifest_path}:{line_number}"
        if row.get("schema_version") != "graphrestore.agenticir_online_canonical.v1":
            raise MiO100EvaluationError(f"{context}: schema drifted")
        if row.get("input_mode") != "agenticir_online_canonical":
            raise MiO100EvaluationError(f"{context}: input_mode drifted")
        if row.get("source") != "AgenticIR" or row.get("split") != "test":
            raise MiO100EvaluationError(f"{context}: source/split drifted")
        for forbidden in ("canonical_lq_path", "legacy_opencv_canonical_lq_path"):
            if row.get(forbidden) is not None:
                raise MiO100EvaluationError(
                    f"{context}: legacy canonical path is forbidden"
                )
        sample_id = row.get("sample_id")
        clean_id = row.get("clean_id")
        group = row.get("group")
        degradations = row.get("degradations")
        if not isinstance(sample_id, str) or not sample_id:
            raise MiO100EvaluationError(f"{context}: invalid sample_id")
        if sample_id in sample_ids:
            raise MiO100EvaluationError(f"{context}: duplicate sample_id {sample_id}")
        if not isinstance(clean_id, str) or not clean_id:
            raise MiO100EvaluationError(f"{context}: invalid clean_id")
        if group not in expected_group_counts:
            raise MiO100EvaluationError(f"{context}: unexpected group {group!r}")
        if not isinstance(degradations, list) or not all(
            isinstance(item, str) and item for item in degradations
        ):
            raise MiO100EvaluationError(f"{context}: invalid degradations")
        degradation_tuple = tuple(degradations)
        expected_depth = 3 if group == "C" else 2
        if len(degradation_tuple) != expected_depth:
            raise MiO100EvaluationError(
                f"{context}: degradation depth disagrees with group {group}"
            )
        combination = "+".join(degradation_tuple)
        if official_group_for.get(combination) != group:
            raise MiO100EvaluationError(
                f"{context}: non-official combination/group {combination}/{group}"
            )

        native_text = row.get("native_lq_path")
        target_text = row.get("gt_path")
        if not isinstance(native_text, str) or not isinstance(target_text, str):
            raise MiO100EvaluationError(f"{context}: missing native/GT path")
        native_path = Path(native_text)
        target_path = Path(target_text)
        if not native_path.is_absolute() or not target_path.is_absolute():
            raise MiO100EvaluationError(f"{context}: native/GT paths must be absolute")
        if row.get("input_path") != native_text:
            raise MiO100EvaluationError(f"{context}: input_path != native_lq_path")
        if target_path.name != f"{clean_id}.png" or not _SAFE_FILENAME_RE.fullmatch(
            target_path.name
        ):
            raise MiO100EvaluationError(f"{context}: unsafe/noncanonical GT filename")
        contains_lr = row.get("contains_low_resolution")
        expected_lr = "low resolution" in degradation_tuple
        if not isinstance(contains_lr, bool) or contains_lr is not expected_lr:
            raise MiO100EvaluationError(f"{context}: low-resolution flag drifted")
        expected_scale = 0.25 if contains_lr else 1.0
        expected_factor = 4 if contains_lr else 1
        expected_operation = (
            "mioir_basicsr_native_uint8_to_rgb_float_x4"
            if contains_lr
            else "native_uint8_to_rgb_float_identity"
        )
        if (
            row.get("native_scale") != expected_scale
            or row.get("scale_factor") != expected_factor
            or row.get("online_scale_factor") != expected_factor
            or row.get("requantize_after_online_resize") is not False
            or row.get("online_canonicalization") != expected_operation
            or row.get("mioir_matlab_functions_sha256") != MIOIR_MATLAB_FUNCTIONS_SHA256
        ):
            raise MiO100EvaluationError(f"{context}: canonicalization metadata drifted")
        if (
            row.get("input_storage_color_order") != "BGR"
            or row.get("model_input_color_order") != "RGB"
            or row.get("model_input_dtype") != "float32"
        ):
            raise MiO100EvaluationError(f"{context}: color/dtype metadata drifted")
        output_key = (combination, target_path.name)
        if output_key in output_keys:
            raise MiO100EvaluationError(
                f"{context}: duplicate AgenticIR output key {output_key}"
            )
        sample_ids.add(sample_id)
        output_keys.add(output_key)
        group_counts[group] = group_counts.get(group, 0) + 1
        combination_counts[combination] = combination_counts.get(combination, 0) + 1
        records.append(
            MiO100Record(
                index=index,
                sample_id=sample_id,
                clean_id=clean_id,
                group=group,
                degradations=degradation_tuple,
                combination=combination,
                native_lq_path=native_path,
                target_path=target_path,
                contains_low_resolution=contains_lr,
                row=dict(row),
                row_sha256=hashlib.sha256(_canonical_row_bytes(row)).hexdigest(),
            )
        )

    expected_total = sum(int(value) for value in expected_group_counts.values())
    if len(records) != expected_total:
        raise MiO100EvaluationError(
            f"formal manifest requires {expected_total} rows, got {len(records)}"
        )
    if group_counts != dict(expected_group_counts):
        raise MiO100EvaluationError(
            f"formal manifest group counts drifted: {group_counts}"
        )
    if combination_counts != dict(expected_combination_counts):
        raise MiO100EvaluationError(
            f"formal manifest combination counts drifted: {combination_counts}"
        )
    return tuple(records)


def load_and_bind_formal_data_inventory(
    path: str | Path,
    records: Sequence[MiO100Record],
    *,
    expected_sha256: str,
    manifest_path: str | Path,
    manifest_sha256: str,
    authorization_protocol_path: str | Path = FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    authorization_protocol_sha256: str = FORMAL_AUTHORIZATION_PROTOCOL_SHA256,
    verify_file_bytes: bool = True,
    inventory_validation_kwargs: Mapping[str, Any] | None = None,
) -> tuple[tuple[MiO100Record, ...], FormalDataInventory]:
    """Validate the pre-registered byte inventory and attach it to every row."""

    try:
        inventory = load_strict_formal_data_inventory(
            path,
            expected_manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha256,
            expected_authorization_protocol_path=authorization_protocol_path,
            expected_authorization_protocol_sha256=authorization_protocol_sha256,
            verify_file_bytes=verify_file_bytes,
            **dict(inventory_validation_kwargs or {}),
        )
    except FormalInventoryError as exc:
        raise MiO100EvaluationError(f"formal data inventory rejected: {exc}") from exc
    if inventory.sha256 != expected_sha256:
        raise MiO100EvaluationError("formal data inventory authorization SHA drifted")
    if len(records) != len(inventory.rows):
        raise MiO100EvaluationError("formal data inventory row count drifted")
    bound = []
    for record, row in zip(records, inventory.rows, strict=True):
        if (
            row.index != record.index
            or row.sample_id != record.sample_id
            or row.row_sha256 != record.row_sha256
            or row.native_lq_path != record.native_lq_path
            or row.target_path != record.target_path
        ):
            raise MiO100EvaluationError(
                f"formal data inventory row binding drifted at {record.index}"
            )
        native_identity = inventory.files.get(record.native_lq_path)
        target_identity = inventory.files.get(record.target_path)
        if (
            native_identity is None
            or target_identity is None
            or native_identity.sha256 != row.native_lq_sha256
            or target_identity.sha256 != row.target_sha256
        ):
            raise MiO100EvaluationError(
                f"formal data inventory file binding drifted at {record.index}"
            )
        bound.append(
            replace(
                record,
                expected_native_sha256=row.native_lq_sha256,
                expected_target_sha256=row.target_sha256,
                native_file_identity=native_identity,
                target_file_identity=target_identity,
            )
        )
    return tuple(bound), inventory


def _verify_record_file_identity(
    record: MiO100Record,
    *,
    role: str,
) -> str:
    if role == "native_lq":
        path = record.native_lq_path
        expected_sha = record.expected_native_sha256
        expected = record.native_file_identity
    elif role == "target":
        path = record.target_path
        expected_sha = record.expected_target_sha256
        expected = record.target_file_identity
    else:  # pragma: no cover - private caller invariant
        raise MiO100EvaluationError(f"unknown formal data role: {role}")
    if expected is None or not is_sha256(expected_sha):
        raise MiO100EvaluationError(
            f"record {record.sample_id} is not bound to formal data inventory"
        )
    try:
        actual = stream_file_identity(path, field=f"formal {role}")
    except FormalInventoryError as exc:
        raise MiO100EvaluationError(f"formal {role} identity rejected: {exc}") from exc
    for key, value in (
        ("sha256", expected_sha),
        ("size_bytes", expected.size_bytes),
        ("mode", expected.mode),
        ("device", expected.device),
        ("inode", expected.inode),
    ):
        if actual[key] != value:
            raise MiO100EvaluationError(
                f"formal {role} inventory drifted for {record.sample_id}: {key}"
            )
    return str(actual["sha256"])


def load_stage4_best_ema(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_config_sha256: str | None = None,
    expected_tensor_count: int | None = 1_640,
) -> Stage4Checkpoint:
    checkpoint_path = _canonical_regular_file(path, field="Stage4 best EMA")
    if checkpoint_path.name != "best_ema.pth":
        raise MiO100EvaluationError("formal checkpoint must be named best_ema.pth")
    actual_sha = _hash_stable_file(checkpoint_path, field="Stage4 best EMA")
    if actual_sha != expected_sha256:
        raise MiO100EvaluationError(
            f"Stage4 checkpoint hash drifted: {actual_sha} != {expected_sha256}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise MiO100EvaluationError(
            f"could not CPU-load Stage4 best EMA: {exc}"
        ) from exc
    payload = _mapping(payload, "Stage4 checkpoint")
    expected_header: Mapping[str, object] = {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage4",
        "step": 40_000,
        "model_role": "ema_selection",
        "resumable": False,
        "pending_validation_step": None,
    }
    header_mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected_header.items()
        if payload.get(key) != value
    }
    if header_mismatches:
        raise MiO100EvaluationError(
            f"Stage4 best EMA header drifted: {header_mismatches}"
        )
    model = _mapping(payload.get("model"), "Stage4 checkpoint model")
    ema = _mapping(payload.get("ema"), "Stage4 checkpoint EMA")
    shadow = _mapping(ema.get("shadow"), "Stage4 checkpoint EMA shadow")
    if set(model) != set(shadow):
        raise MiO100EvaluationError("Stage4 best model/EMA key sets differ")
    if expected_tensor_count is not None and len(model) != expected_tensor_count:
        raise MiO100EvaluationError(
            f"Stage4 best tensor count drifted: {len(model)} != {expected_tensor_count}"
        )
    normalized: dict[str, Tensor] = {}
    for name in sorted(model):
        model_value = model[name]
        ema_value = shadow[name]
        if not torch.is_tensor(model_value) or not torch.is_tensor(ema_value):
            raise MiO100EvaluationError(f"non-tensor Stage4 state at {name}")
        if (
            model_value.shape != ema_value.shape
            or model_value.dtype != ema_value.dtype
            or model_value.layout != ema_value.layout
            or not torch.equal(model_value, ema_value)
        ):
            raise MiO100EvaluationError(
                f"Stage4 best model is not bit-exact to EMA shadow at {name}"
            )
        if model_value.is_floating_point() and not bool(
            torch.isfinite(model_value).all().item()
        ):
            raise MiO100EvaluationError(f"non-finite Stage4 best tensor at {name}")
        normalized[name] = model_value.detach().cpu()
    if ema.get("num_updates") != 40_000:
        raise MiO100EvaluationError("Stage4 EMA update count is not 40000")
    provenance = _mapping(payload.get("provenance"), "Stage4 checkpoint provenance")
    if (
        provenance.get("schema_version") != "graphrestore-stage4-runtime-v1"
        or provenance.get("protocol_id") != PROTOCOL_ID
    ):
        raise MiO100EvaluationError("Stage4 checkpoint provenance protocol drifted")
    if (
        expected_config_sha256 is not None
        and provenance.get("config_sha256") != expected_config_sha256
    ):
        raise MiO100EvaluationError("Stage4 checkpoint/config binding drifted")
    return Stage4Checkpoint(
        path=checkpoint_path,
        sha256=actual_sha,
        model_state=normalized,
        provenance=dict(provenance),
    )


def _write_new_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise MiO100EvaluationError(
                f"refusing to write through a symlink: {ancestor}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    for ancestor in (path.parent, *path.parent.parents):
        if ancestor.is_symlink():
            raise MiO100EvaluationError(
                f"refusing to write through a symlink: {ancestor}"
            )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise MiO100EvaluationError(
            f"refusing to overwrite existing artifact: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
        fsync_directory(path.parent)
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_or_verify_immutable(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise MiO100EvaluationError(
                f"existing immutable artifact differs from expected bytes: {path}"
            )
        _require_read_only(path, field="immutable evaluation artifact")
        return
    _write_new_bytes(path, payload)


def _require_data_disk_output(path: Path) -> Path:
    if not path.is_absolute():
        raise MiO100EvaluationError("formal output_root must be absolute")
    canonical = path.resolve(strict=False)
    data_root = Path("/root/autodl-tmp").resolve(strict=True)
    if canonical == data_root or data_root not in canonical.parents:
        raise MiO100EvaluationError(
            f"formal outputs must stay on the data disk under {data_root}"
        )
    for parent in (canonical, *canonical.parents):
        if parent == data_root.parent:
            break
        if parent.exists() and parent.is_symlink():
            raise MiO100EvaluationError(f"formal output path crosses symlink: {parent}")
    return canonical


def prepare_run_contract(
    authorization: FormalAuthorization,
    *,
    manifest_sha256: str,
    data_inventory_sha256: str,
    data_inventory_rows_digest: str,
    data_inventory_files_digest: str,
    checkpoint_sha256: str,
    config_sha256: str,
    shard_count: int,
    enforce_data_disk: bool = True,
) -> EvaluationRun:
    if (
        _hash_stable_file(authorization.path, field="formal authorization")
        != authorization.sha256
    ):
        raise MiO100EvaluationError("formal authorization changed before run setup")
    if shard_count != authorization.shard_count:
        raise MiO100EvaluationError("shard_count differs from formal authorization")
    if (
        data_inventory_sha256 != authorization.bindings["formal_data_inventory"].sha256
        or not is_sha256(data_inventory_rows_digest)
        or not is_sha256(data_inventory_files_digest)
    ):
        raise MiO100EvaluationError("formal data inventory run binding drifted")
    if not _SAFE_METHOD_RE.fullmatch(authorization.method_name):
        raise MiO100EvaluationError("unsafe formal method_name")
    root = (
        _require_data_disk_output(authorization.output_root)
        if enforce_data_disk
        else authorization.output_root.resolve(strict=False)
    )
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise MiO100EvaluationError("formal output_root is not a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "run_contract.json"
    core = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "authorization": {
            "path": str(authorization.path),
            "sha256": authorization.sha256,
        },
        "authorization_bindings": {
            name: {"path": str(binding.path), "sha256": binding.sha256}
            for name, binding in sorted(authorization.bindings.items())
        },
        "manifest_sha256": manifest_sha256,
        "formal_data_inventory": {
            "path": str(authorization.bindings["formal_data_inventory"].path),
            "sha256": data_inventory_sha256,
            "rows_digest": data_inventory_rows_digest,
            "files_digest": data_inventory_files_digest,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "method_name": authorization.method_name,
        "output_root": str(root),
        "manifest_row_count": FORMAL_ROW_COUNT,
        "groups": dict(FORMAL_GROUP_COUNTS),
        "combination_counts": dict(FORMAL_COMBINATION_COUNTS),
        "shard_count": shard_count,
        "assignment": "manifest_index_mod_shard_count",
        "inference": {
            "autonomous_graphrestore": True,
            "task_label_routing": False,
            "max_rounds": 3,
            "amp_dtype": "bf16",
            "tf32": True,
            "tta": False,
            "model_soup": False,
        },
        "output_protocol": {
            "crop": "top_left_to_gt_shape",
            "quantization": "clamp_round_uint8",
            "encoding": "lossless_png",
            "score_source": "png_readback",
            "layout": "methods/<method>/d2|d3/<combination>/<gt_basename>",
            "overwrite": False,
        },
        "vram_maximum_peak_reserved_fraction": MAXIMUM_VRAM_RESERVED_FRACTION,
    }
    if contract_path.exists():
        payload = _mapping(load_json(contract_path), "formal run contract")
        if set(payload) != set(core) | {"created_utc"}:
            raise MiO100EvaluationError("existing formal run-contract schema drifted")
        for key, value in core.items():
            if payload.get(key) != value:
                raise MiO100EvaluationError(
                    f"existing formal run-contract drifted at {key}"
                )
        _validate_utc(payload.get("created_utc"), field="run_contract.created_utc")
        _require_read_only(contract_path, field="formal run contract")
        contract = dict(payload)
    else:
        unexpected = [
            item for item in root.iterdir() if item.name != "run_contract.json"
        ]
        if unexpected:
            raise MiO100EvaluationError(
                "formal output_root is non-empty without a run contract: "
                f"{sorted(str(item) for item in unexpected)}"
            )
        contract = {**core, "created_utc": utc_now_iso()}
        _write_new_bytes(contract_path, _json_bytes(contract))
    return EvaluationRun(
        root=root,
        method_name=authorization.method_name,
        contract_path=contract_path,
        contract_sha256=_hash_stable_file(contract_path, field="formal run contract"),
        contract=contract,
    )


def _extract_pair_prior(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    prior = payload.get("pair_prior")
    if not isinstance(prior, Mapping):
        raise MiO100EvaluationError("pair_prior artifact lacks pair_prior mapping")
    return prior


def _extract_global_priority(payload: Mapping[str, Any]) -> Mapping[str, float]:
    priority = payload.get("priority")
    if not isinstance(priority, Mapping) or set(priority) != set(SKILLS):
        raise MiO100EvaluationError("global_priority artifact lacks eight skills")
    values = {str(key): float(value) for key, value in priority.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise MiO100EvaluationError("global_priority contains non-finite values")
    return values


def _load_threshold_values(path: Path) -> tuple[float, ...]:
    payload = _mapping(load_json(path), "planner thresholds")
    if (
        payload.get("schema_version") != "graphrestore-presence-thresholds-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("frozen") is not True
        or payload.get("skills") != list(SKILLS)
    ):
        raise MiO100EvaluationError("planner threshold artifact drifted")
    raw = _mapping(payload.get("thresholds"), "planner threshold values")
    if set(raw) != set(SKILLS):
        raise MiO100EvaluationError("planner threshold skill set drifted")
    values = tuple(float(raw[name]) for name in SKILLS)
    if not all(math.isfinite(value) and 0.2 <= value <= 0.8 for value in values):
        raise MiO100EvaluationError("planner thresholds are invalid")
    return values


def build_formal_graphrestore(
    checkpoint: Stage4Checkpoint,
    *,
    config_path: str | Path,
    thresholds_path: str | Path,
    pair_prior_path: str | Path,
    global_priority_path: str | Path,
) -> GraphRestore:
    config_file = _canonical_regular_file(config_path, field="Stage4 config")
    config = _mapping(load_yaml(config_file), "Stage4 config")
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or config.get("stage") != "stage4"
        or _mapping(config.get("training"), "training").get("max_steps") != 40_000
        or _mapping(config.get("runtime"), "runtime").get("amp_dtype") != "bf16"
    ):
        raise MiO100EvaluationError(
            "Stage4 config is not the frozen 40000-step protocol"
        )
    pair_file = _canonical_regular_file(pair_prior_path, field="pair prior")
    priority_file = _canonical_regular_file(
        global_priority_path, field="global priority"
    )
    threshold_file = _canonical_regular_file(
        thresholds_path, field="planner thresholds"
    )
    pair_payload = _mapping(load_json(pair_file), "pair prior")
    priority_payload = _mapping(load_json(priority_file), "global priority")
    model = GraphRestore(
        gradient_checkpointing=True,
        pair_prior=_extract_pair_prior(pair_payload),
        global_priority=_extract_global_priority(priority_payload),
        max_active_skills=3,
        kmax_train=2,
        kmax_test=3,
        allow_skill_reentry=False,
        max_calls_per_skill=1,
    )
    incompatible = model.load_state_dict(dict(checkpoint.model_state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise MiO100EvaluationError(
            "strict Stage4 model loading returned incompatibilities"
        )
    thresholds = _load_threshold_values(threshold_file)
    checkpoint_thresholds = checkpoint.model_state.get("presence_thresholds")
    if checkpoint_thresholds is None or not torch.equal(
        checkpoint_thresholds.float(), torch.tensor(thresholds, dtype=torch.float32)
    ):
        raise MiO100EvaluationError(
            "frozen threshold artifact differs from selected Stage4 checkpoint"
        )
    model.set_presence_thresholds(thresholds)
    model.eval()
    return model


def configure_formal_runtime() -> None:
    allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator != "backend:native,expandable_segments:True":
        raise MiO100EvaluationError(
            "formal evaluation requires "
            "PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True"
        )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def assert_exclusive_gpu_process(
    *,
    expected_pid: int | None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Fail unless compute-process ownership is empty or exactly ``expected_pid``."""

    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MiO100EvaluationError(
            f"could not prove exclusive GPU ownership: {exc}"
        ) from exc
    if result.returncode != 0:
        raise MiO100EvaluationError(
            "could not prove exclusive GPU ownership: "
            f"nvidia-smi exit={result.returncode}, stderr={result.stderr.strip()!r}"
        )
    pids: set[int] = set()
    for raw in result.stdout.splitlines():
        value = raw.strip()
        if not value or "No running processes" in value:
            continue
        if not value.isdigit():
            raise MiO100EvaluationError(
                f"unexpected nvidia-smi compute PID row: {value!r}"
            )
        pids.add(int(value))
    expected = set() if expected_pid is None else {expected_pid}
    if pids != expected:
        raise MiO100EvaluationError(
            f"exclusive GPU process gate failed: expected={sorted(expected)}, "
            f"actual={sorted(pids)}"
        )


@torch.inference_mode()
def autonomous_graphrestore_inference(
    model: GraphRestore,
    image: Tensor,
    *,
    device: torch.device,
    use_bf16: bool = True,
) -> InferenceResult:
    """Run only image-conditioned planning; no task labels are accepted."""

    if device.type != "cuda":
        raise MiO100EvaluationError("formal GraphRestore inference requires CUDA")
    if tuple(image.shape[:2]) != (1, 3) or not image.is_floating_point():
        raise MiO100EvaluationError("formal input must be RGB float [1,3,H,W]")
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with context:
        output = model(
            image.to(device=device, dtype=torch.float32),
            return_trace=True,
            max_rounds=3,
        )
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1_000.0
    if not isinstance(output, GraphRestoreOutput):
        raise MiO100EvaluationError(
            "autonomous GraphRestore inference returned no trace"
        )
    program_levels = sum(
        int(trace.active_mask.any(dim=1).sum().item()) for trace in output.trace
    )
    parallel_levels = sum(
        int((trace.active_mask.sum(dim=1) > 1).sum().item()) for trace in output.trace
    )
    active_calls = sum(int(trace.active_mask.sum().item()) for trace in output.trace)
    diagnostics: dict[str, int | float] = {
        "program_levels": program_levels,
        "parallel_levels": parallel_levels,
        "active_skill_calls": active_calls,
        "reentry_requests": sum(
            int(trace.reentry_request_mask.sum().item()) for trace in output.trace
        ),
        "unexpected_activations": sum(
            int(trace.unexpected_activation_mask.sum().item()) for trace in output.trace
        ),
        "precycle_graphs": sum(
            bool(graph.dropped_edges) for graph in output.compiled_graphs
        ),
        "dropped_edges": sum(
            len(graph.dropped_edges) for graph in output.compiled_graphs
        ),
    }
    return InferenceResult(
        prediction=output.final.detach().float().cpu(),
        diagnostics=diagnostics,
        latency_ms=latency_ms,
    )


def _read_bgr_png(path: Path, *, field: str) -> tuple[np.ndarray, str]:
    canonical = _canonical_regular_file(path, field=field)
    before = sha256_file(canonical)
    image = cv2.imread(str(canonical), cv2.IMREAD_COLOR)
    after = sha256_file(canonical)
    if before != after:
        raise MiO100EvaluationError(f"{field} changed while reading: {canonical}")
    if (
        image is None
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
    ):
        raise MiO100EvaluationError(f"invalid uint8 BGR PNG for {field}: {canonical}")
    return image, before


def _encode_prediction_png(
    prediction: Tensor, *, target_shape: tuple[int, int]
) -> bytes:
    if prediction.ndim != 4 or tuple(prediction.shape[:2]) != (1, 3):
        raise MiO100EvaluationError("prediction must be RGB [1,3,H,W]")
    if not prediction.is_floating_point() or not bool(
        torch.isfinite(prediction).all().item()
    ):
        raise MiO100EvaluationError("prediction contains non-finite values")
    target_h, target_w = target_shape
    if prediction.shape[-2] < target_h or prediction.shape[-1] < target_w:
        raise MiO100EvaluationError(
            f"prediction is smaller than GT: {tuple(prediction.shape[-2:])} < {target_shape}"
        )
    cropped = prediction[0, :, :target_h, :target_w]
    rgb = (
        cropped.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise MiO100EvaluationError("OpenCV failed to encode lossless PNG")
    return bytes(encoded)


def _decode_png_bytes(payload: bytes) -> Tensor:
    array = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None or bgr.dtype != np.uint8:
        raise MiO100EvaluationError("could not read back committed PNG bytes")
    return bgr_uint8_to_rgb_float(bgr).unsqueeze(0)


def _output_path(run: EvaluationRun, record: MiO100Record) -> Path:
    return (
        run.root
        / "methods"
        / run.method_name
        / record.depth_dir
        / record.combination
        / record.output_filename
    )


def _bundle_path(run: EvaluationRun, record: MiO100Record) -> Path:
    return run.root / "records" / record.record_key


def _peak_reserved_fraction(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    total = int(torch.cuda.get_device_properties(device).total_memory)
    if total <= 0:
        raise MiO100EvaluationError("CUDA device reports invalid total memory")
    return float(torch.cuda.max_memory_reserved(device) / total)


def _receipt_from_bundle(
    run: EvaluationRun,
    record: MiO100Record,
    *,
    verify_metric: bool = True,
) -> Mapping[str, Any]:
    bundle = _bundle_path(run, record)
    if bundle.is_symlink() or not bundle.is_dir():
        raise MiO100EvaluationError(f"missing committed image bundle: {bundle}")
    prediction_file = bundle / "prediction.png"
    receipt_file = bundle / "receipt.json"
    if set(item.name for item in bundle.iterdir()) != {
        "prediction.png",
        "receipt.json",
    }:
        raise MiO100EvaluationError(
            f"image bundle contains unexpected entries: {bundle}"
        )
    prediction_file = _canonical_regular_file(
        prediction_file, field="bundle prediction"
    )
    receipt_file = _canonical_regular_file(receipt_file, field="image receipt")
    _require_read_only(prediction_file, field="bundle prediction")
    _require_read_only(receipt_file, field="image receipt")
    receipt = _mapping(load_json(receipt_file), "image receipt")
    if set(receipt) != _RECEIPT_KEYS:
        raise MiO100EvaluationError(
            f"image receipt fields drifted for {record.sample_id}"
        )
    _validate_utc(receipt.get("created_utc"), field="image receipt created_utc")
    expected_scalars: Mapping[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_sha256": run.contract_sha256,
        "index": record.index,
        "sample_id": record.sample_id,
        "row_sha256": record.row_sha256,
        "clean_id": record.clean_id,
        "group": record.group,
        "combination": record.combination,
        "native_lq_path": str(record.native_lq_path),
        "target_png": str(record.target_path),
        "prediction_png": str(_output_path(run, record)),
    }
    for key, value in expected_scalars.items():
        if receipt.get(key) != value:
            raise MiO100EvaluationError(
                f"image receipt drifted for {record.sample_id} at {key}"
            )
    for key in ("native_lq_sha256", "target_sha256", "prediction_sha256"):
        if not is_sha256(receipt.get(key)):
            raise MiO100EvaluationError(f"image receipt has malformed {key}")
    if (
        receipt["native_lq_sha256"] != record.expected_native_sha256
        or receipt["target_sha256"] != record.expected_target_sha256
    ):
        raise MiO100EvaluationError("image receipt differs from formal data inventory")
    for key in ("psnr", "ssim", "latency_ms", "peak_reserved_fraction"):
        value = receipt.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise MiO100EvaluationError(f"image receipt has invalid {key}")
    if float(receipt["latency_ms"]) < 0.0:
        raise MiO100EvaluationError("image receipt has negative latency")
    if not 0.0 <= float(receipt["peak_reserved_fraction"]) < 0.90:
        raise MiO100EvaluationError("image receipt violates the VRAM ceiling")
    diagnostics = _mapping(receipt.get("diagnostics"), "image diagnostics")
    if set(diagnostics) != _DIAGNOSTIC_KEYS:
        raise MiO100EvaluationError("image diagnostic fields drifted")
    for key, value in diagnostics.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MiO100EvaluationError(f"invalid image diagnostic {key}")
    if sha256_file(prediction_file) != receipt["prediction_sha256"]:
        raise MiO100EvaluationError("committed prediction hash drifted")
    native_sha = _verify_record_file_identity(record, role="native_lq")
    target_sha = _verify_record_file_identity(record, role="target")
    if native_sha != receipt["native_lq_sha256"]:
        raise MiO100EvaluationError("native LQ changed after image commit")
    if target_sha != receipt["target_sha256"]:
        raise MiO100EvaluationError("GT changed after image commit")
    output = _output_path(run, record)
    if output.is_symlink():
        raise MiO100EvaluationError(f"AgenticIR output must not be a symlink: {output}")
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise MiO100EvaluationError(f"invalid AgenticIR output path: {output}")
        if sha256_file(output) != receipt["prediction_sha256"]:
            raise MiO100EvaluationError(f"AgenticIR output hash drifted: {output}")
    else:
        for ancestor in output.parents:
            if ancestor.is_symlink():
                raise MiO100EvaluationError(
                    f"AgenticIR output path crosses symlink: {ancestor}"
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(prediction_file, output)
        except FileExistsError as exc:
            raise MiO100EvaluationError(f"output publication race: {output}") from exc
        fsync_directory(output.parent)
    output_stat = output.stat()
    bundle_stat = prediction_file.stat()
    if (output_stat.st_dev, output_stat.st_ino) != (
        bundle_stat.st_dev,
        bundle_stat.st_ino,
    ):
        raise MiO100EvaluationError(
            f"AgenticIR output is not the committed prediction hard link: {output}"
        )
    _require_read_only(output, field="AgenticIR prediction output")
    if verify_metric:
        prediction_bgr, prediction_sha = _read_bgr_png(output, field="prediction")
        target_bgr, target_sha = _read_bgr_png(record.target_path, field="GT")
        if (
            prediction_sha != receipt["prediction_sha256"]
            or target_sha != receipt["target_sha256"]
        ):
            raise MiO100EvaluationError("readback hashes disagree with receipt")
        prediction = bgr_uint8_to_rgb_float(prediction_bgr).unsqueeze(0)
        target = bgr_uint8_to_rgb_float(target_bgr).unsqueeze(0)
        metric = official_psnr_ssim(prediction, target, quantize=True)
        for key, actual in (
            ("psnr", float(metric.psnr.item())),
            ("ssim", float(metric.ssim.item())),
        ):
            recorded = receipt.get(key)
            if not isinstance(recorded, (int, float)) or not math.isclose(
                float(recorded), actual, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise MiO100EvaluationError(
                    f"readback metric drifted for {record.sample_id}: {key}"
                )
    return receipt


def _commit_image_bundle(
    run: EvaluationRun,
    record: MiO100Record,
    *,
    png_payload: bytes,
    native_sha256: str,
    target_sha256: str,
    psnr: float,
    ssim: float,
    inference: InferenceResult,
    peak_reserved_fraction: float,
) -> Mapping[str, Any]:
    final_bundle = _bundle_path(run, record)
    pending = final_bundle.parent / f".pending-{record.record_key}"
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    if pending.exists() or pending.is_symlink():
        raise MiO100EvaluationError(
            f"incomplete prior image transaction requires audit: {pending}"
        )
    if final_bundle.exists() or final_bundle.is_symlink():
        return _receipt_from_bundle(run, record)
    pending.mkdir(mode=0o700)
    try:
        prediction_path = pending / "prediction.png"
        _write_new_bytes(prediction_path, png_payload)
        prediction_sha = sha256_file(prediction_path)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "contract_sha256": run.contract_sha256,
            "created_utc": utc_now_iso(),
            "index": record.index,
            "sample_id": record.sample_id,
            "row_sha256": record.row_sha256,
            "clean_id": record.clean_id,
            "group": record.group,
            "combination": record.combination,
            "native_lq_path": str(record.native_lq_path),
            "native_lq_sha256": native_sha256,
            "target_png": str(record.target_path),
            "target_sha256": target_sha256,
            "prediction_png": str(_output_path(run, record)),
            "prediction_sha256": prediction_sha,
            "psnr": psnr,
            "ssim": ssim,
            "latency_ms": inference.latency_ms,
            "peak_reserved_fraction": peak_reserved_fraction,
            "diagnostics": dict(inference.diagnostics),
        }
        _write_new_bytes(pending / "receipt.json", _json_bytes(receipt))
        os.chmod(pending, 0o555)
        fsync_directory(pending)
        try:
            os.rename(pending, final_bundle)
        except FileExistsError as exc:
            raise MiO100EvaluationError(
                f"image bundle publication race: {final_bundle}"
            ) from exc
        fsync_directory(final_bundle.parent)
    except BaseException:
        # Deliberately retain a non-empty pending transaction for audit.  An
        # empty directory is safe to remove because no prediction was created.
        if pending.is_dir() and not any(pending.iterdir()):
            pending.rmdir()
            fsync_directory(pending.parent)
        raise
    return _receipt_from_bundle(run, record)


def process_record(
    run: EvaluationRun,
    record: MiO100Record,
    *,
    infer: Callable[[Tensor], InferenceResult],
    device: torch.device,
    input_loader: Callable[[Mapping[str, Any]], Tensor] = (
        load_agenticir_online_canonical_input
    ),
) -> Mapping[str, Any]:
    bundle = _bundle_path(run, record)
    output = _output_path(run, record)
    pending = bundle.parent / f".pending-{record.record_key}"
    for candidate in (bundle, output, pending):
        for ancestor in (candidate, *candidate.parents):
            if ancestor.is_symlink():
                raise MiO100EvaluationError(
                    f"formal image path crosses a symlink: {ancestor}"
                )
    if bundle.exists() or bundle.is_symlink():
        return _receipt_from_bundle(run, record)
    if pending.exists() or pending.is_symlink():
        raise MiO100EvaluationError(
            f"incomplete prior image transaction requires audit: {pending}"
        )
    if output.exists() or output.is_symlink():
        raise MiO100EvaluationError(
            "unreceipted output exists; refusing result-driven rerun or overwrite: "
            f"{output}"
        )
    native_before = _verify_record_file_identity(record, role="native_lq")
    target_before = _verify_record_file_identity(record, role="target")
    target_bgr, target_sha = _read_bgr_png(record.target_path, field="GT")
    target_after = _verify_record_file_identity(record, role="target")
    if target_before != target_sha or target_after != target_sha:
        raise MiO100EvaluationError("GT changed during formal decode")
    image = input_loader(record.row)
    native_after = _verify_record_file_identity(record, role="native_lq")
    if native_before != native_after:
        raise MiO100EvaluationError("native LQ changed during canonical input loading")
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise MiO100EvaluationError("canonical input loader returned invalid RGB CHW")
    if not bool(torch.isfinite(image).all().item()):
        raise MiO100EvaluationError("canonical model input contains non-finite values")
    inference = infer(image.unsqueeze(0))
    if not math.isfinite(inference.latency_ms) or inference.latency_ms < 0.0:
        raise MiO100EvaluationError("inference latency is invalid")
    peak = _peak_reserved_fraction(device)
    if not math.isfinite(peak) or peak >= MAXIMUM_VRAM_RESERVED_FRACTION:
        raise MiO100EvaluationError(
            f"formal inference peak VRAM fraction {peak:.6f} is not below 0.90"
        )
    png_payload = _encode_prediction_png(
        inference.prediction,
        target_shape=(int(target_bgr.shape[0]), int(target_bgr.shape[1])),
    )
    prediction = _decode_png_bytes(png_payload)
    target = bgr_uint8_to_rgb_float(target_bgr).unsqueeze(0)
    if prediction.shape != target.shape:
        raise MiO100EvaluationError("PNG readback shape differs from GT")
    metric = official_psnr_ssim(prediction, target, quantize=True)
    psnr = float(metric.psnr.item())
    ssim = float(metric.ssim.item())
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        raise MiO100EvaluationError("formal PNG-readback metric is non-finite")
    return _commit_image_bundle(
        run,
        record,
        png_payload=png_payload,
        native_sha256=native_before,
        target_sha256=target_sha,
        psnr=psnr,
        ssim=ssim,
        inference=inference,
        peak_reserved_fraction=peak,
    )


def _shard_path(run: EvaluationRun, shard_index: int, shard_count: int) -> Path:
    return run.root / "shards" / f"shard-{shard_index:04d}-of-{shard_count:04d}.json"


def _validate_shard_complete(
    run: EvaluationRun,
    records: Sequence[MiO100Record],
    *,
    shard_index: int,
    shard_count: int,
) -> Mapping[str, Any]:
    path = _canonical_regular_file(
        _shard_path(run, shard_index, shard_count), field="shard completion"
    )
    _require_read_only(path, field="shard completion")
    payload = _mapping(load_json(path), "shard completion")
    expected_indices = [
        record.index for record in records if record.index % shard_count == shard_index
    ]
    expected: Mapping[str, object] = {
        "schema_version": SHARD_SCHEMA,
        "contract_sha256": run.contract_sha256,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "sample_indices": expected_indices,
        "image_count": len(expected_indices),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise MiO100EvaluationError(f"shard completion drifted at {key}")
    _validate_utc(payload.get("created_utc"), field="shard completion created_utc")
    receipt_bindings = payload.get("receipt_sha256")
    if not isinstance(receipt_bindings, list) or len(receipt_bindings) != len(
        expected_indices
    ):
        raise MiO100EvaluationError("shard receipt bindings are malformed")
    for index, binding in zip(expected_indices, receipt_bindings, strict=True):
        record = records[index]
        receipt_path = _bundle_path(run, record) / "receipt.json"
        if binding != {
            "index": index,
            "sha256": sha256_file(receipt_path),
        }:
            raise MiO100EvaluationError("shard receipt hash binding drifted")
    return payload


def run_shard(
    run: EvaluationRun,
    records: Sequence[MiO100Record],
    *,
    shard_index: int,
    shard_count: int,
    infer: Callable[[Tensor], InferenceResult],
    device: torch.device,
    input_loader: Callable[[Mapping[str, Any]], Tensor] = (
        load_agenticir_online_canonical_input
    ),
) -> Mapping[str, Any]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise MiO100EvaluationError("invalid shard index/count")
    if shard_count != int(run.contract["shard_count"]):
        raise MiO100EvaluationError("runtime shard_count differs from run contract")
    shard_path = _shard_path(run, shard_index, shard_count)
    if shard_path.exists():
        return _validate_shard_complete(
            run, records, shard_index=shard_index, shard_count=shard_count
        )
    selected = [
        record for record in records if record.index % shard_count == shard_index
    ]
    for record in selected:
        process_record(
            run,
            record,
            infer=infer,
            device=device,
            input_loader=input_loader,
        )
    receipt_sha = [
        {
            "index": record.index,
            "sha256": sha256_file(_bundle_path(run, record) / "receipt.json"),
        }
        for record in selected
    ]
    payload = {
        "schema_version": SHARD_SCHEMA,
        "contract_sha256": run.contract_sha256,
        "created_utc": utc_now_iso(),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "sample_indices": [record.index for record in selected],
        "image_count": len(selected),
        "receipt_sha256": receipt_sha,
    }
    _write_new_bytes(shard_path, _json_bytes(payload))
    return _validate_shard_complete(
        run, records, shard_index=shard_index, shard_count=shard_count
    )


_CSV_COLUMNS = (
    "sample_id",
    "group",
    "combination",
    "clean_id",
    "prediction_png",
    "prediction_sha256",
    "target_png",
    "target_sha256",
    "psnr",
    "ssim",
    "latency_ms",
    "program_levels",
    "parallel_levels",
    "active_skill_calls",
    "reentry_requests",
    "unexpected_activations",
    "precycle_graphs",
    "dropped_edges",
    "peak_reserved_fraction",
)


def _csv_bytes(receipts: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for receipt in receipts:
        diagnostics = _mapping(receipt.get("diagnostics"), "receipt diagnostics")
        row = {
            key: receipt[key]
            for key in (
                "sample_id",
                "group",
                "combination",
                "clean_id",
                "prediction_png",
                "prediction_sha256",
                "target_png",
                "target_sha256",
                "psnr",
                "ssim",
                "latency_ms",
                "peak_reserved_fraction",
            )
        }
        for key in (
            "program_levels",
            "parallel_levels",
            "active_skill_calls",
            "reentry_requests",
            "unexpected_activations",
            "precycle_graphs",
            "dropped_edges",
        ):
            row[key] = diagnostics.get(key, 0)
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _table1_jsonl_bytes(receipts: Sequence[Mapping[str, Any]]) -> bytes:
    official_order = {
        combination: index
        for index, combination in enumerate(
            combination
            for combinations in OFFICIAL_GROUPS.values()
            for combination in combinations
        )
    }
    ordered = sorted(
        receipts,
        key=lambda receipt: (
            official_order[str(receipt["combination"])],
            str(receipt["sample_id"]),
        ),
    )
    rows = []
    for receipt in ordered:
        row = {
            "schema_version": TABLE1_INPUT_SCHEMA,
            "sample_id": receipt["sample_id"],
            "group": receipt["group"],
            "combination": receipt["combination"],
            "prediction_png": receipt["prediction_png"],
            "prediction_sha256": receipt["prediction_sha256"],
            "target_png": receipt["target_png"],
            "target_sha256": receipt["target_sha256"],
        }
        rows.append(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _aggregate_runtime(receipts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    diagnostics_keys = (
        "program_levels",
        "parallel_levels",
        "active_skill_calls",
        "reentry_requests",
        "unexpected_activations",
        "precycle_graphs",
        "dropped_edges",
    )
    totals = {key: 0 for key in diagnostics_keys}
    for receipt in receipts:
        diagnostics = _mapping(receipt.get("diagnostics"), "receipt diagnostics")
        for key in diagnostics_keys:
            value = diagnostics.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MiO100EvaluationError(f"invalid runtime diagnostic {key}")
            totals[key] += value
    image_count = len(receipts)
    latency = [float(receipt["latency_ms"]) for receipt in receipts]
    peak = max(float(receipt["peak_reserved_fraction"]) for receipt in receipts)
    if peak >= MAXIMUM_VRAM_RESERVED_FRACTION:
        raise MiO100EvaluationError("aggregate formal VRAM peak is not below 0.90")
    return {
        "total_latency_ms": math.fsum(latency),
        "mean_latency_ms": math.fsum(latency) / image_count,
        "peak_reserved_fraction": peak,
        "mean_program_levels": float(totals["program_levels"]) / image_count,
        "parallel_level_fraction": float(totals["parallel_levels"])
        / max(float(totals["program_levels"]), 1.0),
        **totals,
    }


def finalize_evaluation(
    run: EvaluationRun,
    records: Sequence[MiO100Record],
    *,
    authorization: FormalAuthorization,
) -> Mapping[str, Any]:
    shard_count = int(run.contract["shard_count"])
    for shard_index in range(shard_count):
        _validate_shard_complete(
            run,
            records,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    receipts = [
        _receipt_from_bundle(run, record, verify_metric=True) for record in records
    ]
    metric_rows = [
        {
            "combination": receipt["combination"],
            "psnr": receipt["psnr"],
            "ssim": receipt["ssim"],
        }
        for receipt in receipts
    ]
    aggregate = aggregate_official_records(
        metric_rows,
        required_combinations=tuple(
            combination
            for combinations in OFFICIAL_GROUPS.values()
            for combination in combinations
        ),
        expected_counts=FORMAL_COMBINATION_COUNTS,
    )
    if aggregate["image_count"] != FORMAL_ROW_COUNT:
        raise MiO100EvaluationError("formal aggregation did not cover 1440 images")
    for output in (_output_path(run, record) for record in records):
        _require_read_only(output, field="formal prediction PNG")
    for directory in {_output_path(run, record).parent for record in records}:
        fsync_directory(directory)

    table1_path = run.root / "table1_input.jsonl"
    csv_path = run.root / "per_image.csv"
    summary_path = run.root / "summary.json"
    complete_path = run.root / "complete.json"
    _write_or_verify_immutable(table1_path, _table1_jsonl_bytes(receipts))
    _write_or_verify_immutable(csv_path, _csv_bytes(receipts))
    predictions_digest = sha256_json(
        [
            {
                "sample_id": receipt["sample_id"],
                "prediction_sha256": receipt["prediction_sha256"],
                "target_sha256": receipt["target_sha256"],
            }
            for receipt in receipts
        ]
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": run.contract["created_utc"],
        "method_name": run.method_name,
        "image_count": FORMAL_ROW_COUNT,
        "manifest_sha256": run.contract["manifest_sha256"],
        "formal_data_inventory": dict(run.contract["formal_data_inventory"]),
        "checkpoint_sha256": run.contract["checkpoint_sha256"],
        "authorization_sha256": authorization.sha256,
        "run_contract_sha256": run.contract_sha256,
        "predictions_digest": predictions_digest,
        "aggregation": aggregate,
        "runtime": _aggregate_runtime(receipts),
        "metric_protocol": {
            "prediction_source": "lossless_png_readback",
            "psnr": "AgenticIR/pyiqa-0.1.10 RGB parity",
            "ssim": "AgenticIR/pyiqa-0.1.10 Y parity",
            "group_reduction": "equal_combination_mean",
            "weighted_all_images": "additional_only",
        },
        "outputs": {
            "agenticir_methods_root": str(run.root / "methods" / run.method_name),
            "per_image_csv": str(csv_path),
            "table1_input_jsonl": str(table1_path),
        },
    }
    _write_or_verify_immutable(summary_path, _json_bytes(summary))
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": run.contract["created_utc"],
        "status": "COMPLETE",
        "image_count": FORMAL_ROW_COUNT,
        "method_name": run.method_name,
        "authorization_sha256": authorization.sha256,
        "run_contract_sha256": run.contract_sha256,
        "checkpoint_sha256": run.contract["checkpoint_sha256"],
        "manifest_sha256": run.contract["manifest_sha256"],
        "formal_data_inventory": dict(run.contract["formal_data_inventory"]),
        "predictions_digest": predictions_digest,
        "bindings": {
            "run_contract": {
                "path": str(run.contract_path),
                "sha256": run.contract_sha256,
            },
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "per_image_csv": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
            "table1_input_jsonl": {
                "path": str(table1_path),
                "sha256": sha256_file(table1_path),
            },
        },
    }
    _write_or_verify_immutable(complete_path, _json_bytes(complete))
    return summary


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MiO100EvaluationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise MiO100EvaluationError(f"non-finite JSON constant is forbidden: {value}")


def _load_strict_json(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MiO100EvaluationError(f"could not read strict {field}: {exc}") from exc
    return _mapping(value, field)


def _load_strict_jsonl(path: Path, *, field: str) -> tuple[Mapping[str, Any], ...]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise MiO100EvaluationError(
                        f"{field}:{line_number}: blank row is forbidden"
                    )
                try:
                    value = json.loads(
                        raw,
                        object_pairs_hook=_strict_json_object,
                        parse_constant=_reject_json_constant,
                    )
                except json.JSONDecodeError as exc:
                    raise MiO100EvaluationError(
                        f"{field}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                rows.append(_mapping(value, f"{field}:{line_number}"))
    except (OSError, UnicodeError) as exc:
        raise MiO100EvaluationError(f"could not read strict {field}: {exc}") from exc
    return tuple(rows)


def _complete_artifact_binding(
    raw: object,
    *,
    field: str,
    expected_path: Path,
) -> ArtifactBinding:
    binding = _mapping(raw, field)
    if set(binding) != {"path", "sha256"}:
        raise MiO100EvaluationError(f"{field} must contain only path/sha256")
    path_value = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(path_value, str) or not is_sha256(digest):
        raise MiO100EvaluationError(f"{field} is malformed")
    path = _canonical_regular_file(path_value, field=field)
    if path != expected_path:
        raise MiO100EvaluationError(f"{field} path drifted: {path} != {expected_path}")
    _require_read_only(path, field=field)
    actual = _hash_stable_file(path, field=field)
    if actual != digest:
        raise MiO100EvaluationError(f"{field} SHA256 drifted")
    return ArtifactBinding(path=path, sha256=actual)


def _finite_float(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise MiO100EvaluationError(f"{field} is not numeric")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise MiO100EvaluationError(f"{field} is not numeric") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise MiO100EvaluationError(f"{field} is outside its finite domain")
    return parsed


def _strict_nonnegative_int(value: str, *, field: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise MiO100EvaluationError(f"{field} is not a canonical nonnegative integer")
    return int(value)


def validate_formal_evaluator_complete(
    complete_path: str | Path = FORMAL_OUTPUT_ROOT / "complete.json",
    *,
    authorization_path: str | Path,
    expected_bindings: Mapping[str, str | Path] | None = None,
    expected_output_root: str | Path = FORMAL_OUTPUT_ROOT,
    expected_method_name: str = FORMAL_METHOD_NAME,
    expected_row_count: int | None = None,
    expected_group_counts: Mapping[str, int] | None = None,
    expected_combination_counts: Mapping[str, int] | None = None,
    inventory_validation_kwargs: Mapping[str, Any] | None = None,
    verify_data_files: bool = True,
) -> FormalEvaluatorCompletion:
    """Validate the immutable evaluator terminal chain without CUDA inference.

    The returned ``evidence`` mapping is stable and JSON-serializable.  It is
    intended as the only production hand-off to the independent Table-1
    scorer; no path or hash supplied by a worker request is trusted.
    """

    row_count = FORMAL_ROW_COUNT if expected_row_count is None else expected_row_count
    group_counts = dict(expected_group_counts or FORMAL_GROUP_COUNTS)
    combination_counts = dict(expected_combination_counts or FORMAL_COMBINATION_COUNTS)
    root = Path(expected_output_root).resolve(strict=False)
    authorization = validate_formal_authorization(
        authorization_path,
        expected_bindings=expected_bindings,
        expected_output_root=root,
        expected_method_name=expected_method_name,
        expected_shard_count=1,
    )
    validate_protocol_bindings(authorization)
    validate_stage4_completion(
        authorization.bindings["stage4_complete"].path,
        checkpoint_sha256=authorization.bindings["stage4_checkpoint"].sha256,
        authorization=authorization,
    )
    expected_complete = root / "complete.json"
    complete_file = _canonical_regular_file(complete_path, field="evaluator complete")
    if complete_file != expected_complete:
        raise MiO100EvaluationError("formal evaluator complete path drifted")
    _require_read_only(complete_file, field="evaluator complete")
    complete_sha = _hash_stable_file(complete_file, field="evaluator complete")
    complete = _load_strict_json(complete_file, field="evaluator complete")
    complete_keys = {
        "schema_version",
        "protocol_id",
        "created_utc",
        "status",
        "image_count",
        "method_name",
        "authorization_sha256",
        "run_contract_sha256",
        "checkpoint_sha256",
        "manifest_sha256",
        "formal_data_inventory",
        "predictions_digest",
        "bindings",
    }
    if set(complete) != complete_keys:
        raise MiO100EvaluationError("formal evaluator complete fields drifted")
    _validate_utc(complete.get("created_utc"), field="evaluator complete created_utc")
    expected_complete_scalars = {
        "schema_version": COMPLETE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "image_count": row_count,
        "method_name": expected_method_name,
        "authorization_sha256": authorization.sha256,
        "checkpoint_sha256": authorization.bindings["stage4_checkpoint"].sha256,
        "manifest_sha256": authorization.bindings["formal_manifest"].sha256,
    }
    if any(
        complete.get(key) != value for key, value in expected_complete_scalars.items()
    ):
        raise MiO100EvaluationError("formal evaluator complete scope drifted")
    if not is_sha256(complete.get("run_contract_sha256")) or not is_sha256(
        complete.get("predictions_digest")
    ):
        raise MiO100EvaluationError("formal evaluator complete hashes are malformed")

    inventory_raw = _mapping(
        complete.get("formal_data_inventory"), "evaluator complete data inventory"
    )
    if set(inventory_raw) != {"path", "sha256", "rows_digest", "files_digest"}:
        raise MiO100EvaluationError("evaluator complete data inventory fields drifted")
    inventory_binding = authorization.bindings["formal_data_inventory"]
    if (
        inventory_raw.get("path") != str(inventory_binding.path)
        or inventory_raw.get("sha256") != inventory_binding.sha256
        or not is_sha256(inventory_raw.get("rows_digest"))
        or not is_sha256(inventory_raw.get("files_digest"))
    ):
        raise MiO100EvaluationError("evaluator complete data inventory binding drifted")
    inventory_kwargs = dict(inventory_validation_kwargs or {})
    inventory_kwargs.setdefault("expected_row_count", row_count)
    inventory_kwargs.setdefault("expected_group_counts", group_counts)
    inventory_kwargs.setdefault("expected_combination_counts", combination_counts)
    try:
        inventory = load_strict_formal_data_inventory(
            inventory_binding.path,
            expected_manifest_path=authorization.bindings["formal_manifest"].path,
            expected_manifest_sha256=authorization.bindings["formal_manifest"].sha256,
            expected_authorization_protocol_path=authorization.bindings[
                "formal_authorization_protocol"
            ].path,
            expected_authorization_protocol_sha256=authorization.bindings[
                "formal_authorization_protocol"
            ].sha256,
            verify_file_bytes=verify_data_files,
            **inventory_kwargs,
        )
    except FormalInventoryError as exc:
        raise MiO100EvaluationError(
            f"formal evaluator data inventory rejected: {exc}"
        ) from exc
    if (
        inventory.sha256 != inventory_binding.sha256
        or inventory.rows_digest != inventory_raw["rows_digest"]
        or inventory.files_digest != inventory_raw["files_digest"]
    ):
        raise MiO100EvaluationError("formal evaluator inventory digest drifted")

    raw_bindings = _mapping(complete.get("bindings"), "evaluator complete bindings")
    if set(raw_bindings) != {
        "run_contract",
        "summary",
        "per_image_csv",
        "table1_input_jsonl",
    }:
        raise MiO100EvaluationError("formal evaluator complete binding keys drifted")
    run_contract_binding = _complete_artifact_binding(
        raw_bindings["run_contract"],
        field="evaluator run contract",
        expected_path=root / "run_contract.json",
    )
    summary_binding = _complete_artifact_binding(
        raw_bindings["summary"],
        field="evaluator summary",
        expected_path=root / "summary.json",
    )
    per_image_binding = _complete_artifact_binding(
        raw_bindings["per_image_csv"],
        field="evaluator per-image CSV",
        expected_path=root / "per_image.csv",
    )
    table1_binding = _complete_artifact_binding(
        raw_bindings["table1_input_jsonl"],
        field="evaluator Table-1 input",
        expected_path=root / "table1_input.jsonl",
    )
    if run_contract_binding.sha256 != complete["run_contract_sha256"]:
        raise MiO100EvaluationError("evaluator complete/run-contract SHA drifted")

    contract = _load_strict_json(
        run_contract_binding.path, field="evaluator run contract"
    )
    contract_keys = {
        "schema_version",
        "protocol_id",
        "authorization",
        "authorization_bindings",
        "manifest_sha256",
        "formal_data_inventory",
        "checkpoint_sha256",
        "config_sha256",
        "method_name",
        "output_root",
        "manifest_row_count",
        "groups",
        "combination_counts",
        "shard_count",
        "assignment",
        "inference",
        "output_protocol",
        "vram_maximum_peak_reserved_fraction",
        "created_utc",
    }
    if set(contract) != contract_keys:
        raise MiO100EvaluationError("formal evaluator run-contract fields drifted")
    expected_authorization_bindings = {
        name: {"path": str(binding.path), "sha256": binding.sha256}
        for name, binding in sorted(authorization.bindings.items())
    }
    expected_contract = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "authorization": {
            "path": str(authorization.path),
            "sha256": authorization.sha256,
        },
        "authorization_bindings": expected_authorization_bindings,
        "manifest_sha256": authorization.bindings["formal_manifest"].sha256,
        "formal_data_inventory": dict(inventory_raw),
        "checkpoint_sha256": authorization.bindings["stage4_checkpoint"].sha256,
        "config_sha256": authorization.bindings["stage4_config"].sha256,
        "method_name": expected_method_name,
        "output_root": str(root),
        "manifest_row_count": row_count,
        "groups": group_counts,
        "combination_counts": combination_counts,
        "shard_count": 1,
        "assignment": "manifest_index_mod_shard_count",
        "inference": {
            "autonomous_graphrestore": True,
            "task_label_routing": False,
            "max_rounds": 3,
            "amp_dtype": "bf16",
            "tf32": True,
            "tta": False,
            "model_soup": False,
        },
        "output_protocol": {
            "crop": "top_left_to_gt_shape",
            "quantization": "clamp_round_uint8",
            "encoding": "lossless_png",
            "score_source": "png_readback",
            "layout": "methods/<method>/d2|d3/<combination>/<gt_basename>",
            "overwrite": False,
        },
        "vram_maximum_peak_reserved_fraction": MAXIMUM_VRAM_RESERVED_FRACTION,
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise MiO100EvaluationError(f"formal run contract drifted at {key}")
    if contract.get("created_utc") != complete.get("created_utc"):
        raise MiO100EvaluationError("formal run/complete UTC binding drifted")
    _validate_utc(contract.get("created_utc"), field="run contract created_utc")

    records = load_formal_manifest(
        authorization.bindings["formal_manifest"].path,
        expected_sha256=authorization.bindings["formal_manifest"].sha256,
        expected_group_counts=group_counts,
        expected_combination_counts=combination_counts,
    )
    if len(records) != row_count or len(inventory.rows) != row_count:
        raise MiO100EvaluationError("formal completion row count drifted")
    for record, identity in zip(records, inventory.rows, strict=True):
        if (
            record.index != identity.index
            or record.sample_id != identity.sample_id
            or record.row_sha256 != identity.row_sha256
            or record.native_lq_path != identity.native_lq_path
            or record.target_path != identity.target_path
        ):
            raise MiO100EvaluationError("formal completion manifest/inventory drifted")

    csv_rows = []
    try:
        with per_image_binding.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _CSV_COLUMNS:
                raise MiO100EvaluationError("formal per-image CSV header drifted")
            csv_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MiO100EvaluationError(
            f"could not read formal per-image CSV: {exc}"
        ) from exc
    if len(csv_rows) != row_count:
        raise MiO100EvaluationError("formal per-image CSV count drifted")
    sample_ids: set[str] = set()
    prediction_paths: set[Path] = set()
    metric_rows = []
    runtime_receipts = []
    prediction_digest_rows = []
    per_image_by_sample: dict[str, Mapping[str, str]] = {}
    actual_groups: dict[str, int] = {}
    actual_combinations: dict[str, int] = {}
    diagnostic_names = _CSV_COLUMNS[11:18]
    for index, (row, record, identity) in enumerate(
        zip(csv_rows, records, inventory.rows, strict=True)
    ):
        if set(row) != set(_CSV_COLUMNS):
            raise MiO100EvaluationError("formal per-image CSV row fields drifted")
        sample_id = row["sample_id"]
        if sample_id != record.sample_id or sample_id in sample_ids:
            raise MiO100EvaluationError(
                "formal per-image sample order/uniqueness drifted"
            )
        if (
            row["group"] != record.group
            or row["combination"] != record.combination
            or row["clean_id"] != record.clean_id
            or row["target_png"] != str(record.target_path)
            or row["target_sha256"] != identity.target_sha256
        ):
            raise MiO100EvaluationError(
                f"formal per-image row {index} metadata drifted"
            )
        prediction_path = _canonical_regular_file(
            row["prediction_png"], field="formal prediction PNG"
        )
        expected_prediction = (
            root
            / "methods"
            / expected_method_name
            / record.depth_dir
            / record.combination
            / record.output_filename
        )
        if (
            prediction_path != expected_prediction
            or prediction_path in prediction_paths
        ):
            raise MiO100EvaluationError("formal prediction path/uniqueness drifted")
        if stat.S_IMODE(prediction_path.stat().st_mode) != 0o444:
            raise MiO100EvaluationError("formal prediction PNG must have mode 0444")
        if not is_sha256(row["prediction_sha256"]) or (
            _hash_stable_file(prediction_path, field="formal prediction PNG")
            != row["prediction_sha256"]
        ):
            raise MiO100EvaluationError("formal prediction SHA256 drifted")
        if not is_sha256(row["target_sha256"]):
            raise MiO100EvaluationError("formal target SHA256 is malformed")
        psnr = _finite_float(row["psnr"], field="per-image PSNR")
        ssim = _finite_float(row["ssim"], field="per-image SSIM")
        latency = _finite_float(
            row["latency_ms"], field="per-image latency", minimum=0.0
        )
        peak = _finite_float(
            row["peak_reserved_fraction"], field="per-image VRAM", minimum=0.0
        )
        if peak >= MAXIMUM_VRAM_RESERVED_FRACTION:
            raise MiO100EvaluationError("formal per-image VRAM ceiling drifted")
        diagnostics = {
            name: _strict_nonnegative_int(row[name], field=f"per-image {name}")
            for name in diagnostic_names
        }
        sample_ids.add(sample_id)
        prediction_paths.add(prediction_path)
        actual_groups[record.group] = actual_groups.get(record.group, 0) + 1
        actual_combinations[record.combination] = (
            actual_combinations.get(record.combination, 0) + 1
        )
        metric_rows.append(
            {"combination": record.combination, "psnr": psnr, "ssim": ssim}
        )
        runtime_receipts.append(
            {
                "latency_ms": latency,
                "peak_reserved_fraction": peak,
                "diagnostics": diagnostics,
            }
        )
        prediction_digest_rows.append(
            {
                "sample_id": sample_id,
                "prediction_sha256": row["prediction_sha256"],
                "target_sha256": row["target_sha256"],
            }
        )
        per_image_by_sample[sample_id] = row
    if actual_groups != group_counts or actual_combinations != combination_counts:
        raise MiO100EvaluationError("formal per-image group/combination counts drifted")
    predictions_digest = sha256_json(prediction_digest_rows)
    if predictions_digest != complete["predictions_digest"]:
        raise MiO100EvaluationError("formal predictions digest drifted")

    table_rows = _load_strict_jsonl(table1_binding.path, field="formal Table-1 input")
    if len(table_rows) != row_count:
        raise MiO100EvaluationError("formal Table-1 input count drifted")
    official_order = {
        combination: index
        for index, combination in enumerate(
            combination
            for combinations in OFFICIAL_GROUPS.values()
            for combination in combinations
        )
    }
    expected_table_records = sorted(
        records,
        key=lambda record: (official_order[record.combination], record.sample_id),
    )
    table_keys = {
        "schema_version",
        "sample_id",
        "group",
        "combination",
        "prediction_png",
        "prediction_sha256",
        "target_png",
        "target_sha256",
    }
    for row, record in zip(table_rows, expected_table_records, strict=True):
        csv_row = per_image_by_sample.get(record.sample_id)
        expected_table = {
            "schema_version": TABLE1_INPUT_SCHEMA,
            "sample_id": record.sample_id,
            "group": record.group,
            "combination": record.combination,
            "prediction_png": csv_row["prediction_png"] if csv_row else None,
            "prediction_sha256": csv_row["prediction_sha256"] if csv_row else None,
            "target_png": csv_row["target_png"] if csv_row else None,
            "target_sha256": csv_row["target_sha256"] if csv_row else None,
        }
        if set(row) != table_keys or dict(row) != expected_table:
            raise MiO100EvaluationError("formal Table-1/per-image binding drifted")

    summary = _load_strict_json(summary_binding.path, field="evaluator summary")
    summary_keys = {
        "schema_version",
        "protocol_id",
        "created_utc",
        "method_name",
        "image_count",
        "manifest_sha256",
        "formal_data_inventory",
        "checkpoint_sha256",
        "authorization_sha256",
        "run_contract_sha256",
        "predictions_digest",
        "aggregation",
        "runtime",
        "metric_protocol",
        "outputs",
    }
    if set(summary) != summary_keys:
        raise MiO100EvaluationError("formal evaluator summary fields drifted")
    expected_summary_scalars = {
        "schema_version": SUMMARY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": contract["created_utc"],
        "method_name": expected_method_name,
        "image_count": row_count,
        "manifest_sha256": authorization.bindings["formal_manifest"].sha256,
        "formal_data_inventory": dict(inventory_raw),
        "checkpoint_sha256": authorization.bindings["stage4_checkpoint"].sha256,
        "authorization_sha256": authorization.sha256,
        "run_contract_sha256": run_contract_binding.sha256,
        "predictions_digest": predictions_digest,
        "metric_protocol": {
            "prediction_source": "lossless_png_readback",
            "psnr": "AgenticIR/pyiqa-0.1.10 RGB parity",
            "ssim": "AgenticIR/pyiqa-0.1.10 Y parity",
            "group_reduction": "equal_combination_mean",
            "weighted_all_images": "additional_only",
        },
        "outputs": {
            "agenticir_methods_root": str(root / "methods" / expected_method_name),
            "per_image_csv": str(per_image_binding.path),
            "table1_input_jsonl": str(table1_binding.path),
        },
    }
    for key, value in expected_summary_scalars.items():
        if summary.get(key) != value:
            raise MiO100EvaluationError(f"formal evaluator summary drifted at {key}")
    recomputed_aggregate = aggregate_official_records(
        metric_rows,
        required_combinations=tuple(
            combination
            for combinations in OFFICIAL_GROUPS.values()
            for combination in combinations
        ),
        expected_counts=combination_counts,
    )
    if summary.get("aggregation") != recomputed_aggregate:
        raise MiO100EvaluationError("formal evaluator aggregate metrics drifted")
    recomputed_runtime = _aggregate_runtime(runtime_receipts)
    if summary.get("runtime") != recomputed_runtime:
        raise MiO100EvaluationError("formal evaluator runtime diagnostics drifted")

    def evidence_binding(binding: ArtifactBinding) -> Mapping[str, str]:
        return {"path": str(binding.path), "sha256": binding.sha256}

    authorization_binding = ArtifactBinding(
        path=authorization.path, sha256=authorization.sha256
    )
    complete_binding = ArtifactBinding(path=complete_file, sha256=complete_sha)
    evidence = {
        "authorization": evidence_binding(authorization_binding),
        "evaluator_complete": evidence_binding(complete_binding),
        "run_contract": evidence_binding(run_contract_binding),
        "summary": evidence_binding(summary_binding),
        "per_image": evidence_binding(per_image_binding),
        "table1_input": evidence_binding(table1_binding),
        "checkpoint": evidence_binding(authorization.bindings["stage4_checkpoint"]),
        "manifest": evidence_binding(authorization.bindings["formal_manifest"]),
        "formal_data_inventory": evidence_binding(inventory_binding),
        "predictions_digest": predictions_digest,
    }
    return FormalEvaluatorCompletion(
        complete_path=complete_file,
        complete_sha256=complete_sha,
        authorization=authorization_binding,
        run_contract=run_contract_binding,
        summary=summary_binding,
        per_image=per_image_binding,
        table1_input=table1_binding,
        checkpoint=authorization.bindings["stage4_checkpoint"],
        manifest=authorization.bindings["formal_manifest"],
        formal_data_inventory=inventory_binding,
        predictions_digest=predictions_digest,
        evidence=evidence,
    )


def bind_default_authorization_paths(
    project_root: str | Path,
    *,
    manifest: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    stage4_complete: str | Path,
    thresholds: str | Path,
    pair_prior: str | Path,
    global_priority: str | Path,
    formal_data_inventory: str | Path = FORMAL_DATA_INVENTORY_PATH,
) -> Mapping[str, Path]:
    """Known canonical paths; externally produced bindings remain self-verified."""

    return authorization_binding_paths(
        project_root,
        manifest=manifest,
        formal_data_inventory=formal_data_inventory,
        checkpoint=checkpoint,
        config=config,
        stage4_complete=stage4_complete,
        thresholds=thresholds,
        pair_prior=pair_prior,
        global_priority=global_priority,
    )


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "COMPLETE_SCHEMA",
    "FORMAL_DATA_INVENTORY_PATH",
    "FORMAL_COMBINATION_COUNTS",
    "FORMAL_GROUP_COUNTS",
    "FORMAL_MANIFEST_FILENAME",
    "FORMAL_MANIFEST_SHA256",
    "FORMAL_METHOD_NAME",
    "FORMAL_OUTPUT_ROOT",
    "FORMAL_ROW_COUNT",
    "FormalAuthorization",
    "FormalEvaluatorCompletion",
    "InferenceResult",
    "MiO100EvaluationError",
    "MiO100Record",
    "MIOIR_MATLAB_FUNCTIONS_SHA256",
    "Stage4Checkpoint",
    "assert_exclusive_gpu_process",
    "autonomous_graphrestore_inference",
    "bind_default_authorization_paths",
    "build_formal_graphrestore",
    "configure_formal_runtime",
    "finalize_evaluation",
    "load_formal_manifest",
    "load_and_bind_formal_data_inventory",
    "load_stage4_best_ema",
    "prepare_run_contract",
    "process_record",
    "run_shard",
    "validate_formal_authorization",
    "validate_formal_evaluator_complete",
    "validate_protocol_bindings",
    "validate_stage4_completion",
]
