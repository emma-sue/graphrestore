#!/usr/bin/env python3
"""Gate and publish the one-off Stage4 40k -> 48k extension.

The extension is deliberately a two-stage authorization:

1. ``STAGE4_EXTENSION_CONDITIONAL_APPROVED.json`` is written before the
   step-40000 result exists.  It authorizes only an exact Decimal gate.
2. This module snapshots the stopped step-40000 boundary as independent 0444
   copies, recomputes ``PSNR(40000) - PSNR(36000)`` from the canonical CSV
   strings, and writes ``STAGE4_EXTENSION_GATE_RECEIPT.json``.  A result below
   ``Decimal("0.20")`` is permanently recorded as ``DO_NOT_EXTEND``.

Only an immutable ``ACTIVATE_EXTENSION`` gate receipt can authorize the
provenance migration.  The live run contract and both checkpoints receive the
same new provenance.  Outside provenance, the EMA selection checkpoint and
run contract remain bit-exact; the raw checkpoint changes only the
``metrics.best_checkpoint_sha256`` leaf so it binds the migrated EMA file.

The optimizer, EMA, scheduler, RNG and sampler are never reset or rewritten.
The original cosine schedule horizon remains 40000 and its reached 5e-7 floor
is held through the two additional validation cycles at 44000 and 48000.
"""

from __future__ import annotations

import argparse
import copy
import csv
from decimal import Decimal, InvalidOperation, localcontext
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

# This assignment must precede imports that can transitively import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.provenance import semantic_source_hashes  # noqa: E402
from src.utils.hashing import is_sha256, sha256_file, sha256_json  # noqa: E402
from src.utils.io import atomic_write_json, fsync_directory, load_json, utc_now_iso  # noqa: E402


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
CHECKPOINT_SCHEMA = "graphrestore-checkpoint-v1"
STAGE4_RUNTIME_SCHEMA = "graphrestore-stage4-runtime-v1"
CONDITIONAL_SCHEMA = "graphrestore-stage4-extension-conditional-approval-v1"
GATE_SCHEMA = "graphrestore-stage4-extension-gate-receipt-v1"
MIGRATION_RECEIPT_SCHEMA = "graphrestore-stage4-extension-migration-v1"

CONDITIONAL_KIND = "stage4_extension_conditional_approval"
GATE_KIND = "stage4_extension_gate_receipt"
MIGRATION_KIND = "stage4_40000_to_48000_extension_provenance"

BASE_STEP = 40_000
TARGET_STEP = 48_000
HARD_TERMINAL_STEP = 48_000
ADDITIONAL_OPTIMIZER_STEPS = 8_000
CYCLES = 2
VALIDATION_EVERY_STEPS = 4_000
VALIDATION_STEPS = (44_000, 48_000)
PRE_EXTENSION_VALIDATION_STEPS = tuple(range(4_000, BASE_STEP + 1, 4_000))
SCHEDULE_HORIZON_STEPS = 40_000
MIN_LR = 5.0e-7
LR_POLICY = "hold_original_cosine_floor_after_schedule_horizon"

TRIGGER_METRIC = "group_a_psnr"
TRIGGER_LHS_STEP = 40_000
TRIGGER_RHS_STEP = 36_000
TRIGGER_OPERATOR = "lhs_minus_rhs_greater_than_or_equal"
TRIGGER_THRESHOLD_DECIMAL = "0.20"
TRIGGER_ARITHMETIC = "decimal_exact_from_canonical_csv_strings"

DECISION_ACTIVATE = "ACTIVATE_EXTENSION"
DECISION_DO_NOT_EXTEND = "DO_NOT_EXTEND"

CONDITIONAL_NAME = "STAGE4_EXTENSION_CONDITIONAL_APPROVED.json"
GATE_NAME = "STAGE4_EXTENSION_GATE_RECEIPT.json"
BACKUP_DIR_NAME = "stage4_extension_40000_to_48000_v1"
RECEIPT_NAME = "MIGRATION_RECEIPT.json"

GATE_CONFIRMATION_TOKEN = "WRITE_STAGE4_EXTENSION_40000_GATE_RECEIPT"
MIGRATION_CONFIRMATION_TOKEN = "MIGRATE_STAGE4_40000_TO_48000_EXTENSION_PROVENANCE"

ENTRYPOINTS = ("scripts/train_stage4_e2e.py",)
ALLOWED_CHANGED_SOURCE_PATHS = (
    "scripts/train_stage4_e2e.py",
    "src/training/orchestration.py",
    "src/training/stage4_engine.py",
)

CALIBRATION_COLUMNS = (
    "step",
    "single_psnr",
    "single_ssim",
    "group_a_psnr",
    "group_a_ssim",
    "planner_macro_f1",
    "relation_accuracy",
    "parallel_precision",
    "parallel_recall",
    "pre_cycle_rate",
    "dropped_edge_rate",
    "guard_spearman_rain",
    "guard_spearman_haze",
    "guard_mae_rain",
    "guard_mae_haze",
    "guard_std_rain",
    "guard_std_haze",
    "guard_high_frac_rain",
    "guard_high_frac_haze",
    "clean_misuse_psnr",
    "clean_misuse_ssim",
    "clean_misuse_residual_norm",
    "wrong_skill_identity_psnr",
    "wrong_skill_identity_ssim",
    "wrong_skill_residual_norm",
    "reentry_request_rate",
    "unexpected_skill_activation_rate",
    "mean_program_levels",
)

SNAPSHOT_FILENAMES = {
    "run_contract": "pre_extension_run_contract.json",
    "last_checkpoint": "pre_extension_last.pth",
    "best_checkpoint": "pre_extension_best_ema.pth",
    "calibration_history": "pre_extension_stage4_calibration_history.csv",
    "validation_latest": "pre_extension_validation_latest.json",
    "report": "pre_extension_STAGE4_E2E.md",
    "train_log": "pre_extension_train.jsonl",
    "orchestration_state": "pre_extension_orchestration_state.json",
    "pipeline_log": "pre_extension_main_pipeline.log",
    "config": "pre_extension_stage4_graphrestore_e2e.yaml",
}


class Stage4ExtensionMigrationError(RuntimeError):
    """A Stage4 extension gate or migration invariant failed."""


def _fail(message: str) -> NoReturn:
    raise Stage4ExtensionMigrationError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _assert_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        _fail("Stage4 extension migration must remain CPU-only")


def _validate_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not is_sha256(value):
        _fail(f"{field} must be a lowercase SHA256")
    return value


def _absolute_lexical(path: str | Path) -> Path:
    raw = Path(path)
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if not raw.is_absolute() or str(raw) != str(absolute):
        _fail(f"path must be absolute and lexically canonical: {raw}")
    return absolute


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            _fail(f"symlink is forbidden in {label} path: {current}")


def _canonical_path(path: str | Path, *, label: str) -> Path:
    absolute = _absolute_lexical(path)
    _reject_symlink_chain(absolute, label=label)
    if absolute.resolve(strict=False) != absolute:
        _fail(f"{label} is not a canonical path: {absolute}")
    return absolute


@contextmanager
def _single_writer_lock(migrations_directory: Path) -> Iterator[None]:
    descriptor = os.open(
        migrations_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Stage4ExtensionMigrationError(
                "another Stage4 extension writer holds the migration lock"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _qualified_type(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _tensor_bytes(value: torch.Tensor) -> bytes:
    if value.layout is not torch.strided:
        _fail(f"unsupported tensor layout: {value.layout}")
    flat = value.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _assert_bit_exact(before: object, after: object, *, path: str) -> None:
    if type(before) is not type(after):
        _fail(
            f"type mutation at {path}: {_qualified_type(before)} != "
            f"{_qualified_type(after)}"
        )
    if isinstance(before, torch.Tensor):
        assert isinstance(after, torch.Tensor)
        old_meta = (
            before.dtype,
            before.layout,
            tuple(before.shape),
            tuple(before.stride()),
            before.storage_offset(),
            before.requires_grad,
        )
        new_meta = (
            after.dtype,
            after.layout,
            tuple(after.shape),
            tuple(after.stride()),
            after.storage_offset(),
            after.requires_grad,
        )
        if old_meta != new_meta or _tensor_bytes(before) != _tensor_bytes(after):
            _fail(f"tensor mutation at {path}")
        return
    if isinstance(before, np.ndarray):
        assert isinstance(after, np.ndarray)
        if (
            before.dtype != after.dtype
            or before.shape != after.shape
            or before.strides != after.strides
            or before.tobytes(order="A") != after.tobytes(order="A")
        ):
            _fail(f"numpy mutation at {path}")
        return
    if isinstance(before, Mapping):
        assert isinstance(after, Mapping)
        if list(before) != list(after):
            _fail(f"mapping key/order mutation at {path}")
        for key in before:
            _assert_bit_exact(before[key], after[key], path=f"{path}.{key}")
        return
    if isinstance(before, (list, tuple)):
        assert isinstance(after, (list, tuple))
        if len(before) != len(after):
            _fail(f"sequence length mutation at {path}")
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            _assert_bit_exact(old, new, path=f"{path}[{index}]")
        return
    if isinstance(before, float):
        assert isinstance(after, float)
        if struct.pack(">d", before) != struct.pack(">d", after):
            _fail(f"float mutation at {path}")
        return
    if before != after:
        _fail(f"value mutation at {path}: {before!r} != {after!r}")


def _walk_finite(value: object, *, path: str = "checkpoint") -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            _fail(f"non-finite tensor at {path}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(
            np.isfinite(value).all()
        ):
            _fail(f"non-finite numpy value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_finite(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_finite(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(f"non-finite scalar at {path}")


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        raise Stage4ExtensionMigrationError(
            f"could not load checkpoint on CPU: {type(exc).__name__}: {exc}"
        ) from exc
    return _mapping(value, field=f"checkpoint {path}")


def _validate_source_map(value: Mapping[str, str], *, field: str) -> dict[str, str]:
    result = dict(value)
    if len(result) != 47:
        _fail(f"{field} must contain the exact 47 semantic-source paths")
    for path, digest in result.items():
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or Path(path).as_posix() != path
            or not is_sha256(digest)
        ):
            _fail(f"{field} contains an invalid path/SHA256 entry")
    return dict(sorted(result.items()))


def _validate_source_transition(
    old_value: Mapping[str, str], new_value: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    old = _validate_source_map(old_value, field="old source map")
    new = _validate_source_map(new_value, field="new source map")
    if old.keys() != new.keys():
        _fail("old/new semantic-source maps have different path sets")
    changed = tuple(path for path in old if old[path] != new[path])
    if changed != ALLOWED_CHANGED_SOURCE_PATHS:
        _fail(
            "semantic-source changed-path set must be exactly "
            f"{list(ALLOWED_CHANGED_SOURCE_PATHS)}, got {list(changed)}"
        )
    return old, new


def _resolve_paths(project_root: str | Path) -> dict[str, Path]:
    root = _canonical_path(project_root, label="project root")
    paths = {
        "project_root": root,
        "run_contract": root / "artifacts/checkpoints/stage4/run_contract.json",
        "last_checkpoint": root / "artifacts/checkpoints/stage4/last.pth",
        "best_checkpoint": root / "artifacts/checkpoints/stage4/best_ema.pth",
        "calibration_history": root
        / "artifacts/metrics/stage4_calibration_history.csv",
        "validation_latest": root
        / "artifacts/checkpoints/stage4/validation_latest.json",
        "report": root / "reports/STAGE4_E2E.md",
        "train_log": root / "artifacts/checkpoints/stage4/train.jsonl",
        "orchestration_state": root / "artifacts/orchestration/state.json",
        "pipeline_log": root / "artifacts/logs/main_pipeline.log",
        "config": root / "configs/stage4_graphrestore_e2e.yaml",
        "instruction_protocol": root
        / "reports/STAGE4_CONDITIONAL_EXTENSION_PROTOCOL.md",
        "conditional": root / "artifacts/approvals" / CONDITIONAL_NAME,
        "gate": root / "artifacts/approvals" / GATE_NAME,
        "backup_dir": root / "artifacts/migrations" / BACKUP_DIR_NAME,
        "complete": root / "artifacts/checkpoints/stage4/complete.json",
        "diagnostics_json": root / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.json",
        "diagnostics_report": root / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.md",
    }
    for label, path in paths.items():
        _reject_symlink_chain(path, label=label)
    return paths


def _require_file(path: Path, *, label: str, expected_sha256: str | None = None) -> str:
    if path.is_symlink() or not path.is_file():
        _fail(f"missing or symlinked canonical {label}: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        _fail(f"{label} SHA256 drifted")
    return digest


def _require_immutable_json(
    path: Path, *, label: str, digest: str
) -> Mapping[str, Any]:
    _require_file(path, label=label, expected_sha256=digest)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        _fail(f"{label} must be chmod 0444")
    return _mapping(load_json(path), field=label)


def _expected_conditional_keys() -> set[str]:
    return {
        "schema_version",
        "kind",
        "protocol_id",
        "approved",
        "conditional",
        "created_utc",
        "cycles",
        "additional_optimizer_steps",
        "base_step",
        "target_step",
        "hard_terminal_step",
        "validation_every_steps",
        "validation_steps",
        "schedule_horizon_steps",
        "min_lr",
        "lr_policy",
        "exact_resume",
        "reset_optimizer",
        "reset_ema",
        "reset_scheduler",
        "reset_rng",
        "reset_sampler",
        "further_extension_authorized",
        "trigger_metric",
        "trigger_lhs_step",
        "trigger_rhs_step",
        "trigger_operator",
        "trigger_threshold_decimal",
        "trigger_arithmetic",
        "formal_mio100_authorized",
        "group_b_or_c_authorized",
        "authorized_pipeline",
        "base_stage4_config",
        "user_instruction_protocol",
        "base_stage4_run_contract",
        "preauthorization_ledger_prefix",
    }


def build_stage4_extension_conditional_authorization(
    *,
    project_root: str | Path,
    config_sha256: str,
    run_contract_sha256: str,
    instruction_protocol_sha256: str,
    preauthorization_ledger_prefix_byte_length: int,
    preauthorization_ledger_prefix_sha256: str,
    created_utc: str,
) -> dict[str, Any]:
    """Build the exact pre-40k conditional authorization payload."""

    paths = _resolve_paths(project_root)
    _validate_sha(config_sha256, field="Stage4 config")
    _validate_sha(run_contract_sha256, field="Stage4 run contract")
    _validate_sha(instruction_protocol_sha256, field="user instruction protocol")
    _validate_sha(
        preauthorization_ledger_prefix_sha256,
        field="preauthorization ledger prefix",
    )
    if (
        isinstance(preauthorization_ledger_prefix_byte_length, bool)
        or not isinstance(preauthorization_ledger_prefix_byte_length, int)
        or preauthorization_ledger_prefix_byte_length <= 0
    ):
        _fail("preauthorization ledger prefix byte length must be positive")
    payload = {
        "schema_version": CONDITIONAL_SCHEMA,
        "kind": CONDITIONAL_KIND,
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "conditional": True,
        "created_utc": created_utc,
        "cycles": CYCLES,
        "additional_optimizer_steps": ADDITIONAL_OPTIMIZER_STEPS,
        "base_step": BASE_STEP,
        "target_step": TARGET_STEP,
        "hard_terminal_step": HARD_TERMINAL_STEP,
        "validation_every_steps": VALIDATION_EVERY_STEPS,
        "validation_steps": list(VALIDATION_STEPS),
        "schedule_horizon_steps": SCHEDULE_HORIZON_STEPS,
        "min_lr": MIN_LR,
        "lr_policy": LR_POLICY,
        "exact_resume": True,
        "reset_optimizer": False,
        "reset_ema": False,
        "reset_scheduler": False,
        "reset_rng": False,
        "reset_sampler": False,
        "further_extension_authorized": False,
        "trigger_metric": TRIGGER_METRIC,
        "trigger_lhs_step": TRIGGER_LHS_STEP,
        "trigger_rhs_step": TRIGGER_RHS_STEP,
        "trigger_operator": TRIGGER_OPERATOR,
        "trigger_threshold_decimal": TRIGGER_THRESHOLD_DECIMAL,
        "trigger_arithmetic": TRIGGER_ARITHMETIC,
        "formal_mio100_authorized": False,
        "group_b_or_c_authorized": False,
        "authorized_pipeline": [
            "stage4_extension_gate",
            "stage4_extension",
            "stage4_zero_training_diagnostics",
        ],
        "base_stage4_config": {
            "path": str(paths["config"]),
            "sha256": config_sha256,
        },
        "user_instruction_protocol": {
            "path": str(paths["instruction_protocol"]),
            "sha256": instruction_protocol_sha256,
        },
        "base_stage4_run_contract": {
            "path": str(paths["run_contract"]),
            "sha256": run_contract_sha256,
        },
        "preauthorization_ledger_prefix": {
            "path": str(paths["calibration_history"]),
            "byte_length": preauthorization_ledger_prefix_byte_length,
            "sha256": preauthorization_ledger_prefix_sha256,
        },
    }
    if set(payload) != _expected_conditional_keys():
        _fail("conditional authorization field set drifted")
    return payload


def _validate_conditional(
    paths: Mapping[str, Path], expected_sha256: str
) -> dict[str, Any]:
    conditional = _require_immutable_json(
        paths["conditional"],
        label="conditional authorization",
        digest=expected_sha256,
    )
    protocol_binding = _mapping(
        conditional.get("user_instruction_protocol"),
        field="conditional user instruction protocol",
    )
    run_binding = _mapping(
        conditional.get("base_stage4_run_contract"),
        field="conditional base Stage4 run contract",
    )
    prefix_binding = _mapping(
        conditional.get("preauthorization_ledger_prefix"),
        field="conditional ledger prefix",
    )
    protocol_sha = _validate_sha(
        protocol_binding.get("sha256"), field="conditional instruction protocol"
    )
    run_sha = _validate_sha(run_binding.get("sha256"), field="conditional run contract")
    prefix_sha = _validate_sha(
        prefix_binding.get("sha256"), field="conditional ledger prefix"
    )
    prefix_length = prefix_binding.get("byte_length")
    if (
        protocol_binding.get("path") != str(paths["instruction_protocol"])
        or run_binding.get("path") != str(paths["run_contract"])
        or prefix_binding.get("path") != str(paths["calibration_history"])
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or prefix_length <= 0
    ):
        _fail("conditional pre-result binding path/schema drifted")
    config_sha = sha256_file(paths["config"])
    expected = build_stage4_extension_conditional_authorization(
        project_root=paths["project_root"],
        config_sha256=config_sha,
        run_contract_sha256=run_sha,
        instruction_protocol_sha256=protocol_sha,
        preauthorization_ledger_prefix_byte_length=prefix_length,
        preauthorization_ledger_prefix_sha256=prefix_sha,
        created_utc=str(conditional.get("created_utc")),
    )
    if dict(conditional) != expected:
        _fail("conditional Stage4 extension authorization contract drifted")
    _require_file(
        paths["instruction_protocol"],
        label="Stage4 conditional extension instruction protocol",
        expected_sha256=protocol_sha,
    )
    if stat.S_IMODE(paths["instruction_protocol"].stat().st_mode) != 0o444:
        _fail("Stage4 conditional extension instruction protocol must be chmod 0444")
    _require_file(
        paths["run_contract"],
        label="conditional base Stage4 run contract",
        expected_sha256=run_sha,
    )
    _require_file(paths["calibration_history"], label="conditional Stage4 ledger")
    if _sha256_prefix(paths["calibration_history"], prefix_length) != prefix_sha:
        _fail("Stage4 calibration ledger preauthorization prefix drifted")
    return dict(conditional)


def _sha256_prefix(path: Path, byte_length: int) -> str:
    digest = hashlib.sha256()
    remaining = byte_length
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                _fail("preauthorization ledger prefix exceeds the live file length")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _decimal_string(raw: object, *, field: str) -> tuple[str, Decimal]:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        _fail(f"{field} must be a non-empty canonical CSV string")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise Stage4ExtensionMigrationError(f"{field} is not Decimal") from exc
    if not value.is_finite():
        _fail(f"{field} is not finite")
    return raw, value


def _decimal_delta(lhs: Decimal, rhs: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        return lhs - rhs


def _validate_calibration_history(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    _require_file(
        path, label="Stage4 calibration history", expected_sha256=expected_sha256
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CALIBRATION_COLUMNS:
            _fail("Stage4 calibration history is not the exact 28-column schema")
        rows = [dict(row) for row in reader]
    if (
        tuple(int(row.get("step", "-1")) for row in rows)
        != PRE_EXTENSION_VALIDATION_STEPS
    ):
        _fail("Stage4 calibration history is not the exact 4k..40k prefix")
    if any(
        set(row) != set(CALIBRATION_COLUMNS) or any(v is None for v in row.values())
        for row in rows
    ):
        _fail("Stage4 calibration history contains a malformed row")
    by_step = {int(row["step"]): row for row in rows}
    lhs_raw, lhs = _decimal_string(
        by_step[TRIGGER_LHS_STEP][TRIGGER_METRIC], field="step-40000 Group-A PSNR"
    )
    rhs_raw, rhs = _decimal_string(
        by_step[TRIGGER_RHS_STEP][TRIGGER_METRIC], field="step-36000 Group-A PSNR"
    )
    delta = _decimal_delta(lhs, rhs)
    threshold = Decimal(TRIGGER_THRESHOLD_DECIMAL)
    return {
        "row_count": len(rows),
        "steps": list(PRE_EXTENSION_VALIDATION_STEPS),
        "observed_lhs_decimal": lhs_raw,
        "observed_rhs_decimal": rhs_raw,
        "observed_delta_decimal": str(delta),
        "threshold_decimal": TRIGGER_THRESHOLD_DECIMAL,
        "decision": (
            DECISION_ACTIVATE if delta >= threshold else DECISION_DO_NOT_EXTEND
        ),
    }


def _validate_validation_latest(
    path: Path,
    *,
    expected_sha256: str,
    gate_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    _require_file(
        path, label="Stage4 validation latest", expected_sha256=expected_sha256
    )
    value = _mapping(load_json(path), field="Stage4 validation latest")
    group = _mapping(
        value.get("group_a_equal_combination_mean"), field="latest Group-A"
    )
    single = _mapping(value.get("single_equal_task_mean"), field="latest single")
    if (
        value.get("schema_version") != "graphrestore-stage4-validation-v1"
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("image_count") != 1600
        or group.get("combination_count") != 8
        or group.get("count") != 800
        or single.get("task_count") != 8
        or single.get("count") != 800
    ):
        _fail("Stage4 validation_latest is not the exact 1600-image contract")
    latest_psnr = group.get("psnr")
    if (
        isinstance(latest_psnr, bool)
        or not isinstance(latest_psnr, (int, float))
        or not math.isfinite(float(latest_psnr))
        or Decimal(str(latest_psnr))
        != Decimal(str(gate_evidence["observed_lhs_decimal"]))
    ):
        _fail("validation_latest Group-A PSNR differs from the step-40000 CSV row")
    return {
        "image_count": 1600,
        "group_a_psnr": str(latest_psnr),
        "single_psnr": str(single.get("psnr")),
    }


def _validate_train_log(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    _require_file(path, label="Stage4 train log", expected_sha256=expected_sha256)
    validation_count = 0
    optimizer_steps: list[int] = []
    last_event: Mapping[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                _fail(f"blank Stage4 train log row at line {line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise Stage4ExtensionMigrationError(
                    f"invalid Stage4 train log JSON at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                _fail(f"non-object Stage4 train log row at line {line_number}")
            event, step = row.get("event"), row.get("step")
            if event is None:
                if (
                    isinstance(step, bool)
                    or not isinstance(step, int)
                    or row.get("schema_version") != STAGE4_RUNTIME_SCHEMA
                    or row.get("samples") != 4
                    or any(
                        field not in row
                        for field in (
                            "created_utc",
                            "loss",
                            "grad_norm",
                            "learning_rates",
                            "seconds",
                        )
                    )
                ):
                    _fail(
                        f"malformed Stage4 optimizer row at train log line {line_number}"
                    )
                optimizer_steps.append(step)
            if event == "validation" and step == BASE_STEP:
                validation_count += 1
            last_event = row
    if (
        validation_count != 1
        or optimizer_steps != list(range(1, BASE_STEP + 1))
        or last_event is None
    ):
        _fail(
            "Stage4 train log is not continuous 1..40000 with one committed "
            "step-40000 validation"
        )
    return {
        "validation_step40000_count": validation_count,
        "optimizer_row_count": len(optimizer_steps),
        "minimum_train_step": optimizer_steps[0],
        "maximum_train_step": optimizer_steps[-1],
        "last_event": str(last_event.get("event")),
        "last_event_step": last_event.get("step"),
    }


def _validate_stopped_state(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    _require_file(path, label="orchestration state", expected_sha256=expected_sha256)
    state = _mapping(load_json(path), field="orchestration state")
    command = state.get("last_command")
    if (
        state.get("schema_version") != "graphrestore-orchestration-v1"
        or state.get("protocol_id") != PROTOCOL_ID
        or state.get("status") != "FAILED"
        or state.get("current_stage") != "FAILED"
        or state.get("gpu") != "released"
        or state.get("last_exit_code") != 130
        or not isinstance(command, list)
        or "scripts/train_stage4_e2e.py" not in command
    ):
        _fail("orchestration is not the exact fail-closed post-40000 stop")
    return {
        key: state.get(key)
        for key in ("status", "current_stage", "gpu", "last_exit_code", "next_command")
    }


def _validate_checkpoint(payload: Mapping[str, Any], *, role: str) -> None:
    if len(payload) != 17:
        _fail(f"{role} Stage4 checkpoint must have exactly 17 top-level sections")
    step = payload.get("step")
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("stage") != "stage4"
        or payload.get("model_role") != role
        or payload.get("resumable") is not (role == "raw_training_state")
        or payload.get("pending_validation_step") is not None
        or payload.get("scaler") is not None
        or payload.get("amp") != {"dtype": "bfloat16", "scaler_required": False}
        or isinstance(step, bool)
        or not isinstance(step, int)
    ):
        _fail(f"{role} Stage4 checkpoint header drifted")
    if role == "raw_training_state" and step != BASE_STEP:
        _fail("raw Stage4 checkpoint is not at the committed step-40000 boundary")
    if role == "ema_selection" and step not in PRE_EXTENSION_VALIDATION_STEPS:
        _fail("best Stage4 checkpoint is not a valid pre-extension boundary")
    ema = _mapping(payload.get("ema"), field=f"{role}.ema")
    scheduler = _mapping(payload.get("scheduler"), field=f"{role}.scheduler")
    sampler = _mapping(payload.get("sampler_state"), field=f"{role}.sampler")
    if (
        ema.get("num_updates") != step
        or scheduler.get("max_steps") != SCHEDULE_HORIZON_STEPS
        or scheduler.get("min_lr") != MIN_LR
        or scheduler.get("last_epoch") != step
        or scheduler.get("_step_count") != step + 1
        or sampler.get("consumed_optimizer_step") != step
        or sampler.get("sample_cursor") != step * 4
        or sampler.get("effective_batch_size") != 4
        or sampler.get("num_samples") != BASE_STEP * 4
    ):
        _fail(f"{role} Stage4 optimizer boundary metadata drifted")
    _walk_finite(payload)


def _validate_checkpoint_pair(
    last: Mapping[str, Any], best: Mapping[str, Any], *, expected_best_sha256: str
) -> None:
    _validate_checkpoint(last, role="raw_training_state")
    _validate_checkpoint(best, role="ema_selection")
    if last.get("provenance") != best.get("provenance"):
        _fail("Stage4 last/best provenance differs")
    best_ema = _mapping(best.get("ema"), field="best.ema")
    best_model = _mapping(best.get("model"), field="best.model")
    best_shadow = _mapping(best_ema.get("shadow"), field="best.ema.shadow")
    if list(best_model) != list(best_shadow):
        _fail("Stage4 best model/EMA keys differ")
    for name in best_model:
        _assert_bit_exact(
            best_model[name], best_shadow[name], path=f"best.model_ema.{name}"
        )
    metrics = _mapping(last.get("metrics"), field="last.metrics")
    if (
        metrics.get("validation_step") != float(BASE_STEP)
        or metrics.get("best_checkpoint_sha256") != expected_best_sha256
    ):
        _fail("Stage4 raw checkpoint incumbent binding drifted")


def _validate_provenance_anchor(
    provenance: Mapping[str, Any], *, old_source_map: Mapping[str, str]
) -> None:
    runtime = _mapping(provenance.get("runtime"), field="provenance.runtime")
    routing = _mapping(
        provenance.get("calibration_history_routing"),
        field="provenance.calibration_history_routing",
    )
    if (
        provenance.get("schema_version") != STAGE4_RUNTIME_SCHEMA
        or provenance.get("protocol_id") != PROTOCOL_ID
        or dict(
            _mapping(provenance.get("semantic_source_sha256"), field="semantic sources")
        )
        != dict(old_source_map)
        or runtime.get("max_steps") != BASE_STEP
        or runtime.get("schedule_max_steps") != SCHEDULE_HORIZON_STEPS
        or routing.get("validation_steps") != list(PRE_EXTENSION_VALIDATION_STEPS)
        or "stage4_extension" in provenance
    ):
        _fail("pre-extension Stage4 provenance identity/schedule drifted")


def _snapshot_paths(paths: Mapping[str, Path]) -> dict[str, Path]:
    return {
        label: paths["backup_dir"] / filename
        for label, filename in SNAPSHOT_FILENAMES.items()
    }


def _copy_backup(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        _fail(f"refusing existing Stage4 extension snapshot: {destination}")
    source_mode = stat.S_IMODE(source.stat().st_mode)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with (
            source.open("rb") as reader,
            os.fdopen(descriptor, "wb", closefd=False) as writer,
        ):
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        os.close(descriptor)
    os.chmod(destination, 0o444, follow_symlinks=False)
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    fsync_directory(destination.parent)
    source_info, copy_info = source.stat(), destination.stat()
    if (
        source_info.st_dev != copy_info.st_dev
        or source_info.st_ino == copy_info.st_ino
        or destination.is_symlink()
        or stat.S_IMODE(copy_info.st_mode) != 0o444
        or sha256_file(source) != sha256_file(destination)
    ):
        _fail(f"snapshot is not a distinct immutable same-disk copy: {destination}")
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "source_mode": source_mode,
        "mode": 0o444,
        "device": copy_info.st_dev,
        "inode": copy_info.st_ino,
        "distinct_inode_from_live_at_creation": True,
    }


def _validate_snapshot(
    paths: Mapping[str, Path], label: str, raw: object
) -> dict[str, Any]:
    evidence = _mapping(raw, field=f"gate snapshot {label}")
    expected_path = _snapshot_paths(paths)[label]
    if evidence.get("path") != str(expected_path):
        _fail(f"gate snapshot path drifted: {label}")
    _reject_symlink_chain(expected_path, label=f"snapshot {label}")
    info = expected_path.stat() if expected_path.is_file() else None
    digest = _validate_sha(evidence.get("sha256"), field=f"snapshot {label}")
    if (
        info is None
        or expected_path.is_symlink()
        or stat.S_IMODE(info.st_mode) != 0o444
        or evidence.get("mode") != 0o444
        or evidence.get("device") != info.st_dev
        or evidence.get("inode") != info.st_ino
        or evidence.get("distinct_inode_from_live_at_creation") is not True
        or sha256_file(expected_path) != digest
        or not isinstance(evidence.get("source_mode"), int)
    ):
        _fail(f"gate snapshot content/identity drifted: {label}")
    return dict(evidence)


def _expected_live_hashes(
    *,
    run_contract: str,
    last_checkpoint: str,
    best_checkpoint: str,
    calibration_history: str,
    validation_latest: str,
    report: str,
    train_log: str,
    orchestration_state: str,
    pipeline_log: str,
    config: str,
) -> dict[str, str]:
    values = {
        "run_contract": run_contract,
        "last_checkpoint": last_checkpoint,
        "best_checkpoint": best_checkpoint,
        "calibration_history": calibration_history,
        "validation_latest": validation_latest,
        "report": report,
        "train_log": train_log,
        "orchestration_state": orchestration_state,
        "pipeline_log": pipeline_log,
        "config": config,
    }
    for label, digest in values.items():
        _validate_sha(digest, field=label)
    return values


def _verify_live_hashes(paths: Mapping[str, Path], expected: Mapping[str, str]) -> None:
    for label, digest in expected.items():
        _require_file(paths[label], label=label, expected_sha256=digest)


def _build_gate_receipt(
    *,
    paths: Mapping[str, Path],
    conditional_sha256: str,
    gate_evidence: Mapping[str, Any],
    snapshots: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": GATE_SCHEMA,
        "kind": GATE_KIND,
        "protocol_id": PROTOCOL_ID,
        "decision": gate_evidence["decision"],
        "created_utc": utc_now_iso(),
        "conditional_authorization": {
            "path": str(paths["conditional"]),
            "sha256": conditional_sha256,
        },
        "cycles": CYCLES,
        "additional_optimizer_steps": ADDITIONAL_OPTIMIZER_STEPS,
        "base_step": BASE_STEP,
        "target_step": TARGET_STEP,
        "hard_terminal_step": HARD_TERMINAL_STEP,
        "validation_every_steps": VALIDATION_EVERY_STEPS,
        "validation_steps": list(VALIDATION_STEPS),
        "schedule_horizon_steps": SCHEDULE_HORIZON_STEPS,
        "min_lr": MIN_LR,
        "lr_policy": LR_POLICY,
        "trigger_metric": TRIGGER_METRIC,
        "trigger_lhs_step": TRIGGER_LHS_STEP,
        "trigger_rhs_step": TRIGGER_RHS_STEP,
        "trigger_operator": TRIGGER_OPERATOR,
        "trigger_threshold_decimal": TRIGGER_THRESHOLD_DECIMAL,
        "trigger_arithmetic": TRIGGER_ARITHMETIC,
        "observed_lhs_decimal": gate_evidence["observed_lhs_decimal"],
        "observed_rhs_decimal": gate_evidence["observed_rhs_decimal"],
        "observed_delta_decimal": gate_evidence["observed_delta_decimal"],
        "exact_resume": True,
        "reset_optimizer": False,
        "reset_ema": False,
        "reset_scheduler": False,
        "reset_rng": False,
        "reset_sampler": False,
        "further_extension_authorized": False,
        "formal_mio100_authorized": False,
        "group_b_or_c_authorized": False,
        "snapshots": dict(snapshots),
    }


def evaluate_stage4_extension_gate(
    *,
    project_root: str | Path,
    expected_conditional_sha256: str,
    expected_run_contract_sha256: str,
    expected_last_checkpoint_sha256: str,
    expected_best_checkpoint_sha256: str,
    expected_calibration_history_sha256: str,
    expected_validation_latest_sha256: str,
    expected_report_sha256: str,
    expected_train_log_sha256: str,
    expected_state_sha256: str,
    expected_pipeline_log_sha256: str,
    expected_config_sha256: str,
    execute: bool = False,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Evaluate and optionally durably publish the exact step-40000 gate."""

    _assert_cpu_only()
    _validate_sha(expected_conditional_sha256, field="conditional authorization")
    expected = _expected_live_hashes(
        run_contract=expected_run_contract_sha256,
        last_checkpoint=expected_last_checkpoint_sha256,
        best_checkpoint=expected_best_checkpoint_sha256,
        calibration_history=expected_calibration_history_sha256,
        validation_latest=expected_validation_latest_sha256,
        report=expected_report_sha256,
        train_log=expected_train_log_sha256,
        orchestration_state=expected_state_sha256,
        pipeline_log=expected_pipeline_log_sha256,
        config=expected_config_sha256,
    )
    if execute and confirmation_token != GATE_CONFIRMATION_TOKEN:
        _fail("gate publication requires the exact confirmation token")
    paths = _resolve_paths(project_root)
    migrations = paths["backup_dir"].parent
    if migrations.is_symlink() or not migrations.is_dir():
        _fail("canonical migrations directory is missing or symlinked")
    if paths["gate"].exists() or paths["gate"].is_symlink():
        _fail("Stage4 extension gate receipt already exists")
    if paths["backup_dir"].exists() or paths["backup_dir"].is_symlink():
        _fail("dedicated Stage4 extension snapshot directory already exists")
    for forbidden in ("complete", "diagnostics_json", "diagnostics_report"):
        if paths[forbidden].exists() or paths[forbidden].is_symlink():
            _fail(f"Stage4 finalization artifact already exists: {forbidden}")

    with _single_writer_lock(migrations):
        _verify_live_hashes(paths, expected)
        conditional = _validate_conditional(paths, expected_conditional_sha256)
        if conditional["base_stage4_config"] != {
            "path": str(paths["config"]),
            "sha256": expected_config_sha256,
        }:
            _fail("conditional authorization config binding drifted")
        gate_evidence = _validate_calibration_history(
            paths["calibration_history"],
            expected_sha256=expected_calibration_history_sha256,
        )
        validation_evidence = _validate_validation_latest(
            paths["validation_latest"],
            expected_sha256=expected_validation_latest_sha256,
            gate_evidence=gate_evidence,
        )
        train_evidence = _validate_train_log(
            paths["train_log"], expected_sha256=expected_train_log_sha256
        )
        state_evidence = _validate_stopped_state(
            paths["orchestration_state"], expected_sha256=expected_state_sha256
        )
        run = _mapping(load_json(paths["run_contract"]), field="Stage4 run contract")
        last = _load_checkpoint(paths["last_checkpoint"])
        best = _load_checkpoint(paths["best_checkpoint"])
        _validate_checkpoint_pair(
            last, best, expected_best_sha256=expected_best_checkpoint_sha256
        )
        if (
            run.get("schema_version") != STAGE4_RUNTIME_SCHEMA
            or run.get("provenance") != last.get("provenance")
            or last.get("provenance") != best.get("provenance")
        ):
            _fail("pre-extension run/last/best provenance differs")
        report_text = paths["report"].read_text(encoding="utf-8")
        if "Validation step: 40000" not in report_text:
            _fail("Stage4 report is not the committed step-40000 report")
        _verify_live_hashes(paths, expected)

        if not execute:
            return {
                "schema_version": GATE_SCHEMA,
                "status": "DRY_RUN",
                "decision": gate_evidence["decision"],
                "gate_evidence": gate_evidence,
                "validation_evidence": validation_evidence,
                "train_evidence": train_evidence,
                "orchestration_state": state_evidence,
                "planned_snapshot_paths": {
                    label: str(path) for label, path in _snapshot_paths(paths).items()
                },
            }

        paths["backup_dir"].mkdir(parents=False, exist_ok=False)
        fsync_directory(migrations)
        devices = {paths["backup_dir"].stat().st_dev} | {
            paths[label].stat().st_dev for label in SNAPSHOT_FILENAMES
        }
        if len(devices) != 1:
            _fail("Stage4 extension snapshots are not on the live-artifact filesystem")
        snapshots: dict[str, Any] = {}
        for label, destination in _snapshot_paths(paths).items():
            snapshots[label] = _copy_backup(paths[label], destination)
        if len({(row["device"], row["inode"]) for row in snapshots.values()}) != len(
            snapshots
        ):
            _fail("Stage4 extension snapshots alias one another")
        gate = _build_gate_receipt(
            paths=paths,
            conditional_sha256=expected_conditional_sha256,
            gate_evidence=gate_evidence,
            snapshots=snapshots,
        )
        atomic_write_json(paths["gate"], gate)
        os.chmod(paths["gate"], 0o444, follow_symlinks=False)
        with paths["gate"].open("rb") as handle:
            os.fsync(handle.fileno())
        fsync_directory(paths["gate"].parent)
        if dict(load_json(paths["gate"])) != gate:
            _fail("Stage4 extension gate receipt failed JSON round trip")
        _verify_live_hashes(paths, expected)
        _assert_cpu_only()
        return gate | {"sha256": sha256_file(paths["gate"])}


def _validate_gate_receipt(
    paths: Mapping[str, Path],
    *,
    expected_conditional_sha256: str,
    expected_gate_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _validate_conditional(paths, expected_conditional_sha256)
    gate = _require_immutable_json(
        paths["gate"],
        label="Stage4 extension gate receipt",
        digest=expected_gate_sha256,
    )
    if (
        gate.get("schema_version") != GATE_SCHEMA
        or gate.get("kind") != GATE_KIND
        or gate.get("protocol_id") != PROTOCOL_ID
        or gate.get("decision") != DECISION_ACTIVATE
        or gate.get("conditional_authorization")
        != {"path": str(paths["conditional"]), "sha256": expected_conditional_sha256}
        or gate.get("cycles") != CYCLES
        or gate.get("additional_optimizer_steps") != ADDITIONAL_OPTIMIZER_STEPS
        or gate.get("base_step") != BASE_STEP
        or gate.get("target_step") != TARGET_STEP
        or gate.get("hard_terminal_step") != HARD_TERMINAL_STEP
        or gate.get("validation_every_steps") != VALIDATION_EVERY_STEPS
        or gate.get("validation_steps") != list(VALIDATION_STEPS)
        or gate.get("schedule_horizon_steps") != SCHEDULE_HORIZON_STEPS
        or gate.get("min_lr") != MIN_LR
        or gate.get("lr_policy") != LR_POLICY
        or gate.get("trigger_metric") != TRIGGER_METRIC
        or gate.get("trigger_lhs_step") != TRIGGER_LHS_STEP
        or gate.get("trigger_rhs_step") != TRIGGER_RHS_STEP
        or gate.get("trigger_operator") != TRIGGER_OPERATOR
        or gate.get("trigger_threshold_decimal") != TRIGGER_THRESHOLD_DECIMAL
        or gate.get("trigger_arithmetic") != TRIGGER_ARITHMETIC
        or gate.get("exact_resume") is not True
        or any(
            gate.get(name) is not False
            for name in (
                "reset_optimizer",
                "reset_ema",
                "reset_scheduler",
                "reset_rng",
                "reset_sampler",
            )
        )
        or gate.get("further_extension_authorized") is not False
        or gate.get("formal_mio100_authorized") is not False
        or gate.get("group_b_or_c_authorized") is not False
    ):
        _fail("gate receipt is not the exact ACTIVATE_EXTENSION authorization")
    lhs_raw, lhs = _decimal_string(gate.get("observed_lhs_decimal"), field="gate lhs")
    rhs_raw, rhs = _decimal_string(gate.get("observed_rhs_decimal"), field="gate rhs")
    delta_raw, delta = _decimal_string(
        gate.get("observed_delta_decimal"), field="gate delta"
    )
    expected_delta = _decimal_delta(lhs, rhs)
    if (
        delta != expected_delta
        or delta_raw != str(expected_delta)
        or delta < Decimal(TRIGGER_THRESHOLD_DECIMAL)
    ):
        _fail("ACTIVATE_EXTENSION gate Decimal evidence is false")
    raw_snapshots = _mapping(gate.get("snapshots"), field="gate snapshots")
    if set(raw_snapshots) != set(SNAPSHOT_FILENAMES):
        _fail("gate snapshot label set drifted")
    snapshots = {
        label: _validate_snapshot(paths, label, raw_snapshots[label])
        for label in SNAPSHOT_FILENAMES
    }
    # Recompute the trigger from the immutable CSV, not from receipt narration.
    evidence = _validate_calibration_history(
        Path(snapshots["calibration_history"]["path"]),
        expected_sha256=snapshots["calibration_history"]["sha256"],
    )
    if (
        evidence["decision"] != DECISION_ACTIVATE
        or evidence["observed_lhs_decimal"] != lhs_raw
        or evidence["observed_rhs_decimal"] != rhs_raw
        or evidence["observed_delta_decimal"] != delta_raw
    ):
        _fail("gate receipt differs from its immutable calibration snapshot")
    return dict(gate), snapshots


def _extension_binding(
    paths: Mapping[str, Path], *, conditional_sha256: str, gate_sha256: str
) -> dict[str, Any]:
    return {
        "conditional_authorization": {
            "path": str(paths["conditional"]),
            "sha256": conditional_sha256,
        },
        "gate_receipt": {"path": str(paths["gate"]), "sha256": gate_sha256},
        "cycles": CYCLES,
        "additional_optimizer_steps": ADDITIONAL_OPTIMIZER_STEPS,
        "base_step": BASE_STEP,
        "target_step": TARGET_STEP,
        "hard_terminal_step": HARD_TERMINAL_STEP,
        "validation_every_steps": VALIDATION_EVERY_STEPS,
        "validation_steps": list(VALIDATION_STEPS),
        "schedule_horizon_steps": SCHEDULE_HORIZON_STEPS,
        "min_lr": MIN_LR,
        "lr_policy": LR_POLICY,
        "exact_resume": True,
        "reset_optimizer": False,
        "reset_ema": False,
        "reset_scheduler": False,
        "reset_rng": False,
        "reset_sampler": False,
        "further_extension_authorized": False,
    }


def _build_new_provenance(
    old: Mapping[str, Any],
    *,
    old_source_map: Mapping[str, str],
    new_source_map: Mapping[str, str],
    extension_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_provenance_anchor(old, old_source_map=old_source_map)
    new = copy.deepcopy(dict(old))
    new["semantic_source_sha256"] = dict(new_source_map)
    runtime_old = _mapping(old["runtime"], field="old runtime")
    runtime_new = copy.deepcopy(dict(runtime_old))
    runtime_new["max_steps"] = TARGET_STEP
    new["runtime"] = runtime_new
    routing_old = _mapping(
        old["calibration_history_routing"], field="old calibration routing"
    )
    routing_new = copy.deepcopy(dict(routing_old))
    routing_new["validation_steps"] = [
        *PRE_EXTENSION_VALIDATION_STEPS,
        *VALIDATION_STEPS,
    ]
    new["calibration_history_routing"] = routing_new
    new["stage4_extension"] = dict(extension_binding)

    for key in old:
        if key not in {
            "semantic_source_sha256",
            "runtime",
            "calibration_history_routing",
        }:
            _assert_bit_exact(old[key], new[key], path=f"provenance.{key}")
    if set(new) != {*old, "stage4_extension"}:
        _fail("unexpected Stage4 provenance top-level change")
    for key in runtime_old:
        if key != "max_steps":
            _assert_bit_exact(runtime_old[key], runtime_new[key], path=f"runtime.{key}")
    if set(runtime_new) != set(runtime_old):
        _fail("Stage4 runtime provenance key set changed")
    for key in routing_old:
        if key != "validation_steps":
            _assert_bit_exact(routing_old[key], routing_new[key], path=f"routing.{key}")
    if set(routing_new) != set(routing_old):
        _fail("Stage4 calibration routing key set changed")
    return new


def _make_candidate(parent: Path, name: str, suffix: str) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=f".{name}.", suffix=suffix, dir=parent)
    os.close(descriptor)
    path = Path(raw)
    path.unlink()
    return path


def _replace_and_fsync(candidate: Path, destination: Path) -> None:
    os.replace(candidate, destination)
    fsync_directory(destination.parent)


def _restore_snapshot(snapshot: Mapping[str, Any], destination: Path) -> None:
    source = Path(str(snapshot["path"]))
    candidate = _make_candidate(destination.parent, destination.name, ".rollback")
    try:
        shutil.copyfile(source, candidate)
        os.chmod(candidate, int(snapshot["source_mode"]), follow_symlinks=False)
        with candidate.open("rb") as handle:
            os.fsync(handle.fileno())
        _replace_and_fsync(candidate, destination)
    finally:
        candidate.unlink(missing_ok=True)


def migrate_stage4_extension_provenance(
    *,
    project_root: str | Path,
    expected_conditional_sha256: str,
    expected_gate_sha256: str,
    expected_old_source_map: Mapping[str, str],
    expected_new_source_map: Mapping[str, str],
    execute: bool = False,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Dry-run or publish the exact ACTIVATE_EXTENSION provenance migration."""

    _assert_cpu_only()
    _validate_sha(expected_conditional_sha256, field="conditional authorization")
    _validate_sha(expected_gate_sha256, field="gate receipt")
    old_sources, new_sources = _validate_source_transition(
        expected_old_source_map, expected_new_source_map
    )
    if execute and confirmation_token != MIGRATION_CONFIRMATION_TOKEN:
        _fail("migration publication requires the exact confirmation token")
    paths = _resolve_paths(project_root)
    migrations = paths["backup_dir"].parent
    if migrations.is_symlink() or not migrations.is_dir():
        _fail("canonical migrations directory is missing or symlinked")
    receipt_path = paths["backup_dir"] / RECEIPT_NAME
    if receipt_path.exists() or receipt_path.is_symlink():
        _fail("Stage4 extension migration receipt already exists")
    for forbidden in ("complete", "diagnostics_json", "diagnostics_report"):
        if paths[forbidden].exists() or paths[forbidden].is_symlink():
            _fail(f"Stage4 finalization artifact already exists: {forbidden}")

    with _single_writer_lock(migrations):
        gate, snapshots = _validate_gate_receipt(
            paths,
            expected_conditional_sha256=expected_conditional_sha256,
            expected_gate_sha256=expected_gate_sha256,
        )
        old_hashes = {
            label: snapshots[label]["sha256"]
            for label in ("run_contract", "last_checkpoint", "best_checkpoint")
        }
        # All gate-snapshotted live evidence must still be the exact stopped
        # boundary.  Only semantic source files may have moved to their audited
        # new hashes between gate publication and provenance migration.
        for label, evidence in snapshots.items():
            if sha256_file(paths[label]) != evidence["sha256"]:
                _fail(f"live artifact changed after gate publication: {label}")
        physical_sources = dict(
            semantic_source_hashes(paths["project_root"], entrypoints=ENTRYPOINTS)
        )
        if physical_sources != new_sources:
            _fail("physical semantic-source map differs from the exact new map")

        run = _mapping(load_json(paths["run_contract"]), field="run contract")
        last = _load_checkpoint(paths["last_checkpoint"])
        best = _load_checkpoint(paths["best_checkpoint"])
        _validate_checkpoint_pair(
            last, best, expected_best_sha256=old_hashes["best_checkpoint"]
        )
        old_provenance = _mapping(run.get("provenance"), field="run provenance")
        if (
            last.get("provenance") != old_provenance
            or best.get("provenance") != old_provenance
        ):
            _fail("pre-migration run/last/best provenance differs")
        binding = _extension_binding(
            paths,
            conditional_sha256=expected_conditional_sha256,
            gate_sha256=expected_gate_sha256,
        )
        new_provenance = _build_new_provenance(
            old_provenance,
            old_source_map=old_sources,
            new_source_map=new_sources,
            extension_binding=binding,
        )

        candidates = {
            "run_contract": _make_candidate(
                paths["run_contract"].parent, "run_contract", ".extension.json"
            ),
            "best_checkpoint": _make_candidate(
                paths["best_checkpoint"].parent, "best_ema", ".extension.pth"
            ),
            "last_checkpoint": _make_candidate(
                paths["last_checkpoint"].parent, "last", ".extension.pth"
            ),
        }
        try:
            new_run = copy.deepcopy(dict(run))
            new_run["provenance"] = new_provenance
            new_best = copy.copy(best)
            new_best["provenance"] = new_provenance
            atomic_write_json(candidates["run_contract"], new_run)
            atomic_torch_save(new_best, candidates["best_checkpoint"])
            new_best_sha = sha256_file(candidates["best_checkpoint"])
            new_last = copy.copy(last)
            new_last["provenance"] = new_provenance
            new_last_metrics = copy.deepcopy(
                dict(_mapping(last["metrics"], field="last.metrics"))
            )
            new_last_metrics["best_checkpoint_sha256"] = new_best_sha
            new_last["metrics"] = new_last_metrics
            atomic_torch_save(new_last, candidates["last_checkpoint"])

            reloaded_run = _mapping(
                load_json(candidates["run_contract"]), field="candidate run"
            )
            reloaded_best = _load_checkpoint(candidates["best_checkpoint"])
            reloaded_last = _load_checkpoint(candidates["last_checkpoint"])
            _validate_checkpoint_pair(
                reloaded_last, reloaded_best, expected_best_sha256=new_best_sha
            )
            if (
                reloaded_run.get("provenance") != new_provenance
                or reloaded_best.get("provenance") != new_provenance
                or reloaded_last.get("provenance") != new_provenance
            ):
                _fail("candidate run/last/best provenance identity failed")
            _assert_bit_exact(
                {k: v for k, v in run.items() if k != "provenance"},
                {k: v for k, v in reloaded_run.items() if k != "provenance"},
                path="run_contract.outside_provenance",
            )
            _assert_bit_exact(
                {k: v for k, v in best.items() if k != "provenance"},
                {k: v for k, v in reloaded_best.items() if k != "provenance"},
                path="best.outside_provenance",
            )
            for key in last:
                if key not in {"provenance", "metrics"}:
                    _assert_bit_exact(last[key], reloaded_last[key], path=f"last.{key}")
            for key in _mapping(last["metrics"], field="old last.metrics"):
                if key != "best_checkpoint_sha256":
                    _assert_bit_exact(
                        last["metrics"][key],
                        reloaded_last["metrics"][key],
                        path=f"last.metrics.{key}",
                    )
            if set(last["metrics"]) != set(reloaded_last["metrics"]):
                _fail("raw checkpoint metrics key set changed")

            new_hashes = {
                label: sha256_file(path) for label, path in candidates.items()
            }
            receipt: dict[str, Any] = {
                "schema_version": MIGRATION_RECEIPT_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "migration": MIGRATION_KIND,
                "status": "PREPARED" if execute else "DRY_RUN",
                "created_utc": utc_now_iso(),
                "cpu_only": True,
                "conditional_authorization": {
                    "path": str(paths["conditional"]),
                    "sha256": expected_conditional_sha256,
                },
                "gate_receipt": {
                    "path": str(paths["gate"]),
                    "sha256": expected_gate_sha256,
                    "decision": gate["decision"],
                },
                "base_step": BASE_STEP,
                "target_step": TARGET_STEP,
                "validation_steps": list(VALIDATION_STEPS),
                "schedule_horizon_steps": SCHEDULE_HORIZON_STEPS,
                "lr_policy": LR_POLICY,
                "old": old_hashes
                | {"provenance_json_sha256": sha256_json(dict(old_provenance))},
                "new": new_hashes
                | {"provenance_json_sha256": sha256_json(new_provenance)},
                "semantic_source_changed_paths": list(ALLOWED_CHANGED_SOURCE_PATHS),
                "run_contract_bit_exact_outside_provenance": True,
                "best_checkpoint_bit_exact_outside_provenance": True,
                "last_checkpoint_only_nonprovenance_change": (
                    "metrics.best_checkpoint_sha256"
                ),
                "optimizer_ema_scheduler_rng_sampler_reset": False,
                "snapshots": snapshots,
                "migration_script_sha256": sha256_file(Path(__file__).resolve()),
            }
            if not execute:
                return receipt

            for label, evidence in snapshots.items():
                if sha256_file(paths[label]) != evidence["sha256"]:
                    _fail(f"protected artifact changed before publication: {label}")
            if (
                dict(
                    semantic_source_hashes(
                        paths["project_root"], entrypoints=ENTRYPOINTS
                    )
                )
                != new_sources
            ):
                _fail("semantic sources changed before publication")
            atomic_write_json(receipt_path, receipt)
            published: list[str] = []
            try:
                for label in ("best_checkpoint", "last_checkpoint", "run_contract"):
                    _replace_and_fsync(candidates[label], paths[label])
                    published.append(label)
                    if sha256_file(paths[label]) != new_hashes[label]:
                        _fail(f"published {label} hash differs from candidate")
                pub_run = _mapping(
                    load_json(paths["run_contract"]), field="published run"
                )
                pub_best = _load_checkpoint(paths["best_checkpoint"])
                pub_last = _load_checkpoint(paths["last_checkpoint"])
                _validate_checkpoint_pair(
                    pub_last, pub_best, expected_best_sha256=new_best_sha
                )
                if (
                    pub_run.get("provenance") != new_provenance
                    or pub_best.get("provenance") != new_provenance
                    or pub_last.get("provenance") != new_provenance
                ):
                    _fail("published run/last/best provenance identity failed")
                for label, evidence in snapshots.items():
                    if (
                        label
                        not in {"run_contract", "last_checkpoint", "best_checkpoint"}
                        and sha256_file(paths[label]) != evidence["sha256"]
                    ):
                        _fail(f"protected artifact changed during publication: {label}")
                receipt["status"] = "COMPLETE"
                receipt["completed_utc"] = utc_now_iso()
                receipt["backup_read_only_after_publication"] = True
                atomic_write_json(receipt_path, receipt)
                os.chmod(receipt_path, 0o444, follow_symlinks=False)
                with receipt_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                fsync_directory(receipt_path.parent)
                if (
                    receipt_path.is_symlink()
                    or not receipt_path.is_file()
                    or stat.S_IMODE(receipt_path.stat().st_mode) != 0o444
                    or dict(load_json(receipt_path)) != receipt
                ):
                    _fail("COMPLETE migration receipt publication verification failed")
                _assert_cpu_only()
                return receipt
            except BaseException as original_error:
                rollback_errors: list[str] = []
                for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
                    try:
                        _restore_snapshot(snapshots[label], paths[label])
                        if sha256_file(paths[label]) != snapshots[label]["sha256"]:
                            _fail(f"rollback hash mismatch: {label}")
                    except BaseException as exc:
                        rollback_errors.append(f"{label}: {type(exc).__name__}: {exc}")
                atomic_write_json(
                    receipt_path,
                    {
                        "schema_version": MIGRATION_RECEIPT_SCHEMA,
                        "protocol_id": PROTOCOL_ID,
                        "migration": MIGRATION_KIND,
                        "status": "ROLLBACK_FAILED"
                        if rollback_errors
                        else "ROLLED_BACK",
                        "rolled_back_utc": utc_now_iso(),
                        "published_before_failure": published,
                        "rollback_errors": rollback_errors,
                        "snapshots": snapshots,
                    },
                )
                if rollback_errors:
                    raise Stage4ExtensionMigrationError(
                        "publication failed and rollback was incomplete: "
                        + "; ".join(rollback_errors)
                    ) from original_error
                raise
        finally:
            for candidate in candidates.values():
                candidate.unlink(missing_ok=True)


def _load_source_map(path: str, *, field: str) -> dict[str, str]:
    raw = load_json(_canonical_path(path, label=field))
    return _validate_source_map(_mapping(raw, field=field), field=field)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    gate = subparsers.add_parser("gate", help="evaluate/publish the 40k Decimal gate")
    for name in (
        "conditional",
        "run-contract",
        "last-checkpoint",
        "best-checkpoint",
        "calibration-history",
        "validation-latest",
        "report",
        "train-log",
        "state",
        "pipeline-log",
        "config",
    ):
        gate.add_argument(f"--expected-{name}-sha256", required=True)
    gate.add_argument("--execute", action="store_true")
    gate.add_argument("--confirmation-token")

    migration = subparsers.add_parser("migrate", help="publish ACTIVATE provenance")
    migration.add_argument("--expected-conditional-sha256", required=True)
    migration.add_argument("--expected-gate-sha256", required=True)
    migration.add_argument("--old-source-map-json", required=True)
    migration.add_argument("--new-source-map-json", required=True)
    migration.add_argument("--execute", action="store_true")
    migration.add_argument("--confirmation-token")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "gate":
            result = evaluate_stage4_extension_gate(
                project_root=args.project_root,
                expected_conditional_sha256=args.expected_conditional_sha256,
                expected_run_contract_sha256=args.expected_run_contract_sha256,
                expected_last_checkpoint_sha256=args.expected_last_checkpoint_sha256,
                expected_best_checkpoint_sha256=args.expected_best_checkpoint_sha256,
                expected_calibration_history_sha256=args.expected_calibration_history_sha256,
                expected_validation_latest_sha256=args.expected_validation_latest_sha256,
                expected_report_sha256=args.expected_report_sha256,
                expected_train_log_sha256=args.expected_train_log_sha256,
                expected_state_sha256=args.expected_state_sha256,
                expected_pipeline_log_sha256=args.expected_pipeline_log_sha256,
                expected_config_sha256=args.expected_config_sha256,
                execute=args.execute,
                confirmation_token=args.confirmation_token,
            )
        else:
            result = migrate_stage4_extension_provenance(
                project_root=args.project_root,
                expected_conditional_sha256=args.expected_conditional_sha256,
                expected_gate_sha256=args.expected_gate_sha256,
                expected_old_source_map=_load_source_map(
                    args.old_source_map_json, field="old source map JSON"
                ),
                expected_new_source_map=_load_source_map(
                    args.new_source_map_json, field="new source map JSON"
                ),
                execute=args.execute,
                confirmation_token=args.confirmation_token,
            )
    except Stage4ExtensionMigrationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
