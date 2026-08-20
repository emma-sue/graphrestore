#!/usr/bin/env python3
"""One-time Stage3 pending-step guard-alignment provenance migration.

The Stage3 step-2000 optimizer transaction is scientifically valid, but its
first validation failed after all 1,600 examples because diagnostic guard maps
did not crop the model's right/bottom padding.  This CPU-only tool permits the
resulting code repair to change exactly two semantic-source SHA leaves:

* ``src/training/stage3_engine.py``
* ``src/training/stage4_engine.py``

No training state may change.  The run contract and pending raw checkpoint are
updated to the same provenance mapping only after candidates round-trip and
every non-provenance checkpoint section compares bit-for-bit.  Execution is
opt-in behind an exact confirmation token; without ``--execute`` the command is
a non-publishing dry run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

# Enforce CPU-only behavior before importing torch.  The migration never calls
# a torch.cuda API and all checkpoint loads explicitly map to CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.provenance import semantic_source_hashes  # noqa: E402
from src.utils.hashing import is_sha256, sha256_file, sha256_json  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    fsync_directory,
    load_json,
    utc_now_iso,
)


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
CHECKPOINT_SCHEMA = "graphrestore-checkpoint-v1"
STAGE3_SCHEMA = "graphrestore-stage3-runtime-v1"
APPROVAL_SCHEMA = "graphrestore-stage3-approval-v1"
RECEIPT_SCHEMA = "graphrestore-stage3-guard-alignment-migration-v1"
MIGRATION_STEP = 2_000
EXPECTED_BINDING_COUNT = 22
CONFIRMATION_TOKEN = "MIGRATE_STAGE3_PENDING_2000_GUARD_ALIGNMENT"
RECOVERY_CONFIRMATION_TOKEN = "RECOVER_STAGE3_PENDING_2000_GUARD_ALIGNMENT"
ENTRYPOINTS = (
    "scripts/train_stage3_planner.py",
    "scripts/eval_guard_diagnostics.py",
)
ALLOWED_SOURCE_PATHS = (
    "src/training/stage3_engine.py",
    "src/training/stage4_engine.py",
)
EXPECTED_PROVENANCE_DIFF_PATHS = tuple(
    f"semantic_source_sha256.{path}" for path in ALLOWED_SOURCE_PATHS
)


class Stage3GuardAlignmentMigrationError(RuntimeError):
    """The requested one-time migration does not satisfy its exact contract."""


def _fail(message: str) -> NoReturn:
    raise Stage3GuardAlignmentMigrationError(message)


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _qualified_type(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _tensor_raw_bytes(value: torch.Tensor) -> bytes:
    if value.layout is not torch.strided:
        _fail(f"unsupported tensor layout: {value.layout}")
    flat = value.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


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
            _fail(f"non-finite numpy array at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_finite(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_finite(child, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"non-finite Python float at {path}")


def _update_fingerprint(
    digest: Any,
    value: object,
    counts: Counter[str],
) -> None:
    counts["nodes"] += 1
    type_name = _qualified_type(value).encode("utf-8")
    digest.update(struct.pack(">I", len(type_name)))
    digest.update(type_name)
    if isinstance(value, torch.Tensor):
        raw = _tensor_raw_bytes(value)
        counts["tensors"] += 1
        counts["tensor_numel"] += value.numel()
        counts["tensor_bytes"] += len(raw)
        metadata = json.dumps(
            {
                "dtype": str(value.dtype),
                "layout": str(value.layout),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "requires_grad": value.requires_grad,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
        return
    if isinstance(value, np.ndarray):
        raw = value.tobytes(order="A")
        counts["numpy_arrays"] += 1
        counts["numpy_bytes"] += len(raw)
        metadata = json.dumps(
            {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "strides": list(value.strides),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
        return
    if isinstance(value, Mapping):
        counts["mappings"] += 1
        digest.update(struct.pack(">Q", len(value)))
        for key, child in value.items():
            _update_fingerprint(digest, key, counts)
            _update_fingerprint(digest, child, counts)
        return
    if isinstance(value, (list, tuple)):
        counts["sequences"] += 1
        digest.update(struct.pack(">Q", len(value)))
        for child in value:
            _update_fingerprint(digest, child, counts)
        return
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bool):
        digest.update(b"true" if value else b"false")
        return
    if isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        return
    if isinstance(value, float):
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        return
    _fail(f"unsupported value type: {_qualified_type(value)}")


def _fingerprint(value: object) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    _update_fingerprint(digest, value, counts)
    return {"sha256": digest.hexdigest(), "counts": dict(sorted(counts.items()))}


def _assert_bit_exact(before: object, after: object, *, path: str) -> None:
    if type(before) is not type(after):
        _fail(
            f"state type mutation at {path}: "
            f"{_qualified_type(before)} != {_qualified_type(after)}"
        )
    if isinstance(before, torch.Tensor):
        assert isinstance(after, torch.Tensor)
        old_metadata = (
            before.dtype,
            before.layout,
            tuple(before.shape),
            tuple(before.stride()),
            before.storage_offset(),
            before.requires_grad,
        )
        new_metadata = (
            after.dtype,
            after.layout,
            tuple(after.shape),
            tuple(after.stride()),
            after.storage_offset(),
            after.requires_grad,
        )
        if old_metadata != new_metadata or _tensor_raw_bytes(
            before
        ) != _tensor_raw_bytes(after):
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
            _fail(f"numpy state mutation at {path}")
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
        if struct.pack(">d", before) != struct.pack(">d", after):
            _fail(f"float mutation at {path}")
        return
    if before != after:
        _fail(f"state mutation at {path}: {before!r} != {after!r}")


def _flatten_provenance(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            _fail("provenance mappings require non-empty string keys")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            flattened.update(_flatten_provenance(child, prefix=path))
        else:
            flattened[path] = child
    return flattened


def _exact_provenance_diff(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> list[dict[str, str]]:
    old_flat = _flatten_provenance(old)
    new_flat = _flatten_provenance(new)
    if old_flat.keys() != new_flat.keys():
        _fail(
            "provenance leaf set changed: "
            f"only_old={sorted(old_flat.keys() - new_flat.keys())}, "
            f"only_new={sorted(new_flat.keys() - old_flat.keys())}"
        )
    changed = sorted(path for path in old_flat if old_flat[path] != new_flat[path])
    if changed != sorted(EXPECTED_PROVENANCE_DIFF_PATHS):
        _fail(
            "unexpected provenance diff: "
            f"expected={sorted(EXPECTED_PROVENANCE_DIFF_PATHS)}, actual={changed}"
        )
    result: list[dict[str, str]] = []
    for path in changed:
        old_value, new_value = old_flat[path], new_flat[path]
        if not is_sha256(old_value) or not is_sha256(new_value):
            _fail(f"provenance diff at {path} is not SHA256-to-SHA256")
        result.append({"path": path, "old": old_value, "new": new_value})
    return result


def _load_cpu_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise Stage3GuardAlignmentMigrationError(
            f"could not load checkpoint on CPU: {type(exc).__name__}: {exc}"
        ) from exc
    return _require_mapping(payload, field="checkpoint")


def _validate_checkpoint(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": "stage3",
        "step": MIGRATION_STEP,
        "model_role": "raw_training_state",
        "resumable": True,
        "pending_validation_step": MIGRATION_STEP,
        "optimizer_transaction_active": False,
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "scaler": None,
    }
    for key, value in expected.items():
        if payload.get(key, object()) != value:
            _fail(
                f"checkpoint header mismatch at {key}: "
                f"expected {value!r}, got {payload.get(key, '<missing>')!r}"
            )
    if payload.get("metrics") != {}:
        _fail("first pending Stage3 validation must still have empty metrics")
    sampler = _require_mapping(payload.get("sampler_state"), field="sampler_state")
    if (
        sampler.get("consumed_optimizer_step") != MIGRATION_STEP
        or sampler.get("sample_cursor") != MIGRATION_STEP * 8
    ):
        _fail("checkpoint sampler is not the exact step-2000/cursor-16000 boundary")
    ema = _require_mapping(payload.get("ema"), field="ema")
    if (
        ema.get("num_updates") != MIGRATION_STEP
        or ema.get("scope") != "planner_parameters_only_executor_bitwise_frozen"
    ):
        _fail("checkpoint EMA is not the exact Stage3 step-2000 policy")
    _walk_finite(payload)


def _validate_failed_state(path: Path) -> dict[str, Any]:
    state = _require_mapping(load_json(path), field="orchestration state")
    last_command = state.get("last_command")
    if (
        state.get("schema_version") != "graphrestore-orchestration-v1"
        or state.get("protocol_id") != PROTOCOL_ID
        or state.get("status") != "FAILED"
        or state.get("current_stage") != "FAILED"
        or state.get("gpu") != "released"
        or state.get("last_exit_code") != 3
        or state.get("next_command")
        != "python scripts/orchestrate.py --resume_post_approval_pipeline"
        or not isinstance(last_command, list)
        or "scripts/train_stage3_planner.py" not in last_command
    ):
        _fail("orchestration state is not the exact failed Stage3 recovery boundary")
    return dict(state)


def _validate_approval(
    approval_path: Path,
    approval_required_path: Path,
    *,
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
) -> dict[str, Any]:
    if sha256_file(approval_path) != expected_approval_sha256:
        _fail("Stage3 approval SHA256 drifted")
    if sha256_file(approval_required_path) != expected_approval_required_sha256:
        _fail("Stage3 approval-required SHA256 drifted")
    approval = _require_mapping(load_json(approval_path), field="Stage3 approval")
    required = _require_mapping(
        load_json(approval_required_path), field="Stage3 approval-required"
    )
    bindings = _require_mapping(approval.get("bindings"), field="approval.bindings")
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA
        or approval.get("kind") != "stage3_approval"
        or approval.get("protocol_id") != PROTOCOL_ID
        or approval.get("approved") is not True
        or approval.get("approval_required_sha256") != expected_approval_required_sha256
        or required.get("schema_version") != APPROVAL_SCHEMA
        or required.get("kind") != "stage3_approval_required"
        or required.get("protocol_id") != PROTOCOL_ID
        or required.get("approved") is not False
        or required.get("bindings") != bindings
        or len(bindings) != EXPECTED_BINDING_COUNT
    ):
        _fail("Stage3 approval/approval-required semantic contract drifted")
    verified: dict[str, str] = {}
    for logical, raw in bindings.items():
        binding = _require_mapping(raw, field=f"approval.bindings.{logical}")
        path_raw, expected_sha = binding.get("path"), binding.get("sha256")
        if not isinstance(path_raw, str) or not is_sha256(expected_sha):
            _fail(f"invalid approved binding: {logical}")
        path = Path(path_raw)
        if not path.is_file() or sha256_file(path) != expected_sha:
            _fail(f"approved binding changed: {logical}")
        verified[str(logical)] = str(expected_sha)
    return {
        "approval_sha256": expected_approval_sha256,
        "approval_required_sha256": expected_approval_required_sha256,
        "binding_count": len(bindings),
        "binding_sha256": verified,
    }


def _make_candidate(parent: Path, name: str, suffix: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=suffix, dir=parent
    )
    os.close(descriptor)
    candidate = Path(temporary_name)
    candidate.unlink()
    return candidate


def _section_evidence(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> dict[str, Any]:
    if list(old) != list(new):
        _fail("checkpoint top-level structure changed")
    result: dict[str, Any] = {}
    for key in old:
        old_fingerprint = _fingerprint(old[key])
        new_fingerprint = _fingerprint(new[key])
        if key != "provenance" and old_fingerprint != new_fingerprint:
            _fail(f"checkpoint section changed outside provenance: {key}")
        result[key] = {
            "old": old_fingerprint,
            "new": new_fingerprint,
            "bit_exact": old_fingerprint == new_fingerprint,
        }
    return result


def _hardlink_backup(source: Path, destination: Path) -> dict[str, Any]:
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise Stage3GuardAlignmentMigrationError(
            f"refusing existing backup: {destination}"
        ) from exc
    fsync_directory(destination.parent)
    source_stat, backup_stat = source.stat(), destination.stat()
    if (
        source_stat.st_dev != backup_stat.st_dev
        or source_stat.st_ino != backup_stat.st_ino
        or sha256_file(source) != sha256_file(destination)
    ):
        _fail(f"backup is not an exact same-disk hard link: {destination}")
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "mode": stat.S_IMODE(source_stat.st_mode),
        "device": backup_stat.st_dev,
        "inode": backup_stat.st_ino,
        "hard_link_verified": True,
    }


def _replace_and_fsync(candidate: Path, destination: Path) -> None:
    os.replace(candidate, destination)
    fsync_directory(destination.parent)


def _restore_from_backup(backup: Path, destination: Path, *, mode: int) -> None:
    rollback = _make_candidate(destination.parent, destination.name, ".rollback")
    try:
        shutil.copyfile(backup, rollback)
        os.chmod(rollback, mode, follow_symlinks=False)
        with rollback.open("rb") as stream:
            os.fsync(stream.fileno())
        _replace_and_fsync(rollback, destination)
    finally:
        rollback.unlink(missing_ok=True)


def _make_backup_read_only(path: Path) -> None:
    os.chmod(
        path,
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH,
        follow_symlinks=False,
    )
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def recover_prepared_stage3_guard_alignment(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    checkpoint: str | Path,
    state: str | Path,
    approval: str | Path,
    approval_required: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_checkpoint_sha256: str,
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
    expected_old_source_sha256: Mapping[str, str],
    expected_new_source_sha256: Mapping[str, str],
    confirmation_token: str | None,
) -> dict[str, Any]:
    """Recover both old files from an interrupted PREPARED transaction."""

    if confirmation_token != RECOVERY_CONFIRMATION_TOKEN:
        _fail("PREPARED recovery requires the exact recovery confirmation token")
    root = Path(project_root).resolve()
    contract_path = Path(run_contract).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    state_path = Path(state).resolve()
    approval_path = Path(approval).resolve()
    required_path = Path(approval_required).resolve()
    backup_requested = Path(backup_dir)
    artifacts_requested = root / "artifacts"
    migrations_requested = artifacts_requested / "migrations"
    if (
        artifacts_requested.is_symlink()
        or migrations_requested.is_symlink()
        or backup_requested.is_symlink()
    ):
        _fail("PREPARED recovery path must not traverse a symlink")
    backup_path = backup_requested.resolve()
    if backup_path.parent != migrations_requested.resolve():
        _fail("PREPARED recovery backup must be under project artifacts/migrations")
    receipt_path = backup_path / "MIGRATION_RECEIPT.json"
    if not backup_path.is_dir() or not receipt_path.is_file():
        _fail("PREPARED recovery requires an existing backup directory and receipt")
    if (
        set(expected_old_source_sha256) != set(ALLOWED_SOURCE_PATHS)
        or set(expected_new_source_sha256) != set(ALLOWED_SOURCE_PATHS)
        or any(
            not is_sha256(value)
            for value in (
                *expected_old_source_sha256.values(),
                *expected_new_source_sha256.values(),
            )
        )
    ):
        _fail("recovery source expectations must name exact Stage3/Stage4 SHA256")
    expected_hashes = (
        expected_run_contract_sha256,
        expected_checkpoint_sha256,
        expected_approval_sha256,
        expected_approval_required_sha256,
    )
    if any(not is_sha256(value) for value in expected_hashes):
        _fail("every recovery anchor must be a lowercase SHA256")

    state_before_sha = sha256_file(state_path)
    _validate_failed_state(state_path)
    _validate_approval(
        approval_path,
        required_path,
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_approval_required_sha256,
    )
    current_semantic = semantic_source_hashes(root, entrypoints=ENTRYPOINTS)
    for relative in ALLOWED_SOURCE_PATHS:
        if current_semantic.get(relative) != expected_new_source_sha256[relative]:
            _fail(f"recovery semantic source drifted at {relative}")

    receipt = _require_mapping(load_json(receipt_path), field="migration receipt")
    old = _require_mapping(receipt.get("old"), field="migration receipt.old")
    new = _require_mapping(receipt.get("new"), field="migration receipt.new")
    receipt_backups = _require_mapping(
        receipt.get("backup"), field="migration receipt.backup"
    )
    receipt_approval = _require_mapping(
        receipt.get("approval_and_bindings_unchanged"),
        field="migration receipt.approval_and_bindings_unchanged",
    )
    old_contract = _require_mapping(
        old.get("run_contract"), field="migration receipt.old.run_contract"
    )
    old_checkpoint = _require_mapping(
        old.get("checkpoint"), field="migration receipt.old.checkpoint"
    )
    raw_diff = receipt.get("exact_provenance_leaf_diff")
    if not isinstance(raw_diff, list):
        _fail("PREPARED receipt provenance diff must be a list")
    expected_diff = [
        {
            "path": f"semantic_source_sha256.{relative}",
            "old": expected_old_source_sha256[relative],
            "new": expected_new_source_sha256[relative],
        }
        for relative in sorted(ALLOWED_SOURCE_PATHS)
    ]
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("migration")
        != "stage3_pending_2000_guard_alignment_provenance_only"
        or receipt.get("status") not in {"PREPARED", "ROLLED_BACK_FROM_PREPARED"}
        or receipt.get("migration_script_sha256")
        != sha256_file(Path(__file__).resolve())
        or receipt.get("orchestration_state_sha256") != state_before_sha
        or receipt_approval.get("approval_sha256") != expected_approval_sha256
        or receipt_approval.get("approval_required_sha256")
        != expected_approval_required_sha256
        or receipt_approval.get("binding_count") != EXPECTED_BINDING_COUNT
        or old_contract.get("path") != str(contract_path)
        or old_contract.get("sha256") != expected_run_contract_sha256
        or old_checkpoint.get("path") != str(checkpoint_path)
        or old_checkpoint.get("sha256") != expected_checkpoint_sha256
        or not is_sha256(new.get("run_contract_sha256"))
        or not is_sha256(new.get("checkpoint_sha256"))
        or raw_diff != expected_diff
    ):
        _fail("PREPARED receipt does not match the audited migration transaction")

    recovery_items: list[tuple[str, Path, Path, int, str, str]] = []
    for label, destination, expected_old, expected_new in (
        (
            "run_contract",
            contract_path,
            expected_run_contract_sha256,
            str(new["run_contract_sha256"]),
        ),
        (
            "checkpoint",
            checkpoint_path,
            expected_checkpoint_sha256,
            str(new["checkpoint_sha256"]),
        ),
    ):
        backup = _require_mapping(
            receipt_backups.get(label), field=f"migration receipt.backup.{label}"
        )
        backup_raw, backup_sha, backup_mode = (
            backup.get("path"),
            backup.get("sha256"),
            backup.get("mode"),
        )
        if not isinstance(backup_raw, str) or not isinstance(backup_mode, int):
            _fail(f"invalid PREPARED backup evidence for {label}")
        backup_file = Path(backup_raw).resolve()
        if (
            backup_file.parent != backup_path
            or backup_sha != expected_old
            or not backup_file.is_file()
            or sha256_file(backup_file) != expected_old
            or backup_mode < 0
            or backup_mode > 0o777
            or stat.S_IMODE(backup_file.stat().st_mode) not in {backup_mode, 0o444}
        ):
            _fail(f"PREPARED backup drifted for {label}")
        live_sha = sha256_file(destination)
        if live_sha not in {expected_old, expected_new}:
            _fail(f"PREPARED live {label} is neither audited old nor candidate new")
        recovery_items.append(
            (
                label,
                destination,
                backup_file,
                backup_mode,
                expected_old,
                live_sha,
            )
        )

    contract_backup_file = next(
        item[2] for item in recovery_items if item[0] == "run_contract"
    )
    backup_contract = _require_mapping(
        load_json(contract_backup_file), field="PREPARED run-contract backup"
    )
    backup_provenance = _require_mapping(
        backup_contract.get("provenance"),
        field="PREPARED run-contract backup provenance",
    )
    backup_semantic = _require_mapping(
        backup_provenance.get("semantic_source_sha256"),
        field="PREPARED run-contract backup semantic sources",
    )
    if backup_semantic.keys() != current_semantic.keys():
        _fail("PREPARED recovery semantic-source path set changed")
    physical_diffs = sorted(
        path
        for path in backup_semantic
        if backup_semantic[path] != current_semantic[path]
    )
    if physical_diffs != sorted(ALLOWED_SOURCE_PATHS):
        _fail(
            "PREPARED recovery physical source drift is not exactly "
            f"Stage3+Stage4: {physical_diffs}"
        )
    for relative in ALLOWED_SOURCE_PATHS:
        if backup_semantic.get(relative) != expected_old_source_sha256[relative]:
            _fail(f"PREPARED old semantic source drifted at {relative}")

    def validate_recovered_files() -> None:
        for (
            label,
            destination,
            backup_file,
            mode,
            expected_old,
            _,
        ) in recovery_items:
            if (
                sha256_file(destination) != expected_old
                or stat.S_IMODE(destination.stat().st_mode) != mode
                or sha256_file(backup_file) != expected_old
                or stat.S_IMODE(backup_file.stat().st_mode) != 0o444
                or os.path.samestat(destination.stat(), backup_file.stat())
            ):
                _fail(f"recovered PREPARED files drifted for {label}")

    if receipt.get("status") == "ROLLED_BACK_FROM_PREPARED":
        recovered_from = _require_mapping(
            receipt.get("recovered_from_live_sha256"),
            field="finalized recovery live SHA evidence",
        )
        expected_recovery_token_sha = hashlib.sha256(
            RECOVERY_CONFIRMATION_TOKEN.encode("utf-8")
        ).hexdigest()
        if (
            not isinstance(receipt.get("recovered_utc"), str)
            or not receipt.get("recovered_utc")
            or receipt.get("recovery_confirmation_token_sha256")
            != expected_recovery_token_sha
            or receipt.get("backup_read_only_after_recovery") is not True
            or set(recovered_from) != {"run_contract", "checkpoint"}
        ):
            _fail("finalized PREPARED recovery evidence is incomplete")
        for (
            label,
            destination,
            backup_file,
            mode,
            expected_old,
            _,
        ) in recovery_items:
            allowed_initial = {
                expected_old,
                str(
                    new[
                        "run_contract_sha256"
                        if label == "run_contract"
                        else "checkpoint_sha256"
                    ]
                ),
            }
            if recovered_from.get(label) not in allowed_initial:
                _fail(f"finalized PREPARED recovery drifted for {label}")
        validate_recovered_files()
        return dict(receipt)

    recovered_from = {label: live_sha for label, *_, live_sha in recovery_items}
    for label, destination, backup_file, mode, expected_old, _ in recovery_items:
        _restore_from_backup(backup_file, destination, mode=mode)
        if sha256_file(destination) != expected_old:
            _fail(f"PREPARED recovery SHA256 mismatch for {label}")
        if os.path.samestat(destination.stat(), backup_file.stat()):
            _fail(f"PREPARED recovery still aliases {label} backup inode")
        if stat.S_IMODE(destination.stat().st_mode) != mode:
            _fail(f"PREPARED recovery mode mismatch for {label}")
        _make_backup_read_only(backup_file)
    fsync_directory(backup_path)
    if sha256_file(state_path) != state_before_sha:
        _fail("orchestration state changed during PREPARED recovery")
    if semantic_source_hashes(root, entrypoints=ENTRYPOINTS) != current_semantic:
        _fail("semantic sources changed during PREPARED recovery")
    _validate_approval(
        approval_path,
        required_path,
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_approval_required_sha256,
    )
    validate_recovered_files()
    recovered_receipt = dict(receipt)
    recovered_receipt["status"] = "ROLLED_BACK_FROM_PREPARED"
    recovered_receipt["recovered_utc"] = utc_now_iso()
    recovered_receipt["recovered_from_live_sha256"] = recovered_from
    recovered_receipt["recovery_confirmation_token_sha256"] = hashlib.sha256(
        RECOVERY_CONFIRMATION_TOKEN.encode("utf-8")
    ).hexdigest()
    recovered_receipt["backup_read_only_after_recovery"] = True
    atomic_write_json(receipt_path, recovered_receipt)
    return recovered_receipt


def migrate_stage3_guard_alignment(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    checkpoint: str | Path,
    state: str | Path,
    approval: str | Path,
    approval_required: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_checkpoint_sha256: str,
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
    expected_old_source_sha256: Mapping[str, str],
    expected_new_source_sha256: Mapping[str, str],
    execute: bool = False,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Prepare or execute the exact two-leaf Stage3 provenance migration."""

    root = Path(project_root).resolve()
    contract_path = Path(run_contract).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    state_path = Path(state).resolve()
    approval_path = Path(approval).resolve()
    required_path = Path(approval_required).resolve()
    backup_path = Path(backup_dir).resolve()
    artifacts_requested = root / "artifacts"
    migrations_requested = artifacts_requested / "migrations"
    if artifacts_requested.is_symlink() or migrations_requested.is_symlink():
        _fail("artifacts/migrations backup path must not traverse a symlink")
    expected_backup_parent = migrations_requested.resolve()
    if backup_path.parent != expected_backup_parent:
        _fail(f"backup directory must be one direct child of {expected_backup_parent}")
    expected_hashes = (
        expected_run_contract_sha256,
        expected_checkpoint_sha256,
        expected_approval_sha256,
        expected_approval_required_sha256,
        *expected_old_source_sha256.values(),
        *expected_new_source_sha256.values(),
    )
    if any(not is_sha256(value) for value in expected_hashes):
        _fail("every expected hash must be a lowercase SHA256")
    if set(expected_old_source_sha256) != set(ALLOWED_SOURCE_PATHS) or set(
        expected_new_source_sha256
    ) != set(ALLOWED_SOURCE_PATHS):
        _fail("old/new source expectations must name exactly Stage3 and Stage4")
    if execute and confirmation_token != CONFIRMATION_TOKEN:
        _fail("execution requires the exact Stage3 migration confirmation token")
    for path, label in (
        (contract_path, "run contract"),
        (checkpoint_path, "checkpoint"),
        (state_path, "orchestration state"),
        (approval_path, "approval"),
        (required_path, "approval-required"),
    ):
        if not path.is_file():
            _fail(f"missing {label}: {path}")
    if backup_path.exists():
        _fail(f"backup directory already exists: {backup_path}")
    if sha256_file(contract_path) != expected_run_contract_sha256:
        _fail("run-contract SHA256 differs from the audited pre-migration anchor")
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        _fail("checkpoint SHA256 differs from the audited pre-migration anchor")

    state_before_sha = sha256_file(state_path)
    state_evidence = _validate_failed_state(state_path)
    approval_evidence = _validate_approval(
        approval_path,
        required_path,
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_approval_required_sha256,
    )
    contract = _require_mapping(load_json(contract_path), field="run contract")
    if contract.get("schema_version") != STAGE3_SCHEMA:
        _fail("Stage3 run-contract schema drifted")
    old_provenance = _require_mapping(
        contract.get("provenance"), field="run contract provenance"
    )
    checkpoint_payload = _load_cpu_checkpoint(checkpoint_path)
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        _fail("checkpoint changed during CPU load")
    _validate_checkpoint(checkpoint_payload)
    checkpoint_provenance = _require_mapping(
        checkpoint_payload.get("provenance"), field="checkpoint provenance"
    )
    if checkpoint_provenance != old_provenance:
        _fail("run-contract and checkpoint provenance differ before migration")
    stage3_approval = _require_mapping(
        old_provenance.get("stage3_approval"), field="provenance.stage3_approval"
    )
    if (
        old_provenance.get("protocol_id") != PROTOCOL_ID
        or stage3_approval.get("sha256") != expected_approval_sha256
        or stage3_approval.get("approval_required_sha256")
        != expected_approval_required_sha256
        or old_provenance.get("bindings")
        != _require_mapping(load_json(approval_path), field="approval").get("bindings")
    ):
        _fail("checkpoint provenance approval/binding identity drifted")

    old_semantic = _require_mapping(
        old_provenance.get("semantic_source_sha256"),
        field="provenance.semantic_source_sha256",
    )
    current_semantic = semantic_source_hashes(root, entrypoints=ENTRYPOINTS)
    if old_semantic.keys() != current_semantic.keys():
        _fail("semantic-source path set changed")
    physical_diffs = sorted(
        path for path in old_semantic if old_semantic[path] != current_semantic[path]
    )
    if physical_diffs != sorted(ALLOWED_SOURCE_PATHS):
        _fail(
            "physical semantic-source drift is not exactly Stage3+Stage4: "
            f"{physical_diffs}"
        )
    for path in ALLOWED_SOURCE_PATHS:
        if old_semantic.get(path) != expected_old_source_sha256[path]:
            _fail(f"old source SHA differs at {path}")
        if current_semantic.get(path) != expected_new_source_sha256[path]:
            _fail(f"current source SHA differs at {path}")
        if expected_old_source_sha256[path] == expected_new_source_sha256[path]:
            _fail(f"source repair did not change SHA at {path}")

    new_provenance = copy.deepcopy(dict(old_provenance))
    new_semantic = dict(old_semantic)
    for path in ALLOWED_SOURCE_PATHS:
        new_semantic[path] = current_semantic[path]
    new_provenance["semantic_source_sha256"] = new_semantic
    provenance_diff = _exact_provenance_diff(old_provenance, new_provenance)

    new_contract = copy.deepcopy(dict(contract))
    new_contract["provenance"] = new_provenance
    _assert_bit_exact(
        {key: value for key, value in contract.items() if key != "provenance"},
        {key: value for key, value in new_contract.items() if key != "provenance"},
        path="run_contract.outside_provenance",
    )
    new_checkpoint = copy.copy(checkpoint_payload)
    new_checkpoint["provenance"] = new_provenance
    for key in checkpoint_payload:
        if key != "provenance":
            _assert_bit_exact(
                checkpoint_payload[key],
                new_checkpoint[key],
                path=f"checkpoint.{key}",
            )

    contract_candidate = _make_candidate(
        contract_path.parent, contract_path.name, ".candidate.json"
    )
    checkpoint_candidate = _make_candidate(
        checkpoint_path.parent, checkpoint_path.name, ".candidate.pth"
    )
    backups: dict[str, Any] = {}
    receipt_path = backup_path / "MIGRATION_RECEIPT.json"
    try:
        atomic_write_json(contract_candidate, new_contract)
        if load_json(contract_candidate) != new_contract:
            _fail("run-contract candidate failed JSON round trip")
        atomic_torch_save(new_checkpoint, checkpoint_candidate)
        reloaded = _load_cpu_checkpoint(checkpoint_candidate)
        _validate_checkpoint(reloaded)
        if (
            _require_mapping(reloaded.get("provenance"), field="candidate provenance")
            != new_provenance
        ):
            _fail("checkpoint candidate provenance differs from run contract")
        section_evidence = _section_evidence(checkpoint_payload, reloaded)
        new_contract_sha = sha256_file(contract_candidate)
        new_checkpoint_sha = sha256_file(checkpoint_candidate)
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "migration": "stage3_pending_2000_guard_alignment_provenance_only",
            "status": "DRY_RUN" if not execute else "PREPARED",
            "created_utc": utc_now_iso(),
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "step": MIGRATION_STEP,
            "pending_validation_step": MIGRATION_STEP,
            "old": {
                "run_contract": {
                    "path": str(contract_path),
                    "sha256": expected_run_contract_sha256,
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": expected_checkpoint_sha256,
                },
                "provenance_json_sha256": sha256_json(dict(old_provenance)),
            },
            "new": {
                "run_contract_sha256": new_contract_sha,
                "checkpoint_sha256": new_checkpoint_sha,
                "provenance_json_sha256": sha256_json(new_provenance),
            },
            "exact_provenance_leaf_diff": provenance_diff,
            "checkpoint_section_fingerprints": section_evidence,
            "checkpoint_state_bit_exact_outside_provenance": True,
            "run_contract_bit_exact_outside_provenance": True,
            "all_checkpoint_tensors_finite": True,
            "approval_and_bindings_unchanged": approval_evidence,
            "orchestration_state_sha256": state_before_sha,
            "orchestration_state": {
                key: state_evidence.get(key)
                for key in (
                    "status",
                    "current_stage",
                    "gpu",
                    "last_exit_code",
                    "next_command",
                )
            },
            "backup": backups,
            "migration_script_sha256": sha256_file(Path(__file__).resolve()),
        }
        if not execute:
            return receipt

        # Repeat all mutable external gates immediately before publication.
        if (
            sha256_file(contract_path) != expected_run_contract_sha256
            or sha256_file(checkpoint_path) != expected_checkpoint_sha256
            or sha256_file(state_path) != state_before_sha
        ):
            _fail("migration inputs changed before publication")
        _validate_failed_state(state_path)
        _validate_approval(
            approval_path,
            required_path,
            expected_approval_sha256=expected_approval_sha256,
            expected_approval_required_sha256=expected_approval_required_sha256,
        )

        artifacts_path = artifacts_requested.resolve()
        if not artifacts_path.is_dir():
            _fail(f"missing project artifacts directory: {artifacts_path}")
        expected_backup_parent.mkdir(parents=False, exist_ok=True)
        fsync_directory(artifacts_path)
        backup_path.mkdir(parents=False, exist_ok=False)
        fsync_directory(backup_path.parent)
        if (
            backup_path.stat().st_dev != contract_path.stat().st_dev
            or backup_path.stat().st_dev != checkpoint_path.stat().st_dev
        ):
            _fail("backup directory is not on the same filesystem as both sources")
        contract_backup = backup_path / (
            f"run_contract.pre_guard_alignment.{expected_run_contract_sha256}.json"
        )
        checkpoint_backup = backup_path / (
            f"last.pre_guard_alignment.{expected_checkpoint_sha256}.pth"
        )
        backups["run_contract"] = _hardlink_backup(contract_path, contract_backup)
        backups["checkpoint"] = _hardlink_backup(checkpoint_path, checkpoint_backup)
        if (
            backups["run_contract"]["sha256"] != expected_run_contract_sha256
            or backups["checkpoint"]["sha256"] != expected_checkpoint_sha256
        ):
            _fail("same-disk backup differs from an audited pre-migration anchor")
        receipt["backup"] = backups
        atomic_write_json(receipt_path, receipt)

        _replace_and_fsync(checkpoint_candidate, checkpoint_path)
        _replace_and_fsync(contract_candidate, contract_path)
        if (
            sha256_file(checkpoint_path) != new_checkpoint_sha
            or sha256_file(contract_path) != new_contract_sha
        ):
            _fail("published migration hash differs from verified candidate")
        published_contract_value = _require_mapping(
            load_json(contract_path), field="published run contract"
        )
        published_checkpoint_value = _load_cpu_checkpoint(checkpoint_path)
        if (
            published_contract_value.get("provenance") != new_provenance
            or published_checkpoint_value.get("provenance") != new_provenance
        ):
            _fail("published provenance pair is not identical")
        _section_evidence(checkpoint_payload, published_checkpoint_value)
        _assert_bit_exact(
            {key: value for key, value in contract.items() if key != "provenance"},
            {
                key: value
                for key, value in published_contract_value.items()
                if key != "provenance"
            },
            path="published_run_contract.outside_provenance",
        )
        if semantic_source_hashes(root, entrypoints=ENTRYPOINTS) != current_semantic:
            _fail("semantic sources changed during migration")
        _validate_approval(
            approval_path,
            required_path,
            expected_approval_sha256=expected_approval_sha256,
            expected_approval_required_sha256=(expected_approval_required_sha256),
        )
        for backup in (contract_backup, checkpoint_backup):
            _make_backup_read_only(backup)
        fsync_directory(backup_path)
        if sha256_file(state_path) != state_before_sha:
            _fail("orchestration state changed during migration")
        receipt["status"] = "COMPLETE"
        receipt["completed_utc"] = utc_now_iso()
        receipt["backup_read_only_after_publication"] = True
        atomic_write_json(receipt_path, receipt)
        return receipt
    except BaseException as original_error:
        if execute and backups:
            contract_backup_raw = backups.get("run_contract", {}).get("path")
            checkpoint_backup_raw = backups.get("checkpoint", {}).get("path")
            rollback_errors: list[str] = []
            for label, destination, backup_raw, backup_mode, expected_sha in (
                (
                    "run_contract",
                    contract_path,
                    contract_backup_raw,
                    backups.get("run_contract", {}).get("mode"),
                    expected_run_contract_sha256,
                ),
                (
                    "checkpoint",
                    checkpoint_path,
                    checkpoint_backup_raw,
                    backups.get("checkpoint", {}).get("mode"),
                    expected_checkpoint_sha256,
                ),
            ):
                if not isinstance(backup_raw, str) or not isinstance(backup_mode, int):
                    continue
                try:
                    backup_file = Path(backup_raw)
                    if sha256_file(backup_file) != expected_sha:
                        raise Stage3GuardAlignmentMigrationError(
                            f"{label} backup SHA256 mismatch"
                        )
                    _restore_from_backup(backup_file, destination, mode=backup_mode)
                    if sha256_file(destination) != expected_sha:
                        raise Stage3GuardAlignmentMigrationError(
                            f"{label} rollback SHA256 mismatch"
                        )
                    if os.path.samestat(destination.stat(), backup_file.stat()):
                        raise Stage3GuardAlignmentMigrationError(
                            f"{label} rollback still aliases backup inode"
                        )
                    if stat.S_IMODE(destination.stat().st_mode) != backup_mode:
                        raise Stage3GuardAlignmentMigrationError(
                            f"{label} rollback mode mismatch"
                        )
                    _make_backup_read_only(backup_file)
                except BaseException as rollback_error:
                    rollback_errors.append(
                        f"{label}: {type(rollback_error).__name__}: {rollback_error}"
                    )
            if receipt_path.parent.is_dir():
                rollback_receipt = {
                    "schema_version": RECEIPT_SCHEMA,
                    "protocol_id": PROTOCOL_ID,
                    "migration": "stage3_pending_2000_guard_alignment_provenance_only",
                    "status": ("ROLLBACK_FAILED" if rollback_errors else "ROLLED_BACK"),
                    "rolled_back_utc": utc_now_iso(),
                    "old_run_contract_sha256": expected_run_contract_sha256,
                    "old_checkpoint_sha256": expected_checkpoint_sha256,
                    "backup": backups,
                    "rollback_errors": rollback_errors,
                }
                atomic_write_json(receipt_path, rollback_receipt)
            if rollback_errors:
                raise Stage3GuardAlignmentMigrationError(
                    "migration publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from original_error
        raise
    finally:
        contract_candidate.unlink(missing_ok=True)
        checkpoint_candidate.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-required", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--expected-run-contract-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--expected-approval-required-sha256", required=True)
    for stage in ("stage3", "stage4"):
        parser.add_argument(f"--expected-old-{stage}-source-sha256", required=True)
        parser.add_argument(f"--expected-new-{stage}-source-sha256", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--recover-prepared", action="store_true")
    parser.add_argument("--confirmation-token")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        common = {
            "project_root": arguments.project_root,
            "run_contract": arguments.run_contract,
            "checkpoint": arguments.checkpoint,
            "state": arguments.state,
            "approval": arguments.approval,
            "approval_required": arguments.approval_required,
            "backup_dir": arguments.backup_dir,
            "expected_run_contract_sha256": (arguments.expected_run_contract_sha256),
            "expected_checkpoint_sha256": arguments.expected_checkpoint_sha256,
            "expected_approval_sha256": arguments.expected_approval_sha256,
            "expected_approval_required_sha256": (
                arguments.expected_approval_required_sha256
            ),
        }
        new_sources = {
            ALLOWED_SOURCE_PATHS[0]: arguments.expected_new_stage3_source_sha256,
            ALLOWED_SOURCE_PATHS[1]: arguments.expected_new_stage4_source_sha256,
        }
        if arguments.recover_prepared:
            receipt = recover_prepared_stage3_guard_alignment(
                **common,
                expected_old_source_sha256={
                    ALLOWED_SOURCE_PATHS[0]: (
                        arguments.expected_old_stage3_source_sha256
                    ),
                    ALLOWED_SOURCE_PATHS[1]: (
                        arguments.expected_old_stage4_source_sha256
                    ),
                },
                expected_new_source_sha256=new_sources,
                confirmation_token=arguments.confirmation_token,
            )
        else:
            receipt = migrate_stage3_guard_alignment(
                **common,
                expected_old_source_sha256={
                    ALLOWED_SOURCE_PATHS[0]: (
                        arguments.expected_old_stage3_source_sha256
                    ),
                    ALLOWED_SOURCE_PATHS[1]: (
                        arguments.expected_old_stage4_source_sha256
                    ),
                },
                expected_new_source_sha256=new_sources,
                execute=arguments.execute,
                confirmation_token=arguments.confirmation_token,
            )
    except Stage3GuardAlignmentMigrationError as exc:
        print(f"Stage3 guard-alignment migration refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
