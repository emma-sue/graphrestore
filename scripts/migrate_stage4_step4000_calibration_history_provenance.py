#!/usr/bin/env python3
"""Audited Stage4 step-4000 calibration-history routing migration.

The first formal Stage4 validation finished and published its validation
summary and EMA candidate, but the transaction failed closed before committing
``last.pth`` because the shared calibration ledger contained legitimate Stage0
and Stage3 rows with the same numeric step.  The repaired Stage4 entrypoint
identifies Stage4 rows by its six Stage4-only diagnostic columns.

This CPU-only tool changes exactly two provenance leaves in the Stage4 run
contract, pending raw checkpoint and partial EMA checkpoint:

* ``semantic_source_sha256.scripts/train_stage4_e2e.py``;
* the new top-level ``calibration_history_routing`` mapping.

Every checkpoint section outside ``provenance`` must remain bit-exact.  The
failed orchestration state, validation summary, shared calibration history,
training-log tail and failure log are hash-bound and copied to immutable
read-only evidence files.  Publication is opt-in behind an exact token, uses a
directory flock, writes a PREPARED receipt before replacing any canonical
artifact, rolls back on caught failures, and supports explicit recovery of an
interrupted PREPARED transaction.

Without ``--execute`` this command is a non-publishing dry run.  It never calls
a CUDA API and forces CUDA invisibility before importing PyTorch.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, NoReturn

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
STAGE4_SCHEMA = "graphrestore-stage4-runtime-v1"
VALIDATION_SCHEMA = "graphrestore-stage4-validation-v1"
RECEIPT_SCHEMA = "graphrestore-stage4-step4000-calibration-routing-migration-v1"
MIGRATION_KIND = "stage4_pending_4000_calibration_history_routing_provenance_only"
MIGRATION_STEP = 4_000
EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT = 17
EXPECTED_UNCHANGED_TOP_LEVEL_COUNT = 16
EXPECTED_SEMANTIC_SOURCE_COUNT = 47
EXPECTED_TRAIN_LOG_LINE_COUNT = 4_001
TRAIN_TAIL_LINE_COUNT = 16
ALLOWED_SOURCE_PATH = "scripts/train_stage4_e2e.py"
ROUTING_PROVENANCE_KEY = "calibration_history_routing"
ENTRYPOINTS = (ALLOWED_SOURCE_PATH,)
BACKUP_DIR_NAME = "stage4_step4000_calibration_history_routing_v1"
CONFIRMATION_TOKEN = "MIGRATE_STAGE4_PENDING_4000_CALIBRATION_HISTORY_ROUTING"
RECOVERY_CONFIRMATION_TOKEN = "RECOVER_STAGE4_PENDING_4000_CALIBRATION_HISTORY_ROUTING"

STAGE4_MARKER_COLUMNS = (
    "clean_misuse_psnr",
    "clean_misuse_ssim",
    "clean_misuse_residual_norm",
    "wrong_skill_identity_psnr",
    "wrong_skill_identity_ssim",
    "wrong_skill_residual_norm",
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
    *STAGE4_MARKER_COLUMNS,
    "reentry_request_rate",
    "unexpected_skill_activation_rate",
    "mean_program_levels",
)

# Frozen failure-boundary anchors.  The repaired source leaf and routing
# mapping are intentionally supplied explicitly until the implementation is
# frozen and independently audited.
AUDITED_RUN_CONTRACT_SHA256 = (
    "46aca21b891b5da7194546a04a44d156d713315c06965c53b53e6334e14ca0ab"
)
AUDITED_LAST_CHECKPOINT_SHA256 = (
    "22d8254d1833efd267d897ba2ddcc4addee93c0000d7dbded72df9a2000193cb"
)
AUDITED_BEST_CHECKPOINT_SHA256 = (
    "5465e55b99923e55a00e2ac70f4ee61399e2e4c1ff2e4d651da0d41321529989"
)
AUDITED_VALIDATION_LATEST_SHA256 = (
    "c2e560ebf2929b3c8933628b78a3591de471524394fadbac9afecbe02dc39a77"
)
AUDITED_CALIBRATION_HISTORY_SHA256 = (
    "b282987c3f77034f76788a412e91823cd4570ce8c6c10cd93030ee181612e034"
)
AUDITED_TRAIN_LOG_SHA256 = (
    "6cf0f60f34f1820c2626c085bfa79b2facece3c2cff2235e33760d08e83a26f3"
)
AUDITED_STATE_SHA256 = (
    "ef3d144b7cdc71417bc813ebf0fb1f1a4a45656d490189ac85f55aef83bb155c"
)
AUDITED_FAILURE_LOG_SHA256 = (
    "28c578ff9095f92938be52cc6c547d54afc900afb972aad5762de20b198559f3"
)
AUDITED_OLD_STAGE4_SOURCE_SHA256 = (
    "6eaaef9d6a88b85d1cce7339927064f7ad70529a63f2dbaa465654c578b0629b"
)
AUDITED_NEW_STAGE4_SOURCE_SHA256 = (
    "884487c1ba6b39706e92e52f748ad6aa5bbca5f4aea8fde701915c55a031b104"
)
AUDITED_ROUTING_SCHEMA = "graphrestore-stage4-calibration-ledger-v1"
AUDITED_CALIBRATION_HISTORY_ROUTING: dict[str, Any] = {
    "schema_version": AUDITED_ROUTING_SCHEMA,
    "frozen_stage3_history": {
        "path": (
            "/root/autodl-tmp/aaa/graphrestore/artifacts/metrics/"
            "calibration_history.csv"
        ),
        "sha256": AUDITED_CALIBRATION_HISTORY_SHA256,
    },
    "stage4_history_path": (
        "/root/autodl-tmp/aaa/graphrestore/artifacts/metrics/"
        "stage4_calibration_history.csv"
    ),
    "columns": list(CALIBRATION_COLUMNS),
    "stage4_marker_columns": list(STAGE4_MARKER_COLUMNS),
    "validation_steps": list(range(4_000, 40_001, 4_000)),
}
AUDITED_ROUTING_SHA256 = (
    "6fa3a7f6eb6c5ad3790ed7ea2d332c9d422e3999f2adc7bfa47495830e3802a0"
)
STAGE4_REFUSAL_MARKER = (
    "STAGE4_REFUSED: multiple Stage4 calibration rows already exist for step 4000"
)


class Stage4CalibrationRoutingMigrationError(RuntimeError):
    """The requested migration does not satisfy its exact contract."""


def _fail(message: str) -> NoReturn:
    raise Stage4CalibrationRoutingMigrationError(message)


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _assert_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        _fail("migration must remain CPU-only with CUDA uninitialized")


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = _absolute_without_symlink_resolution(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            _fail(f"symlink is forbidden in {label} path: {current}")


def _qualified_type(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _tensor_raw_bytes(value: torch.Tensor) -> bytes:
    if value.layout is not torch.strided:
        _fail(f"unsupported tensor layout: {value.layout}")
    return (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


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


def _update_fingerprint(digest: Any, value: object, counts: Counter[str]) -> None:
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
        ).encode()
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
        ).encode()
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
    elif isinstance(value, bool):
        digest.update(b"true" if value else b"false")
    elif isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    elif isinstance(value, float):
        digest.update(struct.pack(">d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    else:
        _fail(f"unsupported value type: {_qualified_type(value)}")


def _fingerprint(value: object) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    _update_fingerprint(digest, value, counts)
    return {"sha256": digest.hexdigest(), "counts": dict(sorted(counts.items()))}


def _assert_bit_exact(before: object, after: object, *, path: str) -> None:
    old = _fingerprint(before)
    new = _fingerprint(after)
    if old != new:
        _fail(f"state mutation at {path}")


def _section_evidence(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
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


def _load_cpu_checkpoint(path: Path) -> Mapping[str, Any]:
    _assert_cpu_only()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    _assert_cpu_only()
    return _require_mapping(payload, field=f"checkpoint {path}")


def _make_candidate(parent: Path, name: str, suffix: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=suffix, dir=parent
    )
    os.close(descriptor)
    candidate = Path(temporary_name)
    candidate.unlink()
    return candidate


def _replace_and_fsync(candidate: Path, destination: Path) -> None:
    os.replace(candidate, destination)
    fsync_directory(destination.parent)


def _restore_from_backup(backup: Path, destination: Path, *, mode: int) -> None:
    candidate = _make_candidate(destination.parent, destination.name, ".rollback")
    try:
        shutil.copyfile(backup, candidate)
        os.chmod(candidate, mode, follow_symlinks=False)
        with candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        _replace_and_fsync(candidate, destination)
    finally:
        candidate.unlink(missing_ok=True)


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        destination.write(chunk)


def _archive_file(
    source: Path, destination: Path, *, canonical_sha256: str
) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        _fail(f"protected evidence is not a regular file: {source}")
    if sha256_file(source) != canonical_sha256:
        _fail(f"protected evidence changed before archive: {source}")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with (
            source.open("rb") as input_stream,
            os.fdopen(descriptor, "wb", closefd=True) as output_stream,
        ):
            descriptor = -1
            _copy_stream(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if sha256_file(destination) != canonical_sha256:
        _fail(f"archive copy differs from canonical evidence: {source}")
    if source.stat().st_dev != destination.stat().st_dev:
        _fail(f"archive is not on the canonical artifact filesystem: {source}")
    os.chmod(destination, 0o444, follow_symlinks=False)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    fsync_directory(destination.parent)
    return {
        "canonical_path": str(source),
        "canonical_sha256": canonical_sha256,
        "canonical_mode": stat.S_IMODE(source.stat().st_mode),
        "archive_path": str(destination),
        "archive_sha256": canonical_sha256,
        "archive_mode": 0o444,
        "archive_device": destination.stat().st_dev,
        "archive_inode": destination.stat().st_ino,
        "byte_exact": True,
        "same_filesystem": True,
    }


def _archive_bytes(
    value: bytes,
    destination: Path,
    *,
    canonical_path: Path,
    canonical_sha256: str,
) -> dict[str, Any]:
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    archive_sha = sha256_file(destination)
    if archive_sha != hashlib.sha256(value).hexdigest():
        _fail("training-log tail archive round trip failed")
    if canonical_path.stat().st_dev != destination.stat().st_dev:
        _fail("training-log tail archive is not on the canonical filesystem")
    os.chmod(destination, 0o444, follow_symlinks=False)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    fsync_directory(destination.parent)
    return {
        "canonical_path": str(canonical_path),
        "canonical_sha256": canonical_sha256,
        "canonical_mode": stat.S_IMODE(canonical_path.stat().st_mode),
        "archive_path": str(destination),
        "archive_sha256": archive_sha,
        "archive_mode": 0o444,
        "archive_device": destination.stat().st_dev,
        "archive_inode": destination.stat().st_ino,
        "byte_exact_tail": True,
        "same_filesystem": True,
    }


def _resolve_requested_paths(**raw: str | Path) -> dict[str, Path]:
    requested = {key: Path(value) for key, value in raw.items()}
    for label, path in requested.items():
        _reject_symlink_chain(path, label=label.replace("_", " "))
    return {key: path.resolve(strict=False) for key, path in requested.items()}


def _validate_paths(paths: Mapping[str, Path]) -> None:
    root = paths["project_root"]
    expected = {
        "run_contract": root / "artifacts/checkpoints/stage4/run_contract.json",
        "last_checkpoint": root / "artifacts/checkpoints/stage4/last.pth",
        "best_checkpoint": root / "artifacts/checkpoints/stage4/best_ema.pth",
        "validation_latest": root
        / "artifacts/checkpoints/stage4/validation_latest.json",
        "train_log": root / "artifacts/checkpoints/stage4/train.jsonl",
        "calibration_history": root / "artifacts/metrics/calibration_history.csv",
        "state": root / "artifacts/orchestration/state.json",
        "failure_log": root / "artifacts/logs/main_pipeline.log",
        "backup_dir": root / "artifacts/migrations" / BACKUP_DIR_NAME,
    }
    if not root.is_dir() or root.is_symlink():
        _fail("project root must be an existing non-symlink directory")
    for label, expected_path in expected.items():
        if paths[label] != expected_path.resolve(strict=False):
            _fail(f"{label} is not the exact project-local canonical path")
    for label in expected:
        if label == "backup_dir":
            continue
        path = paths[label]
        if path.is_symlink() or not path.is_file():
            _fail(f"{label} must be an existing non-symlink regular file")
    migrations = (root / "artifacts/migrations").resolve(strict=False)
    if migrations.is_symlink() or not migrations.is_dir():
        _fail("artifacts/migrations must be an existing non-symlink directory")


@contextmanager
def _migration_lock(project_root: Path) -> Iterator[dict[str, Any]]:
    directory = project_root / "artifacts/migrations"
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("another provenance migration holds the artifacts/migrations flock")
        evidence = {
            "path": str(directory.resolve()),
            "device": os.fstat(descriptor).st_dev,
            "inode": os.fstat(descriptor).st_ino,
            "exclusive_nonblocking": True,
        }
        yield evidence
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_hashes(*values: str) -> None:
    if any(not is_sha256(value) for value in values):
        _fail("every expected hash must be a lowercase SHA256")


def _validate_state(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        _fail("orchestration state SHA256 drifted")
    state = _require_mapping(load_json(path), field="orchestration state")
    command = state.get("last_command")
    if (
        state.get("schema_version") != "graphrestore-orchestration-v1"
        or state.get("protocol_id") != PROTOCOL_ID
        or state.get("status") != "FAILED"
        or state.get("current_stage") != "FAILED"
        or state.get("gpu") != "released"
        or state.get("last_exit_code") != 2
        or not isinstance(command, list)
        or "scripts/train_stage4_e2e.py" not in command
        or not isinstance(state.get("next_command"), str)
        or "--resume_post_approval_pipeline" not in state["next_command"]
    ):
        _fail("orchestration state is not the exact Stage4 exit-2 recovery boundary")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "status": state["status"],
        "current_stage": state["current_stage"],
        "gpu": state["gpu"],
        "last_exit_code": state["last_exit_code"],
        "next_command": state["next_command"],
    }


def _validate_failure_log(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        _fail("Stage4 failure log SHA256 drifted")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        _fail(f"Stage4 failure log is not UTF-8: {exc}")
    if (
        lines.count(STAGE4_REFUSAL_MARKER) != 1
        or len(lines) < 3
        or not lines[-3].startswith("[")
        or not lines[-3].endswith(
            "] START stage4: /root/miniconda3/bin/python "
            "scripts/train_stage4_e2e.py --config configs/stage4_graphrestore_e2e.yaml"
        )
        or lines[-2] != STAGE4_REFUSAL_MARKER
        or not lines[-1].startswith("[")
        or not lines[-1].endswith("] END stage4: exit=2")
    ):
        _fail("Stage4 failure log lacks the exact terminal refusal transaction")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "failure_marker": STAGE4_REFUSAL_MARKER,
        "failure_marker_count": 1,
        "failure_marker_sha256": hashlib.sha256(
            STAGE4_REFUSAL_MARKER.encode()
        ).hexdigest(),
        "terminal_start": lines[-3],
        "terminal_end": lines[-1],
    }


def _validate_validation_latest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        _fail("Stage4 validation_latest SHA256 drifted")
    summary = _require_mapping(load_json(path), field="Stage4 validation_latest")
    if (
        summary.get("schema_version") != VALIDATION_SCHEMA
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("image_count") != 1_600
        or summary.get("dataset") != "primary_val_single_and_group_a_only"
        or not isinstance(summary.get("single_equal_task_mean"), Mapping)
        or not isinstance(summary.get("group_a_equal_combination_mean"), Mapping)
        or not isinstance(summary.get("diagnostics"), Mapping)
    ):
        _fail("Stage4 validation_latest semantic boundary drifted")
    return dict(summary)


def _validate_train_log(
    path: Path, expected_sha256: str, *, expected_line_count: int
) -> tuple[dict[str, Any], bytes]:
    if sha256_file(path) != expected_sha256:
        _fail("Stage4 train log SHA256 drifted")
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if (
        len(raw_lines) != expected_line_count
        or expected_line_count != MIGRATION_STEP + 1
    ):
        _fail("Stage4 train log is not exactly steps 1..4000 plus PRE")
    for expected_step, raw in enumerate(raw_lines[:-1], start=1):
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(f"invalid Stage4 train JSONL at line {expected_step}: {exc}")
        if (
            not isinstance(row, Mapping)
            or row.get("step") != expected_step
            or "event" in row
        ):
            _fail(f"Stage4 train log step sequence drifted at {expected_step}")
    final = json.loads(raw_lines[-1])
    if (
        not isinstance(final, Mapping)
        or final.get("event") != "pre_validation_checkpoint"
        or final.get("step") != MIGRATION_STEP
        or any(b'"event": "validation"' in line for line in raw_lines)
    ):
        _fail("Stage4 train log has no exact uncommitted step-4000 PRE boundary")
    tail = b"".join(raw_lines[-TRAIN_TAIL_LINE_COUNT:])
    return (
        {
            "path": str(path),
            "sha256": expected_sha256,
            "line_count": len(raw_lines),
            "tail_line_count": min(TRAIN_TAIL_LINE_COUNT, len(raw_lines)),
            "tail_sha256": hashlib.sha256(tail).hexdigest(),
            "final_event": dict(final),
        },
        tail,
    )


def _validate_calibration_history(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        _fail("shared calibration history SHA256 drifted")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CALIBRATION_COLUMNS:
            _fail("shared calibration history header drifted")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = dict(raw)
            if set(row) != set(CALIBRATION_COLUMNS) or None in row:
                _fail(
                    "shared calibration history row width drifted at line "
                    f"{line_number}"
                )
            if any(value is None for value in row.values()):
                _fail(
                    "shared calibration history row has missing columns at line "
                    f"{line_number}"
                )
            rows.append(row)
    historical_at_step = 0
    stage4_at_step = 0
    for row in rows:
        if row.get("step") != str(MIGRATION_STEP):
            continue
        presence = tuple(bool(row.get(column)) for column in STAGE4_MARKER_COLUMNS)
        if any(presence) and not all(presence):
            _fail("partial Stage4 calibration row exists at the failure boundary")
        if all(presence):
            stage4_at_step += 1
        else:
            historical_at_step += 1
    if historical_at_step != 2 or stage4_at_step != 0:
        _fail("shared history is not the exact two-historical-row collision boundary")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "row_count": len(rows),
        "historical_non_stage4_rows_at_step4000": historical_at_step,
        "stage4_rows_at_step4000": stage4_at_step,
        "marker_columns": list(STAGE4_MARKER_COLUMNS),
    }


def _validate_checkpoint(
    payload: Mapping[str, Any], *, role: str, validation: Mapping[str, Any]
) -> None:
    if len(payload) != EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT:
        _fail(f"{role} checkpoint top-level count drifted")
    expected_common = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": "stage4",
        "step": MIGRATION_STEP,
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "scaler": None,
    }
    for key, value in expected_common.items():
        if payload.get(key, object()) != value:
            _fail(f"{role} checkpoint header drifted at {key}")
    if role == "last":
        expected = {
            "model_role": "raw_training_state",
            "resumable": True,
            "pending_validation_step": MIGRATION_STEP,
            "metrics": {},
        }
    else:
        expected = {
            "model_role": "ema_selection",
            "resumable": False,
            "pending_validation_step": None,
        }
    for key, value in expected.items():
        if payload.get(key, object()) != value:
            _fail(f"{role} checkpoint boundary drifted at {key}")
    sampler = _require_mapping(payload.get("sampler_state"), field=f"{role}.sampler")
    ema = _require_mapping(payload.get("ema"), field=f"{role}.ema")
    if (
        sampler.get("consumed_optimizer_step") != MIGRATION_STEP
        or sampler.get("sample_cursor") != 16_000
        or ema.get("num_updates") != MIGRATION_STEP
    ):
        _fail(f"{role} checkpoint optimizer/sampler boundary drifted")
    if role == "best":
        metrics = _require_mapping(payload.get("metrics"), field="best.metrics")
        single = _require_mapping(
            validation.get("single_equal_task_mean"), field="validation.single"
        )
        group = _require_mapping(
            validation.get("group_a_equal_combination_mean"), field="validation.group"
        )
        expected_metrics = {
            "validation_step": float(MIGRATION_STEP),
            "single_psnr": single.get("psnr"),
            "single_ssim": single.get("ssim"),
            "group_a_psnr": group.get("psnr"),
            "group_a_ssim": group.get("ssim"),
            "best_step": float(MIGRATION_STEP),
            "best_single_psnr": single.get("psnr"),
            "best_single_ssim": single.get("ssim"),
            "best_group_a_psnr": group.get("psnr"),
            "best_group_a_ssim": group.get("ssim"),
        }
        if set(metrics) != set(expected_metrics) or any(
            metrics[key] != value for key, value in expected_metrics.items()
        ):
            _fail("partial best metrics differ from validation_latest")
    _walk_finite(payload)


def _validate_semantic_sources(
    *,
    root: Path,
    old_provenance: Mapping[str, Any],
    expected_old_source_sha256: str,
    expected_new_source_sha256: str,
    expected_count: int,
) -> tuple[dict[str, str], dict[str, str]]:
    old_raw = _require_mapping(
        old_provenance.get("semantic_source_sha256"),
        field="old provenance semantic sources",
    )
    old = {str(key): str(value) for key, value in old_raw.items()}
    current = semantic_source_hashes(root, entrypoints=ENTRYPOINTS)
    if (
        len(old) != expected_count
        or len(current) != expected_count
        or set(old) != set(current)
        or old.get(ALLOWED_SOURCE_PATH) != expected_old_source_sha256
        or current.get(ALLOWED_SOURCE_PATH) != expected_new_source_sha256
    ):
        _fail("semantic source boundary/count drifted")
    changed = [path for path in old if old[path] != current[path]]
    if changed != [ALLOWED_SOURCE_PATH]:
        _fail("semantic sources changed outside the exact Stage4 entrypoint repair")
    return current, old


def _validate_routing_mapping(
    value: Mapping[str, Any],
    *,
    project_root: Path,
    expected_schema: str,
    expected_sha256: str,
    expected_frozen_history_sha256: str,
) -> dict[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(
                dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
    except (TypeError, ValueError) as exc:
        _fail(f"calibration history routing mapping is not canonical JSON: {exc}")
    frozen_path = (project_root / "artifacts/metrics/calibration_history.csv").resolve(
        strict=False
    )
    stage4_path = (
        project_root / "artifacts/metrics/stage4_calibration_history.csv"
    ).resolve(strict=False)
    expected_keys = {
        "schema_version",
        "frozen_stage3_history",
        "stage4_history_path",
        "columns",
        "stage4_marker_columns",
        "validation_steps",
    }
    frozen = normalized.get("frozen_stage3_history")
    if (
        not isinstance(normalized, dict)
        or set(normalized) != expected_keys
        or expected_schema != "graphrestore-stage4-calibration-ledger-v1"
        or normalized.get("schema_version") != expected_schema
        or frozen
        != {
            "path": str(frozen_path),
            "sha256": expected_frozen_history_sha256,
        }
        or normalized.get("stage4_history_path") != str(stage4_path)
        or normalized.get("columns") != list(CALIBRATION_COLUMNS)
        or normalized.get("stage4_marker_columns") != list(STAGE4_MARKER_COLUMNS)
        or normalized.get("validation_steps") != list(range(4_000, 40_001, 4_000))
        or sha256_json(normalized) != expected_sha256
    ):
        _fail("calibration history routing mapping/schema/hash drifted")
    if os.path.lexists(stage4_path):
        _fail("Stage4-only calibration history must be absent at migration boundary")
    return normalized


def _exact_provenance_diff(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> list[dict[str, Any]]:
    old_semantic = _require_mapping(
        old.get("semantic_source_sha256"), field="old semantic"
    )
    new_semantic = _require_mapping(
        new.get("semantic_source_sha256"), field="new semantic"
    )
    if set(old) | {ROUTING_PROVENANCE_KEY} != set(new):
        _fail("provenance top-level keys changed outside routing addition")
    for key in old:
        if key == "semantic_source_sha256":
            continue
        if old[key] != new[key]:
            _fail(f"provenance changed outside allowed leaves: {key}")
    source_changes = [
        key for key in old_semantic if old_semantic[key] != new_semantic.get(key)
    ]
    if set(old_semantic) != set(new_semantic) or source_changes != [
        ALLOWED_SOURCE_PATH
    ]:
        _fail("provenance semantic-source diff is not exactly one entrypoint leaf")
    return [
        {
            "path": ROUTING_PROVENANCE_KEY,
            "old": "<absent>",
            "new": copy.deepcopy(new[ROUTING_PROVENANCE_KEY]),
        },
        {
            "path": f"semantic_source_sha256.{ALLOWED_SOURCE_PATH}",
            "old": old_semantic[ALLOWED_SOURCE_PATH],
            "new": new_semantic[ALLOWED_SOURCE_PATH],
        },
    ]


def _protected_hashes(paths: Mapping[str, Path], expected: Mapping[str, str]) -> None:
    for label, expected_sha in expected.items():
        if sha256_file(paths[label]) != expected_sha:
            _fail(f"protected {label} changed during migration")


def _verify_archive_evidence(
    evidence: Mapping[str, Any], *, backup_dir: Path, expected_sha256: str
) -> Path:
    raw_path = evidence.get("archive_path")
    if not isinstance(raw_path, str):
        _fail("archive evidence path is invalid")
    requested = Path(raw_path)
    _reject_symlink_chain(requested, label="archive evidence")
    path = requested.resolve(strict=False)
    canonical_raw = evidence.get("canonical_path")
    if not isinstance(canonical_raw, str):
        _fail("archive canonical evidence path is invalid")
    canonical_requested = Path(canonical_raw)
    _reject_symlink_chain(canonical_requested, label="archive canonical evidence")
    canonical = canonical_requested.resolve(strict=False)
    if (
        path.parent != backup_dir
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != expected_sha256
        or evidence.get("archive_sha256") != expected_sha256
        or evidence.get("archive_mode") != 0o444
        or stat.S_IMODE(path.stat().st_mode) != 0o444
        or evidence.get("archive_device") != path.stat().st_dev
        or evidence.get("archive_inode") != path.stat().st_ino
        or not isinstance(evidence.get("canonical_mode"), int)
        or not canonical.is_file()
        or canonical.is_symlink()
        or canonical.stat().st_dev != path.stat().st_dev
        or evidence.get("same_filesystem") is not True
    ):
        _fail(f"immutable archive evidence drifted: {path}")
    return path


def _publish_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(receipt))
    fsync_directory(path.parent)


def _common_validation(
    *,
    paths: Mapping[str, Path],
    expected: Mapping[str, str],
    expected_old_source_sha256: str,
    expected_new_source_sha256: str,
    expected_semantic_source_count: int,
    calibration_history_routing: Mapping[str, Any],
    expected_routing_schema: str,
    expected_routing_sha256: str,
    expected_train_log_line_count: int,
) -> dict[str, Any]:
    _assert_cpu_only()
    _validate_hashes(
        *expected.values(),
        expected_old_source_sha256,
        expected_new_source_sha256,
        expected_routing_sha256,
    )
    if expected_semantic_source_count != EXPECTED_SEMANTIC_SOURCE_COUNT:
        _fail("Stage4 semantic-source count must remain exactly 47")
    if expected_train_log_line_count != EXPECTED_TRAIN_LOG_LINE_COUNT:
        _fail("Stage4 train-log line count must remain exactly 4001")
    routing = _validate_routing_mapping(
        calibration_history_routing,
        project_root=paths["project_root"],
        expected_schema=expected_routing_schema,
        expected_sha256=expected_routing_sha256,
        expected_frozen_history_sha256=expected["calibration_history"],
    )
    for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
        if sha256_file(paths[label]) != expected[label]:
            _fail(f"{label} SHA256 differs from the audited failure boundary")
    state = _validate_state(paths["state"], expected["state"])
    failure = _validate_failure_log(paths["failure_log"], expected["failure_log"])
    validation = _validate_validation_latest(
        paths["validation_latest"], expected["validation_latest"]
    )
    train, tail = _validate_train_log(
        paths["train_log"],
        expected["train_log"],
        expected_line_count=expected_train_log_line_count,
    )
    history = _validate_calibration_history(
        paths["calibration_history"], expected["calibration_history"]
    )
    contract = _require_mapping(
        load_json(paths["run_contract"]), field="Stage4 run contract"
    )
    if contract.get("schema_version") != STAGE4_SCHEMA:
        _fail("Stage4 run-contract schema drifted")
    old_provenance = _require_mapping(
        contract.get("provenance"), field="run provenance"
    )
    if ROUTING_PROVENANCE_KEY in old_provenance:
        _fail("old Stage4 provenance already contains calibration history routing")
    last = _load_cpu_checkpoint(paths["last_checkpoint"])
    best = _load_cpu_checkpoint(paths["best_checkpoint"])
    if (
        sha256_file(paths["last_checkpoint"]) != expected["last_checkpoint"]
        or sha256_file(paths["best_checkpoint"]) != expected["best_checkpoint"]
    ):
        _fail("checkpoint changed during CPU load")
    _validate_checkpoint(last, role="last", validation=validation)
    _validate_checkpoint(best, role="best", validation=validation)
    if (
        last.get("provenance") != old_provenance
        or best.get("provenance") != old_provenance
    ):
        _fail("run contract, raw checkpoint and partial best provenance differ")
    current_semantic, old_semantic = _validate_semantic_sources(
        root=paths["project_root"],
        old_provenance=old_provenance,
        expected_old_source_sha256=expected_old_source_sha256,
        expected_new_source_sha256=expected_new_source_sha256,
        expected_count=expected_semantic_source_count,
    )
    new_provenance = copy.deepcopy(dict(old_provenance))
    new_semantic = dict(old_semantic)
    new_semantic[ALLOWED_SOURCE_PATH] = current_semantic[ALLOWED_SOURCE_PATH]
    new_provenance["semantic_source_sha256"] = new_semantic
    new_provenance[ROUTING_PROVENANCE_KEY] = routing
    provenance_diff = _exact_provenance_diff(old_provenance, new_provenance)
    return {
        "routing": routing,
        "contract": contract,
        "last": last,
        "best": best,
        "old_provenance": old_provenance,
        "new_provenance": new_provenance,
        "provenance_diff": provenance_diff,
        "state": state,
        "failure": failure,
        "validation": validation,
        "train": train,
        "train_tail": tail,
        "history": history,
    }


def migrate_stage4_step4000_calibration_history_provenance(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    last_checkpoint: str | Path,
    best_checkpoint: str | Path,
    validation_latest: str | Path,
    train_log: str | Path,
    calibration_history: str | Path,
    state: str | Path,
    failure_log: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_last_checkpoint_sha256: str,
    expected_best_checkpoint_sha256: str,
    expected_validation_latest_sha256: str,
    expected_train_log_sha256: str,
    expected_calibration_history_sha256: str,
    expected_state_sha256: str,
    expected_failure_log_sha256: str,
    expected_old_stage4_source_sha256: str,
    expected_new_stage4_source_sha256: str,
    calibration_history_routing: Mapping[str, Any],
    expected_routing_schema: str,
    expected_routing_sha256: str,
    expected_semantic_source_count: int = EXPECTED_SEMANTIC_SOURCE_COUNT,
    expected_train_log_line_count: int = EXPECTED_TRAIN_LOG_LINE_COUNT,
    execute: bool = False,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Build or publish the exact three-artifact Stage4 migration."""

    paths = _resolve_requested_paths(
        project_root=project_root,
        run_contract=run_contract,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
        validation_latest=validation_latest,
        train_log=train_log,
        calibration_history=calibration_history,
        state=state,
        failure_log=failure_log,
        backup_dir=backup_dir,
    )
    _validate_paths(paths)
    if execute and confirmation_token != CONFIRMATION_TOKEN:
        _fail("execution requires the exact Stage4 calibration routing token")
    if paths["backup_dir"].exists():
        _fail(f"dedicated backup directory already exists: {paths['backup_dir']}")
    expected = {
        "run_contract": expected_run_contract_sha256,
        "last_checkpoint": expected_last_checkpoint_sha256,
        "best_checkpoint": expected_best_checkpoint_sha256,
        "validation_latest": expected_validation_latest_sha256,
        "train_log": expected_train_log_sha256,
        "calibration_history": expected_calibration_history_sha256,
        "state": expected_state_sha256,
        "failure_log": expected_failure_log_sha256,
    }
    with _migration_lock(paths["project_root"]) as lock_evidence:
        values = _common_validation(
            paths=paths,
            expected=expected,
            expected_old_source_sha256=expected_old_stage4_source_sha256,
            expected_new_source_sha256=expected_new_stage4_source_sha256,
            expected_semantic_source_count=expected_semantic_source_count,
            calibration_history_routing=calibration_history_routing,
            expected_routing_schema=expected_routing_schema,
            expected_routing_sha256=expected_routing_sha256,
            expected_train_log_line_count=expected_train_log_line_count,
        )
        contract = values["contract"]
        last = values["last"]
        best = values["best"]
        new_provenance = values["new_provenance"]
        new_contract = copy.deepcopy(dict(contract))
        new_contract["provenance"] = new_provenance
        _assert_bit_exact(
            {key: value for key, value in contract.items() if key != "provenance"},
            {key: value for key, value in new_contract.items() if key != "provenance"},
            path="run_contract.outside_provenance",
        )
        new_last = copy.copy(last)
        new_last["provenance"] = new_provenance
        new_best = copy.copy(best)
        new_best["provenance"] = new_provenance
        for label, old_payload, new_payload in (
            ("last", last, new_last),
            ("best", best, new_best),
        ):
            unchanged = 0
            for key in old_payload:
                if key != "provenance":
                    _assert_bit_exact(
                        old_payload[key], new_payload[key], path=f"{label}.{key}"
                    )
                    unchanged += 1
            if unchanged != EXPECTED_UNCHANGED_TOP_LEVEL_COUNT:
                _fail(f"{label} unchanged checkpoint section count drifted")

        candidates = {
            "run_contract": _make_candidate(
                paths["run_contract"].parent,
                paths["run_contract"].name,
                ".stage4-calibration-routing.candidate.json",
            ),
            "last_checkpoint": _make_candidate(
                paths["last_checkpoint"].parent,
                paths["last_checkpoint"].name,
                ".stage4-calibration-routing.candidate.pth",
            ),
            "best_checkpoint": _make_candidate(
                paths["best_checkpoint"].parent,
                paths["best_checkpoint"].name,
                ".stage4-calibration-routing.candidate.pth",
            ),
        }
        receipt_path = paths["backup_dir"] / "MIGRATION_RECEIPT.json"
        backups: dict[str, Any] = {}
        prepared_written = False
        try:
            atomic_write_json(candidates["run_contract"], new_contract)
            atomic_torch_save(new_last, candidates["last_checkpoint"])
            atomic_torch_save(new_best, candidates["best_checkpoint"])
            reloaded_contract = _require_mapping(
                load_json(candidates["run_contract"]), field="candidate run contract"
            )
            reloaded_last = _load_cpu_checkpoint(candidates["last_checkpoint"])
            reloaded_best = _load_cpu_checkpoint(candidates["best_checkpoint"])
            if (
                reloaded_contract != new_contract
                or reloaded_contract.get("provenance") != new_provenance
                or reloaded_last.get("provenance") != new_provenance
                or reloaded_best.get("provenance") != new_provenance
            ):
                _fail("candidate three-way provenance identity failed")
            _validate_checkpoint(
                reloaded_last, role="last", validation=values["validation"]
            )
            _validate_checkpoint(
                reloaded_best, role="best", validation=values["validation"]
            )
            section_evidence = {
                "last_checkpoint": _section_evidence(last, reloaded_last),
                "best_checkpoint": _section_evidence(best, reloaded_best),
            }
            new_hashes = {
                label: sha256_file(path) for label, path in candidates.items()
            }
            receipt: dict[str, Any] = {
                "schema_version": RECEIPT_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "migration": MIGRATION_KIND,
                "status": "DRY_RUN" if not execute else "PREPARED",
                "created_utc": utc_now_iso(),
                "cpu_only": True,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "step": MIGRATION_STEP,
                "flock": lock_evidence,
                "old": {
                    label: {"path": str(paths[label]), "sha256": expected[label]}
                    for label in ("run_contract", "last_checkpoint", "best_checkpoint")
                }
                | {
                    "provenance_json_sha256": sha256_json(
                        dict(values["old_provenance"])
                    )
                },
                "new": new_hashes
                | {"provenance_json_sha256": sha256_json(new_provenance)},
                "exact_provenance_leaf_diff": values["provenance_diff"],
                "calibration_history_routing": {
                    "schema_version": expected_routing_schema,
                    "sha256": expected_routing_sha256,
                    "mapping": values["routing"],
                },
                "semantic_source_count": expected_semantic_source_count,
                "unchanged_semantic_source_count": expected_semantic_source_count - 1,
                "checkpoint_top_level_count": EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT,
                "checkpoint_top_level_bit_exact_outside_provenance_count": EXPECTED_UNCHANGED_TOP_LEVEL_COUNT,
                "both_checkpoints_bit_exact_outside_provenance": True,
                "run_contract_bit_exact_outside_provenance": True,
                "checkpoint_section_fingerprints": section_evidence,
                "all_checkpoint_tensors_finite": True,
                "protected_evidence": {
                    "validation_latest": {
                        "path": str(paths["validation_latest"]),
                        "sha256": expected["validation_latest"],
                        "image_count": values["validation"]["image_count"],
                    },
                    "train_log": values["train"],
                    "calibration_history": values["history"],
                    "orchestration_state": values["state"],
                    "failure_log": values["failure"],
                },
                "backup": backups,
                "execution_confirmation_token_sha256": (
                    hashlib.sha256(CONFIRMATION_TOKEN.encode()).hexdigest()
                    if execute
                    else None
                ),
                "migration_script_sha256": sha256_file(Path(__file__).resolve()),
            }
            if not execute:
                _protected_hashes(paths, expected)
                _assert_cpu_only()
                return receipt

            paths["backup_dir"].mkdir(mode=0o700)
            fsync_directory(paths["backup_dir"].parent)
            archive_specs = (
                ("run_contract", "old-run_contract.json"),
                ("last_checkpoint", "old-last.pth"),
                ("best_checkpoint", "old-best_ema.pth"),
                ("validation_latest", "validation_latest.json"),
                ("calibration_history", "calibration_history.csv"),
                ("state", "orchestration_state.json"),
                ("failure_log", "main_pipeline.log"),
            )
            for label, filename in archive_specs:
                backups[label] = _archive_file(
                    paths[label],
                    paths["backup_dir"] / filename,
                    canonical_sha256=expected[label],
                )
            backups["train_tail"] = _archive_bytes(
                values["train_tail"],
                paths["backup_dir"] / "train_tail.jsonl",
                canonical_path=paths["train_log"],
                canonical_sha256=expected["train_log"],
            ) | {
                "tail_sha256": values["train"]["tail_sha256"],
                "tail_line_count": values["train"]["tail_line_count"],
            }
            receipt["backup"] = backups
            _protected_hashes(paths, expected)
            _publish_receipt(receipt_path, receipt)
            prepared_written = True

            # Checkpoints first, contract last.  Any process observing the old
            # contract still rejects the new checkpoint provenance; PREPARED
            # recovery covers interruption between replacements.
            for label in ("last_checkpoint", "best_checkpoint", "run_contract"):
                if sha256_file(paths[label]) != expected[label]:
                    _fail(f"{label} changed immediately before publication")
                _replace_and_fsync(candidates[label], paths[label])
            for label, expected_sha in new_hashes.items():
                if sha256_file(paths[label]) != expected_sha:
                    _fail(f"published {label} candidate hash drifted")
            _protected_hashes(
                paths,
                {
                    key: value
                    for key, value in expected.items()
                    if key not in new_hashes
                },
            )
            published_contract = _require_mapping(
                load_json(paths["run_contract"]), field="published run contract"
            )
            published_last = _load_cpu_checkpoint(paths["last_checkpoint"])
            published_best = _load_cpu_checkpoint(paths["best_checkpoint"])
            if (
                published_contract.get("provenance") != new_provenance
                or published_last.get("provenance") != new_provenance
                or published_best.get("provenance") != new_provenance
            ):
                _fail("published three-way provenance identity failed")
            _section_evidence(last, published_last)
            _section_evidence(best, published_best)
            for evidence in backups.values():
                _verify_archive_evidence(
                    evidence,
                    backup_dir=paths["backup_dir"],
                    expected_sha256=evidence["archive_sha256"],
                )
            receipt["status"] = "COMPLETE"
            receipt["completed_utc"] = utc_now_iso()
            receipt["backup_read_only_after_publication"] = True
            receipt["protected_evidence_unchanged_after_publication"] = True
            _assert_cpu_only()
            _publish_receipt(receipt_path, receipt)
            return receipt
        except BaseException as original_error:
            if execute and prepared_written:
                rollback_errors: list[str] = []
                for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
                    evidence = backups.get(label)
                    if not isinstance(evidence, Mapping):
                        rollback_errors.append(f"missing backup evidence for {label}")
                        continue
                    try:
                        _restore_from_backup(
                            Path(str(evidence["archive_path"])),
                            paths[label],
                            mode=int(evidence["canonical_mode"]),
                        )
                        if sha256_file(paths[label]) != expected[label]:
                            rollback_errors.append(f"{label} rollback hash mismatch")
                    except BaseException as rollback_error:
                        rollback_errors.append(f"{label}: {rollback_error}")
                try:
                    _protected_hashes(
                        paths,
                        {
                            key: value
                            for key, value in expected.items()
                            if key
                            not in {
                                "run_contract",
                                "last_checkpoint",
                                "best_checkpoint",
                            }
                        },
                    )
                except BaseException as protected_error:
                    rollback_errors.append(str(protected_error))
                rollback_receipt = {
                    "schema_version": RECEIPT_SCHEMA,
                    "protocol_id": PROTOCOL_ID,
                    "migration": MIGRATION_KIND,
                    "status": "ROLLBACK_FAILED" if rollback_errors else "ROLLED_BACK",
                    "rolled_back_utc": utc_now_iso(),
                    "old": receipt.get("old") if "receipt" in locals() else {},
                    "new": receipt.get("new") if "receipt" in locals() else {},
                    "backup": backups,
                    "rollback_errors": rollback_errors,
                    "migration_script_sha256": sha256_file(Path(__file__).resolve()),
                }
                _publish_receipt(receipt_path, rollback_receipt)
                if rollback_errors:
                    raise Stage4CalibrationRoutingMigrationError(
                        "publication failed and rollback was incomplete: "
                        + "; ".join(rollback_errors)
                    ) from original_error
            raise
        finally:
            for candidate in candidates.values():
                candidate.unlink(missing_ok=True)


def recover_prepared_stage4_step4000_calibration_history_provenance(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    last_checkpoint: str | Path,
    best_checkpoint: str | Path,
    validation_latest: str | Path,
    train_log: str | Path,
    calibration_history: str | Path,
    state: str | Path,
    failure_log: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_last_checkpoint_sha256: str,
    expected_best_checkpoint_sha256: str,
    expected_validation_latest_sha256: str,
    expected_train_log_sha256: str,
    expected_calibration_history_sha256: str,
    expected_state_sha256: str,
    expected_failure_log_sha256: str,
    expected_old_stage4_source_sha256: str,
    expected_new_stage4_source_sha256: str,
    calibration_history_routing: Mapping[str, Any],
    expected_routing_schema: str,
    expected_routing_sha256: str,
    expected_semantic_source_count: int = EXPECTED_SEMANTIC_SOURCE_COUNT,
    expected_train_log_line_count: int = EXPECTED_TRAIN_LOG_LINE_COUNT,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Roll an interrupted PREPARED three-file transaction back exactly."""

    if confirmation_token != RECOVERY_CONFIRMATION_TOKEN:
        _fail("PREPARED recovery requires the exact distinct recovery token")
    paths = _resolve_requested_paths(
        project_root=project_root,
        run_contract=run_contract,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
        validation_latest=validation_latest,
        train_log=train_log,
        calibration_history=calibration_history,
        state=state,
        failure_log=failure_log,
        backup_dir=backup_dir,
    )
    _validate_paths(paths)
    expected = {
        "run_contract": expected_run_contract_sha256,
        "last_checkpoint": expected_last_checkpoint_sha256,
        "best_checkpoint": expected_best_checkpoint_sha256,
        "validation_latest": expected_validation_latest_sha256,
        "train_log": expected_train_log_sha256,
        "calibration_history": expected_calibration_history_sha256,
        "state": expected_state_sha256,
        "failure_log": expected_failure_log_sha256,
    }
    _validate_hashes(
        *expected.values(),
        expected_old_stage4_source_sha256,
        expected_new_stage4_source_sha256,
        expected_routing_sha256,
    )
    routing = _validate_routing_mapping(
        calibration_history_routing,
        project_root=paths["project_root"],
        expected_schema=expected_routing_schema,
        expected_sha256=expected_routing_sha256,
        expected_frozen_history_sha256=expected["calibration_history"],
    )
    receipt_path = paths["backup_dir"] / "MIGRATION_RECEIPT.json"
    if not paths["backup_dir"].is_dir() or not receipt_path.is_file():
        _fail("PREPARED recovery requires its dedicated receipt")
    with _migration_lock(paths["project_root"]):
        receipt = _require_mapping(load_json(receipt_path), field="migration receipt")
        already_recovered = receipt.get("status") == "ROLLED_BACK_FROM_PREPARED"
        if (
            receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("migration") != MIGRATION_KIND
            or receipt.get("status") not in {"PREPARED", "ROLLED_BACK_FROM_PREPARED"}
            or receipt.get("migration_script_sha256")
            != sha256_file(Path(__file__).resolve())
            or receipt.get("execution_confirmation_token_sha256")
            != hashlib.sha256(CONFIRMATION_TOKEN.encode()).hexdigest()
        ):
            _fail("receipt is not the exact PREPARED Stage4 transaction")
        routing_evidence = _require_mapping(
            receipt.get("calibration_history_routing"), field="receipt routing"
        )
        if (
            routing_evidence.get("schema_version") != expected_routing_schema
            or routing_evidence.get("sha256") != expected_routing_sha256
            or routing_evidence.get("mapping") != routing
        ):
            _fail("PREPARED routing evidence drifted")
        current_semantic = semantic_source_hashes(
            paths["project_root"], entrypoints=ENTRYPOINTS
        )
        if (
            len(current_semantic) != expected_semantic_source_count
            or current_semantic.get(ALLOWED_SOURCE_PATH)
            != expected_new_stage4_source_sha256
        ):
            _fail("semantic sources drifted before PREPARED recovery")
        protected_expected = {
            key: value
            for key, value in expected.items()
            if key not in {"run_contract", "last_checkpoint", "best_checkpoint"}
        }
        _protected_hashes(paths, protected_expected)
        _validate_state(paths["state"], expected["state"])
        _validate_failure_log(paths["failure_log"], expected["failure_log"])
        _validate_validation_latest(
            paths["validation_latest"], expected["validation_latest"]
        )
        train_evidence, tail = _validate_train_log(
            paths["train_log"],
            expected["train_log"],
            expected_line_count=expected_train_log_line_count,
        )
        _validate_calibration_history(
            paths["calibration_history"], expected["calibration_history"]
        )

        old = _require_mapping(receipt.get("old"), field="receipt.old")
        new = _require_mapping(receipt.get("new"), field="receipt.new")
        backups = _require_mapping(receipt.get("backup"), field="receipt.backup")
        backup_paths: dict[str, Path] = {}
        canonical_modes: dict[str, int] = {}
        live_before: dict[str, str] = {}
        for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
            old_entry = _require_mapping(old.get(label), field=f"receipt.old.{label}")
            new_sha = new.get(label)
            evidence = _require_mapping(
                backups.get(label), field=f"receipt.backup.{label}"
            )
            if (
                old_entry.get("path") != str(paths[label])
                or old_entry.get("sha256") != expected[label]
                or not is_sha256(new_sha)
            ):
                _fail(f"PREPARED old/new evidence drifted: {label}")
            backup_paths[label] = _verify_archive_evidence(
                evidence,
                backup_dir=paths["backup_dir"],
                expected_sha256=expected[label],
            )
            canonical_modes[label] = int(evidence["canonical_mode"])
            live_sha = sha256_file(paths[label])
            if live_sha not in {expected[label], str(new_sha)}:
                _fail(f"PREPARED live artifact is neither old nor new: {label}")
            live_before[label] = live_sha
        for label in (
            "validation_latest",
            "calibration_history",
            "state",
            "failure_log",
        ):
            _verify_archive_evidence(
                _require_mapping(backups.get(label), field=f"receipt.backup.{label}"),
                backup_dir=paths["backup_dir"],
                expected_sha256=expected[label],
            )
        tail_evidence = _require_mapping(
            backups.get("train_tail"), field="receipt train tail"
        )
        if (
            tail_evidence.get("canonical_sha256") != expected["train_log"]
            or tail_evidence.get("tail_sha256") != train_evidence["tail_sha256"]
        ):
            _fail("PREPARED train-tail evidence drifted")
        _verify_archive_evidence(
            tail_evidence,
            backup_dir=paths["backup_dir"],
            expected_sha256=hashlib.sha256(tail).hexdigest(),
        )

        backup_contract = _require_mapping(
            load_json(backup_paths["run_contract"]), field="backup run contract"
        )
        backup_last = _load_cpu_checkpoint(backup_paths["last_checkpoint"])
        backup_best = _load_cpu_checkpoint(backup_paths["best_checkpoint"])
        if backup_contract.get("schema_version") != STAGE4_SCHEMA:
            _fail("PREPARED backup run-contract schema drifted")
        old_provenance = _require_mapping(
            backup_contract.get("provenance"), field="backup old provenance"
        )
        _validate_checkpoint(
            backup_last,
            role="last",
            validation=_validate_validation_latest(
                paths["validation_latest"], expected["validation_latest"]
            ),
        )
        _validate_checkpoint(
            backup_best,
            role="best",
            validation=_validate_validation_latest(
                paths["validation_latest"], expected["validation_latest"]
            ),
        )
        if (
            backup_last.get("provenance") != old_provenance
            or backup_best.get("provenance") != old_provenance
        ):
            _fail("PREPARED backup three-way provenance identity drifted")
        current_semantic, old_semantic = _validate_semantic_sources(
            root=paths["project_root"],
            old_provenance=old_provenance,
            expected_old_source_sha256=expected_old_stage4_source_sha256,
            expected_new_source_sha256=expected_new_stage4_source_sha256,
            expected_count=expected_semantic_source_count,
        )
        new_provenance = copy.deepcopy(dict(old_provenance))
        new_semantic = dict(old_semantic)
        new_semantic[ALLOWED_SOURCE_PATH] = current_semantic[ALLOWED_SOURCE_PATH]
        new_provenance["semantic_source_sha256"] = new_semantic
        new_provenance[ROUTING_PROVENANCE_KEY] = routing
        if (
            receipt.get("exact_provenance_leaf_diff")
            != _exact_provenance_diff(old_provenance, new_provenance)
            or old.get("provenance_json_sha256") != sha256_json(dict(old_provenance))
            or new.get("provenance_json_sha256") != sha256_json(new_provenance)
            or receipt.get("semantic_source_count") != expected_semantic_source_count
            or receipt.get("unchanged_semantic_source_count")
            != expected_semantic_source_count - 1
            or receipt.get("both_checkpoints_bit_exact_outside_provenance") is not True
            or receipt.get("run_contract_bit_exact_outside_provenance") is not True
        ):
            _fail("PREPARED exact provenance/bit-exact evidence drifted")
        for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
            if live_before[label] == expected[label]:
                continue
            if label == "run_contract":
                live_contract = _require_mapping(
                    load_json(paths[label]), field="live new run contract"
                )
                live_provenance = _require_mapping(
                    live_contract.get("provenance"),
                    field="live new run provenance",
                )
                _assert_bit_exact(
                    {
                        key: value
                        for key, value in backup_contract.items()
                        if key != "provenance"
                    },
                    {
                        key: value
                        for key, value in live_contract.items()
                        if key != "provenance"
                    },
                    path="PREPARED.live_run_contract.outside_provenance",
                )
            else:
                live_checkpoint = _load_cpu_checkpoint(paths[label])
                live_provenance = _require_mapping(
                    live_checkpoint.get("provenance"),
                    field=f"live new {label} provenance",
                )
                _section_evidence(
                    backup_last if label == "last_checkpoint" else backup_best,
                    live_checkpoint,
                )
            if live_provenance != new_provenance:
                _fail(f"PREPARED live new provenance drifted: {label}")

        if already_recovered:
            if (
                any(
                    live_before[label] != expected[label]
                    for label in ("run_contract", "last_checkpoint", "best_checkpoint")
                )
                or receipt.get("recovery_confirmation_token_sha256")
                != hashlib.sha256(RECOVERY_CONFIRMATION_TOKEN.encode()).hexdigest()
                or receipt.get("backup_read_only_after_recovery") is not True
                or receipt.get("protected_evidence_unchanged_after_recovery")
                is not True
            ):
                _fail("finalized PREPARED recovery evidence drifted")
            return dict(receipt)

        rollback_errors: list[str] = []
        for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
            try:
                _restore_from_backup(
                    backup_paths[label], paths[label], mode=canonical_modes[label]
                )
                if sha256_file(paths[label]) != expected[label]:
                    rollback_errors.append(f"{label} rollback hash mismatch")
            except BaseException as exc:
                rollback_errors.append(f"{label}: {exc}")
        if rollback_errors:
            _fail("PREPARED recovery was incomplete: " + "; ".join(rollback_errors))
        _protected_hashes(paths, protected_expected)
        recovered = dict(receipt)
        recovered["status"] = "ROLLED_BACK_FROM_PREPARED"
        recovered["recovered_utc"] = utc_now_iso()
        recovered["recovered_from_live_sha256"] = live_before
        recovered["recovery_confirmation_token_sha256"] = hashlib.sha256(
            RECOVERY_CONFIRMATION_TOKEN.encode()
        ).hexdigest()
        recovered["backup_read_only_after_recovery"] = True
        recovered["protected_evidence_unchanged_after_recovery"] = True
        _assert_cpu_only()
        _publish_receipt(receipt_path, recovered)
        return recovered


def _load_routing_json(path: Path) -> Mapping[str, Any]:
    _reject_symlink_chain(path, label="calibration routing JSON")
    return _require_mapping(load_json(path), field="calibration routing JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--last-checkpoint", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-latest", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--calibration-history", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--calibration-history-routing-json", type=Path)
    parser.add_argument("--expected-routing-schema", default=AUDITED_ROUTING_SCHEMA)
    parser.add_argument("--expected-routing-sha256", default=AUDITED_ROUTING_SHA256)
    parser.add_argument(
        "--expected-run-contract-sha256", default=AUDITED_RUN_CONTRACT_SHA256
    )
    parser.add_argument(
        "--expected-last-checkpoint-sha256", default=AUDITED_LAST_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--expected-best-checkpoint-sha256", default=AUDITED_BEST_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--expected-validation-latest-sha256",
        default=AUDITED_VALIDATION_LATEST_SHA256,
    )
    parser.add_argument("--expected-train-log-sha256", default=AUDITED_TRAIN_LOG_SHA256)
    parser.add_argument(
        "--expected-calibration-history-sha256",
        default=AUDITED_CALIBRATION_HISTORY_SHA256,
    )
    parser.add_argument("--expected-state-sha256", default=AUDITED_STATE_SHA256)
    parser.add_argument(
        "--expected-failure-log-sha256", default=AUDITED_FAILURE_LOG_SHA256
    )
    parser.add_argument(
        "--expected-old-stage4-source-sha256",
        default=AUDITED_OLD_STAGE4_SOURCE_SHA256,
    )
    parser.add_argument(
        "--expected-new-stage4-source-sha256",
        default=AUDITED_NEW_STAGE4_SOURCE_SHA256,
    )
    parser.add_argument(
        "--expected-semantic-source-count",
        type=int,
        default=EXPECTED_SEMANTIC_SOURCE_COUNT,
    )
    parser.add_argument(
        "--expected-train-log-line-count",
        type=int,
        default=EXPECTED_TRAIN_LOG_LINE_COUNT,
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--recover-prepared", action="store_true")
    parser.add_argument("--confirmation-token")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        routing = (
            AUDITED_CALIBRATION_HISTORY_ROUTING
            if arguments.calibration_history_routing_json is None
            else _load_routing_json(arguments.calibration_history_routing_json)
        )
    except (Stage4CalibrationRoutingMigrationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    common = {
        "project_root": arguments.project_root,
        "run_contract": arguments.run_contract,
        "last_checkpoint": arguments.last_checkpoint,
        "best_checkpoint": arguments.best_checkpoint,
        "validation_latest": arguments.validation_latest,
        "train_log": arguments.train_log,
        "calibration_history": arguments.calibration_history,
        "state": arguments.state,
        "failure_log": arguments.failure_log,
        "backup_dir": arguments.backup_dir,
        "expected_run_contract_sha256": arguments.expected_run_contract_sha256,
        "expected_last_checkpoint_sha256": arguments.expected_last_checkpoint_sha256,
        "expected_best_checkpoint_sha256": arguments.expected_best_checkpoint_sha256,
        "expected_validation_latest_sha256": arguments.expected_validation_latest_sha256,
        "expected_train_log_sha256": arguments.expected_train_log_sha256,
        "expected_calibration_history_sha256": arguments.expected_calibration_history_sha256,
        "expected_state_sha256": arguments.expected_state_sha256,
        "expected_failure_log_sha256": arguments.expected_failure_log_sha256,
        "expected_old_stage4_source_sha256": arguments.expected_old_stage4_source_sha256,
        "expected_new_stage4_source_sha256": arguments.expected_new_stage4_source_sha256,
        "calibration_history_routing": routing,
        "expected_routing_schema": arguments.expected_routing_schema,
        "expected_routing_sha256": arguments.expected_routing_sha256,
        "expected_semantic_source_count": arguments.expected_semantic_source_count,
        "expected_train_log_line_count": arguments.expected_train_log_line_count,
    }
    try:
        if arguments.recover_prepared:
            receipt = recover_prepared_stage4_step4000_calibration_history_provenance(
                **common, confirmation_token=arguments.confirmation_token
            )
        else:
            receipt = migrate_stage4_step4000_calibration_history_provenance(
                **common,
                execute=arguments.execute,
                confirmation_token=arguments.confirmation_token,
            )
    except (Stage4CalibrationRoutingMigrationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
