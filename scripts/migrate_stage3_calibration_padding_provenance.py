#!/usr/bin/env python3
"""Migrate one Stage3 source leaf after the final calibration padding fix.

Stage3 training and all ranking validations completed at step 12000.  The
one-time presence-only calibration then failed because non-multiple-of-eight
inputs were not padded before the planner forward.  The fix changes only
``src/training/stage3_engine.py``.  This CPU-only tool publishes that single
semantic-source leaf to the canonical run contract, raw/resumable ``last``,
and EMA-selection ``best`` checkpoint while requiring every other checkpoint
section to remain bit-exact.

Both earlier Stage3 migration receipts and their read-only, same-filesystem
backups are immutable prerequisites.  Execution and interrupted-PREPARED
recovery use separate exact confirmation tokens.  Without ``--execute`` this
tool only constructs, reloads, and verifies temporary candidates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

# Must precede torch transitively imported by the prior migration module.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AUDITED_PRIOR_MIGRATION_SCRIPT_SHA256 = (
    "0d03b1f1a1529d5fed38bc70a1c0c741aed9e46904df2ca77289bc0945970a1f"
)
PRIOR_MIGRATION_SCRIPT_PATH = (
    PROJECT_ROOT / "scripts/migrate_stage3_ema_device_provenance.py"
)


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if (
    PRIOR_MIGRATION_SCRIPT_PATH.is_symlink()
    or not PRIOR_MIGRATION_SCRIPT_PATH.is_file()
    or _raw_sha256(PRIOR_MIGRATION_SCRIPT_PATH) != AUDITED_PRIOR_MIGRATION_SCRIPT_SHA256
):
    raise RuntimeError("audited EMA-device migration implementation drifted")

from scripts import migrate_stage3_ema_device_provenance as prior  # noqa: E402
from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.provenance import semantic_source_hashes  # noqa: E402
from src.utils.hashing import is_sha256, sha256_file, sha256_json  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    fsync_directory,
    load_json,
    utc_now_iso,
)


PROTOCOL_ID = prior.PROTOCOL_ID
CHECKPOINT_SCHEMA = prior.CHECKPOINT_SCHEMA
STAGE3_SCHEMA = prior.STAGE3_SCHEMA
APPROVAL_SCHEMA = prior.APPROVAL_SCHEMA
EXPECTED_BINDING_COUNT = prior.EXPECTED_BINDING_COUNT
ENTRYPOINTS = prior.ENTRYPOINTS

MIGRATION_STEP = 12_000
EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT = 20
EXPECTED_UNCHANGED_TOP_LEVEL_COUNT = 19
EXPECTED_SEMANTIC_SOURCE_COUNT = 47

RECEIPT_SCHEMA = "graphrestore-stage3-calibration-padding-migration-v1"
MIGRATION_KIND = "stage3_complete_12000_calibration_padding_provenance_only"
CONFIRMATION_TOKEN = "MIGRATE_STAGE3_COMPLETE_12000_CALIBRATION_PADDING_PROVENANCE"
RECOVERY_CONFIRMATION_TOKEN = (
    "RECOVER_STAGE3_COMPLETE_12000_CALIBRATION_PADDING_PROVENANCE"
)
BACKUP_DIR_NAME = "stage3_calibration_padding_complete12000_v1"

GUARD_BACKUP_DIR_NAME = "stage3_guard_alignment_pending2000_v1"
GUARD_RECEIPT_SCHEMA = "graphrestore-stage3-guard-alignment-migration-v1"
GUARD_MIGRATION_KIND = "stage3_pending_2000_guard_alignment_provenance_only"
EMA_BACKUP_DIR_NAME = "stage3_ema_device_pending2000_v1"
EMA_RECEIPT_SCHEMA = "graphrestore-stage3-ema-device-migration-v1"
EMA_MIGRATION_KIND = "stage3_pending_2000_ema_device_provenance_only"

ALLOWED_SOURCE_PATH = "src/training/stage3_engine.py"
PRESERVED_STAGE4_SOURCE_PATH = "src/training/stage4_engine.py"
EXPECTED_PROVENANCE_DIFF_PATH = f"semantic_source_sha256.{ALLOWED_SOURCE_PATH}"

# Audited canonical anchors at the calibration failure boundary.
AUDITED_RUN_CONTRACT_SHA256 = (
    "d98b7493b41a0ace9fcb228c50b3acbdf855f092bb2ddc9c9f479730cecf053f"
)
AUDITED_LAST_CHECKPOINT_SHA256 = (
    "39733371064c282e46e858aaf50df7b0d4a9fdf3c49c5bc8838798b4958e2438"
)
AUDITED_BEST_CHECKPOINT_SHA256 = (
    "b26ebca987fae140bbaff8a7b530692f7a4e0113bdeea863547b6aaec8958b20"
)
AUDITED_STATE_SHA256 = (
    "876a3fffada00db1ad9c87891f94a23d751fb626005c9b7e5818a5a2e31b888d"
)
AUDITED_APPROVAL_SHA256 = (
    "7b351c0958aa681dc1f65114e801c58e3a5bc4bb7cc73c06507c0b647e51a08b"
)
AUDITED_APPROVAL_REQUIRED_SHA256 = (
    "33be4aba2c4229175ac33edef7a5914a48a249b8c733d86338c64a8662072825"
)
AUDITED_GUARD_RECEIPT_SHA256 = (
    "449bd49b3e31a430eed1d4c6e217c4299084beb272d9845648ded95b7f8718e6"
)
AUDITED_EMA_RECEIPT_SHA256 = (
    "9848708c1a2dc91a99230a68ebf630c8574c64b6cbc8bad97700b5846efc21cb"
)
AUDITED_OLD_STAGE3_SOURCE_SHA256 = (
    "908bcd7ff829aabba8376ec949156890983f51924aaa7e2313e013648d817b49"
)
AUDITED_NEW_STAGE3_SOURCE_SHA256 = (
    "2ba4c211476b2aa8a374e608000660dc024966c4094d79aeff8adc506431f796"
)
AUDITED_STAGE4_SOURCE_SHA256 = (
    "e2fbfbc2ee580b90cb92c48e6b289d6bc6d3d4651c42d34295ce07fc664814b6"
)

Stage3CalibrationPaddingMigrationError = prior.Stage3EMADeviceMigrationError
_assert_bit_exact = prior._assert_bit_exact
_fingerprint = prior._fingerprint
_hardlink_backup = prior._hardlink_backup
_load_cpu_checkpoint = prior._load_cpu_checkpoint
_make_backup_read_only = prior._make_backup_read_only
_make_candidate = prior._make_candidate
_replace_and_fsync = prior._replace_and_fsync
_require_mapping = prior._require_mapping
_restore_from_backup = prior._restore_from_backup
_section_evidence = prior._section_evidence
_walk_finite = prior._walk_finite


def _fail(message: str) -> NoReturn:
    raise Stage3CalibrationPaddingMigrationError(message)


def _assert_cpu_only() -> None:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or prior.shared.torch.cuda.is_initialized()
    ):
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


def _validate_prior_module() -> dict[str, Any]:
    path = PRIOR_MIGRATION_SCRIPT_PATH
    required = (
        "_assert_bit_exact",
        "_fingerprint",
        "_hardlink_backup",
        "_load_cpu_checkpoint",
        "_make_backup_read_only",
        "_make_candidate",
        "_replace_and_fsync",
        "_require_mapping",
        "_restore_from_backup",
        "_section_evidence",
        "_walk_finite",
    )
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != AUDITED_PRIOR_MIGRATION_SCRIPT_SHA256
        or Path(prior.__file__).resolve() != path.resolve()
        or any(not callable(getattr(prior, name, None)) for name in required)
    ):
        _fail("audited EMA-device migration implementation drifted")
    return {
        "path": str(path.resolve()),
        "sha256": AUDITED_PRIOR_MIGRATION_SCRIPT_SHA256,
        "primitive_names": list(required),
        "protected_unchanged": True,
    }


def _resolve_requested_paths(**raw: str | Path) -> dict[str, Path]:
    paths = {key: Path(value) for key, value in raw.items()}
    for label, path in paths.items():
        _reject_symlink_chain(path, label=label.replace("_", " "))
    return {key: path.resolve() for key, path in paths.items()}


def _validate_hashes(*values: str) -> None:
    if any(not is_sha256(value) for value in values):
        _fail("every expected hash must be a lowercase SHA256")


def _validate_paths(paths: Mapping[str, Path]) -> None:
    root = paths["project_root"]
    migrations = (root / "artifacts/migrations").resolve()
    backup_dir = paths["backup_dir"]
    if (
        (root / "artifacts").is_symlink()
        or (root / "artifacts/migrations").is_symlink()
        or backup_dir.is_symlink()
        or backup_dir.parent != migrations
        or backup_dir.name != BACKUP_DIR_NAME
    ):
        _fail("backup directory is not the dedicated calibration-padding path")
    if backup_dir in {
        paths["guard_receipt"].parent,
        paths["ema_receipt"].parent,
    }:
        _fail("calibration migration may not reuse a prior backup directory")
    for label in (
        "run_contract",
        "last_checkpoint",
        "best_checkpoint",
        "state",
        "approval",
        "approval_required",
        "guard_receipt",
        "ema_receipt",
    ):
        path = paths[label]
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise Stage3CalibrationPaddingMigrationError(
                f"{label} escaped project root: {path}"
            ) from exc
        if path.is_symlink() or not path.is_file():
            _fail(f"missing or symlinked {label}: {path}")


def _validate_failed_state(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        _fail("orchestration-state SHA256 drifted")
    state = _require_mapping(load_json(path), field="orchestration state")
    command = state.get("last_command")
    if (
        state.get("schema_version") != "graphrestore-orchestration-v1"
        or state.get("protocol_id") != PROTOCOL_ID
        or state.get("status") != "FAILED"
        or state.get("current_stage") != "FAILED"
        or state.get("gpu") != "released"
        or state.get("last_exit_code") != 1
        or state.get("next_command")
        != "python scripts/orchestrate.py --resume_post_approval_pipeline"
        or not isinstance(command, list)
        or "scripts/train_stage3_planner.py" not in command
        or "--resume" not in command
    ):
        _fail("orchestration state is not the exact exit-1 Stage3 boundary")
    return dict(state)


def _validate_approval(
    approval: Path,
    required: Path,
    *,
    expected_approval_sha256: str,
    expected_required_sha256: str,
) -> dict[str, Any]:
    return prior._validate_approval(
        approval,
        required,
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_required_sha256,
    )


def _verify_receipt_backups(
    receipt_path: Path,
    raw_backups: object,
    *,
    expected_labels: set[str],
) -> dict[str, dict[str, Any]]:
    backups = _require_mapping(raw_backups, field="prior receipt.backup")
    if set(backups) != expected_labels:
        _fail(f"prior receipt backup labels drifted: {receipt_path}")
    verified: dict[str, dict[str, Any]] = {}
    for label, raw in backups.items():
        evidence = _require_mapping(raw, field=f"prior receipt.backup.{label}")
        raw_path = evidence.get("path")
        expected_sha = evidence.get("sha256")
        inode = evidence.get("inode")
        device = evidence.get("device")
        if (
            not isinstance(raw_path, str)
            or not is_sha256(expected_sha)
            or not isinstance(inode, int)
            or not isinstance(device, int)
        ):
            _fail(f"invalid prior backup evidence: {label}")
        requested = Path(raw_path)
        _reject_symlink_chain(requested, label=f"prior {label} backup")
        path = requested.resolve()
        if (
            path.parent != receipt_path.parent
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_sha
            or stat.S_IMODE(path.stat().st_mode) != 0o444
            or path.stat().st_ino != inode
            or path.stat().st_dev != device
            or evidence.get("hard_link_verified") is not True
        ):
            _fail(f"prior migration backup drifted: {label}")
        verified[str(label)] = {
            "path": str(path),
            "sha256": str(expected_sha),
            "mode": 0o444,
            "inode": inode,
            "device": device,
        }
    return verified


def _validate_prior_migrations(
    *,
    root: Path,
    guard_receipt: Path,
    ema_receipt: Path,
    expected_guard_sha256: str,
    expected_ema_sha256: str,
    expected_old_stage3_sha256: str,
    expected_stage4_sha256: str,
) -> dict[str, Any]:
    expected_guard_path = (
        root / "artifacts/migrations" / GUARD_BACKUP_DIR_NAME / "MIGRATION_RECEIPT.json"
    ).resolve()
    expected_ema_path = (
        root / "artifacts/migrations" / EMA_BACKUP_DIR_NAME / "MIGRATION_RECEIPT.json"
    ).resolve()
    if guard_receipt != expected_guard_path or ema_receipt != expected_ema_path:
        _fail("prior migration receipt path drifted")
    if (
        sha256_file(guard_receipt) != expected_guard_sha256
        or sha256_file(ema_receipt) != expected_ema_sha256
    ):
        _fail("prior COMPLETE migration receipt SHA256 drifted")

    guard = _require_mapping(load_json(guard_receipt), field="guard receipt")
    guard_diff = guard.get("exact_provenance_leaf_diff")
    if (
        guard.get("schema_version") != GUARD_RECEIPT_SCHEMA
        or guard.get("protocol_id") != PROTOCOL_ID
        or guard.get("migration") != GUARD_MIGRATION_KIND
        or guard.get("status") != "COMPLETE"
        or guard.get("backup_read_only_after_publication") is not True
        or not isinstance(guard_diff, list)
        or len(guard_diff) != 2
        or any(not isinstance(row, Mapping) for row in guard_diff)
        or sorted(row.get("path") for row in guard_diff if isinstance(row, Mapping))
        != sorted(
            [
                f"semantic_source_sha256.{ALLOWED_SOURCE_PATH}",
                f"semantic_source_sha256.{PRESERVED_STAGE4_SOURCE_PATH}",
            ]
        )
    ):
        _fail("prior guard-alignment COMPLETE receipt contract drifted")
    guard_backups = _verify_receipt_backups(
        guard_receipt,
        guard.get("backup"),
        expected_labels={"run_contract", "checkpoint"},
    )

    ema = _require_mapping(load_json(ema_receipt), field="EMA-device receipt")
    ema_diff = ema.get("exact_provenance_leaf_diff")
    embedded_guard = _require_mapping(
        ema.get("prior_guard_alignment_migration"),
        field="EMA-device receipt prior guard migration",
    )
    preserved_stage4 = _require_mapping(
        ema.get("preserved_stage4_source"),
        field="EMA-device receipt preserved Stage4",
    )
    if (
        ema.get("schema_version") != EMA_RECEIPT_SCHEMA
        or ema.get("protocol_id") != PROTOCOL_ID
        or ema.get("migration") != EMA_MIGRATION_KIND
        or ema.get("status") != "COMPLETE"
        or ema.get("migration_script_sha256") != AUDITED_PRIOR_MIGRATION_SCRIPT_SHA256
        or ema.get("backup_read_only_after_publication") is not True
        or ema.get("prior_guard_alignment_receipt_unchanged_after_publication")
        is not True
        or ema_diff
        != [
            {
                "path": EXPECTED_PROVENANCE_DIFF_PATH,
                "old": prior.AUDITED_OLD_STAGE3_SOURCE_SHA256,
                "new": expected_old_stage3_sha256,
            }
        ]
        or embedded_guard.get("path") != str(guard_receipt)
        or embedded_guard.get("sha256") != expected_guard_sha256
        or embedded_guard.get("status") != "COMPLETE"
        or embedded_guard.get("protected_unchanged") is not True
        or preserved_stage4.get("path") != PRESERVED_STAGE4_SOURCE_PATH
        or preserved_stage4.get("sha256") != expected_stage4_sha256
    ):
        _fail("prior EMA-device COMPLETE receipt contract drifted")
    ema_backups = _verify_receipt_backups(
        ema_receipt,
        ema.get("backup"),
        expected_labels={"run_contract", "checkpoint"},
    )
    return {
        "guard_alignment": {
            "path": str(guard_receipt),
            "sha256": expected_guard_sha256,
            "schema_version": GUARD_RECEIPT_SCHEMA,
            "migration": GUARD_MIGRATION_KIND,
            "status": "COMPLETE",
            "backup": guard_backups,
            "protected_unchanged": True,
        },
        "ema_device": {
            "path": str(ema_receipt),
            "sha256": expected_ema_sha256,
            "schema_version": EMA_RECEIPT_SCHEMA,
            "migration": EMA_MIGRATION_KIND,
            "status": "COMPLETE",
            "backup": ema_backups,
            "protected_unchanged": True,
        },
    }


def _validate_checkpoint(payload: Mapping[str, Any], *, role: str) -> None:
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": "stage3",
        "step": MIGRATION_STEP,
        "model_role": role,
        "resumable": role == "raw_training_state",
        "pending_validation_step": None,
        "optimizer_transaction_active": False,
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "scaler": None,
    }
    for key, value in expected.items():
        if payload.get(key, object()) != value:
            _fail(f"{role} checkpoint header mismatch at {key}")
    if len(payload) != EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT:
        _fail(f"{role} checkpoint top-level section count drifted")
    metrics = _require_mapping(payload.get("metrics"), field=f"{role}.metrics")
    if (
        metrics.get("validation_step") != MIGRATION_STEP
        or metrics.get("best_step") != MIGRATION_STEP
    ):
        _fail(f"{role} checkpoint is not the selected step-12000 validation")
    sampler = _require_mapping(
        payload.get("sampler_state"), field=f"{role}.sampler_state"
    )
    if (
        sampler.get("consumed_optimizer_step") != MIGRATION_STEP
        or sampler.get("sample_cursor") != MIGRATION_STEP * 8
    ):
        _fail(f"{role} checkpoint sampler boundary drifted")
    ema = _require_mapping(payload.get("ema"), field=f"{role}.ema")
    if (
        ema.get("num_updates") != MIGRATION_STEP
        or ema.get("scope") != "planner_parameters_only_executor_bitwise_frozen"
    ):
        _fail(f"{role} checkpoint EMA policy drifted")
    _walk_finite(payload)


def _validate_checkpoint_pair(last: Mapping[str, Any], best: Mapping[str, Any]) -> None:
    _validate_checkpoint(last, role="raw_training_state")
    _validate_checkpoint(best, role="ema_selection")
    if last.get("provenance") != best.get("provenance"):
        _fail("last/best provenance differs before migration")
    last_ema = _require_mapping(last.get("ema"), field="last.ema")
    best_ema = _require_mapping(best.get("ema"), field="best.ema")
    _assert_bit_exact(
        last_ema.get("shadow"), best.get("model"), path="last_ema.best_model"
    )
    _assert_bit_exact(best.get("model"), best_ema.get("shadow"), path="best.model_ema")


def _validate_provenance_identity(
    provenance: Mapping[str, Any],
    approval: Mapping[str, Any],
    *,
    expected_approval_sha256: str,
    expected_required_sha256: str,
) -> None:
    stage3_approval = _require_mapping(
        provenance.get("stage3_approval"), field="provenance.stage3_approval"
    )
    if (
        provenance.get("protocol_id") != PROTOCOL_ID
        or stage3_approval.get("sha256") != expected_approval_sha256
        or stage3_approval.get("approval_required_sha256") != expected_required_sha256
        or provenance.get("bindings") != approval.get("bindings")
    ):
        _fail("checkpoint provenance approval/binding identity drifted")


def _validate_semantic_sources(
    *,
    root: Path,
    old_provenance: Mapping[str, Any],
    expected_old_stage3_sha256: str,
    expected_new_stage3_sha256: str,
    expected_stage4_sha256: str,
    expected_count: int,
) -> tuple[dict[str, str], Mapping[str, Any]]:
    old = _require_mapping(
        old_provenance.get("semantic_source_sha256"),
        field="provenance.semantic_source_sha256",
    )
    current = semantic_source_hashes(root, entrypoints=ENTRYPOINTS)
    if (
        len(old) != expected_count
        or len(current) != expected_count
        or old.keys() != current.keys()
    ):
        _fail("semantic-source path/count contract drifted")
    changed = sorted(path for path in old if old[path] != current[path])
    if changed != [ALLOWED_SOURCE_PATH]:
        _fail(f"physical semantic-source drift is not exactly Stage3: {changed}")
    if (
        old.get(ALLOWED_SOURCE_PATH) != expected_old_stage3_sha256
        or current.get(ALLOWED_SOURCE_PATH) != expected_new_stage3_sha256
        or expected_old_stage3_sha256 == expected_new_stage3_sha256
        or old.get(PRESERVED_STAGE4_SOURCE_PATH) != expected_stage4_sha256
        or current.get(PRESERVED_STAGE4_SOURCE_PATH) != expected_stage4_sha256
    ):
        _fail("Stage3 old/new or preserved Stage4 source SHA drifted")
    return current, old


def _single_leaf_diff(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> list[dict[str, str]]:
    old_flat = prior.shared._flatten_provenance(old)
    new_flat = prior.shared._flatten_provenance(new)
    if old_flat.keys() != new_flat.keys():
        _fail("provenance leaf set changed")
    changed = sorted(path for path in old_flat if old_flat[path] != new_flat[path])
    if changed != [EXPECTED_PROVENANCE_DIFF_PATH]:
        _fail(f"unexpected provenance diff: {changed}")
    return [
        {
            "path": EXPECTED_PROVENANCE_DIFF_PATH,
            "old": str(old_flat[EXPECTED_PROVENANCE_DIFF_PATH]),
            "new": str(new_flat[EXPECTED_PROVENANCE_DIFF_PATH]),
        }
    ]


def _restore_three(
    *,
    backups: Mapping[str, Any],
    destinations: Mapping[str, Path],
    expected_old: Mapping[str, str],
) -> list[str]:
    errors: list[str] = []
    for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
        destination = destinations[label]
        expected_sha = expected_old[label]
        try:
            raw = backups.get(label)
            if not isinstance(raw, Mapping):
                if destination.is_file() and sha256_file(destination) == expected_sha:
                    continue
                _fail(f"missing rollback backup evidence: {label}")
            evidence = raw
            raw_path, mode = evidence.get("path"), evidence.get("mode")
            if not isinstance(raw_path, str) or not isinstance(mode, int):
                _fail(f"invalid rollback backup evidence: {label}")
            requested = Path(raw_path)
            _reject_symlink_chain(requested, label=f"rollback {label} backup")
            backup = requested.resolve()
            if sha256_file(backup) != expected_sha:
                _fail(f"rollback backup SHA256 mismatch: {label}")
            _restore_from_backup(backup, destination, mode=mode)
            if (
                sha256_file(destination) != expected_sha
                or stat.S_IMODE(destination.stat().st_mode) != mode
                or os.path.samestat(destination.stat(), backup.stat())
            ):
                _fail(f"rollback output mismatch: {label}")
            _make_backup_read_only(backup)
        except BaseException as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    return errors


def migrate_stage3_calibration_padding_provenance(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    last_checkpoint: str | Path,
    best_checkpoint: str | Path,
    state: str | Path,
    approval: str | Path,
    approval_required: str | Path,
    guard_migration_receipt: str | Path,
    ema_migration_receipt: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_last_checkpoint_sha256: str,
    expected_best_checkpoint_sha256: str,
    expected_state_sha256: str,
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
    expected_guard_migration_receipt_sha256: str,
    expected_ema_migration_receipt_sha256: str,
    expected_old_stage3_source_sha256: str,
    expected_new_stage3_source_sha256: str,
    expected_unchanged_stage4_source_sha256: str,
    expected_semantic_source_count: int = EXPECTED_SEMANTIC_SOURCE_COUNT,
    execute: bool = False,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Build or publish the exact three-artifact, one-leaf migration."""

    paths = _resolve_requested_paths(
        project_root=project_root,
        run_contract=run_contract,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
        state=state,
        approval=approval,
        approval_required=approval_required,
        guard_receipt=guard_migration_receipt,
        ema_receipt=ema_migration_receipt,
        backup_dir=backup_dir,
    )
    _assert_cpu_only()
    _validate_hashes(
        expected_run_contract_sha256,
        expected_last_checkpoint_sha256,
        expected_best_checkpoint_sha256,
        expected_state_sha256,
        expected_approval_sha256,
        expected_approval_required_sha256,
        expected_guard_migration_receipt_sha256,
        expected_ema_migration_receipt_sha256,
        expected_old_stage3_source_sha256,
        expected_new_stage3_source_sha256,
        expected_unchanged_stage4_source_sha256,
    )
    if (
        isinstance(expected_semantic_source_count, bool)
        or not isinstance(expected_semantic_source_count, int)
        or expected_semantic_source_count < 2
    ):
        _fail("semantic-source count must be an integer >= 2")
    if execute and confirmation_token != CONFIRMATION_TOKEN:
        _fail("execution requires the exact calibration-padding migration token")
    _validate_paths(paths)
    if paths["backup_dir"].exists():
        _fail(f"dedicated backup directory already exists: {paths['backup_dir']}")

    expected_old = {
        "run_contract": expected_run_contract_sha256,
        "last_checkpoint": expected_last_checkpoint_sha256,
        "best_checkpoint": expected_best_checkpoint_sha256,
    }
    for label, expected_sha in expected_old.items():
        if sha256_file(paths[label]) != expected_sha:
            _fail(f"{label} SHA256 differs from the audited step-12000 anchor")

    prior_module_evidence = _validate_prior_module()
    state_evidence = _validate_failed_state(paths["state"], expected_state_sha256)
    approval_evidence = _validate_approval(
        paths["approval"],
        paths["approval_required"],
        expected_approval_sha256=expected_approval_sha256,
        expected_required_sha256=expected_approval_required_sha256,
    )
    prior_evidence = _validate_prior_migrations(
        root=paths["project_root"],
        guard_receipt=paths["guard_receipt"],
        ema_receipt=paths["ema_receipt"],
        expected_guard_sha256=expected_guard_migration_receipt_sha256,
        expected_ema_sha256=expected_ema_migration_receipt_sha256,
        expected_old_stage3_sha256=expected_old_stage3_source_sha256,
        expected_stage4_sha256=expected_unchanged_stage4_source_sha256,
    )

    contract = _require_mapping(
        load_json(paths["run_contract"]), field="Stage3 run contract"
    )
    if contract.get("schema_version") != STAGE3_SCHEMA:
        _fail("Stage3 run-contract schema drifted")
    old_provenance = _require_mapping(
        contract.get("provenance"), field="run contract provenance"
    )
    last = _load_cpu_checkpoint(paths["last_checkpoint"])
    best = _load_cpu_checkpoint(paths["best_checkpoint"])
    if (
        sha256_file(paths["last_checkpoint"]) != expected_last_checkpoint_sha256
        or sha256_file(paths["best_checkpoint"]) != expected_best_checkpoint_sha256
    ):
        _fail("checkpoint changed during CPU load")
    _validate_checkpoint_pair(last, best)
    if last.get("provenance") != old_provenance:
        _fail("run-contract and checkpoint provenance differ before migration")
    approval_value = _require_mapping(
        load_json(paths["approval"]), field="Stage3 approval"
    )
    _validate_provenance_identity(
        old_provenance,
        approval_value,
        expected_approval_sha256=expected_approval_sha256,
        expected_required_sha256=expected_approval_required_sha256,
    )
    current_semantic, old_semantic = _validate_semantic_sources(
        root=paths["project_root"],
        old_provenance=old_provenance,
        expected_old_stage3_sha256=expected_old_stage3_source_sha256,
        expected_new_stage3_sha256=expected_new_stage3_source_sha256,
        expected_stage4_sha256=expected_unchanged_stage4_source_sha256,
        expected_count=expected_semantic_source_count,
    )

    new_provenance = copy.deepcopy(dict(old_provenance))
    new_semantic = dict(old_semantic)
    new_semantic[ALLOWED_SOURCE_PATH] = current_semantic[ALLOWED_SOURCE_PATH]
    new_provenance["semantic_source_sha256"] = new_semantic
    provenance_diff = _single_leaf_diff(old_provenance, new_provenance)

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
            _fail(f"{label} unchanged top-level section count drifted")

    candidates = {
        "run_contract": _make_candidate(
            paths["run_contract"].parent,
            paths["run_contract"].name,
            ".calibration-padding.candidate.json",
        ),
        "last_checkpoint": _make_candidate(
            paths["last_checkpoint"].parent,
            paths["last_checkpoint"].name,
            ".calibration-padding.candidate.pth",
        ),
        "best_checkpoint": _make_candidate(
            paths["best_checkpoint"].parent,
            paths["best_checkpoint"].name,
            ".calibration-padding.candidate.pth",
        ),
    }
    backups: dict[str, Any] = {}
    receipt_path = paths["backup_dir"] / "MIGRATION_RECEIPT.json"
    try:
        atomic_write_json(candidates["run_contract"], new_contract)
        if load_json(candidates["run_contract"]) != new_contract:
            _fail("run-contract candidate failed JSON round trip")
        atomic_torch_save(new_last, candidates["last_checkpoint"])
        atomic_torch_save(new_best, candidates["best_checkpoint"])
        reloaded_last = _load_cpu_checkpoint(candidates["last_checkpoint"])
        reloaded_best = _load_cpu_checkpoint(candidates["best_checkpoint"])
        _validate_checkpoint_pair(reloaded_last, reloaded_best)
        if (
            reloaded_last.get("provenance") != new_provenance
            or reloaded_best.get("provenance") != new_provenance
        ):
            _fail("checkpoint candidate provenance differs from run contract")
        section_evidence = {
            "last_checkpoint": _section_evidence(last, reloaded_last),
            "best_checkpoint": _section_evidence(best, reloaded_best),
        }
        new_hashes = {label: sha256_file(path) for label, path in candidates.items()}
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "migration": MIGRATION_KIND,
            "status": "PREPARED" if execute else "DRY_RUN",
            "created_utc": utc_now_iso(),
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "step": MIGRATION_STEP,
            "old": {
                label: {"path": str(paths[label]), "sha256": sha}
                for label, sha in expected_old.items()
            }
            | {"provenance_json_sha256": sha256_json(dict(old_provenance))},
            "new": new_hashes | {"provenance_json_sha256": sha256_json(new_provenance)},
            "exact_provenance_leaf_diff": provenance_diff,
            "semantic_source_count": expected_semantic_source_count,
            "unchanged_semantic_source_count": expected_semantic_source_count - 1,
            "preserved_stage4_source": {
                "path": PRESERVED_STAGE4_SOURCE_PATH,
                "sha256": expected_unchanged_stage4_source_sha256,
            },
            "checkpoint_section_fingerprints": section_evidence,
            "checkpoint_top_level_count": EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT,
            "checkpoint_top_level_bit_exact_outside_provenance_count": (
                EXPECTED_UNCHANGED_TOP_LEVEL_COUNT
            ),
            "both_checkpoints_bit_exact_outside_provenance": True,
            "run_contract_bit_exact_outside_provenance": True,
            "all_checkpoint_tensors_finite": True,
            "approval_and_22_bindings_unchanged": approval_evidence,
            "prior_complete_migrations": prior_evidence,
            "prior_migration_implementation": prior_module_evidence,
            "orchestration_state": {
                "path": str(paths["state"]),
                "sha256": expected_state_sha256,
                **{
                    key: state_evidence.get(key)
                    for key in (
                        "status",
                        "current_stage",
                        "gpu",
                        "last_exit_code",
                        "next_command",
                    )
                },
            },
            "execution_confirmation_token_sha256": (
                hashlib.sha256(CONFIRMATION_TOKEN.encode()).hexdigest()
                if execute
                else None
            ),
            "backup": backups,
            "migration_script_sha256": sha256_file(Path(__file__).resolve()),
        }
        if not execute:
            _assert_cpu_only()
            return receipt

        protected_before = {
            "state": expected_state_sha256,
            "approval": expected_approval_sha256,
            "approval_required": expected_approval_required_sha256,
            "guard_receipt": expected_guard_migration_receipt_sha256,
            "ema_receipt": expected_ema_migration_receipt_sha256,
        }
        for label, expected_sha in expected_old.items():
            if sha256_file(paths[label]) != expected_sha:
                _fail(f"{label} changed before publication")
        for label, expected_sha in protected_before.items():
            if sha256_file(paths[label]) != expected_sha:
                _fail(f"protected {label} changed before publication")
        if (
            semantic_source_hashes(paths["project_root"], entrypoints=ENTRYPOINTS)
            != current_semantic
        ):
            _fail("semantic sources changed before publication")
        if _validate_prior_module() != prior_module_evidence:
            _fail("prior migration implementation changed before publication")
        _validate_failed_state(paths["state"], expected_state_sha256)
        _validate_approval(
            paths["approval"],
            paths["approval_required"],
            expected_approval_sha256=expected_approval_sha256,
            expected_required_sha256=expected_approval_required_sha256,
        )
        _validate_prior_migrations(
            root=paths["project_root"],
            guard_receipt=paths["guard_receipt"],
            ema_receipt=paths["ema_receipt"],
            expected_guard_sha256=expected_guard_migration_receipt_sha256,
            expected_ema_sha256=expected_ema_migration_receipt_sha256,
            expected_old_stage3_sha256=expected_old_stage3_source_sha256,
            expected_stage4_sha256=expected_unchanged_stage4_source_sha256,
        )

        artifacts = (paths["project_root"] / "artifacts").resolve()
        migrations = (artifacts / "migrations").resolve()
        if not artifacts.is_dir():
            _fail("project artifacts directory is missing")
        migrations.mkdir(parents=False, exist_ok=True)
        fsync_directory(artifacts)
        paths["backup_dir"].mkdir(parents=False, exist_ok=False)
        fsync_directory(migrations)
        devices = {
            paths["backup_dir"].stat().st_dev,
            paths["run_contract"].stat().st_dev,
            paths["last_checkpoint"].stat().st_dev,
            paths["best_checkpoint"].stat().st_dev,
        }
        if len(devices) != 1:
            _fail("backup directory is not on the same filesystem as all sources")

        backup_paths = {
            "run_contract": paths["backup_dir"]
            / f"run_contract.pre_calibration_padding.{expected_run_contract_sha256}.json",
            "last_checkpoint": paths["backup_dir"]
            / f"last.pre_calibration_padding.{expected_last_checkpoint_sha256}.pth",
            "best_checkpoint": paths["backup_dir"]
            / f"best_ema.pre_calibration_padding.{expected_best_checkpoint_sha256}.pth",
        }
        for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
            backups[label] = _hardlink_backup(paths[label], backup_paths[label])
        receipt["backup"] = backups
        atomic_write_json(receipt_path, receipt)

        for label in ("best_checkpoint", "last_checkpoint", "run_contract"):
            _replace_and_fsync(candidates[label], paths[label])
        for label, expected_sha in new_hashes.items():
            if sha256_file(paths[label]) != expected_sha:
                _fail(f"published {label} hash differs from verified candidate")

        published_contract = _require_mapping(
            load_json(paths["run_contract"]), field="published run contract"
        )
        published_last = _load_cpu_checkpoint(paths["last_checkpoint"])
        published_best = _load_cpu_checkpoint(paths["best_checkpoint"])
        _validate_checkpoint_pair(published_last, published_best)
        if (
            published_contract.get("provenance") != new_provenance
            or published_last.get("provenance") != new_provenance
            or published_best.get("provenance") != new_provenance
        ):
            _fail("published three-way provenance identity failed")
        _section_evidence(last, published_last)
        _section_evidence(best, published_best)
        _assert_bit_exact(
            {key: value for key, value in contract.items() if key != "provenance"},
            {
                key: value
                for key, value in published_contract.items()
                if key != "provenance"
            },
            path="published_run_contract.outside_provenance",
        )
        for label, expected_sha in protected_before.items():
            if sha256_file(paths[label]) != expected_sha:
                _fail(f"protected {label} changed during publication")
        if (
            semantic_source_hashes(paths["project_root"], entrypoints=ENTRYPOINTS)
            != current_semantic
        ):
            _fail("semantic sources changed during publication")
        if _validate_prior_module() != prior_module_evidence:
            _fail("prior migration implementation changed during publication")
        _validate_prior_migrations(
            root=paths["project_root"],
            guard_receipt=paths["guard_receipt"],
            ema_receipt=paths["ema_receipt"],
            expected_guard_sha256=expected_guard_migration_receipt_sha256,
            expected_ema_sha256=expected_ema_migration_receipt_sha256,
            expected_old_stage3_sha256=expected_old_stage3_source_sha256,
            expected_stage4_sha256=expected_unchanged_stage4_source_sha256,
        )
        for path in backup_paths.values():
            _make_backup_read_only(path)
        fsync_directory(paths["backup_dir"])
        receipt["status"] = "COMPLETE"
        receipt["completed_utc"] = utc_now_iso()
        receipt["backup_read_only_after_publication"] = True
        receipt["both_prior_receipts_unchanged_after_publication"] = True
        receipt["orchestration_state_unchanged_after_publication"] = True
        _assert_cpu_only()
        atomic_write_json(receipt_path, receipt)
        return receipt
    except BaseException as original_error:
        if execute and backups:
            rollback_errors = _restore_three(
                backups=backups,
                destinations={
                    label: paths[label]
                    for label in (
                        "run_contract",
                        "last_checkpoint",
                        "best_checkpoint",
                    )
                },
                expected_old=expected_old,
            )
            for label, expected_sha in (
                ("state", expected_state_sha256),
                ("approval", expected_approval_sha256),
                ("approval_required", expected_approval_required_sha256),
                ("guard_receipt", expected_guard_migration_receipt_sha256),
                ("ema_receipt", expected_ema_migration_receipt_sha256),
            ):
                if sha256_file(paths[label]) != expected_sha:
                    rollback_errors.append(f"protected {label} SHA256 changed")
            if receipt_path.parent.is_dir():
                atomic_write_json(
                    receipt_path,
                    {
                        "schema_version": RECEIPT_SCHEMA,
                        "protocol_id": PROTOCOL_ID,
                        "migration": MIGRATION_KIND,
                        "status": (
                            "ROLLBACK_FAILED" if rollback_errors else "ROLLED_BACK"
                        ),
                        "rolled_back_utc": utc_now_iso(),
                        "old": expected_old,
                        "prior_receipts": {
                            "guard": expected_guard_migration_receipt_sha256,
                            "ema": expected_ema_migration_receipt_sha256,
                        },
                        "backup": backups,
                        "rollback_errors": rollback_errors,
                    },
                )
            if rollback_errors:
                raise Stage3CalibrationPaddingMigrationError(
                    "publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from original_error
        raise
    finally:
        for candidate in candidates.values():
            candidate.unlink(missing_ok=True)


def recover_prepared_stage3_calibration_padding_provenance(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    last_checkpoint: str | Path,
    best_checkpoint: str | Path,
    state: str | Path,
    approval: str | Path,
    approval_required: str | Path,
    guard_migration_receipt: str | Path,
    ema_migration_receipt: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_last_checkpoint_sha256: str,
    expected_best_checkpoint_sha256: str,
    expected_state_sha256: str,
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
    expected_guard_migration_receipt_sha256: str,
    expected_ema_migration_receipt_sha256: str,
    expected_old_stage3_source_sha256: str,
    expected_new_stage3_source_sha256: str,
    expected_unchanged_stage4_source_sha256: str,
    expected_semantic_source_count: int = EXPECTED_SEMANTIC_SOURCE_COUNT,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Roll an interrupted PREPARED three-file transaction back exactly."""

    if confirmation_token != RECOVERY_CONFIRMATION_TOKEN:
        _fail("PREPARED recovery requires the exact calibration recovery token")
    paths = _resolve_requested_paths(
        project_root=project_root,
        run_contract=run_contract,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
        state=state,
        approval=approval,
        approval_required=approval_required,
        guard_receipt=guard_migration_receipt,
        ema_receipt=ema_migration_receipt,
        backup_dir=backup_dir,
    )
    _assert_cpu_only()
    _validate_hashes(
        expected_run_contract_sha256,
        expected_last_checkpoint_sha256,
        expected_best_checkpoint_sha256,
        expected_state_sha256,
        expected_approval_sha256,
        expected_approval_required_sha256,
        expected_guard_migration_receipt_sha256,
        expected_ema_migration_receipt_sha256,
        expected_old_stage3_source_sha256,
        expected_new_stage3_source_sha256,
        expected_unchanged_stage4_source_sha256,
    )
    _validate_paths(paths)
    receipt_path = paths["backup_dir"] / "MIGRATION_RECEIPT.json"
    if not paths["backup_dir"].is_dir() or not receipt_path.is_file():
        _fail("PREPARED recovery requires its dedicated receipt")

    prior_module_evidence = _validate_prior_module()
    _validate_failed_state(paths["state"], expected_state_sha256)
    approval_evidence = _validate_approval(
        paths["approval"],
        paths["approval_required"],
        expected_approval_sha256=expected_approval_sha256,
        expected_required_sha256=expected_approval_required_sha256,
    )
    prior_evidence = _validate_prior_migrations(
        root=paths["project_root"],
        guard_receipt=paths["guard_receipt"],
        ema_receipt=paths["ema_receipt"],
        expected_guard_sha256=expected_guard_migration_receipt_sha256,
        expected_ema_sha256=expected_ema_migration_receipt_sha256,
        expected_old_stage3_sha256=expected_old_stage3_source_sha256,
        expected_stage4_sha256=expected_unchanged_stage4_source_sha256,
    )
    current_semantic = semantic_source_hashes(
        paths["project_root"], entrypoints=ENTRYPOINTS
    )
    if (
        len(current_semantic) != expected_semantic_source_count
        or current_semantic.get(ALLOWED_SOURCE_PATH)
        != expected_new_stage3_source_sha256
        or current_semantic.get(PRESERVED_STAGE4_SOURCE_PATH)
        != expected_unchanged_stage4_source_sha256
    ):
        _fail("semantic sources drifted before PREPARED recovery")

    receipt = _require_mapping(load_json(receipt_path), field="migration receipt")
    old = _require_mapping(receipt.get("old"), field="migration receipt.old")
    new = _require_mapping(receipt.get("new"), field="migration receipt.new")
    backups = _require_mapping(receipt.get("backup"), field="migration receipt.backup")
    preserved_stage4 = _require_mapping(
        receipt.get("preserved_stage4_source"),
        field="migration receipt preserved Stage4",
    )
    state_evidence = _require_mapping(
        receipt.get("orchestration_state"),
        field="migration receipt orchestration state",
    )
    expected_old = {
        "run_contract": expected_run_contract_sha256,
        "last_checkpoint": expected_last_checkpoint_sha256,
        "best_checkpoint": expected_best_checkpoint_sha256,
    }
    expected_diff = [
        {
            "path": EXPECTED_PROVENANCE_DIFF_PATH,
            "old": expected_old_stage3_source_sha256,
            "new": expected_new_stage3_source_sha256,
        }
    ]
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("migration") != MIGRATION_KIND
        or receipt.get("status") not in {"PREPARED", "ROLLED_BACK_FROM_PREPARED"}
        or receipt.get("migration_script_sha256")
        != sha256_file(Path(__file__).resolve())
        or receipt.get("execution_confirmation_token_sha256")
        != hashlib.sha256(CONFIRMATION_TOKEN.encode()).hexdigest()
        or receipt.get("exact_provenance_leaf_diff") != expected_diff
        or receipt.get("semantic_source_count") != expected_semantic_source_count
        or receipt.get("unchanged_semantic_source_count")
        != expected_semantic_source_count - 1
        or receipt.get("checkpoint_top_level_count")
        != EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT
        or receipt.get("checkpoint_top_level_bit_exact_outside_provenance_count")
        != EXPECTED_UNCHANGED_TOP_LEVEL_COUNT
        or receipt.get("both_checkpoints_bit_exact_outside_provenance") is not True
        or receipt.get("run_contract_bit_exact_outside_provenance") is not True
        or receipt.get("approval_and_22_bindings_unchanged") != approval_evidence
        or receipt.get("prior_complete_migrations") != prior_evidence
        or receipt.get("prior_migration_implementation") != prior_module_evidence
        or state_evidence.get("path") != str(paths["state"])
        or state_evidence.get("sha256") != expected_state_sha256
        or preserved_stage4.get("path") != PRESERVED_STAGE4_SOURCE_PATH
        or preserved_stage4.get("sha256") != expected_unchanged_stage4_source_sha256
    ):
        _fail("PREPARED receipt does not match the audited three-file transaction")

    new_hashes: dict[str, str] = {}
    backup_paths: dict[str, Path] = {}
    recovered_from: dict[str, str] = {}
    for label, expected_sha in expected_old.items():
        old_entry = _require_mapping(old.get(label), field=f"receipt.old.{label}")
        new_sha = new.get(label)
        evidence = _require_mapping(backups.get(label), field=f"receipt.backup.{label}")
        raw_path = evidence.get("path")
        raw_mode = evidence.get("mode")
        inode = evidence.get("inode")
        device = evidence.get("device")
        if (
            old_entry.get("path") != str(paths[label])
            or old_entry.get("sha256") != expected_sha
            or not is_sha256(new_sha)
            or not isinstance(raw_path, str)
            or not isinstance(raw_mode, int)
            or not isinstance(inode, int)
            or not isinstance(device, int)
        ):
            _fail(f"invalid PREPARED old/new/backup evidence: {label}")
        requested = Path(raw_path)
        _reject_symlink_chain(requested, label=f"PREPARED {label} backup")
        backup = requested.resolve()
        live_sha = sha256_file(paths[label])
        if (
            backup.parent != paths["backup_dir"]
            or not backup.is_file()
            or backup.is_symlink()
            or sha256_file(backup) != expected_sha
            or backup.stat().st_ino != inode
            or backup.stat().st_dev != device
            or stat.S_IMODE(backup.stat().st_mode) not in {raw_mode, 0o444}
            or evidence.get("hard_link_verified") is not True
            or live_sha not in {expected_sha, str(new_sha)}
        ):
            _fail(f"PREPARED backup/live state drifted: {label}")
        new_hashes[label] = str(new_sha)
        backup_paths[label] = backup
        recovered_from[label] = live_sha

    backup_contract = _require_mapping(
        load_json(backup_paths["run_contract"]), field="backup run contract"
    )
    backup_last = _load_cpu_checkpoint(backup_paths["last_checkpoint"])
    backup_best = _load_cpu_checkpoint(backup_paths["best_checkpoint"])
    _validate_checkpoint_pair(backup_last, backup_best)
    backup_provenance = _require_mapping(
        backup_contract.get("provenance"), field="backup contract provenance"
    )
    if (
        backup_contract.get("schema_version") != STAGE3_SCHEMA
        or backup_last.get("provenance") != backup_provenance
        or backup_best.get("provenance") != backup_provenance
    ):
        _fail("PREPARED backup three-way provenance differs")
    _validate_provenance_identity(
        backup_provenance,
        _require_mapping(load_json(paths["approval"]), field="Stage3 approval"),
        expected_approval_sha256=expected_approval_sha256,
        expected_required_sha256=expected_approval_required_sha256,
    )
    verified_semantic, _ = _validate_semantic_sources(
        root=paths["project_root"],
        old_provenance=backup_provenance,
        expected_old_stage3_sha256=expected_old_stage3_source_sha256,
        expected_new_stage3_sha256=expected_new_stage3_source_sha256,
        expected_stage4_sha256=expected_unchanged_stage4_source_sha256,
        expected_count=expected_semantic_source_count,
    )
    if verified_semantic != current_semantic:
        _fail("semantic sources changed during PREPARED verification")

    if receipt.get("status") == "ROLLED_BACK_FROM_PREPARED":
        if (
            receipt.get("recovery_confirmation_token_sha256")
            != hashlib.sha256(RECOVERY_CONFIRMATION_TOKEN.encode()).hexdigest()
            or receipt.get("both_prior_receipts_unchanged_after_recovery") is not True
            or receipt.get("orchestration_state_unchanged_after_recovery") is not True
        ):
            _fail("finalized PREPARED recovery evidence is incomplete")
        for label, expected_sha in expected_old.items():
            if (
                sha256_file(paths[label]) != expected_sha
                or stat.S_IMODE(backup_paths[label].stat().st_mode) != 0o444
                or os.path.samestat(paths[label].stat(), backup_paths[label].stat())
            ):
                _fail(f"finalized PREPARED recovery drifted: {label}")
        return dict(receipt)

    rollback_errors = _restore_three(
        backups=backups,
        destinations={label: paths[label] for label in expected_old},
        expected_old=expected_old,
    )
    if rollback_errors:
        _fail("PREPARED recovery was incomplete: " + "; ".join(rollback_errors))
    protected = (
        ("state", expected_state_sha256),
        ("approval", expected_approval_sha256),
        ("approval_required", expected_approval_required_sha256),
        ("guard_receipt", expected_guard_migration_receipt_sha256),
        ("ema_receipt", expected_ema_migration_receipt_sha256),
    )
    if any(
        sha256_file(paths[label]) != expected_sha for label, expected_sha in protected
    ):
        _fail("protected state changed during PREPARED recovery")
    if (
        semantic_source_hashes(paths["project_root"], entrypoints=ENTRYPOINTS)
        != current_semantic
        or _validate_prior_module() != prior_module_evidence
    ):
        _fail("protected source or migration implementation changed during recovery")
    recovered = dict(receipt)
    recovered["status"] = "ROLLED_BACK_FROM_PREPARED"
    recovered["recovered_utc"] = utc_now_iso()
    recovered["recovered_from_live_sha256"] = recovered_from
    recovered["recovery_confirmation_token_sha256"] = hashlib.sha256(
        RECOVERY_CONFIRMATION_TOKEN.encode()
    ).hexdigest()
    recovered["backup_read_only_after_recovery"] = True
    recovered["both_prior_receipts_unchanged_after_recovery"] = True
    recovered["orchestration_state_unchanged_after_recovery"] = True
    _assert_cpu_only()
    atomic_write_json(receipt_path, recovered)
    return recovered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--last-checkpoint", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-required", type=Path, required=True)
    parser.add_argument("--guard-migration-receipt", type=Path, required=True)
    parser.add_argument("--ema-migration-receipt", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-run-contract-sha256", default=AUDITED_RUN_CONTRACT_SHA256
    )
    parser.add_argument(
        "--expected-last-checkpoint-sha256",
        default=AUDITED_LAST_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-best-checkpoint-sha256",
        default=AUDITED_BEST_CHECKPOINT_SHA256,
    )
    parser.add_argument("--expected-state-sha256", default=AUDITED_STATE_SHA256)
    parser.add_argument("--expected-approval-sha256", default=AUDITED_APPROVAL_SHA256)
    parser.add_argument(
        "--expected-approval-required-sha256",
        default=AUDITED_APPROVAL_REQUIRED_SHA256,
    )
    parser.add_argument(
        "--expected-guard-migration-receipt-sha256",
        default=AUDITED_GUARD_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--expected-ema-migration-receipt-sha256",
        default=AUDITED_EMA_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--expected-old-stage3-source-sha256",
        default=AUDITED_OLD_STAGE3_SOURCE_SHA256,
    )
    parser.add_argument(
        "--expected-new-stage3-source-sha256",
        default=AUDITED_NEW_STAGE3_SOURCE_SHA256,
    )
    parser.add_argument(
        "--expected-unchanged-stage4-source-sha256",
        default=AUDITED_STAGE4_SOURCE_SHA256,
    )
    parser.add_argument(
        "--expected-semantic-source-count",
        type=int,
        default=EXPECTED_SEMANTIC_SOURCE_COUNT,
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--recover-prepared", action="store_true")
    parser.add_argument("--confirmation-token")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    common = {
        "project_root": arguments.project_root,
        "run_contract": arguments.run_contract,
        "last_checkpoint": arguments.last_checkpoint,
        "best_checkpoint": arguments.best_checkpoint,
        "state": arguments.state,
        "approval": arguments.approval,
        "approval_required": arguments.approval_required,
        "guard_migration_receipt": arguments.guard_migration_receipt,
        "ema_migration_receipt": arguments.ema_migration_receipt,
        "backup_dir": arguments.backup_dir,
        "expected_run_contract_sha256": arguments.expected_run_contract_sha256,
        "expected_last_checkpoint_sha256": (arguments.expected_last_checkpoint_sha256),
        "expected_best_checkpoint_sha256": (arguments.expected_best_checkpoint_sha256),
        "expected_state_sha256": arguments.expected_state_sha256,
        "expected_approval_sha256": arguments.expected_approval_sha256,
        "expected_approval_required_sha256": (
            arguments.expected_approval_required_sha256
        ),
        "expected_guard_migration_receipt_sha256": (
            arguments.expected_guard_migration_receipt_sha256
        ),
        "expected_ema_migration_receipt_sha256": (
            arguments.expected_ema_migration_receipt_sha256
        ),
        "expected_old_stage3_source_sha256": (
            arguments.expected_old_stage3_source_sha256
        ),
        "expected_new_stage3_source_sha256": (
            arguments.expected_new_stage3_source_sha256
        ),
        "expected_unchanged_stage4_source_sha256": (
            arguments.expected_unchanged_stage4_source_sha256
        ),
        "expected_semantic_source_count": arguments.expected_semantic_source_count,
    }
    try:
        if arguments.recover_prepared:
            receipt = recover_prepared_stage3_calibration_padding_provenance(
                **common, confirmation_token=arguments.confirmation_token
            )
        else:
            receipt = migrate_stage3_calibration_padding_provenance(
                **common,
                execute=arguments.execute,
                confirmation_token=arguments.confirmation_token,
            )
    except (Stage3CalibrationPaddingMigrationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json_dumps(receipt), flush=True)
    return 0


def json_dumps(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
