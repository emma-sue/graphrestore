#!/usr/bin/env python3
"""Audited source-only Stage4 step-4000 provenance migration.

The completed calibration-routing migration is a frozen historical
transaction.  A later CPU preflight found that the Stage4 entrypoint read the
Stage3 finalization calibration binding from the wrong level of the validated
return value.  This second transaction preserves the existing
``calibration_history_routing`` mapping byte-for-byte and changes only
``semantic_source_sha256.scripts/train_stage4_e2e.py`` in the Stage4 run
contract, pending raw checkpoint and partial EMA checkpoint.

Without ``--execute`` this command is a non-publishing dry run.  It forces CUDA
invisibility before importing PyTorch through the audited transaction helper.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import (  # noqa: E402
    migrate_stage4_step4000_calibration_history_provenance as framework,
)
from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.provenance import semantic_source_hashes  # noqa: E402
from src.utils.hashing import is_sha256, sha256_file, sha256_json  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    fsync_directory,
    load_json,
    utc_now_iso,
)


PROTOCOL_ID = framework.PROTOCOL_ID
CHECKPOINT_SCHEMA = framework.CHECKPOINT_SCHEMA
STAGE4_SCHEMA = framework.STAGE4_SCHEMA
RECEIPT_SCHEMA = (
    "graphrestore-stage4-step4000-finalization-binding-source-fix-migration-v1"
)
MIGRATION_KIND = "stage4_pending_4000_finalization_binding_source_fix_provenance_only"
MIGRATION_STEP = framework.MIGRATION_STEP
EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT = framework.EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT
EXPECTED_UNCHANGED_TOP_LEVEL_COUNT = framework.EXPECTED_UNCHANGED_TOP_LEVEL_COUNT
EXPECTED_SEMANTIC_SOURCE_COUNT = framework.EXPECTED_SEMANTIC_SOURCE_COUNT
EXPECTED_TRAIN_LOG_LINE_COUNT = framework.EXPECTED_TRAIN_LOG_LINE_COUNT
ALLOWED_SOURCE_PATH = framework.ALLOWED_SOURCE_PATH
ROUTING_PROVENANCE_KEY = framework.ROUTING_PROVENANCE_KEY
ENTRYPOINTS = framework.ENTRYPOINTS
BACKUP_DIR_NAME = "stage4_step4000_finalization_binding_source_fix_v1"
PRIOR_BACKUP_DIR_NAME = framework.BACKUP_DIR_NAME
CONFIRMATION_TOKEN = "MIGRATE_STAGE4_PENDING_4000_FINALIZATION_BINDING_SOURCE_FIX"
RECOVERY_CONFIRMATION_TOKEN = (
    "RECOVER_STAGE4_PENDING_4000_FINALIZATION_BINDING_SOURCE_FIX"
)

AUDITED_RUN_CONTRACT_SHA256 = (
    "522c0f855db85af69617bbc8e2c17544be2b3485371d6adcf8e9ede9ef4624ea"
)
AUDITED_LAST_CHECKPOINT_SHA256 = (
    "02d7f3266f9db67e65c9e96d34e7e587ed708e6af36ebf619b81158f76795f30"
)
AUDITED_BEST_CHECKPOINT_SHA256 = (
    "a98cfb7ccc4e5472b15deeb5ece4306e0554dca92287099aef5ec699ba431384"
)
AUDITED_VALIDATION_LATEST_SHA256 = framework.AUDITED_VALIDATION_LATEST_SHA256
AUDITED_TRAIN_LOG_SHA256 = framework.AUDITED_TRAIN_LOG_SHA256
AUDITED_CALIBRATION_HISTORY_SHA256 = framework.AUDITED_CALIBRATION_HISTORY_SHA256
AUDITED_STATE_SHA256 = framework.AUDITED_STATE_SHA256
AUDITED_FAILURE_LOG_SHA256 = framework.AUDITED_FAILURE_LOG_SHA256
AUDITED_PRIOR_MIGRATION_RECEIPT_SHA256 = (
    "795982a5f607c147e25a2553a63c4b24306fd0fe2753cdcd6ca0cab0af8c190d"
)
AUDITED_OLD_PROVENANCE_SHA256 = (
    "a28650bc3cd1e5a47e0007d400a4e50e0450d131c5d40bf03d78c4d169a911fc"
)
AUDITED_TRANSACTION_FRAMEWORK_SCRIPT_SHA256 = (
    "16998f44b5c16fa108d70f1861dedafc1ff97b738f8a298868896734debf0bcb"
)
AUDITED_OLD_STAGE4_SOURCE_SHA256 = framework.AUDITED_NEW_STAGE4_SOURCE_SHA256
AUDITED_NEW_STAGE4_SOURCE_SHA256 = (
    "9224ee0abb62f919aed7f372e3f66395c74cb59aef0fb707ab91b7bcff43222b"
)
AUDITED_ROUTING_SCHEMA = framework.AUDITED_ROUTING_SCHEMA
AUDITED_ROUTING_SHA256 = framework.AUDITED_ROUTING_SHA256
AUDITED_CANONICAL_MODES = {
    "run_contract": 0o600,
    "last_checkpoint": 0o600,
    "best_checkpoint": 0o600,
    "validation_latest": 0o600,
    "train_log": 0o644,
    "calibration_history": 0o600,
    "state": 0o600,
    "failure_log": 0o644,
    "prior_migration_receipt": 0o600,
    "migration_script": 0o644,
    "transaction_framework_script": 0o644,
}
ARCHIVE_FILENAMES = {
    "run_contract": "old-run_contract.json",
    "last_checkpoint": "old-last.pth",
    "best_checkpoint": "old-best_ema.pth",
    "validation_latest": "validation_latest.json",
    "calibration_history": "calibration_history.csv",
    "state": "orchestration_state.json",
    "failure_log": "main_pipeline.log",
    "prior_migration_receipt": ("prior-calibration-routing-MIGRATION_RECEIPT.json"),
    "migration_script": "migration_script.py",
    "transaction_framework_script": "transaction_framework_script.py",
}
PREPARED_RECEIPT_EXACT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "migration",
        "status",
        "created_utc",
        "cpu_only",
        "cuda_visible_devices",
        "step",
        "flock",
        "old",
        "new",
        "exact_provenance_leaf_diff",
        "calibration_history_routing_preserved",
        "prior_migration_receipt",
        "semantic_source_count",
        "unchanged_semantic_source_count",
        "checkpoint_top_level_count",
        "checkpoint_top_level_bit_exact_outside_provenance_count",
        "both_checkpoints_bit_exact_outside_provenance",
        "run_contract_bit_exact_outside_provenance",
        "checkpoint_section_fingerprints",
        "all_checkpoint_tensors_finite",
        "stage4_calibration_sidecar_absent",
        "protected_evidence",
        "backup",
        "execution_confirmation_token_sha256",
        "migration_script_sha256",
        "transaction_framework_script_sha256",
    }
)
ROLLED_BACK_FROM_PREPARED_RECEIPT_EXACT_KEYS = PREPARED_RECEIPT_EXACT_KEYS | {
    "recovered_utc",
    "recovered_from_live_sha256",
    "recovery_confirmation_token_sha256",
    "backup_read_only_after_recovery",
    "protected_evidence_unchanged_after_recovery",
}

Stage4FinalizationBindingSourceMigrationError = (
    framework.Stage4CalibrationRoutingMigrationError
)
_fail = framework._fail
_mapping = framework._require_mapping
_assert_cpu_only = framework._assert_cpu_only
_reject_symlink_chain = framework._reject_symlink_chain
_load_cpu_checkpoint = framework._load_cpu_checkpoint
_make_candidate = framework._make_candidate
_replace_and_fsync = framework._replace_and_fsync
_restore_from_backup = framework._restore_from_backup
_archive_file = framework._archive_file
_archive_bytes = framework._archive_bytes
_migration_lock = framework._migration_lock
_validate_hashes = framework._validate_hashes
_validate_state = framework._validate_state
_validate_failure_log = framework._validate_failure_log
_validate_validation_latest = framework._validate_validation_latest
_validate_train_log = framework._validate_train_log
_validate_calibration_history = framework._validate_calibration_history
_validate_checkpoint = framework._validate_checkpoint
_validate_semantic_sources = framework._validate_semantic_sources
_validate_routing_mapping = framework._validate_routing_mapping
_protected_hashes = framework._protected_hashes
_verify_archive_evidence = framework._verify_archive_evidence
_publish_receipt = framework._publish_receipt
_assert_bit_exact = framework._assert_bit_exact
_section_evidence = framework._section_evidence


def _is_strict_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 20:
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value


def _validate_prepared_transaction_directory(paths: Mapping[str, Path]) -> Path:
    backup_dir = paths["backup_dir"]
    _reject_symlink_chain(backup_dir, label="PREPARED transaction directory")
    try:
        backup_stat = backup_dir.lstat()
    except FileNotFoundError:
        _fail("PREPARED recovery requires its dedicated transaction directory")
    migrations_dir = paths["project_root"] / "artifacts/migrations"
    migrations_stat = migrations_dir.lstat()
    if (
        not stat.S_ISDIR(backup_stat.st_mode)
        or backup_dir.is_symlink()
        or stat.S_IMODE(backup_stat.st_mode) != 0o700
        or backup_stat.st_dev != migrations_stat.st_dev
    ):
        _fail(
            "PREPARED transaction directory must be exact mode-0700, "
            "non-symlink, and same-filesystem"
        )

    receipt_path = backup_dir / "MIGRATION_RECEIPT.json"
    _reject_symlink_chain(receipt_path, label="PREPARED migration receipt")
    expected_names = {
        "MIGRATION_RECEIPT.json",
        "train_tail.jsonl",
        *ARCHIVE_FILENAMES.values(),
    }
    actual_names = {entry.name for entry in os.scandir(backup_dir)}
    if actual_names != expected_names:
        _fail("PREPARED transaction directory entry set drifted")
    archived_canonical_paths = {
        filename: paths[label] for label, filename in ARCHIVE_FILENAMES.items()
    } | {"train_tail.jsonl": paths["train_log"]}
    for name in expected_names:
        entry_path = backup_dir / name
        entry_stat = entry_path.lstat()
        if (
            not stat.S_ISREG(entry_stat.st_mode)
            or entry_path.is_symlink()
            or entry_stat.st_dev != backup_stat.st_dev
            or entry_stat.st_nlink != 1
        ):
            _fail(
                "PREPARED transaction directory entries must be regular, "
                "non-symlink, single-link, and same-filesystem"
            )
        canonical_path = archived_canonical_paths.get(name)
        if canonical_path is None:
            continue
        canonical_stat = canonical_path.lstat()
        if (
            not stat.S_ISREG(canonical_stat.st_mode)
            or canonical_path.is_symlink()
            or canonical_stat.st_nlink != 1
            or canonical_stat.st_dev != backup_stat.st_dev
            or (entry_stat.st_dev, entry_stat.st_ino)
            == (canonical_stat.st_dev, canonical_stat.st_ino)
        ):
            _fail(
                "PREPARED archive/canonical files must be distinct single-link "
                "regular files on the transaction filesystem"
            )
    if stat.S_IMODE(receipt_path.lstat().st_mode) != 0o600:
        _fail("PREPARED migration receipt mode must be exactly 0600")
    return receipt_path


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
        "prior_migration_receipt": root
        / "artifacts/migrations"
        / PRIOR_BACKUP_DIR_NAME
        / "MIGRATION_RECEIPT.json",
        "migration_script": root
        / "scripts/migrate_stage4_step4000_finalization_binding_source_fix.py",
        "transaction_framework_script": root
        / "scripts/migrate_stage4_step4000_calibration_history_provenance.py",
        "backup_dir": root / "artifacts/migrations" / BACKUP_DIR_NAME,
    }
    if root.is_symlink() or not root.is_dir():
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


def _validate_prior_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected_artifacts: Mapping[str, str],
    expected_routing_sha256: str,
    expected_framework_script_sha256: str,
    expected_old_provenance_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        _fail("prior calibration-routing migration receipt SHA256 drifted")
    receipt = _mapping(load_json(path), field="prior migration receipt")
    new = _mapping(receipt.get("new"), field="prior receipt new artifacts")
    routing = _mapping(
        receipt.get("calibration_history_routing"), field="prior receipt routing"
    )
    if (
        receipt.get("schema_version") != framework.RECEIPT_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("migration") != framework.MIGRATION_KIND
        or receipt.get("status") != "COMPLETE"
        or receipt.get("both_checkpoints_bit_exact_outside_provenance") is not True
        or receipt.get("run_contract_bit_exact_outside_provenance") is not True
        or receipt.get("protected_evidence_unchanged_after_publication") is not True
        or routing.get("sha256") != expected_routing_sha256
        or receipt.get("migration_script_sha256") != expected_framework_script_sha256
        or new.get("provenance_json_sha256") != expected_old_provenance_sha256
        or any(
            new.get(label) != expected_artifacts[label]
            for label in ("run_contract", "last_checkpoint", "best_checkpoint")
        )
    ):
        _fail("prior calibration-routing COMPLETE receipt drifted")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "schema_version": receipt["schema_version"],
        "migration": receipt["migration"],
        "status": receipt["status"],
        "new": {
            label: new[label]
            for label in ("run_contract", "last_checkpoint", "best_checkpoint")
        },
        "routing_sha256": routing["sha256"],
        "migration_script_sha256": receipt["migration_script_sha256"],
    }


def _require_stage4_sidecar_absent(project_root: Path) -> dict[str, Any]:
    path = project_root / "artifacts/metrics/stage4_calibration_history.csv"
    _reject_symlink_chain(path, label="Stage4 calibration sidecar")
    if os.path.lexists(path):
        _fail("Stage4-only calibration history must be absent at migration boundary")
    return {"path": str(path.resolve(strict=False)), "absent": True}


def _verify_bound_archive_evidence(
    evidence: Mapping[str, Any],
    *,
    label: str,
    paths: Mapping[str, Path],
    expected_sha256: str,
) -> Path:
    expected_keys = {
        "canonical_path",
        "canonical_sha256",
        "canonical_mode",
        "archive_path",
        "archive_sha256",
        "archive_mode",
        "archive_device",
        "archive_inode",
        "byte_exact",
        "same_filesystem",
    }
    expected_archive = paths["backup_dir"] / ARCHIVE_FILENAMES[label]
    if (
        set(evidence) != expected_keys
        or evidence.get("canonical_path") != str(paths[label])
        or evidence.get("canonical_sha256") != expected_sha256
        or evidence.get("canonical_mode") != AUDITED_CANONICAL_MODES[label]
        or evidence.get("archive_path") != str(expected_archive)
        or evidence.get("archive_sha256") != expected_sha256
        or evidence.get("archive_mode") != 0o444
        or evidence.get("byte_exact") is not True
        or evidence.get("same_filesystem") is not True
        or stat.S_IMODE(paths[label].stat().st_mode) != AUDITED_CANONICAL_MODES[label]
    ):
        _fail(f"bound archive evidence drifted: {label}")
    return _verify_archive_evidence(
        evidence,
        backup_dir=paths["backup_dir"],
        expected_sha256=expected_sha256,
    )


def _verify_bound_train_tail_evidence(
    evidence: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    expected_train_sha256: str,
    expected_tail_sha256: str,
    expected_tail_line_count: int,
) -> Path:
    expected_keys = {
        "canonical_path",
        "canonical_sha256",
        "canonical_mode",
        "archive_path",
        "archive_sha256",
        "archive_mode",
        "archive_device",
        "archive_inode",
        "byte_exact_tail",
        "same_filesystem",
        "tail_sha256",
        "tail_line_count",
    }
    expected_archive = paths["backup_dir"] / "train_tail.jsonl"
    if (
        set(evidence) != expected_keys
        or evidence.get("canonical_path") != str(paths["train_log"])
        or evidence.get("canonical_sha256") != expected_train_sha256
        or evidence.get("canonical_mode") != AUDITED_CANONICAL_MODES["train_log"]
        or evidence.get("archive_path") != str(expected_archive)
        or evidence.get("archive_sha256") != expected_tail_sha256
        or evidence.get("archive_mode") != 0o444
        or evidence.get("byte_exact_tail") is not True
        or evidence.get("same_filesystem") is not True
        or evidence.get("tail_sha256") != expected_tail_sha256
        or evidence.get("tail_line_count") != expected_tail_line_count
        or stat.S_IMODE(paths["train_log"].stat().st_mode)
        != AUDITED_CANONICAL_MODES["train_log"]
    ):
        _fail("bound archive evidence drifted: train_tail")
    return _verify_archive_evidence(
        evidence,
        backup_dir=paths["backup_dir"],
        expected_sha256=expected_tail_sha256,
    )


def _exact_source_only_provenance_diff(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if list(old) != list(new) or set(old) != set(new):
        _fail("provenance top-level structure changed")
    old_semantic = _mapping(
        old.get("semantic_source_sha256"), field="old semantic sources"
    )
    new_semantic = _mapping(
        new.get("semantic_source_sha256"), field="new semantic sources"
    )
    for key in old:
        if key != "semantic_source_sha256" and old[key] != new[key]:
            _fail(f"provenance changed outside the source leaf: {key}")
    changed = [
        key for key in old_semantic if old_semantic[key] != new_semantic.get(key)
    ]
    if set(old_semantic) != set(new_semantic) or changed != [ALLOWED_SOURCE_PATH]:
        _fail("provenance semantic-source diff is not exactly one entrypoint leaf")
    if old[ROUTING_PROVENANCE_KEY] != new[ROUTING_PROVENANCE_KEY]:
        _fail("calibration history routing changed during source-only migration")
    return [
        {
            "path": f"semantic_source_sha256.{ALLOWED_SOURCE_PATH}",
            "old": old_semantic[ALLOWED_SOURCE_PATH],
            "new": new_semantic[ALLOWED_SOURCE_PATH],
        }
    ]


def _common_validation(
    *,
    paths: Mapping[str, Path],
    expected: Mapping[str, str],
    expected_old_source_sha256: str,
    expected_new_source_sha256: str,
    expected_semantic_source_count: int,
    expected_routing_schema: str,
    expected_routing_sha256: str,
    expected_old_provenance_sha256: str,
    expected_train_log_line_count: int,
) -> dict[str, Any]:
    _assert_cpu_only()
    _validate_hashes(
        *expected.values(),
        expected_old_source_sha256,
        expected_new_source_sha256,
        expected_routing_sha256,
        expected_old_provenance_sha256,
    )
    if expected_semantic_source_count != EXPECTED_SEMANTIC_SOURCE_COUNT:
        _fail("Stage4 semantic-source count must remain exactly 47")
    if expected_train_log_line_count != EXPECTED_TRAIN_LOG_LINE_COUNT:
        _fail("Stage4 train-log line count must remain exactly 4001")
    for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
        if sha256_file(paths[label]) != expected[label]:
            _fail(f"{label} SHA256 differs from the audited post-migration boundary")
    if (
        sha256_file(paths["migration_script"]) != expected["migration_script"]
        or sha256_file(Path(__file__).resolve()) != expected["migration_script"]
        or sha256_file(paths["transaction_framework_script"])
        != expected["transaction_framework_script"]
        or sha256_file(Path(framework.__file__).resolve())
        != expected["transaction_framework_script"]
    ):
        _fail("migration implementation source binding drifted")
    if set(AUDITED_CANONICAL_MODES) != set(expected) or any(
        stat.S_IMODE(paths[label].stat().st_mode) != mode
        for label, mode in AUDITED_CANONICAL_MODES.items()
    ):
        _fail("audited canonical artifact mode drifted")
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
    prior = _validate_prior_receipt(
        paths["prior_migration_receipt"],
        expected_sha256=expected["prior_migration_receipt"],
        expected_artifacts=expected,
        expected_routing_sha256=expected_routing_sha256,
        expected_framework_script_sha256=expected["transaction_framework_script"],
        expected_old_provenance_sha256=expected_old_provenance_sha256,
    )
    sidecar = _require_stage4_sidecar_absent(paths["project_root"])
    contract = _mapping(load_json(paths["run_contract"]), field="Stage4 run contract")
    if contract.get("schema_version") != STAGE4_SCHEMA:
        _fail("Stage4 run-contract schema drifted")
    old_provenance = _mapping(contract.get("provenance"), field="run provenance")
    if sha256_json(dict(old_provenance)) != expected_old_provenance_sha256:
        _fail("post-first-migration provenance SHA256 drifted")
    if ROUTING_PROVENANCE_KEY not in old_provenance:
        _fail("post-migration Stage4 provenance lacks calibration history routing")
    routing = _validate_routing_mapping(
        _mapping(
            old_provenance.get(ROUTING_PROVENANCE_KEY),
            field="existing calibration history routing",
        ),
        project_root=paths["project_root"],
        expected_schema=expected_routing_schema,
        expected_sha256=expected_routing_sha256,
        expected_frozen_history_sha256=expected["calibration_history"],
    )
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
    provenance_diff = _exact_source_only_provenance_diff(old_provenance, new_provenance)
    if sha256_json(routing) != sha256_json(new_provenance[ROUTING_PROVENANCE_KEY]):
        _fail("calibration history routing is not bit-exact")
    return {
        "contract": contract,
        "last": last,
        "best": best,
        "old_provenance": old_provenance,
        "new_provenance": new_provenance,
        "provenance_diff": provenance_diff,
        "routing": routing,
        "prior": prior,
        "sidecar": sidecar,
        "state": state,
        "failure": failure,
        "validation": validation,
        "train": train,
        "train_tail": tail,
        "history": history,
    }


def migrate_stage4_step4000_finalization_binding_source_fix(
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
    prior_migration_receipt: str | Path,
    migration_script: str | Path,
    transaction_framework_script: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_last_checkpoint_sha256: str,
    expected_best_checkpoint_sha256: str,
    expected_validation_latest_sha256: str,
    expected_train_log_sha256: str,
    expected_calibration_history_sha256: str,
    expected_state_sha256: str,
    expected_failure_log_sha256: str,
    expected_prior_migration_receipt_sha256: str,
    expected_old_provenance_sha256: str,
    expected_transaction_framework_script_sha256: str,
    expected_old_stage4_source_sha256: str,
    expected_new_stage4_source_sha256: str,
    expected_routing_schema: str,
    expected_routing_sha256: str,
    expected_semantic_source_count: int = EXPECTED_SEMANTIC_SOURCE_COUNT,
    expected_train_log_line_count: int = EXPECTED_TRAIN_LOG_LINE_COUNT,
    execute: bool = False,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Build or publish the exact source-only three-artifact migration."""

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
        prior_migration_receipt=prior_migration_receipt,
        migration_script=migration_script,
        transaction_framework_script=transaction_framework_script,
        backup_dir=backup_dir,
    )
    _validate_paths(paths)
    if execute and confirmation_token != CONFIRMATION_TOKEN:
        _fail("execution requires the exact Stage4 source-fix token")
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
        "prior_migration_receipt": expected_prior_migration_receipt_sha256,
        "migration_script": sha256_file(Path(__file__).resolve()),
        "transaction_framework_script": expected_transaction_framework_script_sha256,
    }
    with _migration_lock(paths["project_root"]) as lock_evidence:
        values = _common_validation(
            paths=paths,
            expected=expected,
            expected_old_source_sha256=expected_old_stage4_source_sha256,
            expected_new_source_sha256=expected_new_stage4_source_sha256,
            expected_semantic_source_count=expected_semantic_source_count,
            expected_routing_schema=expected_routing_schema,
            expected_routing_sha256=expected_routing_sha256,
            expected_old_provenance_sha256=expected_old_provenance_sha256,
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
            evidence = _section_evidence(old_payload, new_payload)
            unchanged = sum(
                item["bit_exact"]
                for key, item in evidence.items()
                if key != "provenance"
            )
            if unchanged != EXPECTED_UNCHANGED_TOP_LEVEL_COUNT:
                _fail(f"{label} unchanged checkpoint section count drifted")

        candidates = {
            "run_contract": _make_candidate(
                paths["run_contract"].parent,
                paths["run_contract"].name,
                ".stage4-finalization-binding-source-fix.candidate.json",
            ),
            "last_checkpoint": _make_candidate(
                paths["last_checkpoint"].parent,
                paths["last_checkpoint"].name,
                ".stage4-finalization-binding-source-fix.candidate.pth",
            ),
            "best_checkpoint": _make_candidate(
                paths["best_checkpoint"].parent,
                paths["best_checkpoint"].name,
                ".stage4-finalization-binding-source-fix.candidate.pth",
            ),
        }
        receipt_path = paths["backup_dir"] / "MIGRATION_RECEIPT.json"
        backups: dict[str, Any] = {}
        prepared_written = False
        try:
            atomic_write_json(candidates["run_contract"], new_contract)
            atomic_torch_save(new_last, candidates["last_checkpoint"])
            atomic_torch_save(new_best, candidates["best_checkpoint"])
            reloaded_contract = _mapping(
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
                "status": "PREPARED" if execute else "DRY_RUN",
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
                "calibration_history_routing_preserved": {
                    "schema_version": expected_routing_schema,
                    "sha256": expected_routing_sha256,
                    "mapping": values["routing"],
                    "bit_exact": True,
                },
                "prior_migration_receipt": values["prior"],
                "semantic_source_count": expected_semantic_source_count,
                "unchanged_semantic_source_count": expected_semantic_source_count - 1,
                "checkpoint_top_level_count": EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT,
                "checkpoint_top_level_bit_exact_outside_provenance_count": EXPECTED_UNCHANGED_TOP_LEVEL_COUNT,
                "both_checkpoints_bit_exact_outside_provenance": True,
                "run_contract_bit_exact_outside_provenance": True,
                "checkpoint_section_fingerprints": section_evidence,
                "all_checkpoint_tensors_finite": True,
                "stage4_calibration_sidecar_absent": values["sidecar"],
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
                    "prior_migration_receipt": values["prior"],
                    "stage4_calibration_sidecar": values["sidecar"],
                    "migration_script": {
                        "path": str(paths["migration_script"]),
                        "sha256": expected["migration_script"],
                    },
                    "transaction_framework_script": {
                        "path": str(paths["transaction_framework_script"]),
                        "sha256": expected["transaction_framework_script"],
                    },
                },
                "backup": backups,
                "execution_confirmation_token_sha256": (
                    hashlib.sha256(CONFIRMATION_TOKEN.encode()).hexdigest()
                    if execute
                    else None
                ),
                "migration_script_sha256": sha256_file(Path(__file__).resolve()),
                "transaction_framework_script_sha256": sha256_file(
                    Path(framework.__file__).resolve()
                ),
            }
            if not execute:
                _protected_hashes(paths, expected)
                _require_stage4_sidecar_absent(paths["project_root"])
                _assert_cpu_only()
                return receipt

            paths["backup_dir"].mkdir(mode=0o700)
            fsync_directory(paths["backup_dir"].parent)
            for label, filename in ARCHIVE_FILENAMES.items():
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
            published_contract = _mapping(
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
            for label in ARCHIVE_FILENAMES:
                _verify_bound_archive_evidence(
                    _mapping(backups[label], field=f"backup.{label}"),
                    label=label,
                    paths=paths,
                    expected_sha256=expected[label],
                )
            _verify_bound_train_tail_evidence(
                _mapping(backups["train_tail"], field="backup.train_tail"),
                paths=paths,
                expected_train_sha256=expected["train_log"],
                expected_tail_sha256=values["train"]["tail_sha256"],
                expected_tail_line_count=values["train"]["tail_line_count"],
            )
            _require_stage4_sidecar_absent(paths["project_root"])
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
                    _require_stage4_sidecar_absent(paths["project_root"])
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
                    "transaction_framework_script_sha256": sha256_file(
                        Path(framework.__file__).resolve()
                    ),
                }
                _publish_receipt(receipt_path, rollback_receipt)
                if rollback_errors:
                    raise Stage4FinalizationBindingSourceMigrationError(
                        "publication failed and rollback was incomplete: "
                        + "; ".join(rollback_errors)
                    ) from original_error
            raise
        finally:
            for candidate in candidates.values():
                candidate.unlink(missing_ok=True)


def recover_prepared_stage4_step4000_finalization_binding_source_fix(
    *,
    confirmation_token: str | None = None,
    **arguments: Any,
) -> dict[str, Any]:
    """Roll an interrupted PREPARED source-only transaction back exactly."""

    if confirmation_token != RECOVERY_CONFIRMATION_TOKEN:
        _fail("PREPARED recovery requires the exact distinct recovery token")
    paths = _resolve_requested_paths(
        **{
            key: arguments[key]
            for key in (
                "project_root",
                "run_contract",
                "last_checkpoint",
                "best_checkpoint",
                "validation_latest",
                "train_log",
                "calibration_history",
                "state",
                "failure_log",
                "prior_migration_receipt",
                "migration_script",
                "transaction_framework_script",
                "backup_dir",
            )
        }
    )
    _validate_paths(paths)
    expected = {
        "run_contract": arguments["expected_run_contract_sha256"],
        "last_checkpoint": arguments["expected_last_checkpoint_sha256"],
        "best_checkpoint": arguments["expected_best_checkpoint_sha256"],
        "validation_latest": arguments["expected_validation_latest_sha256"],
        "train_log": arguments["expected_train_log_sha256"],
        "calibration_history": arguments["expected_calibration_history_sha256"],
        "state": arguments["expected_state_sha256"],
        "failure_log": arguments["expected_failure_log_sha256"],
        "prior_migration_receipt": arguments["expected_prior_migration_receipt_sha256"],
        "migration_script": sha256_file(Path(__file__).resolve()),
        "transaction_framework_script": arguments[
            "expected_transaction_framework_script_sha256"
        ],
    }
    _validate_hashes(
        *expected.values(),
        arguments["expected_old_stage4_source_sha256"],
        arguments["expected_new_stage4_source_sha256"],
        arguments["expected_routing_sha256"],
        arguments["expected_old_provenance_sha256"],
    )
    with _migration_lock(paths["project_root"]) as recovery_lock:
        receipt_path = _validate_prepared_transaction_directory(paths)
        receipt = _mapping(load_json(receipt_path), field="migration receipt")
        already_recovered = receipt.get("status") == "ROLLED_BACK_FROM_PREPARED"
        expected_receipt_keys = (
            ROLLED_BACK_FROM_PREPARED_RECEIPT_EXACT_KEYS
            if already_recovered
            else PREPARED_RECEIPT_EXACT_KEYS
        )
        if (
            set(receipt) != expected_receipt_keys
            or not _is_strict_utc_timestamp(receipt.get("created_utc"))
            or receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("migration") != MIGRATION_KIND
            or receipt.get("status") not in {"PREPARED", "ROLLED_BACK_FROM_PREPARED"}
            or type(receipt.get("step")) is not int
            or receipt.get("step") != MIGRATION_STEP
            or receipt.get("cpu_only") is not True
            or receipt.get("cuda_visible_devices") != ""
            or receipt.get("flock") != recovery_lock
            or receipt.get("migration_script_sha256")
            != sha256_file(Path(__file__).resolve())
            or receipt.get("transaction_framework_script_sha256")
            != sha256_file(Path(framework.__file__).resolve())
            or receipt.get("execution_confirmation_token_sha256")
            != hashlib.sha256(CONFIRMATION_TOKEN.encode()).hexdigest()
        ):
            _fail("receipt is not the exact PREPARED Stage4 source-fix transaction")
        current_semantic = semantic_source_hashes(
            paths["project_root"], entrypoints=ENTRYPOINTS
        )
        if (
            len(current_semantic) != arguments["expected_semantic_source_count"]
            or current_semantic.get(ALLOWED_SOURCE_PATH)
            != arguments["expected_new_stage4_source_sha256"]
        ):
            _fail("semantic sources drifted before PREPARED recovery")
        protected_expected = {
            key: value
            for key, value in expected.items()
            if key not in {"run_contract", "last_checkpoint", "best_checkpoint"}
        }
        _protected_hashes(paths, protected_expected)
        state_evidence = _validate_state(paths["state"], expected["state"])
        failure_evidence = _validate_failure_log(
            paths["failure_log"], expected["failure_log"]
        )
        validation = _validate_validation_latest(
            paths["validation_latest"], expected["validation_latest"]
        )
        train_evidence, tail = _validate_train_log(
            paths["train_log"],
            expected["train_log"],
            expected_line_count=arguments["expected_train_log_line_count"],
        )
        history_evidence = _validate_calibration_history(
            paths["calibration_history"], expected["calibration_history"]
        )
        prior_evidence = _validate_prior_receipt(
            paths["prior_migration_receipt"],
            expected_sha256=expected["prior_migration_receipt"],
            expected_artifacts=expected,
            expected_routing_sha256=arguments["expected_routing_sha256"],
            expected_framework_script_sha256=expected["transaction_framework_script"],
            expected_old_provenance_sha256=arguments["expected_old_provenance_sha256"],
        )
        sidecar_evidence = _require_stage4_sidecar_absent(paths["project_root"])
        expected_protected_evidence = {
            "validation_latest": {
                "path": str(paths["validation_latest"]),
                "sha256": expected["validation_latest"],
                "image_count": validation["image_count"],
            },
            "train_log": train_evidence,
            "calibration_history": history_evidence,
            "orchestration_state": state_evidence,
            "failure_log": failure_evidence,
            "prior_migration_receipt": prior_evidence,
            "stage4_calibration_sidecar": sidecar_evidence,
            "migration_script": {
                "path": str(paths["migration_script"]),
                "sha256": expected["migration_script"],
            },
            "transaction_framework_script": {
                "path": str(paths["transaction_framework_script"]),
                "sha256": expected["transaction_framework_script"],
            },
        }

        old = _mapping(receipt.get("old"), field="receipt.old")
        new = _mapping(receipt.get("new"), field="receipt.new")
        backups = _mapping(receipt.get("backup"), field="receipt.backup")
        backup_paths: dict[str, Path] = {}
        canonical_modes: dict[str, int] = {}
        live_before: dict[str, str] = {}
        if set(backups) != {*ARCHIVE_FILENAMES, "train_tail"}:
            _fail("PREPARED backup evidence key set drifted")
        for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
            old_entry = _mapping(old.get(label), field=f"receipt.old.{label}")
            new_sha = new.get(label)
            evidence = _mapping(backups.get(label), field=f"receipt.backup.{label}")
            if (
                old_entry.get("path") != str(paths[label])
                or old_entry.get("sha256") != expected[label]
                or not is_sha256(new_sha)
            ):
                _fail(f"PREPARED old/new evidence drifted: {label}")
            backup_paths[label] = _verify_bound_archive_evidence(
                evidence,
                label=label,
                paths=paths,
                expected_sha256=expected[label],
            )
            canonical_modes[label] = AUDITED_CANONICAL_MODES[label]
            live_sha = sha256_file(paths[label])
            if live_sha not in {expected[label], str(new_sha)}:
                _fail(f"PREPARED live artifact is neither old nor new: {label}")
            live_before[label] = live_sha
        for label in (
            "validation_latest",
            "calibration_history",
            "state",
            "failure_log",
            "prior_migration_receipt",
            "migration_script",
            "transaction_framework_script",
        ):
            _verify_bound_archive_evidence(
                _mapping(backups.get(label), field=f"receipt.backup.{label}"),
                label=label,
                paths=paths,
                expected_sha256=expected[label],
            )
        tail_evidence = _mapping(backups.get("train_tail"), field="receipt train tail")
        if (
            tail_evidence.get("canonical_sha256") != expected["train_log"]
            or tail_evidence.get("tail_sha256") != train_evidence["tail_sha256"]
        ):
            _fail("PREPARED train-tail evidence drifted")
        _verify_bound_train_tail_evidence(
            tail_evidence,
            paths=paths,
            expected_train_sha256=expected["train_log"],
            expected_tail_sha256=hashlib.sha256(tail).hexdigest(),
            expected_tail_line_count=train_evidence["tail_line_count"],
        )

        backup_contract = _mapping(
            load_json(backup_paths["run_contract"]), field="backup run contract"
        )
        backup_last = _load_cpu_checkpoint(backup_paths["last_checkpoint"])
        backup_best = _load_cpu_checkpoint(backup_paths["best_checkpoint"])
        old_provenance = _mapping(
            backup_contract.get("provenance"), field="backup old provenance"
        )
        if (
            sha256_json(dict(old_provenance))
            != arguments["expected_old_provenance_sha256"]
        ):
            _fail("PREPARED backup provenance SHA256 drifted")
        _validate_checkpoint(backup_last, role="last", validation=validation)
        _validate_checkpoint(backup_best, role="best", validation=validation)
        if (
            backup_last.get("provenance") != old_provenance
            or backup_best.get("provenance") != old_provenance
        ):
            _fail("PREPARED backup three-way provenance identity drifted")
        routing = _validate_routing_mapping(
            _mapping(
                old_provenance.get(ROUTING_PROVENANCE_KEY),
                field="backup calibration history routing",
            ),
            project_root=paths["project_root"],
            expected_schema=arguments["expected_routing_schema"],
            expected_sha256=arguments["expected_routing_sha256"],
            expected_frozen_history_sha256=expected["calibration_history"],
        )
        current_semantic, old_semantic = _validate_semantic_sources(
            root=paths["project_root"],
            old_provenance=old_provenance,
            expected_old_source_sha256=arguments["expected_old_stage4_source_sha256"],
            expected_new_source_sha256=arguments["expected_new_stage4_source_sha256"],
            expected_count=arguments["expected_semantic_source_count"],
        )
        new_provenance = copy.deepcopy(dict(old_provenance))
        new_semantic = dict(old_semantic)
        new_semantic[ALLOWED_SOURCE_PATH] = current_semantic[ALLOWED_SOURCE_PATH]
        new_provenance["semantic_source_sha256"] = new_semantic
        expected_new_contract = copy.deepcopy(dict(backup_contract))
        expected_new_contract["provenance"] = new_provenance
        expected_new_last = copy.copy(backup_last)
        expected_new_last["provenance"] = new_provenance
        expected_new_best = copy.copy(backup_best)
        expected_new_best["provenance"] = new_provenance
        expected_section_evidence = {
            "last_checkpoint": _section_evidence(backup_last, expected_new_last),
            "best_checkpoint": _section_evidence(backup_best, expected_new_best),
        }
        recovery_candidates = {
            "run_contract": _make_candidate(
                paths["run_contract"].parent,
                paths["run_contract"].name,
                ".stage4-source-fix.recovery-check.json",
            ),
            "last_checkpoint": _make_candidate(
                paths["last_checkpoint"].parent,
                paths["last_checkpoint"].name,
                ".stage4-source-fix.recovery-check.pth",
            ),
            "best_checkpoint": _make_candidate(
                paths["best_checkpoint"].parent,
                paths["best_checkpoint"].name,
                ".stage4-source-fix.recovery-check.pth",
            ),
        }
        try:
            atomic_write_json(
                recovery_candidates["run_contract"], expected_new_contract
            )
            atomic_torch_save(expected_new_last, recovery_candidates["last_checkpoint"])
            atomic_torch_save(expected_new_best, recovery_candidates["best_checkpoint"])
            expected_new_hashes = {
                label: sha256_file(path) for label, path in recovery_candidates.items()
            }
        finally:
            for candidate in recovery_candidates.values():
                candidate.unlink(missing_ok=True)
        routing_evidence = _mapping(
            receipt.get("calibration_history_routing_preserved"),
            field="receipt routing preservation",
        )
        protected_evidence = _mapping(
            receipt.get("protected_evidence"), field="receipt protected evidence"
        )
        expected_routing_evidence = {
            "schema_version": arguments["expected_routing_schema"],
            "sha256": arguments["expected_routing_sha256"],
            "mapping": routing,
            "bit_exact": True,
        }
        if (
            set(old)
            != {
                "run_contract",
                "last_checkpoint",
                "best_checkpoint",
                "provenance_json_sha256",
            }
            or set(new)
            != {
                "run_contract",
                "last_checkpoint",
                "best_checkpoint",
                "provenance_json_sha256",
            }
            or any(
                new.get(label) != expected_new_hashes[label]
                for label in expected_new_hashes
            )
            or receipt.get("exact_provenance_leaf_diff")
            != _exact_source_only_provenance_diff(old_provenance, new_provenance)
            or old.get("provenance_json_sha256") != sha256_json(dict(old_provenance))
            or new.get("provenance_json_sha256") != sha256_json(new_provenance)
            or dict(routing_evidence) != expected_routing_evidence
            or receipt.get("prior_migration_receipt") != prior_evidence
            or receipt.get("stage4_calibration_sidecar_absent") != sidecar_evidence
            or dict(protected_evidence) != expected_protected_evidence
            or receipt.get("checkpoint_section_fingerprints")
            != expected_section_evidence
            or receipt.get("semantic_source_count")
            != arguments["expected_semantic_source_count"]
            or receipt.get("unchanged_semantic_source_count")
            != arguments["expected_semantic_source_count"] - 1
            or receipt.get("checkpoint_top_level_count")
            != EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT
            or receipt.get("checkpoint_top_level_bit_exact_outside_provenance_count")
            != EXPECTED_UNCHANGED_TOP_LEVEL_COUNT
            or receipt.get("all_checkpoint_tensors_finite") is not True
            or receipt.get("both_checkpoints_bit_exact_outside_provenance") is not True
            or receipt.get("run_contract_bit_exact_outside_provenance") is not True
        ):
            _fail("PREPARED exact source-only/bit-exact evidence drifted")
        for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
            if live_before[label] == expected[label]:
                continue
            if label == "run_contract":
                live_payload = _mapping(
                    load_json(paths[label]), field="live new run contract"
                )
                _assert_bit_exact(
                    {
                        key: value
                        for key, value in backup_contract.items()
                        if key != "provenance"
                    },
                    {
                        key: value
                        for key, value in live_payload.items()
                        if key != "provenance"
                    },
                    path="PREPARED.live_run_contract.outside_provenance",
                )
            else:
                live_payload = _load_cpu_checkpoint(paths[label])
                _section_evidence(
                    backup_last if label == "last_checkpoint" else backup_best,
                    live_payload,
                )
            if live_payload.get("provenance") != new_provenance:
                _fail(f"PREPARED live new provenance drifted: {label}")

        if already_recovered:
            recovered_from_live = _mapping(
                receipt.get("recovered_from_live_sha256"),
                field="receipt recovered-from-live evidence",
            )
            if (
                any(
                    live_before[label] != expected[label]
                    for label in ("run_contract", "last_checkpoint", "best_checkpoint")
                )
                or not _is_strict_utc_timestamp(receipt.get("recovered_utc"))
                or set(recovered_from_live)
                != {"run_contract", "last_checkpoint", "best_checkpoint"}
                or any(
                    recovered_from_live.get(label)
                    not in {expected[label], new.get(label)}
                    for label in ("run_contract", "last_checkpoint", "best_checkpoint")
                )
                or receipt.get("recovery_confirmation_token_sha256")
                != hashlib.sha256(RECOVERY_CONFIRMATION_TOKEN.encode()).hexdigest()
                or receipt.get("backup_read_only_after_recovery") is not True
                or receipt.get("protected_evidence_unchanged_after_recovery")
                is not True
            ):
                _fail("finalized PREPARED recovery evidence drifted")
            _require_stage4_sidecar_absent(paths["project_root"])
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
        _require_stage4_sidecar_absent(paths["project_root"])
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    for name in (
        "run-contract",
        "last-checkpoint",
        "best-checkpoint",
        "validation-latest",
        "train-log",
        "calibration-history",
        "state",
        "failure-log",
        "prior-migration-receipt",
        "migration-script",
        "transaction-framework-script",
        "backup-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
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
        "--expected-prior-migration-receipt-sha256",
        default=AUDITED_PRIOR_MIGRATION_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--expected-old-provenance-sha256",
        default=AUDITED_OLD_PROVENANCE_SHA256,
    )
    parser.add_argument(
        "--expected-transaction-framework-script-sha256",
        default=AUDITED_TRANSACTION_FRAMEWORK_SCRIPT_SHA256,
    )
    parser.add_argument(
        "--expected-old-stage4-source-sha256",
        default=AUDITED_OLD_STAGE4_SOURCE_SHA256,
    )
    parser.add_argument(
        "--expected-new-stage4-source-sha256",
        default=AUDITED_NEW_STAGE4_SOURCE_SHA256,
    )
    parser.add_argument("--expected-routing-schema", default=AUDITED_ROUTING_SCHEMA)
    parser.add_argument("--expected-routing-sha256", default=AUDITED_ROUTING_SHA256)
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
    common = {
        key: value
        for key, value in vars(arguments).items()
        if key not in {"execute", "recover_prepared", "confirmation_token"}
    }
    try:
        if arguments.recover_prepared:
            receipt = recover_prepared_stage4_step4000_finalization_binding_source_fix(
                **common, confirmation_token=arguments.confirmation_token
            )
        else:
            receipt = migrate_stage4_step4000_finalization_binding_source_fix(
                **common,
                execute=arguments.execute,
                confirmation_token=arguments.confirmation_token,
            )
    except (Stage4FinalizationBindingSourceMigrationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
