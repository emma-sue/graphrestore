#!/usr/bin/env python3
"""Migrate one Stage3 semantic-source leaf after the EMA device fix.

The pending Stage3 step-2000 checkpoint was produced before a resume-only
validation compared a CPU-mapped frozen checkpoint tensor with the live CUDA
Stage1 parent.  The comparison failure did not perform an optimizer update or
publish validation metrics.  This CPU-only tool permits exactly one provenance
change, ``src/training/stage3_engine.py``, while preserving the Stage4 leaf and
all other semantic sources.

The first guard-alignment migration is an immutable prerequisite.  Its
COMPLETE receipt, read-only backups, and receipt SHA are checked before and
after every publishing operation.  Execution requires an exact confirmation
token; otherwise this command only builds and verifies non-publishing
candidates.  A separate recovery token rolls an interrupted PREPARED
transaction back to both audited old files.
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

# This must precede both torch and the shared migration helper import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The shared primitives are executable code, so bind them before importing the
# module rather than trusting a post-import receipt field alone.
AUDITED_SHARED_MIGRATION_SCRIPT_SHA256 = (
    "96b28d294e9632882afafb1a93d23fd10c21f8458fa675a8d37276d41d0a90db"
)
SHARED_MIGRATION_SCRIPT_PATH = (
    PROJECT_ROOT / "scripts/migrate_stage3_guard_alignment_provenance.py"
)


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if (
    SHARED_MIGRATION_SCRIPT_PATH.is_symlink()
    or not SHARED_MIGRATION_SCRIPT_PATH.is_file()
    or _raw_sha256(SHARED_MIGRATION_SCRIPT_PATH)
    != AUDITED_SHARED_MIGRATION_SCRIPT_SHA256
):
    raise RuntimeError(
        "audited shared guard-migration primitives drifted before import"
    )

from scripts import migrate_stage3_guard_alignment_provenance as shared  # noqa: E402
from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.provenance import semantic_source_hashes  # noqa: E402
from src.utils.hashing import is_sha256, sha256_file, sha256_json  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    fsync_directory,
    load_json,
    utc_now_iso,
)


PROTOCOL_ID = shared.PROTOCOL_ID
CHECKPOINT_SCHEMA = shared.CHECKPOINT_SCHEMA
STAGE3_SCHEMA = shared.STAGE3_SCHEMA
APPROVAL_SCHEMA = shared.APPROVAL_SCHEMA
MIGRATION_STEP = shared.MIGRATION_STEP
EXPECTED_BINDING_COUNT = shared.EXPECTED_BINDING_COUNT
ENTRYPOINTS = shared.ENTRYPOINTS

RECEIPT_SCHEMA = "graphrestore-stage3-ema-device-migration-v1"
MIGRATION_KIND = "stage3_pending_2000_ema_device_provenance_only"
CONFIRMATION_TOKEN = "MIGRATE_STAGE3_PENDING_2000_EMA_DEVICE_PROVENANCE"
RECOVERY_CONFIRMATION_TOKEN = "RECOVER_STAGE3_PENDING_2000_EMA_DEVICE_PROVENANCE"
BACKUP_DIR_NAME = "stage3_ema_device_pending2000_v1"
PRIOR_BACKUP_DIR_NAME = "stage3_guard_alignment_pending2000_v1"
PRIOR_RECEIPT_SCHEMA = "graphrestore-stage3-guard-alignment-migration-v1"
PRIOR_MIGRATION_KIND = "stage3_pending_2000_guard_alignment_provenance_only"

ALLOWED_SOURCE_PATH = "src/training/stage3_engine.py"
PRESERVED_STAGE4_SOURCE_PATH = "src/training/stage4_engine.py"
EXPECTED_PROVENANCE_DIFF_PATH = f"semantic_source_sha256.{ALLOWED_SOURCE_PATH}"
EXPECTED_SEMANTIC_SOURCE_COUNT = 47
SHARED_PRIMITIVE_NAMES = (
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

# Audited production anchors.  Public functions still receive explicit values
# so synthetic fault tests never need to touch canonical artifacts.
AUDITED_RUN_CONTRACT_SHA256 = (
    "156a57b5f74659c45d2123e98c3e89c02b4611136e960d1134d0d88b092084b5"
)
AUDITED_CHECKPOINT_SHA256 = (
    "39bc85036a372df040774bf93d3000d0a5e36853e0e07b4648d7a01953a30d16"
)
AUDITED_OLD_STAGE3_SOURCE_SHA256 = (
    "65a0812ea60dba4721e1dc4f744282ef23990ac78c32106ad8774d7dafa71a14"
)
AUDITED_STAGE4_SOURCE_SHA256 = (
    "e2fbfbc2ee580b90cb92c48e6b289d6bc6d3d4651c42d34295ce07fc664814b6"
)
AUDITED_APPROVAL_SHA256 = (
    "7b351c0958aa681dc1f65114e801c58e3a5bc4bb7cc73c06507c0b647e51a08b"
)
AUDITED_APPROVAL_REQUIRED_SHA256 = (
    "33be4aba2c4229175ac33edef7a5914a48a249b8c733d86338c64a8662072825"
)
AUDITED_PRIOR_RECEIPT_SHA256 = (
    "449bd49b3e31a430eed1d4c6e217c4299084beb272d9845648ded95b7f8718e6"
)


# Reuse the already fault-tested binary-state and filesystem primitives.  The
# transaction schema, allowed diff, tokens, and receipt directory below remain
# wholly independent from the first migration.
Stage3EMADeviceMigrationError = shared.Stage3GuardAlignmentMigrationError
_assert_bit_exact = shared._assert_bit_exact
_fingerprint = shared._fingerprint
_hardlink_backup = shared._hardlink_backup
_load_cpu_checkpoint = shared._load_cpu_checkpoint
_make_backup_read_only = shared._make_backup_read_only
_make_candidate = shared._make_candidate
_replace_and_fsync = shared._replace_and_fsync
_require_mapping = shared._require_mapping
_restore_from_backup = shared._restore_from_backup
_section_evidence = shared._section_evidence
_walk_finite = shared._walk_finite


def _fail(message: str) -> NoReturn:
    raise Stage3EMADeviceMigrationError(message)


def _validate_shared_primitives() -> dict[str, Any]:
    path = SHARED_MIGRATION_SCRIPT_PATH
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != AUDITED_SHARED_MIGRATION_SCRIPT_SHA256
        or Path(shared.__file__).resolve() != path.resolve()
        or any(
            not callable(getattr(shared, name, None)) for name in SHARED_PRIMITIVE_NAMES
        )
    ):
        _fail("audited shared guard-migration primitives drifted")
    return {
        "path": str(path.resolve()),
        "sha256": AUDITED_SHARED_MIGRATION_SCRIPT_SHA256,
        "primitive_names": list(SHARED_PRIMITIVE_NAMES),
        "protected_unchanged": True,
    }


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = _absolute_without_symlink_resolution(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            _fail(f"symlink is forbidden in {label} path: {current}")


def _resolve_requested_paths(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    checkpoint: str | Path,
    state: str | Path,
    approval: str | Path,
    approval_required: str | Path,
    prior_migration_receipt: str | Path,
    backup_dir: str | Path,
) -> dict[str, Path]:
    requested = {
        "project_root": Path(project_root),
        "run_contract": Path(run_contract),
        "checkpoint": Path(checkpoint),
        "state": Path(state),
        "approval": Path(approval),
        "approval_required": Path(approval_required),
        "prior_migration_receipt": Path(prior_migration_receipt),
        "backup_dir": Path(backup_dir),
    }
    for label, path in requested.items():
        _reject_symlink_chain(path, label=label.replace("_", " "))
    return {label: path.resolve() for label, path in requested.items()}


def _validate_expected_hashes(*values: str) -> None:
    if any(not is_sha256(value) for value in values):
        _fail("every expected hash must be a lowercase SHA256")


def _validate_failed_state(path: Path) -> dict[str, Any]:
    state = _require_mapping(load_json(path), field="orchestration state")
    last_command = state.get("last_command")
    if (
        state.get("schema_version") != "graphrestore-orchestration-v1"
        or state.get("protocol_id") != PROTOCOL_ID
        or state.get("status") != "FAILED"
        or state.get("current_stage") != "FAILED"
        or state.get("gpu") != "released"
        or state.get("last_exit_code") != 1
        or state.get("next_command")
        != "python scripts/orchestrate.py --resume_post_approval_pipeline"
        or not isinstance(last_command, list)
        or "scripts/train_stage3_planner.py" not in last_command
        or "--resume" not in last_command
    ):
        _fail("orchestration state is not the exact exit-1 Stage3 recovery boundary")
    return dict(state)


def _validate_checkpoint(payload: Mapping[str, Any]) -> None:
    shared._validate_checkpoint(payload)
    if payload.get("metrics") != {}:
        _fail("first pending Stage3 validation must still have empty metrics")
    _walk_finite(payload)


def _validate_approval(
    approval_path: Path,
    approval_required_path: Path,
    *,
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
) -> dict[str, Any]:
    return shared._validate_approval(
        approval_path,
        approval_required_path,
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_approval_required_sha256,
    )


def _validate_prior_guard_receipt(
    *,
    project_root: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    expected_path = (
        project_root
        / "artifacts"
        / "migrations"
        / PRIOR_BACKUP_DIR_NAME
        / "MIGRATION_RECEIPT.json"
    ).resolve()
    if receipt_path != expected_path or receipt_path.is_symlink():
        _fail("prior guard-alignment receipt path drifted")
    if (
        not receipt_path.is_file()
        or sha256_file(receipt_path) != expected_receipt_sha256
    ):
        _fail("prior COMPLETE guard-alignment receipt SHA256 drifted")
    receipt = _require_mapping(load_json(receipt_path), field="prior migration receipt")
    backups = _require_mapping(
        receipt.get("backup"), field="prior migration receipt.backup"
    )
    raw_diff = receipt.get("exact_provenance_leaf_diff")
    if not isinstance(raw_diff, list):
        _fail("prior guard-alignment provenance diff is missing")
    expected_paths = sorted(
        (
            f"semantic_source_sha256.{ALLOWED_SOURCE_PATH}",
            f"semantic_source_sha256.{PRESERVED_STAGE4_SOURCE_PATH}",
        )
    )
    if any(
        not isinstance(row, Mapping) or not isinstance(row.get("path"), str)
        for row in raw_diff
    ):
        _fail("prior guard-alignment provenance diff is malformed")
    actual_paths = sorted(str(row["path"]) for row in raw_diff)
    if (
        receipt.get("schema_version") != PRIOR_RECEIPT_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("migration") != PRIOR_MIGRATION_KIND
        or receipt.get("status") != "COMPLETE"
        or receipt.get("backup_read_only_after_publication") is not True
        or not is_sha256(receipt.get("migration_script_sha256"))
        or actual_paths != expected_paths
        or set(backups) != {"run_contract", "checkpoint"}
    ):
        _fail("prior guard-alignment COMPLETE receipt contract drifted")
    verified_backups: dict[str, dict[str, Any]] = {}
    for label, raw in backups.items():
        evidence = _require_mapping(raw, field=f"prior receipt.backup.{label}")
        raw_path, expected_sha = evidence.get("path"), evidence.get("sha256")
        if not isinstance(raw_path, str) or not is_sha256(expected_sha):
            _fail(f"prior guard-alignment backup evidence invalid: {label}")
        requested_backup = Path(raw_path)
        _reject_symlink_chain(
            requested_backup, label=f"prior guard-alignment {label} backup"
        )
        backup = requested_backup.resolve()
        expected_inode, expected_device = evidence.get("inode"), evidence.get("device")
        if (
            backup.parent != receipt_path.parent
            or not backup.is_file()
            or sha256_file(backup) != expected_sha
            or stat.S_IMODE(backup.stat().st_mode) != 0o444
            or evidence.get("hard_link_verified") is not True
            or not isinstance(expected_inode, int)
            or not isinstance(expected_device, int)
            or backup.stat().st_ino != expected_inode
            or backup.stat().st_dev != expected_device
        ):
            _fail(f"prior guard-alignment backup drifted: {label}")
        verified_backups[str(label)] = {
            "path": str(backup),
            "sha256": str(expected_sha),
            "mode": stat.S_IMODE(backup.stat().st_mode),
            "inode": backup.stat().st_ino,
            "device": backup.stat().st_dev,
        }
    return {
        "path": str(receipt_path),
        "sha256": expected_receipt_sha256,
        "schema_version": PRIOR_RECEIPT_SCHEMA,
        "migration": PRIOR_MIGRATION_KIND,
        "status": "COMPLETE",
        "protected_unchanged": True,
        "backup": verified_backups,
    }


def _exact_single_leaf_diff(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> list[dict[str, str]]:
    old_flat = shared._flatten_provenance(old)
    new_flat = shared._flatten_provenance(new)
    if old_flat.keys() != new_flat.keys():
        _fail("provenance leaf set changed")
    changed = sorted(path for path in old_flat if old_flat[path] != new_flat[path])
    if changed != [EXPECTED_PROVENANCE_DIFF_PATH]:
        _fail(
            "unexpected provenance diff: "
            f"expected={[EXPECTED_PROVENANCE_DIFF_PATH]}, actual={changed}"
        )
    old_value = old_flat[EXPECTED_PROVENANCE_DIFF_PATH]
    new_value = new_flat[EXPECTED_PROVENANCE_DIFF_PATH]
    if not is_sha256(old_value) or not is_sha256(new_value):
        _fail("single provenance diff is not SHA256-to-SHA256")
    return [
        {
            "path": EXPECTED_PROVENANCE_DIFF_PATH,
            "old": old_value,
            "new": new_value,
        }
    ]


def _validate_paths(
    *,
    project_root: Path,
    run_contract: Path,
    checkpoint: Path,
    state: Path,
    approval: Path,
    approval_required: Path,
    prior_receipt: Path,
    backup_dir: Path,
) -> None:
    migrations = (project_root / "artifacts" / "migrations").resolve()
    if (
        (project_root / "artifacts").is_symlink()
        or (project_root / "artifacts" / "migrations").is_symlink()
        or backup_dir.is_symlink()
    ):
        _fail("artifacts/migrations path must not traverse a symlink")
    if backup_dir.parent != migrations or backup_dir.name != BACKUP_DIR_NAME:
        _fail("backup directory is not the dedicated EMA-device migration path")
    if backup_dir == prior_receipt.parent:
        _fail("EMA-device migration may not reuse the guard-alignment backup directory")
    for path, label in (
        (run_contract, "run contract"),
        (checkpoint, "checkpoint"),
        (state, "orchestration state"),
        (approval, "approval"),
        (approval_required, "approval-required"),
        (prior_receipt, "prior guard-alignment receipt"),
    ):
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise Stage3EMADeviceMigrationError(
                f"{label} escaped project root: {path}"
            ) from exc
        if not path.is_file() or path.is_symlink():
            _fail(f"missing or symlinked {label}: {path}")


def _validate_semantic_sources(
    *,
    project_root: Path,
    old_provenance: Mapping[str, Any],
    expected_old_stage3_source_sha256: str,
    expected_new_stage3_source_sha256: str,
    expected_unchanged_stage4_source_sha256: str,
    expected_semantic_source_count: int,
) -> tuple[dict[str, str], Mapping[str, Any]]:
    old_semantic = _require_mapping(
        old_provenance.get("semantic_source_sha256"),
        field="provenance.semantic_source_sha256",
    )
    current = semantic_source_hashes(project_root, entrypoints=ENTRYPOINTS)
    if (
        len(old_semantic) != expected_semantic_source_count
        or len(current) != expected_semantic_source_count
        or old_semantic.keys() != current.keys()
    ):
        _fail("semantic-source path/count contract drifted")
    physical_diffs = sorted(
        path for path in old_semantic if old_semantic[path] != current[path]
    )
    if physical_diffs != [ALLOWED_SOURCE_PATH]:
        _fail(
            "physical semantic-source drift is not exactly the Stage3 engine: "
            f"{physical_diffs}"
        )
    if (
        old_semantic.get(ALLOWED_SOURCE_PATH) != expected_old_stage3_source_sha256
        or current.get(ALLOWED_SOURCE_PATH) != expected_new_stage3_source_sha256
        or expected_old_stage3_source_sha256 == expected_new_stage3_source_sha256
    ):
        _fail("Stage3 source old/new SHA contract drifted")
    if (
        old_semantic.get(PRESERVED_STAGE4_SOURCE_PATH)
        != expected_unchanged_stage4_source_sha256
        or current.get(PRESERVED_STAGE4_SOURCE_PATH)
        != expected_unchanged_stage4_source_sha256
    ):
        _fail("Stage4 semantic-source leaf changed")
    unchanged_count = sum(old_semantic[path] == current[path] for path in old_semantic)
    if unchanged_count != expected_semantic_source_count - 1:
        _fail("the other semantic-source leaves are not all unchanged")
    return current, old_semantic


def _validate_provenance_identity(
    *,
    provenance: Mapping[str, Any],
    approval: Mapping[str, Any],
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
) -> None:
    stage3_approval = _require_mapping(
        provenance.get("stage3_approval"), field="provenance.stage3_approval"
    )
    if (
        provenance.get("protocol_id") != PROTOCOL_ID
        or stage3_approval.get("sha256") != expected_approval_sha256
        or stage3_approval.get("approval_required_sha256")
        != expected_approval_required_sha256
        or provenance.get("bindings") != approval.get("bindings")
    ):
        _fail("checkpoint provenance approval/binding identity drifted")


def _restore_pair(
    *,
    backups: Mapping[str, Any],
    run_contract: Path,
    checkpoint: Path,
    expected_run_contract_sha256: str,
    expected_checkpoint_sha256: str,
) -> list[str]:
    errors: list[str] = []
    for label, destination, expected_sha in (
        ("run_contract", run_contract, expected_run_contract_sha256),
        ("checkpoint", checkpoint, expected_checkpoint_sha256),
    ):
        try:
            raw_evidence = backups.get(label)
            if not isinstance(raw_evidence, Mapping):
                if destination.is_file() and sha256_file(destination) == expected_sha:
                    continue
                _fail(f"missing rollback backup evidence: {label}")
            evidence = raw_evidence
            raw_path, raw_mode = evidence.get("path"), evidence.get("mode")
            if not isinstance(raw_path, str) or not isinstance(raw_mode, int):
                _fail(f"invalid rollback backup evidence: {label}")
            requested_backup = Path(raw_path)
            _reject_symlink_chain(requested_backup, label=f"rollback {label} backup")
            backup = requested_backup.resolve()
            if sha256_file(backup) != expected_sha:
                _fail(f"rollback backup SHA256 mismatch: {label}")
            _restore_from_backup(backup, destination, mode=raw_mode)
            if (
                sha256_file(destination) != expected_sha
                or stat.S_IMODE(destination.stat().st_mode) != raw_mode
                or os.path.samestat(destination.stat(), backup.stat())
            ):
                _fail(f"rollback output mismatch: {label}")
            _make_backup_read_only(backup)
        except BaseException as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    return errors


def migrate_stage3_ema_device_provenance(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    checkpoint: str | Path,
    state: str | Path,
    approval: str | Path,
    approval_required: str | Path,
    prior_migration_receipt: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_checkpoint_sha256: str,
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
    expected_prior_migration_receipt_sha256: str,
    expected_old_stage3_source_sha256: str,
    expected_new_stage3_source_sha256: str,
    expected_unchanged_stage4_source_sha256: str,
    expected_semantic_source_count: int = EXPECTED_SEMANTIC_SOURCE_COUNT,
    execute: bool = False,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Prepare or execute the exact one-leaf Stage3 provenance migration."""

    resolved = _resolve_requested_paths(
        project_root=project_root,
        run_contract=run_contract,
        checkpoint=checkpoint,
        state=state,
        approval=approval,
        approval_required=approval_required,
        prior_migration_receipt=prior_migration_receipt,
        backup_dir=backup_dir,
    )
    root = resolved["project_root"]
    contract_path = resolved["run_contract"]
    checkpoint_path = resolved["checkpoint"]
    state_path = resolved["state"]
    approval_path = resolved["approval"]
    required_path = resolved["approval_required"]
    prior_receipt_path = resolved["prior_migration_receipt"]
    backup_path = resolved["backup_dir"]
    _validate_expected_hashes(
        expected_run_contract_sha256,
        expected_checkpoint_sha256,
        expected_approval_sha256,
        expected_approval_required_sha256,
        expected_prior_migration_receipt_sha256,
        expected_old_stage3_source_sha256,
        expected_new_stage3_source_sha256,
        expected_unchanged_stage4_source_sha256,
    )
    shared_evidence = _validate_shared_primitives()
    if (
        isinstance(expected_semantic_source_count, bool)
        or not isinstance(expected_semantic_source_count, int)
        or expected_semantic_source_count < 2
    ):
        _fail("semantic-source count must be an integer >= 2")
    if execute and confirmation_token != CONFIRMATION_TOKEN:
        _fail("execution requires the exact EMA-device migration token")
    _validate_paths(
        project_root=root,
        run_contract=contract_path,
        checkpoint=checkpoint_path,
        state=state_path,
        approval=approval_path,
        approval_required=required_path,
        prior_receipt=prior_receipt_path,
        backup_dir=backup_path,
    )
    if backup_path.exists():
        _fail(f"dedicated backup directory already exists: {backup_path}")
    if sha256_file(contract_path) != expected_run_contract_sha256:
        _fail("run-contract SHA256 differs from the post-guard anchor")
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        _fail("checkpoint SHA256 differs from the post-guard anchor")

    state_before_sha = sha256_file(state_path)
    state_evidence = _validate_failed_state(state_path)
    approval_evidence = _validate_approval(
        approval_path,
        required_path,
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_approval_required_sha256,
    )
    prior_evidence = _validate_prior_guard_receipt(
        project_root=root,
        receipt_path=prior_receipt_path,
        expected_receipt_sha256=expected_prior_migration_receipt_sha256,
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
    approval_value = _require_mapping(load_json(approval_path), field="approval")
    _validate_provenance_identity(
        provenance=old_provenance,
        approval=approval_value,
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_approval_required_sha256,
    )
    current_semantic, old_semantic = _validate_semantic_sources(
        project_root=root,
        old_provenance=old_provenance,
        expected_old_stage3_source_sha256=expected_old_stage3_source_sha256,
        expected_new_stage3_source_sha256=expected_new_stage3_source_sha256,
        expected_unchanged_stage4_source_sha256=(
            expected_unchanged_stage4_source_sha256
        ),
        expected_semantic_source_count=expected_semantic_source_count,
    )

    new_provenance = copy.deepcopy(dict(old_provenance))
    new_semantic = dict(old_semantic)
    new_semantic[ALLOWED_SOURCE_PATH] = current_semantic[ALLOWED_SOURCE_PATH]
    new_provenance["semantic_source_sha256"] = new_semantic
    provenance_diff = _exact_single_leaf_diff(old_provenance, new_provenance)
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
                checkpoint_payload[key], new_checkpoint[key], path=f"checkpoint.{key}"
            )

    contract_candidate = _make_candidate(
        contract_path.parent, contract_path.name, ".ema-device.candidate.json"
    )
    checkpoint_candidate = _make_candidate(
        checkpoint_path.parent, checkpoint_path.name, ".ema-device.candidate.pth"
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
        if reloaded.get("provenance") != new_provenance:
            _fail("checkpoint candidate provenance differs from run contract")
        section_evidence = _section_evidence(checkpoint_payload, reloaded)
        new_contract_sha = sha256_file(contract_candidate)
        new_checkpoint_sha = sha256_file(checkpoint_candidate)
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "migration": MIGRATION_KIND,
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
            "semantic_source_count": expected_semantic_source_count,
            "unchanged_semantic_source_count": expected_semantic_source_count - 1,
            "preserved_stage4_source": {
                "path": PRESERVED_STAGE4_SOURCE_PATH,
                "sha256": expected_unchanged_stage4_source_sha256,
            },
            "checkpoint_section_fingerprints": section_evidence,
            "checkpoint_state_bit_exact_outside_provenance": True,
            "run_contract_bit_exact_outside_provenance": True,
            "all_checkpoint_tensors_finite": True,
            "approval_and_bindings_unchanged": approval_evidence,
            "prior_guard_alignment_migration": prior_evidence,
            "shared_guard_migration_primitives": shared_evidence,
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

        if (
            sha256_file(contract_path) != expected_run_contract_sha256
            or sha256_file(checkpoint_path) != expected_checkpoint_sha256
            or sha256_file(state_path) != state_before_sha
            or sha256_file(prior_receipt_path)
            != expected_prior_migration_receipt_sha256
            or semantic_source_hashes(root, entrypoints=ENTRYPOINTS) != current_semantic
        ):
            _fail("migration inputs changed before publication")
        _validate_failed_state(state_path)
        _validate_approval(
            approval_path,
            required_path,
            expected_approval_sha256=expected_approval_sha256,
            expected_approval_required_sha256=expected_approval_required_sha256,
        )
        _validate_prior_guard_receipt(
            project_root=root,
            receipt_path=prior_receipt_path,
            expected_receipt_sha256=expected_prior_migration_receipt_sha256,
        )
        if _validate_shared_primitives() != shared_evidence:
            _fail("shared guard-migration primitives changed before publication")

        artifacts_path = (root / "artifacts").resolve()
        migrations_path = (artifacts_path / "migrations").resolve()
        if not artifacts_path.is_dir():
            _fail("project artifacts directory is missing")
        migrations_path.mkdir(parents=False, exist_ok=True)
        fsync_directory(artifacts_path)
        backup_path.mkdir(parents=False, exist_ok=False)
        fsync_directory(migrations_path)
        if (
            backup_path.stat().st_dev != contract_path.stat().st_dev
            or backup_path.stat().st_dev != checkpoint_path.stat().st_dev
        ):
            _fail("backup directory is not on the same filesystem as both sources")
        contract_backup = backup_path / (
            f"run_contract.pre_ema_device.{expected_run_contract_sha256}.json"
        )
        checkpoint_backup = backup_path / (
            f"last.pre_ema_device.{expected_checkpoint_sha256}.pth"
        )
        backups["run_contract"] = _hardlink_backup(contract_path, contract_backup)
        backups["checkpoint"] = _hardlink_backup(checkpoint_path, checkpoint_backup)
        receipt["backup"] = backups
        atomic_write_json(receipt_path, receipt)

        _replace_and_fsync(checkpoint_candidate, checkpoint_path)
        _replace_and_fsync(contract_candidate, contract_path)
        if (
            sha256_file(checkpoint_path) != new_checkpoint_sha
            or sha256_file(contract_path) != new_contract_sha
        ):
            _fail("published migration hash differs from verified candidate")
        published_contract = _require_mapping(
            load_json(contract_path), field="published run contract"
        )
        published_checkpoint = _load_cpu_checkpoint(checkpoint_path)
        if (
            published_contract.get("provenance") != new_provenance
            or published_checkpoint.get("provenance") != new_provenance
        ):
            _fail("published provenance pair is not identical")
        _section_evidence(checkpoint_payload, published_checkpoint)
        _assert_bit_exact(
            {key: value for key, value in contract.items() if key != "provenance"},
            {
                key: value
                for key, value in published_contract.items()
                if key != "provenance"
            },
            path="published_run_contract.outside_provenance",
        )
        if (
            semantic_source_hashes(root, entrypoints=ENTRYPOINTS) != current_semantic
            or sha256_file(state_path) != state_before_sha
            or sha256_file(prior_receipt_path)
            != expected_prior_migration_receipt_sha256
        ):
            _fail("protected sources, state, or prior receipt changed during migration")
        _validate_approval(
            approval_path,
            required_path,
            expected_approval_sha256=expected_approval_sha256,
            expected_approval_required_sha256=expected_approval_required_sha256,
        )
        _validate_prior_guard_receipt(
            project_root=root,
            receipt_path=prior_receipt_path,
            expected_receipt_sha256=expected_prior_migration_receipt_sha256,
        )
        if _validate_shared_primitives() != shared_evidence:
            _fail("shared guard-migration primitives changed during publication")
        for backup in (contract_backup, checkpoint_backup):
            _make_backup_read_only(backup)
        fsync_directory(backup_path)
        receipt["status"] = "COMPLETE"
        receipt["completed_utc"] = utc_now_iso()
        receipt["backup_read_only_after_publication"] = True
        receipt["prior_guard_alignment_receipt_unchanged_after_publication"] = True
        atomic_write_json(receipt_path, receipt)
        return receipt
    except BaseException as original_error:
        if execute and backups:
            rollback_errors = _restore_pair(
                backups=backups,
                run_contract=contract_path,
                checkpoint=checkpoint_path,
                expected_run_contract_sha256=expected_run_contract_sha256,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
            )
            if (
                sha256_file(prior_receipt_path)
                != expected_prior_migration_receipt_sha256
            ):
                rollback_errors.append("prior guard-alignment receipt SHA256 changed")
            try:
                if _validate_shared_primitives() != shared_evidence:
                    rollback_errors.append("shared guard-migration primitives changed")
            except BaseException as shared_error:
                rollback_errors.append(
                    "shared primitive verification failed: "
                    f"{type(shared_error).__name__}: {shared_error}"
                )
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
                        "old_run_contract_sha256": expected_run_contract_sha256,
                        "old_checkpoint_sha256": expected_checkpoint_sha256,
                        "prior_guard_alignment_receipt_sha256": (
                            expected_prior_migration_receipt_sha256
                        ),
                        "shared_guard_migration_script_sha256": (
                            AUDITED_SHARED_MIGRATION_SCRIPT_SHA256
                        ),
                        "backup": backups,
                        "rollback_errors": rollback_errors,
                    },
                )
            if rollback_errors:
                raise Stage3EMADeviceMigrationError(
                    "migration publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from original_error
        raise
    finally:
        contract_candidate.unlink(missing_ok=True)
        checkpoint_candidate.unlink(missing_ok=True)


def recover_prepared_stage3_ema_device_provenance(
    *,
    project_root: str | Path,
    run_contract: str | Path,
    checkpoint: str | Path,
    state: str | Path,
    approval: str | Path,
    approval_required: str | Path,
    prior_migration_receipt: str | Path,
    backup_dir: str | Path,
    expected_run_contract_sha256: str,
    expected_checkpoint_sha256: str,
    expected_approval_sha256: str,
    expected_approval_required_sha256: str,
    expected_prior_migration_receipt_sha256: str,
    expected_old_stage3_source_sha256: str,
    expected_new_stage3_source_sha256: str,
    expected_unchanged_stage4_source_sha256: str,
    expected_semantic_source_count: int = EXPECTED_SEMANTIC_SOURCE_COUNT,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Restore both audited old files from an interrupted PREPARED receipt."""

    if confirmation_token != RECOVERY_CONFIRMATION_TOKEN:
        _fail("PREPARED recovery requires the exact EMA-device recovery token")
    resolved = _resolve_requested_paths(
        project_root=project_root,
        run_contract=run_contract,
        checkpoint=checkpoint,
        state=state,
        approval=approval,
        approval_required=approval_required,
        prior_migration_receipt=prior_migration_receipt,
        backup_dir=backup_dir,
    )
    root = resolved["project_root"]
    contract_path = resolved["run_contract"]
    checkpoint_path = resolved["checkpoint"]
    state_path = resolved["state"]
    approval_path = resolved["approval"]
    required_path = resolved["approval_required"]
    prior_receipt_path = resolved["prior_migration_receipt"]
    backup_path = resolved["backup_dir"]
    _validate_expected_hashes(
        expected_run_contract_sha256,
        expected_checkpoint_sha256,
        expected_approval_sha256,
        expected_approval_required_sha256,
        expected_prior_migration_receipt_sha256,
        expected_old_stage3_source_sha256,
        expected_new_stage3_source_sha256,
        expected_unchanged_stage4_source_sha256,
    )
    shared_evidence = _validate_shared_primitives()
    _validate_paths(
        project_root=root,
        run_contract=contract_path,
        checkpoint=checkpoint_path,
        state=state_path,
        approval=approval_path,
        approval_required=required_path,
        prior_receipt=prior_receipt_path,
        backup_dir=backup_path,
    )
    receipt_path = backup_path / "MIGRATION_RECEIPT.json"
    if not backup_path.is_dir() or not receipt_path.is_file():
        _fail("PREPARED recovery requires its dedicated receipt")
    state_before_sha = sha256_file(state_path)
    _validate_failed_state(state_path)
    approval_evidence = _validate_approval(
        approval_path,
        required_path,
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_approval_required_sha256,
    )
    _validate_prior_guard_receipt(
        project_root=root,
        receipt_path=prior_receipt_path,
        expected_receipt_sha256=expected_prior_migration_receipt_sha256,
    )
    current_semantic = semantic_source_hashes(root, entrypoints=ENTRYPOINTS)
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
    prior = _require_mapping(
        receipt.get("prior_guard_alignment_migration"),
        field="migration receipt.prior_guard_alignment_migration",
    )
    receipt_approval = _require_mapping(
        receipt.get("approval_and_bindings_unchanged"),
        field="migration receipt.approval_and_bindings_unchanged",
    )
    preserved_stage4 = _require_mapping(
        receipt.get("preserved_stage4_source"),
        field="migration receipt.preserved_stage4_source",
    )
    receipt_shared = _require_mapping(
        receipt.get("shared_guard_migration_primitives"),
        field="migration receipt.shared_guard_migration_primitives",
    )
    old_contract = _require_mapping(
        old.get("run_contract"), field="migration receipt.old.run_contract"
    )
    old_checkpoint = _require_mapping(
        old.get("checkpoint"), field="migration receipt.old.checkpoint"
    )
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
        or receipt.get("orchestration_state_sha256") != state_before_sha
        or receipt.get("exact_provenance_leaf_diff") != expected_diff
        or receipt.get("semantic_source_count") != expected_semantic_source_count
        or receipt.get("unchanged_semantic_source_count")
        != expected_semantic_source_count - 1
        or old_contract.get("path") != str(contract_path)
        or old_contract.get("sha256") != expected_run_contract_sha256
        or old_checkpoint.get("path") != str(checkpoint_path)
        or old_checkpoint.get("sha256") != expected_checkpoint_sha256
        or not is_sha256(new.get("run_contract_sha256"))
        or not is_sha256(new.get("checkpoint_sha256"))
        or prior.get("sha256") != expected_prior_migration_receipt_sha256
        or prior.get("protected_unchanged") is not True
        or prior.get("path") != str(prior_receipt_path)
        or receipt_approval.get("approval_sha256") != expected_approval_sha256
        or receipt_approval.get("approval_required_sha256")
        != expected_approval_required_sha256
        or receipt_approval.get("binding_count") != EXPECTED_BINDING_COUNT
        or receipt_approval.get("binding_sha256")
        != approval_evidence.get("binding_sha256")
        or preserved_stage4.get("path") != PRESERVED_STAGE4_SOURCE_PATH
        or preserved_stage4.get("sha256") != expected_unchanged_stage4_source_sha256
        or receipt_shared != shared_evidence
    ):
        _fail("PREPARED receipt does not match the audited one-leaf transaction")

    expected_new_hashes = {
        "run_contract": str(new["run_contract_sha256"]),
        "checkpoint": str(new["checkpoint_sha256"]),
    }
    recovered_from: dict[str, str] = {}
    backup_paths: dict[str, Path] = {}
    for label, destination, expected_old in (
        ("run_contract", contract_path, expected_run_contract_sha256),
        ("checkpoint", checkpoint_path, expected_checkpoint_sha256),
    ):
        evidence = _require_mapping(
            backups.get(label), field=f"migration receipt.backup.{label}"
        )
        raw_path, raw_sha, raw_mode = (
            evidence.get("path"),
            evidence.get("sha256"),
            evidence.get("mode"),
        )
        raw_inode, raw_device = evidence.get("inode"), evidence.get("device")
        if (
            not isinstance(raw_path, str)
            or not isinstance(raw_mode, int)
            or not isinstance(raw_inode, int)
            or not isinstance(raw_device, int)
        ):
            _fail(f"invalid PREPARED backup evidence: {label}")
        requested_backup = Path(raw_path)
        _reject_symlink_chain(requested_backup, label=f"PREPARED {label} backup")
        backup = requested_backup.resolve()
        live_sha = sha256_file(destination)
        if (
            backup.parent != backup_path
            or raw_sha != expected_old
            or not backup.is_file()
            or sha256_file(backup) != expected_old
            or backup.stat().st_ino != raw_inode
            or backup.stat().st_dev != raw_device
            or stat.S_IMODE(backup.stat().st_mode) not in {raw_mode, 0o444}
            or live_sha not in {expected_old, expected_new_hashes[label]}
        ):
            _fail(f"PREPARED backup/live state drifted: {label}")
        recovered_from[label] = live_sha
        backup_paths[label] = backup

    backup_contract = _require_mapping(
        load_json(backup_paths["run_contract"]),
        field="PREPARED run-contract backup",
    )
    if backup_contract.get("schema_version") != STAGE3_SCHEMA:
        _fail("PREPARED run-contract backup schema drifted")
    backup_checkpoint = _load_cpu_checkpoint(backup_paths["checkpoint"])
    _validate_checkpoint(backup_checkpoint)
    backup_provenance = _require_mapping(
        backup_contract.get("provenance"),
        field="PREPARED run-contract backup provenance",
    )
    if backup_checkpoint.get("provenance") != backup_provenance:
        _fail("PREPARED backup provenance pair differs")
    _validate_provenance_identity(
        provenance=backup_provenance,
        approval=_require_mapping(load_json(approval_path), field="approval"),
        expected_approval_sha256=expected_approval_sha256,
        expected_approval_required_sha256=expected_approval_required_sha256,
    )
    verified_current, _ = _validate_semantic_sources(
        project_root=root,
        old_provenance=backup_provenance,
        expected_old_stage3_source_sha256=expected_old_stage3_source_sha256,
        expected_new_stage3_source_sha256=expected_new_stage3_source_sha256,
        expected_unchanged_stage4_source_sha256=(
            expected_unchanged_stage4_source_sha256
        ),
        expected_semantic_source_count=expected_semantic_source_count,
    )
    if verified_current != current_semantic:
        _fail("semantic sources changed during PREPARED receipt verification")

    if receipt.get("status") == "ROLLED_BACK_FROM_PREPARED":
        if (
            receipt.get("recovery_confirmation_token_sha256")
            != hashlib.sha256(RECOVERY_CONFIRMATION_TOKEN.encode()).hexdigest()
            or receipt.get("prior_guard_alignment_receipt_unchanged_after_recovery")
            is not True
        ):
            _fail("finalized PREPARED recovery evidence is incomplete")
        for label, destination, expected_old in (
            ("run_contract", contract_path, expected_run_contract_sha256),
            ("checkpoint", checkpoint_path, expected_checkpoint_sha256),
        ):
            evidence = _require_mapping(
                backups.get(label), field=f"migration receipt.backup.{label}"
            )
            requested_backup = Path(str(evidence["path"]))
            _reject_symlink_chain(
                requested_backup, label=f"finalized PREPARED {label} backup"
            )
            backup = requested_backup.resolve()
            if (
                sha256_file(destination) != expected_old
                or stat.S_IMODE(backup.stat().st_mode) != 0o444
                or os.path.samestat(destination.stat(), backup.stat())
            ):
                _fail(f"finalized PREPARED recovery drifted: {label}")
        if _validate_shared_primitives() != shared_evidence:
            _fail("shared primitives changed during finalized recovery validation")
        return dict(receipt)

    rollback_errors = _restore_pair(
        backups=backups,
        run_contract=contract_path,
        checkpoint=checkpoint_path,
        expected_run_contract_sha256=expected_run_contract_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if rollback_errors:
        _fail("PREPARED recovery was incomplete: " + "; ".join(rollback_errors))
    if (
        sha256_file(state_path) != state_before_sha
        or sha256_file(prior_receipt_path) != expected_prior_migration_receipt_sha256
        or semantic_source_hashes(root, entrypoints=ENTRYPOINTS) != current_semantic
        or _validate_shared_primitives() != shared_evidence
    ):
        _fail("protected state changed during PREPARED recovery")
    recovered_receipt = dict(receipt)
    recovered_receipt["status"] = "ROLLED_BACK_FROM_PREPARED"
    recovered_receipt["recovered_utc"] = utc_now_iso()
    recovered_receipt["recovered_from_live_sha256"] = recovered_from
    recovered_receipt["recovery_confirmation_token_sha256"] = hashlib.sha256(
        RECOVERY_CONFIRMATION_TOKEN.encode()
    ).hexdigest()
    recovered_receipt["backup_read_only_after_recovery"] = True
    recovered_receipt["prior_guard_alignment_receipt_unchanged_after_recovery"] = True
    recovered_receipt["shared_guard_migration_primitives_unchanged_after_recovery"] = (
        True
    )
    atomic_write_json(receipt_path, recovered_receipt)
    return recovered_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-required", type=Path, required=True)
    parser.add_argument("--prior-migration-receipt", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-run-contract-sha256", default=AUDITED_RUN_CONTRACT_SHA256
    )
    parser.add_argument(
        "--expected-checkpoint-sha256", default=AUDITED_CHECKPOINT_SHA256
    )
    parser.add_argument("--expected-approval-sha256", default=AUDITED_APPROVAL_SHA256)
    parser.add_argument(
        "--expected-approval-required-sha256",
        default=AUDITED_APPROVAL_REQUIRED_SHA256,
    )
    parser.add_argument(
        "--expected-prior-migration-receipt-sha256",
        default=AUDITED_PRIOR_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--expected-old-stage3-source-sha256",
        default=AUDITED_OLD_STAGE3_SOURCE_SHA256,
    )
    parser.add_argument("--expected-new-stage3-source-sha256", required=True)
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
        "checkpoint": arguments.checkpoint,
        "state": arguments.state,
        "approval": arguments.approval,
        "approval_required": arguments.approval_required,
        "prior_migration_receipt": arguments.prior_migration_receipt,
        "backup_dir": arguments.backup_dir,
        "expected_run_contract_sha256": arguments.expected_run_contract_sha256,
        "expected_checkpoint_sha256": arguments.expected_checkpoint_sha256,
        "expected_approval_sha256": arguments.expected_approval_sha256,
        "expected_approval_required_sha256": (
            arguments.expected_approval_required_sha256
        ),
        "expected_prior_migration_receipt_sha256": (
            arguments.expected_prior_migration_receipt_sha256
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
            receipt = recover_prepared_stage3_ema_device_provenance(
                **common, confirmation_token=arguments.confirmation_token
            )
        else:
            receipt = migrate_stage3_ema_device_provenance(
                **common,
                execute=arguments.execute,
                confirmation_token=arguments.confirmation_token,
            )
    except (Stage3EMADeviceMigrationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json_dumps(receipt),
        flush=True,
    )
    return 0


def json_dumps(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
