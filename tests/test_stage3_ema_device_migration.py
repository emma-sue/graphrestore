from __future__ import annotations

import copy
import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import migrate_stage3_ema_device_provenance as migration

import torch

from src.training.checkpointing import atomic_torch_save
from src.training.provenance import semantic_source_hashes
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, load_json


@dataclass(frozen=True)
class _Case:
    root: Path
    contract: Path
    checkpoint: Path
    state: Path
    approval: Path
    approval_required: Path
    prior_receipt: Path
    backup_dir: Path
    contract_sha: str
    checkpoint_sha: str
    approval_sha: str
    approval_required_sha: str
    prior_receipt_sha: str
    old_stage3_sha: str
    new_stage3_sha: str
    stage4_sha: str
    original_payload: dict[str, Any]
    source_count: int

    def arguments(self, *, execute: bool = False) -> dict[str, Any]:
        return {
            "project_root": self.root,
            "run_contract": self.contract,
            "checkpoint": self.checkpoint,
            "state": self.state,
            "approval": self.approval,
            "approval_required": self.approval_required,
            "prior_migration_receipt": self.prior_receipt,
            "backup_dir": self.backup_dir,
            "expected_run_contract_sha256": self.contract_sha,
            "expected_checkpoint_sha256": self.checkpoint_sha,
            "expected_approval_sha256": self.approval_sha,
            "expected_approval_required_sha256": self.approval_required_sha,
            "expected_prior_migration_receipt_sha256": self.prior_receipt_sha,
            "expected_old_stage3_source_sha256": self.old_stage3_sha,
            "expected_new_stage3_source_sha256": self.new_stage3_sha,
            "expected_unchanged_stage4_source_sha256": self.stage4_sha,
            "expected_semantic_source_count": self.source_count,
            "execute": execute,
            "confirmation_token": (migration.CONFIRMATION_TOKEN if execute else None),
        }

    def recovery_arguments(self) -> dict[str, Any]:
        values = self.arguments()
        values.pop("execute")
        values["confirmation_token"] = migration.RECOVERY_CONFIRMATION_TOKEN
        return values


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


def _make_prior_receipt(root: Path) -> Path:
    directory = root / "artifacts" / "migrations" / migration.PRIOR_BACKUP_DIR_NAME
    directory.mkdir(parents=True)
    backups: dict[str, dict[str, Any]] = {}
    for label, suffix in (("run_contract", "json"), ("checkpoint", "pth")):
        path = directory / f"prior-{label}.{suffix}"
        path.write_bytes(f"immutable prior {label}\n".encode())
        os.chmod(path, 0o444)
        backups[label] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "mode": 0o600,
            "device": path.stat().st_dev,
            "inode": path.stat().st_ino,
            "hard_link_verified": True,
        }
    receipt = directory / "MIGRATION_RECEIPT.json"
    atomic_write_json(
        receipt,
        {
            "schema_version": migration.PRIOR_RECEIPT_SCHEMA,
            "protocol_id": migration.PROTOCOL_ID,
            "migration": migration.PRIOR_MIGRATION_KIND,
            "status": "COMPLETE",
            "migration_script_sha256": "d" * 64,
            "backup_read_only_after_publication": True,
            "backup": backups,
            "exact_provenance_leaf_diff": [
                {
                    "path": (f"semantic_source_sha256.{migration.ALLOWED_SOURCE_PATH}"),
                    "old": "a" * 64,
                    "new": "b" * 64,
                },
                {
                    "path": (
                        "semantic_source_sha256."
                        f"{migration.PRESERVED_STAGE4_SOURCE_PATH}"
                    ),
                    "old": "b" * 64,
                    "new": "c" * 64,
                },
            ],
        },
    )
    return receipt


def _make_case(tmp_path: Path) -> _Case:
    root = tmp_path / "project"
    (root / "src/training").mkdir(parents=True)
    (root / "scripts").mkdir()
    stage3_old = b"stage3 guard-alignment implementation\n"
    stage3_new = b"stage3 exact cpu/cuda ema comparison\n"
    stage4 = b"stage4 guard-alignment implementation unchanged\n"
    source_content = {
        migration.ALLOWED_SOURCE_PATH: stage3_new,
        migration.PRESERVED_STAGE4_SOURCE_PATH: stage4,
        "src/training/unchanged.py": b"unchanged semantic source\n",
        "scripts/train_stage3_planner.py": b"stage3 entrypoint\n",
        "scripts/eval_guard_diagnostics.py": b"guard entrypoint\n",
    }
    # 45 Python files under src plus two explicit entrypoints = 47 sources.
    for index in range(42):
        source_content[f"src/dummy_{index:02d}.py"] = f"DUMMY = {index}\n".encode()
    for relative, content in source_content.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    current = semantic_source_hashes(root, entrypoints=migration.ENTRYPOINTS)
    assert len(current) == migration.EXPECTED_SEMANTIC_SOURCE_COUNT
    old_semantic = dict(current)
    old_semantic[migration.ALLOWED_SOURCE_PATH] = _sha_bytes(stage3_old)

    binding_dir = root / "bindings"
    binding_dir.mkdir()
    bindings: dict[str, dict[str, str]] = {}
    for index in range(migration.EXPECTED_BINDING_COUNT):
        path = binding_dir / f"binding_{index:02d}.txt"
        path.write_text(f"binding {index}\n", encoding="utf-8")
        bindings[f"binding_{index:02d}"] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    approval_required = root / "artifacts/approvals/STAGE3_APPROVAL_REQUIRED.json"
    approval_required.parent.mkdir(parents=True)
    atomic_write_json(
        approval_required,
        {
            "schema_version": migration.APPROVAL_SCHEMA,
            "kind": "stage3_approval_required",
            "protocol_id": migration.PROTOCOL_ID,
            "approved": False,
            "bindings": bindings,
        },
    )
    required_sha = sha256_file(approval_required)
    approval = approval_required.parent / "STAGE3_APPROVED.json"
    atomic_write_json(
        approval,
        {
            "schema_version": migration.APPROVAL_SCHEMA,
            "kind": "stage3_approval",
            "protocol_id": migration.PROTOCOL_ID,
            "approved": True,
            "approval_required_sha256": required_sha,
            "bindings": bindings,
        },
    )
    approval_sha = sha256_file(approval)
    provenance: dict[str, Any] = {
        "schema_version": migration.STAGE3_SCHEMA,
        "protocol_id": migration.PROTOCOL_ID,
        "semantic_source_sha256": old_semantic,
        "stage3_approval": {
            "sha256": approval_sha,
            "approval_required_sha256": required_sha,
        },
        "bindings": bindings,
        "parent_checkpoint": {"sha256": "a" * 64, "step": 30_000},
        "runtime": {"micro_batch": 4, "accumulation_steps": 2},
    }
    contract = root / "artifacts/checkpoints/stage3/run_contract.json"
    contract.parent.mkdir(parents=True)
    atomic_write_json(
        contract,
        {
            "schema_version": migration.STAGE3_SCHEMA,
            "created_utc": "2026-08-17T09:30:11Z",
            "provenance": provenance,
            "parent_load": {"loaded_count": 1535},
            "micro_batch_trials": [{"micro_batch": 4, "passed": True}],
            "validation_vram_gate": {"passed": True, "peak": 0.5},
        },
    )
    model = OrderedDict(
        {
            "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "planner.weight": torch.tensor([-0.0, 1.5], dtype=torch.float32),
        }
    )
    original_payload: dict[str, Any] = {
        "schema_version": migration.CHECKPOINT_SCHEMA,
        "stage": "stage3",
        "step": migration.MIGRATION_STEP,
        "model": model,
        "ema": {
            "decay": 0.9999,
            "num_updates": migration.MIGRATION_STEP,
            "scope": "planner_parameters_only_executor_bitwise_frozen",
            "shadow": OrderedDict(
                (name, value.clone()) for name, value in model.items()
            ),
        },
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(2000.0),
                    "exp_avg": torch.tensor([-0.0, 0.25]),
                    "exp_avg_sq": torch.tensor([0.5, 1.0]),
                }
            },
            "param_groups": [{"lr": 1.0e-4, "params": [0]}],
        },
        "scheduler": {"last_epoch": 2000, "_step_count": 2001},
        "scaler": None,
        "rng_states": {
            "python": (3, (1, 2, 3), None),
            "numpy": ("MT19937", np.arange(8, dtype=np.uint32), 0, 0, 0.0),
            "torch_cpu": torch.arange(12, dtype=torch.uint8),
            "torch_cuda_all": [torch.arange(8, dtype=torch.uint8)],
        },
        "sampler_state": {
            "consumed_optimizer_step": 2000,
            "sample_cursor": 16000,
        },
        "provenance": provenance,
        "metrics": {},
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
        "model_role": "raw_training_state",
        "resumable": True,
        "pending_validation_step": 2000,
        "optimizer_transaction_active": False,
    }
    checkpoint = contract.parent / "last.pth"
    atomic_torch_save(original_payload, checkpoint)
    state = root / "artifacts/orchestration/state.json"
    atomic_write_json(
        state,
        {
            "schema_version": "graphrestore-orchestration-v1",
            "protocol_id": migration.PROTOCOL_ID,
            "status": "FAILED",
            "current_stage": "FAILED",
            "gpu": "released",
            "last_exit_code": 1,
            "last_command": [
                "/python",
                "scripts/train_stage3_planner.py",
                "--config",
                "configs/stage3.yaml",
                "--resume",
                str(checkpoint),
            ],
            "next_command": (
                "python scripts/orchestrate.py --resume_post_approval_pipeline"
            ),
        },
    )
    prior_receipt = _make_prior_receipt(root)
    return _Case(
        root=root,
        contract=contract,
        checkpoint=checkpoint,
        state=state,
        approval=approval,
        approval_required=approval_required,
        prior_receipt=prior_receipt,
        backup_dir=(root / "artifacts/migrations" / migration.BACKUP_DIR_NAME),
        contract_sha=sha256_file(contract),
        checkpoint_sha=sha256_file(checkpoint),
        approval_sha=approval_sha,
        approval_required_sha=required_sha,
        prior_receipt_sha=sha256_file(prior_receipt),
        old_stage3_sha=_sha_bytes(stage3_old),
        new_stage3_sha=current[migration.ALLOWED_SOURCE_PATH],
        stage4_sha=current[migration.PRESERVED_STAGE4_SOURCE_PATH],
        original_payload=original_payload,
        source_count=len(current),
    )


def test_dry_run_is_exactly_one_leaf_and_writes_nothing(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    state_sha = sha256_file(case.state)
    result = migration.migrate_stage3_ema_device_provenance(**case.arguments())

    assert result["status"] == "DRY_RUN"
    assert result["schema_version"] == migration.RECEIPT_SCHEMA
    assert result["exact_provenance_leaf_diff"] == [
        {
            "path": migration.EXPECTED_PROVENANCE_DIFF_PATH,
            "old": case.old_stage3_sha,
            "new": case.new_stage3_sha,
        }
    ]
    assert result["unchanged_semantic_source_count"] == case.source_count - 1
    assert result["preserved_stage4_source"]["sha256"] == case.stage4_sha
    assert result["checkpoint_state_bit_exact_outside_provenance"] is True
    assert result["run_contract_bit_exact_outside_provenance"] is True
    assert result["prior_guard_alignment_migration"]["sha256"] == (
        case.prior_receipt_sha
    )
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert sha256_file(case.state) == state_sha
    assert sha256_file(case.prior_receipt) == case.prior_receipt_sha
    assert not case.backup_dir.exists()
    assert migration.os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_execute_publishes_one_leaf_with_independent_receipt(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    prior_sha = sha256_file(case.prior_receipt)
    state_sha = sha256_file(case.state)
    result = migration.migrate_stage3_ema_device_provenance(
        **case.arguments(execute=True)
    )

    assert result["status"] == "COMPLETE"
    assert result["schema_version"] != migration.PRIOR_RECEIPT_SCHEMA
    assert result["migration"] != migration.PRIOR_MIGRATION_KIND
    assert sha256_file(case.state) == state_sha
    assert sha256_file(case.prior_receipt) == prior_sha
    assert sha256_file(case.contract) == result["new"]["run_contract_sha256"]
    assert sha256_file(case.checkpoint) == result["new"]["checkpoint_sha256"]
    contract = load_json(case.contract)
    checkpoint = torch.load(case.checkpoint, map_location="cpu", weights_only=False)
    assert contract["provenance"] == checkpoint["provenance"]
    semantic = checkpoint["provenance"]["semantic_source_sha256"]
    assert semantic[migration.ALLOWED_SOURCE_PATH] == case.new_stage3_sha
    assert semantic[migration.PRESERVED_STAGE4_SOURCE_PATH] == case.stage4_sha
    for key in case.original_payload:
        if key != "provenance":
            migration._assert_bit_exact(
                case.original_payload[key], checkpoint[key], path=f"checkpoint.{key}"
            )
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "COMPLETE"
    assert receipt["prior_guard_alignment_receipt_unchanged_after_publication"]
    assert receipt["shared_guard_migration_primitives"]["sha256"] == (
        migration.AUDITED_SHARED_MIGRATION_SCRIPT_SHA256
    )
    for evidence in receipt["backup"].values():
        assert _mode(Path(evidence["path"])) == 0o444


def test_wrong_execution_token_fails_before_writes(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    arguments = case.arguments(execute=True)
    arguments["confirmation_token"] = "wrong"
    with pytest.raises(migration.Stage3EMADeviceMigrationError, match="exact"):
        migration.migrate_stage3_ema_device_provenance(**arguments)
    assert not case.backup_dir.exists()


@pytest.mark.parametrize("drift", ["stage4", "other_source"])
def test_any_second_physical_source_change_is_rejected(
    tmp_path: Path, drift: str
) -> None:
    case = _make_case(tmp_path)
    relative = (
        migration.PRESERVED_STAGE4_SOURCE_PATH
        if drift == "stage4"
        else "src/training/unchanged.py"
    )
    (case.root / relative).write_text("drift\n", encoding="utf-8")
    with pytest.raises(
        migration.Stage3EMADeviceMigrationError,
        match="exactly the Stage3 engine",
    ):
        migration.migrate_stage3_ema_device_provenance(**case.arguments())


@pytest.mark.parametrize("drift", ["state", "binding", "prior_backup"])
def test_state_approval_and_prior_receipt_closure_are_fail_closed(
    tmp_path: Path, drift: str
) -> None:
    case = _make_case(tmp_path)
    if drift == "state":
        state = load_json(case.state)
        state["last_exit_code"] = 3
        atomic_write_json(case.state, state)
        match = "exit-1"
    elif drift == "binding":
        approval = load_json(case.approval)
        bound = Path(next(iter(approval["bindings"].values()))["path"])
        bound.write_text("changed\n", encoding="utf-8")
        match = "binding changed"
    else:
        receipt = load_json(case.prior_receipt)
        backup = Path(receipt["backup"]["checkpoint"]["path"])
        os.chmod(backup, 0o644)
        match = "backup drifted"
    with pytest.raises(migration.Stage3EMADeviceMigrationError, match=match):
        migration.migrate_stage3_ema_device_provenance(**case.arguments())
    assert not case.backup_dir.exists()


@pytest.mark.parametrize(
    "field",
    [
        "project_root",
        "run_contract",
        "checkpoint",
        "state",
        "approval",
        "approval_required",
        "prior_migration_receipt",
        "backup_dir",
    ],
)
def test_every_raw_input_symlink_is_rejected_before_resolution(
    tmp_path: Path, field: str
) -> None:
    case = _make_case(tmp_path)
    arguments = case.arguments()
    target = Path(arguments[field])
    if field == "backup_dir":
        target = case.root / "unrelated-migration-directory"
        target.mkdir()
    link_parent = tmp_path / "raw-links"
    link_parent.mkdir(exist_ok=True)
    link = link_parent / field
    link.symlink_to(target, target_is_directory=target.is_dir())
    arguments[field] = link
    with pytest.raises(migration.Stage3EMADeviceMigrationError, match="symlink"):
        migration.migrate_stage3_ema_device_provenance(**arguments)


def test_symlinked_parent_component_is_rejected_before_resolution(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    alias = case.root / "artifacts-alias"
    alias.symlink_to(case.root / "artifacts", target_is_directory=True)
    arguments = case.arguments()
    arguments["run_contract"] = alias / "checkpoints/stage3/run_contract.json"
    with pytest.raises(migration.Stage3EMADeviceMigrationError, match="symlink"):
        migration.migrate_stage3_ema_device_provenance(**arguments)


def test_prior_backup_symlink_is_rejected_even_when_bytes_match(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    receipt = load_json(case.prior_receipt)
    backup = Path(receipt["backup"]["checkpoint"]["path"])
    content = backup.read_bytes()
    target = case.root / "same-prior-backup-bytes.pth"
    target.write_bytes(content)
    os.chmod(target, 0o444)
    backup.unlink()
    backup.symlink_to(target)
    with pytest.raises(migration.Stage3EMADeviceMigrationError, match="symlink"):
        migration.migrate_stage3_ema_device_provenance(**case.arguments())


def test_prior_backup_inode_and_device_are_receipt_bound(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    receipt = load_json(case.prior_receipt)
    receipt["backup"]["run_contract"]["inode"] += 1
    atomic_write_json(case.prior_receipt, receipt)
    arguments = case.arguments()
    arguments["expected_prior_migration_receipt_sha256"] = sha256_file(
        case.prior_receipt
    )
    with pytest.raises(migration.Stage3EMADeviceMigrationError, match="backup drifted"):
        migration.migrate_stage3_ema_device_provenance(**arguments)


def test_shared_primitive_script_hash_is_a_hard_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    real_sha256_file = migration.sha256_file

    def drift_shared(path: str | Path) -> str:
        if Path(path).resolve() == migration.SHARED_MIGRATION_SCRIPT_PATH.resolve():
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(migration, "sha256_file", drift_shared)
    with pytest.raises(
        migration.Stage3EMADeviceMigrationError, match="shared guard-migration"
    ):
        migration.migrate_stage3_ema_device_provenance(**case.arguments())


def test_publication_fault_rolls_both_files_back_and_preserves_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    original_replace = migration._replace_and_fsync
    calls = 0

    def fail_after_first(candidate: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second-publication failure")
        original_replace(candidate, destination)

    monkeypatch.setattr(migration, "_replace_and_fsync", fail_after_first)
    with pytest.raises(OSError, match="synthetic"):
        migration.migrate_stage3_ema_device_provenance(**case.arguments(execute=True))
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert sha256_file(case.prior_receipt) == case.prior_receipt_sha
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "ROLLED_BACK"
    assert receipt["rollback_errors"] == []
    for evidence in receipt["backup"].values():
        assert _mode(Path(evidence["path"])) == 0o444


def test_second_hardlink_failure_keeps_canonical_pair_and_clean_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    original_hardlink = migration._hardlink_backup
    calls = 0

    def fail_second(source: Path, destination: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second-hardlink failure")
        return original_hardlink(source, destination)

    monkeypatch.setattr(migration, "_hardlink_backup", fail_second)
    with pytest.raises(OSError, match="second-hardlink"):
        migration.migrate_stage3_ema_device_provenance(**case.arguments(execute=True))
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert sha256_file(case.prior_receipt) == case.prior_receipt_sha
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "ROLLED_BACK"
    assert receipt["rollback_errors"] == []
    assert set(receipt["backup"]) == {"run_contract"}


def _simulate_prepared(case: _Case, *, publication_state: str) -> dict[str, str]:
    contract = load_json(case.contract)
    checkpoint = torch.load(case.checkpoint, map_location="cpu", weights_only=False)
    new_provenance = copy.deepcopy(contract["provenance"])
    new_provenance["semantic_source_sha256"][migration.ALLOWED_SOURCE_PATH] = (
        case.new_stage3_sha
    )
    new_contract = copy.deepcopy(contract)
    new_contract["provenance"] = new_provenance
    new_checkpoint = copy.copy(checkpoint)
    new_checkpoint["provenance"] = new_provenance
    contract_candidate = case.root / "prepared-run-contract.json"
    checkpoint_candidate = case.root / "prepared-last.pth"
    atomic_write_json(contract_candidate, new_contract)
    atomic_torch_save(new_checkpoint, checkpoint_candidate)
    new_hashes = {
        "run_contract": sha256_file(contract_candidate),
        "checkpoint": sha256_file(checkpoint_candidate),
    }
    case.backup_dir.mkdir(parents=True)
    contract_backup = case.backup_dir / "old-run-contract.json"
    checkpoint_backup = case.backup_dir / "old-last.pth"
    backups = {
        "run_contract": migration._hardlink_backup(case.contract, contract_backup),
        "checkpoint": migration._hardlink_backup(case.checkpoint, checkpoint_backup),
    }
    atomic_write_json(
        case.backup_dir / "MIGRATION_RECEIPT.json",
        {
            "schema_version": migration.RECEIPT_SCHEMA,
            "protocol_id": migration.PROTOCOL_ID,
            "migration": migration.MIGRATION_KIND,
            "status": "PREPARED",
            "migration_script_sha256": sha256_file(Path(migration.__file__)),
            "orchestration_state_sha256": sha256_file(case.state),
            "old": {
                "run_contract": {
                    "path": str(case.contract.resolve()),
                    "sha256": case.contract_sha,
                },
                "checkpoint": {
                    "path": str(case.checkpoint.resolve()),
                    "sha256": case.checkpoint_sha,
                },
            },
            "new": {
                "run_contract_sha256": new_hashes["run_contract"],
                "checkpoint_sha256": new_hashes["checkpoint"],
            },
            "exact_provenance_leaf_diff": [
                {
                    "path": migration.EXPECTED_PROVENANCE_DIFF_PATH,
                    "old": case.old_stage3_sha,
                    "new": case.new_stage3_sha,
                }
            ],
            "semantic_source_count": case.source_count,
            "unchanged_semantic_source_count": case.source_count - 1,
            "preserved_stage4_source": {
                "path": migration.PRESERVED_STAGE4_SOURCE_PATH,
                "sha256": case.stage4_sha,
            },
            "approval_and_bindings_unchanged": {
                "approval_sha256": case.approval_sha,
                "approval_required_sha256": case.approval_required_sha,
                "binding_count": migration.EXPECTED_BINDING_COUNT,
                "binding_sha256": {
                    key: value["sha256"]
                    for key, value in load_json(case.approval)["bindings"].items()
                },
            },
            "prior_guard_alignment_migration": {
                "path": str(case.prior_receipt.resolve()),
                "sha256": case.prior_receipt_sha,
                "protected_unchanged": True,
            },
            "shared_guard_migration_primitives": migration._validate_shared_primitives(),
            "backup": backups,
        },
    )
    if publication_state in {"checkpoint", "both"}:
        os.replace(checkpoint_candidate, case.checkpoint)
    if publication_state in {"contract", "both"}:
        os.replace(contract_candidate, case.contract)
    contract_candidate.unlink(missing_ok=True)
    checkpoint_candidate.unlink(missing_ok=True)
    return new_hashes


@pytest.mark.parametrize(
    "publication_state", ["none", "checkpoint", "contract", "both"]
)
def test_prepared_recovery_handles_every_old_new_pair_and_is_idempotent(
    tmp_path: Path, publication_state: str
) -> None:
    case = _make_case(tmp_path)
    _simulate_prepared(case, publication_state=publication_state)
    result = migration.recover_prepared_stage3_ema_device_provenance(
        **case.recovery_arguments()
    )
    assert result["status"] == "ROLLED_BACK_FROM_PREPARED"
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert sha256_file(case.prior_receipt) == case.prior_receipt_sha
    for evidence in result["backup"].values():
        backup = Path(evidence["path"])
        assert _mode(backup) == 0o444
        destination = (
            case.contract if "run_contract" in backup.name else case.checkpoint
        )
        assert not os.path.samestat(destination.stat(), backup.stat())
    again = migration.recover_prepared_stage3_ema_device_provenance(
        **case.recovery_arguments()
    )
    assert again == result


def test_prepared_recovery_requires_its_distinct_token(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    _simulate_prepared(case, publication_state="checkpoint")
    arguments = case.recovery_arguments()
    arguments["confirmation_token"] = migration.CONFIRMATION_TOKEN
    with pytest.raises(migration.Stage3EMADeviceMigrationError, match="recovery token"):
        migration.recover_prepared_stage3_ema_device_provenance(**arguments)
    assert sha256_file(case.checkpoint) != case.checkpoint_sha


def test_production_anchor_constants_are_post_guard_values() -> None:
    assert migration.AUDITED_RUN_CONTRACT_SHA256 == (
        "156a57b5f74659c45d2123e98c3e89c02b4611136e960d1134d0d88b092084b5"
    )
    assert migration.AUDITED_CHECKPOINT_SHA256 == (
        "39bc85036a372df040774bf93d3000d0a5e36853e0e07b4648d7a01953a30d16"
    )
    assert migration.AUDITED_OLD_STAGE3_SOURCE_SHA256 == (
        "65a0812ea60dba4721e1dc4f744282ef23990ac78c32106ad8774d7dafa71a14"
    )
    assert migration.AUDITED_STAGE4_SOURCE_SHA256 == (
        "e2fbfbc2ee580b90cb92c48e6b289d6bc6d3d4651c42d34295ce07fc664814b6"
    )
    assert sha256_file(migration.SHARED_MIGRATION_SCRIPT_PATH) == (
        migration.AUDITED_SHARED_MIGRATION_SCRIPT_SHA256
    )
