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
import torch

from scripts import migrate_stage3_guard_alignment_provenance as migration
from src.training.checkpointing import atomic_torch_save
from src.training.provenance import semantic_source_hashes
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, load_json


@dataclass(frozen=True)
class _MigrationCase:
    root: Path
    contract: Path
    checkpoint: Path
    state: Path
    approval: Path
    approval_required: Path
    backup_dir: Path
    old_sources: dict[str, str]
    new_sources: dict[str, str]
    contract_sha: str
    checkpoint_sha: str
    approval_sha: str
    approval_required_sha: str
    original_payload: dict[str, Any]

    def arguments(self, *, execute: bool = False) -> dict[str, Any]:
        return {
            "project_root": self.root,
            "run_contract": self.contract,
            "checkpoint": self.checkpoint,
            "state": self.state,
            "approval": self.approval,
            "approval_required": self.approval_required,
            "backup_dir": self.backup_dir,
            "expected_run_contract_sha256": self.contract_sha,
            "expected_checkpoint_sha256": self.checkpoint_sha,
            "expected_approval_sha256": self.approval_sha,
            "expected_approval_required_sha256": self.approval_required_sha,
            "expected_old_source_sha256": self.old_sources,
            "expected_new_source_sha256": self.new_sources,
            "execute": execute,
            "confirmation_token": (migration.CONFIRMATION_TOKEN if execute else None),
        }


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _make_case(tmp_path: Path) -> _MigrationCase:
    root = tmp_path / "project"
    (root / "src/training").mkdir(parents=True)
    (root / "scripts").mkdir()
    stage3_old = b"stage3 diagnostic before alignment\n"
    stage3_new = b"stage3 diagnostic crops right bottom padding\n"
    stage4_old = b"stage4 diagnostic before alignment\n"
    stage4_new = b"stage4 diagnostic crops right bottom padding\n"
    source_content = {
        "src/training/stage3_engine.py": stage3_new,
        "src/training/stage4_engine.py": stage4_new,
        "src/training/unchanged.py": b"unchanged semantic source\n",
        "scripts/train_stage3_planner.py": b"stage3 entrypoint\n",
        "scripts/eval_guard_diagnostics.py": b"guard entrypoint\n",
    }
    for relative, content in source_content.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    current = semantic_source_hashes(root, entrypoints=migration.ENTRYPOINTS)
    old_semantic = dict(current)
    old_semantic[migration.ALLOWED_SOURCE_PATHS[0]] = _sha_bytes(stage3_old)
    old_semantic[migration.ALLOWED_SOURCE_PATHS[1]] = _sha_bytes(stage4_old)
    old_sources = {
        migration.ALLOWED_SOURCE_PATHS[0]: _sha_bytes(stage3_old),
        migration.ALLOWED_SOURCE_PATHS[1]: _sha_bytes(stage4_old),
    }
    new_sources = {path: current[path] for path in migration.ALLOWED_SOURCE_PATHS}

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
    approval_required = root / "artifacts/orchestration/STAGE3_APPROVAL_REQUIRED.json"
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
    approval = root / "artifacts/approvals/STAGE3_APPROVED.json"
    approval.parent.mkdir(parents=True)
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
            "last_exit_code": 3,
            "last_command": [
                "/python",
                "scripts/train_stage3_planner.py",
                "--config",
                "configs/stage3.yaml",
            ],
            "next_command": (
                "python scripts/orchestrate.py --resume_post_approval_pipeline"
            ),
        },
    )
    return _MigrationCase(
        root=root,
        contract=contract,
        checkpoint=checkpoint,
        state=state,
        approval=approval,
        approval_required=approval_required,
        backup_dir=root / "artifacts/migrations/stage3_guard_alignment",
        old_sources=old_sources,
        new_sources=new_sources,
        contract_sha=sha256_file(contract),
        checkpoint_sha=sha256_file(checkpoint),
        approval_sha=approval_sha,
        approval_required_sha=required_sha,
        original_payload=original_payload,
    )


def test_dry_run_proves_exact_two_leaf_plan_without_writes(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    state_sha = sha256_file(case.state)
    result = migration.migrate_stage3_guard_alignment(**case.arguments())

    assert result["status"] == "DRY_RUN"
    assert [row["path"] for row in result["exact_provenance_leaf_diff"]] == sorted(
        migration.EXPECTED_PROVENANCE_DIFF_PATHS
    )
    assert result["checkpoint_state_bit_exact_outside_provenance"] is True
    assert result["run_contract_bit_exact_outside_provenance"] is True
    assert all(
        evidence["bit_exact"]
        for key, evidence in result["checkpoint_section_fingerprints"].items()
        if key != "provenance"
    )
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert sha256_file(case.state) == state_sha
    assert not case.backup_dir.exists()


def test_execute_backs_up_then_publishes_provenance_only(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    state_sha = sha256_file(case.state)
    result = migration.migrate_stage3_guard_alignment(**case.arguments(execute=True))

    assert result["status"] == "COMPLETE"
    assert sha256_file(case.state) == state_sha
    assert sha256_file(case.approval) == case.approval_sha
    assert sha256_file(case.approval_required) == case.approval_required_sha
    assert sha256_file(case.contract) == result["new"]["run_contract_sha256"]
    assert sha256_file(case.checkpoint) == result["new"]["checkpoint_sha256"]
    contract = load_json(case.contract)
    migrated = torch.load(case.checkpoint, map_location="cpu", weights_only=False)
    assert contract["provenance"] == migrated["provenance"]
    for path in migration.ALLOWED_SOURCE_PATHS:
        assert (
            migrated["provenance"]["semantic_source_sha256"][path]
            == case.new_sources[path]
        )
    for key in case.original_payload:
        if key != "provenance":
            migration._assert_bit_exact(
                case.original_payload[key], migrated[key], path=f"checkpoint.{key}"
            )
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "COMPLETE"
    for evidence in receipt["backup"].values():
        backup = Path(evidence["path"])
        assert backup.is_file()
        assert stat_mode(backup) == 0o444
    assert receipt["backup"]["run_contract"]["sha256"] == case.contract_sha
    assert receipt["backup"]["checkpoint"]["sha256"] == case.checkpoint_sha


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


def _simulate_prepared_interruption(
    case: _MigrationCase, *, publication_state: str
) -> tuple[dict[str, str], dict[str, int]]:
    contract = load_json(case.contract)
    checkpoint = torch.load(case.checkpoint, map_location="cpu", weights_only=False)
    new_provenance = copy.deepcopy(contract["provenance"])
    for relative, source_sha in case.new_sources.items():
        new_provenance["semantic_source_sha256"][relative] = source_sha
    new_contract = copy.deepcopy(contract)
    new_contract["provenance"] = new_provenance
    new_checkpoint = copy.copy(checkpoint)
    new_checkpoint["provenance"] = new_provenance
    contract_candidate = case.root / "new_run_contract.json"
    checkpoint_candidate = case.root / "new_last.pth"
    atomic_write_json(contract_candidate, new_contract)
    atomic_torch_save(new_checkpoint, checkpoint_candidate)
    new_hashes = {
        "run_contract_sha256": sha256_file(contract_candidate),
        "checkpoint_sha256": sha256_file(checkpoint_candidate),
    }
    original_modes = {
        "run_contract": stat_mode(case.contract),
        "checkpoint": stat_mode(case.checkpoint),
    }
    case.backup_dir.mkdir(parents=True)
    contract_backup = case.backup_dir / (
        f"run_contract.pre_guard_alignment.{case.contract_sha}.json"
    )
    checkpoint_backup = case.backup_dir / (
        f"last.pre_guard_alignment.{case.checkpoint_sha}.pth"
    )
    backups = {
        "run_contract": migration._hardlink_backup(case.contract, contract_backup),
        "checkpoint": migration._hardlink_backup(case.checkpoint, checkpoint_backup),
    }
    atomic_write_json(
        case.backup_dir / "MIGRATION_RECEIPT.json",
        {
            "schema_version": migration.RECEIPT_SCHEMA,
            "protocol_id": migration.PROTOCOL_ID,
            "migration": "stage3_pending_2000_guard_alignment_provenance_only",
            "status": "PREPARED",
            "migration_script_sha256": sha256_file(Path(migration.__file__).resolve()),
            "orchestration_state_sha256": sha256_file(case.state),
            "approval_and_bindings_unchanged": {
                "approval_sha256": case.approval_sha,
                "approval_required_sha256": case.approval_required_sha,
                "binding_count": migration.EXPECTED_BINDING_COUNT,
            },
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
            "new": new_hashes,
            "exact_provenance_leaf_diff": [
                {
                    "path": f"semantic_source_sha256.{relative}",
                    "old": case.old_sources[relative],
                    "new": case.new_sources[relative],
                }
                for relative in sorted(migration.ALLOWED_SOURCE_PATHS)
            ],
            "backup": backups,
        },
    )
    if publication_state in {"mixed", "both_new"}:
        os.replace(checkpoint_candidate, case.checkpoint)
    if publication_state == "both_new":
        os.replace(contract_candidate, case.contract)
    if publication_state not in {"both_old", "mixed", "both_new"}:
        raise AssertionError(publication_state)
    contract_candidate.unlink(missing_ok=True)
    checkpoint_candidate.unlink(missing_ok=True)
    return new_hashes, original_modes


def test_refuses_any_third_semantic_source_drift(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    (case.root / "src/training/unchanged.py").write_text(
        "unexpected third source change\n", encoding="utf-8"
    )
    with pytest.raises(
        migration.Stage3GuardAlignmentMigrationError,
        match="physical semantic-source drift",
    ):
        migration.migrate_stage3_guard_alignment(**case.arguments())
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert not case.backup_dir.exists()


def test_refuses_binding_drift(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    approval = load_json(case.approval)
    first = next(iter(approval["bindings"].values()))
    Path(first["path"]).write_text("changed after approval\n", encoding="utf-8")
    with pytest.raises(
        migration.Stage3GuardAlignmentMigrationError,
        match="approved binding changed",
    ):
        migration.migrate_stage3_guard_alignment(**case.arguments())
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert not case.backup_dir.exists()


def test_refuses_execute_without_exact_confirmation(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    arguments = case.arguments(execute=True)
    arguments["confirmation_token"] = "wrong"
    with pytest.raises(
        migration.Stage3GuardAlignmentMigrationError,
        match="exact Stage3 migration confirmation token",
    ):
        migration.migrate_stage3_guard_alignment(**arguments)
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert not case.backup_dir.exists()


def test_candidate_state_mutation_is_rejected_before_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    real_save = atomic_torch_save

    def corrupt_save(payload: dict[str, Any], destination: str | Path) -> None:
        corrupted = copy.copy(payload)
        corrupted["model"] = copy.copy(payload["model"])
        corrupted["model"]["planner.weight"] = payload["model"]["planner.weight"] + 1.0
        real_save(corrupted, destination)

    monkeypatch.setattr(migration, "atomic_torch_save", corrupt_save)
    with pytest.raises(
        migration.Stage3GuardAlignmentMigrationError,
        match="checkpoint section changed outside provenance",
    ):
        migration.migrate_stage3_guard_alignment(**case.arguments())
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert not case.backup_dir.exists()


def test_partial_publication_failure_rolls_back_both_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    original_modes = {
        "run_contract": stat_mode(case.contract),
        "checkpoint": stat_mode(case.checkpoint),
    }
    real_replace = migration._replace_and_fsync
    calls = 0

    def fail_second(candidate: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic contract publication failure")
        real_replace(candidate, destination)

    monkeypatch.setattr(migration, "_replace_and_fsync", fail_second)
    with pytest.raises(OSError, match="synthetic contract publication failure"):
        migration.migrate_stage3_guard_alignment(**case.arguments(execute=True))
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "ROLLED_BACK"
    for label, live in (
        ("run_contract", case.contract),
        ("checkpoint", case.checkpoint),
    ):
        backup = Path(receipt["backup"][label]["path"])
        assert stat_mode(backup) == 0o444
        assert stat_mode(live) == original_modes[label]
        assert os.stat(live).st_ino != os.stat(backup).st_ino


def test_replace_then_raise_still_rolls_back_using_destination_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    original_modes = {
        "run_contract": stat_mode(case.contract),
        "checkpoint": stat_mode(case.checkpoint),
    }
    real_replace = migration._replace_and_fsync
    calls = 0

    def replace_then_fail(candidate: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        real_replace(candidate, destination)
        if calls == 1:
            raise OSError("synthetic failure after checkpoint replacement")

    monkeypatch.setattr(migration, "_replace_and_fsync", replace_then_fail)
    with pytest.raises(OSError, match="synthetic failure after checkpoint replacement"):
        migration.migrate_stage3_guard_alignment(**case.arguments(execute=True))
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "ROLLED_BACK"
    assert receipt["rollback_errors"] == []
    for label, live in (
        ("run_contract", case.contract),
        ("checkpoint", case.checkpoint),
    ):
        backup = Path(receipt["backup"][label]["path"])
        assert stat_mode(backup) == 0o444
        assert stat_mode(live) == original_modes[label]
        assert os.stat(live).st_ino != os.stat(backup).st_ino


def test_refuses_symlinked_migrations_parent(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    migrations = case.root / "artifacts/migrations"
    external = tmp_path / "external-migrations"
    external.mkdir()
    migrations.symlink_to(external, target_is_directory=True)

    with pytest.raises(
        migration.Stage3GuardAlignmentMigrationError,
        match="must not traverse a symlink",
    ):
        migration.migrate_stage3_guard_alignment(**case.arguments())
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha


def test_failure_after_backups_become_read_only_restores_independent_live_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    real_atomic_write_json = migration.atomic_write_json
    original_modes = {
        "run_contract": stat_mode(case.contract),
        "checkpoint": stat_mode(case.checkpoint),
    }
    failed = False

    def fail_complete_receipt(destination: str | Path, payload: Any) -> None:
        nonlocal failed
        if (
            Path(destination).name == "MIGRATION_RECEIPT.json"
            and isinstance(payload, dict)
            and payload.get("status") == "COMPLETE"
            and not failed
        ):
            failed = True
            raise OSError("synthetic final receipt failure")
        real_atomic_write_json(destination, payload)

    monkeypatch.setattr(migration, "atomic_write_json", fail_complete_receipt)
    with pytest.raises(OSError, match="synthetic final receipt failure"):
        migration.migrate_stage3_guard_alignment(**case.arguments(execute=True))

    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "ROLLED_BACK"
    for label, live in (
        ("run_contract", case.contract),
        ("checkpoint", case.checkpoint),
    ):
        backup = Path(receipt["backup"][label]["path"])
        assert stat_mode(backup) == 0o444
        assert stat_mode(live) == original_modes[label]
        assert os.stat(live).st_ino != os.stat(backup).st_ino


@pytest.mark.parametrize("publication_state", ["both_old", "mixed", "both_new"])
def test_explicit_prepared_recovery_restores_all_crash_states(
    tmp_path: Path, publication_state: str
) -> None:
    case = _make_case(tmp_path)
    state_sha = sha256_file(case.state)
    approval_sha = sha256_file(case.approval)
    required_sha = sha256_file(case.approval_required)
    _, original_modes = _simulate_prepared_interruption(
        case, publication_state=publication_state
    )

    receipt = migration.recover_prepared_stage3_guard_alignment(
        project_root=case.root,
        run_contract=case.contract,
        checkpoint=case.checkpoint,
        state=case.state,
        approval=case.approval,
        approval_required=case.approval_required,
        backup_dir=case.backup_dir,
        expected_run_contract_sha256=case.contract_sha,
        expected_checkpoint_sha256=case.checkpoint_sha,
        expected_approval_sha256=case.approval_sha,
        expected_approval_required_sha256=case.approval_required_sha,
        expected_old_source_sha256=case.old_sources,
        expected_new_source_sha256=case.new_sources,
        confirmation_token=migration.RECOVERY_CONFIRMATION_TOKEN,
    )

    assert receipt["status"] == "ROLLED_BACK_FROM_PREPARED"
    assert sha256_file(case.contract) == case.contract_sha
    assert sha256_file(case.checkpoint) == case.checkpoint_sha
    assert sha256_file(case.state) == state_sha
    assert sha256_file(case.approval) == approval_sha
    assert sha256_file(case.approval_required) == required_sha
    for label, live in (
        ("run_contract", case.contract),
        ("checkpoint", case.checkpoint),
    ):
        backup = Path(receipt["backup"][label]["path"])
        assert stat_mode(backup) == 0o444
        assert stat_mode(live) == original_modes[label]
        assert os.stat(live).st_ino != os.stat(backup).st_ino

    receipt_sha = sha256_file(case.backup_dir / "MIGRATION_RECEIPT.json")
    repeated = migration.recover_prepared_stage3_guard_alignment(
        project_root=case.root,
        run_contract=case.contract,
        checkpoint=case.checkpoint,
        state=case.state,
        approval=case.approval,
        approval_required=case.approval_required,
        backup_dir=case.backup_dir,
        expected_run_contract_sha256=case.contract_sha,
        expected_checkpoint_sha256=case.checkpoint_sha,
        expected_approval_sha256=case.approval_sha,
        expected_approval_required_sha256=case.approval_required_sha,
        expected_old_source_sha256=case.old_sources,
        expected_new_source_sha256=case.new_sources,
        confirmation_token=migration.RECOVERY_CONFIRMATION_TOKEN,
    )
    assert repeated == receipt
    assert sha256_file(case.backup_dir / "MIGRATION_RECEIPT.json") == receipt_sha


def test_prepared_recovery_retry_after_committed_receipt_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    _simulate_prepared_interruption(case, publication_state="mixed")
    real_atomic_write_json = migration.atomic_write_json
    raised = False

    def commit_then_raise(destination: str | Path, payload: Any) -> None:
        nonlocal raised
        real_atomic_write_json(destination, payload)
        if (
            Path(destination).name == "MIGRATION_RECEIPT.json"
            and isinstance(payload, dict)
            and payload.get("status") == "ROLLED_BACK_FROM_PREPARED"
            and not raised
        ):
            raised = True
            raise OSError("synthetic crash after recovery receipt commit")

    monkeypatch.setattr(migration, "atomic_write_json", commit_then_raise)
    arguments = {
        "project_root": case.root,
        "run_contract": case.contract,
        "checkpoint": case.checkpoint,
        "state": case.state,
        "approval": case.approval,
        "approval_required": case.approval_required,
        "backup_dir": case.backup_dir,
        "expected_run_contract_sha256": case.contract_sha,
        "expected_checkpoint_sha256": case.checkpoint_sha,
        "expected_approval_sha256": case.approval_sha,
        "expected_approval_required_sha256": case.approval_required_sha,
        "expected_old_source_sha256": case.old_sources,
        "expected_new_source_sha256": case.new_sources,
        "confirmation_token": migration.RECOVERY_CONFIRMATION_TOKEN,
    }
    with pytest.raises(OSError, match="synthetic crash after recovery receipt commit"):
        migration.recover_prepared_stage3_guard_alignment(**arguments)
    committed_sha = sha256_file(case.backup_dir / "MIGRATION_RECEIPT.json")
    repeated = migration.recover_prepared_stage3_guard_alignment(**arguments)
    assert repeated["status"] == "ROLLED_BACK_FROM_PREPARED"
    assert sha256_file(case.backup_dir / "MIGRATION_RECEIPT.json") == committed_sha
