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

from scripts import migrate_stage3_calibration_padding_provenance as migration
from src.training.checkpointing import atomic_torch_save
from src.training.provenance import semantic_source_hashes
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, load_json


@dataclass(frozen=True)
class _Case:
    root: Path
    contract: Path
    last: Path
    best: Path
    state: Path
    approval: Path
    approval_required: Path
    guard_receipt: Path
    ema_receipt: Path
    backup_dir: Path
    hashes: dict[str, str]
    old_stage3_sha: str
    new_stage3_sha: str
    stage4_sha: str
    source_count: int
    original_contract: dict[str, Any]
    original_last: dict[str, Any]
    original_best: dict[str, Any]

    def arguments(self, *, execute: bool = False) -> dict[str, Any]:
        return {
            "project_root": self.root,
            "run_contract": self.contract,
            "last_checkpoint": self.last,
            "best_checkpoint": self.best,
            "state": self.state,
            "approval": self.approval,
            "approval_required": self.approval_required,
            "guard_migration_receipt": self.guard_receipt,
            "ema_migration_receipt": self.ema_receipt,
            "backup_dir": self.backup_dir,
            "expected_run_contract_sha256": self.hashes["run_contract"],
            "expected_last_checkpoint_sha256": self.hashes["last_checkpoint"],
            "expected_best_checkpoint_sha256": self.hashes["best_checkpoint"],
            "expected_state_sha256": self.hashes["state"],
            "expected_approval_sha256": self.hashes["approval"],
            "expected_approval_required_sha256": self.hashes["approval_required"],
            "expected_guard_migration_receipt_sha256": self.hashes["guard_receipt"],
            "expected_ema_migration_receipt_sha256": self.hashes["ema_receipt"],
            "expected_old_stage3_source_sha256": self.old_stage3_sha,
            "expected_new_stage3_source_sha256": self.new_stage3_sha,
            "expected_unchanged_stage4_source_sha256": self.stage4_sha,
            "expected_semantic_source_count": self.source_count,
            "execute": execute,
            "confirmation_token": migration.CONFIRMATION_TOKEN if execute else None,
        }

    def recovery_arguments(self) -> dict[str, Any]:
        values = self.arguments()
        values.pop("execute")
        values["confirmation_token"] = migration.RECOVERY_CONFIRMATION_TOKEN
        return values


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _make_read_only_backups(directory: Path, labels: tuple[str, ...]) -> dict[str, Any]:
    directory.mkdir(parents=True)
    result: dict[str, Any] = {}
    for label in labels:
        path = directory / f"prior-{label}.bin"
        path.write_bytes(f"immutable prior {label}\n".encode())
        os.chmod(path, 0o444)
        result[label] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "mode": 0o600,
            "device": path.stat().st_dev,
            "inode": path.stat().st_ino,
            "hard_link_verified": True,
        }
    return result


def _make_prior_receipts(
    root: Path, old_stage3_sha: str, stage4_sha: str
) -> tuple[Path, Path]:
    guard_dir = root / "artifacts/migrations" / migration.GUARD_BACKUP_DIR_NAME
    guard_backups = _make_read_only_backups(guard_dir, ("run_contract", "checkpoint"))
    guard_receipt = guard_dir / "MIGRATION_RECEIPT.json"
    atomic_write_json(
        guard_receipt,
        {
            "schema_version": migration.GUARD_RECEIPT_SCHEMA,
            "protocol_id": migration.PROTOCOL_ID,
            "migration": migration.GUARD_MIGRATION_KIND,
            "status": "COMPLETE",
            "backup_read_only_after_publication": True,
            "backup": guard_backups,
            "exact_provenance_leaf_diff": [
                {
                    "path": f"semantic_source_sha256.{migration.ALLOWED_SOURCE_PATH}",
                    "old": "1" * 64,
                    "new": "2" * 64,
                },
                {
                    "path": (
                        "semantic_source_sha256."
                        f"{migration.PRESERVED_STAGE4_SOURCE_PATH}"
                    ),
                    "old": "3" * 64,
                    "new": stage4_sha,
                },
            ],
        },
    )
    guard_sha = sha256_file(guard_receipt)

    ema_dir = root / "artifacts/migrations" / migration.EMA_BACKUP_DIR_NAME
    ema_backups = _make_read_only_backups(ema_dir, ("run_contract", "checkpoint"))
    ema_receipt = ema_dir / "MIGRATION_RECEIPT.json"
    atomic_write_json(
        ema_receipt,
        {
            "schema_version": migration.EMA_RECEIPT_SCHEMA,
            "protocol_id": migration.PROTOCOL_ID,
            "migration": migration.EMA_MIGRATION_KIND,
            "status": "COMPLETE",
            "migration_script_sha256": (
                migration.AUDITED_PRIOR_MIGRATION_SCRIPT_SHA256
            ),
            "backup_read_only_after_publication": True,
            "prior_guard_alignment_receipt_unchanged_after_publication": True,
            "backup": ema_backups,
            "exact_provenance_leaf_diff": [
                {
                    "path": migration.EXPECTED_PROVENANCE_DIFF_PATH,
                    "old": migration.prior.AUDITED_OLD_STAGE3_SOURCE_SHA256,
                    "new": old_stage3_sha,
                }
            ],
            "prior_guard_alignment_migration": {
                "path": str(guard_receipt.resolve()),
                "sha256": guard_sha,
                "status": "COMPLETE",
                "protected_unchanged": True,
            },
            "preserved_stage4_source": {
                "path": migration.PRESERVED_STAGE4_SOURCE_PATH,
                "sha256": stage4_sha,
            },
        },
    )
    return guard_receipt, ema_receipt


def _make_case(tmp_path: Path) -> _Case:
    root = tmp_path / "project"
    (root / "src/training").mkdir(parents=True)
    (root / "scripts").mkdir()
    old_stage3 = b"stage3 exact ema-device implementation\n"
    new_stage3 = b"stage3 calibration input padding implementation\n"
    stage4 = b"stage4 unchanged implementation\n"
    source_content = {
        migration.ALLOWED_SOURCE_PATH: new_stage3,
        migration.PRESERVED_STAGE4_SOURCE_PATH: stage4,
        "src/training/unchanged.py": b"unchanged source\n",
        "scripts/train_stage3_planner.py": b"stage3 entrypoint\n",
        "scripts/eval_guard_diagnostics.py": b"guard entrypoint\n",
    }
    for index in range(42):
        source_content[f"src/dummy_{index:02d}.py"] = f"DUMMY = {index}\n".encode()
    for relative, content in source_content.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    current = semantic_source_hashes(root, entrypoints=migration.ENTRYPOINTS)
    assert len(current) == migration.EXPECTED_SEMANTIC_SOURCE_COUNT
    old_stage3_sha = _sha_bytes(old_stage3)
    old_semantic = dict(current)
    old_semantic[migration.ALLOWED_SOURCE_PATH] = old_stage3_sha

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
    original_contract = {
        "schema_version": migration.STAGE3_SCHEMA,
        "created_utc": "2026-08-17T09:30:11Z",
        "provenance": provenance,
        "parent_load": {"loaded_count": 1535},
        "micro_batch_trials": [{"micro_batch": 4, "passed": True}],
        "validation_vram_gate": {"passed": True, "peak": 0.5},
    }
    atomic_write_json(contract, original_contract)

    model = OrderedDict(
        {
            "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "planner.weight": torch.tensor([-0.0, 1.5], dtype=torch.float32),
        }
    )
    ema_shadow = OrderedDict((name, value.clone()) for name, value in model.items())
    original_last: dict[str, Any] = {
        "schema_version": migration.CHECKPOINT_SCHEMA,
        "stage": "stage3",
        "step": migration.MIGRATION_STEP,
        "model": model,
        "ema": {
            "decay": 0.9999,
            "num_updates": migration.MIGRATION_STEP,
            "scope": "planner_parameters_only_executor_bitwise_frozen",
            "shadow": ema_shadow,
        },
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(12_000.0),
                    "exp_avg": torch.tensor([-0.0, 0.25]),
                    "exp_avg_sq": torch.tensor([0.5, 1.0]),
                }
            },
            "param_groups": [{"lr": 1.0e-6, "params": [0]}],
        },
        "scheduler": {"last_epoch": 12_000, "_step_count": 12_001},
        "scaler": None,
        "rng_states": {
            "python": (3, (1, 2, 3), None),
            "numpy": ("MT19937", np.arange(8, dtype=np.uint32), 0, 0, 0.0),
            "torch_cpu": torch.arange(12, dtype=torch.uint8),
            "torch_cuda_all": [torch.arange(8, dtype=torch.uint8)],
        },
        "sampler_state": {
            "consumed_optimizer_step": 12_000,
            "sample_cursor": 96_000,
        },
        "provenance": provenance,
        "metrics": {
            "validation_step": 12_000,
            "best_step": 12_000,
            "group_a_psnr": 22.027,
            "group_a_ssim": 0.711,
        },
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
        "model_role": "raw_training_state",
        "resumable": True,
        "pending_validation_step": None,
        "optimizer_transaction_active": False,
        "optimizer_state_name_ledger": {"0": "planner.weight"},
    }
    assert len(original_last) == migration.EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT
    original_best = copy.copy(original_last)
    original_best["model"] = OrderedDict(
        (name, value.clone()) for name, value in ema_shadow.items()
    )
    original_best["ema"] = copy.deepcopy(original_last["ema"])
    original_best["model_role"] = "ema_selection"
    original_best["resumable"] = False
    last = contract.parent / "last.pth"
    best = contract.parent / "best_ema.pth"
    atomic_torch_save(original_last, last)
    atomic_torch_save(original_best, best)

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
                str(last),
            ],
            "next_command": (
                "python scripts/orchestrate.py --resume_post_approval_pipeline"
            ),
        },
    )
    guard_receipt, ema_receipt = _make_prior_receipts(
        root, old_stage3_sha, current[migration.PRESERVED_STAGE4_SOURCE_PATH]
    )
    hashes = {
        "run_contract": sha256_file(contract),
        "last_checkpoint": sha256_file(last),
        "best_checkpoint": sha256_file(best),
        "state": sha256_file(state),
        "approval": approval_sha,
        "approval_required": required_sha,
        "guard_receipt": sha256_file(guard_receipt),
        "ema_receipt": sha256_file(ema_receipt),
    }
    return _Case(
        root=root,
        contract=contract,
        last=last,
        best=best,
        state=state,
        approval=approval,
        approval_required=approval_required,
        guard_receipt=guard_receipt,
        ema_receipt=ema_receipt,
        backup_dir=(root / "artifacts/migrations" / migration.BACKUP_DIR_NAME),
        hashes=hashes,
        old_stage3_sha=old_stage3_sha,
        new_stage3_sha=current[migration.ALLOWED_SOURCE_PATH],
        stage4_sha=current[migration.PRESERVED_STAGE4_SOURCE_PATH],
        source_count=len(current),
        original_contract=original_contract,
        original_last=original_last,
        original_best=original_best,
    )


def test_dry_run_is_one_leaf_three_artifacts_and_writes_nothing(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    result = migration.migrate_stage3_calibration_padding_provenance(**case.arguments())
    assert result["status"] == "DRY_RUN"
    assert result["exact_provenance_leaf_diff"] == [
        {
            "path": migration.EXPECTED_PROVENANCE_DIFF_PATH,
            "old": case.old_stage3_sha,
            "new": case.new_stage3_sha,
        }
    ]
    assert result["checkpoint_top_level_count"] == 20
    assert result["checkpoint_top_level_bit_exact_outside_provenance_count"] == 19
    assert set(result["new"]) == {
        "run_contract",
        "last_checkpoint",
        "best_checkpoint",
        "provenance_json_sha256",
    }
    for label, path in (
        ("run_contract", case.contract),
        ("last_checkpoint", case.last),
        ("best_checkpoint", case.best),
        ("state", case.state),
        ("guard_receipt", case.guard_receipt),
        ("ema_receipt", case.ema_receipt),
    ):
        assert sha256_file(path) == case.hashes[label]
    assert not case.backup_dir.exists()
    assert migration.os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_execute_publishes_three_way_identity_and_read_only_backups(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    result = migration.migrate_stage3_calibration_padding_provenance(
        **case.arguments(execute=True)
    )
    assert result["status"] == "COMPLETE"
    contract = load_json(case.contract)
    last = torch.load(case.last, map_location="cpu", weights_only=False)
    best = torch.load(case.best, map_location="cpu", weights_only=False)
    assert contract["provenance"] == last["provenance"] == best["provenance"]
    semantic = contract["provenance"]["semantic_source_sha256"]
    assert semantic[migration.ALLOWED_SOURCE_PATH] == case.new_stage3_sha
    assert semantic[migration.PRESERVED_STAGE4_SOURCE_PATH] == case.stage4_sha
    for old, new, label in (
        (case.original_last, last, "last"),
        (case.original_best, best, "best"),
    ):
        for key in old:
            if key != "provenance":
                migration._assert_bit_exact(old[key], new[key], path=f"{label}.{key}")
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "COMPLETE"
    assert set(receipt["backup"]) == {
        "run_contract",
        "last_checkpoint",
        "best_checkpoint",
    }
    assert receipt["both_prior_receipts_unchanged_after_publication"] is True
    for evidence in receipt["backup"].values():
        assert _mode(Path(evidence["path"])) == 0o444
    for label, path in (
        ("state", case.state),
        ("guard_receipt", case.guard_receipt),
        ("ema_receipt", case.ema_receipt),
    ):
        assert sha256_file(path) == case.hashes[label]


def test_wrong_execution_token_fails_before_writes(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    arguments = case.arguments(execute=True)
    arguments["confirmation_token"] = "wrong"
    with pytest.raises(migration.Stage3CalibrationPaddingMigrationError, match="exact"):
        migration.migrate_stage3_calibration_padding_provenance(**arguments)
    assert not case.backup_dir.exists()


@pytest.mark.parametrize("drift", ["stage4", "other_source"])
def test_any_second_semantic_source_change_is_rejected(
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
        migration.Stage3CalibrationPaddingMigrationError,
        match="exactly Stage3",
    ):
        migration.migrate_stage3_calibration_padding_provenance(**case.arguments())


@pytest.mark.parametrize("drift", ["state", "binding", "guard_backup", "ema_backup"])
def test_failed_state_approval_and_two_prior_receipts_are_closed(
    tmp_path: Path, drift: str
) -> None:
    case = _make_case(tmp_path)
    if drift == "state":
        state = load_json(case.state)
        state["last_exit_code"] = 3
        atomic_write_json(case.state, state)
        match = "state SHA256"
    elif drift == "binding":
        approval = load_json(case.approval)
        bound = Path(next(iter(approval["bindings"].values()))["path"])
        bound.write_text("drift\n", encoding="utf-8")
        match = "binding changed"
    else:
        receipt_path = (
            case.guard_receipt if drift == "guard_backup" else case.ema_receipt
        )
        receipt = load_json(receipt_path)
        backup = Path(receipt["backup"]["checkpoint"]["path"])
        os.chmod(backup, 0o644)
        match = "backup drifted"
    with pytest.raises(migration.Stage3CalibrationPaddingMigrationError, match=match):
        migration.migrate_stage3_calibration_padding_provenance(**case.arguments())


def test_publication_fault_rolls_all_three_files_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    original_replace = migration._replace_and_fsync
    calls = 0

    def fail_second(candidate: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second-publication failure")
        original_replace(candidate, destination)

    monkeypatch.setattr(migration, "_replace_and_fsync", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        migration.migrate_stage3_calibration_padding_provenance(
            **case.arguments(execute=True)
        )
    for label, path in (
        ("run_contract", case.contract),
        ("last_checkpoint", case.last),
        ("best_checkpoint", case.best),
    ):
        assert sha256_file(path) == case.hashes[label]
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "ROLLED_BACK"
    assert receipt["rollback_errors"] == []
    for evidence in receipt["backup"].values():
        assert _mode(Path(evidence["path"])) == 0o444


def test_third_hardlink_fault_keeps_canonical_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    original_hardlink = migration._hardlink_backup
    calls = 0

    def fail_third(source: Path, destination: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic third-hardlink failure")
        return original_hardlink(source, destination)

    monkeypatch.setattr(migration, "_hardlink_backup", fail_third)
    with pytest.raises(OSError, match="third-hardlink"):
        migration.migrate_stage3_calibration_padding_provenance(
            **case.arguments(execute=True)
        )
    assert sha256_file(case.contract) == case.hashes["run_contract"]
    assert sha256_file(case.last) == case.hashes["last_checkpoint"]
    assert sha256_file(case.best) == case.hashes["best_checkpoint"]
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "ROLLED_BACK"
    assert set(receipt["backup"]) == {"run_contract", "last_checkpoint"}


def _simulate_prepared(case: _Case, *, published_mask: int) -> None:
    dry = migration.migrate_stage3_calibration_padding_provenance(**case.arguments())
    new_provenance = copy.deepcopy(case.original_contract["provenance"])
    new_provenance["semantic_source_sha256"][migration.ALLOWED_SOURCE_PATH] = (
        case.new_stage3_sha
    )
    values = {
        "run_contract": copy.deepcopy(case.original_contract),
        "last_checkpoint": copy.copy(case.original_last),
        "best_checkpoint": copy.copy(case.original_best),
    }
    values["run_contract"]["provenance"] = new_provenance
    values["last_checkpoint"]["provenance"] = new_provenance
    values["best_checkpoint"]["provenance"] = new_provenance
    candidates = {
        "run_contract": case.root / "new-run-contract.json",
        "last_checkpoint": case.root / "new-last.pth",
        "best_checkpoint": case.root / "new-best.pth",
    }
    atomic_write_json(candidates["run_contract"], values["run_contract"])
    atomic_torch_save(values["last_checkpoint"], candidates["last_checkpoint"])
    atomic_torch_save(values["best_checkpoint"], candidates["best_checkpoint"])
    new_hashes = {label: sha256_file(path) for label, path in candidates.items()}

    case.backup_dir.mkdir(parents=True)
    destinations = {
        "run_contract": case.contract,
        "last_checkpoint": case.last,
        "best_checkpoint": case.best,
    }
    backups = {
        label: migration._hardlink_backup(
            destination, case.backup_dir / f"old-{label}.bin"
        )
        for label, destination in destinations.items()
    }
    receipt = dict(dry)
    receipt["status"] = "PREPARED"
    receipt["execution_confirmation_token_sha256"] = hashlib.sha256(
        migration.CONFIRMATION_TOKEN.encode()
    ).hexdigest()
    receipt["new"] = new_hashes | {
        "provenance_json_sha256": dry["new"]["provenance_json_sha256"]
    }
    receipt["backup"] = backups
    atomic_write_json(case.backup_dir / "MIGRATION_RECEIPT.json", receipt)
    for index, label in enumerate(
        ("run_contract", "last_checkpoint", "best_checkpoint")
    ):
        if published_mask & (1 << index):
            os.replace(candidates[label], destinations[label])
    for path in candidates.values():
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("published_mask", range(8))
def test_prepared_recovery_handles_all_old_new_triplets_and_is_idempotent(
    tmp_path: Path, published_mask: int
) -> None:
    case = _make_case(tmp_path)
    _simulate_prepared(case, published_mask=published_mask)
    result = migration.recover_prepared_stage3_calibration_padding_provenance(
        **case.recovery_arguments()
    )
    assert result["status"] == "ROLLED_BACK_FROM_PREPARED"
    for label, path in (
        ("run_contract", case.contract),
        ("last_checkpoint", case.last),
        ("best_checkpoint", case.best),
    ):
        assert sha256_file(path) == case.hashes[label]
        backup = Path(result["backup"][label]["path"])
        assert _mode(backup) == 0o444
        assert not os.path.samestat(path.stat(), backup.stat())
    again = migration.recover_prepared_stage3_calibration_padding_provenance(
        **case.recovery_arguments()
    )
    assert again == result


def test_prepared_recovery_requires_distinct_token(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    _simulate_prepared(case, published_mask=0b010)
    arguments = case.recovery_arguments()
    arguments["confirmation_token"] = migration.CONFIRMATION_TOKEN
    with pytest.raises(
        migration.Stage3CalibrationPaddingMigrationError, match="recovery token"
    ):
        migration.recover_prepared_stage3_calibration_padding_provenance(**arguments)
    assert sha256_file(case.last) != case.hashes["last_checkpoint"]


def test_production_defaults_bind_exact_calibration_boundary() -> None:
    assert migration.AUDITED_RUN_CONTRACT_SHA256 == (
        "d98b7493b41a0ace9fcb228c50b3acbdf855f092bb2ddc9c9f479730cecf053f"
    )
    assert migration.AUDITED_LAST_CHECKPOINT_SHA256 == (
        "39733371064c282e46e858aaf50df7b0d4a9fdf3c49c5bc8838798b4958e2438"
    )
    assert migration.AUDITED_BEST_CHECKPOINT_SHA256 == (
        "b26ebca987fae140bbaff8a7b530692f7a4e0113bdeea863547b6aaec8958b20"
    )
    assert migration.AUDITED_OLD_STAGE3_SOURCE_SHA256 == (
        "908bcd7ff829aabba8376ec949156890983f51924aaa7e2313e013648d817b49"
    )
    assert migration.AUDITED_NEW_STAGE3_SOURCE_SHA256 == (
        "2ba4c211476b2aa8a374e608000660dc024966c4094d79aeff8adc506431f796"
    )
    assert migration.AUDITED_STAGE4_SOURCE_SHA256 == (
        "e2fbfbc2ee580b90cb92c48e6b289d6bc6d3d4651c42d34295ce07fc664814b6"
    )
    assert sha256_file(migration.PRIOR_MIGRATION_SCRIPT_PATH) == (
        migration.AUDITED_PRIOR_MIGRATION_SCRIPT_SHA256
    )
