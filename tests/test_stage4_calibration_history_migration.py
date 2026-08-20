from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts import migrate_stage4_step4000_calibration_history_provenance as migration
from scripts import (
    migrate_stage4_step4000_finalization_binding_source_fix as source_migration,
)
from src.training.checkpointing import atomic_torch_save
from src.training.provenance import semantic_source_hashes
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json, load_json


@dataclass(frozen=True)
class _Case:
    root: Path
    contract: Path
    last: Path
    best: Path
    validation: Path
    train: Path
    history: Path
    state: Path
    failure_log: Path
    backup_dir: Path
    hashes: dict[str, str]
    old_source_sha: str
    new_source_sha: str
    routing: dict[str, Any]
    routing_sha: str
    source_count: int
    old_contract: dict[str, Any]
    old_last: dict[str, Any]
    old_best: dict[str, Any]

    def arguments(self, *, execute: bool = False) -> dict[str, Any]:
        return {
            "project_root": self.root,
            "run_contract": self.contract,
            "last_checkpoint": self.last,
            "best_checkpoint": self.best,
            "validation_latest": self.validation,
            "train_log": self.train,
            "calibration_history": self.history,
            "state": self.state,
            "failure_log": self.failure_log,
            "backup_dir": self.backup_dir,
            "expected_run_contract_sha256": self.hashes["run_contract"],
            "expected_last_checkpoint_sha256": self.hashes["last_checkpoint"],
            "expected_best_checkpoint_sha256": self.hashes["best_checkpoint"],
            "expected_validation_latest_sha256": self.hashes["validation_latest"],
            "expected_train_log_sha256": self.hashes["train_log"],
            "expected_calibration_history_sha256": self.hashes["calibration_history"],
            "expected_state_sha256": self.hashes["state"],
            "expected_failure_log_sha256": self.hashes["failure_log"],
            "expected_old_stage4_source_sha256": self.old_source_sha,
            "expected_new_stage4_source_sha256": self.new_source_sha,
            "calibration_history_routing": self.routing,
            "expected_routing_schema": self.routing["schema_version"],
            "expected_routing_sha256": self.routing_sha,
            "expected_semantic_source_count": self.source_count,
            "expected_train_log_line_count": migration.EXPECTED_TRAIN_LOG_LINE_COUNT,
            "execute": execute,
            "confirmation_token": migration.CONFIRMATION_TOKEN if execute else None,
        }

    def recovery_arguments(self) -> dict[str, Any]:
        values = self.arguments()
        values.pop("execute")
        values["confirmation_token"] = migration.RECOVERY_CONFIRMATION_TOKEN
        return values


@dataclass(frozen=True)
class _SourceFixCase:
    base: _Case
    prior_receipt: Path
    migration_script: Path
    framework_script: Path
    backup_dir: Path
    hashes: dict[str, str]
    old_source_sha: str
    new_source_sha: str
    old_contract: dict[str, Any]
    old_last: dict[str, Any]
    old_best: dict[str, Any]

    def arguments(self, *, execute: bool = False) -> dict[str, Any]:
        return {
            "project_root": self.base.root,
            "run_contract": self.base.contract,
            "last_checkpoint": self.base.last,
            "best_checkpoint": self.base.best,
            "validation_latest": self.base.validation,
            "train_log": self.base.train,
            "calibration_history": self.base.history,
            "state": self.base.state,
            "failure_log": self.base.failure_log,
            "prior_migration_receipt": self.prior_receipt,
            "migration_script": self.migration_script,
            "transaction_framework_script": self.framework_script,
            "backup_dir": self.backup_dir,
            "expected_run_contract_sha256": self.hashes["run_contract"],
            "expected_last_checkpoint_sha256": self.hashes["last_checkpoint"],
            "expected_best_checkpoint_sha256": self.hashes["best_checkpoint"],
            "expected_validation_latest_sha256": self.hashes["validation_latest"],
            "expected_train_log_sha256": self.hashes["train_log"],
            "expected_calibration_history_sha256": self.hashes["calibration_history"],
            "expected_state_sha256": self.hashes["state"],
            "expected_failure_log_sha256": self.hashes["failure_log"],
            "expected_prior_migration_receipt_sha256": self.hashes[
                "prior_migration_receipt"
            ],
            "expected_old_provenance_sha256": self.hashes["old_provenance"],
            "expected_transaction_framework_script_sha256": self.hashes[
                "transaction_framework_script"
            ],
            "expected_old_stage4_source_sha256": self.old_source_sha,
            "expected_new_stage4_source_sha256": self.new_source_sha,
            "expected_routing_schema": self.base.routing["schema_version"],
            "expected_routing_sha256": self.base.routing_sha,
            "expected_semantic_source_count": self.base.source_count,
            "expected_train_log_line_count": migration.EXPECTED_TRAIN_LOG_LINE_COUNT,
            "execute": execute,
            "confirmation_token": (
                source_migration.CONFIRMATION_TOKEN if execute else None
            ),
        }

    def recovery_arguments(self) -> dict[str, Any]:
        values = self.arguments()
        values.pop("execute")
        values["confirmation_token"] = source_migration.RECOVERY_CONFIRMATION_TOKEN
        return values


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _write_history(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=migration.CALIBRATION_COLUMNS)
        writer.writeheader()
        for origin in ("stage0", "stage3"):
            row = {column: "" for column in migration.CALIBRATION_COLUMNS}
            row.update(
                {
                    "step": str(migration.MIGRATION_STEP),
                    "single_psnr": "25.0" if origin == "stage0" else "25.5",
                    "group_a_psnr": "22.0" if origin == "stage0" else "22.6",
                }
            )
            writer.writerow(row)


def _make_case(tmp_path: Path) -> _Case:
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "artifacts/migrations").mkdir(parents=True)
    old_source = b"old Stage4 calibration step-only routing\n"
    new_source = b"new Stage4 marker-column routing\n"
    (root / migration.ALLOWED_SOURCE_PATH).write_bytes(new_source)
    for index in range(46):
        (root / "src" / f"source_{index:02d}.py").write_text(
            f"VALUE = {index}\n", encoding="utf-8"
        )
    current = semantic_source_hashes(root, entrypoints=migration.ENTRYPOINTS)
    assert len(current) == migration.EXPECTED_SEMANTIC_SOURCE_COUNT
    old_source_sha = hashlib.sha256(old_source).hexdigest()
    old_semantic = dict(current)
    old_semantic[migration.ALLOWED_SOURCE_PATH] = old_source_sha
    provenance: dict[str, Any] = {
        "schema_version": migration.STAGE4_SCHEMA,
        "protocol_id": migration.PROTOCOL_ID,
        "semantic_source_sha256": old_semantic,
        "parents": {"stage3_checkpoint": {"sha256": "1" * 64}},
        "runtime": {"max_steps": 40_000, "crop_size": 160},
    }

    checkpoint_dir = root / "artifacts/checkpoints/stage4"
    checkpoint_dir.mkdir(parents=True)
    contract = checkpoint_dir / "run_contract.json"
    old_contract = {
        "schema_version": migration.STAGE4_SCHEMA,
        "created_utc": "2026-08-18T08:44:48Z",
        "provenance": provenance,
        "micro_batch_trials": [{"micro_batch": 2, "passed": True}],
        "validation_vram_gate": {"passed": True},
    }
    atomic_write_json(contract, old_contract)

    validation_summary = {
        "schema_version": migration.VALIDATION_SCHEMA,
        "protocol_id": migration.PROTOCOL_ID,
        "created_utc": "2026-08-18T12:08:04Z",
        "dataset": "primary_val_single_and_group_a_only",
        "image_count": 1_600,
        "single_equal_task_mean": {
            "count": 800,
            "task_count": 8,
            "psnr": 25.72357422590256,
            "ssim": 0.8427473554359649,
        },
        "group_a_equal_combination_mean": {
            "count": 800,
            "combination_count": 8,
            "psnr": 22.798064711093904,
            "ssim": 0.744995057692371,
        },
        "diagnostics": {
            "planner_macro_f1": 0.9088870656889629,
            "clean_misuse": {"psnr": 48.0},
            "wrong_skill_identity": {"psnr": 42.0},
        },
    }
    validation = checkpoint_dir / "validation_latest.json"
    atomic_write_json(validation, validation_summary)
    single = validation_summary["single_equal_task_mean"]
    group = validation_summary["group_a_equal_combination_mean"]
    best_metrics = {
        "group_a_psnr": group["psnr"],
        "group_a_ssim": group["ssim"],
        "single_psnr": single["psnr"],
        "single_ssim": single["ssim"],
        "validation_step": 4_000.0,
        "best_group_a_psnr": group["psnr"],
        "best_group_a_ssim": group["ssim"],
        "best_single_psnr": single["psnr"],
        "best_single_ssim": single["ssim"],
        "best_step": 4_000.0,
    }
    model = OrderedDict(
        {
            "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "planner.weight": torch.tensor([-0.0, 1.5], dtype=torch.float32),
        }
    )
    shadow = OrderedDict((name, value.clone()) for name, value in model.items())
    old_last: dict[str, Any] = {
        "schema_version": migration.CHECKPOINT_SCHEMA,
        "stage": "stage4",
        "step": migration.MIGRATION_STEP,
        "model": model,
        "ema": {
            "decay": 0.9999,
            "num_updates": migration.MIGRATION_STEP,
            "scope": "stage4",
            "shadow": shadow,
        },
        "optimizer": {
            "state": {
                0: {"step": torch.tensor(4_000.0), "exp_avg": torch.tensor([0.1])}
            },
            "param_groups": [{"lr": 1.0e-5, "params": [0]}],
        },
        "scheduler": {"last_epoch": 4_000, "_step_count": 4_001},
        "scaler": None,
        "rng_states": {
            "python": (3, (1, 2, 3), None),
            "numpy": ("MT19937", np.arange(8, dtype=np.uint32), 0, 0, 0.0),
            "torch_cpu": torch.arange(12, dtype=torch.uint8),
        },
        "sampler_state": {
            "consumed_optimizer_step": 4_000,
            "sample_cursor": 16_000,
        },
        "provenance": provenance,
        "metrics": {},
        "model_role": "raw_training_state",
        "resumable": True,
        "pending_validation_step": 4_000,
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "optimizer_state_name_ledger": {"0": "planner.weight"},
    }
    assert len(old_last) == migration.EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT
    old_best = copy.copy(old_last)
    old_best["model"] = OrderedDict(
        (name, value.clone()) for name, value in shadow.items()
    )
    old_best["ema"] = copy.deepcopy(old_last["ema"])
    old_best["metrics"] = best_metrics
    old_best["model_role"] = "ema_selection"
    old_best["resumable"] = False
    old_best["pending_validation_step"] = None
    last = checkpoint_dir / "last.pth"
    best = checkpoint_dir / "best_ema.pth"
    atomic_torch_save(old_last, last)
    atomic_torch_save(old_best, best)

    train = checkpoint_dir / "train.jsonl"
    with train.open("w", encoding="utf-8") as stream:
        for step in range(1, migration.MIGRATION_STEP + 1):
            stream.write(json.dumps({"step": step, "loss": 1.0 / step}) + "\n")
        stream.write(
            json.dumps(
                {"event": "pre_validation_checkpoint", "step": 4_000},
                sort_keys=True,
            )
            + "\n"
        )

    history = root / "artifacts/metrics/calibration_history.csv"
    _write_history(history)
    state = root / "artifacts/orchestration/state.json"
    state.parent.mkdir()
    atomic_write_json(
        state,
        {
            "schema_version": "graphrestore-orchestration-v1",
            "protocol_id": migration.PROTOCOL_ID,
            "status": "FAILED",
            "current_stage": "FAILED",
            "gpu": "released",
            "last_exit_code": 2,
            "last_command": [
                "/python",
                "scripts/train_stage4_e2e.py",
                "--config",
                "stage4.yaml",
            ],
            "next_command": "python scripts/orchestrate.py --resume_post_approval_pipeline --stage3_finalization_authorization auth.json",
        },
    )
    failure_log = root / "artifacts/logs/main_pipeline.log"
    failure_log.parent.mkdir()
    failure_log.write_text(
        "[2026-08-18T08:43:48Z] START stage4: /root/miniconda3/bin/python "
        "scripts/train_stage4_e2e.py --config configs/stage4_graphrestore_e2e.yaml\n"
        f"{migration.STAGE4_REFUSAL_MARKER}\n"
        "[2026-08-18T12:08:08Z] END stage4: exit=2\n",
        encoding="utf-8",
    )
    routing = {
        "schema_version": "graphrestore-stage4-calibration-ledger-v1",
        "frozen_stage3_history": {
            "path": str(history.resolve()),
            "sha256": sha256_file(history),
        },
        "stage4_history_path": str(
            (root / "artifacts/metrics/stage4_calibration_history.csv").resolve()
        ),
        "columns": list(migration.CALIBRATION_COLUMNS),
        "stage4_marker_columns": list(migration.STAGE4_MARKER_COLUMNS),
        "validation_steps": list(range(4_000, 40_001, 4_000)),
    }
    hashes = {
        "run_contract": sha256_file(contract),
        "last_checkpoint": sha256_file(last),
        "best_checkpoint": sha256_file(best),
        "validation_latest": sha256_file(validation),
        "train_log": sha256_file(train),
        "calibration_history": sha256_file(history),
        "state": sha256_file(state),
        "failure_log": sha256_file(failure_log),
    }
    return _Case(
        root=root,
        contract=contract,
        last=last,
        best=best,
        validation=validation,
        train=train,
        history=history,
        state=state,
        failure_log=failure_log,
        backup_dir=root / "artifacts/migrations" / migration.BACKUP_DIR_NAME,
        hashes=hashes,
        old_source_sha=old_source_sha,
        new_source_sha=current[migration.ALLOWED_SOURCE_PATH],
        routing=routing,
        routing_sha=sha256_json(routing),
        source_count=len(current),
        old_contract=old_contract,
        old_last=old_last,
        old_best=old_best,
    )


def _make_source_fix_case(tmp_path: Path) -> _SourceFixCase:
    base = _make_case(tmp_path)
    # Match the audited production mode at the post-first-migration boundary.
    base.history.chmod(0o600)
    first_receipt = migration.migrate_stage4_step4000_calibration_history_provenance(
        **base.arguments(execute=True)
    )
    assert first_receipt["status"] == "COMPLETE"
    prior_receipt = base.backup_dir / "MIGRATION_RECEIPT.json"

    migration_script = (
        base.root / "scripts/migrate_stage4_step4000_finalization_binding_source_fix.py"
    )
    framework_script = (
        base.root / "scripts/migrate_stage4_step4000_calibration_history_provenance.py"
    )
    shutil.copyfile(Path(source_migration.__file__).resolve(), migration_script)
    shutil.copyfile(Path(migration.__file__).resolve(), framework_script)

    old_source_sha = base.new_source_sha
    repaired_source = b"fixed nested Stage3 finalization payload binding\n"
    (base.root / source_migration.ALLOWED_SOURCE_PATH).write_bytes(repaired_source)
    new_source_sha = hashlib.sha256(repaired_source).hexdigest()
    current = semantic_source_hashes(
        base.root, entrypoints=source_migration.ENTRYPOINTS
    )
    assert current[source_migration.ALLOWED_SOURCE_PATH] == new_source_sha
    assert len(current) == base.source_count

    old_contract = load_json(base.contract)
    old_last = torch.load(base.last, map_location="cpu", weights_only=False)
    old_best = torch.load(base.best, map_location="cpu", weights_only=False)
    hashes = {
        "run_contract": sha256_file(base.contract),
        "last_checkpoint": sha256_file(base.last),
        "best_checkpoint": sha256_file(base.best),
        "validation_latest": sha256_file(base.validation),
        "train_log": sha256_file(base.train),
        "calibration_history": sha256_file(base.history),
        "state": sha256_file(base.state),
        "failure_log": sha256_file(base.failure_log),
        "prior_migration_receipt": sha256_file(prior_receipt),
        "old_provenance": sha256_json(old_contract["provenance"]),
        "transaction_framework_script": sha256_file(framework_script),
    }
    return _SourceFixCase(
        base=base,
        prior_receipt=prior_receipt,
        migration_script=migration_script,
        framework_script=framework_script,
        backup_dir=(
            base.root / "artifacts/migrations" / source_migration.BACKUP_DIR_NAME
        ),
        hashes=hashes,
        old_source_sha=old_source_sha,
        new_source_sha=new_source_sha,
        old_contract=old_contract,
        old_last=old_last,
        old_best=old_best,
    )


def test_dry_run_has_exact_two_provenance_leaves_and_no_writes(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    result = migration.migrate_stage4_step4000_calibration_history_provenance(
        **case.arguments()
    )
    assert result["status"] == "DRY_RUN"
    assert result["exact_provenance_leaf_diff"] == [
        {
            "path": migration.ROUTING_PROVENANCE_KEY,
            "old": "<absent>",
            "new": case.routing,
        },
        {
            "path": f"semantic_source_sha256.{migration.ALLOWED_SOURCE_PATH}",
            "old": case.old_source_sha,
            "new": case.new_source_sha,
        },
    ]
    assert result["checkpoint_top_level_bit_exact_outside_provenance_count"] == 16
    assert (
        result["protected_evidence"]["calibration_history"][
            "historical_non_stage4_rows_at_step4000"
        ]
        == 2
    )
    assert (
        result["protected_evidence"]["calibration_history"]["stage4_rows_at_step4000"]
        == 0
    )
    assert not case.backup_dir.exists()
    for label, path in (
        ("run_contract", case.contract),
        ("last_checkpoint", case.last),
        ("best_checkpoint", case.best),
        ("validation_latest", case.validation),
        ("train_log", case.train),
        ("calibration_history", case.history),
        ("state", case.state),
        ("failure_log", case.failure_log),
    ):
        assert sha256_file(path) == case.hashes[label]
    assert migration.os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_execute_publishes_three_way_identity_and_immutable_evidence(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    result = migration.migrate_stage4_step4000_calibration_history_provenance(
        **case.arguments(execute=True)
    )
    assert result["status"] == "COMPLETE"
    contract = load_json(case.contract)
    last = torch.load(case.last, map_location="cpu", weights_only=False)
    best = torch.load(case.best, map_location="cpu", weights_only=False)
    assert contract["provenance"] == last["provenance"] == best["provenance"]
    assert contract["provenance"][migration.ROUTING_PROVENANCE_KEY] == case.routing
    assert (
        contract["provenance"]["semantic_source_sha256"][migration.ALLOWED_SOURCE_PATH]
        == case.new_source_sha
    )
    for old, new, label in (
        (case.old_last, last, "last"),
        (case.old_best, best, "best"),
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
        "validation_latest",
        "calibration_history",
        "state",
        "failure_log",
        "train_tail",
    }
    for evidence in receipt["backup"].values():
        assert _mode(Path(evidence["archive_path"])) == 0o444
    assert Path(receipt["backup"]["failure_log"]["archive_path"]).name == (
        "main_pipeline.log"
    )
    for label, path in (
        ("validation_latest", case.validation),
        ("train_log", case.train),
        ("calibration_history", case.history),
        ("state", case.state),
        ("failure_log", case.failure_log),
    ):
        assert sha256_file(path) == case.hashes[label]


def test_wrong_execution_token_fails_before_backup(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    arguments = case.arguments(execute=True)
    arguments["confirmation_token"] = "wrong"
    with pytest.raises(migration.Stage4CalibrationRoutingMigrationError, match="exact"):
        migration.migrate_stage4_step4000_calibration_history_provenance(**arguments)
    assert not case.backup_dir.exists()


@pytest.mark.parametrize(
    ("target", "match"),
    [
        ("history", "history SHA256"),
        ("validation", "validation_latest SHA256"),
        ("train", "train log SHA256"),
        ("state", "state SHA256"),
        ("failure_log", "failure log SHA256"),
    ],
)
def test_protected_evidence_drift_is_rejected(
    tmp_path: Path, target: str, match: str
) -> None:
    case = _make_case(tmp_path)
    path = getattr(case, target)
    path.write_bytes(path.read_bytes() + b"drift\n")
    with pytest.raises(migration.Stage4CalibrationRoutingMigrationError, match=match):
        migration.migrate_stage4_step4000_calibration_history_provenance(
            **case.arguments()
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_marker",
        "marker_typo",
        "duplicate_marker",
        "trailing_line",
        "wrong_start_command",
    ),
)
def test_failure_log_requires_exact_terminal_stage4_refusal(
    tmp_path: Path, mutation: str
) -> None:
    case = _make_case(tmp_path)
    lines = case.failure_log.read_text(encoding="utf-8").splitlines()
    if mutation == "missing_marker":
        lines.pop(-2)
    elif mutation == "marker_typo":
        lines[-2] += "!"
    elif mutation == "duplicate_marker":
        lines.insert(-1, migration.STAGE4_REFUSAL_MARKER)
    elif mutation == "trailing_line":
        lines.append("unexpected trailing event")
    else:
        lines[-3] = lines[-3].replace("stage4_graphrestore_e2e.yaml", "wrong.yaml")
    case.failure_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    arguments = case.arguments()
    arguments["expected_failure_log_sha256"] = sha256_file(case.failure_log)
    with pytest.raises(
        migration.Stage4CalibrationRoutingMigrationError,
        match="exact terminal refusal transaction",
    ):
        migration.migrate_stage4_step4000_calibration_history_provenance(**arguments)
    assert not case.backup_dir.exists()
    for label, path in (
        ("run_contract", case.contract),
        ("last_checkpoint", case.last),
        ("best_checkpoint", case.best),
        ("validation_latest", case.validation),
        ("train_log", case.train),
        ("calibration_history", case.history),
        ("state", case.state),
    ):
        assert sha256_file(path) == case.hashes[label]
    assert sha256_file(case.failure_log) == arguments["expected_failure_log_sha256"]


def test_legacy_console_log_path_is_not_an_accepted_failure_binding(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    console = case.failure_log.with_name("stage3_stage4_orchestrator_console.log")
    console.write_bytes(case.failure_log.read_bytes())
    arguments = case.arguments()
    arguments["failure_log"] = console
    arguments["expected_failure_log_sha256"] = sha256_file(console)
    with pytest.raises(
        migration.Stage4CalibrationRoutingMigrationError,
        match="not the exact project-local canonical path",
    ):
        migration.migrate_stage4_step4000_calibration_history_provenance(**arguments)
    assert not case.backup_dir.exists()
    for label, path in (
        ("run_contract", case.contract),
        ("last_checkpoint", case.last),
        ("best_checkpoint", case.best),
        ("validation_latest", case.validation),
        ("train_log", case.train),
        ("calibration_history", case.history),
        ("state", case.state),
        ("failure_log", case.failure_log),
    ):
        assert sha256_file(path) == case.hashes[label]


def test_second_source_change_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    (case.root / "src/source_00.py").write_text("DRIFT = True\n", encoding="utf-8")
    with pytest.raises(
        migration.Stage4CalibrationRoutingMigrationError, match="outside the exact"
    ):
        migration.migrate_stage4_step4000_calibration_history_provenance(
            **case.arguments()
        )


def test_routing_mapping_is_exact_and_stage4_sidecar_must_be_absent(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    bad = copy.deepcopy(case.routing)
    bad["validation_steps"] = [4_000]
    arguments = case.arguments()
    arguments["calibration_history_routing"] = bad
    arguments["expected_routing_sha256"] = sha256_json(bad)
    with pytest.raises(
        migration.Stage4CalibrationRoutingMigrationError, match="mapping/schema/hash"
    ):
        migration.migrate_stage4_step4000_calibration_history_provenance(**arguments)
    sidecar = case.root / "artifacts/metrics/stage4_calibration_history.csv"
    sidecar.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(
        migration.Stage4CalibrationRoutingMigrationError, match="must be absent"
    ):
        migration.migrate_stage4_step4000_calibration_history_provenance(
            **case.arguments()
        )


def test_shared_history_rejects_extra_csv_fields(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    lines = case.history.read_text(encoding="utf-8").splitlines()
    lines[1] += ",unexpected-extra-field"
    case.history.write_text("\n".join(lines) + "\n", encoding="utf-8")
    changed_history_sha = sha256_file(case.history)
    routing = copy.deepcopy(case.routing)
    routing["frozen_stage3_history"]["sha256"] = changed_history_sha
    arguments = case.arguments()
    arguments["expected_calibration_history_sha256"] = changed_history_sha
    arguments["calibration_history_routing"] = routing
    arguments["expected_routing_sha256"] = sha256_json(routing)
    with pytest.raises(
        migration.Stage4CalibrationRoutingMigrationError, match="row width drifted"
    ):
        migration.migrate_stage4_step4000_calibration_history_provenance(**arguments)


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
        migration.migrate_stage4_step4000_calibration_history_provenance(
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


@pytest.mark.parametrize("published_mask", range(8))
def test_prepared_recovery_handles_every_old_new_triplet_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published_mask: int
) -> None:
    case = _make_case(tmp_path)
    original_publish = migration._publish_receipt
    stashed: dict[str, Path] = {}

    def crash_after_prepared(path: Path, receipt: dict[str, Any]) -> None:
        original_publish(path, receipt)
        if receipt.get("status") == "PREPARED":
            patterns = {
                "run_contract": ".run_contract.json.*.candidate.json",
                "last_checkpoint": ".last.pth.*.candidate.pth",
                "best_checkpoint": ".best_ema.pth.*.candidate.pth",
            }
            for label, pattern in patterns.items():
                matches = list(case.contract.parent.glob(pattern))
                assert len(matches) == 1
                destination = case.root / f"stashed-{label}"
                shutil.copyfile(matches[0], destination)
                stashed[label] = destination
            raise KeyboardInterrupt("synthetic abrupt stop after PREPARED")

    monkeypatch.setattr(migration, "_publish_receipt", crash_after_prepared)
    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        migration.migrate_stage4_step4000_calibration_history_provenance(
            **case.arguments(execute=True)
        )
    assert load_json(case.backup_dir / "MIGRATION_RECEIPT.json")["status"] == "PREPARED"
    monkeypatch.setattr(migration, "_publish_receipt", original_publish)

    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    destinations = {
        "run_contract": case.contract,
        "last_checkpoint": case.last,
        "best_checkpoint": case.best,
    }
    for index, label in enumerate(
        ("run_contract", "last_checkpoint", "best_checkpoint")
    ):
        if not published_mask & (1 << index):
            continue
        shutil.copyfile(stashed[label], destinations[label])
        assert sha256_file(destinations[label]) == receipt["new"][label]

    recovered = (
        migration.recover_prepared_stage4_step4000_calibration_history_provenance(
            **case.recovery_arguments()
        )
    )
    assert recovered["status"] == "ROLLED_BACK_FROM_PREPARED"
    for label, path in (
        ("run_contract", case.contract),
        ("last_checkpoint", case.last),
        ("best_checkpoint", case.best),
    ):
        assert sha256_file(path) == case.hashes[label]
    again = migration.recover_prepared_stage4_step4000_calibration_history_provenance(
        **case.recovery_arguments()
    )
    assert again == recovered


def test_directory_flock_rejects_concurrent_migration(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    descriptor = os.open(case.root / "artifacts/migrations", os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            migration.Stage4CalibrationRoutingMigrationError, match="flock"
        ):
            migration.migrate_stage4_step4000_calibration_history_provenance(
                **case.arguments()
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_symlinked_canonical_path_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    real = case.validation.with_name("validation-real.json")
    case.validation.rename(real)
    case.validation.symlink_to(real.name)
    with pytest.raises(
        migration.Stage4CalibrationRoutingMigrationError, match="symlink"
    ):
        migration.migrate_stage4_step4000_calibration_history_provenance(
            **case.arguments()
        )


def test_production_defaults_bind_the_observed_failure_boundary() -> None:
    assert migration.AUDITED_RUN_CONTRACT_SHA256 == (
        "46aca21b891b5da7194546a04a44d156d713315c06965c53b53e6334e14ca0ab"
    )
    assert migration.AUDITED_LAST_CHECKPOINT_SHA256 == (
        "22d8254d1833efd267d897ba2ddcc4addee93c0000d7dbded72df9a2000193cb"
    )
    assert migration.AUDITED_BEST_CHECKPOINT_SHA256 == (
        "5465e55b99923e55a00e2ac70f4ee61399e2e4c1ff2e4d651da0d41321529989"
    )
    assert migration.AUDITED_VALIDATION_LATEST_SHA256 == (
        "c2e560ebf2929b3c8933628b78a3591de471524394fadbac9afecbe02dc39a77"
    )
    assert migration.AUDITED_CALIBRATION_HISTORY_SHA256 == (
        "b282987c3f77034f76788a412e91823cd4570ce8c6c10cd93030ee181612e034"
    )
    assert migration.AUDITED_TRAIN_LOG_SHA256 == (
        "6cf0f60f34f1820c2626c085bfa79b2facece3c2cff2235e33760d08e83a26f3"
    )
    assert migration.AUDITED_STATE_SHA256 == (
        "ef3d144b7cdc71417bc813ebf0fb1f1a4a45656d490189ac85f55aef83bb155c"
    )
    assert migration.AUDITED_FAILURE_LOG_SHA256 == (
        "28c578ff9095f92938be52cc6c547d54afc900afb972aad5762de20b198559f3"
    )
    assert migration.AUDITED_OLD_STAGE4_SOURCE_SHA256 == (
        "6eaaef9d6a88b85d1cce7339927064f7ad70529a63f2dbaa465654c578b0629b"
    )
    assert migration.AUDITED_NEW_STAGE4_SOURCE_SHA256 == (
        "884487c1ba6b39706e92e52f748ad6aa5bbca5f4aea8fde701915c55a031b104"
    )
    assert sha256_json(migration.AUDITED_CALIBRATION_HISTORY_ROUTING) == (
        migration.AUDITED_ROUTING_SHA256
    )


def test_source_fix_dry_run_is_one_leaf_and_zero_write(tmp_path: Path) -> None:
    case = _make_source_fix_case(tmp_path)
    before = {
        label: sha256_file(path)
        for label, path in (
            ("run_contract", case.base.contract),
            ("last_checkpoint", case.base.last),
            ("best_checkpoint", case.base.best),
            ("validation_latest", case.base.validation),
            ("train_log", case.base.train),
            ("calibration_history", case.base.history),
            ("state", case.base.state),
            ("failure_log", case.base.failure_log),
            ("prior_migration_receipt", case.prior_receipt),
            ("migration_script", case.migration_script),
            ("transaction_framework_script", case.framework_script),
        )
    }
    receipt = source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
        **case.arguments()
    )
    repeated = source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
        **case.arguments()
    )
    assert receipt["status"] == "DRY_RUN"
    assert repeated["status"] == "DRY_RUN"
    assert repeated["old"] == receipt["old"]
    assert repeated["new"] == receipt["new"]
    assert (
        repeated["exact_provenance_leaf_diff"] == receipt["exact_provenance_leaf_diff"]
    )
    assert (
        repeated["calibration_history_routing_preserved"]
        == receipt["calibration_history_routing_preserved"]
    )
    assert receipt["exact_provenance_leaf_diff"] == [
        {
            "path": ("semantic_source_sha256.scripts/train_stage4_e2e.py"),
            "old": case.old_source_sha,
            "new": case.new_source_sha,
        }
    ]
    assert receipt["calibration_history_routing_preserved"] == {
        "schema_version": case.base.routing["schema_version"],
        "sha256": case.base.routing_sha,
        "mapping": case.base.routing,
        "bit_exact": True,
    }
    assert (
        receipt["prior_migration_receipt"]["sha256"]
        == case.hashes["prior_migration_receipt"]
    )
    assert receipt["stage4_calibration_sidecar_absent"] == {
        "path": str(
            (
                case.base.root / "artifacts/metrics/stage4_calibration_history.csv"
            ).resolve()
        ),
        "absent": True,
    }
    assert not case.backup_dir.exists()
    assert not (
        case.base.root / "artifacts/metrics/stage4_calibration_history.csv"
    ).exists()
    for label, path in (
        ("run_contract", case.base.contract),
        ("last_checkpoint", case.base.last),
        ("best_checkpoint", case.base.best),
        ("validation_latest", case.base.validation),
        ("train_log", case.base.train),
        ("calibration_history", case.base.history),
        ("state", case.base.state),
        ("failure_log", case.base.failure_log),
        ("prior_migration_receipt", case.prior_receipt),
        ("migration_script", case.migration_script),
        ("transaction_framework_script", case.framework_script),
    ):
        assert sha256_file(path) == before[label]


def test_source_fix_execute_preserves_routing_and_all_nonprovenance(
    tmp_path: Path,
) -> None:
    case = _make_source_fix_case(tmp_path)
    prior_receipt_sha = sha256_file(case.prior_receipt)
    receipt = source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
        **case.arguments(execute=True)
    )
    assert receipt["status"] == "COMPLETE"
    assert sha256_file(case.prior_receipt) == prior_receipt_sha

    contract = load_json(case.base.contract)
    last = torch.load(case.base.last, map_location="cpu", weights_only=False)
    best = torch.load(case.base.best, map_location="cpu", weights_only=False)
    provenance = contract["provenance"]
    assert last["provenance"] == provenance == best["provenance"]
    assert provenance[source_migration.ROUTING_PROVENANCE_KEY] == case.base.routing
    assert (
        provenance["semantic_source_sha256"][source_migration.ALLOWED_SOURCE_PATH]
        == case.new_source_sha
    )
    assert {key: value for key, value in contract.items() if key != "provenance"} == {
        key: value for key, value in case.old_contract.items() if key != "provenance"
    }
    for old, new in ((case.old_last, last), (case.old_best, best)):
        assert list(old) == list(new)
        for key in old:
            if key != "provenance":
                framework_evidence = migration._fingerprint(old[key])
                assert migration._fingerprint(new[key]) == framework_evidence

    assert set(receipt["backup"]) == {
        "run_contract",
        "last_checkpoint",
        "best_checkpoint",
        "validation_latest",
        "calibration_history",
        "state",
        "failure_log",
        "prior_migration_receipt",
        "migration_script",
        "transaction_framework_script",
        "train_tail",
    }
    for evidence in receipt["backup"].values():
        assert _mode(Path(evidence["archive_path"])) == 0o444
    assert receipt["checkpoint_top_level_bit_exact_outside_provenance_count"] == 16
    assert receipt["both_checkpoints_bit_exact_outside_provenance"] is True
    assert receipt["run_contract_bit_exact_outside_provenance"] is True


def test_source_fix_wrong_token_and_prior_receipt_drift_are_zero_write(
    tmp_path: Path,
) -> None:
    case = _make_source_fix_case(tmp_path)
    old_hashes = {
        "run_contract": sha256_file(case.base.contract),
        "last_checkpoint": sha256_file(case.base.last),
        "best_checkpoint": sha256_file(case.base.best),
    }
    arguments = case.arguments(execute=True)
    arguments["confirmation_token"] = "wrong"
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match="exact Stage4 source-fix token",
    ):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **arguments
        )
    assert not case.backup_dir.exists()

    case.prior_receipt.write_bytes(case.prior_receipt.read_bytes() + b"drift\n")
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match="receipt SHA256 drifted",
    ):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **case.arguments()
        )
    assert not case.backup_dir.exists()
    assert sha256_file(case.base.contract) == old_hashes["run_contract"]
    assert sha256_file(case.base.last) == old_hashes["last_checkpoint"]
    assert sha256_file(case.base.best) == old_hashes["best_checkpoint"]


def test_source_fix_rejects_sidecar_and_second_semantic_change(
    tmp_path: Path,
) -> None:
    sidecar_case = _make_source_fix_case(tmp_path / "sidecar")
    sidecar = (
        sidecar_case.base.root / "artifacts/metrics/stage4_calibration_history.csv"
    )
    sidecar.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match="must be absent",
    ):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **sidecar_case.arguments()
        )
    assert not sidecar_case.backup_dir.exists()

    semantic_case = _make_source_fix_case(tmp_path / "semantic")
    (semantic_case.base.root / "src/source_00.py").write_text(
        "DRIFT = True\n", encoding="utf-8"
    )
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match="outside the exact Stage4 entrypoint repair",
    ):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **semantic_case.arguments()
        )
    assert not semantic_case.backup_dir.exists()


def test_source_fix_guards_flock_paths_symlinks_and_framework_source(
    tmp_path: Path,
) -> None:
    wrong_path_case = _make_source_fix_case(tmp_path / "wrong-path")
    wrong_arguments = wrong_path_case.arguments()
    wrong_arguments["backup_dir"] = (
        wrong_path_case.base.root / "artifacts/migrations/wrong-source-fix"
    )
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match="not the exact project-local canonical path",
    ):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **wrong_arguments
        )
    assert not wrong_path_case.backup_dir.exists()

    symlink_case = _make_source_fix_case(tmp_path / "symlink")
    real_receipt = symlink_case.prior_receipt.with_name("receipt-real.json")
    symlink_case.prior_receipt.rename(real_receipt)
    symlink_case.prior_receipt.symlink_to(real_receipt.name)
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match="symlink is forbidden",
    ):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **symlink_case.arguments()
        )
    assert not symlink_case.backup_dir.exists()

    framework_case = _make_source_fix_case(tmp_path / "framework")
    framework_case.framework_script.write_bytes(
        framework_case.framework_script.read_bytes() + b"\n# drift\n"
    )
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match="implementation source binding drifted",
    ):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **framework_case.arguments()
        )
    assert not framework_case.backup_dir.exists()

    flock_case = _make_source_fix_case(tmp_path / "flock")
    descriptor = os.open(flock_case.base.root / "artifacts/migrations", os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            source_migration.Stage4FinalizationBindingSourceMigrationError,
            match="flock",
        ):
            source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
                **flock_case.arguments()
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not flock_case.backup_dir.exists()


def _leave_source_fix_prepared(
    case: _SourceFixCase, monkeypatch: pytest.MonkeyPatch
) -> Path:
    original_publish = source_migration._publish_receipt

    def crash_after_prepared(path: Path, receipt: dict[str, Any]) -> None:
        original_publish(path, receipt)
        if receipt.get("status") == "PREPARED":
            raise KeyboardInterrupt("synthetic source-fix PREPARED receipt")

    monkeypatch.setattr(source_migration, "_publish_receipt", crash_after_prepared)
    with pytest.raises(KeyboardInterrupt, match="PREPARED receipt"):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **case.arguments(execute=True)
        )
    monkeypatch.setattr(source_migration, "_publish_receipt", original_publish)
    receipt_path = case.backup_dir / "MIGRATION_RECEIPT.json"
    assert load_json(receipt_path)["status"] == "PREPARED"
    return receipt_path


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("protected_validation_sha", "exact source-only"),
        ("canonical_mode", "bound archive evidence"),
        ("canonical_path", "bound archive evidence"),
        ("canonical_sha256", "bound archive evidence"),
        ("byte_exact", "bound archive evidence"),
        ("archive_filename", "bound archive evidence"),
        ("new_hash", "exact source-only"),
        ("checkpoint_fingerprint", "exact source-only"),
        ("step", "exact PREPARED"),
        ("cpu_only", "exact PREPARED"),
        ("cuda_visible_devices", "exact PREPARED"),
        ("flock", "exact PREPARED"),
        ("created_utc", "exact PREPARED"),
        ("extra_key", "exact PREPARED"),
    ),
)
def test_source_fix_prepared_recovery_rejects_receipt_tamper_zero_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    case = _make_source_fix_case(tmp_path)
    receipt_path = _leave_source_fix_prepared(case, monkeypatch)
    receipt = load_json(receipt_path)
    if mutation == "protected_validation_sha":
        receipt["protected_evidence"]["validation_latest"]["sha256"] = "0" * 64
    elif mutation == "canonical_mode":
        receipt["backup"]["run_contract"]["canonical_mode"] = 0o777
    elif mutation == "canonical_path":
        receipt["backup"]["run_contract"]["canonical_path"] = str(case.base.best)
    elif mutation == "canonical_sha256":
        receipt["backup"]["run_contract"]["canonical_sha256"] = "0" * 64
    elif mutation == "byte_exact":
        receipt["backup"]["run_contract"]["byte_exact"] = False
    elif mutation == "archive_filename":
        receipt["backup"]["run_contract"]["archive_path"] = str(
            case.backup_dir / "wrong-old-run_contract.json"
        )
    elif mutation == "new_hash":
        receipt["new"]["run_contract"] = "0" * 64
    elif mutation == "checkpoint_fingerprint":
        receipt["checkpoint_section_fingerprints"]["last_checkpoint"]["model"]["old"][
            "sha256"
        ] = "0" * 64
    elif mutation == "step":
        receipt["step"] = 9999
    elif mutation == "cpu_only":
        receipt["cpu_only"] = False
    elif mutation == "cuda_visible_devices":
        receipt["cuda_visible_devices"] = "0"
    elif mutation == "flock":
        receipt["flock"]["inode"] += 1
    elif mutation == "created_utc":
        receipt["created_utc"] = "not-utc"
    else:
        assert mutation == "extra_key"
        receipt["unexpected"] = "tampered"
    atomic_write_json(receipt_path, receipt)

    anchors = {
        "run_contract": (case.base.contract, sha256_file(case.base.contract)),
        "last_checkpoint": (case.base.last, sha256_file(case.base.last)),
        "best_checkpoint": (case.base.best, sha256_file(case.base.best)),
    }
    modes = {label: _mode(path) for label, (path, _) in anchors.items()}
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match=match,
    ):
        source_migration.recover_prepared_stage4_step4000_finalization_binding_source_fix(
            **case.recovery_arguments()
        )
    for label, (path, digest) in anchors.items():
        assert sha256_file(path) == digest
        assert _mode(path) == modes[label] == 0o600


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("extra_key", "exact PREPARED"),
        ("recovered_utc", "finalized PREPARED"),
        ("recovered_labels", "finalized PREPARED"),
        ("recovered_value", "finalized PREPARED"),
    ),
)
def test_source_fix_finalized_prepared_recovery_rejects_receipt_tamper_zero_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    case = _make_source_fix_case(tmp_path)
    receipt_path = _leave_source_fix_prepared(case, monkeypatch)
    recovered = source_migration.recover_prepared_stage4_step4000_finalization_binding_source_fix(
        **case.recovery_arguments()
    )
    assert recovered["status"] == "ROLLED_BACK_FROM_PREPARED"
    receipt = load_json(receipt_path)
    if mutation == "extra_key":
        receipt["unexpected"] = "tampered"
    elif mutation == "recovered_utc":
        receipt["recovered_utc"] = "not-utc"
    elif mutation == "recovered_labels":
        del receipt["recovered_from_live_sha256"]["best_checkpoint"]
    else:
        assert mutation == "recovered_value"
        receipt["recovered_from_live_sha256"]["last_checkpoint"] = "0" * 64
    atomic_write_json(receipt_path, receipt)

    anchors = {
        "run_contract": (case.base.contract, sha256_file(case.base.contract)),
        "last_checkpoint": (case.base.last, sha256_file(case.base.last)),
        "best_checkpoint": (case.base.best, sha256_file(case.base.best)),
    }
    modes = {label: _mode(path) for label, (path, _) in anchors.items()}
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match=match,
    ):
        source_migration.recover_prepared_stage4_step4000_finalization_binding_source_fix(
            **case.recovery_arguments()
        )
    for label, (path, digest) in anchors.items():
        assert sha256_file(path) == digest
        assert _mode(path) == modes[label] == 0o600


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("receipt_symlink", "symlink"),
        ("receipt_mode", "receipt mode"),
        ("backup_mode", "transaction directory"),
        ("extra_entry", "entry set"),
        ("archive_hardlink", "single-link"),
        ("receipt_hardlink", "single-link"),
        ("canonical_hardlink", "single-link"),
    ),
)
def test_source_fix_prepared_recovery_rejects_transaction_directory_tamper_zero_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    case = _make_source_fix_case(tmp_path)
    receipt_path = _leave_source_fix_prepared(case, monkeypatch)
    if mutation == "receipt_symlink":
        target = case.backup_dir / "MIGRATION_RECEIPT.target.json"
        receipt_path.rename(target)
        receipt_path.symlink_to(target.name)
    elif mutation == "receipt_mode":
        receipt_path.chmod(0o644)
    elif mutation == "backup_mode":
        case.backup_dir.chmod(0o755)
    elif mutation == "extra_entry":
        (case.backup_dir / "unexpected-entry").write_bytes(b"tampered")
    elif mutation == "archive_hardlink":
        archive = case.backup_dir / "old-run_contract.json"
        archive.unlink()
        os.link(case.base.contract, archive)
    elif mutation == "receipt_hardlink":
        target = case.base.root / "receipt-hardlink-target.json"
        shutil.copyfile(receipt_path, target)
        receipt_path.unlink()
        os.link(target, receipt_path)
    else:
        assert mutation == "canonical_hardlink"
        os.link(case.base.contract, case.base.root / "canonical-hardlink-alias.json")

    anchors = {
        "run_contract": (case.base.contract, sha256_file(case.base.contract)),
        "last_checkpoint": (case.base.last, sha256_file(case.base.last)),
        "best_checkpoint": (case.base.best, sha256_file(case.base.best)),
    }
    modes = {label: _mode(path) for label, (path, _) in anchors.items()}
    with pytest.raises(
        source_migration.Stage4FinalizationBindingSourceMigrationError,
        match=match,
    ):
        source_migration.recover_prepared_stage4_step4000_finalization_binding_source_fix(
            **case.recovery_arguments()
        )
    for label, (path, digest) in anchors.items():
        assert sha256_file(path) == digest
        assert _mode(path) == modes[label] == 0o600


def test_source_fix_publication_fault_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_source_fix_case(tmp_path)
    original_replace = source_migration._replace_and_fsync
    calls = 0

    def fail_second(candidate: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic source-fix publication failure")
        original_replace(candidate, destination)

    monkeypatch.setattr(source_migration, "_replace_and_fsync", fail_second)
    with pytest.raises(OSError, match="synthetic source-fix"):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **case.arguments(execute=True)
        )
    for label, path in (
        ("run_contract", case.base.contract),
        ("last_checkpoint", case.base.last),
        ("best_checkpoint", case.base.best),
    ):
        assert sha256_file(path) == case.hashes[label]
    rolled_back = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert rolled_back["status"] == "ROLLED_BACK"
    assert rolled_back["rollback_errors"] == []


@pytest.mark.parametrize("published_mask", range(8))
def test_source_fix_prepared_recovery_all_triplets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published_mask: int
) -> None:
    case = _make_source_fix_case(tmp_path)
    original_publish = source_migration._publish_receipt
    stashed: dict[str, Path] = {}

    def crash_after_prepared(path: Path, receipt: dict[str, Any]) -> None:
        original_publish(path, receipt)
        if receipt.get("status") != "PREPARED":
            return
        patterns = {
            "run_contract": ".run_contract.json.*.candidate.json",
            "last_checkpoint": ".last.pth.*.candidate.pth",
            "best_checkpoint": ".best_ema.pth.*.candidate.pth",
        }
        for label, pattern in patterns.items():
            matches = list(case.base.contract.parent.glob(pattern))
            assert len(matches) == 1
            destination = case.base.root / f"stashed-source-fix-{label}"
            shutil.copyfile(matches[0], destination)
            stashed[label] = destination
        raise KeyboardInterrupt("synthetic abrupt stop after source-fix PREPARED")

    monkeypatch.setattr(source_migration, "_publish_receipt", crash_after_prepared)
    with pytest.raises(KeyboardInterrupt, match="source-fix PREPARED"):
        source_migration.migrate_stage4_step4000_finalization_binding_source_fix(
            **case.arguments(execute=True)
        )
    monkeypatch.setattr(source_migration, "_publish_receipt", original_publish)
    receipt = load_json(case.backup_dir / "MIGRATION_RECEIPT.json")
    assert receipt["status"] == "PREPARED"

    destinations = {
        "run_contract": case.base.contract,
        "last_checkpoint": case.base.last,
        "best_checkpoint": case.base.best,
    }
    for index, label in enumerate(
        ("run_contract", "last_checkpoint", "best_checkpoint")
    ):
        if published_mask & (1 << index):
            shutil.copyfile(stashed[label], destinations[label])
            assert sha256_file(destinations[label]) == receipt["new"][label]

    recovered = source_migration.recover_prepared_stage4_step4000_finalization_binding_source_fix(
        **case.recovery_arguments()
    )
    assert recovered["status"] == "ROLLED_BACK_FROM_PREPARED"
    for label, path in destinations.items():
        assert sha256_file(path) == case.hashes[label]
    again = source_migration.recover_prepared_stage4_step4000_finalization_binding_source_fix(
        **case.recovery_arguments()
    )
    assert again == recovered


def test_source_fix_defaults_bind_post_first_migration_boundary() -> None:
    assert source_migration.BACKUP_DIR_NAME == (
        "stage4_step4000_finalization_binding_source_fix_v1"
    )
    assert source_migration.AUDITED_RUN_CONTRACT_SHA256 == (
        "522c0f855db85af69617bbc8e2c17544be2b3485371d6adcf8e9ede9ef4624ea"
    )
    assert source_migration.AUDITED_LAST_CHECKPOINT_SHA256 == (
        "02d7f3266f9db67e65c9e96d34e7e587ed708e6af36ebf619b81158f76795f30"
    )
    assert source_migration.AUDITED_BEST_CHECKPOINT_SHA256 == (
        "a98cfb7ccc4e5472b15deeb5ece4306e0554dca92287099aef5ec699ba431384"
    )
    assert source_migration.AUDITED_PRIOR_MIGRATION_RECEIPT_SHA256 == (
        "795982a5f607c147e25a2553a63c4b24306fd0fe2753cdcd6ca0cab0af8c190d"
    )
    assert source_migration.AUDITED_OLD_PROVENANCE_SHA256 == (
        "a28650bc3cd1e5a47e0007d400a4e50e0450d131c5d40bf03d78c4d169a911fc"
    )
    assert source_migration.AUDITED_TRANSACTION_FRAMEWORK_SCRIPT_SHA256 == (
        "16998f44b5c16fa108d70f1861dedafc1ff97b738f8a298868896734debf0bcb"
    )
    assert source_migration.AUDITED_OLD_STAGE4_SOURCE_SHA256 == (
        "884487c1ba6b39706e92e52f748ad6aa5bbca5f4aea8fde701915c55a031b104"
    )
    assert source_migration.AUDITED_NEW_STAGE4_SOURCE_SHA256 == (
        "9224ee0abb62f919aed7f372e3f66395c74cb59aef0fb707ab91b7bcff43222b"
    )
    assert source_migration.AUDITED_ROUTING_SHA256 == (
        "6fa3a7f6eb6c5ad3790ed7ea2d332c9d422e3999f2adc7bfa47495830e3802a0"
    )
