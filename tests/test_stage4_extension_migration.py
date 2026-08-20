from __future__ import annotations

import copy
import csv
import hashlib
import json
import stat
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import migrate_stage4_extension_provenance as migration
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, load_json


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _source_maps() -> tuple[dict[str, str], dict[str, str]]:
    paths = [*migration.ALLOWED_CHANGED_SOURCE_PATHS]
    paths.extend(f"src/frozen/source_{index:02d}.py" for index in range(44))
    old = {path: hashlib.sha256(f"old:{path}".encode()).hexdigest() for path in paths}
    new = dict(old)
    for path in migration.ALLOWED_CHANGED_SOURCE_PATHS:
        new[path] = hashlib.sha256(f"new:{path}".encode()).hexdigest()
    return dict(sorted(old.items())), dict(sorted(new.items()))


def _checkpoint(
    provenance: dict[str, Any],
    *,
    role: str,
    best_sha256: str | None,
) -> dict[str, Any]:
    model = OrderedDict(
        [
            ("planner.weight", torch.tensor([1.25], dtype=torch.float32)),
            ("decoder.weight", torch.tensor([2.5], dtype=torch.float32)),
        ]
    )
    shadow = OrderedDict((name, tensor.clone()) for name, tensor in model.items())
    metrics: dict[str, Any] = {
        "group_a_psnr": 24.2,
        "group_a_ssim": 0.8,
        "single_psnr": 26.0,
        "single_ssim": 0.85,
        "validation_step": float(migration.BASE_STEP),
        "best_group_a_psnr": 24.2,
        "best_group_a_ssim": 0.8,
        "best_single_psnr": 26.0,
        "best_single_ssim": 0.85,
        "best_step": float(migration.BASE_STEP),
    }
    if role == "raw_training_state":
        assert best_sha256 is not None
        metrics["best_checkpoint_sha256"] = best_sha256
    return {
        "schema_version": migration.CHECKPOINT_SCHEMA,
        "stage": "stage4",
        "step": migration.BASE_STEP,
        "model": model,
        "ema": {
            "decay": 0.9999,
            "num_updates": migration.BASE_STEP,
            "shadow": shadow,
            "scope": "stage4-test",
            "policy": {"scope": "stage4-test"},
        },
        "optimizer": {
            "state": {0: {"step": torch.tensor(float(migration.BASE_STEP))}},
            "param_groups": [{"lr": migration.MIN_LR, "params": [0]}],
        },
        "scheduler": {
            "warmup_steps": 800,
            "max_steps": migration.SCHEDULE_HORIZON_STEPS,
            "min_lr": migration.MIN_LR,
            "last_epoch": migration.BASE_STEP,
            "_step_count": migration.BASE_STEP + 1,
        },
        "scaler": None,
        "rng_states": {
            "torch_cpu": torch.tensor([1, 2, 3], dtype=torch.uint8),
            "numpy": ("MT19937",),
        },
        "sampler_state": {
            "schema_version": migration.STAGE4_RUNTIME_SCHEMA,
            "stage": "stage4",
            "seed": 2027,
            "num_samples": migration.BASE_STEP * 4,
            "effective_batch_size": 4,
            "consumed_optimizer_step": migration.BASE_STEP,
            "sample_cursor": migration.BASE_STEP * 4,
        },
        "provenance": provenance,
        "metrics": metrics,
        "model_role": role,
        "resumable": role == "raw_training_state",
        "pending_validation_step": None,
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "optimizer_state_name_ledger": {"0": "planner.weight"},
    }


def _calibration_row(step: int, group_a_psnr: str) -> dict[str, str]:
    row = {column: "0" for column in migration.CALIBRATION_COLUMNS}
    row.update(
        {
            "step": str(step),
            "single_psnr": "26.0",
            "single_ssim": "0.85",
            "group_a_psnr": group_a_psnr,
            "group_a_ssim": "0.8",
            "planner_macro_f1": "0.9",
            "relation_accuracy": "0.65",
            "clean_misuse_psnr": "70.0",
            "clean_misuse_ssim": "0.99",
            "clean_misuse_residual_norm": "0.01",
            "wrong_skill_identity_psnr": "60.0",
            "wrong_skill_identity_ssim": "0.98",
            "wrong_skill_residual_norm": "0.02",
        }
    )
    return row


def _write_calibration(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=migration.CALIBRATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_train_log(path: Path, *, omit_step: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for step in range(1, migration.BASE_STEP + 1):
            if step == omit_step:
                continue
            handle.write(
                json.dumps(
                    {
                        "schema_version": migration.STAGE4_RUNTIME_SCHEMA,
                        "created_utc": "2026-08-20T00:00:00Z",
                        "step": step,
                        "samples": 4,
                        "loss": 0.1,
                        "grad_norm": 0.2,
                        "learning_rates": {"planner": migration.MIN_LR},
                        "seconds": 1.0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        handle.write(
            json.dumps(
                {
                    "schema_version": migration.STAGE4_RUNTIME_SCHEMA,
                    "event": "validation",
                    "step": migration.BASE_STEP,
                    "created_utc": "2026-08-20T01:00:00Z",
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "schema_version": migration.STAGE4_RUNTIME_SCHEMA,
                    "event": "interrupted",
                    "step": migration.BASE_STEP,
                    "created_utc": "2026-08-20T01:00:01Z",
                },
                sort_keys=True,
            )
            + "\n"
        )


def _fixture(
    tmp_path: Path,
    *,
    lhs: str = "24.20",
    rhs: str = "24.00",
    omit_train_step: int | None = None,
) -> dict[str, Any]:
    root = (tmp_path / "project").resolve()
    for directory in (
        "artifacts/checkpoints/stage4",
        "artifacts/metrics",
        "artifacts/approvals",
        "artifacts/migrations",
        "artifacts/orchestration",
        "artifacts/logs",
        "configs",
        "reports",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)

    config = root / "configs/stage4_graphrestore_e2e.yaml"
    _write(config, "training:\n  max_steps: 40000\n")
    instruction = root / "reports/STAGE4_CONDITIONAL_EXTENSION_PROTOCOL.md"
    _write(instruction, "# Immutable user-authorized Stage4 conditional extension\n")
    instruction.chmod(0o444)

    old_sources, new_sources = _source_maps()
    provenance = {
        "schema_version": migration.STAGE4_RUNTIME_SCHEMA,
        "protocol_id": migration.PROTOCOL_ID,
        "config_sha256": sha256_file(config),
        "semantic_source_sha256": old_sources,
        "runtime": {
            "crop_size": 160,
            "micro_batch": 2,
            "effective_batch_size": 4,
            "accumulation_steps": 2,
            "max_steps": migration.BASE_STEP,
            "schedule_max_steps": migration.SCHEDULE_HORIZON_STEPS,
        },
        "calibration_history_routing": {
            "schema_version": "graphrestore-stage4-calibration-ledger-v1",
            "validation_steps": list(migration.PRE_EXTENSION_VALIDATION_STEPS),
            "stage4_history_path": str(
                root / "artifacts/metrics/stage4_calibration_history.csv"
            ),
        },
        "frozen_parent_state_sha256": "f" * 64,
    }
    run_path = root / "artifacts/checkpoints/stage4/run_contract.json"
    atomic_write_json(
        run_path,
        {
            "schema_version": migration.STAGE4_RUNTIME_SCHEMA,
            "created_utc": "2026-08-18T00:00:00Z",
            "provenance": provenance,
            "approval": {"approved": True},
        },
    )
    best_path = root / "artifacts/checkpoints/stage4/best_ema.pth"
    torch.save(
        _checkpoint(provenance, role="ema_selection", best_sha256=None), best_path
    )
    last_path = root / "artifacts/checkpoints/stage4/last.pth"
    torch.save(
        _checkpoint(
            provenance,
            role="raw_training_state",
            best_sha256=sha256_file(best_path),
        ),
        last_path,
    )

    calibration = root / "artifacts/metrics/stage4_calibration_history.csv"
    prefix_rows = [
        _calibration_row(
            step,
            rhs if step == migration.TRIGGER_RHS_STEP else f"{20 + step / 10000:.2f}",
        )
        for step in migration.PRE_EXTENSION_VALIDATION_STEPS
        if step < migration.BASE_STEP
    ]
    _write_calibration(calibration, prefix_rows)
    prefix_bytes = calibration.read_bytes()
    prefix_sha = hashlib.sha256(prefix_bytes).hexdigest()
    conditional_path = root / "artifacts/approvals" / migration.CONDITIONAL_NAME
    conditional = migration.build_stage4_extension_conditional_authorization(
        project_root=root,
        config_sha256=sha256_file(config),
        run_contract_sha256=sha256_file(run_path),
        instruction_protocol_sha256=sha256_file(instruction),
        preauthorization_ledger_prefix_byte_length=len(prefix_bytes),
        preauthorization_ledger_prefix_sha256=prefix_sha,
        created_utc="2026-08-19T00:00:00Z",
    )
    atomic_write_json(conditional_path, conditional)
    conditional_path.chmod(0o444)
    _write_calibration(
        calibration,
        [*prefix_rows, _calibration_row(migration.BASE_STEP, lhs)],
    )

    latest = root / "artifacts/checkpoints/stage4/validation_latest.json"
    atomic_write_json(
        latest,
        {
            "schema_version": "graphrestore-stage4-validation-v1",
            "protocol_id": migration.PROTOCOL_ID,
            "image_count": 1600,
            "group_a_equal_combination_mean": {
                "combination_count": 8,
                "count": 800,
                "psnr": float(lhs),
                "ssim": 0.8,
            },
            "single_equal_task_mean": {
                "task_count": 8,
                "count": 800,
                "psnr": 26.0,
                "ssim": 0.85,
            },
        },
    )
    report = root / "reports/STAGE4_E2E.md"
    _write(report, "# Stage4\n\n- Validation step: 40000\n")
    train_log = root / "artifacts/checkpoints/stage4/train.jsonl"
    _write_train_log(train_log, omit_step=omit_train_step)
    state = root / "artifacts/orchestration/state.json"
    atomic_write_json(
        state,
        {
            "schema_version": "graphrestore-orchestration-v1",
            "protocol_id": migration.PROTOCOL_ID,
            "status": "FAILED",
            "current_stage": "FAILED",
            "gpu": "released",
            "last_exit_code": 130,
            "last_command": [
                "python",
                "scripts/train_stage4_e2e.py",
                "--resume",
                "last.pth",
            ],
            "next_command": "await_stage4_extension_gate",
        },
    )
    pipeline_log = root / "artifacts/logs/main_pipeline.log"
    _write(pipeline_log, "Stage4 stopped after committed step-40000 validation\n")

    hashes = {
        "conditional": sha256_file(conditional_path),
        "run_contract": sha256_file(run_path),
        "last_checkpoint": sha256_file(last_path),
        "best_checkpoint": sha256_file(best_path),
        "calibration_history": sha256_file(calibration),
        "validation_latest": sha256_file(latest),
        "report": sha256_file(report),
        "train_log": sha256_file(train_log),
        "state": sha256_file(state),
        "pipeline_log": sha256_file(pipeline_log),
        "config": sha256_file(config),
    }
    return {
        "root": root,
        "paths": migration._resolve_paths(root),
        "hashes": hashes,
        "old_sources": old_sources,
        "new_sources": new_sources,
    }


def _gate_kwargs(fixture: dict[str, Any]) -> dict[str, Any]:
    hashes = fixture["hashes"]
    return {
        "project_root": fixture["root"],
        "expected_conditional_sha256": hashes["conditional"],
        "expected_run_contract_sha256": hashes["run_contract"],
        "expected_last_checkpoint_sha256": hashes["last_checkpoint"],
        "expected_best_checkpoint_sha256": hashes["best_checkpoint"],
        "expected_calibration_history_sha256": hashes["calibration_history"],
        "expected_validation_latest_sha256": hashes["validation_latest"],
        "expected_report_sha256": hashes["report"],
        "expected_train_log_sha256": hashes["train_log"],
        "expected_state_sha256": hashes["state"],
        "expected_pipeline_log_sha256": hashes["pipeline_log"],
        "expected_config_sha256": hashes["config"],
    }


def _publish_gate(fixture: dict[str, Any]) -> dict[str, Any]:
    return migration.evaluate_stage4_extension_gate(
        **_gate_kwargs(fixture),
        execute=True,
        confirmation_token=migration.GATE_CONFIRMATION_TOKEN,
    )


def test_gate_dry_run_uses_exact_decimal_equality_and_is_non_mutating(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, lhs="24.20", rhs="24.00")
    before = {
        label: sha256_file(fixture["paths"][label])
        for label in migration.SNAPSHOT_FILENAMES
    }
    result = migration.evaluate_stage4_extension_gate(**_gate_kwargs(fixture))
    assert result["status"] == "DRY_RUN"
    assert result["decision"] == migration.DECISION_ACTIVATE
    assert result["gate_evidence"]["observed_delta_decimal"] == "0.20"
    assert not fixture["paths"]["gate"].exists()
    assert not fixture["paths"]["backup_dir"].exists()
    assert before == {
        label: sha256_file(fixture["paths"][label])
        for label in migration.SNAPSHOT_FILENAMES
    }


def test_gate_execute_writes_immutable_receipt_and_distinct_0444_snapshots(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    gate = _publish_gate(fixture)
    assert gate["decision"] == migration.DECISION_ACTIVATE
    assert gate["sha256"] == sha256_file(fixture["paths"]["gate"])
    assert stat.S_IMODE(fixture["paths"]["gate"].stat().st_mode) == 0o444
    identities: set[tuple[int, int]] = set()
    for label, evidence in gate["snapshots"].items():
        snapshot = Path(evidence["path"])
        assert snapshot.is_file() and not snapshot.is_symlink()
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o444
        hash_label = "state" if label == "orchestration_state" else label
        assert sha256_file(snapshot) == fixture["hashes"][hash_label]
        assert snapshot.stat().st_ino != fixture["paths"][label].stat().st_ino
        identities.add((snapshot.stat().st_dev, snapshot.stat().st_ino))
    assert len(identities) == len(migration.SNAPSHOT_FILENAMES)


def test_below_threshold_publishes_do_not_extend_and_migration_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, lhs="24.19", rhs="24.00")
    gate = _publish_gate(fixture)
    assert gate["decision"] == migration.DECISION_DO_NOT_EXTEND
    assert gate["observed_delta_decimal"] == "0.19"
    monkeypatch.setattr(
        migration,
        "semantic_source_hashes",
        lambda *_args, **_kwargs: dict(fixture["new_sources"]),
    )
    with pytest.raises(
        migration.Stage4ExtensionMigrationError, match="ACTIVATE_EXTENSION"
    ):
        migration.migrate_stage4_extension_provenance(
            project_root=fixture["root"],
            expected_conditional_sha256=fixture["hashes"]["conditional"],
            expected_gate_sha256=gate["sha256"],
            expected_old_source_map=fixture["old_sources"],
            expected_new_source_map=fixture["new_sources"],
        )


def test_execute_migrates_exact_three_way_provenance_and_raw_best_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    gate = _publish_gate(fixture)
    snapshots = gate["snapshots"]
    old_run = load_json(Path(snapshots["run_contract"]["path"]))
    old_last = torch.load(
        Path(snapshots["last_checkpoint"]["path"]),
        map_location="cpu",
        weights_only=False,
    )
    old_best = torch.load(
        Path(snapshots["best_checkpoint"]["path"]),
        map_location="cpu",
        weights_only=False,
    )
    monkeypatch.setattr(
        migration,
        "semantic_source_hashes",
        lambda *_args, **_kwargs: dict(fixture["new_sources"]),
    )
    dry = migration.migrate_stage4_extension_provenance(
        project_root=fixture["root"],
        expected_conditional_sha256=fixture["hashes"]["conditional"],
        expected_gate_sha256=gate["sha256"],
        expected_old_source_map=fixture["old_sources"],
        expected_new_source_map=fixture["new_sources"],
    )
    assert dry["status"] == "DRY_RUN"
    receipt = migration.migrate_stage4_extension_provenance(
        project_root=fixture["root"],
        expected_conditional_sha256=fixture["hashes"]["conditional"],
        expected_gate_sha256=gate["sha256"],
        expected_old_source_map=fixture["old_sources"],
        expected_new_source_map=fixture["new_sources"],
        execute=True,
        confirmation_token=migration.MIGRATION_CONFIRMATION_TOKEN,
    )
    assert receipt["status"] == "COMPLETE"
    run = load_json(fixture["paths"]["run_contract"])
    last = torch.load(
        fixture["paths"]["last_checkpoint"], map_location="cpu", weights_only=False
    )
    best = torch.load(
        fixture["paths"]["best_checkpoint"], map_location="cpu", weights_only=False
    )
    provenance = run["provenance"]
    assert provenance == last["provenance"] == best["provenance"]
    assert provenance["runtime"]["max_steps"] == migration.TARGET_STEP
    assert provenance["runtime"]["schedule_max_steps"] == migration.BASE_STEP
    assert provenance["calibration_history_routing"]["validation_steps"] == [
        *migration.PRE_EXTENSION_VALIDATION_STEPS,
        *migration.VALIDATION_STEPS,
    ]
    extension = provenance["stage4_extension"]
    assert set(extension) == {
        "conditional_authorization",
        "gate_receipt",
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
    }
    assert (
        extension["conditional_authorization"]["sha256"]
        == fixture["hashes"]["conditional"]
    )
    assert extension["gate_receipt"]["sha256"] == gate["sha256"]
    assert last["metrics"]["best_checkpoint_sha256"] == sha256_file(
        fixture["paths"]["best_checkpoint"]
    )
    from src.training.stage4_engine import validate_stage4_extension_authorization

    runtime_evidence = validate_stage4_extension_authorization(
        fixture["paths"]["gate"],
        project_root=fixture["root"],
        config_path=fixture["paths"]["config"],
    )
    assert runtime_evidence.provenance_binding() == extension
    migration._assert_bit_exact(
        {key: value for key, value in old_run.items() if key != "provenance"},
        {key: value for key, value in run.items() if key != "provenance"},
        path="test.run",
    )
    migration._assert_bit_exact(
        {key: value for key, value in old_best.items() if key != "provenance"},
        {key: value for key, value in best.items() if key != "provenance"},
        path="test.best",
    )
    for key in old_last:
        if key not in {"provenance", "metrics"}:
            migration._assert_bit_exact(
                old_last[key], last[key], path=f"test.last.{key}"
            )


def test_gate_rejects_noncontinuous_optimizer_log_and_prefix_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, omit_train_step=17)
    with pytest.raises(
        migration.Stage4ExtensionMigrationError, match="continuous 1..40000"
    ):
        migration.evaluate_stage4_extension_gate(**_gate_kwargs(fixture))

    second = _fixture(tmp_path / "second")
    ledger = second["paths"]["calibration_history"]
    payload = bytearray(ledger.read_bytes())
    payload[0] ^= 1
    ledger.write_bytes(payload)
    second["hashes"]["calibration_history"] = sha256_file(ledger)
    with pytest.raises(
        migration.Stage4ExtensionMigrationError,
        match="preauthorization prefix drifted",
    ):
        migration.evaluate_stage4_extension_gate(**_gate_kwargs(second))


def test_tokens_and_changed_source_set_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(
        migration.Stage4ExtensionMigrationError, match="confirmation token"
    ):
        migration.evaluate_stage4_extension_gate(
            **_gate_kwargs(fixture), execute=True, confirmation_token="wrong"
        )
    gate = _publish_gate(fixture)
    monkeypatch.setattr(
        migration,
        "semantic_source_hashes",
        lambda *_args, **_kwargs: dict(fixture["new_sources"]),
    )
    widened = copy.deepcopy(fixture["new_sources"])
    frozen = next(
        path for path in widened if path not in migration.ALLOWED_CHANGED_SOURCE_PATHS
    )
    widened[frozen] = "e" * 64
    with pytest.raises(
        migration.Stage4ExtensionMigrationError, match="changed-path set"
    ):
        migration.migrate_stage4_extension_provenance(
            project_root=fixture["root"],
            expected_conditional_sha256=fixture["hashes"]["conditional"],
            expected_gate_sha256=gate["sha256"],
            expected_old_source_map=fixture["old_sources"],
            expected_new_source_map=widened,
        )
    with pytest.raises(
        migration.Stage4ExtensionMigrationError, match="confirmation token"
    ):
        migration.migrate_stage4_extension_provenance(
            project_root=fixture["root"],
            expected_conditional_sha256=fixture["hashes"]["conditional"],
            expected_gate_sha256=gate["sha256"],
            expected_old_source_map=fixture["old_sources"],
            expected_new_source_map=fixture["new_sources"],
            execute=True,
            confirmation_token="wrong",
        )
