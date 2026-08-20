from __future__ import annotations

import copy
import fcntl
import itertools
import os
import shutil
import stat
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import migrate_stage3_extension_provenance as migration
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, load_json


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _prior_receipt(
    root: Path,
    *,
    directory: str,
    schema: str,
    kind: str,
) -> Path:
    parent = root / "artifacts/migrations" / directory
    parent.mkdir(parents=True)
    backups: dict[str, Any] = {}
    for label in ("run_contract", "checkpoint"):
        path = parent / f"{label}.backup"
        path.write_bytes(f"{directory}:{label}".encode())
        path.chmod(0o444)
        info = path.stat()
        backups[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "mode": 0o444,
            "device": info.st_dev,
            "inode": info.st_ino,
            "hard_link_verified": True,
        }
    receipt = parent / migration.RECEIPT_NAME
    atomic_write_json(
        receipt,
        {
            "schema_version": schema,
            "protocol_id": migration.PROTOCOL_ID,
            "migration": kind,
            "status": "COMPLETE",
            "backup_read_only_after_publication": True,
            "backup": backups,
        },
    )
    return receipt


def _checkpoint(
    provenance: dict[str, Any],
    *,
    role: str,
    model_value: float,
    ema_value: float,
) -> dict[str, Any]:
    model = OrderedDict(
        [("planner.weight", torch.tensor([model_value], dtype=torch.float32))]
    )
    shadow = {"planner.weight": torch.tensor([ema_value], dtype=torch.float32)}
    return {
        "schema_version": migration.CHECKPOINT_SCHEMA,
        "stage": "stage3",
        "step": migration.BASE_STEP,
        "model": model,
        "ema": {
            "shadow": shadow,
            "num_updates": migration.BASE_STEP,
            "scope": "planner_parameters_only_executor_bitwise_frozen",
            "decay": 0.999,
        },
        "optimizer": {
            "state": {0: {"step": torch.tensor(12000.0)}},
            "param_groups": [{"lr": migration.MIN_LR, "params": [0]}],
        },
        "scheduler": {
            "max_steps": migration.SCHEDULE_HORIZON_STEPS,
            "last_epoch": migration.BASE_STEP,
            "min_lr": migration.MIN_LR,
        },
        "scaler": None,
        "rng_states": {
            "torch": torch.tensor([1, 2, 3], dtype=torch.uint8),
            "numpy": ("MT19937",),
        },
        "sampler_state": {
            "consumed_optimizer_step": migration.BASE_STEP,
            "sample_cursor": migration.BASE_STEP * 8,
        },
        "provenance": provenance,
        "metrics": {
            "validation_step": migration.BASE_STEP,
            "best_step": migration.BASE_STEP,
            "group_a_psnr": 22.027225,
        },
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
        "model_role": role,
        "resumable": role == "raw_training_state",
        "pending_validation_step": None,
        "optimizer_transaction_active": False,
        "optimizer_state_name_ledger": {"0": "planner.weight"},
    }


def _source_maps() -> tuple[dict[str, str], dict[str, str]]:
    return (
        dict(migration.AUDITED_OLD_SEMANTIC_SOURCE_SHA256),
        dict(migration.AUDITED_NEW_SEMANTIC_SOURCE_SHA256),
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = (tmp_path / "project").resolve()
    (root / "artifacts/checkpoints/stage3").mkdir(parents=True)
    (root / "artifacts/approvals").mkdir(parents=True)
    (root / "artifacts/orchestration").mkdir(parents=True)
    (root / "artifacts/migrations").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)

    config = root / "configs/stage3_planner.yaml"
    _write(config, "training:\n  max_steps: 12000\n")
    bindings: dict[str, dict[str, str]] = {}
    for index in range(21):
        path = root / "bound" / f"artifact_{index:02d}.bin"
        _write(path, f"artifact-{index}")
        bindings[f"artifact_{index:02d}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    bindings["config_stage3"] = {
        "path": str(config),
        "sha256": sha256_file(config),
    }
    required_path = root / "artifacts/approvals/STAGE3_APPROVAL_REQUIRED.json"
    atomic_write_json(
        required_path,
        {
            "schema_version": migration.BASE_APPROVAL_SCHEMA,
            "kind": "stage3_approval_required",
            "protocol_id": migration.PROTOCOL_ID,
            "approved": False,
            "bindings": bindings,
        },
    )
    approval_path = root / "artifacts/approvals/STAGE3_APPROVED.json"
    atomic_write_json(
        approval_path,
        {
            "schema_version": migration.BASE_APPROVAL_SCHEMA,
            "kind": "stage3_approval",
            "protocol_id": migration.PROTOCOL_ID,
            "approved": True,
            "approval_required_sha256": sha256_file(required_path),
            "bindings": bindings,
        },
    )

    old_sources, new_sources = _source_maps()
    provenance = {
        "schema_version": migration.STAGE3_RUNTIME_SCHEMA,
        "protocol_id": migration.PROTOCOL_ID,
        "config_sha256": sha256_file(config),
        "config_semantic_sha256": "a" * 64,
        "semantic_source_sha256": old_sources,
        "stage3_approval": {
            "path": str(approval_path),
            "sha256": sha256_file(approval_path),
            "approval_required_sha256": sha256_file(required_path),
        },
        "bindings": bindings,
        "runtime": {
            "max_steps": migration.SCHEDULE_HORIZON_STEPS,
            "micro_batch": 4,
            "accumulation_steps": 2,
        },
        "data_exposure": {"mio100": False},
    }
    run_contract = root / "artifacts/checkpoints/stage3/run_contract.json"
    atomic_write_json(
        run_contract,
        {
            "schema_version": migration.STAGE3_RUNTIME_SCHEMA,
            "created_utc": "2026-08-18T00:00:00Z",
            "provenance": provenance,
            "parent_load": {"loaded_count": 1535},
            "micro_batch_trials": [],
            "validation_vram_gate": {"passed": True},
        },
    )
    last_path = root / "artifacts/checkpoints/stage3/last.pth"
    best_path = root / "artifacts/checkpoints/stage3/best_ema.pth"
    torch.save(
        _checkpoint(
            provenance,
            role="raw_training_state",
            model_value=1.0,
            ema_value=2.0,
        ),
        last_path,
    )
    torch.save(
        _checkpoint(
            provenance,
            role="ema_selection",
            model_value=2.0,
            ema_value=2.0,
        ),
        best_path,
    )

    state_path = root / "artifacts/orchestration/state.json"
    atomic_write_json(
        state_path,
        {
            "schema_version": "graphrestore-orchestration-v1",
            "protocol_id": migration.PROTOCOL_ID,
            "status": "FAILED",
            "current_stage": "FAILED",
            "gpu": "released",
            "last_exit_code": 1,
            "next_command": "python scripts/orchestrate.py --resume_post_approval_pipeline",
            "last_command": [
                "python",
                "scripts/train_stage3_planner.py",
                "--resume",
                "auto",
            ],
        },
    )
    guard = _prior_receipt(
        root,
        directory=migration.GUARD_BACKUP_DIR_NAME,
        schema=migration.GUARD_RECEIPT_SCHEMA,
        kind=migration.GUARD_MIGRATION_KIND,
    )
    ema = _prior_receipt(
        root,
        directory=migration.EMA_BACKUP_DIR_NAME,
        schema=migration.EMA_RECEIPT_SCHEMA,
        kind=migration.EMA_MIGRATION_KIND,
    )
    monkeypatch.setattr(
        migration,
        "semantic_source_hashes",
        lambda project_root, *, entrypoints: dict(new_sources),
    )
    expected = {
        "project_root": root,
        "expected_run_contract_sha256": sha256_file(run_contract),
        "expected_last_checkpoint_sha256": sha256_file(last_path),
        "expected_best_checkpoint_sha256": sha256_file(best_path),
        "expected_state_sha256": sha256_file(state_path),
        "expected_base_approval_sha256": sha256_file(approval_path),
        "expected_approval_required_sha256": sha256_file(required_path),
        "expected_stage3_config_sha256": sha256_file(config),
        "expected_guard_receipt_sha256": sha256_file(guard),
        "expected_ema_receipt_sha256": sha256_file(ema),
        "expected_old_source_map": old_sources,
        "expected_new_source_map": new_sources,
    }
    return {
        "root": root,
        "expected": expected,
        "old_sources": old_sources,
        "new_sources": new_sources,
        "run_contract": run_contract,
        "last": last_path,
        "best": best_path,
        "state": state_path,
        "approval": approval_path,
        "required": required_path,
        "config": config,
        "guard": guard,
        "ema": ema,
        "extension": root / "artifacts/approvals" / migration.EXTENSION_APPROVAL_NAME,
        "backup_dir": root / "artifacts/migrations" / migration.BACKUP_DIR_NAME,
    }


def _execute(fixture: dict[str, Any]) -> dict[str, Any]:
    return migration.migrate_stage3_extension_provenance(
        **fixture["expected"],
        execute=True,
        confirmation_token=migration.CONFIRMATION_TOKEN,
    )


def test_dry_run_is_non_mutating_and_records_exact_three_change_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    before = {
        label: sha256_file(fixture[label])
        for label in ("run_contract", "last", "best", "state", "approval")
    }
    receipt = migration.migrate_stage3_extension_provenance(**fixture["expected"])
    assert receipt["status"] == "DRY_RUN"
    assert receipt["extension_approval_field_count"] == 20
    assert receipt["extension_provenance_field_count"] == 10
    assert receipt["checkpoint_bit_exact_outside_provenance_count"] == 19
    changes = receipt["provenance_changes"]
    assert set(changes) == {
        "semantic_source_leaf_diffs",
        "runtime_training_target_step",
        "added_stage3_extension",
    }
    assert len(changes["semantic_source_leaf_diffs"]) == 4
    assert changes["runtime_training_target_step"] == {
        "path": "runtime.training_target_step",
        "old_present": False,
        "old": None,
        "new": 18000,
    }
    assert len(changes["added_stage3_extension"]) == 10
    for label in ("last_checkpoint", "best_checkpoint"):
        model_evidence = receipt["checkpoint_section_fingerprints"][label]["model"]
        assert model_evidence["bit_exact"] is True
        assert model_evidence["old"]["counts"]["collections.OrderedDict"] == 1
        assert model_evidence["new"]["counts"]["collections.OrderedDict"] == 1
    assert not fixture["extension"].exists()
    assert not fixture["backup_dir"].exists()
    assert before == {
        label: sha256_file(fixture[label])
        for label in ("run_contract", "last", "best", "state", "approval")
    }
    assert torch.cuda.is_initialized() is False


def test_execute_publishes_exact_flat_approval_backups_and_three_way_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    protected = {
        label: sha256_file(fixture[label])
        for label in ("state", "approval", "required", "config", "guard", "ema")
    }
    receipt = _execute(fixture)
    assert receipt["status"] == "COMPLETE"
    artifact = load_json(fixture["extension"])
    assert len(artifact) == 20
    assert set(artifact) == {
        "schema_version",
        "kind",
        "protocol_id",
        "approved",
        "cycles",
        "base_step",
        "target_step",
        "validation_every_steps",
        "validation_steps",
        "schedule_horizon_steps",
        "min_lr",
        "lr_policy",
        "formal_mio100_authorized",
        "authorized_pipeline",
        "base_stage3_approval",
        "base_approval_required",
        "base_stage3_config",
        "pre_extension_run_contract",
        "pre_extension_last_checkpoint",
        "pre_extension_best_checkpoint",
    }
    contract = load_json(fixture["run_contract"])
    last = torch.load(fixture["last"], map_location="cpu", weights_only=False)
    best = torch.load(fixture["best"], map_location="cpu", weights_only=False)
    assert type(last["model"]) is OrderedDict
    assert type(best["model"]) is OrderedDict
    assert type(last["ema"]["shadow"]) is dict
    assert type(best["ema"]["shadow"]) is dict
    provenance = contract["provenance"]
    assert provenance == last["provenance"] == best["provenance"]
    assert provenance["runtime"]["max_steps"] == 12000
    assert provenance["runtime"]["training_target_step"] == 18000
    assert provenance["semantic_source_sha256"] == fixture["new_sources"]
    extension = provenance["stage3_extension"]
    assert len(extension) == 10
    assert extension["sha256"] == sha256_file(fixture["extension"])
    assert extension["validation_steps"] == [14000, 16000, 18000]
    backups = [
        fixture["backup_dir"] / "run_contract.json",
        fixture["backup_dir"] / "last.pth",
        fixture["backup_dir"] / "best_ema.pth",
    ]
    assert all(path.is_file() and not path.is_symlink() for path in backups)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in backups)
    assert len({(path.stat().st_dev, path.stat().st_ino) for path in backups}) == 3
    assert protected == {
        label: sha256_file(fixture[label])
        for label in ("state", "approval", "required", "config", "guard", "ema")
    }
    assert torch.cuda.is_initialized() is False


def test_execute_requires_exact_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    with pytest.raises(migration.Stage3ExtensionMigrationError, match="exact"):
        migration.migrate_stage3_extension_provenance(
            **fixture["expected"], execute=True, confirmation_token="yes"
        )
    assert not fixture["backup_dir"].exists()


def test_rejects_physical_source_map_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        migration,
        "semantic_source_hashes",
        lambda project_root, *, entrypoints: fixture["old_sources"],
    )
    with pytest.raises(migration.Stage3ExtensionMigrationError, match="physical"):
        migration.migrate_stage3_extension_provenance(**fixture["expected"])


@pytest.mark.parametrize("mutation", ["fifth_leaf", "missing", "swapped"])
def test_frozen_transition_override_is_rejected_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    arguments = copy.deepcopy(fixture["expected"])
    if mutation == "fifth_leaf":
        arguments["expected_new_source_map"]["scripts/eval_guard_diagnostics.py"] = (
            "f" * 64
        )
    elif mutation == "missing":
        arguments["expected_old_source_map"].pop("src/utils/paths.py")
    else:
        arguments["expected_old_source_map"], arguments["expected_new_source_map"] = (
            arguments["expected_new_source_map"],
            arguments["expected_old_source_map"],
        )
    before = {
        label: sha256_file(fixture[label])
        for label in ("run_contract", "last", "best", "state", "approval")
    }
    with pytest.raises(migration.Stage3ExtensionMigrationError):
        migration.migrate_stage3_extension_provenance(**arguments)
    assert not fixture["extension"].exists()
    assert not fixture["backup_dir"].exists()
    assert before == {
        label: sha256_file(fixture[label])
        for label in ("run_contract", "last", "best", "state", "approval")
    }


def test_builtin_default_value_drift_is_rejected_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    drifted = dict(migration.AUDITED_NEW_SEMANTIC_SOURCE_SHA256)
    drifted["src/utils/paths.py"] = "f" * 64
    monkeypatch.setattr(migration, "AUDITED_NEW_SEMANTIC_SOURCE_SHA256", drifted)
    before = sha256_file(fixture["run_contract"])
    with pytest.raises(
        migration.Stage3ExtensionMigrationError,
        match="built-in audited source-transition values drifted",
    ):
        migration.migrate_stage3_extension_provenance(**fixture["expected"])
    assert sha256_file(fixture["run_contract"]) == before
    assert not fixture["extension"].exists()
    assert not fixture["backup_dir"].exists()


def test_rejects_symlink_anywhere_in_canonical_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    real = fixture["root"] / "artifacts/checkpoints/stage3"
    moved = fixture["root"] / "artifacts/checkpoints/stage3.real"
    real.rename(moved)
    real.symlink_to(moved, target_is_directory=True)
    with pytest.raises(migration.Stage3ExtensionMigrationError, match="symlink"):
        migration.migrate_stage3_extension_provenance(**fixture["expected"])


def test_single_writer_lock_refuses_a_second_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    descriptor = os.open(
        fixture["root"] / "artifacts/migrations",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(migration.Stage3ExtensionMigrationError, match="writer"):
            migration.migrate_stage3_extension_provenance(**fixture["expected"])
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_candidate_checkpoint_mutation_outside_provenance_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original = migration.atomic_torch_save

    def corrupt(payload: dict[str, Any], path: Path) -> None:
        mutated = copy.deepcopy(payload)
        mutated["optimizer"]["param_groups"][0]["lr"] = 9.0
        original(mutated, path)

    monkeypatch.setattr(migration, "atomic_torch_save", corrupt)
    with pytest.raises(migration.Stage3ExtensionMigrationError, match="mutation"):
        migration.migrate_stage3_extension_provenance(**fixture["expected"])
    assert not fixture["backup_dir"].exists()


def test_publication_failure_rolls_back_and_removes_extension_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    old = {
        label: sha256_file(fixture[label]) for label in ("run_contract", "last", "best")
    }
    original = migration._replace_and_fsync
    calls = 0

    def fail_after_first(candidate: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        original(candidate, destination)
        if calls == 2:
            raise OSError("synthetic publication interruption")

    monkeypatch.setattr(migration, "_replace_and_fsync", fail_after_first)
    with pytest.raises(OSError, match="synthetic"):
        _execute(fixture)
    assert not fixture["extension"].exists()
    assert old == {
        label: sha256_file(fixture[label]) for label in ("run_contract", "last", "best")
    }
    rolled_last = torch.load(fixture["last"], map_location="cpu", weights_only=False)
    rolled_best = torch.load(fixture["best"], map_location="cpu", weights_only=False)
    assert type(rolled_last["model"]) is OrderedDict
    assert type(rolled_best["model"]) is OrderedDict
    assert type(rolled_last["ema"]["shadow"]) is dict
    assert type(rolled_best["ema"]["shadow"]) is dict
    assert (
        load_json(fixture["backup_dir"] / migration.RECEIPT_NAME)["status"]
        == "ROLLED_BACK"
    )


MIXTURES = tuple(itertools.product((False, True), repeat=4))


@pytest.mark.parametrize(
    ("new_contract", "new_last", "new_best", "new_approval"), MIXTURES
)
def test_recover_prepared_accepts_every_old_new_publication_mixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    new_contract: bool,
    new_last: bool,
    new_best: bool,
    new_approval: bool,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    old_hashes = {
        label: fixture["expected"][f"expected_{label}_sha256"]
        for label in ("run_contract", "last_checkpoint", "best_checkpoint")
    }
    _execute(fixture)
    snapshots = fixture["root"] / "snapshots"
    snapshots.mkdir()
    new_paths = {
        "run_contract": fixture["run_contract"],
        "last_checkpoint": fixture["last"],
        "best_checkpoint": fixture["best"],
        "extension_approval": fixture["extension"],
    }
    saved: dict[str, Path] = {}
    for label, source in new_paths.items():
        target = snapshots / source.name
        shutil.copyfile(source, target)
        saved[label] = target
    receipt_path = fixture["backup_dir"] / migration.RECEIPT_NAME
    receipt = load_json(receipt_path)
    receipt["status"] = "PREPARED"
    receipt.pop("completed_utc", None)
    receipt.pop("backup_read_only_after_publication", None)
    receipt.pop("protected_artifacts_unchanged_after_publication", None)
    atomic_write_json(receipt_path, receipt)

    choices = {
        "run_contract": new_contract,
        "last_checkpoint": new_last,
        "best_checkpoint": new_best,
    }
    live = {
        "run_contract": fixture["run_contract"],
        "last_checkpoint": fixture["last"],
        "best_checkpoint": fixture["best"],
    }
    backup = {
        "run_contract": fixture["backup_dir"] / "run_contract.json",
        "last_checkpoint": fixture["backup_dir"] / "last.pth",
        "best_checkpoint": fixture["backup_dir"] / "best_ema.pth",
    }
    for label, use_new in choices.items():
        source = saved[label] if use_new else backup[label]
        shutil.copyfile(source, live[label])
    if new_approval:
        shutil.copyfile(saved["extension_approval"], fixture["extension"])
    else:
        fixture["extension"].unlink()

    recovered = migration.recover_prepared_stage3_extension_provenance(
        **fixture["expected"],
        confirmation_token=migration.RECOVERY_CONFIRMATION_TOKEN,
    )
    assert recovered["status"] == "ROLLED_BACK_FROM_PREPARED"
    assert not fixture["extension"].exists()
    assert sha256_file(fixture["run_contract"]) == old_hashes["run_contract"]
    assert sha256_file(fixture["last"]) == old_hashes["last_checkpoint"]
    assert sha256_file(fixture["best"]) == old_hashes["best_checkpoint"]
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in backup.values())


def test_prepared_recovery_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _execute(fixture)
    receipt_path = fixture["backup_dir"] / migration.RECEIPT_NAME
    receipt = load_json(receipt_path)
    receipt["status"] = "PREPARED"
    atomic_write_json(receipt_path, receipt)
    first = migration.recover_prepared_stage3_extension_provenance(
        **fixture["expected"],
        confirmation_token=migration.RECOVERY_CONFIRMATION_TOKEN,
    )
    second = migration.recover_prepared_stage3_extension_provenance(
        **fixture["expected"],
        confirmation_token=migration.RECOVERY_CONFIRMATION_TOKEN,
    )
    assert first == second


def test_prepared_recovery_rejects_unknown_live_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _execute(fixture)
    receipt_path = fixture["backup_dir"] / migration.RECEIPT_NAME
    receipt = load_json(receipt_path)
    receipt["status"] = "PREPARED"
    atomic_write_json(receipt_path, receipt)
    fixture["last"].write_bytes(b"unknown")
    with pytest.raises(
        migration.Stage3ExtensionMigrationError, match="neither old nor new"
    ):
        migration.recover_prepared_stage3_extension_provenance(
            **fixture["expected"],
            confirmation_token=migration.RECOVERY_CONFIRMATION_TOKEN,
        )


def test_recovery_requires_separate_exact_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _execute(fixture)
    with pytest.raises(migration.Stage3ExtensionMigrationError, match="exact"):
        migration.recover_prepared_stage3_extension_provenance(
            **fixture["expected"], confirmation_token=migration.CONFIRMATION_TOKEN
        )


def test_recovery_override_map_drift_is_rejected_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _execute(fixture)
    receipt_path = fixture["backup_dir"] / migration.RECEIPT_NAME
    receipt = load_json(receipt_path)
    receipt["status"] = "PREPARED"
    atomic_write_json(receipt_path, receipt)
    arguments = copy.deepcopy(fixture["expected"])
    arguments["expected_new_source_map"]["src/utils/paths.py"] = "f" * 64
    before = {
        label: sha256_file(fixture[label])
        for label in ("run_contract", "last", "best", "extension")
    }
    receipt_before = sha256_file(receipt_path)
    with pytest.raises(migration.Stage3ExtensionMigrationError):
        migration.recover_prepared_stage3_extension_provenance(
            **arguments,
            confirmation_token=migration.RECOVERY_CONFIRMATION_TOKEN,
        )
    assert receipt_before == sha256_file(receipt_path)
    assert before == {
        label: sha256_file(fixture[label])
        for label in ("run_contract", "last", "best", "extension")
    }


def test_prior_complete_backup_mode_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = load_json(fixture["guard"])
    backup = Path(receipt["backup"]["run_contract"]["path"])
    backup.chmod(0o644)
    fixture["expected"]["expected_guard_receipt_sha256"] = sha256_file(fixture["guard"])
    with pytest.raises(migration.Stage3ExtensionMigrationError, match="backup drifted"):
        migration.migrate_stage3_extension_provenance(**fixture["expected"])


def test_extension_backup_symlink_is_rejected_during_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _execute(fixture)
    receipt_path = fixture["backup_dir"] / migration.RECEIPT_NAME
    receipt = load_json(receipt_path)
    receipt["status"] = "PREPARED"
    atomic_write_json(receipt_path, receipt)
    backup = fixture["backup_dir"] / "last.pth"
    moved = fixture["backup_dir"] / "last.real.pth"
    backup.rename(moved)
    backup.symlink_to(moved)
    with pytest.raises(migration.Stage3ExtensionMigrationError, match="symlink"):
        migration.recover_prepared_stage3_extension_provenance(
            **fixture["expected"],
            confirmation_token=migration.RECOVERY_CONFIRMATION_TOKEN,
        )


def test_cli_defaults_lock_complete_47_entry_old_and_new_source_maps() -> None:
    old = migration.AUDITED_OLD_SEMANTIC_SOURCE_SHA256
    new = migration.AUDITED_NEW_SEMANTIC_SOURCE_SHA256
    assert len(old) == len(new) == 47
    assert old.keys() == new.keys()
    assert {path: (old[path], new[path]) for path in old if old[path] != new[path]} == {
        "scripts/train_stage3_planner.py": (
            "3d498fcea7cdc52480e6ff8e3e2d85596d2bde94ed289f14deeee66f9d9beabc",
            "1e7db4c46f640d62501e91eb50862073f8c6473b9090018771b41fe1bdfc4b9d",
        ),
        "src/training/orchestration.py": (
            "7979ae0feedc1677a02fe2bd2ac76432185881a75b80122dc8bcd936b9cbff1f",
            "8691c56fafafd6f5f2b37d53ab01009b092ca0395735a69ab71ab97f34a9b622",
        ),
        "src/training/stage3_engine.py": (
            "908bcd7ff829aabba8376ec949156890983f51924aaa7e2313e013648d817b49",
            "7c65d89f9778dd3f49250774fcfaa4f3f6209d62ac6e9f9f507991fe22427e0a",
        ),
        "src/training/stage4_engine.py": (
            "e2fbfbc2ee580b90cb92c48e6b289d6bc6d3d4651c42d34295ce07fc664814b6",
            "518b10b49320fd24879febc3483d30f7a8b28e96037588102ddb65f89a958845",
        ),
    }
    with pytest.raises(
        migration.Stage3ExtensionMigrationError, match="must not be empty"
    ):
        migration._validate_source_map({}, field="unfrozen map")
