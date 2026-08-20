from __future__ import annotations

import copy
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import train_stage3_planner as trainer
from src.training import stage3_finalization as contract
from src.utils.hashing import is_sha256, sha256_file
from src.utils.io import atomic_write_json


def _write(path: Path, value: str, *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    path.chmod(mode)
    return path


def _save(path: Path, value: object, *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    path.chmod(mode)
    return path


def _checkpoint(
    provenance: dict[str, Any],
    *,
    step: int,
    role: str,
    pending: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage3",
        "step": step,
        "model": {"planner.weight": torch.tensor([1.0])},
        "ema": {"shadow": {"planner.weight": torch.tensor([1.0])}},
        "optimizer": {},
        "scheduler": {},
        "scaler": None,
        "rng_states": {},
        "sampler_state": {},
        "provenance": provenance,
        "metrics": {"validation_step": step, "best_step": 12_000},
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
        "model_role": role,
        "resumable": role == "raw_training_state",
        "pending_validation_step": pending,
        "optimizer_transaction_active": False,
        "optimizer_state_name_ledger": {},
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = (tmp_path / "project").resolve()
    source = _write(root / "src/core.py", "VALUE = 1\n")
    trainer_source = _write(
        root / "scripts/train_stage3_planner.py", "# frozen trainer\n"
    )
    _write(root / contract.FINALIZER_ENTRYPOINT, "# frozen finalizer\n")
    historical = {
        "scripts/train_stage3_planner.py": sha256_file(trainer_source),
        "src/core.py": sha256_file(source),
    }
    monkeypatch.setattr(contract, "HISTORICAL_SEMANTIC_SOURCE_COUNT", 2)
    monkeypatch.setattr(contract, "ALLOWED_SEMANTIC_SOURCE_DRIFT", ())

    relative = dict(contract.EXPECTED_RELATIVE_PATHS)
    absolute = {
        "user_instruction": str(root / "external/user_instruction.txt"),
        "target_contract": str(root / "external/target.md"),
        "primary_val_manifest": str(root / "external/primary_val.jsonl"),
    }
    monkeypatch.setattr(contract, "EXPECTED_RELATIVE_PATHS", relative)
    monkeypatch.setattr(contract, "EXPECTED_ABSOLUTE_PATHS", absolute)

    paths = {
        logical: Path(absolute[logical])
        if logical in absolute
        else root / relative[logical]
        for logical in contract.BINDING_KEYS
    }
    for logical in (
        "user_instruction",
        "target_contract",
        "primary_val_manifest",
        "stage3_config",
        "relation_val",
        "pair_prior",
        "global_priority",
        "stage1_checkpoint",
        "pre_extension_run_contract",
        "pre_extension_last_checkpoint",
        "pre_extension_best_checkpoint",
    ):
        mode = 0o444 if logical in contract.IMMUTABLE_BINDINGS else 0o600
        _write(paths[logical], logical, mode=mode)

    approval_binding_paths = {
        "config_stage3": paths["stage3_config"],
        "primary_val_manifest": paths["primary_val_manifest"],
        "relation_val": paths["relation_val"],
        "pair_prior": paths["pair_prior"],
        "global_priority": paths["global_priority"],
        "stage1_checkpoint": paths["stage1_checkpoint"],
    }
    approval_bindings = {
        logical: {"path": str(path), "sha256": sha256_file(path)}
        for logical, path in approval_binding_paths.items()
    }
    for index in range(16):
        path = _write(root / f"bound/extra_{index}.bin", f"extra-{index}")
        approval_bindings[f"extra_{index}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    atomic_write_json(
        paths["approval_required"],
        {
            "schema_version": "graphrestore-stage3-approval-v1",
            "kind": "stage3_approval_required",
            "protocol_id": contract.PROTOCOL_ID,
            "approved": False,
            "bindings": approval_bindings,
        },
    )
    atomic_write_json(
        paths["stage3_approval"],
        {
            "schema_version": "graphrestore-stage3-approval-v1",
            "kind": "stage3_approval",
            "protocol_id": contract.PROTOCOL_ID,
            "approved": True,
            "approval_required_sha256": sha256_file(paths["approval_required"]),
            "bindings": approval_bindings,
        },
    )
    atomic_write_json(
        paths["historical_extension_authorization"],
        {
            "schema_version": "graphrestore-stage3-extension-approval-v1",
            "kind": "stage3_extension_approval",
            "approved": True,
            "authorized_pipeline": ["stage3_extension", "stage4"],
            "formal_mio100_authorized": False,
        },
    )
    extension_sha = sha256_file(paths["historical_extension_authorization"])
    provenance = {
        "schema_version": "graphrestore-stage3-runtime-v1",
        "protocol_id": contract.PROTOCOL_ID,
        "semantic_source_sha256": historical,
        "runtime": {"max_steps": 12_000, "training_target_step": 18_000},
        "stage3_extension": {
            "path": str(paths["historical_extension_authorization"]),
            "sha256": extension_sha,
            "cycles": 3,
            "base_step": 12_000,
            "target_step": 18_000,
            "validation_every_steps": 2_000,
            "validation_steps": [14_000, 16_000, 18_000],
            "schedule_horizon_steps": 12_000,
            "min_lr": 2.0e-6,
            "lr_policy": "hold_original_cosine_floor_after_schedule_horizon",
        },
    }
    atomic_write_json(
        paths["run_contract"],
        {
            "schema_version": "graphrestore-stage3-runtime-v1",
            "provenance": provenance,
        },
    )
    paths["run_contract"].chmod(0o444)
    live_run = root / "artifacts/checkpoints/stage3/run_contract.json"
    live_run.parent.mkdir(parents=True, exist_ok=True)
    live_run.write_bytes(paths["run_contract"].read_bytes())

    _save(
        paths["abandoned_last_checkpoint"],
        _checkpoint(
            provenance,
            step=14_000,
            role="raw_training_state",
            pending=14_000,
        ),
        mode=0o444,
    )
    live_last = root / "artifacts/checkpoints/stage3/last.pth"
    live_last.write_bytes(paths["abandoned_last_checkpoint"].read_bytes())
    _save(
        paths["selected_checkpoint"],
        _checkpoint(
            provenance,
            step=12_000,
            role="ema_selection",
            pending=None,
        ),
    )
    atomic_write_json(
        paths["selected_validation"],
        {
            "protocol_id": contract.PROTOCOL_ID,
            "checkpoint_presence_threshold": 0.5,
            "planner": {"sample_count": 1_600},
            "graph": {"sample_count": 1_600},
        },
    )
    history = "step,planner_macro_f1\n" + "".join(
        f"{step},0.5\n" for step in (2_000, 4_000, 6_000, 8_000, 10_000, 12_000)
    )
    _write(paths["calibration_history"], history)

    atomic_write_json(
        paths["historical_extension_migration_receipt"],
        {
            "status": "COMPLETE",
            "cpu_only": True,
            "three_live_artifacts_share_exact_provenance": True,
            "new": {
                "extension_approval": extension_sha,
                "run_contract": sha256_file(paths["run_contract"]),
                "best_checkpoint": sha256_file(paths["selected_checkpoint"]),
            },
        },
    )
    audited = {logical: sha256_file(path) for logical, path in paths.items()}
    assert len(audited["global_priority"]) == 64
    assert all(is_sha256(value) for value in audited.values())
    monkeypatch.setattr(contract, "AUDITED_BINDING_SHA256", audited)
    return {"root": root, "paths": paths}


def _publish(fixture: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root, paths = fixture["root"], fixture["paths"]
    payload = contract.build_stage3_extension_revocation_payload(
        project_root=root,
        binding_paths=paths,
        allowed_semantic_source_drift=[],
        created_utc="2026-08-18T12:00:00Z",
    )
    canonical = root / contract.REVOCATION_RELATIVE_PATH
    atomic_write_json(canonical, payload)
    return canonical, payload


def test_production_audited_hash_constants_are_exact_sha256() -> None:
    assert set(contract.AUDITED_BINDING_SHA256) == contract.BINDING_KEYS
    assert len(contract.AUDITED_BINDING_SHA256["global_priority"]) == 64
    assert all(is_sha256(value) for value in contract.AUDITED_BINDING_SHA256.values())


def test_builder_and_validator_round_trip_cpu_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert torch.cuda.is_initialized() is False
    fixture = _fixture(tmp_path, monkeypatch)
    canonical, payload = _publish(fixture)
    evidence = contract.validate_stage3_extension_revocation(
        canonical, project_root=fixture["root"], require_present=True
    )
    assert evidence is not None
    assert evidence.sha256 == sha256_file(canonical)
    assert evidence.payload["bindings"] == payload["bindings"]
    assert (
        evidence.bindings["selected_checkpoint"]["sha256"]
        == (contract.AUDITED_BINDING_SHA256["selected_checkpoint"])
    )
    assert evidence.provenance_binding() == {
        "path": str(canonical),
        "sha256": sha256_file(canonical),
    }
    assert torch.cuda.is_initialized() is False


def test_builder_is_pure_and_optional_absence_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    payload = contract.build_stage3_extension_revocation_payload(
        project_root=fixture["root"],
        binding_paths=fixture["paths"],
        allowed_semantic_source_drift=[],
        created_utc="2026-08-18T12:00:00Z",
    )
    canonical = fixture["root"] / contract.REVOCATION_RELATIVE_PATH
    assert set(payload) == contract.REVOCATION_KEYS
    assert not canonical.exists()
    assert (
        contract.validate_stage3_extension_revocation(
            canonical, project_root=fixture["root"], require_present=False
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage3_training_authorized", True),
        ("optimizer_steps_authorized", False),
        ("selected_step", 14_000),
        ("authorized_pipeline", ["stage3_extension", "stage4"]),
    ],
)
def test_validator_rejects_permission_or_step_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    canonical, payload = _publish(fixture)
    changed = copy.deepcopy(payload)
    changed[field] = value
    atomic_write_json(canonical, changed)
    with pytest.raises(contract.Stage3FinalizationContractError):
        contract.validate_stage3_extension_revocation(
            canonical, project_root=fixture["root"]
        )


def test_validator_rejects_live_state_drift_despite_intact_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    canonical, _ = _publish(fixture)
    live = fixture["root"] / "artifacts/checkpoints/stage3/run_contract.json"
    live.write_text("drift", encoding="utf-8")
    with pytest.raises(
        contract.Stage3FinalizationContractError,
        match="canonical live run_contract changed",
    ):
        contract.validate_stage3_extension_revocation(
            canonical, project_root=fixture["root"]
        )


def test_validator_rejects_live_archive_shared_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    canonical, _ = _publish(fixture)
    live = fixture["root"] / "artifacts/checkpoints/stage3/last.pth"
    live.unlink()
    os.link(fixture["paths"]["abandoned_last_checkpoint"], live)
    with pytest.raises(
        contract.Stage3FinalizationContractError, match="must not share an inode"
    ):
        contract.validate_stage3_extension_revocation(
            canonical, project_root=fixture["root"]
        )


def test_validator_rejects_unlisted_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    canonical, _ = _publish(fixture)
    (fixture["root"] / "src/core.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(contract.Stage3FinalizationContractError):
        contract.validate_stage3_extension_revocation(
            canonical, project_root=fixture["root"]
        )


@pytest.mark.parametrize("entry_kind", ["file", "directory", "dangling_symlink"])
def test_trainer_refuses_any_tombstone_before_approval_or_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    root = (tmp_path / "trainer").resolve()
    tombstone = root / contract.REVOCATION_RELATIVE_PATH
    tombstone.parent.mkdir(parents=True)
    if entry_kind == "file":
        tombstone.write_text("not even valid json", encoding="utf-8")
    elif entry_kind == "directory":
        tombstone.mkdir()
    else:
        tombstone.symlink_to(root / "missing")
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("called")
        raise AssertionError("later Stage3 operation was reached")

    monkeypatch.setattr(trainer, "PROJECT_ROOT", root)
    monkeypatch.setattr(trainer, "validate_stage3_approval", forbidden)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(trainer, "build_stage3_optimizer", forbidden)
    with pytest.raises(
        contract.Stage3FinalizationContractError, match="permanently disabled"
    ):
        trainer.run(Namespace())
    assert calls == []


def test_trainer_without_tombstone_reaches_approval_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "trainer").resolve()
    root.mkdir()

    class ExpectedApprovalStop(RuntimeError):
        pass

    def stop(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ExpectedApprovalStop

    monkeypatch.setattr(trainer, "PROJECT_ROOT", root)
    monkeypatch.setattr(trainer, "validate_stage3_approval", stop)
    with pytest.raises(ExpectedApprovalStop):
        trainer.run(Namespace(config=Path("unused"), output_dir=None, resume=None))
    assert not os.path.lexists(root / contract.REVOCATION_RELATIVE_PATH)
