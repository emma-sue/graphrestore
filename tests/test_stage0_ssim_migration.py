from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from scripts import migrate_stage0_ssim_fp32_checkpoint as migration
from src.training.checkpointing import atomic_torch_save
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Case:
    source: Path
    destination: Path
    config: Path
    receipt: Path
    expected_source_sha256: str
    expected_role: str
    old_provenance: dict[str, Any]
    expected_provenance: dict[str, Any]

    def arguments(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "config": self.config,
            "receipt": self.receipt,
            "expected_source_sha256": self.expected_source_sha256,
            "expected_role": self.expected_role,
            "project_root": PROJECT_ROOT,
        }


def _write_train_log(path: Path) -> None:
    rows = (
        {
            "event": "train_step",
            "step": 1,
            "lambda_ssim": 0.0,
            "ssim_loss": 0.0,
        },
        {
            "event": "train_step",
            "step": 12_000,
            "lambda_ssim": 0.0,
            "ssim_loss": 0.0,
        },
        {"event": "validation_committed", "step": 12_000},
        {
            "event": "train_step",
            "step": 12_020,
            "lambda_ssim": 0.05,
            "ssim_loss": -0.2,
        },
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tensor_state(
    *, ema_selection: bool
) -> tuple[OrderedDict[str, Any], dict[str, Any]]:
    model: OrderedDict[str, Any] = OrderedDict()
    shadow: dict[str, Any] = {}
    for index in range(migration.EXPECTED_TENSOR_KEYS):
        value = torch.tensor([index, -index], dtype=torch.float32)
        model[f"tensor_{index:03d}"] = value
        shadow[f"tensor_{index:03d}"] = value.clone() if ema_selection else value + 1.0
    return model, shadow


def _make_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str = "raw_training_state",
    mutate_payload: Callable[[dict[str, Any]], None] | None = None,
    mutate_builder: Callable[[dict[str, Any]], None] | None = None,
) -> _Case:
    config = tmp_path / "stage0.yaml"
    config.write_text("synthetic: true\n", encoding="utf-8")
    metric_git = migration._metric_git_facts(PROJECT_ROOT)
    metric_sha = sha256_file(PROJECT_ROOT / migration.METRIC_SOURCE_RELATIVE)
    compile_path = PROJECT_ROOT / migration.COMPILE_ARTIFACT_RELATIVE
    compile_sha = sha256_file(compile_path)
    compile_artifact = load_json(compile_path)
    decision = compile_artifact["decision"]
    runtime = {
        "crop_size": 192,
        "micro_batch": 4,
        "effective_batch": 8,
        "accumulation_steps": 2,
        "gradient_checkpointing": False,
        "schedule_max_steps": 60_000,
        "target_step": 60_000,
        "integration": False,
        "torch_compile": False,
    }
    old_provenance: dict[str, Any] = {
        "protocol_id": migration.PROTOCOL_ID,
        "stage": "stage0",
        "config_path": str(config.resolve()),
        "config_sha256": sha256_file(config),
        "runtime": runtime,
        "semantic_source_sha256": {
            migration.METRIC_SOURCE_RELATIVE: metric_git["before_sha256"]
        },
        "compile_ab": {
            "path": str(compile_path),
            "sha256": "a" * 64,
            "recommend_torch_compile": False,
            "decision": decision,
            "code_sha256": {
                migration.METRIC_SOURCE_RELATIVE: metric_git["before_sha256"]
            },
        },
        "warm_start_load": copy.deepcopy(migration.EXPECTED_OLD_WARM_START),
    }
    builder_provenance = copy.deepcopy(old_provenance)
    builder_provenance["semantic_source_sha256"][migration.METRIC_SOURCE_RELATIVE] = (
        metric_sha
    )
    builder_provenance["compile_ab"]["sha256"] = compile_sha
    builder_provenance["compile_ab"]["code_sha256"][
        migration.METRIC_SOURCE_RELATIVE
    ] = metric_sha
    builder_provenance["warm_start_load"] = copy.deepcopy(
        migration.EXPECTED_BUILDER_NONE_WARM_START
    )
    if mutate_builder is not None:
        mutate_builder(builder_provenance)

    monkeypatch.setattr(
        migration,
        "load_and_validate_stage0_config",
        lambda path: ({"synthetic": True}, {"synthetic": True}),
    )

    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["project_root"] == PROJECT_ROOT
        assert kwargs["config_path"] == config.resolve()
        assert kwargs["load_report"] is None
        assert kwargs["runtime"].torch_compile is False
        return copy.deepcopy(builder_provenance)

    monkeypatch.setattr(migration, "build_stage0_provenance", fake_builder)

    ema_selection = role == "ema_selection"
    model, shadow = _tensor_state(ema_selection=ema_selection)
    payload: dict[str, Any] = {
        "schema_version": migration.CHECKPOINT_SCHEMA,
        "stage": "stage0",
        "step": 12_000,
        "model": model,
        "ema": {"decay": 0.9999, "num_updates": 12_000, "shadow": shadow},
        "optimizer": {
            "state": {0: {"exp_avg": torch.tensor([-0.0, 1.25])}},
            "param_groups": [{"lr": 1.0e-4, "params": [0]}],
        },
        "scheduler": {"last_epoch": 12_000},
        "scaler": None,
        "rng_states": {
            "python": (3, (1, 2, 3), None),
            "numpy": (
                "MT19937",
                np.arange(8, dtype=np.uint32),
                7,
                0,
                0.0,
            ),
            "torch_cpu": torch.arange(8, dtype=torch.uint8),
        },
        "sampler_state": {
            "consumed_optimizer_step": 12_000,
            "sample_cursor": 96_000,
        },
        "provenance": old_provenance,
        "metrics": {"best_group_a_psnr": 22.1},
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "model_role": role,
        "resumable": role == "raw_training_state",
        "pending_validation_step": None,
    }
    if mutate_payload is not None:
        mutate_payload(payload)
    expected_provenance = copy.deepcopy(builder_provenance)
    expected_provenance["warm_start_load"] = copy.deepcopy(
        migration.EXPECTED_OLD_WARM_START
    )
    monkeypatch.setattr(
        migration,
        "EXPECTED_OLD_PROVENANCE_JSON_SHA256",
        sha256_json(payload["provenance"]),
    )
    monkeypatch.setattr(
        migration,
        "EXPECTED_NEW_PROVENANCE_JSON_SHA256",
        sha256_json(expected_provenance),
    )
    monkeypatch.setattr(migration, "FRESH_GATE_ARTIFACTS", ())
    source = tmp_path / f"{role}.source.pth"
    atomic_torch_save(payload, source)
    _write_train_log(tmp_path / "train.jsonl")
    return _Case(
        source=source,
        destination=tmp_path / f"{role}.migrated.pth",
        config=config,
        receipt=tmp_path / f"{role}.receipt.json",
        expected_source_sha256=sha256_file(source),
        expected_role=role,
        old_provenance=old_provenance,
        expected_provenance=expected_provenance,
    )


@pytest.mark.parametrize("role", ["raw_training_state", "ema_selection"])
def test_stage0_ssim_migration_happy_path_is_provenance_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    case = _make_case(tmp_path, monkeypatch, role=role)
    receipt = migration.migrate_stage0_checkpoint(**case.arguments())

    assert case.destination.is_file()
    assert case.receipt.is_file()
    migrated = torch.load(
        case.destination, map_location="cpu", weights_only=False, mmap=True
    )
    source = torch.load(case.source, map_location="cpu", weights_only=False, mmap=True)
    assert migrated["provenance"] == case.expected_provenance
    assert migrated["provenance"] != source["provenance"]
    for key in source:
        if key != "provenance":
            migration._assert_bit_exact(source[key], migrated[key], path=key)
    assert [row["path"] for row in receipt["exact_provenance_leaf_diff"]] == sorted(
        migration.EXPECTED_DIFF_PATHS
    )
    assert receipt["discarded_log_range"] == {
        "classification": "discarded_transient_not_in_step12000_checkpoint",
        "record_count": 1,
        "train_step_count": 1,
        "minimum_step": 12_020,
        "maximum_step": 12_020,
        "event_counts": {"train_step": 1},
    }
    assert receipt["old_checkpoint_sha256"] == case.expected_source_sha256
    assert receipt["new_checkpoint_sha256"] == sha256_file(case.destination)
    assert load_json(case.receipt) == receipt
    assert all(
        row["bit_exact"]
        for key, row in receipt["section_fingerprints_and_counts"].items()
        if key != "provenance"
    )


def test_stage0_ssim_migration_rejects_unexpected_provenance_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate_builder(provenance: dict[str, Any]) -> None:
        provenance["config_sha256"] = "b" * 64

    case = _make_case(tmp_path, monkeypatch, mutate_builder=mutate_builder)
    with pytest.raises(
        migration.Stage0SsimMigrationError, match="unexpected provenance diff"
    ):
        migration.migrate_stage0_checkpoint(**case.arguments())
    assert not case.destination.exists()
    assert not case.receipt.exists()


def test_stage0_ssim_migration_rejects_round_trip_state_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, monkeypatch)
    real_atomic_torch_save = migration.atomic_torch_save

    def corrupt_save(payload: Any, destination: Any) -> None:
        corrupted = copy.deepcopy(dict(payload))
        first_key = next(iter(corrupted["model"]))
        corrupted["model"][first_key] = corrupted["model"][first_key] + 1.0
        real_atomic_torch_save(corrupted, destination)

    monkeypatch.setattr(migration, "atomic_torch_save", corrupt_save)
    with pytest.raises(migration.Stage0SsimMigrationError, match="tensor mutation"):
        migration.migrate_stage0_checkpoint(**case.arguments())
    assert not case.destination.exists()
    assert not case.receipt.exists()


def test_stage0_ssim_migration_rejects_bad_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_step(payload: dict[str, Any]) -> None:
        payload["step"] = 11_999

    case = _make_case(tmp_path, monkeypatch, mutate_payload=bad_step)
    with pytest.raises(
        migration.Stage0SsimMigrationError, match="header mismatch at step"
    ):
        migration.migrate_stage0_checkpoint(**case.arguments())
    assert not case.destination.exists()
    assert not case.receipt.exists()


def test_stage0_ssim_migration_rejects_bad_best_ema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_ema(payload: dict[str, Any]) -> None:
        first_key = next(iter(payload["ema"]["shadow"]))
        payload["ema"]["shadow"][first_key] += 1.0

    case = _make_case(
        tmp_path,
        monkeypatch,
        role="ema_selection",
        mutate_payload=bad_ema,
    )
    with pytest.raises(
        migration.Stage0SsimMigrationError,
        match="model differs from ema.shadow",
    ):
        migration.migrate_stage0_checkpoint(**case.arguments())
    assert not case.destination.exists()
    assert not case.receipt.exists()


def test_stage0_ssim_migration_rejects_warm_start_premise_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_warm_start(payload: dict[str, Any]) -> None:
        payload["provenance"]["warm_start_load"]["missing_keys"] = ["encoder.bad"]

    case = _make_case(tmp_path, monkeypatch, mutate_payload=bad_warm_start)
    with pytest.raises(
        migration.Stage0SsimMigrationError,
        match="old warm_start_load is not the exact",
    ):
        migration.migrate_stage0_checkpoint(**case.arguments())
    assert not case.destination.exists()
    assert not case.receipt.exists()


def test_stage0_ssim_migration_rejects_builder_none_warm_start_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_builder_warm_start(provenance: dict[str, Any]) -> None:
        provenance["warm_start_load"]["missing_keys"] = []

    case = _make_case(
        tmp_path,
        monkeypatch,
        mutate_builder=bad_builder_warm_start,
    )
    with pytest.raises(
        migration.Stage0SsimMigrationError,
        match="load_report=None warm-start builder behavior drifted",
    ):
        migration.migrate_stage0_checkpoint(**case.arguments())
    assert not case.destination.exists()
    assert not case.receipt.exists()


def test_stage0_ssim_migration_refuses_existing_destination_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, monkeypatch)
    case.destination.write_bytes(b"user-owned")
    with pytest.raises(
        migration.Stage0SsimMigrationError,
        match="refusing to overwrite existing destination",
    ):
        migration.migrate_stage0_checkpoint(**case.arguments())
    assert case.destination.read_bytes() == b"user-owned"
    assert not case.receipt.exists()


def test_stage0_ssim_migration_rejects_nonzero_preboundary_ssim_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, monkeypatch)
    log_path = case.source.parent / "train.jsonl"
    rows = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["ssim_loss"] = -0.25
    log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(
        migration.Stage0SsimMigrationError,
        match="pre-boundary ssim_loss must be finite zero",
    ):
        migration.migrate_stage0_checkpoint(**case.arguments())
    assert not case.destination.exists()
    assert not case.receipt.exists()
