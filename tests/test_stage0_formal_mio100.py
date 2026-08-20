from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import torch

from scripts import authorize_stage0_formal_mio100 as authorizer
from src.evaluation import mio100 as shared
from src.evaluation import stage0_formal
from src.evaluation import stage0_formal_inventory as inventory_contract
from src.evaluation.formal_inventory import (
    FORMAL_APPROVAL_PATH,
    FormalInventoryError,
    write_new_read_only_json,
)
from src.evaluation.mio100 import ArtifactBinding, FormalAuthorization
from src.metrics.agenticir_official import OFFICIAL_GROUPS, aggregate_official_records
from src.utils.hashing import sha256_file, sha256_json


def _write(path: Path, payload: bytes, *, mode: int = 0o444) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path.resolve()


def _write_json(path: Path, payload: object, *, mode: int = 0o444) -> Path:
    return _write(
        path,
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode(),
        mode=mode,
    )


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _user_authorization(path: Path) -> Path:
    text = "\n".join(
        (
            "# Formal MiO100 Stage0 Control User Authorization",
            "",
            "- Status: USER-AUTHORIZED",
            "- Authorized UTC: 2026-08-20T12:00:00Z",
            f"- Scope: {inventory_contract.STAGE0_USER_AUTHORIZATION_SCOPE}",
            "- Stage0 checkpoint SHA256: "
            f"{inventory_contract.FROZEN_STAGE0_CHECKPOINT_SHA256}",
            "- Stage0 control protocol SHA256: "
            f"{inventory_contract.FROZEN_STAGE0_CONTROL_PROTOCOL_SHA256}",
            "- Training authorized: false",
            "- Checkpoint or threshold selection authorized: false",
            "- Stage4 mutation authorized: false",
            "- TTA or model fusion authorized: false",
            "- Result-driven rerun authorized: false",
            "- Blind-test status restored: false",
            "",
            "## Exact user instruction",
            "Run the one frozen Stage0 control exactly as scoped above.",
            "",
        )
    )
    return _write(path, text.encode())


def test_legacy_stage4_sources_remain_byte_exact_and_roots_are_independent() -> None:
    root = Path(__file__).resolve().parents[1]
    assert sha256_file(root / "src/evaluation/mio100.py") == (
        "f0c64d636821e43e639e1bbbb125db9ee718750a030938032d3d107470a42d50"
    )
    assert sha256_file(root / "src/evaluation/agenticir_table1.py") == (
        "22c8f48607b631ab9ddf2e0565012be2be5f52674eae18e2b2f09ad02faa8d73"
    )
    assert inventory_contract.STAGE0_OUTPUT_ROOT != shared.FORMAL_OUTPUT_ROOT
    assert inventory_contract.STAGE0_APPROVAL_PATH != FORMAL_APPROVAL_PATH
    assert inventory_contract.STAGE0_METHOD_NAME != shared.FORMAL_METHOD_NAME
    assert "stage0_formal_readiness" in (
        inventory_contract.REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    )
    assert "inventory_origin_protocol" in (
        inventory_contract.REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    )
    assert "stage0_control_protocol" in (
        inventory_contract.REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    )
    assert "stage0_user_authorization_protocol" in (
        inventory_contract.REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    )


def test_stage0_approval_binds_exact_external_user_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths: dict[str, Path] = {}
    for index, name in enumerate(
        inventory_contract.REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    ):
        paths[name] = _write(
            tmp_path / "bindings" / f"{index:02d}-{name}.txt",
            f"{name}\n".encode(),
        )

    frozen = {
        "FROZEN_STAGE0_CHECKPOINT_SHA256": "stage0_checkpoint",
        "FROZEN_STAGE0_CONFIG_SHA256": "stage0_config",
        "FORMAL_MANIFEST_SHA256": "formal_manifest",
        "FROZEN_FORMAL_DATA_INVENTORY_SHA256": "formal_data_inventory",
        "FROZEN_METRIC_WEIGHT_INVENTORY_SHA256": "metric_weight_inventory",
        "FROZEN_STAGE4_TABLE1_COMPLETE_SHA256": "stage4_table1_complete",
        "FROZEN_STAGE4_TABLE1_PER_IMAGE_SHA256": "stage4_table1_per_image",
        "FROZEN_STAGE4_TABLE1_SUMMARY_SHA256": "stage4_table1_summary",
        "FROZEN_STAGE0_CONTROL_PROTOCOL_SHA256": "stage0_control_protocol",
    }
    for constant, name in frozen.items():
        monkeypatch.setattr(inventory_contract, constant, sha256_file(paths[name]))
    paths["stage0_user_authorization_protocol"].chmod(0o644)
    paths["stage0_user_authorization_protocol"] = _user_authorization(
        paths["stage0_user_authorization_protocol"]
    )

    payload = inventory_contract.build_stage0_authorization_payload(
        paths, approved_utc="2026-08-20T12:01:00Z"
    )
    approval = (tmp_path / "FORMAL_MIO100_STAGE0_CONTROL_APPROVED.json").resolve()
    write_new_read_only_json(approval, payload)
    accepted = inventory_contract.validate_stage0_lightweight_authorization(
        approval, expected_binding_paths=paths
    )
    assert accepted["bindings"]["stage0_user_authorization_protocol"] == {
        "path": str(paths["stage0_user_authorization_protocol"]),
        "sha256": sha256_file(paths["stage0_user_authorization_protocol"]),
    }
    paths["stage0_user_authorization_protocol"].chmod(0o644)
    with pytest.raises(FormalInventoryError, match="exact mode 0444"):
        inventory_contract.validate_stage0_lightweight_authorization(
            approval, expected_binding_paths=paths
        )


def test_authorizer_rejects_missing_user_evidence_before_gpu_or_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(authorizer, "assert_standard_library_only", lambda: None)

    def forbidden_gpu(*_args: object, **_kwargs: object) -> object:
        pytest.fail("GPU gate must not run before explicit user evidence")

    missing = (tmp_path / "missing-user-authorization.md").resolve()
    with pytest.raises(FormalInventoryError, match="missing Stage0 user authorization"):
        authorizer.run_approval_phase(
            execute_token=authorizer.APPROVAL_EXECUTE_TOKEN,
            manifest=(tmp_path / "manifest.jsonl").resolve(),
            inventory_path=(tmp_path / "inventory.json").resolve(),
            inventory_protocol=(tmp_path / "inventory-protocol.md").resolve(),
            authorization_protocol=(tmp_path / "control.md").resolve(),
            user_authorization_protocol=missing,
            summary=(tmp_path / "summary.json").resolve(),
            checkpoint=(tmp_path / "best_ema.pth").resolve(),
            config=(tmp_path / "config.yaml").resolve(),
            primary_validation=(tmp_path / "validation.json").resolve(),
            calibration_history=(tmp_path / "calibration.csv").resolve(),
            report=(tmp_path / "report.md").resolve(),
            readiness=(tmp_path / "readiness.json").resolve(),
            gpu_runner=forbidden_gpu,
        )


def test_authorizer_never_reads_formal_file_bytes_before_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory_path = _write_json(tmp_path / "inventory.json", {"frozen": True})
    approval_path = (tmp_path / "approval.json").resolve()
    output_root = (tmp_path / "future-output").resolve()
    paths = {
        name: _write(tmp_path / "bindings" / name, f"{name}\n".encode())
        for name in inventory_contract.REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    }
    paths["formal_data_inventory"] = inventory_path
    paths["inventory_origin_protocol"] = (tmp_path / "inventory-protocol.md").resolve()
    paths["stage0_control_protocol"] = (tmp_path / "control.md").resolve()
    paths["stage0_user_authorization_protocol"] = (tmp_path / "user.md").resolve()
    for name in (
        "inventory_origin_protocol",
        "stage0_control_protocol",
        "stage0_user_authorization_protocol",
    ):
        _write(paths[name], f"{name}\n".encode())

    monkeypatch.setattr(authorizer, "assert_standard_library_only", lambda: None)
    monkeypatch.setattr(
        authorizer, "validate_stage0_user_authorization_protocol", lambda *_: {}
    )
    monkeypatch.setattr(
        authorizer, "validate_stage0_ready_without_torch", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        authorizer,
        "validate_stage0_readiness_receipt_without_torch",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        authorizer,
        "build_stage0_authorization_payload",
        lambda *_a, **_k: {"approved": True},
    )

    def inventory_gate(*_args: object, **kwargs: object) -> object:
        assert kwargs["verify_file_bytes"] is False
        return SimpleNamespace(
            path=inventory_path,
            sha256=sha256_file(inventory_path),
        )

    monkeypatch.setattr(authorizer, "load_formal_data_inventory", inventory_gate)
    monkeypatch.setattr(
        authorizer,
        "validate_stage0_lightweight_authorization",
        lambda *_a, **_k: {
            "bindings": {
                "formal_data_inventory": {"sha256": sha256_file(inventory_path)}
            }
        },
    )
    result = authorizer.run_approval_phase(
        execute_token=authorizer.APPROVAL_EXECUTE_TOKEN,
        manifest=(tmp_path / "manifest.jsonl").resolve(),
        inventory_path=inventory_path,
        inventory_protocol=paths["inventory_origin_protocol"],
        authorization_protocol=paths["stage0_control_protocol"],
        user_authorization_protocol=paths["stage0_user_authorization_protocol"],
        summary=(tmp_path / "summary.json").resolve(),
        checkpoint=(tmp_path / "best_ema.pth").resolve(),
        config=(tmp_path / "config.yaml").resolve(),
        primary_validation=(tmp_path / "validation.json").resolve(),
        calibration_history=(tmp_path / "calibration.csv").resolve(),
        report=(tmp_path / "report.md").resolve(),
        readiness=(tmp_path / "readiness.json").resolve(),
        stage1_run_contract=paths["stage1_run_contract"],
        approval_path=approval_path,
        output_root=output_root,
        gpu_runner=lambda *_a, **_k: SimpleNamespace(stdout="", returncode=0),
        binding_paths_override=paths,
    )
    assert result["status"] == "FORMAL_MIO100_STAGE0_CONTROL_APPROVED"
    assert approval_path.is_file()
    assert not output_root.exists()


def _stage0_checkpoint_payload() -> dict[str, Any]:
    model = {
        "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "counter": torch.tensor(7, dtype=torch.int64),
    }
    return {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage0",
        "step": 60_000,
        "model_role": "ema_selection",
        "resumable": False,
        "pending_validation_step": None,
        "model": model,
        "ema": {
            "shadow": {name: value.clone() for name, value in model.items()},
            "num_updates": 60_000,
        },
        "provenance": {
            "protocol_id": inventory_contract.PROTOCOL_ID,
            "stage": "stage0",
            "config_path": "/tmp/stage0.yaml",
            "config_sha256": "a" * 64,
            "resolved_paths_sha256": "b" * 64,
            "semantic_source_sha256": {
                "scripts/train_stage0.py": "1" * 64,
                "src/data/scale_canonicalizer.py": "2" * 64,
                "src/metrics/agenticir_official.py": "3" * 64,
                "src/net/mio_stagea.py": "4" * 64,
                "src/training/stage0_engine.py": "5" * 64,
            },
            "manifests": {
                name: {"path": f"/data/{name}.jsonl", "sha256": "6" * 64}
                for name in ("clean_train", "clean_val", "primary_train", "primary_val")
            },
            "parent_checkpoint": {"path": "/parent.pth", "sha256": "7" * 64},
            "repositories": {
                "agenticir_commit": "9640a291480dee3ba8f2974125d4ee9e3440f3d6",
                "mioir_commit": "4d5f6ca0235cf2c307319673242d5722ee35d73f",
            },
            "runtime": {
                "target_step": 60_000,
                "schedule_max_steps": 60_000,
                "integration": False,
            },
            "compile_ab": {
                "recommend_torch_compile": False,
                "sha256": "8" * 64,
                "profile_script_sha256": "9" * 64,
            },
            "warm_start_load": {
                "source_tensor_count": 495,
                "loaded_count": 495,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            },
            "dependency_versions": {"torch": "synthetic"},
        },
        "metrics": {
            "best_step": 60_000,
            "best_single_psnr": 28.0,
            "best_single_ssim": 0.87,
            "best_group_a_psnr": 24.8,
            "best_group_a_ssim": 0.78,
        },
    }


def test_cpu_readiness_binds_ema_summary_calibration_and_stage1_parent(
    tmp_path: Path,
) -> None:
    assert torch.cuda.is_initialized() is False
    config = _write(tmp_path / "stage0.yaml", b"stage0: frozen\n", mode=0o644)
    config_sha = sha256_file(config)
    source_names = {
        "scripts/train_stage0.py",
        "src/data/scale_canonicalizer.py",
        "src/metrics/agenticir_official.py",
        "src/net/mio_stagea.py",
        "src/training/stage0_engine.py",
        *stage0_formal.STAGE0_PROVENANCE_COMPATIBILITY,
    }
    semantic_sources = {}
    source_payloads = {}
    for index, name in enumerate(sorted(source_names)):
        payload_bytes = f"synthetic source {index}: {name}\n".encode()
        source = _write(tmp_path / name, payload_bytes, mode=0o644)
        source_payloads[name] = payload_bytes
        semantic_sources[name] = (
            _sha_bytes(f"historical {name}\n".encode())
            if name in stage0_formal.STAGE0_PROVENANCE_COMPATIBILITY
            else sha256_file(source)
        )
    synthetic_compatibility = {
        name: {
            "checkpoint_sha256": semantic_sources[name],
            "current_sha256": sha256_file(tmp_path / name),
            "rationale": "post_stage0_downstream_not_imported_by_formal_stage0",
        }
        for name in stage0_formal.STAGE0_PROVENANCE_COMPATIBILITY
    }
    provenance_manifests = {}
    for name in ("clean_train", "clean_val", "primary_train", "primary_val"):
        manifest = _write(
            tmp_path / "training-manifests" / f"{name}.jsonl",
            f"{name}\n".encode(),
            mode=0o644,
        )
        provenance_manifests[name] = {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        }
    checkpoint = (tmp_path / "best_ema.pth").resolve()
    payload = _stage0_checkpoint_payload()
    payload["provenance"]["config_path"] = str(config)
    payload["provenance"]["config_sha256"] = config_sha
    payload["provenance"]["semantic_source_sha256"] = semantic_sources
    payload["provenance"]["manifests"] = provenance_manifests
    synthetic_digest_expectations = {
        "expected_provenance_sha256": sha256_json(payload["provenance"]),
        "expected_semantic_source_map_sha256": sha256_json(semantic_sources),
        "expected_manifest_map_sha256": sha256_json(provenance_manifests),
    }
    torch.save(payload, checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()

    validation = {
        "schema_version": "graphrestore-stage0-primary-val-v1",
        "protocol_id": "agenticir_official_parity",
        "step": 60_000,
        "image_count": 1_600,
        "single_psnr": 28.0,
        "single_ssim": 0.87,
        "group_a_psnr": 24.8,
        "group_a_ssim": 0.78,
        "task_means": {"synthetic": {"psnr": 24.8, "ssim": 0.78}},
    }
    validation_path = _write_json(tmp_path / "validation.json", validation)
    summary = {
        "schema_version": "graphrestore-stage0-run-v1",
        "protocol_id": inventory_contract.PROTOCOL_ID,
        "completed_step": 60_000,
        "target_step": 60_000,
        "integration": False,
        "finite": True,
        "best_checkpoint": str(checkpoint),
        "maximum_train_peak_reserved_fraction": 0.5,
        "maximum_validation_peak_reserved_fraction": 0.4,
        "runtime": {
            "schedule_max_steps": 60_000,
            "target_step": 60_000,
            "integration": False,
        },
        "validation": validation,
    }
    summary_path = _write_json(tmp_path / "summary.json", summary)
    calibration = _write(
        tmp_path / "calibration.csv",
        (
            "step,single_psnr,single_ssim,group_a_psnr,group_a_ssim\n"
            "60000,28.0,0.87,24.8,0.78\n"
        ).encode(),
        mode=0o644,
    )
    report = _write(tmp_path / "STAGE0.md", b"frozen report\n", mode=0o644)
    stage1_contract = _write_json(
        tmp_path / "stage1-run-contract.json",
        {
            "provenance": {
                "parent_checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "source": "stage0_best_ema_shadow",
                    "allowed_new_prefixes": ["decoder.skill_bank."],
                }
            },
            "stage0_backbone_load": {
                "source_tensor_count": 2,
                "loaded_count": 2,
                "missing_count": 3,
                "missing_prefixes": ["decoder.skill_bank."],
                "unexpected_keys": [],
                "shape_mismatches": [],
            },
        },
    )
    receipt = stage0_formal.build_stage0_readiness_payload(
        checkpoint_path=checkpoint,
        config_path=config,
        summary_path=summary_path,
        primary_validation_path=validation_path,
        calibration_history_path=calibration,
        report_path=report,
        stage1_run_contract_path=stage1_contract,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_config_sha256=config_sha,
        expected_tensor_count=2,
        expected_stage1_missing_count=3,
        expected_project_root=tmp_path,
        expected_semantic_source_count=len(source_names),
        expected_provenance_compatibility=synthetic_compatibility,
        **synthetic_digest_expectations,
    )
    readiness = tmp_path / "readiness.json"
    write_new_read_only_json(readiness, receipt)
    accepted = inventory_contract.validate_stage0_readiness_receipt_without_torch(
        readiness,
        checkpoint_path=checkpoint,
        config_path=config,
        summary_path=summary_path,
        primary_validation_path=validation_path,
        calibration_history_path=calibration,
        report_path=report,
        stage1_run_contract_path=stage1_contract,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_config_sha256=config_sha,
        expected_tensor_count=2,
        expected_stage1_missing_count=3,
        expected_project_root=tmp_path,
        expected_semantic_source_count=len(source_names),
        expected_provenance_compatibility=synthetic_compatibility,
        **synthetic_digest_expectations,
    )
    assert accepted["checkpoint"]["model_equals_ema_shadow"] is True
    assert accepted["stage1_parent_receipt"]["loaded_count"] == 2
    assert len(accepted["provenance_verification"]["semantic_sources"]) == len(
        source_names
    )
    assert set(accepted["provenance_verification"]["compatibility_mismatches"]) == set(
        stage0_formal.STAGE0_PROVENANCE_COMPATIBILITY
    )
    assert accepted["cuda_initialized_before"] is False
    assert accepted["cuda_initialized_after"] is False
    assert torch.cuda.is_initialized() is False

    forged_receipt_payload = json.loads(json.dumps(receipt))
    forged_receipt_payload["provenance_verification"]["compatibility_mismatches"][
        "src/training/stage4_engine.py"
    ]["checkpoint_sha256"] = "e" * 64
    forged_readiness = tmp_path / "forged-readiness.json"
    write_new_read_only_json(forged_readiness, forged_receipt_payload)
    with pytest.raises(FormalInventoryError, match="provenance compatibility drifted"):
        inventory_contract.validate_stage0_readiness_receipt_without_torch(
            forged_readiness,
            checkpoint_path=checkpoint,
            config_path=config,
            summary_path=summary_path,
            primary_validation_path=validation_path,
            calibration_history_path=calibration,
            report_path=report,
            stage1_run_contract_path=stage1_contract,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
            expected_tensor_count=2,
            expected_stage1_missing_count=3,
            expected_project_root=tmp_path,
            expected_semantic_source_count=len(source_names),
            expected_provenance_compatibility=synthetic_compatibility,
            **synthetic_digest_expectations,
        )

    replacement_source = _write(
        tmp_path / "pyproject.toml",
        b"[project]\nname='forged-replacement'\n",
        mode=0o644,
    )
    substituted_source_receipt = json.loads(json.dumps(receipt))
    substituted_sources = substituted_source_receipt["provenance_verification"][
        "semantic_sources"
    ]
    del substituted_sources["src/net/mio_stagea.py"]
    substituted_sources["pyproject.toml"] = {
        "path": str(replacement_source),
        "sha256": sha256_file(replacement_source),
    }
    substituted_source_readiness = tmp_path / "substituted-source-readiness.json"
    write_new_read_only_json(substituted_source_readiness, substituted_source_receipt)
    with pytest.raises(
        FormalInventoryError, match="semantic-source identity map drifted"
    ):
        inventory_contract.validate_stage0_readiness_receipt_without_torch(
            substituted_source_readiness,
            checkpoint_path=checkpoint,
            config_path=config,
            summary_path=summary_path,
            primary_validation_path=validation_path,
            calibration_history_path=calibration,
            report_path=report,
            stage1_run_contract_path=stage1_contract,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
            expected_tensor_count=2,
            expected_stage1_missing_count=3,
            expected_project_root=tmp_path,
            expected_semantic_source_count=len(source_names),
            expected_provenance_compatibility=synthetic_compatibility,
            **synthetic_digest_expectations,
        )

    replacement_manifest = _write(
        tmp_path / "replacement-primary-val.jsonl",
        Path(provenance_manifests["primary_val"]["path"]).read_bytes(),
        mode=0o644,
    )
    substituted_manifest_receipt = json.loads(json.dumps(receipt))
    substituted_manifest_receipt["provenance_verification"]["manifests"][
        "primary_val"
    ] = {
        "path": str(replacement_manifest),
        "sha256": sha256_file(replacement_manifest),
    }
    substituted_manifest_readiness = tmp_path / "substituted-manifest-readiness.json"
    write_new_read_only_json(
        substituted_manifest_readiness, substituted_manifest_receipt
    )
    with pytest.raises(FormalInventoryError, match="manifest identity map drifted"):
        inventory_contract.validate_stage0_readiness_receipt_without_torch(
            substituted_manifest_readiness,
            checkpoint_path=checkpoint,
            config_path=config,
            summary_path=summary_path,
            primary_validation_path=validation_path,
            calibration_history_path=calibration,
            report_path=report,
            stage1_run_contract_path=stage1_contract,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
            expected_tensor_count=2,
            expected_stage1_missing_count=3,
            expected_project_root=tmp_path,
            expected_semantic_source_count=len(source_names),
            expected_provenance_compatibility=synthetic_compatibility,
            **synthetic_digest_expectations,
        )

    forward_source = tmp_path / "src/net/mio_stagea.py"
    forward_source.write_bytes(b"shape-preserving forward implementation drift\n")
    with pytest.raises(
        shared.MiO100EvaluationError, match="semantic source bytes drifted"
    ):
        stage0_formal.build_stage0_readiness_payload(
            checkpoint_path=checkpoint,
            config_path=config,
            summary_path=summary_path,
            primary_validation_path=validation_path,
            calibration_history_path=calibration,
            report_path=report,
            stage1_run_contract_path=stage1_contract,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
            expected_tensor_count=2,
            expected_stage1_missing_count=3,
            expected_project_root=tmp_path,
            expected_semantic_source_count=len(source_names),
            expected_provenance_compatibility=synthetic_compatibility,
            **synthetic_digest_expectations,
        )
    with pytest.raises(FormalInventoryError, match="semantic source bytes drifted"):
        inventory_contract.validate_stage0_readiness_receipt_without_torch(
            readiness,
            checkpoint_path=checkpoint,
            config_path=config,
            summary_path=summary_path,
            primary_validation_path=validation_path,
            calibration_history_path=calibration,
            report_path=report,
            stage1_run_contract_path=stage1_contract,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
            expected_tensor_count=2,
            expected_stage1_missing_count=3,
            expected_project_root=tmp_path,
            expected_semantic_source_count=len(source_names),
            expected_provenance_compatibility=synthetic_compatibility,
            **synthetic_digest_expectations,
        )
    forward_source.write_bytes(source_payloads["src/net/mio_stagea.py"])

    compatibility_source = tmp_path / "src/training/stage4_engine.py"
    compatibility_source.write_bytes(b"allowlisted current source drift\n")
    with pytest.raises(
        shared.MiO100EvaluationError, match="provenance compatibility drifted"
    ):
        stage0_formal.build_stage0_readiness_payload(
            checkpoint_path=checkpoint,
            config_path=config,
            summary_path=summary_path,
            primary_validation_path=validation_path,
            calibration_history_path=calibration,
            report_path=report,
            stage1_run_contract_path=stage1_contract,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
            expected_tensor_count=2,
            expected_stage1_missing_count=3,
            expected_project_root=tmp_path,
            expected_semantic_source_count=len(source_names),
            expected_provenance_compatibility=synthetic_compatibility,
            **synthetic_digest_expectations,
        )
    compatibility_source.write_bytes(source_payloads["src/training/stage4_engine.py"])

    compatibility_name = "src/training/stage4_engine.py"
    historical_sha = semantic_sources[compatibility_name]
    for forged_checkpoint_sha in (
        synthetic_compatibility[compatibility_name]["current_sha256"],
        "f" * 64,
    ):
        payload["provenance"]["semantic_source_sha256"][compatibility_name] = (
            forged_checkpoint_sha
        )
        torch.save(payload, checkpoint)
        with pytest.raises(
            shared.MiO100EvaluationError,
            match="whole provenance digest drifted",
        ):
            stage0_formal.build_stage0_readiness_payload(
                checkpoint_path=checkpoint,
                config_path=config,
                summary_path=summary_path,
                primary_validation_path=validation_path,
                calibration_history_path=calibration,
                report_path=report,
                stage1_run_contract_path=stage1_contract,
                expected_checkpoint_sha256=sha256_file(checkpoint),
                expected_config_sha256=config_sha,
                expected_tensor_count=2,
                expected_stage1_missing_count=3,
                expected_project_root=tmp_path,
                expected_semantic_source_count=len(source_names),
                expected_provenance_compatibility=synthetic_compatibility,
                **synthetic_digest_expectations,
            )
    payload["provenance"]["semantic_source_sha256"][compatibility_name] = historical_sha
    checkpoint.write_bytes(checkpoint_bytes)
    assert sha256_file(checkpoint) == checkpoint_sha

    primary_manifest = Path(provenance_manifests["primary_val"]["path"])
    primary_manifest.write_bytes(b"manifest drift\n")
    with pytest.raises(shared.MiO100EvaluationError, match="manifest bytes drifted"):
        stage0_formal.build_stage0_readiness_payload(
            checkpoint_path=checkpoint,
            config_path=config,
            summary_path=summary_path,
            primary_validation_path=validation_path,
            calibration_history_path=calibration,
            report_path=report,
            stage1_run_contract_path=stage1_contract,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
            expected_tensor_count=2,
            expected_stage1_missing_count=3,
            expected_project_root=tmp_path,
            expected_semantic_source_count=len(source_names),
            expected_provenance_compatibility=synthetic_compatibility,
            **synthetic_digest_expectations,
        )


def _make_table1_complete(
    path: Path,
    *,
    per_image: Path,
    summary: Path,
    authorization: ArtifactBinding,
    image_count: int,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "graphrestore.agenticir_table1_complete.v1",
            "status": "COMPLETE",
            "image_count": image_count,
            "no_selective_rerun": True,
            "all_values_finite": True,
            "maximum_peak_reserved_fraction": 0.5,
            "per_image": {
                "path": str(per_image),
                "sha256": sha256_file(per_image),
            },
            "summary": {"path": str(summary), "sha256": sha256_file(summary)},
            "formal_evidence": {
                "authorization": {
                    "path": str(authorization.path),
                    "sha256": authorization.sha256,
                }
            },
        },
    )


def _table1_csv(path: Path, *, stage4: bool) -> Path:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=stage0_formal.SCORER_CSV_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    index = 0
    for group, combinations in OFFICIAL_GROUPS.items():
        for combination in combinations:
            gain = 1.0 if stage4 else 0.0
            writer.writerow(
                {
                    "sample_id": f"test/{group}/{combination}/{index:03d}",
                    "group": group,
                    "combination": combination,
                    "prediction_png": f"/prediction/{'stage4' if stage4 else 'stage0'}/{index}.png",
                    "prediction_sha256": f"{index + (100 if stage4 else 0):064x}",
                    "target_png": f"/target/{index}.png",
                    "target_sha256": f"{index + 200:064x}",
                    "psnr": 20.0 + 0.3 * gain,
                    "ssim": 0.7 + 0.01 * gain,
                    "lpips": 0.4 - 0.1 * gain,
                    "maniqa": 0.2 + 0.02 * gain,
                    "clipiqa": 0.3 + 0.02 * gain,
                    "musiq": 40.0 + 2.0 * gain,
                }
            )
            index += 1
    return _write(path, stream.getvalue().encode())


def test_six_metric_paired_stage4_minus_stage0_comparison_is_bound(
    tmp_path: Path,
) -> None:
    stage0_csv = _table1_csv(tmp_path / "stage0/per_image.csv", stage4=False)
    stage4_csv = _table1_csv(tmp_path / "stage4/per_image.csv", stage4=True)
    stage0_summary = _write_json(tmp_path / "stage0/summary.json", {"ok": True})
    stage4_summary = _write_json(tmp_path / "stage4/summary.json", {"ok": True})
    stage0_auth_path = _write_json(tmp_path / "stage0-approval.json", {"ok": True})
    stage4_auth_path = _write_json(tmp_path / "stage4-approval.json", {"ok": True})
    stage0_auth = ArtifactBinding(stage0_auth_path, sha256_file(stage0_auth_path))
    stage4_auth = ArtifactBinding(stage4_auth_path, sha256_file(stage4_auth_path))
    stage0_complete = _make_table1_complete(
        tmp_path / "stage0/complete.json",
        per_image=stage0_csv,
        summary=stage0_summary,
        authorization=stage0_auth,
        image_count=16,
    )
    stage4_complete = _make_table1_complete(
        tmp_path / "stage4/complete.json",
        per_image=stage4_csv,
        summary=stage4_summary,
        authorization=stage4_auth,
        image_count=16,
    )
    authorization = FormalAuthorization(
        path=stage0_auth_path,
        sha256=sha256_file(stage0_auth_path),
        approved_utc="2026-08-20T12:00:00Z",
        output_root=tmp_path / "unused",
        method_name=inventory_contract.STAGE0_METHOD_NAME,
        shard_count=1,
        bindings={
            "stage4_formal_authorization": stage4_auth,
            "stage4_table1_complete": ArtifactBinding(
                stage4_complete, sha256_file(stage4_complete)
            ),
            "stage4_table1_per_image": ArtifactBinding(
                stage4_csv, sha256_file(stage4_csv)
            ),
            "stage4_table1_summary": ArtifactBinding(
                stage4_summary, sha256_file(stage4_summary)
            ),
        },
    )
    result = stage0_formal.publish_stage0_vs_stage4_comparison(
        stage0_per_image=stage0_csv,
        stage4_per_image=stage4_csv,
        stage0_table1_complete=stage0_complete,
        stage4_table1_complete=stage4_complete,
        stage4_table1_summary=stage4_summary,
        output_root=tmp_path / "comparison",
        authorization=authorization,
        expected_count=16,
        expected_combination_counts={
            combination: 1
            for combinations in OFFICIAL_GROUPS.values()
            for combination in combinations
        },
        enforce_fixed_root=False,
    )
    assert result["decision"] == "PASS_INCREMENTAL_EFFICACY"
    summary = json.loads((tmp_path / "comparison/summary.json").read_text())
    assert summary["groups"]["A"]["stage4_minus_stage0"]["psnr"] == pytest.approx(0.3)
    assert summary["groups"]["B"]["oriented_stage4_gain"]["lpips"] == (
        pytest.approx(0.1)
    )
    assert summary["stage0_parameter_count"] == 25_437_220
    assert set(summary["stage0_graph_diagnostic_columns"]) == {
        "program_levels",
        "parallel_levels",
        "active_skill_calls",
        "reentry_requests",
        "unexpected_activations",
        "precycle_graphs",
        "dropped_edges",
    }
    assert all(
        value == "N/A (prompt-free Stage0 compatibility placeholder)"
        for value in summary["stage0_graph_diagnostic_columns"].values()
    )
    comparison_root = tmp_path / "comparison"
    assert {path.name for path in comparison_root.iterdir()} == {
        "paired_per_image.csv",
        "summary.json",
        "complete.json",
    }
    assert all(
        path.stat().st_mode & 0o777 == 0o444 for path in comparison_root.iterdir()
    )
    verified = stage0_formal.validate_stage0_vs_stage4_comparison_complete(
        comparison_root / "complete.json",
        stage0_table1_complete=stage0_complete,
        stage4_table1_complete=stage4_complete,
        stage4_table1_summary=stage4_summary,
        authorization=authorization,
        output_root=comparison_root,
        expected_count=16,
        expected_combination_counts={
            combination: 1
            for combinations in OFFICIAL_GROUPS.values()
            for combination in combinations
        },
        enforce_fixed_root=False,
    )
    assert verified == result

    unexpected = _write(comparison_root / "unregistered.json", b"{}\n")
    with pytest.raises(shared.MiO100EvaluationError, match="tree drifted"):
        stage0_formal.validate_stage0_vs_stage4_comparison_complete(
            comparison_root / "complete.json",
            stage0_table1_complete=stage0_complete,
            stage4_table1_complete=stage4_complete,
            stage4_table1_summary=stage4_summary,
            authorization=authorization,
            output_root=comparison_root,
            expected_count=16,
            expected_combination_counts={
                combination: 1
                for combinations in OFFICIAL_GROUPS.values()
                for combination in combinations
            },
            enforce_fixed_root=False,
        )
    unexpected.unlink()

    comparison_summary_path = comparison_root / "summary.json"
    comparison_summary_path.chmod(0o644)
    comparison_summary_path.write_text("{}\n", encoding="utf-8")
    comparison_summary_path.chmod(0o444)
    with pytest.raises(shared.MiO100EvaluationError, match="summary drifted"):
        stage0_formal.validate_stage0_vs_stage4_comparison_complete(
            comparison_root / "complete.json",
            stage0_table1_complete=stage0_complete,
            stage4_table1_complete=stage4_complete,
            stage4_table1_summary=stage4_summary,
            authorization=authorization,
            output_root=comparison_root,
            expected_count=16,
            expected_combination_counts={
                combination: 1
                for combinations in OFFICIAL_GROUPS.values()
                for combination in combinations
            },
            enforce_fixed_root=False,
        )
    assert set(stage0_formal.METRICS) == {
        "psnr",
        "ssim",
        "lpips",
        "maniqa",
        "clipiqa",
        "musiq",
    }


def test_stage0_model_parameter_count_is_frozen() -> None:
    from src.net.mio_stagea import MiOStageA

    assert torch.cuda.is_initialized() is False
    model = MiOStageA(gradient_checkpointing=False)
    assert sum(parameter.numel() for parameter in model.parameters()) == 25_437_220
    assert stage0_formal.STAGE0_PARAMETER_COUNT == 25_437_220
    assert torch.cuda.is_initialized() is False


def test_stage0_scorer_has_fixed_formal_roots_and_rejects_fake_worker() -> None:
    from scripts import score_stage0_agenticir_table1 as score_cli

    assert torch.cuda.is_initialized() is False
    scorer = score_cli.SCORER
    assert scorer.FORMAL_AUTHORIZATION_PATH == inventory_contract.STAGE0_APPROVAL_PATH
    assert scorer.FORMAL_EVALUATOR_ROOT == inventory_contract.STAGE0_OUTPUT_ROOT
    assert scorer.FORMAL_SCORE_ROOT == inventory_contract.STAGE0_SCORE_ROOT
    assert scorer.FORMAL_EVALUATOR_ROOT != shared.FORMAL_OUTPUT_ROOT
    with pytest.raises(
        scorer.Table1ContractError, match="forbids the internal worker-launcher"
    ):
        scorer.score_table1(
            input_manifest=scorer.FORMAL_TABLE1_INPUT_PATH,
            output_root=scorer.FORMAL_SCORE_ROOT,
            cache_root=scorer.DEFAULT_CACHE_ROOT,
            reference_python=scorer.DEFAULT_REFERENCE_PYTHON,
            source_paths=scorer.default_source_paths(),
            device=scorer.FORMAL_DEVICE,
            shard_size=scorer.FORMAL_SHARD_SIZE,
            worker_launcher=lambda *_args: None,
            enforce_formal=True,
        )
    assert torch.cuda.is_initialized() is False


def test_stage0_scorer_hashes_legacy_bytes_before_module_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import score_stage0_agenticir_table1 as score_cli

    tampered = _write(
        tmp_path / "agenticir_table1.py",
        b"raise RuntimeError('this source must never execute')\n",
        mode=0o444,
    )

    def forbidden_spec(*_args: object, **_kwargs: object) -> object:
        pytest.fail("drifted scorer source reached exec_module setup")

    monkeypatch.setattr(
        score_cli.importlib.util, "spec_from_file_location", forbidden_spec
    )
    cuda_before = torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match="SHA256 drifted"):
        score_cli._load_fixed_scorer(tampered)  # noqa: SLF001
    assert torch.cuda.is_initialized() is cuda_before is False
    assert list(tmp_path.iterdir()) == [tampered]


def test_stage0_formal_score_cli_routes_only_canonical_internal_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import score_stage0_agenticir_table1 as score_cli

    score_root = (tmp_path / "scores").resolve()
    request_root = score_root / ".worker"
    request_root.mkdir(parents=True)
    request = _write_json(request_root / "request-00000.json", {})
    score_work = (tmp_path / "score-work").resolve()
    score_result_parent = score_work / "agenticir-table1-score"
    score_result_parent.mkdir(parents=True)
    score_result = score_result_parent / "result.json"
    cache_root = (tmp_path / "cache" / "weights").resolve()
    inspect_work = cache_root.parent / ".agenticir_table1_check_work"
    inspect_result_parent = inspect_work / "agenticir-table1-inspect"
    inspect_result_parent.mkdir(parents=True)
    inspect_result = inspect_result_parent / "result.json"
    monkeypatch.setattr(score_cli.SCORER, "FORMAL_SCORE_ROOT", score_root)
    monkeypatch.setattr(score_cli.SCORER, "FORMAL_WORK_ROOT", score_work)
    monkeypatch.setattr(score_cli.SCORER, "DEFAULT_CACHE_ROOT", cache_root)
    calls: list[list[str]] = []

    def worker_main(arguments: Sequence[str]) -> int:
        calls.append(list(arguments))
        return 0

    monkeypatch.setattr(score_cli.SCORER, "main", worker_main)
    inspect_arguments = [
        "_worker-inspect",
        "--worker-result",
        str(inspect_result),
    ]
    score_arguments = [
        "_worker-score",
        "--request",
        str(request),
        "--worker-result",
        str(score_result),
    ]
    assert score_cli.main(inspect_arguments) == 0
    assert score_cli.main(score_arguments) == 0
    assert calls == [inspect_arguments, score_arguments]

    call_count = len(calls)
    assert (
        score_cli.main(
            [
                "_worker-score",
                "--request",
                str(tmp_path / "forged-request.json"),
                "--worker-result",
                str(score_result),
            ]
        )
        == 2
    )
    assert score_cli.main(["score", "--output", "/tmp/forbidden"]) == 2
    assert (
        score_cli.main(
            ["_worker-inspect", "--worker-result", str(tmp_path / "escape.json")]
        )
        == 2
    )
    assert len(calls) == call_count

    monkeypatch.setattr(
        score_cli.SCORER,
        "main",
        lambda *_a, **_k: pytest.fail("legacy variable-path parser was reached"),
    )
    cuda_before = torch.cuda.is_initialized()
    assert score_cli.main(["_worker-prefetch", "--cache-root", "/tmp"]) == 2
    assert torch.cuda.is_initialized() is cuda_before is False


def _synthetic_stage0_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, FormalAuthorization, dict[str, Path]]:
    root = (tmp_path / "evaluator").resolve()
    root.mkdir()
    bound_paths = {
        name: _write(tmp_path / "bindings" / f"{name}.txt", f"{name}\n".encode())
        for name in inventory_contract.REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    }
    authorization_path = _write_json(tmp_path / "approval.json", {"ok": True})
    bindings = {
        name: ArtifactBinding(path, sha256_file(path))
        for name, path in bound_paths.items()
    }
    authorization = FormalAuthorization(
        path=authorization_path,
        sha256=sha256_file(authorization_path),
        approved_utc="2026-08-20T12:00:00Z",
        output_root=root,
        method_name=inventory_contract.STAGE0_METHOD_NAME,
        shard_count=1,
        bindings=bindings,
    )
    rows = []
    records = []
    identities = []
    metric_rows = []
    runtime_rows = []
    digest_rows = []
    table_rows = []
    index = 0
    for group, combinations in OFFICIAL_GROUPS.items():
        for combination in combinations:
            sample_id = f"test/{group}/{combination}/{index:03d}"
            target = _write(tmp_path / "targets" / f"{index:03d}.png", b"target")
            native = _write(tmp_path / "native" / f"{index:03d}.png", b"native")
            depth = "d3" if group == "C" else "d2"
            output_name = f"{index:03d}.png"
            prediction = _write(
                root
                / "methods"
                / inventory_contract.STAGE0_METHOD_NAME
                / depth
                / combination
                / output_name,
                f"prediction-{index}".encode(),
            )
            target_sha = sha256_file(target)
            prediction_sha = sha256_file(prediction)
            row_sha = f"{index + 300:064x}"
            record = SimpleNamespace(
                index=index,
                sample_id=sample_id,
                group=group,
                combination=combination,
                clean_id=f"{index:03d}",
                target_path=target,
                native_lq_path=native,
                row_sha256=row_sha,
                depth_dir=depth,
                output_filename=output_name,
            )
            identity = SimpleNamespace(
                index=index,
                sample_id=sample_id,
                target_path=target,
                native_lq_path=native,
                row_sha256=row_sha,
                target_sha256=target_sha,
            )
            csv_row = {
                "sample_id": sample_id,
                "group": group,
                "combination": combination,
                "clean_id": f"{index:03d}",
                "prediction_png": str(prediction),
                "prediction_sha256": prediction_sha,
                "target_png": str(target),
                "target_sha256": target_sha,
                "psnr": "20.0",
                "ssim": "0.7",
                "latency_ms": "1.0",
                "program_levels": "0",
                "parallel_levels": "0",
                "active_skill_calls": "0",
                "reentry_requests": "0",
                "unexpected_activations": "0",
                "precycle_graphs": "0",
                "dropped_edges": "0",
                "peak_reserved_fraction": "0.1",
            }
            rows.append(csv_row)
            records.append(record)
            identities.append(identity)
            metric_rows.append({"combination": combination, "psnr": 20.0, "ssim": 0.7})
            runtime_rows.append(
                {
                    "latency_ms": 1.0,
                    "peak_reserved_fraction": 0.1,
                    "diagnostics": {
                        name: 0
                        for name in shared._CSV_COLUMNS[11:18]  # noqa: SLF001
                    },
                }
            )
            digest_rows.append(
                {
                    "sample_id": sample_id,
                    "prediction_sha256": prediction_sha,
                    "target_sha256": target_sha,
                }
            )
            table_rows.append(
                {
                    "schema_version": shared.TABLE1_INPUT_SCHEMA,
                    "sample_id": sample_id,
                    "group": group,
                    "combination": combination,
                    "prediction_png": str(prediction),
                    "prediction_sha256": prediction_sha,
                    "target_png": str(target),
                    "target_sha256": target_sha,
                }
            )
            index += 1
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_stream,
        fieldnames=shared._CSV_COLUMNS,
        lineterminator="\n",  # noqa: SLF001
    )
    writer.writeheader()
    writer.writerows(rows)
    per_image = _write(root / "per_image.csv", csv_stream.getvalue().encode())
    table = _write(
        root / "table1_input.jsonl",
        (
            "\n".join(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in table_rows
            )
            + "\n"
        ).encode(),
    )
    inventory_raw = {
        "path": str(bindings["formal_data_inventory"].path),
        "sha256": bindings["formal_data_inventory"].sha256,
        "rows_digest": "a" * 64,
        "files_digest": "b" * 64,
    }
    created = "2026-08-20T12:00:00Z"
    contract = {
        "schema_version": stage0_formal.STAGE0_RUN_CONTRACT_SCHEMA,
        "protocol_id": inventory_contract.PROTOCOL_ID,
        "authorization": {
            "path": str(authorization.path),
            "sha256": authorization.sha256,
        },
        "authorization_bindings": {
            name: {"path": str(binding.path), "sha256": binding.sha256}
            for name, binding in sorted(bindings.items())
        },
        "manifest_sha256": bindings["formal_manifest"].sha256,
        "formal_data_inventory": inventory_raw,
        "checkpoint_sha256": bindings["stage0_checkpoint"].sha256,
        "config_sha256": bindings["stage0_config"].sha256,
        "method_name": inventory_contract.STAGE0_METHOD_NAME,
        "output_root": str(root),
        "manifest_row_count": 16,
        "groups": {"A": 8, "B": 4, "C": 4},
        "combination_counts": {
            combination: 1
            for combinations in OFFICIAL_GROUPS.values()
            for combination in combinations
        },
        "shard_count": 1,
        "assignment": "manifest_index_mod_shard_count",
        "inference": dict(stage0_formal.STAGE0_FORMAL_INFERENCE),
        "output_protocol": {
            "crop": "top_left_to_gt_shape",
            "quantization": "clamp_round_uint8",
            "encoding": "lossless_png",
            "score_source": "png_readback",
            "layout": "methods/<method>/d2|d3/<combination>/<gt_basename>",
            "overwrite": False,
        },
        "vram_maximum_peak_reserved_fraction": 0.90,
        "created_utc": created,
    }
    run_contract = _write_json(root / "run_contract.json", contract)
    predictions_digest = sha256_json(digest_rows)
    summary = {
        "schema_version": shared.SUMMARY_SCHEMA,
        "protocol_id": inventory_contract.PROTOCOL_ID,
        "created_utc": created,
        "method_name": inventory_contract.STAGE0_METHOD_NAME,
        "image_count": 16,
        "manifest_sha256": bindings["formal_manifest"].sha256,
        "formal_data_inventory": inventory_raw,
        "checkpoint_sha256": bindings["stage0_checkpoint"].sha256,
        "authorization_sha256": authorization.sha256,
        "run_contract_sha256": sha256_file(run_contract),
        "predictions_digest": predictions_digest,
        "aggregation": aggregate_official_records(
            metric_rows,
            required_combinations=tuple(
                combination
                for combinations in OFFICIAL_GROUPS.values()
                for combination in combinations
            ),
            expected_counts={
                combination: 1
                for combinations in OFFICIAL_GROUPS.values()
                for combination in combinations
            },
        ),
        "runtime": shared._aggregate_runtime(runtime_rows),  # noqa: SLF001
        "metric_protocol": {
            "prediction_source": "lossless_png_readback",
            "psnr": "AgenticIR/pyiqa-0.1.10 RGB parity",
            "ssim": "AgenticIR/pyiqa-0.1.10 Y parity",
            "group_reduction": "equal_combination_mean",
            "weighted_all_images": "additional_only",
        },
        "outputs": {
            "agenticir_methods_root": str(
                root / "methods" / inventory_contract.STAGE0_METHOD_NAME
            ),
            "per_image_csv": str(per_image),
            "table1_input_jsonl": str(table),
        },
    }
    summary_path = _write_json(root / "summary.json", summary)
    complete = {
        "schema_version": shared.COMPLETE_SCHEMA,
        "protocol_id": inventory_contract.PROTOCOL_ID,
        "created_utc": created,
        "status": "COMPLETE",
        "image_count": 16,
        "method_name": inventory_contract.STAGE0_METHOD_NAME,
        "authorization_sha256": authorization.sha256,
        "run_contract_sha256": sha256_file(run_contract),
        "checkpoint_sha256": bindings["stage0_checkpoint"].sha256,
        "manifest_sha256": bindings["formal_manifest"].sha256,
        "formal_data_inventory": inventory_raw,
        "predictions_digest": predictions_digest,
        "bindings": {
            "run_contract": {
                "path": str(run_contract),
                "sha256": sha256_file(run_contract),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "per_image_csv": {
                "path": str(per_image),
                "sha256": sha256_file(per_image),
            },
            "table1_input_jsonl": {
                "path": str(table),
                "sha256": sha256_file(table),
            },
        },
    }
    complete_path = _write_json(root / "complete.json", complete)
    monkeypatch.setattr(
        stage0_formal,
        "validate_stage0_formal_authorization",
        lambda *_args, **_kwargs: authorization,
    )
    monkeypatch.setattr(
        stage0_formal,
        "validate_stage0_readiness_receipt_without_torch",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        stage0_formal,
        "load_formal_data_inventory",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha256=bindings["formal_data_inventory"].sha256,
            rows_digest="a" * 64,
            files_digest="b" * 64,
            rows=tuple(identities),
        ),
    )
    monkeypatch.setattr(
        shared, "load_formal_manifest", lambda *_args, **_kwargs: tuple(records)
    )
    return complete_path, authorization, bound_paths


def test_stage0_evaluator_completion_is_independent_and_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete, _authorization, bindings = _synthetic_stage0_completion(
        tmp_path, monkeypatch
    )
    counts = {
        combination: 1
        for combinations in OFFICIAL_GROUPS.values()
        for combination in combinations
    }
    result = stage0_formal.validate_stage0_evaluator_complete(
        complete,
        authorization_path=tmp_path / "approval.json",
        expected_bindings=bindings,
        verify_data_files=False,
        expected_output_root=tmp_path / "evaluator",
        expected_row_count=16,
        expected_group_counts={"A": 8, "B": 4, "C": 4},
        expected_combination_counts=counts,
        validate_protocol=False,
    )
    assert result.complete_path == complete
    assert (
        result.predictions_digest
        == json.loads(complete.read_text())["predictions_digest"]
    )
    summary = tmp_path / "evaluator/summary.json"
    summary.chmod(0o644)
    summary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        shared.MiO100EvaluationError, match="immutable/read-only|binding drifted"
    ):
        stage0_formal.validate_stage0_evaluator_complete(
            complete,
            authorization_path=tmp_path / "approval.json",
            expected_bindings=bindings,
            verify_data_files=False,
            expected_output_root=tmp_path / "evaluator",
            expected_row_count=16,
            expected_group_counts={"A": 8, "B": 4, "C": 4},
            expected_combination_counts=counts,
            validate_protocol=False,
        )
