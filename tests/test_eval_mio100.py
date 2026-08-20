from __future__ import annotations

import json
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from src.evaluation import mio100
from src.evaluation.formal_inventory import (
    FormalDataInventory,
    InventoryFileIdentity,
    InventoryRowIdentity,
    stream_file_identity,
)
from src.evaluation.mio100 import (
    AUTHORIZATION_SCHEMA,
    InferenceResult,
    ArtifactBinding,
    FormalAuthorization,
    MiO100EvaluationError,
    MiO100Record,
    REQUIRED_AUTHORIZATION_BINDINGS,
    TABLE1_INPUT_SCHEMA,
    finalize_evaluation,
    load_formal_manifest,
    load_stage4_best_ema,
    prepare_run_contract,
    process_record,
    run_shard,
    validate_formal_authorization,
    validate_formal_evaluator_complete,
)
from src.metrics.agenticir_official import OFFICIAL_GROUPS
from src.utils.hashing import sha256_file


def test_gpu_process_gate_requires_exact_ownership() -> None:
    def result(stdout: str, *, returncode: int = 0):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    mio100.assert_exclusive_gpu_process(
        expected_pid=None,
        runner=lambda *args, **kwargs: result(""),
    )
    mio100.assert_exclusive_gpu_process(
        expected_pid=123,
        runner=lambda *args, **kwargs: result("123\n123\n"),
    )
    with pytest.raises(MiO100EvaluationError, match="exclusive GPU process gate"):
        mio100.assert_exclusive_gpu_process(
            expected_pid=123,
            runner=lambda *args, **kwargs: result("123\n456\n"),
        )
    with pytest.raises(MiO100EvaluationError, match="unexpected nvidia-smi"):
        mio100.assert_exclusive_gpu_process(
            expected_pid=None,
            runner=lambda *args, **kwargs: result("not-a-pid\n"),
        )


def test_protocol_bindings_cross_check_inventory_and_metric_parity(
    tmp_path: Path,
) -> None:
    manifest = (tmp_path / mio100.FORMAL_MANIFEST_FILENAME).resolve()
    manifest.write_text("frozen manifest\n", encoding="utf-8")
    mioir = (tmp_path / "matlab_functions.py").resolve()
    mioir.write_text("# frozen canonicalizer\n", encoding="utf-8")
    scorer = (tmp_path / "scorer.py").resolve()
    scorer.write_text("# frozen scorer\n", encoding="utf-8")
    inventory = (tmp_path / "inventory.json").resolve()
    _write_json(
        inventory,
        {
            "schema_version": "graphrestore.agenticir_online_canonical.inventory.v1",
            "canonicalizer": {
                "implementation": str(mioir),
                "operation": (
                    "native BGR uint8 -> RGB float -> MiOIR imresize x4 -> clamp"
                ),
                "requantized_after_resize": "false",
                "sha256": sha256_file(mioir),
            },
            "manifests": {
                mio100.FORMAL_MANIFEST_FILENAME: {
                    "path": str(manifest),
                    "rows": 1_440,
                    "sha256": sha256_file(manifest),
                }
            },
        },
    )
    parity = (tmp_path / "parity.json").resolve()
    _write_json(
        parity,
        {
            "protocol": "graphrestore-v7.1-agenticir-metric-parity",
            "passed": True,
            "failure_count": 0,
            "facts": {
                "canonical_float_exact": True,
                "canonical_uint8_exact": True,
                "max_psnr_abs_diff": 0.0,
                "max_ssim_abs_diff": 3.9e-7,
                "versions": {
                    "agenticir_scorer_sha256": sha256_file(scorer),
                    "reference_environment": {"pyiqa": "0.1.10"},
                },
            },
        },
    )
    authorization = FormalAuthorization(
        path=parity,
        sha256=sha256_file(parity),
        approved_utc="2026-08-20T00:00:00Z",
        output_root=tmp_path,
        method_name=mio100.FORMAL_METHOD_NAME,
        shard_count=1,
        bindings={
            "manifest_inventory": ArtifactBinding(inventory, sha256_file(inventory)),
            "formal_manifest": ArtifactBinding(manifest, sha256_file(manifest)),
            "mioir_matlab_functions": ArtifactBinding(mioir, sha256_file(mioir)),
            "metric_parity_summary": ArtifactBinding(parity, sha256_file(parity)),
            "agenticir_scorer": ArtifactBinding(scorer, sha256_file(scorer)),
            "formal_authorization_protocol": ArtifactBinding(
                mio100.FORMAL_AUTHORIZATION_PROTOCOL_PATH,
                mio100.FORMAL_AUTHORIZATION_PROTOCOL_SHA256,
            ),
        },
    )
    mio100.validate_protocol_bindings(authorization)


def test_stage4_completion_binds_selected_step40000_and_six_modes(
    tmp_path: Path,
) -> None:
    files = {}
    for name in (
        "stage4_checkpoint",
        "stage4_validation",
        "stage4_calibration_history",
        "stage4_report",
        "stage4_diagnostics_report",
    ):
        path = (tmp_path / f"{name}.txt").resolve()
        path.write_text(f"{name}\n", encoding="utf-8")
        files[name] = path
    checkpoint_sha = sha256_file(files["stage4_checkpoint"])
    mode = {"image_count": 1_600, "peak_reserved_fraction": 0.5}
    diagnostics_path = (tmp_path / "diagnostics.json").resolve()
    _write_json(
        diagnostics_path,
        {
            "schema_version": "graphrestore-stage4-zero-training-diagnostics-v1",
            "protocol_id": mio100.PROTOCOL_ID,
            "selected_best_ema_path": str(files["stage4_checkpoint"]),
            "selected_best_ema_sha256": checkpoint_sha,
            "optimizer_updates": 0,
            "model_ema_rng_unchanged": True,
            "compiler_modes": {
                "full_partial_order": mode,
                "forced_total_order": mode,
                "parallel_only": mode,
            },
            "guard_modes": {
                "predicted_spatial": mode,
                "global_mean": mode,
                "all_one": mode,
            },
        },
    )
    files["stage4_diagnostics_json"] = diagnostics_path
    bindings = {
        name: ArtifactBinding(path, sha256_file(path)) for name, path in files.items()
    }
    authorization = FormalAuthorization(
        path=diagnostics_path,
        sha256=sha256_file(diagnostics_path),
        approved_utc="2026-08-20T00:00:00Z",
        output_root=tmp_path,
        method_name=mio100.FORMAL_METHOD_NAME,
        shard_count=1,
        bindings=bindings,
    )
    complete = (tmp_path / "complete.json").resolve()
    _write_json(
        complete,
        {
            "schema_version": "graphrestore-stage4-runtime-v1",
            "protocol_id": mio100.PROTOCOL_ID,
            "step": 40_000,
            "formal_mio100_started": False,
            "waiting_for": "new_user_authorization_for_formal_mio100",
            "best_ema_path": str(files["stage4_checkpoint"]),
            "best_ema_sha256": checkpoint_sha,
            "best_score": {"step": 40_000},
            "latest_score": {"step": 40_000},
            "maximum_train_peak_reserved_fraction": 0.5,
            "maximum_validation_peak_reserved_fraction": 0.6,
            "diagnostics_selected_best_ema_sha256": checkpoint_sha,
            "validation": str(files["stage4_validation"]),
            "validation_sha256": sha256_file(files["stage4_validation"]),
            "calibration_history": str(files["stage4_calibration_history"]),
            "calibration_history_sha256": sha256_file(
                files["stage4_calibration_history"]
            ),
            "report": str(files["stage4_report"]),
            "report_sha256": sha256_file(files["stage4_report"]),
            "diagnostics_json": str(files["stage4_diagnostics_json"]),
            "diagnostics_json_sha256": sha256_file(files["stage4_diagnostics_json"]),
            "diagnostics_report": str(files["stage4_diagnostics_report"]),
            "diagnostics_report_sha256": sha256_file(
                files["stage4_diagnostics_report"]
            ),
        },
    )
    mio100.validate_stage4_completion(
        complete,
        checkpoint_sha256=checkpoint_sha,
        authorization=authorization,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _authorization_payload(
    tmp_path: Path,
    *,
    output_root: Path,
    shard_count: int = 1,
) -> tuple[Path, dict[str, Any]]:
    bindings: dict[str, dict[str, str]] = {}
    for index, name in enumerate(REQUIRED_AUTHORIZATION_BINDINGS):
        bound = (tmp_path / "bindings" / f"{index:02d}-{name}.txt").resolve()
        bound.parent.mkdir(parents=True, exist_ok=True)
        bound.write_text(f"{name}\n", encoding="utf-8")
        bound.chmod(0o444)
        bindings[name] = {"path": str(bound), "sha256": sha256_file(bound)}
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "kind": "formal_mio100_approval",
        "protocol_id": mio100.PROTOCOL_ID,
        "approved": True,
        "formal_mio100_authorized": True,
        "one_shot": True,
        "inference_only": True,
        "authorized_groups": ["A", "B", "C"],
        "manifest_row_count": 1_440,
        "method_name": mio100.FORMAL_METHOD_NAME,
        "shard_count": shard_count,
        "output_root": str(output_root.resolve()),
        "approved_utc": "2026-08-20T00:00:00Z",
        "restrictions": {
            "task_label_routing": False,
            "tta": False,
            "model_soup": False,
            "threshold_tuning": False,
            "result_driven_rerun": False,
            "overwrite": False,
        },
        "bindings": bindings,
    }
    path = (tmp_path / "FORMAL_MIO100_APPROVED.json").resolve()
    _write_json(path, payload)
    path.chmod(0o444)
    return path, payload


def test_authorization_requires_immutable_exact_scope_and_hashes(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "formal-output"
    path, payload = _authorization_payload(tmp_path, output_root=output_root)
    expected_manifest = Path(payload["bindings"]["formal_manifest"]["path"])
    authorization = validate_formal_authorization(
        path,
        expected_bindings={"formal_manifest": expected_manifest},
        expected_output_root=output_root,
    )
    assert authorization.path == path
    assert authorization.output_root == output_root.resolve()
    assert set(authorization.bindings) == set(REQUIRED_AUTHORIZATION_BINDINGS)

    path.chmod(0o644)
    with pytest.raises(MiO100EvaluationError, match="immutable/read-only"):
        validate_formal_authorization(path, expected_output_root=output_root)
    path.chmod(0o444)

    bound = Path(payload["bindings"]["metric_weight_inventory"]["path"])
    bound.chmod(0o644)
    bound.write_text("drift\n", encoding="utf-8")
    with pytest.raises(MiO100EvaluationError, match="hash drifted"):
        validate_formal_authorization(path, expected_output_root=output_root)


def _manifest_row(
    *,
    index: int,
    group: str,
    combination: str,
    root: Path,
) -> dict[str, object]:
    clean_id = f"{index:06d}"
    degradations = combination.split("+")
    contains_lr = "low resolution" in degradations
    native = (root / "native" / group / combination / f"{clean_id}.png").resolve()
    target = (root / "gt" / f"{clean_id}.png").resolve()
    return {
        "schema_version": "graphrestore.agenticir_online_canonical.v1",
        "sample_id": f"test/{group}/{combination}/{clean_id}",
        "clean_id": clean_id,
        "group": group,
        "degradations": degradations,
        "source": "AgenticIR",
        "split": "test",
        "native_lq_path": str(native),
        "input_path": str(native),
        "gt_path": str(target),
        "input_mode": "agenticir_online_canonical",
        "contains_low_resolution": contains_lr,
        "native_scale": 0.25 if contains_lr else 1.0,
        "scale_factor": 4 if contains_lr else 1,
        "online_scale_factor": 4 if contains_lr else 1,
        "online_canonicalization": (
            "mioir_basicsr_native_uint8_to_rgb_float_x4"
            if contains_lr
            else "native_uint8_to_rgb_float_identity"
        ),
        "mioir_matlab_functions_sha256": mio100.MIOIR_MATLAB_FUNCTIONS_SHA256,
        "requantize_after_online_resize": False,
        "input_storage_color_order": "BGR",
        "model_input_color_order": "RGB",
        "model_input_dtype": "float32",
    }


def _write_formal_manifest(tmp_path: Path) -> Path:
    path = tmp_path / mio100.FORMAL_MANIFEST_FILENAME
    rows = []
    index = 0
    for group, combinations in OFFICIAL_GROUPS.items():
        count = 80 if group == "A" else 100
        for combination in combinations:
            for _ in range(count):
                rows.append(
                    _manifest_row(
                        index=index,
                        group=group,
                        combination=combination,
                        root=tmp_path,
                    )
                )
                index += 1
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return path.resolve()


def test_formal_manifest_requires_exact_1440_online_canonical_inventory(
    tmp_path: Path,
) -> None:
    path = _write_formal_manifest(tmp_path)
    records = load_formal_manifest(path, expected_sha256=sha256_file(path))
    assert len(records) == 1_440
    assert len({record.sample_id for record in records}) == 1_440
    assert sum(record.contains_low_resolution for record in records) == 460
    assert records[0].combination in OFFICIAL_GROUPS["A"]

    rows = path.read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(rows[1])
    duplicate["sample_id"] = json.loads(rows[0])["sample_id"]
    rows[1] = json.dumps(duplicate, sort_keys=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(MiO100EvaluationError, match="duplicate sample_id"):
        load_formal_manifest(path, expected_sha256=sha256_file(path))


def test_formal_manifest_rejects_stale_filename(tmp_path: Path) -> None:
    path = (tmp_path / "mio100_test_1440_agenticir_canonical.jsonl").resolve()
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MiO100EvaluationError, match="only the full online-canonical"):
        load_formal_manifest(path, expected_sha256=sha256_file(path))


def _checkpoint_payload(*, equal: bool = True) -> dict[str, object]:
    model = {
        "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "counter": torch.tensor(7, dtype=torch.int64),
    }
    shadow = {key: value.clone() for key, value in model.items()}
    if not equal:
        shadow["weight"][0, 0] += 1
    return {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage4",
        "step": 40_000,
        "model_role": "ema_selection",
        "resumable": False,
        "pending_validation_step": None,
        "model": model,
        "ema": {"shadow": shadow, "num_updates": 40_000},
        "provenance": {
            "schema_version": "graphrestore-stage4-runtime-v1",
            "protocol_id": mio100.PROTOCOL_ID,
            "config_sha256": "a" * 64,
        },
    }


def test_stage4_checkpoint_requires_step40000_best_model_equal_ema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "best_ema.pth"
    torch.save(_checkpoint_payload(), path)
    snapshot = load_stage4_best_ema(
        path.resolve(),
        expected_sha256=sha256_file(path),
        expected_config_sha256="a" * 64,
        expected_tensor_count=2,
    )
    assert snapshot.model_state["counter"].item() == 7

    torch.save(_checkpoint_payload(equal=False), path)
    with pytest.raises(MiO100EvaluationError, match="not bit-exact"):
        load_stage4_best_ema(
            path.resolve(),
            expected_sha256=sha256_file(path),
            expected_tensor_count=2,
        )


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _record(
    root: Path,
    *,
    index: int,
    group: str,
    combination: str,
) -> MiO100Record:
    clean_id = f"{index:06d}"
    native = (root / "inputs" / f"{clean_id}.png").resolve()
    target = (root / "targets" / f"{clean_id}.png").resolve()
    _write_png(native, 64)
    _write_png(target, 96)
    row = _manifest_row(
        index=index,
        group=group,
        combination=combination,
        root=root,
    )
    row["native_lq_path"] = str(native)
    row["input_path"] = str(native)
    row["gt_path"] = str(target)
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    native_raw = stream_file_identity(native, field="test native")
    target_raw = stream_file_identity(target, field="test target")
    native_identity = InventoryFileIdentity(
        path=native,
        sha256=str(native_raw["sha256"]),
        size_bytes=int(native_raw["size_bytes"]),
        mode=int(native_raw["mode"]),
        device=int(native_raw["device"]),
        inode=int(native_raw["inode"]),
        roles=("native_lq",),
        reference_count=1,
    )
    target_identity = InventoryFileIdentity(
        path=target,
        sha256=str(target_raw["sha256"]),
        size_bytes=int(target_raw["size_bytes"]),
        mode=int(target_raw["mode"]),
        device=int(target_raw["device"]),
        inode=int(target_raw["inode"]),
        roles=("target",),
        reference_count=1,
    )
    return MiO100Record(
        index=index,
        sample_id=str(row["sample_id"]),
        clean_id=clean_id,
        group=group,
        degradations=tuple(combination.split("+")),
        combination=combination,
        native_lq_path=native,
        target_path=target,
        contains_low_resolution="low resolution" in combination,
        row=row,
        row_sha256=__import__("hashlib").sha256(encoded).hexdigest(),
        expected_native_sha256=native_identity.sha256,
        expected_target_sha256=target_identity.sha256,
        native_file_identity=native_identity,
        target_file_identity=target_identity,
    )


def _fake_authorization(
    tmp_path: Path,
    *,
    output_root: Path,
    shard_count: int,
) -> FormalAuthorization:
    auth = (tmp_path / "authorization.json").resolve()
    auth.write_text("{}\n", encoding="utf-8")
    return FormalAuthorization(
        path=auth,
        sha256=sha256_file(auth),
        approved_utc="2026-08-20T00:00:00Z",
        output_root=output_root.resolve(),
        method_name=mio100.FORMAL_METHOD_NAME,
        shard_count=shard_count,
        bindings={
            "formal_data_inventory": ArtifactBinding(auth, sha256_file(auth)),
            "formal_manifest": ArtifactBinding(auth, sha256_file(auth)),
            "formal_authorization_protocol": ArtifactBinding(auth, sha256_file(auth)),
            "stage4_checkpoint": ArtifactBinding(auth, sha256_file(auth)),
            "stage4_config": ArtifactBinding(auth, sha256_file(auth)),
            "stage4_complete": ArtifactBinding(auth, sha256_file(auth)),
        },
    )


def _run(
    tmp_path: Path,
    *,
    shard_count: int = 1,
) -> tuple[mio100.EvaluationRun, FormalAuthorization]:
    authorization = _fake_authorization(
        tmp_path,
        output_root=tmp_path / "run",
        shard_count=shard_count,
    )
    run = prepare_run_contract(
        authorization,
        manifest_sha256=authorization.bindings["formal_manifest"].sha256,
        data_inventory_sha256=authorization.bindings["formal_data_inventory"].sha256,
        data_inventory_rows_digest="a" * 64,
        data_inventory_files_digest="b" * 64,
        checkpoint_sha256=authorization.bindings["stage4_checkpoint"].sha256,
        config_sha256=authorization.bindings["stage4_config"].sha256,
        shard_count=shard_count,
        enforce_data_disk=False,
    )
    return run, authorization


def _input_loader(_: Any) -> torch.Tensor:
    return torch.full((3, 16, 16), 64.0 / 255.0)


def _inference(value: float = 96.0 / 255.0) -> InferenceResult:
    return InferenceResult(
        prediction=torch.full((1, 3, 16, 16), value),
        diagnostics={
            "program_levels": 1,
            "parallel_levels": 0,
            "active_skill_calls": 1,
            "reentry_requests": 0,
            "unexpected_activations": 0,
            "precycle_graphs": 0,
            "dropped_edges": 0,
        },
        latency_ms=1.25,
    )


def test_image_transaction_is_atomic_readback_scored_and_resume_skips_inference(
    tmp_path: Path,
) -> None:
    run, _ = _run(tmp_path)
    record = _record(
        tmp_path,
        index=0,
        group="A",
        combination=OFFICIAL_GROUPS["A"][0],
    )
    calls = 0

    def infer(_: torch.Tensor) -> InferenceResult:
        nonlocal calls
        calls += 1
        return _inference()

    receipt = process_record(
        run,
        record,
        infer=infer,
        device=torch.device("cpu"),
        input_loader=_input_loader,
    )
    assert calls == 1
    assert receipt["psnr"] == pytest.approx(80.0)
    assert receipt["ssim"] == pytest.approx(1.0)
    output = Path(receipt["prediction_png"])
    bundle = run.root / "records" / record.record_key
    bundle_png = bundle / "prediction.png"
    assert output.stat().st_ino == bundle_png.stat().st_ino
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o555

    def must_not_infer(_: torch.Tensor) -> InferenceResult:
        raise AssertionError("resume must use the committed receipt")

    resumed = process_record(
        run,
        record,
        infer=must_not_infer,
        device=torch.device("cpu"),
        input_loader=_input_loader,
    )
    assert resumed["prediction_sha256"] == receipt["prediction_sha256"]


def test_unreceipted_output_refuses_overwrite_or_rerun(tmp_path: Path) -> None:
    run, _ = _run(tmp_path)
    record = _record(
        tmp_path,
        index=0,
        group="A",
        combination=OFFICIAL_GROUPS["A"][0],
    )
    output = (
        run.root
        / "methods"
        / run.method_name
        / "d2"
        / record.combination
        / record.output_filename
    )
    _write_png(output, 1)
    with pytest.raises(MiO100EvaluationError, match="unreceipted output"):
        process_record(
            run,
            record,
            infer=lambda _: _inference(),
            device=torch.device("cpu"),
            input_loader=_input_loader,
        )


def test_two_shard_exact_resume_finalizes_agenticir_tree_and_table1_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {
        combination: 1
        for combinations in OFFICIAL_GROUPS.values()
        for combination in combinations
    }
    monkeypatch.setattr(mio100, "FORMAL_ROW_COUNT", 16)
    monkeypatch.setattr(mio100, "FORMAL_GROUP_COUNTS", {"A": 8, "B": 4, "C": 4})
    monkeypatch.setattr(mio100, "FORMAL_COMBINATION_COUNTS", counts)
    records = []
    manifest_order = sorted(
        (
            (group, combination)
            for group, combinations in OFFICIAL_GROUPS.items()
            for combination in combinations
        ),
        key=lambda item: (item[0], item[1]),
    )
    for index, (group, combination) in enumerate(manifest_order):
        records.append(
            _record(
                tmp_path,
                index=index,
                group=group,
                combination=combination,
            )
        )
    run, authorization = _run(tmp_path, shard_count=2)
    for shard_index in (0, 1):
        run_shard(
            run,
            records,
            shard_index=shard_index,
            shard_count=2,
            infer=lambda _: _inference(),
            device=torch.device("cpu"),
            input_loader=_input_loader,
        )
    summary = finalize_evaluation(run, records, authorization=authorization)
    assert summary["image_count"] == 16
    assert list(summary["aggregation"]["groups"]) == ["A", "B", "C"]
    assert summary["aggregation"]["weighted_all_images"]["psnr"] == pytest.approx(80.0)
    table_path = run.root / "table1_input.jsonl"
    table_rows = [json.loads(line) for line in table_path.read_text().splitlines()]
    assert len(table_rows) == 16
    assert set(table_rows[0]) == {
        "schema_version",
        "sample_id",
        "group",
        "combination",
        "prediction_png",
        "prediction_sha256",
        "target_png",
        "target_sha256",
    }
    assert table_rows[0]["schema_version"] == TABLE1_INPUT_SCHEMA
    assert [row["combination"] for row in table_rows] == [
        combination
        for combinations in OFFICIAL_GROUPS.values()
        for combination in combinations
    ]
    assert stat.S_IMODE(table_path.stat().st_mode) == 0o444
    assert (run.root / "complete.json").is_file()
    for group, combinations in OFFICIAL_GROUPS.items():
        depth = "d3" if group == "C" else "d2"
        for combination in combinations:
            assert (
                run.root / "methods" / run.method_name / depth / combination
            ).is_dir()

    # A completed invocation verifies the same immutable bytes and never calls
    # inference again.
    again = finalize_evaluation(run, records, authorization=authorization)
    assert again == summary


def test_formal_completion_validator_cross_binds_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {
        combination: 1
        for combinations in OFFICIAL_GROUPS.values()
        for combination in combinations
    }
    groups = {"A": 8, "B": 4, "C": 4}
    monkeypatch.setattr(mio100, "FORMAL_ROW_COUNT", 16)
    monkeypatch.setattr(mio100, "FORMAL_GROUP_COUNTS", groups)
    monkeypatch.setattr(mio100, "FORMAL_COMBINATION_COUNTS", counts)
    records = tuple(
        _record(tmp_path, index=index, group=group, combination=combination)
        for index, (group, combination) in enumerate(
            (group, combination)
            for group, combinations in OFFICIAL_GROUPS.items()
            for combination in combinations
        )
    )
    run, authorization = _run(tmp_path)
    run_shard(
        run,
        records,
        shard_index=0,
        shard_count=1,
        infer=lambda _: _inference(),
        device=torch.device("cpu"),
        input_loader=_input_loader,
    )
    finalize_evaluation(run, records, authorization=authorization)

    inventory_files = {
        identity.path: identity
        for record in records
        for identity in (record.native_file_identity, record.target_file_identity)
        if identity is not None
    }
    inventory_rows = tuple(
        InventoryRowIdentity(
            index=record.index,
            sample_id=record.sample_id,
            row_sha256=record.row_sha256,
            native_lq_path=record.native_lq_path,
            native_lq_sha256=str(record.expected_native_sha256),
            target_path=record.target_path,
            target_sha256=str(record.expected_target_sha256),
        )
        for record in records
    )
    inventory = FormalDataInventory(
        path=authorization.bindings["formal_data_inventory"].path,
        sha256=authorization.bindings["formal_data_inventory"].sha256,
        manifest_sha256=authorization.bindings["formal_manifest"].sha256,
        rows_digest="a" * 64,
        files_digest="b" * 64,
        rows=inventory_rows,
        files=inventory_files,
    )
    monkeypatch.setattr(
        mio100, "validate_formal_authorization", lambda *args, **kwargs: authorization
    )
    monkeypatch.setattr(mio100, "validate_protocol_bindings", lambda *args: None)
    monkeypatch.setattr(
        mio100, "validate_stage4_completion", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        mio100,
        "load_strict_formal_data_inventory",
        lambda *args, **kwargs: inventory,
    )
    monkeypatch.setattr(mio100, "load_formal_manifest", lambda *args, **kwargs: records)
    completion = validate_formal_evaluator_complete(
        run.root / "complete.json",
        authorization_path=authorization.path,
        expected_output_root=run.root,
        expected_row_count=16,
        expected_group_counts=groups,
        expected_combination_counts=counts,
    )
    assert set(completion.evidence) == {
        "authorization",
        "evaluator_complete",
        "run_contract",
        "summary",
        "per_image",
        "table1_input",
        "checkpoint",
        "manifest",
        "formal_data_inventory",
        "predictions_digest",
    }
    complete_path = run.root / "complete.json"
    complete_path.chmod(0o644)
    tampered = json.loads(complete_path.read_text(encoding="utf-8"))
    tampered["predictions_digest"] = "f" * 64
    _write_json(complete_path, tampered)
    complete_path.chmod(0o444)
    with pytest.raises(MiO100EvaluationError, match="predictions digest drifted"):
        validate_formal_evaluator_complete(
            complete_path,
            authorization_path=authorization.path,
            expected_output_root=run.root,
            expected_row_count=16,
            expected_group_counts=groups,
            expected_combination_counts=counts,
        )


def test_pending_image_transaction_fails_closed(tmp_path: Path) -> None:
    run, _ = _run(tmp_path)
    record = _record(
        tmp_path,
        index=0,
        group="A",
        combination=OFFICIAL_GROUPS["A"][0],
    )
    pending = run.root / "records" / f".pending-{record.record_key}"
    pending.mkdir(parents=True)
    (pending / "prediction.png").write_bytes(b"incomplete")
    calls = 0

    def infer(_: torch.Tensor) -> InferenceResult:
        nonlocal calls
        calls += 1
        return _inference()

    with pytest.raises(
        MiO100EvaluationError, match="incomplete prior image transaction"
    ):
        process_record(
            run,
            record,
            infer=infer,
            device=torch.device("cpu"),
            input_loader=_input_loader,
        )
    assert calls == 0
