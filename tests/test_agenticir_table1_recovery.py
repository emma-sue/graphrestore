from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.evaluation import agenticir_table1_recovery as recovery


@pytest.fixture(autouse=True)
def _isolate_cpu_only_guard_from_pytest_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Other collected test modules import Torch before these unit tests run."""

    monkeypatch.setattr(recovery, "assert_cpu_only_entrypoint", lambda: None)


def test_cpu_only_entrypoint_contract_in_fresh_process() -> None:
    project_root = Path(__file__).resolve().parents[1]
    clean = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.evaluation.agenticir_table1_recovery import "
                "assert_cpu_only_entrypoint; assert_cpu_only_entrypoint()"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert clean.returncode == 0, clean.stderr

    contaminated = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, types; "
                "sys.modules['torch'] = types.ModuleType('torch'); "
                "from src.evaluation.agenticir_table1_recovery import "
                "assert_cpu_only_entrypoint; assert_cpu_only_entrypoint()"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert contaminated.returncode != 0
    assert "forbidden modules: ['torch']" in contaminated.stderr


def _json_text(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write(path: Path, payload: str | bytes, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    if immutable:
        path.chmod(0o444)


def _write_json(path: Path, value: object, *, immutable: bool = False) -> None:
    _write(path, _json_text(value), immutable=immutable)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_sha(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": _sha(path)}


def _full_binding(path: Path, root: Path) -> dict[str, object]:
    return recovery._binding(path, confinement_root=root)  # noqa: SLF001


def _score_row(row: dict[str, object], *, base: float) -> dict[str, object]:
    return {
        **row,
        "psnr": base + 0.100003,
        "ssim": 0.80001 + base / 1000.0,
        "lpips": 0.2 + base / 1000.0,
        "maniqa": 0.3 + base / 1000.0,
        "clipiqa": 0.4 + base / 1000.0,
        "musiq": 50.0 + base,
    }


def _fixture(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "data"
    score_root = root / "formal/table1_scores"
    shards_dir = score_root / "shards"
    worker_dir = score_root / ".worker"
    shards_dir.mkdir(parents=True)
    worker_dir.mkdir()

    paths = recovery.RecoveryPaths(
        confinement_root=root,
        formal_authorization=root / "approvals/formal.json",
        evaluator_complete=root / "formal/complete.json",
        evaluator_per_image=root / "formal/per_image.csv",
        table1_input=root / "formal/table1_input.jsonl",
        weights_lock=root / "cache/weights_lock.json",
        metric_parity_summary=root / "metrics/parity.json",
        run_contract=score_root / "run_contract.json",
        input_lock=score_root / "input_lock.json",
        worker_request=worker_dir / "request-00000.json",
        shards_dir=shards_dir,
        score_root=score_root,
        legacy_module=root / "sources/legacy.py",
        legacy_cli=root / "sources/legacy_cli.py",
        failure_log=root / "logs/failure.log",
        official_compare_methods=root / "sources/compare_methods.py",
        recovery_module=root / "sources/recovery.py",
        recovery_cli=root / "sources/recovery_cli.py",
        approval=root / "approvals/recovery.json",
        remediation_receipt=root / "migrations/recovery/COMPLETE.json",
        output_per_image=score_root / "per_image.csv",
        output_summary=score_root / "summary.json",
        output_complete=score_root / "complete.json",
    )
    spec = recovery.RecoverySpec(
        groups={"A": ("combo-a",), "B": ("combo-b",)},
        expected_counts={"combo-a": 1, "combo-b": 1},
        shard_size=1,
    )

    for path, text in (
        (paths.legacy_module, "# pinned legacy scorer\n"),
        (paths.legacy_cli, "# pinned legacy CLI\n"),
        (paths.official_compare_methods, 'print(f"{1.0:.4}")\n'),
        (paths.recovery_module, "# recovery module\n"),
        (paths.recovery_cli, "# recovery CLI\n"),
    ):
        _write(path, text)
    _write(
        paths.failure_log,
        "AgenticIR Table-1 contract error: evaluator/scorer ssim drift\n",
    )

    rows: list[dict[str, object]] = []
    for group, combination, suffix in (
        ("A", "combo-a", "001"),
        ("B", "combo-b", "002"),
    ):
        rows.append(
            {
                "schema_version": recovery.INPUT_SCHEMA,
                "sample_id": f"test/{group}/{combination}/{suffix}",
                "group": group,
                "combination": combination,
                "prediction_png": str(root / f"never-open/pred-{suffix}.png"),
                "prediction_sha256": hashlib.sha256(
                    f"pred-{suffix}".encode()
                ).hexdigest(),
                "target_png": str(root / f"never-open/target-{suffix}.png"),
                "target_sha256": hashlib.sha256(
                    f"target-{suffix}".encode()
                ).hexdigest(),
            }
        )
    _write(
        paths.table1_input,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        immutable=True,
    )

    evaluator_values = [(20.0, 0.8), (30.0, 0.81)]
    evaluator_io = io.StringIO(newline="")
    writer = csv.DictWriter(
        evaluator_io,
        fieldnames=recovery.EVALUATOR_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row, (psnr, ssim) in zip(rows, evaluator_values, strict=True):
        writer.writerow(
            {
                **{key: row[key] for key in recovery.IDENTITY_FIELDS},
                "clean_id": row["sample_id"],
                "psnr": psnr,
                "ssim": ssim,
                "latency_ms": 1,
                "program_levels": 1,
                "parallel_levels": 1,
                "active_skill_calls": 1,
                "reentry_requests": 0,
                "unexpected_activations": 0,
                "precycle_graphs": 0,
                "dropped_edges": 0,
                "peak_reserved_fraction": 0.1,
            }
        )
    _write(paths.evaluator_per_image, evaluator_io.getvalue(), immutable=True)

    for path, payload in (
        (paths.formal_authorization, {"approved": True}),
        (root / "formal/run_contract.json", {"formal": "run"}),
        (root / "formal/summary.json", {"formal": "summary"}),
        (root / "formal/checkpoint.pth", {"fake": "checkpoint"}),
        (root / "formal/manifest.jsonl", {"fake": "manifest"}),
        (root / "formal/inventory.json", {"fake": "inventory"}),
    ):
        _write_json(path, payload, immutable=True)
    _write_json(
        paths.metric_parity_summary,
        {
            "passed": True,
            "facts": {
                "max_psnr_abs_diff": 0.0,
                "max_ssim_abs_diff": 0.0000001,
            },
        },
    )

    digest_rows = [
        {
            "sample_id": row["sample_id"],
            "prediction_sha256": row["prediction_sha256"],
            "target_sha256": row["target_sha256"],
        }
        for row in rows
    ]
    predictions_digest = recovery._sha256_json(digest_rows)  # noqa: SLF001
    evaluator_complete = {
        "schema_version": "graphrestore-formal-mio100-complete-v1",
        "status": "COMPLETE",
        "image_count": spec.image_count,
        "predictions_digest": predictions_digest,
        "authorization_sha256": _sha(paths.formal_authorization),
        "bindings": {
            "per_image_csv": _path_sha(paths.evaluator_per_image),
            "table1_input_jsonl": _path_sha(paths.table1_input),
        },
    }
    _write_json(paths.evaluator_complete, evaluator_complete, immutable=True)

    initial_rng_core = {
        "python": "python-state",
        "numpy": "numpy-state",
        "torch_cpu": "torch-state",
    }
    metric_runtime = [
        {"name": name, "mode": "FR", "lower_better": name == "lpips"}
        for name in recovery.METRICS
    ]
    _write_json(
        paths.weights_lock,
        {"initial_rng_core": initial_rng_core, "metric_runtime": metric_runtime},
        immutable=True,
    )

    locked_rows = []
    for index, row in enumerate(rows, start=1):
        locked_rows.append(
            {
                **row,
                "prediction_mode": 0o444,
                "prediction_device": 1,
                "prediction_inode": index,
                "prediction_size": 10,
                "target_mode": 0o644,
                "target_device": 1,
                "target_inode": index + 10,
                "target_size": 20,
            }
        )
    input_lock = {
        "schema_version": recovery.INPUT_LOCK_SCHEMA,
        "created_utc": "2026-08-20T00:00:00Z",
        "manifest": _full_binding(paths.table1_input, root),
        "image_count": spec.image_count,
        "expected_counts": dict(spec.expected_counts),
        "ordering": "OFFICIAL_GROUPS order, then strictly increasing sample_id",
        "rows": locked_rows,
    }
    _write_json(paths.input_lock, input_lock, immutable=True)

    formal_evidence = {
        "authorization": _path_sha(paths.formal_authorization),
        "evaluator_complete": _path_sha(paths.evaluator_complete),
        "run_contract": _path_sha(root / "formal/run_contract.json"),
        "summary": _path_sha(root / "formal/summary.json"),
        "per_image": _path_sha(paths.evaluator_per_image),
        "table1_input": _path_sha(paths.table1_input),
        "checkpoint": _path_sha(root / "formal/checkpoint.pth"),
        "manifest": _path_sha(root / "formal/manifest.jsonl"),
        "formal_data_inventory": _path_sha(root / "formal/inventory.json"),
        "metric_parity_summary": _path_sha(paths.metric_parity_summary),
        "predictions_digest": predictions_digest,
    }
    implementation = {
        "table1_scorer_module": _full_binding(paths.legacy_module, root),
        "table1_scorer_cli": _full_binding(paths.legacy_cli, root),
    }
    run_contract = {
        "schema_version": recovery.RUN_CONTRACT_SCHEMA,
        "metrics": list(recovery.METRICS),
        "metric_directions": recovery.METRIC_DIRECTIONS,
        "device": "cuda:0",
        "shard_size": spec.shard_size,
        "image_count": spec.image_count,
        "expected_counts": dict(spec.expected_counts),
        "formal_mio100_only": True,
        "input_lock": _path_sha(paths.input_lock),
        "weights_lock": _path_sha(paths.weights_lock),
        "implementation": implementation,
        "agenticir_sources": {
            "official_compare_methods": _full_binding(
                paths.official_compare_methods, root
            )
        },
        "formal_evidence": formal_evidence,
    }
    _write_json(paths.run_contract, run_contract, immutable=True)

    request = {
        "schema_version": "graphrestore.agenticir_table1_worker_request.v1",
        "device": "cuda:0",
        "expected_metric_runtime": metric_runtime,
        "expected_runtime": None,
        "formal_evidence": formal_evidence,
        "implementation": implementation,
        "initial_rng_core": initial_rng_core,
        "input_lock": run_contract["input_lock"],
        "input_lock_sha256": _sha(paths.input_lock),
        "previous_rng": None,
        "rows": rows,
        "run_contract": _path_sha(paths.run_contract),
        "run_contract_sha256": _sha(paths.run_contract),
        "score_root": str(score_root),
        "shard_size": spec.shard_size,
        "shards_dir": str(shards_dir),
        "start_shard": 0,
        "weights_lock": run_contract["weights_lock"],
    }
    _write_json(paths.worker_request, request, immutable=True)

    score_rows = [
        _score_row(row, base=base) for row, base in zip(rows, (20.0, 30.0), strict=True)
    ]
    previous_rng: dict[str, object] | None = None
    for index, score_row in enumerate(score_rows):
        rng_before = (
            {**initial_rng_core, "torch_cuda": f"cuda-{index}"}
            if previous_rng is None
            else previous_rng
        )
        rng_after = {**initial_rng_core, "torch_cuda": f"cuda-{index + 1}"}
        shard = {
            "schema_version": recovery.SHARD_SCHEMA,
            "shard_index": index,
            "start_index": index,
            "end_index": index + 1,
            "run_contract_sha256": _sha(paths.run_contract),
            "input_lock_sha256": _sha(paths.input_lock),
            "runtime": {"device": "cuda:0", "backend": "pinned-pyiqa"},
            "rng_before": rng_before,
            "rng_before_sha256": recovery._sha256_json(rng_before),  # noqa: SLF001
            "rng_after": rng_after,
            "rng_after_sha256": recovery._sha256_json(rng_after),  # noqa: SLF001
            "peak_reserved_bytes": 10,
            "total_memory_bytes": 100,
            "peak_reserved_fraction": 0.1,
            "rows": [score_row],
        }
        _write_json(shards_dir / f"shard-{index:05d}.json", shard, immutable=True)
        previous_rng = rng_after

    labels = {
        "formal_authorization": paths.formal_authorization,
        "evaluator_complete": paths.evaluator_complete,
        "evaluator_per_image": paths.evaluator_per_image,
        "table1_input": paths.table1_input,
        "weights_lock": paths.weights_lock,
        "metric_parity_summary": paths.metric_parity_summary,
        "run_contract": paths.run_contract,
        "input_lock": paths.input_lock,
        "worker_request": paths.worker_request,
        "legacy_module": paths.legacy_module,
        "legacy_cli": paths.legacy_cli,
        "failure_log": paths.failure_log,
        "official_compare_methods": paths.official_compare_methods,
    }
    anchors = {label: _sha(path) for label, path in labels.items()}
    simple_shards = [
        {"name": path.name, "sha256": _sha(path)}
        for path in sorted(shards_dir.iterdir())
    ]
    anchors["shard_sha_list"] = recovery._sha256_json(simple_shards)  # noqa: SLF001
    return SimpleNamespace(
        paths=paths,
        spec=spec,
        anchors=anchors,
        rows=rows,
        score_rows=score_rows,
    )


def _approve(fixture: SimpleNamespace) -> dict[str, object]:
    return recovery.publish_approval(
        fixture.paths,
        execute_token=recovery.APPROVAL_EXECUTE_TOKEN,
        spec=fixture.spec,
        expected_anchors=fixture.anchors,
        approved_utc="2026-08-20T12:00:00Z",
    )


def test_recovery_publishes_only_official_shard_values(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    approval = _approve(fixture)
    receipt = recovery.finalize_recovery(
        fixture.paths,
        execute_token=recovery.FINALIZE_EXECUTE_TOKEN,
        spec=fixture.spec,
        expected_anchors=fixture.anchors,
    )

    with fixture.paths.output_per_image.open(newline="", encoding="utf-8") as handle:
        published = list(csv.DictReader(handle))
    assert [float(row["psnr"]) for row in published] == [
        row["psnr"] for row in fixture.score_rows
    ]
    assert float(published[0]["psnr"]) != 20.0
    summary = json.loads(fixture.paths.output_summary.read_text(encoding="utf-8"))
    crosscheck = summary["evaluator_psnr_ssim_crosscheck"]
    assert crosscheck["identity_passed"] is True
    assert crosscheck["numeric_parity_claim"] is False
    assert crosscheck["numeric_gate_applied"] is False
    assert crosscheck["tolerance_changed"] is False
    assert crosscheck["table_display_diagnostic"]["scientific_acceptance_gate"] is False
    assert len(receipt["per_image_drift"]) == fixture.spec.image_count
    assert receipt["shard_inventory_unchanged"] is True
    assert approval["shard_count"] == fixture.spec.shard_count
    for path in (
        fixture.paths.approval,
        fixture.paths.output_per_image,
        fixture.paths.output_summary,
        fixture.paths.output_complete,
        fixture.paths.remediation_receipt,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_table_display_difference_is_diagnostic_not_numeric_gate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state = recovery.audit_recovery_state(
        fixture.paths,
        spec=fixture.spec,
        expected_anchors=fixture.anchors,
        allow_outputs=False,
    )
    diagnostic = state["crosscheck"]["table_display_diagnostic"]
    assert diagnostic["mismatch_count"] > 0
    assert diagnostic["all_equal"] is False
    assert state["crosscheck"]["numeric_gate_applied"] is False


def test_verify_only_is_read_only_and_wrong_tokens_fail(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before = sorted(
        str(path.relative_to(fixture.paths.confinement_root))
        for path in fixture.paths.confinement_root.rglob("*")
    )
    result = recovery.approval_verify_only(
        fixture.paths,
        spec=fixture.spec,
        expected_anchors=fixture.anchors,
    )
    after = sorted(
        str(path.relative_to(fixture.paths.confinement_root))
        for path in fixture.paths.confinement_root.rglob("*")
    )
    assert result["status"] == "READY_FOR_APPROVAL"
    assert before == after
    with pytest.raises(recovery.Table1RecoveryError, match="approval requires"):
        recovery.publish_approval(
            fixture.paths,
            execute_token="wrong",
            spec=fixture.spec,
            expected_anchors=fixture.anchors,
        )
    _approve(fixture)
    with pytest.raises(recovery.Table1RecoveryError, match="finalization requires"):
        recovery.finalize_recovery(
            fixture.paths,
            execute_token="wrong",
            spec=fixture.spec,
            expected_anchors=fixture.anchors,
        )


def test_interrupted_exact_prefix_resumes_without_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _approve(fixture)
    original = recovery._publish_or_verify  # noqa: SLF001
    calls = 0

    def interrupted(path: Path, payload: str, *, confinement_root: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise recovery.Table1RecoveryError("injected interruption")
        original(path, payload, confinement_root=confinement_root)

    monkeypatch.setattr(recovery, "_publish_or_verify", interrupted)
    with pytest.raises(recovery.Table1RecoveryError, match="injected interruption"):
        recovery.finalize_recovery(
            fixture.paths,
            execute_token=recovery.FINALIZE_EXECUTE_TOKEN,
            spec=fixture.spec,
            expected_anchors=fixture.anchors,
        )
    first_stat = fixture.paths.output_per_image.stat()
    monkeypatch.setattr(recovery, "_publish_or_verify", original)
    recovery.finalize_recovery(
        fixture.paths,
        execute_token=recovery.FINALIZE_EXECUTE_TOKEN,
        spec=fixture.spec,
        expected_anchors=fixture.anchors,
    )
    resumed_stat = fixture.paths.output_per_image.stat()
    assert (first_stat.st_ino, first_stat.st_mtime_ns) == (
        resumed_stat.st_ino,
        resumed_stat.st_mtime_ns,
    )


def test_output_tamper_and_unknown_score_entry_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _approve(fixture)
    recovery.finalize_recovery(
        fixture.paths,
        execute_token=recovery.FINALIZE_EXECUTE_TOKEN,
        spec=fixture.spec,
        expected_anchors=fixture.anchors,
    )
    fixture.paths.output_summary.chmod(0o644)
    fixture.paths.output_summary.write_text("{}\n", encoding="utf-8")
    fixture.paths.output_summary.chmod(0o444)
    with pytest.raises(recovery.Table1RecoveryError, match="standard output drifted"):
        recovery.verify_recovery(
            fixture.paths,
            spec=fixture.spec,
            expected_anchors=fixture.anchors,
            require_complete=True,
        )

    other = _fixture(tmp_path / "other")
    _write(other.paths.score_root / "unexpected.bin", b"x")
    with pytest.raises(recovery.Table1RecoveryError, match="unauthorized"):
        recovery.audit_recovery_state(
            other.paths,
            spec=other.spec,
            expected_anchors=other.anchors,
            allow_outputs=False,
        )


def test_evidence_or_shard_tamper_fails_before_publication(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.paths.failure_log.write_text("different failure\n", encoding="utf-8")
    with pytest.raises(recovery.Table1RecoveryError, match="identity anchor"):
        recovery.approval_candidate(
            fixture.paths,
            spec=fixture.spec,
            expected_anchors=fixture.anchors,
        )
    assert not fixture.paths.approval.exists()
    assert not fixture.paths.output_per_image.exists()

    other = _fixture(tmp_path / "other")
    shard = other.paths.shards_dir / "shard-00000.json"
    shard.chmod(0o644)
    with pytest.raises(recovery.Table1RecoveryError, match="mode is not 0444"):
        recovery.approval_candidate(
            other.paths,
            spec=other.spec,
            expected_anchors=other.anchors,
        )


def test_no_png_is_opened_and_heavy_metric_modules_are_not_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    original = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.suffix.lower() == ".png":
            raise AssertionError(f"PNG access is forbidden: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    recovery.approval_candidate(
        fixture.paths,
        spec=fixture.spec,
        expected_anchors=fixture.anchors,
    )
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import torch",
        "import cv2",
        "import pyiqa",
        "from PIL",
        "import socket",
        "import subprocess",
    ):
        assert forbidden not in source


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = fixture.paths.worker_request
    request.chmod(0o644)
    request.unlink()
    request.symlink_to(fixture.paths.failure_log)
    with pytest.raises(recovery.Table1RecoveryError, match="invalid score-tree"):
        recovery.approval_candidate(
            fixture.paths,
            spec=fixture.spec,
            expected_anchors=fixture.anchors,
        )
