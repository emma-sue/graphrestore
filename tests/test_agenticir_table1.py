from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.evaluation import agenticir_table1 as table1
from src.utils.hashing import sha256_file, sha256_json


def _all_one_counts() -> dict[str, int]:
    return {combination: 1 for combination in table1.EXPECTED_COUNTS}


def _manifest_rows(
    counts: dict[str, int],
    *,
    prediction_root: Path = Path("/formal/predictions"),
    target_root: Path = Path("/formal/targets"),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    global_index = 0
    for group, combinations in table1.OFFICIAL_GROUPS.items():
        for combination in combinations:
            for sample_index in range(counts[combination]):
                rows.append(
                    {
                        "schema_version": table1.INPUT_SCHEMA,
                        "sample_id": f"{global_index:06d}",
                        "group": group,
                        "combination": combination,
                        "prediction_png": str(
                            prediction_root / f"prediction-{global_index:06d}.png"
                        ),
                        "prediction_sha256": "a" * 64,
                        "target_png": str(target_root / f"{sample_index:06d}.png"),
                        "target_sha256": "b" * 64,
                    }
                )
                global_index += 1
    return rows


def _score_row(input_row: dict[str, object], value: float) -> dict[str, object]:
    return {**input_row, **{metric: value for metric in table1.METRICS}}


def _rng_state(marker: int) -> dict[str, object]:
    return {
        "python": {"marker": marker},
        "numpy": {"marker": marker},
        "torch_cpu": [marker],
    }


def _publish_shard(
    directory: Path,
    *,
    index: int,
    shard_size: int,
    expected_rows: list[dict[str, object]],
    before: dict[str, object],
    after: dict[str, object],
    run_sha: str,
    input_sha: str,
) -> Path:
    start = index * shard_size
    end = min(start + shard_size, len(expected_rows))
    payload = {
        "schema_version": table1.SHARD_SCHEMA,
        "shard_index": index,
        "start_index": start,
        "end_index": end,
        "run_contract_sha256": run_sha,
        "input_lock_sha256": input_sha,
        "runtime": {"device": "cpu"},
        "rng_before": before,
        "rng_before_sha256": sha256_json(before),
        "rng_after": after,
        "rng_after_sha256": sha256_json(after),
        "peak_reserved_bytes": 0,
        "total_memory_bytes": 1,
        "peak_reserved_fraction": 0.0,
        "rows": [
            _score_row(dict(row), float(offset + start))
            for offset, row in enumerate(expected_rows[start:end])
        ],
    }
    path = directory / f"shard-{index:05d}.json"
    table1._atomic_create_json(path, payload)
    return path


def test_official_manifest_cardinality_order_and_counts() -> None:
    rows = _manifest_rows(dict(table1.EXPECTED_COUNTS))
    canonical, locked = table1.validate_manifest_records(rows, verify_files=False)
    assert len(canonical) == table1.EXPECTED_IMAGE_COUNT == 1440
    assert canonical == locked
    assert [row["group"] for row in canonical[:640]] == ["A"] * 640
    assert [row["group"] for row in canonical[640:1040]] == ["B"] * 400
    assert [row["group"] for row in canonical[1040:]] == ["C"] * 400

    swapped = list(rows)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(table1.Table1ContractError, match="sample_id order"):
        table1.validate_manifest_records(swapped, verify_files=False)


def test_manifest_verifies_frozen_prediction_and_both_hashes(tmp_path: Path) -> None:
    counts = _all_one_counts()
    prediction_root = tmp_path / "predictions"
    target_root = tmp_path / "targets"
    prediction_root.mkdir()
    target_root.mkdir()
    target = target_root / "000000.png"
    target.write_bytes(b"target-png-bytes")
    rows = _manifest_rows(
        counts, prediction_root=prediction_root, target_root=target_root
    )
    for index, row in enumerate(rows):
        prediction = Path(str(row["prediction_png"]))
        prediction.write_bytes(f"prediction-{index}".encode())
        os.chmod(prediction, 0o444)
        row["prediction_sha256"] = sha256_file(prediction)
        row["target_sha256"] = sha256_file(target)
        row["target_png"] = str(target)

    canonical, locked = table1.validate_manifest_records(
        rows, expected_counts=counts, verify_files=True
    )
    assert len(canonical) == 16
    assert all(row["prediction_mode"] == 0o444 for row in locked)
    os.chmod(Path(str(rows[3]["prediction_png"])), 0o644)
    with pytest.raises(table1.Table1ContractError, match="expected mode 0444"):
        table1.validate_manifest_records(
            rows, expected_counts=counts, verify_files=True
        )


def test_manifest_rejects_duplicate_prediction_and_nonfinite_hash_shape() -> None:
    rows = _manifest_rows(_all_one_counts())
    rows[1]["prediction_png"] = rows[0]["prediction_png"]
    with pytest.raises(table1.Table1ContractError, match="duplicate prediction"):
        table1.validate_manifest_records(
            rows, expected_counts=_all_one_counts(), verify_files=False
        )
    rows = _manifest_rows(_all_one_counts())
    rows[0]["target_sha256"] = "ABC"
    with pytest.raises(table1.Table1ContractError, match="lowercase SHA256"):
        table1.validate_manifest_records(
            rows, expected_counts=_all_one_counts(), verify_files=False
        )


def test_six_metric_aggregation_is_combo_then_equal_combo_group_mean() -> None:
    counts = _all_one_counts()
    inputs = _manifest_rows(counts)
    records = [_score_row(row, float(index + 1)) for index, row in enumerate(inputs)]
    result = table1.aggregate_table1_records(records, expected_counts=counts)
    assert result["image_count"] == 16
    assert result["groups"]["A"]["combination_count"] == 8
    assert result["groups"]["A"]["psnr"] == pytest.approx(4.5)
    assert result["groups"]["B"]["lpips"] == pytest.approx(10.5)
    assert result["groups"]["C"]["musiq"] == pytest.approx(14.5)

    records[5]["maniqa"] = math.nan
    with pytest.raises(table1.Table1ContractError, match="non-finite"):
        table1.aggregate_table1_records(records, expected_counts=counts)


def test_atomic_publication_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "immutable.json"
    table1._atomic_create_json(destination, {"value": 1})
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    with pytest.raises(table1.Table1ContractError, match="refusing to overwrite"):
        table1._atomic_create_json(destination, {"value": 2})
    assert json.loads(destination.read_text())["value"] == 1


def test_shards_are_an_exact_contiguous_rng_chain(tmp_path: Path) -> None:
    expected = _manifest_rows(_all_one_counts())[:3]
    run_sha = "1" * 64
    input_sha = "2" * 64
    state0, state1, state2 = _rng_state(0), _rng_state(1), _rng_state(2)
    shards = tmp_path / "shards"
    _publish_shard(
        shards,
        index=0,
        shard_size=2,
        expected_rows=expected,
        before=state0,
        after=state1,
        run_sha=run_sha,
        input_sha=input_sha,
    )
    partial = table1.scan_score_shards(
        shards_dir=shards,
        expected_rows=expected,
        shard_size=2,
        run_contract_sha256=run_sha,
        input_lock_sha256=input_sha,
        initial_rng_core=state0,
    )
    assert partial["completed_shards"] == 1
    assert len(partial["records"]) == 2
    assert partial["previous_rng"] == state1

    _publish_shard(
        shards,
        index=1,
        shard_size=2,
        expected_rows=expected,
        before=state1,
        after=state2,
        run_sha=run_sha,
        input_sha=input_sha,
    )
    complete = table1.scan_score_shards(
        shards_dir=shards,
        expected_rows=expected,
        shard_size=2,
        run_contract_sha256=run_sha,
        input_lock_sha256=input_sha,
        initial_rng_core=state0,
    )
    assert complete["completed_shards"] == complete["expected_shards"] == 2
    assert len(complete["records"]) == 3

    broken = tmp_path / "broken"
    _publish_shard(
        broken,
        index=1,
        shard_size=2,
        expected_rows=expected,
        before=state1,
        after=state2,
        run_sha=run_sha,
        input_sha=input_sha,
    )
    with pytest.raises(table1.Table1ContractError, match="non-contiguous"):
        table1.scan_score_shards(
            shards_dir=broken,
            expected_rows=expected,
            shard_size=2,
            run_contract_sha256=run_sha,
            input_lock_sha256=input_sha,
            initial_rng_core=state0,
        )


def test_shard_identity_or_rng_tamper_fails_without_rewrite(tmp_path: Path) -> None:
    expected = _manifest_rows(_all_one_counts())[:2]
    state0, state1 = _rng_state(0), _rng_state(1)
    shard = _publish_shard(
        tmp_path,
        index=0,
        shard_size=2,
        expected_rows=expected,
        before=state0,
        after=state1,
        run_sha="1" * 64,
        input_sha="2" * 64,
    )
    original = json.loads(shard.read_text())
    os.chmod(shard, 0o644)
    original["rows"][0]["sample_id"] = "forged"
    shard.write_text(json.dumps(original))
    os.chmod(shard, 0o444)
    with pytest.raises(table1.Table1ContractError, match="identity mismatch"):
        table1.scan_score_shards(
            shards_dir=tmp_path,
            expected_rows=expected,
            shard_size=2,
            run_contract_sha256="1" * 64,
            input_lock_sha256="2" * 64,
            initial_rng_core=state0,
        )


def test_cache_environment_never_repurposes_home_or_system_cache(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "data-cache"
    temporary = tmp_path / "work"
    before_home = os.environ.get("HOME")
    environment = table1._cache_environment(
        cache, offline=True, temporary_root=temporary, cpu_only=True
    )
    assert environment.get("HOME") == before_home
    for key in (
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "MPLCONFIGDIR",
        "NUMBA_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        Path(environment[key]).relative_to(cache)
    assert environment["TMPDIR"] == str(temporary)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_cache_inventory_hashes_files_and_internal_symlinks(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    nested = cache / "nested"
    nested.mkdir(parents=True)
    weight = nested / "weight.pth"
    weight.write_bytes(b"frozen-weight")
    (cache / "alias.pth").symlink_to(weight.relative_to(cache))
    table1._freeze_cache(cache)
    inventory = table1._cache_inventory(cache)
    assert {entry["type"] for entry in inventory} == {
        "directory",
        "file",
        "symlink",
    }
    assert next(entry for entry in inventory if entry["type"] == "file")[
        "sha256"
    ] == sha256_file(weight)
    table1._verify_frozen_cache_directories(cache)


def test_synthetic_reference_launcher_binds_link_target_and_prefix(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "base" / "bin" / "python3.12"
    trusted.parent.mkdir(parents=True)
    trusted.write_bytes(b"synthetic-python")
    os.chmod(trusted, 0o755)
    prefix = tmp_path / "reference"
    (prefix / "bin").mkdir(parents=True)
    launcher = prefix / "bin" / "python"
    launcher.symlink_to(trusted)
    (prefix / "pyvenv.cfg").write_text(
        f"home = {trusted.parent}\n"
        "include-system-site-packages = true\n"
        "version = 3.12.3\n"
        f"executable = {trusted}\n",
        encoding="utf-8",
    )

    binding = table1._reference_launcher_binding(launcher, trusted_base_python=trusted)
    assert binding["launcher"]["path"] == str(launcher)
    assert binding["launcher"]["symlink_target"] == str(trusted)
    assert binding["resolved_target"]["path"] == str(trusted)
    assert binding["reference_prefix"]["path"] == str(prefix)

    replacement = tmp_path / "base" / "bin" / "other-python"
    replacement.write_bytes(b"other")
    os.chmod(replacement, 0o755)
    launcher.unlink()
    launcher.symlink_to(replacement)
    with pytest.raises(table1.Table1ContractError, match="target mismatch"):
        table1._reference_launcher_binding(launcher, trusted_base_python=trusted)


def test_real_reference_launcher_executes_unresolved_venv_entry(
    tmp_path: Path,
) -> None:
    binding = table1._reference_launcher_binding(table1.DEFAULT_REFERENCE_PYTHON)
    script = (
        "from importlib import metadata\n"
        "import json, os, sys\n"
        "print(json.dumps({"
        "'sys_executable': sys.executable, "
        "'sys_prefix': sys.prefix, "
        "'sys_base_prefix': sys.base_prefix, "
        "'basicsr': metadata.version('basicsr'), "
        "'pyiqa': metadata.version('pyiqa'), "
        "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'torch_imported': 'torch' in sys.modules}))\n"
    )
    environment = table1._cache_environment(
        tmp_path / "cache",
        offline=True,
        temporary_root=tmp_path,
        cpu_only=True,
    )
    completed = subprocess.run(
        [str(table1.DEFAULT_REFERENCE_PYTHON), "-c", script],
        cwd=table1.PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "sys_executable": binding["launcher"]["path"],
        "sys_prefix": binding["reference_prefix"]["path"],
        "sys_base_prefix": binding["expected_sys_base_prefix"],
        "basicsr": "1.4.2",
        "pyiqa": "0.1.10",
        "cuda_visible_devices": "",
        "torch_imported": False,
    }


def test_json_worker_executes_launcher_path_without_resolving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "python-target"
    target.write_bytes(b"target")
    launcher = tmp_path / "python"
    launcher.symlink_to(target)

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == str(launcher)
        assert command[0] != str(launcher.resolve())
        result_path = Path(command[command.index("--worker-result") + 1])
        result_path.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(table1.subprocess, "run", fake_run)
    assert (
        table1._run_json_worker(
            launcher,
            ["_worker-inspect"],
            environment={},
            work_parent=tmp_path,
        )
        == {}
    )


def test_partial_cache_recovery_reopens_modes_without_losing_bytes(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    nested = cache / "nested"
    nested.mkdir(parents=True)
    weight = nested / "weight.pth"
    weight.write_bytes(b"already-downloaded-weight")
    alias = cache / "alias.pth"
    alias.symlink_to(weight.relative_to(cache))
    original_inode = weight.stat().st_ino
    original_sha = sha256_file(weight)
    table1._freeze_cache(cache)

    receipt = table1._recover_unlocked_cache_for_prefetch(cache, data_root=tmp_path)
    table1._validate_partial_cache_recovery(receipt)
    assert receipt["cache_preexisting"] is True
    assert receipt["regular_file_count"] == 1
    assert receipt["symlink_count"] == 1
    assert receipt["modes_reopened"] >= 3
    assert weight.stat().st_ino == original_inode
    assert sha256_file(weight) == original_sha
    assert weight.read_bytes() == b"already-downloaded-weight"
    assert stat.S_IMODE(cache.stat().st_mode) == 0o755
    assert stat.S_IMODE(nested.stat().st_mode) == 0o755
    assert stat.S_IMODE(weight.stat().st_mode) == 0o644
    assert alias.is_symlink()

    second = table1._recover_unlocked_cache_for_prefetch(cache, data_root=tmp_path)
    assert second["modes_reopened"] == 0
    assert second["content_sha256_before"] == receipt["content_sha256_before"]


def test_partial_cache_recovery_rejects_escape_and_partial_lock(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    cache = tmp_path / "cache"
    cache.mkdir()
    escaped = cache / "escaped"
    escaped.symlink_to(outside)
    with pytest.raises(table1.Table1ContractError, match="escapes"):
        table1._recover_unlocked_cache_for_prefetch(cache, data_root=tmp_path)
    assert outside.read_bytes() == b"outside"

    escaped.unlink()
    partial_lock = cache / ".weights_lock.json.interrupted"
    partial_lock.write_bytes(b"do-not-delete")
    with pytest.raises(table1.Table1ContractError, match="weights-lock"):
        table1._recover_unlocked_cache_for_prefetch(cache, data_root=tmp_path)
    assert partial_lock.read_bytes() == b"do-not-delete"


def test_pinned_agenticir_sources_and_cli_help_are_cpu_only() -> None:
    bindings = table1.validate_pinned_sources(table1.default_source_paths())
    assert set(bindings) == set(table1.PINNED_SOURCE_SHA256)
    assert {
        label: binding["sha256"] for label, binding in bindings.items()
    } == table1.PINNED_SOURCE_SHA256

    completed = subprocess.run(
        [sys.executable, str(table1.DEFAULT_CLI_PATH), "score", "--help"],
        cwd=table1.PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--input-manifest" not in completed.stdout
    assert "--device" not in completed.stdout
    assert "--cache-root" not in completed.stdout
    assert "shard-index" not in completed.stdout
    rejected = subprocess.run(
        [sys.executable, str(table1.DEFAULT_CLI_PATH), "score", "--device", "cpu"],
        cwd=table1.PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert rejected.returncode == 2


def test_score_controller_publishes_and_resumes_without_worker_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _manifest_rows(dict(table1.EXPECTED_COUNTS))
    initial = _rng_state(0)
    source_bindings = {"pinned": {"sha256": "3" * 64}}
    implementation = {"module": {"sha256": "4" * 64}}
    cache_lock_path = tmp_path / "cache" / "weights_lock.json"
    cache_lock_path.parent.mkdir()
    input_lock = {
        "schema_version": table1.INPUT_LOCK_SCHEMA,
        "manifest": {"path": "/formal/table1_input.jsonl", "sha256": "5" * 64},
        "image_count": table1.EXPECTED_IMAGE_COUNT,
        "expected_counts": dict(table1.EXPECTED_COUNTS),
        "ordering": "test",
        "rows": rows,
    }
    cache_binding = {
        "path": str(cache_lock_path),
        "sha256": "6" * 64,
        "mode": 0o444,
        "lock": {
            "initial_rng_core": initial,
            "metric_runtime": table1.EXPECTED_METRIC_RUNTIME,
        },
    }
    monkeypatch.setattr(
        table1, "validate_pinned_sources", lambda _paths: source_bindings
    )
    monkeypatch.setattr(table1, "_implementation_bindings", lambda: implementation)
    monkeypatch.setattr(table1, "check_cache", lambda **_kwargs: cache_binding)
    monkeypatch.setattr(
        table1, "validate_input_manifest", lambda _path: (rows, input_lock)
    )

    launches = 0

    def fake_worker(
        request_path: Path, environment: dict[str, str], _work_root: Path
    ) -> None:
        nonlocal launches
        launches += 1
        request = json.loads(request_path.read_text())
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert environment["HF_HUB_OFFLINE"] == "1"
        before = initial
        for shard_index in range(
            request["start_shard"],
            (len(rows) + request["shard_size"] - 1) // request["shard_size"],
        ):
            after = _rng_state(shard_index + 1)
            _publish_shard(
                Path(request["shards_dir"]),
                index=shard_index,
                shard_size=request["shard_size"],
                expected_rows=rows,
                before=before,
                after=after,
                run_sha=request["run_contract_sha256"],
                input_sha=request["input_lock_sha256"],
            )
            before = after

    output = tmp_path / "scores"
    first = table1.score_table1(
        input_manifest=tmp_path / "unused.jsonl",
        output_root=output,
        cache_root=cache_lock_path.parent,
        reference_python=Path(sys.executable),
        source_paths={},
        device="cpu",
        shard_size=100,
        worker_launcher=fake_worker,
        enforce_data_disk=False,
    )
    assert launches == 1
    assert first["status"] == "COMPLETE"
    assert first["image_count"] == 1440
    assert stat.S_IMODE((output / "per_image.csv").stat().st_mode) == 0o444
    assert stat.S_IMODE((output / "summary.json").stat().st_mode) == 0o444
    summary = json.loads((output / "summary.json").read_text())
    assert summary["groups"]["A"]["combination_count"] == 8
    assert summary["metric_directions"]["lpips"] == "lower"

    second = table1.score_table1(
        input_manifest=tmp_path / "unused.jsonl",
        output_root=output,
        cache_root=cache_lock_path.parent,
        reference_python=Path(sys.executable),
        source_paths={},
        device="cpu",
        shard_size=100,
        worker_launcher=lambda *_args: pytest.fail("completed run launched a worker"),
        enforce_data_disk=False,
    )
    assert second == first
    assert launches == 1


def test_formal_score_forbids_worker_launcher_test_seam() -> None:
    with pytest.raises(table1.Table1ContractError, match="forbids"):
        table1.score_table1(
            input_manifest=table1.FORMAL_TABLE1_INPUT_PATH,
            output_root=table1.FORMAL_SCORE_ROOT,
            cache_root=table1.DEFAULT_CACHE_ROOT,
            reference_python=table1.DEFAULT_REFERENCE_PYTHON,
            source_paths=table1.default_source_paths(),
            device=table1.FORMAL_DEVICE,
            shard_size=table1.FORMAL_SHARD_SIZE,
            worker_launcher=lambda *_args: None,
            enforce_formal=True,
        )


def test_gpu_ownership_and_vram_ceiling_are_fail_closed() -> None:
    def runner_with(output: str):
        def fake_runner(
            *_args: object, **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, stdout=output, stderr="")

        return fake_runner

    table1._assert_gpu_ownership(set(), runner=runner_with(""))
    table1._assert_gpu_ownership({123}, runner=runner_with("123\n"))
    with pytest.raises(table1.Table1ContractError, match="ownership mismatch"):
        table1._assert_gpu_ownership({123}, runner=runner_with("123\n456\n"))
    assert table1._validate_peak_reserved(899, 1000)[
        "peak_reserved_fraction"
    ] == pytest.approx(0.899)
    with pytest.raises(table1.Table1ContractError, match="not below"):
        table1._validate_peak_reserved(900, 1000)


def test_score_root_rejects_symlink_escape_and_unknown_entries(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = data_root / "escaped"
    escaped.symlink_to(outside, target_is_directory=True)
    with pytest.raises(table1.Table1ContractError, match="symlink"):
        table1._prepare_score_root(escaped, data_root=data_root)

    unknown_root = data_root / "unknown"
    unknown_root.mkdir()
    (unknown_root / "surprise.bin").write_bytes(b"x")
    with pytest.raises(table1.Table1ContractError, match="unauthorized"):
        table1._assert_score_tree_shape(unknown_root, data_root=data_root)

    for dirname in ("shards", ".worker"):
        root = data_root / dirname.replace(".", "worker-")
        root.mkdir()
        lock = root / "input_lock.json"
        lock.write_text("{}\n")
        os.chmod(lock, 0o444)
        contract = root / "run_contract.json"
        contract.write_text("{}\n")
        os.chmod(contract, 0o444)
        (root / dirname).symlink_to(outside, target_is_directory=True)
        with pytest.raises(table1.Table1ContractError, match="symlink"):
            table1._assert_score_tree_shape(root, data_root=data_root)


def test_formal_evidence_missing_or_tampered_is_rejected() -> None:
    binding = {"path": "/frozen", "sha256": "a" * 64}
    evidence = {
        key: ("b" * 64 if key == "predictions_digest" else dict(binding))
        for key in table1._FORMAL_EVIDENCE_KEYS
    }

    def missing() -> dict[str, object]:
        raise table1.Table1ContractError("authorization missing")

    with pytest.raises(table1.Table1ContractError, match="authorization missing"):
        table1._revalidate_formal_evidence(evidence, loader=missing)
    changed = dict(evidence)
    changed["predictions_digest"] = "c" * 64
    with pytest.raises(table1.Table1ContractError, match="binding changed"):
        table1._revalidate_formal_evidence(evidence, loader=lambda: changed)


def test_public_completion_helper_is_mandatory_and_hash_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.evaluation import mio100

    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    paths = {
        "authorization": tmp_path / "FORMAL_MIO100_APPROVED.json",
        "evaluator_complete": formal_root / "complete.json",
        "run_contract": formal_root / "run_contract.json",
        "summary": formal_root / "summary.json",
        "per_image": formal_root / "per_image.csv",
        "table1_input": formal_root / "table1_input.jsonl",
        "checkpoint": tmp_path / "best.pth",
        "manifest": tmp_path / "manifest.jsonl",
        "formal_data_inventory": tmp_path / "formal_data_inventory.json",
        "metric_parity_summary": tmp_path / "metric_parity_summary.json",
    }
    for label, path in paths.items():
        path.write_text(f"{label}\n", encoding="utf-8")
        os.chmod(path, 0o444)
    os.chmod(paths["metric_parity_summary"], 0o600)
    helper_evidence = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in paths.items()
        if label != "metric_parity_summary"
    }
    helper_evidence["predictions_digest"] = "d" * 64
    authorization = SimpleNamespace(
        bindings={
            name: SimpleNamespace(path=paths[name], sha256=sha256_file(paths[name]))
            for name in (
                "checkpoint",
                "manifest",
                "formal_data_inventory",
                "metric_parity_summary",
            )
        }
    )
    authorization.bindings["stage4_checkpoint"] = authorization.bindings.pop(
        "checkpoint"
    )
    authorization.bindings["formal_manifest"] = authorization.bindings.pop("manifest")
    monkeypatch.setattr(table1, "FORMAL_AUTHORIZATION_PATH", paths["authorization"])
    monkeypatch.setattr(table1, "FORMAL_EVALUATOR_ROOT", formal_root)
    monkeypatch.setattr(
        table1, "FORMAL_EVALUATOR_COMPLETE_PATH", paths["evaluator_complete"]
    )
    monkeypatch.setattr(table1, "FORMAL_TABLE1_INPUT_PATH", paths["table1_input"])
    monkeypatch.setattr(
        mio100,
        "validate_formal_authorization",
        lambda *_args, **_kwargs: authorization,
    )
    monkeypatch.setattr(
        mio100,
        "validate_formal_evaluator_complete",
        lambda *_args, **_kwargs: {"evidence": helper_evidence},
    )
    accepted = table1._load_formal_evidence()
    assert accepted["predictions_digest"] == "d" * 64
    assert accepted["metric_parity_summary"]["sha256"] == sha256_file(
        paths["metric_parity_summary"]
    )

    tampered = json.loads(json.dumps(helper_evidence))
    tampered["summary"]["sha256"] = "e" * 64
    monkeypatch.setattr(
        mio100,
        "validate_formal_evaluator_complete",
        lambda *_args, **_kwargs: {"evidence": tampered},
    )
    with pytest.raises(table1.Table1ContractError, match="hash drifted"):
        table1._load_formal_evidence()

    def missing(*_args: object, **_kwargs: object) -> object:
        raise mio100.MiO100EvaluationError("authorization missing")

    monkeypatch.setattr(mio100, "validate_formal_authorization", missing)
    with pytest.raises(table1.Table1ContractError, match="authorization missing"):
        table1._load_formal_evidence()


def test_evaluator_psnr_ssim_crosscheck_detects_per_image_drift(
    tmp_path: Path,
) -> None:
    inputs = _manifest_rows(_all_one_counts())[:2]
    records: list[dict[str, object]] = []
    evaluator_rows: list[dict[str, object]] = []
    digest_rows: list[dict[str, str]] = []
    for index, input_row in enumerate(inputs):
        record = _score_row(input_row, 0.5)
        record["psnr"] = 30.0 + index
        record["ssim"] = 0.8 + index * 0.01
        records.append(record)
        evaluator_rows.append(
            {
                "sample_id": input_row["sample_id"],
                "group": input_row["group"],
                "combination": input_row["combination"],
                "clean_id": f"clean-{index}",
                "prediction_png": input_row["prediction_png"],
                "prediction_sha256": input_row["prediction_sha256"],
                "target_png": input_row["target_png"],
                "target_sha256": input_row["target_sha256"],
                "psnr": record["psnr"],
                "ssim": record["ssim"],
                "latency_ms": 1.0,
                "program_levels": 1,
                "parallel_levels": 0,
                "active_skill_calls": 1,
                "reentry_requests": 0,
                "unexpected_activations": 0,
                "precycle_graphs": 0,
                "dropped_edges": 0,
                "peak_reserved_fraction": 0.1,
            }
        )
        digest_rows.append(
            {
                "sample_id": str(input_row["sample_id"]),
                "prediction_sha256": str(input_row["prediction_sha256"]),
                "target_sha256": str(input_row["target_sha256"]),
            }
        )
    evaluator_csv = tmp_path / "per_image.csv"

    def write_evaluator() -> None:
        with evaluator_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=table1._EVALUATOR_PER_IMAGE_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(evaluator_rows)
        os.chmod(evaluator_csv, 0o444)

    write_evaluator()
    parity = tmp_path / "metric_parity.json"
    parity.write_text(
        json.dumps(
            {
                "passed": True,
                "facts": {
                    "max_psnr_abs_diff": 0.0,
                    "max_ssim_abs_diff": 3.8790801337729164e-7,
                },
            }
        )
    )
    # This long-lived project evidence is hash-bound by authorization but is
    # not itself part of the scorer's newly frozen output tree.
    os.chmod(parity, 0o600)
    result = table1.crosscheck_evaluator_psnr_ssim(
        records,
        evaluator_csv=evaluator_csv,
        metric_parity_summary=parity,
        predictions_digest=sha256_json(digest_rows),
        expected_count=2,
    )
    assert result["passed"] is True
    assert result["psnr_max_abs_difference"] == 0.0

    os.chmod(evaluator_csv, 0o644)
    evaluator_rows[0]["psnr"] = 30.01
    write_evaluator()
    with pytest.raises(table1.Table1ContractError, match="psnr drift"):
        table1.crosscheck_evaluator_psnr_ssim(
            records,
            evaluator_csv=evaluator_csv,
            metric_parity_summary=parity,
            predictions_digest=sha256_json(digest_rows),
            expected_count=2,
        )
