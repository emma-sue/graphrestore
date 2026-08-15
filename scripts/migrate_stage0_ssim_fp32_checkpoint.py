#!/usr/bin/env python3
"""One-time, fail-closed Stage0 step-12000 SSIM provenance migration.

This tool does not advance training state.  It accepts only the exact
step-12000 transaction boundary produced before the FP32 training-SSIM repair,
replaces only the checkpoint provenance, proves that every other value survived
the save/load round trip bit-for-bit, and then publishes a receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.stage0_engine import (  # noqa: E402
    PROTOCOL_ID,
    STAGE0_CHECKPOINT_STAGE,
    Stage0Runtime,
    build_stage0_provenance,
    load_and_validate_stage0_config,
)
from src.utils.git import git_commit  # noqa: E402
from src.utils.hashing import is_sha256, sha256_file, sha256_json  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    fsync_directory,
    iter_jsonl,
    load_json,
    utc_now_iso,
)


CHECKPOINT_SCHEMA = "graphrestore-checkpoint-v1"
RECEIPT_SCHEMA = "graphrestore-stage0-ssim-fp32-migration-v1"
MIGRATION_STEP = 12_000
EXPECTED_TENSOR_KEYS = 495
METRIC_SOURCE_RELATIVE = "src/metrics/agenticir_official.py"
COMPILE_ARTIFACT_RELATIVE = "artifacts/audits/stage0_compile_ab.json"
EXPECTED_DIFF_PATHS = (
    f"semantic_source_sha256.{METRIC_SOURCE_RELATIVE}",
    f"compile_ab.code_sha256.{METRIC_SOURCE_RELATIVE}",
    "compile_ab.sha256",
)
EXPECTED_OLD_PROVENANCE_JSON_SHA256 = (
    "25c56dfde62484cdb55462d6c327a1933e1391e4eb353c83043c507a85975187"
)
EXPECTED_NEW_PROVENANCE_JSON_SHA256 = (
    "aa38a9175489423eea103c7520e1c871f9b621b7d6c79e18d97db1967ee59200"
)
FRESH_GATE_ARTIFACTS = (
    "artifacts/audits/data_audit.json",
    "artifacts/audits/degradation_parity.json",
    "artifacts/metrics/metric_parity_summary.json",
    "artifacts/metrics/metric_parity_per_image.csv",
    "artifacts/audits/validation_vram_probe.json",
    COMPILE_ARTIFACT_RELATIVE,
    "artifacts/integration/stage0_ssim_fp32_recovery_100_steps/summary.json",
    "artifacts/integration/stage0_ssim_fp32_recovery_100_steps/micro_batch_probe.json",
    "artifacts/integration/stage0_ssim_fp32_recovery_100_steps/train.jsonl",
    "artifacts/integration/stage0_ssim_fp32_recovery_100_steps/INTEGRATION_REPORT.md",
    "artifacts/integration/stage0_ssim_fp32_recovery_100_steps/last.pth",
)
EXPECTED_OLD_WARM_START = {
    "source_tensor_count": EXPECTED_TENSOR_KEYS,
    "loaded_count": EXPECTED_TENSOR_KEYS,
    "missing_keys": [],
    "unexpected_keys": [],
    "shape_mismatches": [],
}
EXPECTED_BUILDER_NONE_WARM_START = {
    "source_tensor_count": EXPECTED_TENSOR_KEYS,
    "loaded_count": EXPECTED_TENSOR_KEYS,
}


class Stage0SsimMigrationError(RuntimeError):
    """The candidate does not satisfy the one-time migration contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-role",
        choices=("raw_training_state", "ema_selection"),
        required=True,
    )
    return parser


def _fail(message: str) -> NoReturn:
    raise Stage0SsimMigrationError(message)


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _qualified_type(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _tensor_raw_bytes(value: torch.Tensor) -> bytes:
    if value.layout is not torch.strided:
        _fail(f"unsupported tensor layout in migration state: {value.layout}")
    flat = value.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _walk_finite(value: object, *, path: str = "checkpoint") -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            _fail(f"non-finite tensor at {path}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(
            np.isfinite(value).all()
        ):
            _fail(f"non-finite numpy array at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_finite(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_finite(child, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"non-finite Python float at {path}")


def _update_fingerprint(
    digest: Any,
    value: object,
    counts: Counter[str],
) -> None:
    counts["nodes"] += 1
    type_name = _qualified_type(value).encode("utf-8")
    digest.update(struct.pack(">I", len(type_name)))
    digest.update(type_name)

    if isinstance(value, torch.Tensor):
        raw = _tensor_raw_bytes(value)
        counts["tensors"] += 1
        counts["tensor_numel"] += value.numel()
        counts["tensor_bytes"] += len(raw)
        metadata = json.dumps(
            {
                "dtype": str(value.dtype),
                "layout": str(value.layout),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "requires_grad": value.requires_grad,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
        return
    if isinstance(value, np.ndarray):
        raw = value.tobytes(order="A")
        counts["numpy_arrays"] += 1
        counts["numpy_bytes"] += len(raw)
        metadata = json.dumps(
            {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "strides": list(value.strides),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
        return
    if isinstance(value, Mapping):
        counts["mappings"] += 1
        digest.update(struct.pack(">Q", len(value)))
        for key, child in value.items():
            _update_fingerprint(digest, key, counts)
            _update_fingerprint(digest, child, counts)
        return
    if isinstance(value, (list, tuple)):
        counts["sequences"] += 1
        digest.update(struct.pack(">Q", len(value)))
        for child in value:
            _update_fingerprint(digest, child, counts)
        return
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bool):
        digest.update(b"true" if value else b"false")
        return
    if isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        return
    if isinstance(value, float):
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, str):
        encoded = str(value).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        return
    _fail(f"unsupported checkpoint value type: {_qualified_type(value)}")


def _fingerprint(value: object) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    _update_fingerprint(digest, value, counts)
    return {"sha256": digest.hexdigest(), "counts": dict(sorted(counts.items()))}


def _assert_bit_exact(before: object, after: object, *, path: str) -> None:
    if type(before) is not type(after):
        _fail(
            f"state type mutation at {path}: "
            f"{_qualified_type(before)} != {_qualified_type(after)}"
        )
    if isinstance(before, torch.Tensor):
        assert isinstance(after, torch.Tensor)
        metadata_before = (
            before.dtype,
            before.layout,
            tuple(before.shape),
            tuple(before.stride()),
            before.storage_offset(),
            before.requires_grad,
        )
        metadata_after = (
            after.dtype,
            after.layout,
            tuple(after.shape),
            tuple(after.stride()),
            after.storage_offset(),
            after.requires_grad,
        )
        if metadata_before != metadata_after or _tensor_raw_bytes(
            before
        ) != _tensor_raw_bytes(after):
            _fail(f"tensor mutation at {path}")
        return
    if isinstance(before, np.ndarray):
        assert isinstance(after, np.ndarray)
        if (
            before.dtype != after.dtype
            or before.shape != after.shape
            or before.strides != after.strides
            or before.tobytes(order="A") != after.tobytes(order="A")
        ):
            _fail(f"numpy state mutation at {path}")
        return
    if isinstance(before, Mapping):
        assert isinstance(after, Mapping)
        before_keys = list(before.keys())
        after_keys = list(after.keys())
        if before_keys != after_keys:
            _fail(f"mapping key/order mutation at {path}")
        for key in before_keys:
            _assert_bit_exact(before[key], after[key], path=f"{path}.{key}")
        return
    if isinstance(before, (list, tuple)):
        assert isinstance(after, (list, tuple))
        if len(before) != len(after):
            _fail(f"sequence length mutation at {path}")
        for index, (old_child, new_child) in enumerate(zip(before, after, strict=True)):
            _assert_bit_exact(old_child, new_child, path=f"{path}[{index}]")
        return
    if isinstance(before, float):
        if struct.pack(">d", before) != struct.pack(">d", after):
            _fail(f"float mutation at {path}")
        return
    if before != after:
        _fail(f"state mutation at {path}: {before!r} != {after!r}")


def _assert_tensor_mapping_equal(
    model: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> None:
    if list(model) != list(shadow):
        _fail("EMA selection model keys differ from ema.shadow keys")
    for key in model:
        model_tensor = model[key]
        shadow_tensor = shadow[key]
        if not isinstance(model_tensor, torch.Tensor) or not isinstance(
            shadow_tensor, torch.Tensor
        ):
            _fail(f"EMA selection state is not tensor-valued at {key}")
        if (
            model_tensor.dtype != shadow_tensor.dtype
            or model_tensor.layout != shadow_tensor.layout
            or tuple(model_tensor.shape) != tuple(shadow_tensor.shape)
            or tuple(model_tensor.stride()) != tuple(shadow_tensor.stride())
            or model_tensor.storage_offset() != shadow_tensor.storage_offset()
            or model_tensor.requires_grad != shadow_tensor.requires_grad
            or _tensor_raw_bytes(model_tensor) != _tensor_raw_bytes(shadow_tensor)
        ):
            _fail(f"EMA selection model differs from ema.shadow at {key}")


def _validate_checkpoint_header(
    payload: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    required = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": STAGE0_CHECKPOINT_STAGE,
        "step": MIGRATION_STEP,
        "model_role": expected_role,
        "pending_validation_step": None,
    }
    for key, expected in required.items():
        if key not in payload:
            _fail(f"checkpoint header is missing required field: {key}")
        actual = payload.get(key)
        if key == "step" and isinstance(actual, bool):
            _fail("checkpoint step must be an integer, not bool")
        if actual != expected:
            _fail(
                f"checkpoint header mismatch at {key}: "
                f"expected {expected!r}, got {actual!r}"
            )
    expected_resumable = expected_role == "raw_training_state"
    if payload.get("resumable") is not expected_resumable:
        _fail(
            "checkpoint resumable mismatch: "
            f"role {expected_role!r} requires {expected_resumable}"
        )

    sampler = _require_mapping(payload.get("sampler_state"), field="sampler_state")
    if sampler.get("consumed_optimizer_step") != MIGRATION_STEP:
        _fail("sampler consumed_optimizer_step must equal 12000")
    if sampler.get("sample_cursor") != MIGRATION_STEP * 8:
        _fail("sampler sample_cursor must equal 96000")

    model = _require_mapping(payload.get("model"), field="model")
    ema = _require_mapping(payload.get("ema"), field="ema")
    shadow = _require_mapping(ema.get("shadow"), field="ema.shadow")
    if len(model) != EXPECTED_TENSOR_KEYS or len(shadow) != EXPECTED_TENSOR_KEYS:
        _fail(
            "checkpoint model/ema.shadow must contain exactly "
            f"{EXPECTED_TENSOR_KEYS}/{EXPECTED_TENSOR_KEYS} entries"
        )
    if expected_role == "ema_selection":
        _assert_tensor_mapping_equal(model, shadow)


def _flatten_provenance(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            _fail("provenance mappings require non-empty string keys")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            flattened.update(_flatten_provenance(child, prefix=path))
        else:
            flattened[path] = child
    return flattened


def _exact_provenance_diff(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> list[dict[str, str]]:
    old_flat = _flatten_provenance(old)
    new_flat = _flatten_provenance(new)
    if old_flat.keys() != new_flat.keys():
        only_old = sorted(old_flat.keys() - new_flat.keys())
        only_new = sorted(new_flat.keys() - old_flat.keys())
        _fail(f"provenance leaf set changed: only_old={only_old}, only_new={only_new}")
    changed = sorted(path for path in old_flat if old_flat[path] != new_flat[path])
    if tuple(changed) != tuple(sorted(EXPECTED_DIFF_PATHS)):
        _fail(
            "unexpected provenance diff: "
            f"expected={sorted(EXPECTED_DIFF_PATHS)}, actual={changed}"
        )
    result: list[dict[str, str]] = []
    for path in changed:
        old_value = old_flat[path]
        new_value = new_flat[path]
        if not is_sha256(old_value) or not is_sha256(new_value):
            _fail(f"provenance diff at {path} is not SHA256-to-SHA256")
        result.append({"path": path, "old": old_value, "new": new_value})
    return result


def _git_bytes(project_root: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        _fail(f"git {' '.join(arguments)} failed: {detail}")
    return process.stdout


def _is_git_object_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _metric_git_facts(project_root: Path) -> dict[str, str]:
    change_commit = (
        _git_bytes(
            project_root,
            "log",
            "-1",
            "--format=%H",
            "--",
            METRIC_SOURCE_RELATIVE,
        )
        .decode("ascii")
        .strip()
    )
    if not _is_git_object_id(change_commit):
        _fail("could not resolve the metric repair commit")
    old_source = _git_bytes(
        project_root,
        "show",
        f"{change_commit}^:{METRIC_SOURCE_RELATIVE}",
    )
    new_source = _git_bytes(
        project_root,
        "show",
        f"{change_commit}:{METRIC_SOURCE_RELATIVE}",
    )
    return {
        "change_commit": change_commit,
        "before_sha256": hashlib.sha256(old_source).hexdigest(),
        "after_sha256": hashlib.sha256(new_source).hexdigest(),
    }


def _validate_provenance_facts(
    *,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    runtime: Stage0Runtime,
    project_root: Path,
) -> dict[str, Any]:
    old_semantic = _require_mapping(
        old.get("semantic_source_sha256"), field="old.semantic_source_sha256"
    )
    new_semantic = _require_mapping(
        new.get("semantic_source_sha256"), field="new.semantic_source_sha256"
    )
    old_compile = _require_mapping(old.get("compile_ab"), field="old.compile_ab")
    new_compile = _require_mapping(new.get("compile_ab"), field="new.compile_ab")
    old_code = _require_mapping(
        old_compile.get("code_sha256"), field="old.compile_ab.code_sha256"
    )
    new_code = _require_mapping(
        new_compile.get("code_sha256"), field="new.compile_ab.code_sha256"
    )

    metric_path = project_root / METRIC_SOURCE_RELATIVE
    compile_path = project_root / COMPILE_ARTIFACT_RELATIVE
    metric_sha = sha256_file(metric_path)
    compile_sha = sha256_file(compile_path)
    compile_artifact = _require_mapping(
        load_json(compile_path), field="current compile A/B artifact"
    )
    artifact_code = _require_mapping(
        compile_artifact.get("code_sha256"), field="compile artifact code_sha256"
    )
    git_facts = _metric_git_facts(project_root)

    if metric_sha != git_facts["after_sha256"]:
        _fail("working metric source differs from the repair commit")
    if old_semantic.get(METRIC_SOURCE_RELATIVE) != git_facts["before_sha256"]:
        _fail("old semantic metric hash differs from the pre-repair Git source")
    if old_code.get(METRIC_SOURCE_RELATIVE) != git_facts["before_sha256"]:
        _fail("old compile code hash differs from the pre-repair Git source")
    for actual, label in (
        (new_semantic.get(METRIC_SOURCE_RELATIVE), "new semantic metric hash"),
        (new_code.get(METRIC_SOURCE_RELATIVE), "new compile code hash"),
        (artifact_code.get(METRIC_SOURCE_RELATIVE), "compile artifact metric hash"),
    ):
        if actual != metric_sha:
            _fail(f"{label} differs from the current metric source")
    if new_compile.get("sha256") != compile_sha:
        _fail("new compile provenance hash differs from the current artifact")
    if old_compile.get("sha256") == compile_sha:
        _fail("old compile artifact hash unexpectedly equals the repaired artifact")

    artifact_decision = compile_artifact.get("decision")
    if (
        compile_artifact.get("recommend_torch_compile") is not False
        or compile_artifact.get("safe_default") != "eager"
        or not isinstance(artifact_decision, str)
        or not artifact_decision.startswith("disable:")
    ):
        _fail("current compile A/B decision is not the locked false/eager decision")
    for provenance, label in ((old_compile, "old"), (new_compile, "new")):
        if (
            provenance.get("recommend_torch_compile") is not False
            or provenance.get("decision") != artifact_decision
        ):
            _fail(f"{label} provenance changed the false/eager compile decision")
    if runtime.torch_compile is not False:
        _fail("Stage0 migration requires the frozen eager runtime")
    if (
        runtime.schedule_max_steps != 60_000
        or runtime.target_step != 60_000
        or runtime.integration is not False
    ):
        _fail("Stage0 migration requires the formal 60k runtime")
    if old.get("protocol_id") != PROTOCOL_ID or new.get("protocol_id") != PROTOCOL_ID:
        _fail("Stage0 protocol_id drifted")
    if (
        old.get("stage") != STAGE0_CHECKPOINT_STAGE
        or new.get("stage") != STAGE0_CHECKPOINT_STAGE
    ):
        _fail("Stage0 provenance stage drifted")
    if old.get("runtime") != asdict(runtime) or new.get("runtime") != asdict(runtime):
        _fail("Stage0 runtime changed during provenance reconstruction")

    return {
        "metric_source": {
            "path": str(metric_path),
            "sha256": metric_sha,
            **git_facts,
        },
        "compile_ab": {
            "path": str(compile_path),
            "sha256": compile_sha,
            "recommend_torch_compile": False,
            "safe_default": "eager",
            "decision": artifact_decision,
        },
    }


def _reconstruct_current_provenance(
    *,
    old: Mapping[str, Any],
    config_path: Path,
    project_root: Path,
) -> tuple[dict[str, Any], Stage0Runtime, dict[str, Any]]:
    old_runtime = _require_mapping(old.get("runtime"), field="old.runtime")
    try:
        runtime = Stage0Runtime(**dict(old_runtime))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Stage0SsimMigrationError(
            f"invalid frozen Stage0 runtime: {type(exc).__name__}: {exc}"
        ) from exc

    old_warm = _require_mapping(old.get("warm_start_load"), field="old.warm_start_load")
    if dict(old_warm) != EXPECTED_OLD_WARM_START:
        _fail("old warm_start_load is not the exact 495/495 zero-diagnostic report")

    config_sha_before = sha256_file(config_path)
    config, resolved = load_and_validate_stage0_config(config_path)
    expected = build_stage0_provenance(
        project_root=project_root,
        config_path=config_path,
        config=config,
        resolved=resolved,
        runtime=runtime,
        load_report=None,
    )
    if not isinstance(expected, dict):
        _fail("current Stage0 provenance builder did not return a dict")
    builder_warm = _require_mapping(
        expected.get("warm_start_load"), field="builder warm_start_load"
    )
    if dict(builder_warm) != EXPECTED_BUILDER_NONE_WARM_START:
        _fail("load_report=None warm-start builder behavior drifted")

    # A real train_stage0 resume reconstructs provenance with the strict load
    # report, whose three diagnostics are empty at this 495/495 parent.  Keep
    # those already-proven empty leaves so the migrated checkpoint remains
    # resumable while still calling the required current builder with None.
    expected = copy.deepcopy(expected)
    expected["warm_start_load"] = dict(EXPECTED_OLD_WARM_START)
    if sha256_file(config_path) != config_sha_before:
        _fail("Stage0 config changed while reconstructing provenance")
    return expected, runtime, {"config_sha256": config_sha_before}


def _audit_train_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing Stage0 train log beside checkpoint: {path}")
    digest_before = sha256_file(path)
    accepted_train_steps: list[int] = []
    discarded_rows: list[tuple[int, str]] = []
    discarded_events: Counter[str] = Counter()
    for line_number, row in iter_jsonl(path):
        step = row.get("step")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
            _fail(f"{path}:{line_number}: step must be an integer when present")
        if isinstance(step, int) and step > MIGRATION_STEP:
            event = row.get("event")
            event_name = event if isinstance(event, str) else "<invalid>"
            discarded_rows.append((step, event_name))
            discarded_events[event_name] += 1
        if row.get("event") != "train_step":
            continue
        if not isinstance(step, int) or step <= 0:
            _fail(f"{path}:{line_number}: train_step has invalid step")
        if step <= MIGRATION_STEP:
            for field in ("lambda_ssim", "ssim_loss"):
                value = row.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) != 0.0
                ):
                    _fail(
                        f"{path}:{line_number}: pre-boundary {field} must be finite zero"
                    )
            accepted_train_steps.append(step)
    if not accepted_train_steps:
        _fail("Stage0 train log has no pre-boundary train_step evidence")
    if min(accepted_train_steps) != 1 or max(accepted_train_steps) != MIGRATION_STEP:
        _fail("Stage0 train log must cover recorded train_step endpoints 1..12000")
    if sha256_file(path) != digest_before:
        _fail("Stage0 train log changed during audit")

    discarded_steps = [step for step, _ in discarded_rows]
    discarded = {
        "classification": "discarded_transient_not_in_step12000_checkpoint",
        "record_count": len(discarded_rows),
        "train_step_count": discarded_events.get("train_step", 0),
        "minimum_step": min(discarded_steps) if discarded_steps else None,
        "maximum_step": max(discarded_steps) if discarded_steps else None,
        "event_counts": dict(sorted(discarded_events.items())),
    }
    return {
        "path": str(path),
        "sha256": digest_before,
        "accepted_train_step_record_count": len(accepted_train_steps),
        "accepted_minimum_step": min(accepted_train_steps),
        "accepted_maximum_step": max(accepted_train_steps),
        "pre_boundary_lambda_ssim_all_zero": True,
        "pre_boundary_ssim_loss_all_zero": True,
        "discarded_transient": discarded,
    }


def _fresh_gate_hashes(project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in FRESH_GATE_ARTIFACTS:
        path = project_root / relative
        if not path.is_file():
            _fail(f"missing fresh recovery gate artifact: {path}")
        result[relative] = sha256_file(path)
    return result


def _load_cpu_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise Stage0SsimMigrationError(
            f"could not load source checkpoint on CPU: {type(exc).__name__}: {exc}"
        ) from exc
    return _require_mapping(payload, field="checkpoint")


def _section_evidence(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if list(before) != list(after):
        _fail("top-level checkpoint structure changed")
    result: dict[str, Any] = {}
    for key in before:
        old_fingerprint = _fingerprint(before[key])
        new_fingerprint = _fingerprint(after[key])
        if key != "provenance" and old_fingerprint != new_fingerprint:
            _fail(f"section fingerprint changed outside provenance: {key}")
        result[key] = {
            "old": old_fingerprint,
            "new": new_fingerprint,
            "bit_exact": old_fingerprint == new_fingerprint,
        }
    return result


def _make_candidate_path(parent: Path, name: str, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=suffix, dir=parent
    )
    os.close(descriptor)
    candidate = Path(temporary_name)
    candidate.unlink()
    return candidate


def _publish_link(candidate: Path, destination: Path) -> None:
    try:
        os.link(candidate, destination)
    except FileExistsError as exc:
        raise Stage0SsimMigrationError(
            f"refusing to overwrite existing output: {destination}"
        ) from exc
    fsync_directory(destination.parent)


def migrate_stage0_checkpoint(
    *,
    source: str | Path,
    destination: str | Path,
    config: str | Path,
    expected_source_sha256: str,
    receipt: str | Path,
    expected_role: str,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate, migrate, round-trip-check, and atomically publish one checkpoint."""

    root = Path(project_root).resolve()
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    config_path = Path(config).resolve()
    receipt_path = Path(receipt).resolve()
    if expected_role not in {"raw_training_state", "ema_selection"}:
        _fail(f"unsupported expected role: {expected_role!r}")
    if not is_sha256(expected_source_sha256):
        _fail("--expected-source-sha256 must be a lowercase SHA256")
    if not source_path.is_file():
        _fail(f"source checkpoint is missing: {source_path}")
    if not config_path.is_file():
        _fail(f"Stage0 config is missing: {config_path}")
    if destination_path.exists():
        _fail(f"refusing to overwrite existing destination: {destination_path}")
    if receipt_path.exists():
        _fail(f"refusing to overwrite existing receipt: {receipt_path}")
    if destination_path == receipt_path:
        _fail("destination and receipt must be different paths")

    source_sha = sha256_file(source_path)
    if source_sha != expected_source_sha256:
        _fail(
            "source checkpoint SHA256 mismatch: "
            f"expected {expected_source_sha256}, got {source_sha}"
        )
    payload = _load_cpu_checkpoint(source_path)
    if sha256_file(source_path) != source_sha:
        _fail("source checkpoint changed while loading")
    _validate_checkpoint_header(payload, expected_role=expected_role)
    _walk_finite(payload)

    old_provenance = _require_mapping(
        payload.get("provenance"), field="checkpoint.provenance"
    )
    expected_provenance, runtime, config_gate = _reconstruct_current_provenance(
        old=old_provenance,
        config_path=config_path,
        project_root=root,
    )
    exact_diff = _exact_provenance_diff(old_provenance, expected_provenance)
    old_provenance_sha = sha256_json(dict(old_provenance))
    new_provenance_sha = sha256_json(expected_provenance)
    if old_provenance_sha != EXPECTED_OLD_PROVENANCE_JSON_SHA256:
        _fail(
            "old provenance JSON hash is not the audited step12000 provenance: "
            f"{old_provenance_sha}"
        )
    if new_provenance_sha != EXPECTED_NEW_PROVENANCE_JSON_SHA256:
        _fail(
            "new provenance JSON hash is not the audited FP32-SSIM provenance: "
            f"{new_provenance_sha}"
        )
    fact_gates = _validate_provenance_facts(
        old=old_provenance,
        new=expected_provenance,
        runtime=runtime,
        project_root=root,
    )
    train_log = _audit_train_log(source_path.parent / "train.jsonl")
    fresh_gate_artifact_hashes = _fresh_gate_hashes(root)
    if sha256_file(source_path) != source_sha:
        _fail("source checkpoint changed during gate validation")

    migrated_payload = copy.copy(payload)
    migrated_payload["provenance"] = expected_provenance
    _walk_finite(migrated_payload)
    for key in payload:
        if key != "provenance":
            _assert_bit_exact(
                payload[key], migrated_payload[key], path=f"checkpoint.{key}"
            )

    destination_candidate = _make_candidate_path(
        destination_path.parent, destination_path.name, ".candidate.pth"
    )
    receipt_candidate: Path | None = None
    destination_published = False
    receipt_published = False
    try:
        atomic_torch_save(migrated_payload, destination_candidate)
        reloaded = _load_cpu_checkpoint(destination_candidate)
        _walk_finite(reloaded)
        for key in payload:
            if key == "provenance":
                continue
            if key not in reloaded:
                _fail(f"reloaded checkpoint lost section: {key}")
            _assert_bit_exact(payload[key], reloaded[key], path=f"checkpoint.{key}")
        if list(reloaded) != list(migrated_payload):
            _fail("reloaded checkpoint top-level keys/order changed")
        reloaded_provenance = _require_mapping(
            reloaded.get("provenance"), field="reloaded provenance"
        )
        if reloaded_provenance != expected_provenance:
            _fail(
                "reloaded checkpoint provenance differs from expected current provenance"
            )
        _exact_provenance_diff(old_provenance, reloaded_provenance)
        section_evidence = _section_evidence(payload, reloaded)

        new_checkpoint_sha = sha256_file(destination_candidate)
        gate_hashes = {
            "source_checkpoint_sha256": source_sha,
            "config_sha256": config_gate["config_sha256"],
            "train_log_sha256": train_log["sha256"],
            "metric_source_sha256": fact_gates["metric_source"]["sha256"],
            "compile_ab_sha256": fact_gates["compile_ab"]["sha256"],
            "migration_script_sha256": sha256_file(Path(__file__).resolve()),
            "fresh_gate_artifact_sha256": fresh_gate_artifact_hashes,
        }
        receipt_payload: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "created_utc": utc_now_iso(),
            "migration": "stage0_step12000_fp32_training_ssim_provenance_only",
            "expected_role": expected_role,
            "source": {
                "path": str(source_path),
                "sha256": source_sha,
                "expected_sha256": expected_source_sha256,
            },
            "destination": {
                "path": str(destination_path),
                "sha256": new_checkpoint_sha,
            },
            "receipt": {"path": str(receipt_path)},
            "config": {"path": str(config_path), **config_gate},
            "old_checkpoint_sha256": source_sha,
            "new_checkpoint_sha256": new_checkpoint_sha,
            "old_provenance_json_sha256": old_provenance_sha,
            "new_provenance_json_sha256": new_provenance_sha,
            "exact_provenance_leaf_diff": exact_diff,
            "section_fingerprints_and_counts": section_evidence,
            "gate_hashes": gate_hashes,
            "gate_hashes_sha256": sha256_json(gate_hashes),
            "metric_git_facts": fact_gates["metric_source"],
            "compile_decision": fact_gates["compile_ab"],
            "train_log_audit": train_log,
            "discarded_log_range": train_log["discarded_transient"],
            "code_commit": git_commit(root),
            "state_round_trip_bit_exact_outside_provenance": True,
            "all_checkpoint_tensors_finite": True,
            "destination_preexisted": False,
            "receipt_written_after_destination": True,
        }
        receipt_candidate = _make_candidate_path(
            receipt_path.parent, receipt_path.name, ".candidate.json"
        )
        atomic_write_json(receipt_candidate, receipt_payload)
        loaded_receipt = load_json(receipt_candidate)
        if loaded_receipt != receipt_payload:
            _fail("receipt JSON round trip changed its payload")

        # Hard-link publication is atomic and refuses replacement.  The receipt
        # is linked only after the verified checkpoint is present.  On any
        # catchable receipt publication failure, the newly-created destination
        # is rolled back.
        _publish_link(destination_candidate, destination_path)
        destination_published = True
        if sha256_file(destination_path) != new_checkpoint_sha:
            _fail("published destination hash differs from verified candidate")
        _publish_link(receipt_candidate, receipt_path)
        receipt_published = True
        return receipt_payload
    except BaseException:
        if destination_published and not receipt_published:
            destination_path.unlink(missing_ok=True)
            fsync_directory(destination_path.parent)
        raise
    finally:
        destination_candidate.unlink(missing_ok=True)
        if receipt_candidate is not None:
            receipt_candidate.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        receipt = migrate_stage0_checkpoint(
            source=arguments.source,
            destination=arguments.destination,
            config=arguments.config,
            expected_source_sha256=arguments.expected_source_sha256,
            receipt=arguments.receipt,
            expected_role=arguments.expected_role,
        )
    except Stage0SsimMigrationError as exc:
        print(f"Stage0 SSIM migration refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
