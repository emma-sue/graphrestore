"""Independent formal MiO100 control for the frozen Stage0 MiO-StageA.

The legacy Stage4 evaluator/scorer files are SHA-bound terminal evidence and
must remain byte-identical.  This module therefore owns the Stage0 approval,
checkpoint audit, run contract, completion validation, Table-1 evidence
adapter, and paired Stage4-minus-Stage0 comparison.  It reuses only the
already-audited immutable image transaction primitives.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import stat
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor

from src.evaluation import mio100 as shared
from src.evaluation.formal_inventory import (
    FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    FORMAL_DATA_INVENTORY_PATH,
    FORMAL_GROUP_COUNTS,
    FormalInventoryError,
    load_formal_data_inventory,
    write_new_read_only_json,
)
from src.evaluation.stage0_formal_inventory import (
    FROZEN_STAGE0_CHECKPOINT_SHA256,
    FROZEN_STAGE0_CONFIG_SHA256,
    FROZEN_STAGE0_MANIFEST_MAP_SHA256,
    FROZEN_STAGE0_PROVENANCE_SHA256,
    FROZEN_STAGE0_SEMANTIC_SOURCE_MAP_SHA256,
    PROJECT_ROOT,
    PROTOCOL_ID,
    REQUIRED_STAGE0_AUTHORIZATION_BINDINGS,
    STAGE0_APPROVAL_PATH,
    STAGE0_AUTHORIZATION_PROTOCOL_PATH,
    STAGE0_COMPARISON_ROOT,
    STAGE0_METHOD_NAME,
    STAGE0_OUTPUT_ROOT,
    STAGE0_PROVENANCE_COMPATIBILITY,
    STAGE0_SCORE_ROOT,
    STAGE0_USER_AUTHORIZATION_PROTOCOL_PATH,
    stage0_authorization_binding_paths,
    validate_stage0_lightweight_authorization,
    validate_stage0_readiness_receipt_without_torch,
    validate_stage0_ready_without_torch,
)
from src.metrics.agenticir_official import OFFICIAL_GROUPS, aggregate_official_records
from src.net.mio_stagea import MiOStageA
from src.utils.hashing import is_sha256, sha256_file, sha256_json
from src.utils.io import fsync_directory, load_json, utc_now_iso


STAGE0_RUN_CONTRACT_SCHEMA = "graphrestore-stage0-formal-mio100-run-contract-v1"
STAGE0_READINESS_SCHEMA = "graphrestore-stage0-formal-readiness-v1"
STAGE0_COMPARISON_SCHEMA = "graphrestore-stage0-vs-stage4-comparison-v1"
STAGE0_COMPARISON_COMPLETE_SCHEMA = (
    "graphrestore-stage0-vs-stage4-comparison-complete-v1"
)
STAGE0_PARAMETER_COUNT = 25_437_220
STAGE0_GRAPH_DIAGNOSTIC_COMPATIBILITY = {
    name: "N/A (prompt-free Stage0 compatibility placeholder)"
    for name in (
        "program_levels",
        "parallel_levels",
        "active_skill_calls",
        "reentry_requests",
        "unexpected_activations",
        "precycle_graphs",
        "dropped_edges",
    )
}
STAGE0_FORMAL_INFERENCE = {
    "prompt_free_mio_stagea": True,
    "task_label_input": False,
    "planner": False,
    "skill_bank": False,
    "amp_dtype": "bf16",
    "tf32": True,
    "tta": False,
    "model_soup": False,
    "parameter_count": STAGE0_PARAMETER_COUNT,
    "graph_diagnostic_columns": STAGE0_GRAPH_DIAGNOSTIC_COMPATIBILITY,
}
FORMAL_SHARD_COUNT = 1
MAXIMUM_VRAM_RESERVED_FRACTION = 0.90
METRICS = ("psnr", "ssim", "lpips", "maniqa", "clipiqa", "musiq")
SCORER_CSV_COLUMNS = (
    "sample_id",
    "group",
    "combination",
    "prediction_png",
    "prediction_sha256",
    "target_png",
    "target_sha256",
    *METRICS,
)


@dataclass(frozen=True)
class Stage0Checkpoint:
    path: Path
    sha256: str
    model_state: Mapping[str, Tensor]
    provenance: Mapping[str, Any]
    provenance_verification: Mapping[str, Any]
    metrics: Mapping[str, Any]


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise shared.MiO100EvaluationError(f"{field} must be a mapping")
    return value


def _validate_stage0_checkpoint_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    expected_config_path: Path | None,
    expected_project_root: Path,
    expected_semantic_source_count: int,
    expected_provenance_sha256: str,
    expected_semantic_source_map_sha256: str,
    expected_manifest_map_sha256: str,
    expected_provenance_compatibility: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Any]:
    required = {
        "protocol_id",
        "stage",
        "config_path",
        "config_sha256",
        "resolved_paths_sha256",
        "semantic_source_sha256",
        "manifests",
        "parent_checkpoint",
        "repositories",
        "runtime",
        "compile_ab",
        "warm_start_load",
        "dependency_versions",
    }
    if set(provenance) != required:
        raise shared.MiO100EvaluationError("Stage0 provenance fields drifted")
    if sha256_json(dict(provenance)) != expected_provenance_sha256:
        raise shared.MiO100EvaluationError("Stage0 whole provenance digest drifted")
    config_path = Path(str(provenance.get("config_path"))).resolve(strict=False)
    if expected_config_path is not None and config_path != expected_config_path:
        raise shared.MiO100EvaluationError("Stage0 provenance config path drifted")
    if (
        provenance.get("protocol_id") != PROTOCOL_ID
        or provenance.get("stage") != "stage0"
        or provenance.get("config_sha256") != expected_config_sha256
        or not is_sha256(provenance.get("resolved_paths_sha256"))
    ):
        raise shared.MiO100EvaluationError("Stage0 checkpoint provenance drifted")
    semantic = _mapping(
        provenance.get("semantic_source_sha256"),
        field="Stage0 semantic-source provenance",
    )
    mandatory_sources = {
        "scripts/train_stage0.py",
        "src/data/scale_canonicalizer.py",
        "src/metrics/agenticir_official.py",
        "src/net/mio_stagea.py",
        "src/training/stage0_engine.py",
    }
    if (
        len(semantic) != expected_semantic_source_count
        or not mandatory_sources.issubset(semantic)
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not is_sha256(digest)
            for name, digest in semantic.items()
        )
    ):
        raise shared.MiO100EvaluationError("Stage0 semantic-source provenance drifted")
    if sha256_json(dict(semantic)) != expected_semantic_source_map_sha256:
        raise shared.MiO100EvaluationError(
            "Stage0 semantic-source identity map drifted"
        )
    verified_sources: dict[str, Mapping[str, str]] = {}
    compatibility_mismatches: dict[str, Mapping[str, str]] = {}
    for name, expected_sha256 in sorted(semantic.items()):
        source = _canonical_file(
            expected_project_root / name,
            field=f"Stage0 semantic source {name}",
        )
        if source != expected_project_root / name:
            raise shared.MiO100EvaluationError(
                f"Stage0 semantic source path drifted at {name}"
            )
        actual_sha256 = sha256_file(source)
        compatibility = expected_provenance_compatibility.get(name)
        if compatibility is None:
            if actual_sha256 != expected_sha256:
                raise shared.MiO100EvaluationError(
                    f"Stage0 semantic source bytes drifted at {name}"
                )
        else:
            if (
                set(compatibility)
                != {"checkpoint_sha256", "current_sha256", "rationale"}
                or expected_sha256 != compatibility["checkpoint_sha256"]
                or actual_sha256 != compatibility["current_sha256"]
                or expected_sha256 == actual_sha256
                or not is_sha256(compatibility["checkpoint_sha256"])
                or not is_sha256(compatibility["current_sha256"])
                or compatibility["rationale"]
                != "post_stage0_downstream_not_imported_by_formal_stage0"
            ):
                raise shared.MiO100EvaluationError(
                    f"Stage0 provenance compatibility drifted at {name}"
                )
            compatibility_mismatches[name] = {
                "path": str(source),
                "checkpoint_sha256": str(expected_sha256),
                "current_sha256": actual_sha256,
                "rationale": compatibility["rationale"],
            }
        verified_sources[name] = {
            "path": str(source),
            "sha256": actual_sha256,
        }
    if set(compatibility_mismatches) != set(expected_provenance_compatibility):
        raise shared.MiO100EvaluationError(
            "Stage0 provenance compatibility mismatch set drifted"
        )
    manifests = _mapping(provenance.get("manifests"), field="Stage0 manifests")
    if set(manifests) != {"clean_train", "clean_val", "primary_train", "primary_val"}:
        raise shared.MiO100EvaluationError("Stage0 manifest provenance drifted")
    manifest_identity = {
        name: dict(_mapping(raw, field=f"Stage0 manifest {name}"))
        for name, raw in manifests.items()
    }
    if sha256_json(manifest_identity) != expected_manifest_map_sha256:
        raise shared.MiO100EvaluationError("Stage0 manifest identity map drifted")
    verified_manifests: dict[str, Mapping[str, str]] = {}
    for name, raw in sorted(manifests.items()):
        binding = _mapping(raw, field=f"Stage0 manifest {name}")
        path = binding.get("path")
        if (
            set(binding) != {"path", "sha256"}
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            or not is_sha256(binding.get("sha256"))
        ):
            raise shared.MiO100EvaluationError(
                f"Stage0 manifest provenance drifted at {name}"
            )
        manifest_path = _canonical_file(path, field=f"Stage0 manifest {name}")
        actual_sha256 = sha256_file(manifest_path)
        if actual_sha256 != binding["sha256"]:
            raise shared.MiO100EvaluationError(
                f"Stage0 manifest bytes drifted at {name}"
            )
        verified_manifests[name] = {
            "path": str(manifest_path),
            "sha256": actual_sha256,
        }
    parent = _mapping(
        provenance.get("parent_checkpoint"), field="Stage0 parent checkpoint"
    )
    if (
        set(parent) != {"path", "sha256"}
        or not isinstance(parent.get("path"), str)
        or not Path(str(parent["path"])).is_absolute()
        or not is_sha256(parent.get("sha256"))
    ):
        raise shared.MiO100EvaluationError("Stage0 parent provenance drifted")
    repositories = _mapping(provenance.get("repositories"), field="Stage0 repositories")
    if repositories != {
        "agenticir_commit": "9640a291480dee3ba8f2974125d4ee9e3440f3d6",
        "mioir_commit": "4d5f6ca0235cf2c307319673242d5722ee35d73f",
    }:
        raise shared.MiO100EvaluationError("Stage0 repository provenance drifted")
    warm_start = _mapping(
        provenance.get("warm_start_load"), field="Stage0 warm-start receipt"
    )
    if warm_start != {
        "source_tensor_count": 495,
        "loaded_count": 495,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
    }:
        raise shared.MiO100EvaluationError("Stage0 warm-start provenance drifted")
    compile_ab = _mapping(provenance.get("compile_ab"), field="Stage0 compile A/B")
    if (
        compile_ab.get("recommend_torch_compile") is not False
        or not is_sha256(compile_ab.get("sha256"))
        or not is_sha256(compile_ab.get("profile_script_sha256"))
    ):
        raise shared.MiO100EvaluationError("Stage0 compile provenance drifted")
    dependencies = _mapping(
        provenance.get("dependency_versions"), field="Stage0 dependencies"
    )
    if not dependencies:
        raise shared.MiO100EvaluationError("Stage0 dependency provenance is empty")
    return {
        "project_root": str(expected_project_root),
        "semantic_source_count": len(verified_sources),
        "manifest_binding_count": len(verified_manifests),
        "semantic_sources": verified_sources,
        "compatibility_mismatches": compatibility_mismatches,
        "manifests": verified_manifests,
    }


def _canonical_file(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise shared.MiO100EvaluationError(f"{field} must be canonical and absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise shared.MiO100EvaluationError(f"missing {field}: {candidate}") from exc
    if resolved != candidate or not resolved.is_file():
        raise shared.MiO100EvaluationError(f"{field} is not a canonical regular file")
    return resolved


def _require_read_only(path: Path, *, field: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise shared.MiO100EvaluationError(f"{field} must be immutable/read-only")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_bytes(path: Path, payload: bytes) -> None:
    # Delegate the already-audited no-clobber/fsync implementation.  This is a
    # private import by design; the legacy source itself remains byte-exact.
    shared._write_new_bytes(path, payload)  # noqa: SLF001


def _write_or_verify(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        existing = _canonical_file(path, field=f"published artifact {path.name}")
        _require_read_only(existing, field=f"published artifact {path.name}")
        if existing.read_bytes() != payload:
            raise shared.MiO100EvaluationError(
                f"published artifact content drifted: {path}"
            )
        return
    _write_new_bytes(path, payload)


def bind_default_stage0_authorization_paths(
    project_root: str | Path,
    *,
    manifest: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    summary: str | Path,
    primary_validation: str | Path,
    calibration_history: str | Path,
    report: str | Path,
    readiness: str | Path,
    stage1_run_contract: str | Path | None = None,
    formal_data_inventory: str | Path = FORMAL_DATA_INVENTORY_PATH,
) -> Mapping[str, Path]:
    return stage0_authorization_binding_paths(
        project_root,
        manifest=manifest,
        formal_data_inventory=formal_data_inventory,
        checkpoint=checkpoint,
        config=config,
        summary=summary,
        primary_validation=primary_validation,
        calibration_history=calibration_history,
        report=report,
        readiness=readiness,
        stage1_run_contract=stage1_run_contract,
        authorization_protocol=STAGE0_AUTHORIZATION_PROTOCOL_PATH,
    )


def default_stage0_authorization_paths() -> Mapping[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return bind_default_stage0_authorization_paths(
        root,
        manifest=root / "manifests/mio100_test_1440_agenticir_online_canonical.jsonl",
        checkpoint=root / "artifacts/checkpoints/stage0/best_ema.pth",
        config=root / "configs/stage0_mio_stagea.yaml",
        summary=root / "artifacts/checkpoints/stage0/summary.json",
        primary_validation=root
        / "artifacts/metrics/stage0_primary_val_step_060000.json",
        calibration_history=root / "artifacts/metrics/calibration_history.csv",
        report=root / "reports/STAGE0_MIO_STAGEA.md",
        readiness=root / "artifacts/audits/stage0_formal_readiness.json",
    )


def validate_stage0_formal_authorization(
    path: str | Path,
    *,
    expected_bindings: Mapping[str, str | Path],
    expected_output_root: str | Path = STAGE0_OUTPUT_ROOT,
    expected_method_name: str = STAGE0_METHOD_NAME,
) -> shared.FormalAuthorization:
    expected_paths = {
        name: Path(value).resolve(strict=False)
        for name, value in expected_bindings.items()
    }
    if set(expected_paths) != set(REQUIRED_STAGE0_AUTHORIZATION_BINDINGS):
        raise shared.MiO100EvaluationError(
            "Stage0 expected authorization binding keys drifted"
        )
    try:
        payload = validate_stage0_lightweight_authorization(
            path,
            expected_binding_paths=expected_paths,
        )
    except FormalInventoryError as exc:
        raise shared.MiO100EvaluationError(
            f"Stage0 formal authorization rejected: {exc}"
        ) from exc
    if (
        payload.get("output_root")
        != str(Path(expected_output_root).resolve(strict=False))
        or payload.get("method_name") != expected_method_name
        or payload.get("shard_count") != 1
    ):
        raise shared.MiO100EvaluationError("Stage0 formal authorization scope drifted")
    raw_bindings = _mapping(payload.get("bindings"), field="Stage0 bindings")
    bindings = {
        name: shared.ArtifactBinding(
            path=expected_paths[name], sha256=str(raw_bindings[name]["sha256"])
        )
        for name in REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    }
    approval = _canonical_file(path, field="Stage0 formal authorization")
    return shared.FormalAuthorization(
        path=approval,
        sha256=sha256_file(approval),
        approved_utc=str(payload["approved_utc"]),
        output_root=Path(str(payload["output_root"])),
        method_name=str(payload["method_name"]),
        shard_count=1,
        bindings=bindings,
    )


def validate_stage0_protocol_bindings(
    authorization: shared.FormalAuthorization,
) -> None:
    """Reuse Stage4's data/metric semantic validator with an explicit alias."""

    origin = authorization.bindings["inventory_origin_protocol"]
    if origin.path != FORMAL_AUTHORIZATION_PROTOCOL_PATH:
        raise shared.MiO100EvaluationError("inventory origin protocol path drifted")
    control = authorization.bindings["stage0_control_protocol"]
    if control.path != STAGE0_AUTHORIZATION_PROTOCOL_PATH:
        raise shared.MiO100EvaluationError("Stage0 control protocol path drifted")
    user = authorization.bindings["stage0_user_authorization_protocol"]
    if user.path != STAGE0_USER_AUTHORIZATION_PROTOCOL_PATH:
        raise shared.MiO100EvaluationError(
            "Stage0 user authorization protocol path drifted"
        )
    alias_bindings = dict(authorization.bindings)
    alias_bindings["formal_authorization_protocol"] = origin
    aliased = shared.FormalAuthorization(
        path=authorization.path,
        sha256=authorization.sha256,
        approved_utc=authorization.approved_utc,
        output_root=authorization.output_root,
        method_name=authorization.method_name,
        shard_count=authorization.shard_count,
        bindings=alias_bindings,
    )
    shared.validate_protocol_bindings(aliased)


def load_stage0_best_ema(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_config_sha256: str,
    expected_config_path: str | Path | None = None,
    expected_tensor_count: int | None = 495,
    expected_project_root: str | Path = PROJECT_ROOT,
    expected_semantic_source_count: int = 46,
    expected_provenance_sha256: str = FROZEN_STAGE0_PROVENANCE_SHA256,
    expected_semantic_source_map_sha256: str = (
        FROZEN_STAGE0_SEMANTIC_SOURCE_MAP_SHA256
    ),
    expected_manifest_map_sha256: str = FROZEN_STAGE0_MANIFEST_MAP_SHA256,
    expected_provenance_compatibility: Mapping[
        str, Mapping[str, str]
    ] = STAGE0_PROVENANCE_COMPATIBILITY,
) -> Stage0Checkpoint:
    if torch.cuda.is_initialized():
        raise shared.MiO100EvaluationError(
            "CUDA initialized before Stage0 CPU checkpoint audit"
        )
    checkpoint_path = _canonical_file(path, field="Stage0 best EMA")
    if checkpoint_path.name != "best_ema.pth":
        raise shared.MiO100EvaluationError("Stage0 checkpoint must be best_ema.pth")
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_sha256:
        raise shared.MiO100EvaluationError("Stage0 checkpoint SHA256 drifted")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise shared.MiO100EvaluationError(
            f"could not CPU-load Stage0 best EMA: {exc}"
        ) from exc
    payload = _mapping(payload, field="Stage0 checkpoint")
    header = {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage0",
        "step": 60_000,
        "model_role": "ema_selection",
        "resumable": False,
        "pending_validation_step": None,
    }
    if any(payload.get(key) != value for key, value in header.items()):
        raise shared.MiO100EvaluationError("Stage0 best EMA header drifted")
    model = _mapping(payload.get("model"), field="Stage0 model")
    ema = _mapping(payload.get("ema"), field="Stage0 EMA")
    shadow = _mapping(ema.get("shadow"), field="Stage0 EMA shadow")
    if set(model) != set(shadow):
        raise shared.MiO100EvaluationError("Stage0 model/EMA key sets differ")
    if expected_tensor_count is not None and len(model) != expected_tensor_count:
        raise shared.MiO100EvaluationError("Stage0 tensor count drifted")
    normalized: dict[str, Tensor] = {}
    for name in sorted(model):
        current, ema_value = model[name], shadow[name]
        if not torch.is_tensor(current) or not torch.is_tensor(ema_value):
            raise shared.MiO100EvaluationError(f"non-tensor Stage0 state at {name}")
        if current.device.type != "cpu" or ema_value.device.type != "cpu":
            raise shared.MiO100EvaluationError(f"non-CPU Stage0 state at {name}")
        if (
            current.shape != ema_value.shape
            or current.dtype != ema_value.dtype
            or current.layout != ema_value.layout
            or not torch.equal(current, ema_value)
        ):
            raise shared.MiO100EvaluationError(
                f"Stage0 model is not bit-exact to EMA shadow at {name}"
            )
        if current.is_floating_point() and not bool(torch.isfinite(current).all()):
            raise shared.MiO100EvaluationError(f"non-finite Stage0 tensor at {name}")
        normalized[name] = current.detach().cpu()
    if ema.get("num_updates") != 60_000:
        raise shared.MiO100EvaluationError("Stage0 EMA update count is not 60000")
    provenance = _mapping(payload.get("provenance"), field="Stage0 provenance")
    runtime = _mapping(provenance.get("runtime"), field="Stage0 runtime provenance")
    config_path = (
        None
        if expected_config_path is None
        else _canonical_file(expected_config_path, field="Stage0 config")
    )
    project_root = Path(expected_project_root).resolve(strict=True)
    if not project_root.is_dir():
        raise shared.MiO100EvaluationError("Stage0 project root is not a directory")
    provenance_verification = _validate_stage0_checkpoint_provenance(
        provenance,
        expected_config_sha256=expected_config_sha256,
        expected_config_path=config_path,
        expected_project_root=project_root,
        expected_semantic_source_count=expected_semantic_source_count,
        expected_provenance_sha256=expected_provenance_sha256,
        expected_semantic_source_map_sha256=expected_semantic_source_map_sha256,
        expected_manifest_map_sha256=expected_manifest_map_sha256,
        expected_provenance_compatibility=expected_provenance_compatibility,
    )
    if (
        provenance.get("protocol_id") != PROTOCOL_ID
        or provenance.get("stage") != "stage0"
        or provenance.get("config_sha256") != expected_config_sha256
        or runtime.get("target_step") != 60_000
        or runtime.get("schedule_max_steps") != 60_000
        or runtime.get("integration") is not False
    ):
        raise shared.MiO100EvaluationError("Stage0 checkpoint provenance drifted")
    metrics = _mapping(payload.get("metrics"), field="Stage0 checkpoint metrics")
    if torch.cuda.is_initialized():
        raise shared.MiO100EvaluationError("Stage0 CPU audit initialized CUDA")
    return Stage0Checkpoint(
        path=checkpoint_path,
        sha256=actual_sha,
        model_state=normalized,
        provenance=dict(provenance),
        provenance_verification=provenance_verification,
        metrics=dict(metrics),
    )


def build_stage0_readiness_payload(
    *,
    checkpoint_path: Path,
    config_path: Path,
    summary_path: Path,
    primary_validation_path: Path,
    calibration_history_path: Path,
    report_path: Path,
    stage1_run_contract_path: Path,
    expected_checkpoint_sha256: str = FROZEN_STAGE0_CHECKPOINT_SHA256,
    expected_config_sha256: str = FROZEN_STAGE0_CONFIG_SHA256,
    expected_tensor_count: int = 495,
    expected_stage1_missing_count: int = 1040,
    expected_project_root: str | Path = PROJECT_ROOT,
    expected_semantic_source_count: int = 46,
    expected_provenance_sha256: str = FROZEN_STAGE0_PROVENANCE_SHA256,
    expected_semantic_source_map_sha256: str = (
        FROZEN_STAGE0_SEMANTIC_SOURCE_MAP_SHA256
    ),
    expected_manifest_map_sha256: str = FROZEN_STAGE0_MANIFEST_MAP_SHA256,
    expected_provenance_compatibility: Mapping[
        str, Mapping[str, str]
    ] = STAGE0_PROVENANCE_COMPATIBILITY,
) -> Mapping[str, Any]:
    """Construct CPU-only readiness evidence; caller decides where to publish."""

    cuda_before = bool(torch.cuda.is_initialized())
    if cuda_before:
        raise shared.MiO100EvaluationError("CUDA initialized before readiness audit")
    config = _canonical_file(config_path, field="Stage0 config")
    if sha256_file(config) != expected_config_sha256:
        raise shared.MiO100EvaluationError("Stage0 config SHA256 drifted")
    snapshot = load_stage0_best_ema(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_config_path=config,
        expected_tensor_count=expected_tensor_count,
        expected_project_root=expected_project_root,
        expected_semantic_source_count=expected_semantic_source_count,
        expected_provenance_sha256=expected_provenance_sha256,
        expected_semantic_source_map_sha256=expected_semantic_source_map_sha256,
        expected_manifest_map_sha256=expected_manifest_map_sha256,
        expected_provenance_compatibility=expected_provenance_compatibility,
    )
    try:
        summary = validate_stage0_ready_without_torch(
            summary_path,
            checkpoint_path=checkpoint_path,
            primary_validation_path=primary_validation_path,
        )
    except FormalInventoryError as exc:
        raise shared.MiO100EvaluationError(
            f"Stage0 summary gate failed: {exc}"
        ) from exc
    summary_validation = _mapping(summary.get("validation"), field="summary validation")
    metric_pairs = {
        "single_psnr": "best_single_psnr",
        "single_ssim": "best_single_ssim",
        "group_a_psnr": "best_group_a_psnr",
        "group_a_ssim": "best_group_a_ssim",
    }
    for summary_key, checkpoint_key in metric_pairs.items():
        if float(summary_validation[summary_key]) != float(
            snapshot.metrics[checkpoint_key]
        ):
            raise shared.MiO100EvaluationError(
                f"Stage0 checkpoint/summary metric drifted at {summary_key}"
            )
    if float(snapshot.metrics.get("best_step", -1.0)) != 60_000.0:
        raise shared.MiO100EvaluationError("Stage0 selected best step drifted")

    calibration = _canonical_file(
        calibration_history_path, field="Stage0 calibration history"
    )
    try:
        with calibration.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise shared.MiO100EvaluationError(
            f"could not read Stage0 calibration history: {exc}"
        ) from exc
    final_rows = [row for row in rows if row.get("step") == "60000"]
    if len(final_rows) != 1:
        raise shared.MiO100EvaluationError(
            "Stage0 calibration history must contain one step-60000 row"
        )
    final_row = final_rows[0]
    for key in metric_pairs:
        if not math.isclose(
            float(final_row[key]),
            float(summary_validation[key]),
            rel_tol=0.0,
            abs_tol=5.0e-10,
        ):
            raise shared.MiO100EvaluationError(
                f"Stage0 calibration/summary drifted at {key}"
            )

    stage1_contract_file = _canonical_file(
        stage1_run_contract_path, field="Stage1 run contract"
    )
    stage1_contract = _mapping(
        load_json(stage1_contract_file), field="Stage1 run contract"
    )
    stage1_provenance = _mapping(
        stage1_contract.get("provenance"), field="Stage1 provenance"
    )
    parent = _mapping(stage1_provenance.get("parent_checkpoint"), field="Stage1 parent")
    load_receipt = _mapping(
        stage1_contract.get("stage0_backbone_load"), field="Stage1 load receipt"
    )
    expected_parent = {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "source": "stage0_best_ema_shadow",
    }
    if dict(parent) != {
        **expected_parent,
        "allowed_new_prefixes": ["decoder.skill_bank."],
    }:
        raise shared.MiO100EvaluationError("Stage1 parent checkpoint binding drifted")
    expected_load = {
        "source_tensor_count": expected_tensor_count,
        "loaded_count": expected_tensor_count,
        "missing_count": expected_stage1_missing_count,
        "missing_prefixes": ["decoder.skill_bank."],
        "unexpected_keys": [],
        "shape_mismatches": [],
    }
    if dict(load_receipt) != expected_load:
        raise shared.MiO100EvaluationError("Stage1 early backbone receipt drifted")
    files = {
        "config": config,
        "summary": _canonical_file(summary_path, field="Stage0 summary"),
        "primary_validation": _canonical_file(
            primary_validation_path, field="Stage0 primary validation"
        ),
        "calibration_history": calibration,
        "report": _canonical_file(report_path, field="Stage0 report"),
        "stage1_run_contract": stage1_contract_file,
    }
    cuda_after = bool(torch.cuda.is_initialized())
    if cuda_after:
        raise shared.MiO100EvaluationError("readiness audit initialized CUDA")
    return {
        "schema_version": STAGE0_READINESS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "READY",
        "created_utc": utc_now_iso(),
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "checkpoint": {
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
            "stage": "stage0",
            "step": 60_000,
            "model_role": "ema_selection",
            "resumable": False,
            "pending_validation_step": None,
            "tensor_count": len(snapshot.model_state),
            "ema_num_updates": 60_000,
            "model_equals_ema_shadow": True,
            "all_tensors_finite": True,
            "all_tensors_cpu": True,
            "provenance_config_sha256": snapshot.provenance["config_sha256"],
            "provenance_sha256": sha256_json(snapshot.provenance),
            "semantic_source_count": len(snapshot.provenance["semantic_source_sha256"]),
            "manifest_binding_count": len(snapshot.provenance["manifests"]),
            "parent_checkpoint_sha256": snapshot.provenance["parent_checkpoint"][
                "sha256"
            ],
            "warm_start_loaded_count": snapshot.provenance["warm_start_load"][
                "loaded_count"
            ],
        },
        **{
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "stage1_parent_receipt": {**expected_parent, **expected_load},
        "provenance_verification": dict(snapshot.provenance_verification),
    }


def publish_stage0_readiness(
    destination: Path,
    **kwargs: Any,
) -> Mapping[str, Any]:
    payload = build_stage0_readiness_payload(**kwargs)
    write_new_read_only_json(destination, payload)
    try:
        return validate_stage0_readiness_receipt_without_torch(
            destination,
            checkpoint_path=kwargs["checkpoint_path"],
            config_path=kwargs["config_path"],
            summary_path=kwargs["summary_path"],
            primary_validation_path=kwargs["primary_validation_path"],
            calibration_history_path=kwargs["calibration_history_path"],
            report_path=kwargs["report_path"],
            stage1_run_contract_path=kwargs["stage1_run_contract_path"],
        )
    except FormalInventoryError as exc:
        raise shared.MiO100EvaluationError(
            f"published Stage0 readiness receipt rejected: {exc}"
        ) from exc


def build_formal_stage0(checkpoint: Stage0Checkpoint) -> MiOStageA:
    model = MiOStageA(gradient_checkpointing=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != STAGE0_PARAMETER_COUNT:
        raise shared.MiO100EvaluationError(
            "Stage0 model parameter count drifted: "
            f"{parameter_count} != {STAGE0_PARAMETER_COUNT}"
        )
    target = model.state_dict()
    if set(target) != set(checkpoint.model_state):
        raise shared.MiO100EvaluationError("Stage0 architecture/checkpoint keys differ")
    mismatches = [
        name
        for name in target
        if target[name].shape != checkpoint.model_state[name].shape
        or target[name].dtype != checkpoint.model_state[name].dtype
    ]
    if mismatches:
        raise shared.MiO100EvaluationError(
            f"Stage0 architecture/checkpoint shapes drifted: {mismatches[:8]}"
        )
    model.load_state_dict(checkpoint.model_state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def stage0_formal_inference(
    model: MiOStageA,
    image: Tensor,
    *,
    device: torch.device,
    use_bf16: bool = True,
) -> shared.InferenceResult:
    if device.type != "cuda":
        raise shared.MiO100EvaluationError("formal Stage0 inference requires CUDA")
    if tuple(image.shape[:2]) != (1, 3) or not image.is_floating_point():
        raise shared.MiO100EvaluationError("Stage0 input must be RGB float [1,3,H,W]")
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with context:
        prediction = model(image.to(device=device, dtype=torch.float32))
    torch.cuda.synchronize(device)
    if not torch.is_tensor(prediction):
        raise shared.MiO100EvaluationError("Stage0 inference returned no tensor")
    return shared.InferenceResult(
        prediction=prediction.detach().float().cpu(),
        diagnostics={
            "program_levels": 0,
            "parallel_levels": 0,
            "active_skill_calls": 0,
            "reentry_requests": 0,
            "unexpected_activations": 0,
            "precycle_graphs": 0,
            "dropped_edges": 0,
        },
        latency_ms=(time.perf_counter() - started) * 1_000.0,
    )


def _data_disk_root(path: Path) -> Path:
    root = Path("/root/autodl-tmp").resolve(strict=True)
    candidate = path.resolve(strict=False)
    if candidate == root or root not in candidate.parents:
        raise shared.MiO100EvaluationError("Stage0 formal output must use data disk")
    for ancestor in (candidate, *candidate.parents):
        if ancestor == root.parent:
            break
        if ancestor.exists() and ancestor.is_symlink():
            raise shared.MiO100EvaluationError(
                f"Stage0 formal output crosses symlink: {ancestor}"
            )
    return candidate


def prepare_stage0_run_contract(
    authorization: shared.FormalAuthorization,
    *,
    manifest_sha256: str,
    data_inventory_sha256: str,
    data_inventory_rows_digest: str,
    data_inventory_files_digest: str,
    checkpoint_sha256: str,
    config_sha256: str,
    shard_count: int,
    enforce_data_disk: bool = True,
) -> shared.EvaluationRun:
    if sha256_file(authorization.path) != authorization.sha256:
        raise shared.MiO100EvaluationError("Stage0 authorization changed before setup")
    if shard_count != authorization.shard_count or shard_count != 1:
        raise shared.MiO100EvaluationError("Stage0 formal run is exactly shard 0/1")
    inventory_binding = authorization.bindings["formal_data_inventory"]
    if (
        data_inventory_sha256 != inventory_binding.sha256
        or not is_sha256(data_inventory_rows_digest)
        or not is_sha256(data_inventory_files_digest)
    ):
        raise shared.MiO100EvaluationError("Stage0 inventory run binding drifted")
    root = (
        _data_disk_root(authorization.output_root)
        if enforce_data_disk
        else authorization.output_root.resolve(strict=False)
    )
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise shared.MiO100EvaluationError("Stage0 output root is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "run_contract.json"
    core = {
        "schema_version": STAGE0_RUN_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "authorization": {
            "path": str(authorization.path),
            "sha256": authorization.sha256,
        },
        "authorization_bindings": {
            name: {"path": str(binding.path), "sha256": binding.sha256}
            for name, binding in sorted(authorization.bindings.items())
        },
        "manifest_sha256": manifest_sha256,
        "formal_data_inventory": {
            "path": str(inventory_binding.path),
            "sha256": data_inventory_sha256,
            "rows_digest": data_inventory_rows_digest,
            "files_digest": data_inventory_files_digest,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "method_name": authorization.method_name,
        "output_root": str(root),
        "manifest_row_count": 1_440,
        "groups": dict(FORMAL_GROUP_COUNTS),
        "combination_counts": dict(shared.FORMAL_COMBINATION_COUNTS),
        "shard_count": 1,
        "assignment": "manifest_index_mod_shard_count",
        "inference": dict(STAGE0_FORMAL_INFERENCE),
        "output_protocol": {
            "crop": "top_left_to_gt_shape",
            "quantization": "clamp_round_uint8",
            "encoding": "lossless_png",
            "score_source": "png_readback",
            "layout": "methods/<method>/d2|d3/<combination>/<gt_basename>",
            "overwrite": False,
        },
        "vram_maximum_peak_reserved_fraction": MAXIMUM_VRAM_RESERVED_FRACTION,
    }
    if contract_path.exists() or contract_path.is_symlink():
        contract = _mapping(load_json(contract_path), field="Stage0 run contract")
        if set(contract) != {*core, "created_utc"}:
            raise shared.MiO100EvaluationError("Stage0 run-contract schema drifted")
        if any(contract.get(key) != value for key, value in core.items()):
            raise shared.MiO100EvaluationError("Stage0 run-contract content drifted")
        _require_read_only(contract_path, field="Stage0 run contract")
    else:
        unexpected = list(root.iterdir())
        if unexpected:
            raise shared.MiO100EvaluationError(
                "Stage0 output root is non-empty without run contract"
            )
        contract = {**core, "created_utc": utc_now_iso()}
        _write_new_bytes(contract_path, _json_bytes(contract))
    return shared.EvaluationRun(
        root=root,
        method_name=authorization.method_name,
        contract_path=contract_path,
        contract_sha256=sha256_file(contract_path),
        contract=contract,
    )


def _complete_binding(
    raw: object, *, field: str, expected_path: Path
) -> shared.ArtifactBinding:
    value = _mapping(raw, field=field)
    if set(value) != {"path", "sha256"}:
        raise shared.MiO100EvaluationError(f"{field} binding fields drifted")
    path = _canonical_file(value.get("path"), field=field)
    _require_read_only(path, field=field)
    if path != expected_path or value.get("sha256") != sha256_file(path):
        raise shared.MiO100EvaluationError(f"{field} binding drifted")
    return shared.ArtifactBinding(path, str(value["sha256"]))


def validate_stage0_evaluator_complete(
    complete_path: str | Path = STAGE0_OUTPUT_ROOT / "complete.json",
    *,
    authorization_path: str | Path = STAGE0_APPROVAL_PATH,
    expected_bindings: Mapping[str, str | Path],
    verify_data_files: bool = True,
    expected_output_root: str | Path = STAGE0_OUTPUT_ROOT,
    expected_row_count: int = 1_440,
    expected_group_counts: Mapping[str, int] = FORMAL_GROUP_COUNTS,
    expected_combination_counts: Mapping[str, int] = shared.FORMAL_COMBINATION_COUNTS,
    inventory_validation_kwargs: Mapping[str, Any] | None = None,
    readiness_validation_kwargs: Mapping[str, Any] | None = None,
    validate_protocol: bool = True,
) -> shared.FormalEvaluatorCompletion:
    authorization = validate_stage0_formal_authorization(
        authorization_path,
        expected_bindings=expected_bindings,
        expected_output_root=expected_output_root,
    )
    if validate_protocol:
        validate_stage0_protocol_bindings(authorization)
    try:
        validate_stage0_readiness_receipt_without_torch(
            authorization.bindings["stage0_formal_readiness"].path,
            checkpoint_path=authorization.bindings["stage0_checkpoint"].path,
            config_path=authorization.bindings["stage0_config"].path,
            summary_path=authorization.bindings["stage0_summary"].path,
            primary_validation_path=authorization.bindings[
                "stage0_primary_validation"
            ].path,
            calibration_history_path=authorization.bindings[
                "stage0_calibration_history"
            ].path,
            report_path=authorization.bindings["stage0_report"].path,
            stage1_run_contract_path=authorization.bindings["stage1_run_contract"].path,
            **dict(readiness_validation_kwargs or {}),
        )
    except FormalInventoryError as exc:
        raise shared.MiO100EvaluationError(
            f"Stage0 readiness binding rejected: {exc}"
        ) from exc
    root = Path(expected_output_root).resolve(strict=False)
    complete_file = _canonical_file(complete_path, field="Stage0 evaluator complete")
    if complete_file != root / "complete.json":
        raise shared.MiO100EvaluationError("Stage0 evaluator complete path drifted")
    _require_read_only(complete_file, field="Stage0 evaluator complete")
    complete = _mapping(load_json(complete_file), field="Stage0 evaluator complete")
    required_complete = {
        "schema_version",
        "protocol_id",
        "created_utc",
        "status",
        "image_count",
        "method_name",
        "authorization_sha256",
        "run_contract_sha256",
        "checkpoint_sha256",
        "manifest_sha256",
        "formal_data_inventory",
        "predictions_digest",
        "bindings",
    }
    if set(complete) != required_complete:
        raise shared.MiO100EvaluationError("Stage0 evaluator complete fields drifted")
    shared._validate_utc(  # noqa: SLF001
        complete.get("created_utc"), field="Stage0 evaluator complete UTC"
    )
    expected_complete = {
        "schema_version": shared.COMPLETE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "image_count": expected_row_count,
        "method_name": STAGE0_METHOD_NAME,
        "authorization_sha256": authorization.sha256,
        "checkpoint_sha256": authorization.bindings["stage0_checkpoint"].sha256,
        "manifest_sha256": authorization.bindings["formal_manifest"].sha256,
    }
    if any(complete.get(key) != value for key, value in expected_complete.items()):
        raise shared.MiO100EvaluationError("Stage0 evaluator complete scope drifted")
    if not is_sha256(complete.get("predictions_digest")):
        raise shared.MiO100EvaluationError("Stage0 predictions digest is malformed")
    inventory_raw = _mapping(
        complete.get("formal_data_inventory"), field="Stage0 inventory completion"
    )
    inventory_binding = authorization.bindings["formal_data_inventory"]
    if (
        set(inventory_raw) != {"path", "sha256", "rows_digest", "files_digest"}
        or inventory_raw.get("path") != str(inventory_binding.path)
        or inventory_raw.get("sha256") != inventory_binding.sha256
        or not is_sha256(inventory_raw.get("rows_digest"))
        or not is_sha256(inventory_raw.get("files_digest"))
    ):
        raise shared.MiO100EvaluationError("Stage0 inventory completion drifted")
    try:
        inventory = load_formal_data_inventory(
            inventory_binding.path,
            expected_manifest_path=authorization.bindings["formal_manifest"].path,
            expected_manifest_sha256=authorization.bindings["formal_manifest"].sha256,
            expected_authorization_protocol_path=authorization.bindings[
                "inventory_origin_protocol"
            ].path,
            expected_authorization_protocol_sha256=authorization.bindings[
                "inventory_origin_protocol"
            ].sha256,
            verify_file_bytes=verify_data_files,
            expected_row_count=expected_row_count,
            expected_group_counts=expected_group_counts,
            expected_combination_counts=expected_combination_counts,
            **dict(inventory_validation_kwargs or {}),
        )
    except FormalInventoryError as exc:
        raise shared.MiO100EvaluationError(f"Stage0 inventory rejected: {exc}") from exc
    if inventory.rows_digest != inventory_raw.get(
        "rows_digest"
    ) or inventory.files_digest != inventory_raw.get("files_digest"):
        raise shared.MiO100EvaluationError("Stage0 inventory digest drifted")
    raw_bindings = _mapping(complete.get("bindings"), field="completion bindings")
    if set(raw_bindings) != {
        "run_contract",
        "summary",
        "per_image_csv",
        "table1_input_jsonl",
    }:
        raise shared.MiO100EvaluationError("Stage0 completion binding keys drifted")
    run_binding = _complete_binding(
        raw_bindings["run_contract"],
        field="Stage0 run contract",
        expected_path=root / "run_contract.json",
    )
    summary_binding = _complete_binding(
        raw_bindings["summary"],
        field="Stage0 evaluator summary",
        expected_path=root / "summary.json",
    )
    csv_binding = _complete_binding(
        raw_bindings["per_image_csv"],
        field="Stage0 evaluator per-image CSV",
        expected_path=root / "per_image.csv",
    )
    table_binding = _complete_binding(
        raw_bindings["table1_input_jsonl"],
        field="Stage0 evaluator Table-1 input",
        expected_path=root / "table1_input.jsonl",
    )
    if run_binding.sha256 != complete.get("run_contract_sha256"):
        raise shared.MiO100EvaluationError("Stage0 run contract SHA drifted")
    contract = _mapping(load_json(run_binding.path), field="Stage0 run contract")
    expected_contract = {
        "schema_version": STAGE0_RUN_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "authorization": {
            "path": str(authorization.path),
            "sha256": authorization.sha256,
        },
        "authorization_bindings": {
            name: {"path": str(binding.path), "sha256": binding.sha256}
            for name, binding in sorted(authorization.bindings.items())
        },
        "manifest_sha256": authorization.bindings["formal_manifest"].sha256,
        "formal_data_inventory": dict(inventory_raw),
        "checkpoint_sha256": authorization.bindings["stage0_checkpoint"].sha256,
        "config_sha256": authorization.bindings["stage0_config"].sha256,
        "method_name": STAGE0_METHOD_NAME,
        "output_root": str(root),
        "manifest_row_count": expected_row_count,
        "groups": dict(expected_group_counts),
        "combination_counts": dict(expected_combination_counts),
        "shard_count": 1,
        "assignment": "manifest_index_mod_shard_count",
        "inference": dict(STAGE0_FORMAL_INFERENCE),
        "output_protocol": {
            "crop": "top_left_to_gt_shape",
            "quantization": "clamp_round_uint8",
            "encoding": "lossless_png",
            "score_source": "png_readback",
            "layout": "methods/<method>/d2|d3/<combination>/<gt_basename>",
            "overwrite": False,
        },
        "vram_maximum_peak_reserved_fraction": 0.90,
    }
    if set(contract) != {*expected_contract, "created_utc"} or any(
        contract.get(key) != value for key, value in expected_contract.items()
    ):
        raise shared.MiO100EvaluationError("Stage0 formal run contract drifted")
    if contract.get("created_utc") != complete.get("created_utc"):
        raise shared.MiO100EvaluationError("Stage0 run/complete UTC drifted")

    records = shared.load_formal_manifest(
        authorization.bindings["formal_manifest"].path,
        expected_sha256=authorization.bindings["formal_manifest"].sha256,
        expected_group_counts=expected_group_counts,
        expected_combination_counts=expected_combination_counts,
    )
    if len(records) != expected_row_count or len(inventory.rows) != expected_row_count:
        raise shared.MiO100EvaluationError("Stage0 formal row count drifted")
    for record, identity in zip(records, inventory.rows, strict=True):
        if (
            record.index != identity.index
            or record.sample_id != identity.sample_id
            or record.row_sha256 != identity.row_sha256
            or record.native_lq_path != identity.native_lq_path
            or record.target_path != identity.target_path
        ):
            raise shared.MiO100EvaluationError(
                "Stage0 manifest/inventory identity drifted"
            )
    try:
        with csv_binding.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != shared._CSV_COLUMNS:  # noqa: SLF001
                raise shared.MiO100EvaluationError("Stage0 per-image header drifted")
            csv_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise shared.MiO100EvaluationError(
            f"Stage0 per-image CSV failed: {exc}"
        ) from exc
    if len(csv_rows) != expected_row_count:
        raise shared.MiO100EvaluationError("Stage0 per-image count drifted")
    digest_rows: list[Mapping[str, str]] = []
    per_image_by_sample: dict[str, Mapping[str, str]] = {}
    metric_rows: list[Mapping[str, Any]] = []
    runtime_receipts: list[Mapping[str, Any]] = []
    sample_ids: set[str] = set()
    prediction_paths: set[Path] = set()
    actual_groups: dict[str, int] = {}
    actual_combinations: dict[str, int] = {}
    diagnostic_names = shared._CSV_COLUMNS[11:18]  # noqa: SLF001
    for row, record, identity in zip(csv_rows, records, inventory.rows, strict=True):
        if set(row) != set(shared._CSV_COLUMNS):  # noqa: SLF001
            raise shared.MiO100EvaluationError("Stage0 per-image fields drifted")
        prediction = _canonical_file(row["prediction_png"], field="Stage0 prediction")
        expected_prediction = (
            root
            / "methods"
            / STAGE0_METHOD_NAME
            / record.depth_dir
            / record.combination
            / record.output_filename
        )
        if (
            row["sample_id"] != record.sample_id
            or record.sample_id in sample_ids
            or row["group"] != record.group
            or row["combination"] != record.combination
            or row["clean_id"] != record.clean_id
            or row["target_png"] != str(record.target_path)
            or row["target_sha256"] != identity.target_sha256
            or prediction != expected_prediction
            or prediction in prediction_paths
            or not is_sha256(row["prediction_sha256"])
            or not is_sha256(row["target_sha256"])
            or sha256_file(prediction) != row["prediction_sha256"]
        ):
            raise shared.MiO100EvaluationError("Stage0 per-image identity drifted")
        if stat.S_IMODE(prediction.stat().st_mode) != 0o444:
            raise shared.MiO100EvaluationError(
                "Stage0 prediction must have exact mode 0444"
            )
        psnr = shared._finite_float(row["psnr"], field="Stage0 PSNR")  # noqa: SLF001
        ssim = shared._finite_float(row["ssim"], field="Stage0 SSIM")  # noqa: SLF001
        latency = shared._finite_float(  # noqa: SLF001
            row["latency_ms"], field="Stage0 latency", minimum=0.0
        )
        peak = shared._finite_float(  # noqa: SLF001
            row["peak_reserved_fraction"], field="Stage0 VRAM", minimum=0.0
        )
        if peak >= MAXIMUM_VRAM_RESERVED_FRACTION:
            raise shared.MiO100EvaluationError("Stage0 per-image VRAM ceiling drifted")
        diagnostics = {
            name: shared._strict_nonnegative_int(  # noqa: SLF001
                row[name], field=f"Stage0 {name}"
            )
            for name in diagnostic_names
        }
        if any(diagnostics.values()):
            raise shared.MiO100EvaluationError(
                "prompt-free Stage0 emitted graph diagnostics"
            )
        sample_ids.add(record.sample_id)
        prediction_paths.add(prediction)
        actual_groups[record.group] = actual_groups.get(record.group, 0) + 1
        actual_combinations[record.combination] = (
            actual_combinations.get(record.combination, 0) + 1
        )
        metric_rows.append(
            {"combination": record.combination, "psnr": psnr, "ssim": ssim}
        )
        runtime_receipts.append(
            {
                "latency_ms": latency,
                "peak_reserved_fraction": peak,
                "diagnostics": diagnostics,
            }
        )
        digest_rows.append(
            {
                "sample_id": record.sample_id,
                "prediction_sha256": row["prediction_sha256"],
                "target_sha256": row["target_sha256"],
            }
        )
        per_image_by_sample[record.sample_id] = row
    if actual_groups != dict(expected_group_counts) or actual_combinations != dict(
        expected_combination_counts
    ):
        raise shared.MiO100EvaluationError(
            "Stage0 per-image group/combination counts drifted"
        )
    predictions_digest = sha256_json(digest_rows)
    if predictions_digest != complete["predictions_digest"]:
        raise shared.MiO100EvaluationError("Stage0 prediction digest drifted")
    table_rows = shared._load_strict_jsonl(  # noqa: SLF001
        table_binding.path, field="Stage0 Table-1 input"
    )
    official_order = {
        combination: index
        for index, combination in enumerate(
            name for names in OFFICIAL_GROUPS.values() for name in names
        )
    }
    ordered_records = sorted(
        records,
        key=lambda record: (official_order[record.combination], record.sample_id),
    )
    table_keys = {
        "schema_version",
        "sample_id",
        "group",
        "combination",
        "prediction_png",
        "prediction_sha256",
        "target_png",
        "target_sha256",
    }
    if len(table_rows) != expected_row_count:
        raise shared.MiO100EvaluationError("Stage0 Table-1 input count drifted")
    for row, record in zip(table_rows, ordered_records, strict=True):
        per_image = per_image_by_sample.get(record.sample_id)
        expected = {
            "schema_version": shared.TABLE1_INPUT_SCHEMA,
            "sample_id": record.sample_id,
            "group": record.group,
            "combination": record.combination,
            "prediction_png": per_image["prediction_png"] if per_image else None,
            "prediction_sha256": (
                per_image["prediction_sha256"] if per_image else None
            ),
            "target_png": per_image["target_png"] if per_image else None,
            "target_sha256": per_image["target_sha256"] if per_image else None,
        }
        if set(row) != table_keys or dict(row) != expected:
            raise shared.MiO100EvaluationError("Stage0 Table-1 input drifted")

    summary = _mapping(
        shared._load_strict_json(  # noqa: SLF001
            summary_binding.path, field="Stage0 evaluator summary"
        ),
        field="Stage0 evaluator summary",
    )
    summary_keys = {
        "schema_version",
        "protocol_id",
        "created_utc",
        "method_name",
        "image_count",
        "manifest_sha256",
        "formal_data_inventory",
        "checkpoint_sha256",
        "authorization_sha256",
        "run_contract_sha256",
        "predictions_digest",
        "aggregation",
        "runtime",
        "metric_protocol",
        "outputs",
    }
    if set(summary) != summary_keys:
        raise shared.MiO100EvaluationError("Stage0 evaluator summary fields drifted")
    expected_summary = {
        "schema_version": shared.SUMMARY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": contract["created_utc"],
        "method_name": STAGE0_METHOD_NAME,
        "image_count": expected_row_count,
        "manifest_sha256": authorization.bindings["formal_manifest"].sha256,
        "formal_data_inventory": dict(inventory_raw),
        "checkpoint_sha256": authorization.bindings["stage0_checkpoint"].sha256,
        "authorization_sha256": authorization.sha256,
        "run_contract_sha256": run_binding.sha256,
        "predictions_digest": predictions_digest,
        "metric_protocol": {
            "prediction_source": "lossless_png_readback",
            "psnr": "AgenticIR/pyiqa-0.1.10 RGB parity",
            "ssim": "AgenticIR/pyiqa-0.1.10 Y parity",
            "group_reduction": "equal_combination_mean",
            "weighted_all_images": "additional_only",
        },
        "outputs": {
            "agenticir_methods_root": str(root / "methods" / STAGE0_METHOD_NAME),
            "per_image_csv": str(csv_binding.path),
            "table1_input_jsonl": str(table_binding.path),
        },
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise shared.MiO100EvaluationError("Stage0 evaluator summary drifted")
    aggregate = aggregate_official_records(
        metric_rows,
        required_combinations=tuple(
            name for names in OFFICIAL_GROUPS.values() for name in names
        ),
        expected_counts=expected_combination_counts,
    )
    if summary.get("aggregation") != aggregate:
        raise shared.MiO100EvaluationError("Stage0 aggregate metrics drifted")
    runtime = shared._aggregate_runtime(runtime_receipts)  # noqa: SLF001
    if summary.get("runtime") != runtime:
        raise shared.MiO100EvaluationError("Stage0 runtime summary drifted")

    def evidence(binding: shared.ArtifactBinding) -> Mapping[str, str]:
        return {"path": str(binding.path), "sha256": binding.sha256}

    approval_binding = shared.ArtifactBinding(authorization.path, authorization.sha256)
    complete_binding = shared.ArtifactBinding(complete_file, sha256_file(complete_file))
    stable = {
        "authorization": evidence(approval_binding),
        "evaluator_complete": evidence(complete_binding),
        "run_contract": evidence(run_binding),
        "summary": evidence(summary_binding),
        "per_image": evidence(csv_binding),
        "table1_input": evidence(table_binding),
        "checkpoint": evidence(authorization.bindings["stage0_checkpoint"]),
        "manifest": evidence(authorization.bindings["formal_manifest"]),
        "formal_data_inventory": evidence(inventory_binding),
        "predictions_digest": predictions_digest,
    }
    return shared.FormalEvaluatorCompletion(
        complete_path=complete_file,
        complete_sha256=complete_binding.sha256,
        authorization=approval_binding,
        run_contract=run_binding,
        summary=summary_binding,
        per_image=csv_binding,
        table1_input=table_binding,
        checkpoint=authorization.bindings["stage0_checkpoint"],
        manifest=authorization.bindings["formal_manifest"],
        formal_data_inventory=inventory_binding,
        predictions_digest=predictions_digest,
        evidence=stable,
    )


def load_stage0_table1_evidence(
    *,
    scorer_module_path: Path,
    scorer_cli_path: Path,
) -> Mapping[str, Any]:
    expected = dict(default_stage0_authorization_paths())
    expected["table1_scorer_module"] = scorer_module_path.resolve(strict=True)
    expected["stage0_table1_scorer_cli"] = scorer_cli_path.resolve(strict=True)
    authorization = validate_stage0_formal_authorization(
        STAGE0_APPROVAL_PATH,
        expected_bindings=expected,
    )
    completion = validate_stage0_evaluator_complete(
        authorization_path=STAGE0_APPROVAL_PATH,
        expected_bindings=expected,
        verify_data_files=False,
    )
    evidence = dict(completion.evidence)
    parity = authorization.bindings["metric_parity_summary"]
    evidence["metric_parity_summary"] = {
        "path": str(parity.path),
        "sha256": parity.sha256,
    }
    return evidence


def configure_stage0_table1_module(module: Any, *, cli_path: Path) -> Any:
    """Retarget a private standalone legacy scorer module to Stage0 paths."""

    root = Path(__file__).resolve().parents[2]
    module.DEFAULT_CLI_PATH = cli_path.resolve(strict=True)
    module.FORMAL_AUTHORIZATION_PATH = STAGE0_APPROVAL_PATH
    module.FORMAL_EVALUATOR_ROOT = STAGE0_OUTPUT_ROOT
    module.FORMAL_EVALUATOR_COMPLETE_PATH = STAGE0_OUTPUT_ROOT / "complete.json"
    module.FORMAL_TABLE1_INPUT_PATH = STAGE0_OUTPUT_ROOT / "table1_input.jsonl"
    module.FORMAL_SCORE_ROOT = STAGE0_SCORE_ROOT
    module.FORMAL_WORK_ROOT = root / "artifacts/work/agenticir_table1_stage0"
    module._load_formal_evidence = lambda: load_stage0_table1_evidence(  # noqa: SLF001
        scorer_module_path=Path(module.__file__).resolve(strict=True),
        scorer_cli_path=cli_path,
    )
    return module


def _load_scorer_csv(path: Path) -> list[dict[str, str]]:
    file = _canonical_file(path, field="Table-1 per-image CSV")
    _require_read_only(file, field="Table-1 per-image CSV")
    try:
        with file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SCORER_CSV_COLUMNS:
                raise shared.MiO100EvaluationError("Table-1 CSV header drifted")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise shared.MiO100EvaluationError(f"Table-1 CSV read failed: {exc}") from exc
    return rows


def _validate_table1_complete_for_comparison(
    complete_path: Path,
    *,
    expected_per_image: Path,
    expected_summary: Path,
    expected_authorization: shared.ArtifactBinding,
    expected_count: int,
) -> Mapping[str, Any]:
    complete_file = _canonical_file(complete_path, field="Table-1 complete")
    _require_read_only(complete_file, field="Table-1 complete")
    complete = _mapping(
        shared._load_strict_json(  # noqa: SLF001
            complete_file, field="Table-1 complete"
        ),
        field="Table-1 complete",
    )
    peak = complete.get("maximum_peak_reserved_fraction")
    if (
        complete.get("schema_version") != "graphrestore.agenticir_table1_complete.v1"
        or complete.get("status") != "COMPLETE"
        or complete.get("image_count") != expected_count
        or complete.get("no_selective_rerun") is not True
        or complete.get("all_values_finite") is not True
        or isinstance(peak, bool)
        or not isinstance(peak, (int, float))
        or not math.isfinite(float(peak))
        or not 0.0 <= float(peak) < MAXIMUM_VRAM_RESERVED_FRACTION
    ):
        raise shared.MiO100EvaluationError("Table-1 completion scope drifted")
    per_image = _mapping(complete.get("per_image"), field="Table-1 per-image")
    summary = _mapping(complete.get("summary"), field="Table-1 summary")
    for raw, wanted, field in (
        (per_image, expected_per_image, "Table-1 per-image"),
        (summary, expected_summary, "Table-1 summary"),
    ):
        if set(raw) != {"path", "sha256"}:
            raise shared.MiO100EvaluationError(f"{field} binding fields drifted")
        actual = _canonical_file(raw.get("path"), field=field)
        _require_read_only(actual, field=field)
        if actual != wanted or raw.get("sha256") != sha256_file(actual):
            raise shared.MiO100EvaluationError(f"{field} binding drifted")
    evidence = _mapping(
        complete.get("formal_evidence"), field="Table-1 formal evidence"
    )
    bound_authorization = _mapping(
        evidence.get("authorization"), field="Table-1 formal authorization"
    )
    if bound_authorization != {
        "path": str(expected_authorization.path),
        "sha256": expected_authorization.sha256,
    }:
        raise shared.MiO100EvaluationError(
            "Table-1 formal authorization binding drifted"
        )
    return complete


def _recompute_paired_comparison(
    stage0_rows: Sequence[Mapping[str, str]],
    stage4_rows: Sequence[Mapping[str, str]],
    *,
    expected_count: int,
    expected_combination_counts: Mapping[str, int],
) -> tuple[
    list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]
]:
    if len(stage0_rows) != expected_count or len(stage4_rows) != expected_count:
        raise shared.MiO100EvaluationError("paired comparison row count drifted")
    stage4_by_id = {row["sample_id"]: row for row in stage4_rows}
    if len(stage4_by_id) != expected_count:
        raise shared.MiO100EvaluationError("duplicate Stage4 paired sample")
    official_group = {
        combination: group
        for group, combinations in OFFICIAL_GROUPS.items()
        for combination in combinations
    }
    paired: list[dict[str, Any]] = []
    for stage0 in stage0_rows:
        sample_id = stage0["sample_id"]
        stage4 = stage4_by_id.get(sample_id)
        if stage4 is None:
            raise shared.MiO100EvaluationError("Stage0/4 paired sample sets differ")
        for field in ("group", "combination", "target_png", "target_sha256"):
            if stage0[field] != stage4[field]:
                raise shared.MiO100EvaluationError(
                    f"Stage0/4 paired identity drifted: {sample_id}/{field}"
                )
        if (
            official_group.get(stage0["combination"]) != stage0["group"]
            or not is_sha256(stage0["target_sha256"])
            or not is_sha256(stage0["prediction_sha256"])
            or not is_sha256(stage4["prediction_sha256"])
        ):
            raise shared.MiO100EvaluationError(
                f"Stage0/4 paired scope drifted: {sample_id}"
            )
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "group": stage0["group"],
            "combination": stage0["combination"],
            "target_png": stage0["target_png"],
            "target_sha256": stage0["target_sha256"],
        }
        for metric in METRICS:
            try:
                before = float(stage0[metric])
                after = float(stage4[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise shared.MiO100EvaluationError(
                    f"invalid paired metric {sample_id}/{metric}"
                ) from exc
            if not math.isfinite(before) or not math.isfinite(after):
                raise shared.MiO100EvaluationError("non-finite paired metric")
            delta = after - before
            oriented = -delta if metric == "lpips" else delta
            row[f"stage0_{metric}"] = before
            row[f"stage4_{metric}"] = after
            row[f"stage4_minus_stage0_{metric}"] = delta
            row[f"oriented_stage4_gain_{metric}"] = oriented
        paired.append(row)
    if len({row["sample_id"] for row in paired}) != expected_count:
        raise shared.MiO100EvaluationError("duplicate Stage0 paired sample")
    actual_counts: dict[str, int] = {}
    for row in paired:
        combination = str(row["combination"])
        actual_counts[combination] = actual_counts.get(combination, 0) + 1
    if actual_counts != dict(expected_combination_counts):
        raise shared.MiO100EvaluationError(
            "paired comparison combination counts drifted"
        )

    combination_summary: dict[str, Any] = {}
    group_summary: dict[str, Any] = {}
    for combination in expected_combination_counts:
        selected = [row for row in paired if row["combination"] == combination]
        combination_summary[combination] = {
            "count": len(selected),
            "stage4_minus_stage0": {
                metric: math.fsum(
                    float(row[f"stage4_minus_stage0_{metric}"]) for row in selected
                )
                / len(selected)
                for metric in METRICS
            },
            "oriented_stage4_gain": {
                metric: math.fsum(
                    float(row[f"oriented_stage4_gain_{metric}"]) for row in selected
                )
                / len(selected)
                for metric in METRICS
            },
            "stage4_win_rate": {
                metric: math.fsum(
                    float(row[f"oriented_stage4_gain_{metric}"] > 0.0)
                    for row in selected
                )
                / len(selected)
                for metric in METRICS
            },
        }
    for group, combinations in OFFICIAL_GROUPS.items():
        group_summary[group] = {
            "combination_count": len(combinations),
            "stage4_minus_stage0": {
                metric: math.fsum(
                    combination_summary[name]["stage4_minus_stage0"][metric]
                    for name in combinations
                )
                / len(combinations)
                for metric in METRICS
            },
            "oriented_stage4_gain": {
                metric: math.fsum(
                    combination_summary[name]["oriented_stage4_gain"][metric]
                    for name in combinations
                )
                / len(combinations)
                for metric in METRICS
            },
        }
    directional = all(
        group_summary[group]["stage4_minus_stage0"][metric] > 0.0
        for group in ("A", "B", "C")
        for metric in ("psnr", "ssim")
    )
    ideal = directional and all(
        group_summary[group]["stage4_minus_stage0"]["psnr"] >= 0.20
        for group in ("A", "B", "C")
    )
    decision = (
        "PASS_INCREMENTAL_EFFICACY"
        if ideal
        else "INTERMEDIATE_DIRECTIONAL_ONLY"
        if directional
        else "FAIL_V7_1_FORMAL_TARGET"
    )
    success = {
        "all_group_psnr_ssim_positive": directional,
        "all_group_psnr_at_least_0_20_db": ideal,
        "decision": decision,
    }
    return paired, combination_summary, group_summary, success


def publish_stage0_vs_stage4_comparison(
    *,
    stage0_per_image: Path,
    stage4_per_image: Path,
    stage0_table1_complete: Path = STAGE0_SCORE_ROOT / "complete.json",
    stage4_table1_complete: Path | None = None,
    stage4_table1_summary: Path | None = None,
    output_root: Path = STAGE0_COMPARISON_ROOT,
    authorization: shared.FormalAuthorization,
    expected_count: int = 1_440,
    expected_combination_counts: Mapping[str, int] = shared.FORMAL_COMBINATION_COUNTS,
    enforce_fixed_root: bool = True,
) -> Mapping[str, Any]:
    """Publish immutable paired Stage4-minus-Stage0 six-metric deltas."""

    if enforce_fixed_root and output_root != STAGE0_COMPARISON_ROOT:
        raise shared.MiO100EvaluationError("Stage0 comparison root is fixed")
    if sha256_file(authorization.path) != authorization.sha256:
        raise shared.MiO100EvaluationError("Stage0 authorization changed")
    stage4_complete = (
        authorization.bindings["stage4_table1_complete"].path
        if stage4_table1_complete is None
        else stage4_table1_complete
    )
    stage4_summary = (
        authorization.bindings["stage4_table1_summary"].path
        if stage4_table1_summary is None
        else stage4_table1_summary
    )
    for name, path in (
        ("stage4_table1_complete", stage4_complete),
        ("stage4_table1_summary", stage4_summary),
    ):
        binding = authorization.bindings[name]
        if path != binding.path or sha256_file(path) != binding.sha256:
            raise shared.MiO100EvaluationError(f"{name} binding drifted")
    if (
        stage4_per_image != authorization.bindings["stage4_table1_per_image"].path
        or sha256_file(stage4_per_image)
        != authorization.bindings["stage4_table1_per_image"].sha256
    ):
        raise shared.MiO100EvaluationError("Stage4 paired reference drifted")
    stage0_summary = stage0_table1_complete.parent / "summary.json"
    _validate_table1_complete_for_comparison(
        stage0_table1_complete,
        expected_per_image=stage0_per_image,
        expected_summary=stage0_summary,
        expected_authorization=shared.ArtifactBinding(
            authorization.path, authorization.sha256
        ),
        expected_count=expected_count,
    )
    _validate_table1_complete_for_comparison(
        stage4_complete,
        expected_per_image=stage4_per_image,
        expected_summary=stage4_summary,
        expected_authorization=authorization.bindings["stage4_formal_authorization"],
        expected_count=expected_count,
    )
    stage0_rows = _load_scorer_csv(stage0_per_image)
    stage4_rows = _load_scorer_csv(stage4_per_image)
    paired, combination_summary, group_summary, success_rule = (
        _recompute_paired_comparison(
            stage0_rows,
            stage4_rows,
            expected_count=expected_count,
            expected_combination_counts=expected_combination_counts,
        )
    )
    decision = str(success_rule["decision"])
    if enforce_fixed_root:
        _data_disk_root(output_root)
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir()):
        raise shared.MiO100EvaluationError("Stage0 comparison root is invalid")
    if output_root.exists():
        allowed = {"paired_per_image.csv", "summary.json", "complete.json"}
        unexpected = {entry.name for entry in output_root.iterdir()} - allowed
        if unexpected or any(
            entry.is_symlink() or not entry.is_file() for entry in output_root.iterdir()
        ):
            raise shared.MiO100EvaluationError(
                f"unexpected Stage0 comparison tree: {sorted(unexpected)}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "paired_per_image.csv"
    summary_path = output_root / "summary.json"
    complete_path = output_root / "complete.json"
    fields = list(paired[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(paired)
    _write_or_verify(csv_path, stream.getvalue().encode("utf-8"))
    summary = {
        "schema_version": STAGE0_COMPARISON_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "image_count": expected_count,
        "delta_orientation": "stage4_minus_stage0",
        "lpips_oriented_gain": "stage0_minus_stage4",
        "stage0_parameter_count": STAGE0_PARAMETER_COUNT,
        "stage0_graph_diagnostic_columns": dict(STAGE0_GRAPH_DIAGNOSTIC_COMPATIBILITY),
        "stage0_per_image": {
            "path": str(stage0_per_image),
            "sha256": sha256_file(stage0_per_image),
        },
        "stage4_per_image": {
            "path": str(stage4_per_image),
            "sha256": sha256_file(stage4_per_image),
        },
        "combinations": combination_summary,
        "groups": group_summary,
        "success_rule": success_rule,
    }
    _write_or_verify(summary_path, _json_bytes(summary))
    complete = {
        "schema_version": STAGE0_COMPARISON_COMPLETE_SCHEMA,
        "status": "COMPLETE",
        "image_count": expected_count,
        "authorization_sha256": authorization.sha256,
        "bindings": {
            "paired_per_image": {
                "path": str(csv_path),
                "sha256": sha256_file(csv_path),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "stage0_table1_complete": {
                "path": str(stage0_table1_complete),
                "sha256": sha256_file(stage0_table1_complete),
            },
            "stage4_table1_complete": {
                "path": str(authorization.bindings["stage4_table1_complete"].path),
                "sha256": authorization.bindings["stage4_table1_complete"].sha256,
            },
        },
        "decision": decision,
    }
    _write_or_verify(complete_path, _json_bytes(complete))
    fsync_directory(output_root)
    return validate_stage0_vs_stage4_comparison_complete(
        complete_path,
        stage0_table1_complete=stage0_table1_complete,
        stage4_table1_complete=stage4_complete,
        stage4_table1_summary=stage4_summary,
        authorization=authorization,
        output_root=output_root,
        expected_count=expected_count,
        expected_combination_counts=expected_combination_counts,
        enforce_fixed_root=enforce_fixed_root,
    )


def validate_stage0_vs_stage4_comparison_complete(
    complete_path: Path = STAGE0_COMPARISON_ROOT / "complete.json",
    *,
    stage0_table1_complete: Path = STAGE0_SCORE_ROOT / "complete.json",
    stage4_table1_complete: Path | None = None,
    stage4_table1_summary: Path | None = None,
    authorization: shared.FormalAuthorization,
    output_root: Path = STAGE0_COMPARISON_ROOT,
    expected_count: int = 1_440,
    expected_combination_counts: Mapping[str, int] = shared.FORMAL_COMBINATION_COUNTS,
    enforce_fixed_root: bool = True,
) -> Mapping[str, Any]:
    """Recompute and verify the immutable paired comparison terminal tree."""

    if enforce_fixed_root and output_root != STAGE0_COMPARISON_ROOT:
        raise shared.MiO100EvaluationError("Stage0 comparison root is fixed")
    root = output_root.resolve(strict=True)
    if root != output_root or root.is_symlink() or not root.is_dir():
        raise shared.MiO100EvaluationError("Stage0 comparison root drifted")
    expected_tree = {"paired_per_image.csv", "summary.json", "complete.json"}
    entries = {entry.name: entry for entry in root.iterdir()}
    if set(entries) != expected_tree or any(
        entry.is_symlink() or not entry.is_file() for entry in entries.values()
    ):
        raise shared.MiO100EvaluationError("Stage0 comparison tree drifted")
    for name, path in entries.items():
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise shared.MiO100EvaluationError(
                f"Stage0 comparison artifact must be 0444: {name}"
            )
    if sha256_file(authorization.path) != authorization.sha256:
        raise shared.MiO100EvaluationError("Stage0 authorization changed")
    stage4_complete = (
        authorization.bindings["stage4_table1_complete"].path
        if stage4_table1_complete is None
        else stage4_table1_complete
    )
    stage4_summary = (
        authorization.bindings["stage4_table1_summary"].path
        if stage4_table1_summary is None
        else stage4_table1_summary
    )
    stage0_per_image = stage0_table1_complete.parent / "per_image.csv"
    stage0_summary = stage0_table1_complete.parent / "summary.json"
    stage4_per_image = authorization.bindings["stage4_table1_per_image"].path
    _validate_table1_complete_for_comparison(
        stage0_table1_complete,
        expected_per_image=stage0_per_image,
        expected_summary=stage0_summary,
        expected_authorization=shared.ArtifactBinding(
            authorization.path, authorization.sha256
        ),
        expected_count=expected_count,
    )
    _validate_table1_complete_for_comparison(
        stage4_complete,
        expected_per_image=stage4_per_image,
        expected_summary=stage4_summary,
        expected_authorization=authorization.bindings["stage4_formal_authorization"],
        expected_count=expected_count,
    )
    for name, path in (
        ("stage4_table1_complete", stage4_complete),
        ("stage4_table1_per_image", stage4_per_image),
        ("stage4_table1_summary", stage4_summary),
    ):
        binding = authorization.bindings[name]
        if path != binding.path or sha256_file(path) != binding.sha256:
            raise shared.MiO100EvaluationError(f"{name} binding drifted")

    stage0_rows = _load_scorer_csv(stage0_per_image)
    stage4_rows = _load_scorer_csv(stage4_per_image)
    paired, combinations, groups, success_rule = _recompute_paired_comparison(
        stage0_rows,
        stage4_rows,
        expected_count=expected_count,
        expected_combination_counts=expected_combination_counts,
    )
    paired_path = root / "paired_per_image.csv"
    expected_columns = (
        "sample_id",
        "group",
        "combination",
        "target_png",
        "target_sha256",
        *(
            column
            for metric in METRICS
            for column in (
                f"stage0_{metric}",
                f"stage4_{metric}",
                f"stage4_minus_stage0_{metric}",
                f"oriented_stage4_gain_{metric}",
            )
        ),
    )
    try:
        with paired_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_columns:
                raise shared.MiO100EvaluationError(
                    "Stage0 paired comparison CSV header drifted"
                )
            published_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise shared.MiO100EvaluationError(
            f"Stage0 paired comparison CSV failed: {exc}"
        ) from exc
    if len(published_rows) != expected_count:
        raise shared.MiO100EvaluationError("Stage0 paired comparison count drifted")
    identity_columns = expected_columns[:5]
    numeric_columns = expected_columns[5:]
    for raw, expected in zip(published_rows, paired, strict=True):
        if any(raw[name] != str(expected[name]) for name in identity_columns):
            raise shared.MiO100EvaluationError(
                "Stage0 paired comparison identity drifted"
            )
        try:
            numeric_equal = all(
                math.isfinite(float(raw[name]))
                and float(raw[name]) == float(expected[name])
                for name in numeric_columns
            )
        except (TypeError, ValueError) as exc:
            raise shared.MiO100EvaluationError(
                "Stage0 paired comparison metric is invalid"
            ) from exc
        if not numeric_equal:
            raise shared.MiO100EvaluationError(
                "Stage0 paired comparison metric drifted"
            )

    expected_summary = {
        "schema_version": STAGE0_COMPARISON_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "image_count": expected_count,
        "delta_orientation": "stage4_minus_stage0",
        "lpips_oriented_gain": "stage0_minus_stage4",
        "stage0_parameter_count": STAGE0_PARAMETER_COUNT,
        "stage0_graph_diagnostic_columns": dict(STAGE0_GRAPH_DIAGNOSTIC_COMPATIBILITY),
        "stage0_per_image": {
            "path": str(stage0_per_image),
            "sha256": sha256_file(stage0_per_image),
        },
        "stage4_per_image": {
            "path": str(stage4_per_image),
            "sha256": sha256_file(stage4_per_image),
        },
        "combinations": combinations,
        "groups": groups,
        "success_rule": success_rule,
    }
    summary_path = root / "summary.json"
    summary = _mapping(
        shared._load_strict_json(summary_path, field="Stage0 comparison summary"),  # noqa: SLF001
        field="Stage0 comparison summary",
    )
    if dict(summary) != expected_summary:
        raise shared.MiO100EvaluationError("Stage0 comparison summary drifted")

    complete_file = _canonical_file(complete_path, field="Stage0 comparison complete")
    if complete_file != root / "complete.json":
        raise shared.MiO100EvaluationError("Stage0 comparison complete path drifted")
    complete = _mapping(
        shared._load_strict_json(  # noqa: SLF001
            complete_file, field="Stage0 comparison complete"
        ),
        field="Stage0 comparison complete",
    )
    expected_complete = {
        "schema_version": STAGE0_COMPARISON_COMPLETE_SCHEMA,
        "status": "COMPLETE",
        "image_count": expected_count,
        "authorization_sha256": authorization.sha256,
        "bindings": {
            "paired_per_image": {
                "path": str(paired_path),
                "sha256": sha256_file(paired_path),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "stage0_table1_complete": {
                "path": str(stage0_table1_complete),
                "sha256": sha256_file(stage0_table1_complete),
            },
            "stage4_table1_complete": {
                "path": str(stage4_complete),
                "sha256": sha256_file(stage4_complete),
            },
        },
        "decision": success_rule["decision"],
    }
    if dict(complete) != expected_complete:
        raise shared.MiO100EvaluationError("Stage0 comparison completion drifted")
    return complete


__all__ = [
    "STAGE0_FORMAL_INFERENCE",
    "STAGE0_GRAPH_DIAGNOSTIC_COMPATIBILITY",
    "STAGE0_PARAMETER_COUNT",
    "STAGE0_RUN_CONTRACT_SCHEMA",
    "Stage0Checkpoint",
    "bind_default_stage0_authorization_paths",
    "build_formal_stage0",
    "build_stage0_readiness_payload",
    "configure_stage0_table1_module",
    "default_stage0_authorization_paths",
    "load_stage0_best_ema",
    "load_stage0_table1_evidence",
    "prepare_stage0_run_contract",
    "publish_stage0_readiness",
    "publish_stage0_vs_stage4_comparison",
    "stage0_formal_inference",
    "validate_stage0_evaluator_complete",
    "validate_stage0_formal_authorization",
    "validate_stage0_protocol_bindings",
    "validate_stage0_vs_stage4_comparison_complete",
]
