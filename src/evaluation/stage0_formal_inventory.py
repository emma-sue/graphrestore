"""Standard-library-only authorization contract for formal Stage0 MiO100.

Stage0 deliberately reuses the immutable Stage4-era formal data inventory.
It never creates a second image inventory: the same manifest, row digest,
file digest, and per-file byte/stat identities are therefore shared by both
models.  The Stage0 approval, method name, output root, evaluator, scorer, and
comparison artifacts are nevertheless independent and cannot alias Stage4.

This module imports neither torch nor OpenCV.  Authorization may hash files,
but it may not decode images or initialize CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.formal_inventory import (
    AUTHORIZATION_RESTRICTIONS,
    FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    FORMAL_DATA_INVENTORY_PATH,
    FORMAL_MANIFEST_FILENAME,
    FORMAL_MANIFEST_SHA256,
    FORMAL_ROW_COUNT,
    FormalInventoryError,
    canonical_regular_file,
    require_mode_0444,
    sha256_file,
)


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
STAGE0_AUTHORIZATION_SCHEMA = "graphrestore-stage0-formal-mio100-approval-v1"
STAGE0_AUTHORIZATION_KIND = "stage0_formal_mio100_approval"
STAGE0_METHOD_NAME = "mio_stagea_v7_1_stage0_step060000"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE0_OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts/formal_mio100/mio_stagea_v7_1_stage0_step060000"
)
STAGE0_APPROVAL_PATH = (
    PROJECT_ROOT / "artifacts/approvals/FORMAL_MIO100_STAGE0_CONTROL_APPROVED.json"
)
STAGE0_AUTHORIZATION_PROTOCOL_PATH = (
    PROJECT_ROOT / "reports/FORMAL_MIO100_STAGE0_CONTROL_PROTOCOL.md"
)
STAGE0_USER_AUTHORIZATION_PROTOCOL_PATH = (
    PROJECT_ROOT / "reports/FORMAL_MIO100_STAGE0_CONTROL_USER_AUTHORIZATION.md"
)
STAGE0_SCORE_ROOT = STAGE0_OUTPUT_ROOT / "table1_scores"
STAGE0_COMPARISON_ROOT = STAGE0_OUTPUT_ROOT / "stage0_vs_stage4"

STAGE4_FORMAL_ROOT = (
    PROJECT_ROOT / "artifacts/formal_mio100/graphrestore_v7_1_stage4_step040000"
)

FROZEN_STAGE0_CHECKPOINT_SHA256 = (
    "52a8744582e39e4f1aa052cc84924ad486289c0b97fc30c89fc6489e69dfac8a"
)
FROZEN_STAGE0_CONFIG_SHA256 = (
    "1bfb0444e311d110c6929ce30fdcf888a73d0f10167ed27e3328492a08406283"
)
FROZEN_STAGE0_PROVENANCE_SHA256 = (
    "aa38a9175489423eea103c7520e1c871f9b621b7d6c79e18d97db1967ee59200"
)
FROZEN_STAGE0_SEMANTIC_SOURCE_MAP_SHA256 = (
    "80ba30db9b0558ad6bd476f50c6b8a76bb52af6f2fca75abbd68ed265c98e32a"
)
FROZEN_STAGE0_MANIFEST_MAP_SHA256 = (
    "b6af0a85c4893aeca61fa44bbca5eef019d4f819a499f8968d34bb91be64ad75"
)
FROZEN_FORMAL_DATA_INVENTORY_SHA256 = (
    "489d9c216589bb73f4b99ec8301abd57d77b7d418489cef02c162bc135aa91ae"
)
FROZEN_METRIC_WEIGHT_INVENTORY_SHA256 = (
    "796e39eddc51c28e57b9c40b393f99fd73bd14fde2bab138987c2ddcde746e7d"
)
FROZEN_STAGE4_TABLE1_COMPLETE_SHA256 = (
    "eb1468705d2591709565b3461aa630ae826f1efc311cddaf6bc06feca309a60b"
)
FROZEN_STAGE4_TABLE1_PER_IMAGE_SHA256 = (
    "f8bcbd463eb7113ccf632b5e03f0a34951650a8b0e66e1e63edabbfdd781a6d5"
)
FROZEN_STAGE4_TABLE1_SUMMARY_SHA256 = (
    "68301ea9f0ef52f062a8fdaa763865c3fb0374ee49f65ec9d99cdd9cf5ac7be8"
)
FROZEN_STAGE0_CONTROL_PROTOCOL_SHA256 = (
    "90dcf2307e37fb10a6325952fa1b25ecb16ade1d4e135ac0d1f05e092dba4955"
)
STAGE0_USER_AUTHORIZATION_SCOPE = (
    "frozen-stage0-step060000-inference-only-six-metric-paired-comparison"
)
STAGE0_PROVENANCE_COMPATIBILITY = {
    "src/training/orchestration.py": {
        "checkpoint_sha256": "597333101407451f83aaede9e6c23be9f222209de007b78a05ce4684f42b0584",
        "current_sha256": "1aed52f7780b574756e2620092686be55f66ef60619d133cf8265b79439ee66b",
        "rationale": "post_stage0_downstream_not_imported_by_formal_stage0",
    },
    "src/training/stage1_engine.py": {
        "checkpoint_sha256": "017b031a2424f7bdd2f7481fe223c455607a53e908cef4a37ef007d04c85e961",
        "current_sha256": "ab76a61422532d0deac8d7c01da69dae9f1d8154a277a4c0c26658b936c36f6c",
        "rationale": "post_stage0_downstream_not_imported_by_formal_stage0",
    },
    "src/training/stage2_distillation.py": {
        "checkpoint_sha256": "b9c8816b0ad67fbb8ff9c39851b190466307ba44970ec2cf5f69c9085c8be7bf",
        "current_sha256": "4cd37e4a1e5725e9b948c758eab893ff9a95f1c23357a2cdea3215df93d1b06c",
        "rationale": "post_stage0_downstream_not_imported_by_formal_stage0",
    },
    "src/training/stage3_engine.py": {
        "checkpoint_sha256": "eecfeecc087b735d085562b26047f99d90160a5cd2938075d1277cf09d9477f5",
        "current_sha256": "1d373d8d3e416e6431d52721c2cd2eef15541241a2d7c28af8a688181a972548",
        "rationale": "post_stage0_downstream_not_imported_by_formal_stage0",
    },
    "src/training/stage4_engine.py": {
        "checkpoint_sha256": "7079296d2ab27a09982303c6d609cbba1694966a515474d3b21eeb707b9b669f",
        "current_sha256": "2cd111a2ed5181fab864784916ead8d889fceaac02fee64f1aae40f50bde0d6c",
        "rationale": "post_stage0_downstream_not_imported_by_formal_stage0",
    },
}

STAGE0_AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_id",
        "approved",
        "stage0_formal_mio100_authorized",
        "one_shot",
        "inference_only",
        "authorized_groups",
        "manifest_row_count",
        "method_name",
        "reference_method_name",
        "shard_count",
        "output_root",
        "approved_utc",
        "restrictions",
        "bindings",
    }
)

REQUIRED_STAGE0_AUTHORIZATION_BINDINGS = (
    "stage0_summary",
    "stage0_checkpoint",
    "stage0_config",
    "stage0_primary_validation",
    "stage0_calibration_history",
    "stage0_report",
    "stage0_formal_readiness",
    "stage1_run_contract",
    "stage0_engine_source",
    "stage0_model_source",
    "formal_manifest",
    "manifest_inventory",
    "formal_data_inventory",
    "inventory_origin_protocol",
    "stage0_control_protocol",
    "stage0_user_authorization_protocol",
    "metric_parity_summary",
    "metric_protocol",
    "stage0_evaluator_module",
    "stage0_evaluator_cli",
    "stage0_authorizer_cli",
    "shared_evaluator_module",
    "shared_formal_inventory_module",
    "canonicalizer_source",
    "mioir_matlab_functions",
    "agenticir_scorer",
    "agenticir_compute_scores",
    "agenticir_compare_methods",
    "table1_scorer_module",
    "stage0_table1_scorer_cli",
    "metric_weight_inventory",
    "stage4_formal_authorization",
    "stage4_formal_evaluator_complete",
    "stage4_table1_complete",
    "stage4_table1_per_image",
    "stage4_table1_summary",
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalInventoryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalInventoryError(f"could not read {field}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FormalInventoryError(f"{field} must be a JSON object")
    return value


def _validate_utc(value: object) -> str:
    if not isinstance(value, str) or "T" not in value or not value.endswith("Z"):
        raise FormalInventoryError("approved_utc must be an RFC3339 UTC timestamp")
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_stage0_user_authorization_protocol(
    path: str | Path,
) -> Mapping[str, str]:
    """Validate future user evidence without inventing or publishing it.

    The Markdown file keeps the user's exact instruction verbatim while the
    fixed machine-readable preamble prevents that prose from broadening the
    preregistered one-shot scope.
    """

    protocol = canonical_regular_file(path, field="Stage0 user authorization")
    require_mode_0444(protocol, field="Stage0 user authorization")
    try:
        text = protocol.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise FormalInventoryError(
            f"could not read Stage0 user authorization: {exc}"
        ) from exc
    lines = text.splitlines()
    required_prefix = [
        "# Formal MiO100 Stage0 Control User Authorization",
        "",
        "- Status: USER-AUTHORIZED",
    ]
    if lines[:3] != required_prefix:
        raise FormalInventoryError("Stage0 user authorization header drifted")
    marker = "## Exact user instruction"
    if lines.count(marker) != 1:
        raise FormalInventoryError(
            "Stage0 user authorization must contain one exact-instruction section"
        )
    marker_index = lines.index(marker)
    values: dict[str, str] = {}
    for line in lines[2:marker_index]:
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            if key in values:
                raise FormalInventoryError(
                    f"duplicate Stage0 user authorization field: {key}"
                )
            values[key] = value
    expected = {
        "Status": "USER-AUTHORIZED",
        "Scope": STAGE0_USER_AUTHORIZATION_SCOPE,
        "Stage0 checkpoint SHA256": FROZEN_STAGE0_CHECKPOINT_SHA256,
        "Stage0 control protocol SHA256": FROZEN_STAGE0_CONTROL_PROTOCOL_SHA256,
        "Training authorized": "false",
        "Checkpoint or threshold selection authorized": "false",
        "Stage4 mutation authorized": "false",
        "TTA or model fusion authorized": "false",
        "Result-driven rerun authorized": "false",
        "Blind-test status restored": "false",
    }
    if set(values) != {*expected, "Authorized UTC"}:
        raise FormalInventoryError("Stage0 user authorization fields drifted")
    for key, wanted in expected.items():
        if values.get(key) != wanted:
            raise FormalInventoryError(
                f"Stage0 user authorization scope drifted at {key}"
            )
    _validate_utc(values.get("Authorized UTC"))
    instruction = "\n".join(lines[marker_index + 1 :]).strip()
    if not instruction:
        raise FormalInventoryError("exact Stage0 user instruction is empty")
    return {
        "path": str(protocol),
        "sha256": sha256_file(protocol),
        "authorized_utc": values["Authorized UTC"],
        "scope": values["Scope"],
        "exact_instruction": instruction,
    }


def stage0_authorization_binding_paths(
    project_root: str | Path,
    *,
    manifest: str | Path,
    formal_data_inventory: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    summary: str | Path,
    primary_validation: str | Path,
    calibration_history: str | Path,
    report: str | Path,
    readiness: str | Path,
    stage1_run_contract: str | Path | None = None,
    authorization_protocol: str | Path = STAGE0_AUTHORIZATION_PROTOCOL_PATH,
    user_authorization_protocol: str | Path = (STAGE0_USER_AUTHORIZATION_PROTOCOL_PATH),
) -> Mapping[str, Path]:
    root = Path(project_root).resolve(strict=True)
    return {
        "stage0_summary": Path(summary).resolve(strict=False),
        "stage0_checkpoint": Path(checkpoint).resolve(strict=False),
        "stage0_config": Path(config).resolve(strict=False),
        "stage0_primary_validation": Path(primary_validation).resolve(strict=False),
        "stage0_calibration_history": Path(calibration_history).resolve(strict=False),
        "stage0_report": Path(report).resolve(strict=False),
        "stage0_formal_readiness": Path(readiness).resolve(strict=False),
        "stage1_run_contract": (
            root / "artifacts/checkpoints/stage1/run_contract.json"
            if stage1_run_contract is None
            else Path(stage1_run_contract).resolve(strict=False)
        ),
        "stage0_engine_source": root / "src/training/stage0_engine.py",
        "stage0_model_source": root / "src/net/mio_stagea.py",
        "formal_manifest": Path(manifest).resolve(strict=False),
        "manifest_inventory": root
        / "manifests/agenticir_online_canonical_inventory.json",
        "formal_data_inventory": Path(formal_data_inventory).resolve(strict=False),
        # The shared inventory was generated under this already-immutable data
        # protocol.  Stage0 authorization is a separate, additional binding.
        "inventory_origin_protocol": FORMAL_AUTHORIZATION_PROTOCOL_PATH,
        "stage0_control_protocol": Path(authorization_protocol).resolve(strict=False),
        "stage0_user_authorization_protocol": Path(user_authorization_protocol).resolve(
            strict=False
        ),
        "metric_parity_summary": root / "artifacts/metrics/metric_parity_summary.json",
        "metric_protocol": root / "reports/METRIC_PROTOCOL.md",
        "stage0_evaluator_module": root / "src/evaluation/stage0_formal.py",
        "stage0_evaluator_cli": root / "scripts/eval_stage0_mio100.py",
        "stage0_authorizer_cli": root / "scripts/authorize_stage0_formal_mio100.py",
        "shared_evaluator_module": root / "src/evaluation/mio100.py",
        "shared_formal_inventory_module": root / "src/evaluation/formal_inventory.py",
        "canonicalizer_source": root / "src/data/scale_canonicalizer.py",
        "mioir_matlab_functions": Path(
            "/root/autodl-tmp/graph/upstream/MiOIR/basicsr/utils/matlab_functions.py"
        ),
        "agenticir_scorer": Path(
            "/root/autodl-tmp/graph/upstream/AgenticIR/utils/scorer.py"
        ),
        "agenticir_compute_scores": Path(
            "/root/autodl-tmp/graph/upstream/AgenticIR/eval/compute_scores.py"
        ),
        "agenticir_compare_methods": Path(
            "/root/autodl-tmp/graph/upstream/AgenticIR/eval/compare_methods.py"
        ),
        "table1_scorer_module": root / "src/evaluation/agenticir_table1.py",
        "stage0_table1_scorer_cli": root / "scripts/score_stage0_agenticir_table1.py",
        "metric_weight_inventory": root
        / "artifacts/formal_mio100/cache/weights_lock.json",
        "stage4_formal_authorization": root
        / "artifacts/approvals/FORMAL_MIO100_APPROVED.json",
        "stage4_formal_evaluator_complete": STAGE4_FORMAL_ROOT / "complete.json",
        "stage4_table1_complete": STAGE4_FORMAL_ROOT / "table1_scores/complete.json",
        "stage4_table1_per_image": STAGE4_FORMAL_ROOT / "table1_scores/per_image.csv",
        "stage4_table1_summary": STAGE4_FORMAL_ROOT / "table1_scores/summary.json",
    }


def validate_stage0_ready_without_torch(
    summary_path: str | Path,
    *,
    checkpoint_path: str | Path,
    primary_validation_path: str | Path,
) -> Mapping[str, Any]:
    """Cross-check Stage0 selection evidence without loading torch/CUDA."""

    summary_file = canonical_regular_file(summary_path, field="Stage0 summary")
    checkpoint = canonical_regular_file(checkpoint_path, field="Stage0 best EMA")
    validation_file = canonical_regular_file(
        primary_validation_path, field="Stage0 final primary validation"
    )
    summary = _load_json(summary_file, field="Stage0 summary")
    validation = _load_json(validation_file, field="Stage0 final validation")
    runtime = summary.get("runtime")
    if not isinstance(runtime, Mapping):
        raise FormalInventoryError("Stage0 summary runtime is malformed")
    if (
        summary.get("schema_version") != "graphrestore-stage0-run-v1"
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("completed_step") != 60_000
        or summary.get("target_step") != 60_000
        or summary.get("integration") is not False
        or summary.get("finite") is not True
        or summary.get("best_checkpoint") != str(checkpoint)
        or runtime.get("schedule_max_steps") != 60_000
        or runtime.get("target_step") != 60_000
        or runtime.get("integration") is not False
    ):
        raise FormalInventoryError("Stage0 is not ready for formal authorization")
    for key in (
        "maximum_train_peak_reserved_fraction",
        "maximum_validation_peak_reserved_fraction",
    ):
        value = summary.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 0.90
        ):
            raise FormalInventoryError(f"Stage0 summary has invalid {key}")
    if (
        validation.get("schema_version") != "graphrestore-stage0-primary-val-v1"
        or validation.get("protocol_id") != "agenticir_official_parity"
        or validation.get("step") != 60_000
        or validation.get("image_count") != 1_600
    ):
        raise FormalInventoryError("Stage0 final validation scope drifted")
    summary_validation = summary.get("validation")
    if not isinstance(summary_validation, Mapping):
        raise FormalInventoryError("Stage0 summary validation is malformed")
    for key in (
        "single_psnr",
        "single_ssim",
        "group_a_psnr",
        "group_a_ssim",
        "image_count",
        "task_means",
    ):
        if summary_validation.get(key) != validation.get(key):
            raise FormalInventoryError(
                f"Stage0 summary/final validation drifted at {key}"
            )
    return summary


def validate_stage0_readiness_receipt_without_torch(
    path: str | Path,
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    summary_path: str | Path,
    primary_validation_path: str | Path,
    calibration_history_path: str | Path,
    report_path: str | Path,
    stage1_run_contract_path: str | Path,
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
    """Verify a CPU checkpoint audit receipt without importing torch itself."""

    receipt_path = canonical_regular_file(path, field="Stage0 readiness receipt")
    require_mode_0444(receipt_path, field="Stage0 readiness receipt")
    receipt = _load_json(receipt_path, field="Stage0 readiness receipt")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "created_utc",
        "cuda_initialized_before",
        "cuda_initialized_after",
        "checkpoint",
        "config",
        "summary",
        "primary_validation",
        "calibration_history",
        "report",
        "stage1_run_contract",
        "stage1_parent_receipt",
        "provenance_verification",
    }
    if set(receipt) != expected_keys:
        raise FormalInventoryError("Stage0 readiness receipt fields drifted")
    if (
        receipt.get("schema_version") != "graphrestore-stage0-formal-readiness-v1"
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "READY"
        or receipt.get("cuda_initialized_before") is not False
        or receipt.get("cuda_initialized_after") is not False
    ):
        raise FormalInventoryError("Stage0 readiness receipt scope drifted")
    _validate_utc(receipt.get("created_utc"))
    expected_files = {
        "checkpoint": (checkpoint_path, expected_checkpoint_sha256),
        "config": (config_path, expected_config_sha256),
        "summary": (summary_path, None),
        "primary_validation": (primary_validation_path, None),
        "calibration_history": (calibration_history_path, None),
        "report": (report_path, None),
        "stage1_run_contract": (stage1_run_contract_path, None),
    }
    for name, (raw_path, frozen_sha) in expected_files.items():
        expected_path = canonical_regular_file(raw_path, field=f"expected {name}")
        raw = receipt.get(name)
        required = {"path", "sha256"}
        if name == "checkpoint":
            required |= {
                "stage",
                "step",
                "model_role",
                "resumable",
                "pending_validation_step",
                "tensor_count",
                "ema_num_updates",
                "model_equals_ema_shadow",
                "all_tensors_finite",
                "all_tensors_cpu",
                "provenance_config_sha256",
                "provenance_sha256",
                "semantic_source_count",
                "manifest_binding_count",
                "parent_checkpoint_sha256",
                "warm_start_loaded_count",
            }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise FormalInventoryError(f"Stage0 readiness {name} fields drifted")
        actual_sha = sha256_file(expected_path)
        if raw.get("path") != str(expected_path) or raw.get("sha256") != actual_sha:
            raise FormalInventoryError(f"Stage0 readiness {name} binding drifted")
        if frozen_sha is not None and actual_sha != frozen_sha:
            raise FormalInventoryError(f"Stage0 readiness frozen {name} drifted")
    checkpoint = receipt["checkpoint"]
    if (
        checkpoint.get("stage") != "stage0"
        or checkpoint.get("step") != 60_000
        or checkpoint.get("model_role") != "ema_selection"
        or checkpoint.get("resumable") is not False
        or checkpoint.get("pending_validation_step") is not None
        or checkpoint.get("tensor_count") != expected_tensor_count
        or checkpoint.get("ema_num_updates") != 60_000
        or checkpoint.get("model_equals_ema_shadow") is not True
        or checkpoint.get("all_tensors_finite") is not True
        or checkpoint.get("all_tensors_cpu") is not True
        or checkpoint.get("provenance_config_sha256") != expected_config_sha256
        or checkpoint.get("semantic_source_count") != expected_semantic_source_count
        or checkpoint.get("manifest_binding_count") != 4
        or checkpoint.get("provenance_sha256") != expected_provenance_sha256
        or not _is_sha256(checkpoint.get("parent_checkpoint_sha256"))
        or checkpoint.get("warm_start_loaded_count") != 495
    ):
        raise FormalInventoryError("Stage0 readiness checkpoint audit drifted")
    parent = receipt.get("stage1_parent_receipt")
    if not isinstance(parent, Mapping) or set(parent) != {
        "path",
        "sha256",
        "source",
        "source_tensor_count",
        "loaded_count",
        "missing_count",
        "missing_prefixes",
        "unexpected_keys",
        "shape_mismatches",
    }:
        raise FormalInventoryError("Stage0 readiness Stage1 parent receipt drifted")
    if (
        parent.get("path") != str(Path(checkpoint_path).resolve(strict=True))
        or parent.get("sha256") != expected_checkpoint_sha256
        or parent.get("source") != "stage0_best_ema_shadow"
        or parent.get("source_tensor_count") != expected_tensor_count
        or parent.get("loaded_count") != expected_tensor_count
        or parent.get("missing_count") != expected_stage1_missing_count
        or parent.get("missing_prefixes") != ["decoder.skill_bank."]
        or parent.get("unexpected_keys") != []
        or parent.get("shape_mismatches") != []
    ):
        raise FormalInventoryError("Stage1 early parent receipt is not exact")
    project_root = Path(expected_project_root).resolve(strict=True)
    if not project_root.is_dir():
        raise FormalInventoryError("Stage0 readiness project root is not a directory")
    verification = receipt.get("provenance_verification")
    if not isinstance(verification, Mapping) or set(verification) != {
        "project_root",
        "semantic_source_count",
        "manifest_binding_count",
        "semantic_sources",
        "compatibility_mismatches",
        "manifests",
    }:
        raise FormalInventoryError("Stage0 provenance verification fields drifted")
    raw_sources = verification.get("semantic_sources")
    raw_compatibility = verification.get("compatibility_mismatches")
    raw_manifests = verification.get("manifests")
    if (
        verification.get("project_root") != str(project_root)
        or verification.get("semantic_source_count") != expected_semantic_source_count
        or verification.get("manifest_binding_count") != 4
        or not isinstance(raw_sources, Mapping)
        or len(raw_sources) != expected_semantic_source_count
        or not isinstance(raw_compatibility, Mapping)
        or set(raw_compatibility) != set(expected_provenance_compatibility)
        or not isinstance(raw_manifests, Mapping)
        or set(raw_manifests)
        != {"clean_train", "clean_val", "primary_train", "primary_val"}
    ):
        raise FormalInventoryError("Stage0 provenance verification scope drifted")
    for name, raw in raw_sources.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(raw, Mapping)
            or set(raw) != {"path", "sha256"}
        ):
            raise FormalInventoryError("Stage0 semantic-source receipt drifted")
        expected_path = canonical_regular_file(
            project_root / name, field=f"Stage0 semantic source {name}"
        )
        if (
            expected_path != project_root / name
            or raw.get("path") != str(expected_path)
            or raw.get("sha256") != sha256_file(expected_path)
        ):
            raise FormalInventoryError(
                f"Stage0 semantic source bytes drifted at {name}"
            )
    for name, raw in raw_compatibility.items():
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "checkpoint_sha256",
            "current_sha256",
            "rationale",
        }:
            raise FormalInventoryError("Stage0 provenance compatibility fields drifted")
        source = raw_sources[name]
        expected = expected_provenance_compatibility[name]
        if set(expected) != {
            "checkpoint_sha256",
            "current_sha256",
            "rationale",
        }:
            raise FormalInventoryError(
                "Stage0 frozen provenance compatibility contract drifted"
            )
        if (
            raw.get("path") != source["path"]
            or raw.get("current_sha256") != source["sha256"]
            or raw.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or raw.get("current_sha256") != expected["current_sha256"]
            or raw.get("rationale") != expected["rationale"]
            or not _is_sha256(expected["checkpoint_sha256"])
            or not _is_sha256(expected["current_sha256"])
            or expected["checkpoint_sha256"] == expected["current_sha256"]
        ):
            raise FormalInventoryError(
                f"Stage0 provenance compatibility drifted at {name}"
            )
    reconstructed_semantic = {
        name: (
            expected_provenance_compatibility[name]["checkpoint_sha256"]
            if name in expected_provenance_compatibility
            else raw["sha256"]
        )
        for name, raw in raw_sources.items()
    }
    if (
        not _is_sha256(expected_semantic_source_map_sha256)
        or _sha256_json(reconstructed_semantic) != expected_semantic_source_map_sha256
    ):
        raise FormalInventoryError("Stage0 semantic-source identity map drifted")
    for name, raw in raw_manifests.items():
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise FormalInventoryError("Stage0 manifest receipt fields drifted")
        manifest_path = canonical_regular_file(
            raw.get("path"), field=f"Stage0 manifest {name}"
        )
        if raw.get("sha256") != sha256_file(manifest_path):
            raise FormalInventoryError(f"Stage0 manifest bytes drifted at {name}")
    reconstructed_manifests = {
        name: {"path": raw["path"], "sha256": raw["sha256"]}
        for name, raw in raw_manifests.items()
    }
    if (
        not _is_sha256(expected_manifest_map_sha256)
        or _sha256_json(reconstructed_manifests) != expected_manifest_map_sha256
    ):
        raise FormalInventoryError("Stage0 manifest identity map drifted")
    return receipt


def _expected_scope() -> Mapping[str, Any]:
    return {
        "schema_version": STAGE0_AUTHORIZATION_SCHEMA,
        "kind": STAGE0_AUTHORIZATION_KIND,
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "stage0_formal_mio100_authorized": True,
        "one_shot": True,
        "inference_only": True,
        "authorized_groups": ["A", "B", "C"],
        "manifest_row_count": FORMAL_ROW_COUNT,
        "method_name": STAGE0_METHOD_NAME,
        "reference_method_name": "graphrestore_v7_1_stage4_step040000",
        "shard_count": 1,
        "output_root": str(STAGE0_OUTPUT_ROOT),
        "restrictions": dict(AUTHORIZATION_RESTRICTIONS),
    }


def build_stage0_authorization_payload(
    binding_paths: Mapping[str, str | Path],
    *,
    approved_utc: str,
) -> Mapping[str, Any]:
    if set(binding_paths) != set(REQUIRED_STAGE0_AUTHORIZATION_BINDINGS):
        raise FormalInventoryError("Stage0 authorization binding keys drifted")
    _validate_utc(approved_utc)
    bindings: dict[str, Mapping[str, str]] = {}
    for name in REQUIRED_STAGE0_AUTHORIZATION_BINDINGS:
        path = canonical_regular_file(binding_paths[name], field=f"binding {name}")
        if name in {
            "formal_data_inventory",
            "inventory_origin_protocol",
            "stage0_control_protocol",
            "stage0_user_authorization_protocol",
            "metric_weight_inventory",
        }:
            require_mode_0444(path, field=f"binding {name}")
        if name == "stage0_user_authorization_protocol":
            validate_stage0_user_authorization_protocol(path)
        bindings[name] = {"path": str(path), "sha256": sha256_file(path)}
    frozen_hashes = {
        "stage0_checkpoint": FROZEN_STAGE0_CHECKPOINT_SHA256,
        "stage0_config": FROZEN_STAGE0_CONFIG_SHA256,
        "formal_manifest": FORMAL_MANIFEST_SHA256,
        "formal_data_inventory": FROZEN_FORMAL_DATA_INVENTORY_SHA256,
        "metric_weight_inventory": FROZEN_METRIC_WEIGHT_INVENTORY_SHA256,
        "stage4_table1_complete": FROZEN_STAGE4_TABLE1_COMPLETE_SHA256,
        "stage4_table1_per_image": FROZEN_STAGE4_TABLE1_PER_IMAGE_SHA256,
        "stage4_table1_summary": FROZEN_STAGE4_TABLE1_SUMMARY_SHA256,
        "stage0_control_protocol": FROZEN_STAGE0_CONTROL_PROTOCOL_SHA256,
    }
    for name, expected_sha256 in frozen_hashes.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise FormalInventoryError(f"frozen Stage0 control binding drifted: {name}")
    return {**_expected_scope(), "approved_utc": approved_utc, "bindings": bindings}


def validate_stage0_lightweight_authorization(
    path: str | Path,
    *,
    expected_binding_paths: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    approval = canonical_regular_file(path, field="Stage0 formal authorization")
    require_mode_0444(approval, field="Stage0 formal authorization")
    payload = _load_json(approval, field="Stage0 formal authorization")
    if set(payload) != STAGE0_AUTHORIZATION_KEYS:
        raise FormalInventoryError("Stage0 authorization fields drifted")
    expected_scope = _expected_scope()
    if any(payload.get(key) != value for key, value in expected_scope.items()):
        raise FormalInventoryError("Stage0 authorization scope drifted")
    _validate_utc(payload.get("approved_utc"))
    raw_bindings = payload.get("bindings")
    if not isinstance(raw_bindings, Mapping) or set(raw_bindings) != set(
        REQUIRED_STAGE0_AUTHORIZATION_BINDINGS
    ):
        raise FormalInventoryError("Stage0 authorization binding keys drifted")
    expected_paths = {
        name: Path(raw).resolve(strict=False)
        for name, raw in expected_binding_paths.items()
    }
    if set(expected_paths) != set(REQUIRED_STAGE0_AUTHORIZATION_BINDINGS):
        raise FormalInventoryError("expected Stage0 binding keys drifted")
    for name in REQUIRED_STAGE0_AUTHORIZATION_BINDINGS:
        raw = raw_bindings[name]
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise FormalInventoryError(f"binding {name} fields drifted")
        bound = canonical_regular_file(raw.get("path"), field=f"binding {name}")
        if bound != expected_paths[name] or raw.get("sha256") != sha256_file(bound):
            raise FormalInventoryError(f"binding {name} path/hash drifted")
    frozen_hashes = {
        "stage0_checkpoint": FROZEN_STAGE0_CHECKPOINT_SHA256,
        "stage0_config": FROZEN_STAGE0_CONFIG_SHA256,
        "formal_manifest": FORMAL_MANIFEST_SHA256,
        "formal_data_inventory": FROZEN_FORMAL_DATA_INVENTORY_SHA256,
        "metric_weight_inventory": FROZEN_METRIC_WEIGHT_INVENTORY_SHA256,
        "stage4_table1_complete": FROZEN_STAGE4_TABLE1_COMPLETE_SHA256,
        "stage4_table1_per_image": FROZEN_STAGE4_TABLE1_PER_IMAGE_SHA256,
        "stage4_table1_summary": FROZEN_STAGE4_TABLE1_SUMMARY_SHA256,
        "stage0_control_protocol": FROZEN_STAGE0_CONTROL_PROTOCOL_SHA256,
    }
    for name, expected_sha256 in frozen_hashes.items():
        if raw_bindings[name]["sha256"] != expected_sha256:
            raise FormalInventoryError(f"frozen Stage0 control binding drifted: {name}")
    # These authorization/data/metric gates remain immutable after publication.
    for name in (
        "formal_data_inventory",
        "inventory_origin_protocol",
        "stage0_control_protocol",
        "stage0_user_authorization_protocol",
        "metric_weight_inventory",
    ):
        require_mode_0444(
            expected_paths[name], field=f"Stage0 authorization binding {name}"
        )
    validate_stage0_user_authorization_protocol(
        expected_paths["stage0_user_authorization_protocol"]
    )
    return payload


__all__ = [
    "FROZEN_STAGE0_MANIFEST_MAP_SHA256",
    "FROZEN_STAGE0_PROVENANCE_SHA256",
    "FROZEN_STAGE0_SEMANTIC_SOURCE_MAP_SHA256",
    "FORMAL_DATA_INVENTORY_PATH",
    "FORMAL_MANIFEST_FILENAME",
    "FORMAL_MANIFEST_SHA256",
    "REQUIRED_STAGE0_AUTHORIZATION_BINDINGS",
    "STAGE0_APPROVAL_PATH",
    "STAGE0_AUTHORIZATION_KIND",
    "STAGE0_AUTHORIZATION_PROTOCOL_PATH",
    "STAGE0_AUTHORIZATION_SCHEMA",
    "STAGE0_USER_AUTHORIZATION_PROTOCOL_PATH",
    "STAGE0_USER_AUTHORIZATION_SCOPE",
    "STAGE0_COMPARISON_ROOT",
    "STAGE0_METHOD_NAME",
    "STAGE0_OUTPUT_ROOT",
    "STAGE0_PROVENANCE_COMPATIBILITY",
    "STAGE0_SCORE_ROOT",
    "build_stage0_authorization_payload",
    "stage0_authorization_binding_paths",
    "validate_stage0_lightweight_authorization",
    "validate_stage0_user_authorization_protocol",
    "validate_stage0_readiness_receipt_without_torch",
    "validate_stage0_ready_without_torch",
]
