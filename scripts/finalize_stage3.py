#!/usr/bin/env python3
"""Finalize frozen Stage3 step-12000 without any further training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GraphRestoreEpisodeDataset  # noqa: E402
from src.data.manifests import SKILLS  # noqa: E402
from src.training import stage3_engine  # noqa: E402
from src.training.selection import ValidationScore  # noqa: E402
from src.training.stage3_engine import (  # noqa: E402
    PROTOCOL_ID,
    STAGE3_SCHEMA,
    THRESHOLD_SCHEMA,
    THRESHOLD_TIE_BREAK,
    THRESHOLD_F1_TOLERANCE,
    Stage3ContractError,
    calibrate_presence_thresholds,
    collect_primary_val_presence,
    configure_stage3_reproducibility,
    enforce_stage3_peak_memory,
    freeze_presence_thresholds,
    load_relation_records,
    load_stage3_best_ema,
    relation_baseline_audit,
    reset_stage3_peak_memory,
    validate_stage3,
    validate_stage3_allocator_conf,
    validate_stage3_approval,
    validate_stage3_finalization_outputs,
)
from src.training.stage3_finalization import (  # noqa: E402
    Stage3FinalizationContractError,
    Stage3RevocationAuthorization,
    validate_stage3_extension_revocation,
)
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.io import atomic_write_json, atomic_write_text, load_json, utc_now_iso  # noqa: E402


SELECTED_STEP = 12_000
CALIBRATED_VALIDATION_FILENAME = "selected_validation_calibrated.json"
COMPLETE_FILENAME = "complete.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3_planner.yaml"),
    )
    parser.add_argument(
        "--finalization_authorization",
        type=Path,
        required=True,
        help="canonical STAGE3_EXTENSION_REVOKED.json finalize-only authorization",
    )
    return parser


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _binding(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise Stage3ContractError(f"{field} binding is missing")
    path_raw, digest = value.get("path"), value.get("sha256")
    if not isinstance(path_raw, str) or not isinstance(digest, str):
        raise Stage3ContractError(f"{field} binding is invalid")
    path = Path(path_raw).resolve(strict=False)
    if not path.is_file() or sha256_file(path) != digest:
        raise Stage3ContractError(f"{field} physical hash drifted")
    return {"path": str(path), "sha256": digest}


def _assert_frozen_inputs(
    authorization: Stage3RevocationAuthorization,
) -> Stage3RevocationAuthorization:
    refreshed = validate_stage3_extension_revocation(
        authorization.path,
        project_root=PROJECT_ROOT,
    )
    if refreshed.sha256 != authorization.sha256:
        raise Stage3ContractError("Stage3 finalization authorization changed")
    required = (
        "run_contract",
        "abandoned_last_checkpoint",
        "selected_checkpoint",
        "selected_validation",
        "calibration_history",
        "historical_extension_authorization",
        "stage3_config",
        "primary_val_manifest",
        "relation_val",
        "pair_prior",
        "global_priority",
    )
    for logical in required:
        _binding(refreshed.bindings.get(logical), field=logical)
    return refreshed


def _finite_tree(value: object, *, field: str) -> None:
    if value is None:
        raise Stage3ContractError(f"{field} contains null/non-finite output")
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Stage3ContractError(f"{field} contains non-finite output")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _finite_tree(item, field=f"{field}[{index}]")
        return
    raise Stage3ContractError(f"{field} contains unsupported output type")


def _publish_json_once(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = load_json(path)
        if not isinstance(existing, Mapping):
            raise Stage3ContractError(f"existing {path.name} is invalid")
        expected = json.loads(json.dumps(dict(payload), sort_keys=True))
        current = json.loads(json.dumps(dict(existing), sort_keys=True))
        expected.pop("created_utc", None)
        current.pop("created_utc", None)
        if current != expected:
            raise Stage3ContractError(
                f"existing {path.name} scientific content drifted"
            )
        return dict(existing)
    atomic_write_json(path, payload)
    return dict(payload)


def _selected_score(summary: Mapping[str, Any]) -> ValidationScore:
    restoration = summary.get("restoration")
    if not isinstance(restoration, Mapping):
        raise Stage3ContractError("selected validation restoration is missing")
    single, group_a = restoration.get("single"), restoration.get("group_a")
    if not isinstance(single, Mapping) or not isinstance(group_a, Mapping):
        raise Stage3ContractError("selected validation task metrics are missing")
    score = ValidationScore(
        group_a_psnr=float(group_a["psnr"]),
        group_a_ssim=float(group_a["ssim"]),
        single_psnr=float(single["psnr"]),
        single_ssim=float(single["ssim"]),
        step=SELECTED_STEP,
    )
    _finite_tree(
        {
            "group_a_psnr": score.group_a_psnr,
            "group_a_ssim": score.group_a_ssim,
            "single_psnr": score.single_psnr,
            "single_ssim": score.single_ssim,
        },
        field="selected score",
    )
    return score


def _validated_threshold_values(
    payload: Mapping[str, Any],
    *,
    authorization: Stage3RevocationAuthorization,
    selected_sha256: str,
    primary_val_sha256: str,
    stage3_approval_sha256: str,
) -> list[float]:
    if (
        payload.get("schema_version") != THRESHOLD_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("frozen") is not True
        or payload.get("skills") != list(SKILLS)
        or payload.get("baseline_threshold") != 0.50
        or payload.get("search_grid") != [value / 100.0 for value in range(20, 81, 2)]
        or payload.get("tie_break") != THRESHOLD_TIE_BREAK
        or payload.get("numerical_tolerance") != THRESHOLD_F1_TOLERANCE
        or payload.get("calibration_runs") != 1
        or payload.get("checkpoint_sha256") != selected_sha256
        or payload.get("primary_val_manifest_sha256") != primary_val_sha256
        or payload.get("stage3_approval_sha256") != stage3_approval_sha256
        or payload.get("stage3_extension_authorization_sha256")
        != authorization.bindings["historical_extension_authorization"]["sha256"]
        or payload.get("stage3_finalization_authorization_sha256")
        != authorization.sha256
        or payload.get("mio100_rows_read") != 0
        or payload.get("group_b_rows_read") != 0
        or payload.get("group_c_rows_read") != 0
    ):
        raise Stage3ContractError("frozen Stage3 thresholds fail finalizer contract")
    values = payload.get("thresholds")
    metrics = payload.get("per_skill_metrics")
    per_skill_f1 = payload.get("per_skill_f1")
    code = payload.get("calibration_code")
    if (
        not isinstance(values, Mapping)
        or not isinstance(metrics, Mapping)
        or not isinstance(per_skill_f1, Mapping)
        or set(values) != set(SKILLS)
        or set(metrics) != set(SKILLS)
        or set(per_skill_f1) != set(SKILLS)
    ):
        raise Stage3ContractError("frozen Stage3 threshold metrics are missing")
    if not isinstance(code, Mapping):
        raise Stage3ContractError("frozen Stage3 calibration code binding is missing")
    code_path = Path(str(code.get("path"))).resolve(strict=False)
    if (
        code_path != Path(stage3_engine.__file__).resolve()
        or not code_path.is_file()
        or code.get("sha256") != sha256_file(code_path)
    ):
        raise Stage3ContractError("frozen Stage3 calibration code binding drifted")
    grid = {value / 100.0 for value in range(20, 81, 2)}
    ordered: list[float] = []
    before_values: list[float] = []
    after_values: list[float] = []
    for skill in SKILLS:
        threshold = float(values[skill])
        if threshold not in grid:
            raise Stage3ContractError(f"{skill}: frozen threshold escaped grid")
        row = metrics.get(skill)
        if not isinstance(row, Mapping):
            raise Stage3ContractError(f"{skill}: frozen threshold metrics are missing")
        baseline, calibrated = row.get("baseline"), row.get("calibrated")
        if not isinstance(baseline, Mapping) or not isinstance(calibrated, Mapping):
            raise Stage3ContractError(f"{skill}: threshold metric roles are missing")
        expected_metric_fields = {"threshold", "precision", "recall", "f1"}
        if (
            set(baseline) != expected_metric_fields
            or set(calibrated) != expected_metric_fields
        ):
            raise Stage3ContractError(f"{skill}: threshold metric fields drifted")
        before, after = float(baseline["f1"]), float(calibrated["f1"])
        metrics_to_check = [
            before,
            after,
            float(baseline["precision"]),
            float(baseline["recall"]),
            float(calibrated["precision"]),
            float(calibrated["recall"]),
        ]
        if (
            not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in metrics_to_check
            )
            or after + THRESHOLD_F1_TOLERANCE < before
            or float(calibrated["threshold"]) != threshold
            or float(baseline["threshold"]) != 0.50
            or float(per_skill_f1[skill]) != after
        ):
            raise Stage3ContractError(f"{skill}: calibrated F1 contract drifted")
        ordered.append(threshold)
        before_values.append(before)
        after_values.append(after)
    macro_before = float(payload["macro_f1_before"])
    macro_after = float(payload["macro_f1_after"])
    if (
        macro_after + THRESHOLD_F1_TOLERANCE < macro_before
        or macro_before != math.fsum(before_values) / len(SKILLS)
        or macro_after != math.fsum(after_values) / len(SKILLS)
    ):
        raise Stage3ContractError("calibrated Stage3 macro F1 regressed")
    return ordered


def _render_report(
    calibrated: Mapping[str, Any],
    *,
    original: Mapping[str, Any],
    score: ValidationScore,
    checkpoint: Path,
    thresholds: Path,
    authorization: Stage3RevocationAuthorization,
) -> str:
    planner = calibrated["planner"]
    relation = calibrated["relation"]
    guard = calibrated["guard"]
    graph = calibrated["graph"]
    baseline = relation["cpu_baseline_audit"]
    restoration = calibrated["restoration"]
    original_restoration = original["restoration"]
    threshold_payload = load_json(thresholds)
    skill_lines = "".join(
        f"- {skill} threshold/P/R/F1/activation: "
        f"{planner['per_skill'][skill]['threshold']:.2f} / "
        f"{planner['per_skill'][skill]['precision']:.10f} / "
        f"{planner['per_skill'][skill]['recall']:.10f} / "
        f"{planner['per_skill'][skill]['f1']:.10f} / "
        f"{planner['per_skill'][skill]['activation_rate']:.10f}\n"
        for skill in SKILLS
    )
    return (
        "# Stage3 Planner and Guard\n\n"
        f"- protocol: `{PROTOCOL_ID}`\n"
        "- finalization mode: `step12000_finalize_only_no_training`\n"
        f"- selected step: {SELECTED_STEP}\n"
        f"- selected best checkpoint: `{checkpoint.resolve()}`\n"
        f"- selected best checkpoint SHA256: `{sha256_file(checkpoint)}`\n"
        "- Stage4 parent role: only_stage3_parent\n"
        f"- finalization authorization SHA256: `{authorization.sha256}`\n"
        f"- frozen thresholds: `{thresholds.resolve()}`\n"
        "- optimizer / scheduler / train loader created: false / false / false\n"
        "- checkpoint written: false\n"
        "- optimizer steps executed / checkpoint writes / sampler steps advanced: 0 / 0 / 0\n"
        "- step14000 pending checkpoint role: abandoned_unselected_extension_state\n"
        "- MiO100 / Group B / Group C rows read: 0 / 0 / 0\n"
        "- original selection and six-validation history remain byte-exact\n\n"
        "## Original selected diagnostic at threshold 0.50 (selection unchanged)\n\n"
        f"- Single PSNR/SSIM: {original_restoration['single']['psnr']:.10f} / "
        f"{original_restoration['single']['ssim']:.10f}\n"
        f"- Group-A PSNR/SSIM: {score.group_a_psnr:.10f} / {score.group_a_ssim:.10f}\n\n"
        "## Post-calibration full primary_val diagnostic\n\n"
        f"- Single PSNR/SSIM: {restoration['single']['psnr']:.10f} / "
        f"{restoration['single']['ssim']:.10f}\n"
        f"- Group-A PSNR/SSIM: {restoration['group_a']['psnr']:.10f} / "
        f"{restoration['group_a']['ssim']:.10f}\n"
        f"- planner macro F1: {planner['macro_f1']:.10f}\n"
        f"- macro F1 before/after calibration: "
        f"{threshold_payload['macro_f1_before']:.10f} / "
        f"{threshold_payload['macro_f1_after']:.10f}\n"
        f"- planner activation rate (skill slots): {planner['activation_rate']:.10f}\n"
        f"{skill_lines}"
        f"- learned raw relation accuracy: {relation['relation_accuracy_non_ambiguous']:.10f}\n"
        f"- learned raw relation macro-F1/balanced accuracy: "
        f"{relation['learned_raw']['macro_f1']:.10f} / "
        f"{relation['learned_raw']['balanced_accuracy']:.10f}\n"
        f"- parallel precision/recall: {relation['parallel_precision_non_ambiguous']:.10f} / "
        f"{relation['parallel_recall_non_ambiguous']:.10f}\n"
        f"- always-parallel baseline accuracy: {baseline['always_parallel']['accuracy']:.10f}\n"
        f"- always-parallel baseline macro-F1/balanced accuracy: "
        f"{baseline['always_parallel']['macro_f1']:.10f} / "
        f"{baseline['always_parallel']['balanced_accuracy']:.10f}\n"
        f"- per-pair majority-prior baseline accuracy: "
        f"{baseline['per_pair_majority_prior']['accuracy']:.10f}\n"
        f"- per-pair majority-prior macro-F1/balanced accuracy: "
        f"{baseline['per_pair_majority_prior']['macro_f1']:.10f} / "
        f"{baseline['per_pair_majority_prior']['balanced_accuracy']:.10f}\n"
        f"- rain guard Spearman/MAE: {guard['guard_spearman_rain']:.10f} / "
        f"{guard['guard_mae_rain']:.10f}\n"
        f"- haze guard Spearman/MAE: {guard['guard_spearman_haze']:.10f} / "
        f"{guard['guard_mae_haze']:.10f}\n"
        f"- mean program levels: {graph['mean_program_levels']:.10f}\n"
        f"- STOP rate: {graph['sample_stop_rate']:.10f}\n"
        "- STOP-rate definition: fraction of primary_val samples whose "
        "stopped_mask fired in any formal inference round\n"
        f"- post-compiler cycle rate: {graph['post_compiler_cycle_rate']:.10f}\n"
    )


def _run_post_calibration_diagnostic(
    model: Any,
    validation_dataset: GraphRestoreEpisodeDataset,
    val_relations: Mapping[str, Mapping[str, Any]],
    *,
    device: torch.device,
    threshold_values: Sequence[float],
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Invoke the production validator with its exact one-model API."""

    values = [float(value) for value in threshold_values]
    if len(values) != len(SKILLS) or not all(math.isfinite(value) for value in values):
        raise Stage3ContractError(
            "finalize-only diagnostic requires eight finite thresholds"
        )
    summary = validate_stage3(
        model,
        validation_dataset,
        val_relations,
        device=device,
        use_bf16=use_bf16,
        presence_threshold=values,
    )
    # Finalization artifacts always distinguish the calibrated eight-skill
    # vector from the scalar 0.50 checkpoint-selection protocol, even when the
    # calibrated values happen to be numerically identical.
    summary["checkpoint_presence_threshold"] = values
    summary["presence_thresholds"] = {
        skill: values[index] for index, skill in enumerate(SKILLS)
    }
    planner = summary.get("planner")
    if not isinstance(planner, Mapping) or not isinstance(
        planner.get("per_skill"), Mapping
    ):
        raise Stage3ContractError("finalize-only planner diagnostics are missing")
    for index, skill in enumerate(SKILLS):
        row = planner["per_skill"].get(skill)
        if not isinstance(row, dict):
            raise Stage3ContractError(
                f"finalize-only planner diagnostic is missing {skill}"
            )
        row["threshold"] = values[index]
    return summary


def _completion_bindings(
    *,
    authorization: Stage3RevocationAuthorization,
    paths: Any,
    calibrated_path: Path,
) -> dict[str, dict[str, str]]:
    source = authorization.bindings
    return {
        "best_checkpoint": _binding(source["selected_checkpoint"], field="selected"),
        "abandoned_last_checkpoint": _binding(
            source["abandoned_last_checkpoint"], field="abandoned last"
        ),
        "selected_validation": _binding(
            source["selected_validation"], field="selected validation"
        ),
        "thresholds": {
            "path": str(paths.thresholds.resolve()),
            "sha256": sha256_file(paths.thresholds),
        },
        "selected_validation_calibrated": {
            "path": str(calibrated_path.resolve()),
            "sha256": sha256_file(calibrated_path),
        },
        "report": {
            "path": str(paths.report.resolve()),
            "sha256": sha256_file(paths.report),
        },
        "finalization_authorization": authorization.provenance_binding(),
        "historical_extension_authorization": _binding(
            source["historical_extension_authorization"],
            field="historical extension authorization",
        ),
        "stage3_approval": _binding(source["stage3_approval"], field="Stage3 approval"),
        "approval_required": _binding(
            source["approval_required"], field="Stage3 approval-required"
        ),
        "stage1_checkpoint": _binding(
            source["stage1_checkpoint"], field="Stage1 checkpoint"
        ),
        "run_contract": _binding(source["run_contract"], field="Stage3 run contract"),
        "stage3_config": _binding(source["stage3_config"], field="Stage3 config"),
        "primary_val_manifest": _binding(
            source["primary_val_manifest"], field="primary_val"
        ),
        "relation_val": _binding(source["relation_val"], field="relation val"),
        "pair_prior": _binding(source["pair_prior"], field="pair prior"),
        "global_priority": _binding(source["global_priority"], field="global priority"),
        "calibration_history": _binding(
            source["calibration_history"], field="calibration history"
        ),
    }


def run(arguments: argparse.Namespace) -> int:
    authorization = validate_stage3_extension_revocation(
        _project_path(arguments.finalization_authorization),
        project_root=PROJECT_ROOT,
    )
    authorization = _assert_frozen_inputs(authorization)
    paths = validate_stage3_approval(
        _project_path(arguments.config),
        project_root=PROJECT_ROOT,
        require_orchestrator_running=False,
    )
    if int(paths.config["training"]["max_steps"]) != SELECTED_STEP:
        raise Stage3ContractError("Stage3 finalization max_steps drifted from 12000")
    if (
        sha256_file(paths.config_path)
        != authorization.bindings["stage3_config"]["sha256"]
    ):
        raise Stage3ContractError(
            "Stage3 finalization config differs from authorization"
        )

    complete_path = paths.output_dir / COMPLETE_FILENAME
    latest_validation_path = paths.output_dir / "validation_latest.json"
    latest_validation_sha256 = (
        sha256_file(latest_validation_path)
        if latest_validation_path.is_file()
        else None
    )
    if complete_path.is_file():
        validate_stage3_finalization_outputs(
            PROJECT_ROOT,
            finalization_authorization_sha256=authorization.sha256,
            historical_extension_authorization_sha256=authorization.bindings[
                "historical_extension_authorization"
            ]["sha256"],
        )
        return 0

    selected_binding = _binding(
        authorization.bindings["selected_checkpoint"], field="selected checkpoint"
    )
    selected_path = Path(selected_binding["path"])
    original_path = Path(authorization.bindings["selected_validation"]["path"])
    original = load_json(original_path)
    if not isinstance(original, Mapping):
        raise Stage3ContractError("original selected validation is invalid")
    score = _selected_score(original)

    parent_sha = paths.approval.bindings["stage1_checkpoint"]["sha256"]
    val_relations = load_relation_records(
        paths.relation_val,
        split="val",
        parent_checkpoint_sha256=parent_sha,
        interaction_manifest_sha256=paths.approval.bindings["interaction_val_manifest"][
            "sha256"
        ],
    )
    pair_prior = load_json(paths.pair_prior)
    if not isinstance(pair_prior, Mapping):
        raise Stage3ContractError("Stage3 pair prior is invalid")

    # Still before the first CUDA API query: every authorization, source,
    # selected snapshot, manifest and label identity is now closed.
    validate_stage3_allocator_conf()
    authorization = _assert_frozen_inputs(authorization)
    if not torch.cuda.is_available():
        raise Stage3ContractError("formal Stage3 finalization requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    configure_stage3_reproducibility(int(paths.config["seed"]))
    model = load_stage3_best_ema(
        paths,
        selected_path,
        device=device,
        load_frozen_thresholds=False,
        historical_extension_authorization=authorization.bindings[
            "historical_extension_authorization"
        ],
    )
    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(selected_payload, Mapping)
        or selected_payload.get("step") != SELECTED_STEP
    ):
        raise Stage3ContractError("Stage3 selected checkpoint is not step12000")
    del selected_payload

    depth_compat = PROJECT_ROOT / "artifacts/cache/agenticir_depth_compat"
    validation_dataset = GraphRestoreEpisodeDataset(
        paths.val_manifest,
        paths.training_data_root,
        depth_compat,
        crop_size=None,
        training=False,
        stage="stage3",
        base_seed=int(paths.config["seed"]),
        agenticir_repo=paths.resolved["agenticir_repo"],
        mioir_repo=paths.resolved["mioir_repo"],
    )
    if len(validation_dataset) != 1600:
        raise Stage3ContractError("formal Stage3 primary_val must contain 1600 rows")

    calibration_peak_bytes = calibration_peak_fraction = None
    if not paths.thresholds.exists():
        reset_stage3_peak_memory(device)
        probabilities, targets = collect_primary_val_presence(
            model,
            validation_dataset,
            device=device,
            use_bf16=True,
        )
        calibration_peak_bytes, calibration_peak_fraction = enforce_stage3_peak_memory(
            device,
            phase="step12000_presence_threshold_calibration",
            maximum_reserved_fraction=float(
                paths.config["runtime"]["vram_maximum_peak_reserved_fraction"]
            ),
        )
        calibration = calibrate_presence_thresholds(probabilities, targets)
        freeze_presence_thresholds(
            paths.thresholds,
            calibration,
            primary_val_manifest=paths.val_manifest,
            selected_checkpoint=selected_path,
            approval_sha256=paths.approval.approval_sha256,
            extension_authorization_sha256=authorization.bindings[
                "historical_extension_authorization"
            ]["sha256"],
            finalization_authorization_sha256=authorization.sha256,
        )
        del probabilities, targets
    threshold_payload = load_json(paths.thresholds)
    if not isinstance(threshold_payload, Mapping):
        raise Stage3ContractError("frozen Stage3 thresholds are invalid")
    threshold_values = _validated_threshold_values(
        threshold_payload,
        authorization=authorization,
        selected_sha256=selected_binding["sha256"],
        primary_val_sha256=authorization.bindings["primary_val_manifest"]["sha256"],
        stage3_approval_sha256=authorization.bindings["stage3_approval"]["sha256"],
    )
    authorization = _assert_frozen_inputs(authorization)

    calibrated_path = paths.output_dir / CALIBRATED_VALIDATION_FILENAME
    diagnostic_peak_bytes = diagnostic_peak_fraction = None
    if not calibrated_path.exists():
        reset_stage3_peak_memory(device)
        calibrated = _run_post_calibration_diagnostic(
            model,
            validation_dataset,
            val_relations,
            device=device,
            threshold_values=threshold_values,
            use_bf16=True,
        )
        diagnostic_peak_bytes, diagnostic_peak_fraction = enforce_stage3_peak_memory(
            device,
            phase="step12000_post_calibration_full_diagnostic",
            maximum_reserved_fraction=float(
                paths.config["runtime"]["vram_maximum_peak_reserved_fraction"]
            ),
        )
        relation = calibrated.get("relation")
        if not isinstance(relation, dict):
            raise Stage3ContractError("calibrated Stage3 relation metrics are missing")
        relation["cpu_baseline_audit"] = relation_baseline_audit(
            val_relations,
            pair_prior,
            learned_raw_accuracy=float(relation["relation_accuracy_non_ambiguous"]),
        )
        calibrated["diagnostic_role"] = "post_calibration_non_selection_diagnostic"
        calibrated["selected_step"] = SELECTED_STEP
        calibrated["selected_checkpoint_sha256"] = selected_binding["sha256"]
        calibrated["thresholds_sha256"] = sha256_file(paths.thresholds)
        calibrated["stage3_finalization_authorization_sha256"] = authorization.sha256
        calibrated["post_calibration_diagnostic_runs"] = 1
        calibrated["group_b_rows_read"] = 0
        calibrated["group_c_rows_read"] = 0
        _finite_tree(calibrated, field="post-calibration Stage3 diagnostic")
        _publish_json_once(calibrated_path, calibrated)
    calibrated = load_json(calibrated_path)
    if not isinstance(calibrated, Mapping):
        raise Stage3ContractError("post-calibration Stage3 diagnostic is invalid")
    _finite_tree(calibrated, field="post-calibration Stage3 diagnostic")
    authorization = _assert_frozen_inputs(authorization)

    atomic_write_text(
        paths.report,
        _render_report(
            calibrated,
            original=original,
            score=score,
            checkpoint=selected_path,
            thresholds=paths.thresholds,
            authorization=authorization,
        ),
    )
    completion: dict[str, Any] = {
        "schema_version": STAGE3_SCHEMA,
        "kind": "stage3_finalize_only",
        "protocol_id": PROTOCOL_ID,
        "completed_utc": utc_now_iso(),
        "step": SELECTED_STEP,
        "best_checkpoint": str(selected_path),
        "best_checkpoint_sha256": selected_binding["sha256"],
        "best_score": {
            "group_a_psnr": score.group_a_psnr,
            "group_a_ssim": score.group_a_ssim,
            "single_psnr": score.single_psnr,
            "single_ssim": score.single_ssim,
            "step": score.step,
        },
        "thresholds": str(paths.thresholds.resolve()),
        "thresholds_sha256": sha256_file(paths.thresholds),
        "optimizer_created": False,
        "scheduler_created": False,
        "train_loader_created": False,
        "checkpoint_written": False,
        "optimizer_steps_executed": 0,
        "checkpoint_writes": 0,
        "sampler_steps_advanced": 0,
        "abandoned_last_checkpoint_role": "abandoned_unselected_extension_state",
        "stage4_parent_role": "only_stage3_parent",
        "threshold_calibration_runs": 1,
        "post_calibration_diagnostic_runs": 1,
        "mio100_rows_read": 0,
        "group_b_rows_read": 0,
        "group_c_rows_read": 0,
        "bindings": _completion_bindings(
            authorization=authorization,
            paths=paths,
            calibrated_path=calibrated_path,
        ),
    }
    if calibration_peak_bytes is not None and calibration_peak_fraction is not None:
        completion.update(
            {
                "calibration_peak_reserved_bytes": calibration_peak_bytes,
                "calibration_peak_reserved_fraction": calibration_peak_fraction,
            }
        )
    if diagnostic_peak_bytes is not None and diagnostic_peak_fraction is not None:
        completion.update(
            {
                "diagnostic_peak_reserved_bytes": diagnostic_peak_bytes,
                "diagnostic_peak_reserved_fraction": diagnostic_peak_fraction,
            }
        )
    _publish_json_once(complete_path, completion)
    authorization = _assert_frozen_inputs(authorization)
    if (latest_validation_sha256 is None and latest_validation_path.exists()) or (
        latest_validation_sha256 is not None
        and sha256_file(latest_validation_path) != latest_validation_sha256
    ):
        raise Stage3ContractError("Stage3 validation_latest.json was modified")
    validate_stage3_finalization_outputs(
        PROJECT_ROOT,
        finalization_authorization_sha256=authorization.sha256,
        historical_extension_authorization_sha256=authorization.bindings[
            "historical_extension_authorization"
        ]["sha256"],
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (
        Stage3ContractError,
        Stage3FinalizationContractError,
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as exc:
        print(f"Stage3 finalization refused: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
