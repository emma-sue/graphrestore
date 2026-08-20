from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch

from src.training.orchestration import (
    D017_ACCEPTANCE,
    STAGE0_GROUP_A_PSNR_ANCHOR,
    STAGE0_GROUP_A_SSIM_ANCHOR,
    STAGE3_EXTENSION_APPROVAL_SCHEMA,
    STAGE3_EXTENSION_BACKUP_DIRECTORY,
    STAGE3_EXTENSION_BASE_STEP,
    STAGE3_EXTENSION_CYCLES,
    STAGE3_EXTENSION_LR_POLICY,
    STAGE3_EXTENSION_MIN_LR,
    STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS,
    STAGE3_EXTENSION_TARGET_STEP,
    STAGE3_EXTENSION_VALIDATION_EVERY_STEPS,
    STAGE3_EXTENSION_VALIDATION_STEPS,
    STAGE4_EXTENSION_BASE_STEP,
    STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS,
    STAGE4_EXTENSION_TARGET_STEP,
    STAGE4_EXTENSION_VALIDATION_EVERY_STEPS,
    STAGE4_EXTENSION_VALIDATION_STEPS,
    ApprovalError,
    ChildCommandError,
    CommandSpec,
    GraphRestoreOrchestrator,
    OrchestrationError,
    PipelineStatus,
    Stage4ExtensionAuthorization,
    SubprocessCommandRunner,
    command_plan,
    recommended_tmux_argv,
)
from src.training.stage3_finalization import Stage3RevocationAuthorization
from src.utils.hashing import sha256_file
from src.utils.io import load_json


class RecordingRunner:
    def __init__(
        self,
        callback: Callable[[CommandSpec, Path], None] | None = None,
        returncodes: dict[str, int] | None = None,
    ) -> None:
        self.commands: list[CommandSpec] = []
        self.callback = callback
        self.returncodes = returncodes or {}

    def run(self, command: CommandSpec, *, cwd: Path, log_path: Path) -> int:
        self.commands.append(command)
        if self.callback is not None:
            self.callback(command, cwd)
        return self.returncodes.get(command.name, 0)


def _write(path: Path, value: str | bytes = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _write_checkpoint(
    path: Path,
    *,
    stage: str,
    model_role: str,
    resumable: bool,
    approval_sha256: str,
    step: int,
    extension_authorization: Mapping[str, object] | None = None,
    stage4_extension_authorization: (Stage4ExtensionAuthorization | None) = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if stage == "stage3":
        provenance = {
            "stage3_approval": {"sha256": approval_sha256},
            "runtime": {"max_steps": STAGE3_EXTENSION_BASE_STEP},
        }
        if extension_authorization is not None:
            provenance["stage3_extension"] = dict(extension_authorization)
            provenance["runtime"]["training_target_step"] = STAGE3_EXTENSION_TARGET_STEP
    else:
        provenance = {"parents": {"stage3_approval": {"sha256": approval_sha256}}}
        if extension_authorization is not None:
            provenance["stage3_extension"] = {
                "path": extension_authorization["path"],
                "sha256": extension_authorization["sha256"],
            }
        if stage4_extension_authorization is not None:
            provenance["stage4_extension"] = (
                stage4_extension_authorization.provenance_binding()
            )
            provenance["runtime"] = {
                "max_steps": STAGE4_EXTENSION_TARGET_STEP,
                "schedule_max_steps": STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS,
            }
    torch.save(
        {
            "schema_version": "graphrestore-checkpoint-v1",
            "stage": stage,
            "step": step,
            "model": {},
            "ema": {},
            "optimizer": {},
            "scheduler": {},
            "rng_states": {},
            "sampler_state": {},
            "provenance": provenance,
            "model_role": model_role,
            "resumable": resumable,
            "metrics": {"validation_step": step, "best_step": step},
            "pending_validation_step": None,
            "optimizer_transaction_active": False,
        },
        path,
    )


def _write_raw_post_approval_checkpoint(
    root: Path,
    stage: str,
    *,
    model_role: str = "raw_training_state",
    resumable: bool = True,
) -> None:
    approval_sha = sha256_file(root / "artifacts/approvals/STAGE3_APPROVED.json")
    _write_checkpoint(
        root / f"artifacts/checkpoints/{stage}/last.pth",
        stage=stage,
        model_role=model_role,
        resumable=resumable,
        approval_sha256=approval_sha,
        step=12_000 if stage == "stage3" else 40_000,
    )


def _minimal_stage4_diagnostic_mode() -> dict[str, object]:
    return {
        "single_equal_task_mean": {"psnr": 25.0, "ssim": 0.80},
        "group_a_equal_combination_mean": {"psnr": 24.0, "ssim": 0.75},
        "diagnostics": {"sentinel": 1.0},
        "image_count": 1,
        "peak_reserved_bytes": 1,
        "peak_reserved_fraction": 0.10,
    }


def _stage4_extension_authorization(root: Path) -> Stage4ExtensionAuthorization:
    conditional = (
        root / "artifacts/approvals/STAGE4_EXTENSION_CONDITIONAL_APPROVED.json"
    )
    gate = root / "artifacts/approvals/STAGE4_EXTENSION_GATE_RECEIPT.json"
    _write(conditional, '{"approved":true}\n')
    _write(gate, '{"decision":"ACTIVATE_EXTENSION"}\n')
    return Stage4ExtensionAuthorization(
        conditional_path=conditional.resolve(),
        conditional_sha256=sha256_file(conditional),
        gate_path=gate.resolve(),
        gate_sha256=sha256_file(gate),
        payload={"decision": "ACTIVATE_EXTENSION"},
    )


def _write_post_approval_completion(
    root: Path,
    stage: str,
    *,
    best_step: int | None = None,
    extension_authorization: Mapping[str, object] | None = None,
    stage4_extension_authorization: (Stage4ExtensionAuthorization | None) = None,
) -> None:
    approval_sha = sha256_file(root / "artifacts/approvals/STAGE3_APPROVED.json")
    step = (
        STAGE3_EXTENSION_TARGET_STEP
        if stage == "stage3" and extension_authorization is not None
        else 12_000
        if stage == "stage3"
        else STAGE4_EXTENSION_TARGET_STEP
        if stage4_extension_authorization is not None
        else 40_000
    )
    selected_step = step if best_step is None else best_step
    directory = root / f"artifacts/checkpoints/{stage}"
    best = directory / "best_ema.pth"
    last = directory / "last.pth"
    _write_checkpoint(
        best,
        stage=stage,
        model_role="ema_selection",
        resumable=False,
        approval_sha256=approval_sha,
        step=selected_step,
        extension_authorization=extension_authorization,
        stage4_extension_authorization=stage4_extension_authorization,
    )
    _write_checkpoint(
        last,
        stage=stage,
        model_role="raw_training_state",
        resumable=True,
        approval_sha256=approval_sha,
        step=step,
        extension_authorization=extension_authorization,
        stage4_extension_authorization=stage4_extension_authorization,
    )
    if stage == "stage4" and stage4_extension_authorization is not None:
        last_payload = torch.load(last, map_location="cpu", weights_only=False)
        _write(
            directory / "run_contract.json",
            json.dumps(
                {
                    "schema_version": "graphrestore-stage4-runtime-v1",
                    "provenance": last_payload["provenance"],
                },
                sort_keys=True,
            )
            + "\n",
        )
    if stage == "stage3":
        thresholds = root / "artifacts/planner_thresholds.json"
        threshold_payload: dict[str, object] = {
            "schema_version": "graphrestore-presence-thresholds-v1",
            "protocol_id": "graphrestore-v7.1-agenticir-locked",
            "frozen": True,
            "checkpoint_sha256": sha256_file(best),
            "stage3_approval_sha256": approval_sha,
            "calibration_runs": 1,
            "mio100_rows_read": 0,
        }
        if extension_authorization is not None:
            threshold_payload.update(
                {
                    "stage3_extension_authorization_sha256": (
                        extension_authorization["sha256"]
                    ),
                    "selected_stage3_checkpoint": {
                        "path": str(best.resolve()),
                        "sha256": sha256_file(best),
                    },
                }
            )
        _write(
            thresholds,
            json.dumps(threshold_payload, sort_keys=True) + "\n",
        )
        report = root / "reports/STAGE3_PLANNER_GUARD.md"
        extension_report = (
            f"- completed training target step: {STAGE3_EXTENSION_TARGET_STEP}\n"
            "- cosine schedule horizon step: "
            f"{STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS}\n"
            "- Stage3 extension authorization SHA256: "
            f"`{extension_authorization['sha256']}`\n"
            if extension_authorization is not None
            else ""
        )
        _write(
            report,
            (
                "# Stage3 Planner and Guard\n\n"
                "- protocol: `graphrestore-v7.1-agenticir-locked`\n"
                + extension_report
                + f"- selected checkpoint SHA256: `{sha256_file(best)}`\n"
                "- Selected Single PSNR/SSIM: 25.0000000000 / 0.8000000000\n"
                "- Selected Group-A PSNR/SSIM: 24.0000000000 / 0.7500000000\n"
            ),
        )
        complete = {
            "schema_version": "graphrestore-stage3-runtime-v1",
            "step": step,
            "best_checkpoint": str(best.resolve()),
            "best_checkpoint_sha256": sha256_file(best),
            "thresholds": str(thresholds.resolve()),
            "thresholds_sha256": sha256_file(thresholds),
            "threshold_calibration_runs": 1,
            "mio100_rows_read": 0,
            "report": str(report.resolve()),
            "report_sha256": sha256_file(report),
            "best_score": {
                "group_a_psnr": 24.0,
                "group_a_ssim": 0.75,
                "single_psnr": 25.0,
                "single_ssim": 0.80,
                "step": selected_step,
            },
        }
        if extension_authorization is not None:
            selected_validation = directory / "selected_validation.json"
            validation = directory / "validation_latest.json"
            train_log = directory / "train.jsonl"
            validation_payload = {
                "protocol_id": "graphrestore-v7.1-agenticir-locked",
                "single_equal_task_mean": {"psnr": 25.0, "ssim": 0.80},
                "group_a_equal_combination_mean": {
                    "psnr": 24.0,
                    "ssim": 0.75,
                },
            }
            _write(
                selected_validation,
                json.dumps(validation_payload, sort_keys=True) + "\n",
            )
            _write(
                validation,
                json.dumps(validation_payload, sort_keys=True) + "\n",
            )
            _write(
                train_log,
                "".join(
                    json.dumps(
                        {"event": "validation", "step": validation_step},
                        sort_keys=True,
                    )
                    + "\n"
                    for validation_step in STAGE3_EXTENSION_VALIDATION_STEPS
                ),
            )
            complete.update(
                {
                    "extension_authorization": extension_authorization["path"],
                    "extension_authorization_sha256": (
                        extension_authorization["sha256"]
                    ),
                    "extension_validation_steps": list(
                        STAGE3_EXTENSION_VALIDATION_STEPS
                    ),
                    "schedule_horizon_steps": (STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS),
                    "training_target_step": STAGE3_EXTENSION_TARGET_STEP,
                    "selected_validation": str(selected_validation.resolve()),
                    "selected_validation_sha256": sha256_file(selected_validation),
                    "validation": str(validation.resolve()),
                    "validation_sha256": sha256_file(validation),
                }
            )
    else:
        selected_group_a_psnr = 24.0
        selected_group_a_ssim = 0.75
        selected_psnr_delta = selected_group_a_psnr - STAGE0_GROUP_A_PSNR_ANCHOR
        selected_ssim_delta = selected_group_a_ssim - STAGE0_GROUP_A_SSIM_ANCHOR
        retention_risk = selected_group_a_ssim < STAGE0_GROUP_A_SSIM_ANCHOR
        report = root / "reports/STAGE4_E2E.md"
        extension_report = (
            "- Conditional Stage4 extension: activated\n"
            "- Conditional authorization SHA256: "
            f"`{stage4_extension_authorization.conditional_sha256}`\n"
            "- Extension gate receipt SHA256: "
            f"`{stage4_extension_authorization.gate_sha256}`\n"
            f"- Completed training target step: {STAGE4_EXTENSION_TARGET_STEP}\n"
            "- Original cosine schedule horizon step: "
            f"{STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS}\n"
            if stage4_extension_authorization is not None
            else ""
        )
        _write(
            report,
            (
                "# Stage4 Full Guarded GraphRestore\n\n"
                "- Protocol: `graphrestore-v7.1-agenticir-locked`\n"
                f"- Selected EMA SHA256: `{sha256_file(best)}`\n"
                + extension_report
                + "- Selected Group-A PSNR/SSIM: 24.000000 / 0.75000000\n"
                "- Selected Single PSNR/SSIM: 25.000000 / 0.80000000\n"
                f"- Stage0 Group-A PSNR anchor: {STAGE0_GROUP_A_PSNR_ANCHOR!r}\n"
                f"- Stage0 Group-A SSIM anchor: {STAGE0_GROUP_A_SSIM_ANCHOR!r}\n"
                f"- Selected Group-A PSNR delta vs Stage0: {selected_psnr_delta!r}\n"
                f"- Selected Group-A SSIM delta vs Stage0: {selected_ssim_delta!r}\n"
                f"- SSIM_RETENTION_RISK: {str(retention_risk).lower()}\n"
                "- SSIM retention interpretation: The selected Group-A SSIM is "
                "below the frozen Stage0 anchor; this risk is not offset by any "
                "average PSNR gain.\n"
            ),
        )
        _write(
            root / "DECISION_MEMO.md",
            (
                "# Decision Memo\n\n"
                "Status: Stage3–4 complete; formal MiO100 remains unauthorized.\n\n"
                "- SSIM_RETENTION_RISK: `true`\n"
                f"- Stage0 Group-A SSIM: `{STAGE0_GROUP_A_SSIM_ANCHOR!r}`\n"
                f"- Stage4 selected Group-A SSIM: `{selected_group_a_ssim!r}`\n"
                f"- SSIM delta: `{selected_ssim_delta!r}`\n"
                f"- PSNR delta: `{selected_psnr_delta!r}`\n"
                "- The selected Group-A PSNR does not offset the SSIM retention "
                "deficit.\n"
            ),
        )
        validation = directory / "validation_latest.json"
        _write(
            validation,
            json.dumps(
                {
                    "protocol_id": "graphrestore-v7.1-agenticir-locked",
                    "single_equal_task_mean": {
                        "count": 800,
                        "psnr": 25.0,
                        "ssim": 0.80,
                    },
                    "group_a_equal_combination_mean": {
                        "count": 800,
                        "psnr": selected_group_a_psnr,
                        "ssim": selected_group_a_ssim,
                    },
                },
                sort_keys=True,
            )
            + "\n",
        )
        diagnostics_report = root / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.md"
        diagnostics_json = root / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.json"
        _write(
            diagnostics_report,
            (
                "# Guard and misuse diagnostics\n\n"
                f"- Selected best EMA SHA256: `{sha256_file(best)}`\n"
                "- full_partial_order\n"
                "- forced_total_order\n"
                "- parallel_only\n"
                "- predicted_spatial\n"
                "- global_mean\n"
                "- all_one\n"
            ),
        )
        _write(
            diagnostics_json,
            json.dumps(
                {
                    "schema_version": (
                        "graphrestore-stage4-zero-training-diagnostics-v1"
                    ),
                    "protocol_id": "graphrestore-v7.1-agenticir-locked",
                    "selected_best_ema_path": str(best.resolve()),
                    "selected_best_ema_sha256": sha256_file(best),
                    "optimizer_updates": 0,
                    "model_ema_rng_unchanged": True,
                    "compiler_modes": {
                        "full_partial_order": _minimal_stage4_diagnostic_mode(),
                        "forced_total_order": _minimal_stage4_diagnostic_mode(),
                        "parallel_only": _minimal_stage4_diagnostic_mode(),
                    },
                    "guard_modes": {
                        "predicted_spatial": _minimal_stage4_diagnostic_mode(),
                        "global_mean": _minimal_stage4_diagnostic_mode(),
                        "all_one": _minimal_stage4_diagnostic_mode(),
                    },
                },
                sort_keys=False,
            )
            + "\n",
        )
        complete = {
            "schema_version": "graphrestore-stage4-runtime-v1",
            "protocol_id": "graphrestore-v7.1-agenticir-locked",
            "step": step,
            "best_ema_path": str(best.resolve()),
            "best_ema_sha256": sha256_file(best),
            "diagnostics_json": str(diagnostics_json.resolve()),
            "diagnostics_json_sha256": sha256_file(diagnostics_json),
            "diagnostics_report": str(diagnostics_report.resolve()),
            "diagnostics_report_sha256": sha256_file(diagnostics_report),
            "diagnostics_selected_best_ema_sha256": sha256_file(best),
            "report": str(report.resolve()),
            "report_sha256": sha256_file(report),
            "best_score": {
                "group_a_psnr": selected_group_a_psnr,
                "group_a_ssim": selected_group_a_ssim,
                "single_psnr": 25.0,
                "single_ssim": 0.80,
                "step": selected_step,
            },
            "validation": str(validation.resolve()),
            "validation_sha256": sha256_file(validation),
            "latest_score": {
                "group_a_psnr": selected_group_a_psnr,
                "group_a_ssim": selected_group_a_ssim,
                "single_psnr": 25.0,
                "single_ssim": 0.80,
                "step": step,
            },
            "stage0_group_a_psnr_anchor": STAGE0_GROUP_A_PSNR_ANCHOR,
            "stage0_group_a_ssim_anchor": STAGE0_GROUP_A_SSIM_ANCHOR,
            "selected_group_a_psnr": selected_group_a_psnr,
            "selected_delta_group_a_psnr_vs_stage0": selected_psnr_delta,
            "selected_group_a_ssim": selected_group_a_ssim,
            "selected_delta_group_a_ssim_vs_stage0": selected_ssim_delta,
            "SSIM_RETENTION_RISK": retention_risk,
            "formal_mio100_started": False,
            "waiting_for": "new_user_authorization_for_formal_mio100",
        }
        if stage4_extension_authorization is not None:
            complete.update(
                {
                    "stage4_extension_conditional_authorization": str(
                        stage4_extension_authorization.conditional_path
                    ),
                    "stage4_extension_conditional_authorization_sha256": (
                        stage4_extension_authorization.conditional_sha256
                    ),
                    "stage4_extension_gate_receipt": str(
                        stage4_extension_authorization.gate_path
                    ),
                    "stage4_extension_gate_receipt_sha256": (
                        stage4_extension_authorization.gate_sha256
                    ),
                    "stage4_extension_validation_steps": list(
                        STAGE4_EXTENSION_VALIDATION_STEPS
                    ),
                    "schedule_horizon_steps": (STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS),
                    "training_target_step": STAGE4_EXTENSION_TARGET_STEP,
                    "additional_optimizer_steps": (
                        STAGE4_EXTENSION_TARGET_STEP - STAGE4_EXTENSION_BASE_STEP
                    ),
                    "further_extension_authorized": False,
                    "calibration_history_steps": list(
                        range(
                            STAGE4_EXTENSION_VALIDATION_EVERY_STEPS,
                            STAGE4_EXTENSION_TARGET_STEP + 1,
                            STAGE4_EXTENSION_VALIDATION_EVERY_STEPS,
                        )
                    ),
                }
            )
    _write(directory / "complete.json", json.dumps(complete, sort_keys=True) + "\n")


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "graphrestore"
    manifests = root / "fixture_manifests"
    manifest_paths: dict[str, Path] = {}
    for name in (
        "clean_train_manifest",
        "clean_val_manifest",
        "primary_train_manifest",
        "primary_val_manifest",
        "primary_all_manifest",
    ):
        path = manifests / f"{name}.jsonl"
        _write(path, '{"sample_id":"fixture"}\n')
        manifest_paths[name] = path

    resolved = "\n".join(f"{name}: {path}" for name, path in manifest_paths.items())
    _write(root / "configs/resolved_paths.yaml", f"{resolved}\n")
    for name in (
        "stage0_mio_stagea.yaml",
        "stage1_skill_bank.yaml",
        "stage2_interaction_distill.yaml",
        "stage3_planner.yaml",
        "stage4_graphrestore_e2e.yaml",
    ):
        _write(root / "configs" / name, f"fixture: {name}\n")
    _write(
        root / "DECISION_MEMO.md",
        "# Decision Memo\n\nStatus: paused before Stage3 pending explicit user approval.\n",
    )
    return root


def _stage_callback(command: CommandSpec, root: Path) -> None:
    if command.name == "integration_100_steps":
        directory = root / "artifacts/integration/stage0_100_steps"
        last = directory / "last.pth"
        _write(last, b"integration")
        _write(directory / "INTEGRATION_REPORT.md", "# pass\n")
        _write(directory / "micro_batch_probe.json", "{}\n")
        summary = {
            "schema_version": "graphrestore-stage0-run-v1",
            "protocol_id": "graphrestore-v7.1-agenticir-locked",
            "integration": True,
            "completed_step": 100,
            "target_step": 100,
            "finite": True,
            "runtime": {
                "crop_size": 192,
                "effective_batch": 8,
                "target_step": 100,
                "integration": True,
            },
            "peak_reserved_fraction": 0.8,
            "last_checkpoint": str(last.resolve()),
        }
        _write(directory / "summary.json", json.dumps(summary) + "\n")
    elif command.name == "stage0":
        _write(root / "artifacts/checkpoints/stage0/best_ema.pth", b"stage0")
    elif command.name == "stage1":
        _write(root / "artifacts/checkpoints/stage1/best_ema.pth", b"stage1")
    elif command.name == "effect_profiles":
        _write(
            root / "artifacts/interaction_labels/skill_effect_profiles.json",
            "{}\n",
        )
    elif command.name == "stage2_distill":
        artifact_root = root / "artifacts/interaction_labels"
        for name in (
            "interaction_train_manifest.jsonl",
            "interaction_val_manifest.jsonl",
            "group_a_relations_train.jsonl",
            "group_a_relations_val.jsonl",
        ):
            _write(artifact_root / name, '{"fixture":true}\n')
        _write(artifact_root / "pair_prior.json", "{}\n")
        _write(artifact_root / "global_priority.json", "{}\n")
        _write(
            root / "artifacts/metrics/stage2_interaction_summary.csv", "split,count\n"
        )
        _write(root / "reports/INTERACTION_DISTILLATION.md", "# fixture\n")
        decision = {
            "approved": False,
            "stage1_checkpoint_sha256": sha256_file(
                root / "artifacts/checkpoints/stage1/best_ema.pth"
            ),
            "interaction_train_manifest_sha256": sha256_file(
                artifact_root / "interaction_train_manifest.jsonl"
            ),
            "interaction_val_manifest_sha256": sha256_file(
                artifact_root / "interaction_val_manifest.jsonl"
            ),
            "relation_train_sha256": sha256_file(
                artifact_root / "group_a_relations_train.jsonl"
            ),
            "relation_val_sha256": sha256_file(
                artifact_root / "group_a_relations_val.jsonl"
            ),
            "pair_prior_sha256": sha256_file(artifact_root / "pair_prior.json"),
            "global_priority_sha256": sha256_file(
                artifact_root / "global_priority.json"
            ),
            "config_sha256": sha256_file(
                root / "configs/stage2_interaction_distill.yaml"
            ),
            "overall": {"non_ambiguous": 1},
            "warnings": [],
        }
        _write(
            artifact_root / "stage2_decision.json",
            json.dumps(decision, sort_keys=True) + "\n",
        )
    elif command.name == "stage3":
        _write_post_approval_completion(root, "stage3")
    elif command.name == "stage4":
        _write_post_approval_completion(root, "stage4")


def _pause_after_stage2(root: Path) -> tuple[GraphRestoreOrchestrator, RecordingRunner]:
    runner = RecordingRunner(_stage_callback)
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)
    orchestrator.run_integration(100)
    state = orchestrator.run_main_pipeline()
    assert state.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    return orchestrator, runner


def _prepare_failed_stage4_extension(
    root: Path,
) -> tuple[
    GraphRestoreOrchestrator,
    Stage3RevocationAuthorization,
    Stage4ExtensionAuthorization,
]:
    def fail_stage4(command: CommandSpec, cwd: Path) -> None:
        if command.name == "stage3":
            _write_post_approval_completion(cwd, "stage3")
        elif command.name == "stage4":
            _write_raw_post_approval_checkpoint(cwd, "stage4")
        else:
            _stage_callback(command, cwd)

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(fail_stage4, returncodes={"stage4": 9})
    with pytest.raises(ChildCommandError):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )

    stage4_extension = _stage4_extension_authorization(root)
    last = root / "artifacts/checkpoints/stage4/last.pth"
    last_payload = torch.load(last, map_location="cpu", weights_only=False)
    last_payload["provenance"]["stage4_extension"] = (
        stage4_extension.provenance_binding()
    )
    last_payload["provenance"]["runtime"] = {
        "max_steps": STAGE4_EXTENSION_TARGET_STEP,
        "schedule_max_steps": STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS,
    }
    torch.save(last_payload, last)

    revocation = orchestrator.paths.stage3_extension_revocation.resolve()
    _write(revocation, "{}\n")
    finalization = Stage3RevocationAuthorization(
        path=revocation,
        sha256=sha256_file(revocation),
        payload={},
        bindings={},
    )
    return orchestrator, finalization, stage4_extension


def _extension_provenance_binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "cycles": STAGE3_EXTENSION_CYCLES,
        "base_step": STAGE3_EXTENSION_BASE_STEP,
        "target_step": STAGE3_EXTENSION_TARGET_STEP,
        "validation_every_steps": STAGE3_EXTENSION_VALIDATION_EVERY_STEPS,
        "validation_steps": list(STAGE3_EXTENSION_VALIDATION_STEPS),
        "schedule_horizon_steps": STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS,
        "min_lr": STAGE3_EXTENSION_MIN_LR,
        "lr_policy": STAGE3_EXTENSION_LR_POLICY,
    }


def _prepare_failed_stage3_extension(
    root: Path,
) -> tuple[
    GraphRestoreOrchestrator,
    RecordingRunner,
    Path,
    dict[str, object],
    bytes,
]:
    orchestrator, _ = _pause_after_stage2(root)

    def fail_at_stage3(command: CommandSpec, cwd: Path) -> None:
        if command.name != "stage3":
            _stage_callback(command, cwd)
            return
        approval_sha = sha256_file(cwd / "artifacts/approvals/STAGE3_APPROVED.json")
        directory = cwd / "artifacts/checkpoints/stage3"
        _write_checkpoint(
            directory / "last.pth",
            stage="stage3",
            model_role="raw_training_state",
            resumable=True,
            approval_sha256=approval_sha,
            step=STAGE3_EXTENSION_BASE_STEP,
        )
        _write_checkpoint(
            directory / "best_ema.pth",
            stage="stage3",
            model_role="ema_selection",
            resumable=False,
            approval_sha256=approval_sha,
            step=STAGE3_EXTENSION_BASE_STEP,
        )
        approval = load_json(cwd / "artifacts/approvals/STAGE3_APPROVED.json")
        config_binding = approval["bindings"]["config_stage3"]
        _write(
            directory / "run_contract.json",
            json.dumps(
                {
                    "schema_version": "graphrestore-stage3-runtime-v1",
                    "provenance": {
                        "config_sha256": config_binding["sha256"],
                        "stage3_approval": {"sha256": approval_sha},
                        "runtime": {"max_steps": STAGE3_EXTENSION_BASE_STEP},
                    },
                },
                sort_keys=True,
            )
            + "\n",
        )

    failing_runner = RecordingRunner(fail_at_stage3, returncodes={"stage3": 9})
    orchestrator.runner = failing_runner
    with pytest.raises(ChildCommandError):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )

    approval_path = orchestrator.paths.approval_granted
    approval_before = approval_path.read_bytes()
    approval = load_json(approval_path)
    approval_required = orchestrator.paths.approval_required
    config_binding = approval["bindings"]["config_stage3"]
    stage3_dir = root / "artifacts/checkpoints/stage3"
    backup_dir = root / STAGE3_EXTENSION_BACKUP_DIRECTORY
    sources = {
        "run_contract.json": stage3_dir / "run_contract.json",
        "last.pth": stage3_dir / "last.pth",
        "best_ema.pth": stage3_dir / "best_ema.pth",
    }
    for name, source in sources.items():
        backup = backup_dir / name
        _write(backup, source.read_bytes())
        backup.chmod(0o444)

    extension_path = orchestrator.paths.stage3_extension_approval
    extension_payload = {
        "schema_version": STAGE3_EXTENSION_APPROVAL_SCHEMA,
        "kind": "stage3_extension_approval",
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "approved": True,
        "cycles": STAGE3_EXTENSION_CYCLES,
        "base_step": STAGE3_EXTENSION_BASE_STEP,
        "target_step": STAGE3_EXTENSION_TARGET_STEP,
        "validation_every_steps": STAGE3_EXTENSION_VALIDATION_EVERY_STEPS,
        "validation_steps": list(STAGE3_EXTENSION_VALIDATION_STEPS),
        "schedule_horizon_steps": STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS,
        "min_lr": STAGE3_EXTENSION_MIN_LR,
        "lr_policy": STAGE3_EXTENSION_LR_POLICY,
        "formal_mio100_authorized": False,
        "authorized_pipeline": ["stage3_extension", "stage4"],
        "base_stage3_approval": {
            "path": str(approval_path.resolve()),
            "sha256": sha256_file(approval_path),
        },
        "base_approval_required": {
            "path": str(approval_required.resolve()),
            "sha256": sha256_file(approval_required),
        },
        "base_stage3_config": dict(config_binding),
        "pre_extension_run_contract": {
            "path": str((backup_dir / "run_contract.json").resolve()),
            "sha256": sha256_file(backup_dir / "run_contract.json"),
        },
        "pre_extension_last_checkpoint": {
            "path": str((backup_dir / "last.pth").resolve()),
            "sha256": sha256_file(backup_dir / "last.pth"),
        },
        "pre_extension_best_checkpoint": {
            "path": str((backup_dir / "best_ema.pth").resolve()),
            "sha256": sha256_file(backup_dir / "best_ema.pth"),
        },
    }
    _write(extension_path, json.dumps(extension_payload, sort_keys=True) + "\n")
    extension_binding = _extension_provenance_binding(extension_path)

    approval_sha = sha256_file(approval_path)
    _write_checkpoint(
        stage3_dir / "last.pth",
        stage="stage3",
        model_role="raw_training_state",
        resumable=True,
        approval_sha256=approval_sha,
        step=STAGE3_EXTENSION_BASE_STEP,
        extension_authorization=extension_binding,
    )
    _write_checkpoint(
        stage3_dir / "best_ema.pth",
        stage="stage3",
        model_role="ema_selection",
        resumable=False,
        approval_sha256=approval_sha,
        step=STAGE3_EXTENSION_BASE_STEP,
        extension_authorization=extension_binding,
    )
    live_contract = load_json(stage3_dir / "run_contract.json")
    migrated_last = torch.load(
        stage3_dir / "last.pth", map_location="cpu", weights_only=False
    )
    live_contract["provenance"] = migrated_last["provenance"]
    _write(
        stage3_dir / "run_contract.json",
        json.dumps(live_contract, sort_keys=True) + "\n",
    )
    return (
        orchestrator,
        failing_runner,
        extension_path,
        extension_binding,
        approval_before,
    )


def test_preflight_integration_pause_approval_and_stage3_stage4(tmp_path: Path) -> None:
    root = _project(tmp_path)
    runner = RecordingRunner(_stage_callback)
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)

    ready = orchestrator.run_integration(100)
    assert ready.status == PipelineStatus.READY_FOR_MAIN.value
    assert ready.integration_steps == 100
    assert runner.commands[-1].name == "integration_100_steps"
    assert "--integration_steps" in runner.commands[-1].argv
    assert runner.commands[-1].argv[-2:] == (
        "--output_dir",
        "artifacts/integration/stage0_100_steps",
    )

    paused = orchestrator.run_main_pipeline()
    assert paused.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    assert paused.gpu == "released"
    assert (root / "RUNNING_STATUS.md").read_text(encoding="utf-8").splitlines()[
        :5
    ] == [
        "status: PAUSED_AFTER_STAGE2",
        "GPU: released",
        "Stage3: NOT STARTED",
        "waiting_for: user approval",
        (
            "resume_command: python scripts/orchestrate.py --approve_stage3 "
            "--resume_from_stage3"
        ),
    ]
    assert [item.name for item in runner.commands[-4:]] == [
        "stage0",
        "stage1",
        "effect_profiles",
        "stage2_distill",
    ]
    assert orchestrator.paths.approval_required.is_file()
    assert not orchestrator.paths.approval_granted.exists()
    required = load_json(orchestrator.paths.approval_required)
    assert required["approved"] is False
    assert required["stage2_decision"]["sha256"] == sha256_file(
        orchestrator.paths.stage2_decision
    )

    command_count_at_pause = len(runner.commands)
    complete = orchestrator.approve_and_resume_stage3(
        approve_stage3=True,
        resume_from_stage3=True,
    )
    assert [item.name for item in runner.commands[command_count_at_pause:]] == [
        "stage3",
        "stage4",
    ]
    assert complete.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )
    assert complete.gpu == "released"
    assert (
        complete.next_command == "await_explicit_user_authorization_for_formal_mio100"
    )
    approval = load_json(orchestrator.paths.approval_granted)
    assert approval["approved"] is True
    assert approval["approved_utc"].endswith("Z")
    assert approval["stage2_decision_sha256"] == paused.stage2_decision_sha256
    assert approval["approval_required_sha256"] == sha256_file(
        orchestrator.paths.approval_required
    )
    assert approval["scientific_adjudications"] == {"D-017": dict(D017_ACCEPTANCE)}
    assert approval["authorized_pipeline"] == ["stage3", "stage4"]
    assert approval["formal_mio100_authorized"] is False
    final_status = (root / "RUNNING_STATUS.md").read_text(encoding="utf-8")
    assert f"stage0_group_a_psnr_anchor: {STAGE0_GROUP_A_PSNR_ANCHOR!r}" in final_status
    assert f"stage0_group_a_ssim_anchor: {STAGE0_GROUP_A_SSIM_ANCHOR!r}" in final_status
    assert "SSIM_RETENTION_RISK: true" in final_status
    assert "SSIM_RETENTION_RISK_NOTE:" in final_status


def test_stage3_refuses_missing_flag_or_changed_hash(tmp_path: Path) -> None:
    root = _project(tmp_path)
    orchestrator, runner = _pause_after_stage2(root)

    with pytest.raises(ApprovalError, match="both explicit flags"):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=False,
        )

    _write(root / "configs/stage2_interaction_distill.yaml", "fixture: changed\n")
    command_count = len(runner.commands)
    with pytest.raises(ApprovalError, match="hashes changed"):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    assert len(runner.commands) == command_count
    assert not orchestrator.paths.approval_granted.exists()
    state = orchestrator.load_state()
    assert state.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    assert state.gpu == "released"
    assert "hashes changed" in (state.last_error or "")


def test_stage3_refuses_approval_required_protocol_drift(tmp_path: Path) -> None:
    root = _project(tmp_path)
    orchestrator, runner = _pause_after_stage2(root)
    required = dict(load_json(orchestrator.paths.approval_required))
    required["protocol_id"] = "graphrestore-v7.1-forged"
    _write(
        orchestrator.paths.approval_required,
        json.dumps(required, sort_keys=True) + "\n",
    )
    command_count = len(runner.commands)

    with pytest.raises(ApprovalError, match="protocol_id mismatch"):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )

    assert len(runner.commands) == command_count
    assert not orchestrator.paths.approval_granted.exists()


@pytest.mark.parametrize(
    "field",
    (
        "relation_train_sha256",
        "relation_val_sha256",
        "pair_prior_sha256",
        "global_priority_sha256",
        "config_sha256",
    ),
)
def test_stage2_pause_rejects_internal_decision_hash_drift(
    tmp_path: Path,
    field: str,
) -> None:
    root = _project(tmp_path)
    orchestrator, _ = _pause_after_stage2(root)
    decision = load_json(orchestrator.paths.stage2_decision)
    assert isinstance(decision, dict)
    decision[field] = "0" * 64
    _write(
        orchestrator.paths.stage2_decision,
        json.dumps(decision, sort_keys=True) + "\n",
    )
    with pytest.raises(OrchestrationError, match=field):
        orchestrator._create_approval_required()


def test_missing_success_artifact_marks_pipeline_failed(tmp_path: Path) -> None:
    root = _project(tmp_path)

    def integration_only(command: CommandSpec, cwd: Path) -> None:
        if command.name == "integration_100_steps":
            _stage_callback(command, cwd)

    orchestrator = GraphRestoreOrchestrator(
        root, runner=RecordingRunner(integration_only)
    )
    orchestrator.run_integration(100)

    with pytest.raises(OrchestrationError, match="required outputs are missing"):
        orchestrator.run_main_pipeline()
    state = orchestrator.load_state()
    assert state.status == PipelineStatus.FAILED.value
    assert state.gpu == "released"


def test_preflight_hard_gates_and_plan_exclude_formal_mio100(tmp_path: Path) -> None:
    orchestrator = GraphRestoreOrchestrator(
        _project(tmp_path), runner=RecordingRunner()
    )
    mandatory = next(
        command
        for command in orchestrator.preflight_commands()
        if command.name == "mandatory_pytests"
    )
    assert mandatory.argv[-1] == "tests"
    for required_test in (
        "tests/test_online_canonical_manifest_semantics.py",
        "tests/test_ambiguous_relation_partial_label.py",
        "tests/test_compile_once_per_sample.py",
        "tests/test_checkpoint_resume.py",
    ):
        assert (Path(__file__).resolve().parents[1] / required_test).is_file()
    preflight_names = {command.name for command in orchestrator.preflight_commands()}
    assert "audit_degradation_parity" in preflight_names
    assert "probe_validation_vram" in preflight_names
    assert "profile_stage0_compile" in preflight_names

    plan = command_plan(orchestrator)
    assert plan["formal_mio100_in_automatic_pipeline"] is False
    all_argv = json.dumps(plan)
    assert "eval_mio100" not in all_argv
    assert "mio100_test" not in all_argv


def test_main_requires_exact_integration_and_integration_requires_100(
    tmp_path: Path,
) -> None:
    orchestrator = GraphRestoreOrchestrator(
        _project(tmp_path), runner=RecordingRunner()
    )
    with pytest.raises(OrchestrationError, match="exactly 100"):
        orchestrator.run_integration(99)
    with pytest.raises(OrchestrationError, match="100-step integration"):
        orchestrator.run_main_pipeline()


def test_explicit_main_resume_uses_last_checkpoint_and_skips_completed(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)

    def integration_only(command: CommandSpec, cwd: Path) -> None:
        if command.name == "integration_100_steps":
            _stage_callback(command, cwd)

    runner = RecordingRunner(
        callback=integration_only,
        returncodes={"stage0": 9},
    )
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)
    orchestrator.run_integration(100)
    with pytest.raises(ChildCommandError):
        orchestrator.run_main_pipeline()
    assert orchestrator.load_state().status == PipelineStatus.FAILED.value
    _write(root / "artifacts/checkpoints/stage0/last.pth", b"last")
    runner.returncodes["stage0"] = 0
    runner.callback = _stage_callback
    paused = orchestrator.resume_main_pipeline()
    assert paused.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    resumed_stage0 = next(
        command for command in runner.commands if "--resume" in command.argv
    )
    assert resumed_stage0.name == "stage0"
    assert resumed_stage0.argv[-2:] == (
        "--resume",
        str(root / "artifacts/checkpoints/stage0/last.pth"),
    )


def test_explicit_main_resume_recovers_proven_stale_running_state(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    runner = RecordingRunner(_stage_callback)
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)
    orchestrator.run_integration(100)
    state = orchestrator.load_state()
    stage0 = orchestrator.main_stage_commands()[0][1]
    state.status = PipelineStatus.STAGE0_RUNNING.value
    state.current_stage = "STAGE0"
    state.last_command = list(stage0.argv)
    state.gpu = "owned_by_child_process"
    orchestrator._persist(state)
    _write(root / "artifacts/checkpoints/stage0/last.pth", b"last")

    paused = orchestrator.resume_main_pipeline()
    assert paused.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    resumed = next(command for command in runner.commands if "--resume" in command.argv)
    assert resumed.name == "stage0"


def test_stale_running_recovery_refuses_an_exact_live_child(tmp_path: Path) -> None:
    root = _project(tmp_path)
    orchestrator = GraphRestoreOrchestrator(
        root,
        runner=RecordingRunner(_stage_callback),
    )
    orchestrator.run_integration(100)
    command = (sys.executable, "-c", "import time; time.sleep(60)")
    process = subprocess.Popen(command, cwd=root)
    try:
        state = orchestrator.load_state()
        state.status = PipelineStatus.STAGE0_RUNNING.value
        state.current_stage = "STAGE0"
        state.last_command = list(command)
        state.gpu = "owned_by_child_process"
        orchestrator._persist(state)
        with pytest.raises(OrchestrationError, match="still live"):
            orchestrator.resume_main_pipeline()
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_post_approval_resume_skips_complete_stage3_and_resumes_raw_stage4(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)

    def fail_stage4(command: CommandSpec, cwd: Path) -> None:
        if command.name == "stage3":
            _write_post_approval_completion(cwd, "stage3")
        elif command.name == "stage4":
            _write_raw_post_approval_checkpoint(cwd, "stage4")
        else:
            _stage_callback(command, cwd)

    orchestrator, pause_runner = _pause_after_stage2(root)
    runner = RecordingRunner(fail_stage4, returncodes={"stage4": 9})
    orchestrator.runner = runner
    with pytest.raises(ChildCommandError):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    failed = orchestrator.load_state()
    assert failed.status == PipelineStatus.FAILED.value
    assert failed.next_command == (
        "python scripts/orchestrate.py --resume_post_approval_pipeline"
    )
    assert "stage3" in failed.completed
    assert "stage4" not in failed.completed
    approval_before = orchestrator.paths.approval_granted.read_bytes()
    command_count = len(runner.commands)

    runner.returncodes["stage4"] = 0
    runner.callback = _stage_callback
    complete = orchestrator.resume_post_approval_pipeline()
    resumed = runner.commands[command_count:]
    assert [command.name for command in resumed] == ["stage4"]
    assert resumed[0].argv[-2:] == (
        "--resume",
        str(root / "artifacts/checkpoints/stage4/last.pth"),
    )
    assert orchestrator.paths.approval_granted.read_bytes() == approval_before
    assert complete.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )
    assert pause_runner.commands[-1].name == "stage2_distill"


def test_stage3_extension_runs_exact_three_cycles_before_stage4(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (
        orchestrator,
        _,
        extension_path,
        extension_binding,
        approval_before,
    ) = _prepare_failed_stage3_extension(root)

    def complete_extension(command: CommandSpec, cwd: Path) -> None:
        _write_post_approval_completion(
            cwd,
            command.name,
            extension_authorization=extension_binding,
        )

    runner = RecordingRunner(complete_extension)
    orchestrator.runner = runner
    state = orchestrator.resume_post_approval_pipeline(
        stage3_extension_authorization=extension_path.resolve()
    )

    assert [command.name for command in runner.commands] == ["stage3", "stage4"]
    assert runner.commands[0].argv[-4:] == (
        "--resume",
        str(root / "artifacts/checkpoints/stage3/last.pth"),
        "--extension_authorization",
        str(extension_path.resolve()),
    )
    assert "--extension_authorization" not in runner.commands[1].argv
    assert state.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )
    assert orchestrator.paths.approval_granted.read_bytes() == approval_before
    assert load_json(root / "artifacts/checkpoints/stage3/complete.json")[
        "extension_validation_steps"
    ] == list(STAGE3_EXTENSION_VALIDATION_STEPS)


def test_stage3_finalize_only_supersedes_failed_extension_and_then_runs_stage4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    orchestrator, _, _, _, _ = _prepare_failed_stage3_extension(root)
    finalization_path = orchestrator.paths.stage3_extension_revocation.resolve()
    _write(finalization_path, "{}\n")
    authorization = Stage3RevocationAuthorization(
        path=finalization_path,
        sha256=sha256_file(finalization_path),
        payload={},
        bindings={},
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_and_verify_stage3_finalization_authorization",
        lambda _state, _path: authorization,
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_post_approval_completion",
        lambda stage, **_kwargs: root / f"artifacts/checkpoints/{stage}/best_ema.pth",
    )
    runner = RecordingRunner()
    orchestrator.runner = runner

    state = orchestrator.resume_post_approval_pipeline(
        stage3_finalization_authorization=finalization_path,
    )

    assert [command.name for command in runner.commands] == ["stage3", "stage4"]
    assert runner.commands[0].argv == (
        orchestrator.python,
        "scripts/finalize_stage3.py",
        "--config",
        "configs/stage3_planner.yaml",
        "--finalization_authorization",
        str(finalization_path),
    )
    assert "--resume" not in runner.commands[0].argv
    assert "train_stage3_planner.py" not in runner.commands[0].argv
    assert runner.commands[1] == orchestrator.post_approval_commands()[1][1]
    assert state.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )


def test_stage3_finalize_only_failure_preserves_exact_authorized_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    orchestrator, _, _, _, _ = _prepare_failed_stage3_extension(root)
    finalization_path = orchestrator.paths.stage3_extension_revocation.resolve()
    _write(finalization_path, "{}\n")
    authorization = Stage3RevocationAuthorization(
        path=finalization_path,
        sha256=sha256_file(finalization_path),
        payload={},
        bindings={},
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_and_verify_stage3_finalization_authorization",
        lambda _state, _path: authorization,
    )
    orchestrator.runner = RecordingRunner(returncodes={"stage3": 19})

    with pytest.raises(ChildCommandError) as caught:
        orchestrator.resume_post_approval_pipeline(
            stage3_finalization_authorization=finalization_path,
        )
    assert caught.value.exit_code == 19
    state = orchestrator.load_state()
    assert state.next_command == (
        "python scripts/orchestrate.py --resume_post_approval_pipeline "
        f"--stage3_finalization_authorization {finalization_path}"
    )
    assert state.last_command is not None
    assert "scripts/finalize_stage3.py" in state.last_command


def test_stage4_extension_resume_coexists_with_stage3_finalization_and_completes_48k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    orchestrator, finalization, stage4_extension = _prepare_failed_stage4_extension(
        root
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_and_verify_stage3_finalization_authorization",
        lambda _state, _path: finalization,
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_and_verify_stage4_extension_authorization",
        lambda _state, _path: stage4_extension,
    )
    original_verify = orchestrator._verify_post_approval_completion

    def verify_completion(stage: str, **kwargs: object) -> Path:
        if stage == "stage3":
            return root / "artifacts/checkpoints/stage3/best_ema.pth"
        return original_verify(stage, **kwargs)

    monkeypatch.setattr(
        orchestrator, "_verify_post_approval_completion", verify_completion
    )

    def complete_extension(command: CommandSpec, cwd: Path) -> None:
        assert command.name == "stage4"
        _write_post_approval_completion(
            cwd,
            "stage4",
            stage4_extension_authorization=stage4_extension,
        )

    runner = RecordingRunner(complete_extension)
    orchestrator.runner = runner
    state = orchestrator.resume_post_approval_pipeline(
        stage3_finalization_authorization=finalization.path,
        stage4_extension_authorization=stage4_extension.gate_path,
    )

    assert [command.name for command in runner.commands] == ["stage4"]
    assert runner.commands[0].argv[-4:] == (
        "--resume",
        str(root / "artifacts/checkpoints/stage4/last.pth"),
        "--extension_authorization",
        str(stage4_extension.gate_path),
    )
    assert state.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )
    complete = load_json(root / "artifacts/checkpoints/stage4/complete.json")
    assert complete["step"] == STAGE4_EXTENSION_TARGET_STEP
    assert complete["stage4_extension_validation_steps"] == list(
        STAGE4_EXTENSION_VALIDATION_STEPS
    )
    assert complete["further_extension_authorized"] is False


def test_stage4_extension_failure_preserves_both_canonical_resume_authorizations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    orchestrator, finalization, stage4_extension = _prepare_failed_stage4_extension(
        root
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_and_verify_stage3_finalization_authorization",
        lambda _state, _path: finalization,
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_and_verify_stage4_extension_authorization",
        lambda _state, _path: stage4_extension,
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_post_approval_completion",
        lambda stage, **_kwargs: root / f"artifacts/checkpoints/{stage}/best_ema.pth",
    )
    orchestrator.runner = RecordingRunner(returncodes={"stage4": 23})

    with pytest.raises(ChildCommandError) as caught:
        orchestrator.resume_post_approval_pipeline(
            stage3_finalization_authorization=finalization.path,
            stage4_extension_authorization=stage4_extension.gate_path,
        )

    assert caught.value.exit_code == 23
    failed = orchestrator.load_state()
    assert failed.next_command == (
        "python scripts/orchestrate.py --resume_post_approval_pipeline "
        f"--stage3_finalization_authorization {finalization.path} "
        f"--stage4_extension_authorization {stage4_extension.gate_path}"
    )
    assert tuple(failed.last_command or ())[-2:] == (
        "--extension_authorization",
        str(stage4_extension.gate_path),
    )


def test_stage3_revocation_tombstone_cannot_be_bypassed_when_dangling(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    orchestrator, _, _, _, _ = _prepare_failed_stage3_extension(root)
    tombstone = orchestrator.paths.stage3_extension_revocation
    tombstone.parent.mkdir(parents=True, exist_ok=True)
    tombstone.symlink_to(tombstone.parent / "missing-revocation-target.json")

    with pytest.raises(ApprovalError, match="permanently disables"):
        orchestrator.resume_post_approval_pipeline()
    assert orchestrator.load_state().next_command == (
        "python scripts/orchestrate.py --resume_post_approval_pipeline "
        f"--stage3_finalization_authorization {tombstone}"
    )


@pytest.mark.parametrize(
    "mutation",
    ("cycles", "cycles_bool", "config_sha", "backup_mode", "formal_mio100"),
)
def test_stage3_extension_refuses_stale_or_expanded_authorization_before_child(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _project(tmp_path)
    orchestrator, _, extension_path, _, approval_before = (
        _prepare_failed_stage3_extension(root)
    )
    payload = load_json(extension_path)
    if mutation == "cycles":
        payload["cycles"] = 4
    elif mutation == "cycles_bool":
        payload["cycles"] = True
    elif mutation == "config_sha":
        payload["base_stage3_config"]["sha256"] = "0" * 64
    elif mutation == "formal_mio100":
        payload["formal_mio100_authorized"] = True
    else:
        backup = Path(payload["pre_extension_last_checkpoint"]["path"])
        backup.chmod(0o644)
    if mutation != "backup_mode":
        _write(extension_path, json.dumps(payload, sort_keys=True) + "\n")

    runner = RecordingRunner()
    orchestrator.runner = runner
    with pytest.raises(ApprovalError):
        orchestrator.resume_post_approval_pipeline(
            stage3_extension_authorization=extension_path.resolve()
        )

    assert runner.commands == []
    assert orchestrator.paths.approval_granted.read_bytes() == approval_before
    state = orchestrator.load_state()
    assert state.status == PipelineStatus.FAILED.value
    assert "--stage3_extension_authorization" in state.next_command
    assert str(extension_path.resolve()) in state.next_command


@pytest.mark.parametrize(
    "mutation",
    (
        "complete_step_12000",
        "missing_16000_log",
        "missing_selected_validation",
        "missing_extension_provenance",
        "stale_calibration_binding",
    ),
)
def test_stage3_extension_completion_gate_blocks_stage4(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _project(tmp_path)
    orchestrator, _, extension_path, extension_binding, _ = (
        _prepare_failed_stage3_extension(root)
    )

    def corrupt_extension_completion(command: CommandSpec, cwd: Path) -> None:
        assert command.name == "stage3"
        _write_post_approval_completion(
            cwd,
            "stage3",
            extension_authorization=extension_binding,
        )
        directory = cwd / "artifacts/checkpoints/stage3"
        if mutation == "complete_step_12000":
            complete = load_json(directory / "complete.json")
            complete["step"] = STAGE3_EXTENSION_BASE_STEP
            _write(
                directory / "complete.json",
                json.dumps(complete, sort_keys=True) + "\n",
            )
        elif mutation == "missing_16000_log":
            rows = [
                row
                for row in (directory / "train.jsonl").read_text().splitlines()
                if json.loads(row)["step"] != 16_000
            ]
            _write(directory / "train.jsonl", "\n".join(rows) + "\n")
        elif mutation == "missing_selected_validation":
            (directory / "selected_validation.json").unlink()
        elif mutation == "missing_extension_provenance":
            approval_sha = sha256_file(cwd / "artifacts/approvals/STAGE3_APPROVED.json")
            _write_checkpoint(
                directory / "last.pth",
                stage="stage3",
                model_role="raw_training_state",
                resumable=True,
                approval_sha256=approval_sha,
                step=STAGE3_EXTENSION_TARGET_STEP,
            )
        else:
            thresholds = load_json(cwd / "artifacts/planner_thresholds.json")
            thresholds["stage3_extension_authorization_sha256"] = "0" * 64
            _write(
                cwd / "artifacts/planner_thresholds.json",
                json.dumps(thresholds, sort_keys=True) + "\n",
            )

    runner = RecordingRunner(corrupt_extension_completion)
    orchestrator.runner = runner
    with pytest.raises(OrchestrationError):
        orchestrator.resume_post_approval_pipeline(
            stage3_extension_authorization=extension_path.resolve()
        )

    assert [command.name for command in runner.commands] == ["stage3"]
    state = orchestrator.load_state()
    assert "stage3" not in state.completed
    assert "stage4" not in state.completed
    assert state.status == PipelineStatus.FAILED.value
    assert "--stage3_extension_authorization" in state.next_command


def test_stage3_extension_failure_and_plain_retry_preserve_exact_resume_command(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    orchestrator, _, extension_path, _, _ = _prepare_failed_stage3_extension(root)
    runner = RecordingRunner(returncodes={"stage3": 7})
    orchestrator.runner = runner

    with pytest.raises(ChildCommandError):
        orchestrator.resume_post_approval_pipeline(
            stage3_extension_authorization=extension_path.resolve()
        )
    failed = orchestrator.load_state()
    expected_next = (
        "python scripts/orchestrate.py --resume_post_approval_pipeline "
        f"--stage3_extension_authorization {extension_path.resolve()}"
    )
    assert failed.next_command == expected_next
    assert tuple(failed.last_command or ())[-2:] == (
        "--extension_authorization",
        str(extension_path.resolve()),
    )

    command_count = len(runner.commands)
    with pytest.raises(OrchestrationError, match="not an exact Stage3/Stage4 child"):
        orchestrator.resume_post_approval_pipeline()
    assert len(runner.commands) == command_count
    assert orchestrator.load_state().next_command == expected_next


@pytest.mark.parametrize(
    ("write_last", "model_role", "resumable", "match"),
    (
        (False, "raw_training_state", True, "missing raw last.pth"),
        (True, "ema_selection", False, "checkpoint role/provenance mismatch"),
    ),
)
def test_post_approval_resume_refuses_missing_or_ema_only_last(
    tmp_path: Path,
    write_last: bool,
    model_role: str,
    resumable: bool,
    match: str,
) -> None:
    root = _project(tmp_path)

    def fail_stage3(command: CommandSpec, cwd: Path) -> None:
        if command.name == "stage3" and write_last:
            _write_raw_post_approval_checkpoint(
                cwd,
                "stage3",
                model_role=model_role,
                resumable=resumable,
            )
        elif command.name != "stage3":
            _stage_callback(command, cwd)

    orchestrator, _ = _pause_after_stage2(root)
    runner = RecordingRunner(fail_stage3, returncodes={"stage3": 7})
    orchestrator.runner = runner
    with pytest.raises(ChildCommandError):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    commands_before = len(runner.commands)
    approval_before = orchestrator.paths.approval_granted.read_bytes()
    with pytest.raises(OrchestrationError, match=match):
        orchestrator.resume_post_approval_pipeline()
    assert len(runner.commands) == commands_before
    assert orchestrator.paths.approval_granted.read_bytes() == approval_before
    state = orchestrator.load_state()
    assert state.status == PipelineStatus.FAILED.value
    assert state.next_command == (
        "python scripts/orchestrate.py --resume_post_approval_pipeline"
    )


def test_post_approval_resume_revalidates_approval_and_all_frozen_bindings(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)

    def fail_stage4(command: CommandSpec, cwd: Path) -> None:
        if command.name == "stage3":
            _write_post_approval_completion(cwd, "stage3")
        elif command.name == "stage4":
            _write_raw_post_approval_checkpoint(cwd, "stage4")
        else:
            _stage_callback(command, cwd)

    orchestrator, _ = _pause_after_stage2(root)
    runner = RecordingRunner(fail_stage4, returncodes={"stage4": 8})
    orchestrator.runner = runner
    with pytest.raises(ChildCommandError):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    approval_before = orchestrator.paths.approval_granted.read_bytes()
    commands_before = len(runner.commands)
    _write(root / "configs/stage4_graphrestore_e2e.yaml", "fixture: drifted\n")
    with pytest.raises(ApprovalError, match="hashes changed"):
        orchestrator.resume_post_approval_pipeline()
    assert len(runner.commands) == commands_before
    assert orchestrator.paths.approval_granted.read_bytes() == approval_before


def test_post_approval_resume_revalidates_approval_required_protocol(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)

    def fail_stage4(command: CommandSpec, cwd: Path) -> None:
        if command.name == "stage3":
            _write_post_approval_completion(cwd, "stage3")
        elif command.name == "stage4":
            _write_raw_post_approval_checkpoint(cwd, "stage4")
        else:
            _stage_callback(command, cwd)

    orchestrator, _ = _pause_after_stage2(root)
    runner = RecordingRunner(fail_stage4, returncodes={"stage4": 8})
    orchestrator.runner = runner
    with pytest.raises(ChildCommandError):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    approval_before = orchestrator.paths.approval_granted.read_bytes()
    commands_before = len(runner.commands)
    required = dict(load_json(orchestrator.paths.approval_required))
    required["protocol_id"] = "graphrestore-v7.1-forged"
    _write(
        orchestrator.paths.approval_required,
        json.dumps(required, sort_keys=True) + "\n",
    )

    with pytest.raises(ApprovalError, match="protocol_id mismatch"):
        orchestrator.resume_post_approval_pipeline()

    assert len(runner.commands) == commands_before
    assert orchestrator.paths.approval_granted.read_bytes() == approval_before


def test_post_approval_resume_rejects_d017_or_authorization_scope_drift(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)

    def fail_stage3(command: CommandSpec, cwd: Path) -> None:
        if command.name == "stage3":
            _write_raw_post_approval_checkpoint(cwd, "stage3")
        else:
            _stage_callback(command, cwd)

    orchestrator, _ = _pause_after_stage2(root)
    runner = RecordingRunner(fail_stage3, returncodes={"stage3": 8})
    orchestrator.runner = runner
    with pytest.raises(ChildCommandError):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    approval = dict(load_json(orchestrator.paths.approval_granted))
    approval["scientific_adjudications"] = {
        "D-017": {**dict(D017_ACCEPTANCE), "parameter_wise_minmax_claimed": True}
    }
    approval["authorized_pipeline"] = ["stage3", "stage4", "formal_mio100"]
    approval["formal_mio100_authorized"] = True
    _write(
        orchestrator.paths.approval_granted,
        json.dumps(approval, sort_keys=True) + "\n",
    )
    state = orchestrator.load_state()
    state.stage3_approval_sha256 = sha256_file(orchestrator.paths.approval_granted)
    orchestrator._persist(state)
    command_count = len(runner.commands)

    with pytest.raises(ApprovalError, match="persisted Stage3 approval/hash"):
        orchestrator.resume_post_approval_pipeline()

    assert len(runner.commands) == command_count


def test_post_approval_output_failure_removes_premature_completed_marker(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)

    def stage3_without_report(command: CommandSpec, cwd: Path) -> None:
        if command.name != "stage3":
            _stage_callback(command, cwd)
            return
        _write_post_approval_completion(cwd, "stage3")
        (cwd / "reports/STAGE3_PLANNER_GUARD.md").unlink()

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(stage3_without_report)
    with pytest.raises(OrchestrationError, match="required outputs are missing"):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    state = orchestrator.load_state()
    assert "stage3" not in state.completed
    assert state.status == PipelineStatus.FAILED.value
    assert state.next_command == (
        "python scripts/orchestrate.py --resume_post_approval_pipeline"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing_json", "required outputs are missing"),
        ("forged_json", "diagnostics.optimizer_updates"),
        ("missing_mode", "diagnostics.compiler_modes"),
        ("complete_hash_drift", "diagnostics_json_sha256"),
    ),
)
def test_stage4_completion_rejects_missing_forged_or_hash_drifted_diagnostics(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = _project(tmp_path)

    def corrupt_stage4_diagnostics(command: CommandSpec, cwd: Path) -> None:
        _stage_callback(command, cwd)
        if command.name != "stage4":
            return
        diagnostics_path = cwd / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.json"
        complete_path = cwd / "artifacts/checkpoints/stage4/complete.json"
        if mutation == "missing_json":
            diagnostics_path.unlink()
            return
        complete = dict(load_json(complete_path))
        if mutation in {"forged_json", "missing_mode"}:
            diagnostics = dict(load_json(diagnostics_path))
            if mutation == "forged_json":
                diagnostics["optimizer_updates"] = 1
            else:
                compiler_modes = dict(diagnostics["compiler_modes"])
                compiler_modes.pop("parallel_only")
                diagnostics["compiler_modes"] = compiler_modes
            _write(
                diagnostics_path,
                json.dumps(diagnostics, sort_keys=True) + "\n",
            )
            complete["diagnostics_json_sha256"] = sha256_file(diagnostics_path)
        else:
            complete["diagnostics_json_sha256"] = "0" * 64
        _write(complete_path, json.dumps(complete, sort_keys=True) + "\n")

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(corrupt_stage4_diagnostics)
    with pytest.raises(OrchestrationError, match=match):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    state = orchestrator.load_state()
    assert "stage3" in state.completed
    assert "stage4" not in state.completed
    assert state.status == PipelineStatus.FAILED.value
    assert state.next_command == (
        "python scripts/orchestrate.py --resume_post_approval_pipeline"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("anchor", "stage0_group_a_ssim_anchor"),
        ("delta", "selected_delta_group_a_ssim_vs_stage0"),
        ("risk", "SSIM_RETENTION_RISK"),
        ("validation_hash", "validation_sha256"),
        ("missing_validation", "required outputs are missing"),
        ("report_risk", "report.retention"),
    ),
)
def test_stage4_completion_rejects_retention_contract_drift(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = _project(tmp_path)

    def corrupt_retention(command: CommandSpec, cwd: Path) -> None:
        _stage_callback(command, cwd)
        if command.name != "stage4":
            return
        directory = cwd / "artifacts/checkpoints/stage4"
        complete_path = directory / "complete.json"
        complete = dict(load_json(complete_path))
        if mutation == "anchor":
            complete["stage0_group_a_ssim_anchor"] = 0.0
        elif mutation == "delta":
            complete["selected_delta_group_a_ssim_vs_stage0"] = 0.0
        elif mutation == "risk":
            complete["SSIM_RETENTION_RISK"] = False
        elif mutation == "validation_hash":
            complete["validation_sha256"] = "0" * 64
        elif mutation == "missing_validation":
            (directory / "validation_latest.json").unlink()
            return
        else:
            report = cwd / "reports/STAGE4_E2E.md"
            text = report.read_text(encoding="utf-8").replace(
                "- SSIM_RETENTION_RISK: true\n", ""
            )
            _write(report, text)
            complete["report_sha256"] = sha256_file(report)
        _write(complete_path, json.dumps(complete, sort_keys=True) + "\n")

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(corrupt_retention)
    with pytest.raises(OrchestrationError, match=match):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            "empty_mode",
            "diagnostics.compiler_modes.full_partial_order.single_equal_task_mean",
        ),
        (
            "nan_metric",
            "diagnostics.compiler_modes.full_partial_order.single_equal_task_mean.psnr",
        ),
        (
            "peak_over_limit",
            "diagnostics.compiler_modes.full_partial_order.peak_reserved_fraction",
        ),
        (
            "missing_field",
            "diagnostics.compiler_modes.full_partial_order.image_count",
        ),
    ),
)
def test_stage4_completion_rejects_invalid_per_mode_diagnostics(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = _project(tmp_path)

    def corrupt_mode(command: CommandSpec, cwd: Path) -> None:
        _stage_callback(command, cwd)
        if command.name != "stage4":
            return
        diagnostics_path = cwd / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.json"
        complete_path = cwd / "artifacts/checkpoints/stage4/complete.json"
        diagnostics = dict(load_json(diagnostics_path))
        compiler_modes = dict(diagnostics["compiler_modes"])
        mode = dict(compiler_modes["full_partial_order"])
        if mutation == "empty_mode":
            mode = {}
        elif mutation == "nan_metric":
            aggregate = dict(mode["single_equal_task_mean"])
            aggregate["psnr"] = float("nan")
            mode["single_equal_task_mean"] = aggregate
        elif mutation == "peak_over_limit":
            mode["peak_reserved_fraction"] = 0.9000001
        else:
            mode.pop("image_count")
        compiler_modes["full_partial_order"] = mode
        diagnostics["compiler_modes"] = compiler_modes
        _write(diagnostics_path, json.dumps(diagnostics, sort_keys=True) + "\n")
        complete = dict(load_json(complete_path))
        complete["diagnostics_json_sha256"] = sha256_file(diagnostics_path)
        _write(complete_path, json.dumps(complete, sort_keys=True) + "\n")

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(corrupt_mode)
    with pytest.raises(OrchestrationError, match=match):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    state = orchestrator.load_state()
    assert "stage3" in state.completed
    assert "stage4" not in state.completed


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing_sha", "diagnostics_report.selected_best_ema_sha256"),
        ("missing_mode", "diagnostics_report.mode.all_one"),
    ),
)
def test_stage4_completion_rejects_semantically_empty_diagnostics_report(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = _project(tmp_path)

    def corrupt_report(command: CommandSpec, cwd: Path) -> None:
        _stage_callback(command, cwd)
        if command.name != "stage4":
            return
        report = cwd / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.md"
        text = report.read_text(encoding="utf-8")
        if mutation == "missing_sha":
            best = cwd / "artifacts/checkpoints/stage4/best_ema.pth"
            text = text.replace(sha256_file(best), "0" * 64)
        else:
            text = text.replace("- all_one\n", "")
        _write(report, text)
        complete_path = cwd / "artifacts/checkpoints/stage4/complete.json"
        complete = dict(load_json(complete_path))
        complete["diagnostics_report_sha256"] = sha256_file(report)
        _write(complete_path, json.dumps(complete, sort_keys=True) + "\n")

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(corrupt_report)
    with pytest.raises(OrchestrationError, match=match):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )


@pytest.mark.parametrize("stage", ("stage3", "stage4"))
def test_post_approval_completion_rejects_arbitrary_nonempty_main_report(
    tmp_path: Path,
    stage: str,
) -> None:
    root = _project(tmp_path)

    def corrupt_report(command: CommandSpec, cwd: Path) -> None:
        _stage_callback(command, cwd)
        if command.name != stage:
            return
        report_name = (
            "STAGE3_PLANNER_GUARD.md" if stage == "stage3" else "STAGE4_E2E.md"
        )
        report = cwd / "reports" / report_name
        _write(report, "# Non-empty but forged report\n")
        complete_path = cwd / f"artifacts/checkpoints/{stage}/complete.json"
        complete = dict(load_json(complete_path))
        complete["report_sha256"] = sha256_file(report)
        _write(complete_path, json.dumps(complete, sort_keys=True) + "\n")

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(corrupt_report)
    with pytest.raises(OrchestrationError, match="report.protocol_id"):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )


@pytest.mark.parametrize(
    ("stage", "mutation", "match"),
    (
        ("stage3", "selected_sha", "report.selected_checkpoint_sha256"),
        ("stage4", "selected_sha", "report.selected_checkpoint_sha256"),
        ("stage3", "metrics", "report.selected_single_psnr_ssim"),
        ("stage4", "metrics", "report.selected_single_psnr_ssim"),
    ),
)
def test_post_approval_completion_rejects_semantically_forged_main_report(
    tmp_path: Path,
    stage: str,
    mutation: str,
    match: str,
) -> None:
    root = _project(tmp_path)

    def corrupt_report(command: CommandSpec, cwd: Path) -> None:
        _stage_callback(command, cwd)
        if command.name != stage:
            return
        report_name = (
            "STAGE3_PLANNER_GUARD.md" if stage == "stage3" else "STAGE4_E2E.md"
        )
        report = cwd / "reports" / report_name
        text = report.read_text(encoding="utf-8")
        if mutation == "selected_sha":
            best = cwd / f"artifacts/checkpoints/{stage}/best_ema.pth"
            text = text.replace(sha256_file(best), "0" * 64)
        else:
            if stage == "stage3":
                text = text.replace(
                    "Selected Single PSNR/SSIM: 25.0000000000 / 0.8000000000",
                    "Selected Single PSNR/SSIM: nan / nan",
                )
            else:
                text = text.replace(
                    "Selected Single PSNR/SSIM: 25.000000 / 0.80000000",
                    "Selected Single PSNR/SSIM: nan / nan",
                )
        _write(report, text)
        complete_path = cwd / f"artifacts/checkpoints/{stage}/complete.json"
        complete = dict(load_json(complete_path))
        complete["report_sha256"] = sha256_file(report)
        _write(complete_path, json.dumps(complete, sort_keys=True) + "\n")

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(corrupt_report)
    with pytest.raises(OrchestrationError, match=match):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )


@pytest.mark.parametrize(
    ("stage", "selected_step"),
    (("stage3", 9_000), ("stage4", 36_000)),
)
def test_post_approval_completion_accepts_selected_metrics_when_final_not_improved(
    tmp_path: Path,
    stage: str,
    selected_step: int,
) -> None:
    root = _project(tmp_path)

    def final_not_improved(command: CommandSpec, cwd: Path) -> None:
        if command.name in {"stage3", "stage4"}:
            _write_post_approval_completion(
                cwd,
                command.name,
                best_step=selected_step if command.name == stage else None,
            )
        else:
            _stage_callback(command, cwd)

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(final_not_improved)
    complete = orchestrator.approve_and_resume_stage3(
        approve_stage3=True,
        resume_from_stage3=True,
    )
    assert complete.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )


@pytest.mark.parametrize(
    ("stage", "selected_step"),
    (("stage3", 9_000), ("stage4", 36_000)),
)
def test_post_approval_completion_rejects_final_metrics_labeled_as_selected(
    tmp_path: Path,
    stage: str,
    selected_step: int,
) -> None:
    root = _project(tmp_path)

    def mislabel_final_metrics(command: CommandSpec, cwd: Path) -> None:
        if command.name not in {"stage3", "stage4"}:
            _stage_callback(command, cwd)
            return
        _write_post_approval_completion(
            cwd,
            command.name,
            best_step=selected_step if command.name == stage else None,
        )
        if command.name != stage:
            return
        report_name = (
            "STAGE3_PLANNER_GUARD.md" if stage == "stage3" else "STAGE4_E2E.md"
        )
        report = cwd / "reports" / report_name
        text = report.read_text(encoding="utf-8")
        if stage == "stage3":
            text = text.replace(
                "Selected Single PSNR/SSIM: 25.0000000000 / 0.8000000000",
                "Selected Single PSNR/SSIM: 24.0000000000 / 0.7000000000",
            )
        else:
            text = text.replace(
                "Selected Single PSNR/SSIM: 25.000000 / 0.80000000",
                "Selected Single PSNR/SSIM: 24.000000 / 0.70000000",
            )
        _write(report, text)
        complete_path = cwd / f"artifacts/checkpoints/{stage}/complete.json"
        complete_payload = dict(load_json(complete_path))
        complete_payload["report_sha256"] = sha256_file(report)
        _write(
            complete_path,
            json.dumps(complete_payload, sort_keys=True) + "\n",
        )

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(mislabel_final_metrics)
    with pytest.raises(
        OrchestrationError,
        match="report.selected_single_psnr_ssim",
    ):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )


@pytest.mark.parametrize("stage", ("stage3", "stage4"))
def test_post_approval_completion_rejects_main_report_hash_drift(
    tmp_path: Path,
    stage: str,
) -> None:
    root = _project(tmp_path)

    def drift_complete_hash(command: CommandSpec, cwd: Path) -> None:
        _stage_callback(command, cwd)
        if command.name != stage:
            return
        complete_path = cwd / f"artifacts/checkpoints/{stage}/complete.json"
        complete = dict(load_json(complete_path))
        complete["report_sha256"] = "0" * 64
        _write(complete_path, json.dumps(complete, sort_keys=True) + "\n")

    orchestrator, _ = _pause_after_stage2(root)
    orchestrator.runner = RecordingRunner(drift_complete_hash)
    with pytest.raises(OrchestrationError, match="report_sha256"):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )


def test_post_approval_resume_recovers_proven_stale_stage3(tmp_path: Path) -> None:
    root = _project(tmp_path)

    def interrupt_stage3(command: CommandSpec, cwd: Path) -> None:
        if command.name == "stage3":
            _write_raw_post_approval_checkpoint(cwd, "stage3")
        else:
            _stage_callback(command, cwd)

    orchestrator, _ = _pause_after_stage2(root)
    runner = RecordingRunner(interrupt_stage3, returncodes={"stage3": 6})
    orchestrator.runner = runner
    with pytest.raises(ChildCommandError):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    state = orchestrator.load_state()
    stage3 = orchestrator.post_approval_commands()[0][1]
    state.status = PipelineStatus.STAGE3_RUNNING.value
    state.current_stage = "STAGE3"
    state.last_command = list(stage3.argv)
    state.gpu = "owned_by_child_process"
    orchestrator._persist(state)
    runner.returncodes["stage3"] = 0
    runner.callback = _stage_callback
    complete = orchestrator.resume_post_approval_pipeline()
    assert complete.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )
    resumed = next(
        command
        for command in runner.commands
        if command.name == "stage3" and "--resume" in command.argv
    )
    assert resumed.argv[-1].endswith("artifacts/checkpoints/stage3/last.pth")


def test_post_approval_resume_finalizes_proven_stale_completed_stage4(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    orchestrator, _ = _pause_after_stage2(root)
    runner = RecordingRunner(_stage_callback)
    orchestrator.runner = runner
    complete = orchestrator.approve_and_resume_stage3(
        approve_stage3=True,
        resume_from_stage3=True,
    )
    state = complete
    stage4 = orchestrator.post_approval_commands()[1][1]
    state.status = PipelineStatus.STAGE4_RUNNING.value
    state.current_stage = "STAGE4"
    state.last_command = list(stage4.argv)
    state.gpu = "owned_by_child_process"
    orchestrator._persist(state)
    commands_before = len(runner.commands)
    recovered = orchestrator.resume_post_approval_pipeline()
    assert len(runner.commands) == commands_before
    assert recovered.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )


def test_subprocess_runner_preserves_child_exit_and_tmux_avoids_double_log(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    runner = SubprocessCommandRunner(output=output)
    command = CommandSpec(
        name="exit_seven",
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
    )
    assert runner.run(command, cwd=tmp_path, log_path=tmp_path / "child.log") == 7
    assert "exit=7" in (tmp_path / "child.log").read_text(encoding="utf-8")

    tmux = recommended_tmux_argv(tmp_path, sys.executable)
    assert tmux[5:9] == ("bash", "-o", "pipefail", "-c")
    assert "--run_main_pipeline" in tmux[-1]
    assert "tee" not in tmux[-1]


@pytest.mark.parametrize(
    "parent_value",
    (
        None,
        "backend:native,expandable_segments:True",
        "backend:cudaMallocAsync",
    ),
)
def test_subprocess_runner_forces_locked_allocator_for_every_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_value: str | None,
) -> None:
    if parent_value is None:
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", parent_value)
    monkeypatch.setenv("GRAPHRESTORE_TEST_INHERITED", "preserved")
    runner = SubprocessCommandRunner(output=io.StringIO())
    command = CommandSpec(
        name="print_allocator",
        argv=(
            sys.executable,
            "-c",
            (
                "import os; print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF')); "
                "print(os.environ.get('GRAPHRESTORE_TEST_INHERITED'))"
            ),
        ),
    )
    log = tmp_path / "allocator.log"
    assert runner.run(command, cwd=tmp_path, log_path=log) == 0
    assert "backend:native,expandable_segments:True" in log.read_text(encoding="utf-8")
    assert "preserved" in log.read_text(encoding="utf-8")
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == parent_value


def test_cli_partial_approval_flag_returns_distinct_refusal_code(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/orchestrate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project_root",
            str(root),
            "--approve_stage3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == ApprovalError.exit_code
    assert "both explicit flags" in result.stderr


def test_cli_exposes_distinct_post_approval_resume_action(tmp_path: Path) -> None:
    root = _project(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/orchestrate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project_root",
            str(root),
            "--resume_post_approval_pipeline",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == OrchestrationError.exit_code
    assert "allowed only from FAILED" in result.stderr


def test_cli_stage3_extension_path_is_only_a_post_approval_resume_modifier(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/orchestrate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project_root",
            str(root),
            "--show_state",
            "--stage3_extension_authorization",
            str(root / "artifacts/approvals/STAGE3_EXTENSION_APPROVED.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires --resume_post_approval_pipeline" in result.stderr


def test_cli_stage4_extension_path_is_only_a_post_approval_resume_modifier(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/orchestrate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project_root",
            str(root),
            "--show_state",
            "--stage4_extension_authorization",
            str(root / "artifacts/approvals/STAGE4_EXTENSION_GATE_RECEIPT.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires --resume_post_approval_pipeline" in result.stderr


def test_cli_stage3_finalization_is_resume_only_and_mutually_exclusive(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/orchestrate.py"
    finalization = root / "artifacts/approvals/STAGE3_EXTENSION_REVOKED.json"
    extension = root / "artifacts/approvals/STAGE3_EXTENSION_APPROVED.json"
    resume_only = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project_root",
            str(root),
            "--show_state",
            "--stage3_finalization_authorization",
            str(finalization),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert resume_only.returncode == 2
    assert "requires --resume_post_approval_pipeline" in resume_only.stderr

    exclusive = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project_root",
            str(root),
            "--resume_post_approval_pipeline",
            "--stage3_extension_authorization",
            str(extension),
            "--stage3_finalization_authorization",
            str(finalization),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert exclusive.returncode == 2
    assert "mutually exclusive" in exclusive.stderr


def test_child_failure_is_recorded_with_exact_exit_code(tmp_path: Path) -> None:
    root = _project(tmp_path)
    runner = RecordingRunner(returncodes={"audit_data": 17})
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)
    with pytest.raises(ChildCommandError) as caught:
        orchestrator.run_preflight()
    assert caught.value.exit_code == 17
    state = orchestrator.load_state()
    assert state.status == PipelineStatus.PREFLIGHT_FAILED.value
    assert state.last_exit_code == 17
    assert state.gpu == "released"
