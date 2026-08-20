#!/usr/bin/env python3
"""Train the approved V7.1 Stage3 planner while freezing the Stage1 executor."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GraphRestoreEpisodeDataset, build_dataloader  # noqa: E402
from src.training.optimization import WarmupCosineScheduler  # noqa: E402
from src.training.selection import ValidationScore, is_better_checkpoint  # noqa: E402
from src.training.stage3_finalization import (  # noqa: E402
    Stage3FinalizationContractError,
    refuse_stage3_training_if_revoked,
)
from src.training.stage3_engine import (  # noqa: E402
    PROTOCOL_ID,
    STAGE3_SCHEMA,
    Stage3ContractError,
    Stage3OptimizerTransaction,
    Stage3PlannerEMA,
    append_calibration_history,
    assert_relation_clean_disjoint,
    build_stage3_model,
    build_stage3_optimizer,
    build_stage3_provenance,
    calibrate_presence_thresholds,
    calibration_history_row,
    collect_primary_val_presence,
    configure_stage3_reproducibility,
    enforce_stage3_peak_memory,
    freeze_presence_thresholds,
    load_relation_records,
    load_stage3_best_ema,
    prepare_stage3_supervision_batch,
    probe_stage3_validation_vram,
    reset_stage3_peak_memory,
    resume_stage3_checkpoint,
    save_stage3_checkpoint,
    select_stage3_micro_batch,
    train_stage3_optimizer_step,
    validate_stage3,
    validate_stage3_allocator_conf,
    validate_stage3_approval,
    validate_stage3_extension_authorization,
    validate_stage3_validation_vram_evidence,
    validation_score,
)
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_json,
    utc_now_iso,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3_planner.yaml"),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        metavar="CHECKPOINT",
        help="resume CHECKPOINT, or output_dir/last.pth when omitted",
    )
    parser.add_argument(
        "--micro_batch",
        type=int,
        choices=(8, 4, 2, 1),
        help="verify and freeze this micro batch; default probes 8,4,2,1",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="override checkpoints/log/report/threshold destinations",
    )
    parser.add_argument(
        "--extension_authorization",
        type=Path,
        help=(
            "canonical user authorization for the audited 12k-to-18k "
            "three-validation Stage3 extension; valid only with --resume"
        ),
    )
    return parser


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )
    handle.flush()


_VOLATILE_TRAIN_STEP_LOG_FIELDS = frozenset({"utc", "seconds", "images_per_second"})


def _canonical_train_step_log(value: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.loads(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    return {
        key: item
        for key, item in canonical.items()
        if key not in _VOLATILE_TRAIN_STEP_LOG_FIELDS
    }


def _append_train_step_idempotent(
    path: Path,
    handle: Any,
    value: Mapping[str, Any],
) -> bool:
    """Commit one scientific train-step row without duplicating a replay.

    A signal can arrive after the row is flushed but before the in-memory
    optimizer transaction becomes checkpointable.  The prior raw checkpoint is
    then replayed, so the repeated row must match the already durable scientific
    record and must not be appended a second time.  Wall-clock fields are
    intentionally excluded from that equality check.
    """

    step = value.get("step")
    if (
        value.get("event") != "train_step"
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
    ):
        raise Stage3ContractError("invalid Stage3 train-step log record")
    handle.flush()
    existing = [
        row
        for _, row in iter_jsonl(path)
        if row.get("event") == "train_step" and row.get("step") == step
    ]
    if len(existing) > 1:
        raise Stage3ContractError(f"duplicate Stage3 train-step rows at step {step}")
    if existing:
        if _canonical_train_step_log(existing[0]) != _canonical_train_step_log(value):
            raise Stage3ContractError(
                f"Stage3 train-step replay drifted at step {step}"
            )
        return False
    _append_jsonl(handle, value)
    return True


def _score_from_metrics(value: object) -> ValidationScore | None:
    if not isinstance(value, Mapping) or "best_group_a_psnr" not in value:
        return None
    score = ValidationScore(
        group_a_psnr=float(value["best_group_a_psnr"]),
        group_a_ssim=float(value["best_group_a_ssim"]),
        single_psnr=float(value["best_single_psnr"]),
        single_ssim=float(value["best_single_ssim"]),
        step=int(value["best_step"]),
    )
    if not all(
        math.isfinite(item)
        for item in (
            score.group_a_psnr,
            score.group_a_ssim,
            score.single_psnr,
            score.single_ssim,
        )
    ):
        raise Stage3ContractError("resume best score is non-finite")
    return score


def _current_score_from_metrics(value: object) -> ValidationScore | None:
    if not isinstance(value, Mapping) or "validation_step" not in value:
        return None
    score = ValidationScore(
        group_a_psnr=float(value["group_a_psnr"]),
        group_a_ssim=float(value["group_a_ssim"]),
        single_psnr=float(value["single_psnr"]),
        single_ssim=float(value["single_ssim"]),
        step=int(value["validation_step"]),
    )
    if not all(
        math.isfinite(item)
        for item in (
            score.group_a_psnr,
            score.group_a_ssim,
            score.single_psnr,
            score.single_ssim,
        )
    ):
        raise Stage3ContractError("resume current score is non-finite")
    return score


def _sigterm_as_keyboard_interrupt(signum: int, frame: object) -> None:
    del signum, frame
    raise KeyboardInterrupt


def _checkpoint_metrics(
    current: ValidationScore | None,
    best: ValidationScore | None,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    if current is not None:
        result.update(
            {
                "group_a_psnr": current.group_a_psnr,
                "group_a_ssim": current.group_a_ssim,
                "single_psnr": current.single_psnr,
                "single_ssim": current.single_ssim,
                "validation_step": current.step,
            }
        )
    if best is not None:
        result.update(
            {
                "best_group_a_psnr": best.group_a_psnr,
                "best_group_a_ssim": best.group_a_ssim,
                "best_single_psnr": best.single_psnr,
                "best_single_ssim": best.single_ssim,
                "best_step": best.step,
            }
        )
    return result


@dataclass
class _PendingValidationState:
    step: int | None = None

    def begin(self, step: int, save: Callable[[], None]) -> None:
        # Publish the in-process marker before the atomic write attempt.  If a
        # signal interrupts fsync/replace, the interrupt handler must preserve
        # the intended replay marker rather than overwrite last.pth with None.
        self.step = step
        save()

    def clear(self, save: Callable[[], None]) -> None:
        # Keep the marker until the clean raw checkpoint itself is durable.
        save()
        self.step = None


def _publish_stage3_train_boundary(
    *,
    log_path: Path,
    log: Any,
    row: Mapping[str, Any],
    optimizer_transaction: Stage3OptimizerTransaction,
    pending_validation: _PendingValidationState,
    validation_due: bool,
) -> bool:
    """Make a peak-checked/logged update safely checkpointable.

    The optimizer transaction remains active through this function.  At a
    validation boundary the local pending marker is published before commit, so
    a signal immediately after commit can only save a replayable pending raw.
    """

    if not optimizer_transaction.active:
        raise Stage3ContractError(
            "Stage3 post-update publication requires an active transaction"
        )
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise Stage3ContractError("Stage3 post-update publication step is invalid")
    appended = _append_train_step_idempotent(log_path, log, row)
    if validation_due:
        if pending_validation.step not in {None, step}:
            raise Stage3ContractError("Stage3 pending validation marker drifted")
        pending_validation.step = step
    elif pending_validation.step is not None:
        raise Stage3ContractError("unexpected Stage3 pending validation marker")
    optimizer_transaction.commit()
    return appended


def _validated_restoration(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if summary.get("protocol_id") != PROTOCOL_ID:
        raise Stage3ContractError("Stage3 report protocol drifted")
    restoration = summary.get("restoration")
    if not isinstance(restoration, Mapping):
        raise Stage3ContractError("Stage3 report restoration metrics are missing")
    for group in ("single", "group_a"):
        metrics = restoration.get(group)
        if not isinstance(metrics, Mapping):
            raise Stage3ContractError(f"Stage3 report {group} metrics are missing")
        for metric in ("psnr", "ssim"):
            value = metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise Stage3ContractError(
                    f"Stage3 report {group} {metric} is non-finite"
                )
    return restoration


def _report_binding(path: Path) -> dict[str, str]:
    report = path.resolve()
    if not report.is_file():
        raise Stage3ContractError("Stage3 completion report is missing")
    return {"report": str(report), "report_sha256": sha256_file(report)}


def _best_score_payload(score: ValidationScore) -> dict[str, float | int]:
    values = {
        "group_a_psnr": score.group_a_psnr,
        "group_a_ssim": score.group_a_ssim,
        "single_psnr": score.single_psnr,
        "single_ssim": score.single_ssim,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise Stage3ContractError("Stage3 selected best score is non-finite")
    return {
        **{name: float(f"{value:.10f}") for name, value in values.items()},
        "step": int(score.step),
    }


def _render_report(
    summary: Mapping[str, Any],
    *,
    best: ValidationScore,
    checkpoint: Path,
    thresholds: Path,
    training_target_step: int = 12_000,
    schedule_horizon_steps: int = 12_000,
    extension_authorization_sha256: str | None = None,
) -> str:
    _validated_restoration(summary)
    checkpoint = checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    best_score = _best_score_payload(best)
    relation = summary["relation"]
    guard = summary["guard"]
    graph = summary["graph"]
    return (
        "# Stage3 Planner and Guard\n\n"
        f"- protocol: `{PROTOCOL_ID}`\n"
        f"- selected step: {best.step}\n"
        f"- completed training target step: {training_target_step}\n"
        f"- cosine schedule horizon step: {schedule_horizon_steps}\n"
        f"- Stage3 extension authorization SHA256: "
        f"`{extension_authorization_sha256 or 'none'}`\n"
        f"- best checkpoint: `{checkpoint}`\n"
        f"- selected best checkpoint SHA256: `{checkpoint_sha256}`\n"
        f"- frozen thresholds: `{thresholds}`\n"
        "- data: primary train/val single + Group A; relation validation uses interaction_val only\n"
        "- MiO100 / Group B / Group C rows read: 0\n"
        "- checkpoint rank: restoration only; guard diagnostics do not alter rank\n\n"
        "## Restoration (fixed presence threshold 0.50)\n\n"
        f"- Selected Single PSNR/SSIM: {best_score['single_psnr']:.10f} / "
        f"{best_score['single_ssim']:.10f}\n"
        f"- Selected Group-A PSNR/SSIM: {best_score['group_a_psnr']:.10f} / "
        f"{best_score['group_a_ssim']:.10f}\n\n"
        "## Planner / Graph\n\n"
        f"- macro F1: {summary['planner']['macro_f1']:.6f}\n"
        f"- non-ambiguous relation accuracy: {relation['relation_accuracy_non_ambiguous']}\n"
        f"- parallel precision/recall: {relation['parallel_precision_non_ambiguous']} / {relation['parallel_recall_non_ambiguous']}\n"
        f"- ambiguous count/fraction: {relation['n_ambiguous']} / {relation['ambiguous_fraction']}\n"
        f"- pre/post compiler cycle rate: {graph['pre_compiler_cycle_rate']} / {graph['post_compiler_cycle_rate']}\n\n"
        "## Continuous Guard Diagnostics (diagnostic only)\n\n"
        f"- rain Spearman/MAE/std/>0.9: {guard['guard_spearman_rain']} / {guard['guard_mae_rain']} / {guard['guard_std_rain']} / {guard['guard_high_frac_rain']}\n"
        f"- haze Spearman/MAE/std/>0.9: {guard['guard_spearman_haze']} / {guard['guard_mae_haze']} / {guard['guard_std_haze']} / {guard['guard_high_frac_haze']}\n"
        f"- rain valid/skipped: {guard['valid_guard_images_rain']} / {guard['skipped_guard_images_rain']}\n"
        f"- haze valid/skipped: {guard['valid_guard_images_haze']} / {guard['skipped_guard_images_haze']}\n"
    )


def _render_validation_report(
    summary: Mapping[str, Any],
    *,
    current: ValidationScore,
    best: ValidationScore,
    checkpoint: Path,
    peak_reserved_bytes: int,
    peak_reserved_fraction: float,
) -> str:
    restoration = _validated_restoration(summary)
    checkpoint = checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    return (
        "# Stage3 Planner and Guard\n\n"
        f"- protocol: `{PROTOCOL_ID}`\n"
        f"- current validation step: {current.step}\n"
        f"- selected step: {best.step}\n"
        f"- best checkpoint: `{checkpoint}`\n"
        f"- selected best checkpoint SHA256: `{checkpoint_sha256}`\n"
        "- presence threshold during checkpoint selection: 0.50\n"
        "- final per-skill threshold calibration: pending Stage3 completion\n"
        f"- validation peak reserved: {peak_reserved_bytes} bytes "
        f"({peak_reserved_fraction:.6%})\n"
        "- MiO100 / Group B / Group C rows read: 0\n\n"
        "## Current Restoration\n\n"
        f"- single PSNR/SSIM: {restoration['single']['psnr']:.6f} / "
        f"{restoration['single']['ssim']:.8f}\n"
        f"- Group A PSNR/SSIM: {restoration['group_a']['psnr']:.6f} / "
        f"{restoration['group_a']['ssim']:.8f}\n"
    )


def run(arguments: argparse.Namespace) -> int:
    # A permanent revocation is a tombstone, not an optional authorization
    # variant.  Refuse on mere directory-entry existence (including malformed
    # files and dangling symlinks) before approval I/O, any CUDA API, checkpoint
    # tensor loading, dataset construction, optimizer creation, or pixel read.
    refuse_stage3_training_if_revoked(PROJECT_ROOT)

    # HARD ORDERING: this complete file/hash approval audit happens before the
    # first CUDA query, checkpoint tensor load, dataset construction, or pixel read.
    paths = validate_stage3_approval(
        _project_path(arguments.config),
        project_root=PROJECT_ROOT,
        output_dir=arguments.output_dir,
        require_orchestrator_running=True,
        allow_failed_resume=arguments.resume is not None,
    )
    canonical_extension = (
        PROJECT_ROOT / "artifacts/approvals/STAGE3_EXTENSION_APPROVED.json"
    ).resolve(strict=False)
    extension_argument = getattr(arguments, "extension_authorization", None)
    if extension_argument is None:
        if canonical_extension.exists():
            raise Stage3ContractError(
                "Stage3 extension authorization exists but was not explicitly passed"
            )
        extension = None
    else:
        if arguments.resume is None:
            raise Stage3ContractError(
                "Stage3 extension authorization is valid only for exact resume"
            )
        extension = validate_stage3_extension_authorization(
            _project_path(extension_argument), paths
        )
        forbidden_completed = [
            path
            for path in (
                paths.thresholds,
                paths.output_dir / "complete.json",
                PROJECT_ROOT / "artifacts/checkpoints/stage4/last.pth",
                PROJECT_ROOT / "artifacts/checkpoints/stage4/best_ema.pth",
                PROJECT_ROOT / "artifacts/checkpoints/stage4/complete.json",
            )
            if path.exists()
        ]
        if forbidden_completed:
            raise Stage3ContractError(
                "Stage3 extension refuses completed calibration/Stage4 artifacts: "
                f"{forbidden_completed}"
            )

    # Semantic label audit remains CPU/file-only and precedes GPU reservation.
    approved_parent_sha = paths.approval.bindings["stage1_checkpoint"]["sha256"]
    train_relations = load_relation_records(
        paths.relation_train,
        split="train",
        parent_checkpoint_sha256=approved_parent_sha,
        interaction_manifest_sha256=paths.approval.bindings[
            "interaction_train_manifest"
        ]["sha256"],
    )
    val_relations = load_relation_records(
        paths.relation_val,
        split="val",
        parent_checkpoint_sha256=approved_parent_sha,
        interaction_manifest_sha256=paths.approval.bindings["interaction_val_manifest"][
            "sha256"
        ],
    )
    assert_relation_clean_disjoint(train_relations, val_relations)

    # Environment-only and therefore still before the first CUDA API query.
    validate_stage3_allocator_conf()

    output_dir = paths.output_dir
    resume_path: Path | None = None
    recover_incomplete_initialization = False
    run_contract_path = output_dir / "run_contract.json"
    frozen_validation_vram_gate: Mapping[str, Any] | None = None
    if arguments.resume is not None:
        resume_path = (
            output_dir / "last.pth"
            if arguments.resume == "auto"
            else _project_path(arguments.resume)
        )
        if not resume_path.is_file() or not run_contract_path.is_file():
            raise Stage3ContractError(
                "Stage3 resume checkpoint/run contract is missing"
            )
        if resume_path.name != "last.pth":
            raise Stage3ContractError(
                "exact Stage3 resume requires last.pth; best_ema.pth is inference/selection only"
            )
        existing_contract = load_json(run_contract_path)
        if not isinstance(existing_contract, Mapping):
            raise Stage3ContractError("Stage3 run contract is invalid")
        contract_provenance = existing_contract.get("provenance")
        if not isinstance(contract_provenance, Mapping) or not isinstance(
            contract_provenance.get("runtime"), Mapping
        ):
            raise Stage3ContractError("Stage3 run contract lacks frozen runtime")
        selected_micro = int(contract_provenance["runtime"].get("micro_batch", -1))
        frozen_validation_vram_gate = validate_stage3_validation_vram_evidence(
            contract_provenance["runtime"].get("validation_vram_gate")
        )
        if (
            validate_stage3_validation_vram_evidence(
                existing_contract.get("validation_vram_gate")
            )
            != frozen_validation_vram_gate
        ):
            raise Stage3ContractError(
                "Stage3 run-contract validation VRAM evidence drifted"
            )
        if (
            arguments.micro_batch is not None
            and arguments.micro_batch != selected_micro
        ):
            raise Stage3ContractError("Stage3 micro batch cannot change across resume")
        trials = ()
    else:
        stale = [
            path
            for path in (
                run_contract_path,
                output_dir / "last.pth",
                output_dir / "best_ema.pth",
                output_dir / "complete.json",
                output_dir / "train.jsonl",
                output_dir / "validation_latest.json",
                output_dir / "selected_validation.json",
            )
            if path.exists()
        ]
        only_contract = stale == [run_contract_path]
        if stale and not only_contract:
            raise Stage3ContractError(
                f"refusing fresh Stage3 over existing run artifacts; use --resume: {stale}"
            )
        if only_contract:
            existing_contract = load_json(run_contract_path)
            if not isinstance(existing_contract, Mapping):
                raise Stage3ContractError(
                    "Stage3 incomplete-initialization run contract is invalid"
                )
            contract_provenance = existing_contract.get("provenance")
            if not isinstance(contract_provenance, Mapping) or not isinstance(
                contract_provenance.get("runtime"), Mapping
            ):
                raise Stage3ContractError(
                    "Stage3 incomplete-initialization contract lacks frozen runtime"
                )
            selected_micro = int(contract_provenance["runtime"].get("micro_batch", -1))
            frozen_validation_vram_gate = validate_stage3_validation_vram_evidence(
                contract_provenance["runtime"].get("validation_vram_gate")
            )
            if (
                validate_stage3_validation_vram_evidence(
                    existing_contract.get("validation_vram_gate")
                )
                != frozen_validation_vram_gate
            ):
                raise Stage3ContractError(
                    "Stage3 run-contract validation VRAM evidence drifted"
                )
            if (
                arguments.micro_batch is not None
                and arguments.micro_batch != selected_micro
            ):
                raise Stage3ContractError(
                    "Stage3 micro batch cannot change during initialization recovery"
                )
            recover_incomplete_initialization = True
            trials = ()

    if not torch.cuda.is_available():
        raise Stage3ContractError("formal Stage3 requires an available CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    configure_stage3_reproducibility(int(paths.config["seed"]))
    model, parent_report = build_stage3_model(paths, device=device)

    if resume_path is None and not recover_incomplete_initialization:
        candidates = (
            (arguments.micro_batch,)
            if arguments.micro_batch is not None
            else (8, 4, 2, 1)
        )
        selected_micro, trials = select_stage3_micro_batch(
            model,
            device=device,
            candidates=candidates,
            required_passes=10,
            maximum_reserved_fraction=float(
                paths.config["runtime"]["vram_maximum_peak_reserved_fraction"]
            ),
        )
    effective_batch = int(paths.config["data"]["effective_batch_size"])
    accumulation_steps = effective_batch // selected_micro
    optimizer = build_stage3_optimizer(
        model,
        lr=float(paths.config["optimization"]["lr"]),
        weight_decay=float(paths.config["optimization"]["weight_decay"]),
    )
    schedule_horizon_steps = int(paths.config["training"]["max_steps"])
    training_target_step = (
        schedule_horizon_steps if extension is None else extension.target_step
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(paths.config["optimization"]["warmup_steps"]),
        max_steps=schedule_horizon_steps,
        min_lr=float(paths.config["optimization"]["min_lr"]),
    )
    ema = Stage3PlannerEMA(model, decay=float(paths.config["ema"]["decay"]))
    if resume_path is None and not recover_incomplete_initialization:
        validation_vram_gate = validate_stage3_validation_vram_evidence(
            asdict(
                probe_stage3_validation_vram(
                    model,
                    optimizer=optimizer,
                    ema=ema,
                    device=device,
                    image_size=2040,
                    max_rounds=3,
                    maximum_reserved_fraction=float(
                        paths.config["runtime"]["vram_maximum_peak_reserved_fraction"]
                    ),
                )
            )
        )
        if optimizer.state:
            raise Stage3ContractError(
                "Stage3 validation gate did not restore an empty step0 optimizer"
            )
    else:
        validation_vram_gate = validate_stage3_validation_vram_evidence(
            frozen_validation_vram_gate
        )
    provenance = build_stage3_provenance(
        paths,
        parent_report,
        micro_batch=selected_micro,
        accumulation_steps=accumulation_steps,
        validation_vram_gate=validation_vram_gate,
        max_steps=schedule_horizon_steps,
        extension=extension,
    )
    if resume_path is not None or recover_incomplete_initialization:
        contract = load_json(run_contract_path)
        if contract.get("provenance") != provenance:
            raise Stage3ContractError("Stage3 resume provenance drifted")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            run_contract_path,
            {
                "schema_version": STAGE3_SCHEMA,
                "created_utc": utc_now_iso(),
                "provenance": provenance,
                "parent_load": {
                    "loaded_count": parent_report.loaded_count,
                    "initialized_planner_key_count": len(
                        parent_report.initialized_planner_keys
                    ),
                },
                "micro_batch_trials": [asdict(trial) for trial in trials],
                "validation_vram_gate": validation_vram_gate,
            },
        )

    if parent_report.checkpoint_sha256 != approved_parent_sha:
        raise Stage3ContractError("loaded Stage1 parent differs from approved binding")
    depth_compat = PROJECT_ROOT / "artifacts/cache/agenticir_depth_compat"
    train_dataset = GraphRestoreEpisodeDataset(
        paths.train_manifest,
        paths.training_data_root,
        depth_compat,
        crop_size=192,
        training=True,
        stage="stage3",
        base_seed=int(paths.config["seed"]),
        agenticir_repo=paths.resolved["agenticir_repo"],
        mioir_repo=paths.resolved["mioir_repo"],
    )
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
    train_loader, sampler = build_dataloader(
        train_dataset,
        batch_size=selected_micro,
        effective_batch_size=effective_batch,
        # Preserve the original sampler contract and absolute cursor sequence.
        # Its iterator still exposes more than the 48k extension samples needed
        # after the step-12000 cursor.
        num_samples=schedule_horizon_steps * effective_batch,
        stage="stage3",
        base_seed=int(paths.config["seed"]),
        start_step=0,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
        training=True,
    )
    if sampler is None:
        raise RuntimeError("Stage3 train loader lacks a stateful sampler")

    validation_every = int(paths.config["runtime"]["validation_every_steps"])
    maximum_reserved_fraction = float(
        paths.config["runtime"]["vram_maximum_peak_reserved_fraction"]
    )
    optimizer_transaction = Stage3OptimizerTransaction()
    step = 0
    best: ValidationScore | None = None
    last_score: ValidationScore | None = None
    pending_validation = _PendingValidationState()
    if resume_path is not None:
        payload = resume_stage3_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_provenance=provenance,
            validation_every_steps=validation_every,
        )
        step = int(payload["step"])
        best = _score_from_metrics(payload.get("metrics"))
        last_score = _current_score_from_metrics(payload.get("metrics"))
        pending_validation.step = payload["pending_validation_step"]
        if extension is not None and not (
            extension.base_step <= step <= extension.target_step
        ):
            raise Stage3ContractError(
                "Stage3 extension resume step escapes the authorized interval"
            )
    else:
        sampler.set_step(0)
        # Fresh runs become exactly resumable before the first sample/prefetch.
        save_stage3_checkpoint(
            output_dir / "last.pth",
            step=0,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            provenance=provenance,
            metrics={},
            pending_validation_step=None,
            optimizer_transaction=optimizer_transaction,
            validation_every_steps=validation_every,
        )
    log_path = output_dir / "train.jsonl"
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        _append_jsonl(
            log,
            {
                "event": "resume" if resume_path else "start",
                "recovered_incomplete_initialization": recover_incomplete_initialization,
                "utc": utc_now_iso(),
                "step": step,
                "micro_batch": selected_micro,
                "accumulation_steps": accumulation_steps,
                "approval_sha256": paths.approval.approval_sha256,
                "pending_validation_step": pending_validation.step,
                "training_target_step": training_target_step,
                "schedule_horizon_steps": schedule_horizon_steps,
                "extension_authorization_sha256": (
                    None if extension is None else extension.authorization_sha256
                ),
            },
        )

        def run_validation_boundary(*, replay: bool) -> None:
            nonlocal best, last_score
            if replay:
                if pending_validation.step != step:
                    raise Stage3ContractError(
                        "Stage3 pending validation replay state drifted"
                    )
            else:
                pending_validation.begin(
                    step,
                    lambda: save_stage3_checkpoint(
                        output_dir / "last.pth",
                        step=step,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler=sampler,
                        provenance=provenance,
                        metrics=_checkpoint_metrics(last_score, best),
                        pending_validation_step=step,
                        optimizer_transaction=optimizer_transaction,
                        validation_every_steps=validation_every,
                    ),
                )
            _append_jsonl(
                log,
                {
                    "event": "validation_replay"
                    if replay
                    else "pre_validation_checkpoint",
                    "utc": utc_now_iso(),
                    "step": step,
                    "pending_validation_step": step,
                },
            )

            # The pending raw checkpoint is durable before any validation work.
            # A VRAM failure below must not publish metrics, best, or report.
            reset_stage3_peak_memory(device)
            with ema.apply_to(model):
                summary = validate_stage3(
                    model,
                    validation_dataset,
                    val_relations,
                    device=device,
                    use_bf16=True,
                    presence_threshold=0.5,
                )
            validation_peak_bytes, validation_peak_fraction = (
                enforce_stage3_peak_memory(
                    device,
                    phase=f"validation_step_{step}",
                    maximum_reserved_fraction=maximum_reserved_fraction,
                )
            )
            score = validation_score(summary, step)
            improved = is_better_checkpoint(score, best)
            candidate_best = score if improved else best
            if candidate_best is None:
                raise Stage3ContractError(
                    "Stage3 validation failed to select a checkpoint"
                )
            metrics = _checkpoint_metrics(score, candidate_best)
            best_path = output_dir / "best_ema.pth"
            if improved:
                save_stage3_checkpoint(
                    best_path,
                    step=step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    metrics=metrics,
                    model_as_ema=True,
                    pending_validation_step=None,
                    optimizer_transaction=optimizer_transaction,
                    validation_every_steps=validation_every,
                )
            atomic_write_json(output_dir / "validation_latest.json", summary)
            row = calibration_history_row(summary, step)
            append_calibration_history(paths.calibration_history, row)
            atomic_write_text(
                paths.report,
                _render_validation_report(
                    summary,
                    current=score,
                    best=candidate_best,
                    checkpoint=best_path,
                    peak_reserved_bytes=validation_peak_bytes,
                    peak_reserved_fraction=validation_peak_fraction,
                ),
            )
            _append_jsonl(
                log,
                {
                    "event": "validation",
                    "utc": utc_now_iso(),
                    "step": step,
                    "improved": improved,
                    "replayed": replay,
                    "peak_reserved_bytes": validation_peak_bytes,
                    "peak_reserved_fraction": validation_peak_fraction,
                    **row,
                },
            )
            best = candidate_best
            last_score = score
            # Clear pending only after metric, best, history, report, and log
            # have all been durably/visibly committed.
            pending_validation.clear(
                lambda: save_stage3_checkpoint(
                    output_dir / "last.pth",
                    step=step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    metrics=metrics,
                    pending_validation_step=None,
                    optimizer_transaction=optimizer_transaction,
                    validation_every_steps=validation_every,
                )
            )

        try:
            # Replaying a durable validation transaction precedes construction
            # of a training iterator and therefore any new sample prefetch.
            if pending_validation.step is not None:
                run_validation_boundary(replay=True)
            iterator = iter(train_loader) if step < training_target_step else None
            while step < training_target_step:
                if iterator is None:
                    raise Stage3ContractError("Stage3 training iterator is unavailable")
                reset_stage3_peak_memory(device)
                batches = [
                    prepare_stage3_supervision_batch(
                        next(iterator),
                        relation_lookup=train_relations,
                        model=model,
                        device=device,
                    )
                    for _ in range(accumulation_steps)
                ]
                result = train_stage3_optimizer_step(
                    model,
                    batches,
                    optimizer,
                    scheduler,
                    ema,
                    device=device,
                    gradient_clip_norm=float(
                        paths.config["optimization"]["gradient_clip_norm"]
                    ),
                    use_bf16=True,
                    audit_gradients=resume_path is None and step == 0,
                    optimizer_transaction=optimizer_transaction,
                )
                step += 1
                sampler.mark_consumed_optimizer_step(step)
                train_peak_bytes, train_peak_fraction = enforce_stage3_peak_memory(
                    device,
                    phase=f"train_step_{step}",
                    maximum_reserved_fraction=maximum_reserved_fraction,
                )
                validation_due = (
                    step % validation_every == 0 or step == training_target_step
                )
                _publish_stage3_train_boundary(
                    log_path=log_path,
                    log=log,
                    row={
                        "event": "train_step",
                        "utc": utc_now_iso(),
                        "step": step,
                        **asdict(result),
                        "images_per_second": result.samples / max(result.seconds, 1e-9),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "peak_reserved_bytes": train_peak_bytes,
                        "peak_reserved_fraction": train_peak_fraction,
                    },
                    optimizer_transaction=optimizer_transaction,
                    pending_validation=pending_validation,
                    validation_due=validation_due,
                )
                if validation_due:
                    run_validation_boundary(replay=False)
        except KeyboardInterrupt:
            if not optimizer_transaction.active:
                save_stage3_checkpoint(
                    output_dir / "last.pth",
                    step=step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    metrics=_checkpoint_metrics(last_score, best),
                    pending_validation_step=pending_validation.step,
                    optimizer_transaction=optimizer_transaction,
                    validation_every_steps=validation_every,
                )
            _append_jsonl(
                log,
                {
                    "event": "interrupted",
                    "utc": utc_now_iso(),
                    "step": step,
                    "mid_optimizer_update": optimizer_transaction.active,
                    "pending_validation_step": pending_validation.step,
                },
            )
            raise

    best_path = output_dir / "best_ema.pth"
    if best is None or not best_path.is_file():
        raise Stage3ContractError("Stage3 completed without a selected EMA snapshot")
    selected_model = load_stage3_best_ema(
        paths,
        best_path,
        device=device,
        load_frozen_thresholds=False,
    )
    reset_stage3_peak_memory(device)
    selected_summary = validate_stage3(
        selected_model,
        validation_dataset,
        val_relations,
        device=device,
        use_bf16=True,
        presence_threshold=0.5,
    )
    selected_peak_bytes, selected_peak_fraction = enforce_stage3_peak_memory(
        device,
        phase="selected_checkpoint_validation",
        maximum_reserved_fraction=maximum_reserved_fraction,
    )
    atomic_write_json(output_dir / "selected_validation.json", selected_summary)
    reset_stage3_peak_memory(device)
    probabilities, targets = collect_primary_val_presence(
        selected_model, validation_dataset, device=device, use_bf16=True
    )
    calibration_peak_bytes, calibration_peak_fraction = enforce_stage3_peak_memory(
        device,
        phase="presence_threshold_calibration",
        maximum_reserved_fraction=maximum_reserved_fraction,
    )
    calibration = calibrate_presence_thresholds(probabilities, targets)
    threshold_payload = freeze_presence_thresholds(
        paths.thresholds,
        calibration,
        primary_val_manifest=paths.val_manifest,
        selected_checkpoint=best_path,
        approval_sha256=paths.approval.approval_sha256,
        extension_authorization_sha256=(
            None if extension is None else extension.authorization_sha256
        ),
    )
    atomic_write_text(
        paths.report,
        _render_report(
            selected_summary,
            best=best,
            checkpoint=best_path,
            thresholds=paths.thresholds,
            training_target_step=training_target_step,
            schedule_horizon_steps=schedule_horizon_steps,
            extension_authorization_sha256=(
                None if extension is None else extension.authorization_sha256
            ),
        ),
    )
    report_binding = _report_binding(paths.report)
    completion: dict[str, Any] = {
        "schema_version": STAGE3_SCHEMA,
        "completed_utc": utc_now_iso(),
        "step": step,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_score": _best_score_payload(best),
        "thresholds": str(paths.thresholds.resolve()),
        "thresholds_sha256": sha256_file(paths.thresholds),
        **report_binding,
        "threshold_calibration_runs": threshold_payload["calibration_runs"],
        "selected_validation_peak_reserved_bytes": selected_peak_bytes,
        "selected_validation_peak_reserved_fraction": selected_peak_fraction,
        "calibration_peak_reserved_bytes": calibration_peak_bytes,
        "calibration_peak_reserved_fraction": calibration_peak_fraction,
        "mio100_rows_read": 0,
    }
    if extension is not None:
        selected_validation_path = output_dir / "selected_validation.json"
        latest_validation_path = output_dir / "validation_latest.json"
        completion.update(
            {
                "extension_authorization": str(extension.authorization_path.resolve()),
                "extension_authorization_sha256": (extension.authorization_sha256),
                "extension_validation_steps": list(extension.validation_steps),
                "schedule_horizon_steps": extension.schedule_horizon_steps,
                "training_target_step": extension.target_step,
                "selected_validation": str(selected_validation_path.resolve()),
                "selected_validation_sha256": sha256_file(selected_validation_path),
                "validation": str(latest_validation_path.resolve()),
                "validation_sha256": sha256_file(latest_validation_path),
            }
        )
    atomic_write_json(output_dir / "complete.json", completion)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    previous_sigterm = signal.signal(signal.SIGTERM, _sigterm_as_keyboard_interrupt)
    try:
        return run(arguments)
    except KeyboardInterrupt:
        print(
            "Stage3 interrupted; an atomic last checkpoint was requested",
            file=sys.stderr,
        )
        return 130
    except (
        Stage3FinalizationContractError,
        Stage3ContractError,
        FileNotFoundError,
        ValueError,
        FloatingPointError,
    ) as exc:
        print(f"Stage3 refused: {exc}", file=sys.stderr)
        return 3
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
