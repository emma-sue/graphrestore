#!/usr/bin/env python3
"""Train the V7.1 teacher-forced guarded skill bank on primary data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GraphRestoreEpisodeDataset, build_dataloader  # noqa: E402
from src.net import GuardedSkillRestormer  # noqa: E402
from src.training.optimization import WarmupCosineScheduler  # noqa: E402
from src.training.selection import ValidationScore, is_better_checkpoint  # noqa: E402
from src.training.stage0_engine import assert_validation_vram_preflight  # noqa: E402
from src.training.stage1_engine import (  # noqa: E402
    STAGE1_SCHEMA,
    Stage1ContractError,
    Stage1PhaseAwareEMA,
    append_stage1_calibration_history,
    append_jsonl,
    build_stage1_ema,
    build_stage1_optimizer,
    build_stage1_provenance,
    choose_micro_batch,
    configure_reproducibility,
    load_stage0_best_ema_backbone,
    lr_by_role,
    micro_batch_trials_json,
    render_stage1_report,
    resume_stage1_checkpoint,
    save_stage1_checkpoint,
    set_stage1_trainability,
    train_stage1_optimizer_step,
    validate_stage1,
    validate_stage1_config,
    validation_score,
)
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_yaml,
    utc_now_iso,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train GraphRestore V7.1 Stage1. Only primary_train/primary_val "
            "single and Group-A recipes are accepted; MiO100 is never opened."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_skill_bank.yaml"),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        metavar="CHECKPOINT",
        help="resume from CHECKPOINT, or output_dir/last.pth when no path is supplied",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="bounded smoke/truncation override; may not exceed the locked 30000 steps",
    )
    parser.add_argument(
        "--micro_batch",
        type=int,
        choices=(8, 4, 2, 1),
        help="require this step-0 selection after the full ten-step VRAM benchmark",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="checkpoint/log directory (default from the locked config)",
    )
    return parser


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1ContractError(f"{context} must be a mapping")
    return value


def _score_from_checkpoint_metrics(metrics: object) -> ValidationScore | None:
    if not isinstance(metrics, Mapping) or "best_group_a_psnr" not in metrics:
        return None
    score = ValidationScore(
        group_a_psnr=float(metrics["best_group_a_psnr"]),
        group_a_ssim=float(metrics["best_group_a_ssim"]),
        single_psnr=float(metrics["best_single_psnr"]),
        single_ssim=float(metrics["best_single_ssim"]),
        step=int(metrics["best_step"]),
    )
    if not all(
        math.isfinite(value)
        for value in (
            score.group_a_psnr,
            score.group_a_ssim,
            score.single_psnr,
            score.single_ssim,
        )
    ):
        raise Stage1ContractError("resume checkpoint contains non-finite best metrics")
    return score


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


def _run_contract_path(output_dir: Path) -> Path:
    return output_dir / "run_contract.json"


def _read_resume_contract(output_dir: Path) -> Mapping[str, Any]:
    path = _run_contract_path(output_dir)
    if not path.is_file():
        raise Stage1ContractError(f"Stage1 resume lacks frozen run contract: {path}")
    value = load_json(path)
    contract = _mapping(value, "Stage1 run contract")
    if contract.get("schema_version") != STAGE1_SCHEMA:
        raise Stage1ContractError("Stage1 run contract schema mismatch")
    return contract


def _validate_resume_contract(
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if contract.get("provenance") != provenance:
        raise Stage1ContractError("Stage1 resume run contract/provenance mismatch")


def _save_validation(
    *,
    model: GuardedSkillRestormer,
    ema: Stage1PhaseAwareEMA,
    validation_dataset: GraphRestoreEpisodeDataset,
    device: torch.device,
    output_dir: Path,
    report_path: Path,
    step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: Any,
    provenance: Mapping[str, Any],
    best_score: ValidationScore | None,
    validation_gate: Callable[[], None] | None = None,
) -> tuple[ValidationScore, bool, Mapping[str, Any]]:
    try:
        with ema.apply_to(model):
            summary = validate_stage1(
                model,
                validation_dataset,
                device=device,
                use_bf16=True,
            )
    finally:
        # Validation switches the module to eval mode.  Restore the exact
        # Stage1 phase even when full-resolution inference raises.
        model.train()
        set_stage1_trainability(model, step)
    if validation_gate is not None:
        validation_gate()
    current = validation_score(summary, step)
    improved = is_better_checkpoint(current, best_score)
    selected = current if improved else best_score
    assert selected is not None
    metrics = _checkpoint_metrics(current, selected)
    if improved:
        save_stage1_checkpoint(
            output_dir / "best_ema.pth",
            step=step,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            provenance=provenance,
            metrics=metrics,
            model_as_ema=True,
        )
    append_stage1_calibration_history(
        PROJECT_ROOT / "artifacts/metrics/calibration_history.csv",
        step=step,
        summary=summary,
    )
    atomic_write_json(output_dir / "validation_latest.json", summary)
    atomic_write_text(
        report_path,
        render_stage1_report(
            summary,
            step=step,
            best_score=selected,
            checkpoint=output_dir / "best_ema.pth",
        ),
    )
    # Clearing pending_validation_step is the final commit record.  If any
    # metric, best-checkpoint, CSV, JSON, or report write above fails, the
    # pre-validation last.pth remains replayable on resume.
    save_stage1_checkpoint(
        output_dir / "last.pth",
        step=step,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        metrics=metrics,
    )
    return selected, improved, summary


def run(arguments: argparse.Namespace) -> int:
    config_path = _project_path(arguments.config)
    config = _mapping(load_yaml(config_path), "Stage1 config")
    validate_stage1_config(config)
    configured_max_steps = int(config["training"]["max_steps"])
    validation_every = int(config["validation"]["every_steps"])
    requested_max_steps = arguments.max_steps
    if requested_max_steps is not None and not 0 < requested_max_steps <= configured_max_steps:
        raise Stage1ContractError("--max_steps must lie in [1, 30000]")

    resolved_path = _project_path(config["paths"]["resolved_paths"])
    resolved = _mapping(load_yaml(resolved_path), "resolved paths")
    parent_checkpoint = _project_path(config["paths"]["parent_checkpoint"])
    if not parent_checkpoint.is_file():
        raise Stage1ContractError(f"Stage0 best EMA checkpoint is missing: {parent_checkpoint}")
    output_dir = _project_path(arguments.output_dir or config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = _project_path(config["paths"]["report"])

    resume_path: Path | None = None
    resume_contract: Mapping[str, Any] | None = None
    recover_incomplete_initialization = False
    if arguments.resume is not None:
        resume_path = (
            output_dir / "last.pth"
            if arguments.resume == "auto"
            else _project_path(arguments.resume)
        )
        if not resume_path.is_file():
            raise Stage1ContractError(f"Stage1 resume checkpoint is missing: {resume_path}")
        resume_contract = _read_resume_contract(output_dir)
        frozen_runtime = _mapping(
            _mapping(resume_contract.get("provenance"), "resume provenance").get("runtime"),
            "resume runtime",
        )
        frozen_max_steps = int(frozen_runtime["max_steps"])
        frozen_micro_batch = int(frozen_runtime["micro_batch"])
        if requested_max_steps is not None and requested_max_steps != frozen_max_steps:
            raise Stage1ContractError("--max_steps cannot change across resume")
        if arguments.micro_batch is not None and arguments.micro_batch != frozen_micro_batch:
            raise Stage1ContractError("--micro_batch cannot change across resume")
        max_steps = frozen_max_steps
        selected_micro_batch = frozen_micro_batch
    else:
        existing_run_artifacts = [
            path
            for path in (
                _run_contract_path(output_dir),
                output_dir / "last.pth",
                output_dir / "best_ema.pth",
                output_dir / "complete.json",
                output_dir / "train.jsonl",
                output_dir / "validation_latest.json",
            )
            if path.exists()
        ]
        only_contract = existing_run_artifacts == [_run_contract_path(output_dir)]
        if existing_run_artifacts and not only_contract:
            raise Stage1ContractError(
                "refusing to overwrite an existing Stage1 run; use --resume or a new "
                f"--output_dir: {[str(path) for path in existing_run_artifacts]}"
            )
        if only_contract:
            # A host/SIGKILL may land after the frozen run contract was made
            # durable but before the step-0 checkpoint.  This is the sole
            # fresh-style recovery case: later we recompute and exactly compare
            # provenance, then materialize the missing step-0 anchor.
            resume_contract = _read_resume_contract(output_dir)
            frozen_runtime = _mapping(
                _mapping(resume_contract.get("provenance"), "initialization provenance").get(
                    "runtime"
                ),
                "initialization runtime",
            )
            max_steps = int(frozen_runtime["max_steps"])
            selected_micro_batch = int(frozen_runtime["micro_batch"])
            if requested_max_steps is not None and requested_max_steps != max_steps:
                raise Stage1ContractError(
                    "--max_steps differs from incomplete initialization contract"
                )
            if arguments.micro_batch is not None and arguments.micro_batch != selected_micro_batch:
                raise Stage1ContractError(
                    "--micro_batch differs from incomplete initialization contract"
                )
            recover_incomplete_initialization = True
        else:
            max_steps = requested_max_steps or configured_max_steps
            selected_micro_batch = -1

    if not torch.cuda.is_available():
        raise Stage1ContractError("formal Stage1 training requires an available CUDA GPU")
    assert_validation_vram_preflight(PROJECT_ROOT)
    device = torch.device("cuda", torch.cuda.current_device())
    configure_reproducibility(int(config["seed"]))

    model = GuardedSkillRestormer(gradient_checkpointing=False)
    load_report = load_stage0_best_ema_backbone(model, parent_checkpoint)
    model.to(device)
    set_stage1_trainability(model, 0)

    trials = ()
    if resume_path is None and not recover_incomplete_initialization:
        selected_micro_batch, trials = choose_micro_batch(
            model,
            device=device,
            candidates=tuple(config["training"]["micro_batch_candidates"]),
            crop_size=int(config["data"]["crop_size"]),
            # Probe the maximum-memory Stage1 phase: decoder/encoder34 are
            # unfrozen and the differentiable SSIM term is active.  The chosen
            # micro batch is then frozen before real optimizer step 0.
            step=6000,
            required_steps=int(config["runtime"]["vram"]["required_consecutive_no_oom_steps"]),
            maximum_reserved_fraction=float(
                config["runtime"]["vram"]["maximum_peak_reserved_fraction"]
            ),
            effective_batch_size=int(config["training"]["effective_batch_size"]),
        )
        if arguments.micro_batch is not None and arguments.micro_batch != selected_micro_batch:
            raise Stage1ContractError(
                f"requested micro batch {arguments.micro_batch} is not the highest-throughput "
                f"passing selection ({selected_micro_batch})"
            )
    effective_batch_size = int(config["training"]["effective_batch_size"])
    if effective_batch_size % selected_micro_batch:
        raise Stage1ContractError("effective batch must be divisible by micro batch")
    accumulation_steps = effective_batch_size // selected_micro_batch

    learning_rates = config["optimization"]["learning_rates"]
    optimizer = build_stage1_optimizer(
        model,
        skill_lr=float(learning_rates["skill_adapters_and_mixers"]),
        decoder_lr=float(learning_rates["decoder_refinement_rgb_head"]),
        encoder34_lr=float(learning_rates["encoder_level3_level4"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
        fused_if_supported=bool(config["runtime"]["fused_adamw_if_supported"]),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(config["optimization"]["warmup_steps"]),
        # ``--max_steps`` is only a bounded stop override.  The learning-rate
        # trajectory remains the locked 30k schedule so a smoke run does not
        # silently become a different optimizer recipe.
        max_steps=configured_max_steps,
        min_lr=float(config["optimization"]["min_lr"]),
    )
    ema = build_stage1_ema(model, decay=float(config["ema"]["decay"]))

    provenance = build_stage1_provenance(
        config_path=config_path,
        config=config,
        resolved_path=resolved_path,
        resolved=resolved,
        parent_checkpoint=parent_checkpoint,
        micro_batch=selected_micro_batch,
        max_steps=max_steps,
        accumulation_steps=accumulation_steps,
    )
    if resume_contract is None:
        atomic_write_json(
            _run_contract_path(output_dir),
            {
                "schema_version": STAGE1_SCHEMA,
                "created_utc": utc_now_iso(),
                "provenance": provenance,
                "stage0_backbone_load": {
                    "source_tensor_count": load_report.source_tensor_count,
                    "loaded_count": load_report.loaded_count,
                    "missing_count": len(load_report.missing_keys),
                    "missing_prefixes": list(load_report.allowed_missing_prefixes),
                    "unexpected_keys": list(load_report.unexpected_keys),
                    "shape_mismatches": list(load_report.shape_mismatches),
                },
                "micro_batch_trials": micro_batch_trials_json(trials),
            },
        )
    else:
        _validate_resume_contract(resume_contract, provenance)

    training_data_root = Path(str(resolved["training_data_root"])).resolve()
    depth_compat_root = PROJECT_ROOT / "artifacts" / "cache" / "agenticir_depth_compat"
    train_manifest = Path(str(resolved[config["paths"]["train_manifest_key"]])).resolve()
    val_manifest = Path(str(resolved[config["paths"]["val_manifest_key"]])).resolve()
    train_dataset = GraphRestoreEpisodeDataset(
        train_manifest,
        training_data_root,
        depth_compat_root,
        crop_size=int(config["data"]["crop_size"]),
        training=True,
        stage="stage1",
        base_seed=int(config["seed"]),
        agenticir_repo=resolved["agenticir_repo"],
        mioir_repo=resolved["mioir_repo"],
    )
    validation_dataset = GraphRestoreEpisodeDataset(
        val_manifest,
        training_data_root,
        depth_compat_root,
        crop_size=None,
        training=False,
        stage="stage1",
        base_seed=int(config["seed"]),
        agenticir_repo=resolved["agenticir_repo"],
        mioir_repo=resolved["mioir_repo"],
    )
    loader_config = config["data"]["loader"]
    train_loader, sampler = build_dataloader(
        train_dataset,
        batch_size=selected_micro_batch,
        effective_batch_size=effective_batch_size,
        num_samples=configured_max_steps * effective_batch_size,
        stage="stage1",
        base_seed=int(config["seed"]),
        start_step=0,
        num_workers=int(loader_config["num_workers"]),
        persistent_workers=bool(loader_config["persistent_workers"]),
        pin_memory=bool(loader_config["pin_memory"]),
        prefetch_factor=int(loader_config["prefetch_factor"]),
        drop_last=True,
        training=True,
    )
    if sampler is None:
        raise RuntimeError("Stage1 training loader has no sampler")

    step = 0
    best_score: ValidationScore | None = None
    last_score: ValidationScore | None = None
    pending_validation_step: int | None = None
    if resume_path is not None:
        payload = resume_stage1_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_provenance=provenance,
            expected_validation_every=validation_every,
            expected_max_steps=max_steps,
        )
        step = int(payload["step"])
        if step > max_steps:
            raise Stage1ContractError(
                f"resume step {step} exceeds configured max_steps {max_steps}"
            )
        pending_value = payload.get("pending_validation_step")
        if pending_value is not None:
            if isinstance(pending_value, bool) or not isinstance(pending_value, int):
                raise Stage1ContractError("pending_validation_step must be an integer or null")
            if pending_value != step:
                raise Stage1ContractError(
                    "pending_validation_step differs from checkpoint optimizer step"
                )
            pending_validation_step = pending_value
        best_score = _score_from_checkpoint_metrics(payload.get("metrics"))
    else:
        sampler.set_step(0)

    log_path = output_dir / "train.jsonl"
    if pending_validation_step is not None and not (
        pending_validation_step % validation_every == 0
        or pending_validation_step == max_steps
    ):
        raise Stage1ContractError(
            "pending_validation_step is not a locked validation boundary"
        )
    iterator = iter(train_loader)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    maximum_peak_fraction = float(
        config["runtime"]["vram"]["maximum_peak_reserved_fraction"]
    )
    maximum_train_peak_reserved = 0
    maximum_validation_peak_reserved = 0
    validation_in_progress_step: int | None = None
    training_update_in_progress = False
    if resume_path is None:
        # Establish a step-0 recovery anchor before the first optimizer update.
        # A signal inside an optimizer/scheduler/EMA transaction must fall back
        # to this (or the latest validation-boundary last.pth), never serialize
        # a partially committed update.
        save_stage1_checkpoint(
            output_dir / "last.pth",
            step=0,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            provenance=provenance,
            metrics={},
        )
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        append_jsonl(
            log,
            {
                "event": "resume" if resume_path is not None else "start",
                "recovered_incomplete_initialization": recover_incomplete_initialization,
                "utc": utc_now_iso(),
                "step": step,
                "max_steps": max_steps,
                "micro_batch": selected_micro_batch,
                "accumulation_steps": accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "parent_checkpoint_sha256": provenance["parent_checkpoint"]["sha256"],
            },
        )

        def run_validation_boundary(*, replay_pending: bool) -> None:
            nonlocal best_score
            nonlocal last_score
            nonlocal maximum_train_peak_reserved
            nonlocal maximum_validation_peak_reserved
            nonlocal validation_in_progress_step

            validation_in_progress_step = step
            if not replay_pending:
                # First commit a raw, exactly resumable optimizer-boundary
                # snapshot. Full-resolution validation must not put the
                # preceding training interval at risk.
                save_stage1_checkpoint(
                    output_dir / "last.pth",
                    step=step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    metrics=_checkpoint_metrics(last_score, best_score),
                    pending_validation_step=step,
                )
            append_jsonl(
                log,
                {
                    "event": (
                        "replay_pending_validation"
                        if replay_pending
                        else "pre_validation_checkpoint"
                    ),
                    "utc": utc_now_iso(),
                    "step": step,
                    "checkpoint": str(output_dir / "last.pth"),
                },
            )
            torch.cuda.synchronize(device)
            maximum_train_peak_reserved = max(
                maximum_train_peak_reserved,
                int(torch.cuda.max_memory_reserved(device)),
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            validation_peak_reserved = 0

            def validation_gate() -> None:
                nonlocal validation_peak_reserved
                nonlocal maximum_validation_peak_reserved

                torch.cuda.synchronize(device)
                validation_peak_reserved = int(torch.cuda.max_memory_reserved(device))
                maximum_validation_peak_reserved = max(
                    maximum_validation_peak_reserved,
                    validation_peak_reserved,
                )
                validation_peak_fraction = validation_peak_reserved / total_memory
                if validation_peak_fraction > maximum_peak_fraction:
                    raise Stage1ContractError(
                        "Stage1 validation peak reserved fraction "
                        f"{validation_peak_fraction:.4f} exceeded the frozen "
                        f"{maximum_peak_fraction:.2f} ceiling"
                    )

            try:
                best_score, improved, summary = _save_validation(
                    model=model,
                    ema=ema,
                    validation_dataset=validation_dataset,
                    device=device,
                    output_dir=output_dir,
                    report_path=report_path,
                    step=step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    best_score=best_score,
                    validation_gate=validation_gate,
                )
                validation_in_progress_step = None
            finally:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            validation_peak_fraction = validation_peak_reserved / total_memory
            last_score = validation_score(summary, step)
            append_jsonl(
                log,
                {
                    "event": "validation",
                    "utc": utc_now_iso(),
                    "step": step,
                    "improved": improved,
                    "replayed_pending": replay_pending,
                    "peak_reserved_bytes": validation_peak_reserved,
                    "peak_reserved_fraction": validation_peak_fraction,
                    "group_a_psnr": last_score.group_a_psnr,
                    "group_a_ssim": last_score.group_a_ssim,
                    "single_psnr": last_score.single_psnr,
                    "single_ssim": last_score.single_ssim,
                    "pair_isolation": summary["episodes"]["pair_isolation"],
                    "pair_parallel_residual_norm": summary["episodes"]["pair_parallel"][
                        "residual_norm"
                    ],
                    "pair_parallel_active_rate": summary["episodes"]["pair_parallel"][
                        "active_rate"
                    ],
                },
            )

        try:
            if pending_validation_step is not None:
                run_validation_boundary(replay_pending=True)
            while step < max_steps:
                micro_batches = [next(iterator) for _ in range(accumulation_steps)]
                training_update_in_progress = True
                result = train_stage1_optimizer_step(
                    model,
                    micro_batches,
                    optimizer,
                    scheduler,
                    ema,
                    step=step,
                    device=device,
                    gradient_clip_norm=float(config["optimization"]["gradient_clip_norm"]),
                    use_bf16=True,
                    audit_first_backward=resume_path is None and step == 0,
                )
                step += 1
                sampler.mark_consumed_optimizer_step(step)
                training_update_in_progress = False
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                maximum_train_peak_reserved = max(
                    maximum_train_peak_reserved,
                    peak_reserved,
                )
                peak_reserved_fraction = peak_reserved / total_memory
                if peak_reserved_fraction > maximum_peak_fraction:
                    raise Stage1ContractError(
                        f"Stage1 peak reserved fraction {peak_reserved_fraction:.4f} "
                        "exceeded the frozen 0.90 ceiling"
                    )
                log_row = {
                    "event": "train_step",
                    "utc": utc_now_iso(),
                    "step": step,
                    "loss": result.loss,
                    "charbonnier": result.charbonnier,
                    "ssim_loss": result.ssim,
                    "lambda_ssim": result.lambda_ssim,
                    "grad_norm_pre_clip": result.grad_norm,
                    "active_rate": result.active_rate,
                    "residual_norm": result.residual_norm,
                    "images_per_second": result.samples / max(result.seconds, 1.0e-9),
                    "peak_reserved_bytes": peak_reserved,
                    "peak_reserved_fraction": peak_reserved_fraction,
                    "learning_rates": lr_by_role(optimizer),
                }
                if not all(
                    math.isfinite(float(log_row[key]))
                    for key in (
                        "loss",
                        "charbonnier",
                        "ssim_loss",
                        "grad_norm_pre_clip",
                        "active_rate",
                        "residual_norm",
                        "images_per_second",
                        "peak_reserved_fraction",
                    )
                ):
                    raise FloatingPointError("non-finite Stage1 JSONL training record")
                append_jsonl(log, log_row)

                if step % validation_every == 0 or step == max_steps:
                    run_validation_boundary(replay_pending=False)
        except KeyboardInterrupt:
            if not training_update_in_progress:
                save_stage1_checkpoint(
                    output_dir / "last.pth",
                    step=step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    metrics=_checkpoint_metrics(last_score, best_score),
                    pending_validation_step=validation_in_progress_step,
                )
            append_jsonl(
                log,
                {
                    "event": "interrupted",
                    "utc": utc_now_iso(),
                    "step": step,
                    "mid_optimizer_update": training_update_in_progress,
                    "checkpoint_advanced": not training_update_in_progress,
                },
            )
            raise

        append_jsonl(
            log,
            {
                "event": "complete",
                "utc": utc_now_iso(),
                "step": step,
                "maximum_train_peak_reserved_bytes": maximum_train_peak_reserved,
                "maximum_train_peak_reserved_fraction": (
                    maximum_train_peak_reserved / total_memory
                ),
                "maximum_validation_peak_reserved_bytes": maximum_validation_peak_reserved,
                "maximum_validation_peak_reserved_fraction": (
                    maximum_validation_peak_reserved / total_memory
                ),
                "best": None
                if best_score is None
                else {
                    "step": best_score.step,
                    "group_a_psnr": best_score.group_a_psnr,
                    "group_a_ssim": best_score.group_a_ssim,
                    "single_psnr": best_score.single_psnr,
                    "single_ssim": best_score.single_ssim,
                },
            },
        )

    atomic_write_json(
        output_dir / "complete.json",
        {
            "schema_version": "graphrestore-stage1-complete-v1",
            "completed_utc": utc_now_iso(),
            "step": step,
            "maximum_train_peak_reserved_bytes": maximum_train_peak_reserved,
            "maximum_train_peak_reserved_fraction": maximum_train_peak_reserved / total_memory,
            "maximum_validation_peak_reserved_bytes": maximum_validation_peak_reserved,
            "maximum_validation_peak_reserved_fraction": (
                maximum_validation_peak_reserved / total_memory
            ),
            "best_checkpoint": str(output_dir / "best_ema.pth"),
            "last_checkpoint": str(output_dir / "last.pth"),
            "report": str(report_path),
            "provenance_sha256": hashlib.sha256(
                json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return run(arguments)
    except KeyboardInterrupt:
        print("Stage1 interrupted; an atomic last checkpoint was requested", file=sys.stderr)
        return 130
    except (Stage1ContractError, FloatingPointError, FileNotFoundError, ValueError) as exc:
        print(f"Stage1 refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
