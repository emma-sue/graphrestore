#!/usr/bin/env python3
"""Train the frozen V7.1 MiO-StageA baseline on primary single/Group-A recipes."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.samplers import build_dataloader  # noqa: E402
from src.training.ema import ExponentialMovingAverage  # noqa: E402
from src.training.runtime import (  # noqa: E402
    MicroBatchSelection,
    MicroBatchTrial,
    configure_torch_runtime,
    seed_everything,
    select_micro_batch,
)
from src.training.selection import ValidationScore, is_better_checkpoint  # noqa: E402
from src.training.stage0_engine import (  # noqa: E402
    Stage0ContractError,
    Stage0Runtime,
    Stage0RestorationDataset,
    Stage0StepEngine,
    Stage0ValidationResult,
    append_stage0_calibration_history,
    assert_stage0_preflight,
    build_stage0_model,
    build_stage0_optimizer,
    build_stage0_provenance,
    checkpoint_metrics,
    evaluate_primary_val,
    load_and_validate_stage0_config,
    load_stage0_compile_ab_decision,
    resume_stage0_checkpoint,
    save_stage0_checkpoint,
    score_from_checkpoint_metrics,
    write_stage0_report,
)
from src.utils.io import atomic_write_json, utc_now_iso  # noqa: E402
from src.utils.paths import ensure_within, resolve_config_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Stage0 only. The default is the locked 60k run; "
            "--integration_steps 100 runs the mandatory recoverability integration."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--integration_steps", type=int)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--micro_batch", type=int, choices=(8, 4, 2, 1))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--log_every", type=int, default=20)
    return parser


def _json_line(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False))
        handle.write("\n")
        handle.flush()


def _load_resume_runtime(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage0ContractError("resume checkpoint lacks provenance")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, Mapping):
        raise Stage0ContractError("resume checkpoint lacks frozen runtime")
    return dict(runtime)


def _make_dataset(
    *,
    resolved: Mapping[str, Any],
    manifest_key: str,
    crop_size: int | None,
    training: bool,
    seed: int,
) -> Stage0RestorationDataset:
    return Stage0RestorationDataset(
        manifest_path=Path(str(resolved[manifest_key])),
        training_data_root=Path(str(resolved["training_data_root"])),
        depth_compat_root=PROJECT_ROOT / "artifacts/cache/mioir_depth_compat",
        crop_size=crop_size,
        training=training,
        stage="stage0",
        base_seed=seed,
        agenticir_repo=Path(str(resolved["agenticir_repo"])),
        mioir_repo=Path(str(resolved["mioir_repo"])),
    )


def _oom_message(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _probe_runtime(
    *,
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    dataset: Stage0RestorationDataset,
    parent_payload: Mapping[str, Any],
    device: torch.device,
    requested_micro_batch: int | None,
    torch_compile_enabled: bool,
) -> MicroBatchSelection:
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    effective_batch = int(config["training"]["effective_batch_size"])
    seed = int(config["seed"])

    def trial(micro_batch: int, *, checkpointing: bool = False) -> MicroBatchTrial:
        model = optimizer = scheduler = ema = engine = None
        completed_steps = 0
        completed_passes = 0
        started = 0.0
        try:
            seed_everything(seed)
            loader, _ = build_dataloader(
                dataset,
                batch_size=micro_batch,
                effective_batch_size=effective_batch,
                num_samples=micro_batch,
                stage="stage0",
                base_seed=seed,
                start_step=0,
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=2,
                drop_last=True,
                training=True,
            )
            fixed_batch = next(iter(loader))
            model, _ = build_stage0_model(
                parent_payload,
                gradient_checkpointing=checkpointing,
            )
            model.to(device)
            optimizer, scheduler = build_stage0_optimizer(model, config)
            ema = ExponentialMovingAverage(model, decay=float(config["ema"]["decay"]))
            if torch_compile_enabled:
                model = torch.compile(
                    model,
                    backend="inductor",
                    mode="default",
                    fullgraph=False,
                    dynamic=False,
                )
            accumulation = effective_batch // micro_batch
            engine = Stage0StepEngine(
                model,
                optimizer,
                scheduler,
                ema,
                device=device,
                accumulation_steps=accumulation,
                micro_batch=micro_batch,
                gradient_clip_norm=float(config["optimization"]["gradient_clip_norm"]),
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            # Probe the worst locked Stage0 phase: all backbone layers are
            # unfrozen and the differentiable SSIM term is active. Ten full
            # optimizer steps give every candidate the same 80 effective images.
            for probe_step in range(10):
                result = engine.train_optimizer_step(
                    [fixed_batch] * accumulation,
                    step=12_000 + probe_step,
                )
                if not math.isfinite(result.loss):
                    raise FloatingPointError("non-finite probe loss")
                completed_steps += 1
                completed_passes += accumulation
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak = int(torch.cuda.max_memory_reserved(device))
            return MicroBatchTrial(
                micro_batch=micro_batch,
                crop_size=192,
                gradient_checkpointing=checkpointing,
                consecutive_optimizer_steps=completed_steps,
                consecutive_forward_backward=completed_passes,
                images_per_second=(completed_steps * effective_batch) / elapsed,
                peak_reserved_bytes=peak,
                total_memory_bytes=total_memory,
                peak_reserved_fraction=peak / total_memory,
                finite=True,
            )
        except (RuntimeError, FloatingPointError) as exc:
            if not _oom_message(exc):
                raise
            peak = int(torch.cuda.max_memory_reserved(device))
            return MicroBatchTrial(
                micro_batch=micro_batch,
                crop_size=192,
                gradient_checkpointing=checkpointing,
                consecutive_optimizer_steps=completed_steps,
                consecutive_forward_backward=completed_passes,
                images_per_second=0.0,
                peak_reserved_bytes=peak,
                total_memory_bytes=total_memory,
                peak_reserved_fraction=peak / total_memory,
                finite=False,
                oom=True,
                error=str(exc).splitlines()[0][:500],
            )
        finally:
            del engine, ema, scheduler, optimizer, model
            torch.cuda.empty_cache()

    maximum = float(config["runtime"]["vram"]["maximum_peak_reserved_fraction"])
    if requested_micro_batch is not None:
        result = trial(requested_micro_batch)
        return select_micro_batch(
            (result,),
            effective_batch=effective_batch,
            maximum_peak_fraction=maximum,
        )
    observed = [
        trial(int(value)) for value in config["training"]["micro_batch_candidates"]
    ]
    try:
        return select_micro_batch(
            observed,
            effective_batch=effective_batch,
            maximum_peak_fraction=maximum,
        )
    except RuntimeError:
        # The only automatic fallback before changing crop is block-level
        # checkpointing at crop192/micro1.  A crop reduction requires an explicit
        # DEVIATIONS entry and is therefore intentionally not hidden here.
        micro_one = next(item for item in observed if item.micro_batch == 1)
        if not micro_one.oom:
            raise
        checkpointed = trial(1, checkpointing=True)
        return select_micro_batch(
            (*observed, checkpointed),
            effective_batch=effective_batch,
            maximum_peak_fraction=maximum,
        )


def _validation_from_metrics(metrics: Mapping[str, object]) -> Stage0ValidationResult | None:
    required = ("single_psnr", "single_ssim", "group_a_psnr", "group_a_ssim")
    if not all(key in metrics for key in required):
        return None
    return Stage0ValidationResult(
        single_psnr=float(metrics["single_psnr"]),
        single_ssim=float(metrics["single_ssim"]),
        group_a_psnr=float(metrics["group_a_psnr"]),
        group_a_ssim=float(metrics["group_a_ssim"]),
        image_count=1600,
        task_means={},
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.integration_steps is not None and arguments.integration_steps != 100:
        raise Stage0ContractError("V7.1 integration must be exactly 100 optimizer steps")
    if arguments.log_every <= 0:
        raise ValueError("--log_every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage0 training requires the configured CUDA GPU")

    config_path = arguments.config.resolve()
    config, resolved = load_and_validate_stage0_config(config_path)
    assert_stage0_preflight(PROJECT_ROOT)
    configure_torch_runtime(tf32=True, cudnn_benchmark=True)
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    schedule_max_steps = int(config["training"]["max_steps"])
    integration = arguments.integration_steps is not None
    target_step = int(arguments.integration_steps or schedule_max_steps)

    if arguments.output_dir is None:
        output_dir = resolve_config_path(
            config_path, str(config["paths"]["output_dir"]), project_root=PROJECT_ROOT
        )
    else:
        output_dir = (
            arguments.output_dir
            if arguments.output_dir.is_absolute()
            else PROJECT_ROOT / arguments.output_dir
        ).resolve()
    ensure_within(output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.pth"
    best_checkpoint = output_dir / "best_ema.pth"
    if last_checkpoint.exists() and arguments.resume is None:
        raise Stage0ContractError(
            f"refusing to overwrite existing run without --resume: {last_checkpoint}"
        )
    resume_path = arguments.resume.resolve() if arguments.resume is not None else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(resume_path)

    parent_path = Path(str(resolved["stage_a_parent_checkpoint"])).resolve()
    parent_payload = torch.load(
        parent_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(parent_payload, Mapping):
        raise Stage0ContractError("parent checkpoint payload must be a mapping")
    compile_ab = load_stage0_compile_ab_decision(
        PROJECT_ROOT,
        config_path=config_path,
        parent_checkpoint=parent_path,
        primary_train_manifest=resolved["primary_train_manifest"],
        device=device,
    )
    torch_compile_enabled = bool(compile_ab["recommend_torch_compile"])

    training_dataset = _make_dataset(
        resolved=resolved,
        manifest_key="primary_train_manifest",
        crop_size=192,
        training=True,
        seed=seed,
    )

    probe: MicroBatchSelection | None = None
    if resume_path is not None:
        frozen = _load_resume_runtime(resume_path)
        micro_batch = int(frozen["micro_batch"])
        if arguments.micro_batch is not None and arguments.micro_batch != micro_batch:
            raise Stage0ContractError("--micro_batch differs from the frozen resume runtime")
        gradient_checkpointing = bool(frozen["gradient_checkpointing"])
        if int(frozen["target_step"]) != target_step or bool(frozen["integration"]) != integration:
            raise Stage0ContractError("resume target/integration mode differs from checkpoint")
    else:
        probe = _probe_runtime(
            config=config,
            resolved=resolved,
            dataset=training_dataset,
            parent_payload=parent_payload,
            device=device,
            requested_micro_batch=arguments.micro_batch,
            torch_compile_enabled=torch_compile_enabled,
        )
        atomic_write_json(output_dir / "micro_batch_probe.json", probe.to_dict())
        micro_batch = probe.micro_batch
        gradient_checkpointing = probe.gradient_checkpointing

    runtime = Stage0Runtime(
        crop_size=192,
        micro_batch=micro_batch,
        effective_batch=8,
        accumulation_steps=8 // micro_batch,
        gradient_checkpointing=gradient_checkpointing,
        schedule_max_steps=schedule_max_steps,
        target_step=target_step,
        integration=integration,
        torch_compile=torch_compile_enabled,
    )
    seed_everything(seed)
    model, load_report = build_stage0_model(
        parent_payload,
        gradient_checkpointing=runtime.gradient_checkpointing,
    )
    model.to(device)
    optimizer, scheduler = build_stage0_optimizer(model, config)
    ema = ExponentialMovingAverage(model, decay=float(config["ema"]["decay"]))
    if runtime.torch_compile:
        model = torch.compile(
            model,
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
    provenance = build_stage0_provenance(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        config=config,
        resolved=resolved,
        runtime=runtime,
        load_report=load_report,
    )

    start_step = 0
    best_score: ValidationScore | None = None
    last_validation: Stage0ValidationResult | None = None
    resume_payload: Mapping[str, Any] | None = None
    pending_validation_step: int | None = None
    resume_checkpoint_metrics: Mapping[str, Any] | None = None
    if resume_path is not None:
        resume_payload = resume_stage0_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_provenance=provenance,
        )
        start_step = int(resume_payload["step"])
        if not 0 <= start_step <= target_step:
            raise Stage0ContractError("resume step is outside this run")
        metrics = resume_payload.get("metrics")
        if isinstance(metrics, Mapping):
            resume_checkpoint_metrics = metrics
            best_score = score_from_checkpoint_metrics(metrics)
            last_validation = _validation_from_metrics(metrics)
        pending_value = resume_payload.get("pending_validation_step")
        if pending_value is not None:
            if isinstance(pending_value, bool) or not isinstance(pending_value, int):
                raise Stage0ContractError("pending_validation_step must be an integer or null")
            if pending_value != start_step:
                raise Stage0ContractError(
                    "pending_validation_step differs from checkpoint optimizer step"
                )
            pending_validation_step = pending_value

    workers = int(
        arguments.num_workers
        if arguments.num_workers is not None
        else config["data"]["loader"]["num_workers"]
    )
    if workers < 0:
        raise ValueError("--num_workers cannot be negative")
    loader, sampler = build_dataloader(
        training_dataset,
        batch_size=runtime.micro_batch,
        effective_batch_size=runtime.effective_batch,
        num_samples=runtime.target_step * runtime.effective_batch,
        stage="stage0",
        base_seed=seed,
        start_step=start_step,
        num_workers=workers,
        persistent_workers=bool(config["data"]["loader"]["persistent_workers"]),
        pin_memory=bool(config["data"]["loader"]["pin_memory"]),
        prefetch_factor=int(config["data"]["loader"]["prefetch_factor"]),
        drop_last=True,
        training=True,
    )
    if sampler is None:
        raise RuntimeError("Stage0 training loader unexpectedly has no sampler")
    if resume_payload is not None:
        sampler_state = resume_payload.get("sampler_state")
        if not isinstance(sampler_state, dict):
            raise Stage0ContractError("resume checkpoint has no sampler state")
        sampler.load_state_dict(sampler_state)
    iterator = iter(loader)
    engine = Stage0StepEngine(
        model,
        optimizer,
        scheduler,
        ema,
        device=device,
        accumulation_steps=runtime.accumulation_steps,
        micro_batch=runtime.micro_batch,
        gradient_clip_norm=float(config["optimization"]["gradient_clip_norm"]),
    )

    if resume_path is None:
        # A durable step-0 anchor guarantees that a signal/OOM inside the first
        # optimizer transaction can restart exactly instead of leaving a
        # partially initialized output directory with no resumable state.
        save_stage0_checkpoint(
            last_checkpoint,
            step=0,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler_state=sampler.state_dict(consumed_optimizer_step=0),
            provenance=provenance,
            metrics={},
            model_as_ema=False,
        )

    log_path = output_dir / "train.jsonl"
    _json_line(
        log_path,
        {
            "event": "start",
            "utc": utc_now_iso(),
            "start_step": start_step,
            "target_step": target_step,
            "runtime": asdict(runtime),
            "gpu": torch.cuda.get_device_name(device),
            "parent_loaded_tensors": load_report.loaded_count,
            "torch_compile": runtime.torch_compile,
        },
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    cumulative_train_seconds = 0.0
    last_step_result = None
    last_checkpoint_written: Path = resume_path or last_checkpoint
    validation_every = int(config["validation"]["every_steps"])
    save_every = int(config["checkpoint"]["save_every_steps"])
    maximum_peak_fraction = float(config["runtime"]["vram"]["maximum_peak_reserved_fraction"])
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    maximum_train_peak_reserved = 0
    maximum_validation_peak_reserved = 0
    if pending_validation_step is not None and (
        integration or pending_validation_step % validation_every != 0
    ):
        raise Stage0ContractError(
            "pending_validation_step is not a locked formal validation boundary"
        )

    def run_validation_boundary(
        *,
        validation_step: int,
        step_result: Any | None,
        replay_pending: bool,
    ) -> None:
        nonlocal best_score
        nonlocal last_checkpoint_written
        nonlocal last_validation
        nonlocal maximum_train_peak_reserved
        nonlocal maximum_validation_peak_reserved

        def boundary_metrics() -> dict[str, float]:
            values = checkpoint_metrics(
                last_step_result=step_result,
                last_validation=last_validation,
                best_score=best_score,
            )
            if step_result is None and resume_checkpoint_metrics is not None:
                for key in ("train_loss", "train_charbonnier", "train_ssim_loss"):
                    value = resume_checkpoint_metrics.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        values[key] = float(value)
            return values

        if not replay_pending:
            save_stage0_checkpoint(
                last_checkpoint,
                step=validation_step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler_state=sampler.state_dict(
                    consumed_optimizer_step=validation_step
                ),
                provenance=provenance,
                metrics=boundary_metrics(),
                model_as_ema=False,
                pending_validation_step=validation_step,
            )
            last_checkpoint_written = last_checkpoint
        _json_line(
            log_path,
            {
                "event": (
                    "replay_pending_validation"
                    if replay_pending
                    else "pre_validation_checkpoint"
                ),
                "utc": utc_now_iso(),
                "step": validation_step,
                "checkpoint": str(last_checkpoint),
            },
        )
        validation_dataset = _make_dataset(
            resolved=resolved,
            manifest_key="primary_val_manifest",
            crop_size=None,
            training=False,
            seed=seed,
        )
        validation_loader, validation_sampler = build_dataloader(
            validation_dataset,
            batch_size=1,
            effective_batch_size=runtime.effective_batch,
            num_samples=None,
            stage="stage0",
            base_seed=seed,
            start_step=0,
            num_workers=workers,
            persistent_workers=bool(config["data"]["loader"]["persistent_workers"]),
            pin_memory=bool(config["data"]["loader"]["pin_memory"]),
            prefetch_factor=int(config["data"]["loader"]["prefetch_factor"]),
            drop_last=False,
            training=False,
        )
        if validation_sampler is not None:
            raise RuntimeError("primary_val loader unexpectedly has a sampler")

        def progress(done: int, total: int) -> None:
            if done % 50 == 0 or done == total:
                _json_line(
                    log_path,
                    {
                        "event": "validation_progress",
                        "utc": utc_now_iso(),
                        "step": validation_step,
                        "done": done,
                        "total": total,
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
        try:
            with ema.apply_to(model):
                last_validation = evaluate_primary_val(
                    model,
                    validation_dataset,
                    device=device,
                    dataloader=validation_loader,
                    progress=progress,
                )
            torch.cuda.synchronize(device)
            validation_peak_reserved = int(torch.cuda.max_memory_reserved(device))
            maximum_validation_peak_reserved = max(
                maximum_validation_peak_reserved,
                validation_peak_reserved,
            )
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        validation_peak_fraction = validation_peak_reserved / total_memory
        if validation_peak_fraction > maximum_peak_fraction:
            raise Stage0ContractError(
                f"validation peak reserved fraction {validation_peak_fraction:.4f} "
                f"exceeded locked ceiling {maximum_peak_fraction:.4f}"
            )
        assert last_validation is not None
        validation_artifact = {
            "schema_version": "graphrestore-stage0-primary-val-v1",
            "protocol_id": "agenticir_official_parity",
            "created_utc": utc_now_iso(),
            "step": validation_step,
            "replayed_pending": replay_pending,
            "peak_reserved_bytes": validation_peak_reserved,
            "peak_reserved_fraction": validation_peak_fraction,
            **last_validation.to_dict(),
        }
        atomic_write_json(
            PROJECT_ROOT
            / "artifacts/metrics"
            / f"stage0_primary_val_step_{validation_step:06d}.json",
            validation_artifact,
        )
        append_stage0_calibration_history(
            PROJECT_ROOT / "artifacts/metrics/calibration_history.csv",
            step=validation_step,
            result=last_validation,
        )
        candidate = last_validation.selection_score(validation_step)
        if is_better_checkpoint(candidate, best_score):
            best_score = candidate
            save_stage0_checkpoint(
                best_checkpoint,
                step=validation_step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler_state=sampler.state_dict(
                    consumed_optimizer_step=validation_step
                ),
                provenance=provenance,
                metrics=boundary_metrics(),
                model_as_ema=True,
            )
            _json_line(
                log_path,
                {
                    "event": "best_ema",
                    "utc": utc_now_iso(),
                    "step": validation_step,
                    "checkpoint": str(best_checkpoint),
                    **last_validation.to_dict(),
                },
            )
        # The raw checkpoint is the validation transaction's final commit:
        # pending_validation_step becomes null only after every required metric
        # and selection artifact above is durable.
        save_stage0_checkpoint(
            last_checkpoint,
            step=validation_step,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler_state=sampler.state_dict(consumed_optimizer_step=validation_step),
            provenance=provenance,
            metrics=boundary_metrics(),
            model_as_ema=False,
        )
        last_checkpoint_written = last_checkpoint
        _json_line(
            log_path,
            {
                "event": "validation_committed",
                "utc": utc_now_iso(),
                "step": validation_step,
                "checkpoint": str(last_checkpoint),
                "peak_reserved_bytes": validation_peak_reserved,
                "peak_reserved_fraction": validation_peak_fraction,
            },
        )

    if pending_validation_step is not None:
        run_validation_boundary(
            validation_step=pending_validation_step,
            step_result=None,
            replay_pending=True,
        )

    for step in range(start_step, target_step):
        step_started = time.perf_counter()
        batches = [next(iterator) for _ in range(runtime.accumulation_steps)]
        last_step_result = engine.train_optimizer_step(batches, step=step)
        sampler.mark_consumed_optimizer_step(last_step_result.step)
        torch.cuda.synchronize(device)
        step_elapsed = time.perf_counter() - step_started
        cumulative_train_seconds += step_elapsed
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        maximum_train_peak_reserved = max(maximum_train_peak_reserved, peak_reserved)
        peak_fraction = peak_reserved / total_memory
        if peak_fraction > maximum_peak_fraction:
            raise Stage0ContractError(
                f"peak reserved fraction {peak_fraction:.4f} exceeded locked "
                f"ceiling {maximum_peak_fraction:.4f}"
            )
        if (
            last_step_result.step == start_step + 1
            or last_step_result.step % arguments.log_every == 0
            or last_step_result.step == target_step
        ):
            processed = (last_step_result.step - start_step) * runtime.effective_batch
            _json_line(
                log_path,
                {
                    "event": "train_step",
                    "utc": utc_now_iso(),
                    "step": last_step_result.step,
                    "loss": last_step_result.loss,
                    "charbonnier": last_step_result.charbonnier,
                    "ssim_loss": last_step_result.ssim_loss,
                    "lambda_ssim": last_step_result.lambda_ssim,
                    "grad_norm": last_step_result.grad_norm,
                    "step_images_per_second": runtime.effective_batch / step_elapsed,
                    "run_images_per_second": processed / cumulative_train_seconds,
                    "peak_reserved_bytes": peak_reserved,
                    "peak_reserved_fraction": peak_fraction,
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                },
            )

        due_validation = not integration and last_step_result.step % validation_every == 0
        validation_committed = False
        if due_validation:
            run_validation_boundary(
                validation_step=last_step_result.step,
                step_result=last_step_result,
                replay_pending=False,
            )
            validation_committed = True

        due_save = last_step_result.step % save_every == 0
        if (due_save or last_step_result.step == target_step) and not validation_committed:
            metrics = checkpoint_metrics(
                last_step_result=last_step_result,
                last_validation=last_validation,
                best_score=best_score,
            )
            save_stage0_checkpoint(
                last_checkpoint,
                step=last_step_result.step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler_state=sampler.state_dict(
                    consumed_optimizer_step=last_step_result.step
                ),
                provenance=provenance,
                metrics=metrics,
                model_as_ema=False,
            )
            last_checkpoint_written = last_checkpoint
            _json_line(
                log_path,
                {
                    "event": "checkpoint",
                    "utc": utc_now_iso(),
                    "step": last_step_result.step,
                    "checkpoint": str(last_checkpoint),
                },
            )

    if last_step_result is None:
        # A fully completed --resume is a successful idempotent verification.
        completed_step = start_step
    else:
        completed_step = last_step_result.step
    torch.cuda.synchronize(device)
    maximum_train_peak_reserved = max(
        maximum_train_peak_reserved,
        int(torch.cuda.max_memory_reserved(device)),
    )
    processed = max(0, completed_step - start_step) * runtime.effective_batch
    throughput = (
        processed / max(cumulative_train_seconds, 1.0e-12) if processed else 0.0
    )
    peak_reserved = max(maximum_train_peak_reserved, maximum_validation_peak_reserved)
    peak_fraction = peak_reserved / total_memory
    report_path = (
        output_dir / "INTEGRATION_REPORT.md"
        if integration
        else resolve_config_path(
            config_path, str(config["paths"]["report"]), project_root=PROJECT_ROOT
        )
    )
    write_stage0_report(
        report_path,
        step=completed_step,
        runtime=runtime,
        checkpoint=last_checkpoint_written,
        validation=last_validation,
        peak_reserved_fraction=peak_fraction,
        images_per_second=throughput,
    )
    summary = {
        "schema_version": "graphrestore-stage0-run-v1",
        "protocol_id": config["protocol_id"],
        "created_utc": utc_now_iso(),
        "integration": integration,
        "completed_step": completed_step,
        "target_step": target_step,
        "runtime": asdict(runtime),
        "finite": True,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_fraction": peak_fraction,
        "maximum_train_peak_reserved_bytes": maximum_train_peak_reserved,
        "maximum_train_peak_reserved_fraction": maximum_train_peak_reserved / total_memory,
        "maximum_validation_peak_reserved_bytes": maximum_validation_peak_reserved,
        "maximum_validation_peak_reserved_fraction": (
            maximum_validation_peak_reserved / total_memory
        ),
        "images_per_second": throughput,
        "last_checkpoint": str(last_checkpoint_written),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint.is_file() else None,
        "validation": last_validation.to_dict() if last_validation is not None else None,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    _json_line(log_path, {"event": "complete", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
