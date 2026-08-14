#!/usr/bin/env python3
"""Train the approved V7.1 Stage3 planner while freezing the Stage1 executor."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GraphRestoreEpisodeDataset, build_dataloader  # noqa: E402
from src.training.optimization import WarmupCosineScheduler  # noqa: E402
from src.training.selection import ValidationScore, is_better_checkpoint  # noqa: E402
from src.training.stage3_engine import (  # noqa: E402
    STAGE3_SCHEMA,
    Stage3ContractError,
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
    freeze_presence_thresholds,
    load_relation_records,
    load_stage3_best_ema,
    prepare_stage3_supervision_batch,
    resume_stage3_checkpoint,
    save_stage3_checkpoint,
    select_stage3_micro_batch,
    train_stage3_optimizer_step,
    validate_stage3,
    validate_stage3_approval,
    validation_score,
)
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.io import atomic_write_json, atomic_write_text, load_json, utc_now_iso  # noqa: E402


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
    return parser


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


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


def _checkpoint_metrics(
    current: ValidationScore | None,
    best: ValidationScore | None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    if current is not None:
        result.update(
            {
                "group_a_psnr": current.group_a_psnr,
                "group_a_ssim": current.group_a_ssim,
                "single_psnr": current.single_psnr,
                "single_ssim": current.single_ssim,
                "validation_step": float(current.step),
            }
        )
    if best is not None:
        result.update(
            {
                "best_group_a_psnr": best.group_a_psnr,
                "best_group_a_ssim": best.group_a_ssim,
                "best_single_psnr": best.single_psnr,
                "best_single_ssim": best.single_ssim,
                "best_step": float(best.step),
            }
        )
    return result


def _render_report(
    summary: Mapping[str, Any],
    *,
    best: ValidationScore,
    checkpoint: Path,
    thresholds: Path,
) -> str:
    restoration = summary["restoration"]
    relation = summary["relation"]
    guard = summary["guard"]
    graph = summary["graph"]
    return (
        "# Stage3 Planner and Guard\n\n"
        f"- protocol: `{summary['protocol_id']}`\n"
        f"- selected step: {best.step}\n"
        f"- best checkpoint: `{checkpoint}`\n"
        f"- frozen thresholds: `{thresholds}`\n"
        "- data: primary train/val single + Group A; relation validation uses interaction_val only\n"
        "- MiO100 / Group B / Group C rows read: 0\n"
        "- checkpoint rank: restoration only; guard diagnostics do not alter rank\n\n"
        "## Restoration (fixed presence threshold 0.50)\n\n"
        f"- single PSNR/SSIM: {restoration['single']['psnr']:.6f} / {restoration['single']['ssim']:.8f}\n"
        f"- Group A PSNR/SSIM: {restoration['group_a']['psnr']:.6f} / {restoration['group_a']['ssim']:.8f}\n\n"
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


def run(arguments: argparse.Namespace) -> int:
    # HARD ORDERING: this complete file/hash approval audit happens before the
    # first CUDA query, checkpoint tensor load, dataset construction, or pixel read.
    paths = validate_stage3_approval(
        _project_path(arguments.config),
        project_root=PROJECT_ROOT,
        output_dir=arguments.output_dir,
        require_orchestrator_running=True,
        allow_failed_resume=arguments.resume is not None,
    )

    # Semantic label audit remains CPU/file-only and precedes GPU reservation.
    approved_parent_sha = paths.approval.bindings["stage1_checkpoint"]["sha256"]
    train_relations = load_relation_records(
        paths.relation_train,
        split="train",
        parent_checkpoint_sha256=approved_parent_sha,
        interaction_manifest_sha256=paths.approval.bindings["interaction_train_manifest"]["sha256"],
    )
    val_relations = load_relation_records(
        paths.relation_val,
        split="val",
        parent_checkpoint_sha256=approved_parent_sha,
        interaction_manifest_sha256=paths.approval.bindings["interaction_val_manifest"]["sha256"],
    )
    assert_relation_clean_disjoint(train_relations, val_relations)

    output_dir = paths.output_dir
    resume_path: Path | None = None
    run_contract_path = output_dir / "run_contract.json"
    if arguments.resume is not None:
        resume_path = (
            output_dir / "last.pth"
            if arguments.resume == "auto"
            else _project_path(arguments.resume)
        )
        if not resume_path.is_file() or not run_contract_path.is_file():
            raise Stage3ContractError("Stage3 resume checkpoint/run contract is missing")
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
        if arguments.micro_batch is not None and arguments.micro_batch != selected_micro:
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
            )
            if path.exists()
        ]
        if stale:
            raise Stage3ContractError(
                f"refusing fresh Stage3 over existing run artifacts; use --resume: {stale}"
            )

    if not torch.cuda.is_available():
        raise Stage3ContractError("formal Stage3 requires an available CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    configure_stage3_reproducibility(int(paths.config["seed"]))
    model, parent_report = build_stage3_model(paths, device=device)

    if resume_path is None:
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
    max_steps = int(paths.config["training"]["max_steps"])
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(paths.config["optimization"]["warmup_steps"]),
        max_steps=max_steps,
        min_lr=float(paths.config["optimization"]["min_lr"]),
    )
    ema = Stage3PlannerEMA(model, decay=float(paths.config["ema"]["decay"]))
    provenance = build_stage3_provenance(
        paths,
        parent_report,
        micro_batch=selected_micro,
        accumulation_steps=accumulation_steps,
        max_steps=max_steps,
    )
    if resume_path is not None:
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
                    "initialized_planner_key_count": len(parent_report.initialized_planner_keys),
                },
                "micro_batch_trials": [asdict(trial) for trial in trials],
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
        num_samples=max_steps * effective_batch,
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

    step = 0
    best: ValidationScore | None = None
    last_score: ValidationScore | None = None
    if resume_path is not None:
        payload = resume_stage3_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_provenance=provenance,
        )
        step = int(payload["step"])
        best = _score_from_metrics(payload.get("metrics"))
    else:
        sampler.set_step(0)
    iterator = iter(train_loader)
    validation_every = int(paths.config["runtime"]["validation_every_steps"])
    log_path = output_dir / "train.jsonl"
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        _append_jsonl(
            log,
            {
                "event": "resume" if resume_path else "start",
                "utc": utc_now_iso(),
                "step": step,
                "micro_batch": selected_micro,
                "accumulation_steps": accumulation_steps,
                "approval_sha256": paths.approval.approval_sha256,
            },
        )
        try:
            while step < max_steps:
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
                    gradient_clip_norm=float(paths.config["optimization"]["gradient_clip_norm"]),
                    use_bf16=True,
                    audit_gradients=resume_path is None and step == 0,
                )
                step += 1
                sampler.mark_consumed_optimizer_step(step)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                _append_jsonl(
                    log,
                    {
                        "event": "train_step",
                        "utc": utc_now_iso(),
                        "step": step,
                        **asdict(result),
                        "images_per_second": result.samples / max(result.seconds, 1e-9),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "peak_reserved_fraction": torch.cuda.max_memory_reserved(device) / total_memory,
                    },
                )
                if step % validation_every == 0 or step == max_steps:
                    with ema.apply_to(model):
                        summary = validate_stage3(
                            model,
                            validation_dataset,
                            val_relations,
                            device=device,
                            use_bf16=True,
                            presence_threshold=0.5,
                        )
                    score = validation_score(summary, step)
                    improved = is_better_checkpoint(score, best)
                    if improved:
                        best = score
                    last_score = score
                    metrics = _checkpoint_metrics(score, best)
                    save_stage3_checkpoint(
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
                    if improved:
                        save_stage3_checkpoint(
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
                    atomic_write_json(output_dir / "validation_latest.json", summary)
                    append_calibration_history(
                        paths.calibration_history,
                        calibration_history_row(summary, step),
                    )
                    _append_jsonl(
                        log,
                        {
                            "event": "validation",
                            "utc": utc_now_iso(),
                            "step": step,
                            "improved": improved,
                            **calibration_history_row(summary, step),
                        },
                    )
        except KeyboardInterrupt:
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
            )
            _append_jsonl(log, {"event": "interrupted", "utc": utc_now_iso(), "step": step})
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
    selected_summary = validate_stage3(
        selected_model,
        validation_dataset,
        val_relations,
        device=device,
        use_bf16=True,
        presence_threshold=0.5,
    )
    atomic_write_json(output_dir / "selected_validation.json", selected_summary)
    probabilities, targets = collect_primary_val_presence(
        selected_model, validation_dataset, device=device, use_bf16=True
    )
    calibration = calibrate_presence_thresholds(probabilities, targets)
    threshold_payload = freeze_presence_thresholds(
        paths.thresholds,
        calibration,
        primary_val_manifest=paths.val_manifest,
        selected_checkpoint=best_path,
        approval_sha256=paths.approval.approval_sha256,
    )
    atomic_write_text(
        paths.report,
        _render_report(
            selected_summary,
            best=best,
            checkpoint=best_path,
            thresholds=paths.thresholds,
        ),
    )
    atomic_write_json(
        output_dir / "complete.json",
        {
            "schema_version": STAGE3_SCHEMA,
            "completed_utc": utc_now_iso(),
            "step": step,
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "thresholds": str(paths.thresholds),
            "thresholds_sha256": sha256_file(paths.thresholds),
            "threshold_calibration_runs": threshold_payload["calibration_runs"],
            "mio100_rows_read": 0,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return run(arguments)
    except KeyboardInterrupt:
        print("Stage3 interrupted; an atomic last checkpoint was requested", file=sys.stderr)
        return 130
    except (Stage3ContractError, FileNotFoundError, ValueError, FloatingPointError) as exc:
        print(f"Stage3 refused: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
