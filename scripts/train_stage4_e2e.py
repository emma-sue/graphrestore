#!/usr/bin/env python3
"""Train V7.1 Full Guarded GraphRestore after the explicit Stage3 approval."""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.episode_dataset import GraphRestoreEpisodeDataset  # noqa: E402
from src.net import GraphRestore  # noqa: E402
from src.training.ema import ExponentialMovingAverage  # noqa: E402
from src.training.optimization import WarmupCosineScheduler  # noqa: E402
from src.training.selection import ValidationScore, is_better_checkpoint  # noqa: E402
from src.training.stage3_engine import (  # noqa: E402
    CALIBRATION_COLUMNS,
    Stage3ContractError,
    append_calibration_history as append_shared_calibration_history,
    validate_stage3_approval as validate_full_stage3_approval_chain,
)
from src.training.stage4_engine import (  # noqa: E402
    STAGE4_SCHEMA,
    Stage4ContractError,
    Stage4EpisodeDataset,
    Stage4EpisodeSampler,
    append_jsonl,
    build_stage4_optimizer,
    build_stage4_provenance,
    choose_stage4_micro_batch,
    load_presence_thresholds,
    load_relation_records,
    load_stage3_best_ema,
    lr_by_role,
    resume_stage4_checkpoint,
    save_stage4_checkpoint,
    set_stage4_trainability,
    stage4_validation_score,
    train_stage4_optimizer_step,
    validate_stage3_approval,
    validate_stage4,
    validate_stage4_config,
)
from src.utils.hashing import sha256_file  # noqa: E402
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
            "Train GraphRestore V7.1 Stage4 on primary single/Group-A recipes. "
            "The entry point rejects execution without the persisted Stage3 approval."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage4_graphrestore_e2e.yaml"),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        metavar="CHECKPOINT",
        help="resume Stage4 from CHECKPOINT, or output_dir/last.pth",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="post-approval bounded run limit; schedule remains the locked 40000-step schedule",
    )
    parser.add_argument(
        "--micro_batch",
        type=int,
        choices=(2, 1),
        help="assert (not override) the measured pre-step0 Stage4 selection",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="checkpoint/log directory (default is the locked config path)",
    )
    return parser


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage4ContractError(f"{field} must be a mapping")
    return value


def _seed_worker(_: int) -> None:
    worker = torch.utils.data.get_worker_info()
    if worker is None:
        return
    seed = int(torch.initial_seed() % 2**32)
    random.seed(seed)
    np.random.seed(seed)
    setter = getattr(worker.dataset, "set_worker_seed", None)
    if callable(setter):
        setter(seed)


def _configure_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _extract_pair_prior(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("pair_prior")
    if not isinstance(value, Mapping):
        raise Stage4ContractError("pair_prior.json lacks compiler pair_prior")
    return value


def _extract_global_priority(payload: Mapping[str, Any]) -> Mapping[str, float]:
    value = payload.get("priority")
    if not isinstance(value, Mapping) or set(value) != set(
        (
            "noise",
            "motion_blur",
            "defocus_blur",
            "jpeg_artifact",
            "rain",
            "haze",
            "low_light",
            "low_resolution",
        )
    ):
        raise Stage4ContractError("global_priority.json lacks eight fitted priorities")
    return {str(key): float(item) for key, item in value.items()}


def _contract_path(output_dir: Path) -> Path:
    return output_dir / "run_contract.json"


def _checkpoint_best_score(payload: Mapping[str, Any]) -> ValidationScore | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or "best_group_a_psnr" not in metrics:
        return None
    result = ValidationScore(
        group_a_psnr=float(metrics["best_group_a_psnr"]),
        group_a_ssim=float(metrics["best_group_a_ssim"]),
        single_psnr=float(metrics["best_single_psnr"]),
        single_ssim=float(metrics["best_single_ssim"]),
        step=int(metrics["best_step"]),
    )
    if not all(
        math.isfinite(value)
        for value in (
            result.group_a_psnr,
            result.group_a_ssim,
            result.single_psnr,
            result.single_ssim,
        )
    ):
        raise Stage4ContractError("resume checkpoint contains non-finite best metrics")
    return result


def _checkpoint_metrics(
    current: ValidationScore | None,
    best: ValidationScore | None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if current is not None:
        values.update(
            {
                "group_a_psnr": current.group_a_psnr,
                "group_a_ssim": current.group_a_ssim,
                "single_psnr": current.single_psnr,
                "single_ssim": current.single_ssim,
                "validation_step": float(current.step),
            }
        )
    if best is not None:
        values.update(
            {
                "best_group_a_psnr": best.group_a_psnr,
                "best_group_a_ssim": best.group_a_ssim,
                "best_single_psnr": best.single_psnr,
                "best_single_ssim": best.single_ssim,
                "best_step": float(best.step),
            }
        )
    return values


def _append_calibration_history(
    path: Path,
    *,
    step: int,
    summary: Mapping[str, Any],
) -> None:
    single = summary["single_equal_task_mean"]
    group = summary["group_a_equal_combination_mean"]
    diag = summary["diagnostics"]
    clean = diag["clean_misuse"]
    wrong = diag["wrong_skill_identity"]
    row = {
        "step": step,
        "single_psnr": single["psnr"],
        "single_ssim": single["ssim"],
        "group_a_psnr": group["psnr"],
        "group_a_ssim": group["ssim"],
        "planner_macro_f1": diag["planner_macro_f1"],
        "relation_accuracy": diag["relation_accuracy"],
        "parallel_precision": diag["parallel_precision"],
        "parallel_recall": diag["parallel_recall"],
        "pre_cycle_rate": diag["pre_cycle_rate"],
        "dropped_edge_rate": diag["dropped_edge_rate"],
        "guard_spearman_rain": diag["guard_spearman_rain"],
        "guard_spearman_haze": diag["guard_spearman_haze"],
        "guard_mae_rain": diag["guard_mae_rain"],
        "guard_mae_haze": diag["guard_mae_haze"],
        "guard_std_rain": diag["guard_std_rain"],
        "guard_std_haze": diag["guard_std_haze"],
        "guard_high_frac_rain": diag["guard_high_frac_rain"],
        "guard_high_frac_haze": diag["guard_high_frac_haze"],
        "clean_misuse_psnr": clean["psnr"],
        "clean_misuse_ssim": clean["ssim"],
        "clean_misuse_residual_norm": clean["residual_norm"],
        "wrong_skill_identity_psnr": wrong["psnr"],
        "wrong_skill_identity_ssim": wrong["ssim"],
        "wrong_skill_residual_norm": wrong["residual_norm"],
        "reentry_request_rate": diag["reentry_request_rate"],
        "unexpected_skill_activation_rate": diag["unexpected_skill_activation_rate"],
        "mean_program_levels": diag["mean_program_levels"],
    }
    if tuple(row) != CALIBRATION_COLUMNS:
        raise Stage4ContractError(
            "Stage4 calibration row drifted from the shared 28-column schema"
        )
    # Stage3 owns the shared append primitive.  It validates any existing
    # header before writing, so a cross-stage schema mismatch cannot silently
    # shift values into the wrong columns.
    append_shared_calibration_history(path, row)


def _render_report(
    summary: Mapping[str, Any],
    *,
    step: int,
    best: ValidationScore,
    checkpoint: Path,
) -> str:
    group = summary["group_a_equal_combination_mean"]
    single = summary["single_equal_task_mean"]
    diag = summary["diagnostics"]
    return (
        "# Stage4 Full Guarded GraphRestore\n\n"
        f"- Protocol: `{summary['protocol_id']}`\n"
        f"- Validation step: {step}\n"
        "- Data: frozen primary_val singles + Group A only; MiO100 B/C were not read\n"
        "- Runtime: compile-once DAG, Kmax_test=3, no skill re-entry\n"
        f"- Selected EMA: `{checkpoint}`\n"
        f"- Group-A PSNR/SSIM: {group['psnr']:.6f} / {group['ssim']:.8f}\n"
        f"- Single PSNR/SSIM: {single['psnr']:.6f} / {single['ssim']:.8f}\n"
        f"- Best Group-A PSNR/SSIM: {best.group_a_psnr:.6f} / {best.group_a_ssim:.8f}\n"
        f"- Planner macro-F1: {diag['planner_macro_f1']:.6f}\n"
        f"- Non-ambiguous relation accuracy: {diag['relation_accuracy']:.6f}\n"
        f"- Re-entry request rate (diagnostic only): {diag['reentry_request_rate']:.8f}\n"
    )


def run(arguments: argparse.Namespace) -> int:
    config_path = _project_path(arguments.config)
    config = _mapping(load_yaml(config_path), "Stage4 config")
    validate_stage4_config(config)
    configured_steps = int(config["training"]["max_steps"])
    if arguments.max_steps is not None and not 0 < arguments.max_steps <= configured_steps:
        raise Stage4ContractError("--max_steps must lie in [1,40000]")

    resolved_path = _project_path(config["paths"]["resolved_paths"])
    resolved = _mapping(load_yaml(resolved_path), "resolved paths")
    output_dir = _project_path(arguments.output_dir or config["paths"]["output_dir"])
    report_path = _project_path(config["paths"]["report"])
    calibration_path = _project_path(config["paths"]["calibration_history"])
    stage1_checkpoint = _project_path(config["paths"]["stage1_checkpoint"])
    stage3_checkpoint = _project_path(config["paths"]["stage3_checkpoint"])
    approval_path = _project_path(config["paths"]["required_approval"])
    thresholds_path = _project_path(config["paths"]["thresholds"])
    pair_prior_path = _project_path(config["paths"]["pair_prior"])
    priority_path = _project_path(config["paths"]["global_priority"])
    relation_train_path = PROJECT_ROOT / "artifacts/interaction_labels/group_a_relations_train.jsonl"
    relation_val_path = PROJECT_ROOT / "artifacts/interaction_labels/group_a_relations_val.jsonl"
    effect_profiles_path = PROJECT_ROOT / "artifacts/interaction_labels/skill_effect_profiles.json"
    stage2_decision_path = PROJECT_ROOT / "artifacts/interaction_labels/stage2_decision.json"

    # This is the first action with scientific state: refuse before any CUDA
    # allocation if any approved Stage2/config/manifest/checkpoint binding is
    # absent or stale.  Stage4 runs under the orchestrator's STAGE4_RUNNING
    # status, so only the Stage3-specific live-status assertion is disabled.
    stage3_paths = validate_full_stage3_approval_chain(
        PROJECT_ROOT / "configs/stage3_planner.yaml",
        project_root=PROJECT_ROOT,
        require_orchestrator_running=False,
    )
    if stage3_paths.approval.approval_path != approval_path:
        raise Stage4ContractError("Stage3/Stage4 approval paths disagree")
    expected_stage3_paths = {
        "stage1 checkpoint": (stage3_paths.executor_checkpoint, stage1_checkpoint),
        "thresholds": (stage3_paths.thresholds, thresholds_path),
        "pair prior": (stage3_paths.pair_prior, pair_prior_path),
        "global priority": (stage3_paths.global_priority, priority_path),
        "relation train": (stage3_paths.relation_train, relation_train_path),
        "relation val": (stage3_paths.relation_val, relation_val_path),
        "effect profiles": (stage3_paths.effect_profiles, effect_profiles_path),
        "Stage2 decision": (stage3_paths.stage2_decision, stage2_decision_path),
    }
    for label, (approved_path, configured_path) in expected_stage3_paths.items():
        if approved_path.resolve() != configured_path.resolve():
            raise Stage4ContractError(f"approved/configured {label} paths disagree")
    approval = validate_stage3_approval(
        approval_path, stage2_decision_path=stage2_decision_path
    )
    approval_sha = sha256_file(approval_path)
    required = (
        stage1_checkpoint,
        stage3_checkpoint,
        pair_prior_path,
        priority_path,
        relation_train_path,
        relation_val_path,
        effect_profiles_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise Stage4ContractError(f"missing frozen Stage4 parent artifacts: {missing}")

    pair_prior_payload = _mapping(load_json(pair_prior_path), "pair prior")
    priority_payload = _mapping(load_json(priority_path), "global priority")
    model = GraphRestore(
        gradient_checkpointing=True,
        pair_prior=_extract_pair_prior(pair_prior_payload),
        global_priority=_extract_global_priority(priority_payload),
        max_active_skills=3,
        kmax_train=2,
        kmax_test=3,
        allow_skill_reentry=False,
        max_calls_per_skill=1,
    )
    snapshot = load_stage3_best_ema(
        stage3_checkpoint,
        model=model,
        approval_sha256=approval_sha,
        required_artifact_hashes=tuple(
            sha256_file(path)
            for path in (
                stage1_checkpoint,
                pair_prior_path,
                priority_path,
                relation_train_path,
                relation_val_path,
                effect_profiles_path,
            )
        ),
    )
    thresholds, _ = load_presence_thresholds(
        thresholds_path,
        stage3_checkpoint_sha256=snapshot.checkpoint_sha256,
        stage3_approval_sha256=approval_sha,
    )
    model.set_presence_thresholds(thresholds)

    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path: Path | None = None
    resume_contract: Mapping[str, Any] | None = None
    if arguments.resume is not None:
        resume_path = (
            output_dir / "last.pth"
            if arguments.resume == "auto"
            else _project_path(arguments.resume)
        )
        if not resume_path.is_file():
            raise Stage4ContractError(f"Stage4 resume checkpoint missing: {resume_path}")
        contract_value = load_json(_contract_path(output_dir))
        resume_contract = _mapping(contract_value, "Stage4 run contract")
        if resume_contract.get("schema_version") != STAGE4_SCHEMA:
            raise Stage4ContractError("Stage4 run contract schema mismatch")
        frozen = _mapping(resume_contract.get("provenance"), "Stage4 provenance")["runtime"]
        frozen = _mapping(frozen, "Stage4 frozen runtime")
        max_steps = int(frozen["max_steps"])
        micro_batch = int(frozen["micro_batch"])
        if arguments.max_steps is not None and arguments.max_steps != max_steps:
            raise Stage4ContractError("--max_steps cannot change across Stage4 resume")
        if arguments.micro_batch is not None and arguments.micro_batch != micro_batch:
            raise Stage4ContractError("--micro_batch cannot change across Stage4 resume")
    else:
        collisions = [
            path
            for path in (
                _contract_path(output_dir),
                output_dir / "last.pth",
                output_dir / "best_ema.pth",
                output_dir / "complete.json",
                output_dir / "train.jsonl",
            )
            if path.exists()
        ]
        if collisions:
            raise Stage4ContractError(
                f"refusing to overwrite Stage4 artifacts; use --resume: {collisions}"
            )
        max_steps = arguments.max_steps or configured_steps
        micro_batch = -1

    if not torch.cuda.is_available():
        raise Stage4ContractError("formal Stage4 requires an available CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    _configure_runtime(int(config["seed"]))
    model.to(device)
    set_stage4_trainability(model)
    trials = ()
    if resume_path is None:
        micro_batch, trials = choose_stage4_micro_batch(
            model,
            device=device,
            candidates=tuple(config["data"]["micro_batch_candidates"]),
            crop_size=int(config["data"]["crop_size"]),
        )
        if arguments.micro_batch is not None and arguments.micro_batch != micro_batch:
            raise Stage4ContractError(
                f"requested micro batch {arguments.micro_batch} differs from measured winner {micro_batch}"
            )
    accumulation = 4 // micro_batch

    learning_rates = config["optimization"]["learning_rates"]
    optimizer = build_stage4_optimizer(
        model,
        planner_lr=float(learning_rates["planner"]),
        skills_lr=float(learning_rates["skill_adapters_and_mixers"]),
        decoder_lr=float(learning_rates["decoder_refinement_rgb_head"]),
        encoder34_lr=float(learning_rates["encoder_level3_level4"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(config["optimization"]["warmup_steps"]),
        max_steps=configured_steps,
        min_lr=float(config["optimization"]["min_lr"]),
    )
    ema = ExponentialMovingAverage(model, decay=float(config["ema"]["decay"]))

    relation_train = load_relation_records(relation_train_path)
    relation_val = load_relation_records(relation_val_path)
    training_root = Path(str(resolved["training_data_root"])).resolve()
    depth_root = PROJECT_ROOT / "artifacts/cache/agenticir_depth_compat"
    train_manifest = Path(str(resolved[config["paths"]["train_manifest_key"]])).resolve()
    val_manifest = Path(str(resolved[config["paths"]["val_manifest_key"]])).resolve()
    train_base = GraphRestoreEpisodeDataset(
        train_manifest,
        training_root,
        depth_root,
        crop_size=160,
        training=True,
        stage="stage4",
        base_seed=int(config["seed"]),
        agenticir_repo=resolved["agenticir_repo"],
        mioir_repo=resolved["mioir_repo"],
    )
    train_dataset = Stage4EpisodeDataset(train_base, relation_train)
    sampler = Stage4EpisodeSampler(
        train_dataset,
        num_samples=max_steps * 4,
        effective_batch_size=4,
        seed=int(config["seed"]),
        start_step=0,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch,
        sampler=sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    validation_dataset = GraphRestoreEpisodeDataset(
        val_manifest,
        training_root,
        depth_root,
        crop_size=None,
        training=False,
        stage="stage4",
        base_seed=int(config["seed"]),
        agenticir_repo=resolved["agenticir_repo"],
        mioir_repo=resolved["mioir_repo"],
    )

    provenance = build_stage4_provenance(
        config_path=config_path,
        config=config,
        resolved_path=resolved_path,
        resolved=resolved,
        stage1_checkpoint=stage1_checkpoint,
        stage3_checkpoint=stage3_checkpoint,
        approval=approval_path,
        thresholds=thresholds_path,
        pair_prior=pair_prior_path,
        global_priority=priority_path,
        relation_train=relation_train_path,
        relation_val=relation_val_path,
        micro_batch=micro_batch,
        max_steps=max_steps,
    )
    if resume_contract is None:
        atomic_write_json(
            _contract_path(output_dir),
            {
                "schema_version": STAGE4_SCHEMA,
                "created_utc": utc_now_iso(),
                "approval": dict(approval),
                "provenance": provenance,
                "micro_batch_trials": [asdict(trial) for trial in trials],
            },
        )
        global_step = 0
        best_score: ValidationScore | None = None
    else:
        if resume_contract.get("provenance") != provenance:
            raise Stage4ContractError("Stage4 resume run-contract provenance mismatch")
        payload = resume_stage4_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_provenance=provenance,
        )
        global_step = int(payload["step"])
        best_score = _checkpoint_best_score(payload)

    train_log_path = output_dir / "train.jsonl"
    train_log = train_log_path.open("a", encoding="utf-8")
    iterator = iter(train_loader)
    latest_score: ValidationScore | None = None
    try:
        while global_step < max_steps:
            micro_batches = [next(iterator) for _ in range(accumulation)]
            result = train_stage4_optimizer_step(
                model,
                micro_batches,
                optimizer,
                scheduler,
                ema,
                step=global_step,
                device=device,
                use_bf16=True,
            )
            global_step += 1
            sampler.mark_consumed_optimizer_step(global_step)
            # Keep every optimizer step: each entry carries the diagnostics for
            # every fixed-DAG execution round represented by its micro-batches.
            append_jsonl(
                train_log,
                {
                    "schema_version": STAGE4_SCHEMA,
                    "created_utc": utc_now_iso(),
                    "step": global_step,
                    **asdict(result),
                    "learning_rates": lr_by_role(optimizer),
                },
            )

            validate_now = (
                global_step % int(config["validation"]["every_steps"]) == 0
                or global_step == max_steps
            )
            if validate_now:
                with ema.apply_to(model):
                    summary = validate_stage4(
                        model,
                        validation_dataset,
                        device=device,
                        relation_val_records=relation_val,
                        use_bf16=True,
                    )
                set_stage4_trainability(model)
                latest_score = stage4_validation_score(summary, global_step)
                improved = is_better_checkpoint(latest_score, best_score)
                if improved:
                    best_score = latest_score
                assert best_score is not None
                metrics = _checkpoint_metrics(latest_score, best_score)
                save_stage4_checkpoint(
                    output_dir / "last.pth",
                    step=global_step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    metrics=metrics,
                )
                if improved:
                    save_stage4_checkpoint(
                        output_dir / "best_ema.pth",
                        step=global_step,
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
                _append_calibration_history(
                    calibration_path, step=global_step, summary=summary
                )
                atomic_write_text(
                    report_path,
                    _render_report(
                        summary,
                        step=global_step,
                        best=best_score,
                        checkpoint=output_dir / "best_ema.pth",
                    ),
                )
            elif global_step % int(config["checkpoint"]["save_every_steps"]) == 0:
                save_stage4_checkpoint(
                    output_dir / "last.pth",
                    step=global_step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    metrics=_checkpoint_metrics(latest_score, best_score),
                )
    finally:
        train_log.close()

    if best_score is None or not (output_dir / "best_ema.pth").is_file():
        raise Stage4ContractError("Stage4 completed without a selected EMA checkpoint")
    atomic_write_json(
        output_dir / "complete.json",
        {
            "schema_version": STAGE4_SCHEMA,
            "protocol_id": config["protocol_id"],
            "completed_utc": utc_now_iso(),
            "step": global_step,
            "best_ema_path": str((output_dir / "best_ema.pth").resolve()),
            "best_ema_sha256": sha256_file(output_dir / "best_ema.pth"),
            "best_score": {
                "group_a_psnr": best_score.group_a_psnr,
                "group_a_ssim": best_score.group_a_ssim,
                "single_psnr": best_score.single_psnr,
                "single_ssim": best_score.single_ssim,
                "step": best_score.step,
            },
            "formal_mio100_started": False,
            "waiting_for": "new_user_authorization_for_formal_mio100",
        },
    )
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (
        Stage3ContractError,
        Stage4ContractError,
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as exc:
        print(f"STAGE4_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
