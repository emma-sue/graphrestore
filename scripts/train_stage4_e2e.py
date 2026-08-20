#!/usr/bin/env python3
"""Train V7.1 Full Guarded GraphRestore after the explicit Stage3 approval."""

from __future__ import annotations

import argparse
import csv
import math
import random
import signal
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE0_GROUP_A_PSNR_ANCHOR = 24.809721372127534
STAGE0_GROUP_A_SSIM_ANCHOR = 0.785909488574689
_STAGE4_STATUS_KEYS = frozenset(
    {
        "latest_group_a_psnr",
        "delta_group_a_psnr_vs_stage0",
        "latest_group_a_ssim",
        "delta_group_a_ssim_vs_stage0",
        "stage0_group_a_psnr_anchor",
        "stage0_group_a_ssim_anchor",
        "selected_group_a_psnr",
        "selected_delta_group_a_psnr_vs_stage0",
        "selected_group_a_ssim",
        "selected_delta_group_a_ssim_vs_stage0",
        "SSIM_RETENTION_RISK",
        "SSIM_RETENTION_RISK_NOTE",
    }
)
STAGE4_CALIBRATION_LEDGER_SCHEMA = "graphrestore-stage4-calibration-ledger-v1"
STAGE4_CALIBRATION_FILENAME = "stage4_calibration_history.csv"
STAGE4_VALIDATION_STEPS = tuple(range(4_000, 40_001, 4_000))
STAGE4_CALIBRATION_MARKER_COLUMNS = (
    "clean_misuse_psnr",
    "clean_misuse_ssim",
    "clean_misuse_residual_norm",
    "wrong_skill_identity_psnr",
    "wrong_skill_identity_ssim",
    "wrong_skill_residual_norm",
)

from src.data.episode_dataset import GraphRestoreEpisodeDataset  # noqa: E402
from src.net import GraphRestore  # noqa: E402
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
    STAGE4_EXTENSION_TARGET_STEP,
    Stage4ContractError,
    Stage4ExtensionEvidence,
    Stage4EpisodeDataset,
    Stage4EpisodeSampler,
    append_jsonl,
    build_stage4_ema,
    build_stage4_optimizer,
    build_stage4_provenance,
    choose_stage4_micro_batch,
    is_stage4_cuda_oom_exception,
    load_presence_thresholds,
    load_relation_records,
    load_stage3_best_ema,
    lr_by_role,
    probe_stage4_validation_vram,
    require_stage4_allocator_conf,
    resume_stage4_checkpoint,
    run_stage4_zero_training_diagnostics,
    save_stage4_checkpoint,
    set_stage4_trainability,
    stage4_fixed_state_digest,
    stage4_runtime_evidence_metadata,
    stage4_validation_score,
    train_stage4_optimizer_step,
    validate_stage3_approval,
    validate_stage3_finalization_for_stage4,
    validate_stage4,
    validate_stage4_config,
    validate_stage4_extension_authorization,
)
from src.utils.hashing import is_sha256, sha256_file  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
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
    parser.add_argument(
        "--extension_authorization",
        type=Path,
        help=(
            "canonical activated Stage4 40k-to-48k conditional extension; "
            "valid only with --resume"
        ),
    )
    return parser


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage4ContractError(f"{field} must be a mapping")
    return value


def _stage3_frozen_calibration_binding(
    stage3_finalization_outputs: object,
    *,
    expected_path: Path,
) -> dict[str, str]:
    outputs = _mapping(stage3_finalization_outputs, "Stage3 finalization outputs")
    payload = _mapping(outputs.get("payload"), "Stage3 finalization payload")
    bindings = _mapping(payload.get("bindings"), "Stage3 finalization payload bindings")
    binding = _mapping(
        bindings.get("calibration_history"),
        "Stage3 frozen calibration history binding",
    )
    if set(binding) != {"path", "sha256"}:
        raise Stage4ContractError(
            "Stage3 frozen calibration history binding schema drifted"
        )
    raw_path = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(raw_path, str) or not is_sha256(digest):
        raise Stage4ContractError(
            "Stage3 frozen calibration history binding is malformed"
        )
    path = Path(raw_path)
    canonical = path.resolve(strict=False)
    if (
        not path.is_absolute()
        or str(canonical) != raw_path
        or canonical != expected_path.resolve(strict=False)
    ):
        raise Stage4ContractError(
            "Stage3 frozen calibration history binding path drifted"
        )
    if path.is_symlink() or not canonical.is_file():
        raise Stage4ContractError(
            "Stage3 frozen calibration history binding is not regular"
        )
    before = sha256_file(canonical)
    if before != digest or sha256_file(canonical) != before:
        raise Stage4ContractError(
            "Stage3 frozen calibration history binding hash drifted"
        )
    return {"path": str(canonical), "sha256": digest}


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


def _sigterm_as_keyboard_interrupt(signum: int, frame: object) -> None:
    del signum, frame
    raise KeyboardInterrupt


def _stage4_status_lines(
    latest: ValidationScore,
    *,
    selected: ValidationScore | None = None,
) -> list[str]:
    lines = [
        f"latest_group_a_psnr: {latest.group_a_psnr!r}",
        "delta_group_a_psnr_vs_stage0: "
        f"{latest.group_a_psnr - STAGE0_GROUP_A_PSNR_ANCHOR!r}",
        f"latest_group_a_ssim: {latest.group_a_ssim!r}",
        "delta_group_a_ssim_vs_stage0: "
        f"{latest.group_a_ssim - STAGE0_GROUP_A_SSIM_ANCHOR!r}",
        f"stage0_group_a_psnr_anchor: {STAGE0_GROUP_A_PSNR_ANCHOR!r}",
        f"stage0_group_a_ssim_anchor: {STAGE0_GROUP_A_SSIM_ANCHOR!r}",
    ]
    if selected is not None:
        risk = selected.group_a_ssim < STAGE0_GROUP_A_SSIM_ANCHOR
        lines.extend(
            (
                f"selected_group_a_psnr: {selected.group_a_psnr!r}",
                "selected_delta_group_a_psnr_vs_stage0: "
                f"{selected.group_a_psnr - STAGE0_GROUP_A_PSNR_ANCHOR!r}",
                f"selected_group_a_ssim: {selected.group_a_ssim!r}",
                "selected_delta_group_a_ssim_vs_stage0: "
                f"{selected.group_a_ssim - STAGE0_GROUP_A_SSIM_ANCHOR!r}",
                f"SSIM_RETENTION_RISK: {str(risk).lower()}",
            )
        )
        if risk:
            lines.append(
                "SSIM_RETENTION_RISK_NOTE: selected Group-A PSNR does not "
                "offset the SSIM retention deficit"
            )
    return lines


def _update_stage4_running_status(
    path: Path,
    *,
    latest: ValidationScore,
    selected: ValidationScore | None = None,
) -> None:
    if not path.is_file():
        raise Stage4ContractError(f"Stage4 running status is missing: {path}")
    retained = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.partition(":")[0] not in _STAGE4_STATUS_KEYS
    ]
    insert_at = 1 if retained and retained[0].startswith("status:") else 0
    updated = (
        retained[:insert_at]
        + _stage4_status_lines(latest, selected=selected)
        + retained[insert_at:]
    )
    atomic_write_text(path, "\n".join(updated) + "\n")


def _update_stage4_decision_memo(path: Path, selected: ValidationScore) -> None:
    if not path.is_file():
        return
    begin = "<!-- STAGE4_SSIM_RETENTION_BEGIN -->"
    end = "<!-- STAGE4_SSIM_RETENTION_END -->"
    current = path.read_text(encoding="utf-8")
    lines = current.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Status:"):
            lines[index] = (
                "Status: Stage3–4 complete; the Stage4 selected checkpoint is "
                "frozen. Formal MiO100 remains unauthorized and pending a "
                "separate user approval."
            )
            break
    current = "\n".join(lines) + ("\n" if current.endswith("\n") else "")
    current = current.replace(
        "The final baseline choice, complete 2–3 contribution assessment, "
        "ablations, formal MiO100 A/B/C table, paper claim strength and "
        "title/abstract package remain pending because the contract forbids "
        "Stage3/4 and formal B/C evaluation without further user approval.",
        "The Stage4 selected checkpoint is now frozen. The formal MiO100 A/B/C "
        "table, final paper claim strength and title/abstract package remain "
        "pending because formal MiO100 still requires separate user approval.",
    )
    if begin in current or end in current:
        if current.count(begin) != 1 or current.count(end) != 1:
            raise Stage4ContractError(
                "Stage4 SSIM retention memo markers are malformed"
            )
        prefix, remainder = current.split(begin, 1)
        _, suffix = remainder.split(end, 1)
        current = prefix.rstrip() + suffix
    risk = selected.group_a_ssim < STAGE0_GROUP_A_SSIM_ANCHOR
    risk_note = (
        "- The selected Group-A PSNR does not offset the SSIM retention deficit.\n"
        if risk
        else ""
    )
    section = (
        f"\n\n{begin}\n"
        "## Stage4 SSIM retention\n\n"
        f"- SSIM_RETENTION_RISK: `{str(risk).lower()}`\n"
        f"- Stage0 Group-A PSNR/SSIM: `{STAGE0_GROUP_A_PSNR_ANCHOR!r} / "
        f"{STAGE0_GROUP_A_SSIM_ANCHOR!r}`\n"
        f"- Stage4 selected Group-A PSNR/SSIM: `{selected.group_a_psnr!r} / "
        f"{selected.group_a_ssim!r}`\n"
        f"- Delta PSNR/SSIM: `{selected.group_a_psnr - STAGE0_GROUP_A_PSNR_ANCHOR!r} / "
        f"{selected.group_a_ssim - STAGE0_GROUP_A_SSIM_ANCHOR!r}`\n"
        f"{risk_note}"
        "- This disclosure does not alter checkpoint selection or authorize MiO100.\n"
        f"{end}\n"
    )
    atomic_write_text(path, current.rstrip() + section)


def _stage4_interrupt_can_checkpoint(
    *,
    mid_optimizer_update: bool,
    pending_validation_step: int | None,
) -> bool:
    """Only a fully committed optimizer boundary may replace ``last.pth``."""

    return not mid_optimizer_update and pending_validation_step is None


def _record_stage4_runtime_oom(
    *,
    output_dir: Path,
    error: RuntimeError,
    attempted_step: int,
    mid_optimizer_update: bool,
    pending_validation_step: int | None,
    crop_size: int,
    micro_batch: int,
    allocator_conf: str,
    deviations_path: Path | None = None,
) -> Mapping[str, Any]:
    """Persist a fail-safe OOM receipt without advancing training state."""

    if not is_stage4_cuda_oom_exception(error):
        raise Stage4ContractError("refusing to record a non-OOM as Stage4 OOM")
    stable = output_dir / "last.pth"
    stable_header: Mapping[str, Any] | None = None
    if stable.is_file():
        loaded = torch.load(stable, map_location="cpu", weights_only=False)
        if isinstance(loaded, Mapping):
            stable_header = loaded
    created = utc_now_iso()
    receipt: dict[str, Any] = {
        "schema_version": "graphrestore-stage4-runtime-oom-v1",
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "created_utc": created,
        "error_type": type(error).__name__,
        "error": str(error),
        "attempted_step": attempted_step,
        "mid_optimizer_update": mid_optimizer_update,
        "pending_validation_step": pending_validation_step,
        "checkpoint_advanced": False,
        "automatic_crop_or_micro_fallback": False,
        "same_process_continuation": False,
        "stable_checkpoint": {
            "path": str(stable.resolve()),
            "exists": stable.is_file(),
            "sha256": sha256_file(stable) if stable.is_file() else None,
            "step": None if stable_header is None else stable_header.get("step"),
            "pending_validation_step": (
                None
                if stable_header is None
                else stable_header.get("pending_validation_step")
            ),
            "model_role": (
                None if stable_header is None else stable_header.get("model_role")
            ),
            "resumable": (
                None if stable_header is None else stable_header.get("resumable")
            ),
        },
        "frozen_runtime": {
            "crop_size": crop_size,
            "micro_batch": micro_batch,
            "allocator_conf": allocator_conf,
        },
        "required_action": "exit_child_and_resume_in_new_process_from_stable_raw",
        "resume_command": "python scripts/orchestrate.py --resume_post_approval_pipeline",
    }
    atomic_write_json(output_dir / "runtime_oom.json", receipt)
    deviation = deviations_path or PROJECT_ROOT / "reports/DEVIATIONS.md"
    existing = (
        deviation.read_text(encoding="utf-8")
        if deviation.is_file()
        else "# Deviations\n"
    )
    section = (
        "\n\n## Stage4 runtime OOM (fail-safe)\n\n"
        f"- UTC: `{created}`\n"
        f"- Attempted optimizer step: `{attempted_step}`\n"
        f"- Mid optimizer update: `{str(mid_optimizer_update).lower()}`\n"
        f"- Pending validation step: `{pending_validation_step}`\n"
        f"- Stable raw checkpoint: `{stable}`\n"
        "- Action: no checkpoint advance, no runtime crop fallback, child exits; "
        "resume in a new process with `python scripts/orchestrate.py "
        "--resume_post_approval_pipeline`.\n"
        f"- Structured receipt: `{output_dir / 'runtime_oom.json'}`\n"
    )
    atomic_write_text(deviation, existing.rstrip() + section)
    return receipt


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


def _checkpoint_current_score(payload: Mapping[str, Any]) -> ValidationScore | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or "group_a_psnr" not in metrics:
        return None
    result = ValidationScore(
        group_a_psnr=float(metrics["group_a_psnr"]),
        group_a_ssim=float(metrics["group_a_ssim"]),
        single_psnr=float(metrics["single_psnr"]),
        single_ssim=float(metrics["single_ssim"]),
        step=int(metrics["validation_step"]),
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
        raise Stage4ContractError(
            "resume checkpoint contains non-finite current metrics"
        )
    return result


def _historical_stage4_peak_reserved(path: Path) -> tuple[int, int]:
    maximum_train = 0
    maximum_validation = 0
    if not path.is_file():
        return maximum_train, maximum_validation
    for line_number, row in iter_jsonl(path):
        peak = row.get("peak_reserved_bytes")
        if peak is None:
            continue
        if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            raise Stage4ContractError(
                f"Stage4 train log has invalid peak at line {line_number}"
            )
        if row.get("event") == "validation":
            maximum_validation = max(maximum_validation, peak)
        else:
            maximum_train = max(maximum_train, peak)
    return maximum_train, maximum_validation


def _checkpoint_metrics(
    current: ValidationScore | None,
    best: ValidationScore | None,
    *,
    best_checkpoint: Path | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
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
        if best_checkpoint is not None:
            path = best_checkpoint.resolve()
            if not path.is_file():
                raise Stage4ContractError(
                    "Stage4 best metric binding requires existing best_ema.pth"
                )
            values["best_checkpoint_sha256"] = sha256_file(path)
    return values


def _stage4_calibration_history_path(frozen_stage3_history: Path) -> Path:
    frozen = frozen_stage3_history.resolve()
    if frozen.name != "calibration_history.csv":
        raise Stage4ContractError(
            "frozen Stage3 calibration history must use calibration_history.csv"
        )
    return frozen.with_name(STAGE4_CALIBRATION_FILENAME)


def _require_calibration_history_boundary(
    *,
    frozen_stage3_history: Path,
    frozen_stage3_sha256: str,
    stage4_history: Path,
) -> None:
    if frozen_stage3_history.is_symlink():
        raise Stage4ContractError("frozen Stage3 calibration history is not regular")
    if stage4_history.is_symlink():
        raise Stage4ContractError("Stage4 calibration history cannot be a symlink")
    frozen = frozen_stage3_history.resolve()
    expected_stage4 = _stage4_calibration_history_path(frozen)
    if stage4_history.resolve() != expected_stage4:
        raise Stage4ContractError("Stage4 calibration history sidecar path drifted")
    if not frozen.is_file():
        raise Stage4ContractError("frozen Stage3 calibration history is not regular")
    if sha256_file(frozen) != frozen_stage3_sha256:
        raise Stage4ContractError("frozen Stage3 calibration history changed")
    if stage4_history.exists():
        if not stage4_history.is_file():
            raise Stage4ContractError("Stage4 calibration history is not regular")
        frozen_stat = frozen.stat()
        stage4_stat = stage4_history.stat()
        if (frozen_stat.st_dev, frozen_stat.st_ino) == (
            stage4_stat.st_dev,
            stage4_stat.st_ino,
        ):
            raise Stage4ContractError(
                "Stage4 calibration history aliases the frozen Stage3 history"
            )


def _calibration_history_routing(
    *,
    frozen_stage3_history: Path,
    frozen_stage3_sha256: str,
    stage4_history: Path,
    validation_steps: Sequence[int] = STAGE4_VALIDATION_STEPS,
) -> dict[str, Any]:
    _require_calibration_history_boundary(
        frozen_stage3_history=frozen_stage3_history,
        frozen_stage3_sha256=frozen_stage3_sha256,
        stage4_history=stage4_history,
    )
    return {
        "schema_version": STAGE4_CALIBRATION_LEDGER_SCHEMA,
        "frozen_stage3_history": {
            "path": str(frozen_stage3_history.resolve()),
            "sha256": frozen_stage3_sha256,
        },
        "stage4_history_path": str(stage4_history.resolve()),
        "columns": list(CALIBRATION_COLUMNS),
        "stage4_marker_columns": list(STAGE4_CALIBRATION_MARKER_COLUMNS),
        "validation_steps": list(validation_steps),
    }


def _load_stage4_calibration_rows(
    path: Path,
    *,
    validation_steps: Sequence[int] = STAGE4_VALIDATION_STEPS,
) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise Stage4ContractError(
            "Stage4 calibration history is missing or not regular"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CALIBRATION_COLUMNS:
            raise Stage3ContractError("calibration history header drifted")
        rows: list[dict[str, str]] = []
        for existing in reader:
            if set(existing) != set(CALIBRATION_COLUMNS) or any(
                value is None for value in existing.values()
            ):
                raise Stage4ContractError(
                    "Stage4 calibration history row is not exactly 28 columns"
                )
            rows.append(dict(existing))
    steps: list[int] = []
    for existing in rows:
        marker_presence = tuple(
            bool(existing.get(column)) for column in STAGE4_CALIBRATION_MARKER_COLUMNS
        )
        if any(marker_presence) and not all(marker_presence):
            raise Stage4ContractError("partial Stage4 calibration row exists")
        if not all(marker_presence):
            raise Stage4ContractError(
                "non-Stage4 row exists in Stage4 calibration history"
            )
        raw_step = existing.get("step")
        try:
            parsed_step = int(str(raw_step))
        except (TypeError, ValueError) as exc:
            raise Stage4ContractError(
                "Stage4 calibration history contains an invalid step"
            ) from exc
        if str(parsed_step) != raw_step or parsed_step not in validation_steps:
            raise Stage4ContractError(
                "Stage4 calibration history contains an off-boundary step"
            )
        steps.append(parsed_step)
    if len(steps) != len(set(steps)):
        raise Stage4ContractError(
            "multiple Stage4 calibration rows already exist for one step"
        )
    if steps != sorted(steps):
        raise Stage4ContractError("Stage4 calibration history steps are non-monotonic")
    return rows


def _require_stage4_calibration_prefix(
    path: Path,
    *,
    checkpoint_step: int,
    pending_validation_step: int | None,
    validation_steps: Sequence[int] = STAGE4_VALIDATION_STEPS,
) -> tuple[int, ...]:
    pending = pending_validation_step
    observed = (
        tuple(
            int(row["step"])
            for row in _load_stage4_calibration_rows(
                path, validation_steps=validation_steps
            )
        )
        if path.exists() or path.is_symlink()
        else ()
    )
    committed = tuple(
        boundary
        for boundary in validation_steps
        if boundary <= checkpoint_step and (pending is None or boundary < pending)
    )
    allowed = {committed}
    if pending is not None:
        allowed.add(committed + (pending,))
    if observed not in allowed:
        raise Stage4ContractError(
            "Stage4 calibration history disagrees with checkpoint transaction"
        )
    return observed


def _append_calibration_history(
    path: Path,
    *,
    step: int,
    summary: Mapping[str, Any],
    validation_steps: Sequence[int] = STAGE4_VALIDATION_STEPS,
) -> None:
    if step not in validation_steps:
        raise Stage4ContractError("Stage4 calibration step is off the frozen schedule")
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
    if path.exists() or path.is_symlink():
        expected = {
            key: "" if row.get(key) is None else str(row.get(key))
            for key in CALIBRATION_COLUMNS
        }
        existing_rows = _load_stage4_calibration_rows(
            path, validation_steps=validation_steps
        )
        existing_steps = tuple(int(existing["step"]) for existing in existing_rows)
        required_predecessors = tuple(
            boundary for boundary in validation_steps if boundary < step
        )
        allowed_steps = (
            required_predecessors,
            required_predecessors + (step,),
        )
        if existing_steps not in allowed_steps:
            raise Stage4ContractError(
                "Stage4 calibration history is not the exact current prefix"
            )
        stage4_rows_at_step = [
            existing for existing in existing_rows if existing.get("step") == str(step)
        ]
        if len(stage4_rows_at_step) > 1:
            raise Stage4ContractError(
                f"multiple Stage4 calibration rows already exist for step {step}"
            )
        if stage4_rows_at_step and stage4_rows_at_step[0] != expected:
            raise Stage4ContractError(
                f"Stage4 calibration row drifted during replay at step {step}"
            )
        if stage4_rows_at_step:
            return
    elif step != validation_steps[0]:
        raise Stage4ContractError(
            "Stage4 calibration history is missing predecessor rows"
        )
    append_shared_calibration_history(path, row)
    committed_steps = tuple(
        int(existing["step"])
        for existing in _load_stage4_calibration_rows(
            path, validation_steps=validation_steps
        )
    )
    expected_steps = tuple(
        boundary for boundary in validation_steps if boundary <= step
    )
    if committed_steps != expected_steps:
        raise Stage4ContractError(
            "Stage4 calibration append did not commit exact prefix"
        )


def _render_report(
    summary: Mapping[str, Any],
    *,
    step: int,
    best: ValidationScore,
    checkpoint: Path,
    calibration_history_routing: Mapping[str, Any],
    stage4_extension: Stage4ExtensionEvidence | None = None,
) -> str:
    group = summary["group_a_equal_combination_mean"]
    single = summary["single_equal_task_mean"]
    diag = summary["diagnostics"]
    selected_checkpoint = checkpoint.resolve()
    selected_sha256 = sha256_file(selected_checkpoint)
    frozen_history = _mapping(
        calibration_history_routing.get("frozen_stage3_history"),
        "frozen Stage3 calibration history routing",
    )
    frozen_history_path_value = frozen_history.get("path")
    frozen_history_sha256 = frozen_history.get("sha256")
    if (
        set(frozen_history) != {"path", "sha256"}
        or not isinstance(frozen_history_path_value, str)
        or not isinstance(frozen_history_sha256, str)
    ):
        raise Stage4ContractError(
            "frozen Stage3 calibration history routing schema drifted"
        )
    frozen_history_path = Path(frozen_history_path_value).resolve()
    stage4_history = Path(
        str(calibration_history_routing.get("stage4_history_path"))
    ).resolve()
    _require_calibration_history_boundary(
        frozen_stage3_history=frozen_history_path,
        frozen_stage3_sha256=frozen_history_sha256,
        stage4_history=stage4_history,
    )
    if (
        calibration_history_routing.get("schema_version")
        != STAGE4_CALIBRATION_LEDGER_SCHEMA
        or not stage4_history.is_file()
    ):
        raise Stage4ContractError("Stage4 calibration history routing is incomplete")
    stage4_history_sha256 = sha256_file(stage4_history)
    psnr_delta = best.group_a_psnr - STAGE0_GROUP_A_PSNR_ANCHOR
    ssim_delta = best.group_a_ssim - STAGE0_GROUP_A_SSIM_ANCHOR
    ssim_retention_risk = best.group_a_ssim < STAGE0_GROUP_A_SSIM_ANCHOR
    retention_interpretation = (
        "The selected Group-A SSIM is below the frozen Stage0 anchor; this "
        "risk is not offset by any average PSNR gain."
        if ssim_retention_risk
        else "The selected Group-A SSIM retains or exceeds the frozen Stage0 anchor."
    )
    extension_lines = ""
    if stage4_extension is not None:
        extension_lines = (
            "- Conditional Stage4 extension: activated\n"
            f"- Conditional authorization SHA256: `{stage4_extension.conditional_sha256}`\n"
            f"- Extension gate receipt SHA256: `{stage4_extension.gate_sha256}`\n"
            f"- Completed training target step: {stage4_extension.target_step}\n"
            f"- Original cosine schedule horizon step: {stage4_extension.schedule_horizon_steps}\n"
            "- Extension budget disclosure: 8,000 additional optimizer steps "
            "(+20%); 48,000 is an unconditional hard terminal and no further "
            "extension is authorized.\n"
            "- Extension interpretation: the >=0.20 dB gate authorized continued "
            "observation only; it is not evidence of Stage0 superiority or paper success.\n"
        )
    return (
        "# Stage4 Full Guarded GraphRestore\n\n"
        f"- Protocol: `{summary['protocol_id']}`\n"
        f"- Validation step: {step}\n"
        "- Data: frozen primary_val singles + Group A only; MiO100 B/C were not read\n"
        "- Runtime: compile-once DAG, Kmax_test=3, no skill re-entry\n"
        f"- Selected EMA: `{selected_checkpoint}`\n"
        f"- Selected EMA SHA256: `{selected_sha256}`\n"
        "- Frozen Stage3 calibration history: "
        f"`{frozen_history.get('path')}`\n"
        "- Frozen Stage3 calibration history SHA256: "
        f"`{frozen_history.get('sha256')}`\n"
        f"- Stage4 calibration history sidecar: `{stage4_history}`\n"
        f"- Stage4 calibration history SHA256: `{stage4_history_sha256}`\n"
        f"{extension_lines}"
        f"- Selected Single PSNR/SSIM: {best.single_psnr:.6f} / "
        f"{best.single_ssim:.8f}\n"
        f"- Selected Group-A PSNR/SSIM: {best.group_a_psnr:.6f} / "
        f"{best.group_a_ssim:.8f}\n"
        f"- Current Single PSNR/SSIM: {single['psnr']:.6f} / "
        f"{single['ssim']:.8f}\n"
        f"- Current Group-A PSNR/SSIM: {group['psnr']:.6f} / "
        f"{group['ssim']:.8f}\n"
        f"- Stage0 Group-A PSNR anchor: {STAGE0_GROUP_A_PSNR_ANCHOR!r}\n"
        f"- Stage0 Group-A SSIM anchor: {STAGE0_GROUP_A_SSIM_ANCHOR!r}\n"
        f"- Selected Group-A PSNR delta vs Stage0: {psnr_delta!r}\n"
        f"- Selected Group-A SSIM delta vs Stage0: {ssim_delta!r}\n"
        f"- SSIM_RETENTION_RISK: {str(ssim_retention_risk).lower()}\n"
        f"- SSIM retention interpretation: {retention_interpretation}\n"
        f"- Planner macro-F1: {diag['planner_macro_f1']:.6f}\n"
        f"- Non-ambiguous relation accuracy: {diag['relation_accuracy']:.6f}\n"
        f"- Re-entry request rate (diagnostic only): {diag['reentry_request_rate']:.8f}\n"
    )


def _stage4_report_binding(
    report_path: Path,
    *,
    selected_best_checkpoint: Path,
    selected_best_score: ValidationScore,
    latest_score: ValidationScore,
    calibration_history_routing: Mapping[str, Any],
    completed_calibration_sha256: str,
    validation_steps: Sequence[int] = STAGE4_VALIDATION_STEPS,
    stage4_extension: Stage4ExtensionEvidence | None = None,
) -> dict[str, str]:
    report = report_path.resolve()
    selected = selected_best_checkpoint.resolve()
    if not report.is_file() or not selected.is_file():
        raise Stage4ContractError("Stage4 completion requires report and selected EMA")
    selected_sha256 = sha256_file(selected)
    payload = torch.load(selected, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "graphrestore-checkpoint-v1"
        or payload.get("stage") != "stage4"
        or payload.get("model_role") != "ema_selection"
        or payload.get("resumable") is not False
        or payload.get("pending_validation_step") is not None
        or payload.get("step") != selected_best_score.step
        or not isinstance(metrics, Mapping)
        or metrics.get("best_step") != float(selected_best_score.step)
        or metrics.get("best_group_a_psnr") != selected_best_score.group_a_psnr
        or metrics.get("best_group_a_ssim") != selected_best_score.group_a_ssim
        or metrics.get("best_single_psnr") != selected_best_score.single_psnr
        or metrics.get("best_single_ssim") != selected_best_score.single_ssim
    ):
        raise Stage4ContractError(
            "Stage4 selected report score differs from best_ema.pth"
        )
    report_text = report.read_text(encoding="utf-8")
    frozen_history = _mapping(
        calibration_history_routing.get("frozen_stage3_history"),
        "frozen Stage3 calibration history routing",
    )
    frozen_history_path_value = frozen_history.get("path")
    frozen_history_sha256 = frozen_history.get("sha256")
    if (
        set(frozen_history) != {"path", "sha256"}
        or not isinstance(frozen_history_path_value, str)
        or not isinstance(frozen_history_sha256, str)
    ):
        raise Stage4ContractError(
            "frozen Stage3 calibration history routing schema drifted"
        )
    frozen_history_path = Path(frozen_history_path_value).resolve()
    stage4_history = Path(
        str(calibration_history_routing.get("stage4_history_path"))
    ).resolve()
    _require_calibration_history_boundary(
        frozen_stage3_history=frozen_history_path,
        frozen_stage3_sha256=frozen_history_sha256,
        stage4_history=stage4_history,
    )
    completed_steps = tuple(
        int(row["step"])
        for row in _load_stage4_calibration_rows(
            stage4_history, validation_steps=validation_steps
        )
    )
    if (
        set(calibration_history_routing)
        != {
            "schema_version",
            "frozen_stage3_history",
            "stage4_history_path",
            "columns",
            "stage4_marker_columns",
            "validation_steps",
        }
        or calibration_history_routing.get("schema_version")
        != STAGE4_CALIBRATION_LEDGER_SCHEMA
        or calibration_history_routing.get("columns") != list(CALIBRATION_COLUMNS)
        or calibration_history_routing.get("stage4_marker_columns")
        != list(STAGE4_CALIBRATION_MARKER_COLUMNS)
        or calibration_history_routing.get("validation_steps") != list(validation_steps)
        or completed_steps != tuple(validation_steps)
        or latest_score.step != validation_steps[-1]
        or sha256_file(stage4_history) != completed_calibration_sha256
    ):
        raise Stage4ContractError(
            "Stage4 report calibration history/final-step binding drifted"
        )
    required_lines = (
        f"Validation step: {latest_score.step}",
        f"Selected EMA SHA256: `{selected_sha256}`",
        "Selected Single PSNR/SSIM: "
        f"{selected_best_score.single_psnr:.6f} / "
        f"{selected_best_score.single_ssim:.8f}",
        "Selected Group-A PSNR/SSIM: "
        f"{selected_best_score.group_a_psnr:.6f} / "
        f"{selected_best_score.group_a_ssim:.8f}",
        f"Stage0 Group-A PSNR anchor: {STAGE0_GROUP_A_PSNR_ANCHOR!r}",
        f"Stage0 Group-A SSIM anchor: {STAGE0_GROUP_A_SSIM_ANCHOR!r}",
        "Selected Group-A PSNR delta vs Stage0: "
        f"{selected_best_score.group_a_psnr - STAGE0_GROUP_A_PSNR_ANCHOR!r}",
        "Selected Group-A SSIM delta vs Stage0: "
        f"{selected_best_score.group_a_ssim - STAGE0_GROUP_A_SSIM_ANCHOR!r}",
        "Current Single PSNR/SSIM: "
        f"{latest_score.single_psnr:.6f} / {latest_score.single_ssim:.8f}",
        "Current Group-A PSNR/SSIM: "
        f"{latest_score.group_a_psnr:.6f} / {latest_score.group_a_ssim:.8f}",
        f"Frozen Stage3 calibration history: `{frozen_history_path}`",
        f"Frozen Stage3 calibration history SHA256: `{frozen_history_sha256}`",
        f"Stage4 calibration history sidecar: `{stage4_history}`",
        f"Stage4 calibration history SHA256: `{completed_calibration_sha256}`",
        "SSIM_RETENTION_RISK: "
        f"{str(selected_best_score.group_a_ssim < STAGE0_GROUP_A_SSIM_ANCHOR).lower()}",
    )
    if any(line not in report_text for line in required_lines):
        raise Stage4ContractError(
            "Stage4 report is not bound to the selected EMA SHA/metrics"
        )
    if stage4_extension is not None:
        required_extension_lines = (
            "- Conditional Stage4 extension: activated",
            "- Conditional authorization SHA256: "
            f"`{stage4_extension.conditional_sha256}`",
            f"- Extension gate receipt SHA256: `{stage4_extension.gate_sha256}`",
            f"- Completed training target step: {stage4_extension.target_step}",
            "- Original cosine schedule horizon step: "
            f"{stage4_extension.schedule_horizon_steps}",
        )
        if any(line not in report_text for line in required_extension_lines):
            raise Stage4ContractError(
                "Stage4 report lacks the activated extension binding"
            )
    if (
        selected_best_score.group_a_ssim < STAGE0_GROUP_A_SSIM_ANCHOR
        and "risk is not offset by any average PSNR gain" not in report_text
    ):
        raise Stage4ContractError(
            "Stage4 report did not disclose that PSNR cannot offset SSIM risk"
        )
    if sha256_file(selected) != selected_sha256:
        raise Stage4ContractError("Stage4 selected EMA changed during report binding")
    return {
        "report": str(report),
        "report_sha256": sha256_file(report),
    }


def run(arguments: argparse.Namespace) -> int:
    config_path = _project_path(arguments.config)
    config = _mapping(load_yaml(config_path), "Stage4 config")
    validate_stage4_config(config)
    allocator_conf = require_stage4_allocator_conf()
    configured_steps = int(config["training"]["max_steps"])
    extension_argument = getattr(arguments, "extension_authorization", None)
    if extension_argument is None:
        extension = None
    else:
        if arguments.resume is None:
            raise Stage4ContractError(
                "Stage4 extension authorization is valid only for exact resume"
            )
        extension = validate_stage4_extension_authorization(
            _project_path(extension_argument),
            project_root=PROJECT_ROOT,
            config_path=config_path,
        )
    training_target_step = (
        configured_steps if extension is None else extension.target_step
    )
    validation_steps = (
        STAGE4_VALIDATION_STEPS
        if extension is None
        else STAGE4_VALIDATION_STEPS + extension.validation_steps
    )
    if (
        arguments.max_steps is not None
        and not 0 < arguments.max_steps <= training_target_step
    ):
        raise Stage4ContractError(f"--max_steps must lie in [1,{training_target_step}]")
    if extension is not None and arguments.max_steps not in (
        None,
        STAGE4_EXTENSION_TARGET_STEP,
    ):
        raise Stage4ContractError(
            "activated Stage4 extension requires the exact 48000-step target"
        )

    resolved_path = _project_path(config["paths"]["resolved_paths"])
    resolved = _mapping(load_yaml(resolved_path), "resolved paths")
    output_dir = _project_path(arguments.output_dir or config["paths"]["output_dir"])
    report_path = _project_path(config["paths"]["report"])
    diagnostics_report_path = _project_path(config["paths"]["diagnostics_report"])
    diagnostics_json_path = diagnostics_report_path.with_suffix(".json")
    # The later Stage3 revocation freezes the shared history byte-for-byte.
    # Stage4 therefore treats the configured path as a parent anchor and owns
    # a deterministic sibling ledger with the same locked 28-column schema.
    frozen_calibration_history_path = _project_path(
        config["paths"]["calibration_history"]
    )
    calibration_path = _stage4_calibration_history_path(frozen_calibration_history_path)
    stage1_checkpoint = _project_path(config["paths"]["stage1_checkpoint"])
    stage3_checkpoint = _project_path(config["paths"]["stage3_checkpoint"])
    approval_path = _project_path(config["paths"]["required_approval"])
    thresholds_path = _project_path(config["paths"]["thresholds"])
    pair_prior_path = _project_path(config["paths"]["pair_prior"])
    priority_path = _project_path(config["paths"]["global_priority"])
    relation_train_path = (
        PROJECT_ROOT / "artifacts/interaction_labels/group_a_relations_train.jsonl"
    )
    relation_val_path = (
        PROJECT_ROOT / "artifacts/interaction_labels/group_a_relations_val.jsonl"
    )
    effect_profiles_path = (
        PROJECT_ROOT / "artifacts/interaction_labels/skill_effect_profiles.json"
    )
    stage2_decision_path = (
        PROJECT_ROOT / "artifacts/interaction_labels/stage2_decision.json"
    )

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
    (
        finalization_authorization,
        stage3_finalization_outputs,
    ) = validate_stage3_finalization_for_stage4(PROJECT_ROOT)
    frozen_calibration_binding = _stage3_frozen_calibration_binding(
        stage3_finalization_outputs,
        expected_path=frozen_calibration_history_path,
    )
    frozen_calibration_sha256 = frozen_calibration_binding["sha256"]
    calibration_history_routing = _calibration_history_routing(
        frozen_stage3_history=frozen_calibration_history_path,
        frozen_stage3_sha256=frozen_calibration_sha256,
        stage4_history=calibration_path,
        validation_steps=validation_steps,
    )
    if calibration_path.exists() or calibration_path.is_symlink():
        _load_stage4_calibration_rows(
            calibration_path, validation_steps=validation_steps
        )
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
        finalization_authorization=finalization_authorization,
    )
    thresholds, _ = load_presence_thresholds(
        thresholds_path,
        stage3_checkpoint_sha256=snapshot.checkpoint_sha256,
        stage3_approval_sha256=approval_sha,
        stage3_extension_authorization_sha256=(
            None
            if snapshot.stage3_extension is None
            else str(snapshot.stage3_extension["sha256"])
        ),
        stage3_finalization_authorization_sha256=(finalization_authorization.sha256),
    )
    model.set_presence_thresholds(thresholds)

    def build_current_provenance(
        *,
        selected_crop_size: int,
        selected_micro_batch: int,
        selected_max_steps: int,
        micro_batch_trials: object,
        validation_vram_gate: object,
    ) -> dict[str, Any]:
        value = dict(
            build_stage4_provenance(
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
                crop_size=selected_crop_size,
                micro_batch=selected_micro_batch,
                max_steps=selected_max_steps,
                allocator_conf=allocator_conf,
                frozen_parent_state_sha256=stage4_fixed_state_digest(model),
                micro_batch_trials=micro_batch_trials,
                validation_vram_gate=validation_vram_gate,
                stage3_extension=snapshot.stage3_extension,
                stage3_finalization=snapshot.stage3_finalization,
                stage3_complete=(Path(stage3_finalization_outputs["complete"]["path"])),
                stage3_calibrated_diagnostic=Path(
                    stage3_finalization_outputs["selected_validation_calibrated"][
                        "path"
                    ]
                ),
                stage3_complete_sha256=stage3_finalization_outputs["complete"][
                    "sha256"
                ],
                stage3_calibrated_diagnostic_sha256=(
                    stage3_finalization_outputs["selected_validation_calibrated"][
                        "sha256"
                    ]
                ),
                stage3_thresholds_sha256=stage3_finalization_outputs["thresholds"][
                    "sha256"
                ],
                stage4_extension=extension,
            )
        )
        value["calibration_history_routing"] = calibration_history_routing
        return value

    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path: Path | None = None
    resume_contract: Mapping[str, Any] | None = None
    prevalidated_resume_provenance: dict[str, Any] | None = None
    micro_batch_trial_evidence: object = None
    validation_vram_gate_evidence: object = None
    if arguments.resume is not None:
        resume_path = (
            output_dir / "last.pth"
            if arguments.resume == "auto"
            else _project_path(arguments.resume)
        )
        if not resume_path.is_file():
            raise Stage4ContractError(
                f"Stage4 resume checkpoint missing: {resume_path}"
            )
        contract_value = load_json(_contract_path(output_dir))
        resume_contract = _mapping(contract_value, "Stage4 run contract")
        if resume_contract.get("schema_version") != STAGE4_SCHEMA:
            raise Stage4ContractError("Stage4 run contract schema mismatch")
        frozen = _mapping(resume_contract.get("provenance"), "Stage4 provenance")[
            "runtime"
        ]
        frozen = _mapping(frozen, "Stage4 frozen runtime")
        max_steps = int(frozen["max_steps"])
        if extension is not None and max_steps != extension.target_step:
            raise Stage4ContractError(
                "Stage4 extension run contract does not carry its exact target"
            )
        if extension is None and max_steps > configured_steps:
            raise Stage4ContractError(
                "plain Stage4 resume cannot load an extended run contract"
            )
        crop_size = int(frozen["crop_size"])
        micro_batch = int(frozen["micro_batch"])
        micro_batch_trial_evidence = resume_contract.get("micro_batch_trials")
        validation_vram_gate_evidence = resume_contract.get("validation_vram_gate")
        # The frozen CUDA gates are validated from their durable evidence on
        # resume; they are never rerun.  This check happens before any CUDA
        # allocation or checkpoint state mutation.
        stage4_runtime_evidence_metadata(
            micro_batch_trial_evidence,
            validation_vram_gate_evidence,
            selected_crop_size=crop_size,
            selected_micro_batch=micro_batch,
        )
        if arguments.max_steps is not None and arguments.max_steps != max_steps:
            raise Stage4ContractError("--max_steps cannot change across Stage4 resume")
        if arguments.micro_batch is not None and arguments.micro_batch != micro_batch:
            raise Stage4ContractError(
                "--micro_batch cannot change across Stage4 resume"
            )
        # Recompute and compare the full physical provenance before CUDA is
        # queried or any checkpoint state can be installed.  In particular,
        # an activated extension must reject a stale semantic-source map or a
        # mismatched 48k provenance migration without touching the GPU.
        prevalidated_resume_provenance = build_current_provenance(
            selected_crop_size=crop_size,
            selected_micro_batch=micro_batch,
            selected_max_steps=max_steps,
            micro_batch_trials=micro_batch_trial_evidence,
            validation_vram_gate=validation_vram_gate_evidence,
        )
        if resume_contract.get("provenance") != prevalidated_resume_provenance:
            raise Stage4ContractError(
                "Stage4 resume run-contract provenance mismatch before CUDA init"
            )
    else:
        collisions = [
            path
            for path in (
                _contract_path(output_dir),
                output_dir / "last.pth",
                output_dir / "best_ema.pth",
                output_dir / "complete.json",
                output_dir / "train.jsonl",
                calibration_path,
            )
            if path.exists()
        ]
        if collisions:
            raise Stage4ContractError(
                f"refusing to overwrite Stage4 artifacts; use --resume: {collisions}"
            )
        max_steps = arguments.max_steps or training_target_step
        crop_size = -1
        micro_batch = -1

    if not torch.cuda.is_available():
        raise Stage4ContractError("formal Stage4 requires an available CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    _configure_runtime(int(config["seed"]))
    model.to(device)
    set_stage4_trainability(model)
    trials = ()
    validation_vram_gate = None
    if resume_path is None:
        crop_size, micro_batch, trials = choose_stage4_micro_batch(
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
    ema = build_stage4_ema(model, decay=float(config["ema"]["decay"]))
    if resume_path is None:
        validation_vram_gate = probe_stage4_validation_vram(
            model,
            optimizer=optimizer,
            ema=ema,
            device=device,
            image_size=2040,
            max_rounds=3,
            maximum_reserved_fraction=float(
                config["runtime"]["vram_maximum_peak_reserved_fraction"]
            ),
        )
        if optimizer.state:
            raise Stage4ContractError(
                "Stage4 validation gate did not restore an empty step0 optimizer"
            )
        torch.cuda.empty_cache()
        micro_batch_trial_evidence = [asdict(trial) for trial in trials]
        validation_vram_gate_evidence = asdict(validation_vram_gate)

    relation_train = load_relation_records(relation_train_path)
    relation_val = load_relation_records(relation_val_path)
    training_root = Path(str(resolved["training_data_root"])).resolve()
    depth_root = PROJECT_ROOT / "artifacts/cache/agenticir_depth_compat"
    train_manifest = Path(
        str(resolved[config["paths"]["train_manifest_key"]])
    ).resolve()
    val_manifest = Path(str(resolved[config["paths"]["val_manifest_key"]])).resolve()
    train_base = GraphRestoreEpisodeDataset(
        train_manifest,
        training_root,
        depth_root,
        crop_size=crop_size,
        training=True,
        stage="stage4",
        base_seed=int(config["seed"]),
        agenticir_repo=resolved["agenticir_repo"],
        mioir_repo=resolved["mioir_repo"],
    )
    train_dataset = Stage4EpisodeDataset(train_base, relation_train)
    sampler = Stage4EpisodeSampler(
        train_dataset,
        # Preserve the original sampler contract across the exact 40k resume.
        # A fresh iterator yields this many additional cursor-addressed samples,
        # which is more than sufficient for the bounded 40k->48k extension.
        num_samples=configured_steps * 4,
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

    provenance = (
        build_current_provenance(
            selected_crop_size=crop_size,
            selected_micro_batch=micro_batch,
            selected_max_steps=max_steps,
            micro_batch_trials=micro_batch_trial_evidence,
            validation_vram_gate=validation_vram_gate_evidence,
        )
        if prevalidated_resume_provenance is None
        else prevalidated_resume_provenance
    )
    if resume_contract is None:
        atomic_write_json(
            _contract_path(output_dir),
            {
                "schema_version": STAGE4_SCHEMA,
                "created_utc": utc_now_iso(),
                "approval": dict(approval),
                "provenance": provenance,
                "micro_batch_trials": micro_batch_trial_evidence,
                "validation_vram_gate": validation_vram_gate_evidence,
            },
        )
        global_step = 0
        best_score: ValidationScore | None = None
        latest_score: ValidationScore | None = None
        pending_validation_step: int | None = None
    else:
        payload = resume_stage4_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_provenance=provenance,
            expected_validation_every=int(config["validation"]["every_steps"]),
            expected_max_steps=max_steps,
        )
        global_step = int(payload["step"])
        best_score = _checkpoint_best_score(payload)
        latest_score = _checkpoint_current_score(payload)
        pending_value = payload.get("pending_validation_step")
        pending_validation_step = None if pending_value is None else int(pending_value)

    _require_calibration_history_boundary(
        frozen_stage3_history=frozen_calibration_history_path,
        frozen_stage3_sha256=frozen_calibration_sha256,
        stage4_history=calibration_path,
    )
    _require_stage4_calibration_prefix(
        calibration_path,
        checkpoint_step=global_step,
        pending_validation_step=pending_validation_step,
        validation_steps=validation_steps,
    )

    train_log_path = output_dir / "train.jsonl"
    historical_train_peak, historical_validation_peak = (
        _historical_stage4_peak_reserved(train_log_path)
    )
    train_log = train_log_path.open("a", encoding="utf-8")
    maximum_peak_fraction = float(
        config["runtime"]["vram_maximum_peak_reserved_fraction"]
    )
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    maximum_train_peak_reserved = historical_train_peak
    maximum_validation_peak_reserved = historical_validation_peak
    # A restored pending transaction is already validation-in-progress for
    # signal safety.  Publish that fact before entering the try/replay window;
    # otherwise a SIGTERM immediately before run_validation_boundary() could
    # overwrite the durable pending marker with ``None``.
    validation_in_progress_step: int | None = pending_validation_step
    training_update_in_progress = False
    if resume_path is None:
        # The fresh step-0 raw anchor is the only legal fallback if a signal
        # lands inside the first optimizer transaction.
        save_stage4_checkpoint(
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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:

        def run_validation_boundary(*, replay_pending: bool) -> None:
            nonlocal best_score
            nonlocal latest_score
            nonlocal maximum_train_peak_reserved
            nonlocal maximum_validation_peak_reserved
            nonlocal validation_in_progress_step

            validation_in_progress_step = global_step
            if not replay_pending:
                save_stage4_checkpoint(
                    output_dir / "last.pth",
                    step=global_step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    provenance=provenance,
                    metrics=_checkpoint_metrics(
                        latest_score,
                        best_score,
                        best_checkpoint=(
                            output_dir / "best_ema.pth"
                            if best_score is not None
                            else None
                        ),
                    ),
                    pending_validation_step=global_step,
                )
            append_jsonl(
                train_log,
                {
                    "schema_version": STAGE4_SCHEMA,
                    "event": (
                        "replay_pending_validation"
                        if replay_pending
                        else "pre_validation_checkpoint"
                    ),
                    "created_utc": utc_now_iso(),
                    "step": global_step,
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
                    summary = validate_stage4(
                        model,
                        validation_dataset,
                        device=device,
                        relation_val_records=relation_val,
                        use_bf16=True,
                    )
                set_stage4_trainability(model)
                torch.cuda.synchronize(device)
                validation_peak_reserved = int(torch.cuda.max_memory_reserved(device))
                maximum_validation_peak_reserved = max(
                    maximum_validation_peak_reserved,
                    validation_peak_reserved,
                )
                validation_peak_fraction = validation_peak_reserved / total_memory
                if validation_peak_fraction > maximum_peak_fraction:
                    raise Stage4ContractError(
                        "Stage4 validation peak reserved fraction "
                        f"{validation_peak_fraction:.4f} exceeded the frozen "
                        f"{maximum_peak_fraction:.2f} ceiling"
                    )
                latest_score = stage4_validation_score(summary, global_step)
                improved = is_better_checkpoint(latest_score, best_score)
                if improved:
                    best_score = latest_score
                assert best_score is not None
                selection_metrics = _checkpoint_metrics(latest_score, best_score)
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
                        metrics=selection_metrics,
                        model_as_ema=True,
                    )
                metrics = _checkpoint_metrics(
                    latest_score,
                    best_score,
                    best_checkpoint=output_dir / "best_ema.pth",
                )
                atomic_write_json(output_dir / "validation_latest.json", summary)
                _require_calibration_history_boundary(
                    frozen_stage3_history=frozen_calibration_history_path,
                    frozen_stage3_sha256=frozen_calibration_sha256,
                    stage4_history=calibration_path,
                )
                _append_calibration_history(
                    calibration_path,
                    step=global_step,
                    summary=summary,
                    validation_steps=validation_steps,
                )
                _require_calibration_history_boundary(
                    frozen_stage3_history=frozen_calibration_history_path,
                    frozen_stage3_sha256=frozen_calibration_sha256,
                    stage4_history=calibration_path,
                )
                atomic_write_text(
                    report_path,
                    _render_report(
                        summary,
                        step=global_step,
                        best=best_score,
                        checkpoint=output_dir / "best_ema.pth",
                        calibration_history_routing=calibration_history_routing,
                        stage4_extension=extension,
                    ),
                )
                append_jsonl(
                    train_log,
                    {
                        "schema_version": STAGE4_SCHEMA,
                        "event": "validation",
                        "created_utc": utc_now_iso(),
                        "step": global_step,
                        "improved": improved,
                        "replayed_pending": replay_pending,
                        "peak_reserved_bytes": validation_peak_reserved,
                        "peak_reserved_fraction": validation_peak_fraction,
                        "group_a_psnr": latest_score.group_a_psnr,
                        "group_a_ssim": latest_score.group_a_ssim,
                        "single_psnr": latest_score.single_psnr,
                        "single_ssim": latest_score.single_ssim,
                    },
                )
                _update_stage4_running_status(
                    PROJECT_ROOT / "RUNNING_STATUS.md",
                    latest=latest_score,
                )
                # Clearing pending is the final validation transaction commit,
                # after best/metric/calibration/report and the recognizable
                # validation event are all visible.  A crash before this save
                # replays the same idempotent artifact writes; the log records
                # the replay explicitly via ``replayed_pending``.
                _require_calibration_history_boundary(
                    frozen_stage3_history=frozen_calibration_history_path,
                    frozen_stage3_sha256=frozen_calibration_sha256,
                    stage4_history=calibration_path,
                )
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
                    pending_validation_step=None,
                )
                validation_in_progress_step = None
            finally:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

        if pending_validation_step is not None:
            run_validation_boundary(replay_pending=True)
            pending_validation_step = None
        # Replaying a durable validation transaction must precede iterator
        # construction and therefore all multi-worker sample prefetch.
        iterator = iter(train_loader) if global_step < max_steps else None
        while global_step < max_steps:
            if iterator is None:
                raise Stage4ContractError("Stage4 training iterator is unavailable")
            micro_batches = [next(iterator) for _ in range(accumulation)]
            training_update_in_progress = True
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
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            maximum_train_peak_reserved = max(
                maximum_train_peak_reserved, peak_reserved
            )
            peak_reserved_fraction = peak_reserved / total_memory
            if (
                peak_reserved < 0
                or not math.isfinite(peak_reserved_fraction)
                or peak_reserved_fraction > maximum_peak_fraction
            ):
                raise Stage4ContractError(
                    "Stage4 training peak reserved fraction is invalid or exceeded "
                    f"the frozen 0.90 ceiling: {peak_reserved_fraction:.4f}"
                )
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
                    "peak_reserved_bytes": peak_reserved,
                    "peak_reserved_fraction": peak_reserved_fraction,
                },
            )
            validate_now = (
                global_step % int(config["validation"]["every_steps"]) == 0
                or global_step == max_steps
            )
            if validate_now:
                # Arm the validation marker while the optimizer transaction is
                # still non-checkpointable.  A SIGTERM between this boundary
                # and the durable pending raw checkpoint must preserve the
                # prior raw checkpoint instead of publishing pending=None and
                # silently skipping the due validation on resume.
                validation_in_progress_step = global_step
            # The optimizer transaction becomes signal-saveable only after
            # step/sampler, the finite <=90% VRAM guard, and the flushed
            # train_step audit record all describe the same boundary.  A due
            # validation is armed before this flag is cleared.
            training_update_in_progress = False
            if validate_now:
                run_validation_boundary(replay_pending=False)
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
                    metrics=_checkpoint_metrics(
                        latest_score,
                        best_score,
                        best_checkpoint=(
                            output_dir / "best_ema.pth"
                            if best_score is not None
                            else None
                        ),
                    ),
                    pending_validation_step=None,
                )
    except RuntimeError as exc:
        if not is_stage4_cuda_oom_exception(exc):
            raise
        _record_stage4_runtime_oom(
            output_dir=output_dir,
            error=exc,
            attempted_step=global_step,
            mid_optimizer_update=training_update_in_progress,
            pending_validation_step=validation_in_progress_step,
            crop_size=crop_size,
            micro_batch=micro_batch,
            allocator_conf=allocator_conf,
        )
        raise
    except KeyboardInterrupt:
        if _stage4_interrupt_can_checkpoint(
            mid_optimizer_update=training_update_in_progress,
            pending_validation_step=validation_in_progress_step,
        ):
            save_stage4_checkpoint(
                output_dir / "last.pth",
                step=global_step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                provenance=provenance,
                metrics=_checkpoint_metrics(
                    latest_score,
                    best_score,
                    best_checkpoint=(
                        output_dir / "best_ema.pth" if best_score is not None else None
                    ),
                ),
            )
        append_jsonl(
            train_log,
            {
                "schema_version": STAGE4_SCHEMA,
                "event": "interrupted",
                "created_utc": utc_now_iso(),
                "step": global_step,
                "mid_optimizer_update": training_update_in_progress,
                "pending_validation_step": validation_in_progress_step,
                "checkpoint_advanced": (
                    not training_update_in_progress
                    and validation_in_progress_step is None
                ),
            },
        )
        raise
    finally:
        train_log.close()

    if (
        best_score is None
        or latest_score is None
        or not (output_dir / "best_ema.pth").is_file()
    ):
        raise Stage4ContractError("Stage4 completed without a selected EMA checkpoint")
    _require_calibration_history_boundary(
        frozen_stage3_history=frozen_calibration_history_path,
        frozen_stage3_sha256=frozen_calibration_sha256,
        stage4_history=calibration_path,
    )
    completed_calibration_steps = tuple(
        int(row["step"])
        for row in _load_stage4_calibration_rows(
            calibration_path, validation_steps=validation_steps
        )
    )
    if completed_calibration_steps != tuple(validation_steps):
        raise Stage4ContractError(
            "Stage4 calibration history does not contain every authorized validation"
        )
    completed_calibration_sha256 = sha256_file(calibration_path)
    try:
        diagnostics = run_stage4_zero_training_diagnostics(
            model,
            ema,
            validation_dataset,
            device=device,
            relation_val_records=relation_val,
            selected_best_checkpoint=output_dir / "best_ema.pth",
            expected_provenance=provenance,
            json_path=diagnostics_json_path,
            report_path=diagnostics_report_path,
            maximum_reserved_fraction=maximum_peak_fraction,
            use_bf16=True,
        )
    except RuntimeError as exc:
        if not is_stage4_cuda_oom_exception(exc):
            raise
        _record_stage4_runtime_oom(
            output_dir=output_dir,
            error=exc,
            attempted_step=global_step,
            mid_optimizer_update=False,
            pending_validation_step=None,
            crop_size=crop_size,
            micro_batch=micro_batch,
            allocator_conf=allocator_conf,
        )
        raise
    if not diagnostics_report_path.is_file() or not diagnostics_json_path.is_file():
        raise Stage4ContractError("Stage4 diagnostics are required before completion")
    report_binding = _stage4_report_binding(
        report_path,
        selected_best_checkpoint=output_dir / "best_ema.pth",
        selected_best_score=best_score,
        latest_score=latest_score,
        calibration_history_routing=calibration_history_routing,
        completed_calibration_sha256=completed_calibration_sha256,
        validation_steps=validation_steps,
        stage4_extension=extension,
    )
    selected_psnr_delta = best_score.group_a_psnr - STAGE0_GROUP_A_PSNR_ANCHOR
    selected_ssim_delta = best_score.group_a_ssim - STAGE0_GROUP_A_SSIM_ANCHOR
    ssim_retention_risk = best_score.group_a_ssim < STAGE0_GROUP_A_SSIM_ANCHOR
    _require_calibration_history_boundary(
        frozen_stage3_history=frozen_calibration_history_path,
        frozen_stage3_sha256=frozen_calibration_sha256,
        stage4_history=calibration_path,
    )
    if sha256_file(calibration_path) != completed_calibration_sha256:
        raise Stage4ContractError(
            "Stage4 calibration history changed during final diagnostics"
        )
    _update_stage4_running_status(
        PROJECT_ROOT / "RUNNING_STATUS.md",
        latest=latest_score,
        selected=best_score,
    )
    _update_stage4_decision_memo(PROJECT_ROOT / "DECISION_MEMO.md", best_score)
    completion_payload: dict[str, Any] = {
        "schema_version": STAGE4_SCHEMA,
        "protocol_id": config["protocol_id"],
        "completed_utc": utc_now_iso(),
        "step": global_step,
        "best_ema_path": str((output_dir / "best_ema.pth").resolve()),
        "best_ema_sha256": sha256_file(output_dir / "best_ema.pth"),
        **report_binding,
        "diagnostics_json": str(diagnostics_json_path),
        "diagnostics_json_sha256": sha256_file(diagnostics_json_path),
        "diagnostics_report": str(diagnostics_report_path),
        "diagnostics_report_sha256": sha256_file(diagnostics_report_path),
        "diagnostics_selected_best_ema_sha256": diagnostics["selected_best_ema_sha256"],
        "maximum_train_peak_reserved_bytes": maximum_train_peak_reserved,
        "maximum_train_peak_reserved_fraction": (
            maximum_train_peak_reserved / total_memory
        ),
        "maximum_validation_peak_reserved_bytes": maximum_validation_peak_reserved,
        "maximum_validation_peak_reserved_fraction": (
            maximum_validation_peak_reserved / total_memory
        ),
        "best_score": {
            "group_a_psnr": best_score.group_a_psnr,
            "group_a_ssim": best_score.group_a_ssim,
            "single_psnr": best_score.single_psnr,
            "single_ssim": best_score.single_ssim,
            "step": best_score.step,
        },
        "validation": str((output_dir / "validation_latest.json").resolve()),
        "validation_sha256": sha256_file(output_dir / "validation_latest.json"),
        "calibration_history": str(calibration_path.resolve()),
        "calibration_history_sha256": completed_calibration_sha256,
        "calibration_history_steps": list(completed_calibration_steps),
        "frozen_stage3_calibration_history": {
            "path": str(frozen_calibration_history_path.resolve()),
            "sha256": frozen_calibration_sha256,
        },
        "latest_score": {
            "group_a_psnr": latest_score.group_a_psnr,
            "group_a_ssim": latest_score.group_a_ssim,
            "single_psnr": latest_score.single_psnr,
            "single_ssim": latest_score.single_ssim,
            "step": latest_score.step,
        },
        "stage0_group_a_psnr_anchor": STAGE0_GROUP_A_PSNR_ANCHOR,
        "stage0_group_a_ssim_anchor": STAGE0_GROUP_A_SSIM_ANCHOR,
        "selected_group_a_psnr": best_score.group_a_psnr,
        "selected_delta_group_a_psnr_vs_stage0": selected_psnr_delta,
        "selected_group_a_ssim": best_score.group_a_ssim,
        "selected_delta_group_a_ssim_vs_stage0": selected_ssim_delta,
        "SSIM_RETENTION_RISK": ssim_retention_risk,
        "formal_mio100_started": False,
        "waiting_for": "new_user_authorization_for_formal_mio100",
    }
    if extension is not None:
        completion_payload.update(
            {
                "stage4_extension_conditional_authorization": str(
                    extension.conditional_path
                ),
                "stage4_extension_conditional_authorization_sha256": (
                    extension.conditional_sha256
                ),
                "stage4_extension_gate_receipt": str(extension.gate_path),
                "stage4_extension_gate_receipt_sha256": extension.gate_sha256,
                "stage4_extension_validation_steps": list(extension.validation_steps),
                "schedule_horizon_steps": extension.schedule_horizon_steps,
                "training_target_step": extension.target_step,
                "additional_optimizer_steps": (
                    extension.target_step - extension.base_step
                ),
                "further_extension_authorized": False,
            }
        )
    atomic_write_json(output_dir / "complete.json", completion_payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    previous_sigterm = signal.signal(signal.SIGTERM, _sigterm_as_keyboard_interrupt)
    try:
        return run(build_parser().parse_args(argv))
    except KeyboardInterrupt:
        print(
            "Stage4 interrupted; the latest complete optimizer boundary remains resumable",
            file=sys.stderr,
        )
        return 130
    except (
        Stage3ContractError,
        Stage4ContractError,
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as exc:
        print(f"STAGE4_REFUSED: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        if not is_stage4_cuda_oom_exception(exc):
            raise
        print(
            "STAGE4_OOM_FAIL_SAFE: raw checkpoint preserved; resume in a new process",
            file=sys.stderr,
        )
        return 3
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
