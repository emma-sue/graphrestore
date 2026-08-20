"""Stage3 planner/guard supervision under the frozen V7.1 contract.

The module is deliberately split into a file-only approval preflight and the
CUDA/data portion.  A caller must successfully validate the approval produced
by :mod:`src.training.orchestration` before constructing a dataset, probing a
GPU, or loading the Stage1 executor.
"""

from __future__ import annotations

import csv
import hashlib
import io
import importlib.metadata
import math
import os
import platform
import random
import stat
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from src.data import GraphRestoreEpisodeDataset, StatefulEpisodeSampler
from src.data.manifests import SKILLS, SKILL_TO_ID
from src.data.subset_targets import DENSE_GUARD_SKILLS
from src.losses.guard_losses import guard_supervision_loss
from src.losses.planner_losses import PlannerLossBreakdown, planner_loss
from src.metrics.agenticir_official import official_psnr_ssim
from src.net import GraphRestore, PAIR_INDICES, PlannerOutput
from src.net.restormer_blocks import pad_to_multiple
from src.training.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    checkpoint_payload,
    restore_rng_state,
    unwrap_model,
)
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import WarmupCosineScheduler
from src.training.provenance import semantic_source_hashes
from src.training.relation_supervision import non_ambiguous_relation_metrics
from src.training.selection import ValidationScore, is_better_checkpoint
from src.training.stage1_engine import STAGE1_EMA_SCOPE, stage1_ema_policy_metadata
from src.utils.git import git_commit
from src.utils.hashing import is_sha256, sha256_file, sha256_json
from src.utils.io import (
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_json,
    load_yaml,
    utc_now_iso,
)


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
APPROVAL_SCHEMA = "graphrestore-stage3-approval-v1"
ORCHESTRATION_SCHEMA = "graphrestore-orchestration-v1"
STAGE3_SCHEMA = "graphrestore-stage3-runtime-v1"
THRESHOLD_SCHEMA = "graphrestore-presence-thresholds-v1"
THRESHOLD_TIE_BREAK = "nearest_0.50_then_higher_threshold"
THRESHOLD_F1_TOLERANCE = 1.0e-15
STAGE3_EXTENSION_SCHEMA = "graphrestore-stage3-extension-approval-v1"
STAGE3_EXTENSION_FILENAME = "STAGE3_EXTENSION_APPROVED.json"
STAGE3_EXTENSION_MIGRATION_NAME = "stage3_extension_12000_to_18000_v1"
STAGE3_BASE_TARGET_STEP = 12_000
STAGE3_EXTENSION_TARGET_STEP = 18_000
STAGE3_EXTENSION_VALIDATION_STEPS = (14_000, 16_000, 18_000)
STAGE3_EXTENSION_LR_POLICY = "hold_original_cosine_floor_after_schedule_horizon"
STAGE3_ALLOCATOR_CONF = "backend:native,expandable_segments:True"
STAGE3_EMA_SCOPE = "planner_parameters_only_executor_bitwise_frozen"
STAGE3_EMA_SCHEMA = "graphrestore-stage3-planner-ema-policy-v1"
RELATION_CLASSES = ("i_before_j", "j_before_i", "parallel")
PAIR_TO_ROW = {pair: index for index, pair in enumerate(PAIR_INDICES)}
_FORBIDDEN_STAGE3_TOKENS = ("mio100", "group_b", "group_c", "exploration")


class Stage3ContractError(RuntimeError):
    """Stage3 would violate approval, data, supervision, or runtime locks."""


def stage3_ema_policy_metadata(decay: float) -> dict[str, object]:
    if not 0.0 < decay < 1.0:
        raise ValueError("Stage3 EMA decay must be in (0,1)")
    return {
        "schema_version": STAGE3_EMA_SCHEMA,
        "scope": STAGE3_EMA_SCOPE,
        "parameter_selector": "state_name_prefix_planner_dot",
        "planner_parameter_update": "standard_fp32_exponential_moving_average",
        "frozen_parameter_update": "copy_current_value_bitwise",
        "buffer_update": "copy_current_value_bitwise",
        "phase_transition": "single_phase_without_shadow_reset",
        "decay": float(decay),
    }


@dataclass
class Stage3OptimizerTransaction:
    """Track whether an optimizer update reached a serializable boundary.

    The caller intentionally keeps the transaction active until the optimizer,
    scheduler, EMA, logical step, sampler cursor, VRAM guard, durable train-step
    log, and any due validation marker have all advanced.  A signal in any
    smaller window must leave the prior atomic checkpoint in place instead of
    serializing a mixed or not-yet-audited step.
    """

    active: bool = False

    def begin(self) -> None:
        if self.active:
            raise Stage3ContractError("Stage3 optimizer transaction is already active")
        self.active = True

    def commit(self) -> None:
        if not self.active:
            raise Stage3ContractError("Stage3 optimizer transaction is not active")
        self.active = False


class Stage3PlannerEMA(ExponentialMovingAverage):
    """EMA planner parameters while preserving the frozen executor bitwise.

    Applying the usual ``a*x + (1-a)*x`` update to an unchanged FP32 executor
    can still round by an ULP.  That would make the selected Stage3 executor no
    longer exactly equal to its frozen Stage1 parent.  Fixed buffers (including
    effect profiles) are copied exactly as well.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__(model, decay=decay)
        self.ema_parameter_names = frozenset(
            name
            for name, _ in unwrap_model(model).named_parameters()
            if name.startswith("planner.")
        )
        if not self.ema_parameter_names:
            raise Stage3ContractError("Stage3 planner EMA has no planner parameters")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = unwrap_model(model).state_dict()
        if source.keys() != self.shadow.keys():
            raise RuntimeError("Stage3 EMA/model state keys drifted")
        self.num_updates += 1
        for name, value in source.items():
            target = self.shadow[name]
            if name in self.ema_parameter_names:
                target.mul_(self.decay).add_(
                    value.detach().to(target), alpha=1.0 - self.decay
                )
            else:
                target.copy_(value.detach().to(target))

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["scope"] = STAGE3_EMA_SCOPE
        state["policy"] = stage3_ema_policy_metadata(self.decay)
        return state


@dataclass(frozen=True)
class Stage3ApprovalEvidence:
    approval_path: Path
    approval_sha256: str
    approval_required_path: Path
    approval_required_sha256: str
    stage2_decision_sha256: str
    bindings: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class Stage3ExtensionEvidence:
    authorization_path: Path
    authorization_sha256: str
    base_step: int
    target_step: int
    cycles: int
    validation_every_steps: int
    validation_steps: tuple[int, ...]
    schedule_horizon_steps: int
    min_lr: float
    lr_policy: str

    def provenance_binding(self) -> dict[str, Any]:
        return {
            "path": str(self.authorization_path),
            "sha256": self.authorization_sha256,
            "cycles": self.cycles,
            "base_step": self.base_step,
            "target_step": self.target_step,
            "validation_every_steps": self.validation_every_steps,
            "validation_steps": list(self.validation_steps),
            "schedule_horizon_steps": self.schedule_horizon_steps,
            "min_lr": self.min_lr,
            "lr_policy": self.lr_policy,
        }


@dataclass(frozen=True)
class Stage3Paths:
    project_root: Path
    config_path: Path
    config: Mapping[str, Any]
    resolved_path: Path
    resolved: Mapping[str, Any]
    training_data_root: Path
    train_manifest: Path
    val_manifest: Path
    executor_checkpoint: Path
    effect_profiles: Path
    relation_train: Path
    relation_val: Path
    pair_prior: Path
    global_priority: Path
    stage2_decision: Path
    output_dir: Path
    thresholds: Path
    calibration_history: Path
    report: Path
    approval: Stage3ApprovalEvidence


@dataclass(frozen=True)
class Stage3ParentLoadReport:
    checkpoint_sha256: str
    checkpoint_step: int
    loaded_count: int
    initialized_planner_keys: tuple[str, ...]


@dataclass
class Stage3SupervisionBatch:
    x0: Tensor
    current: Tensor
    presence_targets: Tensor
    guard_targets: Tensor
    global_severity_targets: Tensor
    dense_skill_mask: Tensor
    global_skill_mask: Tensor
    absent_skill_mask: Tensor
    stop_targets: Tensor
    relation_targets: Tensor
    relation_weights: Tensor
    relation_ambiguous_mask: Tensor
    round_values: Tensor
    sample_ids: tuple[str, ...]
    state_kinds: tuple[str, ...]
    model_intermediate_count: int


@dataclass(frozen=True)
class Stage3StepResult:
    total: float
    presence: float
    guard: float
    relation: float
    stop: float
    cycle: float
    guard_dense: float
    guard_global_mean: float
    guard_absent: float
    grad_norm: float
    samples: int
    model_intermediate_count: int
    state_counts: Mapping[str, int]
    seconds: float


@dataclass(frozen=True)
class Stage3MicroBatchTrial:
    micro_batch: int
    passed: bool
    completed_optimizer_steps: int
    images_per_second: float
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    error: str | None = None


@dataclass(frozen=True)
class Stage3ValidationVRAMTopology:
    compiler_mode: str
    active_skill_count: int
    completed_rounds: int
    active_skill_counts_by_round: tuple[int, ...]
    metric_psnr: float
    metric_ssim: float
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    finite: bool
    passed: bool


@dataclass(frozen=True)
class Stage3ValidationVRAMGate:
    schema_version: str
    image_size: int
    max_rounds: int
    completed_rounds: int
    topologies: tuple[Stage3ValidationVRAMTopology, ...]
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    maximum_peak_reserved_fraction: float
    resident_optimizer_state_entries: int
    resident_optimizer_state_bytes: int
    resident_ema_bytes: int
    optimizer_state_empty_after: bool
    finite: bool
    passed: bool


@dataclass(frozen=True)
class ThresholdCalibration:
    thresholds: tuple[float, ...]
    per_skill_f1: tuple[float, ...]
    grid: tuple[float, ...]
    baseline_diagnostics: Mapping[str, Any]
    calibrated_diagnostics: Mapping[str, Any]
    tie_break: str = THRESHOLD_TIE_BREAK


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage3ContractError(f"{field} must be a mapping")
    return value


def _project_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise Stage3ContractError(f"{field} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve(strict=False)


def _expect(config: Mapping[str, Any], path: Sequence[str], expected: object) -> None:
    value: object = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise Stage3ContractError(f"missing Stage3 config key: {'.'.join(path)}")
        value = value[key]
    if value != expected:
        raise Stage3ContractError(
            f"Stage3 config drift at {'.'.join(path)}: expected {expected!r}, got {value!r}"
        )


def validate_stage3_config(config: Mapping[str, Any]) -> None:
    """Fail closed on all scientifically meaningful Stage3 settings."""

    locked: tuple[tuple[tuple[str, ...], object], ...] = (
        (("schema_version",), "1.0"),
        (("contract_version",), "GraphRestore-V7.1"),
        (("protocol_id",), PROTOCOL_ID),
        (("stage",), "stage3"),
        (("seed",), 2027),
        (("skills", "ordered_names"), list(SKILLS)),
        (("skills", "maximum_active"), 3),
        (("data", "allowed_groups"), ["single", "A"]),
        (("data", "forbidden_groups"), ["B", "C"]),
        (("data", "crop_size"), 192),
        (("data", "crop_multiple"), 4),
        (("data", "effective_batch_size"), 8),
        (("data", "validation_augmentation"), False),
        (("data", "states", "clean", "stop_target"), 1),
        (("data", "states", "clean", "all_presence_zero"), True),
        (("data", "states", "clean", "all_guards_zero"), True),
        (("data", "states", "model_generated_intermediate", "maximum_fraction"), 0.10),
        (("model", "executor"), "frozen_stage1_ema"),
        (("model", "batchnorm_forbidden"), True),
        (("model", "relation_classes"), list(RELATION_CLASSES)),
        (("model", "shared_relation_mlp"), True),
        (("model", "independent_pair_lookup_heads_forbidden"), True),
        (("model", "round_embedding"), "continuous_sinusoidal_mlp"),
        (("loss", "relation", "class_count"), 3),
        (("loss", "relation", "ambiguous_serial_mass_weight"), 0.25),
        (("loss", "relation", "prohibit_double_weighting"), True),
        (("loss", "relation", "ambiguous_excluded_from_metrics_and_priors"), True),
        (("training", "max_steps"), 12_000),
        (("training", "trainable"), ["planner"]),
        (("optimization", "optimizer"), "AdamW"),
        (("optimization", "lr"), 2.0e-4),
        (("optimization", "weight_decay"), 1.0e-4),
        (("optimization", "warmup_steps"), 500),
        (("optimization", "scheduler"), "cosine"),
        (("optimization", "min_lr"), 2.0e-6),
        (("optimization", "gradient_clip_norm"), 1.0),
        (("runtime", "amp_dtype"), "bf16"),
        (("runtime", "tf32"), True),
        (("runtime", "validation_every_steps"), 2000),
        (("runtime", "freeze_crop_micro_accum_after_step0"), True),
        (("ema", "enabled"), True),
        (("ema", "decay"), 0.9999),
        (("validation", "relation_validation_source"), "interaction_val"),
        (("validation", "relation_train_as_validation_forbidden"), True),
        (("validation", "checkpoint_presence_threshold"), 0.50),
        (("threshold_calibration", "when"), "once_after_stage3_snapshot_selection"),
        (("threshold_calibration", "minimum"), 0.20),
        (("threshold_calibration", "maximum"), 0.80),
        (("threshold_calibration", "step"), 0.02),
        (("threshold_calibration", "freeze_after_selection"), True),
        (("threshold_calibration", "mio100_forbidden"), True),
        (("hard_guards", "require_explicit_stage3_approval"), True),
        (("hard_guards", "require_stage2_hash_match"), True),
        (("hard_guards", "allow_mio100_exploration"), False),
        (("hard_guards", "allow_mio100_formal"), False),
        (("hard_guards", "allow_group_b_or_c_training"), False),
        (("hard_guards", "post_compiler_cycle_rate_required"), 0.0),
        (("hard_guards", "fail_on_hash_mismatch"), True),
    )
    for path, expected in locked:
        _expect(config, path, expected)


def validate_stage3_allocator_conf(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Require the allocator proven equivalent during the Stage0 A/B audit.

    This check is deliberately environment-only so the formal CLI can execute
    it before its first CUDA availability/current-device query.
    """

    environment = os.environ if environ is None else environ
    actual = environment.get("PYTORCH_CUDA_ALLOC_CONF")
    if actual != STAGE3_ALLOCATOR_CONF:
        raise Stage3ContractError(
            "formal Stage3 requires exact PYTORCH_CUDA_ALLOC_CONF="
            f"{STAGE3_ALLOCATOR_CONF!s}; got {actual!r}"
        )
    return actual


def validate_stage3_pending_validation_step(
    *,
    step: object,
    pending_validation_step: object,
    max_steps: int,
    validation_every_steps: int = 2_000,
) -> int | None:
    """Validate the raw-checkpoint validation transaction marker."""

    if max_steps <= 0 or validation_every_steps <= 0:
        raise Stage3ContractError("invalid Stage3 checkpoint schedule")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= max_steps
    ):
        raise Stage3ContractError("invalid Stage3 checkpoint step")
    if pending_validation_step is None:
        return None
    if isinstance(pending_validation_step, bool) or not isinstance(
        pending_validation_step, int
    ):
        raise Stage3ContractError("invalid Stage3 pending_validation_step")
    if pending_validation_step != step:
        raise Stage3ContractError(
            "Stage3 pending_validation_step must equal checkpoint step"
        )
    if step <= 0 or not (step % validation_every_steps == 0 or step == max_steps):
        raise Stage3ContractError(
            "Stage3 pending validation is not on a validation boundary"
        )
    return pending_validation_step


def reset_stage3_peak_memory(device: torch.device) -> None:
    """Start an independent Stage3 train/validation peak measurement."""

    if device.type != "cuda":
        raise Stage3ContractError("Stage3 VRAM accounting requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)


def enforce_stage3_peak_memory(
    device: torch.device,
    *,
    phase: str,
    maximum_reserved_fraction: float = 0.90,
) -> tuple[int, float]:
    """Fail closed when one independently reset phase exceeds the VRAM cap."""

    if device.type != "cuda":
        raise Stage3ContractError("Stage3 VRAM accounting requires CUDA")
    if maximum_reserved_fraction != 0.90:
        raise Stage3ContractError("Stage3 VRAM ceiling must remain exactly 0.90")
    if not phase:
        raise ValueError("Stage3 VRAM phase must be non-empty")
    torch.cuda.synchronize(device)
    total = int(torch.cuda.get_device_properties(device).total_memory)
    peak = int(torch.cuda.max_memory_reserved(device))
    if total <= 0 or peak < 0:
        raise Stage3ContractError("invalid Stage3 CUDA memory accounting")
    fraction = peak / total
    if not math.isfinite(fraction) or fraction > maximum_reserved_fraction:
        raise Stage3ContractError(
            f"Stage3 {phase} peak reserved fraction {fraction:.6f} exceeds 0.90"
        )
    return peak, fraction


def _verified_binding(
    bindings: Mapping[str, Any], logical: str, *, expected_path: Path | None = None
) -> Mapping[str, str]:
    value = bindings.get(logical)
    if not isinstance(value, Mapping):
        raise Stage3ContractError(f"approval lacks binding {logical!r}")
    path_raw, digest = value.get("path"), value.get("sha256")
    if not isinstance(path_raw, str) or not is_sha256(digest):
        raise Stage3ContractError(f"invalid approval binding {logical!r}")
    path = Path(path_raw).resolve(strict=False)
    if expected_path is not None and path != expected_path.resolve(strict=False):
        raise Stage3ContractError(
            f"approval binding path mismatch for {logical}: {path} != {expected_path}"
        )
    if not path.is_file():
        raise Stage3ContractError(f"approved artifact disappeared: {logical}={path}")
    actual = sha256_file(path)
    if actual != digest:
        raise Stage3ContractError(
            f"approved artifact hash changed: {logical}: expected {digest}, got {actual}"
        )
    return {"path": str(path), "sha256": str(digest)}


def validate_stage3_approval(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    require_orchestrator_running: bool = True,
    allow_failed_resume: bool = False,
) -> Stage3Paths:
    """Validate explicit approval and every frozen binding using file I/O only.

    This function intentionally performs no CUDA query, tensor allocation,
    checkpoint load, dataset construction, or image read.  It is therefore the
    mandatory first call in both Stage3 CLIs.
    """

    config_file = Path(config_path).resolve()
    root = Path(project_root or config_file.parents[1]).resolve()
    config = _mapping(load_yaml(config_file), field="Stage3 config")
    validate_stage3_config(config)
    path_config = _mapping(config.get("paths"), field="Stage3 paths")
    resolved_path = _project_path(
        root, path_config.get("resolved_paths"), field="resolved_paths"
    )
    resolved = _mapping(load_yaml(resolved_path), field="resolved paths")

    approval_path = _project_path(
        root, path_config.get("required_approval"), field="required_approval"
    )
    if not approval_path.is_file():
        raise Stage3ContractError(
            "Stage3 approval is missing; only the orchestrator command with both "
            "--approve_stage3 and --resume_from_stage3 may create it"
        )
    approval = _mapping(load_json(approval_path), field="STAGE3_APPROVED.json")
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA
        or approval.get("kind") != "stage3_approval"
        or approval.get("protocol_id") != PROTOCOL_ID
        or approval.get("approved") is not True
    ):
        raise Stage3ContractError("invalid Stage3 approval marker")
    command = approval.get("approval_command")
    if not isinstance(command, list) or not {
        "--approve_stage3",
        "--resume_from_stage3",
    }.issubset(command):
        raise Stage3ContractError("approval lacks both explicit orchestrator flags")
    bindings = _mapping(approval.get("bindings"), field="approval.bindings")

    # Hash every recorded Stage2/config/manifest binding, not only the files
    # consumed directly below.  This catches edits after the user approved.
    verified_bindings: dict[str, Mapping[str, str]] = {}
    for logical in sorted(bindings):
        verified_bindings[logical] = _verified_binding(bindings, str(logical))

    _verified_binding(bindings, "config_stage3", expected_path=config_file)
    _verified_binding(bindings, "config_resolved_paths", expected_path=resolved_path)

    train_manifest = _project_path(
        root, resolved.get(path_config.get("train_manifest_key")), field="primary_train"
    )
    val_manifest = _project_path(
        root, resolved.get(path_config.get("val_manifest_key")), field="primary_val"
    )
    executor = _project_path(
        root, path_config.get("executor_checkpoint"), field="executor_checkpoint"
    )
    relation_train = _project_path(
        root, path_config.get("relation_train"), field="relation_train"
    )
    relation_val = _project_path(
        root, path_config.get("relation_val"), field="relation_val"
    )
    pair_prior = _project_path(root, path_config.get("pair_prior"), field="pair_prior")
    global_priority = _project_path(
        root, path_config.get("global_priority"), field="global_priority"
    )
    for logical, path in (
        ("primary_train_manifest", train_manifest),
        ("primary_val_manifest", val_manifest),
        ("stage1_checkpoint", executor),
        ("relation_train", relation_train),
        ("relation_val", relation_val),
        ("pair_prior", pair_prior),
        ("global_priority", global_priority),
    ):
        _verified_binding(bindings, logical, expected_path=path)
        if any(token in str(path).lower() for token in _FORBIDDEN_STAGE3_TOKENS):
            raise Stage3ContractError(f"forbidden Stage3 path: {path}")

    effect_binding = _verified_binding(bindings, "skill_effect_profiles")
    effect_profiles = Path(effect_binding["path"])
    stage2_binding = _verified_binding(bindings, "stage2_decision")
    stage2_decision = Path(stage2_binding["path"])

    required_raw = approval.get("approval_required_path")
    required_sha = approval.get("approval_required_sha256")
    if not isinstance(required_raw, str) or not is_sha256(required_sha):
        raise Stage3ContractError("approval-required binding is invalid")
    approval_required = Path(required_raw).resolve(strict=False)
    if (
        not approval_required.is_file()
        or sha256_file(approval_required) != required_sha
    ):
        raise Stage3ContractError("approval-required marker changed after approval")
    required = _mapping(
        load_json(approval_required), field="STAGE3_APPROVAL_REQUIRED.json"
    )
    if (
        required.get("schema_version") != APPROVAL_SCHEMA
        or required.get("kind") != "stage3_approval_required"
        or required.get("approved") is not False
        or required.get("bindings") != bindings
    ):
        raise Stage3ContractError("approval-required marker no longer matches approval")
    if approval.get("stage2_decision_sha256") != stage2_binding["sha256"]:
        raise Stage3ContractError(
            "approved Stage2 decision SHA does not match its binding"
        )

    decision = _mapping(load_json(stage2_decision), field="stage2_decision.json")
    if decision.get("approved") is not False or not isinstance(
        decision.get("overall"), Mapping
    ):
        raise Stage3ContractError("invalid frozen Stage2 decision")
    expected_decision = {
        "stage1_checkpoint_sha256": sha256_file(executor),
        "interaction_train_manifest_sha256": verified_bindings[
            "interaction_train_manifest"
        ]["sha256"],
        "interaction_val_manifest_sha256": verified_bindings[
            "interaction_val_manifest"
        ]["sha256"],
        "relation_train_sha256": verified_bindings["relation_train"]["sha256"],
        "relation_val_sha256": verified_bindings["relation_val"]["sha256"],
        "pair_prior_sha256": verified_bindings["pair_prior"]["sha256"],
        "global_priority_sha256": verified_bindings["global_priority"]["sha256"],
        "config_sha256": verified_bindings["config_stage2"]["sha256"],
    }
    mismatches = {
        key: {"expected": expected, "actual": decision.get(key)}
        for key, expected in expected_decision.items()
        if decision.get(key) != expected
    }
    if mismatches:
        raise Stage3ContractError(f"Stage2 decision hash mismatch: {mismatches}")

    approval_sha = sha256_file(approval_path)
    if require_orchestrator_running:
        state_path = root / "artifacts/orchestration/state.json"
        if not state_path.is_file():
            raise Stage3ContractError(
                "Stage3 must be launched by the approved orchestrator"
            )
        state = _mapping(load_json(state_path), field="orchestration state")
        status = state.get("status")
        failed_stage3_resume = (
            allow_failed_resume
            and status == "FAILED"
            and isinstance(state.get("last_command"), list)
            and "scripts/train_stage3_planner.py" in state["last_command"]
        )
        if (
            state.get("schema_version") != ORCHESTRATION_SCHEMA
            or state.get("protocol_id") != PROTOCOL_ID
            or (
                status not in {"STAGE3_APPROVED", "STAGE3_RUNNING"}
                and not failed_stage3_resume
            )
            or state.get("stage3_approval_sha256") != approval_sha
        ):
            raise Stage3ContractError(
                "orchestrator state does not prove an approved Stage3 launch/resume"
            )

    formal_output = _project_path(
        root, path_config.get("output_dir"), field="output_dir"
    )
    selected_output = (
        _project_path(root, output_dir, field="output_dir override")
        if output_dir is not None
        else formal_output
    )
    thresholds = (
        selected_output / "planner_thresholds.json"
        if output_dir is not None
        else _project_path(root, path_config.get("thresholds"), field="thresholds")
    )
    report = (
        selected_output / "STAGE3_PLANNER_GUARD.md"
        if output_dir is not None
        else _project_path(root, path_config.get("report"), field="report")
    )
    history = (
        selected_output / "calibration_history.csv"
        if output_dir is not None
        else _project_path(
            root, path_config.get("calibration_history"), field="calibration_history"
        )
    )
    return Stage3Paths(
        project_root=root,
        config_path=config_file,
        config=config,
        resolved_path=resolved_path,
        resolved=resolved,
        training_data_root=_project_path(
            root, resolved.get("training_data_root"), field="training_data_root"
        ),
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        executor_checkpoint=executor,
        effect_profiles=effect_profiles,
        relation_train=relation_train,
        relation_val=relation_val,
        pair_prior=pair_prior,
        global_priority=global_priority,
        stage2_decision=stage2_decision,
        output_dir=selected_output,
        thresholds=thresholds,
        calibration_history=history,
        report=report,
        approval=Stage3ApprovalEvidence(
            approval_path=approval_path,
            approval_sha256=approval_sha,
            approval_required_path=approval_required,
            approval_required_sha256=str(required_sha),
            stage2_decision_sha256=str(stage2_binding["sha256"]),
            bindings=verified_bindings,
        ),
    )


def validate_stage3_extension_authorization(
    authorization_path: str | Path,
    paths: Stage3Paths,
) -> Stage3ExtensionEvidence:
    """Validate the one-off user-authorized 12k -> 18k Stage3 extension.

    The original Stage3 approval and its 22 bindings remain immutable.  The
    extension artifact is a separate, canonical authorization created by the
    controlled provenance migration.  It binds read-only copies of the exact
    pre-extension run contract and checkpoints, avoiding a hash cycle with the
    live files whose provenance points back to this authorization.
    """

    raw_path = Path(authorization_path)
    canonical = (
        paths.project_root / "artifacts/approvals" / STAGE3_EXTENSION_FILENAME
    ).resolve(strict=False)
    if not raw_path.is_absolute():
        raise Stage3ContractError(
            "Stage3 extension authorization must use an absolute canonical path"
        )
    _reject_stage3_extension_symlink_chain(raw_path, field="authorization artifact")
    if str(raw_path.resolve(strict=False)) != str(raw_path) or raw_path != canonical:
        raise Stage3ContractError(
            "Stage3 extension authorization must use the canonical non-symlink path"
        )
    if not canonical.is_file():
        raise Stage3ContractError("Stage3 extension authorization is missing")
    authorization_sha256 = sha256_file(canonical)
    payload = _mapping(load_json(canonical), field="STAGE3_EXTENSION_APPROVED.json")
    if sha256_file(canonical) != authorization_sha256:
        raise Stage3ContractError(
            "Stage3 extension authorization changed while loading"
        )
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_id",
        "approved",
        "cycles",
        "base_step",
        "target_step",
        "validation_every_steps",
        "validation_steps",
        "schedule_horizon_steps",
        "min_lr",
        "lr_policy",
        "authorized_pipeline",
        "formal_mio100_authorized",
        "base_stage3_approval",
        "base_approval_required",
        "base_stage3_config",
        "pre_extension_run_contract",
        "pre_extension_last_checkpoint",
        "pre_extension_best_checkpoint",
    }
    if set(payload) != expected_keys:
        raise Stage3ContractError("Stage3 extension authorization fields drifted")
    expected_values: dict[str, object] = {
        "schema_version": STAGE3_EXTENSION_SCHEMA,
        "kind": "stage3_extension_approval",
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "cycles": 3,
        "base_step": STAGE3_BASE_TARGET_STEP,
        "target_step": STAGE3_EXTENSION_TARGET_STEP,
        "validation_every_steps": 2_000,
        "validation_steps": list(STAGE3_EXTENSION_VALIDATION_STEPS),
        "schedule_horizon_steps": STAGE3_BASE_TARGET_STEP,
        "min_lr": 2.0e-6,
        "lr_policy": STAGE3_EXTENSION_LR_POLICY,
        "authorized_pipeline": ["stage3_extension", "stage4"],
        "formal_mio100_authorized": False,
    }
    mismatches: dict[str, object] = {}
    for key, expected in expected_values.items():
        actual = payload.get(key)
        if isinstance(expected, bool):
            matches = isinstance(actual, bool) and actual is expected
        elif isinstance(expected, int):
            matches = (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and actual == expected
            )
        elif isinstance(expected, float):
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and float(actual) == expected
            )
        else:
            matches = actual == expected
        if not matches:
            mismatches[key] = {"expected": expected, "actual": actual}
    expected_bindings = {
        "base_stage3_approval": {
            "path": str(paths.approval.approval_path),
            "sha256": paths.approval.approval_sha256,
        },
        "base_approval_required": {
            "path": str(paths.approval.approval_required_path),
            "sha256": paths.approval.approval_required_sha256,
        },
        "base_stage3_config": {
            "path": str(paths.config_path),
            "sha256": sha256_file(paths.config_path),
        },
    }
    for field, expected in expected_bindings.items():
        try:
            _, recorded_sha = _stage3_extension_file_binding(
                payload.get(field),
                field=field,
                expected_path=Path(expected["path"]),
            )
            matches = recorded_sha == expected["sha256"]
        except Stage3ContractError:
            matches = False
        if not matches:
            mismatches[field] = {
                "expected": expected,
                "actual": payload.get(field),
            }

    expected_backup_names = {
        "pre_extension_run_contract": "run_contract.json",
        "pre_extension_last_checkpoint": "last.pth",
        "pre_extension_best_checkpoint": "best_ema.pth",
    }
    backup_root = (
        paths.project_root / "artifacts/migrations" / STAGE3_EXTENSION_MIGRATION_NAME
    ).resolve(strict=False)
    backup_identities: set[tuple[int, int]] = set()
    for field, filename in expected_backup_names.items():
        expected_path = (backup_root / filename).resolve(strict=False)
        try:
            checked_path, _ = _stage3_extension_file_binding(
                payload.get(field),
                field=field,
                expected_path=expected_path,
                require_read_only=True,
            )
        except Stage3ContractError:
            mismatches[field] = {
                "expected": {
                    "path": str(expected_path),
                    "sha256": "physical immutable backup SHA256",
                    "mode": "0444",
                },
                "actual": payload.get(field),
            }
            continue
        identity = (checked_path.stat().st_dev, checked_path.stat().st_ino)
        if identity in backup_identities:
            mismatches[field] = {
                "expected": "non-aliasing immutable backup",
                "actual": payload.get(field),
            }
        backup_identities.add(identity)
    if mismatches:
        raise Stage3ContractError(
            f"Stage3 extension authorization mismatch: {mismatches}"
        )
    return Stage3ExtensionEvidence(
        authorization_path=canonical,
        authorization_sha256=authorization_sha256,
        base_step=STAGE3_BASE_TARGET_STEP,
        target_step=STAGE3_EXTENSION_TARGET_STEP,
        cycles=3,
        validation_every_steps=2_000,
        validation_steps=STAGE3_EXTENSION_VALIDATION_STEPS,
        schedule_horizon_steps=STAGE3_BASE_TARGET_STEP,
        min_lr=2.0e-6,
        lr_policy=STAGE3_EXTENSION_LR_POLICY,
    )


def _reject_stage3_extension_symlink_chain(path: Path, *, field: str) -> None:
    """Reject symlinks before resolving an extension-bound path."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise Stage3ContractError(
                f"Stage3 extension {field} path contains a symlink: {current}"
            )


def _stage3_extension_file_binding(
    value: object,
    *,
    field: str,
    expected_path: Path,
    require_read_only: bool = False,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise Stage3ContractError(
            f"Stage3 extension {field} must contain only path/sha256"
        )
    raw = value.get("path")
    digest = value.get("sha256")
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise Stage3ContractError(
            f"Stage3 extension {field}.path must be absolute and canonical"
        )
    if not is_sha256(digest):
        raise Stage3ContractError(
            f"Stage3 extension {field}.sha256 is not a lowercase SHA256"
        )
    raw_path = Path(raw)
    _reject_stage3_extension_symlink_chain(raw_path, field=field)
    canonical = raw_path.resolve(strict=False)
    if str(canonical) != raw or canonical != expected_path.resolve(strict=False):
        raise Stage3ContractError(f"Stage3 extension {field}.path drifted")
    if not canonical.is_file():
        raise Stage3ContractError(f"Stage3 extension {field} file is missing")
    if require_read_only and stat.S_IMODE(canonical.stat().st_mode) != 0o444:
        raise Stage3ContractError(
            f"Stage3 extension {field} immutable backup mode must be 0444"
        )
    if sha256_file(canonical) != digest:
        raise Stage3ContractError(f"Stage3 extension {field} hash drifted")
    return canonical, digest


def configure_stage3_reproducibility(seed: int = 2027) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _strict_tensor_mapping(value: object, *, field: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise Stage3ContractError(f"{field} must be a non-empty tensor mapping")
    if any(
        not isinstance(key, str) or not torch.is_tensor(tensor)
        for key, tensor in value.items()
    ):
        raise Stage3ContractError(f"{field} contains non-tensor values")
    return value  # type: ignore[return-value]


def _load_compiler_evidence(
    paths: Stage3Paths,
) -> tuple[dict[str, Any], dict[str, float], Tensor]:
    prior_document = _mapping(load_json(paths.pair_prior), field="pair_prior.json")
    compiler_prior = _mapping(
        prior_document.get("pair_prior"), field="pair_prior.pair_prior"
    )
    if int(prior_document.get("ambiguous_excluded", -1)) < 0:
        raise Stage3ContractError("pair prior lacks ambiguous exclusion count")
    pairs_audit = _mapping(prior_document.get("pairs"), field="pair_prior.pairs")
    if any(
        not isinstance(value, Mapping) or value.get("ambiguous_in_prior") != 0
        for value in pairs_audit.values()
    ):
        raise Stage3ContractError("ambiguous relation evidence leaked into pair_prior")
    normalized_prior: dict[str, Any] = {}
    for pair_id, probabilities in compiler_prior.items():
        if not isinstance(pair_id, str) or not isinstance(probabilities, Mapping):
            raise Stage3ContractError("invalid compiler pair prior")
        if set(probabilities) != set(RELATION_CLASSES):
            raise Stage3ContractError(f"pair prior class order drifted for {pair_id}")
        values = {name: float(probabilities[name]) for name in RELATION_CLASSES}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise Stage3ContractError(f"invalid pair prior probabilities for {pair_id}")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
            raise Stage3ContractError(
                f"pair prior probabilities do not sum to one: {pair_id}"
            )
        normalized_prior[pair_id] = values

    priority_document = _mapping(
        load_json(paths.global_priority), field="global_priority.json"
    )
    if int(priority_document.get("n_ambiguous_excluded", -1)) < 0:
        raise Stage3ContractError("global priority lacks ambiguous exclusion count")
    priority = _mapping(
        priority_document.get("priority"), field="global_priority.priority"
    )
    if set(priority) != set(SKILLS):
        raise Stage3ContractError("global priority must contain exactly eight skills")
    normalized_priority = {skill: float(priority[skill]) for skill in SKILLS}
    if any(not math.isfinite(value) for value in normalized_priority.values()):
        raise Stage3ContractError("global priority contains non-finite scores")

    profiles_document = _mapping(
        load_json(paths.effect_profiles), field="skill_effect_profiles.json"
    )
    vectors = _mapping(profiles_document.get("effect_vectors"), field="effect_vectors")
    if (
        set(vectors) != set(SKILLS)
        or int(profiles_document.get("effect_vector_dim", -1)) != 40
    ):
        raise Stage3ContractError("Stage2 effect profiles must be exactly 8x40")
    profile_tensor = torch.tensor(
        [[float(value) for value in vectors[skill]] for skill in SKILLS],
        dtype=torch.float32,
    )
    if tuple(profile_tensor.shape) != (8, 40) or not bool(
        torch.isfinite(profile_tensor).all()
    ):
        raise Stage3ContractError("invalid Stage2 effect profile tensor")
    return normalized_prior, normalized_priority, profile_tensor


def set_stage3_trainability(model: nn.Module) -> dict[str, int]:
    """Freeze the complete executor/backbone/skills and train only planner."""

    core = unwrap_model(model)
    if not isinstance(core, GraphRestore):
        raise TypeError("Stage3 requires GraphRestore")
    core.requires_grad_(False)
    core.planner.requires_grad_(True)
    counts = {"planner": 0, "frozen_executor": 0}
    for name, parameter in core.named_parameters():
        if name.startswith("planner."):
            if not parameter.requires_grad:
                raise Stage3ContractError(f"planner parameter remained frozen: {name}")
            counts["planner"] += parameter.numel()
        else:
            if parameter.requires_grad:
                raise Stage3ContractError(
                    f"executor parameter remained trainable: {name}"
                )
            counts["frozen_executor"] += parameter.numel()
    if not counts["planner"] or not counts["frozen_executor"]:
        raise Stage3ContractError("invalid Stage3 trainable/frozen partition")
    return counts


def load_stage1_ema_into_graphrestore(
    model: GraphRestore,
    checkpoint: str | Path,
) -> Stage3ParentLoadReport:
    """Load Stage1 EMA strictly, leaving only planner/threshold state initialized."""

    path = Path(checkpoint).resolve()
    if path.name != "best_ema.pth" or not path.is_file():
        raise Stage3ContractError(f"missing Stage1 best EMA checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "graphrestore-checkpoint-v1"
    ):
        raise Stage3ContractError("Stage1 checkpoint schema mismatch")
    if str(payload.get("stage", "")).lower().replace("-", "_") != "stage1":
        raise Stage3ContractError("Stage3 parent checkpoint is not Stage1")
    if (
        payload.get("model_role") != "ema_selection"
        or payload.get("resumable") is not False
    ):
        raise Stage3ContractError(
            "Stage3 parent must be a non-resumable Stage1 EMA selection checkpoint"
        )
    source = _strict_tensor_mapping(payload.get("model"), field="checkpoint.model")
    ema = _mapping(payload.get("ema"), field="checkpoint.ema")
    if ema.get("scope") != STAGE1_EMA_SCOPE:
        raise Stage3ContractError(
            "Stage1 best EMA phase-aware scope is missing or invalid"
        )
    decay = ema.get("decay")
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or float(decay) != 0.9999
    ):
        raise Stage3ContractError("Stage1 best EMA decay is missing or invalid")
    expected_policy = stage1_ema_policy_metadata(0.9999)
    if ema.get("policy") != expected_policy:
        raise Stage3ContractError(
            "Stage1 best EMA phase-aware policy is missing or invalid"
        )
    provenance = _mapping(payload.get("provenance"), field="checkpoint.provenance")
    if provenance.get("ema_policy") != expected_policy:
        raise Stage3ContractError(
            "Stage1 best provenance EMA policy is missing or invalid"
        )
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Stage3ContractError("Stage1 parent checkpoint step is invalid")
    num_updates = ema.get("num_updates")
    if (
        isinstance(num_updates, bool)
        or not isinstance(num_updates, int)
        or num_updates != step
    ):
        raise Stage3ContractError(
            "Stage1 best EMA update count does not match checkpoint step"
        )
    shadow = _strict_tensor_mapping(ema.get("shadow"), field="checkpoint.ema.shadow")
    if source.keys() != shadow.keys():
        raise Stage3ContractError("Stage1 best model/EMA keys differ")
    for name in source:
        if (
            source[name].shape != shadow[name].shape
            or source[name].dtype != shadow[name].dtype
        ):
            raise Stage3ContractError(
                f"Stage1 best model/EMA metadata differs at {name}"
            )
        if not torch.equal(source[name], shadow[name]):
            raise Stage3ContractError(
                f"Stage1 best checkpoint does not expose EMA at {name}"
            )
    target = model.state_dict()
    unexpected = sorted(set(source) - set(target))
    shape_mismatch = sorted(
        name
        for name in set(source) & set(target)
        if tuple(source[name].shape) != tuple(target[name].shape)
    )
    dtype_mismatch = sorted(
        name
        for name in set(source) & set(target)
        if source[name].dtype != target[name].dtype
    )
    missing = sorted(set(target) - set(source))
    invalid_missing = [
        name
        for name in missing
        if not (name.startswith("planner.") or name == "presence_thresholds")
    ]
    if unexpected or shape_mismatch or dtype_mismatch or invalid_missing:
        raise Stage3ContractError(
            "Stage1->Stage3 strict load failed: "
            f"unexpected={unexpected[:8]}, shape={shape_mismatch[:8]}, "
            f"dtype={dtype_mismatch[:8]}, "
            f"invalid_missing={invalid_missing[:8]}"
        )
    incompatible = model.load_state_dict(source, strict=False)
    if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise Stage3ContractError("PyTorch Stage1->Stage3 load audit differed")
    set_stage3_trainability(model)
    return Stage3ParentLoadReport(
        checkpoint_sha256=sha256_file(path),
        checkpoint_step=step,
        loaded_count=len(source),
        initialized_planner_keys=tuple(missing),
    )


def build_stage3_model(
    paths: Stage3Paths,
    *,
    device: torch.device,
    model_factory: Callable[..., GraphRestore] = GraphRestore,
) -> tuple[GraphRestore, Stage3ParentLoadReport]:
    pair_prior, global_priority, profiles = _load_compiler_evidence(paths)
    model = model_factory(pair_prior=pair_prior, global_priority=global_priority)
    report = load_stage1_ema_into_graphrestore(model, paths.executor_checkpoint)
    model.planner.set_effect_profiles(profiles)
    set_stage3_trainability(model)
    model.to(device)
    return model, report


def build_stage3_optimizer(
    model: nn.Module,
    *,
    lr: float = 2.0e-4,
    weight_decay: float = 1.0e-4,
    fused_if_supported: bool = True,
) -> torch.optim.AdamW:
    if lr != 2.0e-4 or weight_decay != 1.0e-4:
        raise Stage3ContractError("Stage3 optimizer hyperparameters drifted")
    set_stage3_trainability(model)
    parameters = [
        parameter
        for parameter in unwrap_model(model).planner.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise Stage3ContractError("Stage3 planner optimizer is empty")
    kwargs: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.999),
    }
    if fused_if_supported and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def load_relation_records(
    path: str | Path,
    *,
    split: str,
    parent_checkpoint_sha256: str,
    interaction_manifest_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    """Load Stage2 labels with normative ascending-PAIR_INDICES orientation."""

    if split not in {"train", "val"}:
        raise ValueError("relation split must be train or val")
    records: dict[str, Mapping[str, Any]] = {}
    for line, row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in records:
            raise Stage3ContractError(f"{path}:{line}: invalid/duplicate sample_id")
        if row.get("split") != split:
            raise Stage3ContractError(f"{sample_id}: relation split mismatch")
        if row.get("stage1_checkpoint_sha256") != parent_checkpoint_sha256:
            raise Stage3ContractError(f"{sample_id}: Stage1 relation binding mismatch")
        if row.get("interaction_manifest_sha256") != interaction_manifest_sha256:
            raise Stage3ContractError(
                f"{sample_id}: interaction manifest binding mismatch"
            )
        if (
            row.get("pair_orientation")
            != "ProgramPlanner.PAIR_INDICES_ascending_normative_skill_id"
        ):
            raise Stage3ContractError(f"{sample_id}: non-normative pair orientation")
        skill_ids = row.get("skill_ids")
        if (
            not isinstance(skill_ids, list)
            or len(skill_ids) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in skill_ids
            )
            or skill_ids != sorted(skill_ids)
            or tuple(skill_ids) not in PAIR_TO_ROW
        ):
            raise Stage3ContractError(f"{sample_id}: invalid ascending pair IDs")
        label = row.get("label")
        class_index = row.get("relation_class_index")
        weight = row.get("relation_weight")
        if label == "ambiguous":
            if class_index is not None or weight != 0.25:
                raise Stage3ContractError(f"{sample_id}: invalid ambiguous supervision")
        elif label in RELATION_CLASSES:
            if class_index != RELATION_CLASSES.index(str(label)) or weight != 1.0:
                raise Stage3ContractError(
                    f"{sample_id}: invalid one-hot relation supervision"
                )
        else:
            raise Stage3ContractError(f"{sample_id}: unknown relation label")
        clean_id = row.get("clean_id")
        if not isinstance(clean_id, str) or not clean_id:
            raise Stage3ContractError(f"{sample_id}: missing clean_id")
        records[sample_id] = row
    if not records:
        raise Stage3ContractError(f"empty Stage3 relation {split} labels")
    return records


def assert_relation_clean_disjoint(
    train: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Mapping[str, Any]],
) -> None:
    overlap = {str(row["clean_id"]) for row in train.values()} & {
        str(row["clean_id"]) for row in validation.values()
    }
    if overlap:
        raise Stage3ContractError(
            f"interaction_train/interaction_val clean-ID overlap: {sorted(overlap)[:8]}"
        )


def _batch_tensor(
    raw: Mapping[str, Any],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tensor:
    value = raw.get(key)
    if not torch.is_tensor(value):
        raise Stage3ContractError(f"Stage3 batch field {key!r} must be a tensor")
    return value.to(
        device=device,
        dtype=dtype if dtype is not None else value.dtype,
        non_blocking=device.type == "cuda",
    )


def _batch_strings(value: object, batch: int, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        result = (value,)
    elif isinstance(value, Sequence):
        result = tuple(str(item) for item in value)
    else:
        raise Stage3ContractError(f"Stage3 batch field {field!r} must contain strings")
    if len(result) != batch or any(not item for item in result):
        raise Stage3ContractError(f"Stage3 batch field {field!r} has wrong length")
    return result


def _state_bucket(sample_cursor: int, sample_id: str) -> int:
    """Stable ten-way state assignment; bucket 2 caps model states at 10%."""

    if sample_cursor >= 0:
        return sample_cursor % 10
    payload = f"stage3-state:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 10


@torch.no_grad()
def _teacher_model_intermediate(
    model: GraphRestore,
    image: Tensor,
    guards: Tensor,
    skill_id: int,
) -> Tensor:
    active = torch.zeros(1, len(SKILLS), device=image.device, dtype=torch.bool)
    active[0, skill_id] = True
    batched = image.unsqueeze(0)
    features = model.encode(batched)
    result = model.execute_level(
        batched,
        features,
        active_mask=active,
        guards=guards.unsqueeze(0),
    )
    return result.next_image[0].detach()


def prepare_stage3_supervision_batch(
    raw: Mapping[str, Any],
    *,
    relation_lookup: Mapping[str, Mapping[str, Any]],
    model: GraphRestore,
    device: torch.device,
) -> Stage3SupervisionBatch:
    """Build clean/single/pair/ideal/model-intermediate planner states.

    The state policy is stateless in the sampler cursor.  Exactly one of every
    ten global cursors is eligible for a model-generated state, and only Group
    A pairs use that eligibility, so the aggregate fraction can never exceed
    ten percent.  All labels continue to come from the frozen recipe.
    """

    image = _batch_tensor(raw, "input", device=device, dtype=torch.float32)
    clean = _batch_tensor(raw, "gt_clean", device=device, dtype=torch.float32)
    only_i = _batch_tensor(raw, "only_i", device=device, dtype=torch.float32)
    only_j = _batch_tensor(raw, "only_j", device=device, dtype=torch.float32)
    guards = _batch_tensor(raw, "guard_targets", device=device, dtype=torch.float32)
    severities = _batch_tensor(
        raw, "global_severity_targets", device=device, dtype=torch.float32
    )
    present = _batch_tensor(raw, "presence_target", device=device, dtype=torch.float32)
    present_ids = _batch_tensor(raw, "present_skill_ids", device=device).long()
    cursors = (
        _batch_tensor(raw, "sample_cursor", device=torch.device("cpu"))
        .long()
        .reshape(-1)
    )
    if image.ndim != 4 or image.shape[1] != 3:
        raise Stage3ContractError("Stage3 images must be RGB BCHW")
    batch = image.shape[0]
    sample_ids = _batch_strings(raw.get("sample_id"), batch, field="sample_id")
    if (
        tuple(clean.shape) != tuple(image.shape)
        or tuple(only_i.shape) != tuple(image.shape)
        or tuple(only_j.shape) != tuple(image.shape)
    ):
        raise Stage3ContractError("Stage3 subset images must match input shape")
    if tuple(guards.shape[:2]) != (batch, len(SKILLS)):
        raise Stage3ContractError("Stage3 guard targets must be Bx8xH/4xW/4")
    if tuple(present.shape) != (batch, len(SKILLS)) or tuple(present_ids.shape) != (
        batch,
        2,
    ):
        raise Stage3ContractError("Stage3 presence/present_skill_ids shape mismatch")

    x0 = image.clone()
    current = image.clone()
    remaining = present.clone()
    state_kinds: list[str] = []
    round_values = torch.zeros(batch, device=device, dtype=torch.float32)
    relation_enabled = torch.zeros(batch, device=device, dtype=torch.bool)
    model_intermediate_count = 0

    # Executor is frozen even while the planner remains in train mode.
    model.encoder.eval()
    model.decoder.eval()
    for index, sample_id in enumerate(sample_ids):
        ids = [int(value) for value in present_ids[index].tolist() if int(value) >= 0]
        if len(ids) not in {1, 2} or len(ids) != int(present[index].sum().item()):
            raise Stage3ContractError(f"{sample_id}: recipe presence target drifted")
        bucket = _state_bucket(int(cursors[index].item()), sample_id)
        if bucket == 0:
            # A genuine clean state has no artificial x0->xt trace.
            x0[index] = clean[index]
            current[index] = clean[index]
            remaining[index].zero_()
            state_kinds.append("clean")
        elif bucket == 1 and len(ids) == 2:
            # only_i contains degradation i, and only_j contains degradation j.
            keep_slot = (int(cursors[index].item()) // 10) % 2
            keep_id = ids[keep_slot]
            current[index] = only_i[index] if keep_slot == 0 else only_j[index]
            remaining[index].zero_()
            remaining[index, keep_id] = 1.0
            round_values[index] = 0.5
            state_kinds.append("ideal_subset_intermediate")
        elif bucket == 2 and len(ids) == 2:
            execute_slot = (int(cursors[index].item()) // 10) % 2
            execute_id = ids[execute_slot]
            keep_id = ids[1 - execute_slot]
            current[index] = _teacher_model_intermediate(
                model, image[index], guards[index], execute_id
            )
            remaining[index].zero_()
            remaining[index, keep_id] = 1.0
            round_values[index] = 0.5
            model_intermediate_count += 1
            state_kinds.append("model_generated_intermediate")
        elif len(ids) == 1:
            state_kinds.append("single_degradation")
        else:
            state_kinds.append("group_a_pair")
            relation_enabled[index] = True

    # Remaining-only maps and severities; absent maps are exact zeros.
    guard_targets = guards * remaining[:, :, None, None]
    severity_targets = severities * remaining
    dense_ids = torch.tensor(
        [skill in DENSE_GUARD_SKILLS for skill in SKILLS],
        device=device,
        dtype=torch.bool,
    )
    present_mask = remaining.bool()
    dense_mask = present_mask & dense_ids[None, :]
    global_mask = present_mask & ~dense_ids[None, :]
    absent_mask = ~present_mask
    stop_targets = (~present_mask.any(dim=1)).float().unsqueeze(1)

    relation_targets = torch.full(
        (batch, len(PAIR_INDICES)), -2, device=device, dtype=torch.long
    )
    relation_weights = torch.zeros(
        batch, len(PAIR_INDICES), device=device, dtype=torch.float32
    )
    relation_ambiguous = torch.zeros(
        batch, len(PAIR_INDICES), device=device, dtype=torch.bool
    )
    for index, sample_id in enumerate(sample_ids):
        if not bool(relation_enabled[index]):
            continue
        row = relation_lookup.get(sample_id)
        if row is None:
            # Stage2 intentionally labels at most 512 of 900 train rows/pair.
            continue
        pair = tuple(int(value) for value in row["skill_ids"])
        pair_row = PAIR_TO_ROW[pair]
        if row["label"] == "ambiguous":
            relation_targets[index, pair_row] = -1
            relation_weights[index, pair_row] = 0.25
            relation_ambiguous[index, pair_row] = True
        else:
            relation_targets[index, pair_row] = int(row["relation_class_index"])
            relation_weights[index, pair_row] = 1.0

    values = (
        x0,
        current,
        remaining,
        guard_targets,
        severity_targets,
        stop_targets,
        relation_weights,
        round_values,
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in values):
        raise FloatingPointError("non-finite Stage3 supervision batch")
    return Stage3SupervisionBatch(
        x0=x0,
        current=current,
        presence_targets=remaining,
        guard_targets=guard_targets,
        global_severity_targets=severity_targets,
        dense_skill_mask=dense_mask,
        global_skill_mask=global_mask,
        absent_skill_mask=absent_mask,
        stop_targets=stop_targets,
        relation_targets=relation_targets,
        relation_weights=relation_weights,
        relation_ambiguous_mask=relation_ambiguous,
        round_values=round_values,
        sample_ids=sample_ids,
        state_kinds=tuple(state_kinds),
        model_intermediate_count=model_intermediate_count,
    )


def stage3_planner_forward(
    model: GraphRestore,
    batch: Stage3SupervisionBatch,
) -> PlannerOutput:
    """Encode with the frozen executor and retain gradients only in planner."""

    with torch.no_grad():
        features = tuple(feature.detach() for feature in model.encode(batch.current))
    return model.plan_state(
        batch.x0,
        batch.current,
        features,
        round_value=batch.round_values,
        compute_relations=True,
    )


def stage3_supervision_loss(
    output: PlannerOutput,
    batch: Stage3SupervisionBatch,
) -> tuple[PlannerLossBreakdown, Any]:
    guards = guard_supervision_loss(
        output.guard_logits,
        batch.guard_targets,
        batch.global_severity_targets,
        dense_skill_mask=batch.dense_skill_mask,
        global_skill_mask=batch.global_skill_mask,
        absent_skill_mask=batch.absent_skill_mask,
    )
    # The final ambiguity ruling requires FP32 log_softmax+logsumexp.  Casting
    # here keeps the actual BF16 training connection numerically compliant.
    loss = planner_loss(
        presence_logits=output.presence_logits,
        presence_targets=batch.presence_targets,
        relation_logits=output.relation_logits.float(),
        relation_targets=batch.relation_targets,
        relation_weights=batch.relation_weights,
        relation_ambiguous_mask=batch.relation_ambiguous_mask,
        stop_logits=output.stop_logit,
        stop_targets=batch.stop_targets,
        guard=guards,
    )
    return loss, guards


def _stage3_train_mode(model: GraphRestore) -> None:
    model.eval()
    model.planner.train()
    if model.encoder.training or model.decoder.training or not model.planner.training:
        raise Stage3ContractError("Stage3 executor/planner mode partition failed")


def _autocast(device: torch.device, enabled: bool):
    if enabled:
        if device.type != "cuda":
            raise Stage3ContractError("formal Stage3 BF16 requires CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def assert_only_planner_gradients(model: GraphRestore) -> None:
    planner_has_gradient = False
    for name, parameter in model.named_parameters():
        if name.startswith("planner."):
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise FloatingPointError(f"non-finite Stage3 gradient: {name}")
                planner_has_gradient |= bool(torch.count_nonzero(parameter.grad).item())
        elif parameter.grad is not None and bool(
            torch.count_nonzero(parameter.grad).item()
        ):
            raise Stage3ContractError(f"frozen executor received gradient: {name}")
    if not planner_has_gradient:
        raise Stage3ContractError(
            "Stage3 backward produced no nonzero planner gradient"
        )


def train_stage3_optimizer_step(
    model: GraphRestore,
    micro_batches: Sequence[Stage3SupervisionBatch],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler | None,
    ema: ExponentialMovingAverage | None,
    *,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
    use_bf16: bool = True,
    audit_gradients: bool = False,
    optimizer_transaction: Stage3OptimizerTransaction | None = None,
) -> Stage3StepResult:
    if not micro_batches:
        raise ValueError("Stage3 requires at least one micro batch")
    if gradient_clip_norm != 1.0:
        raise Stage3ContractError("Stage3 gradient clipping must be 1.0")
    transaction = optimizer_transaction or Stage3OptimizerTransaction()
    transaction.begin()
    caller_owns_transaction = optimizer_transaction is not None
    set_stage3_trainability(model)
    _stage3_train_mode(model)
    optimizer.zero_grad(set_to_none=True)
    totals: defaultdict[str, float] = defaultdict(float)
    samples = model_intermediate_count = 0
    state_counts: defaultdict[str, int] = defaultdict(int)
    started = time.perf_counter()
    for batch in micro_batches:
        batch_size = int(batch.current.shape[0])
        with _autocast(device, use_bf16):
            output = stage3_planner_forward(model, batch)
            loss, guards = stage3_supervision_loss(output, batch)
        if not bool(torch.isfinite(loss.total).item()):
            raise FloatingPointError("non-finite Stage3 planner loss")
        (loss.total / len(micro_batches)).backward()
        samples += batch_size
        model_intermediate_count += batch.model_intermediate_count
        for state in batch.state_kinds:
            state_counts[state] += 1
        for name, value in (
            ("total", loss.total),
            ("presence", loss.presence),
            ("guard", loss.guard),
            ("relation", loss.relation),
            ("stop", loss.stop),
            ("cycle", loss.cycle),
            ("guard_dense", guards.dense),
            ("guard_global_mean", guards.global_mean),
            ("guard_absent", guards.absent),
        ):
            totals[name] += float(value.detach()) * batch_size
    if audit_gradients:
        assert_only_planner_gradients(model)
    parameters = [
        parameter
        for parameter in model.planner.parameters()
        if parameter.grad is not None
    ]
    if not parameters:
        raise Stage3ContractError("Stage3 planner has no gradients")
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        parameters, gradient_clip_norm, error_if_nonfinite=True
    )
    grad_norm = float(grad_norm_tensor.detach())
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if model_intermediate_count / samples > 0.10 + 1.0 / samples:
        raise Stage3ContractError(
            "model-generated intermediate fraction exceeded schedule"
        )
    result = Stage3StepResult(
        total=totals["total"] / samples,
        presence=totals["presence"] / samples,
        guard=totals["guard"] / samples,
        relation=totals["relation"] / samples,
        stop=totals["stop"] / samples,
        cycle=totals["cycle"] / samples,
        guard_dense=totals["guard_dense"] / samples,
        guard_global_mean=totals["guard_global_mean"] / samples,
        guard_absent=totals["guard_absent"] / samples,
        grad_norm=grad_norm,
        samples=samples,
        model_intermediate_count=model_intermediate_count,
        state_counts=dict(sorted(state_counts.items())),
        seconds=elapsed,
    )
    # A formal caller keeps this active until its logical step and sampler
    # cursor are committed.  Standalone probes/tests own no external state and
    # can close the transaction here.
    if not caller_owns_transaction:
        transaction.commit()
    return result


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def presence_diagnostics(
    probabilities: Tensor,
    targets: Tensor,
    thresholds: Tensor | Sequence[float] | float = 0.5,
) -> dict[str, Any]:
    probabilities = probabilities.detach().float().cpu()
    targets = targets.detach().bool().cpu()
    if (
        tuple(probabilities.shape) != tuple(targets.shape)
        or probabilities.ndim != 2
        or probabilities.shape[1] != len(SKILLS)
    ):
        raise ValueError("presence probabilities/targets must be matching Nx8 tensors")
    threshold = torch.as_tensor(thresholds, dtype=torch.float32)
    if threshold.ndim == 0:
        threshold = threshold.repeat(len(SKILLS))
    if tuple(threshold.shape) != (len(SKILLS),):
        raise ValueError("presence thresholds must be scalar or length eight")
    predicted = probabilities >= threshold[None, :]
    per_skill: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, skill in enumerate(SKILLS):
        prediction = predicted[:, index]
        truth = targets[:, index]
        tp = int((prediction & truth).sum())
        fp = int((prediction & ~truth).sum())
        fn = int((~prediction & truth).sum())
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
        f1_values.append(f1)
        per_skill[skill] = {
            "threshold": float(threshold[index]),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "activation_rate": float(prediction.float().mean().item()),
        }
    return {
        "sample_count": int(probabilities.shape[0]),
        "macro_f1": math.fsum(f1_values) / len(f1_values),
        "activation_rate": float(predicted.float().mean().item()),
        "activation_rate_definition": "fraction_of_sample_skill_slots_predicted_active",
        "per_skill": per_skill,
    }


def calibrate_presence_thresholds(
    probabilities: Tensor,
    targets: Tensor,
    *,
    minimum: float = 0.20,
    maximum: float = 0.80,
    step: float = 0.02,
) -> ThresholdCalibration:
    """Maximize per-skill F1 with the adjudicated deterministic tie-break.

    Exact F1 ties choose the candidate nearest 0.50.  If two candidates are
    equally distant, the higher threshold wins.  The locked grid contains
    0.50, but the second rule remains explicit so future audits do not infer a
    different ordering from loop iteration.
    """

    if (minimum, maximum, step) != (0.20, 0.80, 0.02):
        raise Stage3ContractError("Stage3 threshold grid drifted")
    probabilities = probabilities.detach().float().cpu()
    targets = targets.detach().bool().cpu()
    if (
        tuple(probabilities.shape) != tuple(targets.shape)
        or probabilities.ndim != 2
        or probabilities.shape[1] != 8
    ):
        raise ValueError("calibration requires matching Nx8 probabilities/targets")
    grid = tuple(value / 100.0 for value in range(20, 81, 2))
    if 0.50 not in grid:
        raise Stage3ContractError("Stage3 threshold grid must contain 0.50")
    selected: list[float] = []
    scores: list[float] = []
    for skill in range(8):
        truth = targets[:, skill]
        best_threshold = 0.50
        best_f1 = -1.0
        for threshold in grid:
            prediction = probabilities[:, skill] >= threshold
            tp = int((prediction & truth).sum())
            fp = int((prediction & ~truth).sum())
            fn = int((~prediction & truth).sum())
            f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
            if f1 > best_f1 or (
                f1 == best_f1
                and (
                    abs(threshold - 0.50),
                    -threshold,
                )
                < (
                    abs(best_threshold - 0.50),
                    -best_threshold,
                )
            ):
                best_threshold, best_f1 = threshold, f1
        selected.append(best_threshold)
        scores.append(best_f1)
    baseline = presence_diagnostics(probabilities, targets, 0.50)
    calibrated = presence_diagnostics(probabilities, targets, selected)
    baseline_per_skill = _mapping(
        baseline.get("per_skill"), field="baseline presence diagnostics"
    )
    calibrated_per_skill = _mapping(
        calibrated.get("per_skill"), field="calibrated presence diagnostics"
    )
    for skill in SKILLS:
        before = float(_mapping(baseline_per_skill[skill], field=skill)["f1"])
        after = float(_mapping(calibrated_per_skill[skill], field=skill)["f1"])
        if after + THRESHOLD_F1_TOLERANCE < before:
            raise Stage3ContractError(
                f"calibrated Stage3 F1 regressed for {skill}: {after} < {before}"
            )
    if float(calibrated["macro_f1"]) + THRESHOLD_F1_TOLERANCE < float(
        baseline["macro_f1"]
    ):
        raise Stage3ContractError("calibrated Stage3 macro F1 regressed")
    return ThresholdCalibration(
        thresholds=tuple(selected),
        per_skill_f1=tuple(scores),
        grid=grid,
        baseline_diagnostics=baseline,
        calibrated_diagnostics=calibrated,
    )


def _relation_prediction_metrics(
    predictions: Sequence[int], targets: Sequence[int]
) -> dict[str, float | int]:
    if len(predictions) != len(targets) or not targets:
        raise Stage3ContractError("relation baseline predictions/targets are invalid")
    if any(value not in {0, 1, 2} for value in (*predictions, *targets)):
        raise Stage3ContractError("relation baseline class index is invalid")
    correct = sum(
        int(prediction == target)
        for prediction, target in zip(predictions, targets, strict=True)
    )
    f1_values: list[float] = []
    recalls: list[float] = []
    for class_index in range(3):
        tp = sum(
            int(prediction == class_index and target == class_index)
            for prediction, target in zip(predictions, targets, strict=True)
        )
        fp = sum(
            int(prediction == class_index and target != class_index)
            for prediction, target in zip(predictions, targets, strict=True)
        )
        fn = sum(
            int(prediction != class_index and target == class_index)
            for prediction, target in zip(predictions, targets, strict=True)
        )
        f1_values.append(_safe_ratio(2 * tp, 2 * tp + fp + fn))
        recalls.append(_safe_ratio(tp, tp + fn))
    return {
        "correct": correct,
        "accuracy": correct / len(targets),
        "macro_f1": math.fsum(f1_values) / len(f1_values),
        "balanced_accuracy": math.fsum(recalls) / len(recalls),
    }


def relation_baseline_audit(
    relation_records: Mapping[str, Mapping[str, Any]],
    pair_prior_payload: Mapping[str, Any],
    *,
    learned_raw_accuracy: float,
) -> dict[str, Any]:
    """Audit learned relation accuracy against two CPU-only baselines.

    Ambiguous rows are excluded exactly as in formal Stage3 relation metrics.
    The pair-majority baseline is learned only from the frozen Stage2 train
    prior and evaluated on interaction_val labels; it never reads model output
    or any MiO100/Group-B/Group-C artifact.
    """

    if not math.isfinite(float(learned_raw_accuracy)):
        raise Stage3ContractError("learned raw relation accuracy is non-finite")
    prior_raw = pair_prior_payload.get("pair_prior")
    prior = _mapping(prior_raw, field="pair-prior probabilities")
    if set(prior) != {
        str(row.get("pair_id"))
        for row in relation_records.values()
        if str(row.get("label")) != "ambiguous"
    }:
        raise Stage3ContractError("pair-prior/interaction_val pair set drifted")
    targets: list[int] = []
    pair_predictions: list[int] = []
    ambiguous = 0
    for sample_id, row in sorted(relation_records.items()):
        if str(row.get("label")) == "ambiguous":
            ambiguous += 1
            continue
        target = row.get("relation_class_index")
        pair_id = row.get("pair_id")
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target not in {0, 1, 2}
            or not isinstance(pair_id, str)
            or not pair_id
        ):
            raise Stage3ContractError(
                f"{sample_id}: invalid non-ambiguous relation baseline row"
            )
        probabilities = _mapping(prior.get(pair_id), field=f"pair prior {pair_id}")
        if set(probabilities) != set(RELATION_CLASSES):
            raise Stage3ContractError(f"{pair_id}: pair-prior classes drifted")
        values = [float(probabilities[name]) for name in RELATION_CLASSES]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise Stage3ContractError(f"{pair_id}: pair-prior value is invalid")
        # Stable class-order tie-break is deliberate and recorded by name.
        majority = max(range(3), key=lambda index: (values[index], -index))
        targets.append(target)
        pair_predictions.append(majority)
    if not targets:
        raise Stage3ContractError("relation baseline audit has no eligible rows")
    always_parallel = _relation_prediction_metrics([2] * len(targets), targets)
    pair_majority = _relation_prediction_metrics(pair_predictions, targets)
    return {
        "source": "interaction_val_non_ambiguous_cpu_only",
        "n_total": len(relation_records),
        "n_non_ambiguous": len(targets),
        "n_ambiguous_excluded": ambiguous,
        "learned_raw_accuracy": float(learned_raw_accuracy),
        "always_parallel": always_parallel,
        "per_pair_majority_prior": pair_majority,
        "pair_majority_tie_break": "relation_class_order",
        "mio100_rows_read": 0,
        "group_b_rows_read": 0,
        "group_c_rows_read": 0,
    }


def freeze_presence_thresholds(
    destination: str | Path,
    calibration: ThresholdCalibration,
    *,
    primary_val_manifest: str | Path,
    selected_checkpoint: str | Path,
    approval_sha256: str,
    extension_authorization_sha256: str | None = None,
    finalization_authorization_sha256: str | None = None,
    calibration_code_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = Path(primary_val_manifest).resolve()
    checkpoint = Path(selected_checkpoint).resolve()
    if (
        "mio100" in str(manifest).lower()
        or "group_b" in str(manifest).lower()
        or "group_c" in str(manifest).lower()
    ):
        raise Stage3ContractError("MiO100/Group B/C threshold calibration is forbidden")
    baseline = _mapping(
        calibration.baseline_diagnostics, field="baseline threshold diagnostics"
    )
    calibrated = _mapping(
        calibration.calibrated_diagnostics,
        field="calibrated threshold diagnostics",
    )
    baseline_per_skill = _mapping(
        baseline.get("per_skill"), field="baseline per-skill metrics"
    )
    calibrated_per_skill = _mapping(
        calibrated.get("per_skill"), field="calibrated per-skill metrics"
    )
    code_path = Path(calibration_code_path or __file__).resolve()
    if not code_path.is_file():
        raise Stage3ContractError("Stage3 calibration code disappeared")
    per_skill_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for index, skill in enumerate(SKILLS):
        before = _mapping(baseline_per_skill.get(skill), field=f"baseline {skill}")
        after = _mapping(calibrated_per_skill.get(skill), field=f"calibrated {skill}")
        before_values = {
            "threshold": 0.50,
            "precision": float(before["precision"]),
            "recall": float(before["recall"]),
            "f1": float(before["f1"]),
        }
        after_values = {
            "threshold": float(calibration.thresholds[index]),
            "precision": float(after["precision"]),
            "recall": float(after["recall"]),
            "f1": float(after["f1"]),
        }
        if after_values["f1"] + THRESHOLD_F1_TOLERANCE < before_values["f1"]:
            raise Stage3ContractError(f"frozen calibrated F1 regressed for {skill}")
        per_skill_metrics[skill] = {
            "baseline": before_values,
            "calibrated": after_values,
        }
    macro_before = float(baseline["macro_f1"])
    macro_after = float(calibrated["macro_f1"])
    if macro_after + THRESHOLD_F1_TOLERANCE < macro_before:
        raise Stage3ContractError("frozen calibrated Stage3 macro F1 regressed")
    payload = {
        "schema_version": THRESHOLD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now_iso(),
        "source": "primary_val_presence_f1_only",
        "source_primary_val": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        },
        "selected_stage3_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "checkpoint_sha256": sha256_file(checkpoint),
        "primary_val_manifest_sha256": sha256_file(manifest),
        "stage3_approval_sha256": approval_sha256,
        "skills": list(SKILLS),
        "baseline_threshold": 0.50,
        "thresholds": {
            skill: calibration.thresholds[index] for index, skill in enumerate(SKILLS)
        },
        "per_skill_f1": {
            skill: calibration.per_skill_f1[index] for index, skill in enumerate(SKILLS)
        },
        "search_grid": list(calibration.grid),
        "tie_break": calibration.tie_break,
        "numerical_tolerance": THRESHOLD_F1_TOLERANCE,
        "per_skill_metrics": per_skill_metrics,
        "macro_f1_before": macro_before,
        "macro_f1_after": macro_after,
        "calibration_code": {
            "path": str(code_path),
            "sha256": sha256_file(code_path),
        },
        "calibration_runs": 1,
        "mio100_rows_read": 0,
        "group_b_rows_read": 0,
        "group_c_rows_read": 0,
        "frozen": True,
    }
    if extension_authorization_sha256 is not None:
        if not is_sha256(extension_authorization_sha256):
            raise Stage3ContractError(
                "Stage3 extension authorization SHA256 is invalid"
            )
        payload["stage3_extension_authorization_sha256"] = (
            extension_authorization_sha256
        )
    if finalization_authorization_sha256 is not None:
        if not is_sha256(finalization_authorization_sha256):
            raise Stage3ContractError(
                "Stage3 finalization authorization SHA256 is invalid"
            )
        payload["stage3_finalization_authorization_sha256"] = (
            finalization_authorization_sha256
        )
    path = Path(destination)
    if path.exists():
        existing = _mapping(load_json(path), field="existing planner thresholds")
        comparable = dict(payload)
        comparable.pop("created_utc")
        current = dict(existing)
        current.pop("created_utc", None)
        if current != comparable:
            raise Stage3ContractError(
                "presence thresholds are already frozen and may not be recalibrated"
            )
        return dict(existing)
    atomic_write_json(path, payload)
    return payload


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        ranks[order[cursor:end]] = rank
        cursor = end
    return ranks


def _spearman(
    predicted: Tensor, target: Tensor, variance_threshold: float
) -> float | None:
    x = predicted.detach().float().cpu().reshape(-1).numpy().astype(np.float64)
    y = target.detach().float().cpu().reshape(-1).numpy().astype(np.float64)
    if float(np.var(x)) < variance_threshold or float(np.var(y)) < variance_threshold:
        return None
    x_rank, y_rank = _rankdata_average(x), _rankdata_average(y)
    correlation = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return correlation if math.isfinite(correlation) else None


def align_guard_prediction_to_target(
    predicted: Tensor,
    target: Tensor,
) -> Tensor:
    """Remove only the right/bottom H/4 cells induced by input padding.

    Full-resolution validation inputs are guaranteed to be divisible by four,
    while :class:`GraphRestore` pads them on the right and bottom to a multiple
    of eight.  Consequently a traced planner guard may exceed the unpadded
    target by exactly one H/4 cell on either spatial axis.  Guard diagnostics
    cover the original image support, so that padding-only fringe is cropped;
    every other shape mismatch remains fail-closed.
    """

    if predicted.ndim < 2 or predicted.ndim != target.ndim:
        raise ValueError("predicted/target guard map shape mismatch")
    if tuple(predicted.shape[:-2]) != tuple(target.shape[:-2]):
        raise ValueError("predicted/target guard map shape mismatch")
    spatial_delta = tuple(
        int(predicted_size) - int(target_size)
        for predicted_size, target_size in zip(
            predicted.shape[-2:], target.shape[-2:], strict=True
        )
    )
    if any(delta not in {0, 1} for delta in spatial_delta):
        raise ValueError("predicted/target guard map shape mismatch")
    return predicted[..., : target.shape[-2], : target.shape[-1]]


def guard_structure_diagnostics(
    predicted_guards: Tensor,
    target_guards: Tensor,
    presence_targets: Tensor,
    *,
    variance_threshold: float = 1.0e-8,
) -> dict[str, float | int | None]:
    predicted = predicted_guards.detach().float().cpu()
    target = target_guards.detach().float().cpu()
    presence = presence_targets.detach().bool().cpu()
    if tuple(predicted.shape) != tuple(target.shape) or predicted.ndim != 4:
        raise ValueError("guard diagnostics require matching Nx8xHxW tensors")
    result: dict[str, float | int | None] = {}
    for skill in ("rain", "haze"):
        skill_id = SKILL_TO_ID[skill]
        indices = (
            torch.nonzero(presence[:, skill_id], as_tuple=False).flatten().tolist()
        )
        spearman: list[float] = []
        mae: list[float] = []
        standard_deviation: list[float] = []
        high_fraction: list[float] = []
        skipped = 0
        for index in indices:
            prediction = predicted[index, skill_id]
            truth = target[index, skill_id]
            value = _spearman(prediction, truth, variance_threshold)
            if value is None:
                skipped += 1
            else:
                spearman.append(value)
            mae.append(float((prediction - truth).abs().mean()))
            standard_deviation.append(float(prediction.std(unbiased=False)))
            high_fraction.append(float((prediction > 0.9).float().mean()))
        result.update(
            {
                f"guard_spearman_{skill}": _mean_or_none(spearman),
                f"guard_mae_{skill}": _mean_or_none(mae),
                f"guard_std_{skill}": _mean_or_none(standard_deviation),
                f"guard_high_frac_{skill}": _mean_or_none(high_fraction),
                f"valid_guard_images_{skill}": len(spearman),
                f"skipped_guard_images_{skill}": skipped,
                f"present_guard_images_{skill}": len(indices),
            }
        )
    return result


def _guard_structure_diagnostics_variable_size(
    predicted: Sequence[Tensor],
    target: Sequence[Tensor],
    presence: Tensor,
    *,
    variance_threshold: float,
) -> dict[str, float | int | None]:
    if len(predicted) != len(target) or len(predicted) != presence.shape[0]:
        raise ValueError("variable-size guard diagnostic inputs differ in length")
    aggregate: dict[str, float | int | None] = {}
    for skill in ("rain", "haze"):
        skill_id = SKILL_TO_ID[skill]
        correlations: list[float] = []
        maes: list[float] = []
        deviations: list[float] = []
        highs: list[float] = []
        skipped = 0
        present_count = 0
        for index in range(len(predicted)):
            if not bool(presence[index, skill_id]):
                continue
            present_count += 1
            prediction = predicted[index][skill_id]
            truth = target[index][skill_id]
            if tuple(prediction.shape) != tuple(truth.shape):
                raise ValueError("predicted/target guard map shape mismatch")
            correlation = _spearman(prediction, truth, variance_threshold)
            if correlation is None:
                skipped += 1
            else:
                correlations.append(correlation)
            maes.append(float((prediction - truth).abs().mean()))
            deviations.append(float(prediction.std(unbiased=False)))
            highs.append(float((prediction > 0.9).float().mean()))
        aggregate.update(
            {
                f"guard_spearman_{skill}": _mean_or_none(correlations),
                f"guard_mae_{skill}": _mean_or_none(maes),
                f"guard_std_{skill}": _mean_or_none(deviations),
                f"guard_high_frac_{skill}": _mean_or_none(highs),
                f"valid_guard_images_{skill}": len(correlations),
                f"skipped_guard_images_{skill}": skipped,
                f"present_guard_images_{skill}": present_count,
            }
        )
    return aggregate


def _json_finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


def _equal_task_metric(rows: Sequence[Mapping[str, Any]], group: str) -> dict[str, Any]:
    selected = [row for row in rows if row["group"] == group]
    buckets: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        buckets[str(row["combination"])].append(row)
    if len(buckets) != 8:
        raise Stage3ContractError(
            f"primary_val {group} must contain exactly eight tasks"
        )
    per_task: dict[str, Any] = {}
    for task, values in sorted(buckets.items()):
        per_task[task] = {
            "count": len(values),
            "psnr": math.fsum(float(row["psnr"]) for row in values) / len(values),
            "ssim": math.fsum(float(row["ssim"]) for row in values) / len(values),
        }
    return {
        "count": len(selected),
        "task_count": len(per_task),
        "psnr": math.fsum(value["psnr"] for value in per_task.values()) / len(per_task),
        "ssim": math.fsum(value["ssim"] for value in per_task.values()) / len(per_task),
        "per_task": per_task,
    }


def _validation_autocast(device: torch.device, use_bf16: bool):
    if use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def validate_stage3(
    model: GraphRestore,
    dataset: GraphRestoreEpisodeDataset,
    relation_val: Mapping[str, Mapping[str, Any]],
    *,
    device: torch.device,
    use_bf16: bool = True,
    presence_threshold: Tensor | Sequence[float] | float = 0.5,
) -> dict[str, Any]:
    """Validate restoration on primary_val and relations on interaction_val.

    No Stage2 train relation is accepted by this API and no MiO100 path is
    opened.  Guard diagnostics are comprehensive but returned separately from
    the restoration-first checkpoint score.
    """

    if dataset.training or dataset.crop_size is not None:
        raise Stage3ContractError(
            "Stage3 validation must be full-resolution/no augmentation"
        )
    if any(record.group not in {"single", "A"} for record in dataset.records):
        raise Stage3ContractError("Stage3 validation contains forbidden data groups")
    if any(str(row.get("split")) != "val" for row in relation_val.values()):
        raise Stage3ContractError(
            "Stage3 relation validation must use interaction_val only"
        )
    model.eval()
    requested_thresholds = (
        torch.as_tensor(presence_threshold, dtype=torch.float64).detach().cpu()
    )
    scalar_threshold_input = requested_thresholds.ndim == 0
    if scalar_threshold_input:
        requested_thresholds = requested_thresholds.repeat(len(SKILLS))
    if tuple(requested_thresholds.shape) != (len(SKILLS),):
        raise Stage3ContractError(
            "Stage3 validation presence thresholds must be scalar or length eight"
        )
    if not bool(torch.isfinite(requested_thresholds).all().item()) or bool(
        torch.any((requested_thresholds < 0.0) | (requested_thresholds > 1.0)).item()
    ):
        raise Stage3ContractError("Stage3 validation presence thresholds are invalid")
    fixed_thresholds = requested_thresholds.to(device=device, dtype=torch.float32)
    metric_rows: list[dict[str, Any]] = []
    all_presence_probabilities: list[Tensor] = []
    all_presence_targets: list[Tensor] = []
    all_guard_predictions: list[Tensor] = []
    all_guard_targets: list[Tensor] = []
    relation_logits: list[Tensor] = []
    relation_targets: list[int] = []
    relation_ambiguous: list[bool] = []
    relation_pair_ids: list[str] = []
    pre_cycle_samples = dropped_edges = proposed_edges = 0
    reentry_requests = unexpected_activations = trace_slots = 0
    stopped_samples = 0
    program_levels: list[int] = []

    for index, record in enumerate(dataset.records):
        sample = dataset[index]
        image = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
        target = sample["gt_clean"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with _validation_autocast(device, use_bf16):
            traced = model(
                image,
                presence_thresholds=fixed_thresholds,
                max_rounds=3,
                return_trace=True,
            )
        if torch.is_tensor(traced) or not hasattr(traced, "planner_outputs"):
            raise RuntimeError("Stage3 validation requires a GraphRestore trace")
        prediction = traced.final.detach().float().cpu()
        metric = official_psnr_ssim(
            prediction, target.detach().float().cpu(), quantize=True
        )
        combination = "+".join(record.skill_names)
        metric_rows.append(
            {
                "sample_id": record.sample_id,
                "group": record.group,
                "combination": combination,
                "psnr": float(metric.psnr.item()),
                "ssim": float(metric.ssim.item()),
            }
        )
        if not traced.planner_outputs:
            raise Stage3ContractError(f"{record.sample_id}: missing t=0 planner output")
        plan = traced.planner_outputs[0]
        all_presence_probabilities.append(
            plan.presence_probabilities[0].detach().float().cpu()
        )
        all_presence_targets.append(sample["presence_target"].detach().float().cpu())
        guard_prediction = plan.spatial_guard_probabilities[0].detach().float().cpu()
        guard_target = sample["guard_targets"].detach().float().cpu()
        all_guard_predictions.append(
            align_guard_prediction_to_target(guard_prediction, guard_target)
        )
        all_guard_targets.append(guard_target)

        graph = traced.compiled_graphs[0]
        if not graph.cycle_free:
            raise Stage3ContractError("post-compiler graph must be cycle-free")
        pre_cycle_samples += int(bool(graph.dropped_edges))
        dropped_edges += len(graph.dropped_edges)
        proposed_edges += len(graph.edges) + len(graph.dropped_edges)
        program_levels.append(len(graph.levels))
        sample_stopped = False
        for trace in traced.trace:
            reentry_requests += int(trace.reentry_request_mask.sum().item())
            unexpected_activations += int(trace.unexpected_activation_mask.sum().item())
            trace_slots += trace.reentry_request_mask.numel()
            sample_stopped = sample_stopped or bool(trace.stopped_mask.any().item())
        stopped_samples += int(sample_stopped)

        relation = relation_val.get(record.sample_id)
        if relation is not None:
            pair = tuple(int(value) for value in relation["skill_ids"])
            relation_logits.append(
                plan.relation_logits[0, PAIR_TO_ROW[pair]].detach().float().cpu()
            )
            ambiguous = relation["label"] == "ambiguous"
            relation_ambiguous.append(ambiguous)
            relation_targets.append(
                0 if ambiguous else int(relation["relation_class_index"])
            )
            relation_pair_ids.append(str(relation["pair_id"]))

    if set(relation_val) != set(
        row["sample_id"] for row in metric_rows if row["sample_id"] in relation_val
    ):
        missing = sorted(set(relation_val) - {row["sample_id"] for row in metric_rows})
        raise Stage3ContractError(
            f"interaction_val rows absent from primary_val: {missing[:8]}"
        )
    probability_tensor = torch.stack(all_presence_probabilities)
    presence_target_tensor = torch.stack(all_presence_targets)
    relation_logit_tensor = torch.stack(relation_logits)
    relation_target_tensor = torch.tensor(relation_targets, dtype=torch.long)
    relation_ambiguous_tensor = torch.tensor(relation_ambiguous, dtype=torch.bool)
    relation_metric = non_ambiguous_relation_metrics(
        relation_logit_tensor,
        relation_target_tensor,
        relation_ambiguous_tensor,
        pair_ids=relation_pair_ids,
    )
    relation_metric.pop("pair_prior_non_ambiguous", None)
    relation_metric.pop("majority_label_share_non_ambiguous", None)
    relation_metric["learned_raw"] = _relation_prediction_metrics(
        relation_logit_tensor.argmax(dim=-1)[~relation_ambiguous_tensor].tolist(),
        relation_target_tensor[~relation_ambiguous_tensor].tolist(),
    )
    guard_metric = _guard_structure_diagnostics_variable_size(
        all_guard_predictions,
        all_guard_targets,
        presence_target_tensor,
        variance_threshold=1.0e-8,
    )
    planner_metric = presence_diagnostics(
        probability_tensor, presence_target_tensor, fixed_thresholds.cpu()
    )
    for index, skill in enumerate(SKILLS):
        planner_metric["per_skill"][skill]["threshold"] = float(
            requested_thresholds[index]
        )
    single = _equal_task_metric(metric_rows, "single")
    group_a = _equal_task_metric(metric_rows, "A")
    graph_metric = {
        "sample_count": len(dataset),
        "pre_compiler_cycle_rate": pre_cycle_samples / len(dataset),
        "post_compiler_cycle_rate": 0.0,
        "dropped_edge_rate": dropped_edges / proposed_edges if proposed_edges else 0.0,
        "dropped_edges": dropped_edges,
        "proposed_edges": proposed_edges,
        "reentry_request_rate": reentry_requests / trace_slots if trace_slots else 0.0,
        "unexpected_skill_activation_rate": unexpected_activations / trace_slots
        if trace_slots
        else 0.0,
        "mean_program_levels": math.fsum(program_levels) / len(program_levels),
        "sample_stop_rate": stopped_samples / len(dataset),
        "stopped_samples": stopped_samples,
        "sample_stop_rate_definition": (
            "fraction_of_samples_with_stopped_mask_in_any_formal_inference_round"
        ),
    }
    threshold_values = [float(value) for value in requested_thresholds.tolist()]
    threshold_metadata: float | list[float] = (
        threshold_values[0] if scalar_threshold_input else threshold_values
    )
    return _json_finite(
        {
            "schema_version": STAGE3_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "created_utc": utc_now_iso(),
            "sources": {
                "restoration_presence_guard": "primary_val",
                "relation": "interaction_val",
                "mio100_rows_read": 0,
            },
            "checkpoint_presence_threshold": threshold_metadata,
            "presence_thresholds": {
                skill: threshold_values[index] for index, skill in enumerate(SKILLS)
            },
            "restoration": {"single": single, "group_a": group_a},
            "planner": planner_metric,
            "relation": relation_metric,
            "guard": guard_metric,
            "graph": graph_metric,
        }
    )


@torch.inference_mode()
def collect_primary_val_presence(
    model: GraphRestore,
    dataset: GraphRestoreEpisodeDataset,
    *,
    device: torch.device,
    use_bf16: bool = True,
) -> tuple[Tensor, Tensor]:
    """Collect only primary_val presence outputs for the one-time calibration."""

    if dataset.training or dataset.crop_size is not None:
        raise Stage3ContractError("presence calibration requires full primary_val")
    model.eval()
    probabilities: list[Tensor] = []
    targets: list[Tensor] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
        padded, _ = pad_to_multiple(image, 8)
        with _validation_autocast(device, use_bf16):
            features = model.encode(padded)
            output = model.plan_state(
                padded,
                padded,
                features,
                round_value=0.0,
                compute_relations=False,
            )
        probabilities.append(output.presence_probabilities[0].float().cpu())
        targets.append(sample["presence_target"].float().cpu())
    return torch.stack(probabilities), torch.stack(targets)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def stage3_dependency_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pyiqa": _package_version("pyiqa"),
        "basicsr": _package_version("basicsr"),
    }


def build_stage3_provenance(
    paths: Stage3Paths,
    parent: Stage3ParentLoadReport,
    *,
    micro_batch: int,
    accumulation_steps: int,
    validation_vram_gate: Mapping[str, Any],
    max_steps: int = 12_000,
    extension: Stage3ExtensionEvidence | None = None,
) -> dict[str, Any]:
    if micro_batch not in {1, 2, 4, 8} or 8 % micro_batch:
        raise Stage3ContractError("Stage3 micro batch must divide effective batch 8")
    if accumulation_steps != 8 // micro_batch or max_steps != 12_000:
        raise Stage3ContractError("Stage3 runtime schedule drifted")
    validation_vram_evidence = validate_stage3_validation_vram_evidence(
        validation_vram_gate
    )
    expected = _mapping(
        paths.resolved.get("expected_identity"), field="expected_identity"
    )
    agenticir_commit = git_commit(paths.resolved["agenticir_repo"])
    mioir_commit = git_commit(paths.resolved["mioir_repo"])
    if agenticir_commit != expected.get(
        "agenticir_commit"
    ) or mioir_commit != expected.get("mioir_commit"):
        raise Stage3ContractError("upstream repository commit drifted")
    bindings = {
        logical: dict(value) for logical, value in paths.approval.bindings.items()
    }
    target_step = max_steps if extension is None else extension.target_step
    if extension is not None and (
        extension.base_step != max_steps
        or extension.cycles != 3
        or extension.validation_every_steps != 2_000
        or extension.schedule_horizon_steps != max_steps
        or extension.validation_steps != STAGE3_EXTENSION_VALIDATION_STEPS
        or extension.min_lr != float(paths.config["optimization"]["min_lr"])
        or extension.lr_policy != STAGE3_EXTENSION_LR_POLICY
    ):
        raise Stage3ContractError("Stage3 extension schedule drifted")
    provenance = {
        "schema_version": STAGE3_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": sha256_file(paths.config_path),
        "config_semantic_sha256": sha256_json(paths.config),
        "resolved_paths_sha256": sha256_file(paths.resolved_path),
        "semantic_source_sha256": semantic_source_hashes(
            paths.project_root,
            entrypoints=(
                "scripts/train_stage3_planner.py",
                "scripts/eval_guard_diagnostics.py",
            ),
        ),
        "stage3_approval": {
            "path": str(paths.approval.approval_path),
            "sha256": paths.approval.approval_sha256,
            "stage2_decision_sha256": paths.approval.stage2_decision_sha256,
            "approval_required_sha256": paths.approval.approval_required_sha256,
            "both_explicit_orchestrator_flags_verified": True,
        },
        "bindings": bindings,
        "parent_checkpoint": {
            "path": str(paths.executor_checkpoint),
            "sha256": parent.checkpoint_sha256,
            "step": parent.checkpoint_step,
            "source": "stage1_best_ema_model_equals_shadow",
        },
        "ema_policy": stage3_ema_policy_metadata(float(paths.config["ema"]["decay"])),
        "relation_supervision": {
            "train_sha256": sha256_file(paths.relation_train),
            "validation_sha256": sha256_file(paths.relation_val),
            "class_count": 3,
            "orientation": "ProgramPlanner.PAIR_INDICES_ascending_normative_skill_id",
            "ambiguous": "serial_mass_partial_label_weight_0.25_once",
            "validation_source": "interaction_val_only",
        },
        "effect_profiles_sha256": sha256_file(paths.effect_profiles),
        "pair_prior_sha256": sha256_file(paths.pair_prior),
        "global_priority_sha256": sha256_file(paths.global_priority),
        "repositories": {
            "agenticir_commit": agenticir_commit,
            "mioir_commit": mioir_commit,
        },
        "runtime": {
            "crop_size": 192,
            "micro_batch": micro_batch,
            "effective_batch_size": 8,
            "accumulation_steps": accumulation_steps,
            "max_steps": max_steps,
            "training_target_step": target_step,
            "amp_dtype": "bf16",
            "tf32": True,
            "allocator_conf": validate_stage3_allocator_conf(),
            "model_generated_intermediate_maximum_fraction": 0.10,
            "validation_vram_gate": validation_vram_evidence,
            "validation_vram_gate_sha256": sha256_json(validation_vram_evidence),
        },
        "dependency_versions": stage3_dependency_versions(),
        "data_exposure": {
            "train": "primary_train single/A only",
            "validation": "primary_val single/A + interaction_val labels",
            "mio100": False,
            "group_b_or_c": False,
        },
    }
    if extension is not None:
        provenance["stage3_extension"] = extension.provenance_binding()
    return provenance


def stage3_training_target_step(
    provenance: Mapping[str, Any],
    *,
    schedule_horizon_steps: int,
    validation_every_steps: int = 2_000,
) -> int:
    """Return the authorized training target without altering the LR horizon."""

    runtime = _mapping(provenance.get("runtime"), field="Stage3 provenance runtime")
    if runtime.get("max_steps") != schedule_horizon_steps:
        raise Stage3ContractError("Stage3 scheduler horizon/provenance drifted")
    target = runtime.get("training_target_step", schedule_horizon_steps)
    extension = provenance.get("stage3_extension")
    if extension is None:
        if target != schedule_horizon_steps:
            raise Stage3ContractError(
                "Stage3 target exceeds its schedule without extension authorization"
            )
        return schedule_horizon_steps
    extension = _mapping(extension, field="Stage3 extension provenance")
    expected = {
        "cycles": 3,
        "base_step": STAGE3_BASE_TARGET_STEP,
        "target_step": STAGE3_EXTENSION_TARGET_STEP,
        "validation_every_steps": 2_000,
        "validation_steps": list(STAGE3_EXTENSION_VALIDATION_STEPS),
        "schedule_horizon_steps": STAGE3_BASE_TARGET_STEP,
        "min_lr": 2.0e-6,
        "lr_policy": STAGE3_EXTENSION_LR_POLICY,
    }
    if schedule_horizon_steps != STAGE3_BASE_TARGET_STEP:
        raise Stage3ContractError("Stage3 extension scheduler horizon drifted")
    if target != STAGE3_EXTENSION_TARGET_STEP:
        raise Stage3ContractError("Stage3 extension training target drifted")
    if not isinstance(extension.get("path"), str) or not is_sha256(
        extension.get("sha256")
    ):
        raise Stage3ContractError("Stage3 extension provenance binding is invalid")
    mismatches = {
        key: {"expected": value, "actual": extension.get(key)}
        for key, value in expected.items()
        if extension.get(key) != value
    }
    if set(extension) != {"path", "sha256", *expected}:
        mismatches["fields"] = {
            "expected": sorted({"path", "sha256", *expected}),
            "actual": sorted(extension),
        }
    if tuple(expected["validation_steps"]) != tuple(
        range(
            STAGE3_BASE_TARGET_STEP + validation_every_steps,
            STAGE3_EXTENSION_TARGET_STEP + 1,
            validation_every_steps,
        )
    ):
        raise Stage3ContractError("Stage3 extension validation schedule is invalid")
    if mismatches:
        raise Stage3ContractError(
            f"Stage3 extension provenance schedule mismatch: {mismatches}"
        )
    return STAGE3_EXTENSION_TARGET_STEP


def _stage3_optimizer_serialized_parameter_names(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[int, str], dict[int, nn.Parameter]]:
    canonical_names = {
        id(parameter): name
        for name, parameter in unwrap_model(model).named_parameters()
    }
    serialized = optimizer.state_dict()
    serialized_groups = serialized.get("param_groups")
    if not isinstance(serialized_groups, list) or len(serialized_groups) != len(
        optimizer.param_groups
    ):
        raise Stage3ContractError("Stage3 optimizer parameter-name mapping drifted")
    names: dict[int, str] = {}
    parameters: dict[int, nn.Parameter] = {}
    for serialized_group, live_group in zip(
        serialized_groups, optimizer.param_groups, strict=True
    ):
        if not isinstance(serialized_group, Mapping):
            raise Stage3ContractError("Stage3 optimizer serialized group is invalid")
        serialized_ids = serialized_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(serialized_ids, list) or not isinstance(
            live_parameters, list
        ):
            raise Stage3ContractError("Stage3 optimizer parameter list is invalid")
        if len(serialized_ids) != len(live_parameters):
            raise Stage3ContractError("Stage3 optimizer parameter list size drifted")
        for serialized_id, parameter in zip(
            serialized_ids, live_parameters, strict=True
        ):
            if (
                isinstance(serialized_id, bool)
                or not isinstance(serialized_id, int)
                or not isinstance(parameter, nn.Parameter)
                or serialized_id in names
            ):
                raise Stage3ContractError("Stage3 optimizer serialized ID is invalid")
            name = canonical_names.get(id(parameter))
            if name is None or not name.startswith("planner."):
                raise Stage3ContractError(
                    "Stage3 optimizer contains a non-planner parameter"
                )
            names[serialized_id] = name
            parameters[serialized_id] = parameter
    return names, parameters


def _validate_stage3_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: object,
    ledger: object,
    *,
    step: int,
) -> dict[int, str]:
    """Validate complete planner Adam state and its stable ID-to-name ledger."""

    optimizer_state = _mapping(state, field="checkpoint.optimizer")
    loaded_state = optimizer_state.get("state")
    loaded_groups = optimizer_state.get("param_groups")
    current_state = optimizer.state_dict()
    current_groups = current_state.get("param_groups")
    if not isinstance(loaded_state, Mapping):
        raise Stage3ContractError("Stage3 optimizer state is invalid")
    if not isinstance(loaded_groups, list) or not isinstance(current_groups, list):
        raise Stage3ContractError("Stage3 optimizer groups are invalid")
    if len(loaded_groups) != len(current_groups):
        raise Stage3ContractError("Stage3 optimizer group count drifted")

    serialized_names, live_parameters = _stage3_optimizer_serialized_parameter_names(
        model, optimizer
    )
    all_parameter_ids = set(serialized_names)
    for loaded_group, current_group in zip(loaded_groups, current_groups, strict=True):
        if not isinstance(loaded_group, Mapping) or not isinstance(
            current_group, Mapping
        ):
            raise Stage3ContractError("Stage3 optimizer group is invalid")
        if set(loaded_group) != set(current_group):
            raise Stage3ContractError("Stage3 optimizer group fields drifted")
        loaded_parameters = loaded_group.get("params")
        current_parameters = current_group.get("params")
        if loaded_parameters != current_parameters:
            raise Stage3ContractError("Stage3 optimizer parameter ID order drifted")
        for key in set(current_group) - {"params", "lr"}:
            if loaded_group.get(key) != current_group.get(key):
                raise Stage3ContractError(
                    f"Stage3 optimizer static field drifted: {key}"
                )
        dynamic_lr = loaded_group.get("lr")
        if (
            isinstance(dynamic_lr, bool)
            or not isinstance(dynamic_lr, (int, float))
            or not math.isfinite(float(dynamic_lr))
        ):
            raise Stage3ContractError("Stage3 optimizer LR is non-finite")

    state_ids = set(loaded_state)
    expected_state_ids = set() if step == 0 else all_parameter_ids
    if state_ids != expected_state_ids:
        raise Stage3ContractError(
            "Stage3 optimizer state does not cover every planner parameter"
        )
    if not isinstance(ledger, Mapping):
        raise Stage3ContractError("Stage3 optimizer state-name ledger is invalid")
    if set(ledger) != state_ids:
        raise Stage3ContractError(
            "Stage3 optimizer state-name ledger keys differ from optimizer state"
        )

    normalized: dict[int, str] = {}
    for serialized_id in sorted(state_ids):
        if (
            isinstance(serialized_id, bool)
            or not isinstance(serialized_id, int)
            or serialized_id not in all_parameter_ids
        ):
            raise Stage3ContractError("Stage3 optimizer state ID is invalid")
        name = ledger[serialized_id]
        if not isinstance(name, str) or name != serialized_names[serialized_id]:
            raise Stage3ContractError(
                f"Stage3 optimizer ledger name drifted at ID {serialized_id}"
            )
        parameter_state = loaded_state[serialized_id]
        if not isinstance(parameter_state, Mapping):
            raise Stage3ContractError("Stage3 Adam parameter state is invalid")
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        group = next(
            group for group in loaded_groups if serialized_id in group["params"]
        )
        if group.get("amsgrad") is True:
            expected_keys.add("max_exp_avg_sq")
        if set(parameter_state) != expected_keys:
            raise Stage3ContractError("Stage3 Adam state fields drifted")
        state_step = parameter_state["step"]
        if torch.is_tensor(state_step):
            if state_step.numel() != 1 or not bool(torch.isfinite(state_step).all()):
                raise Stage3ContractError("Stage3 Adam step is invalid")
            state_step_value = float(state_step.item())
        elif isinstance(state_step, (int, float)) and not isinstance(state_step, bool):
            state_step_value = float(state_step)
        else:
            raise Stage3ContractError("Stage3 Adam step is invalid")
        if (
            not math.isfinite(state_step_value)
            or not state_step_value.is_integer()
            or int(state_step_value) != step
        ):
            raise Stage3ContractError("Stage3 Adam step differs from checkpoint step")
        parameter = live_parameters[serialized_id]
        for key in expected_keys - {"step"}:
            tensor = parameter_state[key]
            if (
                not torch.is_tensor(tensor)
                or tuple(tensor.shape) != tuple(parameter.shape)
                or tensor.dtype != parameter.dtype
                or not bool(torch.isfinite(tensor).all())
            ):
                raise Stage3ContractError(f"Stage3 Adam tensor state is invalid: {key}")
        normalized[serialized_id] = name
    return normalized


def _validate_stage3_scheduler_state(
    scheduler: WarmupCosineScheduler,
    state: object,
    optimizer_state: object,
    *,
    step: int,
    context: str,
) -> Mapping[str, Any]:
    """Validate the complete scheduler trajectory before mutating live state."""

    scheduler_state = _mapping(state, field=f"{context}.scheduler")
    optimizer_mapping = _mapping(optimizer_state, field=f"{context}.optimizer")
    current = scheduler.state_dict()
    if set(scheduler_state) != set(current):
        raise Stage3ContractError(f"{context} scheduler state fields drifted")
    dynamic_fields = {"last_epoch", "_step_count", "_last_lr"}
    for key in set(current) - dynamic_fields:
        if scheduler_state.get(key) != current.get(key):
            raise Stage3ContractError(f"{context} scheduler {key} drifted")

    last_epoch = scheduler_state.get("last_epoch")
    step_count = scheduler_state.get("_step_count")
    if (
        isinstance(last_epoch, bool)
        or not isinstance(last_epoch, int)
        or last_epoch != step
    ):
        raise Stage3ContractError(
            f"{context} scheduler.last_epoch must equal checkpoint step"
        )
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count != step + 1
    ):
        raise Stage3ContractError(
            f"{context} scheduler._step_count must equal checkpoint step + 1"
        )

    base_lrs = scheduler_state.get("base_lrs")
    last_lrs = scheduler_state.get("_last_lr")
    optimizer_groups = optimizer_mapping.get("param_groups")
    if not isinstance(base_lrs, list) or not isinstance(last_lrs, list):
        raise Stage3ContractError(f"{context} scheduler LR state is invalid")
    if not isinstance(optimizer_groups, list):
        raise Stage3ContractError(f"{context} optimizer groups are invalid")
    if not len(base_lrs) == len(last_lrs) == len(optimizer_groups):
        raise Stage3ContractError(f"{context} scheduler LR group count drifted")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (*base_lrs, *last_lrs)
    ):
        raise Stage3ContractError(f"{context} scheduler LR state is non-finite")

    warmup_steps = scheduler_state.get("warmup_steps")
    max_steps = scheduler_state.get("max_steps")
    min_lr = scheduler_state.get("min_lr")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps <= warmup_steps
        or isinstance(min_lr, bool)
        or not isinstance(min_lr, (int, float))
        or not math.isfinite(float(min_lr))
        or float(min_lr) < 0.0
    ):
        raise Stage3ContractError(f"{context} scheduler static contract is invalid")

    for index, (base_lr, last_lr, group) in enumerate(
        zip(base_lrs, last_lrs, optimizer_groups, strict=True)
    ):
        if not isinstance(group, Mapping):
            raise Stage3ContractError(f"{context} optimizer group is invalid")
        if group.get("initial_lr") != base_lr:
            raise Stage3ContractError(
                f"{context} optimizer initial_lr/scheduler base_lrs drifted "
                f"at group {index}"
            )
        floor = min(float(min_lr), float(base_lr))
        if warmup_steps and step < warmup_steps:
            scale = float(step + 1) / float(warmup_steps)
            expected_lr = float(base_lr) * scale
        else:
            progress = min(
                1.0,
                max(
                    0.0,
                    (step - warmup_steps) / (max_steps - warmup_steps),
                ),
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            expected_lr = floor + (float(base_lr) - floor) * cosine
        dynamic_lr = group.get("lr")
        if (
            isinstance(dynamic_lr, bool)
            or not isinstance(dynamic_lr, (int, float))
            or not math.isfinite(float(dynamic_lr))
            or dynamic_lr != last_lr
            or dynamic_lr != expected_lr
        ):
            raise Stage3ContractError(
                f"{context} optimizer/scheduler LR trajectory drifted at group {index}"
            )
    return scheduler_state


def _validate_stage3_rng_state(state: object) -> Mapping[str, Any]:
    rng = _mapping(state, field="checkpoint.rng_states")
    try:
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        cpu_state = rng["torch_cpu"]
        if not torch.is_tensor(cpu_state) or cpu_state.dtype != torch.uint8:
            raise TypeError("invalid torch CPU RNG tensor")
        torch.Generator(device="cpu").set_state(cpu_state)
        cuda_states = rng.get("torch_cuda_all")
        if torch.cuda.is_available() and cuda_states is None:
            raise ValueError("CUDA RNG state is missing")
        if cuda_states is not None:
            if not isinstance(cuda_states, (list, tuple)):
                raise TypeError("invalid CUDA RNG state list")
            if (
                torch.cuda.is_available()
                and len(cuda_states) != torch.cuda.device_count()
            ):
                raise ValueError("CUDA RNG state count drifted")
            if any(
                not torch.is_tensor(value) or value.dtype != torch.uint8
                for value in cuda_states
            ):
                raise TypeError("invalid CUDA RNG tensor")
            if torch.cuda.is_available():
                current_cuda_states = torch.cuda.get_rng_state_all()
                if any(
                    tuple(value.shape) != tuple(reference.shape)
                    for value, reference in zip(
                        cuda_states, current_cuda_states, strict=True
                    )
                ):
                    raise ValueError("CUDA RNG state shape drifted")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage3ContractError("Stage3 resume RNG state is invalid") from exc
    return rng


def _validate_stage3_tensor_state(
    state: object,
    reference: Mapping[str, Tensor],
    *,
    field: str,
) -> Mapping[str, Tensor]:
    tensors = _strict_tensor_mapping(state, field=field)
    if tensors.keys() != reference.keys():
        raise Stage3ContractError(f"{field} keys drifted")
    for name, value in tensors.items():
        expected = reference[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise Stage3ContractError(f"{field} tensor contract drifted at {name}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise Stage3ContractError(f"{field} tensor is non-finite at {name}")
    return tensors


def _stage3_tensors_equal_exact(reference: Tensor, candidate: Tensor) -> bool:
    """Compare exact tensor values without weakening cross-device validation."""

    if (
        reference.shape != candidate.shape
        or reference.dtype != candidate.dtype
        or reference.layout != candidate.layout
    ):
        return False
    detached = candidate.detach()
    if detached.device != reference.device:
        # Resume checkpoints are deliberately CPU-mapped while the already
        # loaded Stage1 parent is live on CUDA.  Move the live value to the
        # checkpoint device one tensor at a time: no cast, tolerance, or live
        # mutation is permitted by the frozen-parent exact contract.
        detached = detached.to(device=reference.device)
    return torch.equal(reference.detach(), detached)


_STAGE3_CURRENT_METRIC_FIELDS = (
    "group_a_psnr",
    "group_a_ssim",
    "single_psnr",
    "single_ssim",
    "validation_step",
)
_STAGE3_BEST_METRIC_FIELDS = (
    "best_group_a_psnr",
    "best_group_a_ssim",
    "best_single_psnr",
    "best_single_ssim",
    "best_step",
)


def _validate_stage3_metrics(
    value: object,
    *,
    step: int,
    max_steps: int,
    validation_every_steps: int,
) -> Mapping[str, Any]:
    metrics = _mapping(value, field="checkpoint.metrics")
    allowed = set(_STAGE3_CURRENT_METRIC_FIELDS) | set(_STAGE3_BEST_METRIC_FIELDS)
    if not set(metrics) <= allowed:
        raise Stage3ContractError("Stage3 checkpoint metrics contain unknown fields")
    for label, fields, step_field in (
        ("current", _STAGE3_CURRENT_METRIC_FIELDS, "validation_step"),
        ("best", _STAGE3_BEST_METRIC_FIELDS, "best_step"),
    ):
        present = [field in metrics for field in fields]
        if any(present) and not all(present):
            raise Stage3ContractError(
                f"Stage3 checkpoint metrics have partial {label} fields"
            )
        if not all(present):
            continue
        metric_step = metrics[step_field]
        if (
            isinstance(metric_step, bool)
            or not isinstance(metric_step, int)
            or not 0 <= metric_step <= step
        ):
            raise Stage3ContractError(
                f"Stage3 checkpoint metrics {step_field} is invalid"
            )
        if (
            metric_step != 0
            and metric_step % validation_every_steps != 0
            and metric_step != max_steps
        ):
            raise Stage3ContractError(
                f"Stage3 checkpoint metrics {step_field} is not a validation boundary"
            )
        for field in fields:
            if field == step_field:
                continue
            metric = metrics[field]
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
            ):
                raise Stage3ContractError(
                    f"Stage3 checkpoint metrics {field} is non-finite"
                )
    if (
        "best_step" in metrics
        and "validation_step" in metrics
        and metrics["best_step"] > metrics["validation_step"]
    ):
        raise Stage3ContractError(
            "Stage3 checkpoint metrics best_step exceeds validation_step"
        )
    if "best_step" in metrics and "validation_step" in metrics:
        current = ValidationScore(
            group_a_psnr=float(metrics["group_a_psnr"]),
            group_a_ssim=float(metrics["group_a_ssim"]),
            single_psnr=float(metrics["single_psnr"]),
            single_ssim=float(metrics["single_ssim"]),
            step=int(metrics["validation_step"]),
        )
        best = ValidationScore(
            group_a_psnr=float(metrics["best_group_a_psnr"]),
            group_a_ssim=float(metrics["best_group_a_ssim"]),
            single_psnr=float(metrics["best_single_psnr"]),
            single_ssim=float(metrics["best_single_ssim"]),
            step=int(metrics["best_step"]),
        )
        if is_better_checkpoint(current, best):
            raise Stage3ContractError(
                "Stage3 checkpoint current metrics are better than its incumbent"
            )
    return metrics


def _validate_stage3_ema_state(
    model: GraphRestore,
    ema: Stage3PlannerEMA,
    state: object,
    *,
    step: int,
    raw_state: Mapping[str, Tensor] | None = None,
) -> Mapping[str, Any]:
    ema_state = _mapping(state, field="checkpoint.ema")
    num_updates = ema_state.get("num_updates")
    if (
        ema_state.get("scope") != STAGE3_EMA_SCOPE
        or ema_state.get("policy") != stage3_ema_policy_metadata(ema.decay)
        or ema_state.get("decay") != ema.decay
        or isinstance(num_updates, bool)
        or not isinstance(num_updates, int)
        or num_updates != step
    ):
        raise Stage3ContractError("Stage3 EMA metadata drifted")
    shadow = _strict_tensor_mapping(
        ema_state.get("shadow"), field="checkpoint.ema.shadow"
    )
    live = unwrap_model(model).state_dict()
    raw = live if raw_state is None else raw_state
    if shadow.keys() != raw.keys() or shadow.keys() != ema.shadow.keys():
        raise Stage3ContractError("Stage3 EMA keys drifted")
    for name, value in shadow.items():
        expected = ema.shadow[name]
        raw_value = raw[name]
        if (
            value.shape != expected.shape
            or value.dtype != expected.dtype
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
        ):
            raise Stage3ContractError(f"Stage3 EMA tensor contract drifted at {name}")
        if not name.startswith("planner."):
            if not torch.equal(value, raw_value):
                raise Stage3ContractError(
                    f"Stage3 frozen executor differs from EMA shadow at {name}"
                )
            if raw_state is not None and not _stage3_tensors_equal_exact(
                raw_value, live[name]
            ):
                raise Stage3ContractError(
                    "Stage3 frozen executor differs from the live Stage1 parent "
                    f"at {name}"
                )
    return ema_state


def save_stage3_checkpoint(
    destination: str | Path,
    *,
    step: int,
    model: GraphRestore,
    ema: Stage3PlannerEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: StatefulEpisodeSampler,
    provenance: Mapping[str, Any],
    metrics: Mapping[str, float | int] | None = None,
    model_as_ema: bool = False,
    pending_validation_step: int | None = None,
    optimizer_transaction: Stage3OptimizerTransaction | None = None,
    validation_every_steps: int = 2_000,
) -> None:
    if optimizer_transaction is not None and optimizer_transaction.active:
        raise Stage3ContractError(
            "refusing to serialize a mid-optimizer-update Stage3 state"
        )
    if not isinstance(ema, Stage3PlannerEMA):
        raise Stage3ContractError("Stage3 checkpoints require Stage3PlannerEMA")
    max_steps = stage3_training_target_step(
        provenance,
        schedule_horizon_steps=int(scheduler.max_steps),
        validation_every_steps=validation_every_steps,
    )
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= max_steps
    ):
        raise Stage3ContractError("Stage3 checkpoint step is invalid")
    raw_model_state = _validate_stage3_tensor_state(
        unwrap_model(model).state_dict(),
        unwrap_model(model).state_dict(),
        field="checkpoint.model",
    )
    ema_state = _validate_stage3_ema_state(
        model,
        ema,
        ema.state_dict(),
        step=step,
    )
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    serialized_names, _ = _stage3_optimizer_serialized_parameter_names(model, optimizer)
    state_ids = set(_mapping(optimizer_state.get("state"), field="optimizer.state"))
    optimizer_ledger = {
        serialized_id: serialized_names[serialized_id]
        for serialized_id in sorted(state_ids)
    }
    optimizer_ledger = _validate_stage3_optimizer_state(
        model,
        optimizer,
        optimizer_state,
        optimizer_ledger,
        step=step,
    )
    _validate_stage3_scheduler_state(
        scheduler,
        scheduler_state,
        optimizer_state,
        step=step,
        context="Stage3 checkpoint save",
    )
    validated_metrics = _validate_stage3_metrics(
        {} if metrics is None else metrics,
        step=step,
        max_steps=max_steps,
        validation_every_steps=validation_every_steps,
    )
    pending = validate_stage3_pending_validation_step(
        step=step,
        pending_validation_step=pending_validation_step,
        max_steps=max_steps,
        validation_every_steps=validation_every_steps,
    )
    if model_as_ema and pending is not None:
        raise Stage3ContractError("Stage3 EMA selection cannot be pending validation")
    context = ema.apply_to(model) if model_as_ema else nullcontext()
    with context:
        payload = checkpoint_payload(
            stage="stage3",
            step=step,
            model=model,
            ema_state=ema_state,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            sampler_state=sampler.state_dict(consumed_optimizer_step=step),
            provenance=provenance,
            metrics=validated_metrics,
        )
        payload["amp"] = {"dtype": "bfloat16", "scaler_required": False}
        payload["executor_frozen"] = True
        payload["trainable_prefixes"] = ["planner."]
        payload["model_role"] = (
            "ema_selection" if model_as_ema else "raw_training_state"
        )
        payload["resumable"] = not model_as_ema
        payload["pending_validation_step"] = pending
        payload["optimizer_transaction_active"] = False
        payload["optimizer_state_name_ledger"] = optimizer_ledger
        _validate_stage3_tensor_state(
            payload.get("model"),
            raw_model_state if not model_as_ema else ema.shadow,
            field="checkpoint.model",
        )
        _validate_stage3_rng_state(payload.get("rng_states"))
        atomic_torch_save(payload, destination)
    set_stage3_trainability(model)


def _restore_stage3_ema(ema: ExponentialMovingAverage, value: object) -> None:
    state = _mapping(value, field="checkpoint.ema")
    if isinstance(ema, Stage3PlannerEMA) and state.get("scope") != STAGE3_EMA_SCOPE:
        raise Stage3ContractError("Stage3 resume EMA scope drifted")
    shadow = _strict_tensor_mapping(state.get("shadow"), field="checkpoint.ema.shadow")
    if shadow.keys() != ema.shadow.keys():
        raise Stage3ContractError("Stage3 resume EMA keys drifted")
    ema.load_state_dict(state)


def _validate_stage3_resume_header(
    header: object,
    *,
    model: GraphRestore,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: StatefulEpisodeSampler,
    expected_provenance: Mapping[str, Any],
    validation_every_steps: int,
) -> None:
    """Validate the complete resume contract before mutating live objects."""

    if not isinstance(ema, Stage3PlannerEMA):
        raise Stage3ContractError("Stage3 resume requires Stage3PlannerEMA")
    if (
        not isinstance(header, Mapping)
        or header.get("schema_version") != "graphrestore-checkpoint-v1"
        or header.get("stage") != "stage3"
        or header.get("model_role") != "raw_training_state"
        or header.get("resumable") is not True
        or header.get("scaler") is not None
    ):
        raise Stage3ContractError(
            "Stage3 resume requires raw last.pth; EMA selection checkpoints are non-resumable"
        )
    if header.get("executor_frozen") is not True:
        raise Stage3ContractError(
            "resume checkpoint is not a frozen-executor Stage3 run"
        )
    if header.get("trainable_prefixes") != ["planner."]:
        raise Stage3ContractError("Stage3 resume trainable scope drifted")
    if header.get("amp") != {"dtype": "bfloat16", "scaler_required": False}:
        raise Stage3ContractError("Stage3 resume AMP contract drifted")
    if header.get("optimizer_transaction_active") is not False:
        raise Stage3ContractError(
            "Stage3 resume checkpoint records a mid-optimizer-update state"
        )
    if header.get("provenance") != dict(expected_provenance):
        raise Stage3ContractError("Stage3 resume provenance drifted")

    training_target_step = stage3_training_target_step(
        expected_provenance,
        schedule_horizon_steps=int(scheduler.max_steps),
        validation_every_steps=validation_every_steps,
    )

    step = header.get("step")
    pending_validation_step = validate_stage3_pending_validation_step(
        step=step,
        pending_validation_step=header.get("pending_validation_step", object()),
        max_steps=training_target_step,
        validation_every_steps=validation_every_steps,
    )
    _validate_stage3_metrics(
        header.get("metrics"),
        step=int(step),
        max_steps=training_target_step,
        validation_every_steps=validation_every_steps,
    )

    expected_model_state = unwrap_model(model).state_dict()
    model_state = _validate_stage3_tensor_state(
        header.get("model"),
        expected_model_state,
        field="Stage3 resume model",
    )

    _validate_stage3_ema_state(
        model,
        ema,
        header.get("ema"),
        step=int(step),
        raw_state=model_state,
    )

    optimizer_state = _mapping(header.get("optimizer"), field="checkpoint.optimizer")
    if "optimizer_state_name_ledger" not in header:
        raise Stage3ContractError("Stage3 resume lacks optimizer state-name ledger")
    _validate_stage3_optimizer_state(
        model,
        optimizer,
        optimizer_state,
        header.get("optimizer_state_name_ledger"),
        step=int(step),
    )
    current_optimizer = optimizer.state_dict()
    saved_groups = optimizer_state.get("param_groups")
    current_groups = current_optimizer.get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        raise Stage3ContractError("Stage3 resume optimizer groups are invalid")
    if len(saved_groups) != len(current_groups):
        raise Stage3ContractError("Stage3 resume optimizer parameter groups drifted")
    for saved, current in zip(saved_groups, current_groups, strict=True):
        if not isinstance(saved, Mapping):
            raise Stage3ContractError(
                "Stage3 resume optimizer parameter groups drifted"
            )
        saved_parameters = saved.get("params")
        current_parameters = current.get("params")
        if (
            not isinstance(saved_parameters, list)
            or not isinstance(current_parameters, list)
            or saved_parameters != current_parameters
        ):
            raise Stage3ContractError(
                "Stage3 resume optimizer parameter groups drifted"
            )
    _validate_stage3_scheduler_state(
        scheduler,
        header.get("scheduler"),
        optimizer_state,
        step=int(step),
        context="Stage3 resume",
    )
    _validate_stage3_rng_state(header.get("rng_states"))

    sampler_state = _mapping(
        header.get("sampler_state"), field="checkpoint.sampler_state"
    )
    sampler_expected = sampler.state_dict()
    for key in (
        "schema_version",
        "stage",
        "base_seed",
        "num_samples",
        "effective_batch_size",
    ):
        if sampler_state.get(key) != sampler_expected.get(key):
            raise Stage3ContractError(f"Stage3 resume sampler {key} drifted")
    if sampler_state.get("consumed_optimizer_step") != step:
        raise Stage3ContractError("Stage3 checkpoint/sampler step mismatch")
    if sampler_state.get("sample_cursor") != int(step) * sampler.effective_batch_size:
        raise Stage3ContractError("Stage3 checkpoint/sampler cursor mismatch")

    # Keep the normalized pending value live in this validation scope so a
    # missing key can never be confused with a clean, non-pending checkpoint.
    if pending_validation_step != header.get("pending_validation_step"):
        raise Stage3ContractError("Stage3 pending validation marker drifted")


def _validate_stage3_incumbent_checkpoint(
    checkpoint: Path,
    header: Mapping[str, Any],
    *,
    max_steps: int,
    validation_every_steps: int,
) -> None:
    """Bind resumable metrics to the atomically published EMA incumbent."""

    raw_metrics = _validate_stage3_metrics(
        header.get("metrics"),
        step=int(header["step"]),
        max_steps=max_steps,
        validation_every_steps=validation_every_steps,
    )
    pending = header.get("pending_validation_step")
    best_path = checkpoint.parent / "best_ema.pth"
    raw_has_best = "best_step" in raw_metrics
    if not best_path.is_file():
        if raw_has_best:
            raise Stage3ContractError(
                "Stage3 resumable metrics reference a missing best_ema.pth"
            )
        return

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(best, Mapping)
        or best.get("schema_version") != "graphrestore-checkpoint-v1"
        or best.get("stage") != "stage3"
        or best.get("model_role") != "ema_selection"
        or best.get("resumable") is not False
        or best.get("pending_validation_step") is not None
        or best.get("optimizer_transaction_active") is not False
    ):
        raise Stage3ContractError("Stage3 incumbent best_ema.pth metadata drifted")
    best_step = best.get("step")
    if isinstance(best_step, bool) or not isinstance(best_step, int):
        raise Stage3ContractError("Stage3 incumbent step is invalid")
    best_metrics = _validate_stage3_metrics(
        best.get("metrics"),
        step=best_step,
        max_steps=max_steps,
        validation_every_steps=validation_every_steps,
    )
    if (
        best_metrics.get("validation_step") != best_step
        or best_metrics.get("best_step") != best_step
    ):
        raise Stage3ContractError(
            "Stage3 incumbent metrics are not bound to its checkpoint step"
        )
    for current_field, best_field in zip(
        _STAGE3_CURRENT_METRIC_FIELDS[:-1],
        _STAGE3_BEST_METRIC_FIELDS[:-1],
        strict=True,
    ):
        if best_metrics.get(current_field) != best_metrics.get(best_field):
            raise Stage3ContractError(
                "Stage3 incumbent current/best restoration metrics differ"
            )

    raw_best_step = raw_metrics.get("best_step")
    if best_step == raw_best_step:
        for field in _STAGE3_BEST_METRIC_FIELDS[:-1]:
            if raw_metrics.get(field) != best_metrics.get(field):
                raise Stage3ContractError(
                    "Stage3 resumable metrics differ from best_ema.pth"
                )
        return
    # A signal may arrive after the new best was atomically published but
    # before the pending raw transaction was cleared.  That state is replayable
    # only when the new incumbent is exactly the pending boundary.
    if pending is not None and best_step == pending:
        return
    raise Stage3ContractError(
        "Stage3 best_ema.pth is not the resumable checkpoint incumbent"
    )


def resume_stage3_checkpoint(
    checkpoint: str | Path,
    *,
    model: GraphRestore,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: StatefulEpisodeSampler,
    expected_provenance: Mapping[str, Any],
    validation_every_steps: int = 2_000,
) -> dict[str, Any]:
    # Inspect role metadata before mutating model, optimizer, scheduler, RNG,
    # EMA, or sampler.  Selection EMA checkpoints are never resumable.
    if not isinstance(ema, Stage3PlannerEMA):
        raise Stage3ContractError("Stage3 resume requires Stage3PlannerEMA")
    header = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    _validate_stage3_resume_header(
        header,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        expected_provenance=expected_provenance,
        validation_every_steps=validation_every_steps,
    )
    _validate_stage3_incumbent_checkpoint(
        Path(checkpoint),
        header,
        max_steps=stage3_training_target_step(
            expected_provenance,
            schedule_horizon_steps=int(scheduler.max_steps),
            validation_every_steps=validation_every_steps,
        ),
        validation_every_steps=validation_every_steps,
    )
    # Install the exact in-memory payload that passed every pre-mutation check.
    # Re-reading the path here would create a TOCTOU window in which an atomic
    # replacement with the same provenance but corrupt tensors could bypass the
    # validation above and still mutate the live training state.
    payload = dict(header)
    incompatible = unwrap_model(model).load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Stage3ContractError("Stage3 resume model failed strict installation")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    step = payload.get("step")
    _restore_stage3_ema(ema, payload.get("ema"))
    sampler_state = _mapping(
        payload.get("sampler_state"), field="checkpoint.sampler_state"
    )
    sampler.load_state_dict(dict(sampler_state))
    if sampler_state.get("consumed_optimizer_step") != step:
        raise Stage3ContractError("Stage3 checkpoint/sampler step mismatch")
    restore_rng_state(payload["rng_states"])
    set_stage3_trainability(model)
    return payload


def load_stage3_best_ema(
    paths: Stage3Paths,
    checkpoint: str | Path,
    *,
    device: torch.device,
    model_factory: Callable[..., GraphRestore] = GraphRestore,
    load_frozen_thresholds: bool = True,
    historical_extension_authorization: Mapping[str, str] | None = None,
) -> GraphRestore:
    """Reusable Stage4/evaluation loader for a strictly exposed Stage3 EMA."""

    model, _ = build_stage3_model(
        paths, device=torch.device("cpu"), model_factory=model_factory
    )
    frozen_reference = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    planner_parameter_names = {
        f"planner.{name}" for name, _ in model.planner.named_parameters()
    }
    checkpoint_path = Path(checkpoint).resolve()
    if checkpoint_path.name != "best_ema.pth" or not checkpoint_path.is_file():
        raise Stage3ContractError("Stage3 selected checkpoint must be best_ema.pth")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = payload.get("step") if isinstance(payload, Mapping) else None
    if not isinstance(payload, Mapping):
        raise Stage3ContractError("Stage3 best checkpoint must be a mapping")
    provenance = _mapping(payload.get("provenance"), field="Stage3 best provenance")
    schedule_horizon_steps = int(paths.config["training"]["max_steps"])
    validation_every_steps = int(paths.config["runtime"]["validation_every_steps"])
    max_steps = stage3_training_target_step(
        provenance,
        schedule_horizon_steps=schedule_horizon_steps,
        validation_every_steps=validation_every_steps,
    )
    extension_binding = provenance.get("stage3_extension")
    if extension_binding is not None:
        extension_mapping = _mapping(
            extension_binding, field="Stage3 best extension provenance"
        )
        if historical_extension_authorization is None:
            extension_path = extension_mapping.get("path")
            if not isinstance(extension_path, str):
                raise Stage3ContractError(
                    "Stage3 best extension authorization path is invalid"
                )
            extension = validate_stage3_extension_authorization(extension_path, paths)
            if extension_mapping != extension.provenance_binding():
                raise Stage3ContractError(
                    "Stage3 best extension authorization binding drifted"
                )
        else:
            historical_path_raw = historical_extension_authorization.get("path")
            historical_sha = historical_extension_authorization.get("sha256")
            if not isinstance(historical_path_raw, str) or not is_sha256(
                historical_sha
            ):
                raise Stage3ContractError(
                    "historical Stage3 extension authorization binding is invalid"
                )
            historical_path = Path(historical_path_raw).resolve(strict=False)
            if (
                extension_mapping.get("path") != str(historical_path)
                or not historical_path.is_file()
                or sha256_file(historical_path) != historical_sha
                or extension_mapping.get("sha256") != historical_sha
            ):
                raise Stage3ContractError(
                    "historical Stage3 extension authorization hash drifted"
                )
            historical_payload = _mapping(
                load_json(historical_path),
                field="historical STAGE3_EXTENSION_APPROVED.json",
            )
            if (
                historical_payload.get("schema_version") != STAGE3_EXTENSION_SCHEMA
                or historical_payload.get("kind") != "stage3_extension_approval"
                or historical_payload.get("protocol_id") != PROTOCOL_ID
                or historical_payload.get("approved") is not True
                or historical_payload.get("base_step")
                != extension_mapping.get("base_step")
                or historical_payload.get("target_step")
                != extension_mapping.get("target_step")
                or historical_payload.get("validation_every_steps")
                != extension_mapping.get("validation_every_steps")
                or historical_payload.get("validation_steps")
                != extension_mapping.get("validation_steps")
                or historical_payload.get("schedule_horizon_steps")
                != extension_mapping.get("schedule_horizon_steps")
                or float(historical_payload.get("min_lr", -1.0))
                != float(extension_mapping.get("min_lr", -2.0))
                or historical_payload.get("lr_policy")
                != extension_mapping.get("lr_policy")
            ):
                raise Stage3ContractError(
                    "historical Stage3 extension authorization semantics drifted"
                )
    if (
        payload.get("schema_version") != "graphrestore-checkpoint-v1"
        or payload.get("stage") != "stage3"
        or payload.get("model_role") != "ema_selection"
        or payload.get("resumable") is not False
        or payload.get("pending_validation_step") is not None
        or payload.get("optimizer_transaction_active") is not False
        or payload.get("executor_frozen") is not True
        or payload.get("trainable_prefixes") != ["planner."]
        or payload.get("amp") != {"dtype": "bfloat16", "scaler_required": False}
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or step > max_steps
        or (step % validation_every_steps != 0 and step != max_steps)
    ):
        raise Stage3ContractError("Stage3 best checkpoint schema/stage mismatch")
    source = _validate_stage3_tensor_state(
        payload.get("model"),
        frozen_reference,
        field="Stage3 best model",
    )
    ema = _mapping(payload.get("ema"), field="Stage3 best EMA")
    ema_updates = ema.get("num_updates")
    if (
        ema.get("scope") != STAGE3_EMA_SCOPE
        or ema.get("policy")
        != stage3_ema_policy_metadata(float(paths.config["ema"]["decay"]))
        or ema.get("decay") != float(paths.config["ema"]["decay"])
        or isinstance(ema_updates, bool)
        or not isinstance(ema_updates, int)
        or ema_updates != step
    ):
        raise Stage3ContractError(
            "Stage3 best EMA did not preserve the frozen executor"
        )
    shadow = _validate_stage3_tensor_state(
        ema.get("shadow"),
        frozen_reference,
        field="Stage3 best EMA shadow",
    )
    if source.keys() != shadow.keys() or any(
        not torch.equal(source[name], shadow[name]) for name in source
    ):
        raise Stage3ContractError("Stage3 best_ema.pth does not expose EMA as model")
    metrics = _validate_stage3_metrics(
        payload.get("metrics"),
        step=step,
        max_steps=max_steps,
        validation_every_steps=validation_every_steps,
    )
    if metrics.get("validation_step") != step or metrics.get("best_step") != step:
        raise Stage3ContractError("Stage3 best checkpoint metrics/step drifted")
    for current_field, best_field in zip(
        _STAGE3_CURRENT_METRIC_FIELDS[:-1],
        _STAGE3_BEST_METRIC_FIELDS[:-1],
        strict=True,
    ):
        if metrics.get(current_field) != metrics.get(best_field):
            raise Stage3ContractError(
                "Stage3 best checkpoint current/best metrics drifted"
            )
    fixed_drift = [
        name
        for name in source
        if name not in planner_parameter_names
        and not torch.equal(source[name], frozen_reference[name])
    ]
    if fixed_drift:
        raise Stage3ContractError(
            "Stage3 best checkpoint changed frozen executor/buffer state: "
            f"{fixed_drift[:8]}"
        )
    if provenance.get("ema_policy") != stage3_ema_policy_metadata(
        float(paths.config["ema"]["decay"])
    ):
        raise Stage3ContractError("Stage3 best checkpoint EMA policy drifted")
    approval = _mapping(
        provenance.get("stage3_approval"), field="Stage3 checkpoint approval"
    )
    if approval.get("sha256") != paths.approval.approval_sha256:
        raise Stage3ContractError("Stage3 checkpoint approval hash is stale")
    if provenance.get("bindings") != paths.approval.bindings:
        raise Stage3ContractError(
            "Stage3 checkpoint frozen bindings differ from approval"
        )
    parent_binding = _mapping(
        provenance.get("parent_checkpoint"), field="Stage3 checkpoint parent"
    )
    relation_binding = _mapping(
        provenance.get("relation_supervision"), field="Stage3 relation provenance"
    )
    expected_hashes = {
        "parent_checkpoint.sha256": (
            parent_binding.get("sha256"),
            paths.approval.bindings["stage1_checkpoint"]["sha256"],
        ),
        "effect_profiles_sha256": (
            provenance.get("effect_profiles_sha256"),
            paths.approval.bindings["skill_effect_profiles"]["sha256"],
        ),
        "pair_prior_sha256": (
            provenance.get("pair_prior_sha256"),
            paths.approval.bindings["pair_prior"]["sha256"],
        ),
        "global_priority_sha256": (
            provenance.get("global_priority_sha256"),
            paths.approval.bindings["global_priority"]["sha256"],
        ),
        "relation_train_sha256": (
            relation_binding.get("train_sha256"),
            paths.approval.bindings["relation_train"]["sha256"],
        ),
        "relation_val_sha256": (
            relation_binding.get("validation_sha256"),
            paths.approval.bindings["relation_val"]["sha256"],
        ),
    }
    hash_drift = {
        name: {"checkpoint": actual, "approved": expected}
        for name, (actual, expected) in expected_hashes.items()
        if actual != expected
    }
    if hash_drift:
        raise Stage3ContractError(
            f"Stage3 checkpoint provenance hash drift: {hash_drift}"
        )
    model.load_state_dict(source, strict=True)
    if load_frozen_thresholds:
        thresholds = _mapping(load_json(paths.thresholds), field="planner thresholds")
        expected_extension_sha = (
            None
            if extension_binding is None
            else _mapping(extension_binding, field="Stage3 best extension provenance")[
                "sha256"
            ]
        )
        if (
            thresholds.get("schema_version") != THRESHOLD_SCHEMA
            or thresholds.get("protocol_id") != PROTOCOL_ID
            or thresholds.get("frozen") is not True
            or thresholds.get("skills") != list(SKILLS)
            or thresholds.get("stage3_approval_sha256")
            != paths.approval.approval_sha256
            or thresholds.get("checkpoint_sha256") != sha256_file(checkpoint_path)
            or thresholds.get("primary_val_manifest_sha256")
            != sha256_file(paths.val_manifest)
            or thresholds.get("calibration_runs") != 1
            or thresholds.get("mio100_rows_read") != 0
            or thresholds.get("stage3_extension_authorization_sha256")
            != expected_extension_sha
        ):
            raise Stage3ContractError("invalid frozen Stage3 presence thresholds")
        values = _mapping(thresholds.get("thresholds"), field="threshold values")
        if set(values) != set(SKILLS) or thresholds.get("search_grid") != [
            value / 100.0 for value in range(20, 81, 2)
        ]:
            raise Stage3ContractError("frozen threshold skill/grid schema drifted")
        ordered_thresholds = [float(values[skill]) for skill in SKILLS]
        if any(
            value not in {item / 100.0 for item in range(20, 81, 2)}
            for value in ordered_thresholds
        ):
            raise Stage3ContractError("frozen threshold lies outside the locked grid")
        model.set_presence_thresholds(ordered_thresholds)
    set_stage3_trainability(model)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def probe_stage3_validation_vram(
    model: GraphRestore,
    *,
    optimizer: torch.optim.Optimizer,
    ema: Stage3PlannerEMA,
    device: torch.device,
    image_size: int = 2040,
    max_rounds: int = 3,
    maximum_reserved_fraction: float = 0.90,
) -> Stage3ValidationVRAMGate:
    """Exercise both legal full-resolution Stage3 validation topologies."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise Stage3ContractError("Stage3 validation VRAM gate requires CUDA")
    if image_size != 2040 or max_rounds != 3 or maximum_reserved_fraction != 0.90:
        raise Stage3ContractError("Stage3 validation VRAM gate contract drifted")
    if not isinstance(ema, Stage3PlannerEMA):
        raise Stage3ContractError("Stage3 validation gate requires Stage3PlannerEMA")
    if optimizer.state:
        raise Stage3ContractError(
            "Stage3 validation gate requires a pristine step0 optimizer"
        )
    rng = capture_rng_state()
    module_modes = {name: module.training for name, module in model.named_modules()}
    original_mode = model.compiler.mode
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    if total_memory <= 0:
        raise Stage3ContractError("Stage3 validation VRAM gate saw invalid GPU memory")
    image: Tensor | None = None
    target: Tensor | None = None
    output: Any = None
    topology_results: list[Stage3ValidationVRAMTopology] = []
    resident_optimizer_state_bytes = 0
    resident_optimizer_state_entries = 0
    resident_ema_bytes = sum(
        value.numel() * value.element_size() for value in ema.shadow.values()
    )
    try:
        model.eval()
        # AdamW moments are lazy.  Materialize a conservative full planner-state
        # residency without taking a scientific optimizer step, retain it across
        # both inference topologies, and erase it before the step0 anchor.
        for group in optimizer.param_groups:
            parameters = group.get("params")
            if not isinstance(parameters, list):
                raise Stage3ContractError(
                    "Stage3 validation gate optimizer parameters drifted"
                )
            for parameter in parameters:
                if (
                    not isinstance(parameter, nn.Parameter)
                    or not parameter.is_floating_point()
                ):
                    raise Stage3ContractError(
                        "Stage3 validation gate optimizer parameter is invalid"
                    )
                state = optimizer.state[parameter]
                if state:
                    raise Stage3ContractError(
                        "Stage3 validation gate optimizer was not pristine"
                    )
                state["step"] = torch.zeros(
                    (), dtype=torch.float32, device=parameter.device
                )
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
                resident_optimizer_state_entries += 1
                resident_optimizer_state_bytes += sum(
                    value.numel() * value.element_size()
                    for value in state.values()
                    if torch.is_tensor(value)
                )

        image = torch.rand(1, 3, image_size, image_size, device=device)
        target = torch.rand_like(image)
        thresholds = image.new_ones(len(SKILLS))
        thresholds[:3] = 0.0
        from src.net.graphrestore import GraphRestoreOutput

        for compiler_mode, expected_rounds in (
            ("forced_total_order", max_rounds),
            ("parallel_only", 1),
        ):
            model.compiler.mode = compiler_mode
            output = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    image,
                    presence_thresholds=thresholds,
                    max_rounds=max_rounds,
                    return_trace=True,
                )
            if not isinstance(output, GraphRestoreOutput):
                raise Stage3ContractError("Stage3 validation gate requires a trace")
            if len(output.compiled_graphs) != 1:
                raise Stage3ContractError(
                    "Stage3 validation gate requires one compiled sample graph"
                )
            graph = output.compiled_graphs[0]
            active_skill_count = len(graph.active_skills)
            completed_rounds = len(output.trace)
            active_by_round = tuple(
                int(trace.active_mask[0].sum().item()) for trace in output.trace
            )
            if active_skill_count != 3 or completed_rounds != expected_rounds:
                raise Stage3ContractError(
                    f"Stage3 {compiler_mode} validation gate topology drifted"
                )
            if compiler_mode == "forced_total_order":
                topology_valid = (
                    len(graph.levels) == max_rounds
                    and all(len(level) == 1 for level in graph.levels)
                    and active_by_round == (1, 1, 1)
                )
            else:
                topology_valid = (
                    len(graph.levels) == 1
                    and len(graph.levels[0]) == 3
                    and active_by_round == (3,)
                )
            if not topology_valid:
                raise Stage3ContractError(
                    f"Stage3 {compiler_mode} validation gate execution drifted"
                )
            finite = bool(torch.isfinite(output.final).all())
            if not finite:
                raise FloatingPointError(
                    f"non-finite Stage3 {compiler_mode} validation gate output"
                )
            metric = official_psnr_ssim(
                output.final.detach().float().cpu(),
                target.detach().float().cpu(),
                quantize=True,
            )
            metric_psnr = float(metric.psnr.reshape(-1)[0])
            metric_ssim = float(metric.ssim.reshape(-1)[0])
            if not math.isfinite(metric_psnr) or not math.isfinite(metric_ssim):
                raise FloatingPointError(
                    f"non-finite Stage3 {compiler_mode} validation gate metric"
                )
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_reserved(device))
            if peak < 0:
                raise Stage3ContractError(
                    "Stage3 validation VRAM gate saw negative reserved memory"
                )
            fraction = peak / total_memory
            topology_results.append(
                Stage3ValidationVRAMTopology(
                    compiler_mode=compiler_mode,
                    active_skill_count=active_skill_count,
                    completed_rounds=completed_rounds,
                    active_skill_counts_by_round=active_by_round,
                    metric_psnr=metric_psnr,
                    metric_ssim=metric_ssim,
                    peak_reserved_bytes=peak,
                    peak_reserved_fraction=fraction,
                    finite=finite,
                    passed=fraction <= maximum_reserved_fraction,
                )
            )
            output = None

        peak = max(value.peak_reserved_bytes for value in topology_results)
        fraction = max(value.peak_reserved_fraction for value in topology_results)
        gate = Stage3ValidationVRAMGate(
            schema_version="graphrestore-stage3-validation-vram-gate-v1",
            image_size=image_size,
            max_rounds=max_rounds,
            completed_rounds=max_rounds,
            topologies=tuple(topology_results),
            peak_reserved_bytes=peak,
            peak_reserved_fraction=fraction,
            maximum_peak_reserved_fraction=maximum_reserved_fraction,
            resident_optimizer_state_entries=resident_optimizer_state_entries,
            resident_optimizer_state_bytes=resident_optimizer_state_bytes,
            resident_ema_bytes=resident_ema_bytes,
            optimizer_state_empty_after=True,
            finite=all(value.finite for value in topology_results),
            passed=all(value.passed for value in topology_results),
        )
        if not gate.passed:
            raise Stage3ContractError(
                "Stage3 2040-square validation topology peak "
                f"{fraction:.4f} exceeds 0.90"
            )
        return gate
    finally:
        model.compiler.mode = original_mode
        for name, module in model.named_modules():
            module.training = module_modes[name]
        image = target = output = None
        optimizer.state.clear()
        restore_rng_state(rng)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def validate_stage3_validation_vram_evidence(value: object) -> dict[str, Any]:
    """Normalize and fail closed on frozen run-contract VRAM evidence."""

    evidence = _mapping(value, field="Stage3 validation_vram_gate")
    required = {
        "schema_version",
        "image_size",
        "max_rounds",
        "completed_rounds",
        "topologies",
        "peak_reserved_bytes",
        "peak_reserved_fraction",
        "maximum_peak_reserved_fraction",
        "resident_optimizer_state_entries",
        "resident_optimizer_state_bytes",
        "resident_ema_bytes",
        "optimizer_state_empty_after",
        "finite",
        "passed",
    }
    if set(evidence) != required:
        raise Stage3ContractError("Stage3 validation VRAM evidence fields drifted")
    if (
        evidence.get("schema_version") != "graphrestore-stage3-validation-vram-gate-v1"
        or evidence.get("image_size") != 2040
        or evidence.get("max_rounds") != 3
        or evidence.get("completed_rounds") != 3
        or evidence.get("maximum_peak_reserved_fraction") != 0.90
        or evidence.get("optimizer_state_empty_after") is not True
        or evidence.get("finite") is not True
        or evidence.get("passed") is not True
    ):
        raise Stage3ContractError("Stage3 validation VRAM evidence contract drifted")
    for field in (
        "peak_reserved_bytes",
        "resident_optimizer_state_entries",
        "resident_optimizer_state_bytes",
        "resident_ema_bytes",
    ):
        item = evidence.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise Stage3ContractError(
                f"Stage3 validation VRAM evidence {field} is invalid"
            )
    peak_fraction = evidence.get("peak_reserved_fraction")
    if (
        isinstance(peak_fraction, bool)
        or not isinstance(peak_fraction, (int, float))
        or not math.isfinite(float(peak_fraction))
        or not 0.0 < float(peak_fraction) <= 0.90
    ):
        raise Stage3ContractError("Stage3 validation VRAM evidence peak is invalid")
    topologies = evidence.get("topologies")
    if not isinstance(topologies, (list, tuple)) or len(topologies) != 2:
        raise Stage3ContractError("Stage3 validation VRAM topology evidence drifted")
    normalized_topologies: list[dict[str, Any]] = []
    expected = (
        ("forced_total_order", 3, (1, 1, 1)),
        ("parallel_only", 1, (3,)),
    )
    for row, (mode, rounds, active_by_round) in zip(topologies, expected, strict=True):
        topology = _mapping(row, field="Stage3 validation VRAM topology")
        if (
            topology.get("compiler_mode") != mode
            or topology.get("active_skill_count") != 3
            or topology.get("completed_rounds") != rounds
            or tuple(topology.get("active_skill_counts_by_round", ()))
            != active_by_round
            or topology.get("finite") is not True
            or topology.get("passed") is not True
        ):
            raise Stage3ContractError(f"Stage3 validation VRAM {mode} topology drifted")
        normalized = dict(topology)
        normalized["active_skill_counts_by_round"] = list(active_by_round)
        for field in ("metric_psnr", "metric_ssim", "peak_reserved_fraction"):
            item = topology.get(field)
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise Stage3ContractError(
                    f"Stage3 validation VRAM {mode} {field} is non-finite"
                )
        row_peak = topology.get("peak_reserved_bytes")
        if isinstance(row_peak, bool) or not isinstance(row_peak, int) or row_peak <= 0:
            raise Stage3ContractError(f"Stage3 validation VRAM {mode} peak is invalid")
        if not 0.0 < float(topology["peak_reserved_fraction"]) <= 0.90:
            raise Stage3ContractError(
                f"Stage3 validation VRAM {mode} peak exceeds 0.90"
            )
        normalized_topologies.append(normalized)
    normalized_evidence = dict(evidence)
    normalized_evidence["topologies"] = normalized_topologies
    if normalized_evidence["peak_reserved_bytes"] != max(
        row["peak_reserved_bytes"] for row in normalized_topologies
    ) or normalized_evidence["peak_reserved_fraction"] != max(
        row["peak_reserved_fraction"] for row in normalized_topologies
    ):
        raise Stage3ContractError("Stage3 validation VRAM aggregate peak drifted")
    return normalized_evidence


def _synthetic_probe_batch(
    batch: int,
    device: torch.device,
    *,
    model: GraphRestore | None = None,
    include_teacher_intermediate: bool = False,
) -> Stage3SupervisionBatch:
    image = torch.rand(batch, 3, 192, 192, device=device)
    current = image.clone()
    presence = torch.zeros(batch, 8, device=device)
    presence[:, :2] = 1.0
    guards = torch.rand(batch, 8, 48, 48, device=device) * presence[:, :, None, None]
    relation_targets = torch.full((batch, 28), -2, device=device, dtype=torch.long)
    relation_targets[:, 0] = 0
    relation_weights = torch.zeros(batch, 28, device=device)
    relation_weights[:, 0] = 1.0
    dense_ids = torch.tensor(
        [skill in DENSE_GUARD_SKILLS for skill in SKILLS], device=device
    )
    state_kinds = ["group_a_pair"] * batch
    model_intermediate_count = 0
    if include_teacher_intermediate:
        if model is None:
            raise Stage3ContractError(
                "Stage3 teacher probe requires the frozen executor model"
            )
        teacher_skill_id = 0
        current[0] = _teacher_model_intermediate(
            model,
            image[0],
            guards[0],
            teacher_skill_id,
        )
        presence[0, teacher_skill_id] = 0.0
        guards[0, teacher_skill_id] = 0.0
        relation_weights[0].zero_()
        relation_targets[0].fill_(-2)
        state_kinds[0] = "model_generated_intermediate"
        model_intermediate_count = 1
    return Stage3SupervisionBatch(
        x0=image,
        current=current,
        presence_targets=presence,
        guard_targets=guards,
        global_severity_targets=guards.mean(dim=(-2, -1)),
        dense_skill_mask=presence.bool() & dense_ids[None, :],
        global_skill_mask=presence.bool() & ~dense_ids[None, :],
        absent_skill_mask=~presence.bool(),
        stop_targets=torch.zeros(batch, 1, device=device),
        relation_targets=relation_targets,
        relation_weights=relation_weights,
        relation_ambiguous_mask=torch.zeros(batch, 28, device=device, dtype=torch.bool),
        round_values=torch.zeros(batch, device=device),
        sample_ids=tuple(f"probe-{index}" for index in range(batch)),
        state_kinds=tuple(state_kinds),
        model_intermediate_count=model_intermediate_count,
    )


def select_stage3_micro_batch(
    model: GraphRestore,
    *,
    device: torch.device,
    candidates: Sequence[int] = (8, 4, 2, 1),
    required_passes: int = 10,
    maximum_reserved_fraction: float = 0.90,
) -> tuple[int, tuple[Stage3MicroBatchTrial, ...]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Stage3ContractError("formal Stage3 micro-batch probe requires CUDA")
    if tuple(candidates) not in {(8, 4, 2, 1), (8,), (4,), (2,), (1,)}:
        raise Stage3ContractError("Stage3 micro-batch candidates drifted")
    if required_passes != 10 or maximum_reserved_fraction != 0.90:
        raise Stage3ContractError(
            "Stage3 probe must use ten optimizer steps and <=90% reserved"
        )
    if any(8 % candidate != 0 for candidate in candidates):
        raise Stage3ContractError(
            "Stage3 probe candidates must divide effective batch 8"
        )
    rng = capture_rng_state()
    pristine_model = {
        name: value.detach().clone()
        for name, value in unwrap_model(model).state_dict().items()
    }
    trials: list[Stage3MicroBatchTrial] = []
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    _stage3_train_mode(model)
    try:
        for candidate in candidates:
            unwrap_model(model).load_state_dict(pristine_model, strict=True)
            restore_rng_state(rng)
            set_stage3_trainability(model)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            completed = 0
            error: str | None = None
            throughput = 0.0
            peak = 0
            fraction = 0.0
            optimizer = scheduler = ema = None
            micro_batches: list[Stage3SupervisionBatch] = []
            started = time.perf_counter()
            try:
                optimizer = build_stage3_optimizer(model)
                scheduler = WarmupCosineScheduler(
                    optimizer,
                    warmup_steps=500,
                    max_steps=12_000,
                    min_lr=2.0e-6,
                )
                ema = Stage3PlannerEMA(model, decay=0.9999)
                accumulation_steps = 8 // candidate
                for optimizer_step in range(required_passes):
                    micro_batches = [
                        _synthetic_probe_batch(
                            candidate,
                            device,
                            model=model,
                            include_teacher_intermediate=(
                                optimizer_step == 0 and micro_batch_index == 0
                            ),
                        )
                        for micro_batch_index in range(accumulation_steps)
                    ]
                    train_stage3_optimizer_step(
                        model,
                        micro_batches,
                        optimizer,
                        scheduler,
                        ema,
                        device=device,
                        use_bf16=True,
                        audit_gradients=True,
                    )
                    completed += 1
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                throughput = 8 * completed / max(elapsed, 1e-9)
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                passed = (
                    completed == required_passes
                    and fraction <= maximum_reserved_fraction
                )
                if not passed:
                    error = f"peak reserved fraction {fraction:.4f} exceeds 0.90"
                for name, value in unwrap_model(model).state_dict().items():
                    if not name.startswith("planner.") and not torch.equal(
                        value, pristine_model[name]
                    ):
                        raise Stage3ContractError(
                            "Stage3 micro probe changed frozen executor state"
                        )
            except torch.cuda.OutOfMemoryError as exc:
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                passed = False
                error = f"CUDA OOM: {exc}"
            finally:
                model.zero_grad(set_to_none=True)
                micro_batches.clear()
                optimizer = scheduler = ema = None
                unwrap_model(model).load_state_dict(pristine_model, strict=True)
                torch.cuda.empty_cache()
            trials.append(
                Stage3MicroBatchTrial(
                    micro_batch=candidate,
                    passed=passed,
                    completed_optimizer_steps=completed,
                    images_per_second=throughput,
                    peak_reserved_bytes=peak,
                    peak_reserved_fraction=fraction,
                    error=error,
                )
            )
    finally:
        unwrap_model(model).load_state_dict(pristine_model, strict=True)
        restore_rng_state(rng)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    accepted = [trial for trial in trials if trial.passed]
    if not accepted:
        raise Stage3ContractError(
            "no Stage3 micro batch passed the ten-optimizer-step <=90% gate"
        )
    winner = max(
        accepted, key=lambda trial: (trial.images_per_second, trial.micro_batch)
    )
    return winner.micro_batch, tuple(trials)


def validation_score(summary: Mapping[str, Any], step: int) -> ValidationScore:
    restoration = _mapping(summary.get("restoration"), field="validation.restoration")
    single = _mapping(restoration.get("single"), field="validation.single")
    group_a = _mapping(restoration.get("group_a"), field="validation.group_a")
    return ValidationScore(
        group_a_psnr=float(group_a["psnr"]),
        group_a_ssim=float(group_a["ssim"]),
        single_psnr=float(single["psnr"]),
        single_ssim=float(single["ssim"]),
        step=step,
    )


_STAGE3_FINALIZATION_BINDINGS = frozenset(
    {
        "best_checkpoint",
        "abandoned_last_checkpoint",
        "selected_validation",
        "thresholds",
        "selected_validation_calibrated",
        "report",
        "finalization_authorization",
        "historical_extension_authorization",
        "stage3_approval",
        "approval_required",
        "stage1_checkpoint",
        "run_contract",
        "stage3_config",
        "primary_val_manifest",
        "relation_val",
        "pair_prior",
        "global_priority",
        "calibration_history",
    }
)


def _finalization_output_binding(
    value: object,
    *,
    field: str,
    expected_path: Path | None = None,
) -> dict[str, str]:
    binding = _mapping(value, field=field)
    if set(binding) != {"path", "sha256"}:
        raise Stage3ContractError(f"{field} binding fields drifted")
    raw_path, digest = binding.get("path"), binding.get("sha256")
    if not isinstance(raw_path, str) or not is_sha256(digest):
        raise Stage3ContractError(f"{field} binding is invalid")
    path = Path(raw_path).resolve(strict=False)
    if expected_path is not None and path != expected_path.resolve(strict=False):
        raise Stage3ContractError(f"{field} path drifted")
    if not path.is_file() or sha256_file(path) != digest:
        raise Stage3ContractError(f"{field} physical hash drifted")
    return {"path": str(path), "sha256": str(digest)}


def _required_finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Stage3ContractError(f"{field} is non-finite")
    return float(value)


def validate_stage3_finalization_outputs(
    project_root: str | Path,
    *,
    finalization_authorization_sha256: str,
    historical_extension_authorization_sha256: str,
) -> dict[str, Any]:
    """Validate the finalize-only Stage3 closure using CPU/file I/O only.

    This is shared by the dedicated finalizer, orchestrator and Stage4 parent
    preflight so the three entrypoints cannot silently diverge.
    """

    if not is_sha256(finalization_authorization_sha256) or not is_sha256(
        historical_extension_authorization_sha256
    ):
        raise Stage3ContractError("Stage3 finalization authorization SHA is invalid")
    root = Path(project_root).resolve()
    checkpoint_dir = root / "artifacts/checkpoints/stage3"
    complete_path = checkpoint_dir / "complete.json"
    threshold_path = root / "artifacts/planner_thresholds.json"
    calibrated_path = checkpoint_dir / "selected_validation_calibrated.json"
    report_path = root / "reports/STAGE3_PLANNER_GUARD.md"
    original_path = checkpoint_dir / "selected_validation.json"
    complete = _mapping(load_json(complete_path), field="Stage3 complete.json")
    if (
        complete.get("schema_version") != STAGE3_SCHEMA
        or complete.get("kind") != "stage3_finalize_only"
        or complete.get("protocol_id") != PROTOCOL_ID
        or complete.get("step") != STAGE3_BASE_TARGET_STEP
        or complete.get("optimizer_created") is not False
        or complete.get("scheduler_created") is not False
        or complete.get("train_loader_created") is not False
        or complete.get("checkpoint_written") is not False
        or complete.get("optimizer_steps_executed") != 0
        or complete.get("checkpoint_writes") != 0
        or complete.get("sampler_steps_advanced") != 0
        or complete.get("abandoned_last_checkpoint_role")
        != "abandoned_unselected_extension_state"
        or complete.get("stage4_parent_role") != "only_stage3_parent"
        or complete.get("threshold_calibration_runs") != 1
        or complete.get("post_calibration_diagnostic_runs") != 1
        or complete.get("mio100_rows_read") != 0
        or complete.get("group_b_rows_read") != 0
        or complete.get("group_c_rows_read") != 0
    ):
        raise Stage3ContractError("Stage3 finalize-only completion fields drifted")
    bindings = _mapping(complete.get("bindings"), field="Stage3 complete bindings")
    if set(bindings) != _STAGE3_FINALIZATION_BINDINGS:
        raise Stage3ContractError("Stage3 finalize-only binding set drifted")
    expected_paths = {
        "best_checkpoint": checkpoint_dir / "best_ema.pth",
        "selected_validation": original_path,
        "thresholds": threshold_path,
        "selected_validation_calibrated": calibrated_path,
        "report": report_path,
        "stage3_config": root / "configs/stage3_planner.yaml",
        "stage3_approval": root / "artifacts/approvals/STAGE3_APPROVED.json",
        "approval_required": root / "artifacts/approvals/STAGE3_APPROVAL_REQUIRED.json",
        "stage1_checkpoint": root / "artifacts/checkpoints/stage1/best_ema.pth",
        "relation_val": root
        / "artifacts/interaction_labels/group_a_relations_val.jsonl",
        "pair_prior": root / "artifacts/interaction_labels/pair_prior.json",
        "global_priority": root / "artifacts/interaction_labels/global_priority.json",
        "calibration_history": root / "artifacts/metrics/calibration_history.csv",
    }
    verified: dict[str, dict[str, str]] = {}
    for logical in sorted(bindings):
        verified[logical] = _finalization_output_binding(
            bindings[logical],
            field=f"complete.bindings.{logical}",
            expected_path=expected_paths.get(logical),
        )
    if (
        verified["finalization_authorization"]["sha256"]
        != finalization_authorization_sha256
        or verified["historical_extension_authorization"]["sha256"]
        != historical_extension_authorization_sha256
        or complete.get("best_checkpoint_sha256")
        != verified["best_checkpoint"]["sha256"]
        or complete.get("thresholds_sha256") != verified["thresholds"]["sha256"]
    ):
        raise Stage3ContractError("Stage3 completion identity binding drifted")
    live_last = checkpoint_dir / "last.pth"
    live_run_contract = checkpoint_dir / "run_contract.json"
    if (
        not live_run_contract.is_file()
        or sha256_file(live_run_contract) != verified["run_contract"]["sha256"]
    ):
        raise Stage3ContractError("live Stage3 run contract drifted")
    if (
        not live_last.is_file()
        or sha256_file(live_last) != verified["abandoned_last_checkpoint"]["sha256"]
    ):
        raise Stage3ContractError("live abandoned Stage3 last checkpoint drifted")
    abandoned = torch.load(live_last, map_location="cpu", weights_only=False)
    if (
        not isinstance(abandoned, Mapping)
        or abandoned.get("stage") != "stage3"
        or abandoned.get("model_role") != "raw_training_state"
        or abandoned.get("resumable") is not True
        or abandoned.get("step") != 14_000
        or abandoned.get("pending_validation_step") != 14_000
    ):
        raise Stage3ContractError(
            "abandoned Stage3 last checkpoint role/step/pending drifted"
        )

    thresholds = _mapping(load_json(threshold_path), field="planner thresholds")
    if (
        thresholds.get("schema_version") != THRESHOLD_SCHEMA
        or thresholds.get("protocol_id") != PROTOCOL_ID
        or thresholds.get("source") != "primary_val_presence_f1_only"
        or thresholds.get("frozen") is not True
        or thresholds.get("skills") != list(SKILLS)
        or thresholds.get("baseline_threshold") != 0.50
        or thresholds.get("search_grid")
        != [value / 100.0 for value in range(20, 81, 2)]
        or thresholds.get("tie_break") != THRESHOLD_TIE_BREAK
        or thresholds.get("numerical_tolerance") != THRESHOLD_F1_TOLERANCE
        or thresholds.get("calibration_runs") != 1
        or thresholds.get("mio100_rows_read") != 0
        or thresholds.get("group_b_rows_read") != 0
        or thresholds.get("group_c_rows_read") != 0
        or thresholds.get("checkpoint_sha256") != verified["best_checkpoint"]["sha256"]
        or thresholds.get("primary_val_manifest_sha256")
        != verified["primary_val_manifest"]["sha256"]
        or thresholds.get("stage3_finalization_authorization_sha256")
        != finalization_authorization_sha256
        or thresholds.get("stage3_extension_authorization_sha256")
        != historical_extension_authorization_sha256
    ):
        raise Stage3ContractError("frozen Stage3 threshold contract drifted")
    values = _mapping(thresholds.get("thresholds"), field="threshold values")
    per_skill_f1 = _mapping(
        thresholds.get("per_skill_f1"), field="threshold per-skill F1"
    )
    per_skill_metrics = _mapping(
        thresholds.get("per_skill_metrics"), field="threshold per-skill metrics"
    )
    if (
        set(values) != set(SKILLS)
        or set(per_skill_f1) != set(SKILLS)
        or set(per_skill_metrics) != set(SKILLS)
    ):
        raise Stage3ContractError("frozen Stage3 threshold skill set drifted")
    grid = {value / 100.0 for value in range(20, 81, 2)}
    baseline_f1_values: list[float] = []
    calibrated_f1_values: list[float] = []
    for skill in SKILLS:
        selected_threshold = _required_finite_number(
            values[skill], field=f"thresholds.{skill}"
        )
        if selected_threshold not in grid:
            raise Stage3ContractError(f"{skill}: calibrated threshold left grid")
        metrics = _mapping(per_skill_metrics[skill], field=f"metrics.{skill}")
        if set(metrics) != {"baseline", "calibrated"}:
            raise Stage3ContractError(f"{skill}: threshold metric roles drifted")
        baseline = _mapping(metrics["baseline"], field=f"baseline.{skill}")
        calibrated = _mapping(metrics["calibrated"], field=f"calibrated.{skill}")
        expected_metric_keys = {"threshold", "precision", "recall", "f1"}
        if (
            set(baseline) != expected_metric_keys
            or set(calibrated) != expected_metric_keys
        ):
            raise Stage3ContractError(f"{skill}: threshold metric fields drifted")
        before = _required_finite_number(baseline["f1"], field=f"baseline.{skill}.f1")
        after = _required_finite_number(
            calibrated["f1"], field=f"calibrated.{skill}.f1"
        )
        for role, row in (("baseline", baseline), ("calibrated", calibrated)):
            for metric in ("precision", "recall"):
                value = _required_finite_number(
                    row[metric], field=f"{role}.{skill}.{metric}"
                )
                if not 0.0 <= value <= 1.0:
                    raise Stage3ContractError(f"{role}.{skill}.{metric} escaped [0,1]")
        if (
            baseline["threshold"] != 0.50
            or calibrated["threshold"] != selected_threshold
            or after + THRESHOLD_F1_TOLERANCE < before
            or float(per_skill_f1[skill]) != after
        ):
            raise Stage3ContractError(f"{skill}: calibrated F1 contract drifted")
        baseline_f1_values.append(before)
        calibrated_f1_values.append(after)
    macro_before = _required_finite_number(
        thresholds.get("macro_f1_before"), field="macro_f1_before"
    )
    macro_after = _required_finite_number(
        thresholds.get("macro_f1_after"), field="macro_f1_after"
    )
    if macro_after + THRESHOLD_F1_TOLERANCE < macro_before:
        raise Stage3ContractError("calibrated Stage3 macro F1 regressed")
    if macro_before != math.fsum(baseline_f1_values) / len(
        SKILLS
    ) or macro_after != math.fsum(calibrated_f1_values) / len(SKILLS):
        raise Stage3ContractError("frozen Stage3 macro F1 aggregation drifted")
    calibration_code = _finalization_output_binding(
        thresholds.get("calibration_code"), field="calibration_code"
    )
    if calibration_code["path"] != str(Path(__file__).resolve()):
        raise Stage3ContractError("Stage3 calibration code path drifted")

    diagnostic = _mapping(
        load_json(calibrated_path), field="selected_validation_calibrated.json"
    )
    sources = _mapping(diagnostic.get("sources"), field="diagnostic sources")
    restoration = _mapping(
        diagnostic.get("restoration"), field="diagnostic restoration"
    )
    planner = _mapping(diagnostic.get("planner"), field="diagnostic planner")
    relation = _mapping(diagnostic.get("relation"), field="diagnostic relation")
    guard = _mapping(diagnostic.get("guard"), field="diagnostic guard")
    graph = _mapping(diagnostic.get("graph"), field="diagnostic graph")
    if (
        diagnostic.get("schema_version") != STAGE3_SCHEMA
        or diagnostic.get("protocol_id") != PROTOCOL_ID
        or diagnostic.get("diagnostic_role")
        != "post_calibration_non_selection_diagnostic"
        or diagnostic.get("selected_step") != STAGE3_BASE_TARGET_STEP
        or diagnostic.get("selected_checkpoint_sha256")
        != verified["best_checkpoint"]["sha256"]
        or diagnostic.get("thresholds_sha256") != verified["thresholds"]["sha256"]
        or diagnostic.get("stage3_finalization_authorization_sha256")
        != finalization_authorization_sha256
        or diagnostic.get("post_calibration_diagnostic_runs") != 1
        or sources.get("mio100_rows_read") != 0
        or diagnostic.get("group_b_rows_read") != 0
        or diagnostic.get("group_c_rows_read") != 0
        or planner.get("sample_count") != 1600
        or graph.get("sample_count") != 1600
    ):
        raise Stage3ContractError("post-calibration Stage3 diagnostic contract drifted")
    expected_threshold_values = [float(values[skill]) for skill in SKILLS]
    if diagnostic.get(
        "checkpoint_presence_threshold"
    ) != expected_threshold_values or diagnostic.get("presence_thresholds") != {
        skill: float(values[skill]) for skill in SKILLS
    }:
        raise Stage3ContractError("post-calibration diagnostic thresholds drifted")
    for group in ("single", "group_a"):
        row = _mapping(restoration.get(group), field=f"diagnostic {group}")
        if row.get("count") != 800 or row.get("task_count") != 8:
            raise Stage3ContractError(f"diagnostic {group} coverage drifted")
        _required_finite_number(row.get("psnr"), field=f"{group}.psnr")
        _required_finite_number(row.get("ssim"), field=f"{group}.ssim")
    _required_finite_number(planner.get("macro_f1"), field="planner.macro_f1")
    _required_finite_number(
        planner.get("activation_rate"), field="planner.activation_rate"
    )
    if (
        planner.get("activation_rate_definition")
        != "fraction_of_sample_skill_slots_predicted_active"
    ):
        raise Stage3ContractError("planner activation-rate definition drifted")
    planner_per_skill = _mapping(
        planner.get("per_skill"), field="diagnostic planner per_skill"
    )
    if set(planner_per_skill) != set(SKILLS):
        raise Stage3ContractError("diagnostic planner skill set drifted")
    for skill in SKILLS:
        row = _mapping(planner_per_skill[skill], field=f"planner.{skill}")
        if row.get("threshold") != float(values[skill]):
            raise Stage3ContractError(f"planner.{skill} threshold drifted")
        for metric in ("precision", "recall", "f1", "activation_rate"):
            _required_finite_number(row.get(metric), field=f"planner.{skill}.{metric}")
    for field in (
        "relation_accuracy_non_ambiguous",
        "parallel_precision_non_ambiguous",
        "parallel_recall_non_ambiguous",
    ):
        _required_finite_number(relation.get(field), field=f"relation.{field}")
    learned_raw = _mapping(
        relation.get("learned_raw"), field="learned raw relation metrics"
    )
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        _required_finite_number(
            learned_raw.get(metric), field=f"relation.learned_raw.{metric}"
        )
    if learned_raw.get("accuracy") != relation.get("relation_accuracy_non_ambiguous"):
        raise Stage3ContractError("learned raw relation accuracy drifted")
    baseline = _mapping(
        relation.get("cpu_baseline_audit"), field="relation CPU baseline audit"
    )
    if (
        baseline.get("n_total") != 800
        or baseline.get("n_non_ambiguous") != 735
        or baseline.get("n_ambiguous_excluded") != 65
        or baseline.get("mio100_rows_read") != 0
        or baseline.get("group_b_rows_read") != 0
        or baseline.get("group_c_rows_read") != 0
    ):
        raise Stage3ContractError("relation CPU baseline coverage drifted")
    _required_finite_number(
        baseline.get("learned_raw_accuracy"), field="learned relation accuracy"
    )
    for name in ("always_parallel", "per_pair_majority_prior"):
        row = _mapping(baseline.get(name), field=f"relation baseline {name}")
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            _required_finite_number(row.get(metric), field=f"{name}.{metric}")
    for field in (
        "guard_spearman_rain",
        "guard_mae_rain",
        "guard_spearman_haze",
        "guard_mae_haze",
    ):
        _required_finite_number(guard.get(field), field=f"guard.{field}")
    for field in ("mean_program_levels", "sample_stop_rate"):
        _required_finite_number(graph.get(field), field=f"graph.{field}")
    if (
        graph.get("sample_stop_rate_definition")
        != "fraction_of_samples_with_stopped_mask_in_any_formal_inference_round"
    ):
        raise Stage3ContractError("graph STOP-rate definition drifted")
    if graph.get("post_compiler_cycle_rate") != 0.0:
        raise Stage3ContractError("post-calibration graph contains a cycle")

    return {
        "complete": {
            "path": str(complete_path.resolve()),
            "sha256": sha256_file(complete_path),
        },
        "thresholds": verified["thresholds"],
        "selected_validation_calibrated": verified["selected_validation_calibrated"],
        "report": verified["report"],
        "best_checkpoint": verified["best_checkpoint"],
        "payload": dict(complete),
    }


CALIBRATION_COLUMNS = (
    "step",
    "single_psnr",
    "single_ssim",
    "group_a_psnr",
    "group_a_ssim",
    "planner_macro_f1",
    "relation_accuracy",
    "parallel_precision",
    "parallel_recall",
    "pre_cycle_rate",
    "dropped_edge_rate",
    "guard_spearman_rain",
    "guard_spearman_haze",
    "guard_mae_rain",
    "guard_mae_haze",
    "guard_std_rain",
    "guard_std_haze",
    "guard_high_frac_rain",
    "guard_high_frac_haze",
    "clean_misuse_psnr",
    "clean_misuse_ssim",
    "clean_misuse_residual_norm",
    "wrong_skill_identity_psnr",
    "wrong_skill_identity_ssim",
    "wrong_skill_residual_norm",
    "reentry_request_rate",
    "unexpected_skill_activation_rate",
    "mean_program_levels",
)


def calibration_history_row(summary: Mapping[str, Any], step: int) -> dict[str, Any]:
    restoration = _mapping(summary["restoration"], field="restoration")
    single = _mapping(restoration["single"], field="single")
    group_a = _mapping(restoration["group_a"], field="group_a")
    planner = _mapping(summary["planner"], field="planner")
    relation = _mapping(summary["relation"], field="relation")
    graph = _mapping(summary["graph"], field="graph")
    guard = _mapping(summary["guard"], field="guard")
    row = {
        "step": step,
        "single_psnr": single["psnr"],
        "single_ssim": single["ssim"],
        "group_a_psnr": group_a["psnr"],
        "group_a_ssim": group_a["ssim"],
        "planner_macro_f1": planner["macro_f1"],
        "relation_accuracy": relation["relation_accuracy_non_ambiguous"],
        "parallel_precision": relation["parallel_precision_non_ambiguous"],
        "parallel_recall": relation["parallel_recall_non_ambiguous"],
        "pre_cycle_rate": graph["pre_compiler_cycle_rate"],
        "dropped_edge_rate": graph["dropped_edge_rate"],
        "guard_spearman_rain": guard["guard_spearman_rain"],
        "guard_spearman_haze": guard["guard_spearman_haze"],
        "guard_mae_rain": guard["guard_mae_rain"],
        "guard_mae_haze": guard["guard_mae_haze"],
        "guard_std_rain": guard["guard_std_rain"],
        "guard_std_haze": guard["guard_std_haze"],
        "guard_high_frac_rain": guard["guard_high_frac_rain"],
        "guard_high_frac_haze": guard["guard_high_frac_haze"],
        # Stage3 reports these schema columns explicitly as not measured; they
        # never enter checkpoint ranking. Stage4 provides trajectory variants.
        "clean_misuse_psnr": None,
        "clean_misuse_ssim": None,
        "clean_misuse_residual_norm": None,
        "wrong_skill_identity_psnr": None,
        "wrong_skill_identity_ssim": None,
        "wrong_skill_residual_norm": None,
        "reentry_request_rate": graph["reentry_request_rate"],
        "unexpected_skill_activation_rate": graph["unexpected_skill_activation_rate"],
        "mean_program_levels": graph["mean_program_levels"],
    }
    return row


def append_calibration_history(path: str | Path, row: Mapping[str, Any]) -> None:
    destination = Path(path)
    existing_rows: list[dict[str, str]] = []
    if destination.is_file():
        with destination.open("r", encoding="utf-8", newline="") as existing:
            reader = csv.DictReader(existing)
            if tuple(reader.fieldnames or ()) != CALIBRATION_COLUMNS:
                raise Stage3ContractError(
                    "calibration history header drifted: "
                    f"expected {CALIBRATION_COLUMNS}, got {tuple(reader.fieldnames or ())}"
                )
            existing_rows = [dict(item) for item in reader]

    # Normalize through DictWriter once so replay comparison uses the exact CSV
    # representation (including None -> empty string and float formatting).
    normalized_buffer = io.StringIO(newline="")
    normalized_writer = csv.DictWriter(
        normalized_buffer, fieldnames=CALIBRATION_COLUMNS
    )
    normalized_writer.writerow({key: row.get(key) for key in CALIBRATION_COLUMNS})
    normalized_buffer.seek(0)
    normalized = next(
        csv.DictReader(
            io.StringIO(
                ",".join(CALIBRATION_COLUMNS) + "\n" + normalized_buffer.getvalue()
            )
        )
    )
    if any(existing == normalized for existing in existing_rows):
        return
    if normalized.get("planner_macro_f1", ""):
        conflicting_stage3_rows = [
            existing
            for existing in existing_rows
            if existing.get("step") == normalized.get("step")
            and existing.get("planner_macro_f1", "")
        ]
        if conflicting_stage3_rows:
            raise Stage3ContractError(
                "conflicting Stage3 calibration row for the same step"
            )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CALIBRATION_COLUMNS)
    writer.writeheader()
    writer.writerows(existing_rows)
    writer.writerow(normalized)
    atomic_write_text(destination, output.getvalue())


__all__ = [
    "APPROVAL_SCHEMA",
    "CALIBRATION_COLUMNS",
    "PROTOCOL_ID",
    "STAGE3_ALLOCATOR_CONF",
    "STAGE3_BASE_TARGET_STEP",
    "STAGE3_EMA_SCHEMA",
    "STAGE3_EMA_SCOPE",
    "STAGE3_EXTENSION_FILENAME",
    "STAGE3_EXTENSION_LR_POLICY",
    "STAGE3_EXTENSION_MIGRATION_NAME",
    "STAGE3_EXTENSION_SCHEMA",
    "STAGE3_EXTENSION_TARGET_STEP",
    "STAGE3_EXTENSION_VALIDATION_STEPS",
    "STAGE3_SCHEMA",
    "Stage3ApprovalEvidence",
    "Stage3ContractError",
    "Stage3ExtensionEvidence",
    "Stage3MicroBatchTrial",
    "Stage3OptimizerTransaction",
    "Stage3ParentLoadReport",
    "Stage3PlannerEMA",
    "Stage3Paths",
    "Stage3StepResult",
    "Stage3SupervisionBatch",
    "Stage3ValidationVRAMGate",
    "Stage3ValidationVRAMTopology",
    "THRESHOLD_SCHEMA",
    "THRESHOLD_F1_TOLERANCE",
    "THRESHOLD_TIE_BREAK",
    "ThresholdCalibration",
    "align_guard_prediction_to_target",
    "append_calibration_history",
    "assert_relation_clean_disjoint",
    "assert_only_planner_gradients",
    "build_stage3_model",
    "build_stage3_optimizer",
    "build_stage3_provenance",
    "calibrate_presence_thresholds",
    "calibration_history_row",
    "collect_primary_val_presence",
    "configure_stage3_reproducibility",
    "enforce_stage3_peak_memory",
    "freeze_presence_thresholds",
    "guard_structure_diagnostics",
    "load_relation_records",
    "load_stage1_ema_into_graphrestore",
    "load_stage3_best_ema",
    "presence_diagnostics",
    "relation_baseline_audit",
    "prepare_stage3_supervision_batch",
    "probe_stage3_validation_vram",
    "resume_stage3_checkpoint",
    "reset_stage3_peak_memory",
    "save_stage3_checkpoint",
    "select_stage3_micro_batch",
    "set_stage3_trainability",
    "stage3_planner_forward",
    "stage3_ema_policy_metadata",
    "stage3_supervision_loss",
    "stage3_training_target_step",
    "train_stage3_optimizer_step",
    "validate_stage3",
    "validate_stage3_approval",
    "validate_stage3_allocator_conf",
    "validate_stage3_config",
    "validate_stage3_extension_authorization",
    "validate_stage3_finalization_outputs",
    "validate_stage3_pending_validation_step",
    "validate_stage3_validation_vram_evidence",
    "validation_score",
]
