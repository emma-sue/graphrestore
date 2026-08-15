"""Stage3 planner/guard supervision under the frozen V7.1 contract.

The module is deliberately split into a file-only approval preflight and the
CUDA/data portion.  A caller must successfully validate the approval produced
by :mod:`src.training.orchestration` before constructing a dataset, probing a
GPU, or loading the Stage1 executor.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import math
import platform
import random
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
from src.training.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    checkpoint_payload,
    load_checkpoint,
    restore_rng_state,
    unwrap_model,
)
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import WarmupCosineScheduler
from src.training.provenance import semantic_source_hashes
from src.training.relation_supervision import non_ambiguous_relation_metrics
from src.training.selection import ValidationScore
from src.training.stage1_engine import STAGE1_EMA_SCOPE, stage1_ema_policy_metadata
from src.utils.git import git_commit
from src.utils.hashing import is_sha256, sha256_file, sha256_json
from src.utils.io import (
    atomic_write_json,
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
RELATION_CLASSES = ("i_before_j", "j_before_i", "parallel")
PAIR_TO_ROW = {pair: index for index, pair in enumerate(PAIR_INDICES)}
_FORBIDDEN_STAGE3_TOKENS = ("mio100", "group_b", "group_c", "exploration")


class Stage3ContractError(RuntimeError):
    """Stage3 would violate approval, data, supervision, or runtime locks."""


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
        state["scope"] = "planner_parameters_only_executor_bitwise_frozen"
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
    completed_passes: int
    images_per_second: float
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    error: str | None = None


@dataclass(frozen=True)
class ThresholdCalibration:
    thresholds: tuple[float, ...]
    per_skill_f1: tuple[float, ...]
    grid: tuple[float, ...]
    tie_break: str = "lowest_threshold"


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
    resolved_path = _project_path(root, path_config.get("resolved_paths"), field="resolved_paths")
    resolved = _mapping(load_yaml(resolved_path), field="resolved paths")

    approval_path = _project_path(root, path_config.get("required_approval"), field="required_approval")
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

    train_manifest = _project_path(root, resolved.get(path_config.get("train_manifest_key")), field="primary_train")
    val_manifest = _project_path(root, resolved.get(path_config.get("val_manifest_key")), field="primary_val")
    executor = _project_path(root, path_config.get("executor_checkpoint"), field="executor_checkpoint")
    relation_train = _project_path(root, path_config.get("relation_train"), field="relation_train")
    relation_val = _project_path(root, path_config.get("relation_val"), field="relation_val")
    pair_prior = _project_path(root, path_config.get("pair_prior"), field="pair_prior")
    global_priority = _project_path(root, path_config.get("global_priority"), field="global_priority")
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
    if not approval_required.is_file() or sha256_file(approval_required) != required_sha:
        raise Stage3ContractError("approval-required marker changed after approval")
    required = _mapping(load_json(approval_required), field="STAGE3_APPROVAL_REQUIRED.json")
    if (
        required.get("schema_version") != APPROVAL_SCHEMA
        or required.get("kind") != "stage3_approval_required"
        or required.get("approved") is not False
        or required.get("bindings") != bindings
    ):
        raise Stage3ContractError("approval-required marker no longer matches approval")
    if approval.get("stage2_decision_sha256") != stage2_binding["sha256"]:
        raise Stage3ContractError("approved Stage2 decision SHA does not match its binding")

    decision = _mapping(load_json(stage2_decision), field="stage2_decision.json")
    if decision.get("approved") is not False or not isinstance(decision.get("overall"), Mapping):
        raise Stage3ContractError("invalid frozen Stage2 decision")
    expected_decision = {
        "stage1_checkpoint_sha256": sha256_file(executor),
        "interaction_train_manifest_sha256": verified_bindings["interaction_train_manifest"]["sha256"],
        "interaction_val_manifest_sha256": verified_bindings["interaction_val_manifest"]["sha256"],
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
            raise Stage3ContractError("Stage3 must be launched by the approved orchestrator")
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

    formal_output = _project_path(root, path_config.get("output_dir"), field="output_dir")
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
        else _project_path(root, path_config.get("calibration_history"), field="calibration_history")
    )
    return Stage3Paths(
        project_root=root,
        config_path=config_file,
        config=config,
        resolved_path=resolved_path,
        resolved=resolved,
        training_data_root=_project_path(root, resolved.get("training_data_root"), field="training_data_root"),
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
    if any(not isinstance(key, str) or not torch.is_tensor(tensor) for key, tensor in value.items()):
        raise Stage3ContractError(f"{field} contains non-tensor values")
    return value  # type: ignore[return-value]


def _load_compiler_evidence(paths: Stage3Paths) -> tuple[dict[str, Any], dict[str, float], Tensor]:
    prior_document = _mapping(load_json(paths.pair_prior), field="pair_prior.json")
    compiler_prior = _mapping(prior_document.get("pair_prior"), field="pair_prior.pair_prior")
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
            raise Stage3ContractError(f"pair prior probabilities do not sum to one: {pair_id}")
        normalized_prior[pair_id] = values

    priority_document = _mapping(load_json(paths.global_priority), field="global_priority.json")
    if int(priority_document.get("n_ambiguous_excluded", -1)) < 0:
        raise Stage3ContractError("global priority lacks ambiguous exclusion count")
    priority = _mapping(priority_document.get("priority"), field="global_priority.priority")
    if set(priority) != set(SKILLS):
        raise Stage3ContractError("global priority must contain exactly eight skills")
    normalized_priority = {skill: float(priority[skill]) for skill in SKILLS}
    if any(not math.isfinite(value) for value in normalized_priority.values()):
        raise Stage3ContractError("global priority contains non-finite scores")

    profiles_document = _mapping(load_json(paths.effect_profiles), field="skill_effect_profiles.json")
    vectors = _mapping(profiles_document.get("effect_vectors"), field="effect_vectors")
    if set(vectors) != set(SKILLS) or int(profiles_document.get("effect_vector_dim", -1)) != 40:
        raise Stage3ContractError("Stage2 effect profiles must be exactly 8x40")
    profile_tensor = torch.tensor(
        [[float(value) for value in vectors[skill]] for skill in SKILLS],
        dtype=torch.float32,
    )
    if tuple(profile_tensor.shape) != (8, 40) or not bool(torch.isfinite(profile_tensor).all()):
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
                raise Stage3ContractError(f"executor parameter remained trainable: {name}")
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
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "graphrestore-checkpoint-v1":
        raise Stage3ContractError("Stage1 checkpoint schema mismatch")
    if str(payload.get("stage", "")).lower().replace("-", "_") != "stage1":
        raise Stage3ContractError("Stage3 parent checkpoint is not Stage1")
    if payload.get("model_role") != "ema_selection" or payload.get("resumable") is not False:
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
    if ema.get("num_updates") != step:
        raise Stage3ContractError(
            "Stage1 best EMA update count does not match checkpoint step"
        )
    shadow = _strict_tensor_mapping(ema.get("shadow"), field="checkpoint.ema.shadow")
    if source.keys() != shadow.keys():
        raise Stage3ContractError("Stage1 best model/EMA keys differ")
    for name in source:
        if source[name].shape != shadow[name].shape or source[name].dtype != shadow[name].dtype:
            raise Stage3ContractError(
                f"Stage1 best model/EMA metadata differs at {name}"
            )
        if not torch.equal(source[name], shadow[name]):
            raise Stage3ContractError(f"Stage1 best checkpoint does not expose EMA at {name}")
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
    parameters = [parameter for parameter in unwrap_model(model).planner.parameters() if parameter.requires_grad]
    if not parameters:
        raise Stage3ContractError("Stage3 planner optimizer is empty")
    kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.999)}
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
            raise Stage3ContractError(f"{sample_id}: interaction manifest binding mismatch")
        if row.get("pair_orientation") != "ProgramPlanner.PAIR_INDICES_ascending_normative_skill_id":
            raise Stage3ContractError(f"{sample_id}: non-normative pair orientation")
        skill_ids = row.get("skill_ids")
        if (
            not isinstance(skill_ids, list)
            or len(skill_ids) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in skill_ids)
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
                raise Stage3ContractError(f"{sample_id}: invalid one-hot relation supervision")
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
    cursors = _batch_tensor(raw, "sample_cursor", device=torch.device("cpu")).long().reshape(-1)
    if image.ndim != 4 or image.shape[1] != 3:
        raise Stage3ContractError("Stage3 images must be RGB BCHW")
    batch = image.shape[0]
    sample_ids = _batch_strings(raw.get("sample_id"), batch, field="sample_id")
    if tuple(clean.shape) != tuple(image.shape) or tuple(only_i.shape) != tuple(image.shape) or tuple(only_j.shape) != tuple(image.shape):
        raise Stage3ContractError("Stage3 subset images must match input shape")
    if tuple(guards.shape[:2]) != (batch, len(SKILLS)):
        raise Stage3ContractError("Stage3 guard targets must be Bx8xH/4xW/4")
    if tuple(present.shape) != (batch, len(SKILLS)) or tuple(present_ids.shape) != (batch, 2):
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
        elif parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item()):
            raise Stage3ContractError(f"frozen executor received gradient: {name}")
    if not planner_has_gradient:
        raise Stage3ContractError("Stage3 backward produced no nonzero planner gradient")


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
) -> Stage3StepResult:
    if not micro_batches:
        raise ValueError("Stage3 requires at least one micro batch")
    if gradient_clip_norm != 1.0:
        raise Stage3ContractError("Stage3 gradient clipping must be 1.0")
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
    parameters = [parameter for parameter in model.planner.parameters() if parameter.grad is not None]
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
        raise Stage3ContractError("model-generated intermediate fraction exceeded schedule")
    return Stage3StepResult(
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
    if tuple(probabilities.shape) != tuple(targets.shape) or probabilities.ndim != 2 or probabilities.shape[1] != len(SKILLS):
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
        }
    return {
        "sample_count": int(probabilities.shape[0]),
        "macro_f1": math.fsum(f1_values) / len(f1_values),
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
    """Maximize per-skill F1; exact ties choose the lowest threshold."""

    if (minimum, maximum, step) != (0.20, 0.80, 0.02):
        raise Stage3ContractError("Stage3 threshold grid drifted")
    probabilities = probabilities.detach().float().cpu()
    targets = targets.detach().bool().cpu()
    if tuple(probabilities.shape) != tuple(targets.shape) or probabilities.ndim != 2 or probabilities.shape[1] != 8:
        raise ValueError("calibration requires matching Nx8 probabilities/targets")
    grid = tuple(value / 100.0 for value in range(20, 81, 2))
    selected: list[float] = []
    scores: list[float] = []
    for skill in range(8):
        truth = targets[:, skill]
        best_threshold = grid[0]
        best_f1 = -1.0
        for threshold in grid:
            prediction = probabilities[:, skill] >= threshold
            tp = int((prediction & truth).sum())
            fp = int((prediction & ~truth).sum())
            fn = int((~prediction & truth).sum())
            f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
            if f1 > best_f1:
                best_threshold, best_f1 = threshold, f1
        selected.append(best_threshold)
        scores.append(best_f1)
    return ThresholdCalibration(
        thresholds=tuple(selected),
        per_skill_f1=tuple(scores),
        grid=grid,
    )


def freeze_presence_thresholds(
    destination: str | Path,
    calibration: ThresholdCalibration,
    *,
    primary_val_manifest: str | Path,
    selected_checkpoint: str | Path,
    approval_sha256: str,
) -> dict[str, Any]:
    manifest = Path(primary_val_manifest).resolve()
    checkpoint = Path(selected_checkpoint).resolve()
    if "mio100" in str(manifest).lower() or "group_b" in str(manifest).lower() or "group_c" in str(manifest).lower():
        raise Stage3ContractError("MiO100/Group B/C threshold calibration is forbidden")
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
        "thresholds": {
            skill: calibration.thresholds[index]
            for index, skill in enumerate(SKILLS)
        },
        "per_skill_f1": {
            skill: calibration.per_skill_f1[index]
            for index, skill in enumerate(SKILLS)
        },
        "search_grid": list(calibration.grid),
        "tie_break": calibration.tie_break,
        "calibration_runs": 1,
        "mio100_rows_read": 0,
        "frozen": True,
    }
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


def _spearman(predicted: Tensor, target: Tensor, variance_threshold: float) -> float | None:
    x = predicted.detach().float().cpu().reshape(-1).numpy().astype(np.float64)
    y = target.detach().float().cpu().reshape(-1).numpy().astype(np.float64)
    if float(np.var(x)) < variance_threshold or float(np.var(y)) < variance_threshold:
        return None
    x_rank, y_rank = _rankdata_average(x), _rankdata_average(y)
    correlation = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return correlation if math.isfinite(correlation) else None


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
        indices = torch.nonzero(presence[:, skill_id], as_tuple=False).flatten().tolist()
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
        raise Stage3ContractError(f"primary_val {group} must contain exactly eight tasks")
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
    presence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Validate restoration on primary_val and relations on interaction_val.

    No Stage2 train relation is accepted by this API and no MiO100 path is
    opened.  Guard diagnostics are comprehensive but returned separately from
    the restoration-first checkpoint score.
    """

    if dataset.training or dataset.crop_size is not None:
        raise Stage3ContractError("Stage3 validation must be full-resolution/no augmentation")
    if any(record.group not in {"single", "A"} for record in dataset.records):
        raise Stage3ContractError("Stage3 validation contains forbidden data groups")
    if any(str(row.get("split")) != "val" for row in relation_val.values()):
        raise Stage3ContractError("Stage3 relation validation must use interaction_val only")
    model.eval()
    fixed_thresholds = torch.full((len(SKILLS),), float(presence_threshold), device=device)
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
        metric = official_psnr_ssim(prediction, target.detach().float().cpu(), quantize=True)
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
        all_presence_probabilities.append(plan.presence_probabilities[0].detach().float().cpu())
        all_presence_targets.append(sample["presence_target"].detach().float().cpu())
        all_guard_predictions.append(plan.spatial_guard_probabilities[0].detach().float().cpu())
        all_guard_targets.append(sample["guard_targets"].detach().float().cpu())

        graph = traced.compiled_graphs[0]
        if not graph.cycle_free:
            raise Stage3ContractError("post-compiler graph must be cycle-free")
        pre_cycle_samples += int(bool(graph.dropped_edges))
        dropped_edges += len(graph.dropped_edges)
        proposed_edges += len(graph.edges) + len(graph.dropped_edges)
        program_levels.append(len(graph.levels))
        for trace in traced.trace:
            reentry_requests += int(trace.reentry_request_mask.sum().item())
            unexpected_activations += int(trace.unexpected_activation_mask.sum().item())
            trace_slots += trace.reentry_request_mask.numel()

        relation = relation_val.get(record.sample_id)
        if relation is not None:
            pair = tuple(int(value) for value in relation["skill_ids"])
            relation_logits.append(plan.relation_logits[0, PAIR_TO_ROW[pair]].detach().float().cpu())
            ambiguous = relation["label"] == "ambiguous"
            relation_ambiguous.append(ambiguous)
            relation_targets.append(0 if ambiguous else int(relation["relation_class_index"]))
            relation_pair_ids.append(str(relation["pair_id"]))

    if set(relation_val) != set(
        row["sample_id"] for row in metric_rows if row["sample_id"] in relation_val
    ):
        missing = sorted(set(relation_val) - {row["sample_id"] for row in metric_rows})
        raise Stage3ContractError(f"interaction_val rows absent from primary_val: {missing[:8]}")
    probability_tensor = torch.stack(all_presence_probabilities)
    presence_target_tensor = torch.stack(all_presence_targets)
    relation_metric = non_ambiguous_relation_metrics(
        torch.stack(relation_logits),
        torch.tensor(relation_targets, dtype=torch.long),
        torch.tensor(relation_ambiguous, dtype=torch.bool),
        pair_ids=relation_pair_ids,
    )
    relation_metric.pop("pair_prior_non_ambiguous", None)
    relation_metric.pop("majority_label_share_non_ambiguous", None)
    guard_metric = _guard_structure_diagnostics_variable_size(
        all_guard_predictions,
        all_guard_targets,
        presence_target_tensor,
        variance_threshold=1.0e-8,
    )
    planner_metric = presence_diagnostics(
        probability_tensor, presence_target_tensor, presence_threshold
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
        "unexpected_skill_activation_rate": unexpected_activations / trace_slots if trace_slots else 0.0,
        "mean_program_levels": math.fsum(program_levels) / len(program_levels),
    }
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
            "checkpoint_presence_threshold": presence_threshold,
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
        with _validation_autocast(device, use_bf16):
            features = model.encode(image)
            output = model.plan_state(
                image, image, features, round_value=0.0, compute_relations=False
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
    max_steps: int = 12_000,
) -> dict[str, Any]:
    if micro_batch not in {1, 2, 4, 8} or 8 % micro_batch:
        raise Stage3ContractError("Stage3 micro batch must divide effective batch 8")
    if accumulation_steps != 8 // micro_batch or max_steps != 12_000:
        raise Stage3ContractError("Stage3 runtime schedule drifted")
    expected = _mapping(paths.resolved.get("expected_identity"), field="expected_identity")
    agenticir_commit = git_commit(paths.resolved["agenticir_repo"])
    mioir_commit = git_commit(paths.resolved["mioir_repo"])
    if agenticir_commit != expected.get("agenticir_commit") or mioir_commit != expected.get("mioir_commit"):
        raise Stage3ContractError("upstream repository commit drifted")
    bindings = {
        logical: dict(value) for logical, value in paths.approval.bindings.items()
    }
    return {
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
            "amp_dtype": "bf16",
            "tf32": True,
            "model_generated_intermediate_maximum_fraction": 0.10,
        },
        "dependency_versions": stage3_dependency_versions(),
        "data_exposure": {
            "train": "primary_train single/A only",
            "validation": "primary_val single/A + interaction_val labels",
            "mio100": False,
            "group_b_or_c": False,
        },
    }


def save_stage3_checkpoint(
    destination: str | Path,
    *,
    step: int,
    model: GraphRestore,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: StatefulEpisodeSampler,
    provenance: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    model_as_ema: bool = False,
) -> None:
    context = ema.apply_to(model) if model_as_ema else nullcontext()
    with context:
        payload = checkpoint_payload(
            stage="stage3",
            step=step,
            model=model,
            ema_state=ema.state_dict(),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            sampler_state=sampler.state_dict(consumed_optimizer_step=step),
            provenance=provenance,
            metrics=metrics,
        )
        payload["amp"] = {"dtype": "bfloat16", "scaler_required": False}
        payload["executor_frozen"] = True
        payload["trainable_prefixes"] = ["planner."]
        payload["model_role"] = (
            "ema_selection" if model_as_ema else "raw_training_state"
        )
        payload["resumable"] = not model_as_ema
        atomic_torch_save(payload, destination)
    set_stage3_trainability(model)


def _restore_stage3_ema(
    ema: ExponentialMovingAverage, value: object
) -> None:
    state = _mapping(value, field="checkpoint.ema")
    if isinstance(ema, Stage3PlannerEMA) and state.get("scope") != (
        "planner_parameters_only_executor_bitwise_frozen"
    ):
        raise Stage3ContractError("Stage3 resume EMA scope drifted")
    shadow = _strict_tensor_mapping(state.get("shadow"), field="checkpoint.ema.shadow")
    if shadow.keys() != ema.shadow.keys():
        raise Stage3ContractError("Stage3 resume EMA keys drifted")
    ema.load_state_dict(state)


def resume_stage3_checkpoint(
    checkpoint: str | Path,
    *,
    model: GraphRestore,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: StatefulEpisodeSampler,
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    # Inspect role metadata before mutating model, optimizer, scheduler, RNG,
    # EMA, or sampler.  Selection EMA checkpoints are never resumable.
    header = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if (
        not isinstance(header, Mapping)
        or header.get("stage") != "stage3"
        or header.get("model_role") != "raw_training_state"
        or header.get("resumable") is not True
    ):
        raise Stage3ContractError(
            "Stage3 resume requires raw last.pth; EMA selection checkpoints are non-resumable"
        )
    payload = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_provenance=expected_provenance,
        restore_rng=True,
        map_location="cpu",
    )
    if payload.get("executor_frozen") is not True:
        raise Stage3ContractError("resume checkpoint is not a frozen-executor Stage3 run")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 12_000:
        raise Stage3ContractError("invalid Stage3 resume step")
    _restore_stage3_ema(ema, payload.get("ema"))
    sampler_state = _mapping(payload.get("sampler_state"), field="checkpoint.sampler_state")
    sampler.load_state_dict(dict(sampler_state))
    if sampler_state.get("consumed_optimizer_step") != step:
        raise Stage3ContractError("Stage3 checkpoint/sampler step mismatch")
    set_stage3_trainability(model)
    return payload


def load_stage3_best_ema(
    paths: Stage3Paths,
    checkpoint: str | Path,
    *,
    device: torch.device,
    model_factory: Callable[..., GraphRestore] = GraphRestore,
    load_frozen_thresholds: bool = True,
) -> GraphRestore:
    """Reusable Stage4/evaluation loader for a strictly exposed Stage3 EMA."""

    model, _ = build_stage3_model(paths, device=torch.device("cpu"), model_factory=model_factory)
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
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "graphrestore-checkpoint-v1"
        or payload.get("stage") != "stage3"
        or payload.get("model_role") != "ema_selection"
        or payload.get("resumable") is not False
    ):
        raise Stage3ContractError("Stage3 best checkpoint schema/stage mismatch")
    source = _strict_tensor_mapping(payload.get("model"), field="Stage3 best model")
    ema = _mapping(payload.get("ema"), field="Stage3 best EMA")
    if ema.get("scope") != "planner_parameters_only_executor_bitwise_frozen":
        raise Stage3ContractError("Stage3 best EMA did not preserve the frozen executor")
    shadow = _strict_tensor_mapping(ema.get("shadow"), field="Stage3 best EMA shadow")
    if source.keys() != shadow.keys() or any(not torch.equal(source[name], shadow[name]) for name in source):
        raise Stage3ContractError("Stage3 best_ema.pth does not expose EMA as model")
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
    provenance = _mapping(payload.get("provenance"), field="Stage3 best provenance")
    approval = _mapping(provenance.get("stage3_approval"), field="Stage3 checkpoint approval")
    if approval.get("sha256") != paths.approval.approval_sha256:
        raise Stage3ContractError("Stage3 checkpoint approval hash is stale")
    if provenance.get("bindings") != paths.approval.bindings:
        raise Stage3ContractError("Stage3 checkpoint frozen bindings differ from approval")
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
        raise Stage3ContractError(f"Stage3 checkpoint provenance hash drift: {hash_drift}")
    model.load_state_dict(source, strict=True)
    if load_frozen_thresholds:
        thresholds = _mapping(load_json(paths.thresholds), field="planner thresholds")
        if (
            thresholds.get("schema_version") != THRESHOLD_SCHEMA
            or thresholds.get("protocol_id") != PROTOCOL_ID
            or thresholds.get("frozen") is not True
            or thresholds.get("skills") != list(SKILLS)
            or thresholds.get("stage3_approval_sha256") != paths.approval.approval_sha256
            or thresholds.get("checkpoint_sha256") != sha256_file(checkpoint_path)
            or thresholds.get("primary_val_manifest_sha256") != sha256_file(paths.val_manifest)
            or thresholds.get("calibration_runs") != 1
            or thresholds.get("mio100_rows_read") != 0
        ):
            raise Stage3ContractError("invalid frozen Stage3 presence thresholds")
        values = _mapping(thresholds.get("thresholds"), field="threshold values")
        if set(values) != set(SKILLS) or thresholds.get("search_grid") != [
            value / 100.0 for value in range(20, 81, 2)
        ]:
            raise Stage3ContractError("frozen threshold skill/grid schema drifted")
        ordered_thresholds = [float(values[skill]) for skill in SKILLS]
        if any(value not in {item / 100.0 for item in range(20, 81, 2)} for value in ordered_thresholds):
            raise Stage3ContractError("frozen threshold lies outside the locked grid")
        model.set_presence_thresholds(ordered_thresholds)
    set_stage3_trainability(model)
    model.to(device)
    model.eval()
    return model


def _synthetic_probe_batch(batch: int, device: torch.device) -> Stage3SupervisionBatch:
    image = torch.rand(batch, 3, 192, 192, device=device)
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
    return Stage3SupervisionBatch(
        x0=image,
        current=image.clone(),
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
        state_kinds=tuple("group_a_pair" for _ in range(batch)),
        model_intermediate_count=0,
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
        raise Stage3ContractError("Stage3 probe must use ten passes and <=90% reserved")
    rng = capture_rng_state()
    trials: list[Stage3MicroBatchTrial] = []
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    _stage3_train_mode(model)
    try:
        for candidate in candidates:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            completed = 0
            error: str | None = None
            throughput = 0.0
            started = time.perf_counter()
            try:
                probe = _synthetic_probe_batch(candidate, device)
                for _ in range(required_passes):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = stage3_planner_forward(model, probe)
                        loss, _ = stage3_supervision_loss(output, probe)
                    loss.total.backward()
                    assert_only_planner_gradients(model)
                    completed += 1
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                throughput = candidate * completed / max(elapsed, 1e-9)
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                passed = completed == required_passes and fraction <= maximum_reserved_fraction
                if not passed:
                    error = f"peak reserved fraction {fraction:.4f} exceeds 0.90"
            except torch.OutOfMemoryError as exc:
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                passed = False
                error = f"CUDA OOM: {exc}"
            finally:
                model.zero_grad(set_to_none=True)
                probe = output = loss = None
                torch.cuda.empty_cache()
            trials.append(
                Stage3MicroBatchTrial(
                    micro_batch=candidate,
                    passed=passed,
                    completed_passes=completed,
                    images_per_second=throughput,
                    peak_reserved_bytes=peak,
                    peak_reserved_fraction=fraction,
                    error=error,
                )
            )
    finally:
        restore_rng_state(rng)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    accepted = [trial for trial in trials if trial.passed]
    if not accepted:
        raise Stage3ContractError("no Stage3 micro batch passed the ten-pass <=90% gate")
    winner = max(accepted, key=lambda trial: (trial.images_per_second, trial.micro_batch))
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists = destination.is_file()
    if exists:
        with destination.open("r", encoding="utf-8", newline="") as existing:
            reader = csv.reader(existing)
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise Stage3ContractError("existing calibration history is empty") from exc
        if header != CALIBRATION_COLUMNS:
            raise Stage3ContractError(
                "calibration history header drifted: "
                f"expected {CALIBRATION_COLUMNS}, got {header}"
            )
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CALIBRATION_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CALIBRATION_COLUMNS})
        handle.flush()


__all__ = [
    "APPROVAL_SCHEMA",
    "CALIBRATION_COLUMNS",
    "PROTOCOL_ID",
    "STAGE3_SCHEMA",
    "Stage3ApprovalEvidence",
    "Stage3ContractError",
    "Stage3MicroBatchTrial",
    "Stage3ParentLoadReport",
    "Stage3PlannerEMA",
    "Stage3Paths",
    "Stage3StepResult",
    "Stage3SupervisionBatch",
    "THRESHOLD_SCHEMA",
    "ThresholdCalibration",
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
    "freeze_presence_thresholds",
    "guard_structure_diagnostics",
    "load_relation_records",
    "load_stage1_ema_into_graphrestore",
    "load_stage3_best_ema",
    "presence_diagnostics",
    "prepare_stage3_supervision_batch",
    "resume_stage3_checkpoint",
    "save_stage3_checkpoint",
    "select_stage3_micro_batch",
    "set_stage3_trainability",
    "stage3_planner_forward",
    "stage3_supervision_loss",
    "train_stage3_optimizer_step",
    "validate_stage3",
    "validate_stage3_approval",
    "validate_stage3_config",
    "validation_score",
]
