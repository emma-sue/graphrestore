"""Contract-bound Stage4 end-to-end GraphRestore training.

Stage4 is the only training stage that follows the model's discrete program
trajectory.  This module keeps that discrete decision honest: each sample's
partial-order graph is compiled exactly once at ``t=0`` and later rounds only
refresh presence, spatial guards, and stop.  Restoration gradients flow through
the selected executor path; relation logits receive their explicit planner
supervision and are never advertised as differentiable through the compiler.

Only frozen ``primary_train``/``primary_val`` recipes are consumed.  The
counterfactual episodes below are views of those recipes, not a new data source.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import csv
import json
import math
import os
import platform
import random
import stat
import time
from decimal import Decimal, InvalidOperation, localcontext
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, TextIO

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from src.data.episode_dataset import GraphRestoreEpisodeDataset
from src.data.manifests import SKILLS, task_buckets
from src.data.samplers import EpisodeRequest
from src.losses.guard_losses import guard_supervision_loss
from src.losses.planner_losses import (
    PlannerLossBreakdown,
    focal_binary_cross_entropy,
    planner_loss,
)
from src.metrics.agenticir_official import official_psnr_ssim, train_ssim_y
from src.net.graph_compiler import CompiledGraph, PAIR_TO_ROW
from src.net.graphrestore import GraphRestore, ProgramGraphState
from src.net.program_planner import PAIR_INDICES, PlannerOutput
from src.net.restormer_blocks import crop_to_shape, pad_to_multiple
from src.net.skill_adapter import SKILL_TO_INDEX
from src.training.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    checkpoint_payload,
    restore_rng_state,
    unwrap_model,
    verify_provenance,
)
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import WarmupCosineScheduler
from src.training.provenance import semantic_source_hashes
from src.training.selection import ValidationScore, is_better_checkpoint
from src.training.stage3_engine import (
    STAGE3_EMA_SCOPE,
    align_guard_prediction_to_target,
    stage3_ema_policy_metadata,
    validate_stage3_finalization_outputs,
)
from src.training.stage3_finalization import (
    Stage3RevocationAuthorization,
    validate_stage3_extension_revocation,
)
from src.utils.git import git_commit
from src.utils.hashing import is_sha256, sha256_file, sha256_json
from src.utils.io import (
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_json,
    utc_now_iso,
)


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
STAGE4_SCHEMA = "graphrestore-stage4-runtime-v1"
STAGE4_CHECKPOINT_STAGE = "stage4"
STAGE3_APPROVAL_SCHEMA = "graphrestore-stage3-approval-v1"
STAGE3_EXTENSION_APPROVAL_SCHEMA = "graphrestore-stage3-extension-approval-v1"
STAGE3_EXTENSION_APPROVAL_NAME = "STAGE3_EXTENSION_APPROVED.json"
STAGE3_EXTENSION_BACKUP_DIR_NAME = "stage3_extension_12000_to_18000_v1"
STAGE3_BASE_STEP = 12_000
STAGE3_EXTENSION_TARGET_STEP = 18_000
STAGE3_VALIDATION_EVERY_STEPS = 2_000
STAGE3_BASE_VALIDATION_STEPS = tuple(
    range(
        STAGE3_VALIDATION_EVERY_STEPS,
        STAGE3_BASE_STEP + 1,
        STAGE3_VALIDATION_EVERY_STEPS,
    )
)
STAGE3_EXTENSION_VALIDATION_STEPS = (14_000, 16_000, 18_000)
STAGE3_EXTENSION_LR_POLICY = "hold_original_cosine_floor_after_schedule_horizon"
STAGE4_EMA_SCHEMA = "graphrestore-stage4-phase-aware-ema-v1"
STAGE4_EMA_SCOPE = (
    "stage4_trainable_named_parameters_ema_"
    "frozen_parameters_and_all_buffers_bitwise_copy"
)
STAGE4_ALLOCATOR_CONF = "backend:native,expandable_segments:True"
STAGE4_EXTENSION_CONDITIONAL_SCHEMA = (
    "graphrestore-stage4-extension-conditional-approval-v1"
)
STAGE4_EXTENSION_GATE_SCHEMA = "graphrestore-stage4-extension-gate-receipt-v1"
STAGE4_EXTENSION_CONDITIONAL_FILENAME = "STAGE4_EXTENSION_CONDITIONAL_APPROVED.json"
STAGE4_EXTENSION_GATE_FILENAME = "STAGE4_EXTENSION_GATE_RECEIPT.json"
STAGE4_EXTENSION_BACKUP_DIR_NAME = "stage4_extension_40000_to_48000_v1"
STAGE4_EXTENSION_BASE_STEP = 40_000
STAGE4_EXTENSION_TARGET_STEP = 48_000
STAGE4_EXTENSION_CYCLES = 2
STAGE4_EXTENSION_VALIDATION_EVERY_STEPS = 4_000
STAGE4_EXTENSION_VALIDATION_STEPS = (44_000, 48_000)
STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS = 40_000
STAGE4_EXTENSION_MIN_LR = 5.0e-7
STAGE4_EXTENSION_LR_POLICY = "hold_original_cosine_floor_after_schedule_horizon"
STAGE4_EXTENSION_THRESHOLD_DECIMAL = "0.20"
STAGE4_EXTENSION_ADDITIONAL_OPTIMIZER_STEPS = 8_000
STAGE4_EXTENSION_TRIGGER_METRIC = "group_a_psnr"
STAGE4_EXTENSION_TRIGGER_LHS_STEP = 40_000
STAGE4_EXTENSION_TRIGGER_RHS_STEP = 36_000
STAGE4_EXTENSION_TRIGGER_OPERATOR = "lhs_minus_rhs_greater_than_or_equal"
STAGE4_EXTENSION_TRIGGER_ARITHMETIC = "decimal_exact_from_canonical_csv_strings"
STAGE4_EXTENSION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "scripts/train_stage4_e2e.py",
    "src/training/orchestration.py",
    "src/training/stage4_engine.py",
)
STAGE4_EXTENSION_SNAPSHOT_FILENAMES = {
    "run_contract": "pre_extension_run_contract.json",
    "last_checkpoint": "pre_extension_last.pth",
    "best_checkpoint": "pre_extension_best_ema.pth",
    "calibration_history": "pre_extension_stage4_calibration_history.csv",
    "validation_latest": "pre_extension_validation_latest.json",
    "report": "pre_extension_STAGE4_E2E.md",
    "train_log": "pre_extension_train.jsonl",
    "orchestration_state": "pre_extension_orchestration_state.json",
    "pipeline_log": "pre_extension_main_pipeline.log",
    "config": "pre_extension_stage4_graphrestore_e2e.yaml",
}
STAGE4_EXTENSION_CALIBRATION_COLUMNS = (
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
EPISODE_TYPES = (
    "single_restoration",
    "group_a_pair_restoration",
    "clean_misuse",
    "wrong_skill",
)
COUNTERFACTUAL_TYPES = frozenset({"clean_misuse", "wrong_skill"})


class Stage4ContractError(RuntimeError):
    """A requested action would diverge from the frozen Stage4 contract."""


@dataclass(frozen=True)
class Stage4ExtensionEvidence:
    """Verified one-shot authorization for the conditional 40k -> 48k run."""

    conditional_path: Path
    conditional_sha256: str
    gate_path: Path
    gate_sha256: str
    base_step: int = STAGE4_EXTENSION_BASE_STEP
    target_step: int = STAGE4_EXTENSION_TARGET_STEP
    cycles: int = STAGE4_EXTENSION_CYCLES
    additional_optimizer_steps: int = STAGE4_EXTENSION_ADDITIONAL_OPTIMIZER_STEPS
    hard_terminal_step: int = STAGE4_EXTENSION_TARGET_STEP
    validation_every_steps: int = STAGE4_EXTENSION_VALIDATION_EVERY_STEPS
    validation_steps: tuple[int, ...] = STAGE4_EXTENSION_VALIDATION_STEPS
    schedule_horizon_steps: int = STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS
    min_lr: float = STAGE4_EXTENSION_MIN_LR
    lr_policy: str = STAGE4_EXTENSION_LR_POLICY
    exact_resume: bool = True
    reset_optimizer: bool = False
    reset_ema: bool = False
    reset_scheduler: bool = False
    reset_rng: bool = False
    reset_sampler: bool = False
    further_extension_authorized: bool = False

    def provenance_binding(self) -> dict[str, Any]:
        return {
            "conditional_authorization": {
                "path": str(self.conditional_path),
                "sha256": self.conditional_sha256,
            },
            "gate_receipt": {
                "path": str(self.gate_path),
                "sha256": self.gate_sha256,
            },
            "cycles": self.cycles,
            "additional_optimizer_steps": self.additional_optimizer_steps,
            "base_step": self.base_step,
            "target_step": self.target_step,
            "hard_terminal_step": self.hard_terminal_step,
            "validation_every_steps": self.validation_every_steps,
            "validation_steps": list(self.validation_steps),
            "schedule_horizon_steps": self.schedule_horizon_steps,
            "min_lr": self.min_lr,
            "lr_policy": self.lr_policy,
            "exact_resume": self.exact_resume,
            "reset_optimizer": self.reset_optimizer,
            "reset_ema": self.reset_ema,
            "reset_scheduler": self.reset_scheduler,
            "reset_rng": self.reset_rng,
            "reset_sampler": self.reset_sampler,
            "further_extension_authorized": self.further_extension_authorized,
        }


def stage4_ema_policy_metadata(decay: float) -> dict[str, object]:
    """Return the exact, checkpointed Stage4 EMA update contract."""

    if not 0.0 < decay < 1.0:
        raise ValueError("Stage4 EMA decay must be in (0,1)")
    return {
        "schema_version": STAGE4_EMA_SCHEMA,
        "scope": STAGE4_EMA_SCOPE,
        "parameter_selector": "stage4_parameter_role_and_requires_grad",
        "trainable_parameter_update": "standard_fp32_exponential_moving_average",
        "frozen_parameter_update": "copy_current_value_bitwise",
        "buffer_update": "copy_current_value_bitwise",
        "trainability_schedule": "static_for_all_stage4_optimizer_steps",
        "decay": float(decay),
    }


class Stage4PhaseAwareEMA(ExponentialMovingAverage):
    """EMA trainable Stage4 parameters and exactly copy all fixed state.

    Multiplying an unchanged floating-point value by the EMA coefficients can
    still move it by an ULP.  Stage4 therefore applies ordinary FP32 EMA only
    to parameters assigned to a Stage4 optimizer role.  Frozen parameters and
    every buffer are copied exactly on every committed optimizer step.
    """

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        core = unwrap_model(model)
        source = core.state_dict()
        if source.keys() != self.shadow.keys():
            raise RuntimeError("Stage4 EMA/model state keys drifted")
        parameters = dict(core.named_parameters(remove_duplicate=False))
        if any(name not in source for name in parameters):
            raise RuntimeError("Stage4 EMA named parameters escaped model state")

        self.num_updates += 1
        for name, value in source.items():
            target = self.shadow[name]
            parameter = parameters.get(name)
            trainable = (
                parameter is not None
                and stage4_parameter_role(name) is not None
                and parameter.requires_grad
            )
            if trainable:
                if not target.is_floating_point():
                    raise Stage4ContractError(
                        f"trainable Stage4 EMA parameter is not floating point: {name}"
                    )
                target.mul_(self.decay).add_(
                    value.detach().to(target), alpha=1.0 - self.decay
                )
            else:
                target.copy_(value.detach().to(target))

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["scope"] = STAGE4_EMA_SCOPE
        state["policy"] = stage4_ema_policy_metadata(self.decay)
        return state

    def validate_state_metadata(self, state: Mapping[str, object]) -> None:
        expected_keys = {"decay", "num_updates", "shadow", "scope", "policy"}
        if set(state) != expected_keys:
            raise Stage4ContractError(
                "Stage4 EMA state fields drifted: "
                f"expected {sorted(expected_keys)}, got {sorted(state)}"
            )
        if state.get("scope") != STAGE4_EMA_SCOPE:
            raise Stage4ContractError("Stage4 resume EMA scope drifted")
        if state.get("policy") != stage4_ema_policy_metadata(self.decay):
            raise Stage4ContractError("Stage4 resume EMA policy drifted")
        decay = state.get("decay")
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise Stage4ContractError("Stage4 resume EMA decay is invalid")
        if float(decay) != self.decay:
            raise Stage4ContractError("Stage4 resume EMA decay drifted")
        updates = state.get("num_updates")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise Stage4ContractError("Stage4 resume EMA update count is invalid")

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.validate_state_metadata(state)
        super().load_state_dict(state)


def build_stage4_ema(
    model: nn.Module,
    *,
    decay: float = 0.9999,
) -> Stage4PhaseAwareEMA:
    return Stage4PhaseAwareEMA(model, decay=decay)


def require_stage4_allocator_conf(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Reject Stage4 before the first CUDA query if allocator policy drifted."""

    environment = os.environ if environ is None else environ
    actual = environment.get("PYTORCH_CUDA_ALLOC_CONF")
    if actual != STAGE4_ALLOCATOR_CONF:
        raise Stage4ContractError(
            "PYTORCH_CUDA_ALLOC_CONF must be exactly "
            f"{STAGE4_ALLOCATOR_CONF!r} before Stage4 CUDA initialization; "
            f"got {actual!r}"
        )
    return actual


def _expect(config: Mapping[str, Any], path: Sequence[str], expected: Any) -> None:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise Stage4ContractError(f"missing Stage4 config key: {'.'.join(path)}")
        value = value[key]
    if value != expected:
        raise Stage4ContractError(
            f"Stage4 config drift at {'.'.join(path)}: "
            f"expected {expected!r}, got {value!r}"
        )


def validate_stage4_config(config: Mapping[str, Any]) -> None:
    """Fail closed on every Stage4 scientific/data/optimizer constant."""

    locked: tuple[tuple[tuple[str, ...], Any], ...] = (
        (("schema_version",), "1.0"),
        (("contract_version",), "GraphRestore-V7.1"),
        (("protocol_id",), PROTOCOL_ID),
        (("stage",), "stage4"),
        (("seed",), 2027),
        (("skills", "ordered_names"), list(SKILLS)),
        (("skills", "maximum_active"), 3),
        (("skills", "allow_skill_reentry"), False),
        (("skills", "max_calls_per_skill"), 1),
        (("program", "compile_relations_once_at_t0"), True),
        (("program", "delete_executed_nodes_after_level"), True),
        (("program", "insert_late_skills_into_frozen_dag"), False),
        (("program", "reencode_current_state_each_round"), True),
        (("program", "update_presence_guard_stop_each_round"), True),
        (("program", "kmax_train"), 2),
        (("program", "kmax_test"), 3),
        (("data", "allowed_groups"), ["single", "A"]),
        (("data", "forbidden_groups"), ["B", "C"]),
        (("data", "sampling", "single_restoration"), 0.20),
        (("data", "sampling", "group_a_pair_restoration"), 0.70),
        (("data", "sampling", "clean_misuse"), 0.05),
        (("data", "sampling", "wrong_skill_misuse"), 0.05),
        (("data", "group_a_sampling"), "uniform_8_combinations"),
        (("data", "single_sampling"), "uniform_8_classes"),
        (("data", "wrong_skill_pair_sampling"), "uniform_i_not_equal_j"),
        (("data", "crop_size"), 160),
        (("data", "minimum_crop_after_oom"), 128),
        (("data", "micro_batch_candidates"), [2, 1]),
        (("data", "effective_batch_size"), 4),
        (("model", "frozen"), ["encoder_level1", "encoder_level2"]),
        (("model", "discrete_graph_gradient_claim"), "forbidden"),
        (("teacher_forcing", "preserve_written_discontinuity_at_step12000"), True),
        (("training", "max_steps"), 40_000),
        (("training", "intermediate_levels_train_max"), 2),
        (("optimization", "optimizer"), "AdamW"),
        (("optimization", "betas"), [0.9, 0.999]),
        (("optimization", "weight_decay"), 1.0e-4),
        (("optimization", "weight_decay_norm_bias"), 0.0),
        (("optimization", "learning_rates", "planner"), 5.0e-5),
        (("optimization", "learning_rates", "skill_adapters_and_mixers"), 3.0e-5),
        (("optimization", "learning_rates", "decoder_refinement_rgb_head"), 1.0e-5),
        (("optimization", "learning_rates", "encoder_level3_level4"), 2.0e-6),
        (("optimization", "warmup_steps"), 800),
        (("optimization", "scheduler"), "cosine"),
        (("optimization", "min_lr"), 5.0e-7),
        (("optimization", "gradient_clip_norm"), 0.5),
        (("loss", "ordinary", "final_charbonnier_weight"), 1.0),
        (("loss", "ordinary", "intermediate_subset_charbonnier_weight"), 0.30),
        (("loss", "ordinary", "final_ssim_weight", "start"), 0.0),
        (("loss", "ordinary", "final_ssim_weight", "end"), 0.05),
        (("loss", "ordinary", "final_ssim_weight", "ramp_end_step"), 8000),
        (("loss", "counterfactual", "identity_charbonnier_weight"), 1.0),
        (("loss", "counterfactual", "identity_ssim_weight"), 0.05),
        (("loss", "planner_total_weight"), 0.05),
        (("loss", "training_quantization"), False),
        (("loss", "hard_clamp_forward"), False),
        (("runtime", "amp_dtype"), "bf16"),
        (("runtime", "tf32"), True),
        (("runtime", "channels_last"), False),
        (("runtime", "gradient_checkpointing"), "block_level"),
        (("runtime", "torch_compile"), False),
        (("runtime", "vram_maximum_peak_reserved_fraction"), 0.90),
        (("runtime", "freeze_crop_micro_accum_after_step0"), True),
        (("ema", "enabled"), True),
        (("ema", "decay"), 0.9999),
        (("validation", "every_steps"), 4000),
        (("validation", "manifest_key"), "primary_val_manifest"),
        (("validation", "groups"), ["single", "A"]),
        (("validation", "protocol"), "agenticir_official_parity"),
        (("checkpoint", "save_every_steps"), 4000),
        (("hard_guards", "require_stage3_approval"), True),
        (("hard_guards", "require_all_parent_hashes_match"), True),
        (("hard_guards", "allow_mio100_exploration"), False),
        (("hard_guards", "allow_mio100_formal_during_training"), False),
        (("hard_guards", "allow_group_b_or_c_training"), False),
        (("hard_guards", "fail_on_hash_mismatch"), True),
    )
    for path, expected in locked:
        _expect(config, path, expected)

    expected_teacher = [
        {
            "start_step": 0,
            "end_step_exclusive": 4000,
            "probability_start": 1.0,
            "probability_end": 1.0,
            "source": "true_active_set_and_distilled_relation",
        },
        {
            "start_step": 4000,
            "end_step_exclusive": 12000,
            "probability_start": 1.0,
            "probability_end": 0.5,
            "interpolation": "linear",
        },
        {
            "start_step": 12000,
            "end_step_exclusive": 40000,
            "probability_start": 0.25,
            "probability_end": 0.25,
            "source": "mixed_teacher_and_predicted_graph",
        },
    ]
    _expect(config, ("teacher_forcing", "schedule"), expected_teacher)

    forbidden = set(config["loss"]["forbidden"])
    expected_forbidden = {
        "gan",
        "lpips",
        "clip_iqa",
        "musiq",
        "dino_perceptual",
        "llm_reward",
        "reinforcement_learning",
        "independent_commit_verifier",
    }
    if forbidden != expected_forbidden:
        raise Stage4ContractError("Stage4 forbidden-loss set drifted")


def teacher_forcing_probability(step: int) -> float:
    """The written V7.1 schedule, including its deliberate step-12000 jump."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if step < 4000:
        return 1.0
    if step < 12_000:
        progress = (step - 4000) / 8000.0
        return 1.0 - 0.5 * progress
    return 0.25


def stage4_ssim_weight(step: int) -> float:
    """Cosine-ramp the ordinary SSIM term over the first 20% of Stage4."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if step >= 8000:
        return 0.05
    progress = step / 8000.0
    return 0.025 * (1.0 - math.cos(math.pi * progress))


def _cursor_rng(seed: int, cursor: int) -> random.Random:
    digest = hashlib.sha256(
        f"graphrestore-stage4:{seed}:{cursor}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class Stage4Request:
    index: int
    episode_type: str
    absolute_step: int
    sample_cursor: int
    use_teacher: bool
    forced_skill_ids: tuple[int, ...] = ()


def _relation_mapping(
    records: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(records, Mapping):
        result = {str(key): value for key, value in records.items()}
    else:
        result = {str(row.get("sample_id", "")): row for row in records}
    if not result or "" in result:
        raise Stage4ContractError(
            "relation records require unique non-empty sample IDs"
        )
    if len(result) != (
        len(records) if not isinstance(records, Mapping) else len(records)
    ):
        raise Stage4ContractError("duplicate relation sample ID")
    return result


def load_relation_records(path: str | Path) -> dict[str, Mapping[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise Stage4ContractError(f"missing relation labels: {source}")
    rows = [row for _, row in iter_jsonl(source)]
    return _relation_mapping(rows)


class Stage4EpisodeDataset(Dataset[dict[str, Any]]):
    """Stage4 views over the frozen primary recipe dataset."""

    def __init__(
        self,
        base: GraphRestoreEpisodeDataset,
        relation_records: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if not base.training or base.crop_size not in {(160, 160), (128, 128)}:
            raise Stage4ContractError(
                "Stage4 train dataset must use the frozen crop160 or gated crop128 fallback"
            )
        if any(record.group not in {"single", "A"} for record in base.records):
            raise Stage4ContractError("Stage4 dataset contains forbidden groups")
        self.base = base
        self.records = base.records
        self.relation_records = _relation_mapping(relation_records)

    def __len__(self) -> int:
        return len(self.base)

    def set_worker_seed(self, seed: int) -> None:
        self.base.set_worker_seed(seed)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __getitem__(self, request: Stage4Request) -> dict[str, Any]:
        if not isinstance(request, Stage4Request):
            raise TypeError("Stage4EpisodeDataset requires Stage4Request indices")
        if request.episode_type not in EPISODE_TYPES:
            raise Stage4ContractError(f"unknown Stage4 episode: {request.episode_type}")
        record = self.records[request.index]
        if (
            request.episode_type in {"single_restoration", "wrong_skill"}
            and record.is_pair
        ):
            raise Stage4ContractError(
                "single/wrong-skill request selected a pair recipe"
            )
        if request.episode_type == "group_a_pair_restoration" and not record.is_pair:
            raise Stage4ContractError("Group-A request selected a single recipe")

        sample = dict(
            self.base[
                EpisodeRequest(
                    index=request.index,
                    episode_type="restoration",
                    absolute_step=request.absolute_step,
                    sample_cursor=request.sample_cursor,
                )
            ]
        )
        input_image = sample["input"]
        if not torch.is_tensor(input_image):
            raise Stage4ContractError("base episode returned a non-tensor input")
        forced = torch.zeros(len(SKILLS), dtype=torch.bool)
        for skill_id in request.forced_skill_ids:
            if not 0 <= skill_id < len(SKILLS):
                raise Stage4ContractError(f"invalid forced skill ID: {skill_id}")
            forced[skill_id] = True

        if request.episode_type == "clean_misuse":
            if len(request.forced_skill_ids) not in {1, 2}:
                raise Stage4ContractError("clean misuse must force one or two skills")
            clean = sample["gt_clean"]
            sample["input"] = clean
            sample["x_both"] = clean
            sample["target"] = clean
            sample["presence_target"] = torch.zeros(len(SKILLS), dtype=torch.float32)
            sample["guard_targets"] = torch.zeros_like(sample["guard_targets"])
            sample["global_severity_targets"] = torch.zeros_like(
                sample["global_severity_targets"]
            )
            sample["present_skill_ids"] = torch.full((2,), -1, dtype=torch.long)
        elif request.episode_type == "wrong_skill":
            if len(request.forced_skill_ids) != 1:
                raise Stage4ContractError(
                    "wrong-skill misuse must force exactly one skill"
                )
            present = int(sample["present_skill_ids"][0])
            if request.forced_skill_ids[0] == present:
                raise Stage4ContractError(
                    "wrong-skill misuse cannot force the true skill"
                )
            # Identity target is the degraded input, not clean.
            sample["target"] = input_image
        elif request.forced_skill_ids:
            raise Stage4ContractError("ordinary restoration cannot force a skill")

        relation_row = -1
        relation_label = -2
        relation_weight = 0.0
        relation_ambiguous = False
        if request.episode_type == "group_a_pair_restoration":
            try:
                relation = self.relation_records[record.sample_id]
            except KeyError as exc:
                raise Stage4ContractError(
                    f"Group-A Stage4 sample lacks distilled relation: {record.sample_id}"
                ) from exc
            ids = tuple(sorted(record.skill_ids))
            relation_row = PAIR_TO_ROW[ids]
            label_name = str(relation.get("label", ""))
            if label_name == "ambiguous":
                relation_label = -1
                relation_weight = 0.25
                relation_ambiguous = True
            elif label_name in {"i_before_j", "j_before_i", "parallel"}:
                relation_label = ("i_before_j", "j_before_i", "parallel").index(
                    label_name
                )
                relation_weight = 1.0
            else:
                raise Stage4ContractError(
                    f"invalid distilled relation label for {record.sample_id}: {label_name!r}"
                )
            if relation.get("relation_weight") != relation_weight:
                raise Stage4ContractError("distilled relation weight drifted")

        sample.update(
            {
                "stage4_episode_type": request.episode_type,
                "use_teacher": torch.tensor(request.use_teacher, dtype=torch.bool),
                "forced_skill_mask": forced,
                "relation_row": torch.tensor(relation_row, dtype=torch.long),
                "relation_label": torch.tensor(relation_label, dtype=torch.long),
                "relation_weight": torch.tensor(relation_weight, dtype=torch.float32),
                "relation_ambiguous": torch.tensor(
                    relation_ambiguous, dtype=torch.bool
                ),
            }
        )
        return sample


class Stage4EpisodeSampler(Sampler[Stage4Request]):
    """Checkpointable 20/70/5/5 sampler with uniform task identities."""

    def __init__(
        self,
        dataset: Stage4EpisodeDataset,
        *,
        num_samples: int,
        effective_batch_size: int = 4,
        seed: int = 2027,
        start_step: int = 0,
    ) -> None:
        if num_samples <= 0 or effective_batch_size != 4:
            raise ValueError(
                "Stage4 requires positive samples and effective batch four"
            )
        if seed != 2027 or start_step < 0:
            raise ValueError("Stage4 seed/start step drifted")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.effective_batch_size = effective_batch_size
        self.seed = seed
        self._sample_cursor = start_step * effective_batch_size
        self._consumed_optimizer_step = start_step

        buckets = task_buckets(dataset.records)
        self.single_tasks = tuple(sorted(key for key in buckets if len(key) == 1))
        all_pair_tasks = tuple(sorted(key for key in buckets if len(key) == 2))
        if len(self.single_tasks) != 8 or len(all_pair_tasks) != 8:
            raise Stage4ContractError(
                "Stage4 requires eight single and eight Group-A tasks"
            )
        self.buckets = buckets
        relation_ids = set(dataset.relation_records)
        labelled: dict[tuple[str, ...], tuple[int, ...]] = {}
        for task in all_pair_tasks:
            indices = tuple(
                index
                for index in buckets[task]
                if dataset.records[index].sample_id in relation_ids
            )
            if not indices:
                raise Stage4ContractError(
                    f"no distilled Stage4 examples for pair {task}"
                )
            labelled[task] = indices
        self.pair_tasks = all_pair_tasks
        self.labelled_pair_buckets = labelled

    @property
    def step(self) -> int:
        return self._sample_cursor // self.effective_batch_size

    def mark_consumed_optimizer_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("consumed step must be non-negative")
        self._consumed_optimizer_step = int(step)

    def state_dict(
        self, *, consumed_optimizer_step: int | None = None
    ) -> dict[str, Any]:
        if consumed_optimizer_step is not None:
            self.mark_consumed_optimizer_step(consumed_optimizer_step)
        return {
            "schema_version": STAGE4_SCHEMA,
            "stage": "stage4",
            "seed": self.seed,
            "num_samples": self.num_samples,
            "effective_batch_size": self.effective_batch_size,
            "consumed_optimizer_step": self._consumed_optimizer_step,
            "sample_cursor": self._consumed_optimizer_step * self.effective_batch_size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": STAGE4_SCHEMA,
            "stage": "stage4",
            "seed": self.seed,
            "num_samples": self.num_samples,
            "effective_batch_size": self.effective_batch_size,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise Stage4ContractError(
                    f"Stage4 sampler {key} mismatch: {state.get(key)!r} != {value!r}"
                )
        step = state.get("consumed_optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise Stage4ContractError("invalid consumed Stage4 optimizer step")
        if state.get("sample_cursor") != step * self.effective_batch_size:
            raise Stage4ContractError("Stage4 sampler cursor is not step*4")
        self._consumed_optimizer_step = step
        self._sample_cursor = step * self.effective_batch_size

    @staticmethod
    def _pick(
        rng: random.Random,
        tasks: Sequence[tuple[str, ...]],
        buckets: Mapping[tuple[str, ...], Sequence[int]],
    ) -> int:
        task = tasks[rng.randrange(len(tasks))]
        values = buckets[task]
        return int(values[rng.randrange(len(values))])

    def _request(self, cursor: int) -> Stage4Request:
        step = cursor // self.effective_batch_size
        rng = _cursor_rng(self.seed, cursor)
        draw = rng.random()
        teacher = rng.random() < teacher_forcing_probability(step)
        if draw < 0.20:
            index = self._pick(rng, self.single_tasks, self.buckets)
            return Stage4Request(index, "single_restoration", step, cursor, teacher)
        if draw < 0.90:
            index = self._pick(rng, self.pair_tasks, self.labelled_pair_buckets)
            return Stage4Request(
                index, "group_a_pair_restoration", step, cursor, teacher
            )
        if draw < 0.95:
            # Any frozen single recipe provides a clean image.  Skill choice is
            # independent and uniform; sample_without_replacement handles 1/2.
            index = self._pick(rng, self.single_tasks, self.buckets)
            count = 1 + rng.randrange(2)
            forced = tuple(sorted(rng.sample(range(len(SKILLS)), count)))
            return Stage4Request(index, "clean_misuse", step, cursor, False, forced)
        true_skill = rng.randrange(len(SKILLS))
        task = next(
            key
            for key in self.single_tasks
            if self.dataset.records[self.buckets[key][0]].skill_ids[0] == true_skill
        )
        values = self.buckets[task]
        index = int(values[rng.randrange(len(values))])
        wrong_draw = rng.randrange(len(SKILLS) - 1)
        wrong_skill = wrong_draw + int(wrong_draw >= true_skill)
        return Stage4Request(index, "wrong_skill", step, cursor, False, (wrong_skill,))

    def __iter__(self) -> Iterator[Stage4Request]:
        for _ in range(self.num_samples):
            cursor = self._sample_cursor
            self._sample_cursor += 1
            yield self._request(cursor)

    def __len__(self) -> int:
        return self.num_samples


def _mapping_of_tensors(value: object, *, field: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise Stage4ContractError(f"{field} must be a non-empty tensor mapping")
    if any(
        not isinstance(key, str) or not torch.is_tensor(item)
        for key, item in value.items()
    ):
        raise Stage4ContractError(f"{field} contains non-tensor entries")
    return value  # type: ignore[return-value]


def _flatten_values(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten_values(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _flatten_values(item)
    else:
        yield value


def validate_stage3_approval(
    approval_path: str | Path,
    *,
    stage2_decision_path: str | Path | None = None,
) -> Mapping[str, Any]:
    path = Path(approval_path).resolve()
    if not path.is_file():
        raise Stage4ContractError(
            f"Stage4 is forbidden before explicit Stage3 approval: {path}"
        )
    approval = load_json(path)
    if not isinstance(approval, Mapping):
        raise Stage4ContractError("Stage3 approval must be a JSON mapping")
    if (
        approval.get("schema_version") != STAGE3_APPROVAL_SCHEMA
        or approval.get("kind") != "stage3_approval"
        or approval.get("protocol_id") != PROTOCOL_ID
        or approval.get("approved") is not True
    ):
        raise Stage4ContractError("invalid or non-approved Stage3 approval artifact")
    if stage2_decision_path is not None:
        decision = Path(stage2_decision_path).resolve()
        if not decision.is_file():
            raise Stage4ContractError(f"missing frozen Stage2 decision: {decision}")
        if approval.get("stage2_decision_sha256") != sha256_file(decision):
            raise Stage4ContractError("Stage3 approval/Stage2 decision hash mismatch")
    return approval


def validate_stage3_finalization_for_stage4(
    project_root: str | Path,
) -> tuple[Stage3RevocationAuthorization, Mapping[str, Any]]:
    """Close the finalize-only Stage3 evidence before any Stage4 CUDA work."""

    root = Path(project_root).resolve()
    try:
        authorization = validate_stage3_extension_revocation(
            root / "artifacts/approvals/STAGE3_EXTENSION_REVOKED.json",
            project_root=root,
            require_present=True,
        )
        if authorization is None:  # pragma: no cover - require_present contract
            raise Stage4ContractError("Stage3 finalization authorization is missing")
        outputs = validate_stage3_finalization_outputs(
            root,
            finalization_authorization_sha256=authorization.sha256,
            historical_extension_authorization_sha256=(
                authorization.bindings["historical_extension_authorization"]["sha256"]
            ),
        )
        report_binding = outputs.get("report")
        best_binding = outputs.get("best_checkpoint")
        if not isinstance(report_binding, Mapping) or not isinstance(
            best_binding, Mapping
        ):
            raise Stage4ContractError(
                "Stage3 finalization output validator omitted report/best bindings"
            )
        report_text = Path(str(report_binding.get("path"))).read_text(encoding="utf-8")
        required_report_fragments = (
            PROTOCOL_ID,
            authorization.sha256,
            str(best_binding.get("sha256")),
            "step12000_finalize_only_no_training",
            "optimizer / scheduler / train loader created: false / false / false",
            "checkpoint written: false",
            "MiO100 / Group B / Group C rows read: 0 / 0 / 0",
            "learned raw relation accuracy",
            "always-parallel baseline accuracy",
            "per-pair majority-prior baseline accuracy",
            "STOP-rate definition",
        )
        if any(fragment not in report_text for fragment in required_report_fragments):
            raise Stage4ContractError(
                "Stage3 finalize-only report lacks required scientific disclosures"
            )
    except Stage4ContractError:
        raise
    except Exception as exc:
        raise Stage4ContractError(
            f"Stage3 finalize-only evidence is incomplete or stale: {exc}"
        ) from exc
    return authorization, outputs


def load_presence_thresholds(
    path: str | Path,
    *,
    stage3_checkpoint_sha256: str,
    stage3_approval_sha256: str | None = None,
    stage3_extension_authorization_sha256: str | None = None,
    stage3_finalization_authorization_sha256: str | None = None,
) -> tuple[Tensor, Mapping[str, Any]]:
    threshold_path = Path(path).resolve()
    if not threshold_path.is_file():
        raise Stage4ContractError(
            f"missing frozen planner thresholds: {threshold_path}"
        )
    payload = load_json(threshold_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "graphrestore-presence-thresholds-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("frozen") is not True
        or payload.get("source") != "primary_val_presence_f1_only"
        or payload.get("calibration_runs") != 1
        or payload.get("mio100_rows_read") != 0
    ):
        raise Stage4ContractError("planner thresholds are not marked frozen")
    if payload.get("skills") != list(SKILLS):
        raise Stage4ContractError("planner threshold skill ordering drifted")
    bound_checkpoint = payload.get(
        "checkpoint_sha256", payload.get("stage3_checkpoint_sha256")
    )
    if bound_checkpoint != stage3_checkpoint_sha256:
        raise Stage4ContractError("thresholds are not bound to the Stage3 parent")
    selected = payload.get("selected_stage3_checkpoint")
    if (
        not isinstance(selected, Mapping)
        or selected.get("sha256") != stage3_checkpoint_sha256
    ):
        raise Stage4ContractError("selected Stage3 checkpoint binding drifted")
    if (
        stage3_approval_sha256 is not None
        and payload.get("stage3_approval_sha256") != stage3_approval_sha256
    ):
        raise Stage4ContractError("thresholds are not bound to current Stage3 approval")
    if (
        payload.get("stage3_extension_authorization_sha256")
        != stage3_extension_authorization_sha256
    ):
        raise Stage4ContractError(
            "thresholds are not bound to the active Stage3 extension authorization"
        )
    raw = payload.get("thresholds")
    if isinstance(raw, Mapping):
        if set(raw) != set(SKILLS):
            raise Stage4ContractError(
                "threshold mapping must contain exactly eight skills"
            )
        values = [float(raw[name]) for name in SKILLS]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [float(item) for item in raw]
    else:
        raise Stage4ContractError("thresholds must be a skill mapping or list")
    tensor = torch.tensor(values, dtype=torch.float32)
    if tuple(tensor.shape) != (len(SKILLS),) or not bool(torch.isfinite(tensor).all()):
        raise Stage4ContractError("thresholds must contain eight finite values")
    if bool(torch.any((tensor < 0.20) | (tensor > 0.80))):
        raise Stage4ContractError("thresholds escape the frozen [0.20,0.80] grid")
    grid_units = (tensor - 0.20) / 0.02
    if not torch.allclose(grid_units, grid_units.round(), atol=2.0e-5, rtol=0.0):
        raise Stage4ContractError("thresholds are not on the frozen 0.02 grid")
    expected_grid = [0.20 + 0.02 * index for index in range(31)]
    actual_grid = payload.get("search_grid")
    if not isinstance(actual_grid, Sequence) or isinstance(actual_grid, (str, bytes)):
        raise Stage4ContractError("threshold artifact lacks the frozen search grid")
    if len(actual_grid) != len(expected_grid) or any(
        not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9)
        for actual, expected in zip(actual_grid, expected_grid, strict=True)
    ):
        raise Stage4ContractError("threshold search grid drifted")
    expected_tie_break = (
        "nearest_0.50_then_higher_threshold"
        if stage3_finalization_authorization_sha256 is not None
        else "lowest_threshold"
    )
    if payload.get("tie_break") != expected_tie_break:
        raise Stage4ContractError("threshold tie-break drifted")
    if stage3_finalization_authorization_sha256 is not None:
        if (
            payload.get("stage3_finalization_authorization_sha256")
            != stage3_finalization_authorization_sha256
            or payload.get("stage3_extension_authorization_sha256")
            != stage3_extension_authorization_sha256
        ):
            raise Stage4ContractError(
                "thresholds are not bound to the Stage3 finalize-only authorization"
            )
        per_skill = payload.get("per_skill_metrics")
        if not isinstance(per_skill, Mapping) or set(per_skill) != set(SKILLS):
            raise Stage4ContractError(
                "finalized thresholds lack exact per-skill baseline/calibrated metrics"
            )
        tolerance = payload.get("numerical_tolerance", 1.0e-12)
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise Stage4ContractError("threshold numerical tolerance is invalid")
        for skill, threshold in zip(SKILLS, values, strict=True):
            metrics = per_skill.get(skill)
            if not isinstance(metrics, Mapping) or set(metrics) != {
                "baseline",
                "calibrated",
            }:
                raise Stage4ContractError(
                    f"threshold metrics schema drifted for {skill}"
                )
            baseline = metrics.get("baseline")
            calibrated = metrics.get("calibrated")
            for label, metric, expected_threshold in (
                ("baseline", baseline, 0.50),
                ("calibrated", calibrated, threshold),
            ):
                if not isinstance(metric, Mapping) or set(metric) != {
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                }:
                    raise Stage4ContractError(
                        f"threshold {label} metrics schema drifted for {skill}"
                    )
                numeric = tuple(metric.get(name) for name in metric)
                if any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in numeric
                ) or not math.isclose(
                    float(metric["threshold"]),
                    float(expected_threshold),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise Stage4ContractError(
                        f"threshold {label} metrics are non-finite/drifted for {skill}"
                    )
            if float(calibrated["f1"]) < (float(baseline["f1"]) - float(tolerance)):
                raise Stage4ContractError(
                    f"calibrated F1 regressed below baseline for {skill}"
                )
        for key in ("macro_f1_before", "macro_f1_after"):
            value = payload.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise Stage4ContractError(f"threshold {key} is not finite")
    return tensor, payload


def _reject_stage3_extension_symlink_chain(path: Path, *, field: str) -> None:
    """Reject symlinks before resolving an extension-bound artifact path."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise Stage4ContractError(
                f"Stage3 extension {field} path contains a symlink: {current}"
            )


def _stage3_extension_binding(
    value: object,
    *,
    field: str,
    expected_path: Path | None = None,
    require_read_only: bool = False,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise Stage4ContractError(
            f"Stage3 extension {field} must contain only path/sha256"
        )
    raw_path = value.get("path")
    expected_sha = value.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not Path(raw_path).is_absolute()
    ):
        raise Stage4ContractError(
            f"Stage3 extension {field}.path must be an absolute canonical path"
        )
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise Stage4ContractError(
            f"Stage3 extension {field}.sha256 is not a lowercase SHA256"
        )
    path = Path(raw_path)
    _reject_stage3_extension_symlink_chain(path, field=field)
    canonical = path.resolve(strict=False)
    if str(canonical) != raw_path:
        raise Stage4ContractError(
            f"Stage3 extension {field}.path is not lexically canonical"
        )
    if expected_path is not None and canonical != expected_path.resolve(strict=False):
        raise Stage4ContractError(f"Stage3 extension {field}.path drifted")
    if not canonical.is_file():
        raise Stage4ContractError(f"Stage3 extension {field} file is missing")
    if require_read_only and stat.S_IMODE(canonical.stat().st_mode) != 0o444:
        raise Stage4ContractError(
            f"Stage3 extension {field} immutable backup mode must be 0444"
        )
    if sha256_file(canonical) != expected_sha:
        raise Stage4ContractError(f"Stage3 extension {field} hash drifted")
    return canonical, expected_sha


def _reject_stage4_extension_symlink_chain(path: Path, *, field: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise Stage4ContractError(
                f"Stage4 extension {field} path contains a symlink: {current}"
            )


def _stage4_extension_file_binding(
    value: object,
    *,
    field: str,
    expected_path: Path,
    require_read_only: bool = False,
    verify_content: bool = True,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise Stage4ContractError(
            f"Stage4 extension {field} must contain only path/sha256"
        )
    raw_path, expected_sha = value.get("path"), value.get("sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise Stage4ContractError(f"Stage4 extension {field}.path must be absolute")
    if not isinstance(expected_sha, str) or not is_sha256(expected_sha):
        raise Stage4ContractError(
            f"Stage4 extension {field}.sha256 is not a lowercase SHA256"
        )
    path = Path(raw_path)
    _reject_stage4_extension_symlink_chain(path, field=field)
    canonical = path.resolve(strict=False)
    if (
        str(canonical) != raw_path
        or canonical != expected_path.resolve(strict=False)
        or path.is_symlink()
        or not canonical.is_file()
    ):
        raise Stage4ContractError(f"Stage4 extension {field} path drifted")
    if require_read_only and stat.S_IMODE(canonical.stat().st_mode) != 0o444:
        raise Stage4ContractError(
            f"Stage4 extension {field} immutable mode must be 0444"
        )
    if verify_content and sha256_file(canonical) != expected_sha:
        raise Stage4ContractError(f"Stage4 extension {field} hash drifted")
    return canonical, expected_sha


def _stage4_extension_decimal(raw: object, *, field: str) -> tuple[str, Decimal]:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise Stage4ContractError(
            f"Stage4 extension {field} must be a non-empty Decimal lexeme"
        )
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise Stage4ContractError(f"Stage4 extension {field} is not Decimal") from exc
    if not value.is_finite():
        raise Stage4ContractError(f"Stage4 extension {field} is not finite")
    return raw, value


def validate_stage4_extension_authorization(
    authorization_path: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> Stage4ExtensionEvidence:
    """Validate the activated one-shot Stage4 40k -> 48k authorization.

    This validator is deliberately CPU-only and runs before model/CUDA state is
    installed.  It trusts neither the narrated decision nor binary floating
    point: the 40k-minus-36k Group-A PSNR gate is recomputed from the immutable
    CSV snapshot with Decimal precision 80.
    """

    if torch.cuda.is_initialized():
        raise Stage4ContractError(
            "Stage4 extension authorization must be verified before CUDA init"
        )
    root = Path(project_root).resolve()
    config = Path(config_path).resolve()
    conditional_path = (
        root / "artifacts/approvals" / STAGE4_EXTENSION_CONDITIONAL_FILENAME
    ).resolve(strict=False)
    gate_path = (root / "artifacts/approvals" / STAGE4_EXTENSION_GATE_FILENAME).resolve(
        strict=False
    )
    requested = Path(authorization_path)
    if not requested.is_absolute():
        raise Stage4ContractError(
            "Stage4 extension gate receipt must use its absolute canonical path"
        )
    _reject_stage4_extension_symlink_chain(requested, field="gate receipt")
    if requested != gate_path or requested.resolve(strict=False) != requested:
        raise Stage4ContractError("Stage4 extension gate receipt path is not canonical")
    _, gate_sha = _stage4_extension_file_binding(
        {"path": str(gate_path), "sha256": sha256_file(gate_path)},
        field="gate receipt",
        expected_path=gate_path,
        require_read_only=True,
    )
    gate_value = load_json(gate_path)
    if not isinstance(gate_value, Mapping) or sha256_file(gate_path) != gate_sha:
        raise Stage4ContractError("Stage4 extension gate receipt changed while loading")
    gate = dict(gate_value)
    gate_keys = {
        "schema_version",
        "kind",
        "protocol_id",
        "decision",
        "created_utc",
        "conditional_authorization",
        "cycles",
        "additional_optimizer_steps",
        "base_step",
        "target_step",
        "hard_terminal_step",
        "validation_every_steps",
        "validation_steps",
        "schedule_horizon_steps",
        "min_lr",
        "lr_policy",
        "trigger_metric",
        "trigger_lhs_step",
        "trigger_rhs_step",
        "trigger_operator",
        "trigger_threshold_decimal",
        "trigger_arithmetic",
        "observed_lhs_decimal",
        "observed_rhs_decimal",
        "observed_delta_decimal",
        "exact_resume",
        "reset_optimizer",
        "reset_ema",
        "reset_scheduler",
        "reset_rng",
        "reset_sampler",
        "further_extension_authorized",
        "formal_mio100_authorized",
        "group_b_or_c_authorized",
        "snapshots",
    }
    expected_scalars: dict[str, Any] = {
        "schema_version": STAGE4_EXTENSION_GATE_SCHEMA,
        "kind": "stage4_extension_gate_receipt",
        "protocol_id": PROTOCOL_ID,
        "decision": "ACTIVATE_EXTENSION",
        "cycles": STAGE4_EXTENSION_CYCLES,
        "additional_optimizer_steps": STAGE4_EXTENSION_ADDITIONAL_OPTIMIZER_STEPS,
        "base_step": STAGE4_EXTENSION_BASE_STEP,
        "target_step": STAGE4_EXTENSION_TARGET_STEP,
        "hard_terminal_step": STAGE4_EXTENSION_TARGET_STEP,
        "validation_every_steps": STAGE4_EXTENSION_VALIDATION_EVERY_STEPS,
        "validation_steps": list(STAGE4_EXTENSION_VALIDATION_STEPS),
        "schedule_horizon_steps": STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS,
        "min_lr": STAGE4_EXTENSION_MIN_LR,
        "lr_policy": STAGE4_EXTENSION_LR_POLICY,
        "trigger_metric": STAGE4_EXTENSION_TRIGGER_METRIC,
        "trigger_lhs_step": STAGE4_EXTENSION_TRIGGER_LHS_STEP,
        "trigger_rhs_step": STAGE4_EXTENSION_TRIGGER_RHS_STEP,
        "trigger_operator": STAGE4_EXTENSION_TRIGGER_OPERATOR,
        "trigger_threshold_decimal": STAGE4_EXTENSION_THRESHOLD_DECIMAL,
        "trigger_arithmetic": STAGE4_EXTENSION_TRIGGER_ARITHMETIC,
        "exact_resume": True,
        "reset_optimizer": False,
        "reset_ema": False,
        "reset_scheduler": False,
        "reset_rng": False,
        "reset_sampler": False,
        "further_extension_authorized": False,
        "formal_mio100_authorized": False,
        "group_b_or_c_authorized": False,
    }
    if set(gate) != gate_keys or any(
        gate.get(key) != expected for key, expected in expected_scalars.items()
    ):
        raise Stage4ContractError(
            "Stage4 extension gate receipt contract drifted or was not activated"
        )
    created_utc = gate.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc.endswith("Z"):
        raise Stage4ContractError("Stage4 extension gate UTC is invalid")

    conditional_binding = gate.get("conditional_authorization")
    conditional_path, conditional_sha = _stage4_extension_file_binding(
        conditional_binding,
        field="conditional authorization",
        expected_path=conditional_path,
        require_read_only=True,
    )
    conditional_value = load_json(conditional_path)
    if (
        not isinstance(conditional_value, Mapping)
        or sha256_file(conditional_path) != conditional_sha
    ):
        raise Stage4ContractError(
            "Stage4 conditional authorization changed while loading"
        )
    conditional = dict(conditional_value)
    conditional_keys = {
        "schema_version",
        "kind",
        "protocol_id",
        "approved",
        "conditional",
        "created_utc",
        "cycles",
        "additional_optimizer_steps",
        "base_step",
        "target_step",
        "hard_terminal_step",
        "validation_every_steps",
        "validation_steps",
        "schedule_horizon_steps",
        "min_lr",
        "lr_policy",
        "exact_resume",
        "reset_optimizer",
        "reset_ema",
        "reset_scheduler",
        "reset_rng",
        "reset_sampler",
        "further_extension_authorized",
        "trigger_metric",
        "trigger_lhs_step",
        "trigger_rhs_step",
        "trigger_operator",
        "trigger_threshold_decimal",
        "trigger_arithmetic",
        "formal_mio100_authorized",
        "group_b_or_c_authorized",
        "authorized_pipeline",
        "base_stage4_config",
        "user_instruction_protocol",
        "base_stage4_run_contract",
        "preauthorization_ledger_prefix",
    }
    conditional_expected = {
        key: value
        for key, value in expected_scalars.items()
        if key
        not in {
            "schema_version",
            "kind",
            "decision",
        }
    }
    conditional_expected.update(
        {
            "schema_version": STAGE4_EXTENSION_CONDITIONAL_SCHEMA,
            "kind": "stage4_extension_conditional_approval",
            "protocol_id": PROTOCOL_ID,
            "approved": True,
            "conditional": True,
            "authorized_pipeline": [
                "stage4_extension_gate",
                "stage4_extension",
                "stage4_zero_training_diagnostics",
            ],
        }
    )
    if set(conditional) != conditional_keys or any(
        conditional.get(key) != expected
        for key, expected in conditional_expected.items()
    ):
        raise Stage4ContractError(
            "Stage4 conditional extension authorization contract drifted"
        )
    conditional_utc = conditional.get("created_utc")
    if not isinstance(conditional_utc, str) or not conditional_utc.endswith("Z"):
        raise Stage4ContractError("Stage4 conditional authorization UTC is invalid")
    _stage4_extension_file_binding(
        conditional["base_stage4_config"],
        field="base Stage4 config",
        expected_path=config,
    )
    instruction_path = root / "reports/STAGE4_CONDITIONAL_EXTENSION_PROTOCOL.md"
    _stage4_extension_file_binding(
        conditional["user_instruction_protocol"],
        field="user instruction protocol",
        expected_path=instruction_path,
    )

    snapshot_root = (
        root / "artifacts/migrations" / STAGE4_EXTENSION_BACKUP_DIR_NAME
    ).resolve(strict=False)
    raw_snapshots = gate.get("snapshots")
    if not isinstance(raw_snapshots, Mapping) or set(raw_snapshots) != set(
        STAGE4_EXTENSION_SNAPSHOT_FILENAMES
    ):
        raise Stage4ContractError("Stage4 extension snapshot label set drifted")
    snapshots: dict[str, tuple[Path, str]] = {}
    identities: set[tuple[int, int]] = set()
    for label, filename in STAGE4_EXTENSION_SNAPSHOT_FILENAMES.items():
        raw = raw_snapshots[label]
        expected_path = snapshot_root / filename
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "sha256",
            "source_mode",
            "mode",
            "device",
            "inode",
            "distinct_inode_from_live_at_creation",
        }:
            raise Stage4ContractError(
                f"Stage4 extension snapshot {label} schema drifted"
            )
        path, digest = _stage4_extension_file_binding(
            {"path": raw.get("path"), "sha256": raw.get("sha256")},
            field=f"snapshot {label}",
            expected_path=expected_path,
            require_read_only=True,
        )
        info = path.stat()
        identity = (info.st_dev, info.st_ino)
        if (
            raw.get("mode") != 0o444
            or raw.get("device") != info.st_dev
            or raw.get("inode") != info.st_ino
            or raw.get("distinct_inode_from_live_at_creation") is not True
            or isinstance(raw.get("source_mode"), bool)
            or not isinstance(raw.get("source_mode"), int)
            or identity in identities
        ):
            raise Stage4ContractError(
                f"Stage4 extension snapshot {label} identity drifted"
            )
        identities.add(identity)
        snapshots[label] = (path, digest)

    _, conditional_run_sha = _stage4_extension_file_binding(
        conditional["base_stage4_run_contract"],
        field="base Stage4 run contract",
        expected_path=root / "artifacts/checkpoints/stage4/run_contract.json",
        verify_content=False,
    )
    if conditional_run_sha != snapshots["run_contract"][1]:
        raise Stage4ContractError(
            "conditional Stage4 run-contract binding differs from its snapshot"
        )
    prefix = conditional.get("preauthorization_ledger_prefix")
    if not isinstance(prefix, Mapping) or set(prefix) != {
        "path",
        "byte_length",
        "sha256",
    }:
        raise Stage4ContractError(
            "Stage4 preauthorization ledger prefix schema drifted"
        )
    prefix_path = root / "artifacts/metrics/stage4_calibration_history.csv"
    if prefix.get("path") != str(prefix_path.resolve(strict=False)):
        raise Stage4ContractError("Stage4 preauthorization ledger prefix path drifted")
    prefix_length, prefix_sha = prefix.get("byte_length"), prefix.get("sha256")
    if (
        isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or prefix_length <= 0
        or not isinstance(prefix_sha, str)
        or not is_sha256(prefix_sha)
    ):
        raise Stage4ContractError(
            "Stage4 preauthorization ledger prefix metadata is invalid"
        )
    with snapshots["calibration_history"][0].open("rb") as handle:
        prefix_bytes = handle.read(prefix_length)
    if (
        len(prefix_bytes) != prefix_length
        or hashlib.sha256(prefix_bytes).hexdigest() != prefix_sha
    ):
        raise Stage4ContractError(
            "Stage4 ledger does not preserve its preauthorization prefix"
        )

    with snapshots["calibration_history"][0].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != STAGE4_EXTENSION_CALIBRATION_COLUMNS:
            raise Stage4ContractError(
                "Stage4 extension calibration snapshot schema drifted"
            )
        rows = [dict(row) for row in reader]
    if [int(row["step"]) for row in rows] != list(range(4_000, 40_001, 4_000)):
        raise Stage4ContractError(
            "Stage4 extension calibration snapshot is not the exact 4k..40k ledger"
        )
    by_step = {int(row["step"]): row for row in rows}
    lhs_raw, lhs = _stage4_extension_decimal(
        by_step[STAGE4_EXTENSION_TRIGGER_LHS_STEP][STAGE4_EXTENSION_TRIGGER_METRIC],
        field="step-40000 Group-A PSNR",
    )
    rhs_raw, rhs = _stage4_extension_decimal(
        by_step[STAGE4_EXTENSION_TRIGGER_RHS_STEP][STAGE4_EXTENSION_TRIGGER_METRIC],
        field="step-36000 Group-A PSNR",
    )
    gate_lhs_raw, gate_lhs = _stage4_extension_decimal(
        gate.get("observed_lhs_decimal"), field="gate lhs"
    )
    gate_rhs_raw, gate_rhs = _stage4_extension_decimal(
        gate.get("observed_rhs_decimal"), field="gate rhs"
    )
    gate_delta_raw, gate_delta = _stage4_extension_decimal(
        gate.get("observed_delta_decimal"), field="gate delta"
    )
    with localcontext() as context:
        context.prec = 80
        delta = lhs - rhs
    if (
        lhs_raw != gate_lhs_raw
        or rhs_raw != gate_rhs_raw
        or lhs != gate_lhs
        or rhs != gate_rhs
        or gate_delta != delta
        or gate_delta_raw != str(delta)
        or delta < Decimal(STAGE4_EXTENSION_THRESHOLD_DECIMAL)
    ):
        raise Stage4ContractError(
            "Stage4 extension Decimal gate is false or differs from its snapshot"
        )

    receipt_path = snapshot_root / "MIGRATION_RECEIPT.json"
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or stat.S_IMODE(receipt_path.stat().st_mode) != 0o444
    ):
        raise Stage4ContractError(
            "Stage4 extension migration COMPLETE receipt is missing or mutable"
        )
    receipt_sha = sha256_file(receipt_path)
    receipt_value = load_json(receipt_path)
    if not isinstance(receipt_value, Mapping):
        raise Stage4ContractError("Stage4 extension migration receipt is invalid")
    receipt = dict(receipt_value)
    receipt_keys = {
        "schema_version",
        "protocol_id",
        "migration",
        "status",
        "created_utc",
        "cpu_only",
        "conditional_authorization",
        "gate_receipt",
        "base_step",
        "target_step",
        "validation_steps",
        "schedule_horizon_steps",
        "lr_policy",
        "old",
        "new",
        "semantic_source_changed_paths",
        "run_contract_bit_exact_outside_provenance",
        "best_checkpoint_bit_exact_outside_provenance",
        "last_checkpoint_only_nonprovenance_change",
        "optimizer_ema_scheduler_rng_sampler_reset",
        "snapshots",
        "migration_script_sha256",
        "completed_utc",
        "backup_read_only_after_publication",
    }
    receipt_conditional = receipt.get("conditional_authorization")
    receipt_gate = receipt.get("gate_receipt")
    receipt_old = receipt.get("old")
    receipt_new = receipt.get("new")
    run_contract = root / "artifacts/checkpoints/stage4/run_contract.json"
    last_checkpoint = root / "artifacts/checkpoints/stage4/last.pth"
    best_checkpoint = root / "artifacts/checkpoints/stage4/best_ema.pth"
    live_run_value = load_json(run_contract)
    live_run = dict(live_run_value) if isinstance(live_run_value, Mapping) else None
    if (
        set(receipt) != receipt_keys
        or receipt.get("schema_version") != "graphrestore-stage4-extension-migration-v1"
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("migration") != "stage4_40000_to_48000_extension_provenance"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("cpu_only") is not True
        or receipt.get("base_step") != STAGE4_EXTENSION_BASE_STEP
        or receipt.get("target_step") != STAGE4_EXTENSION_TARGET_STEP
        or receipt.get("validation_steps") != list(STAGE4_EXTENSION_VALIDATION_STEPS)
        or receipt.get("schedule_horizon_steps")
        != STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS
        or receipt.get("lr_policy") != STAGE4_EXTENSION_LR_POLICY
        or receipt_conditional
        != {"path": str(conditional_path), "sha256": conditional_sha}
        or not isinstance(receipt_gate, Mapping)
        or receipt_gate.get("path") != str(gate_path)
        or receipt_gate.get("sha256") != gate_sha
        or receipt_gate.get("decision") != "ACTIVATE_EXTENSION"
        or not isinstance(receipt_old, Mapping)
        or set(receipt_old)
        != {
            "run_contract",
            "last_checkpoint",
            "best_checkpoint",
            "provenance_json_sha256",
        }
        or receipt_old.get("run_contract") != snapshots["run_contract"][1]
        or receipt_old.get("last_checkpoint") != snapshots["last_checkpoint"][1]
        or receipt_old.get("best_checkpoint") != snapshots["best_checkpoint"][1]
        or not isinstance(receipt_new, Mapping)
        or set(receipt_new)
        != {
            "run_contract",
            "last_checkpoint",
            "best_checkpoint",
            "provenance_json_sha256",
        }
        or receipt_new.get("run_contract") != sha256_file(run_contract)
        or receipt_new.get("last_checkpoint") != sha256_file(last_checkpoint)
        or receipt_new.get("best_checkpoint") != sha256_file(best_checkpoint)
        or live_run is None
        or not isinstance(live_run.get("provenance"), Mapping)
        or receipt_new.get("provenance_json_sha256")
        != sha256_json(dict(live_run["provenance"]))
        or receipt.get("semantic_source_changed_paths")
        != list(STAGE4_EXTENSION_ALLOWED_CHANGED_SOURCE_PATHS)
        or receipt.get("run_contract_bit_exact_outside_provenance") is not True
        or receipt.get("best_checkpoint_bit_exact_outside_provenance") is not True
        or receipt.get("last_checkpoint_only_nonprovenance_change")
        != "metrics.best_checkpoint_sha256"
        or receipt.get("optimizer_ema_scheduler_rng_sampler_reset") is not False
        or receipt.get("snapshots") != raw_snapshots
        or receipt.get("backup_read_only_after_publication") is not True
        or not isinstance(receipt.get("created_utc"), str)
        or not str(receipt["created_utc"]).endswith("Z")
        or not isinstance(receipt.get("completed_utc"), str)
        or not str(receipt["completed_utc"]).endswith("Z")
        or not is_sha256(receipt.get("migration_script_sha256"))
        or sha256_file(receipt_path) != receipt_sha
    ):
        raise Stage4ContractError(
            "Stage4 extension migration receipt/live run contract drifted"
        )
    return Stage4ExtensionEvidence(
        conditional_path=conditional_path,
        conditional_sha256=conditional_sha,
        gate_path=gate_path,
        gate_sha256=gate_sha,
    )


def _validate_stage3_extension_parent(
    checkpoint: Path,
    provenance: Mapping[str, Any],
    *,
    checkpoint_step: int,
    approval_sha256: str,
) -> Mapping[str, Any] | None:
    """Validate the user-authorized 12k->18k Stage3 extension, when present."""

    extension = provenance.get("stage3_extension")
    if extension is None:
        if checkpoint_step not in STAGE3_BASE_VALIDATION_STEPS:
            raise Stage4ContractError(
                "Stage3 checkpoint without extension is outside the original "
                "2k..12k validation boundaries"
            )
        return None
    expected_extension_keys = {
        "path",
        "sha256",
        "cycles",
        "base_step",
        "target_step",
        "validation_every_steps",
        "validation_steps",
        "schedule_horizon_steps",
        "min_lr",
        "lr_policy",
    }
    if not isinstance(extension, Mapping) or set(extension) != expected_extension_keys:
        raise Stage4ContractError(
            "Stage3 extension checkpoint provenance has an unknown/partial schema"
        )
    runtime = provenance.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or isinstance(runtime.get("max_steps"), bool)
        or not isinstance(runtime.get("max_steps"), int)
        or runtime.get("max_steps") != STAGE3_BASE_STEP
        or isinstance(runtime.get("training_target_step"), bool)
        or not isinstance(runtime.get("training_target_step"), int)
        or runtime.get("training_target_step") != STAGE3_EXTENSION_TARGET_STEP
    ):
        raise Stage4ContractError(
            "Stage3 extension runtime must preserve the 12k schedule horizon "
            "and authorize the 18k training target"
        )
    # An extended parent is accepted only from the canonical Stage3 output tree.
    if len(checkpoint.parents) < 4:
        raise Stage4ContractError("Stage3 extension checkpoint path is too shallow")
    project_root = checkpoint.parents[3]
    expected_checkpoint = (
        project_root / "artifacts/checkpoints/stage3/best_ema.pth"
    ).resolve(strict=False)
    if checkpoint != expected_checkpoint:
        raise Stage4ContractError(
            "Stage3 extension parent must be canonical artifacts/checkpoints/stage3/best_ema.pth"
        )
    extension_path = (
        project_root / "artifacts/approvals" / STAGE3_EXTENSION_APPROVAL_NAME
    )
    if extension.get("path") != str(extension_path):
        raise Stage4ContractError(
            "Stage3 extension provenance does not name the canonical approval artifact"
        )
    extension_sha = extension.get("sha256")
    if (
        not isinstance(extension_sha, str)
        or len(extension_sha) != 64
        or any(character not in "0123456789abcdef" for character in extension_sha)
    ):
        raise Stage4ContractError(
            "Stage3 extension provenance SHA is not a lowercase SHA256"
        )
    _reject_stage3_extension_symlink_chain(extension_path, field="approval artifact")
    extension_path = extension_path.resolve(strict=False)
    if not extension_path.is_file() or sha256_file(extension_path) != extension_sha:
        raise Stage4ContractError(
            "Stage3 extension approval artifact is missing or hash-drifted"
        )
    artifact = load_json(extension_path)
    if sha256_file(extension_path) != extension_sha:
        raise Stage4ContractError(
            "Stage3 extension approval artifact changed while loading"
        )
    artifact_keys = {
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
        "formal_mio100_authorized",
        "authorized_pipeline",
        "base_stage3_approval",
        "base_approval_required",
        "base_stage3_config",
        "pre_extension_run_contract",
        "pre_extension_last_checkpoint",
        "pre_extension_best_checkpoint",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != artifact_keys:
        raise Stage4ContractError(
            "Stage3 extension approval has an unknown/partial schema"
        )
    scalar_expected = {
        "schema_version": STAGE3_EXTENSION_APPROVAL_SCHEMA,
        "kind": "stage3_extension_approval",
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "cycles": 3,
        "base_step": STAGE3_BASE_STEP,
        "target_step": STAGE3_EXTENSION_TARGET_STEP,
        "validation_every_steps": STAGE3_VALIDATION_EVERY_STEPS,
        "validation_steps": list(STAGE3_EXTENSION_VALIDATION_STEPS),
        "schedule_horizon_steps": STAGE3_BASE_STEP,
        "min_lr": 2.0e-6,
        "lr_policy": STAGE3_EXTENSION_LR_POLICY,
        "formal_mio100_authorized": False,
        "authorized_pipeline": ["stage3_extension", "stage4"],
    }
    for key, expected in scalar_expected.items():
        actual = artifact.get(key)
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
            raise Stage4ContractError(f"Stage3 extension approval {key} drifted")

    provenance_expected = {
        "path": str(extension_path),
        "sha256": extension_sha,
        **{
            key: artifact[key]
            for key in (
                "cycles",
                "base_step",
                "target_step",
                "validation_every_steps",
                "validation_steps",
                "schedule_horizon_steps",
                "min_lr",
                "lr_policy",
            )
        },
    }
    if dict(extension) != provenance_expected:
        raise Stage4ContractError(
            "Stage3 extension checkpoint provenance differs from its approval"
        )
    if checkpoint_step not in (STAGE3_BASE_STEP, *STAGE3_EXTENSION_VALIDATION_STEPS):
        raise Stage4ContractError(
            "Stage3 extension best is not on an allowed 12k/14k/16k/18k boundary"
        )

    canonical_bindings = {
        "base_stage3_approval": project_root
        / "artifacts/approvals/STAGE3_APPROVED.json",
        "base_approval_required": project_root
        / "artifacts/approvals/STAGE3_APPROVAL_REQUIRED.json",
        "base_stage3_config": project_root / "configs/stage3_planner.yaml",
    }
    for field, expected_path in canonical_bindings.items():
        _, digest = _stage3_extension_binding(
            artifact[field], field=field, expected_path=expected_path
        )
        if field == "base_stage3_approval" and digest != approval_sha256:
            raise Stage4ContractError(
                "Stage3 extension approval is not bound to the active base approval"
            )

    backup_root = (
        project_root / "artifacts/migrations" / STAGE3_EXTENSION_BACKUP_DIR_NAME
    ).resolve(strict=False)
    backup_names = {
        "pre_extension_run_contract": "run_contract.json",
        "pre_extension_last_checkpoint": "last.pth",
        "pre_extension_best_checkpoint": "best_ema.pth",
    }
    backup_identities: set[tuple[int, int]] = set()
    for field, basename in backup_names.items():
        path, _ = _stage3_extension_binding(
            artifact[field],
            field=field,
            expected_path=backup_root / basename,
            require_read_only=True,
        )
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in backup_identities:
            raise Stage4ContractError(
                "Stage3 extension immutable backups alias the same file"
            )
        backup_identities.add(identity)
    return dict(extension)


def _validate_stage3_finalization_parent(
    checkpoint: Path,
    *,
    checkpoint_step: int,
    stage3_extension: Mapping[str, Any] | None,
    authorization: Stage3RevocationAuthorization | None,
) -> Mapping[str, Any] | None:
    """Require the permanent finalize-only tombstone when it exists."""

    if len(checkpoint.parents) < 4:
        raise Stage4ContractError("Stage3 checkpoint path is too shallow")
    project_root = checkpoint.parents[3]
    canonical_raw = project_root / "artifacts/approvals/STAGE3_EXTENSION_REVOKED.json"
    canonical = canonical_raw.resolve(strict=False)
    if authorization is None:
        if os.path.lexists(canonical_raw):
            raise Stage4ContractError(
                "Stage4 requires the validated Stage3 finalize-only authorization"
            )
        return None
    try:
        current = validate_stage3_extension_revocation(
            authorization.path,
            project_root=project_root,
            require_present=True,
        )
    except Exception as exc:
        raise Stage4ContractError(
            f"Stage3 finalize-only authorization is invalid: {exc}"
        ) from exc
    if (
        current.path.resolve() != canonical
        or current.sha256 != authorization.sha256
        or checkpoint_step != STAGE3_BASE_STEP
        or stage3_extension is None
    ):
        raise Stage4ContractError(
            "Stage4 finalize-only parent must be the historical extended step12000 best"
        )
    selected = authorization.bindings.get("selected_checkpoint")
    historical_extension = authorization.bindings.get(
        "historical_extension_authorization"
    )
    if (
        not isinstance(selected, Mapping)
        or set(selected) != {"path", "sha256"}
        or selected.get("path") != str(checkpoint)
        or selected.get("sha256") != sha256_file(checkpoint)
        or not isinstance(historical_extension, Mapping)
        or historical_extension.get("sha256") != stage3_extension.get("sha256")
    ):
        raise Stage4ContractError(
            "Stage3 finalize-only parent/revocation/historical extension binding drifted"
        )
    return authorization.provenance_binding()


@dataclass(frozen=True)
class FrozenStage3Snapshot:
    model: GraphRestore
    checkpoint_sha256: str
    checkpoint_step: int
    provenance: Mapping[str, Any]
    stage3_extension: Mapping[str, Any] | None = None
    stage3_finalization: Mapping[str, Any] | None = None


def load_stage3_best_ema(
    checkpoint: str | Path,
    *,
    model: GraphRestore,
    approval_sha256: str,
    required_artifact_hashes: Sequence[str] = (),
    finalization_authorization: Stage3RevocationAuthorization | None = None,
) -> FrozenStage3Snapshot:
    """Strictly load Stage3 best EMA without inheriting its optimizer state."""

    path = Path(checkpoint).resolve()
    if path.name != "best_ema.pth" or not path.is_file():
        raise Stage4ContractError(f"missing Stage3 best_ema.pth: {path}")
    digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise Stage4ContractError("Stage3 checkpoint must be a mapping")
    if payload.get("schema_version") != "graphrestore-checkpoint-v1":
        raise Stage4ContractError("Stage3 checkpoint schema mismatch")
    if payload.get("stage") != "stage3":
        raise Stage4ContractError("Stage4 parent must have stage='stage3'")
    if (
        payload.get("model_role") != "ema_selection"
        or payload.get("resumable") is not False
        or payload.get("pending_validation_step") is not None
        or payload.get("optimizer_transaction_active") is not False
        or payload.get("scaler") is not None
        or payload.get("amp") != {"dtype": "bfloat16", "scaler_required": False}
    ):
        raise Stage4ContractError(
            "Stage4 parent must be a non-resumable Stage3 EMA selection"
        )
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise Stage4ContractError("Stage3 checkpoint has invalid step")
    state = _mapping_of_tensors(payload.get("model"), field="Stage3 model")
    ema = payload.get("ema")
    if not isinstance(ema, Mapping):
        raise Stage4ContractError("Stage3 best checkpoint lacks EMA state")
    expected_decay = 0.9999
    expected_policy = stage3_ema_policy_metadata(expected_decay)
    ema_decay = ema.get("decay")
    ema_updates = ema.get("num_updates")
    if (
        set(ema) != {"decay", "num_updates", "shadow", "scope", "policy"}
        or ema.get("scope") != STAGE3_EMA_SCOPE
        or ema.get("policy") != expected_policy
        or isinstance(ema_decay, bool)
        or not isinstance(ema_decay, (int, float))
        or not math.isfinite(float(ema_decay))
        or float(ema_decay) != expected_decay
        or isinstance(ema_updates, bool)
        or not isinstance(ema_updates, int)
        or ema_updates != step
    ):
        raise Stage4ContractError(
            "Stage3 EMA policy/decay/update count did not preserve the frozen executor"
        )
    if payload.get("executor_frozen") is not True or payload.get(
        "trainable_prefixes"
    ) != ["planner."]:
        raise Stage4ContractError(
            "Stage3 checkpoint executor/trainable boundary drifted"
        )
    shadow = _mapping_of_tensors(ema.get("shadow"), field="Stage3 EMA shadow")
    target_state = model.state_dict()
    if state.keys() != shadow.keys() or state.keys() != target_state.keys():
        raise Stage4ContractError("Stage3 best model/EMA/target keys differ")
    for name in state:
        expected_shadow_dtype = (
            torch.float32
            if target_state[name].is_floating_point()
            else target_state[name].dtype
        )
        if (
            state[name].shape != target_state[name].shape
            or state[name].dtype != target_state[name].dtype
            or shadow[name].shape != target_state[name].shape
            or shadow[name].dtype != expected_shadow_dtype
        ):
            raise Stage4ContractError(f"Stage3 best model/EMA metadata differs: {name}")
        if state[name].is_floating_point() and not bool(
            torch.isfinite(state[name]).all()
        ):
            raise Stage4ContractError(f"Stage3 best model is non-finite: {name}")
        if shadow[name].is_floating_point() and not bool(
            torch.isfinite(shadow[name]).all()
        ):
            raise Stage4ContractError(f"Stage3 best EMA shadow is non-finite: {name}")
        if not torch.equal(state[name], shadow[name]):
            raise Stage4ContractError(
                f"Stage3 best model is not its EMA snapshot: {name}"
            )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage4ContractError("Stage3 checkpoint lacks provenance")
    if provenance.get("ema_policy") != expected_policy:
        raise Stage4ContractError("Stage3 checkpoint provenance EMA policy drifted")
    stage3_extension = _validate_stage3_extension_parent(
        path,
        provenance,
        checkpoint_step=step,
        approval_sha256=approval_sha256,
    )
    stage3_finalization = _validate_stage3_finalization_parent(
        path,
        checkpoint_step=step,
        stage3_extension=stage3_extension,
        authorization=finalization_authorization,
    )
    flat_values = set(_flatten_values(provenance))
    if approval_sha256 not in flat_values:
        raise Stage4ContractError("Stage3 checkpoint is not bound to current approval")
    for artifact_hash in required_artifact_hashes:
        if artifact_hash not in flat_values:
            raise Stage4ContractError(
                f"Stage3 checkpoint lacks required artifact binding: {artifact_hash}"
            )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Stage4ContractError("Stage3 EMA did not load strictly into GraphRestore")
    if sha256_file(path) != digest:
        raise Stage4ContractError("Stage3 checkpoint changed while loading")
    return FrozenStage3Snapshot(
        model,
        digest,
        step,
        provenance,
        stage3_extension=stage3_extension,
        stage3_finalization=stage3_finalization,
    )


def stage4_parameter_role(name: str) -> str | None:
    if name.startswith("planner."):
        return "planner"
    if name.startswith("decoder.skill_bank."):
        return "skills_mixers"
    if name.startswith("decoder."):
        return "decoder_refine_head"
    if name.startswith(
        (
            "encoder.down23.",
            "encoder.level3.",
            "encoder.down34.",
            "encoder.level4.",
        )
    ):
        return "encoder34"
    return None


def set_stage4_trainability(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for name, parameter in unwrap_model(model).named_parameters():
        role = stage4_parameter_role(name)
        parameter.requires_grad_(role is not None)
        counts[f"{role or 'frozen'}:{'trainable' if role else 'frozen'}"] += (
            parameter.numel()
        )
    return dict(counts)


def _validate_stage4_fixed_ema_state(
    model: nn.Module,
    model_state: Mapping[str, Tensor],
    ema_shadow: Mapping[str, Tensor],
    *,
    context: str,
    require_frozen_live_match: bool = False,
) -> None:
    """Validate all state and preserve the frozen Stage3-derived lineage."""

    core = unwrap_model(model)
    live_state = core.state_dict()
    if (
        model_state.keys() != ema_shadow.keys()
        or model_state.keys() != live_state.keys()
    ):
        raise Stage4ContractError(f"{context} EMA/model state keys drifted")
    parameters = dict(core.named_parameters(remove_duplicate=False))
    for name, value in model_state.items():
        live = live_state[name]
        shadow = ema_shadow[name]
        expected_shadow_dtype = (
            torch.float32 if live.is_floating_point() else live.dtype
        )
        if tuple(value.shape) != tuple(live.shape) or value.dtype != live.dtype:
            raise Stage4ContractError(f"{context} raw tensor metadata drifted: {name}")
        if (
            tuple(shadow.shape) != tuple(live.shape)
            or shadow.dtype != expected_shadow_dtype
        ):
            raise Stage4ContractError(f"{context} EMA tensor metadata drifted: {name}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise Stage4ContractError(f"{context} raw tensor is non-finite: {name}")
        if shadow.is_floating_point() and not bool(torch.isfinite(shadow).all()):
            raise Stage4ContractError(f"{context} EMA tensor is non-finite: {name}")
        parameter = parameters.get(name)
        if parameter is not None and stage4_parameter_role(name) is not None:
            continue
        if not torch.equal(value.detach().to(shadow), shadow):
            kind = "buffer" if parameter is None else "frozen parameter"
            raise Stage4ContractError(f"{context} Stage4 EMA {kind} drifted: {name}")
        if require_frozen_live_match and not torch.equal(
            value.detach().cpu(), live.detach().cpu()
        ):
            kind = "buffer" if parameter is None else "frozen parameter"
            raise Stage4ContractError(
                f"{context} Stage4 checkpoint {kind} drifted from live parent: {name}"
            )


def stage4_fixed_state_digest(model: nn.Module) -> str:
    """Hash every frozen parameter and every buffer in canonical state order."""

    core = unwrap_model(model)
    parameters = dict(core.named_parameters(remove_duplicate=False))
    fixed = {
        name: value
        for name, value in core.state_dict().items()
        if name not in parameters or stage4_parameter_role(name) is None
    }
    if not fixed:
        raise Stage4ContractError("Stage4 frozen-parent state is empty")
    return _tensor_mapping_digest(fixed)


def _validate_stage4_rng_state(state: Mapping[str, Any]) -> None:
    expected_keys = {"python", "numpy", "torch_cpu"}
    if torch.cuda.is_available():
        expected_keys.add("torch_cuda_all")
    if set(state) != expected_keys:
        raise Stage4ContractError("Stage4 resume RNG state fields drifted")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        cpu_state = state["torch_cpu"]
        if not torch.is_tensor(cpu_state) or cpu_state.dtype != torch.uint8:
            raise TypeError("invalid torch CPU RNG tensor")
        torch.Generator(device="cpu").set_state(cpu_state)
        cuda_states = state.get("torch_cuda_all")
        if cuda_states is not None:
            if not isinstance(cuda_states, (list, tuple)):
                raise TypeError("invalid CUDA RNG state list")
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError("CUDA RNG state count drifted")
            current_cuda_states = torch.cuda.get_rng_state_all()
            if len(current_cuda_states) != len(cuda_states):
                raise ValueError("CUDA RNG state count drifted")
            for value, reference in zip(cuda_states, current_cuda_states, strict=True):
                if (
                    not torch.is_tensor(value)
                    or value.dtype != torch.uint8
                    or tuple(value.shape) != tuple(reference.shape)
                ):
                    raise TypeError("invalid CUDA RNG tensor")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4ContractError("Stage4 resume RNG state is invalid") from exc


def _validate_stage4_metrics(
    value: object,
    *,
    step: int,
    resumable: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage4ContractError("Stage4 checkpoint metrics must be a mapping")
    metrics = dict(value)
    current_fields = {
        "group_a_psnr",
        "group_a_ssim",
        "single_psnr",
        "single_ssim",
        "validation_step",
    }
    best_fields = {
        "best_group_a_psnr",
        "best_group_a_ssim",
        "best_single_psnr",
        "best_single_ssim",
        "best_step",
    }
    allowed = current_fields | best_fields | {"best_checkpoint_sha256"}
    if set(metrics) - allowed:
        raise Stage4ContractError("Stage4 checkpoint metrics contain unknown fields")
    has_current = bool(current_fields & set(metrics))
    has_best = bool(best_fields & set(metrics))
    if has_current != has_best or (has_current and not current_fields <= set(metrics)):
        raise Stage4ContractError("Stage4 current/best metric groups are incomplete")
    if has_best and not best_fields <= set(metrics):
        raise Stage4ContractError("Stage4 best metric group is incomplete")
    if not has_current:
        if metrics:
            raise Stage4ContractError("Stage4 checkpoint has an orphan best binding")
        return metrics
    numeric_fields = current_fields | best_fields
    if any(
        isinstance(metrics[field], bool)
        or not isinstance(metrics[field], (int, float))
        or not math.isfinite(float(metrics[field]))
        for field in numeric_fields
    ):
        raise Stage4ContractError("Stage4 checkpoint metrics are non-finite")
    validation_step = float(metrics["validation_step"])
    best_step = float(metrics["best_step"])
    if (
        not validation_step.is_integer()
        or not best_step.is_integer()
        or not 0 <= int(best_step) <= int(validation_step) <= step
    ):
        raise Stage4ContractError("Stage4 checkpoint metric step boundary drifted")
    current = ValidationScore(
        group_a_psnr=float(metrics["group_a_psnr"]),
        group_a_ssim=float(metrics["group_a_ssim"]),
        single_psnr=float(metrics["single_psnr"]),
        single_ssim=float(metrics["single_ssim"]),
        step=int(validation_step),
    )
    best = ValidationScore(
        group_a_psnr=float(metrics["best_group_a_psnr"]),
        group_a_ssim=float(metrics["best_group_a_ssim"]),
        single_psnr=float(metrics["best_single_psnr"]),
        single_ssim=float(metrics["best_single_ssim"]),
        step=int(best_step),
    )
    if is_better_checkpoint(current, best):
        raise Stage4ContractError(
            "Stage4 recorded current validation is better than its incumbent"
        )
    binding = metrics.get("best_checkpoint_sha256")
    if resumable:
        if not isinstance(binding, str) or len(binding) != 64:
            raise Stage4ContractError(
                "Stage4 resumable metrics lack best checkpoint SHA256"
            )
        try:
            int(binding, 16)
        except ValueError as exc:
            raise Stage4ContractError(
                "Stage4 best checkpoint SHA256 is malformed"
            ) from exc
    elif binding is not None:
        raise Stage4ContractError(
            "Stage4 selection checkpoint metrics must not self-bind by SHA256"
        )
    return metrics


def _validate_stage4_best_incumbent_binding(
    raw_checkpoint: Path,
    metrics: Mapping[str, Any],
    *,
    model: nn.Module,
    ema: Stage4PhaseAwareEMA,
    expected_provenance: Mapping[str, Any],
    pending_validation_step: int | None,
) -> None:
    binding = metrics.get("best_checkpoint_sha256")
    best_path = raw_checkpoint.resolve().parent / "best_ema.pth"
    if binding is None and (pending_validation_step is None or not best_path.is_file()):
        return
    if not best_path.is_file():
        raise Stage4ContractError(
            "Stage4 raw checkpoint incumbent best_ema.pth is missing"
        )
    digest = sha256_file(best_path)
    binding_matches = digest == binding
    interrupted_candidate = not binding_matches and pending_validation_step is not None
    if not binding_matches and not interrupted_candidate:
        raise Stage4ContractError(
            "Stage4 raw checkpoint incumbent SHA does not match disk best_ema.pth"
        )
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(best_payload, Mapping)
        or best_payload.get("schema_version") != "graphrestore-checkpoint-v1"
        or best_payload.get("stage") != STAGE4_CHECKPOINT_STAGE
        or best_payload.get("model_role") != "ema_selection"
        or best_payload.get("resumable") is not False
        or best_payload.get("pending_validation_step") is not None
        or best_payload.get("scaler") is not None
        or best_payload.get("amp") != {"dtype": "bfloat16", "scaler_required": False}
    ):
        raise Stage4ContractError("Stage4 disk incumbent role metadata drifted")
    disk_step = best_payload.get("step")
    if isinstance(disk_step, bool) or not isinstance(disk_step, int) or disk_step < 0:
        raise Stage4ContractError("Stage4 disk incumbent has invalid step")
    expected_disk_step = (
        pending_validation_step
        if interrupted_candidate
        else int(float(metrics["best_step"]))
    )
    if disk_step != expected_disk_step:
        raise Stage4ContractError("Stage4 disk incumbent step drifted")
    provenance = best_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage4ContractError("Stage4 disk incumbent lacks provenance")
    verify_provenance(provenance, expected_provenance)
    best_metrics = _validate_stage4_metrics(
        best_payload.get("metrics"), step=disk_step, resumable=False
    )
    if not best_metrics:
        raise Stage4ContractError("Stage4 disk incumbent lacks validation metrics")
    if binding_matches:
        for field in (
            "best_group_a_psnr",
            "best_group_a_ssim",
            "best_single_psnr",
            "best_single_ssim",
            "best_step",
        ):
            if best_metrics.get(field) != metrics.get(field):
                raise Stage4ContractError(
                    f"Stage4 raw/disk incumbent metric drifted: {field}"
                )
    elif metrics:
        raw_best = ValidationScore(
            group_a_psnr=float(metrics["best_group_a_psnr"]),
            group_a_ssim=float(metrics["best_group_a_ssim"]),
            single_psnr=float(metrics["best_single_psnr"]),
            single_ssim=float(metrics["best_single_ssim"]),
            step=int(float(metrics["best_step"])),
        )
        pending_candidate = ValidationScore(
            group_a_psnr=float(best_metrics["best_group_a_psnr"]),
            group_a_ssim=float(best_metrics["best_group_a_ssim"]),
            single_psnr=float(best_metrics["best_single_psnr"]),
            single_ssim=float(best_metrics["best_single_ssim"]),
            step=int(float(best_metrics["best_step"])),
        )
        if not is_better_checkpoint(pending_candidate, raw_best):
            raise Stage4ContractError(
                "Stage4 pending disk candidate is not better than raw incumbent"
            )
    best_ema = best_payload.get("ema")
    if not isinstance(best_ema, Mapping):
        raise Stage4ContractError("Stage4 disk incumbent lacks EMA state")
    ema.validate_state_metadata(best_ema)
    if best_ema.get("num_updates") != disk_step:
        raise Stage4ContractError("Stage4 disk incumbent EMA step drifted")
    best_model = _mapping_of_tensors(
        best_payload.get("model"), field="Stage4 disk incumbent model"
    )
    best_shadow = _mapping_of_tensors(
        best_ema.get("shadow"), field="Stage4 disk incumbent EMA shadow"
    )
    _validate_stage4_fixed_ema_state(
        model,
        best_model,
        best_shadow,
        context="Stage4 disk incumbent",
        require_frozen_live_match=True,
    )
    if any(
        not torch.equal(value.detach().to(best_shadow[name]), best_shadow[name])
        for name, value in best_model.items()
    ):
        raise Stage4ContractError("Stage4 disk incumbent model is not its EMA shadow")
    if sha256_file(best_path) != digest:
        raise Stage4ContractError("Stage4 disk incumbent changed while validating")


def _is_norm_or_bias(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith(".bias") or ".norm" in name.lower()


def build_stage4_optimizer(
    model: nn.Module,
    *,
    planner_lr: float = 5.0e-5,
    skills_lr: float = 3.0e-5,
    decoder_lr: float = 1.0e-5,
    encoder34_lr: float = 2.0e-6,
    weight_decay: float = 1.0e-4,
    fused_if_supported: bool = True,
) -> torch.optim.AdamW:
    """Build a fresh, exhaustive, role-exclusive Stage4 AdamW."""

    learning_rates = {
        "planner": planner_lr,
        "skills_mixers": skills_lr,
        "decoder_refine_head": decoder_lr,
        "encoder34": encoder34_lr,
    }
    if min(learning_rates.values()) <= 0 or weight_decay < 0:
        raise ValueError("invalid Stage4 optimizer settings")
    set_stage4_trainability(model)
    grouped: dict[tuple[str, float], list[nn.Parameter]] = defaultdict(list)
    seen: set[int] = set()
    for name, parameter in unwrap_model(model).named_parameters():
        role = stage4_parameter_role(name)
        if role is None:
            if parameter.requires_grad:
                raise Stage4ContractError(
                    f"unexpected Stage4 trainable parameter: {name}"
                )
            continue
        if not parameter.requires_grad or id(parameter) in seen:
            raise Stage4ContractError(f"invalid Stage4 parameter assignment: {name}")
        decay = 0.0 if _is_norm_or_bias(name, parameter) else weight_decay
        grouped[(role, decay)].append(parameter)
        seen.add(id(parameter))
    expected = {
        id(parameter)
        for name, parameter in unwrap_model(model).named_parameters()
        if stage4_parameter_role(name) is not None
    }
    if seen != expected:
        raise Stage4ContractError(
            "Stage4 optimizer does not cover each trainable tensor once"
        )
    present_roles = {role for role, _ in grouped}
    if present_roles != set(learning_rates):
        raise Stage4ContractError(f"Stage4 optimizer roles incomplete: {present_roles}")
    groups = [
        {
            "params": parameters,
            "lr": float(learning_rates[role]),
            "initial_lr": float(learning_rates[role]),
            "weight_decay": decay,
            "role": role,
        }
        for (role, decay), parameters in sorted(grouped.items())
    ]
    kwargs: dict[str, Any] = {"betas": (0.9, 0.999)}
    if fused_if_supported and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def _optimizer_serialized_parameter_names(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[int, str]:
    canonical_names = {
        id(parameter): name
        for name, parameter in unwrap_model(model).named_parameters()
    }
    serialized = optimizer.state_dict().get("param_groups")
    if not isinstance(serialized, list) or len(serialized) != len(
        optimizer.param_groups
    ):
        raise Stage4ContractError("Stage4 optimizer parameter-name mapping drifted")
    result: dict[int, str] = {}
    for serialized_group, live_group in zip(
        serialized, optimizer.param_groups, strict=True
    ):
        if not isinstance(serialized_group, Mapping):
            raise Stage4ContractError("Stage4 optimizer serialized group is invalid")
        serialized_ids = serialized_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(serialized_ids, list) or not isinstance(
            live_parameters, list
        ):
            raise Stage4ContractError("Stage4 optimizer parameter list is invalid")
        if len(serialized_ids) != len(live_parameters):
            raise Stage4ContractError("Stage4 optimizer parameter list size drifted")
        for serialized_id, parameter in zip(
            serialized_ids, live_parameters, strict=True
        ):
            if isinstance(serialized_id, bool) or not isinstance(serialized_id, int):
                raise Stage4ContractError("Stage4 optimizer serialized ID is invalid")
            name = canonical_names.get(id(parameter))
            if name is None or stage4_parameter_role(name) is None:
                raise Stage4ContractError(
                    "Stage4 optimizer parameter lacks a legal canonical role"
                )
            if serialized_id in result:
                raise Stage4ContractError(
                    "Stage4 optimizer serialized ID is duplicated"
                )
            result[serialized_id] = name
    return result


def _validate_stage4_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
    *,
    step: int,
) -> None:
    current = optimizer.state_dict()
    loaded_state = state.get("state")
    loaded_groups = state.get("param_groups")
    current_groups = current.get("param_groups")
    if not isinstance(loaded_state, Mapping):
        raise Stage4ContractError("Stage4 optimizer state is invalid")
    if not isinstance(loaded_groups, list) or not isinstance(current_groups, list):
        raise Stage4ContractError("Stage4 optimizer groups are invalid")
    if len(loaded_groups) != len(current_groups):
        raise Stage4ContractError("Stage4 optimizer group count drifted")
    loaded_parameter_ids: set[int] = set()
    live_parameters: dict[int, nn.Parameter] = {}
    group_by_parameter_id: dict[int, Mapping[str, Any]] = {}
    for loaded_group, current_group in zip(loaded_groups, current_groups, strict=True):
        if not isinstance(loaded_group, Mapping) or not isinstance(
            current_group, Mapping
        ):
            raise Stage4ContractError("Stage4 optimizer group is invalid")
        loaded_parameters = loaded_group.get("params")
        current_parameters = current_group.get("params")
        if not isinstance(loaded_parameters, list) or not isinstance(
            current_parameters, list
        ):
            raise Stage4ContractError("Stage4 optimizer parameter IDs are invalid")
        if loaded_parameters != current_parameters:
            raise Stage4ContractError("Stage4 optimizer parameter ID order drifted")
        if loaded_group.get("role") != current_group.get("role"):
            raise Stage4ContractError("Stage4 optimizer role drifted")
        if set(loaded_group) != set(current_group):
            raise Stage4ContractError("Stage4 optimizer group fields drifted")
        for key in set(current_group) - {"params", "lr"}:
            if loaded_group.get(key) != current_group.get(key):
                raise Stage4ContractError(
                    f"Stage4 optimizer static field drifted: {key}"
                )
        dynamic_lr = loaded_group.get("lr")
        if (
            isinstance(dynamic_lr, bool)
            or not isinstance(dynamic_lr, (int, float))
            or not math.isfinite(float(dynamic_lr))
        ):
            raise Stage4ContractError("Stage4 optimizer LR is non-finite")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in loaded_parameters
        ):
            raise Stage4ContractError("Stage4 optimizer parameter ID is invalid")
        loaded_parameter_ids.update(loaded_parameters)
    for current_group, live_group in zip(
        current_groups, optimizer.param_groups, strict=True
    ):
        for parameter_id, parameter in zip(
            current_group["params"], live_group["params"], strict=True
        ):
            if not isinstance(parameter, nn.Parameter):
                raise Stage4ContractError("Stage4 optimizer has a non-parameter")
            live_parameters[parameter_id] = parameter
            group_by_parameter_id[parameter_id] = current_group
    if step == 0 and loaded_state:
        raise Stage4ContractError("Stage4 step0 optimizer state must be empty")
    if any(
        isinstance(key, bool)
        or not isinstance(key, int)
        or key not in loaded_parameter_ids
        or not isinstance(value, Mapping)
        for key, value in loaded_state.items()
    ):
        raise Stage4ContractError("Stage4 optimizer parameter state is invalid")
    if step > 0:
        missing_dense_state = [
            parameter_id
            for parameter_id, group in group_by_parameter_id.items()
            if group.get("role") != "skills_mixers" and parameter_id not in loaded_state
        ]
        if missing_dense_state:
            raise Stage4ContractError(
                "Stage4 optimizer lacks required non-skill Adam state: "
                f"{missing_dense_state[:8]}"
            )
    for parameter_id, parameter_state in loaded_state.items():
        parameter = live_parameters[parameter_id]
        group = group_by_parameter_id[parameter_id]
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad") is True:
            expected_keys.add("max_exp_avg_sq")
        if set(parameter_state) != expected_keys:
            raise Stage4ContractError("Stage4 Adam state fields drifted")
        state_step = parameter_state["step"]
        if torch.is_tensor(state_step):
            if state_step.numel() != 1 or not bool(torch.isfinite(state_step).all()):
                raise Stage4ContractError("Stage4 Adam step is invalid")
            state_step_value = float(state_step.item())
        elif isinstance(state_step, (int, float)) and not isinstance(state_step, bool):
            state_step_value = float(state_step)
        else:
            raise Stage4ContractError("Stage4 Adam step is invalid")
        if (
            not math.isfinite(state_step_value)
            or not state_step_value.is_integer()
            or not 1 <= int(state_step_value) <= step
        ):
            raise Stage4ContractError("Stage4 Adam step is outside checkpoint bounds")
        if group.get("role") != "skills_mixers" and int(state_step_value) != step:
            raise Stage4ContractError(
                "Stage4 non-skill Adam step differs from checkpoint boundary"
            )
        for key in expected_keys - {"step"}:
            tensor = parameter_state[key]
            if (
                not torch.is_tensor(tensor)
                or tuple(tensor.shape) != tuple(parameter.shape)
                or tensor.dtype != parameter.dtype
                or not bool(torch.isfinite(tensor).all())
            ):
                raise Stage4ContractError(f"Stage4 Adam tensor state is invalid: {key}")


def _validate_stage4_optimizer_state_ledger(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    ledger: object,
    *,
    create: bool,
) -> dict[int, dict[str, Any]]:
    loaded_state = optimizer_state.get("state")
    if not isinstance(loaded_state, Mapping):
        raise Stage4ContractError("Stage4 optimizer state is invalid for ledger")
    names = _optimizer_serialized_parameter_names(model, optimizer)
    if create:
        ledger_mapping: Mapping[object, object] = {
            parameter_id: {
                "name": name,
                "role": stage4_parameter_role(name),
                "has_state": parameter_id in loaded_state,
            }
            for parameter_id, name in names.items()
        }
    elif isinstance(ledger, Mapping):
        ledger_mapping = ledger
    else:
        raise Stage4ContractError("Stage4 optimizer state-name ledger is invalid")
    if set(ledger_mapping) != set(names):
        raise Stage4ContractError(
            "Stage4 optimizer ledger keys differ from optimizer parameter IDs"
        )
    normalized: dict[int, dict[str, Any]] = {}
    for parameter_id, expected_name in names.items():
        value = ledger_mapping[parameter_id]
        if not isinstance(value, Mapping) or set(value) != {
            "name",
            "role",
            "has_state",
        }:
            raise Stage4ContractError("Stage4 optimizer ledger entry is invalid")
        expected_role = stage4_parameter_role(expected_name)
        if (
            value.get("name") != expected_name
            or value.get("role") != expected_role
            or not isinstance(value.get("has_state"), bool)
        ):
            raise Stage4ContractError(
                f"Stage4 optimizer ledger name/role drifted at ID {parameter_id}"
            )
        if value["has_state"] is not (parameter_id in loaded_state):
            raise Stage4ContractError(
                f"Stage4 optimizer ledger state-presence drifted at ID {parameter_id}"
            )
        normalized[parameter_id] = dict(value)
    return normalized


def _validate_stage4_scheduler_state(
    scheduler: WarmupCosineScheduler,
    state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    *,
    step: int,
) -> None:
    current = scheduler.state_dict()
    if set(state) != set(current):
        raise Stage4ContractError("Stage4 scheduler state fields drifted")
    dynamic_fields = {"last_epoch", "_step_count", "_last_lr"}
    for key in set(current) - dynamic_fields:
        if state.get(key) != current.get(key):
            raise Stage4ContractError(f"Stage4 scheduler {key} drifted")
    if state.get("last_epoch") != step or state.get("_step_count") != step + 1:
        raise Stage4ContractError(
            "Stage4 scheduler epoch/count must match checkpoint boundary"
        )
    base_lrs = state.get("base_lrs")
    last_lrs = state.get("_last_lr")
    groups = optimizer_state.get("param_groups")
    if not isinstance(base_lrs, list) or not isinstance(last_lrs, list):
        raise Stage4ContractError("Stage4 scheduler LR state is invalid")
    if not isinstance(groups, list) or not len(base_lrs) == len(last_lrs) == len(
        groups
    ):
        raise Stage4ContractError("Stage4 scheduler LR group count drifted")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (*base_lrs, *last_lrs)
    ):
        raise Stage4ContractError("Stage4 scheduler LR state is non-finite")
    for index, (base_lr, last_lr, group) in enumerate(
        zip(base_lrs, last_lrs, groups, strict=True)
    ):
        if not isinstance(group, Mapping) or group.get("initial_lr") != base_lr:
            raise Stage4ContractError(
                f"Stage4 scheduler base LR drifted at group {index}"
            )
        warmup_steps = int(state["warmup_steps"])
        max_steps = int(state["max_steps"])
        min_lr = float(state["min_lr"])
        floor = min(min_lr, float(base_lr))
        if warmup_steps and step < warmup_steps:
            expected_lr = float(base_lr) * (float(step + 1) / float(warmup_steps))
        else:
            progress = min(
                1.0,
                max(0.0, (step - warmup_steps) / (max_steps - warmup_steps)),
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            expected_lr = floor + (float(base_lr) - floor) * cosine
        dynamic_lr = group.get("lr")
        if dynamic_lr != last_lr or dynamic_lr != expected_lr:
            raise Stage4ContractError(
                f"Stage4 optimizer/scheduler LR trajectory drifted at group {index}"
            )


def _validate_stage4_sampler_state(
    sampler: Stage4EpisodeSampler,
    state: Mapping[str, Any],
    *,
    step: int,
) -> None:
    current = sampler.state_dict()
    for key in (
        "schema_version",
        "stage",
        "seed",
        "num_samples",
        "effective_batch_size",
    ):
        if state.get(key) != current.get(key):
            raise Stage4ContractError(f"Stage4 resume sampler {key} drifted")
    if state.get("consumed_optimizer_step") != step:
        raise Stage4ContractError("Stage4 resume sampler consumed step drifted")
    if state.get("sample_cursor") != step * 4:
        raise Stage4ContractError("Stage4 resume sampler cursor drifted")


def _require_tensor(batch: Mapping[str, Any], key: str, device: torch.device) -> Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise Stage4ContractError(f"Stage4 batch field {key!r} must be a tensor")
    return value.to(device=device, non_blocking=device.type == "cuda")


@dataclass(frozen=True)
class Stage4Batch:
    input: Tensor
    target: Tensor
    gt_clean: Tensor
    target_after_i: Tensor
    target_after_j: Tensor
    only_i: Tensor
    only_j: Tensor
    guard_targets: Tensor
    global_severity_targets: Tensor
    presence_target: Tensor
    dense_guard_mask: Tensor
    global_guard_mask: Tensor
    present_skill_ids: Tensor
    forced_skill_mask: Tensor
    use_teacher: Tensor
    relation_row: Tensor
    relation_label: Tensor
    relation_weight: Tensor
    relation_ambiguous: Tensor
    episode_types: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.input.shape[0])


def prepare_stage4_batch(batch: Mapping[str, Any], device: torch.device) -> Stage4Batch:
    float_fields = (
        "input",
        "target",
        "gt_clean",
        "target_after_i",
        "target_after_j",
        "only_i",
        "only_j",
        "guard_targets",
        "global_severity_targets",
        "presence_target",
    )
    tensors = {key: _require_tensor(batch, key, device).float() for key in float_fields}
    bool_fields = (
        "dense_guard_mask",
        "global_guard_mask",
        "forced_skill_mask",
        "use_teacher",
        "relation_ambiguous",
    )
    bools = {key: _require_tensor(batch, key, device).bool() for key in bool_fields}
    longs = {
        key: _require_tensor(batch, key, device).long()
        for key in ("present_skill_ids", "relation_row", "relation_label")
    }
    relation_weight = _require_tensor(batch, "relation_weight", device).float()
    raw_types = batch.get("stage4_episode_type")
    if not isinstance(raw_types, (list, tuple)):
        raise Stage4ContractError("collated Stage4 episode types must be a sequence")
    episode_types = tuple(str(value) for value in raw_types)
    batch_size = int(tensors["input"].shape[0])
    if len(episode_types) != batch_size or any(
        value not in EPISODE_TYPES for value in episode_types
    ):
        raise Stage4ContractError("invalid collated Stage4 episode types")
    if tensors["input"].shape != tensors["gt_clean"].shape:
        raise Stage4ContractError("Stage4 input/clean shape mismatch")
    if tuple(bools["forced_skill_mask"].shape) != (batch_size, len(SKILLS)):
        raise Stage4ContractError("forced_skill_mask must be [B,8]")
    if tuple(tensors["presence_target"].shape) != (batch_size, len(SKILLS)):
        raise Stage4ContractError("presence_target must be [B,8]")
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"non-finite Stage4 batch tensor: {name}")
    return Stage4Batch(
        **tensors,
        **bools,
        **longs,
        relation_weight=relation_weight,
        episode_types=episode_types,
    )


def _teacher_relation_logits(batch: Stage4Batch, sample: int) -> Tensor:
    logits = batch.input.new_zeros((len(PAIR_INDICES), 3))
    row = int(batch.relation_row[sample])
    if row < 0:
        return logits
    label = int(batch.relation_label[sample])
    if bool(batch.relation_ambiguous[sample]):
        # Equal serial evidence is deliberately not a hard pseudo-direction.
        logits[row] = logits.new_tensor((10.0, 10.0, -10.0))
    elif 0 <= label < 3:
        logits[row].fill_(-20.0)
        logits[row, label] = 20.0
    else:
        raise Stage4ContractError("teacher pair lacks a legal distilled relation")
    return logits


def _compile_initial_graphs(
    model: GraphRestore,
    planner: PlannerOutput,
    batch: Stage4Batch,
    thresholds: Tensor,
) -> tuple[tuple[CompiledGraph, ...], tuple[bool, ...]]:
    graphs: list[CompiledGraph] = []
    teacher_flags: list[bool] = []
    probabilities = planner.presence_probabilities.detach()
    for sample, episode_type in enumerate(batch.episode_types):
        counterfactual = episode_type in COUNTERFACTUAL_TYPES
        use_teacher = bool(batch.use_teacher[sample]) and not counterfactual
        teacher_flags.append(use_teacher)
        if counterfactual:
            graphs.append(model.compiler.compile((), planner.relation_logits[sample]))
            continue
        if use_teacher:
            active = (
                torch.nonzero(batch.presence_target[sample] > 0.5, as_tuple=False)
                .flatten()
                .tolist()
            )
            if not 1 <= len(active) <= 2:
                raise Stage4ContractError(
                    "ordinary teacher graph must have one or two skills"
                )
            graphs.append(
                model.compiler.compile(active, _teacher_relation_logits(batch, sample))
            )
        else:
            active = model._select_active(probabilities[sample], thresholds)
            graphs.append(
                model.compiler.compile(active, planner.relation_logits[sample])
            )
    return tuple(graphs), tuple(teacher_flags)


def _remaining_target(batch: Stage4Batch, sample: int, remaining: Tensor) -> Tensor:
    if batch.episode_types[sample] in COUNTERFACTUAL_TYPES:
        return batch.input[sample]
    present = [
        int(value)
        for value in batch.present_skill_ids[sample].tolist()
        if int(value) >= 0
    ]
    remaining_ids = [skill for skill in present if bool(remaining[skill])]
    if not remaining_ids:
        return batch.gt_clean[sample]
    if len(present) == 1:
        return batch.input[sample]
    if len(remaining_ids) == 2:
        return batch.input[sample]
    if remaining_ids[0] == present[0]:
        return batch.only_i[sample]
    if remaining_ids[0] == present[1]:
        return batch.only_j[sample]
    raise Stage4ContractError("remaining subset target cannot be resolved")


def _relation_supervision(batch: Stage4Batch) -> tuple[Tensor, Tensor, Tensor]:
    labels = torch.full(
        (batch.batch_size, len(PAIR_INDICES)),
        -2,
        device=batch.input.device,
        dtype=torch.long,
    )
    weights = torch.zeros_like(labels, dtype=torch.float32)
    ambiguous = torch.zeros_like(labels, dtype=torch.bool)
    for sample in range(batch.batch_size):
        row = int(batch.relation_row[sample])
        if row < 0:
            continue
        labels[sample, row] = batch.relation_label[sample]
        weights[sample, row] = batch.relation_weight[sample]
        ambiguous[sample, row] = batch.relation_ambiguous[sample]
    return labels, weights, ambiguous


def _planner_supervision(
    planner: PlannerOutput,
    batch: Stage4Batch,
    remaining: Tensor,
    *,
    include_relations: bool,
) -> PlannerLossBreakdown:
    target_guard = batch.guard_targets * remaining[:, :, None, None].to(
        batch.guard_targets
    )
    target_severity = batch.global_severity_targets * remaining.to(
        batch.global_severity_targets
    )
    dense_mask = batch.dense_guard_mask & remaining
    global_mask = batch.global_guard_mask & remaining
    absent = ~remaining
    guard = guard_supervision_loss(
        planner.guard_logits,
        target_guard,
        target_severity,
        dense_skill_mask=dense_mask,
        global_skill_mask=global_mask,
        absent_skill_mask=absent,
    )
    stop_target = (~remaining.any(dim=1)).to(planner.stop_logit.dtype)[:, None]
    if include_relations:
        labels, weights, ambiguous = _relation_supervision(batch)
        return planner_loss(
            presence_logits=planner.presence_logits,
            presence_targets=remaining.to(planner.presence_logits),
            # Preserve the final ambiguity ruling's mandatory FP32
            # log_softmax/logsumexp path under the outer BF16 autocast.
            relation_logits=planner.relation_logits.float(),
            relation_targets=labels,
            relation_weights=weights,
            relation_ambiguous_mask=ambiguous,
            stop_logits=planner.stop_logit,
            stop_targets=stop_target,
            guard=guard,
        )
    presence = focal_binary_cross_entropy(
        planner.presence_logits, remaining.to(planner.presence_logits), gamma=2.0
    )
    stop = F.binary_cross_entropy_with_logits(planner.stop_logit, stop_target)
    zero = planner.presence_logits.sum() * 0.0
    total = presence + 0.5 * guard.total + 0.25 * stop
    return PlannerLossBreakdown(total, presence, guard.total, zero, stop, zero)


@dataclass(frozen=True)
class Stage4RoundDiagnostics:
    round_index: int
    active_skill_counts: tuple[int, ...]
    active_sample_count: int
    skipped_node_count: int
    guard_mean_per_skill: tuple[float, ...]
    guard_max_per_skill: tuple[float, ...]
    union_guard_mean: float
    union_guard_std: float
    union_guard_high_fraction: float
    rgb_residual_norm: float
    identity_fraction: float


@dataclass(frozen=True)
class Stage4ProgramOutput:
    final: Tensor
    step_images: tuple[Tensor, ...]
    step_targets: tuple[Tensor, ...]
    step_valid_masks: tuple[Tensor, ...]
    planner_losses: tuple[PlannerLossBreakdown, ...]
    compiled_graphs: tuple[CompiledGraph, ...]
    graph_states: tuple[ProgramGraphState, ...]
    teacher_flags: tuple[bool, ...]
    executed_masks: tuple[Tensor, ...]
    round_diagnostics: tuple[Stage4RoundDiagnostics, ...]
    reentry_request_count: int
    unexpected_activation_count: int


def run_stage4_program(
    model: GraphRestore,
    batch: Stage4Batch,
    *,
    presence_thresholds: Tensor | Sequence[float] | None = None,
) -> Stage4ProgramOutput:
    """Run a two-round Stage4 trajectory with one t=0 compilation per sample."""

    core = unwrap_model(model)
    if not isinstance(core, GraphRestore):
        raise TypeError("Stage4 program requires GraphRestore")
    padded, original_shape = pad_to_multiple(batch.input, 8)
    current = padded
    x0 = padded
    features = core.encode(current)
    planner = core.plan_state(
        x0, current, features, round_value=0.0, compute_relations=True
    )
    thresholds = core._threshold_tensor(presence_thresholds, planner.presence_logits)
    compiled, teacher_flags = _compile_initial_graphs(core, planner, batch, thresholds)
    states = [ProgramGraphState.from_compiled(graph) for graph in compiled]
    remaining = batch.presence_target > 0.5
    initial_graph_mask = torch.zeros_like(remaining)
    for sample, graph in enumerate(compiled):
        for skill in graph.active_skills:
            initial_graph_mask[sample, SKILL_TO_INDEX[skill]] = True

    planner_losses: list[PlannerLossBreakdown] = []
    step_images: list[Tensor] = []
    step_targets: list[Tensor] = []
    step_valid: list[Tensor] = []
    executed_masks: list[Tensor] = []
    round_diagnostics: list[Stage4RoundDiagnostics] = []
    executed = torch.zeros_like(remaining)
    terminal = torch.zeros(
        batch.batch_size, device=batch.input.device, dtype=torch.bool
    )
    reentry_count = 0
    unexpected_count = 0

    for round_index in range(2):
        planner_losses.append(
            _planner_supervision(
                planner,
                batch,
                remaining,
                include_relations=round_index == 0,
            )
        )
        probabilities = planner.presence_probabilities.detach()
        above = probabilities >= thresholds[None, :]
        reentry_count += int((executed & above).sum().item())
        unexpected_count += int(((~initial_graph_mask) & above).sum().item())
        active = torch.zeros_like(remaining)
        forced_presence = torch.zeros_like(remaining)
        processed = torch.zeros(
            batch.batch_size, device=batch.input.device, dtype=torch.bool
        )
        skipped_node_count = 0

        for sample, episode_type in enumerate(batch.episode_types):
            if bool(terminal[sample]):
                continue
            if episode_type in COUNTERFACTUAL_TYPES:
                if round_index == 0:
                    active[sample] = batch.forced_skill_mask[sample]
                    forced_presence[sample] = batch.forced_skill_mask[sample]
                    processed[sample] = bool(active[sample].any())
                terminal[sample] = True
                continue
            state = states[sample]
            if state.complete:
                terminal[sample] = True
                continue
            level = state.current_level
            execute_names: list[str] = []
            skip_names: list[str] = []
            if teacher_flags[sample]:
                execute_names.extend(level)
            else:
                pending_ids = [SKILL_TO_INDEX[name] for name in state.pending]
                confident = bool(
                    pending_ids and torch.any(above[sample, pending_ids]).item()
                )
                if float(planner.stop_logit[sample].sigmoid()) >= 0.5 and not confident:
                    state.skip_all_pending()
                    terminal[sample] = True
                    continue
                for skill in level:
                    skill_id = SKILL_TO_INDEX[skill]
                    if bool(above[sample, skill_id]):
                        execute_names.append(skill)
                    else:
                        skip_names.append(skill)
            skipped_node_count += len(skip_names)
            for skill in execute_names:
                skill_id = SKILL_TO_INDEX[skill]
                active[sample, skill_id] = True
                if teacher_flags[sample]:
                    forced_presence[sample, skill_id] = True
            state.finish_current_level(executed=execute_names, skipped=skip_names)
            processed[sample] = True
            if state.complete:
                terminal[sample] = True

        guards = planner.execution_guards(forced_presence)
        execution = None
        if bool(active.any()):
            execution = core.execute_level(
                current,
                features,
                guards=guards,
                active_mask=active,
                forced_presence_mask=forced_presence,
            )
            current = execution.next_image
        executed = executed | active
        # Only a true skill changes the recipe's remaining-degradation state.
        remaining = remaining & ~active
        targets = torch.stack(
            [
                _remaining_target(batch, sample, remaining[sample])
                for sample in range(batch.batch_size)
            ]
        )
        step_images.append(crop_to_shape(current, original_shape))
        step_targets.append(targets)
        step_valid.append(processed & active.any(dim=1))
        executed_masks.append(active)
        active_samples = active.any(dim=1)
        active_skill_counts = tuple(
            int(active[:, skill].sum().item()) for skill in range(len(SKILLS))
        )
        guard_means: list[float] = []
        guard_maxima: list[float] = []
        for skill in range(len(SKILLS)):
            selected = active[:, skill]
            if bool(selected.any()):
                values = guards[selected, skill].detach().float()
                guard_means.append(float(values.mean().item()))
                guard_maxima.append(float(values.amax().item()))
            else:
                guard_means.append(0.0)
                guard_maxima.append(0.0)
        if execution is not None and bool(active_samples.any()):
            union = execution.union_guard[active_samples].detach().float()
            residual = execution.residual_norm[active_samples].detach().float()
            identity = execution.identity_mask[active_samples].detach().float()
            union_mean = float(union.mean().item())
            union_std = float(union.std(unbiased=False).item())
            union_high = float((union > 0.9).float().mean().item())
            residual_norm = float(residual.mean().item())
            identity_fraction = float(identity.mean().item())
        else:
            union_mean = union_std = union_high = residual_norm = 0.0
            identity_fraction = 1.0
        round_diagnostics.append(
            Stage4RoundDiagnostics(
                round_index=round_index,
                active_skill_counts=active_skill_counts,
                active_sample_count=int(active_samples.sum().item()),
                skipped_node_count=skipped_node_count,
                guard_mean_per_skill=tuple(guard_means),
                guard_max_per_skill=tuple(guard_maxima),
                union_guard_mean=union_mean,
                union_guard_std=union_std,
                union_guard_high_fraction=union_high,
                rgb_residual_norm=residual_norm,
                identity_fraction=identity_fraction,
            )
        )

        if round_index == 0 and not bool(terminal.all()):
            features = core.encode(current)
            planner = core.plan_state(
                x0,
                current,
                features,
                round_value=0.5,
                compute_relations=False,
            )
        elif round_index == 0:
            break

    return Stage4ProgramOutput(
        final=crop_to_shape(current, original_shape),
        step_images=tuple(step_images),
        step_targets=tuple(step_targets),
        step_valid_masks=tuple(step_valid),
        planner_losses=tuple(planner_losses),
        compiled_graphs=compiled,
        graph_states=tuple(states),
        teacher_flags=teacher_flags,
        executed_masks=tuple(executed_masks),
        round_diagnostics=tuple(round_diagnostics),
        reentry_request_count=reentry_count,
        unexpected_activation_count=unexpected_count,
    )


def _per_image_charbonnier(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("per-image Charbonnier shape mismatch")
    return torch.sqrt((prediction - target).square() + 1.0e-6).mean((1, 2, 3))


@dataclass(frozen=True)
class Stage4ImageLoss:
    total: Tensor
    final_pixel: Tensor
    step_pixel: Tensor
    final_ssim: Tensor
    noop_pixel: Tensor
    noop_ssim: Tensor
    lambda_ssim: float


def stage4_image_loss(
    program: Stage4ProgramOutput,
    batch: Stage4Batch,
    *,
    step: int,
) -> Stage4ImageLoss:
    final_pix = _per_image_charbonnier(program.final, batch.gt_clean)
    noop_pix = _per_image_charbonnier(program.final, batch.input)
    counterfactual = torch.tensor(
        [value in COUNTERFACTUAL_TYPES for value in batch.episode_types],
        device=batch.input.device,
        dtype=torch.bool,
    )
    ordinary = ~counterfactual
    # Do not build two full SSIM graphs when each sample belongs to exactly one
    # branch.  This preserves the written loss and materially lowers Stage4's
    # two-round activation peak without changing crop/effective batch.
    final_ssim = torch.zeros(batch.batch_size, device=batch.input.device)
    noop_ssim = torch.zeros_like(final_ssim)
    if bool(ordinary.any()):
        final_ssim[ordinary] = 1.0 - train_ssim_y(
            program.final[ordinary], batch.gt_clean[ordinary]
        )
    if bool(counterfactual.any()):
        noop_ssim[counterfactual] = 1.0 - train_ssim_y(
            program.final[counterfactual], batch.input[counterfactual]
        )
    step_sum = torch.zeros(batch.batch_size, device=batch.input.device)
    step_count = torch.zeros_like(step_sum)
    for prediction, target, valid in zip(
        program.step_images,
        program.step_targets,
        program.step_valid_masks,
        strict=True,
    ):
        value = _per_image_charbonnier(prediction, target)
        step_sum = step_sum + value * valid.to(value)
        step_count = step_count + valid.to(step_count)
    step_pix = step_sum / step_count.clamp_min(1.0)
    lambda_ssim = stage4_ssim_weight(step)
    ordinary_loss = final_pix + 0.30 * step_pix + lambda_ssim * final_ssim
    noop_loss = noop_pix + 0.05 * noop_ssim
    per_image = torch.where(counterfactual, noop_loss, ordinary_loss)
    zero = program.final.new_zeros(())

    def selected_mean(value: Tensor, mask: Tensor) -> Tensor:
        return value[mask].mean() if bool(mask.any()) else zero

    return Stage4ImageLoss(
        total=per_image.mean(),
        final_pixel=selected_mean(final_pix, ordinary),
        step_pixel=selected_mean(step_pix, ordinary),
        final_ssim=selected_mean(final_ssim, ordinary),
        noop_pixel=selected_mean(noop_pix, counterfactual),
        noop_ssim=selected_mean(noop_ssim, counterfactual),
        lambda_ssim=lambda_ssim,
    )


@dataclass(frozen=True)
class Stage4StepResult:
    loss: float
    image_loss: float
    planner_loss: float
    final_pixel: float
    step_pixel: float
    final_ssim: float
    noop_pixel: float
    noop_ssim: float
    lambda_ssim: float
    teacher_fraction: float
    reentry_requests: int
    unexpected_activations: int
    round_diagnostics: tuple[Mapping[str, Any], ...]
    grad_norm: float
    samples: int
    seconds: float


def _autocast(device: torch.device, use_bf16: bool):
    if use_bf16:
        if device.type != "cuda":
            raise Stage4ContractError("formal Stage4 BF16 requires CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_stage4_optimizer_step(
    model: GraphRestore,
    micro_batches: Sequence[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler | None,
    ema: Stage4PhaseAwareEMA | None,
    *,
    step: int,
    device: torch.device,
    use_bf16: bool = True,
) -> Stage4StepResult:
    if not micro_batches:
        raise ValueError("Stage4 requires at least one micro batch")
    if ema is not None and not isinstance(ema, Stage4PhaseAwareEMA):
        raise Stage4ContractError("Stage4 optimizer steps require phase-aware EMA")
    model.train()
    set_stage4_trainability(model)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    reentry = unexpected = 0
    round_diagnostics: list[Mapping[str, Any]] = []

    for micro_index, raw in enumerate(micro_batches):
        batch = prepare_stage4_batch(raw, device)
        with _autocast(device, use_bf16):
            program = run_stage4_program(model, batch)
            image = stage4_image_loss(program, batch, step=step)
            planner_total = torch.stack(
                [item.total for item in program.planner_losses]
            ).mean()
            total = image.total + 0.05 * planner_total
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError("non-finite Stage4 total loss")
        (total / len(micro_batches)).backward()
        count = batch.batch_size
        samples += count
        for key, value in (
            ("loss", total),
            ("image_loss", image.total),
            ("planner_loss", planner_total),
            ("final_pixel", image.final_pixel),
            ("step_pixel", image.step_pixel),
            ("final_ssim", image.final_ssim),
            ("noop_pixel", image.noop_pixel),
            ("noop_ssim", image.noop_ssim),
        ):
            totals[key] += float(value.detach()) * count
        totals["teacher"] += sum(program.teacher_flags)
        reentry += program.reentry_request_count
        unexpected += program.unexpected_activation_count
        for diagnostic in program.round_diagnostics:
            round_diagnostics.append(
                {
                    "micro_batch_index": micro_index,
                    "round_index": diagnostic.round_index,
                    "active_skills": {
                        SKILLS[index]: count
                        for index, count in enumerate(diagnostic.active_skill_counts)
                        if count
                    },
                    "active_sample_count": diagnostic.active_sample_count,
                    "skipped_node_count": diagnostic.skipped_node_count,
                    "guard_mean_per_skill": {
                        SKILLS[index]: value
                        for index, value in enumerate(diagnostic.guard_mean_per_skill)
                    },
                    "guard_max_per_skill": {
                        SKILLS[index]: value
                        for index, value in enumerate(diagnostic.guard_max_per_skill)
                    },
                    "union_guard_mean": diagnostic.union_guard_mean,
                    "union_guard_std": diagnostic.union_guard_std,
                    "union_guard_high_fraction": diagnostic.union_guard_high_fraction,
                    "rgb_residual_norm": diagnostic.rgb_residual_norm,
                    "identity_fraction": diagnostic.identity_fraction,
                }
            )

    if samples != 4:
        raise Stage4ContractError(f"Stage4 effective batch must be four, got {samples}")
    parameters = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    if not parameters:
        raise Stage4ContractError("Stage4 backward produced no gradients")
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=0.5, error_if_nonfinite=True
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
    return Stage4StepResult(
        loss=totals["loss"] / samples,
        image_loss=totals["image_loss"] / samples,
        planner_loss=totals["planner_loss"] / samples,
        final_pixel=totals["final_pixel"] / samples,
        step_pixel=totals["step_pixel"] / samples,
        final_ssim=totals["final_ssim"] / samples,
        noop_pixel=totals["noop_pixel"] / samples,
        noop_ssim=totals["noop_ssim"] / samples,
        lambda_ssim=stage4_ssim_weight(step),
        teacher_fraction=totals["teacher"] / samples,
        reentry_requests=reentry,
        unexpected_activations=unexpected,
        round_diagnostics=tuple(round_diagnostics),
        grad_norm=grad_norm,
        samples=samples,
        seconds=elapsed,
    )


@dataclass(frozen=True)
class Stage4MicroBatchTrial:
    crop_size: int
    micro_batch: int
    passed: bool
    images_per_second: float
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    completed_forward_backward: int
    completed_optimizer_steps: int
    error: str | None = None


def _synthetic_probe_batch(
    micro_batch: int, crop_size: int, device: torch.device
) -> Stage4Batch:
    image = torch.rand(micro_batch, 3, crop_size, crop_size, device=device)
    target = torch.rand_like(image)
    guard = torch.rand(
        micro_batch, len(SKILLS), crop_size // 4, crop_size // 4, device=device
    )
    presence = torch.zeros(micro_batch, len(SKILLS), device=device)
    presence[:, :2] = 1.0
    present = torch.tensor((0, 1), device=device).expand(micro_batch, -1).clone()
    dense = torch.tensor(
        [name in {"rain", "haze", "low_light"} for name in SKILLS],
        device=device,
    ).expand(micro_batch, -1)
    relation_row = torch.full((micro_batch,), PAIR_TO_ROW[(0, 1)], device=device)
    return Stage4Batch(
        input=image,
        target=target,
        gt_clean=target,
        target_after_i=target,
        target_after_j=target,
        only_i=target,
        only_j=target,
        guard_targets=guard,
        global_severity_targets=guard.mean((-2, -1)),
        presence_target=presence,
        dense_guard_mask=dense,
        global_guard_mask=~dense,
        present_skill_ids=present,
        forced_skill_mask=torch.zeros_like(presence, dtype=torch.bool),
        use_teacher=torch.ones(micro_batch, device=device, dtype=torch.bool),
        relation_row=relation_row,
        relation_label=torch.zeros(micro_batch, device=device, dtype=torch.long),
        relation_weight=torch.ones(micro_batch, device=device),
        relation_ambiguous=torch.zeros(micro_batch, device=device, dtype=torch.bool),
        episode_types=tuple("group_a_pair_restoration" for _ in range(micro_batch)),
    )


def _stage4_batch_as_mapping(batch: Stage4Batch) -> dict[str, Any]:
    return {
        "input": batch.input,
        "target": batch.target,
        "gt_clean": batch.gt_clean,
        "target_after_i": batch.target_after_i,
        "target_after_j": batch.target_after_j,
        "only_i": batch.only_i,
        "only_j": batch.only_j,
        "guard_targets": batch.guard_targets,
        "global_severity_targets": batch.global_severity_targets,
        "presence_target": batch.presence_target,
        "dense_guard_mask": batch.dense_guard_mask,
        "global_guard_mask": batch.global_guard_mask,
        "present_skill_ids": batch.present_skill_ids,
        "forced_skill_mask": batch.forced_skill_mask,
        "use_teacher": batch.use_teacher,
        "relation_row": batch.relation_row,
        "relation_label": batch.relation_label,
        "relation_weight": batch.relation_weight,
        "relation_ambiguous": batch.relation_ambiguous,
        "stage4_episode_type": batch.episode_types,
    }


def stage4_probe_candidate_order() -> tuple[tuple[int, int], ...]:
    """The only legal Stage4 crop/micro fallback order."""

    return ((160, 2), (160, 1), (128, 2), (128, 1))


def is_stage4_cuda_oom_exception(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if type(error) is not RuntimeError:
        return False
    message = str(error).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message


def choose_stage4_micro_batch(
    model: GraphRestore,
    *,
    device: torch.device,
    candidates: Sequence[int] = (2, 1),
    crop_size: int = 160,
    required_forward_backward: int = 10,
    maximum_reserved_fraction: float = 0.90,
) -> tuple[int, int, tuple[Stage4MicroBatchTrial, ...]]:
    """Select the first safe candidate using ten full AdamW+EMA steps.

    Every candidate starts from identical model and RNG state.  Crop128 is
    considered only after both crop160 candidates fail; no smaller crop is a
    legal Stage4 fallback.
    """

    if device.type != "cuda" or not torch.cuda.is_available():
        raise Stage4ContractError("Stage4 micro-batch selection requires CUDA")
    if (
        tuple(candidates) != (2, 1)
        or crop_size != 160
        or required_forward_backward != 10
        or maximum_reserved_fraction != 0.90
    ):
        raise Stage4ContractError("Stage4 VRAM-probe contract drifted")
    rng = capture_rng_state()
    pristine_model = {
        name: value.detach().cpu().clone()
        for name, value in unwrap_model(model).state_dict().items()
    }
    total_memory = torch.cuda.get_device_properties(device).total_memory
    if total_memory <= 0:
        raise Stage4ContractError("Stage4 VRAM probe saw invalid GPU memory")
    trials: list[Stage4MicroBatchTrial] = []
    model.train()
    selected: tuple[int, int] | None = None
    try:
        for trial_crop, micro_batch in stage4_probe_candidate_order():
            if trial_crop == 128 and any(
                trial.crop_size == 160 and trial.passed for trial in trials
            ):
                break
            unwrap_model(model).load_state_dict(pristine_model, strict=True)
            restore_rng_state(rng)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            completed = 0
            error: str | None = None
            throughput = 0.0
            peak = 0
            fraction = 1.0
            started = time.perf_counter()
            optimizer: torch.optim.Optimizer | None = None
            ema: Stage4PhaseAwareEMA | None = None
            failed_with_cuda_oom = False
            try:
                optimizer = build_stage4_optimizer(model)
                ema = build_stage4_ema(model, decay=0.9999)
                accumulation = 4 // micro_batch
                for probe_step in range(required_forward_backward):
                    micro_batches = [
                        _stage4_batch_as_mapping(
                            _synthetic_probe_batch(micro_batch, trial_crop, device)
                        )
                        for _ in range(accumulation)
                    ]
                    train_stage4_optimizer_step(
                        model,
                        micro_batches,
                        optimizer,
                        None,
                        ema,
                        step=12_000 + probe_step,
                        device=device,
                        use_bf16=True,
                    )
                    completed += 1
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                throughput = 4 * completed / max(elapsed, 1.0e-9)
                passed = completed == required_forward_backward and fraction <= 0.90
                if not passed:
                    error = f"peak reserved fraction {fraction:.4f} exceeds 0.90"
            except RuntimeError as exc:
                if not is_stage4_cuda_oom_exception(exc):
                    raise
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                passed = False
                failed_with_cuda_oom = True
                error = f"{type(exc).__name__}: {exc}"
            finally:
                model.zero_grad(set_to_none=True)
                micro_batches = None
                optimizer = ema = None
                torch.cuda.empty_cache()
            trials.append(
                Stage4MicroBatchTrial(
                    crop_size=trial_crop,
                    micro_batch=micro_batch,
                    passed=passed,
                    images_per_second=throughput,
                    peak_reserved_bytes=peak,
                    peak_reserved_fraction=fraction,
                    completed_forward_backward=completed,
                    completed_optimizer_steps=completed,
                    error=error,
                )
            )
            if passed:
                selected = (trial_crop, micro_batch)
                break
            if (trial_crop, micro_batch) == (160, 1) and not failed_with_cuda_oom:
                raise Stage4ContractError(
                    "Stage4 crop160/micro1 completed above the 0.90 VRAM ceiling; "
                    "crop128 fallback is authorized only after a true CUDA OOM"
                )
    finally:
        model.zero_grad(set_to_none=True)
        unwrap_model(model).load_state_dict(pristine_model, strict=True)
        restore_rng_state(rng)
        torch.cuda.empty_cache()
    if selected is None:
        raise Stage4ContractError(
            "no legal Stage4 crop/micro candidate passed the 10-step VRAM gate"
        )
    selected_crop, selected_micro = selected
    return selected_crop, selected_micro, tuple(trials)


@dataclass(frozen=True)
class Stage4ValidationVRAMTopology:
    compiler_mode: str
    active_skill_count: int
    completed_rounds: int
    active_skill_counts_by_round: tuple[int, ...]
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    finite: bool
    passed: bool


@dataclass(frozen=True)
class Stage4ValidationVRAMGate:
    image_size: int
    max_rounds: int
    completed_rounds: int
    topologies: tuple[Stage4ValidationVRAMTopology, ...]
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    maximum_peak_reserved_fraction: float
    resident_optimizer_state_entries: int
    resident_optimizer_state_bytes: int
    resident_ema_bytes: int
    optimizer_state_empty_after: bool
    finite: bool
    passed: bool


def stage4_runtime_evidence_metadata(
    micro_batch_trials: object,
    validation_vram_gate: object,
    *,
    selected_crop_size: int,
    selected_micro_batch: int,
) -> dict[str, Any]:
    """Validate and hash the CUDA evidence frozen by the Stage4 run contract.

    Resume never reruns either expensive CUDA gate.  Instead, the raw
    checkpoint binds these canonical hashes in provenance and the run contract
    must reproduce the same strictly validated evidence before any checkpoint
    state can be installed.
    """

    trial_fields = {
        "crop_size",
        "micro_batch",
        "passed",
        "images_per_second",
        "peak_reserved_bytes",
        "peak_reserved_fraction",
        "completed_forward_backward",
        "completed_optimizer_steps",
        "error",
    }
    if not isinstance(micro_batch_trials, Sequence) or isinstance(
        micro_batch_trials, (str, bytes)
    ):
        raise Stage4ContractError("Stage4 micro-batch trial evidence is invalid")
    if not 1 <= len(micro_batch_trials) <= len(stage4_probe_candidate_order()):
        raise Stage4ContractError("Stage4 micro-batch trial count is invalid")
    normalized_trials: list[dict[str, Any]] = []
    for index, raw in enumerate(micro_batch_trials):
        if not isinstance(raw, Mapping) or set(raw) != trial_fields:
            raise Stage4ContractError("Stage4 micro-batch trial schema drifted")
        expected_crop, expected_micro = stage4_probe_candidate_order()[index]
        crop = raw.get("crop_size")
        micro = raw.get("micro_batch")
        passed = raw.get("passed")
        throughput = raw.get("images_per_second")
        peak_bytes = raw.get("peak_reserved_bytes")
        peak_fraction = raw.get("peak_reserved_fraction")
        completed = raw.get("completed_forward_backward")
        completed_steps = raw.get("completed_optimizer_steps")
        error = raw.get("error")
        if (
            isinstance(crop, bool)
            or not isinstance(crop, int)
            or crop != expected_crop
            or isinstance(micro, bool)
            or not isinstance(micro, int)
            or micro != expected_micro
            or not isinstance(passed, bool)
            or isinstance(throughput, bool)
            or not isinstance(throughput, (int, float))
            or not math.isfinite(float(throughput))
            or float(throughput) < 0.0
            or isinstance(peak_bytes, bool)
            or not isinstance(peak_bytes, int)
            or peak_bytes < 0
            or isinstance(peak_fraction, bool)
            or not isinstance(peak_fraction, (int, float))
            or not math.isfinite(float(peak_fraction))
            or not 0.0 <= float(peak_fraction) <= 1.0
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 0 <= completed <= 10
            or isinstance(completed_steps, bool)
            or not isinstance(completed_steps, int)
            or completed_steps != completed
            or (error is not None and (not isinstance(error, str) or not error))
        ):
            raise Stage4ContractError("Stage4 micro-batch trial values drifted")
        expected_pass = (
            completed == 10
            and float(peak_fraction) <= 0.90
            and error is None
            and float(throughput) > 0.0
        )
        if passed is not expected_pass:
            raise Stage4ContractError("Stage4 micro-batch trial pass semantics drifted")
        if not passed:
            oom = error is not None and "out of memory" in error.lower()
            completed_over_limit = (
                completed == 10
                and float(peak_fraction) > 0.90
                and float(throughput) > 0.0
                and error is not None
                and "peak reserved fraction" in error
            )
            if not (oom or completed_over_limit):
                raise Stage4ContractError(
                    "Stage4 failed micro-batch trial lacks an authorized reason"
                )
        normalized_trials.append(
            {
                "crop_size": int(crop),
                "micro_batch": int(micro),
                "passed": passed,
                "images_per_second": float(throughput),
                "peak_reserved_bytes": peak_bytes,
                "peak_reserved_fraction": float(peak_fraction),
                "completed_forward_backward": completed,
                "completed_optimizer_steps": completed_steps,
                "error": error,
            }
        )
    selected = normalized_trials[-1]
    if (
        isinstance(selected_crop_size, bool)
        or not isinstance(selected_crop_size, int)
        or isinstance(selected_micro_batch, bool)
        or not isinstance(selected_micro_batch, int)
        or not selected["passed"]
        or selected["crop_size"] != selected_crop_size
        or selected["micro_batch"] != selected_micro_batch
        or any(trial["passed"] for trial in normalized_trials[:-1])
    ):
        raise Stage4ContractError(
            "Stage4 selected crop/micro is not the first passing trial"
        )
    if selected_crop_size == 128:
        if (
            len(normalized_trials) < 3
            or "out of memory" not in str(normalized_trials[1]["error"]).lower()
        ):
            raise Stage4ContractError(
                "Stage4 crop128 requires a true crop160/micro1 CUDA OOM"
            )

    gate_fields = {
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
    topology_fields = {
        "compiler_mode",
        "active_skill_count",
        "completed_rounds",
        "active_skill_counts_by_round",
        "peak_reserved_bytes",
        "peak_reserved_fraction",
        "finite",
        "passed",
    }
    if (
        not isinstance(validation_vram_gate, Mapping)
        or set(validation_vram_gate) != gate_fields
    ):
        raise Stage4ContractError("Stage4 validation VRAM gate schema drifted")
    topologies = validation_vram_gate.get("topologies")
    if not isinstance(topologies, Sequence) or isinstance(topologies, (str, bytes)):
        raise Stage4ContractError("Stage4 validation VRAM topologies are invalid")
    topology_contracts = (
        ("forced_total_order", 3, (1, 1, 1)),
        ("parallel_only", 1, (3,)),
    )
    if len(topologies) != len(topology_contracts):
        raise Stage4ContractError("Stage4 validation VRAM topology count drifted")
    normalized_topologies: list[dict[str, Any]] = []
    for raw, (mode, rounds, active_by_round) in zip(
        topologies, topology_contracts, strict=True
    ):
        if not isinstance(raw, Mapping) or set(raw) != topology_fields:
            raise Stage4ContractError("Stage4 validation VRAM topology schema drifted")
        peak_bytes = raw.get("peak_reserved_bytes")
        peak_fraction = raw.get("peak_reserved_fraction")
        active_values = raw.get("active_skill_counts_by_round")
        if (
            raw.get("compiler_mode") != mode
            or raw.get("active_skill_count") != 3
            or raw.get("completed_rounds") != rounds
            or not isinstance(active_values, Sequence)
            or isinstance(active_values, (str, bytes))
            or tuple(active_values) != active_by_round
            or isinstance(peak_bytes, bool)
            or not isinstance(peak_bytes, int)
            or peak_bytes < 0
            or isinstance(peak_fraction, bool)
            or not isinstance(peak_fraction, (int, float))
            or not math.isfinite(float(peak_fraction))
            or not 0.0 <= float(peak_fraction) <= 0.90
            or raw.get("finite") is not True
            or raw.get("passed") is not True
        ):
            raise Stage4ContractError("Stage4 validation VRAM topology values drifted")
        normalized_topologies.append(
            {
                "compiler_mode": mode,
                "active_skill_count": 3,
                "completed_rounds": rounds,
                "active_skill_counts_by_round": list(active_by_round),
                "peak_reserved_bytes": peak_bytes,
                "peak_reserved_fraction": float(peak_fraction),
                "finite": True,
                "passed": True,
            }
        )
    peak_bytes = validation_vram_gate.get("peak_reserved_bytes")
    peak_fraction = validation_vram_gate.get("peak_reserved_fraction")
    resident_entries = validation_vram_gate.get("resident_optimizer_state_entries")
    resident_bytes = validation_vram_gate.get("resident_optimizer_state_bytes")
    resident_ema_bytes = validation_vram_gate.get("resident_ema_bytes")
    if (
        validation_vram_gate.get("image_size") != 2040
        or validation_vram_gate.get("max_rounds") != 3
        or validation_vram_gate.get("completed_rounds") != 3
        or isinstance(peak_bytes, bool)
        or not isinstance(peak_bytes, int)
        or peak_bytes < 0
        or peak_bytes
        != max(topology["peak_reserved_bytes"] for topology in normalized_topologies)
        or isinstance(peak_fraction, bool)
        or not isinstance(peak_fraction, (int, float))
        or float(peak_fraction)
        != max(topology["peak_reserved_fraction"] for topology in normalized_topologies)
        or validation_vram_gate.get("maximum_peak_reserved_fraction") != 0.90
        or isinstance(resident_entries, bool)
        or not isinstance(resident_entries, int)
        or resident_entries <= 0
        or isinstance(resident_bytes, bool)
        or not isinstance(resident_bytes, int)
        or resident_bytes <= 0
        or isinstance(resident_ema_bytes, bool)
        or not isinstance(resident_ema_bytes, int)
        or resident_ema_bytes <= 0
        or validation_vram_gate.get("optimizer_state_empty_after") is not True
        or validation_vram_gate.get("finite") is not True
        or validation_vram_gate.get("passed") is not True
    ):
        raise Stage4ContractError("Stage4 validation VRAM gate values drifted")
    normalized_gate = {
        "image_size": 2040,
        "max_rounds": 3,
        "completed_rounds": 3,
        "topologies": normalized_topologies,
        "peak_reserved_bytes": peak_bytes,
        "peak_reserved_fraction": float(peak_fraction),
        "maximum_peak_reserved_fraction": 0.90,
        "resident_optimizer_state_entries": resident_entries,
        "resident_optimizer_state_bytes": resident_bytes,
        "resident_ema_bytes": resident_ema_bytes,
        "optimizer_state_empty_after": True,
        "finite": True,
        "passed": True,
    }
    return {
        "schema_version": "graphrestore-stage4-runtime-evidence-v1",
        "selected_crop_size": selected_crop_size,
        "selected_micro_batch": selected_micro_batch,
        "micro_batch_trials_sha256": sha256_json(normalized_trials),
        "validation_vram_gate_sha256": sha256_json(normalized_gate),
    }


def _validate_stage4_runtime_evidence_binding(
    provenance: Mapping[str, Any],
) -> None:
    value = provenance.get("runtime_evidence")
    expected_fields = {
        "schema_version",
        "selected_crop_size",
        "selected_micro_batch",
        "micro_batch_trials_sha256",
        "validation_vram_gate_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise Stage4ContractError(
            "Stage4 checkpoint provenance lacks runtime gate evidence"
        )
    crop = value.get("selected_crop_size")
    micro = value.get("selected_micro_batch")
    if (
        value.get("schema_version") != "graphrestore-stage4-runtime-evidence-v1"
        or isinstance(crop, bool)
        or not isinstance(crop, int)
        or isinstance(micro, bool)
        or not isinstance(micro, int)
        or (crop, micro) not in stage4_probe_candidate_order()
    ):
        raise Stage4ContractError("Stage4 runtime evidence selection drifted")
    for field in ("micro_batch_trials_sha256", "validation_vram_gate_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise Stage4ContractError("Stage4 runtime evidence SHA256 is invalid")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise Stage4ContractError(
                "Stage4 runtime evidence SHA256 is invalid"
            ) from exc
    runtime = provenance.get("runtime")
    if isinstance(runtime, Mapping) and (
        runtime.get("crop_size") != crop or runtime.get("micro_batch") != micro
    ):
        raise Stage4ContractError(
            "Stage4 runtime evidence differs from frozen runtime selection"
        )


@torch.inference_mode()
def probe_stage4_validation_vram(
    model: GraphRestore,
    *,
    optimizer: torch.optim.Optimizer,
    ema: Stage4PhaseAwareEMA,
    device: torch.device,
    image_size: int = 2040,
    max_rounds: int = 3,
    maximum_reserved_fraction: float = 0.90,
) -> Stage4ValidationVRAMGate:
    """Exercise both legal 2040-square, three-skill Stage4 CUDA topologies."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise Stage4ContractError("Stage4 validation VRAM gate requires CUDA")
    if image_size != 2040 or max_rounds != 3 or maximum_reserved_fraction != 0.90:
        raise Stage4ContractError("Stage4 validation VRAM gate contract drifted")
    if not isinstance(ema, Stage4PhaseAwareEMA):
        raise Stage4ContractError("Stage4 validation gate requires phase-aware EMA")
    if optimizer.state:
        raise Stage4ContractError(
            "Stage4 validation gate requires a pristine step0 optimizer"
        )
    rng = capture_rng_state()
    was_training = model.training
    original_mode = model.compiler.mode
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    if total_memory <= 0:
        raise Stage4ContractError("Stage4 validation VRAM gate saw invalid GPU memory")
    image: Tensor | None = None
    target: Tensor | None = None
    output: Any = None
    topology_results: list[Stage4ValidationVRAMTopology] = []
    resident_optimizer_state_bytes = 0
    resident_optimizer_state_entries = 0
    resident_ema_bytes = sum(
        value.numel() * value.element_size() for value in ema.shadow.values()
    )
    try:
        model.eval()
        # AdamW moments are lazy.  Materialize a conservative full-state
        # residency ledger for every production optimizer parameter, retain it
        # through both inference topologies, then clear it before the legal
        # step0 anchor is written.  No optimizer update or model mutation occurs.
        for group in optimizer.param_groups:
            parameters = group.get("params")
            if not isinstance(parameters, list):
                raise Stage4ContractError(
                    "Stage4 validation gate optimizer parameters drifted"
                )
            for parameter in parameters:
                if (
                    not isinstance(parameter, nn.Parameter)
                    or not parameter.is_floating_point()
                ):
                    raise Stage4ContractError(
                        "Stage4 validation gate optimizer parameter is invalid"
                    )
                state = optimizer.state[parameter]
                if state:
                    raise Stage4ContractError(
                        "Stage4 validation gate optimizer was not pristine"
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
        # The first three thresholds are zero and every other threshold is one.
        # The production compiler is capped at three active skills.  We still
        # assert the exact compiled topology below, so a saturated probability
        # that reaches one fails closed instead of silently changing this gate.
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
                raise Stage4ContractError("Stage4 validation gate requires a trace")
            if len(output.compiled_graphs) != 1:
                raise Stage4ContractError(
                    "Stage4 validation gate requires one compiled sample graph"
                )
            graph = output.compiled_graphs[0]
            active_skill_count = len(graph.active_skills)
            completed_rounds = len(output.trace)
            active_by_round = tuple(
                int(trace.active_mask[0].sum().item()) for trace in output.trace
            )
            if active_skill_count != 3 or completed_rounds != expected_rounds:
                raise Stage4ContractError(
                    f"Stage4 {compiler_mode} validation gate topology drifted"
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
                raise Stage4ContractError(
                    f"Stage4 {compiler_mode} validation gate execution drifted"
                )
            finite = bool(torch.isfinite(output.final).all())
            if not finite:
                raise FloatingPointError(
                    f"non-finite Stage4 {compiler_mode} validation gate output"
                )
            metric = official_psnr_ssim(
                output.final.detach().float().cpu(),
                target.detach().float().cpu(),
                quantize=True,
            )
            if not bool(
                torch.isfinite(metric.psnr).all() & torch.isfinite(metric.ssim).all()
            ):
                raise FloatingPointError(
                    f"non-finite Stage4 {compiler_mode} validation gate metric"
                )
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_reserved(device))
            if peak < 0:
                raise Stage4ContractError(
                    "Stage4 validation VRAM gate saw negative reserved memory"
                )
            fraction = peak / total_memory
            topology_results.append(
                Stage4ValidationVRAMTopology(
                    compiler_mode=compiler_mode,
                    active_skill_count=active_skill_count,
                    completed_rounds=completed_rounds,
                    active_skill_counts_by_round=active_by_round,
                    peak_reserved_bytes=peak,
                    peak_reserved_fraction=fraction,
                    finite=finite,
                    passed=fraction <= maximum_reserved_fraction,
                )
            )
            output = None

        peak = max(value.peak_reserved_bytes for value in topology_results)
        fraction = max(value.peak_reserved_fraction for value in topology_results)
        gate = Stage4ValidationVRAMGate(
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
            raise Stage4ContractError(
                "Stage4 2040-square validation topology peak "
                f"{fraction:.4f} exceeds 0.90"
            )
        return gate
    finally:
        model.compiler.mode = original_mode
        model.train(was_training)
        image = target = output = None
        optimizer.state.clear()
        state = None
        restore_rng_state(rng)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        raise Stage4ContractError("cannot aggregate an empty Stage4 metric bucket")
    result = math.fsum(collected) / len(collected)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite Stage4 validation aggregate")
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(prediction: Tensor, target: Tensor) -> float | None:
    x = prediction.detach().float().cpu().flatten().numpy().astype(np.float64)
    y = target.detach().float().cpu().flatten().numpy().astype(np.float64)
    if x.var() < 1.0e-8 or y.var() < 1.0e-8:
        return None
    result = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    return result if math.isfinite(result) else None


@torch.inference_mode()
def validate_stage4(
    model: GraphRestore,
    dataset: GraphRestoreEpisodeDataset,
    *,
    device: torch.device,
    relation_val_records: Mapping[str, Mapping[str, Any]],
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Full primary-val GraphRestore validation; no MiO100 path is accepted."""

    if dataset.training or dataset.crop_size is not None:
        raise Stage4ContractError(
            "Stage4 validation must be full-resolution/no augmentation"
        )
    if any(record.group not in {"single", "A"} for record in dataset.records):
        raise Stage4ContractError("Stage4 validation contains forbidden groups")
    relation_lookup = _relation_mapping(relation_val_records)
    model.eval()
    rows: list[dict[str, Any]] = []
    presence_tp = torch.zeros(len(SKILLS), dtype=torch.float64)
    presence_fp = torch.zeros_like(presence_tp)
    presence_fn = torch.zeros_like(presence_tp)
    relation_true: list[int] = []
    relation_pred: list[int] = []
    guard_values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("rain", "haze")
    }
    reentry = unexpected = dropped = proposed_edges = precycle_graphs = (
        program_levels
    ) = 0
    clean_examples: dict[int, Mapping[str, Any]] = {}

    for index, record in enumerate(dataset.records):
        sample = dataset[
            EpisodeRequest(index=index, episode_type="restoration", absolute_step=0)
        ]
        image = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
        target = sample["gt_clean"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with _autocast(device, use_bf16):
            output = model(image, return_trace=True, max_rounds=3)
        from src.net.graphrestore import GraphRestoreOutput

        if not isinstance(output, GraphRestoreOutput):
            raise Stage4ContractError("Stage4 validation requires GraphRestore trace")
        metric = official_psnr_ssim(
            output.final.detach().float().cpu(),
            target.detach().float().cpu(),
            quantize=True,
        )
        combination = "+".join(record.operator_order)
        rows.append(
            {
                "sample_id": record.sample_id,
                "group": record.group,
                "combination": combination,
                "psnr": float(metric.psnr.item()),
                "ssim": float(metric.ssim.item()),
            }
        )
        planner0 = output.planner_outputs[0]
        predicted = planner0.presence_probabilities[0] >= model.presence_thresholds
        truth = sample["presence_target"].bool().cpu()
        predicted_cpu = predicted.detach().cpu()
        presence_tp += (predicted_cpu & truth).to(torch.float64)
        presence_fp += (predicted_cpu & ~truth).to(torch.float64)
        presence_fn += (~predicted_cpu & truth).to(torch.float64)
        for skill_name in ("rain", "haze"):
            skill_id = SKILL_TO_INDEX[skill_name]
            if not bool(truth[skill_id]):
                continue
            pred_guard = planner0.spatial_guard_probabilities[0, skill_id]
            gt_guard = sample["guard_targets"][skill_id].to(pred_guard)
            pred_guard = align_guard_prediction_to_target(pred_guard, gt_guard)
            correlation = _spearman(pred_guard, gt_guard)
            if correlation is not None:
                guard_values[skill_name]["spearman"].append(correlation)
            else:
                guard_values[skill_name]["spearman_skipped"].append(1.0)
            guard_values[skill_name]["mae"].append(
                float((pred_guard - gt_guard).abs().mean().cpu())
            )
            guard_values[skill_name]["std"].append(
                float(pred_guard.std(unbiased=False).cpu())
            )
            guard_values[skill_name]["high_frac"].append(
                float((pred_guard > 0.9).float().mean().cpu())
            )

        if record.is_pair and record.sample_id in relation_lookup:
            relation = relation_lookup[record.sample_id]
            label = str(relation.get("label", ""))
            if label != "ambiguous":
                if label not in {"i_before_j", "j_before_i", "parallel"}:
                    raise Stage4ContractError("invalid interaction_val label")
                ids = tuple(sorted(record.skill_ids))
                row = PAIR_TO_ROW[ids]
                relation_true.append(
                    ("i_before_j", "j_before_i", "parallel").index(label)
                )
                relation_pred.append(int(planner0.relation_logits[0, row].argmax()))

        for trace in output.trace:
            reentry += int(trace.reentry_request_mask.sum().item())
            unexpected += int(trace.unexpected_activation_mask.sum().item())
            if trace.execution is not None:
                program_levels += int(trace.active_mask.any(dim=1).sum().item())
        for graph in output.compiled_graphs:
            precycle_graphs += int(bool(graph.dropped_edges))
            dropped += len(graph.dropped_edges)
            proposed_edges += len(graph.edges) + len(graph.dropped_edges)
        if not record.is_pair:
            skill_id = record.skill_ids[0]
            clean_examples.setdefault(skill_id, sample)

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "count": len(selected),
            "psnr": _mean(float(row["psnr"]) for row in selected),
            "ssim": _mean(float(row["ssim"]) for row in selected),
        }

    single_rows = [row for row in rows if row["group"] == "single"]
    pair_rows = [row for row in rows if row["group"] == "A"]
    pair_names = sorted({str(row["combination"]) for row in pair_rows})
    if len(pair_names) != 8:
        raise Stage4ContractError("primary_val lacks eight Group-A combinations")
    group_a_tasks = {
        name: aggregate([row for row in pair_rows if row["combination"] == name])
        for name in pair_names
    }
    group_a = {
        "count": len(pair_rows),
        "combination_count": 8,
        "psnr": _mean(float(row["psnr"]) for row in group_a_tasks.values()),
        "ssim": _mean(float(row["ssim"]) for row in group_a_tasks.values()),
    }
    single_tasks = {
        name: aggregate([row for row in single_rows if row["combination"] == name])
        for name in sorted({str(row["combination"]) for row in single_rows})
    }
    single_equal = {
        "count": len(single_rows),
        "task_count": len(single_tasks),
        "psnr": _mean(float(row["psnr"]) for row in single_tasks.values()),
        "ssim": _mean(float(row["ssim"]) for row in single_tasks.values()),
    }

    precision = presence_tp / (presence_tp + presence_fp).clamp_min(1.0)
    recall = presence_tp / (presence_tp + presence_fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    relation_accuracy = (
        sum(a == b for a, b in zip(relation_true, relation_pred, strict=True))
        / len(relation_true)
        if relation_true
        else 0.0
    )
    parallel_tp = sum(
        a == 2 and b == 2 for a, b in zip(relation_true, relation_pred, strict=True)
    )
    parallel_fp = sum(
        a != 2 and b == 2 for a, b in zip(relation_true, relation_pred, strict=True)
    )
    parallel_fn = sum(
        a == 2 and b != 2 for a, b in zip(relation_true, relation_pred, strict=True)
    )

    # Fixed, bounded identity diagnostics: one clean and one wrong-skill call
    # per true single skill.  They use the same selected EMA snapshot.
    clean_metric_rows: list[tuple[float, float, float]] = []
    wrong_metric_rows: list[tuple[float, float, float]] = []
    for true_skill, sample in sorted(clean_examples.items()):
        clean = sample["gt_clean"].unsqueeze(0).to(device=device, dtype=torch.float32)
        degraded = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
        force_clean = torch.zeros(1, len(SKILLS), device=device, dtype=torch.bool)
        force_clean[0, true_skill] = True
        wrong_skill = (true_skill + 1) % len(SKILLS)
        force_wrong = torch.zeros_like(force_clean)
        force_wrong[0, wrong_skill] = True
        with _autocast(device, use_bf16):
            clean_out = model(clean, forced_counterfactual_mask=force_clean)
            wrong_out = model(degraded, forced_counterfactual_mask=force_wrong)
        if not torch.is_tensor(clean_out) or not torch.is_tensor(wrong_out):
            raise Stage4ContractError(
                "counterfactual validation returned trace unexpectedly"
            )
        for prediction, target_value, sink in (
            (clean_out, clean, clean_metric_rows),
            (wrong_out, degraded, wrong_metric_rows),
        ):
            metric = official_psnr_ssim(
                prediction.detach().float().cpu(),
                target_value.detach().float().cpu(),
                quantize=True,
            )
            residual = float(
                (prediction.float() - target_value).square().mean().sqrt().cpu()
            )
            sink.append(
                (float(metric.psnr.item()), float(metric.ssim.item()), residual)
            )

    diagnostics: dict[str, Any] = {
        "planner_macro_f1": float(f1.mean()),
        "per_skill_f1": {name: float(f1[index]) for index, name in enumerate(SKILLS)},
        "relation_accuracy": relation_accuracy,
        "relation_n_nonambiguous": len(relation_true),
        "parallel_precision": parallel_tp / max(parallel_tp + parallel_fp, 1),
        "parallel_recall": parallel_tp / max(parallel_tp + parallel_fn, 1),
        "pre_cycle_rate": precycle_graphs / len(rows),
        "post_cycle_rate": 0.0,
        "dropped_edge_rate": dropped / proposed_edges if proposed_edges else 0.0,
        "reentry_request_rate": reentry / max(len(rows) * 3 * len(SKILLS), 1),
        "unexpected_skill_activation_rate": unexpected
        / max(len(rows) * 3 * len(SKILLS), 1),
        "mean_program_levels": program_levels / len(rows),
    }
    for name in ("rain", "haze"):
        values = guard_values[name]
        diagnostics.update(
            {
                f"guard_spearman_{name}": (
                    _mean(values["spearman"]) if values["spearman"] else None
                ),
                f"guard_mae_{name}": _mean(values["mae"]),
                f"guard_std_{name}": _mean(values["std"]),
                f"guard_high_frac_{name}": _mean(values["high_frac"]),
                f"valid_guard_images_{name}": len(values["spearman"]),
                f"skipped_guard_images_{name}": len(values["spearman_skipped"]),
            }
        )

    def identity_summary(
        values: Sequence[tuple[float, float, float]],
    ) -> dict[str, float]:
        return {
            "psnr": _mean(row[0] for row in values),
            "ssim": _mean(row[1] for row in values),
            "residual_norm": _mean(row[2] for row in values),
        }

    diagnostics["clean_misuse"] = identity_summary(clean_metric_rows)
    diagnostics["wrong_skill_identity"] = identity_summary(wrong_metric_rows)
    return {
        "schema_version": "graphrestore-stage4-validation-v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now_iso(),
        "dataset": "primary_val_single_and_group_a_only",
        "relation_validation_source": "interaction_val_only",
        "output_quantization": "clamp_round_uint8",
        "single_equal_task_mean": single_equal,
        "single_tasks": single_tasks,
        "group_a_equal_combination_mean": group_a,
        "group_a_combinations": group_a_tasks,
        "diagnostics": diagnostics,
        "image_count": len(rows),
    }


def stage4_validation_score(summary: Mapping[str, Any], step: int) -> ValidationScore:
    group = summary["group_a_equal_combination_mean"]
    single = summary["single_equal_task_mean"]
    return ValidationScore(
        group_a_psnr=float(group["psnr"]),
        group_a_ssim=float(group["ssim"]),
        single_psnr=float(single["psnr"]),
        single_ssim=float(single["ssim"]),
        step=step,
    )


def _tensor_mapping_digest(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in values.items():
        contiguous = value.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _rng_state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode("utf-8"))
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(repr(numpy_state[2:]).encode("utf-8"))
    digest.update(state["torch_cpu"].detach().cpu().numpy().tobytes())
    for value in state.get("torch_cuda_all", ()):
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _stage4_guard_mode(model: GraphRestore, mode: str) -> Iterator[None]:
    if mode not in {"predicted_spatial", "global_mean", "all_one"}:
        raise Stage4ContractError(f"unknown Stage4 diagnostic guard mode: {mode}")
    if mode == "predicted_spatial":
        yield
        return

    had_instance_override = "execute_planned_level" in model.__dict__
    previous = model.__dict__.get("execute_planned_level")

    def execute_with_guard_mode(
        this: GraphRestore,
        current: Tensor,
        encoder_features: Sequence[Tensor],
        planner_output: PlannerOutput,
        *,
        active_mask: Tensor,
        forced_presence_mask: Tensor | None = None,
    ):
        presence = planner_output.presence_probabilities
        if forced_presence_mask is not None:
            forced = forced_presence_mask.to(device=presence.device, dtype=torch.bool)
            presence = torch.where(forced, torch.ones_like(presence), presence)
        spatial = planner_output.spatial_guard_probabilities
        if mode == "global_mean":
            spatial = spatial.mean(dim=(-2, -1), keepdim=True).expand_as(spatial)
        else:
            spatial = torch.ones_like(spatial)
        guards = presence[:, :, None, None] * spatial
        return this.execute_level(
            current,
            encoder_features,
            guards=guards,
            active_mask=active_mask,
            forced_presence_mask=forced_presence_mask,
        )

    model.execute_planned_level = MethodType(execute_with_guard_mode, model)  # type: ignore[method-assign]
    try:
        yield
    finally:
        if had_instance_override:
            model.execute_planned_level = previous  # type: ignore[method-assign,assignment]
        else:
            del model.__dict__["execute_planned_level"]


def _diagnostic_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("single_equal_task_mean", "group_a_equal_combination_mean"):
        value = summary.get(field)
        if not isinstance(value, Mapping):
            raise Stage4ContractError(f"Stage4 diagnostics lacks {field}")
        metric = dict(value)
        for metric_name in ("psnr", "ssim"):
            metric_value = metric.get(metric_name)
            if (
                isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(float(metric_value))
            ):
                raise Stage4ContractError(
                    f"Stage4 diagnostics {field}.{metric_name} is non-finite"
                )
        result[field] = metric
    diagnostics = summary.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise Stage4ContractError("Stage4 diagnostics mapping is empty")
    image_count = summary.get("image_count")
    if (
        isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or image_count <= 0
    ):
        raise Stage4ContractError("Stage4 diagnostics image_count must be positive")
    result["diagnostics"] = dict(diagnostics)
    result["image_count"] = image_count
    return result


def _render_stage4_diagnostics(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Stage4 Guard and Misuse Diagnostics",
        "",
        f"- Selected EMA SHA256: `{payload['selected_best_ema_sha256']}`",
        "- Optimizer updates: `0`",
        "- Model/EMA/RNG unchanged: `true`",
        "- Dataset: frozen primary_val single + Group A only",
        "",
        "| Family | Mode | Group-A PSNR / SSIM | Single PSNR / SSIM |",
        "|---|---|---:|---:|",
    ]
    for family in ("compiler_modes", "guard_modes"):
        for mode, value in payload[family].items():
            group = value["group_a_equal_combination_mean"]
            single = value["single_equal_task_mean"]
            lines.append(
                f"| {family} | {mode} | {group['psnr']:.6f} / "
                f"{group['ssim']:.8f} | {single['psnr']:.6f} / "
                f"{single['ssim']:.8f} |"
            )
    lines.append("")
    return "\n".join(lines)


@torch.inference_mode()
def run_stage4_zero_training_diagnostics(
    model: GraphRestore,
    ema: Stage4PhaseAwareEMA,
    dataset: GraphRestoreEpisodeDataset,
    *,
    device: torch.device,
    relation_val_records: Mapping[str, Mapping[str, Any]],
    selected_best_checkpoint: str | Path,
    expected_provenance: Mapping[str, Any],
    json_path: str | Path,
    report_path: str | Path,
    maximum_reserved_fraction: float = 0.90,
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Run the locked six-mode diagnostic suite with zero optimizer updates."""

    if not isinstance(ema, Stage4PhaseAwareEMA):
        raise Stage4ContractError("Stage4 diagnostics require phase-aware EMA")
    if maximum_reserved_fraction != 0.90:
        raise Stage4ContractError("Stage4 diagnostic VRAM ceiling drifted")
    checkpoint = Path(selected_best_checkpoint).resolve()
    checkpoint_digest = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "graphrestore-checkpoint-v1"
        or payload.get("stage") != STAGE4_CHECKPOINT_STAGE
        or payload.get("model_role") != "ema_selection"
        or payload.get("resumable") is not False
        or payload.get("pending_validation_step") is not None
        or payload.get("scaler") is not None
        or payload.get("amp") != {"dtype": "bfloat16", "scaler_required": False}
    ):
        raise Stage4ContractError("Stage4 diagnostics require selected best_ema.pth")
    selected_step = payload.get("step")
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step < 0
    ):
        raise Stage4ContractError("selected Stage4 EMA has invalid step")
    selected_metrics = _validate_stage4_metrics(
        payload.get("metrics"), step=selected_step, resumable=False
    )
    if not selected_metrics:
        raise Stage4ContractError("selected Stage4 EMA lacks validation metrics")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage4ContractError("selected Stage4 EMA lacks provenance")
    verify_provenance(provenance, expected_provenance)
    selected_state = _mapping_of_tensors(
        payload.get("model"), field="selected Stage4 model"
    )
    selected_ema = payload.get("ema")
    if not isinstance(selected_ema, Mapping):
        raise Stage4ContractError("selected Stage4 EMA lacks EMA state")
    ema.validate_state_metadata(selected_ema)
    if selected_ema.get("num_updates") != selected_step:
        raise Stage4ContractError("selected Stage4 EMA update count drifted")
    if provenance.get("ema_policy") != selected_ema.get("policy"):
        raise Stage4ContractError("selected Stage4 EMA provenance policy drifted")
    selected_shadow = _mapping_of_tensors(
        selected_ema.get("shadow"), field="selected Stage4 EMA shadow"
    )
    if selected_state.keys() != selected_shadow.keys() or any(
        not torch.equal(value.detach().to(selected_shadow[name]), selected_shadow[name])
        for name, value in selected_state.items()
    ):
        raise Stage4ContractError("selected Stage4 best model is not its EMA shadow")
    _validate_stage4_fixed_ema_state(
        model,
        selected_state,
        selected_shadow,
        context="Stage4 diagnostics selected best",
        require_frozen_live_match=True,
    )

    core = unwrap_model(model)
    raw_backup = {
        name: value.detach().cpu().clone() for name, value in core.state_dict().items()
    }
    raw_digest = _tensor_mapping_digest(core.state_dict())
    ema_digest = _tensor_mapping_digest(ema.shadow)
    rng = capture_rng_state()
    rng_digest = _rng_state_digest(rng)
    was_training = model.training
    compiler_mode = model.compiler.mode
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    if total_memory <= 0:
        raise Stage4ContractError("Stage4 diagnostics saw invalid GPU memory")
    compiler_results: dict[str, Any] = {}
    guard_results: dict[str, Any] = {}

    def run_one() -> dict[str, Any]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        summary = validate_stage4(
            model,
            dataset,
            device=device,
            relation_val_records=relation_val_records,
            use_bf16=use_bf16,
        )
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_reserved(device))
        fraction = peak / total_memory
        if peak < 0 or not 0.0 <= fraction <= maximum_reserved_fraction:
            raise Stage4ContractError(
                f"Stage4 diagnostic validation peak {fraction:.4f} exceeds 0.90"
            )
        result = _diagnostic_summary(summary)
        result["peak_reserved_bytes"] = peak
        result["peak_reserved_fraction"] = fraction
        return result

    try:
        core.load_state_dict(selected_state, strict=True)
        for mode in ("full_partial_order", "forced_total_order", "parallel_only"):
            model.compiler.mode = mode
            compiler_results[mode] = run_one()
        model.compiler.mode = "full_partial_order"
        for mode in ("predicted_spatial", "global_mean", "all_one"):
            with _stage4_guard_mode(model, mode):
                guard_results[mode] = run_one()
    finally:
        model.compiler.mode = compiler_mode
        core.load_state_dict(raw_backup, strict=True)
        model.train(was_training)
        restore_rng_state(rng)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if _tensor_mapping_digest(core.state_dict()) != raw_digest:
        raise Stage4ContractError("Stage4 diagnostics changed raw model state")
    if _tensor_mapping_digest(ema.shadow) != ema_digest:
        raise Stage4ContractError("Stage4 diagnostics changed EMA state")
    if _rng_state_digest(capture_rng_state()) != rng_digest:
        raise Stage4ContractError("Stage4 diagnostics changed RNG state")
    if sha256_file(checkpoint) != checkpoint_digest:
        raise Stage4ContractError(
            "selected Stage4 EMA changed during zero-training diagnostics"
        )

    result_payload: dict[str, Any] = {
        "schema_version": "graphrestore-stage4-zero-training-diagnostics-v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now_iso(),
        "selected_best_ema_path": str(checkpoint),
        "selected_best_ema_sha256": checkpoint_digest,
        "optimizer_updates": 0,
        "model_ema_rng_unchanged": True,
        "compiler_modes": compiler_results,
        "guard_modes": guard_results,
    }
    atomic_write_json(json_path, result_payload)
    atomic_write_text(report_path, _render_stage4_diagnostics(result_payload))
    return result_payload


class _null_model_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


def save_stage4_checkpoint(
    destination: str | Path,
    *,
    step: int,
    model: GraphRestore,
    ema: Stage4PhaseAwareEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: Stage4EpisodeSampler,
    provenance: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    model_as_ema: bool = False,
    pending_validation_step: int | None = None,
) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Stage4ContractError("Stage4 checkpoint step must be non-negative")
    if not isinstance(ema, Stage4PhaseAwareEMA):
        raise Stage4ContractError("Stage4 checkpoints require phase-aware EMA")
    ema_state = ema.state_dict()
    ema.validate_state_metadata(ema_state)
    if ema.num_updates != step:
        raise Stage4ContractError("Stage4 checkpoint step/EMA update count mismatch")
    if provenance.get("ema_policy") != ema_state["policy"]:
        raise Stage4ContractError("Stage4 checkpoint provenance EMA policy drifted")
    _validate_stage4_runtime_evidence_binding(provenance)
    frozen_parent_digest = provenance.get("frozen_parent_state_sha256")
    if (
        not isinstance(frozen_parent_digest, str)
        or len(frozen_parent_digest) != 64
        or stage4_fixed_state_digest(model) != frozen_parent_digest
    ):
        raise Stage4ContractError(
            "Stage4 frozen model state drifted from its Stage3-derived parent"
        )
    if pending_validation_step is not None and pending_validation_step != step:
        raise Stage4ContractError(
            "Stage4 pending validation step must equal checkpoint step"
        )
    if model_as_ema and pending_validation_step is not None:
        raise Stage4ContractError("Stage4 best EMA cannot be pending validation")
    _validate_stage4_fixed_ema_state(
        model,
        unwrap_model(model).state_dict(),
        ema.shadow,
        context="Stage4 save",
    )
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    _validate_stage4_optimizer_state(optimizer, optimizer_state, step=step)
    optimizer_ledger = _validate_stage4_optimizer_state_ledger(
        model,
        optimizer,
        optimizer_state,
        None,
        create=True,
    )
    _validate_stage4_scheduler_state(
        scheduler,
        scheduler_state,
        optimizer_state,
        step=step,
    )
    validated_metrics = _validate_stage4_metrics(
        {} if metrics is None else metrics,
        step=step,
        resumable=not model_as_ema,
    )
    context = ema.apply_to(model) if model_as_ema else _null_model_context()
    with context:
        payload = checkpoint_payload(
            stage=STAGE4_CHECKPOINT_STAGE,
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
        payload["model_role"] = (
            "ema_selection" if model_as_ema else "raw_training_state"
        )
        payload["resumable"] = not model_as_ema
        payload["pending_validation_step"] = pending_validation_step
        payload["amp"] = {"dtype": "bfloat16", "scaler_required": False}
        payload["optimizer_state_name_ledger"] = optimizer_ledger
        rng_state = payload.get("rng_states")
        if not isinstance(rng_state, Mapping):
            raise Stage4ContractError("Stage4 checkpoint RNG capture failed")
        _validate_stage4_rng_state(rng_state)
        atomic_torch_save(payload, destination)


def resume_stage4_checkpoint(
    checkpoint: str | Path,
    *,
    model: GraphRestore,
    ema: Stage4PhaseAwareEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: Stage4EpisodeSampler,
    expected_provenance: Mapping[str, Any],
    expected_validation_every: int = 4000,
    expected_max_steps: int = 40_000,
) -> Mapping[str, Any]:
    if not isinstance(ema, Stage4PhaseAwareEMA):
        raise Stage4ContractError("Stage4 resume requires phase-aware EMA")
    # Inspect role metadata before load_checkpoint is allowed to mutate the
    # model, optimizer, scheduler, or RNG.  A selected EMA is an evaluation
    # parent, never an exact continuation point for AdamW.
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_digest = sha256_file(checkpoint_path)
    header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(header, Mapping):
        raise Stage4ContractError("Stage4 resume checkpoint must be a mapping")
    if (
        header.get("schema_version") != "graphrestore-checkpoint-v1"
        or header.get("stage") != STAGE4_CHECKPOINT_STAGE
        or header.get("model_role") != "raw_training_state"
        or header.get("resumable") is not True
        or header.get("scaler") is not None
        or header.get("amp") != {"dtype": "bfloat16", "scaler_required": False}
    ):
        raise Stage4ContractError(
            "Stage4 resume requires resumable raw last.pth, not best_ema.pth"
        )
    step = header.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= expected_max_steps
    ):
        raise Stage4ContractError("invalid Stage4 resume step")
    if "pending_validation_step" not in header:
        raise Stage4ContractError("Stage4 resume lacks pending_validation_step")
    pending = header.get("pending_validation_step")
    if pending is not None:
        if isinstance(pending, bool) or not isinstance(pending, int):
            raise Stage4ContractError(
                "Stage4 pending_validation_step must be an integer or null"
            )
        if pending != step:
            raise Stage4ContractError(
                "Stage4 pending_validation_step differs from checkpoint step"
            )
        if not (
            pending % expected_validation_every == 0 or pending == expected_max_steps
        ):
            raise Stage4ContractError(
                "Stage4 pending_validation_step is not a validation boundary"
            )
    ema_state = header.get("ema")
    if not isinstance(ema_state, Mapping):
        raise Stage4ContractError("Stage4 resume lacks EMA")
    ema.validate_state_metadata(ema_state)
    if ema_state.get("num_updates") != step:
        raise Stage4ContractError("Stage4 resume step/EMA update count mismatch")
    metrics = _validate_stage4_metrics(header.get("metrics"), step=step, resumable=True)
    provenance = header.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage4ContractError("Stage4 resume lacks provenance")
    _validate_stage4_runtime_evidence_binding(provenance)
    _validate_stage4_runtime_evidence_binding(expected_provenance)
    verify_provenance(provenance, expected_provenance)
    if provenance.get("ema_policy") != ema_state.get("policy"):
        raise Stage4ContractError("Stage4 resume provenance EMA policy drifted")
    frozen_parent_digest = provenance.get("frozen_parent_state_sha256")
    if (
        not isinstance(frozen_parent_digest, str)
        or stage4_fixed_state_digest(model) != frozen_parent_digest
    ):
        raise Stage4ContractError(
            "Stage4 live frozen state differs from its provenance parent anchor"
        )
    sampler_state = header.get("sampler_state")
    if not isinstance(sampler_state, Mapping):
        raise Stage4ContractError("Stage4 resume lacks sampler state")
    if sampler_state.get("consumed_optimizer_step") != step:
        raise Stage4ContractError("Stage4 checkpoint/sampler step mismatch")
    _validate_stage4_sampler_state(sampler, sampler_state, step=step)
    rng_state = header.get("rng_states")
    if not isinstance(rng_state, Mapping):
        raise Stage4ContractError("Stage4 resume lacks RNG state")
    _validate_stage4_rng_state(rng_state)
    optimizer_state = header.get("optimizer")
    scheduler_state = header.get("scheduler")
    if not isinstance(optimizer_state, Mapping):
        raise Stage4ContractError("Stage4 resume lacks optimizer state")
    if not isinstance(scheduler_state, Mapping):
        raise Stage4ContractError("Stage4 resume lacks scheduler state")
    _validate_stage4_optimizer_state(optimizer, optimizer_state, step=step)
    if "optimizer_state_name_ledger" not in header:
        raise Stage4ContractError("Stage4 resume lacks optimizer state-name ledger")
    _validate_stage4_optimizer_state_ledger(
        model,
        optimizer,
        optimizer_state,
        header.get("optimizer_state_name_ledger"),
        create=False,
    )
    _validate_stage4_scheduler_state(
        scheduler,
        scheduler_state,
        optimizer_state,
        step=step,
    )
    model_state = _mapping_of_tensors(header.get("model"), field="Stage4 model state")
    shadow = _mapping_of_tensors(ema_state.get("shadow"), field="Stage4 EMA shadow")
    _validate_stage4_fixed_ema_state(
        model,
        model_state,
        shadow,
        context="Stage4 resume",
        require_frozen_live_match=True,
    )
    _validate_stage4_best_incumbent_binding(
        Path(checkpoint),
        metrics,
        model=model,
        ema=ema,
        expected_provenance=expected_provenance,
        pending_validation_step=pending,
    )

    if sha256_file(checkpoint_path) != checkpoint_digest:
        raise Stage4ContractError("Stage4 resume checkpoint changed while validating")
    incompatible = unwrap_model(model).load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Stage4ContractError("Stage4 resume model failed strict load")
    optimizer.load_state_dict(dict(optimizer_state))
    scheduler.load_state_dict(dict(scheduler_state))
    ema.load_state_dict(ema_state)
    sampler.load_state_dict(dict(sampler_state))
    restore_rng_state(rng_state)
    set_stage4_trainability(model)
    return header


def dependency_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "basicsr",
        "numpy",
        "opencv-python",
        "pyiqa",
        "PyYAML",
        "torch",
        "torchvision",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "torch_runtime": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy_runtime": np.__version__,
        "opencv_runtime": cv2.__version__,
        "packages": packages,
    }


def build_stage4_provenance(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    resolved_path: Path,
    resolved: Mapping[str, Any],
    stage1_checkpoint: Path,
    stage3_checkpoint: Path,
    approval: Path,
    thresholds: Path,
    pair_prior: Path,
    global_priority: Path,
    relation_train: Path,
    relation_val: Path,
    crop_size: int,
    micro_batch: int,
    max_steps: int,
    allocator_conf: str,
    frozen_parent_state_sha256: str,
    micro_batch_trials: object,
    validation_vram_gate: object,
    stage3_extension: Mapping[str, Any] | None = None,
    stage3_finalization: Mapping[str, Any] | None = None,
    stage3_complete: Path | None = None,
    stage3_calibrated_diagnostic: Path | None = None,
    stage3_complete_sha256: str | None = None,
    stage3_calibrated_diagnostic_sha256: str | None = None,
    stage3_thresholds_sha256: str | None = None,
    stage4_extension: Stage4ExtensionEvidence | None = None,
) -> dict[str, Any]:
    train = Path(str(resolved[config["paths"]["train_manifest_key"]])).resolve()
    val = Path(str(resolved[config["paths"]["val_manifest_key"]])).resolve()
    expected = resolved.get("expected_identity")
    if not isinstance(expected, Mapping) or not isinstance(
        expected.get("manifests"), Mapping
    ):
        raise Stage4ContractError("resolved paths lacks frozen identities")
    manifest_hashes = expected["manifests"]
    actual_train, actual_val = sha256_file(train), sha256_file(val)
    if actual_train != manifest_hashes.get(
        "primary_train"
    ) or actual_val != manifest_hashes.get("primary_val"):
        raise Stage4ContractError("Stage4 primary manifest hash mismatch")
    agenticir_commit = git_commit(Path(str(resolved["agenticir_repo"])))
    mioir_commit = git_commit(Path(str(resolved["mioir_repo"])))
    if agenticir_commit != expected.get(
        "agenticir_commit"
    ) or mioir_commit != expected.get("mioir_commit"):
        raise Stage4ContractError("Stage4 upstream commit mismatch")
    if crop_size not in {128, 160} or micro_batch not in {1, 2} or 4 % micro_batch:
        raise Stage4ContractError("invalid frozen Stage4 micro batch")
    if allocator_conf != STAGE4_ALLOCATOR_CONF:
        raise Stage4ContractError("invalid frozen Stage4 allocator configuration")
    if (
        not isinstance(frozen_parent_state_sha256, str)
        or len(frozen_parent_state_sha256) != 64
    ):
        raise Stage4ContractError("invalid Stage4 frozen-parent state digest")
    try:
        int(frozen_parent_state_sha256, 16)
    except ValueError as exc:
        raise Stage4ContractError("invalid Stage4 frozen-parent state digest") from exc
    runtime_evidence = stage4_runtime_evidence_metadata(
        micro_batch_trials,
        validation_vram_gate,
        selected_crop_size=crop_size,
        selected_micro_batch=micro_batch,
    )
    artifacts = {
        "stage1_checkpoint": stage1_checkpoint,
        "stage3_checkpoint": stage3_checkpoint,
        "stage3_approval": approval,
        "thresholds": thresholds,
        "pair_prior": pair_prior,
        "global_priority": global_priority,
        "relation_train": relation_train,
        "relation_val": relation_val,
    }
    project_root = config_path.resolve().parents[1]
    stage4_extension_binding: dict[str, Any] | None = None
    if stage4_extension is not None:
        canonical_conditional = (
            project_root / "artifacts/approvals" / STAGE4_EXTENSION_CONDITIONAL_FILENAME
        ).resolve(strict=False)
        canonical_gate = (
            project_root / "artifacts/approvals" / STAGE4_EXTENSION_GATE_FILENAME
        ).resolve(strict=False)
        if (
            max_steps != stage4_extension.target_step
            or stage4_extension.conditional_path != canonical_conditional
            or stage4_extension.gate_path != canonical_gate
            or not canonical_conditional.is_file()
            or not canonical_gate.is_file()
            or sha256_file(canonical_conditional) != stage4_extension.conditional_sha256
            or sha256_file(canonical_gate) != stage4_extension.gate_sha256
        ):
            raise Stage4ContractError(
                "Stage4 extension provenance binding is incomplete or stale"
            )
        stage4_extension_binding = stage4_extension.provenance_binding()
    canonical_extension = (
        project_root / "artifacts/approvals" / STAGE3_EXTENSION_APPROVAL_NAME
    )
    canonical_finalization_raw = (
        project_root / "artifacts/approvals/STAGE3_EXTENSION_REVOKED.json"
    )
    canonical_finalization = canonical_finalization_raw.resolve(strict=False)
    extension_binding: dict[str, str] | None = None
    if stage3_extension is None:
        if canonical_extension.exists():
            raise Stage4ContractError(
                "Stage4 provenance requires the validated Stage3 extension binding"
            )
    else:
        _reject_stage3_extension_symlink_chain(
            canonical_extension, field="approval artifact"
        )
        canonical_extension = canonical_extension.resolve(strict=False)
        extension_keys = {
            "path",
            "sha256",
            "cycles",
            "base_step",
            "target_step",
            "validation_every_steps",
            "validation_steps",
            "schedule_horizon_steps",
            "min_lr",
            "lr_policy",
        }
        if (
            not isinstance(stage3_extension, Mapping)
            or set(stage3_extension) != extension_keys
            or stage3_extension.get("path") != str(canonical_extension)
        ):
            raise Stage4ContractError(
                "Stage4 provenance received an invalid Stage3 extension binding"
            )
        extension_sha = stage3_extension.get("sha256")
        if (
            not isinstance(extension_sha, str)
            or not canonical_extension.is_file()
            or sha256_file(canonical_extension) != extension_sha
        ):
            raise Stage4ContractError("Stage4 provenance Stage3 extension hash drifted")
        extension_binding = {
            "path": str(canonical_extension),
            "sha256": extension_sha,
        }
        artifacts["stage3_extension_approval"] = canonical_extension
    finalization_binding: dict[str, str] | None = None
    if stage3_finalization is None:
        if os.path.lexists(canonical_finalization_raw):
            raise Stage4ContractError(
                "Stage4 provenance requires the validated Stage3 finalization binding"
            )
    else:
        try:
            refreshed_finalization = validate_stage3_extension_revocation(
                canonical_finalization_raw,
                project_root=project_root,
                require_present=True,
            )
        except Exception as exc:
            raise Stage4ContractError(
                f"Stage4 finalization provenance revalidation failed: {exc}"
            ) from exc
        raw_path = stage3_finalization.get("path")
        finalization_sha = stage3_finalization.get("sha256")
        if (
            refreshed_finalization is None
            or not isinstance(raw_path, str)
            or Path(raw_path).resolve(strict=False) != canonical_finalization
            or raw_path != str(canonical_finalization)
            or not isinstance(finalization_sha, str)
            or refreshed_finalization.sha256 != finalization_sha
            or not canonical_finalization.is_file()
            or sha256_file(canonical_finalization) != finalization_sha
        ):
            raise Stage4ContractError(
                "Stage4 provenance received an invalid Stage3 finalization binding"
            )
        if stage3_complete is None or stage3_calibrated_diagnostic is None:
            raise Stage4ContractError(
                "Stage4 finalization provenance requires complete and calibrated diagnostic"
            )
        for label, path, expected_sha in (
            ("stage3_complete", stage3_complete, stage3_complete_sha256),
            (
                "stage3_calibrated_diagnostic",
                stage3_calibrated_diagnostic,
                stage3_calibrated_diagnostic_sha256,
            ),
        ):
            resolved_artifact = path.resolve(strict=False)
            if (
                not is_sha256(expected_sha)
                or not resolved_artifact.is_file()
                or sha256_file(resolved_artifact) != expected_sha
            ):
                raise Stage4ContractError(
                    f"Stage4 finalization provenance {label} hash drifted"
                )
            artifacts[label] = resolved_artifact
        if (
            not is_sha256(stage3_thresholds_sha256)
            or sha256_file(thresholds) != stage3_thresholds_sha256
        ):
            raise Stage4ContractError(
                "Stage4 finalization provenance threshold hash drifted"
            )
        artifacts["stage3_finalization_authorization"] = canonical_finalization
        finalization_binding = {
            "path": str(canonical_finalization),
            "sha256": finalization_sha,
        }
    result = {
        "schema_version": STAGE4_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": sha256_file(config_path),
        "config_semantic_sha256": sha256_json(config),
        "resolved_paths_sha256": sha256_file(resolved_path),
        "semantic_source_sha256": semantic_source_hashes(
            project_root,
            entrypoints=("scripts/train_stage4_e2e.py",),
        ),
        "manifests": {
            "primary_train": {"path": str(train), "sha256": actual_train},
            "primary_val": {"path": str(val), "sha256": actual_val},
        },
        "parents": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "repositories": {
            "agenticir_commit": agenticir_commit,
            "mioir_commit": mioir_commit,
        },
        "ema_policy": stage4_ema_policy_metadata(float(config["ema"]["decay"])),
        "frozen_parent_state_sha256": frozen_parent_state_sha256,
        "runtime_evidence": runtime_evidence,
        "runtime": {
            "crop_size": crop_size,
            "micro_batch": micro_batch,
            "effective_batch_size": 4,
            "accumulation_steps": 4 // micro_batch,
            "max_steps": max_steps,
            "schedule_max_steps": 40_000,
            "kmax_train": 2,
            "kmax_test": 3,
            "gradient_checkpointing": True,
            "torch_compile": False,
            "amp_dtype": "bf16",
            "tf32": True,
            "allocator_conf": allocator_conf,
            "allocator_environment_variable": "PYTORCH_CUDA_ALLOC_CONF",
        },
        "dependency_versions": dependency_versions(),
    }
    if extension_binding is not None:
        result["stage3_extension"] = extension_binding
    if finalization_binding is not None:
        result["stage3_finalization"] = finalization_binding
    if stage4_extension_binding is not None:
        result["stage4_extension"] = stage4_extension_binding
    return result


def append_jsonl(handle: TextIO, value: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )
    handle.flush()


def lr_by_role(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    values: dict[str, float] = {}
    for group in optimizer.param_groups:
        role = str(group.get("role", "unknown"))
        lr = float(group["lr"])
        if role in values and not math.isclose(
            values[role], lr, rel_tol=0.0, abs_tol=0.0
        ):
            raise Stage4ContractError(f"decay splits disagree on {role} LR")
        values[role] = lr
    return values


__all__ = [
    "COUNTERFACTUAL_TYPES",
    "EPISODE_TYPES",
    "FrozenStage3Snapshot",
    "PROTOCOL_ID",
    "STAGE4_ALLOCATOR_CONF",
    "STAGE4_EMA_SCOPE",
    "STAGE4_EXTENSION_TARGET_STEP",
    "STAGE4_SCHEMA",
    "Stage4Batch",
    "Stage4ContractError",
    "Stage4ExtensionEvidence",
    "Stage4EpisodeDataset",
    "Stage4EpisodeSampler",
    "Stage4ImageLoss",
    "Stage4MicroBatchTrial",
    "Stage4PhaseAwareEMA",
    "Stage4ProgramOutput",
    "Stage4Request",
    "Stage4RoundDiagnostics",
    "Stage4StepResult",
    "Stage4ValidationVRAMGate",
    "Stage4ValidationVRAMTopology",
    "append_jsonl",
    "build_stage4_ema",
    "build_stage4_optimizer",
    "build_stage4_provenance",
    "choose_stage4_micro_batch",
    "load_presence_thresholds",
    "load_relation_records",
    "load_stage3_best_ema",
    "is_stage4_cuda_oom_exception",
    "lr_by_role",
    "prepare_stage4_batch",
    "probe_stage4_validation_vram",
    "require_stage4_allocator_conf",
    "resume_stage4_checkpoint",
    "run_stage4_program",
    "run_stage4_zero_training_diagnostics",
    "save_stage4_checkpoint",
    "set_stage4_trainability",
    "stage4_image_loss",
    "stage4_ema_policy_metadata",
    "stage4_fixed_state_digest",
    "stage4_parameter_role",
    "stage4_ssim_weight",
    "stage4_probe_candidate_order",
    "stage4_runtime_evidence_metadata",
    "stage4_validation_score",
    "teacher_forcing_probability",
    "train_stage4_optimizer_step",
    "validate_stage3_approval",
    "validate_stage3_finalization_for_stage4",
    "validate_stage4",
    "validate_stage4_config",
    "validate_stage4_extension_authorization",
]
