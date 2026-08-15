"""Contract-bound Stage1 guarded-skill training for GraphRestore V7.1.

Stage1 is deliberately narrower than the later planner stages: the eight
skills and their continuous spatial guards are teacher-forced, and only the
frozen primary single/Group-A manifests may be read.  This module keeps the
training loop independently testable while the CLI in ``scripts/`` owns path
resolution and process exit behavior.
"""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import math
import platform
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from src.data import EpisodeRequest, GraphRestoreEpisodeDataset, StatefulEpisodeSampler
from src.data.manifests import SKILLS
from src.losses.restoration import charbonnier
from src.metrics.agenticir_official import official_psnr_ssim, train_ssim_y
from src.net import (
    BackboneLoadReport,
    GuardedSkillRestormer,
    MiOStageA,
    SkillExecutionOutput,
    load_parent_backbone,
)
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
from src.training.selection import ValidationScore
from src.training.stage0_engine import CALIBRATION_COLUMNS
from src.utils.git import git_commit
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_text, utc_now_iso


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
STAGE1_SCHEMA = "graphrestore-stage1-runtime-v1"
STAGE1_BACKBONE_WHITELIST = ("decoder.skill_bank.",)
STAGE1_EMA_SCHEMA = "graphrestore-stage1-phase-aware-ema-v1"
STAGE1_EMA_SCOPE = (
    "dynamic_trainable_named_parameters_ema_"
    "frozen_parameters_and_all_buffers_bitwise_copy"
)


class Stage1ContractError(RuntimeError):
    """A Stage1 run would diverge from the frozen V7.1 contract."""


def stage1_ema_policy_metadata(decay: float) -> dict[str, object]:
    """Return the exact, checkpointed Stage1 EMA update contract."""

    if not 0.0 < decay < 1.0:
        raise ValueError("Stage1 EMA decay must be in (0,1)")
    return {
        "schema_version": STAGE1_EMA_SCHEMA,
        "scope": STAGE1_EMA_SCOPE,
        "parameter_selector": "named_parameter_requires_grad_at_each_update",
        "trainable_parameter_update": "standard_fp32_exponential_moving_average",
        "frozen_parameter_update": "copy_current_value_bitwise",
        "buffer_update": "copy_current_value_bitwise",
        "phase_transition": "dynamic_first_ema_without_shadow_reset",
        "optimizer_step_indexing": "zero_based_internal_step",
        "phase0_end_step_exclusive": 5000,
        "phase1_start_step_inclusive": 5000,
        "decay": float(decay),
    }


class Stage1PhaseAwareEMA(ExponentialMovingAverage):
    """EMA currently trainable parameters and exactly copy all fixed state.

    Stage1 changes trainability at internal optimizer step 5000.  Consulting
    ``requires_grad`` on every update lets the newly unfrozen parameters take
    their first ordinary EMA update at that boundary without resetting their
    shadows.  Frozen parameters and every buffer use ``copy_`` because even
    ``a*x + (1-a)*x`` can round an unchanged FP32 backbone by an ULP.
    """

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        core = unwrap_model(model)
        source = core.state_dict()
        if source.keys() != self.shadow.keys():
            raise RuntimeError("Stage1 EMA/model state keys drifted")
        # ``state_dict`` retains every alias key, so retain duplicate named
        # parameters too; an alias of a trainable parameter must not be
        # mistaken for a buffer/frozen value and copied instead of averaged.
        parameters = dict(core.named_parameters(remove_duplicate=False))
        if any(name not in source for name in parameters):
            raise RuntimeError("Stage1 EMA named parameters escaped model state")

        self.num_updates += 1
        for name, value in source.items():
            target = self.shadow[name]
            parameter = parameters.get(name)
            if parameter is not None and parameter.requires_grad:
                if not target.is_floating_point():
                    raise Stage1ContractError(
                        f"trainable Stage1 EMA parameter is not floating point: {name}"
                    )
                target.mul_(self.decay).add_(
                    value.detach().to(target), alpha=1.0 - self.decay
                )
            else:
                target.copy_(value.detach().to(target))

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["scope"] = STAGE1_EMA_SCOPE
        state["policy"] = stage1_ema_policy_metadata(self.decay)
        return state

    def validate_state_metadata(self, state: Mapping[str, object]) -> None:
        expected_keys = {"decay", "num_updates", "shadow", "scope", "policy"}
        if set(state) != expected_keys:
            raise Stage1ContractError(
                "Stage1 EMA state fields drifted: "
                f"expected {sorted(expected_keys)}, got {sorted(state)}"
            )
        if state.get("scope") != STAGE1_EMA_SCOPE:
            raise Stage1ContractError("Stage1 resume EMA scope drifted")
        if state.get("policy") != stage1_ema_policy_metadata(self.decay):
            raise Stage1ContractError("Stage1 resume EMA policy drifted")
        decay = state.get("decay")
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise Stage1ContractError("Stage1 resume EMA decay is invalid")
        if float(decay) != self.decay:
            raise Stage1ContractError("Stage1 resume EMA decay drifted")
        num_updates = state.get("num_updates")
        if isinstance(num_updates, bool) or not isinstance(num_updates, int):
            raise Stage1ContractError("Stage1 resume EMA update count is invalid")
        if num_updates < 0:
            raise Stage1ContractError("Stage1 resume EMA update count is negative")

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.validate_state_metadata(state)
        super().load_state_dict(state)


def build_stage1_ema(
    model: nn.Module,
    *,
    decay: float = 0.9999,
) -> Stage1PhaseAwareEMA:
    """Build the sole EMA implementation permitted for Stage1 state."""

    return Stage1PhaseAwareEMA(model, decay=decay)


@dataclass(frozen=True)
class Stage1Loss:
    total: Tensor
    charbonnier: Tensor
    ssim: Tensor
    lambda_ssim: float


@dataclass(frozen=True)
class MicroBatchTrial:
    micro_batch: int
    passed: bool
    images_per_second: float
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    completed_steps: int
    error: str | None = None


@dataclass(frozen=True)
class Stage1StepResult:
    loss: float
    charbonnier: float
    ssim: float
    lambda_ssim: float
    grad_norm: float
    active_rate: float
    residual_norm: float
    samples: int
    seconds: float


def _expect(config: Mapping[str, Any], path: Sequence[str], expected: Any) -> None:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise Stage1ContractError(f"missing Stage1 config key: {'.'.join(path)}")
        value = value[key]
    if value != expected:
        raise Stage1ContractError(
            f"Stage1 config drift at {'.'.join(path)}: expected {expected!r}, got {value!r}"
        )


def validate_stage1_config(config: Mapping[str, Any]) -> None:
    """Fail closed on scientific, data, and optimizer settings."""

    locked: tuple[tuple[tuple[str, ...], Any], ...] = (
        (("schema_version",), "1.0"),
        (("contract_version",), "GraphRestore-V7.1"),
        (("protocol_id",), PROTOCOL_ID),
        (("stage",), "stage1"),
        (("seed",), 2027),
        (("data", "allowed_groups"), ["single", "A"]),
        (("data", "forbidden_groups"), ["B", "C"]),
        (("data", "episode_sampling", "single_skill"), 0.50),
        (("data", "episode_sampling", "pair_isolation"), 0.25),
        (("data", "episode_sampling", "pair_parallel"), 0.25),
        (("data", "single_skill_target"), "clean"),
        (("data", "pair_isolation_target"), "remaining_only_subset"),
        (("data", "pair_parallel_target"), "clean"),
        (("data", "teacher_forced_presence"), True),
        (("data", "teacher_forced_continuous_guard"), True),
        (("data", "zero_guard_misuse_episode"), False),
        (("data", "crop_size"), 192),
        (("data", "crop_multiple"), 4),
        (("training", "max_steps"), 30_000),
        (("training", "effective_batch_size"), 8),
        (("training", "micro_batch_candidates"), [8, 4, 2, 1]),
        (("training", "accumulation_formula"), "effective_batch_size_divided_by_micro_batch"),
        (("model", "backbone"), "stage0_mio_stagea"),
        (("model", "adapters", "independent_per_skill"), True),
        (("model", "adapters", "insertion"), "after_each_decoder_and_refinement_block"),
        (("model", "adapters", "up_projection_zero_init"), True),
        (("model", "cooperative_mixer", "enabled_when_active_skills_gt"), 1),
        (("model", "cooperative_mixer", "output_projection_zero_init"), True),
        (("model", "skill_sum_normalization"), "divide_by_sqrt_active_count"),
        (("model", "guard_execution", "latent_modulation"), "spatial_multiply_before_sum"),
        (("model", "guard_execution", "rgb_update"), "current_plus_soft_union_guard_times_delta"),
        (("model", "guard_execution", "zero_guard_identity"), "strict"),
        (("optimization", "optimizer"), "AdamW"),
        (("optimization", "learning_rates", "skill_adapters_and_mixers"), 1.0e-4),
        (("optimization", "learning_rates", "decoder_refinement_rgb_head"), 1.0e-5),
        (("optimization", "learning_rates", "encoder_level3_level4"), 2.0e-6),
        (("optimization", "warmup_steps"), 500),
        (("optimization", "scheduler"), "cosine"),
        (("optimization", "min_lr"), 1.0e-6),
        (("optimization", "weight_decay"), 1.0e-4),
        (("optimization", "gradient_clip_norm"), 1.0),
        (("loss", "fidelity"), "charbonnier_rgb"),
        (("loss", "ssim", "channel"), "y"),
        (("loss", "ssim", "window_size"), 11),
        (("loss", "ssim", "downsample"), False),
        (("loss", "ssim", "start_step"), 6000),
        (("loss", "ssim", "weight_before_start"), 0.0),
        (("loss", "ssim", "weight_after_start"), 0.05),
        (("loss", "training_quantization"), False),
        (("loss", "hard_clamp_forward"), False),
        (("runtime", "amp_dtype"), "bf16"),
        (("runtime", "tf32"), True),
        (("runtime", "zero_grad_set_to_none"), True),
        (("runtime", "channels_last"), False),
        (("runtime", "gradient_checkpointing_initial"), False),
        (("runtime", "vram", "maximum_peak_reserved_fraction"), 0.90),
        (("runtime", "vram", "required_consecutive_no_oom_steps"), 10),
        (("runtime", "freeze_crop_micro_accum_after_step0"), True),
        (("ema", "enabled"), True),
        (("ema", "decay"), 0.9999),
        (("validation", "every_steps"), 3000),
        (("validation", "protocol"), "agenticir_official_parity"),
        (("validation", "reports"), [
            "single_skill_to_clean",
            "pair_isolation_to_remaining_only",
            "pair_parallel_to_clean",
            "per_skill_psnr_ssim",
            "skill_residual_norm",
            "actual_activation_rate",
        ]),
        (("hard_guards", "require_adapter_first_backward_nonzero_gradient"), True),
        (("hard_guards", "require_inactive_adapter_gradient_zero_or_none"), True),
        (("hard_guards", "require_zero_guard_identity_fp32_max_abs_lt"), 1.0e-7),
        (("hard_guards", "allow_mio100_exploration"), False),
        (("hard_guards", "allow_mio100_formal"), False),
        (("hard_guards", "allow_group_b_or_c_training"), False),
        (("hard_guards", "fail_on_hash_mismatch"), True),
    )
    for path, expected in locked:
        _expect(config, path, expected)

    _expect(
        config,
        ("skills", "ordered_names"),
        list(SKILLS),
    )
    freeze = config.get("training", {}).get("freeze_schedule")
    if not isinstance(freeze, list) or len(freeze) != 2:
        raise Stage1ContractError("Stage1 requires exactly two freeze phases")
    expected_boundaries = ((0, 5000), (5000, 30_000))
    expected_trainable = (
        ["skill_adapters", "cooperative_mixers"],
        [
            "skill_adapters",
            "cooperative_mixers",
            "decoder",
            "refinement",
            "rgb_head",
            "encoder_level3",
            "encoder_level4",
        ],
    )
    expected_frozen = (["stage0_backbone"], ["encoder_level1", "encoder_level2"])
    for ordinal, (phase, (start, end)) in enumerate(
        zip(freeze, expected_boundaries, strict=True)
    ):
        if not isinstance(phase, Mapping):
            raise Stage1ContractError("invalid Stage1 freeze phase")
        if (phase.get("start_step"), phase.get("end_step_exclusive")) != (start, end):
            raise Stage1ContractError("Stage1 freeze boundary drift")
        if phase.get("trainable") != expected_trainable[ordinal]:
            raise Stage1ContractError("Stage1 trainable module list drift")
        if phase.get("frozen") != expected_frozen[ordinal]:
            raise Stage1ContractError("Stage1 frozen module list drift")


def configure_reproducibility(seed: int = 2027) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def stage1_parameter_role(name: str) -> str | None:
    """Return the only allowed Stage1 optimizer role for a parameter."""

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


def set_stage1_trainability(model: nn.Module, step: int) -> dict[str, int]:
    """Apply the exact 0--5k and >=5k boundary without rebuilding AdamW."""

    if step < 0:
        raise ValueError("step must be non-negative")
    train_backbone = step >= 5000
    counts: dict[str, int] = defaultdict(int)
    for name, parameter in unwrap_model(model).named_parameters():
        role = stage1_parameter_role(name)
        enabled = role == "skills_mixers" or (
            train_backbone and role in {"decoder_refine_head", "encoder34"}
        )
        parameter.requires_grad_(enabled)
        counts[f"{role or 'permanently_frozen'}:{'trainable' if enabled else 'frozen'}"] += (
            parameter.numel()
        )
    return dict(counts)


def _enable_all_stage1_optimizer_parameters(model: nn.Module) -> None:
    """Enable all parameters that can ever train before constructing AdamW."""

    for name, parameter in unwrap_model(model).named_parameters():
        parameter.requires_grad_(stage1_parameter_role(name) is not None)


def build_stage1_optimizer(
    model: nn.Module,
    *,
    skill_lr: float = 1.0e-4,
    decoder_lr: float = 1.0e-5,
    encoder34_lr: float = 2.0e-6,
    weight_decay: float = 1.0e-4,
    fused_if_supported: bool = True,
) -> torch.optim.AdamW:
    """Construct fresh role-exclusive groups; parent optimizer is never read."""

    if min(skill_lr, decoder_lr, encoder34_lr) <= 0 or weight_decay < 0:
        raise ValueError("invalid Stage1 optimizer hyperparameters")
    _enable_all_stage1_optimizer_parameters(model)
    grouped: dict[str, list[nn.Parameter]] = {
        "skills_mixers": [],
        "decoder_refine_head": [],
        "encoder34": [],
    }
    seen: set[int] = set()
    for name, parameter in unwrap_model(model).named_parameters():
        role = stage1_parameter_role(name)
        if role is None:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"duplicate Stage1 optimizer parameter: {name}")
        grouped[role].append(parameter)
        seen.add(id(parameter))
    if any(not parameters for parameters in grouped.values()):
        raise Stage1ContractError("one or more Stage1 optimizer groups are empty")

    learning_rates = {
        "skills_mixers": float(skill_lr),
        "decoder_refine_head": float(decoder_lr),
        "encoder34": float(encoder34_lr),
    }
    groups = [
        {
            "params": parameters,
            "lr": learning_rates[role],
            "initial_lr": learning_rates[role],
            "weight_decay": float(weight_decay),
            "role": role,
        }
        for role, parameters in grouped.items()
    ]
    kwargs: dict[str, Any] = {"betas": (0.9, 0.999)}
    if fused_if_supported and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(groups, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(groups, **kwargs)
    set_stage1_trainability(model, 0)
    return optimizer


def stage1_fidelity_loss(prediction: Tensor, target: Tensor, step: int) -> Stage1Loss:
    if prediction.shape != target.shape:
        raise ValueError("Stage1 prediction/target shape mismatch")
    pixel = charbonnier(prediction, target, eps_squared=1.0e-6)
    lambda_ssim = 0.0 if step < 6000 else 0.05
    if lambda_ssim:
        ssim_term = 1.0 - train_ssim_y(prediction, target).mean()
    else:
        ssim_term = prediction.new_zeros(())
    total = pixel + lambda_ssim * ssim_term
    return Stage1Loss(total=total, charbonnier=pixel, ssim=ssim_term, lambda_ssim=lambda_ssim)


def _require_finite_tensor(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"non-finite tensor in Stage1: {name}")


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    required = ("input", "target", "guard_targets", "active_mask")
    values: dict[str, Tensor] = {}
    for key in required:
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise TypeError(f"Stage1 batch field {key!r} must be a tensor")
        values[key] = value.to(device=device, non_blocking=device.type == "cuda")
    values["input"] = values["input"].float()
    values["target"] = values["target"].float()
    values["guard_targets"] = values["guard_targets"].float()
    values["active_mask"] = values["active_mask"].bool()
    for key, value in values.items():
        if key != "active_mask":
            _require_finite_tensor(key, value)
    if values["active_mask"].ndim != 2 or values["active_mask"].shape[1] != len(SKILLS):
        raise Stage1ContractError("teacher-forced active_mask must be [B,8]")
    if bool((values["active_mask"].sum(dim=1) < 1).any().item()):
        raise Stage1ContractError("Stage1 forbids empty teacher-forced episodes")
    return values


def assert_first_backward_skill_gradients(
    model: nn.Module,
    active_mask: Tensor,
) -> None:
    """Hard gate direct adapter gradients and inactive-skill isolation."""

    core = unwrap_model(model)
    if not isinstance(core, GuardedSkillRestormer):
        raise TypeError("gradient audit requires GuardedSkillRestormer")
    active = set(
        torch.nonzero(active_mask.detach().bool().any(dim=0), as_tuple=False)
        .flatten()
        .cpu()
        .tolist()
    )
    for level in ("level3", "level2", "level1", "refinement"):
        for block_index, block in enumerate(core.decoder.skill_bank.adapters[level]):
            for skill_index, skill in enumerate(SKILLS):
                gradient = block[skill].up.weight.grad
                context = f"{level}[{block_index}].{skill}.up.weight"
                if skill_index in active:
                    if gradient is None or not bool(torch.isfinite(gradient).all().item()):
                        raise Stage1ContractError(f"missing/non-finite active adapter gradient: {context}")
                    if float(gradient.detach().abs().sum()) <= 0.0:
                        raise Stage1ContractError(f"zero active adapter gradient: {context}")
                elif gradient is not None and int(torch.count_nonzero(gradient)) != 0:
                    raise Stage1ContractError(f"inactive adapter received gradient: {context}")


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        if device.type != "cuda":
            raise Stage1ContractError("formal Stage1 bf16 autocast requires CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_stage1_optimizer_step(
    model: nn.Module,
    micro_batches: Sequence[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler | None,
    ema: Stage1PhaseAwareEMA | None,
    *,
    step: int,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
    use_bf16: bool = True,
    audit_first_backward: bool = False,
) -> Stage1StepResult:
    """Consume exactly one effective batch and one optimizer update."""

    if not micro_batches:
        raise ValueError("at least one micro batch is required")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if ema is not None and not isinstance(ema, Stage1PhaseAwareEMA):
        raise Stage1ContractError("Stage1 optimizer steps require phase-aware EMA")
    set_stage1_trainability(model, step)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    totals = defaultdict(float)
    total_samples = 0

    for micro_index, raw_batch in enumerate(micro_batches):
        batch = _batch_to_device(raw_batch, device)
        sample_count = int(batch["input"].shape[0])
        total_samples += sample_count
        with _autocast_context(device, use_bf16):
            output = model(
                batch["input"],
                active_mask=batch["active_mask"],
                guards=batch["guard_targets"],
            )
            if not torch.is_tensor(output):
                raise RuntimeError("Stage1 model unexpectedly returned a trace object")
            loss = stage1_fidelity_loss(output, batch["target"], step)
        _require_finite_tensor("prediction", output)
        _require_finite_tensor("loss", loss.total)
        (loss.total / len(micro_batches)).backward()
        if audit_first_backward and micro_index == 0:
            assert_first_backward_skill_gradients(model, batch["active_mask"])

        residual = (output.detach().float() - batch["input"]).square().mean((1, 2, 3)).sqrt()
        totals["loss"] += float(loss.total.detach()) * sample_count
        totals["charbonnier"] += float(loss.charbonnier.detach()) * sample_count
        totals["ssim"] += float(loss.ssim.detach()) * sample_count
        totals["active_rate"] += float(batch["active_mask"].float().mean()) * sample_count
        totals["residual_norm"] += float(residual.mean()) * sample_count

    parameters_with_grad = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters_with_grad:
        raise Stage1ContractError("Stage1 backward produced no gradients")
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        parameters_with_grad,
        max_norm=gradient_clip_norm,
        error_if_nonfinite=True,
    )
    grad_norm = float(grad_norm_tensor.detach())
    if not math.isfinite(grad_norm):
        raise FloatingPointError("non-finite Stage1 gradient norm")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return Stage1StepResult(
        loss=totals["loss"] / total_samples,
        charbonnier=totals["charbonnier"] / total_samples,
        ssim=totals["ssim"] / total_samples,
        lambda_ssim=0.0 if step < 6000 else 0.05,
        grad_norm=grad_norm,
        active_rate=totals["active_rate"] / total_samples,
        residual_norm=totals["residual_norm"] / total_samples,
        samples=total_samples,
        seconds=elapsed,
    )


def _mapping_of_tensors(value: object, context: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise Stage1ContractError(f"{context} must be a non-empty tensor mapping")
    if any(not isinstance(key, str) or not torch.is_tensor(tensor) for key, tensor in value.items()):
        raise Stage1ContractError(f"{context} contains non-tensor entries")
    return value  # type: ignore[return-value]


def load_stage0_best_ema_backbone(
    model: GuardedSkillRestormer,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    reference_model: MiOStageA | None = None,
) -> BackboneLoadReport:
    """Load only Stage0 EMA backbone tensors with one explicit new-module prefix."""

    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    else:
        payload = checkpoint
    if not isinstance(payload, Mapping):
        raise Stage1ContractError("Stage0 checkpoint must be a mapping")
    if payload.get("schema_version") != "graphrestore-checkpoint-v1":
        raise Stage1ContractError("Stage0 checkpoint schema mismatch")
    if payload.get("stage") != "stage0":
        raise Stage1ContractError("Stage1 parent must have stage='stage0'")
    if payload.get("model_role") != "ema_selection" or payload.get("resumable") is not False:
        raise Stage1ContractError(
            "Stage1 parent must be a non-resumable Stage0 EMA selection checkpoint"
        )
    raw_state = _mapping_of_tensors(payload.get("model"), "Stage0 model state")
    ema = payload.get("ema")
    if not isinstance(ema, Mapping):
        raise Stage1ContractError("Stage0 best_ema checkpoint lacks EMA state")
    shadow = _mapping_of_tensors(ema.get("shadow"), "Stage0 EMA shadow")
    if raw_state.keys() != shadow.keys():
        raise Stage1ContractError("Stage0 raw/EMA backbone keys differ")
    for key in raw_state:
        if tuple(raw_state[key].shape) != tuple(shadow[key].shape):
            raise Stage1ContractError(f"Stage0 raw/EMA shape mismatch: {key}")
        if "skill_bank" in key or key.startswith(("planner.", "presence_thresholds")):
            raise Stage1ContractError(f"Stage0 parent is not a pure backbone: {key}")
    report = load_parent_backbone(
        model,
        {"model": shadow},
        reference_model=reference_model,
        allowed_missing_prefixes=STAGE1_BACKBONE_WHITELIST,
    )
    if not report.missing_keys or any(
        not key.startswith(STAGE1_BACKBONE_WHITELIST) for key in report.missing_keys
    ):
        raise Stage1ContractError("Stage1 parent load escaped decoder.skill_bank whitelist")
    return report


def _move_loaded_ema_to_model(
    ema: Stage1PhaseAwareEMA,
    state: Mapping[str, Any],
) -> None:
    ema.validate_state_metadata(state)
    loaded_shadow = _mapping_of_tensors(state.get("shadow"), "Stage1 EMA shadow")
    if loaded_shadow.keys() != ema.shadow.keys():
        raise Stage1ContractError("Stage1 resume EMA keys drifted")
    for name, tensor in loaded_shadow.items():
        if tuple(tensor.shape) != tuple(ema.shadow[name].shape):
            raise Stage1ContractError(f"Stage1 resume EMA shape drifted: {name}")
        if tensor.dtype != ema.shadow[name].dtype:
            raise Stage1ContractError(f"Stage1 resume EMA dtype drifted: {name}")
    normalized = {
        name: tensor.detach().to(
            device=ema.shadow[name].device,
        )
        for name, tensor in loaded_shadow.items()
    }
    normalized_state = dict(state)
    normalized_state["shadow"] = normalized
    ema.load_state_dict(normalized_state)


def _validate_stage1_scheduler_state(
    scheduler: WarmupCosineScheduler,
    state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    *,
    step: int,
    context: str,
) -> None:
    current = scheduler.state_dict()
    if set(state) != set(current):
        raise Stage1ContractError(f"{context} scheduler state fields drifted")
    dynamic_fields = {"last_epoch", "_step_count", "_last_lr"}
    for key in set(current) - dynamic_fields:
        if state.get(key) != current.get(key):
            raise Stage1ContractError(f"{context} scheduler {key} drifted")
    last_epoch = state.get("last_epoch")
    step_count = state.get("_step_count")
    if (
        isinstance(last_epoch, bool)
        or not isinstance(last_epoch, int)
        or last_epoch != step
    ):
        raise Stage1ContractError(
            f"{context} scheduler.last_epoch must equal checkpoint step"
        )
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count != step + 1
    ):
        raise Stage1ContractError(
            f"{context} scheduler._step_count must equal checkpoint step + 1"
        )
    base_lrs = state.get("base_lrs")
    last_lrs = state.get("_last_lr")
    optimizer_groups = optimizer_state.get("param_groups")
    if not isinstance(base_lrs, list) or not isinstance(last_lrs, list):
        raise Stage1ContractError(f"{context} scheduler LR state is invalid")
    if not isinstance(optimizer_groups, list):
        raise Stage1ContractError(f"{context} optimizer groups are invalid")
    if not len(base_lrs) == len(last_lrs) == len(optimizer_groups):
        raise Stage1ContractError(f"{context} scheduler LR group count drifted")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (*base_lrs, *last_lrs)
    ):
        raise Stage1ContractError(f"{context} scheduler LR state is non-finite")
    for index, (base_lr, last_lr, group) in enumerate(
        zip(base_lrs, last_lrs, optimizer_groups, strict=True)
    ):
        if not isinstance(group, Mapping):
            raise Stage1ContractError(f"{context} optimizer group is invalid")
        if group.get("initial_lr") != base_lr:
            raise Stage1ContractError(
                f"{context} optimizer initial_lr/scheduler base_lrs drifted at group {index}"
            )
        warmup_steps = int(state["warmup_steps"])
        max_steps = int(state["max_steps"])
        min_lr = float(state["min_lr"])
        floor = min(min_lr, float(base_lr))
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
            raise Stage1ContractError(
                f"{context} optimizer/scheduler LR trajectory drifted at group {index}"
            )


def _validate_stage1_fixed_ema_state(
    model: nn.Module,
    raw_state: Mapping[str, Tensor],
    shadow: Mapping[str, Tensor],
    *,
    step: int,
    context: str,
) -> None:
    parameters = dict(unwrap_model(model).named_parameters(remove_duplicate=False))
    for name, raw_value in raw_state.items():
        parameter = parameters.get(name)
        role = stage1_parameter_role(name) if parameter is not None else None
        may_have_ema_history = (step > 0 and role == "skills_mixers") or (
            step > 5000 and role in {"decoder_refine_head", "encoder34"}
        )
        if not may_have_ema_history and not torch.equal(raw_value, shadow[name]):
            raise Stage1ContractError(
                f"{context} fixed parameter/buffer differs from EMA shadow: {name}"
            )


def _validate_stage1_sampler_state(
    sampler: StatefulEpisodeSampler,
    state: Mapping[str, Any],
    *,
    step: int,
) -> None:
    current = sampler.state_dict()
    for key in (
        "schema_version",
        "stage",
        "base_seed",
        "num_samples",
        "effective_batch_size",
    ):
        if state.get(key) != current.get(key):
            raise Stage1ContractError(f"Stage1 resume sampler {key} drifted")
    effective_batch_size = current["effective_batch_size"]
    consumed_step = state.get("consumed_optimizer_step")
    if (
        isinstance(consumed_step, bool)
        or not isinstance(consumed_step, int)
        or consumed_step != step
    ):
        raise Stage1ContractError("Stage1 resume sampler consumed step drifted")
    sample_cursor = state.get("sample_cursor")
    if (
        isinstance(sample_cursor, bool)
        or not isinstance(sample_cursor, int)
        or sample_cursor != step * effective_batch_size
    ):
        raise Stage1ContractError("Stage1 resume sampler cursor drifted")


def _validate_stage1_optimizer_state(
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
        raise Stage1ContractError("Stage1 resume optimizer state is invalid")
    if not isinstance(loaded_groups, list) or not isinstance(current_groups, list):
        raise Stage1ContractError("Stage1 resume optimizer groups are invalid")
    if len(loaded_groups) != len(current_groups):
        raise Stage1ContractError("Stage1 resume optimizer group count drifted")
    loaded_parameter_ids: set[int] = set()
    live_parameters: dict[int, nn.Parameter] = {}
    group_by_parameter_id: dict[int, Mapping[str, Any]] = {}
    for loaded_group, current_group in zip(
        loaded_groups, current_groups, strict=True
    ):
        if not isinstance(loaded_group, Mapping) or not isinstance(current_group, Mapping):
            raise Stage1ContractError("Stage1 resume optimizer group is invalid")
        loaded_parameters = loaded_group.get("params")
        current_parameters = current_group.get("params")
        if not isinstance(loaded_parameters, list) or not isinstance(
            current_parameters, list
        ):
            raise Stage1ContractError("Stage1 resume optimizer parameter IDs are invalid")
        if len(loaded_parameters) != len(current_parameters):
            raise Stage1ContractError("Stage1 resume optimizer group size drifted")
        if loaded_parameters != current_parameters:
            raise Stage1ContractError(
                "Stage1 resume optimizer parameter ID order drifted"
            )
        if loaded_group.get("role") != current_group.get("role"):
            raise Stage1ContractError("Stage1 resume optimizer role drifted")
        if set(loaded_group) != set(current_group):
            raise Stage1ContractError("Stage1 resume optimizer group fields drifted")
        for key in set(current_group) - {"params", "lr"}:
            if loaded_group.get(key) != current_group.get(key):
                raise Stage1ContractError(
                    f"Stage1 resume optimizer static field drifted: {key}"
                )
        dynamic_lr = loaded_group.get("lr")
        if (
            isinstance(dynamic_lr, bool)
            or not isinstance(dynamic_lr, (int, float))
            or not math.isfinite(float(dynamic_lr))
        ):
            raise Stage1ContractError("Stage1 resume optimizer lr is non-finite")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in loaded_parameters):
            raise Stage1ContractError("Stage1 resume optimizer parameter ID is invalid")
        loaded_parameter_ids.update(loaded_parameters)
    for current_group, live_group in zip(
        current_groups, optimizer.param_groups, strict=True
    ):
        current_parameters = current_group["params"]
        live_group_parameters = live_group["params"]
        if len(current_parameters) != len(live_group_parameters):
            raise Stage1ContractError("Stage1 resume optimizer live group size drifted")
        for parameter_id, parameter in zip(
            current_parameters, live_group_parameters, strict=True
        ):
            if not isinstance(parameter, nn.Parameter):
                raise Stage1ContractError("Stage1 resume optimizer has a non-parameter")
            live_parameters[parameter_id] = parameter
            group_by_parameter_id[parameter_id] = current_group
    if any(
        isinstance(key, bool)
        or not isinstance(key, int)
        or key not in loaded_parameter_ids
        or not isinstance(value, Mapping)
        for key, value in loaded_state.items()
    ):
        raise Stage1ContractError("Stage1 resume optimizer parameter state is invalid")
    for parameter_id, parameter_state in loaded_state.items():
        parameter = live_parameters[parameter_id]
        group = group_by_parameter_id[parameter_id]
        expected_state_keys = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad") is True:
            expected_state_keys.add("max_exp_avg_sq")
        if set(parameter_state) != expected_state_keys:
            raise Stage1ContractError("Stage1 resume Adam state fields drifted")
        state_step = parameter_state["step"]
        if torch.is_tensor(state_step):
            if state_step.numel() != 1 or not bool(torch.isfinite(state_step).all()):
                raise Stage1ContractError("Stage1 resume Adam step is invalid")
            state_step_value = float(state_step.item())
        elif isinstance(state_step, (int, float)) and not isinstance(state_step, bool):
            state_step_value = float(state_step)
        else:
            raise Stage1ContractError("Stage1 resume Adam step is invalid")
        if (
            not math.isfinite(state_step_value)
            or not state_step_value.is_integer()
            or not 1 <= int(state_step_value) <= step
        ):
            raise Stage1ContractError("Stage1 resume Adam step is invalid")
        for key in expected_state_keys - {"step"}:
            tensor = parameter_state[key]
            if (
                not torch.is_tensor(tensor)
                or tuple(tensor.shape) != tuple(parameter.shape)
                or tensor.dtype != parameter.dtype
                or not bool(torch.isfinite(tensor).all())
            ):
                raise Stage1ContractError(
                    f"Stage1 resume Adam tensor state is invalid: {key}"
                )


def _optimizer_serialized_parameter_names(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[int, str]:
    canonical_names = {
        id(parameter): name
        for name, parameter in unwrap_model(model).named_parameters()
    }
    optimizer_state = optimizer.state_dict()
    serialized_groups = optimizer_state.get("param_groups")
    if not isinstance(serialized_groups, list) or len(serialized_groups) != len(
        optimizer.param_groups
    ):
        raise Stage1ContractError("Stage1 optimizer parameter-name mapping drifted")
    result: dict[int, str] = {}
    for serialized_group, live_group in zip(
        serialized_groups, optimizer.param_groups, strict=True
    ):
        if not isinstance(serialized_group, Mapping):
            raise Stage1ContractError("Stage1 optimizer serialized group is invalid")
        serialized_ids = serialized_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(serialized_ids, list) or not isinstance(live_parameters, list):
            raise Stage1ContractError("Stage1 optimizer parameter list is invalid")
        if len(serialized_ids) != len(live_parameters):
            raise Stage1ContractError("Stage1 optimizer parameter list size drifted")
        for serialized_id, parameter in zip(
            serialized_ids, live_parameters, strict=True
        ):
            if isinstance(serialized_id, bool) or not isinstance(serialized_id, int):
                raise Stage1ContractError("Stage1 optimizer serialized ID is invalid")
            name = canonical_names.get(id(parameter))
            if name is None:
                raise Stage1ContractError(
                    "Stage1 optimizer parameter lacks a canonical model name"
                )
            if serialized_id in result:
                raise Stage1ContractError("Stage1 optimizer serialized ID is duplicated")
            result[serialized_id] = name
    return result


def _validate_stage1_optimizer_state_ledger(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    ledger: object,
    *,
    step: int,
) -> dict[int, str]:
    loaded_state = optimizer_state.get("state")
    if not isinstance(loaded_state, Mapping):
        raise Stage1ContractError("Stage1 optimizer state is invalid for ledger")
    serialized_names = _optimizer_serialized_parameter_names(model, optimizer)
    state_ids = set(loaded_state)
    if ledger is None:
        ledger_mapping: Mapping[object, object] = {
            serialized_id: serialized_names[serialized_id]
            for serialized_id in sorted(state_ids)
        }
    elif isinstance(ledger, Mapping):
        ledger_mapping = ledger
    else:
        raise Stage1ContractError("Stage1 optimizer state-name ledger is invalid")
    if set(ledger_mapping) != state_ids:
        raise Stage1ContractError(
            "Stage1 optimizer state-name ledger keys differ from optimizer state"
        )
    normalized: dict[int, str] = {}
    for serialized_id in sorted(state_ids):
        if isinstance(serialized_id, bool) or not isinstance(serialized_id, int):
            raise Stage1ContractError("Stage1 optimizer ledger ID is invalid")
        value = ledger_mapping[serialized_id]
        if not isinstance(value, str):
            raise Stage1ContractError("Stage1 optimizer ledger name is invalid")
        role = stage1_parameter_role(value)
        phase_legal = (step > 0 and role == "skills_mixers") or (
            step > 5000 and role in {"decoder_refine_head", "encoder34"}
        )
        if not phase_legal:
            raise Stage1ContractError(
                f"Stage1 optimizer ledger role is illegal at step {step}: {value}"
            )
        parameter_state = loaded_state[serialized_id]
        state_step = parameter_state["step"]
        state_step_value = (
            int(state_step.item()) if torch.is_tensor(state_step) else int(state_step)
        )
        maximum_state_step = step if role == "skills_mixers" else step - 5000
        if state_step_value > maximum_state_step:
            raise Stage1ContractError(
                "Stage1 optimizer ledger Adam step exceeds its phase-local "
                f"maximum at ID {serialized_id}"
            )
        if serialized_names.get(serialized_id) != value:
            raise Stage1ContractError(
                f"Stage1 optimizer ledger name drifted at ID {serialized_id}"
            )
        normalized[serialized_id] = value
    return normalized


def _validate_stage1_rng_state(state: Mapping[str, Any]) -> None:
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        cpu_state = state["torch_cpu"]
        if not torch.is_tensor(cpu_state) or cpu_state.dtype != torch.uint8:
            raise TypeError("invalid torch CPU RNG tensor")
        torch.Generator(device="cpu").set_state(cpu_state)
        cuda_states = state.get("torch_cuda_all")
        if torch.cuda.is_available() and cuda_states is None:
            raise ValueError("CUDA RNG state is missing")
        if cuda_states is not None:
            if not isinstance(cuda_states, (list, tuple)):
                raise TypeError("invalid CUDA RNG state list")
            if torch.cuda.is_available() and len(cuda_states) != torch.cuda.device_count():
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
        raise Stage1ContractError("Stage1 resume RNG state is invalid") from exc


def _validate_stage1_metrics(value: object, *, step: int) -> None:
    if not isinstance(value, Mapping):
        raise Stage1ContractError("Stage1 resume metrics must be a mapping")
    groups = (
        (
            "best",
            (
                "best_group_a_psnr",
                "best_group_a_ssim",
                "best_single_psnr",
                "best_single_ssim",
                "best_step",
            ),
            "best_step",
        ),
        (
            "current",
            (
                "group_a_psnr",
                "group_a_ssim",
                "single_psnr",
                "single_ssim",
                "validation_step",
            ),
            "validation_step",
        ),
    )
    for label, fields, step_field in groups:
        present = [field in value for field in fields]
        if any(present) and not all(present):
            raise Stage1ContractError(f"Stage1 resume metrics has partial {label} fields")
        if not all(present):
            continue
        metric_step = value[step_field]
        if (
            isinstance(metric_step, bool)
            or not isinstance(metric_step, int)
            or not 0 <= metric_step <= step
        ):
            raise Stage1ContractError(
                f"Stage1 resume metrics {step_field} is invalid"
            )
        for field in fields:
            if field == step_field:
                continue
            metric = value[field]
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
            ):
                raise Stage1ContractError(
                    f"Stage1 resume metrics {field} is non-finite"
                )


def save_stage1_checkpoint(
    destination: str | Path,
    *,
    step: int,
    model: nn.Module,
    ema: Stage1PhaseAwareEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: StatefulEpisodeSampler,
    provenance: Mapping[str, Any],
    metrics: Mapping[str, float | int] | None = None,
    model_as_ema: bool = False,
    pending_validation_step: int | None = None,
) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Stage1ContractError(
            "Stage1 checkpoint step must be a non-negative integer"
        )
    if not isinstance(ema, Stage1PhaseAwareEMA):
        raise Stage1ContractError("Stage1 checkpoints require phase-aware EMA")
    ema_state = ema.state_dict()
    ema.validate_state_metadata(ema_state)
    if ema.num_updates != step:
        raise Stage1ContractError(
            "Stage1 checkpoint step/EMA update count mismatch: "
            f"step={step}, num_updates={ema.num_updates}"
        )
    if provenance.get("ema_policy") != ema_state["policy"]:
        raise Stage1ContractError("Stage1 checkpoint provenance EMA policy drifted")
    _validate_stage1_fixed_ema_state(
        model,
        unwrap_model(model).state_dict(),
        ema.shadow,
        step=step,
        context="Stage1 save",
    )
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    _validate_stage1_optimizer_state(optimizer, optimizer_state, step=step)
    optimizer_state_name_ledger = _validate_stage1_optimizer_state_ledger(
        model,
        optimizer,
        optimizer_state,
        None,
        step=step,
    )
    _validate_stage1_scheduler_state(
        scheduler,
        scheduler_state,
        optimizer_state,
        step=step,
        context="Stage1 save",
    )
    if pending_validation_step is not None and pending_validation_step != step:
        raise Stage1ContractError(
            "pending validation step must equal the checkpoint optimizer step"
        )
    sampler_state = sampler.state_dict(consumed_optimizer_step=step)
    context = ema.apply_to(model) if model_as_ema else _null_model_context()
    with context:
        payload = checkpoint_payload(
            stage="stage1",
            step=step,
            model=model,
            ema_state=ema_state,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            sampler_state=sampler_state,
            provenance=provenance,
            metrics=metrics,
        )
        payload["model_role"] = "ema_selection" if model_as_ema else "raw_training_state"
        payload["resumable"] = not model_as_ema
        payload["pending_validation_step"] = pending_validation_step
        payload["optimizer_state_name_ledger"] = optimizer_state_name_ledger
        # Save while EMA weights are installed: state_dict tensors alias model
        # storage and would otherwise observe the restored raw weights.
        atomic_torch_save(payload, destination)


class _null_model_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


def resume_stage1_checkpoint(
    checkpoint: str | Path,
    *,
    model: nn.Module,
    ema: Stage1PhaseAwareEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: StatefulEpisodeSampler,
    expected_provenance: Mapping[str, Any],
    expected_validation_every: int,
    expected_max_steps: int,
) -> dict[str, Any]:
    if not isinstance(ema, Stage1PhaseAwareEMA):
        raise Stage1ContractError("Stage1 resume requires phase-aware EMA")
    if (
        isinstance(expected_validation_every, bool)
        or not isinstance(expected_validation_every, int)
        or expected_validation_every <= 0
    ):
        raise Stage1ContractError("expected_validation_every must be positive")
    if (
        isinstance(expected_max_steps, bool)
        or not isinstance(expected_max_steps, int)
        or expected_max_steps <= 0
    ):
        raise Stage1ContractError("expected_max_steps must be positive")
    # This is the sole payload read.  Validate every Stage1 contract field
    # before mutating model/optimizer/scheduler/EMA/sampler/RNG state.
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "graphrestore-checkpoint-v1"
        or payload.get("stage") != "stage1"
        or payload.get("model_role") != "raw_training_state"
        or payload.get("resumable") is not True
        or payload.get("scaler") is not None
    ):
        raise Stage1ContractError(
            "Stage1 resume requires a resumable raw Stage1 training checkpoint"
        )
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Stage1ContractError("resume checkpoint has invalid step")
    if step > expected_max_steps:
        raise Stage1ContractError("resume checkpoint step exceeds expected max_steps")
    if "pending_validation_step" not in payload:
        raise Stage1ContractError("resume checkpoint lacks pending_validation_step")
    pending_validation_step = payload.get("pending_validation_step")
    if pending_validation_step is not None:
        if isinstance(pending_validation_step, bool) or not isinstance(
            pending_validation_step, int
        ):
            raise Stage1ContractError(
                "resume pending_validation_step must be an integer or null"
            )
        if pending_validation_step != step:
            raise Stage1ContractError(
                "resume pending_validation_step differs from checkpoint step"
            )
        if not (
            pending_validation_step % expected_validation_every == 0
            or pending_validation_step == expected_max_steps
        ):
            raise Stage1ContractError(
                "resume pending_validation_step is not a validation boundary"
            )
    _validate_stage1_metrics(payload.get("metrics"), step=step)
    ema_state = payload.get("ema")
    if not isinstance(ema_state, Mapping):
        raise Stage1ContractError("resume checkpoint lacks EMA")
    ema.validate_state_metadata(ema_state)
    loaded_shadow = _mapping_of_tensors(
        ema_state.get("shadow"), "Stage1 resume EMA shadow"
    )
    if loaded_shadow.keys() != ema.shadow.keys():
        raise Stage1ContractError("Stage1 resume EMA keys drifted")
    if any(
        tuple(tensor.shape) != tuple(ema.shadow[name].shape)
        for name, tensor in loaded_shadow.items()
    ):
        raise Stage1ContractError("Stage1 resume EMA tensor shapes drifted")
    if any(
        tensor.dtype != ema.shadow[name].dtype for name, tensor in loaded_shadow.items()
    ):
        raise Stage1ContractError("Stage1 resume EMA tensor dtypes drifted")
    if ema_state.get("num_updates") != step:
        raise Stage1ContractError("Stage1 resume step/EMA update count mismatch")
    sampler_state = payload.get("sampler_state")
    if not isinstance(sampler_state, Mapping):
        raise Stage1ContractError("resume checkpoint lacks sampler state")
    if sampler_state.get("consumed_optimizer_step") != step:
        raise Stage1ContractError("checkpoint step/sampler consumed step mismatch")
    _validate_stage1_sampler_state(sampler, sampler_state, step=step)
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage1ContractError("resume checkpoint lacks provenance")
    verify_provenance(provenance, expected_provenance)
    if provenance.get("ema_policy") != ema_state.get("policy"):
        raise Stage1ContractError("Stage1 resume provenance EMA policy drifted")
    model_state = _mapping_of_tensors(payload.get("model"), "Stage1 model state")
    current_model_state = unwrap_model(model).state_dict()
    if model_state.keys() != current_model_state.keys():
        raise Stage1ContractError("Stage1 resume model keys drifted")
    for name, tensor in model_state.items():
        reference = current_model_state[name]
        if tuple(tensor.shape) != tuple(reference.shape):
            raise Stage1ContractError(f"Stage1 resume model shape drifted: {name}")
        if tensor.dtype != reference.dtype:
            raise Stage1ContractError(f"Stage1 resume model dtype drifted: {name}")
    optimizer_state = payload.get("optimizer")
    scheduler_state = payload.get("scheduler")
    rng_states = payload.get("rng_states")
    if not isinstance(optimizer_state, Mapping):
        raise Stage1ContractError("resume checkpoint lacks optimizer state")
    if not isinstance(scheduler_state, Mapping):
        raise Stage1ContractError("resume checkpoint lacks scheduler state")
    if not isinstance(rng_states, Mapping):
        raise Stage1ContractError("resume checkpoint lacks RNG state")
    _validate_stage1_optimizer_state(optimizer, optimizer_state, step=step)
    if "optimizer_state_name_ledger" not in payload:
        raise Stage1ContractError("resume checkpoint lacks optimizer state-name ledger")
    _validate_stage1_optimizer_state_ledger(
        model,
        optimizer,
        optimizer_state,
        payload.get("optimizer_state_name_ledger"),
        step=step,
    )
    _validate_stage1_scheduler_state(
        scheduler,
        scheduler_state,
        optimizer_state,
        step=step,
        context="Stage1 resume",
    )
    _validate_stage1_rng_state(rng_states)
    _validate_stage1_fixed_ema_state(
        model,
        model_state,
        loaded_shadow,
        step=step,
        context="Stage1 resume",
    )

    unwrap_model(model).load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(dict(optimizer_state))
    scheduler.load_state_dict(dict(scheduler_state))
    _move_loaded_ema_to_model(ema, ema_state)
    sampler.load_state_dict(dict(sampler_state))
    restore_rng_state(rng_states)
    set_stage1_trainability(model, step)
    return payload


def choose_micro_batch(
    model: GuardedSkillRestormer,
    *,
    device: torch.device,
    candidates: Sequence[int] = (8, 4, 2, 1),
    crop_size: int = 192,
    step: int = 0,
    required_steps: int = 10,
    maximum_reserved_fraction: float = 0.90,
    effective_batch_size: int = 8,
) -> tuple[int, tuple[MicroBatchTrial, ...]]:
    """Select fastest micro batch using ten complete AdamW+EMA optimizer steps."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise Stage1ContractError("automatic Stage1 micro-batch selection requires CUDA")
    if required_steps != 10 or maximum_reserved_fraction != 0.90:
        raise Stage1ContractError("Stage1 VRAM gate must remain 10 steps and <=90% reserved")
    if tuple(candidates) != (8, 4, 2, 1) or crop_size != 192:
        raise Stage1ContractError("Stage1 micro-batch candidates/crop drifted")
    if effective_batch_size != 8:
        raise Stage1ContractError("Stage1 effective batch must remain eight")
    rng = capture_rng_state()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    trials: list[MicroBatchTrial] = []
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in unwrap_model(model).state_dict().items()
    }
    try:
        for micro_batch in candidates:
            if effective_batch_size % micro_batch:
                raise Stage1ContractError("candidate does not divide effective batch")
            optimizer: torch.optim.Optimizer | None = None
            scheduler: WarmupCosineScheduler | None = None
            ema: Stage1PhaseAwareEMA | None = None
            image = target = guards = active = None
            model.load_state_dict(initial_state, strict=True)
            optimizer = build_stage1_optimizer(model)
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_steps=500,
                max_steps=30_000,
                min_lr=1.0e-6,
            )
            ema = build_stage1_ema(model, decay=0.9999)
            set_stage1_trainability(model, step)
            model.train()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            completed = 0
            started = 0.0
            error: str | None = None
            try:
                image = torch.rand(micro_batch, 3, crop_size, crop_size, device=device)
                target = torch.rand_like(image)
                guards = torch.ones(
                    micro_batch,
                    len(SKILLS),
                    crop_size // 4,
                    crop_size // 4,
                    device=device,
                )
                active = torch.zeros(micro_batch, len(SKILLS), dtype=torch.bool, device=device)
                active[:, :2] = True
                fixed_batch = {
                    "input": image,
                    "target": target,
                    "guard_targets": guards,
                    "active_mask": active,
                }
                accumulation = effective_batch_size // micro_batch
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                for probe_index in range(required_steps):
                    result = train_stage1_optimizer_step(
                        model,
                        [fixed_batch] * accumulation,
                        optimizer,
                        scheduler,
                        ema,
                        step=step + probe_index,
                        device=device,
                        gradient_clip_norm=1.0,
                        use_bf16=True,
                    )
                    if not math.isfinite(result.loss):
                        raise FloatingPointError("non-finite Stage1 probe loss")
                    completed += 1
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                passed = completed == required_steps and fraction <= maximum_reserved_fraction
                throughput = effective_batch_size * completed / max(elapsed, 1.0e-9)
                if not passed and fraction > maximum_reserved_fraction:
                    error = f"peak reserved fraction {fraction:.4f} exceeds 0.90"
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in str(
                    exc
                ).lower():
                    raise
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                throughput = 0.0
                passed = False
                error = f"CUDA OOM: {exc}"
            finally:
                model.zero_grad(set_to_none=True)
                image = target = guards = active = None
                del ema, scheduler, optimizer
                torch.cuda.empty_cache()
            trials.append(
                MicroBatchTrial(
                    micro_batch=micro_batch,
                    passed=passed,
                    images_per_second=throughput,
                    peak_reserved_bytes=peak,
                    peak_reserved_fraction=fraction,
                    completed_steps=completed,
                    error=error,
                )
            )
    finally:
        model.load_state_dict(initial_state, strict=True)
        set_stage1_trainability(model, 0)
        model.zero_grad(set_to_none=True)
        restore_rng_state(rng)
        torch.cuda.empty_cache()
    accepted = [trial for trial in trials if trial.passed]
    if not accepted:
        raise Stage1ContractError(
            "no crop192 Stage1 micro batch passed ten-step <=90% VRAM gate"
        )
    winner = max(accepted, key=lambda trial: (trial.images_per_second, trial.micro_batch))
    return winner.micro_batch, tuple(trials)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        raise Stage1ContractError("cannot aggregate an empty Stage1 validation bucket")
    result = math.fsum(collected) / len(collected)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite Stage1 validation aggregate")
    return result


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "count": len(rows),
        "psnr": _mean(float(row["psnr"]) for row in rows),
        "ssim": _mean(float(row["ssim"]) for row in rows),
        "residual_norm": _mean(float(row["residual_norm"]) for row in rows),
        "active_rate": _mean(float(row["active_rate"]) for row in rows),
    }


@torch.inference_mode()
def validate_stage1(
    model: GuardedSkillRestormer,
    dataset: GraphRestoreEpisodeDataset,
    *,
    device: torch.device,
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Validate only primary-val singles and Group A with official metrics."""

    if dataset.training or dataset.crop_size is not None:
        raise Stage1ContractError("Stage1 validation must be full-resolution/no augmentation")
    if any(record.group not in {"single", "A"} for record in dataset.records):
        raise Stage1ContractError("Stage1 validation dataset contains a forbidden group")
    model.eval()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(dataset.records):
        if record.group == "single":
            base = dataset[
                EpisodeRequest(index=index, episode_type="single_skill", active_slot=0)
            ]
            programs = (("single_skill", base["target"], base["active_mask"]),)
        else:
            # Synthesize the frozen recipe once, then evaluate its three
            # teacher-forced programs from the exact same x_both realization.
            # This halves validation degradation work without changing pixels.
            base = dataset[EpisodeRequest(index=index, episode_type="pair_parallel")]
            present_ids = base["present_skill_ids"]
            if not torch.is_tensor(present_ids) or tuple(present_ids.shape) != (2,):
                raise Stage1ContractError("pair validation sample lacks two skill IDs")
            isolation_i = torch.zeros(len(SKILLS), dtype=torch.bool)
            isolation_j = torch.zeros(len(SKILLS), dtype=torch.bool)
            isolation_i[int(present_ids[0])] = True
            isolation_j[int(present_ids[1])] = True
            programs = (
                ("pair_isolation", base["target_after_i"], isolation_i),
                ("pair_isolation", base["target_after_j"], isolation_j),
                ("pair_parallel", base["gt_clean"], base["active_mask"]),
            )
        for episode_type, target_tensor, active_tensor in programs:
            raw_batch = {
                "input": base["input"].unsqueeze(0),
                "target": target_tensor.unsqueeze(0),
                "guard_targets": base["guard_targets"].unsqueeze(0),
                "active_mask": active_tensor.unsqueeze(0),
            }
            batch = _batch_to_device(raw_batch, device)
            with _autocast_context(device, use_bf16):
                traced = model(
                    batch["input"],
                    active_mask=batch["active_mask"],
                    guards=batch["guard_targets"],
                    return_trace=True,
                )
            if not isinstance(traced, SkillExecutionOutput):
                raise RuntimeError("Stage1 validation requires execution trace")
            prediction = traced.final.detach().float().cpu()
            target = batch["target"].detach().float().cpu()
            metric = official_psnr_ssim(prediction, target, quantize=True)
            residual = (prediction - batch["input"].detach().float().cpu()).square().mean().sqrt()
            active_ids = torch.nonzero(active_tensor, as_tuple=False).flatten().tolist()
            row = {
                "sample_id": record.sample_id,
                "episode_type": episode_type,
                "combination": "+".join(record.operator_order),
                "active_skills": [SKILLS[item] for item in active_ids],
                "psnr": float(metric.psnr.item()),
                "ssim": float(metric.ssim.item()),
                "residual_norm": float(residual.item()),
                "active_rate": float(active_tensor.float().mean()),
            }
            if not all(
                math.isfinite(float(row[key]))
                for key in ("psnr", "ssim", "residual_norm", "active_rate")
            ):
                raise FloatingPointError(f"non-finite Stage1 validation row: {record.sample_id}")
            rows.append(row)

    by_episode = {
        episode: _metric_summary([row for row in rows if row["episode_type"] == episode])
        for episode in ("single_skill", "pair_isolation", "pair_parallel")
    }
    pair_rows = [row for row in rows if row["episode_type"] == "pair_parallel"]
    combination_names = sorted({str(row["combination"]) for row in pair_rows})
    if len(combination_names) != 8:
        raise Stage1ContractError("primary-val must contain exactly eight Group-A combinations")
    by_combination = {
        name: _metric_summary([row for row in pair_rows if row["combination"] == name])
        for name in combination_names
    }
    group_a = {
        "count": len(pair_rows),
        "combination_count": len(by_combination),
        "psnr": _mean(float(value["psnr"]) for value in by_combination.values()),
        "ssim": _mean(float(value["ssim"]) for value in by_combination.values()),
        "residual_norm": _mean(float(value["residual_norm"]) for value in by_combination.values()),
        "active_rate": _mean(float(value["active_rate"]) for value in by_combination.values()),
    }
    per_skill: dict[str, dict[str, float | int]] = {}
    for skill in SKILLS:
        selected = [row for row in rows if skill in row["active_skills"]]
        per_skill[skill] = _metric_summary(selected)
    return {
        "schema_version": "graphrestore-stage1-validation-v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now_iso(),
        "dataset": "primary_val_single_and_group_a_only",
        "output_quantization": "clamp_round_uint8",
        "episodes": by_episode,
        "group_a_equal_combination_mean": group_a,
        "group_a_combinations": by_combination,
        "per_skill": per_skill,
        "image_program_evaluations": len(rows),
    }


def validation_score(summary: Mapping[str, Any], step: int) -> ValidationScore:
    group_a = summary["group_a_equal_combination_mean"]
    single = summary["episodes"]["single_skill"]
    return ValidationScore(
        group_a_psnr=float(group_a["psnr"]),
        group_a_ssim=float(group_a["ssim"]),
        single_psnr=float(single["psnr"]),
        single_ssim=float(single["ssim"]),
        step=step,
    )


def append_stage1_calibration_history(
    path: str | Path,
    *,
    step: int,
    summary: Mapping[str, Any],
) -> None:
    """Append Stage1 restoration metrics to the shared 28-column ledger."""

    destination = Path(path)
    rows: list[dict[str, str]] = []
    if destination.is_file():
        with destination.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CALIBRATION_COLUMNS:
                raise Stage1ContractError("calibration history schema drifted")
            rows.extend(dict(row) for row in reader)
    episodes = summary.get("episodes")
    group_a = summary.get("group_a_equal_combination_mean")
    if not isinstance(episodes, Mapping) or not isinstance(group_a, Mapping):
        raise Stage1ContractError("Stage1 validation summary lacks restoration metrics")
    single = episodes.get("single_skill")
    if not isinstance(single, Mapping):
        raise Stage1ContractError("Stage1 validation summary lacks single_skill metrics")
    row = {column: "" for column in CALIBRATION_COLUMNS}
    row.update(
        {
            "step": str(step),
            "single_psnr": f"{float(single['psnr']):.12g}",
            "single_ssim": f"{float(single['ssim']):.12g}",
            "group_a_psnr": f"{float(group_a['psnr']):.12g}",
            "group_a_ssim": f"{float(group_a['ssim']):.12g}",
        }
    )
    if row in rows:
        return
    rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CALIBRATION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(destination, stream.getvalue())


def render_stage1_report(
    summary: Mapping[str, Any],
    *,
    step: int,
    best_score: ValidationScore,
    checkpoint: Path,
) -> str:
    episodes = summary["episodes"]
    lines = [
        "# Stage1 Guarded Skill Bank",
        "",
        f"- Protocol: `{PROTOCOL_ID}`",
        f"- Validation step: {step}",
        "- Validation data: frozen `primary_val` singles + Group A only (no MiO100)",
        "- Metric: AgenticIR parity PSNR-RGB / SSIM-Y, output clamp-round-uint8",
        f"- Selected EMA checkpoint: `{checkpoint}`",
        f"- Best Group-A PSNR/SSIM: {best_score.group_a_psnr:.6f} / {best_score.group_a_ssim:.8f}",
        "",
        "## Episode metrics",
        "",
        "| Episode | Count | PSNR | SSIM | Residual norm | Active rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for episode in ("single_skill", "pair_isolation", "pair_parallel"):
        row = episodes[episode]
        lines.append(
            f"| {episode} | {row['count']} | {row['psnr']:.6f} | "
            f"{row['ssim']:.8f} | {row['residual_norm']:.8f} | {row['active_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-skill activation diagnostics",
            "",
            "| Skill | Evaluations | PSNR | SSIM | Residual norm | Active rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for skill in SKILLS:
        row = summary["per_skill"][skill]
        lines.append(
            f"| {skill} | {row['count']} | {row['psnr']:.6f} | {row['ssim']:.8f} | "
            f"{row['residual_norm']:.8f} | {row['active_rate']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def dependency_versions() -> dict[str, Any]:
    packages = {}
    for package in ("basicsr", "numpy", "opencv-python", "pyiqa", "PyYAML", "torch", "torchvision"):
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


def build_stage1_provenance(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    resolved_path: Path,
    resolved: Mapping[str, Any],
    parent_checkpoint: Path,
    micro_batch: int,
    max_steps: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    train_manifest = Path(str(resolved[config["paths"]["train_manifest_key"]])).resolve()
    val_manifest = Path(str(resolved[config["paths"]["val_manifest_key"]])).resolve()
    expected = resolved.get("expected_identity")
    if not isinstance(expected, Mapping):
        raise Stage1ContractError("resolved paths lacks expected_identity")
    manifests = expected.get("manifests")
    if not isinstance(manifests, Mapping):
        raise Stage1ContractError("resolved paths lacks manifest identities")
    actual_train = sha256_file(train_manifest)
    actual_val = sha256_file(val_manifest)
    if actual_train != manifests.get("primary_train") or actual_val != manifests.get("primary_val"):
        raise Stage1ContractError("primary manifest hash mismatch")
    agenticir_repo = Path(str(resolved["agenticir_repo"])).resolve()
    mioir_repo = Path(str(resolved["mioir_repo"])).resolve()
    agenticir_commit = git_commit(agenticir_repo)
    mioir_commit = git_commit(mioir_repo)
    if agenticir_commit != expected.get("agenticir_commit"):
        raise Stage1ContractError("AgenticIR commit mismatch")
    if mioir_commit != expected.get("mioir_commit"):
        raise Stage1ContractError("MiOIR commit mismatch")
    runtime = {
        "crop_size": 192,
        "micro_batch": micro_batch,
        "effective_batch_size": 8,
        "accumulation_steps": accumulation_steps,
        "max_steps": max_steps,
        "amp_dtype": "bf16",
        "tf32": True,
    }
    return {
        "schema_version": STAGE1_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": sha256_file(config_path),
        "config_semantic_sha256": sha256_json(config),
        "resolved_paths_sha256": sha256_file(resolved_path),
        "semantic_source_sha256": semantic_source_hashes(
            config_path.resolve().parents[1],
            entrypoints=("scripts/train_stage1_skills.py",),
        ),
        "manifests": {
            "primary_train": {"path": str(train_manifest), "sha256": actual_train},
            "primary_val": {"path": str(val_manifest), "sha256": actual_val},
        },
        "parent_checkpoint": {
            "path": str(parent_checkpoint),
            "sha256": sha256_file(parent_checkpoint),
            "source": "stage0_best_ema_shadow",
            "allowed_new_prefixes": list(STAGE1_BACKBONE_WHITELIST),
        },
        "repositories": {
            "agenticir_commit": agenticir_commit,
            "mioir_commit": mioir_commit,
        },
        "ema_policy": stage1_ema_policy_metadata(float(config["ema"]["decay"])),
        "runtime": runtime,
        "dependency_versions": dependency_versions(),
    }


def append_jsonl(handle: TextIO, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    handle.write(payload + "\n")
    handle.flush()


def lr_by_role(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result = {}
    for group in optimizer.param_groups:
        role = str(group.get("role", "unknown"))
        value = float(group["lr"])
        if not math.isfinite(value):
            raise FloatingPointError("non-finite Stage1 learning rate")
        result[role] = value
    return result


def micro_batch_trials_json(trials: Sequence[MicroBatchTrial]) -> list[dict[str, Any]]:
    return [asdict(trial) for trial in trials]


__all__ = [
    "MicroBatchTrial",
    "PROTOCOL_ID",
    "STAGE1_BACKBONE_WHITELIST",
    "STAGE1_EMA_SCHEMA",
    "STAGE1_EMA_SCOPE",
    "STAGE1_SCHEMA",
    "Stage1ContractError",
    "Stage1Loss",
    "Stage1PhaseAwareEMA",
    "Stage1StepResult",
    "append_jsonl",
    "assert_first_backward_skill_gradients",
    "build_stage1_optimizer",
    "build_stage1_ema",
    "build_stage1_provenance",
    "choose_micro_batch",
    "append_stage1_calibration_history",
    "configure_reproducibility",
    "dependency_versions",
    "load_stage0_best_ema_backbone",
    "lr_by_role",
    "micro_batch_trials_json",
    "render_stage1_report",
    "resume_stage1_checkpoint",
    "save_stage1_checkpoint",
    "set_stage1_trainability",
    "stage1_fidelity_loss",
    "stage1_ema_policy_metadata",
    "stage1_parameter_role",
    "train_stage1_optimizer_step",
    "validate_stage1",
    "validate_stage1_config",
    "validation_score",
]
