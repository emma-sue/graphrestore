"""Contract-bound Stage0 training, validation, and checkpoint primitives.

This module deliberately knows nothing about MiO100.  Stage0 validation accepts
only the frozen ``primary_val`` episode dataset and aggregates the eight single
tasks and eight Group-A tasks with equal task weight.
"""

from __future__ import annotations

import csv
import importlib.metadata
import io
import math
import platform
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from src.data.episode_dataset import GraphRestoreEpisodeDataset, EpisodeDatasetError
from src.data.manifests import ALLOWED_GROUP_A, ALLOWED_SINGLE
from src.data.scale_canonicalizer import bgr_uint8_to_rgb_float
from src.losses.restoration import charbonnier, restoration_loss
from src.metrics.agenticir_official import official_psnr_ssim
from src.net.mio_stagea import BackboneLoadReport, MiOStageA, load_parent_backbone
from src.training.checkpointing import (
    atomic_torch_save,
    checkpoint_payload,
    load_checkpoint,
)
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import (
    WarmupCosineScheduler,
    build_adamw,
    parameter_groups,
    set_stage0_trainability,
)
from src.training.provenance import semantic_source_hashes
from src.training.runtime import autocast_context, move_training_batch
from src.training.selection import ValidationScore
from src.utils.git import git_commit
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_text, load_json, load_yaml, utc_now_iso
from src.utils.paths import load_resolved_paths, resolve_config_path


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
STAGE0_CHECKPOINT_STAGE = "stage0"


class Stage0ContractError(RuntimeError):
    """The requested run would diverge from the frozen V7.1 Stage0 contract."""


class Stage0RestorationDataset(GraphRestoreEpisodeDataset):
    """Fast Stage0-only view: synthesize ``both -> clean`` exactly once.

    The general episode dataset intentionally constructs subset targets and
    dense guards for Stage1+.  Stage0 has no skill/guard supervision, so doing
    that would replay both single-operator subsets for every pair and waste
    roughly three times the CPU degradation work.  This view retains the exact
    base-class manifest parsing, optimizer-step geometry, crop-first replay,
    RNG preservation, augmentation, and low-resolution canonicalizer, while
    returning no fabricated subset/guard fields.
    """

    def __getitem__(self, index: object) -> dict[str, Any]:
        (
            actual_index,
            episode_type,
            _active_slot,
            absolute_step,
            sample_cursor,
        ) = self._resolve_request(index)  # type: ignore[arg-type]
        if episode_type not in {"stage0_restoration", "restoration"}:
            raise EpisodeDatasetError(
                f"Stage0 fast dataset cannot serve episode {episode_type!r}"
            )
        recipe = self.records[actual_index]
        clean_bgr = cv2.imread(str(recipe.clean_path), cv2.IMREAD_COLOR)
        if clean_bgr is None:
            raise EpisodeDatasetError(f"unreadable clean image: {recipe.clean_path}")
        if clean_bgr.dtype != np.uint8 or clean_bgr.ndim != 3 or clean_bgr.shape[2] != 3:
            raise EpisodeDatasetError(f"invalid clean image: {recipe.clean_path}")
        height, width = clean_bgr.shape[:2]
        if height % 4 or width % 4:
            raise EpisodeDatasetError(
                f"clean dimensions must be divisible by four: {(height, width)}"
            )
        geometry = self._geometry(recipe.sample_id, absolute_step, height, width)
        top, left, crop_height, crop_width, hflip, vflip, rotation = geometry
        crop_box = (
            (top, left, crop_height, crop_width)
            if self.crop_size is not None
            else None
        )
        if crop_box is None:
            clean_episode = np.ascontiguousarray(clean_bgr)
            applied = self.adapter.apply_sequence(
                clean_bgr,
                recipe.operator_params,
                clean_id=recipe.clean_id,
                capture_traces=False,
            )
        else:
            clean_episode = clean_bgr[
                top : top + crop_height, left : left + crop_width
            ].copy()
            applied = self.adapter.apply_sequence_crop(
                clean_bgr,
                recipe.operator_params,
                clean_id=recipe.clean_id,
                crop_box=crop_box,
                capture_traces=False,
            )
        if applied.contains_low_resolution:
            input_rgb = self.canonicalizer.canonicalize_native_lq(
                applied.output_bgr_uint8, scale=4
            )
        else:
            input_rgb = bgr_uint8_to_rgb_float(applied.output_bgr_uint8)
        target_rgb = bgr_uint8_to_rgb_float(clean_episode)
        expected = tuple(target_rgb.shape)
        if tuple(input_rgb.shape) != expected:
            raise EpisodeDatasetError(
                f"{recipe.sample_id}: Stage0 input {tuple(input_rgb.shape)} "
                f"does not match target {expected}"
            )
        local_geometry = (
            0,
            0,
            crop_height,
            crop_width,
            hflip,
            vflip,
            rotation,
        )
        input_rgb = self._transform_tensor(input_rgb, local_geometry)
        target_rgb = self._transform_tensor(target_rgb, local_geometry)
        return {
            "input": input_rgb,
            "target": target_rgb,
            "gt_clean": target_rgb,
            "episode_type": episode_type,
            "sample_id": recipe.sample_id,
            "operator_order": " + ".join(recipe.operator_order),
            "group": recipe.group,
            "split": recipe.split,
            "has_pair": recipe.is_pair,
            "contains_low_resolution": recipe.contains_low_resolution,
            "sample_index": actual_index,
            "absolute_step": absolute_step,
            "sample_cursor": sample_cursor,
            "crop_box": torch.tensor(
                [top, left, crop_height, crop_width], dtype=torch.long
            ),
            "augmentation": torch.tensor(
                [int(hflip), int(vflip), rotation], dtype=torch.long
            ),
        }


@dataclass(frozen=True)
class Stage0Runtime:
    crop_size: int
    micro_batch: int
    effective_batch: int
    accumulation_steps: int
    gradient_checkpointing: bool
    schedule_max_steps: int
    target_step: int
    integration: bool
    torch_compile: bool = False

    def __post_init__(self) -> None:
        if self.crop_size != 192:
            raise Stage0ContractError("the main Stage0 runtime requires crop_size=192")
        if self.micro_batch not in {8, 4, 2, 1}:
            raise Stage0ContractError("micro_batch must be one of 8,4,2,1")
        if self.effective_batch != 8:
            raise Stage0ContractError("Stage0 effective batch must remain 8")
        if self.accumulation_steps != self.effective_batch // self.micro_batch:
            raise Stage0ContractError("invalid Stage0 accumulation factor")
        if self.target_step <= 0 or self.target_step > self.schedule_max_steps:
            raise Stage0ContractError("target step must be in (0, schedule_max_steps]")
        if not isinstance(self.torch_compile, bool):
            raise Stage0ContractError("torch_compile must be a frozen boolean")


@dataclass(frozen=True)
class Stage0StepResult:
    step: int
    loss: float
    charbonnier: float
    ssim_loss: float
    lambda_ssim: float
    grad_norm: float
    micro_batches: int
    images: int


@dataclass(frozen=True)
class Stage0ValidationResult:
    single_psnr: float
    single_ssim: float
    group_a_psnr: float
    group_a_ssim: float
    image_count: int
    task_means: Mapping[str, Mapping[str, float | int | str]]

    def selection_score(self, step: int) -> ValidationScore:
        return ValidationScore(
            group_a_psnr=self.group_a_psnr,
            group_a_ssim=self.group_a_ssim,
            single_psnr=self.single_psnr,
            single_ssim=self.single_ssim,
            step=step,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "single_psnr": self.single_psnr,
            "single_ssim": self.single_ssim,
            "group_a_psnr": self.group_a_psnr,
            "group_a_ssim": self.group_a_ssim,
            "image_count": self.image_count,
            "task_means": dict(self.task_means),
        }


def _nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise Stage0ContractError(f"missing config field: {'.'.join(keys)}")
        value = value[key]
    return value


def load_and_validate_stage0_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load both YAML files and reject changes to scientific Stage0 constants."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    if not isinstance(raw, Mapping):
        raise Stage0ContractError("Stage0 config must be a YAML mapping")
    config = dict(raw)
    required: tuple[tuple[tuple[str, ...], object], ...] = (
        (("contract_version",), "GraphRestore-V7.1"),
        (("protocol_id",), PROTOCOL_ID),
        (("stage",), "stage0"),
        (("seed",), 2027),
        (("data", "allowed_groups"), ["single", "A"]),
        (("data", "forbidden_groups"), ["B", "C"]),
        (("data", "crop_size"), 192),
        (("model", "widths"), [48, 96, 192, 384]),
        (("model", "encoder_blocks"), [4, 6, 6, 8]),
        (("model", "decoder_blocks"), [6, 6, 4]),
        (("model", "refinement_blocks"), 4),
        (("training", "max_steps"), 60_000),
        (("training", "effective_batch_size"), 8),
        (("training", "micro_batch_candidates"), [8, 4, 2, 1]),
        (("optimization", "optimizer"), "AdamW"),
        (("optimization", "betas"), [0.9, 0.999]),
        (("optimization", "weight_decay"), 1.0e-4),
        (("optimization", "weight_decay_norm_bias"), 0.0),
        (("optimization", "warmup_steps"), 1000),
        (("optimization", "min_lr"), 1.0e-6),
        (("loss", "charbonnier_epsilon_squared"), 1.0e-6),
        (("loss", "ssim", "start_step"), 12_000),
        (("loss", "ssim", "weight_before_start"), 0.0),
        (("loss", "ssim", "weight_after_start"), 0.05),
        (("runtime", "amp_dtype"), "bf16"),
        (("runtime", "vram", "maximum_peak_reserved_fraction"), 0.90),
        (("ema", "decay"), 0.9999),
        (("validation", "every_steps"), 4000),
        (("checkpoint", "save_every_steps"), 4000),
        (("hard_guards", "allow_mio100_exploration"), False),
        (("hard_guards", "allow_mio100_formal"), False),
        (("hard_guards", "allow_group_b_or_c_training"), False),
    )
    for key_path, expected in required:
        actual = _nested(config, *key_path)
        if actual != expected:
            raise Stage0ContractError(
                f"locked config mismatch at {'.'.join(key_path)}: "
                f"expected {expected!r}, got {actual!r}"
            )
    curriculum = _nested(config, "data", "curriculum")
    if curriculum != [
        {
            "start_step": 0,
            "end_step_exclusive": 10_000,
            "single_probability": 0.60,
            "group_a_probability": 0.40,
        },
        {
            "start_step": 10_000,
            "end_step_exclusive": 60_000,
            "single_probability": 0.30,
            "group_a_probability": 0.70,
        },
    ]:
        raise Stage0ContractError("Stage0 curriculum differs from the frozen optimizer-step schedule")
    resolved_value = _nested(config, "paths", "resolved_paths")
    resolved_path = resolve_config_path(config_path, str(resolved_value))
    resolved = load_resolved_paths(resolved_path)
    return config, resolved


def assert_stage0_preflight(project_root: str | Path) -> None:
    """Require the durable data and metric parity artifacts before GPU work."""

    root = Path(project_root).resolve()
    artifacts = (
        (root / "artifacts/audits/data_audit.json", "data audit"),
        (root / "artifacts/audits/degradation_parity.json", "degradation parity"),
        (root / "artifacts/metrics/metric_parity_summary.json", "metric parity"),
    )
    loaded: dict[str, Mapping[str, Any]] = {}
    for path, label in artifacts:
        if not path.is_file():
            raise Stage0ContractError(f"missing required {label} artifact: {path}")
        value = load_json(path)
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            raise Stage0ContractError(f"required {label} did not pass: {path}")
        if int(value.get("failure_count", -1)) != 0:
            raise Stage0ContractError(f"required {label} contains failures: {path}")
        loaded[label] = value
    metric_facts = loaded["metric parity"].get("facts")
    if not isinstance(metric_facts, Mapping):
        raise Stage0ContractError("metric parity artifact lacks facts")
    if float(metric_facts.get("max_psnr_abs_diff", math.inf)) > 1.0e-5:
        raise Stage0ContractError("PSNR parity exceeds 1e-5")
    if float(metric_facts.get("max_ssim_abs_diff", math.inf)) > 1.0e-5:
        raise Stage0ContractError("SSIM parity exceeds 1e-5")
    if metric_facts.get("canonical_float_exact") is not True:
        raise Stage0ContractError("online low-resolution float canonicalization is not exact")
    degradation_facts = loaded["degradation parity"].get("facts")
    if not isinstance(degradation_facts, Mapping):
        raise Stage0ContractError("degradation parity artifact lacks facts")
    if degradation_facts.get("all_exact") is not True:
        raise Stage0ContractError("official degradation parity is not byte exact")
    if int(degradation_facts.get("pairs", 0)) != 16:
        raise Stage0ContractError("degradation parity must cover 2 recipes x 8 operators")
    assert_validation_vram_preflight(root)


def assert_validation_vram_preflight(project_root: str | Path) -> None:
    """Require an exact-code, exact-hardware maximum-size validation PASS."""

    root = Path(project_root).resolve()
    path = root / "artifacts/audits/validation_vram_probe.json"
    if not path.is_file():
        raise Stage0ContractError(f"missing validation VRAM probe artifact: {path}")
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise Stage0ContractError("validation VRAM probe must be a mapping")
    if (
        payload.get("schema_version") != "graphrestore-validation-vram-probe-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("completed") is not True
        or payload.get("passed") is not True
        or payload.get("maximum_peak_reserved_fraction") != 0.90
    ):
        raise Stage0ContractError("validation VRAM probe did not record a locked PASS")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise Stage0ContractError("validation VRAM probe lacks bindings")
    for logical in ("config", "resolved_paths", "clean_val_manifest", "parent_checkpoint"):
        binding = bindings.get(logical)
        if not isinstance(binding, Mapping):
            raise Stage0ContractError(f"validation VRAM probe lacks {logical} binding")
        bound_path = Path(str(binding.get("path", ""))).resolve(strict=False)
        if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
            raise Stage0ContractError(f"validation VRAM probe {logical} binding drifted")
    code = bindings.get("code_sha256")
    if not isinstance(code, Mapping) or not code:
        raise Stage0ContractError("validation VRAM probe lacks code bindings")
    for relative, expected_sha in code.items():
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise Stage0ContractError("validation VRAM probe code binding is malformed")
        source = (root / relative).resolve(strict=False)
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise Stage0ContractError(
                f"validation VRAM code binding escaped project root: {source}"
            ) from exc
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise Stage0ContractError(
                f"validation VRAM probe code binding drifted: {relative}"
            )
    if not torch.cuda.is_available():
        raise Stage0ContractError("validation VRAM probe requires the bound CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = payload.get("hardware")
    if not isinstance(hardware, Mapping):
        raise Stage0ContractError("validation VRAM probe lacks hardware binding")
    current_hardware = {
        "device_index": int(device.index or 0),
        "gpu": torch.cuda.get_device_name(device),
        "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if dict(hardware) != current_hardware:
        raise Stage0ContractError("validation VRAM probe hardware binding drifted")
    probes = payload.get("probes")
    if not isinstance(probes, list) or len(probes) != 2:
        raise Stage0ContractError("validation VRAM probe must contain exactly two models")
    expected_models = {"stage0_mio_stagea", "expanded_guarded_skill_restormer"}
    observed_models: set[str] = set()
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise Stage0ContractError("validation VRAM probe row is malformed")
        model_name = str(probe.get("model"))
        observed_models.add(model_name)
        fraction = probe.get("peak_reserved_fraction")
        if (
            probe.get("passed") is not True
            or probe.get("finite") is not True
            or probe.get("shape_matches_input") is not True
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or float(fraction) > 0.90
        ):
            raise Stage0ContractError("validation VRAM probe row failed the 0.90 gate")
        expected_metric_on_cuda = model_name == "stage0_mio_stagea"
        if (
            probe.get("official_metric_included_on_cuda") is not expected_metric_on_cuda
            or probe.get("official_metric_finite") is not True
        ):
            raise Stage0ContractError(
                f"validation VRAM probe metric coverage drifted for {model_name}"
            )
    if observed_models != expected_models:
        raise Stage0ContractError("validation VRAM probe model coverage drifted")


def load_stage0_compile_ab_decision(
    project_root: str | Path,
    *,
    config_path: str | Path,
    parent_checkpoint: str | Path,
    primary_train_manifest: str | Path,
    device: torch.device,
) -> Mapping[str, Any]:
    """Validate the bound D-011 result and return its immutable decision."""

    root = Path(project_root).resolve()
    config = Path(config_path).resolve()
    parent = Path(parent_checkpoint).resolve()
    primary_manifest = Path(primary_train_manifest).resolve()
    path = root / "artifacts/audits/stage0_compile_ab.json"
    if not path.is_file():
        raise Stage0ContractError(f"missing required Stage0 compile A/B artifact: {path}")
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise Stage0ContractError("Stage0 compile A/B artifact must be a mapping")
    required = {
        "schema_version": "graphrestore-stage0-compile-ab-v1",
        "protocol_id": PROTOCOL_ID,
        "completed": True,
        "safe_default": "eager",
        "tolerances": {
            "output_max_abs": 2.0e-3,
            "output_mean_abs": 1.0e-5,
            "loss_max_abs": 1.0e-5,
            "loss_mean_abs": 2.0e-6,
            "parameter_max_abs": 5.0e-5,
            "parameter_mean_abs": 1.0e-7,
            "minimum_throughput_ratio": 1.05,
        },
        "ab_design": {
            "steps": 20,
            "micro_batch": 8,
            "effective_batch": 8,
            "crop_size": 192,
            "steady_state_excludes_step": 0,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise Stage0ContractError(
                f"Stage0 compile A/B binding drifted at {key}: "
                f"expected {expected!r}, got {payload.get(key)!r}"
            )
    bindings = (
        ("config", config),
        ("parent_checkpoint", parent),
    )
    for name, bound_path in bindings:
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise Stage0ContractError(f"Stage0 compile A/B lacks {name} binding")
        if value.get("path") != str(bound_path) or value.get("sha256") != sha256_file(
            bound_path
        ):
            raise Stage0ContractError(f"Stage0 compile A/B {name} binding drifted")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise Stage0ContractError("Stage0 compile A/B lacks data binding")
    manifest_binding = data.get("primary_train_manifest")
    expected_manifest_binding = {
        "path": str(primary_manifest),
        "sha256": sha256_file(primary_manifest),
    }
    if manifest_binding != expected_manifest_binding:
        raise Stage0ContractError("Stage0 compile A/B primary manifest binding drifted")
    profile_script = root / "scripts/profile_stage0_compile.py"
    if payload.get("profile_script_sha256") != sha256_file(profile_script):
        raise Stage0ContractError("Stage0 compile A/B profiler code changed after measurement")
    code_sha256 = payload.get("code_sha256")
    if not isinstance(code_sha256, Mapping) or not code_sha256:
        raise Stage0ContractError("Stage0 compile A/B lacks measured code bindings")
    for relative, expected_sha in code_sha256.items():
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise Stage0ContractError("Stage0 compile A/B code binding is malformed")
        bound_source = root / relative
        if not bound_source.is_file() or sha256_file(bound_source) != expected_sha:
            raise Stage0ContractError(
                f"Stage0 compile A/B measured source changed: {relative}"
            )
    recommend = payload.get("recommend_torch_compile")
    if not isinstance(recommend, bool):
        raise Stage0ContractError("Stage0 compile A/B recommendation is not boolean")
    numerical_pass = payload.get("numerical_pass") is True
    ratio = float(payload.get("steady_state_throughput_ratio", math.nan))
    valid_enable = (
        payload.get("compiled_error") is None
        and numerical_pass
        and math.isfinite(ratio)
        and ratio >= 1.05
    )
    if recommend != valid_enable:
        raise Stage0ContractError("Stage0 compile A/B recommendation contradicts its evidence")
    return payload


def stage0_lambda_ssim(step: int) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    return 0.0 if step < 12_000 else 0.05


def build_stage0_model(
    parent_checkpoint_or_payload: str | Path | Mapping[str, Any],
    *,
    gradient_checkpointing: bool = False,
) -> tuple[MiOStageA, BackboneLoadReport]:
    """Instantiate the pure host and strictly load only ``payload['model']``."""

    model = MiOStageA(gradient_checkpointing=gradient_checkpointing)
    # Passing the same pure host as reference avoids allocating a redundant
    # 25M-parameter CPU model while retaining the exact 495-key audit.
    report = load_parent_backbone(
        model,
        parent_checkpoint_or_payload,
        reference_model=model,
    )
    if report.source_tensor_count != 495 or report.loaded_count != 495:
        raise Stage0ContractError(
            f"warm start must load all 495 host tensors, got {report.loaded_count}"
        )
    if report.missing_keys or report.unexpected_keys or report.shape_mismatches:
        raise Stage0ContractError("pure Stage0 warm start was not structurally exact")
    return model, report


def build_stage0_optimizer(
    model: nn.Module, config: Mapping[str, Any]
) -> tuple[torch.optim.AdamW, WarmupCosineScheduler]:
    """Register every parameter before the early encoder freeze is applied."""

    learning_rates = _nested(config, "optimization", "learning_rates")
    if not isinstance(learning_rates, Mapping):
        raise Stage0ContractError("optimization.learning_rates must be a mapping")
    groups = parameter_groups(
        model,
        (
            (("decoder.",), float(learning_rates["decoder_refinement_rgb_head"])),
            (
                (
                    "encoder.level3.",
                    "encoder.level4.",
                    "encoder.down23.",
                    "encoder.down34.",
                ),
                float(learning_rates["encoder_level3_level4"]),
            ),
            (
                (
                    "encoder.patch.",
                    "encoder.level1.",
                    "encoder.down12.",
                    "encoder.level2.",
                ),
                float(learning_rates["encoder_level1_level2"]),
            ),
        ),
        weight_decay=float(_nested(config, "optimization", "weight_decay")),
        weight_decay_norm_bias=float(
            _nested(config, "optimization", "weight_decay_norm_bias")
        ),
    )
    optimizer = build_adamw(
        groups,
        betas=tuple(float(value) for value in _nested(config, "optimization", "betas")),
        fused_if_supported=(
            bool(_nested(config, "runtime", "fused_adamw_if_supported"))
            and all(parameter.device.type == "cuda" for parameter in model.parameters())
        ),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(_nested(config, "optimization", "warmup_steps")),
        max_steps=int(_nested(config, "training", "max_steps")),
        min_lr=float(_nested(config, "optimization", "min_lr")),
    )
    # Crucial ordering: the optimizer already owns encoder1/2 parameters.
    set_stage0_trainability(model, 0)
    return optimizer, scheduler


class Stage0StepEngine:
    """One optimizer-step engine with exact accumulation and finite guards."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        ema: ExponentialMovingAverage,
        *,
        device: torch.device,
        accumulation_steps: int,
        micro_batch: int,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        if accumulation_steps <= 0 or micro_batch <= 0:
            raise ValueError("accumulation_steps and micro_batch must be positive")
        if accumulation_steps * micro_batch != 8:
            raise Stage0ContractError("Stage0 accumulation must produce effective batch 8")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.micro_batch = micro_batch
        self.gradient_clip_norm = float(gradient_clip_norm)

    def train_optimizer_step(
        self,
        batches: Sequence[dict[str, object]],
        *,
        step: int,
    ) -> Stage0StepResult:
        if len(batches) != self.accumulation_steps:
            raise ValueError(
                f"expected {self.accumulation_steps} micro-batches, got {len(batches)}"
            )
        set_stage0_trainability(self.model, step)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lambda_ssim = stage0_lambda_ssim(step)
        loss_total = 0.0
        pixel_total = 0.0
        ssim_total = 0.0
        image_count = 0
        for batch in batches:
            input_image, target = move_training_batch(batch, self.device)
            if input_image.shape[0] != self.micro_batch:
                raise Stage0ContractError(
                    f"micro-batch drift: expected {self.micro_batch}, got {input_image.shape[0]}"
                )
            with autocast_context(self.device):
                prediction = self.model(input_image)
                # The first 20% is genuinely Charbonnier-only: do not spend
                # compute on a nominally zero-weight SSIM graph.
                if lambda_ssim == 0.0:
                    pixel = charbonnier(prediction, target)
                    ssim_term = prediction.new_zeros(())
                    loss = pixel
                else:
                    breakdown = restoration_loss(
                        prediction,
                        target,
                        lambda_ssim=lambda_ssim,
                        step_weight=0.0,
                    )
                    pixel = breakdown.final
                    ssim_term = breakdown.ssim
                    loss = breakdown.total
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage0 forward/loss at step {step}")
            (loss / self.accumulation_steps).backward()
            batch_images = int(input_image.shape[0])
            image_count += batch_images
            loss_total += float(loss.detach()) * batch_images
            pixel_total += float(pixel.detach()) * batch_images
            ssim_total += float(ssim_term.detach()) * batch_images

        trainable = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not trainable:
            raise RuntimeError("Stage0 backward produced no trainable gradients")
        if any(not torch.isfinite(parameter.grad).all() for parameter in trainable):
            raise FloatingPointError(f"non-finite Stage0 gradient at step {step}")
        norm = torch.nn.utils.clip_grad_norm_(trainable, self.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"non-finite Stage0 gradient norm at step {step}")
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.ema.update(self.model)
        return Stage0StepResult(
            step=step + 1,
            loss=loss_total / image_count,
            charbonnier=pixel_total / image_count,
            ssim_loss=ssim_total / image_count,
            lambda_ssim=lambda_ssim,
            grad_norm=float(norm.detach()),
            micro_batches=len(batches),
            images=image_count,
        )


def _normalise_task(operator_order: object) -> str:
    if isinstance(operator_order, str):
        parts = [part.strip() for part in operator_order.split("+")]
    elif isinstance(operator_order, Sequence):
        parts = [str(part).strip() for part in operator_order]
    else:
        raise ValueError(f"invalid operator_order: {operator_order!r}")
    return "+".join(parts)


def aggregate_stage0_metric_records(
    records: Iterable[Mapping[str, object]],
    *,
    expected_per_task: int | None = 100,
) -> Stage0ValidationResult:
    """Per image -> per task -> equal-weight single/A aggregation."""

    expected_single = tuple(order[0] for order in ALLOWED_SINGLE)
    expected_a = tuple("+".join(order) for order in ALLOWED_GROUP_A)
    expected = set((*expected_single, *expected_a))
    buckets: dict[str, dict[str, list[float] | str]] = defaultdict(
        lambda: {"group": "", "psnr": [], "ssim": []}
    )
    count = 0
    for row in records:
        task = _normalise_task(row.get("task"))
        group = str(row.get("group"))
        if task not in expected or group not in {"single", "A"}:
            raise Stage0ContractError(f"forbidden Stage0 validation record: {row!r}")
        required_group = "single" if task in expected_single else "A"
        if group != required_group:
            raise Stage0ContractError(f"task/group mismatch for {task}: {group}")
        psnr = float(row["psnr"])
        ssim = float(row["ssim"])
        if not math.isfinite(psnr) or not math.isfinite(ssim):
            raise FloatingPointError(f"non-finite validation metric for {task}")
        bucket = buckets[task]
        bucket["group"] = group
        assert isinstance(bucket["psnr"], list) and isinstance(bucket["ssim"], list)
        bucket["psnr"].append(psnr)
        bucket["ssim"].append(ssim)
        count += 1
    if set(buckets) != expected:
        raise Stage0ContractError(
            f"primary_val task mismatch: missing={sorted(expected-set(buckets))}, "
            f"unexpected={sorted(set(buckets)-expected)}"
        )
    task_means: OrderedDict[str, Mapping[str, float | int | str]] = OrderedDict()
    for task in (*expected_single, *expected_a):
        bucket = buckets[task]
        psnr_values = bucket["psnr"]
        ssim_values = bucket["ssim"]
        assert isinstance(psnr_values, list) and isinstance(ssim_values, list)
        if expected_per_task is not None and len(psnr_values) != expected_per_task:
            raise Stage0ContractError(
                f"{task}: expected {expected_per_task} primary_val images, got {len(psnr_values)}"
            )
        task_means[task] = {
            "group": str(bucket["group"]),
            "count": len(psnr_values),
            "psnr": math.fsum(psnr_values) / len(psnr_values),
            "ssim": math.fsum(ssim_values) / len(ssim_values),
        }

    def task_average(tasks: Sequence[str], metric: str) -> float:
        return math.fsum(float(task_means[task][metric]) for task in tasks) / len(tasks)

    return Stage0ValidationResult(
        single_psnr=task_average(expected_single, "psnr"),
        single_ssim=task_average(expected_single, "ssim"),
        group_a_psnr=task_average(expected_a, "psnr"),
        group_a_ssim=task_average(expected_a, "ssim"),
        image_count=count,
        task_means=task_means,
    )


def evaluate_primary_val(
    model: nn.Module,
    dataset: Any,
    *,
    device: torch.device,
    dataloader: Iterable[Mapping[str, object]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Stage0ValidationResult:
    """Run full-resolution deterministic primary_val only, with official quantization."""

    if getattr(dataset, "training", None) is not False:
        raise Stage0ContractError("Stage0 validation dataset must use training=False")
    if getattr(dataset, "crop_size", object()) is not None:
        raise Stage0ContractError("Stage0 primary_val must be evaluated full-resolution")
    records: list[dict[str, object]] = []
    was_training = model.training
    model.eval()

    def scalar_metadata(value: object, *, name: str) -> object:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise Stage0ContractError(
                    f"primary_val DataLoader must keep batch_size=1 ({name})"
                )
            return value[0]
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise Stage0ContractError(
                    f"primary_val DataLoader must keep batch_size=1 ({name})"
                )
            return value.item()
        return value

    source: Iterable[Mapping[str, object]]
    if dataloader is None:
        source = (dataset[index] for index in range(len(dataset)))
    else:
        source = dataloader
    try:
        with torch.inference_mode():
            for index, sample in enumerate(source):
                group = str(scalar_metadata(sample["group"], name="group"))
                if group not in {"single", "A"}:
                    raise Stage0ContractError(f"primary_val exposed forbidden group {group!r}")
                input_value = sample["input"]
                target_value = sample["target"]
                if not isinstance(input_value, Tensor) or not isinstance(target_value, Tensor):
                    raise TypeError("primary_val input/target must be tensors")
                if input_value.ndim == 3:
                    input_value = input_value.unsqueeze(0)
                    target_value = target_value.unsqueeze(0)
                if input_value.ndim != 4 or input_value.shape[0] != 1:
                    raise Stage0ContractError(
                        "primary_val requires batch_size=1 to preserve variable full resolution"
                    )
                input_image = input_value.to(
                    device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
                )
                target = target_value.to(
                    device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
                )
                with autocast_context(device):
                    prediction = model(input_image)
                # Official metric code expects float arithmetic after PNG-equivalent
                # quantization, independent of the BF16 inference context.
                metric = official_psnr_ssim(
                    prediction.float(), target.float(), quantize=True
                )
                records.append(
                    {
                        "sample_id": str(
                            scalar_metadata(sample["sample_id"], name="sample_id")
                        ),
                        "task": _normalise_task(
                            scalar_metadata(
                                sample["operator_order"], name="operator_order"
                            )
                        ),
                        "group": group,
                        "psnr": float(metric.psnr.item()),
                        "ssim": float(metric.ssim.item()),
                    }
                )
                if progress is not None:
                    progress(index + 1, len(dataset))
                del input_image, target, prediction, metric
    finally:
        model.train(was_training)
    if len(records) != len(dataset):
        raise Stage0ContractError(
            f"primary_val loader yielded {len(records)} images, expected {len(dataset)}"
        )
    return aggregate_stage0_metric_records(records, expected_per_task=100)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_stage0_provenance(
    *,
    project_root: str | Path,
    config_path: str | Path,
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    runtime: Stage0Runtime,
    load_report: BackboneLoadReport | None,
) -> dict[str, Any]:
    """Bind checkpoint state to exact config/data/parent/code/runtime identities."""

    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    identity = resolved.get("expected_identity")
    if not isinstance(identity, Mapping):
        raise Stage0ContractError("resolved paths lack expected_identity")
    expected_manifests = identity.get("manifests")
    if not isinstance(expected_manifests, Mapping):
        raise Stage0ContractError("resolved paths lack expected manifest hashes")
    manifest_bindings: dict[str, dict[str, str]] = {}
    for name, path_key in (
        ("clean_train", "clean_train_manifest"),
        ("clean_val", "clean_val_manifest"),
        ("primary_train", "primary_train_manifest"),
        ("primary_val", "primary_val_manifest"),
    ):
        path = Path(str(resolved[path_key])).resolve()
        actual = sha256_file(path)
        expected = str(expected_manifests[name])
        if actual != expected:
            raise Stage0ContractError(
                f"{name} manifest hash mismatch: expected {expected}, got {actual}"
            )
        manifest_bindings[name] = {"path": str(path), "sha256": actual}
    parent = Path(str(resolved["stage_a_parent_checkpoint"])).resolve()
    parent_hash = sha256_file(parent)
    expected_parent = str(identity["stage_a_parent_sha256"])
    if parent_hash != expected_parent:
        raise Stage0ContractError(
            f"parent checkpoint hash mismatch: expected {expected_parent}, got {parent_hash}"
        )
    agenticir_commit = git_commit(str(resolved["agenticir_repo"]))
    mioir_commit = git_commit(str(resolved["mioir_repo"]))
    if agenticir_commit != identity["agenticir_commit"]:
        raise Stage0ContractError("AgenticIR commit drifted")
    if mioir_commit != identity["mioir_commit"]:
        raise Stage0ContractError("MiOIR commit drifted")
    if not torch.cuda.is_available():
        raise Stage0ContractError("Stage0 provenance requires the bound CUDA device")
    compile_ab = load_stage0_compile_ab_decision(
        root,
        config_path=config_path,
        parent_checkpoint=parent,
        primary_train_manifest=resolved["primary_train_manifest"],
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    if runtime.torch_compile is not bool(compile_ab["recommend_torch_compile"]):
        raise Stage0ContractError(
            "Stage0 runtime compile mode differs from the preregistered A/B decision"
        )
    metric_summary = load_json(root / "artifacts/metrics/metric_parity_summary.json")
    reference_versions: object = "missing"
    if isinstance(metric_summary, Mapping):
        facts = metric_summary.get("facts")
        if isinstance(facts, Mapping):
            versions = facts.get("versions")
            if isinstance(versions, Mapping):
                reference_versions = versions.get("reference_environment", "missing")
    report_value: Mapping[str, Any]
    if load_report is None:
        report_value = {"source_tensor_count": 495, "loaded_count": 495}
    else:
        report_value = {
            "source_tensor_count": load_report.source_tensor_count,
            "loaded_count": load_report.loaded_count,
            "missing_keys": list(load_report.missing_keys),
            "unexpected_keys": list(load_report.unexpected_keys),
            "shape_mismatches": list(load_report.shape_mismatches),
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": STAGE0_CHECKPOINT_STAGE,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_paths_sha256": sha256_file(
            resolve_config_path(config_path, str(_nested(config, "paths", "resolved_paths")))
        ),
        "semantic_source_sha256": semantic_source_hashes(
            root,
            entrypoints=("scripts/train_stage0.py",),
        ),
        "manifests": manifest_bindings,
        "parent_checkpoint": {"path": str(parent), "sha256": parent_hash},
        "repositories": {
            "agenticir_commit": agenticir_commit,
            "mioir_commit": mioir_commit,
        },
        "runtime": asdict(runtime),
        "compile_ab": {
            "path": str(root / "artifacts/audits/stage0_compile_ab.json"),
            "sha256": sha256_file(root / "artifacts/audits/stage0_compile_ab.json"),
            "recommend_torch_compile": bool(compile_ab["recommend_torch_compile"]),
            "decision": str(compile_ab["decision"]),
            "profile_script_sha256": str(compile_ab["profile_script_sha256"]),
            "code_sha256": dict(compile_ab["code_sha256"]),
        },
        "warm_start_load": dict(report_value),
        "dependency_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "pyiqa_runtime_distribution": _package_version("pyiqa"),
            "basicsr_runtime_distribution": _package_version("basicsr"),
            "reference_environment": reference_versions,
        },
    }


def score_from_checkpoint_metrics(metrics: Mapping[str, object]) -> ValidationScore | None:
    required = (
        "best_group_a_psnr",
        "best_group_a_ssim",
        "best_single_psnr",
        "best_single_ssim",
        "best_step",
    )
    if not all(key in metrics for key in required):
        return None
    return ValidationScore(
        group_a_psnr=float(metrics["best_group_a_psnr"]),
        group_a_ssim=float(metrics["best_group_a_ssim"]),
        single_psnr=float(metrics["best_single_psnr"]),
        single_ssim=float(metrics["best_single_ssim"]),
        step=int(metrics["best_step"]),
    )


def checkpoint_metrics(
    *,
    last_step_result: Stage0StepResult | None,
    last_validation: Stage0ValidationResult | None,
    best_score: ValidationScore | None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if last_step_result is not None:
        values.update(
            {
                "train_loss": last_step_result.loss,
                "train_charbonnier": last_step_result.charbonnier,
                "train_ssim_loss": last_step_result.ssim_loss,
            }
        )
    if last_validation is not None:
        values.update(
            {
                "single_psnr": last_validation.single_psnr,
                "single_ssim": last_validation.single_ssim,
                "group_a_psnr": last_validation.group_a_psnr,
                "group_a_ssim": last_validation.group_a_ssim,
            }
        )
    if best_score is not None:
        values.update(
            {
                "best_group_a_psnr": best_score.group_a_psnr,
                "best_group_a_ssim": best_score.group_a_ssim,
                "best_single_psnr": best_score.single_psnr,
                "best_single_ssim": best_score.single_ssim,
                "best_step": float(best_score.step),
            }
        )
    return values


def save_stage0_checkpoint(
    destination: str | Path,
    *,
    step: int,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler_state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    metrics: Mapping[str, float],
    model_as_ema: bool = False,
    pending_validation_step: int | None = None,
) -> None:
    """Atomically save all resumable state; best checkpoints expose EMA as model."""

    if pending_validation_step is not None and pending_validation_step != step:
        raise Stage0ContractError(
            "pending validation step must equal the checkpoint optimizer step"
        )
    if sampler_state.get("consumed_optimizer_step") != step:
        raise Stage0ContractError(
            "checkpoint step/sampler consumed optimizer step mismatch"
        )

    context = ema.apply_to(model) if model_as_ema else _null_model_context()
    with context:
        payload = checkpoint_payload(
            stage=STAGE0_CHECKPOINT_STAGE,
            step=step,
            model=model,
            ema_state=ema.state_dict(),
            optimizer=optimizer,
            scheduler=scheduler,
            # BF16 does not use loss scaling.  The schema field remains present.
            scaler=None,
            sampler_state=sampler_state,
            provenance=provenance,
            metrics=metrics,
        )
        payload["amp"] = {"dtype": "bfloat16", "scaler_required": False}
        payload["model_role"] = "ema_selection" if model_as_ema else "raw_training_state"
        payload["resumable"] = not model_as_ema
        payload["pending_validation_step"] = pending_validation_step
        atomic_torch_save(payload, destination)


class _null_model_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


def resume_stage0_checkpoint(
    checkpoint: str | Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_provenance=expected_provenance,
        require_resumable=True,
        expected_model_role="raw_training_state",
        restore_rng=True,
        map_location="cpu",
    )
    if payload.get("stage") != STAGE0_CHECKPOINT_STAGE:
        raise Stage0ContractError("resume checkpoint is not Stage0")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Stage0ContractError("resume checkpoint has invalid optimizer step")
    sampler_state = payload.get("sampler_state")
    if not isinstance(sampler_state, Mapping):
        raise Stage0ContractError("resume checkpoint has no sampler state")
    if sampler_state.get("consumed_optimizer_step") != step:
        raise Stage0ContractError(
            "resume checkpoint step/sampler consumed optimizer step mismatch"
        )
    ema_state = payload.get("ema")
    if not isinstance(ema_state, Mapping):
        raise Stage0ContractError("resume checkpoint has no EMA state")
    ema.load_state_dict(ema_state)
    return payload


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


def append_stage0_calibration_history(
    path: str | Path, *, step: int, result: Stage0ValidationResult
) -> None:
    """Atomically append the Stage0 restoration fields to the shared history."""

    destination = Path(path)
    rows: list[dict[str, str]] = []
    if destination.is_file():
        with destination.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CALIBRATION_COLUMNS:
                raise Stage0ContractError("calibration history schema drifted")
            rows.extend(dict(row) for row in reader)
    row = {column: "" for column in CALIBRATION_COLUMNS}
    row.update(
        {
            "step": str(step),
            "single_psnr": f"{result.single_psnr:.12g}",
            "single_ssim": f"{result.single_ssim:.12g}",
            "group_a_psnr": f"{result.group_a_psnr:.12g}",
            "group_a_ssim": f"{result.group_a_ssim:.12g}",
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


def write_stage0_report(
    path: str | Path,
    *,
    step: int,
    runtime: Stage0Runtime,
    checkpoint: str | Path,
    validation: Stage0ValidationResult | None,
    peak_reserved_fraction: float,
    images_per_second: float,
) -> None:
    validation_lines = (
        "- validation: skipped (100-step integration contract)\n"
        if validation is None
        else (
            f"- primary_val single: PSNR {validation.single_psnr:.6f}, "
            f"SSIM {validation.single_ssim:.8f}\n"
            f"- primary_val Group A: PSNR {validation.group_a_psnr:.6f}, "
            f"SSIM {validation.group_a_ssim:.8f}\n"
        )
    )
    atomic_write_text(
        path,
        "# Stage0 MiO-StageA\n\n"
        f"- protocol: `{PROTOCOL_ID}`\n"
        f"- updated_utc: `{utc_now_iso()}`\n"
        f"- optimizer step: `{step}` / `{runtime.target_step}`\n"
        f"- crop / micro / accumulation / effective: "
        f"`{runtime.crop_size}` / `{runtime.micro_batch}` / "
        f"`{runtime.accumulation_steps}` / `{runtime.effective_batch}`\n"
        f"- BF16 / TF32 / EMA: `true / true / true`\n"
        f"- peak reserved fraction: `{peak_reserved_fraction:.4f}`\n"
        f"- measured throughput: `{images_per_second:.4f} images/s`\n"
        f"- checkpoint: `{Path(checkpoint)}`\n"
        + validation_lines,
    )
