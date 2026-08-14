#!/usr/bin/env python3
"""Gate full-resolution validation memory before formal GraphRestore Stage0.

The probe deliberately uses the largest clean-validation image and runs the
pure 25M Stage0 host and the expanded two-skill guarded host one at a time.
It performs inference only; it never opens MiO100 or starts a training stage.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifests import CleanRecord, load_clean_manifest  # noqa: E402
from src.data.scale_canonicalizer import bgr_uint8_to_rgb_float  # noqa: E402
from src.net import (  # noqa: E402
    GuardedSkillRestormer,
    MiOStageA,
    SKILL_TO_INDEX,
    load_parent_backbone,
)
from src.metrics.agenticir_official import official_psnr_ssim  # noqa: E402
from src.training.runtime import (  # noqa: E402
    autocast_context,
    configure_torch_runtime,
    seed_everything,
)
from src.training.stage0_engine import (  # noqa: E402
    PROTOCOL_ID,
    build_stage0_model,
    load_and_validate_stage0_config,
)
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.io import atomic_write_json, load_json, utc_now_iso  # noqa: E402
from src.utils.paths import resolve_config_path  # noqa: E402


SCHEMA_VERSION = "graphrestore-validation-vram-probe-v1"
STAGE0_PARAMETER_COUNT = 25_437_220
EXPANDED_PARAMETER_COUNT = 26_465_380
ACTIVE_SKILLS = ("rain", "haze")
CODE_PATHS = (
    "scripts/probe_validation_vram.py",
    "src/data/manifests.py",
    "src/data/scale_canonicalizer.py",
    "src/net/cooperative_executor.py",
    "src/net/graphrestore.py",
    "src/net/latent_skill_bank.py",
    "src/net/mio_stagea.py",
    "src/net/restormer_blocks.py",
    "src/net/skill_adapter.py",
    "src/metrics/agenticir_official.py",
    "src/training/runtime.py",
    "src/training/stage0_engine.py",
    "src/utils/hashing.py",
    "src/utils/io.py",
    "src/utils/paths.py",
)


class ValidationVramProbeError(RuntimeError):
    """The maximum-size validation workload is unsafe or structurally invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe maximum-size clean_val BF16 inference memory for the pure "
            "Stage0 and expanded two-skill models."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/stage0_mio_stagea.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/audits/validation_vram_probe.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="run CUDA again even if an exact bound PASS artifact already exists",
    )
    return parser


def select_largest_clean_record(
    records: Sequence[CleanRecord],
) -> tuple[CleanRecord, int]:
    """Return the first manifest record at the true maximum metadata area."""

    if not records:
        raise ValidationVramProbeError("clean_val manifest is empty")
    maximum_area = max(record.width * record.height for record in records)
    tied = [record for record in records if record.width * record.height == maximum_area]
    return tied[0], len(tied)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationVramProbeError(f"{context} must be a mapping")
    return value


def _binding(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _code_bindings() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in CODE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise ValidationVramProbeError(f"missing validation probe source: {path}")
        result[relative] = sha256_file(path)
    return result


def _hardware(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_index": int(device.index or 0),
        "gpu": torch.cuda.get_device_name(device),
        "total_memory_bytes": int(properties.total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _load_maximum_input(
    manifest_path: Path,
    training_data_root: Path,
) -> tuple[torch.Tensor, dict[str, object]]:
    records = load_clean_manifest(
        manifest_path,
        training_data_root,
        expected_split="val",
        must_exist=True,
    )
    selected, tie_count = select_largest_clean_record(records)
    actual_sha = sha256_file(selected.clean_path)
    if actual_sha != selected.clean_sha256:
        raise ValidationVramProbeError(
            f"largest clean_val image SHA mismatch: {selected.clean_path}"
        )
    image_bgr = cv2.imread(str(selected.clean_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValidationVramProbeError(
            f"largest clean_val image is unreadable: {selected.clean_path}"
        )
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValidationVramProbeError("largest clean_val image is not BGR uint8 HWC")
    actual_height, actual_width = image_bgr.shape[:2]
    if (actual_width, actual_height) != (selected.width, selected.height):
        raise ValidationVramProbeError(
            "largest clean_val manifest/image dimensions disagree: "
            f"manifest={(selected.width, selected.height)}, "
            f"actual={(actual_width, actual_height)}"
        )
    maximum_area = max(record.width * record.height for record in records)
    if actual_width * actual_height != maximum_area:
        raise ValidationVramProbeError("selected image is not maximum-area clean_val")
    rgb = bgr_uint8_to_rgb_float(image_bgr).unsqueeze(0).contiguous()
    if tuple(rgb.shape) != (1, 3, actual_height, actual_width):
        raise ValidationVramProbeError("largest clean_val tensor shape is invalid")
    if rgb.dtype != torch.float32 or not bool(torch.isfinite(rgb).all().item()):
        raise ValidationVramProbeError("largest clean_val RGB tensor is not finite FP32")
    return rgb, {
        "clean_id": selected.clean_id,
        "path": str(selected.clean_path),
        "sha256": actual_sha,
        "width": actual_width,
        "height": actual_height,
        "area": actual_width * actual_height,
        "maximum_area_tie_count": tie_count,
        "manifest_record_count": len(records),
        "input_domain": "native_clean_bgr_uint8_to_rgb_float32_0_1",
    }


def _model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _build_model(
    model_kind: str,
    parent_payload: Mapping[str, Any],
) -> tuple[torch.nn.Module, dict[str, object]]:
    if model_kind == "stage0_mio_stagea":
        model, report = build_stage0_model(parent_payload)
        expected_parameters = STAGE0_PARAMETER_COUNT
    elif model_kind == "expanded_guarded_skill_restormer":
        model = GuardedSkillRestormer(gradient_checkpointing=False)
        reference = MiOStageA(gradient_checkpointing=False)
        try:
            report = load_parent_backbone(
                model,
                parent_payload,
                reference_model=reference,
                allowed_missing_prefixes=("decoder.skill_bank.",),
            )
        finally:
            del reference
        if report.loaded_count != 495 or not report.missing_keys:
            raise ValidationVramProbeError(
                "expanded model did not load exactly the 495 parent backbone tensors"
            )
        if any(not key.startswith("decoder.skill_bank.") for key in report.missing_keys):
            raise ValidationVramProbeError(
                "expanded model has missing tensors outside decoder.skill_bank"
            )
        expected_parameters = EXPANDED_PARAMETER_COUNT
    else:
        raise ValueError(f"unknown model kind: {model_kind}")

    parameter_count = _model_parameter_count(model)
    if parameter_count != expected_parameters:
        raise ValidationVramProbeError(
            f"{model_kind} parameter count drifted: "
            f"expected {expected_parameters}, got {parameter_count}"
        )
    return model, {
        "parameter_count": parameter_count,
        "state_tensor_count": len(model.state_dict()),
        "parent_loaded_tensors": report.loaded_count,
        "new_missing_tensors": len(report.missing_keys),
    }


def _probe_one(
    model_kind: str,
    *,
    parent_payload: Mapping[str, Any],
    input_cpu: torch.Tensor,
    device: torch.device,
    maximum_peak_fraction: float,
) -> dict[str, object]:
    started = time.perf_counter()
    model: torch.nn.Module | None = None
    input_cuda: torch.Tensor | None = None
    guards: torch.Tensor | None = None
    active_mask: torch.Tensor | None = None
    output: torch.Tensor | None = None
    metric_finite: bool | None = None
    metadata: dict[str, object] = {}
    try:
        # Each host is constructed, moved, measured, and destroyed separately.
        model, metadata = _build_model(model_kind, parent_payload)
        model.eval().to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        input_cuda = input_cpu.to(device=device, dtype=torch.float32)
        call_kwargs: dict[str, torch.Tensor] = {}
        active_skills: list[str] = []
        if model_kind == "expanded_guarded_skill_restormer":
            height, width = input_cuda.shape[-2:]
            guards = torch.zeros(
                (1, len(SKILL_TO_INDEX), height, width),
                device=device,
                dtype=torch.float32,
            )
            active_mask = torch.zeros(
                (1, len(SKILL_TO_INDEX)),
                device=device,
                dtype=torch.bool,
            )
            for skill in ACTIVE_SKILLS:
                index = SKILL_TO_INDEX[skill]
                guards[:, index].fill_(1.0)
                active_mask[:, index] = True
            if int(active_mask.sum().item()) != 2:
                raise ValidationVramProbeError("expanded probe must activate two skills")
            call_kwargs = {"active_mask": active_mask, "guards": guards}
            active_skills = list(ACTIVE_SKILLS)

        with torch.inference_mode(), autocast_context(device):
            output_value = model(input_cuda, **call_kwargs)
        if not isinstance(output_value, torch.Tensor):
            raise ValidationVramProbeError(f"{model_kind} did not return a tensor")
        output = output_value
        if model_kind == "stage0_mio_stagea":
            metric = official_psnr_ssim(output.float(), input_cuda.float(), quantize=True)
            metric_finite = bool(
                torch.isfinite(metric.psnr).all().item()
                and torch.isfinite(metric.ssim).all().item()
            )
        else:
            # Stage1 validation transfers prediction/target to CPU before the
            # official metric, so only its expanded forward belongs here.
            metric_finite = True
        torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(output).all().item())
        shape_matches = tuple(output.shape) == tuple(input_cuda.shape)
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        peak_fraction = peak_reserved / total_memory
        passed = (
            finite
            and metric_finite
            and shape_matches
            and math.isfinite(peak_fraction)
            and peak_fraction <= maximum_peak_fraction
        )
        return {
            "model": model_kind,
            **metadata,
            "input_shape": list(input_cuda.shape),
            "output_shape": list(output.shape),
            "shape_matches_input": shape_matches,
            "finite": finite,
            "official_metric_included_on_cuda": model_kind == "stage0_mio_stagea",
            "official_metric_finite": metric_finite,
            "inference_mode": True,
            "autocast_device_type": "cuda",
            "autocast_dtype": "bfloat16",
            "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "input_dtype": str(input_cuda.dtype).removeprefix("torch."),
            "output_dtype": str(output.dtype).removeprefix("torch."),
            "active_skills": active_skills,
            "active_skill_count": len(active_skills),
            "guard_shape": list(guards.shape) if guards is not None else None,
            "guard_values": "two_full_one_six_zero" if guards is not None else None,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_fraction": peak_fraction,
            "maximum_peak_reserved_fraction": maximum_peak_fraction,
            "passed": passed,
            "elapsed_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        return {
            "model": model_kind,
            **metadata,
            "input_shape": list(input_cpu.shape),
            "output_shape": list(output.shape) if output is not None else None,
            "shape_matches_input": False,
            "finite": False,
            "official_metric_included_on_cuda": model_kind == "stage0_mio_stagea",
            "official_metric_finite": metric_finite,
            "inference_mode": True,
            "autocast_device_type": "cuda",
            "autocast_dtype": "bfloat16",
            "active_skills": list(ACTIVE_SKILLS)
            if model_kind == "expanded_guarded_skill_restormer"
            else [],
            "active_skill_count": 2
            if model_kind == "expanded_guarded_skill_restormer"
            else 0,
            "guard_shape": list(guards.shape) if guards is not None else None,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_fraction": peak_reserved / total_memory,
            "maximum_peak_reserved_fraction": maximum_peak_fraction,
            "passed": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        del output, guards, active_mask, input_cuda, model
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_gate(
    probes: Sequence[Mapping[str, Any]],
    *,
    maximum_peak_fraction: float,
) -> bool:
    """Fail closed unless both exact workloads passed all required checks."""

    if [probe.get("model") for probe in probes] != [
        "stage0_mio_stagea",
        "expanded_guarded_skill_restormer",
    ]:
        return False
    for probe in probes:
        fraction = float(probe.get("peak_reserved_fraction", math.inf))
        if (
            probe.get("passed") is not True
            or probe.get("finite") is not True
            or probe.get("shape_matches_input") is not True
            or not math.isfinite(fraction)
            or fraction > maximum_peak_fraction
        ):
            return False
    expanded = probes[1]
    return (
        expanded.get("active_skill_count") == 2
        and expanded.get("active_skills") == list(ACTIVE_SKILLS)
        and expanded.get("guard_shape") is not None
    )


def _reuse_exact_pass(
    output_path: Path,
    *,
    bindings: Mapping[str, Any],
    hardware: Mapping[str, Any],
    selected_image: Mapping[str, Any],
    maximum_peak_fraction: float,
) -> Mapping[str, Any] | None:
    if not output_path.is_file():
        return None
    try:
        value = load_json(output_path)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, Mapping):
        return None
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("completed") is not True
        or value.get("passed") is not True
        or value.get("bindings") != bindings
        or value.get("hardware") != hardware
        or value.get("selected_image") != selected_image
        or value.get("maximum_peak_reserved_fraction") != maximum_peak_fraction
    ):
        return None
    probes = value.get("probes")
    if not isinstance(probes, list) or not evaluate_gate(
        probes, maximum_peak_fraction=maximum_peak_fraction
    ):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise ValidationVramProbeError("validation VRAM gate requires CUDA")
    config_path = arguments.config.resolve()
    output_path = arguments.output.resolve()
    config, resolved = load_and_validate_stage0_config(config_path)
    resolved_path = resolve_config_path(config_path, config["paths"]["resolved_paths"])
    clean_val_manifest = Path(str(resolved["clean_val_manifest"])).resolve(strict=True)
    training_data_root = Path(str(resolved["training_data_root"])).resolve(strict=True)
    parent_checkpoint = Path(str(resolved["stage_a_parent_checkpoint"])).resolve(
        strict=True
    )
    expected_identity = _mapping(resolved.get("expected_identity"), "expected_identity")
    expected_manifests = _mapping(
        expected_identity.get("manifests"), "expected_identity.manifests"
    )
    if sha256_file(clean_val_manifest) != expected_manifests.get("clean_val"):
        raise ValidationVramProbeError("locked clean_val manifest SHA mismatch")
    if sha256_file(parent_checkpoint) != expected_identity.get("stage_a_parent_sha256"):
        raise ValidationVramProbeError("locked Stage-A parent SHA mismatch")

    input_cpu, selected_image = _load_maximum_input(
        clean_val_manifest,
        training_data_root,
    )
    maximum_peak_fraction = float(
        config["runtime"]["vram"]["maximum_peak_reserved_fraction"]
    )
    if maximum_peak_fraction != 0.90:
        raise ValidationVramProbeError("validation VRAM ceiling must remain exactly 0.90")
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = _hardware(device)
    bindings = {
        "config": _binding(config_path),
        "resolved_paths": _binding(resolved_path),
        "clean_val_manifest": _binding(clean_val_manifest),
        "parent_checkpoint": _binding(parent_checkpoint),
        "code_sha256": _code_bindings(),
    }

    if not arguments.force:
        reused = _reuse_exact_pass(
            output_path,
            bindings=bindings,
            hardware=hardware,
            selected_image=selected_image,
            maximum_peak_fraction=maximum_peak_fraction,
        )
        if reused is not None:
            print(
                json.dumps(
                    {
                        "output": str(output_path),
                        "passed": True,
                        "reused_exact_bound_result": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

    seed_everything(int(config["seed"]))
    configure_torch_runtime(tf32=True, cudnn_benchmark=True)
    parent_payload = torch.load(
        parent_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    parent_payload = _mapping(parent_payload, "Stage-A parent checkpoint")
    probes = [
        _probe_one(
            model_kind,
            parent_payload=parent_payload,
            input_cpu=input_cpu,
            device=device,
            maximum_peak_fraction=maximum_peak_fraction,
        )
        for model_kind in (
            "stage0_mio_stagea",
            "expanded_guarded_skill_restormer",
        )
    ]
    passed = evaluate_gate(probes, maximum_peak_fraction=maximum_peak_fraction)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now_iso(),
        "completed": True,
        "passed": passed,
        "maximum_peak_reserved_fraction": maximum_peak_fraction,
        "probe_order": "independent_sequential_models",
        "selected_image": selected_image,
        "hardware": hardware,
        "bindings": bindings,
        "probes": probes,
    }
    atomic_write_json(output_path, artifact)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "passed": passed,
                "selected_image": selected_image,
                "probes": probes,
                "reused_exact_bound_result": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise ValidationVramProbeError(
            f"full-resolution validation VRAM gate failed; see {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
