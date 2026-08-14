#!/usr/bin/env python3
"""Run the preregistered 20-step eager/torch.compile Stage0 A/B gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(PROJECT_ROOT / "artifacts/cache/torchinductor"),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data.samplers import StatefulEpisodeSampler  # noqa: E402
from src.training.checkpointing import unwrap_model  # noqa: E402
from src.training.ema import ExponentialMovingAverage  # noqa: E402
from src.training.runtime import (  # noqa: E402
    autocast_context,
    configure_torch_runtime,
    seed_everything,
)
from src.training.stage0_engine import (  # noqa: E402
    Stage0RestorationDataset,
    Stage0StepEngine,
    assert_stage0_preflight,
    build_stage0_model,
    build_stage0_optimizer,
    load_and_validate_stage0_config,
)
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    load_json,
    utc_now_iso,
)


STEPS = 20
MICRO_BATCH = 8
TOLERANCES = {
    "output_max_abs": 2.0e-3,
    "output_mean_abs": 1.0e-5,
    "loss_max_abs": 1.0e-5,
    "loss_mean_abs": 2.0e-6,
    "parameter_max_abs": 5.0e-5,
    "parameter_mean_abs": 1.0e-7,
    "minimum_throughput_ratio": 1.05,
}
PROFILE_CODE_PATHS = (
    "src/data/agenticir_degradations.py",
    "src/data/episode_dataset.py",
    "src/data/manifests.py",
    "src/data/samplers.py",
    "src/data/scale_canonicalizer.py",
    "src/data/subset_targets.py",
    "src/losses/restoration.py",
    "src/metrics/agenticir_official.py",
    "src/net/mio_stagea.py",
    "src/net/restormer_blocks.py",
    "src/training/checkpointing.py",
    "src/training/ema.py",
    "src/training/optimization.py",
    "src/training/runtime.py",
    "src/training/stage0_engine.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/stage0_mio_stagea.yaml",
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--micro_batch", type=int, default=MICRO_BATCH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/audits/stage0_compile_ab.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/STAGE0_COMPILE_AB.md",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even when a fully bound result already exists",
    )
    return parser.parse_args()


def _hardware_identity(device: torch.device) -> dict[str, object]:
    return {
        "gpu": torch.cuda.get_device_name(device),
        "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _code_bindings() -> dict[str, str]:
    bindings: dict[str, str] = {}
    for relative in PROFILE_CODE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing Stage0 A/B code binding: {path}")
        bindings[relative] = sha256_file(path)
    return bindings


def _reuse_bound_result(
    path: Path,
    *,
    config_path: Path,
    parent_path: Path,
    primary_manifest: Path,
    device: torch.device,
) -> dict[str, Any] | None:
    """Reuse only the exact preregistered result for this code and environment."""

    if not path.is_file():
        return None
    try:
        payload = load_json(path)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    expected = {
        "schema_version": "graphrestore-stage0-compile-ab-v1",
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "completed": True,
        "tolerances": TOLERANCES,
        "hardware": _hardware_identity(device),
        "profile_script_sha256": sha256_file(Path(__file__).resolve()),
        "code_sha256": _code_bindings(),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    config_binding = payload.get("config")
    parent_binding = payload.get("parent_checkpoint")
    ab_design = payload.get("ab_design")
    if not isinstance(config_binding, Mapping) or not isinstance(parent_binding, Mapping):
        return None
    if not isinstance(ab_design, Mapping):
        return None
    if config_binding.get("path") != str(config_path):
        return None
    if config_binding.get("sha256") != sha256_file(config_path):
        return None
    if parent_binding.get("path") != str(parent_path):
        return None
    if parent_binding.get("sha256") != sha256_file(parent_path):
        return None
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return None
    manifest_binding = data.get("primary_train_manifest")
    if not isinstance(manifest_binding, Mapping):
        return None
    if manifest_binding != {
        "path": str(primary_manifest),
        "sha256": sha256_file(primary_manifest),
    }:
        return None
    if ab_design != {
        "steps": STEPS,
        "micro_batch": MICRO_BATCH,
        "effective_batch": MICRO_BATCH,
        "crop_size": 192,
        "steady_state_excludes_step": 0,
    }:
        return None
    return dict(payload)


def _fixed_batch(
    resolved: Mapping[str, Any], *, seed: int
) -> tuple[dict[str, torch.Tensor], list[str]]:
    dataset = Stage0RestorationDataset(
        manifest_path=Path(str(resolved["primary_train_manifest"])),
        training_data_root=Path(str(resolved["training_data_root"])),
        depth_compat_root=PROJECT_ROOT / "artifacts/cache/mioir_depth_compat",
        crop_size=192,
        training=True,
        stage="stage0",
        base_seed=seed,
        agenticir_repo=Path(str(resolved["agenticir_repo"])),
        mioir_repo=Path(str(resolved["mioir_repo"])),
    )
    sampler = StatefulEpisodeSampler(
        dataset,
        num_samples=MICRO_BATCH,
        stage="stage0",
        effective_batch_size=MICRO_BATCH,
        base_seed=seed,
        start_step=0,
    )
    samples = [dataset[request] for request in sampler]
    return (
        {
            "input": torch.stack([sample["input"] for sample in samples]),
            "target": torch.stack([sample["target"] for sample in samples]),
        },
        [str(sample["sample_id"]) for sample in samples],
    )


def _prediction(model: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.train()
    with torch.inference_mode(), autocast_context(device):
        value = model(batch.to(device=device, dtype=torch.float32))
    torch.cuda.synchronize(device)
    return value.detach().float().cpu()


def _run_mode(
    mode: str,
    *,
    parent_payload: Mapping[str, Any],
    config: Mapping[str, Any],
    fixed_batch: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    seed_everything(seed)
    host, report = build_stage0_model(parent_payload)
    host.to(device)
    optimizer, scheduler = build_stage0_optimizer(host, config)
    ema = ExponentialMovingAverage(host, decay=float(config["ema"]["decay"]))
    model: torch.nn.Module = host
    compile_options: dict[str, object] | None = None
    if mode == "compiled":
        compile_options = {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": False,
        }
        model = torch.compile(host, **compile_options)
    elif mode != "eager":
        raise ValueError(mode)
    engine = Stage0StepEngine(
        model,
        optimizer,
        scheduler,
        ema,
        device=device,
        accumulation_steps=1,
        micro_batch=MICRO_BATCH,
        gradient_clip_norm=float(config["optimization"]["gradient_clip_norm"]),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial = _prediction(model, fixed_batch["input"], device)
    losses: list[float] = []
    durations: list[float] = []
    for step in range(STEPS):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = engine.train_optimizer_step([fixed_batch], step=step)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        losses.append(result.loss)
    final = _prediction(model, fixed_batch["input"], device)
    state = {
        name: value.detach().float().cpu().clone()
        for name, value in unwrap_model(model).state_dict().items()
        if value.is_floating_point()
    }
    steady_seconds = math.fsum(durations[1:])
    steady_images = (STEPS - 1) * MICRO_BATCH
    summary = {
        "mode": mode,
        "steps": STEPS,
        "micro_batch": MICRO_BATCH,
        "effective_batch": MICRO_BATCH,
        "parent_loaded_tensors": report.loaded_count,
        "losses": losses,
        "durations_seconds": durations,
        "steady_state_excludes_step": 0,
        "steady_state_images_per_second": steady_images / steady_seconds,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "compile_options": compile_options,
        "finite": bool(
            all(math.isfinite(value) for value in losses)
            and torch.isfinite(initial).all()
            and torch.isfinite(final).all()
        ),
    }
    del engine, ema, scheduler, optimizer, model, host
    torch.cuda.empty_cache()
    return summary, state, initial, final


def _tensor_difference(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    difference = (first.to(torch.float64) - second.to(torch.float64)).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
    }


def _state_difference(
    eager: Mapping[str, torch.Tensor], compiled: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    if eager.keys() != compiled.keys():
        raise RuntimeError("eager/compiled state keys differ")
    maximum = 0.0
    absolute_sum = 0.0
    elements = 0
    worst_key = ""
    for key in eager:
        difference = (eager[key].to(torch.float64) - compiled[key].to(torch.float64)).abs()
        local_max = float(difference.max())
        if local_max > maximum:
            maximum = local_max
            worst_key = key
        absolute_sum += float(difference.sum())
        elements += difference.numel()
    return {
        "max_abs": maximum,
        "mean_abs": absolute_sum / elements,
        "worst_key": worst_key,
        "elements": elements,
    }


def main() -> int:
    args = parse_args()
    if args.steps != STEPS or args.micro_batch != MICRO_BATCH:
        raise ValueError("D-011 is frozen to 20 steps and micro/effective batch 8")
    if not torch.cuda.is_available():
        raise RuntimeError("compile A/B requires the configured CUDA GPU")
    config_path = args.config.resolve()
    config, resolved = load_and_validate_stage0_config(config_path)
    assert_stage0_preflight(PROJECT_ROOT)
    seed = int(config["seed"])
    configure_torch_runtime(tf32=True, cudnn_benchmark=True)
    parent_path = Path(str(resolved["stage_a_parent_checkpoint"])).resolve()
    primary_manifest = Path(str(resolved["primary_train_manifest"])).resolve()
    device = torch.device("cuda", torch.cuda.current_device())
    if not args.force:
        existing = _reuse_bound_result(
            args.output.resolve(),
            config_path=config_path,
            parent_path=parent_path,
            primary_manifest=primary_manifest,
            device=device,
        )
        if existing is not None:
            print(json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    fixed_batch, sample_ids = _fixed_batch(resolved, seed=seed)
    parent_payload = torch.load(
        parent_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(parent_payload, Mapping):
        raise RuntimeError("parent checkpoint is not a mapping")

    eager_summary, eager_state, eager_initial, eager_final = _run_mode(
        "eager",
        parent_payload=parent_payload,
        config=config,
        fixed_batch=fixed_batch,
        device=device,
        seed=seed,
    )
    compiled_error: str | None = None
    try:
        (
            compiled_summary,
            compiled_state,
            compiled_initial,
            compiled_final,
        ) = _run_mode(
            "compiled",
            parent_payload=parent_payload,
            config=config,
            fixed_batch=fixed_batch,
            device=device,
            seed=seed,
        )
    except Exception as error:  # safe fallback is the purpose of this gate.
        compiled_error = f"{type(error).__name__}: {error}"[:4000]
        compiled_summary = {
            "mode": "compiled",
            "steps": 0,
            "finite": False,
            "error": compiled_error,
        }
        compiled_state = {}
        compiled_initial = compiled_final = torch.empty(0)
        torch.cuda.empty_cache()

    numerical: dict[str, Any] | None = None
    throughput_ratio = 0.0
    numerical_pass = False
    if compiled_error is None:
        initial_difference = _tensor_difference(eager_initial, compiled_initial)
        final_difference = _tensor_difference(eager_final, compiled_final)
        losses_eager = torch.tensor(eager_summary["losses"], dtype=torch.float64)
        losses_compiled = torch.tensor(
            compiled_summary["losses"], dtype=torch.float64
        )
        loss_difference = _tensor_difference(losses_eager, losses_compiled)
        parameter_difference = _state_difference(eager_state, compiled_state)
        numerical = {
            "initial_output": initial_difference,
            "post_step20_output": final_difference,
            "per_step_loss": loss_difference,
            "final_parameters": parameter_difference,
        }
        numerical_pass = bool(
            eager_summary["finite"]
            and compiled_summary["finite"]
            and initial_difference["max_abs"] <= TOLERANCES["output_max_abs"]
            and initial_difference["mean_abs"] <= TOLERANCES["output_mean_abs"]
            and final_difference["max_abs"] <= TOLERANCES["output_max_abs"]
            and final_difference["mean_abs"] <= TOLERANCES["output_mean_abs"]
            and loss_difference["max_abs"] <= TOLERANCES["loss_max_abs"]
            and loss_difference["mean_abs"] <= TOLERANCES["loss_mean_abs"]
            and parameter_difference["max_abs"]
            <= TOLERANCES["parameter_max_abs"]
            and parameter_difference["mean_abs"]
            <= TOLERANCES["parameter_mean_abs"]
        )
        throughput_ratio = float(
            compiled_summary["steady_state_images_per_second"]
        ) / float(eager_summary["steady_state_images_per_second"])
    recommend = bool(
        compiled_error is None
        and numerical_pass
        and throughput_ratio >= TOLERANCES["minimum_throughput_ratio"]
    )
    reason = (
        "enable: numerical gates passed and steady-state gain >=5%"
        if recommend
        else (
            "disable: compilation failed; eager remains the safe default"
            if compiled_error is not None
            else (
                "disable: preregistered numerical consistency gate failed"
                if not numerical_pass
                else "disable: steady-state throughput gain was below 5%"
            )
        )
    )
    payload = {
        "schema_version": "graphrestore-stage0-compile-ab-v1",
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "created_utc": utc_now_iso(),
        "completed": True,
        "profile_script_sha256": sha256_file(Path(__file__).resolve()),
        "code_sha256": _code_bindings(),
        "safe_default": "eager",
        "recommend_torch_compile": recommend,
        "decision": reason,
        "tolerances": TOLERANCES,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "parent_checkpoint": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
        },
        "ab_design": {
            "steps": STEPS,
            "micro_batch": MICRO_BATCH,
            "effective_batch": MICRO_BATCH,
            "crop_size": 192,
            "steady_state_excludes_step": 0,
        },
        "data": {
            "sample_ids": sample_ids,
            "crop_size": 192,
            "primary_train_manifest": {
                "path": str(primary_manifest),
                "sha256": sha256_file(primary_manifest),
            },
        },
        "hardware": _hardware_identity(device),
        "eager": eager_summary,
        "compiled": compiled_summary,
        "compiled_error": compiled_error,
        "numerical": numerical,
        "numerical_pass": numerical_pass,
        "steady_state_throughput_ratio": throughput_ratio,
    }
    atomic_write_json(args.output, payload)
    atomic_write_text(
        args.report,
        "# Stage0 torch.compile 20-step A/B\n\n"
        f"- completed: `true`\n"
        f"- recommendation: `{'compile' if recommend else 'eager'}`\n"
        f"- decision: {reason}\n"
        f"- steady-state ratio: `{throughput_ratio:.6f}`\n"
        f"- numerical pass: `{str(numerical_pass).lower()}`\n"
        f"- artifact: `{args.output}`\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
