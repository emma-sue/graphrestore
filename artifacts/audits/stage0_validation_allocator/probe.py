#!/usr/bin/env python3
"""Read-only Stage0 allocator probe against the pending step-4000 checkpoint.

This file intentionally lives outside the project tree.  It loads the exact
resumable model/EMA/AdamW/scheduler state and runs the normal full-resolution
Stage0 forward plus official quantized PSNR/FP64 SSIM without writing any
checkpoint, metric artifact, or orchestration state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path("/root/autodl-tmp/aaa/graphrestore")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.samplers import build_dataloader  # noqa: E402
from src.metrics.agenticir_official import official_psnr_ssim  # noqa: E402
from src.training.ema import ExponentialMovingAverage  # noqa: E402
from src.training.runtime import (  # noqa: E402
    autocast_context,
    configure_torch_runtime,
    seed_everything,
)
from src.training.stage0_engine import (  # noqa: E402
    Stage0RestorationDataset,
    build_stage0_model,
    build_stage0_optimizer,
    load_and_validate_stage0_config,
    resume_stage0_checkpoint,
)
from src.utils.hashing import sha256_file  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "prefix"), required=True)
    parser.add_argument("--index", type=int, default=51)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _dataset(resolved: Mapping[str, Any], seed: int) -> Stage0RestorationDataset:
    return Stage0RestorationDataset(
        manifest_path=Path(str(resolved["primary_val_manifest"])),
        training_data_root=Path(str(resolved["training_data_root"])),
        depth_compat_root=PROJECT_ROOT / "artifacts/cache/mioir_depth_compat",
        crop_size=None,
        training=False,
        stage="stage0",
        base_seed=seed,
        agenticir_repo=Path(str(resolved["agenticir_repo"])),
        mioir_repo=Path(str(resolved["mioir_repo"])),
    )


def _scalar(value: object) -> object:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError("metadata tensor is not scalar")
        return value.item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError("metadata sequence is not scalar")
        return value[0]
    return value


def _prediction_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest()


def _optimizer_tensor_devices(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                key = str(value.device)
                counts[key] = counts.get(key, 0) + 1
    return counts


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    wanted = (
        "active_bytes.all.current",
        "inactive_split_bytes.all.current",
        "inactive_split_bytes.all.peak",
        "num_alloc_retries",
        "num_ooms",
        "segment.all.current",
    )
    result = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    result.update({name: int(stats.get(name, 0)) for name in wanted})
    return result


def main() -> int:
    args = _arguments()
    checkpoint = PROJECT_ROOT / "artifacts/checkpoints/stage0/last.pth"
    config_path = PROJECT_ROOT / "configs/stage0_mio_stagea.yaml"
    parent_path = Path(
        "/root/autodl-tmp/aaa/PromptIR_实验归档汇总_20260813/"
        "06_ProVIR修理检查继续修理/provir_完整工作区/"
        "artifacts/checkpoints/stage_a/final_backbone.ckpt"
    )
    started = time.perf_counter()
    result: dict[str, Any] = {
        "label": args.label,
        "mode": args.mode,
        "allocator_env": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "pytorch_alloc_env": os.environ.get("PYTORCH_ALLOC_CONF"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "success": False,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        configure_torch_runtime(tf32=True, cudnn_benchmark=True)
        if args.deterministic:
            # This isolates allocator semantics from the known cross-process
            # cuDNN benchmark algorithm-selection variation observed in the
            # formal runtime.  Model, checkpoint, BF16 autocast, and official
            # metric implementations remain unchanged.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        seed_everything(2027)
        device = torch.device("cuda", torch.cuda.current_device())
        result["hardware"] = {
            "gpu": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        result["determinism"] = {
            "requested": bool(args.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        config, resolved = load_and_validate_stage0_config(config_path)
        dataset = _dataset(resolved, seed=int(config["seed"]))

        # Obtain the checkpoint's exact frozen provenance without placing any
        # checkpoint tensors on CUDA, then release the mmap payload.
        metadata = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(metadata, Mapping):
            raise RuntimeError("checkpoint payload is not a mapping")
        provenance = metadata.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("checkpoint lacks provenance")
        expected_provenance = dict(provenance)
        runtime = provenance.get("runtime")
        if not isinstance(runtime, Mapping):
            raise RuntimeError("checkpoint lacks frozen runtime")
        result["checkpoint_step"] = int(metadata.get("step", -1))
        result["pending_validation_step"] = metadata.get("pending_validation_step")
        result["model_role"] = metadata.get("model_role")
        result["resumable"] = metadata.get("resumable")
        result["frozen_runtime"] = dict(runtime)
        del metadata

        parent_payload = torch.load(
            parent_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(parent_payload, Mapping):
            raise RuntimeError("parent payload is not a mapping")
        model, load_report = build_stage0_model(
            parent_payload,
            gradient_checkpointing=bool(runtime["gradient_checkpointing"]),
        )
        del parent_payload
        model.to(device)
        optimizer, scheduler = build_stage0_optimizer(model, config)
        ema = ExponentialMovingAverage(model, decay=float(config["ema"]["decay"]))
        loaded = resume_stage0_checkpoint(
            checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_provenance=expected_provenance,
        )
        if int(loaded["step"]) != 4000 or loaded.get("pending_validation_step") != 4000:
            raise RuntimeError("probe did not load the pending step-4000 checkpoint")
        del loaded
        torch.cuda.synchronize(device)
        optimizer_devices = _optimizer_tensor_devices(optimizer)
        if any(not name.startswith("cuda") for name in optimizer_devices):
            raise RuntimeError(f"optimizer state is not CUDA-resident: {optimizer_devices}")
        if any(value.device != device for value in ema.shadow.values()):
            raise RuntimeError("EMA is not fully CUDA-resident")
        result["resident_state"] = {
            "model_parameter_count": sum(p.numel() for p in model.parameters()),
            "model_state_tensors": len(model.state_dict()),
            "parent_loaded_tensors": load_report.loaded_count,
            "ema_tensors": len(ema.shadow),
            "optimizer_state_entries": len(optimizer.state),
            "optimizer_tensor_devices": optimizer_devices,
            "scheduler_state": scheduler.state_dict(),
            "before_empty_cache": _memory_snapshot(device),
        }

        source: Any
        loader = None
        if args.mode == "single":
            if args.index < 0 or args.index >= len(dataset):
                raise ValueError("single index is outside dataset")
            source = (dataset[args.index],)
            requested = 1
        else:
            if args.limit <= 0 or args.limit > len(dataset):
                raise ValueError("prefix limit is outside dataset")
            loader, sampler = build_dataloader(
                dataset,
                batch_size=1,
                effective_batch_size=int(runtime["effective_batch"]),
                num_samples=None,
                stage="stage0",
                base_seed=int(config["seed"]),
                start_step=0,
                num_workers=int(config["data"]["loader"]["num_workers"]),
                persistent_workers=bool(
                    config["data"]["loader"]["persistent_workers"]
                ),
                pin_memory=bool(config["data"]["loader"]["pin_memory"]),
                prefetch_factor=int(config["data"]["loader"]["prefetch_factor"]),
                drop_last=False,
                training=False,
            )
            if sampler is not None:
                raise RuntimeError("validation loader unexpectedly has a sampler")
            source = loader
            requested = args.limit

        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        result["resident_state"]["measurement_baseline"] = _memory_snapshot(device)
        rows: list[dict[str, Any]] = []
        with ema.apply_to(model), torch.inference_mode():
            for index, sample in enumerate(source):
                if index >= requested:
                    break
                input_value = sample["input"]
                target_value = sample["target"]
                if not isinstance(input_value, torch.Tensor) or not isinstance(
                    target_value, torch.Tensor
                ):
                    raise RuntimeError("dataset image fields are not tensors")
                if input_value.ndim == 3:
                    input_value = input_value.unsqueeze(0)
                    target_value = target_value.unsqueeze(0)
                input_image = input_value.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                target = target_value.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                with autocast_context(device):
                    prediction = model(input_image)
                metric = official_psnr_ssim(
                    prediction.float(), target.float(), quantize=True
                )
                psnr = float(metric.psnr.item())
                ssim = float(metric.ssim.item())
                prediction_hash = _prediction_sha256(prediction)
                finite = bool(
                    torch.isfinite(prediction).all().item()
                    and math.isfinite(psnr)
                    and math.isfinite(ssim)
                )
                torch.cuda.synchronize(device)
                rows.append(
                    {
                        "ordinal": index,
                        "dataset_index": args.index if args.mode == "single" else index,
                        "sample_id": str(_scalar(sample["sample_id"])),
                        "shape": list(input_image.shape),
                        "prediction_sha256_float32": prediction_hash,
                        "psnr": psnr,
                        "ssim": ssim,
                        "finite": finite,
                        "memory": _memory_snapshot(device),
                    }
                )
                del input_image, target, prediction, metric
        torch.cuda.synchronize(device)
        if len(rows) != requested:
            raise RuntimeError(f"processed {len(rows)} rows, expected {requested}")
        if not all(row["finite"] for row in rows):
            raise FloatingPointError("non-finite prediction or metric")
        result["rows"] = rows
        result["processed"] = len(rows)
        result["peak"] = _memory_snapshot(device)
        result["all_finite"] = True
        result["success"] = True
        # Keep these objects live until after peak capture; this is the central
        # full-training-state invariant of the probe.
        result["training_state_still_resident"] = bool(
            len(optimizer.state) == 495 and len(ema.shadow) == 495
        )
        del loader, optimizer, scheduler, ema, model
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if torch.cuda.is_available():
            try:
                result["failure_memory"] = _memory_snapshot(
                    torch.device("cuda", torch.cuda.current_device())
                )
            except BaseException:
                pass
    result["elapsed_seconds"] = time.perf_counter() - started
    serialized = json.dumps(
        result, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    if args.output is not None:
        output = args.output.resolve()
        if output.parent != Path("/tmp"):
            raise RuntimeError("probe evidence output must remain directly under /tmp")
        output.write_text(serialized + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "result_path": str(output),
                    "success": result["success"],
                    "processed": result.get("processed", 0),
                    "max_allocated_bytes": result.get("peak", {}).get(
                        "max_allocated_bytes"
                    ),
                    "max_reserved_bytes": result.get("peak", {}).get(
                        "max_reserved_bytes"
                    ),
                    "elapsed_seconds": result["elapsed_seconds"],
                    "error": result.get("error"),
                },
                sort_keys=True,
            )
        )
    else:
        print(serialized)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
