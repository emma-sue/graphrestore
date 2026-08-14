#!/usr/bin/env python3
"""Mandatory real-data/GPU Stage0 one-batch forward/backward checks.

This file is executable by the orchestrator.  It intentionally defines no
pytest test functions, so the CPU unit-test suite does not accidentally launch
a full Restormer CUDA pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.samplers import EpisodeRequest  # noqa: E402
from src.losses.restoration import charbonnier  # noqa: E402
from src.training.runtime import autocast_context, configure_torch_runtime, seed_everything  # noqa: E402
from src.training.stage0_engine import (  # noqa: E402
    Stage0RestorationDataset,
    assert_stage0_preflight,
    build_stage0_model,
    load_and_validate_stage0_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("single", "group_a_low_resolution"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/stage0_mio_stagea.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("mandatory one-batch checks require CUDA")
    config, resolved = load_and_validate_stage0_config(arguments.config)
    assert_stage0_preflight(PROJECT_ROOT)
    seed = int(config["seed"])
    seed_everything(seed)
    configure_torch_runtime(tf32=True, cudnn_benchmark=True)
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
    selected = None
    for index, record in enumerate(dataset.records):
        if arguments.case == "single" and record.group == "single":
            selected = index
            break
        if (
            arguments.case == "group_a_low_resolution"
            and record.group == "A"
            and record.contains_low_resolution
        ):
            selected = index
            break
    if selected is None:
        raise RuntimeError(f"no primary_train recipe found for {arguments.case}")
    sample = dataset[
        EpisodeRequest(
            index=selected,
            episode_type="stage0_restoration",
            absolute_step=0,
            sample_cursor=0,
        )
    ]
    if arguments.case == "group_a_low_resolution":
        assert sample["group"] == "A"
        assert sample["contains_low_resolution"] is True
    if not 0.0 <= float(sample["input"].min()) <= float(sample["input"].max()) <= 1.0:
        raise RuntimeError("Stage0 input escaped RGB [0,1]")
    # The dedicated view must not pretend that Stage0 consumed Stage1 targets.
    forbidden = {"only_i", "only_j", "guard_targets", "presence_target"}
    if forbidden.intersection(sample):
        raise RuntimeError("Stage0 fast batch fabricated unused subset/guard fields")

    parent_payload = torch.load(
        Path(str(resolved["stage_a_parent_checkpoint"])),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(parent_payload, Mapping):
        raise RuntimeError("parent checkpoint is not a mapping")
    model, report = build_stage0_model(parent_payload)
    device = torch.device("cuda", torch.cuda.current_device())
    model.to(device).train()
    input_image = sample["input"].unsqueeze(0).to(device, dtype=torch.float32)
    target = sample["target"].unsqueeze(0).to(device, dtype=torch.float32)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with autocast_context(device):
        prediction = model(input_image)
        loss = charbonnier(prediction, target)
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise FloatingPointError("non-finite one-batch forward")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("non-finite or absent one-batch gradients")
    torch.cuda.synchronize(device)
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak = int(torch.cuda.max_memory_reserved(device))
    result = {
        "case": arguments.case,
        "sample_id": sample["sample_id"],
        "operator_order": sample["operator_order"],
        "shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "gradient_tensors": len(gradients),
        "parent_loaded_tensors": report.loaded_count,
        "peak_reserved_bytes": peak,
        "peak_reserved_fraction": peak / total_memory,
        "finite": math.isfinite(float(loss.detach())),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
