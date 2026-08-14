#!/usr/bin/env python3
"""Evaluate one Stage0 checkpoint on frozen primary_val single + Group A only."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpointing import verify_provenance  # noqa: E402
from src.data.samplers import build_dataloader  # noqa: E402
from src.training.ema import ExponentialMovingAverage  # noqa: E402
from src.training.runtime import configure_torch_runtime, seed_everything  # noqa: E402
from src.training.stage0_engine import (  # noqa: E402
    Stage0ContractError,
    Stage0RestorationDataset,
    Stage0Runtime,
    assert_stage0_preflight,
    build_stage0_model,
    build_stage0_provenance,
    evaluate_primary_val,
    load_and_validate_stage0_config,
)
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.io import atomic_write_json, utc_now_iso  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official-parity in-memory evaluation on primary_val only"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--weights",
        choices=("ema", "model"),
        default="ema",
        help="EMA is the locked validation/checkpoint-selection weight source",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("full-resolution primary_val evaluation requires CUDA")
    config_path = arguments.config.resolve()
    checkpoint_path = arguments.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    config, resolved = load_and_validate_stage0_config(config_path)
    assert_stage0_preflight(PROJECT_ROOT)
    configure_torch_runtime(tf32=True, cudnn_benchmark=True)
    seed_everything(int(config["seed"]))

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise Stage0ContractError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != "graphrestore-checkpoint-v1":
        raise Stage0ContractError("unsupported checkpoint schema")
    if payload.get("stage") != "stage0":
        raise Stage0ContractError("eval_primary_val accepts Stage0 checkpoints only")
    actual_provenance = payload.get("provenance")
    if not isinstance(actual_provenance, Mapping):
        raise Stage0ContractError("checkpoint has no provenance")
    frozen_runtime = actual_provenance.get("runtime")
    if not isinstance(frozen_runtime, Mapping):
        raise Stage0ContractError("checkpoint has no frozen runtime")
    runtime = Stage0Runtime(**dict(frozen_runtime))

    parent_payload = torch.load(
        Path(str(resolved["stage_a_parent_checkpoint"])),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model, load_report = build_stage0_model(
        parent_payload,
        gradient_checkpointing=runtime.gradient_checkpointing,
    )
    expected_provenance = build_stage0_provenance(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        config=config,
        resolved=resolved,
        runtime=runtime,
        load_report=load_report,
    )
    verify_provenance(actual_provenance, expected_provenance)
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping):
        raise Stage0ContractError("checkpoint has no model state")
    model.load_state_dict(model_state, strict=True)
    if arguments.weights == "ema":
        ema_state = payload.get("ema")
        if not isinstance(ema_state, Mapping):
            raise Stage0ContractError("EMA evaluation requested but checkpoint has no EMA")
        ema = ExponentialMovingAverage(model, decay=float(config["ema"]["decay"]))
        ema.load_state_dict(ema_state)
        ema.copy_to(model)
        del ema

    device = torch.device("cuda", torch.cuda.current_device())
    model.to(device)
    dataset = Stage0RestorationDataset(
        manifest_path=Path(str(resolved["primary_val_manifest"])),
        training_data_root=Path(str(resolved["training_data_root"])),
        depth_compat_root=PROJECT_ROOT / "artifacts/cache/mioir_depth_compat",
        crop_size=None,
        training=False,
        stage="stage0",
        base_seed=int(config["seed"]),
        agenticir_repo=Path(str(resolved["agenticir_repo"])),
        mioir_repo=Path(str(resolved["mioir_repo"])),
    )
    loader_config = config["data"]["loader"]
    loader, sampler = build_dataloader(
        dataset,
        batch_size=1,
        effective_batch_size=8,
        num_samples=None,
        stage="stage0",
        base_seed=int(config["seed"]),
        start_step=0,
        num_workers=int(loader_config["num_workers"]),
        persistent_workers=bool(loader_config["persistent_workers"]),
        pin_memory=bool(loader_config["pin_memory"]),
        prefetch_factor=int(loader_config["prefetch_factor"]),
        drop_last=False,
        training=False,
    )
    if sampler is not None:
        raise RuntimeError("primary_val loader unexpectedly has a sampler")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"primary_val {done}/{total}", flush=True)

    result = evaluate_primary_val(
        model,
        dataset,
        device=device,
        dataloader=loader,
        progress=progress,
    )
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else PROJECT_ROOT
        / "artifacts/metrics"
        / f"stage0_primary_val_{checkpoint_path.stem}_{arguments.weights}.json"
    )
    artifact = {
        "schema_version": "graphrestore-stage0-primary-val-v1",
        "protocol_id": "agenticir_official_parity",
        "created_utc": utc_now_iso(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "weights": arguments.weights,
        "output_quantization": "clamp_round_uint8_in_memory",
        "validation_source": "primary_val_single_and_group_a_only",
        **result.to_dict(),
    }
    atomic_write_json(output, artifact)
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
