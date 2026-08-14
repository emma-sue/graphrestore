#!/usr/bin/env python3
"""Enumerate Stage2 Group-A programs and emit frozen relation evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.stage2_distillation import (  # noqa: E402
    Stage2ContractError,
    build_interaction_manifests,
    finalize_interaction_outputs,
    load_frozen_stage1_ema,
    release_stage2_gpu,
    resolve_device,
    resolve_stage2_paths,
    run_interaction_split,
    stage2_resume_bindings,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument(
        "--limit",
        type=int,
        help="bounded per-pair smoke cap; outputs are redirected away from formal artifacts",
    )
    result.add_argument("--device", default="auto", help="auto/cuda for formal execution; cpu is smoke-only")
    result.add_argument("--shard-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--no-resume", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.limit is not None and arguments.limit <= 0:
        raise Stage2ContractError("--limit must be positive")
    smoke_root = (
        PROJECT_ROOT / "artifacts/stage2_smoke" / f"limit_{arguments.limit}"
        if arguments.limit is not None
        else None
    )
    paths = resolve_stage2_paths(
        arguments.config, project_root=PROJECT_ROOT, smoke_root=smoke_root
    )
    sampling = paths.config["data"]["sampling"]
    train_cap = int(sampling["interaction_train_per_group_a_pair_max"])
    val_cap = int(sampling["interaction_val_per_group_a_pair_max"])
    if arguments.limit is not None:
        train_cap = min(train_cap, arguments.limit)
        val_cap = min(val_cap, arguments.limit)
    selection = build_interaction_manifests(
        paths,
        train_per_pair_max=train_cap,
        val_per_pair_max=val_cap,
        seed=int(paths.config["seed"]),
    )
    device = resolve_device(arguments.device)
    snapshot = None
    try:
        snapshot = load_frozen_stage1_ema(paths.checkpoint, device=device)
        common = {
            "model": snapshot.model,
            "training_data_root": paths.training_data_root,
            "depth_compat_root": paths.project_root / "artifacts/cache/depth_compat",
            "checkpoint_sha256": snapshot.checkpoint_sha256,
            "checkpoint_step": snapshot.checkpoint_step,
            "device": device,
            "seed": int(paths.config["seed"]),
            "shard_size": arguments.shard_size,
            "resume": not arguments.no_resume,
            "num_workers": arguments.num_workers,
            "agenticir_repo": paths.resolved["agenticir_repo"],
            "mioir_repo": paths.resolved["mioir_repo"],
            "resume_bindings": stage2_resume_bindings(paths),
        }
        train_records, train_relation_sha = run_interaction_split(
            manifest_path=paths.interaction_train_manifest,
            manifest_sha256=selection["train_sha256"],
            split="train",
            output_path=paths.relation_train,
            **common,
        )
        val_records, val_relation_sha = run_interaction_split(
            manifest_path=paths.interaction_val_manifest,
            manifest_sha256=selection["val_sha256"],
            split="val",
            output_path=paths.relation_val,
            **common,
        )
        decision = finalize_interaction_outputs(
            paths=paths,
            checkpoint_sha256=snapshot.checkpoint_sha256,
            train_manifest_sha256=selection["train_sha256"],
            val_manifest_sha256=selection["val_sha256"],
            train_records=train_records,
            val_records=val_records,
            relation_train_sha256=train_relation_sha,
            relation_val_sha256=val_relation_sha,
        )
        print(
            json.dumps(
                {
                    "status": "PAUSED_AFTER_STAGE2",
                    "decision": str(paths.decision),
                    "train_records": len(train_records),
                    "val_records": len(val_records),
                    "overall": decision["overall"],
                    "warnings": decision["warnings"],
                    "approved": False,
                    "stage3_started": False,
                    "optimizer_created": False,
                    "formal": arguments.limit is None,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        release_stage2_gpu(snapshot.model if snapshot is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
