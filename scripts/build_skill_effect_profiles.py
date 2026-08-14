#!/usr/bin/env python3
"""Build Stage2 single-degradation skill-effect profiles, inference only."""

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
    load_frozen_stage1_ema,
    release_stage2_gpu,
    resolve_device,
    resolve_stage2_paths,
    run_effect_profiles,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument(
        "--limit",
        type=int,
        help="bounded per-source smoke cap; outputs are redirected away from formal artifacts",
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
    contract_cap = int(sampling["single_val_per_source_max"])
    cap = min(contract_cap, arguments.limit) if arguments.limit is not None else contract_cap
    device = resolve_device(arguments.device)
    snapshot = None
    try:
        snapshot = load_frozen_stage1_ema(paths.checkpoint, device=device)
        profile = run_effect_profiles(
            model=snapshot.model,
            paths=paths,
            checkpoint_sha256=snapshot.checkpoint_sha256,
            checkpoint_step=snapshot.checkpoint_step,
            device=device,
            per_source_max=cap,
            seed=int(paths.config["seed"]),
            shard_size=arguments.shard_size,
            resume=not arguments.no_resume,
            num_workers=arguments.num_workers,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "output": str(paths.effect_profiles),
                    "record_count": profile["record_count"],
                    "effect_vector_dim": profile["effect_vector_dim"],
                    "stage1_checkpoint_sha256": snapshot.checkpoint_sha256,
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
