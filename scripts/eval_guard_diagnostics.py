#!/usr/bin/env python3
"""Evaluate Stage3 guard/planner diagnostics on primary_val only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GraphRestoreEpisodeDataset  # noqa: E402
from src.training.stage3_engine import (  # noqa: E402
    Stage3ContractError,
    load_relation_records,
    load_stage3_best_ema,
    validate_stage3,
    validate_stage3_approval,
)
from src.utils.io import atomic_write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage3_planner.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/stage3/best_ema.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/stage3_guard_diagnostics.json"),
    )
    return parser


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run(arguments: argparse.Namespace) -> int:
    # Approval/hashes are checked before CUDA or dataset/pixel access.
    paths = validate_stage3_approval(
        _project_path(arguments.config),
        project_root=PROJECT_ROOT,
        require_orchestrator_running=False,
    )
    relation_val = load_relation_records(
        paths.relation_val,
        split="val",
        parent_checkpoint_sha256=paths.approval.bindings["stage1_checkpoint"]["sha256"],
        interaction_manifest_sha256=paths.approval.bindings["interaction_val_manifest"]["sha256"],
    )
    if not torch.cuda.is_available():
        raise Stage3ContractError("formal Stage3 diagnostics require CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = _project_path(arguments.checkpoint)
    model = load_stage3_best_ema(
        paths,
        checkpoint,
        device=device,
        load_frozen_thresholds=True,
    )
    dataset = GraphRestoreEpisodeDataset(
        paths.val_manifest,
        paths.training_data_root,
        PROJECT_ROOT / "artifacts/cache/agenticir_depth_compat",
        crop_size=None,
        training=False,
        stage="stage3",
        base_seed=int(paths.config["seed"]),
        agenticir_repo=paths.resolved["agenticir_repo"],
        mioir_repo=paths.resolved["mioir_repo"],
    )
    summary = validate_stage3(
        model,
        dataset,
        relation_val,
        device=device,
        use_bf16=True,
        presence_threshold=0.5,
    )
    summary["evaluated_checkpoint"] = str(checkpoint)
    summary["diagnostics_affect_checkpoint_rank"] = False
    atomic_write_json(_project_path(arguments.output), summary)
    print(json.dumps(summary["guard"], ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return run(arguments)
    except (Stage3ContractError, FileNotFoundError, ValueError) as exc:
        print(f"Stage3 diagnostics refused: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
