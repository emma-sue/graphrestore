#!/usr/bin/env python3
"""Approve, finalize, or verify the CPU-only Table-1 backend recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


# Set these before importing any project module.  The recovery implementation
# itself is standard-library only and rejects heavyweight metric/image imports.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.agenticir_table1_recovery import (  # noqa: E402
    APPROVAL_EXECUTE_TOKEN,
    FINALIZE_EXECUTE_TOKEN,
    Table1RecoveryError,
    approval_verify_only,
    assert_cpu_only_entrypoint,
    finalize_recovery,
    production_paths,
    publish_approval,
    verify_recovery,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only, no-image/no-CUDA recovery for the completed immutable "
            "AgenticIR Table-1 score shards"
        )
    )
    parser.add_argument("--phase", choices=("approval", "finalize"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute", metavar="EXACT_TOKEN")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    assert_cpu_only_entrypoint()
    paths = production_paths()
    if args.verify_only:
        result = (
            approval_verify_only(paths)
            if args.phase == "approval"
            else verify_recovery(paths, require_complete=False)
        )
    elif args.phase == "approval":
        if args.execute != APPROVAL_EXECUTE_TOKEN:
            raise Table1RecoveryError(
                f"approval requires --execute {APPROVAL_EXECUTE_TOKEN}"
            )
        result = publish_approval(paths, execute_token=args.execute)
    else:
        if args.execute != FINALIZE_EXECUTE_TOKEN:
            raise Table1RecoveryError(
                f"finalization requires --execute {FINALIZE_EXECUTE_TOKEN}"
            )
        result = finalize_recovery(paths, execute_token=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Table1RecoveryError as exc:
        print(f"AgenticIR Table-1 recovery contract error: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
