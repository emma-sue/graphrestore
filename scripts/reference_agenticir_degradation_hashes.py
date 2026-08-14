#!/usr/bin/env python3
"""Hash locked AgenticIR operator outputs in the isolated reference environment.

This entrypoint is intentionally image-output free: it reads only primary-train
clean images named by an explicit case list and emits hashes of the official
BGR uint8 results.  It exists solely for the mandatory two-recipes-per-operator
parity gate.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.agenticir_degradations import (  # noqa: E402
    AgenticIRDegradationAdapter,
    operator_source_identity,
)
from src.data.manifests import load_primary_manifest  # noqa: E402
from src.utils.io import iter_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--training-data-root", type=Path, required=True)
    parser.add_argument("--depth-compat-root", type=Path, required=True)
    parser.add_argument("--agenticir-repo", type=Path, required=True)
    parser.add_argument("--mioir-repo", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def main() -> int:
    args = parse_args()
    records = load_primary_manifest(
        args.primary_manifest,
        args.training_data_root,
        expected_split="train",
        must_exist=True,
    )
    by_id = {record.sample_id: record for record in records}
    requested = [row for _, row in iter_jsonl(args.cases_jsonl)]
    if not requested:
        raise ValueError("the degradation parity case list is empty")
    adapter = AgenticIRDegradationAdapter(
        agenticir_repo=args.agenticir_repo,
        mioir_repo=args.mioir_repo,
        depth_compat_root=args.depth_compat_root,
    )
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "scipy": metadata.version("scipy"),
        "basicsr": metadata.version("basicsr"),
    }
    source_identity = operator_source_identity(
        args.agenticir_repo, args.mioir_repo
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as destination:
        for request in requested:
            sample_id = request.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in by_id:
                raise ValueError(f"unknown primary-train sample_id: {sample_id!r}")
            record = by_id[sample_id]
            if len(record.operator_params) != 1:
                raise ValueError(f"parity case is not a single operator: {sample_id}")
            clean = cv2.imread(str(record.clean_path), cv2.IMREAD_COLOR)
            if clean is None or clean.dtype != np.uint8:
                raise FileNotFoundError(record.clean_path)
            applied = adapter.apply_sequence(
                clean,
                record.operator_params,
                clean_id=record.clean_id,
                capture_traces=False,
            )
            destination.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "operator": record.operator_order[0],
                        "shape": list(applied.output_bgr_uint8.shape),
                        "dtype": str(applied.output_bgr_uint8.dtype),
                        "sha256": _array_hash(applied.output_bgr_uint8),
                        "versions": versions,
                        "source_identity": source_identity,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
