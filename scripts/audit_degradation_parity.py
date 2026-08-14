#!/usr/bin/env python3
"""Audit all eight AgenticIR degradations across training/reference envs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.agenticir_degradations import (  # noqa: E402
    AgenticIRDegradationAdapter,
    operator_source_identity,
    prepare_depth_compat_tree,
)
from src.data.manifests import load_primary_manifest  # noqa: E402
from src.utils.audit import AuditTrail  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolved-paths",
        type=Path,
        default=PROJECT_ROOT / "configs/resolved_paths.yaml",
    )
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=PROJECT_ROOT / ".venv-reference/bin/python",
    )
    parser.add_argument("--per-operator", type=int, default=2)
    parser.add_argument(
        "--depth-compat-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/runtime/depth_compat",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts/audits/degradation_parity.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/DEGRADATION_PROTOCOL.md",
    )
    return parser.parse_args()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _selected_records(records, per_operator: int):
    selected = defaultdict(list)
    for record in records:
        if len(record.operator_params) != 1:
            continue
        name = record.operator_order[0]
        if len(selected[name]) < per_operator:
            selected[name].append(record)
    if len(selected) != 8 or any(
        len(rows) != per_operator for rows in selected.values()
    ):
        raise RuntimeError(
            f"could not select {per_operator} rows for every operator: "
            f"{dict((key, len(value)) for key, value in selected.items())}"
        )
    return tuple(
        row for name in sorted(selected) for row in selected[name]
    )


def _write_case_list(records, destination: Path) -> None:
    atomic_write_text(
        destination,
        "".join(
            json.dumps({"sample_id": record.sample_id}, sort_keys=True) + "\n"
            for record in records
        ),
    )


def main() -> int:
    args = parse_args()
    if args.per_operator < 2:
        raise ValueError("V7.1 requires at least two recipes for every operator")
    resolved = load_yaml(args.resolved_paths)
    training_root = Path(resolved["training_data_root"])
    manifest = Path(resolved["primary_train_manifest"])
    agenticir_repo = Path(resolved["agenticir_repo"])
    mioir_repo = Path(resolved["mioir_repo"])
    records = load_primary_manifest(
        manifest, training_root, expected_split="train", must_exist=True
    )
    selected = _selected_records(records, args.per_operator)
    prepare_depth_compat_tree(
        training_root / "depth/depth", args.depth_compat_root
    )
    adapter = AgenticIRDegradationAdapter(
        agenticir_repo=agenticir_repo,
        mioir_repo=mioir_repo,
        depth_compat_root=args.depth_compat_root,
    )

    fast: dict[str, dict[str, object]] = {}
    for record in selected:
        clean = cv2.imread(str(record.clean_path), cv2.IMREAD_COLOR)
        if clean is None or clean.dtype != np.uint8:
            raise FileNotFoundError(record.clean_path)
        applied = adapter.apply_sequence(
            clean,
            record.operator_params,
            clean_id=record.clean_id,
            capture_traces=False,
        )
        fast[record.sample_id] = {
            "operator": record.operator_order[0],
            "shape": list(applied.output_bgr_uint8.shape),
            "dtype": str(applied.output_bgr_uint8.dtype),
            "sha256": _hash_array(applied.output_bgr_uint8),
        }

    with tempfile.TemporaryDirectory(
        prefix="graphrestore-degradation-parity-"
    ) as temporary_text:
        temporary = Path(temporary_text)
        cases = temporary / "cases.jsonl"
        reference_output = temporary / "reference.jsonl"
        _write_case_list(selected, cases)
        command = [
            str(args.reference_python),
            str(PROJECT_ROOT / "scripts/reference_agenticir_degradation_hashes.py"),
            "--cases-jsonl",
            str(cases),
            "--primary-manifest",
            str(manifest),
            "--training-data-root",
            str(training_root),
            "--depth-compat-root",
            str(args.depth_compat_root),
            "--agenticir-repo",
            str(agenticir_repo),
            "--mioir-repo",
            str(mioir_repo),
            "--output-jsonl",
            str(reference_output),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        reference = {
            row["sample_id"]: row for _, row in iter_jsonl(reference_output)
        }

    audit = AuditTrail(
        protocol="graphrestore-v7.1-agenticir-degradation-reference-parity"
    )
    output_rows = []
    for record in selected:
        sample_id = record.sample_id
        expected = reference.get(sample_id)
        actual = fast[sample_id]
        exact = bool(
            expected
            and actual["shape"] == expected["shape"]
            and actual["dtype"] == expected["dtype"]
            and actual["sha256"] == expected["sha256"]
        )
        audit.require(
            exact,
            f"operator.{record.operator_order[0]}.{sample_id}",
            "BGR uint8 bytes exact",
            f"fast={actual}, reference={expected}",
        )
        output_rows.append(
            {
                "sample_id": sample_id,
                "operator": record.operator_order[0],
                "exact": exact,
                "fast_sha256": actual["sha256"],
                "reference_sha256": None if expected is None else expected["sha256"],
            }
        )
    first_reference = next(iter(reference.values()))
    audit.facts.update(
        {
            "pairs": len(selected),
            "per_operator": args.per_operator,
            "all_exact": all(row["exact"] for row in output_rows),
            "fast_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "torch": torch.__version__,
            },
            "reference_versions": first_reference["versions"],
            "operator_source_identity": operator_source_identity(
                agenticir_repo, mioir_repo
            ),
            "rows": output_rows,
        }
    )
    atomic_write_json(args.output_json, audit.to_dict())
    atomic_write_text(
        args.report,
        audit.to_markdown(title="AgenticIR Degradation Reference Parity"),
    )
    print(audit.to_markdown(title="AgenticIR Degradation Reference Parity"))
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
