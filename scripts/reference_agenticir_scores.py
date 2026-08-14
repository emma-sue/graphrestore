#!/usr/bin/env python3
"""Run the locked AgenticIR Scorer code path with only PSNR/SSIM instantiated.

This script must be launched with ``.venv-reference/bin/python``.  It loads the
official scorer source after removing only its module-level eager singleton;
the official ``Scorer.__call__``, image IO, shape handling and score method are
executed unchanged.  Instantiating unrelated LPIPS/NR models is intentionally
avoided because those weights are outside the V7.1 protocol.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch


def _install_torchvision_compatibility() -> None:
    from torchvision.transforms.functional import rgb_to_grayscale

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = rgb_to_grayscale
    sys.modules.setdefault("torchvision.transforms.functional_tensor", shim)


def _load_official_scorer_class(source_path: Path):
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    filtered = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "scorer" for target in node.targets
        ):
            continue
        filtered.append(node)
    tree.body = filtered
    module = types.ModuleType("agenticir_locked_scorer_reference")
    module.__file__ = str(source_path)
    exec(compile(tree, str(source_path), "exec"), module.__dict__)
    return module


def _build_two_metric_scorer(scorer_class, device: str):
    import pyiqa

    scorer = scorer_class.__new__(scorer_class)
    scorer.fr_metric_name_lst = ["psnr", "ssim"]
    scorer.nr_metric_name_lst = []
    # No explicit overrides: this intentionally exercises pyiqa 0.1.10's
    # defaults exactly as AgenticIR's locked Scorer does.
    scorer.fr_metrics = [pyiqa.create_metric(name, device=device) for name in scorer.fr_metric_name_lst]
    scorer.nr_metrics = []
    scorer.metrics = scorer.fr_metrics
    scorer.metric_name_lst = scorer.fr_metric_name_lst
    scorer.lower_better_dict = {
        metric.metric_name: metric.lower_better for metric in scorer.metrics
    }
    return scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--scorer-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _install_torchvision_compatibility()
    official_module = _load_official_scorer_class(args.scorer_path.resolve())
    scorer = _build_two_metric_scorer(official_module.Scorer, args.device)
    reference_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "pyiqa": metadata.version("pyiqa"),
        "basicsr": metadata.version("basicsr"),
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs_jsonl.open("r", encoding="utf-8") as source, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            scores = scorer(Path(row["prediction"]), Path(row["target"]))
            values = {name: value for name, _lower_better, value in scores}
            canonical_fields = {}
            if row["kind"] == "native_x4_mismatch":
                native = scorer._get_img_tensor(Path(row["prediction"]))
                target = scorer._get_img_tensor(Path(row["target"]))
                if native.shape[-2] * 4 != target.shape[-2] or native.shape[-1] * 4 != target.shape[-1]:
                    raise ValueError(f"invalid x4 mismatch pair: {row['id']}")
                canonical = official_module.imresize(native[0], scale=4).clamp(0, 1).contiguous()
                raw_float = canonical.cpu().numpy().tobytes(order="C")
                raw_uint8 = canonical.mul(255).round().to(torch.uint8).cpu().numpy().tobytes(order="C")
                canonical_fields = {
                    "canonical_shape": list(canonical.shape),
                    "canonical_float_sha256": hashlib.sha256(raw_float).hexdigest(),
                    "canonical_uint8_sha256": hashlib.sha256(raw_uint8).hexdigest(),
                }
            destination.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "kind": row["kind"],
                        "psnr": values["psnr"],
                        "ssim": values["ssim"],
                        "reference_versions": reference_versions,
                        **canonical_fields,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
