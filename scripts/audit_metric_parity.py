#!/usr/bin/env python3
"""Audit fast metrics/canonicalization against locked AgenticIR code."""

from __future__ import annotations

import argparse
import csv
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

from src.data.scale_canonicalizer import canonicalize_native_lq  # noqa: E402
from src.metrics.agenticir_official import official_psnr_ssim  # noqa: E402
from src.utils.audit import AuditTrail  # noqa: E402
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-paths", type=Path, default=PROJECT_ROOT / "configs/resolved_paths.yaml")
    parser.add_argument("--reference-python", type=Path, default=PROJECT_ROOT / ".venv-reference/bin/python")
    parser.add_argument("--full-pairs", type=int, default=16)
    parser.add_argument("--mismatch-pairs", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "artifacts/metrics/metric_parity_per_image.csv")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "artifacts/metrics/metric_parity_summary.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/METRIC_PROTOCOL.md")
    return parser.parse_args()


def _image_tensor(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255).unsqueeze(0)


def _hash_tensor_bytes(tensor: torch.Tensor, *, uint8: bool) -> str:
    value = tensor.mul(255).round().to(torch.uint8) if uint8 else tensor
    raw = value.contiguous().cpu().numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _write_pairs(rows: list[dict[str, str]], path: Path) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, payload)


def _reference_scores(args: argparse.Namespace, pairs: Path, output: Path, scorer: Path) -> None:
    command = [
        str(args.reference_python),
        str(PROJECT_ROOT / "scripts/reference_agenticir_scores.py"),
        "--pairs-jsonl",
        str(pairs),
        "--output-jsonl",
        str(output),
        "--scorer-path",
        str(scorer),
        "--device",
        "cpu",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.full_pairs < 16 or args.mismatch_pairs < 8:
        raise ValueError("V7.1 requires at least 16 full-size and 8 native-x4 pairs")
    resolved = load_yaml(args.resolved_paths)
    training_root = Path(resolved["training_data_root"])
    clean_rows = [row for _, row in iter_jsonl(Path(resolved["clean_val_manifest"]))]
    needed = max(args.full_pairs, args.mismatch_pairs)
    if len(clean_rows) < needed:
        raise RuntimeError("not enough clean validation images")

    audit = AuditTrail(protocol="graphrestore-v7.1-agenticir-metric-parity")
    audit.facts["versions"] = {
        "torch_fast": torch.__version__,
        "opencv_fast": cv2.__version__,
        "reference_python": str(args.reference_python),
        "agenticir_commit": resolved["expected_identity"]["agenticir_commit"],
        "agenticir_scorer_sha256": sha256_file(resolved["agenticir_scorer"]),
    }

    with tempfile.TemporaryDirectory(prefix="graphrestore-metric-parity-") as temporary_text:
        temporary = Path(temporary_text)
        pairs: list[dict[str, str]] = []
        for index, row in enumerate(clean_rows[:needed]):
            target = (training_root / row["clean_path"]).resolve()
            bgr = cv2.imread(str(target), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(target)
            if index < args.full_pairs:
                prediction = bgr.copy()
                # Deterministic uint8-domain perturbation without resize/colour conversion.
                plane = prediction[index % prediction.shape[0] :: 37, index % prediction.shape[1] :: 41, index % 3]
                prediction[index % prediction.shape[0] :: 37, index % prediction.shape[1] :: 41, index % 3] = np.clip(
                    plane.astype(np.int16) + (index % 5) + 1, 0, 255
                ).astype(np.uint8)
                prediction_path = temporary / f"full_{index:02d}.png"
                if not cv2.imwrite(str(prediction_path), prediction):
                    raise RuntimeError(f"failed to write {prediction_path}")
                pairs.append({"id": f"full_{index:02d}", "kind": "full_size", "prediction": str(prediction_path), "target": str(target)})
            if index < args.mismatch_pairs:
                if bgr.shape[0] % 4 or bgr.shape[1] % 4:
                    raise ValueError(f"clean image not divisible by four: {target}")
                native = cv2.resize(
                    bgr,
                    (bgr.shape[1] // 4, bgr.shape[0] // 4),
                    interpolation=cv2.INTER_AREA,
                )
                native_path = temporary / f"native_{index:02d}.png"
                if not cv2.imwrite(str(native_path), native):
                    raise RuntimeError(f"failed to write {native_path}")
                pairs.append({"id": f"native_{index:02d}", "kind": "native_x4_mismatch", "prediction": str(native_path), "target": str(target)})

        pairs_path = temporary / "pairs.jsonl"
        reference_path = temporary / "reference.jsonl"
        _write_pairs(pairs, pairs_path)
        _reference_scores(args, pairs_path, reference_path, Path(resolved["agenticir_scorer"]))
        reference = {row["id"]: row for _, row in iter_jsonl(reference_path)}
        if not reference:
            raise RuntimeError("locked reference returned no metric rows")
        audit.facts["versions"]["reference_environment"] = next(
            iter(reference.values())
        )["reference_versions"]

        output_rows: list[dict[str, object]] = []
        for pair in pairs:
            target_tensor = _image_tensor(Path(pair["target"]))
            if pair["kind"] == "native_x4_mismatch":
                native_bgr = cv2.imread(pair["prediction"], cv2.IMREAD_COLOR)
                prediction_tensor = canonicalize_native_lq(native_bgr).unsqueeze(0)
            else:
                prediction_tensor = _image_tensor(Path(pair["prediction"]))
            if prediction_tensor.shape != target_tensor.shape:
                raise RuntimeError(
                    f"canonical shape mismatch for {pair['id']}: {prediction_tensor.shape} != {target_tensor.shape}"
                )
            # The reference scorer consumes the decoded PNG directly.  Full-size
            # predictions are already uint8-domain values, while its x4 mismatch
            # branch keeps BasicSR's float result (it does not requantize after
            # interpolation), so no additional round trip is allowed here.
            fast = official_psnr_ssim(prediction_tensor, target_tensor, quantize=False)
            expected = reference[pair["id"]]
            row = {
                "id": pair["id"],
                "kind": pair["kind"],
                "reference_psnr": float(expected["psnr"]),
                "fast_psnr": float(fast.psnr.item()),
                "psnr_abs_diff": abs(float(expected["psnr"]) - float(fast.psnr.item())),
                "reference_ssim": float(expected["ssim"]),
                "fast_ssim": float(fast.ssim.item()),
                "ssim_abs_diff": abs(float(expected["ssim"]) - float(fast.ssim.item())),
            }
            if pair["kind"] == "native_x4_mismatch":
                row["canonical_float_exact"] = _hash_tensor_bytes(prediction_tensor[0], uint8=False) == expected["canonical_float_sha256"]
                row["canonical_uint8_exact"] = _hash_tensor_bytes(prediction_tensor[0], uint8=True) == expected["canonical_uint8_sha256"]
            output_rows.append(row)

    max_psnr = max(float(row["psnr_abs_diff"]) for row in output_rows)
    max_ssim = max(float(row["ssim_abs_diff"]) for row in output_rows)
    mismatch_rows = [row for row in output_rows if row["kind"] == "native_x4_mismatch"]
    canonical_float_exact = all(bool(row["canonical_float_exact"]) for row in mismatch_rows)
    canonical_uint8_exact = all(bool(row["canonical_uint8_exact"]) for row in mismatch_rows)
    audit.require(max_psnr <= 1e-5, "metric.psnr.max_abs", f"max_abs={max_psnr:.12g}", f"max_abs={max_psnr:.12g} exceeds 1e-5")
    audit.require(max_ssim <= 1e-5, "metric.ssim.max_abs", f"max_abs={max_ssim:.12g}", f"max_abs={max_ssim:.12g} exceeds 1e-5")
    audit.require(canonical_float_exact, "low_resolution.canonical_float_exact", f"{len(mismatch_rows)} pairs byte-identical", "float canonical bytes differ")
    audit.require(canonical_uint8_exact, "low_resolution.canonical_uint8_exact", f"{len(mismatch_rows)} pairs byte-identical", "uint8 canonical bytes differ")
    audit.facts.update(
        {
            "full_size_pairs": args.full_pairs,
            "native_x4_pairs": args.mismatch_pairs,
            "max_psnr_abs_diff": max_psnr,
            "max_ssim_abs_diff": max_ssim,
            "canonical_float_exact": canonical_float_exact,
            "canonical_uint8_exact": canonical_uint8_exact,
        }
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    for optional in ("canonical_float_exact", "canonical_uint8_exact"):
        if optional not in fieldnames:
            fieldnames.append(optional)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    atomic_write_json(args.output_json, audit.to_dict())
    atomic_write_text(args.report, audit.to_markdown(title="AgenticIR Official Metric Protocol and Parity"))
    print(audit.to_markdown(title="AgenticIR Official Metric Protocol and Parity"))
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
