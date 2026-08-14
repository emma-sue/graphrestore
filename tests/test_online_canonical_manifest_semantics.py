from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import src.data.scale_canonicalizer as scale_canonicalizer
from src.data.scale_canonicalizer import (
    bgr_uint8_to_rgb_float,
    canonicalize_native_lq,
    load_agenticir_online_canonical_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = PROJECT_ROOT / "scripts/build_agenticir_online_canonical_manifests.py"
CONFIG_PATH = PROJECT_ROOT / "configs/resolved_paths.yaml"


def _load_builder():
    specification = importlib.util.spec_from_file_location(
        "graphrestore_online_manifest_builder_test", BUILDER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_online_canonical_manifest_semantics(
    tmp_path: Path, monkeypatch
) -> None:
    builder = _load_builder()
    inventory = builder.build_manifests(CONFIG_PATH, tmp_path)
    assert len(inventory["manifests"]) == 4
    all_rows = []
    for item in inventory["manifests"].values():
        output = Path(item["path"])
        assert output.suffix == ".jsonl"
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(rows) == item["rows"]
        all_rows.extend(rows)
        for row in rows:
            assert row["input_mode"] == "agenticir_online_canonical"
            assert row["input_path"] == row["native_lq_path"]
            assert row["native_lq_path"]
            assert row["gt_path"]
            assert isinstance(row["contains_low_resolution"], bool)
            assert row["native_scale"] == (
                0.25 if row["contains_low_resolution"] else 1.0
            )
            assert "canonical_lq_path" not in row
            assert "legacy_opencv_canonical_lq_path" not in row
            # No value in the derived row may point at the old processed PNG.
            assert not any(
                isinstance(value, str) and "/processed/" in value
                for value in row.values()
            )
            if row["contains_low_resolution"]:
                assert row["online_canonicalization"] == (
                    "mioir_basicsr_native_uint8_to_rgb_float_x4"
                )
                assert row["online_scale_factor"] == 4
                assert row["requantize_after_online_resize"] is False
            else:
                assert row["online_canonicalization"] == (
                    "native_uint8_to_rgb_float_identity"
                )
                assert row["online_scale_factor"] == 1
    assert any(row["contains_low_resolution"] for row in all_rows)
    assert any(not row["contains_low_resolution"] for row in all_rows)
    # The builder creates metadata only, never a materialized canonical PNG.
    assert not list(tmp_path.rglob("*.png"))

    # Exercise both formal runtime branches.  Instrument decoding to prove
    # that only native_lq_path is opened and make any OpenCV resize a hard fail.
    source_lr_row = next(
        row for row in all_rows if row["contains_low_resolution"]
    )
    source_non_lr_row = next(
        row for row in all_rows if not row["contains_low_resolution"]
    )
    # Do not decode formal MiO100 pixels during preflight.  Clone the frozen
    # metadata modes onto deterministic temporary native uint8 fixtures.
    grid_y, grid_x = np.mgrid[:12, :16]
    lr_native_fixture = np.stack(
        (
            (grid_x * 9 + grid_y * 3) % 256,
            (grid_x * 2 + grid_y * 13) % 256,
            (grid_x * 7 + grid_y * 5) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    non_lr_native_fixture = cv2.resize(
        lr_native_fixture, (64, 48), interpolation=cv2.INTER_NEAREST
    )
    lr_native_path = tmp_path / "fixture_native_lr.png"
    non_lr_native_path = tmp_path / "fixture_native_identity.png"
    assert cv2.imwrite(str(lr_native_path), lr_native_fixture)
    assert cv2.imwrite(str(non_lr_native_path), non_lr_native_fixture)
    lr_row = dict(source_lr_row)
    non_lr_row = dict(source_non_lr_row)
    lr_row.update(
        native_lq_path=str(lr_native_path), input_path=str(lr_native_path)
    )
    non_lr_row.update(
        native_lq_path=str(non_lr_native_path), input_path=str(non_lr_native_path)
    )
    real_imread = cv2.imread
    opened_paths: list[str] = []

    def audited_imread(path: str, flags: int):
        opened_paths.append(path)
        return real_imread(path, flags)

    def forbidden_opencv_resize(*args, **kwargs):  # pragma: no cover - must not run.
        raise AssertionError("formal online canonicalization called cv2.resize")

    monkeypatch.setattr(scale_canonicalizer.cv2, "imread", audited_imread)
    monkeypatch.setattr(scale_canonicalizer.cv2, "resize", forbidden_opencv_resize)
    lr_actual = load_agenticir_online_canonical_input(lr_row)
    non_lr_actual = load_agenticir_online_canonical_input(non_lr_row)

    assert opened_paths == [lr_row["native_lq_path"], non_lr_row["native_lq_path"]]
    assert all("/processed/" not in path for path in opened_paths)
    lr_native = real_imread(lr_row["native_lq_path"], cv2.IMREAD_COLOR)
    non_lr_native = real_imread(non_lr_row["native_lq_path"], cv2.IMREAD_COLOR)
    assert lr_native is not None and non_lr_native is not None
    assert torch.equal(lr_actual, canonicalize_native_lq(lr_native, scale=4))
    assert torch.equal(non_lr_actual, bgr_uint8_to_rgb_float(non_lr_native))
    assert bool(torch.any(lr_actual.mul(255.0).remainder(1.0).abs() > 1e-6))
