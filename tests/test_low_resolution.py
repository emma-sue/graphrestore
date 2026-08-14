from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.agenticir_degradations import (
    AgenticIRDegradationAdapter,
    prepare_depth_compat_tree,
)

GROUP_A_MANIFEST = Path(
    "/root/autodl-tmp/graph/data/graphrestore/manifests/"
    "mio100_group_a_test_640.jsonl"
)
from src.data.scale_canonicalizer import (  # noqa: E402
    LOCKED_MATLAB_FUNCTIONS_SHA256,
    MioIRScaleCanonicalizer,
    canonicalize_native_lq,
    load_mioir_matlab_functions,
)


def _adapter(tmp_path: Path) -> AgenticIRDegradationAdapter:
    depth = tmp_path / "depth"
    depth.mkdir()
    # The adapter requires a prebuilt tree even when this test does not use haze.
    (depth / "placeholder.mat").write_bytes(b"placeholder")
    compat = tmp_path / "compat"
    prepare_depth_compat_tree(depth, compat)
    return AgenticIRDegradationAdapter(depth_compat_root=compat)


def test_native_uint8_to_online_rgb_float_matches_locked_scorer_path(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    image = np.random.RandomState(2027).randint(
        0, 256, size=(80, 96, 3), dtype=np.uint8
    )
    native = adapter.operators.lr(image, keep_size=False)
    assert native.shape == (20, 24, 3)
    actual = canonicalize_native_lq(native, scale=4)

    module = load_mioir_matlab_functions(
        expected_sha256=LOCKED_MATLAB_FUNCTIONS_SHA256
    )
    rgb = np.ascontiguousarray(native[:, :, ::-1])
    scorer_input = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    expected = module.imresize(scorer_input, scale=4).clamp(0, 1)
    assert actual.dtype == torch.float32
    assert actual.shape == (3, 80, 96)
    assert torch.equal(actual, expected)


def test_thirty_two_formal_group_a_native_lq_match_locked_scorer_float_path() -> None:
    """Protocol audit on formal native inputs, without reading GT pixels."""

    rows = [
        json.loads(line)
        for line in GROUP_A_MANIFEST.read_text(encoding="utf-8").splitlines()
    ]
    selected = [
        row for row in rows if "low resolution" in row["degradations"]
    ][:32]
    assert len(selected) == 32
    module = load_mioir_matlab_functions(
        expected_sha256=LOCKED_MATLAB_FUNCTIONS_SHA256
    )
    for row in selected:
        native = cv2.imread(row["native_lq_path"], cv2.IMREAD_COLOR)
        assert native is not None and native.dtype == np.uint8
        actual = canonicalize_native_lq(native, scale=4)
        scorer_rgb = cv2.cvtColor(native, cv2.COLOR_BGR2RGB)
        scorer_input = torch.from_numpy(
            scorer_rgb.transpose(2, 0, 1).copy()
        ).float() / 255.0
        expected = module.imresize(scorer_input, scale=4).clamp(0, 1)
        assert tuple(actual.shape) == (
            3,
            native.shape[0] * 4,
            native.shape[1] * 4,
        )
        assert float((actual - expected).abs().max()) <= 1e-6


def test_online_path_keeps_native_quantization_but_does_not_requantize_output(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    y, x = np.mgrid[:80, :96]
    image = np.stack(
        ((x * 7 + y * 3) % 256, (x * 2 + y * 11) % 256, (x * 13 + y) % 256),
        axis=2,
    ).astype(np.uint8)
    native = adapter.operators.lr(image, keep_size=False)
    online = MioIRScaleCanonicalizer().canonicalize_native_lq(native)
    assert bool(torch.any(online.mul(255.0).remainder(1.0).abs() > 1e-6))

    # The legacy shortcut downsamples and upsamples before one final uint8
    # quantization; it is intentionally not equivalent to the V7.1 path.
    legacy_bgr = adapter.operators.lr(image, keep_size=True)
    legacy_rgb = torch.from_numpy(
        np.ascontiguousarray(legacy_bgr[:, :, ::-1]).transpose(2, 0, 1).copy()
    ).float() / 255.0
    assert not torch.equal(online, legacy_rgb)
    assert float((online - legacy_rgb).abs().max()) > 1e-4


def test_canonicalizer_rejects_non_uint8_native() -> None:
    bad = np.zeros((8, 8, 3), dtype=np.float32)
    try:
        canonicalize_native_lq(bad)
    except TypeError as error:
        assert "uint8" in str(error)
    else:  # pragma: no cover
        raise AssertionError("non-uint8 input was silently accepted")
