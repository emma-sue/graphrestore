from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from src.data.agenticir_degradations import (
    AgenticIRDegradationAdapter,
    prepare_depth_compat_tree,
)
from src.data.manifests import OperatorParameter, PrimaryRecipe, SKILL_TO_ID
from src.data.scale_canonicalizer import (
    MioIRScaleCanonicalizer,
    bgr_uint8_to_rgb_float,
)
from src.data.subset_targets import synthesize_subset_targets


def _adapter(tmp_path: Path, clean_id: str = "synthetic") -> AgenticIRDegradationAdapter:
    depth_source = tmp_path / "depth"
    depth_source.mkdir()
    depth = np.linspace(0.1, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    savemat(depth_source / f"{clean_id}.mat", {"data_obj": depth})
    compat = tmp_path / "compat"
    prepare_depth_compat_tree(depth_source, compat)
    return AgenticIRDegradationAdapter(depth_compat_root=compat)


def _recipe(parameters: tuple[OperatorParameter, ...]) -> PrimaryRecipe:
    return PrimaryRecipe(
        sample_id="A-synthetic-train-0000",
        split="train",
        clean_id="synthetic",
        clean_path=Path("synthetic.png"),
        depth_path=Path("synthetic.mat") if any(p.name == "haze" for p in parameters) else None,
        clean_sha256="0" * 64,
        group="A" if len(parameters) == 2 else "single",
        seed=2027,
        operator_params=parameters,
        raw={},
    )


def test_pair_subset_targets_reuse_operator_seeds_and_are_deterministic(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    canonicalizer = MioIRScaleCanonicalizer()
    rain = OperatorParameter(
        name="rain",
        seed=278319,
        # Actual is provenance for rain; the official RNG sequence is replayed.
        actual={"length": 22, "angle": -4, "value": 71},
    )
    haze = OperatorParameter(
        name="haze",
        seed=99183,
        actual={"A": 0.82, "beta": 1.1},
    )
    recipe = _recipe((rain, haze))
    clean = np.random.RandomState(8).randint(
        0, 256, size=(80, 96, 3), dtype=np.uint8
    )

    first = synthesize_subset_targets(clean, recipe, adapter, canonicalizer)
    second = synthesize_subset_targets(clean, recipe, adapter, canonicalizer)
    for name in (
        "input_rgb",
        "gt_clean_rgb",
        "target_after_i_rgb",
        "target_after_j_rgb",
        "guard_targets",
        "global_severity_targets",
        "presence_target",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name)), name

    only_rain = adapter.apply_sequence(
        clean, (rain,), clean_id="synthetic", capture_traces=False
    )
    only_haze = adapter.apply_sequence(
        clean, (haze,), clean_id="synthetic", capture_traces=False
    )
    assert torch.equal(
        first.target_after_i_rgb,
        bgr_uint8_to_rgb_float(only_haze.output_bgr_uint8),
    )
    assert torch.equal(
        first.target_after_j_rgb,
        bgr_uint8_to_rgb_float(only_rain.output_bgr_uint8),
    )
    rain_id, haze_id = SKILL_TO_ID["rain"], SKILL_TO_ID["haze"]
    assert first.presence_target.sum().item() == 2
    assert first.presence_target[rain_id] == 1
    assert first.presence_target[haze_id] == 1
    assert float(first.guard_targets[rain_id].max()) > 0
    assert float(first.guard_targets[haze_id].max()) > 0
    absent = [index for index in range(8) if index not in {rain_id, haze_id}]
    assert torch.count_nonzero(first.guard_targets[absent]) == 0


def test_low_resolution_subset_target_uses_native_then_float_x4(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    canonicalizer = MioIRScaleCanonicalizer()
    rain = OperatorParameter(
        name="rain",
        seed=1234,
        actual={"length": 20, "angle": 0, "value": 50},
    )
    low_resolution = OperatorParameter(
        name="low resolution",
        seed=5678,
        actual={"scale": 0.25, "model_resize": "AgenticIR/BasicSR imresize x4"},
    )
    recipe = _recipe((rain, low_resolution))
    clean = np.random.RandomState(9).randint(
        0, 256, size=(80, 96, 3), dtype=np.uint8
    )
    result = synthesize_subset_targets(clean, recipe, adapter, canonicalizer)
    native_only_lr = adapter.apply_sequence(
        clean, (low_resolution,), clean_id="synthetic", capture_traces=False
    )
    expected_after_rain = canonicalizer.canonicalize_native_lq(
        native_only_lr.output_bgr_uint8
    )
    assert result.input_rgb.shape == (3, 80, 96)
    assert result.target_after_i_rgb.shape == (3, 80, 96)
    assert torch.equal(result.target_after_i_rgb, expected_after_rain)
    low_id = SKILL_TO_ID["low_resolution"]
    assert result.global_severity_targets[low_id] == 1
    assert torch.all(result.guard_targets[low_id] == 1)

