from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import savemat

from src.data.agenticir_degradations import (
    AgenticIRDegradationAdapter,
    prepare_depth_compat_tree,
    preserved_operator_rng,
)
from src.data.manifests import OperatorParameter, load_primary_manifest

TRAINING_ROOT = Path("/root/autodl-tmp/graph/training_data")
PRIMARY_TRAIN = TRAINING_ROOT / "manifests/primary_train.jsonl"


def _two_parameters_per_operator() -> dict[str, list[tuple[str, OperatorParameter]]]:
    records = load_primary_manifest(PRIMARY_TRAIN, TRAINING_ROOT)
    selected: dict[str, list[tuple[str, OperatorParameter]]] = defaultdict(list)
    for record in records:
        if len(record.operator_params) != 1:
            continue
        parameter = record.operator_params[0]
        if len(selected[parameter.name]) < 2:
            selected[parameter.name].append((record.clean_id, parameter))
        if len(selected) == 8 and all(len(values) == 2 for values in selected.values()):
            break
    assert len(selected) == 8
    assert all(len(values) == 2 for values in selected.values())
    return selected


def _direct_official(
    adapter: AgenticIRDegradationAdapter,
    image: np.ndarray,
    clean_id: str,
    parameter: OperatorParameter,
) -> np.ndarray:
    operators = adapter.operators
    actual = parameter.actual
    with preserved_operator_rng(parameter.seed):
        if parameter.name == "rain":
            result = operators.add_rain(image)
        elif parameter.name == "haze":
            result = operators.add_haze(
                image,
                idx=clean_id,
                depth_dir=adapter.depth_compat_root,
                A=float(actual["A"]),
                beta=float(actual["beta"]),
            )
        elif parameter.name == "motion blur":
            result = operators.add_motion_blur(image)
        elif parameter.name == "low resolution":
            result = operators.lr(image, keep_size=False)
        elif parameter.name == "dark":
            result = operators.darken(
                image,
                darken_type=str(actual["type"]),
                arg=actual["argument"],
            )
        elif parameter.name == "noise":
            noise_type = str(actual["type"])
            key = "sigma" if noise_type == "Gaussian" else "scale"
            result = operators.add_noise(
                image, noise_type=noise_type, arg=float(actual[key])
            )
        elif parameter.name == "defocus blur":
            result = operators.add_defocus_blur(
                image, severity=int(actual["severity"])
            )
        elif parameter.name == "jpeg compression artifact":
            result = operators.add_jpeg_comp_artifacts(
                image, quality_factor=int(actual["quality_factor"])
            )
        else:  # pragma: no cover - selection above is contract-closed.
            raise AssertionError(parameter.name)
    return result


def test_eight_official_single_operators_two_recipes_pixel_exact(tmp_path: Path) -> None:
    selected = _two_parameters_per_operator()
    depth_source = tmp_path / "depth"
    depth_source.mkdir()
    for clean_id, _ in selected["haze"]:
        depth = np.linspace(0.05, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
        savemat(depth_source / f"{clean_id}.mat", {"data_obj": depth})
    compat = tmp_path / "compat"
    prepare_depth_compat_tree(depth_source, compat)
    worker_generator = torch.Generator(device="cpu").manual_seed(991)
    adapter = AgenticIRDegradationAdapter(
        depth_compat_root=compat, worker_generator=worker_generator
    )
    image_rng = np.random.RandomState(41)
    image = image_rng.randint(0, 256, size=(80, 96, 3), dtype=np.uint8)

    for name, parameters in selected.items():
        for clean_id, parameter in parameters:
            actual = _direct_official(adapter, image, clean_id, parameter)
            replay, _ = adapter.apply_operator(
                image, parameter, clean_id=clean_id, capture_trace=True
            )
            assert actual.dtype == np.uint8, name
            assert np.array_equal(actual, replay), (name, parameter.seed)


def test_operator_call_restores_all_rng_states(tmp_path: Path) -> None:
    depth_source = tmp_path / "depth"
    depth_source.mkdir()
    savemat(depth_source / "sample.mat", {"data_obj": np.ones((8, 8), np.float32)})
    compat = tmp_path / "compat"
    prepare_depth_compat_tree(depth_source, compat)
    worker = torch.Generator(device="cpu").manual_seed(123)
    adapter = AgenticIRDegradationAdapter(
        depth_compat_root=compat, worker_generator=worker
    )
    parameter = OperatorParameter(
        name="noise",
        seed=987654,
        actual={"type": "Gaussian", "sigma": 31.25},
    )
    image = np.full((32, 32, 3), 127, dtype=np.uint8)

    random.seed(11)
    np.random.seed(12)
    torch.default_generator.manual_seed(13)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    worker_state = worker.get_state().clone()
    adapter.apply_operator(image, parameter, clean_id="sample")

    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.equal(worker.get_state(), worker_state)


def test_existing_depth_link_always_returns_compat_root(tmp_path: Path) -> None:
    depth_source = tmp_path / "depth"
    depth_source.mkdir()
    savemat(depth_source / "sample.mat", {"data_obj": np.ones((8, 8), np.float32)})
    compat = tmp_path / "compat"
    assert prepare_depth_compat_tree(depth_source, compat) == 1
    assert prepare_depth_compat_tree(depth_source, compat) == 1
    adapter = AgenticIRDegradationAdapter(depth_compat_root=compat)
    parameter = OperatorParameter(
        name="haze",
        seed=7,
        actual={"A": 0.8, "beta": 1.0},
    )
    image = np.full((32, 32, 3), 160, dtype=np.uint8)
    first, _ = adapter.apply_operator(image, parameter, clean_id="sample")
    second, _ = adapter.apply_operator(image, parameter, clean_id="sample")
    assert np.array_equal(first, second)


def test_training_non_haze_replays_official_operator_on_crop_first(
    tmp_path: Path,
) -> None:
    """Rain pixels must come from official crop replay, not full-then-crop."""

    clean_id, parameter = _two_parameters_per_operator()["rain"][0]
    depth_source = tmp_path / "depth"
    depth_source.mkdir()
    # The adapter contract requires a complete non-empty compatibility tree,
    # although this test never invokes haze.
    savemat(depth_source / "placeholder.mat", {"data_obj": np.ones((2, 2))})
    compat = tmp_path / "compat"
    prepare_depth_compat_tree(depth_source, compat)
    adapter = AgenticIRDegradationAdapter(depth_compat_root=compat)
    image = np.random.RandomState(919).randint(
        0, 256, size=(80, 96, 3), dtype=np.uint8
    )
    crop_box = (8, 12, 32, 40)
    top, left, height, width = crop_box
    clean_crop = image[top : top + height, left : left + width].copy()

    expected = _direct_official(adapter, clean_crop, clean_id, parameter)
    replay = adapter.apply_sequence_crop(
        image,
        (parameter,),
        clean_id=clean_id,
        crop_box=crop_box,
        capture_traces=True,
    )
    assert np.array_equal(replay.output_bgr_uint8, expected)

    # Rain generation depends on spatial extent, so this also detects the old
    # synthesize-full-image-then-crop implementation on the frozen recipe.
    full_then_crop = _direct_official(adapter, image, clean_id, parameter)[
        top : top + height, left : left + width
    ]
    assert not np.array_equal(replay.output_bgr_uint8, full_then_crop)


def test_training_haze_uses_full_depth_normalization_then_crops(
    tmp_path: Path,
) -> None:
    clean_id, parameter = _two_parameters_per_operator()["haze"][0]
    depth_source = tmp_path / "depth"
    depth_source.mkdir()
    depth = np.linspace(0.01, 3.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    savemat(depth_source / f"{clean_id}.mat", {"data_obj": depth})
    compat = tmp_path / "compat"
    prepare_depth_compat_tree(depth_source, compat)
    adapter = AgenticIRDegradationAdapter(depth_compat_root=compat)
    image = np.random.RandomState(313).randint(
        0, 256, size=(80, 96, 3), dtype=np.uint8
    )
    crop_box = (8, 12, 32, 40)
    top, left, height, width = crop_box

    full_output, full_trace = adapter.apply_operator(
        image,
        parameter,
        clean_id=clean_id,
        capture_trace=True,
    )
    replay = adapter.apply_sequence_crop(
        image,
        (parameter,),
        clean_id=clean_id,
        crop_box=crop_box,
        capture_traces=True,
    )
    assert np.array_equal(
        replay.output_bgr_uint8,
        full_output[top : top + height, left : left + width],
    )
    assert full_trace.transmission is not None
    assert replay.traces[0].transmission is not None
    assert np.array_equal(
        replay.traces[0].transmission,
        full_trace.transmission[top : top + height, left : left + width],
    )

    # Re-normalizing a cropped depth map would generally differ.  Confirm the
    # chosen nonuniform fixture makes this assertion discriminative.
    full_transmission = full_trace.transmission
    cropped_depth = cv2.resize(
        depth,
        (96, 80),
        interpolation=cv2.INTER_CUBIC,
    )[top : top + height, left : left + width]
    crop_renormalized = np.exp(
        -float(parameter.actual["beta"]) * cropped_depth / cropped_depth.max()
    ).astype(np.float32)
    assert not np.allclose(
        full_transmission[top : top + height, left : left + width],
        crop_renormalized,
        atol=1e-7,
        rtol=0.0,
    )
