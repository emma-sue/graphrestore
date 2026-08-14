"""Thin, deterministic adapter around the locked AgenticIR operators.

The official source file is executed unchanged after its two BasicSR imports
are satisfied by the locked MiOIR source files.  No degradation formula is
copied into GraphRestore.  Each operator runs inside an RNG transaction that
restores Python, NumPy, Torch CPU, and an optional DataLoader worker generator.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import random
import sys
import threading
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np
import torch
from scipy.io import loadmat

from src.utils.hashing import sha256_file

from .manifests import OperatorParameter, normalize_skill_name
from .scale_canonicalizer import (
    LOCKED_MATLAB_FUNCTIONS_SHA256,
    MATLAB_FUNCTIONS_RELATIVE_PATH,
    load_mioir_matlab_functions,
)

DEFAULT_AGENTICIR_REPO = Path(
    os.environ.get(
        "GRAPHRESTORE_AGENTICIR_REPO", "/root/autodl-tmp/graph/upstream/AgenticIR"
    )
)
DEFAULT_MIOIR_REPO = Path(
    os.environ.get("GRAPHRESTORE_MIOIR_REPO", "/root/autodl-tmp/graph/upstream/MiOIR")
)
AGENTICIR_OPERATOR_RELATIVE_PATH = Path("dataset/add_single_degradation.py")
BASICSR_DEGRADATIONS_RELATIVE_PATH = Path("basicsr/data/degradations.py")
LOCKED_AGENTICIR_OPERATOR_SHA256 = (
    "c97450a05acb805e59291a1335a743c77eca3db36f26a444b4033c7f6fe6369c"
)
LOCKED_BASICSR_DEGRADATIONS_SHA256 = (
    "a507295ec9cbe47536bb7530f63ce385fb0ecb0c7b7fbe51b34b5db9d539d2fd"
)

_IMPORT_LOCK = threading.RLock()
_RNG_LOCK = threading.RLock()
_MISSING = object()


class AgenticIRAdapterError(RuntimeError):
    """The official operator path or a recorded recipe cannot be replayed."""


@dataclass
class OperatorTrace:
    """Observable facts from one official operator invocation."""

    name: str
    skill_name: str
    seed: int
    actual: Mapping[str, Any]
    before_bgr_uint8: np.ndarray | None
    after_bgr_uint8: np.ndarray | None
    transmission: np.ndarray | None
    global_severity: float


@dataclass
class AppliedSequence:
    """Result of replaying a frozen official operator sequence."""

    output_bgr_uint8: np.ndarray
    contains_low_resolution: bool
    traces: tuple[OperatorTrace, ...]


def _load_module_from_file(module_name: str, source_path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(module_name, source_path)
    if specification is None or specification.loader is None:
        raise AgenticIRAdapterError(f"cannot import source file: {source_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _temporary_modules(replacements: Mapping[str, ModuleType]) -> Iterator[None]:
    previous = {name: sys.modules.get(name, _MISSING) for name in replacements}
    try:
        sys.modules.update(replacements)
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value  # type: ignore[assignment]


def _minimal_package(name: str) -> ModuleType:
    package = types.ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    return package


@lru_cache(maxsize=4)
def _load_official_operators_cached(
    operator_source_text: str,
    basicsr_source_text: str,
    matlab_source_text: str,
    expected_operator_sha256: str | None,
    expected_basicsr_sha256: str | None,
    expected_matlab_sha256: str | None,
) -> ModuleType:
    operator_source = Path(operator_source_text).resolve()
    basicsr_source = Path(basicsr_source_text).resolve()
    matlab_source = Path(matlab_source_text).resolve()
    identities = (
        (operator_source, expected_operator_sha256, "AgenticIR operator"),
        (basicsr_source, expected_basicsr_sha256, "MiOIR BasicSR degradations"),
        (matlab_source, expected_matlab_sha256, "MiOIR matlab_functions"),
    )
    actual_hashes: dict[Path, str] = {}
    for source, expected, label in identities:
        if not source.is_file():
            raise AgenticIRAdapterError(f"{label} source is missing: {source}")
        actual = sha256_file(source)
        actual_hashes[source] = actual
        if expected is not None and actual != expected:
            raise AgenticIRAdapterError(
                f"{label} SHA256 mismatch: expected {expected}, got {actual}"
            )

    with _IMPORT_LOCK:
        # torchvision removed functional_tensor in newer releases.  The locked
        # BasicSR file imports only rgb_to_grayscale from it; provide the exact
        # in-memory alias without altering either source tree.
        import torchvision.transforms.functional as torchvision_functional

        functional_tensor = types.ModuleType(
            "torchvision.transforms.functional_tensor"
        )
        functional_tensor.rgb_to_grayscale = torchvision_functional.rgb_to_grayscale

        basicsr_pkg = _minimal_package("basicsr")
        basicsr_data_pkg = _minimal_package("basicsr.data")
        basicsr_utils_pkg = _minimal_package("basicsr.utils")
        matlab_module = load_mioir_matlab_functions(
            matlab_source.parents[2], expected_sha256=expected_matlab_sha256
        )

        first_replacements = {
            "torchvision.transforms.functional_tensor": functional_tensor,
            "basicsr": basicsr_pkg,
            "basicsr.data": basicsr_data_pkg,
            "basicsr.utils": basicsr_utils_pkg,
            "basicsr.utils.matlab_functions": matlab_module,
        }
        with _temporary_modules(first_replacements):
            basicsr_module = _load_module_from_file(
                f"graphrestore_basicsr_degradations_{actual_hashes[basicsr_source][:16]}",
                basicsr_source,
            )

        second_replacements = dict(first_replacements)
        second_replacements["basicsr.data.degradations"] = basicsr_module
        with _temporary_modules(second_replacements):
            operator_module = _load_module_from_file(
                f"graphrestore_agenticir_operators_{actual_hashes[operator_source][:16]}",
                operator_source,
            )

    required = (
        "lr",
        "darken",
        "add_noise",
        "add_jpeg_comp_artifacts",
        "add_haze",
        "add_motion_blur",
        "add_defocus_blur",
        "add_rain",
    )
    missing = [name for name in required if not callable(getattr(operator_module, name, None))]
    if missing:
        raise AgenticIRAdapterError(
            f"locked AgenticIR module lacks operators: {', '.join(missing)}"
        )
    return operator_module


def load_official_operators(
    agenticir_repo: str | Path = DEFAULT_AGENTICIR_REPO,
    mioir_repo: str | Path = DEFAULT_MIOIR_REPO,
    *,
    expected_operator_sha256: str | None = LOCKED_AGENTICIR_OPERATOR_SHA256,
    expected_basicsr_sha256: str | None = LOCKED_BASICSR_DEGRADATIONS_SHA256,
    expected_matlab_sha256: str | None = LOCKED_MATLAB_FUNCTIONS_SHA256,
) -> ModuleType:
    """Load unchanged official operators with audited source identities."""

    agenticir = Path(agenticir_repo).resolve()
    mioir = Path(mioir_repo).resolve()
    return _load_official_operators_cached(
        str(agenticir / AGENTICIR_OPERATOR_RELATIVE_PATH),
        str(mioir / BASICSR_DEGRADATIONS_RELATIVE_PATH),
        str(mioir / MATLAB_FUNCTIONS_RELATIVE_PATH),
        expected_operator_sha256,
        expected_basicsr_sha256,
        expected_matlab_sha256,
    )


@contextlib.contextmanager
def preserved_operator_rng(
    seed: int,
    *,
    worker_generator: torch.Generator | None = None,
) -> Iterator[None]:
    """Set independent operator RNGs and restore every caller-visible state."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("operator seed must be a uint32 integer")
    with _RNG_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        worker_state = (
            worker_generator.get_state().clone()
            if worker_generator is not None
            else None
        )
        try:
            random.seed(seed)
            np.random.seed(seed)
            # Seed only the default CPU generator.  No CUDA context is touched.
            torch.default_generator.manual_seed(seed)
            if worker_generator is not None:
                worker_generator.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if worker_generator is not None and worker_state is not None:
                worker_generator.set_state(worker_state)


def prepare_depth_compat_tree(
    depth_source_root: str | Path,
    depth_compat_root: str | Path,
) -> int:
    """Prebuild ``<id>/predict_depth.mat`` links for every MiOIR depth file.

    Existing correct symlinks are accepted.  Regular files, wrong links,
    duplicate IDs, or missing sources fail closed.  The returned path used by
    the adapter is always the compatibility *root*, never an ID subdirectory.
    """

    source_root = Path(depth_source_root).resolve()
    compat_root = Path(depth_compat_root).resolve(strict=False)
    if not source_root.is_dir():
        raise AgenticIRAdapterError(f"depth source root is missing: {source_root}")
    sources = sorted(source_root.glob("*.mat"))
    if not sources:
        raise AgenticIRAdapterError(f"no MiOIR MAT files found in {source_root}")
    ids = [source.stem for source in sources]
    if len(ids) != len(set(ids)):
        raise AgenticIRAdapterError("duplicate MiOIR depth IDs")
    compat_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        destination = compat_root / source.stem / "predict_depth.mat"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise AgenticIRAdapterError(
                    f"wrong depth compatibility link: {destination} -> "
                    f"{os.readlink(destination)}"
                )
            continue
        if destination.exists():
            raise AgenticIRAdapterError(
                f"depth compatibility target is not a symlink: {destination}"
            )
        try:
            destination.symlink_to(source.resolve())
        except FileExistsError:
            # A concurrent preflight may have won the atomic symlink creation.
            if not destination.is_symlink() or destination.resolve() != source.resolve():
                raise AgenticIRAdapterError(
                    f"ambiguous concurrent depth link: {destination}"
                )
    return len(sources)


def _require_bgr_uint8(image: np.ndarray, *, context: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{context}: image must be a NumPy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError(f"{context}: expected BGR uint8 HWC image")
    return np.ascontiguousarray(image)


def _global_severity(parameter: OperatorParameter) -> float:
    actual = parameter.actual
    name = parameter.name
    if name == "noise":
        if actual.get("type") == "Gaussian":
            value = (float(actual["sigma"]) - 20.0) / 30.0
        else:
            value = (float(actual["scale"]) - 1.0) / 2.0
    elif name in {"motion blur", "defocus blur"}:
        value = float(actual["severity"]) / 2.0
    elif name == "jpeg compression artifact":
        value = (30.0 - float(actual["quality_factor"])) / 20.0
    elif name == "low resolution":
        value = 1.0
    else:
        value = 0.0
    return float(np.clip(value, 0.0, 1.0))


class AgenticIRDegradationAdapter:
    """Replay frozen recipes using the unchanged locked official functions."""

    def __init__(
        self,
        *,
        agenticir_repo: str | Path = DEFAULT_AGENTICIR_REPO,
        mioir_repo: str | Path = DEFAULT_MIOIR_REPO,
        depth_compat_root: str | Path,
        worker_generator: torch.Generator | None = None,
    ) -> None:
        self.agenticir_repo = Path(agenticir_repo).resolve()
        self.mioir_repo = Path(mioir_repo).resolve()
        self.depth_compat_root = Path(depth_compat_root).resolve()
        if not self.depth_compat_root.is_dir():
            raise AgenticIRAdapterError(
                f"prebuilt depth compatibility root is missing: {self.depth_compat_root}"
            )
        self.worker_generator = worker_generator
        self.operators = load_official_operators(
            self.agenticir_repo, self.mioir_repo
        )

    def _haze_transmission(
        self, clean_id: str, beta: float, expected_hw: tuple[int, int]
    ) -> np.ndarray:
        path = self.depth_compat_root / clean_id / "predict_depth.mat"
        if not path.is_file():
            raise AgenticIRAdapterError(f"haze depth link is missing: {path}")
        payload = loadmat(path)
        if "data_obj" not in payload:
            raise AgenticIRAdapterError(f"haze MAT lacks data_obj: {path}")
        depth = np.asarray(payload["data_obj"])
        depth = cv2.resize(depth, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        if depth.shape != expected_hw:
            raise AgenticIRAdapterError(
                f"upsampled haze depth {depth.shape} does not match image {expected_hw}"
            )
        maximum = float(depth.max())
        if not np.isfinite(maximum) or maximum <= 0:
            raise AgenticIRAdapterError(f"invalid haze depth maximum in {path}")
        normalized = depth / maximum
        return np.exp(-float(beta) * normalized).astype(np.float32, copy=False)

    def apply_operator(
        self,
        image_bgr_uint8: np.ndarray,
        parameter: OperatorParameter,
        *,
        clean_id: str,
        capture_trace: bool = True,
    ) -> tuple[np.ndarray, OperatorTrace]:
        """Apply one recorded operator and return its optional dense trace."""

        image = _require_bgr_uint8(image_bgr_uint8, context=parameter.name)
        name = parameter.name
        actual = parameter.actual
        normalize_skill_name(name)
        dense = name in {"rain", "haze", "dark"}
        before = image.copy() if capture_trace and dense else None
        transmission: np.ndarray | None = None
        with preserved_operator_rng(
            parameter.seed, worker_generator=self.worker_generator
        ):
            if name == "rain":
                # Do not pass recorded value: doing so skips the official RNG
                # draw and changes the subsequently generated rain pattern.
                output = self.operators.add_rain(image)
            elif name == "haze":
                beta = float(actual["beta"])
                if capture_trace:
                    transmission = self._haze_transmission(
                        clean_id, beta, image.shape[:2]
                    )
                output = self.operators.add_haze(
                    image,
                    idx=clean_id,
                    depth_dir=self.depth_compat_root,
                    A=float(actual["A"]),
                    beta=beta,
                )
            elif name == "motion blur":
                # As for rain, explicit severity would shift the angle draw.
                output = self.operators.add_motion_blur(image)
            elif name == "low resolution":
                output = self.operators.lr(image, keep_size=False)
            elif name == "dark":
                output = self.operators.darken(
                    image,
                    darken_type=str(actual["type"]),
                    arg=actual["argument"],
                )
            elif name == "noise":
                noise_type = str(actual["type"])
                key = "sigma" if noise_type == "Gaussian" else "scale"
                output = self.operators.add_noise(
                    image, noise_type=noise_type, arg=float(actual[key])
                )
            elif name == "defocus blur":
                output = self.operators.add_defocus_blur(
                    image, severity=int(actual["severity"])
                )
            elif name == "jpeg compression artifact":
                output = self.operators.add_jpeg_comp_artifacts(
                    image, quality_factor=int(actual["quality_factor"])
                )
            else:  # normalize_skill_name above makes this defensive only.
                raise AgenticIRAdapterError(f"unsupported operator: {name}")
        result = _require_bgr_uint8(np.asarray(output), context=f"output of {name}")
        after = result.copy() if capture_trace and dense else None
        trace = OperatorTrace(
            name=name,
            skill_name=normalize_skill_name(name),
            seed=parameter.seed,
            actual=dict(actual),
            before_bgr_uint8=before,
            after_bgr_uint8=after,
            transmission=transmission,
            global_severity=_global_severity(parameter),
        )
        return result, trace

    def apply_sequence(
        self,
        image_bgr_uint8: np.ndarray,
        parameters: Sequence[OperatorParameter],
        *,
        clean_id: str,
        capture_traces: bool = True,
    ) -> AppliedSequence:
        """Apply parameters in manifest order; low resolution must be last."""

        if not parameters:
            raise AgenticIRAdapterError("operator sequence is empty")
        current = _require_bgr_uint8(image_bgr_uint8, context="sequence input").copy()
        traces: list[OperatorTrace] = []
        contains_low_resolution = False
        for ordinal, parameter in enumerate(parameters):
            if parameter.name == "low resolution":
                if ordinal != len(parameters) - 1:
                    raise AgenticIRAdapterError(
                        "low resolution must be the last operator in the frozen protocol"
                    )
                contains_low_resolution = True
            current, trace = self.apply_operator(
                current,
                parameter,
                clean_id=clean_id,
                capture_trace=capture_traces,
            )
            traces.append(trace)
        return AppliedSequence(
            output_bgr_uint8=current,
            contains_low_resolution=contains_low_resolution,
            traces=tuple(traces),
        )

    def apply_sequence_crop(
        self,
        full_clean_bgr_uint8: np.ndarray,
        parameters: Sequence[OperatorParameter],
        *,
        clean_id: str,
        crop_box: tuple[int, int, int, int],
        capture_traces: bool = True,
    ) -> AppliedSequence:
        """Replay a training crop while retaining full-depth haze normalization.

        Non-haze operators receive the selected crop directly, matching the
        contract's crop-first training path.  For haze, the current crop is
        pasted into the unchanged full clean canvas and the official function
        is called on that full image; its pixelwise result/transmission is then
        cropped.  Thus the locked full-depth x4 resize and whole-image maximum
        normalization remain exact without reimplementing the haze formula.
        """

        full = _require_bgr_uint8(full_clean_bgr_uint8, context="full clean")
        top, left, height, width = crop_box
        if min(top, left, height, width) < 0 or height <= 0 or width <= 0:
            raise AgenticIRAdapterError(f"invalid crop_box: {crop_box}")
        if any(value % 4 for value in crop_box):
            raise AgenticIRAdapterError(
                f"crop coordinates and dimensions must be multiples of four: {crop_box}"
            )
        if top + height > full.shape[0] or left + width > full.shape[1]:
            raise AgenticIRAdapterError(
                f"crop {crop_box} exceeds full image {full.shape[:2]}"
            )
        if not parameters:
            raise AgenticIRAdapterError("operator sequence is empty")
        current = full[top : top + height, left : left + width].copy()
        traces: list[OperatorTrace] = []
        contains_low_resolution = False
        for ordinal, parameter in enumerate(parameters):
            if parameter.name == "low resolution":
                if ordinal != len(parameters) - 1:
                    raise AgenticIRAdapterError(
                        "low resolution must be the last operator in the frozen protocol"
                    )
                contains_low_resolution = True
            if parameter.name == "haze":
                if current.shape[:2] != (height, width):
                    raise AgenticIRAdapterError(
                        "haze cannot follow a spatial-scale-changing operator"
                    )
                canvas = full.copy()
                canvas[top : top + height, left : left + width] = current
                full_output, full_trace = self.apply_operator(
                    canvas,
                    parameter,
                    clean_id=clean_id,
                    capture_trace=capture_traces,
                )
                current = full_output[top : top + height, left : left + width].copy()
                if capture_traces:
                    before = (
                        None
                        if full_trace.before_bgr_uint8 is None
                        else full_trace.before_bgr_uint8[
                            top : top + height, left : left + width
                        ].copy()
                    )
                    after = (
                        None
                        if full_trace.after_bgr_uint8 is None
                        else full_trace.after_bgr_uint8[
                            top : top + height, left : left + width
                        ].copy()
                    )
                    transmission = (
                        None
                        if full_trace.transmission is None
                        else full_trace.transmission[
                            top : top + height, left : left + width
                        ].copy()
                    )
                    trace = OperatorTrace(
                        name=full_trace.name,
                        skill_name=full_trace.skill_name,
                        seed=full_trace.seed,
                        actual=full_trace.actual,
                        before_bgr_uint8=before,
                        after_bgr_uint8=after,
                        transmission=transmission,
                        global_severity=full_trace.global_severity,
                    )
                else:
                    trace = full_trace
            else:
                current, trace = self.apply_operator(
                    current,
                    parameter,
                    clean_id=clean_id,
                    capture_trace=capture_traces,
                )
            traces.append(trace)
        return AppliedSequence(
            output_bgr_uint8=current,
            contains_low_resolution=contains_low_resolution,
            traces=tuple(traces),
        )


def operator_source_identity(
    agenticir_repo: str | Path = DEFAULT_AGENTICIR_REPO,
    mioir_repo: str | Path = DEFAULT_MIOIR_REPO,
) -> dict[str, dict[str, str]]:
    """Return exact source paths and hashes for protocol reports."""

    agenticir = Path(agenticir_repo).resolve()
    mioir = Path(mioir_repo).resolve()
    paths = {
        "agenticir_add_single_degradation": agenticir
        / AGENTICIR_OPERATOR_RELATIVE_PATH,
        "mioir_basicsr_degradations": mioir / BASICSR_DEGRADATIONS_RELATIVE_PATH,
        "mioir_matlab_functions": mioir / MATLAB_FUNCTIONS_RELATIVE_PATH,
    }
    return {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in paths.items()
    }
