"""AgenticIR/MiOIR native-low-resolution canonicalization.

The formal model input is produced from the official native uint8 PNG in
memory.  It uses the exact locked MiOIR ``matlab_functions.imresize`` source,
returns RGB float CHW, and intentionally performs no second uint8 rounding.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from src.utils.hashing import sha256_file

DEFAULT_MIOIR_REPO = Path(
    os.environ.get("GRAPHRESTORE_MIOIR_REPO", "/root/autodl-tmp/graph/upstream/MiOIR")
)
MATLAB_FUNCTIONS_RELATIVE_PATH = Path("basicsr/utils/matlab_functions.py")
LOCKED_MATLAB_FUNCTIONS_SHA256 = (
    "29a3a3d209ce15724202bfb01415e5d4e574e7b853090551a7938c7b78ec4975"
)
_IMPORT_LOCK = threading.RLock()


class ScaleCanonicalizationError(RuntimeError):
    """The locked BasicSR canonicalization path cannot be used safely."""


@lru_cache(maxsize=4)
def _load_matlab_functions_cached(
    source_path_text: str, expected_sha256: str | None
) -> ModuleType:
    source_path = Path(source_path_text).resolve()
    if not source_path.is_file():
        raise ScaleCanonicalizationError(
            f"locked MiOIR matlab_functions.py is missing: {source_path}"
        )
    actual_sha256 = sha256_file(source_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ScaleCanonicalizationError(
            "MiOIR matlab_functions.py identity mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    module_name = f"graphrestore_mioir_matlab_{actual_sha256[:16]}"
    specification = importlib.util.spec_from_file_location(module_name, source_path)
    if specification is None or specification.loader is None:
        raise ScaleCanonicalizationError(f"cannot import {source_path}")
    module = importlib.util.module_from_spec(specification)
    # The source is a standalone math file (stdlib + NumPy + Torch); loading it
    # by path avoids importing BasicSR's registries or unrelated model code.
    specification.loader.exec_module(module)
    if not callable(getattr(module, "imresize", None)):
        raise ScaleCanonicalizationError(f"imresize is absent from {source_path}")
    return module


def load_mioir_matlab_functions(
    mioir_repo: str | Path = DEFAULT_MIOIR_REPO,
    *,
    expected_sha256: str | None = LOCKED_MATLAB_FUNCTIONS_SHA256,
) -> ModuleType:
    """Load the locked standalone MiOIR resize implementation by file path."""

    source = Path(mioir_repo).resolve() / MATLAB_FUNCTIONS_RELATIVE_PATH
    with _IMPORT_LOCK:
        return _load_matlab_functions_cached(str(source), expected_sha256)


def _validate_native(native_bgr_uint8: np.ndarray, scale: int) -> np.ndarray:
    if not isinstance(native_bgr_uint8, np.ndarray):
        raise TypeError("native_bgr_uint8 must be a NumPy array")
    if native_bgr_uint8.dtype != np.uint8:
        raise TypeError(
            f"native_bgr_uint8 must have dtype uint8, got {native_bgr_uint8.dtype}"
        )
    if native_bgr_uint8.ndim != 3 or native_bgr_uint8.shape[2] != 3:
        raise ValueError(
            "native_bgr_uint8 must have HWC shape with exactly three channels"
        )
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    return np.ascontiguousarray(native_bgr_uint8)


class MioIRScaleCanonicalizer:
    """Callable object binding canonicalization to an audited MiOIR source."""

    def __init__(
        self,
        mioir_repo: str | Path = DEFAULT_MIOIR_REPO,
        *,
        expected_sha256: str | None = LOCKED_MATLAB_FUNCTIONS_SHA256,
    ) -> None:
        self.mioir_repo = Path(mioir_repo).resolve()
        self.source_path = self.mioir_repo / MATLAB_FUNCTIONS_RELATIVE_PATH
        self.expected_sha256 = expected_sha256
        self.source_sha256 = sha256_file(self.source_path)
        self._module = load_mioir_matlab_functions(
            self.mioir_repo, expected_sha256=expected_sha256
        )

    @torch.no_grad()
    def canonicalize_native_lq(
        self, native_bgr_uint8: np.ndarray, scale: int = 4
    ) -> torch.Tensor:
        """Return official scorer-equivalent RGB float CHW without rounding."""

        native = _validate_native(native_bgr_uint8, scale)
        # AgenticIR scorer does cv2 BGR->RGB before calling imresize.  Channel
        # reversal is expressed without cv2 so this function has no OpenCV
        # interpolation path and cannot accidentally use INTER_CUBIC.
        rgb = np.ascontiguousarray(native[:, :, ::-1])
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0)
        resized = self._module.imresize(tensor, scale=scale)
        if not isinstance(resized, torch.Tensor):
            raise ScaleCanonicalizationError("locked imresize returned a non-tensor")
        expected_shape = (3, native.shape[0] * scale, native.shape[1] * scale)
        if tuple(resized.shape) != expected_shape:
            raise ScaleCanonicalizationError(
                f"canonical shape mismatch: expected {expected_shape}, "
                f"got {tuple(resized.shape)}"
            )
        return resized.clamp_(0.0, 1.0).contiguous()

    @torch.no_grad()
    def canonicalize_native_bgr(
        self, native_bgr_uint8: np.ndarray, scale: int = 4
    ) -> torch.Tensor:
        """Return BGR float CHW for internal operator-domain bookkeeping."""

        return self.canonicalize_native_lq(native_bgr_uint8, scale=scale).flip(0).contiguous()


@lru_cache(maxsize=1)
def _default_canonicalizer() -> MioIRScaleCanonicalizer:
    return MioIRScaleCanonicalizer(DEFAULT_MIOIR_REPO)


def canonicalize_native_lq(
    native_bgr_uint8: np.ndarray, scale: int = 4
) -> torch.Tensor:
    """Stable public wrapper: native BGR uint8 -> RGB float CHW.

    No output rounding or uint8 conversion is performed after the locked
    BasicSR/MiOIR x4 resize.
    """

    return _default_canonicalizer().canonicalize_native_lq(
        native_bgr_uint8, scale=scale
    )


def load_agenticir_online_canonical_input(
    row: Mapping[str, Any],
    *,
    canonicalizer: MioIRScaleCanonicalizer | None = None,
) -> torch.Tensor:
    """Load exactly ``native_lq_path`` and return the formal RGB float input.

    The derived online manifest is deliberately fail-closed: a non-null legacy
    canonical path, mismatched ``input_path``, or inconsistent scale metadata
    is rejected before any image is read.  OpenCV is used only for lossless PNG
    decoding into the official BGR uint8 operator domain; interpolation is
    exclusively the locked MiOIR/BasicSR implementation above.
    """

    if row.get("input_mode") != "agenticir_online_canonical":
        raise ScaleCanonicalizationError(
            "input_mode must be 'agenticir_online_canonical'"
        )
    for legacy_key in ("canonical_lq_path", "legacy_opencv_canonical_lq_path"):
        if row.get(legacy_key) is not None:
            raise ScaleCanonicalizationError(
                f"formal online input forbids non-null {legacy_key}"
            )
    native_text = row.get("native_lq_path")
    if not isinstance(native_text, str) or not native_text:
        raise ScaleCanonicalizationError("native_lq_path must be a non-empty string")
    if row.get("input_path") != native_text:
        raise ScaleCanonicalizationError("input_path must equal native_lq_path")
    native_path = Path(native_text)
    if not native_path.is_absolute():
        raise ScaleCanonicalizationError("native_lq_path must be absolute")
    contains_low_resolution = row.get("contains_low_resolution")
    if not isinstance(contains_low_resolution, bool):
        raise ScaleCanonicalizationError(
            "contains_low_resolution must be a boolean"
        )
    expected_native_scale = 0.25 if contains_low_resolution else 1.0
    expected_resize = 4 if contains_low_resolution else 1
    if row.get("native_scale") != expected_native_scale:
        raise ScaleCanonicalizationError(
            "native_scale disagrees with contains_low_resolution"
        )
    if row.get("online_scale_factor") != expected_resize:
        raise ScaleCanonicalizationError(
            "online_scale_factor disagrees with contains_low_resolution"
        )
    native = cv2.imread(str(native_path), cv2.IMREAD_COLOR)
    if native is None:
        raise ScaleCanonicalizationError(f"unreadable native LQ: {native_path}")
    native = _validate_native(native, 1)
    if contains_low_resolution:
        implementation = canonicalizer or _default_canonicalizer()
        return implementation.canonicalize_native_lq(native, scale=4)
    return bgr_uint8_to_rgb_float(native)


def bgr_uint8_to_rgb_float(image_bgr_uint8: np.ndarray) -> torch.Tensor:
    """Convert same-size official BGR uint8 output to RGB float CHW."""

    image = _validate_native(image_bgr_uint8, 1)
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0)


def bgr_uint8_to_bgr_float(image_bgr_uint8: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 HWC to BGR float CHW without changing channels."""

    image = _validate_native(image_bgr_uint8, 1)
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)


def canonicalizer_identity(
    mioir_repo: str | Path = DEFAULT_MIOIR_REPO,
) -> dict[str, str]:
    """Return provenance fields suitable for reports and derived manifests."""

    source = Path(mioir_repo).resolve() / MATLAB_FUNCTIONS_RELATIVE_PATH
    return {
        "implementation": str(source),
        "sha256": sha256_file(source),
        "operation": "native BGR uint8 -> RGB float -> MiOIR imresize x4 -> clamp",
        "requantized_after_resize": "false",
    }
