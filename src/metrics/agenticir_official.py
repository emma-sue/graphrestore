"""Fast PSNR/SSIM with AgenticIR + pyiqa 0.1.10 semantics.

The implementation is intentionally limited to the two full-reference metrics
used for selection and reporting.  It mirrors pyiqa 0.1.10's ``psnr_arch.py``
and ``ssim_arch.py`` without importing AgenticIR's module-level ``Scorer``,
which eagerly constructs four unrelated heavy IQA models.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


OFFICIAL_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    {
        "A": (
            "rain+haze",
            "motion blur+low resolution",
            "dark+noise",
            "defocus blur+jpeg compression artifact",
            "noise+jpeg compression artifact",
            "rain+low resolution",
            "motion blur+dark",
            "defocus blur+haze",
        ),
        "B": (
            "motion blur+jpeg compression artifact",
            "haze+noise",
            "defocus blur+low resolution",
            "rain+dark",
        ),
        "C": (
            "haze+motion blur+low resolution",
            "rain+noise+low resolution",
            "dark+defocus blur+jpeg compression artifact",
            "motion blur+defocus blur+noise",
        ),
    }
)


@dataclass(frozen=True)
class OfficialMetricResult:
    """Per-image metric tensors, one value per batch item."""

    psnr: Tensor
    ssim: Tensor


def _require_images(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}")
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("expected RGB BCHW tensors with three channels")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("metrics expect floating point RGB tensors")


def quantize_uint8_semantics(image: Tensor) -> Tensor:
    """Clamp/round exactly as PNG evaluation, returned in float [0, 1]."""

    if not image.is_floating_point():
        raise TypeError("image must be floating point")
    return image.clamp(0.0, 1.0).mul(255.0).round().div(255.0)


def _rgb_to_yiq_y(image: Tensor, out_data_range: float) -> Tensor:
    # pyiqa 0.1.10 rgb2yiq uses the first column of this transposed matrix.
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True) * out_data_range


def official_psnr(prediction: Tensor, target: Tensor, *, quantize: bool = True) -> Tensor:
    """AgenticIR official RGB PSNR (pyiqa 0.1.10, eps=1e-8)."""

    _require_images(prediction, target)
    if quantize:
        prediction = quantize_uint8_semantics(prediction)
        target = quantize_uint8_semantics(target)
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + 1.0e-8))


def _gaussian_window(channels: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    # Match pyiqa fspecial: NumPy float64 calculation -> float32 -> input dtype.
    radius = 5.0
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * 1.5 * 1.5))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    kernel /= kernel.sum()
    return torch.from_numpy(kernel).float().repeat(channels, 1, 1, 1).to(device=device, dtype=dtype)


def _ssim_from_preprocessed(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float,
) -> Tensor:
    if prediction.shape[-2] < 11 or prediction.shape[-1] < 11:
        raise ValueError("SSIM requires height and width >= 11")
    window = _gaussian_window(
        prediction.shape[1], device=prediction.device, dtype=prediction.dtype
    )
    mu_x = F.conv2d(prediction, window, groups=prediction.shape[1])
    mu_y = F.conv2d(target, window, groups=target.shape[1])
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    var_x = F.conv2d(prediction.square(), window, groups=prediction.shape[1]) - mu_x_sq
    var_y = F.conv2d(target.square(), window, groups=target.shape[1]) - mu_y_sq
    cov_xy = F.conv2d(prediction * target, window, groups=prediction.shape[1]) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    contrast = ((2.0 * cov_xy + c2) / (var_x + var_y + c2)).relu()
    similarity = ((2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)) * contrast
    return similarity.mean(dim=(1, 2, 3))


def _stable_train_ssim_from_preprocessed(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float,
) -> Tensor:
    """SSIM's equivalent, cancellation-resistant form for FP32 training."""

    if prediction.shape[-2] < 11 or prediction.shape[-1] < 11:
        raise ValueError("SSIM requires height and width >= 11")
    window = _gaussian_window(
        prediction.shape[1], device=prediction.device, dtype=prediction.dtype
    )

    # A per-image/channel translation leaves variance and covariance unchanged,
    # while keeping the squared terms away from cancellation around common
    # image offsets such as 0.5.
    prediction_offset = prediction.mean(dim=(-2, -1), keepdim=True)
    target_offset = target.mean(dim=(-2, -1), keepdim=True)
    prediction_centered = prediction - prediction_offset
    target_centered = target - target_offset
    mu_x_centered = F.conv2d(
        prediction_centered, window, groups=prediction.shape[1]
    )
    mu_y_centered = F.conv2d(target_centered, window, groups=target.shape[1])
    mu_x = mu_x_centered + prediction_offset
    mu_y = mu_y_centered + target_offset
    var_x = (
        F.conv2d(
            prediction_centered.square(), window, groups=prediction.shape[1]
        )
        - mu_x_centered.square()
    ).clamp_min(0.0)
    var_y = (
        F.conv2d(target_centered.square(), window, groups=target.shape[1])
        - mu_y_centered.square()
    ).clamp_min(0.0)

    # Var(x-y) = Var(x) + Var(y) - 2 Cov(x,y).  Computing the left side
    # directly avoids a fragile covariance subtraction.  These equivalent
    # difference forms make both SSIM factors at most one without clamping the
    # final score or suppressing its gradients.
    delta_centered = prediction_centered - target_centered
    mu_delta_centered = mu_x_centered - mu_y_centered
    var_delta = (
        F.conv2d(delta_centered.square(), window, groups=prediction.shape[1])
        - mu_delta_centered.square()
    ).clamp_min(0.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    luminance = 1.0 - (mu_x - mu_y).square() / (
        mu_x.square() + mu_y.square() + c1
    )
    contrast = (1.0 - var_delta / (var_x + var_y + c2)).relu()
    return (luminance * contrast).mean(dim=(1, 2, 3))


def official_ssim(prediction: Tensor, target: Tensor, *, quantize: bool = True) -> Tensor:
    """AgenticIR official Y-channel SSIM with no downsampling or crop."""

    _require_images(prediction, target)
    if quantize:
        prediction = quantize_uint8_semantics(prediction)
        target = quantize_uint8_semantics(target)
    # pyiqa first converts RGB [0,1] to YIQ Y * 255 and differentiably rounds,
    # then performs the metric in float64 with an 11x11/1.5 Gaussian window.
    prediction_y = _rgb_to_yiq_y(prediction, 255.0)
    target_y = _rgb_to_yiq_y(target, 255.0)
    prediction_y = prediction_y.round().to(torch.float64)
    target_y = target_y.round().to(torch.float64)
    return _ssim_from_preprocessed(prediction_y, target_y, data_range=255.0)


def train_ssim_y(prediction: Tensor, target: Tensor) -> Tensor:
    """Differentiable non-quantized Y-channel SSIM for training loss."""

    _require_images(prediction, target)
    # This function is called from inside the BF16 model-forward autocast
    # regions in Stage0/Stage1/Stage4.  Merely converting the inputs to FP32 is
    # insufficient: autocast would cast the subsequent conv2d moments back to
    # BF16, where cancellation in E[x^2] - E[x]^2 can yield invalid SSIM (even
    # values above one) and therefore a negative training loss.  Keep the
    # complete differentiable metric path in FP32; ``Tensor.float`` preserves
    # the autograd connection to a BF16 model output.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction_y = _rgb_to_yiq_y(prediction.float(), 1.0)
        target_y = _rgb_to_yiq_y(target.float(), 1.0)
        return _stable_train_ssim_from_preprocessed(
            prediction_y, target_y, data_range=1.0
        )


def official_psnr_ssim(
    prediction: Tensor,
    target: Tensor,
    *,
    quantize: bool = True,
) -> OfficialMetricResult:
    return OfficialMetricResult(
        psnr=official_psnr(prediction, target, quantize=quantize),
        ssim=official_ssim(prediction, target, quantize=quantize),
    )


def aggregate_official_records(
    records: Iterable[Mapping[str, object]],
    *,
    required_combinations: Sequence[str] | None = None,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Strict per-image -> combination -> equal-weight group aggregation.

    Each record must contain ``combination``, ``psnr`` and ``ssim``. Missing or
    unexpected combinations fail closed instead of reproducing AgenticIR's
    silent mean-over-whatever-is-present behavior.
    """

    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"psnr": [], "ssim": []})
    for row in records:
        try:
            combination = str(row["combination"])
            psnr = float(row["psnr"])
            ssim = float(row["ssim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid metric record: {row!r}") from exc
        if not math.isfinite(psnr) or not math.isfinite(ssim):
            raise ValueError(f"non-finite metric for {combination}")
        buckets[combination]["psnr"].append(psnr)
        buckets[combination]["ssim"].append(ssim)

    required = tuple(required_combinations or [task for tasks in OFFICIAL_GROUPS.values() for task in tasks])
    actual = set(buckets)
    missing = set(required) - actual
    unexpected = actual - set(required)
    if missing or unexpected:
        raise ValueError(f"combination mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if expected_counts is not None:
        for combination in required:
            wanted = int(expected_counts[combination])
            got = len(buckets[combination]["psnr"])
            if got != wanted:
                raise ValueError(f"{combination}: expected {wanted} images, got {got}")

    combination_means: OrderedDict[str, dict[str, float | int]] = OrderedDict()
    for combination in required:
        values = buckets[combination]
        combination_means[combination] = {
            "count": len(values["psnr"]),
            "psnr": math.fsum(values["psnr"]) / len(values["psnr"]),
            "ssim": math.fsum(values["ssim"]) / len(values["ssim"]),
        }

    group_means: OrderedDict[str, dict[str, float]] = OrderedDict()
    for group, combinations in OFFICIAL_GROUPS.items():
        selected = [name for name in combinations if name in combination_means]
        if not selected:
            continue
        if len(selected) != len(combinations) and set(required).issuperset(combinations):
            raise ValueError(f"incomplete official group {group}")
        group_means[group] = {
            "psnr": math.fsum(float(combination_means[name]["psnr"]) for name in selected)
            / len(selected),
            "ssim": math.fsum(float(combination_means[name]["ssim"]) for name in selected)
            / len(selected),
        }

    image_count = sum(int(row["count"]) for row in combination_means.values())
    weighted = {
        "psnr": math.fsum(
            float(row["psnr"]) * int(row["count"]) for row in combination_means.values()
        )
        / image_count,
        "ssim": math.fsum(
            float(row["ssim"]) * int(row["count"]) for row in combination_means.values()
        )
        / image_count,
    }
    return {
        "combinations": combination_means,
        "groups": group_means,
        "weighted_all_images": weighted,
        "image_count": image_count,
    }
