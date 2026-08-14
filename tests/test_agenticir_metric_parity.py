from __future__ import annotations

import math

import pytest
import torch

from src.metrics.agenticir_official import (
    OFFICIAL_GROUPS,
    aggregate_official_records,
    official_psnr,
    official_ssim,
    quantize_uint8_semantics,
)


def test_quantization_is_uint8_roundtrip_semantics() -> None:
    values = torch.tensor([[[[-0.1, 0.0, 0.5 / 255, 1.2]]]], dtype=torch.float32)
    got = quantize_uint8_semantics(values)
    expected = torch.tensor([[[[0.0, 0.0, 0.0, 1.0]]]], dtype=torch.float32)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_identical_official_metrics_match_pyiqa_caps() -> None:
    image = torch.rand(2, 3, 32, 40, generator=torch.Generator().manual_seed(7))
    psnr = official_psnr(image, image)
    ssim = official_ssim(image, image)
    torch.testing.assert_close(psnr, torch.full_like(psnr, 80.0), rtol=0, atol=1e-6)
    torch.testing.assert_close(ssim, torch.ones_like(ssim), rtol=0, atol=1e-12)


def test_strict_official_aggregation() -> None:
    records = []
    for group_index, combinations in enumerate(OFFICIAL_GROUPS.values()):
        for combination in combinations:
            records.extend(
                [
                    {"combination": combination, "psnr": 20 + group_index, "ssim": 0.8},
                    {"combination": combination, "psnr": 22 + group_index, "ssim": 0.9},
                ]
            )
    result = aggregate_official_records(records, expected_counts={r["combination"]: 2 for r in records})
    assert result["image_count"] == 32
    assert result["groups"]["A"]["psnr"] == pytest.approx(21.0)
    assert result["groups"]["B"]["psnr"] == pytest.approx(22.0)
    assert result["groups"]["C"]["psnr"] == pytest.approx(23.0)
    assert math.isclose(result["weighted_all_images"]["ssim"], 0.85)


def test_aggregation_rejects_missing_combination() -> None:
    with pytest.raises(ValueError, match="combination mismatch"):
        aggregate_official_records([{"combination": "rain+haze", "psnr": 20, "ssim": 0.8}])
