from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.net import (
    DEFAULT_EXPANDED_MISSING_PREFIXES,
    GraphRestore,
    MiOStageA,
    load_parent_backbone,
)


PARENT = Path(
    "/root/autodl-tmp/aaa/provir/artifacts/checkpoints/stage_a/final_backbone.ckpt"
)


@pytest.fixture(scope="module")
def parent_payload():
    if not PARENT.is_file():
        pytest.skip("contract-bound parent checkpoint is unavailable")
    return torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def test_parent_strictly_matches_all_495_pure_host_keys(parent_payload) -> None:
    model = MiOStageA()
    report = load_parent_backbone(model, parent_payload)
    assert report.source_tensor_count == 495
    assert report.loaded_count == 495
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert report.shape_mismatches == ()


def test_expanded_load_only_leaves_whitelisted_new_modules(parent_payload) -> None:
    model = GraphRestore()
    report = load_parent_backbone(
        model,
        parent_payload,
        allowed_missing_prefixes=DEFAULT_EXPANDED_MISSING_PREFIXES,
    )
    assert report.loaded_count == 495
    assert report.missing_keys
    assert all(
        any(key.startswith(prefix) for prefix in DEFAULT_EXPANDED_MISSING_PREFIXES)
        for key in report.missing_keys
    )
