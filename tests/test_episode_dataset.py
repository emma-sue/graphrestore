from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import savemat

from src.data.episode_dataset import GraphRestoreEpisodeDataset
from src.data.manifests import SKILL_TO_ID
from src.data.samplers import EpisodeRequest


def _make_training_root(tmp_path: Path, split: str) -> tuple[Path, Path]:
    root = tmp_path / "training_data"
    clean_root = root / "source_clean/mioir_gt/GT"
    depth_root = root / "depth/depth"
    clean_root.mkdir(parents=True)
    depth_root.mkdir(parents=True)
    clean_id = "sample"
    y, x = np.mgrid[:80, :96]
    clean = np.stack(
        ((x * 5 + y) % 256, (x + y * 7) % 256, (x * 3 + y * 2) % 256),
        axis=2,
    ).astype(np.uint8)
    assert cv2.imwrite(str(clean_root / f"{clean_id}.png"), clean)
    depth = np.linspace(0.1, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    savemat(depth_root / f"{clean_id}.mat", {"data_obj": depth})
    row = {
        "canonical_resize": None,
        "clean_id": clean_id,
        "clean_path": f"source_clean/mioir_gt/GT/{clean_id}.png",
        "clean_sha256": "1" * 64,
        "degradations": ["rain", "haze"],
        "depth_path": f"depth/depth/{clean_id}.mat",
        "group": "A",
        "lq_model_path": None,
        "lq_native_path": None,
        "native_scale": None,
        "operator_order": ["rain", "haze"],
        "operator_params": [
            {
                "actual": {"angle": 0, "length": 25, "value": 70},
                "name": "rain",
                "seed": 12345,
            },
            {
                "actual": {"A": 0.85, "beta": 1.2},
                "name": "haze",
                "seed": 67890,
            },
        ],
        "sample_id": f"A-synthetic-{split}-0000",
        "seed": 2027,
        "source": "agenticir_official",
        "split": split,
    }
    manifest = root / f"primary_{split}.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root, manifest


def test_pair_isolation_episode_has_aligned_crop_guards_and_target(
    tmp_path: Path,
) -> None:
    root, manifest = _make_training_root(tmp_path, "train")
    dataset = GraphRestoreEpisodeDataset(
        manifest,
        root,
        tmp_path / "compat",
        32,
        True,
        "stage1",
    )
    request = EpisodeRequest(
        index=0,
        episode_type="pair_isolation",
        active_slot=0,
        absolute_step=91,
    )
    first = dataset[request]
    second = dataset[request]
    assert first["input"].shape == (3, 32, 32)
    assert first["target"].shape == (3, 32, 32)
    assert first["guard_targets"].shape == (8, 8, 8)
    assert first["presence_target"].shape == (8,)
    assert first["active_mask"].shape == (8,)
    assert torch.equal(first["target"], first["target_after_i"])
    assert first["skill_ids"].tolist() == [SKILL_TO_ID["rain"], -1]
    assert first["active_mask"].sum().item() == 1
    assert first["presence_target"].sum().item() == 2
    for key in (
        "input",
        "target",
        "gt_clean",
        "target_after_i",
        "target_after_j",
        "guard_targets",
        "crop_box",
        "augmentation",
    ):
        assert torch.equal(first[key], second[key]), key


def test_primary_val_none_crop_is_full_resolution_and_unaugmented(
    tmp_path: Path,
) -> None:
    root, manifest = _make_training_root(tmp_path, "val")
    dataset = GraphRestoreEpisodeDataset(
        manifest,
        root,
        tmp_path / "compat",
        None,
        False,
        "stage0",
    )
    sample = dataset[0]
    assert sample["input"].shape == (3, 80, 96)
    assert sample["target"].shape == (3, 80, 96)
    assert sample["guard_targets"].shape == (8, 20, 24)
    assert sample["crop_box"].tolist() == [0, 0, 80, 96]
    assert sample["augmentation"].tolist() == [0, 0, 0]

