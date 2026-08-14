"""GraphRestore single/Group-A episode Dataset.

Only frozen primary manifests are accepted.  Official degradation synthesis is
performed in BGR uint8, low-resolution inputs are canonicalized in memory, and
all image/guard/subset tensors receive one synchronized crop and augmentation.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import Dataset

from .agenticir_degradations import (
    DEFAULT_AGENTICIR_REPO,
    DEFAULT_MIOIR_REPO,
    AgenticIRDegradationAdapter,
    prepare_depth_compat_tree,
)
from .manifests import SKILLS, PrimaryRecipe, load_primary_manifest
from .samplers import EpisodeRequest
from .scale_canonicalizer import MioIRScaleCanonicalizer
from .subset_targets import SubsetTargets, synthesize_subset_targets


class EpisodeDatasetError(RuntimeError):
    """An episode cannot be generated without violating the data contract."""


def _crop_shape(crop_size: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if crop_size is None:
        return None
    if isinstance(crop_size, bool):
        raise ValueError("crop_size cannot be boolean")
    if isinstance(crop_size, int):
        result = (crop_size, crop_size)
    elif (
        isinstance(crop_size, tuple)
        and len(crop_size) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in crop_size)
    ):
        result = crop_size
    else:
        raise TypeError("crop_size must be None, an integer, or an (H, W) tuple")
    if min(result) <= 0 or result[0] % 4 or result[1] % 4:
        raise ValueError("crop dimensions must be positive multiples of four")
    return result


def _geometry_rng(base_seed: int, sample_id: str, absolute_step: int) -> random.Random:
    payload = (
        f"graphrestore-geometry:{base_seed}:{absolute_step}:{sample_id}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


class GraphRestoreEpisodeDataset(Dataset[dict[str, Any]]):
    """Generate deterministic V7.1 training or full-resolution val episodes.

    Parameters intentionally match the stable training-script integration
    signature.  ``training=False, crop_size=None`` always returns the complete
    primary-val image with no random augmentation.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        training_data_root: str | Path,
        depth_compat_root: str | Path,
        crop_size: int | tuple[int, int] | None,
        training: bool,
        stage: str | int,
        base_seed: int = 2027,
        *,
        agenticir_repo: str | Path = DEFAULT_AGENTICIR_REPO,
        mioir_repo: str | Path = DEFAULT_MIOIR_REPO,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.training_data_root = Path(training_data_root).resolve()
        self.depth_compat_root = Path(depth_compat_root).resolve(strict=False)
        self.crop_size = _crop_shape(crop_size)
        self.training = bool(training)
        self.stage = str(stage).lower().replace("_", "").replace("-", "")
        self.base_seed = int(base_seed)
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        self.agenticir_repo = Path(agenticir_repo).resolve()
        self.mioir_repo = Path(mioir_repo).resolve()
        expected_split = "train" if self.training else "val"
        self.records: tuple[PrimaryRecipe, ...] = load_primary_manifest(
            self.manifest_path,
            self.training_data_root,
            expected_split=expected_split,
            must_exist=True,
        )
        # Prebuild the complete tree in the parent process.  DataLoader workers
        # only read it, eliminating both the old return-path bug and symlink races.
        self.depth_compat_count = prepare_depth_compat_tree(
            self.training_data_root / "depth" / "depth",
            self.depth_compat_root,
        )
        self._step = 0
        self._worker_seed = self.base_seed
        self._worker_generator: torch.Generator | None = None
        self._adapter: AgenticIRDegradationAdapter | None = None
        self._canonicalizer: MioIRScaleCanonicalizer | None = None
        self._init_runtime()

    def _init_runtime(self) -> None:
        self._worker_generator = torch.Generator(device="cpu").manual_seed(
            self._worker_seed
        )
        self._adapter = AgenticIRDegradationAdapter(
            agenticir_repo=self.agenticir_repo,
            mioir_repo=self.mioir_repo,
            depth_compat_root=self.depth_compat_root,
            worker_generator=self._worker_generator,
        )
        self._canonicalizer = MioIRScaleCanonicalizer(self.mioir_repo)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Imported modules and generators should be recreated inside a spawned
        # worker.  The compatibility tree is already complete and read-only.
        state["_adapter"] = None
        state["_canonicalizer"] = None
        state["_worker_generator"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_runtime()

    @property
    def adapter(self) -> AgenticIRDegradationAdapter:
        if self._adapter is None:
            self._init_runtime()
        assert self._adapter is not None
        return self._adapter

    @property
    def canonicalizer(self) -> MioIRScaleCanonicalizer:
        if self._canonicalizer is None:
            self._init_runtime()
        assert self._canonicalizer is not None
        return self._canonicalizer

    def set_worker_seed(self, seed: int) -> None:
        """Set traversal RNG without changing any manifest operator seed."""

        self._worker_seed = int(seed) % 2**32
        self._worker_generator = torch.Generator(device="cpu").manual_seed(
            self._worker_seed
        )
        if self._adapter is not None:
            self._adapter.worker_generator = self._worker_generator

    def set_step(self, step: int) -> None:
        """Set the absolute consumed step used for direct-index train geometry."""

        if step < 0:
            raise ValueError("step must be non-negative")
        self._step = int(step)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_request(
        self, index: int | EpisodeRequest
    ) -> tuple[int, str, int, int, int]:
        if isinstance(index, EpisodeRequest):
            actual_index = index.index
            episode_type = index.episode_type
            active_slot = index.active_slot
            absolute_step = index.absolute_step
            sample_cursor = index.sample_cursor
        else:
            actual_index = int(index)
            record = self.records[actual_index]
            if self.stage in {"0", "stage0"}:
                episode_type = "stage0_restoration"
            elif self.stage in {"1", "stage1"}:
                episode_type = "pair_parallel" if record.is_pair else "single_skill"
            else:
                episode_type = "restoration"
            active_slot = 0 if episode_type == "single_skill" else -1
            absolute_step = self._step
            # Direct indexing has no sampler cursor.  Keep a stable diagnostic
            # value without pretending that the record index is a global
            # traversal position.
            sample_cursor = -1
        if not 0 <= actual_index < len(self.records):
            raise IndexError(actual_index)
        return actual_index, episode_type, active_slot, absolute_step, sample_cursor

    def _geometry(
        self,
        sample_id: str,
        absolute_step: int,
        height: int,
        width: int,
    ) -> tuple[int, int, int, int, bool, bool, int]:
        if self.crop_size is None:
            crop_height, crop_width = height, width
            top = left = 0
        else:
            crop_height, crop_width = self.crop_size
            if crop_height > height or crop_width > width:
                raise EpisodeDatasetError(
                    f"crop {self.crop_size} exceeds image {(height, width)}"
                )
            if self.training:
                rng = _geometry_rng(self.base_seed, sample_id, absolute_step)
                top = 4 * rng.randrange((height - crop_height) // 4 + 1)
                left = 4 * rng.randrange((width - crop_width) // 4 + 1)
            else:
                # Explicit validation crops, if ever requested, are fixed and
                # 4-aligned.  The formal default remains crop_size=None/full.
                top = 4 * ((height - crop_height) // 8)
                left = 4 * ((width - crop_width) // 8)
        if self.training:
            rng = _geometry_rng(
                self.base_seed ^ 0x5EED5EED, sample_id, absolute_step
            )
            horizontal = bool(rng.getrandbits(1))
            vertical = bool(rng.getrandbits(1))
            rotation = rng.randrange(4)
        else:
            horizontal = vertical = False
            rotation = 0
        return top, left, crop_height, crop_width, horizontal, vertical, rotation

    @staticmethod
    def _transform_tensor(
        tensor: torch.Tensor,
        geometry: tuple[int, int, int, int, bool, bool, int],
    ) -> torch.Tensor:
        top, left, height, width, horizontal, vertical, rotation = geometry
        result = tensor[..., top : top + height, left : left + width]
        if horizontal:
            result = result.flip(-1)
        if vertical:
            result = result.flip(-2)
        if rotation:
            result = torch.rot90(result, rotation, dims=(-2, -1))
        return result.contiguous()

    def _active_skills(
        self,
        recipe: PrimaryRecipe,
        episode_type: str,
        active_slot: int,
        subset: SubsetTargets,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        present_ids = recipe.skill_ids
        if episode_type == "single_skill":
            if len(present_ids) != 1:
                raise EpisodeDatasetError("single_skill request selected a pair recipe")
            active_ids = present_ids
        elif episode_type == "pair_isolation":
            if len(present_ids) != 2 or active_slot not in {0, 1}:
                raise EpisodeDatasetError("invalid pair_isolation request")
            active_ids = (present_ids[active_slot],)
        elif episode_type == "pair_parallel":
            if len(present_ids) != 2:
                raise EpisodeDatasetError("pair_parallel request selected a single recipe")
            active_ids = present_ids
        else:
            active_ids = present_ids
        active_mask = torch.zeros(len(SKILLS), dtype=torch.bool)
        active_mask[list(active_ids)] = True
        padded = torch.full((2,), -1, dtype=torch.long)
        padded[: len(active_ids)] = torch.tensor(active_ids, dtype=torch.long)
        return padded, active_mask

    def __getitem__(self, index: int | EpisodeRequest) -> dict[str, Any]:
        (
            actual_index,
            episode_type,
            active_slot,
            absolute_step,
            sample_cursor,
        ) = self._resolve_request(index)
        recipe = self.records[actual_index]
        clean_bgr = cv2.imread(str(recipe.clean_path), cv2.IMREAD_COLOR)
        if clean_bgr is None:
            raise EpisodeDatasetError(f"unreadable clean image: {recipe.clean_path}")
        if clean_bgr.dtype != np.uint8 or clean_bgr.ndim != 3:
            raise EpisodeDatasetError(f"invalid clean image: {recipe.clean_path}")
        height, width = clean_bgr.shape[:2]
        if height % 4 or width % 4:
            raise EpisodeDatasetError(
                f"clean dimensions must be divisible by four: {(height, width)}"
            )
        geometry = self._geometry(
            recipe.sample_id, absolute_step, height, width
        )
        top, left, crop_height, crop_width, hflip, vflip, rotation = geometry
        subset = synthesize_subset_targets(
            clean_bgr,
            recipe,
            self.adapter,
            self.canonicalizer,
            crop_box=(top, left, crop_height, crop_width)
            if self.crop_size is not None
            else None,
        )
        # Explicit training crops were already applied before degradation.
        # Only the synchronized geometric augmentation remains here.
        local_geometry = (
            0,
            0,
            crop_height,
            crop_width,
            hflip,
            vflip,
            rotation,
        )
        transformed = {
            "input": self._transform_tensor(subset.input_rgb, local_geometry),
            "gt_clean": self._transform_tensor(subset.gt_clean_rgb, local_geometry),
            "target_after_i": self._transform_tensor(
                subset.target_after_i_rgb, local_geometry
            ),
            "target_after_j": self._transform_tensor(
                subset.target_after_j_rgb, local_geometry
            ),
            "only_i": self._transform_tensor(subset.only_i_rgb, local_geometry),
            "only_j": self._transform_tensor(subset.only_j_rgb, local_geometry),
            "guards_full": self._transform_tensor(
                subset.guard_targets, local_geometry
            ),
        }
        if episode_type == "pair_isolation":
            target = (
                transformed["target_after_i"]
                if active_slot == 0
                else transformed["target_after_j"]
            )
        else:
            target = transformed["gt_clean"]
        skill_ids, active_mask = self._active_skills(
            recipe, episode_type, active_slot, subset
        )
        guards = transformed["guards_full"]
        guard_height, guard_width = guards.shape[-2] // 4, guards.shape[-1] // 4
        guard_targets = torch_functional.adaptive_avg_pool2d(
            guards, output_size=(guard_height, guard_width)
        )
        present_ids = torch.full((2,), -1, dtype=torch.long)
        present_ids[: len(recipe.skill_ids)] = torch.tensor(
            recipe.skill_ids, dtype=torch.long
        )
        return {
            "input": transformed["input"],
            "x_both": transformed["input"],
            "target": target,
            "gt_clean": transformed["gt_clean"],
            "target_after_i": transformed["target_after_i"],
            "target_after_j": transformed["target_after_j"],
            "only_i": transformed["only_i"],
            "only_j": transformed["only_j"],
            "guard_targets": guard_targets,
            "global_severity_targets": subset.global_severity_targets.clone(),
            "presence_target": subset.presence_target.clone(),
            "dense_guard_mask": subset.dense_guard_mask.clone(),
            "global_guard_mask": subset.global_guard_mask.clone(),
            "active_mask": active_mask,
            "skill_ids": skill_ids,
            "present_skill_ids": present_ids,
            "episode_type": episode_type,
            "sample_id": recipe.sample_id,
            "operator_order": " + ".join(recipe.operator_order),
            "skill_names": tuple((*recipe.skill_names, ""))[:2],
            "group": recipe.group,
            "split": recipe.split,
            "has_pair": recipe.is_pair,
            "contains_low_resolution": recipe.contains_low_resolution,
            "sample_index": actual_index,
            "absolute_step": absolute_step,
            "sample_cursor": sample_cursor,
            "crop_box": torch.tensor(
                [top, left, crop_height, crop_width], dtype=torch.long
            ),
            "augmentation": torch.tensor(
                [int(hflip), int(vflip), rotation], dtype=torch.long
            ),
        }


# Compact alias for scripts that already use the generic name.
EpisodeDataset = GraphRestoreEpisodeDataset
