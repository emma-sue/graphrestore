"""Contract-bound Stage4 end-to-end GraphRestore training.

Stage4 is the only training stage that follows the model's discrete program
trajectory.  This module keeps that discrete decision honest: each sample's
partial-order graph is compiled exactly once at ``t=0`` and later rounds only
refresh presence, spatial guards, and stop.  Restoration gradients flow through
the selected executor path; relation logits receive their explicit planner
supervision and are never advertised as differentiable through the compiler.

Only frozen ``primary_train``/``primary_val`` recipes are consumed.  The
counterfactual episodes below are views of those recipes, not a new data source.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from src.data.episode_dataset import GraphRestoreEpisodeDataset
from src.data.manifests import SKILLS, task_buckets
from src.data.samplers import EpisodeRequest
from src.losses.guard_losses import guard_supervision_loss
from src.losses.planner_losses import (
    PlannerLossBreakdown,
    focal_binary_cross_entropy,
    planner_loss,
)
from src.metrics.agenticir_official import official_psnr_ssim, train_ssim_y
from src.net.graph_compiler import CompiledGraph, PAIR_TO_ROW
from src.net.graphrestore import GraphRestore, ProgramGraphState
from src.net.program_planner import PAIR_INDICES, PlannerOutput
from src.net.restormer_blocks import crop_to_shape, pad_to_multiple
from src.net.skill_adapter import SKILL_TO_INDEX
from src.training.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    checkpoint_payload,
    load_checkpoint,
    restore_rng_state,
    unwrap_model,
)
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import WarmupCosineScheduler
from src.training.provenance import semantic_source_hashes
from src.training.selection import ValidationScore
from src.utils.git import git_commit
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import iter_jsonl, load_json, utc_now_iso


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
STAGE4_SCHEMA = "graphrestore-stage4-runtime-v1"
STAGE4_CHECKPOINT_STAGE = "stage4"
STAGE3_APPROVAL_SCHEMA = "graphrestore-stage3-approval-v1"
EPISODE_TYPES = (
    "single_restoration",
    "group_a_pair_restoration",
    "clean_misuse",
    "wrong_skill",
)
COUNTERFACTUAL_TYPES = frozenset({"clean_misuse", "wrong_skill"})


class Stage4ContractError(RuntimeError):
    """A requested action would diverge from the frozen Stage4 contract."""


def _expect(config: Mapping[str, Any], path: Sequence[str], expected: Any) -> None:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise Stage4ContractError(f"missing Stage4 config key: {'.'.join(path)}")
        value = value[key]
    if value != expected:
        raise Stage4ContractError(
            f"Stage4 config drift at {'.'.join(path)}: "
            f"expected {expected!r}, got {value!r}"
        )


def validate_stage4_config(config: Mapping[str, Any]) -> None:
    """Fail closed on every Stage4 scientific/data/optimizer constant."""

    locked: tuple[tuple[tuple[str, ...], Any], ...] = (
        (("schema_version",), "1.0"),
        (("contract_version",), "GraphRestore-V7.1"),
        (("protocol_id",), PROTOCOL_ID),
        (("stage",), "stage4"),
        (("seed",), 2027),
        (("skills", "ordered_names"), list(SKILLS)),
        (("skills", "maximum_active"), 3),
        (("skills", "allow_skill_reentry"), False),
        (("skills", "max_calls_per_skill"), 1),
        (("program", "compile_relations_once_at_t0"), True),
        (("program", "delete_executed_nodes_after_level"), True),
        (("program", "insert_late_skills_into_frozen_dag"), False),
        (("program", "reencode_current_state_each_round"), True),
        (("program", "update_presence_guard_stop_each_round"), True),
        (("program", "kmax_train"), 2),
        (("program", "kmax_test"), 3),
        (("data", "allowed_groups"), ["single", "A"]),
        (("data", "forbidden_groups"), ["B", "C"]),
        (("data", "sampling", "single_restoration"), 0.20),
        (("data", "sampling", "group_a_pair_restoration"), 0.70),
        (("data", "sampling", "clean_misuse"), 0.05),
        (("data", "sampling", "wrong_skill_misuse"), 0.05),
        (("data", "group_a_sampling"), "uniform_8_combinations"),
        (("data", "single_sampling"), "uniform_8_classes"),
        (("data", "wrong_skill_pair_sampling"), "uniform_i_not_equal_j"),
        (("data", "crop_size"), 160),
        (("data", "minimum_crop_after_oom"), 128),
        (("data", "micro_batch_candidates"), [2, 1]),
        (("data", "effective_batch_size"), 4),
        (("model", "frozen"), ["encoder_level1", "encoder_level2"]),
        (("model", "discrete_graph_gradient_claim"), "forbidden"),
        (("teacher_forcing", "preserve_written_discontinuity_at_step12000"), True),
        (("training", "max_steps"), 40_000),
        (("training", "intermediate_levels_train_max"), 2),
        (("optimization", "optimizer"), "AdamW"),
        (("optimization", "betas"), [0.9, 0.999]),
        (("optimization", "weight_decay"), 1.0e-4),
        (("optimization", "weight_decay_norm_bias"), 0.0),
        (("optimization", "learning_rates", "planner"), 5.0e-5),
        (("optimization", "learning_rates", "skill_adapters_and_mixers"), 3.0e-5),
        (("optimization", "learning_rates", "decoder_refinement_rgb_head"), 1.0e-5),
        (("optimization", "learning_rates", "encoder_level3_level4"), 2.0e-6),
        (("optimization", "warmup_steps"), 800),
        (("optimization", "scheduler"), "cosine"),
        (("optimization", "min_lr"), 5.0e-7),
        (("optimization", "gradient_clip_norm"), 0.5),
        (("loss", "ordinary", "final_charbonnier_weight"), 1.0),
        (("loss", "ordinary", "intermediate_subset_charbonnier_weight"), 0.30),
        (("loss", "ordinary", "final_ssim_weight", "start"), 0.0),
        (("loss", "ordinary", "final_ssim_weight", "end"), 0.05),
        (("loss", "ordinary", "final_ssim_weight", "ramp_end_step"), 8000),
        (("loss", "counterfactual", "identity_charbonnier_weight"), 1.0),
        (("loss", "counterfactual", "identity_ssim_weight"), 0.05),
        (("loss", "planner_total_weight"), 0.05),
        (("loss", "training_quantization"), False),
        (("loss", "hard_clamp_forward"), False),
        (("runtime", "amp_dtype"), "bf16"),
        (("runtime", "tf32"), True),
        (("runtime", "channels_last"), False),
        (("runtime", "gradient_checkpointing"), "block_level"),
        (("runtime", "torch_compile"), False),
        (("runtime", "vram_maximum_peak_reserved_fraction"), 0.90),
        (("runtime", "freeze_crop_micro_accum_after_step0"), True),
        (("ema", "enabled"), True),
        (("ema", "decay"), 0.9999),
        (("validation", "every_steps"), 4000),
        (("validation", "manifest_key"), "primary_val_manifest"),
        (("validation", "groups"), ["single", "A"]),
        (("validation", "protocol"), "agenticir_official_parity"),
        (("checkpoint", "save_every_steps"), 4000),
        (("hard_guards", "require_stage3_approval"), True),
        (("hard_guards", "require_all_parent_hashes_match"), True),
        (("hard_guards", "allow_mio100_exploration"), False),
        (("hard_guards", "allow_mio100_formal_during_training"), False),
        (("hard_guards", "allow_group_b_or_c_training"), False),
        (("hard_guards", "fail_on_hash_mismatch"), True),
    )
    for path, expected in locked:
        _expect(config, path, expected)

    expected_teacher = [
        {
            "start_step": 0,
            "end_step_exclusive": 4000,
            "probability_start": 1.0,
            "probability_end": 1.0,
            "source": "true_active_set_and_distilled_relation",
        },
        {
            "start_step": 4000,
            "end_step_exclusive": 12000,
            "probability_start": 1.0,
            "probability_end": 0.5,
            "interpolation": "linear",
        },
        {
            "start_step": 12000,
            "end_step_exclusive": 40000,
            "probability_start": 0.25,
            "probability_end": 0.25,
            "source": "mixed_teacher_and_predicted_graph",
        },
    ]
    _expect(config, ("teacher_forcing", "schedule"), expected_teacher)

    forbidden = set(config["loss"]["forbidden"])
    expected_forbidden = {
        "gan",
        "lpips",
        "clip_iqa",
        "musiq",
        "dino_perceptual",
        "llm_reward",
        "reinforcement_learning",
        "independent_commit_verifier",
    }
    if forbidden != expected_forbidden:
        raise Stage4ContractError("Stage4 forbidden-loss set drifted")


def teacher_forcing_probability(step: int) -> float:
    """The written V7.1 schedule, including its deliberate step-12000 jump."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if step < 4000:
        return 1.0
    if step < 12_000:
        progress = (step - 4000) / 8000.0
        return 1.0 - 0.5 * progress
    return 0.25


def stage4_ssim_weight(step: int) -> float:
    """Cosine-ramp the ordinary SSIM term over the first 20% of Stage4."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if step >= 8000:
        return 0.05
    progress = step / 8000.0
    return 0.025 * (1.0 - math.cos(math.pi * progress))


def _cursor_rng(seed: int, cursor: int) -> random.Random:
    digest = hashlib.sha256(
        f"graphrestore-stage4:{seed}:{cursor}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class Stage4Request:
    index: int
    episode_type: str
    absolute_step: int
    sample_cursor: int
    use_teacher: bool
    forced_skill_ids: tuple[int, ...] = ()


def _relation_mapping(
    records: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(records, Mapping):
        result = {str(key): value for key, value in records.items()}
    else:
        result = {str(row.get("sample_id", "")): row for row in records}
    if not result or "" in result:
        raise Stage4ContractError("relation records require unique non-empty sample IDs")
    if len(result) != (len(records) if not isinstance(records, Mapping) else len(records)):
        raise Stage4ContractError("duplicate relation sample ID")
    return result


def load_relation_records(path: str | Path) -> dict[str, Mapping[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise Stage4ContractError(f"missing relation labels: {source}")
    rows = [row for _, row in iter_jsonl(source)]
    return _relation_mapping(rows)


class Stage4EpisodeDataset(Dataset[dict[str, Any]]):
    """Stage4 views over the frozen primary recipe dataset."""

    def __init__(
        self,
        base: GraphRestoreEpisodeDataset,
        relation_records: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if not base.training or base.crop_size != (160, 160):
            raise Stage4ContractError("Stage4 train dataset must use augmented crop160")
        if any(record.group not in {"single", "A"} for record in base.records):
            raise Stage4ContractError("Stage4 dataset contains forbidden groups")
        self.base = base
        self.records = base.records
        self.relation_records = _relation_mapping(relation_records)

    def __len__(self) -> int:
        return len(self.base)

    def set_worker_seed(self, seed: int) -> None:
        self.base.set_worker_seed(seed)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __getitem__(self, request: Stage4Request) -> dict[str, Any]:
        if not isinstance(request, Stage4Request):
            raise TypeError("Stage4EpisodeDataset requires Stage4Request indices")
        if request.episode_type not in EPISODE_TYPES:
            raise Stage4ContractError(f"unknown Stage4 episode: {request.episode_type}")
        record = self.records[request.index]
        if request.episode_type in {"single_restoration", "wrong_skill"} and record.is_pair:
            raise Stage4ContractError("single/wrong-skill request selected a pair recipe")
        if request.episode_type == "group_a_pair_restoration" and not record.is_pair:
            raise Stage4ContractError("Group-A request selected a single recipe")

        sample = dict(
            self.base[
                EpisodeRequest(
                    index=request.index,
                    episode_type="restoration",
                    absolute_step=request.absolute_step,
                    sample_cursor=request.sample_cursor,
                )
            ]
        )
        input_image = sample["input"]
        if not torch.is_tensor(input_image):
            raise Stage4ContractError("base episode returned a non-tensor input")
        forced = torch.zeros(len(SKILLS), dtype=torch.bool)
        for skill_id in request.forced_skill_ids:
            if not 0 <= skill_id < len(SKILLS):
                raise Stage4ContractError(f"invalid forced skill ID: {skill_id}")
            forced[skill_id] = True

        if request.episode_type == "clean_misuse":
            if len(request.forced_skill_ids) not in {1, 2}:
                raise Stage4ContractError("clean misuse must force one or two skills")
            clean = sample["gt_clean"]
            sample["input"] = clean
            sample["x_both"] = clean
            sample["target"] = clean
            sample["presence_target"] = torch.zeros(len(SKILLS), dtype=torch.float32)
            sample["guard_targets"] = torch.zeros_like(sample["guard_targets"])
            sample["global_severity_targets"] = torch.zeros_like(
                sample["global_severity_targets"]
            )
            sample["present_skill_ids"] = torch.full((2,), -1, dtype=torch.long)
        elif request.episode_type == "wrong_skill":
            if len(request.forced_skill_ids) != 1:
                raise Stage4ContractError("wrong-skill misuse must force exactly one skill")
            present = int(sample["present_skill_ids"][0])
            if request.forced_skill_ids[0] == present:
                raise Stage4ContractError("wrong-skill misuse cannot force the true skill")
            # Identity target is the degraded input, not clean.
            sample["target"] = input_image
        elif request.forced_skill_ids:
            raise Stage4ContractError("ordinary restoration cannot force a skill")

        relation_row = -1
        relation_label = -2
        relation_weight = 0.0
        relation_ambiguous = False
        if request.episode_type == "group_a_pair_restoration":
            try:
                relation = self.relation_records[record.sample_id]
            except KeyError as exc:
                raise Stage4ContractError(
                    f"Group-A Stage4 sample lacks distilled relation: {record.sample_id}"
                ) from exc
            ids = tuple(sorted(record.skill_ids))
            relation_row = PAIR_TO_ROW[ids]
            label_name = str(relation.get("label", ""))
            if label_name == "ambiguous":
                relation_label = -1
                relation_weight = 0.25
                relation_ambiguous = True
            elif label_name in {"i_before_j", "j_before_i", "parallel"}:
                relation_label = ("i_before_j", "j_before_i", "parallel").index(
                    label_name
                )
                relation_weight = 1.0
            else:
                raise Stage4ContractError(
                    f"invalid distilled relation label for {record.sample_id}: {label_name!r}"
                )
            if relation.get("relation_weight") != relation_weight:
                raise Stage4ContractError("distilled relation weight drifted")

        sample.update(
            {
                "stage4_episode_type": request.episode_type,
                "use_teacher": torch.tensor(request.use_teacher, dtype=torch.bool),
                "forced_skill_mask": forced,
                "relation_row": torch.tensor(relation_row, dtype=torch.long),
                "relation_label": torch.tensor(relation_label, dtype=torch.long),
                "relation_weight": torch.tensor(relation_weight, dtype=torch.float32),
                "relation_ambiguous": torch.tensor(
                    relation_ambiguous, dtype=torch.bool
                ),
            }
        )
        return sample


class Stage4EpisodeSampler(Sampler[Stage4Request]):
    """Checkpointable 20/70/5/5 sampler with uniform task identities."""

    def __init__(
        self,
        dataset: Stage4EpisodeDataset,
        *,
        num_samples: int,
        effective_batch_size: int = 4,
        seed: int = 2027,
        start_step: int = 0,
    ) -> None:
        if num_samples <= 0 or effective_batch_size != 4:
            raise ValueError("Stage4 requires positive samples and effective batch four")
        if seed != 2027 or start_step < 0:
            raise ValueError("Stage4 seed/start step drifted")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.effective_batch_size = effective_batch_size
        self.seed = seed
        self._sample_cursor = start_step * effective_batch_size
        self._consumed_optimizer_step = start_step

        buckets = task_buckets(dataset.records)
        self.single_tasks = tuple(sorted(key for key in buckets if len(key) == 1))
        all_pair_tasks = tuple(sorted(key for key in buckets if len(key) == 2))
        if len(self.single_tasks) != 8 or len(all_pair_tasks) != 8:
            raise Stage4ContractError("Stage4 requires eight single and eight Group-A tasks")
        self.buckets = buckets
        relation_ids = set(dataset.relation_records)
        labelled: dict[tuple[str, ...], tuple[int, ...]] = {}
        for task in all_pair_tasks:
            indices = tuple(
                index
                for index in buckets[task]
                if dataset.records[index].sample_id in relation_ids
            )
            if not indices:
                raise Stage4ContractError(f"no distilled Stage4 examples for pair {task}")
            labelled[task] = indices
        self.pair_tasks = all_pair_tasks
        self.labelled_pair_buckets = labelled

    @property
    def step(self) -> int:
        return self._sample_cursor // self.effective_batch_size

    def mark_consumed_optimizer_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("consumed step must be non-negative")
        self._consumed_optimizer_step = int(step)

    def state_dict(
        self, *, consumed_optimizer_step: int | None = None
    ) -> dict[str, Any]:
        if consumed_optimizer_step is not None:
            self.mark_consumed_optimizer_step(consumed_optimizer_step)
        return {
            "schema_version": STAGE4_SCHEMA,
            "stage": "stage4",
            "seed": self.seed,
            "num_samples": self.num_samples,
            "effective_batch_size": self.effective_batch_size,
            "consumed_optimizer_step": self._consumed_optimizer_step,
            "sample_cursor": self._consumed_optimizer_step
            * self.effective_batch_size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": STAGE4_SCHEMA,
            "stage": "stage4",
            "seed": self.seed,
            "num_samples": self.num_samples,
            "effective_batch_size": self.effective_batch_size,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise Stage4ContractError(
                    f"Stage4 sampler {key} mismatch: {state.get(key)!r} != {value!r}"
                )
        step = state.get("consumed_optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise Stage4ContractError("invalid consumed Stage4 optimizer step")
        if state.get("sample_cursor") != step * self.effective_batch_size:
            raise Stage4ContractError("Stage4 sampler cursor is not step*4")
        self._consumed_optimizer_step = step
        self._sample_cursor = step * self.effective_batch_size

    @staticmethod
    def _pick(
        rng: random.Random,
        tasks: Sequence[tuple[str, ...]],
        buckets: Mapping[tuple[str, ...], Sequence[int]],
    ) -> int:
        task = tasks[rng.randrange(len(tasks))]
        values = buckets[task]
        return int(values[rng.randrange(len(values))])

    def _request(self, cursor: int) -> Stage4Request:
        step = cursor // self.effective_batch_size
        rng = _cursor_rng(self.seed, cursor)
        draw = rng.random()
        teacher = rng.random() < teacher_forcing_probability(step)
        if draw < 0.20:
            index = self._pick(rng, self.single_tasks, self.buckets)
            return Stage4Request(
                index, "single_restoration", step, cursor, teacher
            )
        if draw < 0.90:
            index = self._pick(rng, self.pair_tasks, self.labelled_pair_buckets)
            return Stage4Request(
                index, "group_a_pair_restoration", step, cursor, teacher
            )
        if draw < 0.95:
            # Any frozen single recipe provides a clean image.  Skill choice is
            # independent and uniform; sample_without_replacement handles 1/2.
            index = self._pick(rng, self.single_tasks, self.buckets)
            count = 1 + rng.randrange(2)
            forced = tuple(sorted(rng.sample(range(len(SKILLS)), count)))
            return Stage4Request(
                index, "clean_misuse", step, cursor, False, forced
            )
        true_skill = rng.randrange(len(SKILLS))
        task = next(
            key
            for key in self.single_tasks
            if self.dataset.records[self.buckets[key][0]].skill_ids[0] == true_skill
        )
        values = self.buckets[task]
        index = int(values[rng.randrange(len(values))])
        wrong_draw = rng.randrange(len(SKILLS) - 1)
        wrong_skill = wrong_draw + int(wrong_draw >= true_skill)
        return Stage4Request(
            index, "wrong_skill", step, cursor, False, (wrong_skill,)
        )

    def __iter__(self) -> Iterator[Stage4Request]:
        for _ in range(self.num_samples):
            cursor = self._sample_cursor
            self._sample_cursor += 1
            yield self._request(cursor)

    def __len__(self) -> int:
        return self.num_samples


def _mapping_of_tensors(value: object, *, field: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise Stage4ContractError(f"{field} must be a non-empty tensor mapping")
    if any(not isinstance(key, str) or not torch.is_tensor(item) for key, item in value.items()):
        raise Stage4ContractError(f"{field} contains non-tensor entries")
    return value  # type: ignore[return-value]


def _flatten_values(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten_values(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _flatten_values(item)
    else:
        yield value


def validate_stage3_approval(
    approval_path: str | Path,
    *,
    stage2_decision_path: str | Path | None = None,
) -> Mapping[str, Any]:
    path = Path(approval_path).resolve()
    if not path.is_file():
        raise Stage4ContractError(
            "Stage4 is forbidden before explicit Stage3 approval: " f"{path}"
        )
    approval = load_json(path)
    if not isinstance(approval, Mapping):
        raise Stage4ContractError("Stage3 approval must be a JSON mapping")
    if (
        approval.get("schema_version") != STAGE3_APPROVAL_SCHEMA
        or approval.get("kind") != "stage3_approval"
        or approval.get("protocol_id") != PROTOCOL_ID
        or approval.get("approved") is not True
    ):
        raise Stage4ContractError("invalid or non-approved Stage3 approval artifact")
    if stage2_decision_path is not None:
        decision = Path(stage2_decision_path).resolve()
        if not decision.is_file():
            raise Stage4ContractError(f"missing frozen Stage2 decision: {decision}")
        if approval.get("stage2_decision_sha256") != sha256_file(decision):
            raise Stage4ContractError("Stage3 approval/Stage2 decision hash mismatch")
    return approval


def load_presence_thresholds(
    path: str | Path,
    *,
    stage3_checkpoint_sha256: str,
    stage3_approval_sha256: str | None = None,
) -> tuple[Tensor, Mapping[str, Any]]:
    threshold_path = Path(path).resolve()
    if not threshold_path.is_file():
        raise Stage4ContractError(f"missing frozen planner thresholds: {threshold_path}")
    payload = load_json(threshold_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "graphrestore-presence-thresholds-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("frozen") is not True
        or payload.get("source") != "primary_val_presence_f1_only"
        or payload.get("calibration_runs") != 1
        or payload.get("mio100_rows_read") != 0
    ):
        raise Stage4ContractError("planner thresholds are not marked frozen")
    if payload.get("skills") != list(SKILLS):
        raise Stage4ContractError("planner threshold skill ordering drifted")
    bound_checkpoint = payload.get(
        "checkpoint_sha256", payload.get("stage3_checkpoint_sha256")
    )
    if bound_checkpoint != stage3_checkpoint_sha256:
        raise Stage4ContractError("thresholds are not bound to the Stage3 parent")
    selected = payload.get("selected_stage3_checkpoint")
    if not isinstance(selected, Mapping) or selected.get("sha256") != stage3_checkpoint_sha256:
        raise Stage4ContractError("selected Stage3 checkpoint binding drifted")
    if (
        stage3_approval_sha256 is not None
        and payload.get("stage3_approval_sha256") != stage3_approval_sha256
    ):
        raise Stage4ContractError("thresholds are not bound to current Stage3 approval")
    raw = payload.get("thresholds")
    if isinstance(raw, Mapping):
        if set(raw) != set(SKILLS):
            raise Stage4ContractError("threshold mapping must contain exactly eight skills")
        values = [float(raw[name]) for name in SKILLS]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [float(item) for item in raw]
    else:
        raise Stage4ContractError("thresholds must be a skill mapping or list")
    tensor = torch.tensor(values, dtype=torch.float32)
    if tuple(tensor.shape) != (len(SKILLS),) or not bool(torch.isfinite(tensor).all()):
        raise Stage4ContractError("thresholds must contain eight finite values")
    if bool(torch.any((tensor < 0.20) | (tensor > 0.80))):
        raise Stage4ContractError("thresholds escape the frozen [0.20,0.80] grid")
    grid_units = (tensor - 0.20) / 0.02
    if not torch.allclose(grid_units, grid_units.round(), atol=2.0e-5, rtol=0.0):
        raise Stage4ContractError("thresholds are not on the frozen 0.02 grid")
    expected_grid = [0.20 + 0.02 * index for index in range(31)]
    actual_grid = payload.get("search_grid")
    if not isinstance(actual_grid, Sequence) or isinstance(actual_grid, (str, bytes)):
        raise Stage4ContractError("threshold artifact lacks the frozen search grid")
    if len(actual_grid) != len(expected_grid) or any(
        not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9)
        for actual, expected in zip(actual_grid, expected_grid, strict=True)
    ):
        raise Stage4ContractError("threshold search grid drifted")
    if payload.get("tie_break") != "lowest_threshold":
        raise Stage4ContractError("threshold tie-break drifted")
    return tensor, payload


@dataclass(frozen=True)
class FrozenStage3Snapshot:
    model: GraphRestore
    checkpoint_sha256: str
    checkpoint_step: int
    provenance: Mapping[str, Any]


def load_stage3_best_ema(
    checkpoint: str | Path,
    *,
    model: GraphRestore,
    approval_sha256: str,
    required_artifact_hashes: Sequence[str] = (),
) -> FrozenStage3Snapshot:
    """Strictly load Stage3 best EMA without inheriting its optimizer state."""

    path = Path(checkpoint).resolve()
    if path.name != "best_ema.pth" or not path.is_file():
        raise Stage4ContractError(f"missing Stage3 best_ema.pth: {path}")
    digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise Stage4ContractError("Stage3 checkpoint must be a mapping")
    if payload.get("schema_version") != "graphrestore-checkpoint-v1":
        raise Stage4ContractError("Stage3 checkpoint schema mismatch")
    if payload.get("stage") != "stage3":
        raise Stage4ContractError("Stage4 parent must have stage='stage3'")
    if payload.get("model_role") != "ema_selection" or payload.get("resumable") is not False:
        raise Stage4ContractError("Stage4 parent must be a non-resumable Stage3 EMA selection")
    state = _mapping_of_tensors(payload.get("model"), field="Stage3 model")
    ema = payload.get("ema")
    if not isinstance(ema, Mapping):
        raise Stage4ContractError("Stage3 best checkpoint lacks EMA state")
    if ema.get("scope") != "planner_parameters_only_executor_bitwise_frozen":
        raise Stage4ContractError("Stage3 EMA scope did not preserve the frozen executor")
    if payload.get("executor_frozen") is not True or payload.get("trainable_prefixes") != [
        "planner."
    ]:
        raise Stage4ContractError("Stage3 checkpoint executor/trainable boundary drifted")
    shadow = _mapping_of_tensors(ema.get("shadow"), field="Stage3 EMA shadow")
    if state.keys() != shadow.keys():
        raise Stage4ContractError("Stage3 best model/EMA keys differ")
    for name in state:
        if state[name].shape != shadow[name].shape or state[name].dtype != shadow[name].dtype:
            raise Stage4ContractError(f"Stage3 best model/EMA metadata differs: {name}")
        if not torch.equal(state[name], shadow[name]):
            raise Stage4ContractError(f"Stage3 best model is not its EMA snapshot: {name}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage4ContractError("Stage3 checkpoint lacks provenance")
    flat_values = set(_flatten_values(provenance))
    if approval_sha256 not in flat_values:
        raise Stage4ContractError("Stage3 checkpoint is not bound to current approval")
    for artifact_hash in required_artifact_hashes:
        if artifact_hash not in flat_values:
            raise Stage4ContractError(
                f"Stage3 checkpoint lacks required artifact binding: {artifact_hash}"
            )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Stage4ContractError("Stage3 EMA did not load strictly into GraphRestore")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Stage4ContractError("Stage3 checkpoint has invalid step")
    if sha256_file(path) != digest:
        raise Stage4ContractError("Stage3 checkpoint changed while loading")
    return FrozenStage3Snapshot(model, digest, step, provenance)


def stage4_parameter_role(name: str) -> str | None:
    if name.startswith("planner."):
        return "planner"
    if name.startswith("decoder.skill_bank."):
        return "skills_mixers"
    if name.startswith("decoder."):
        return "decoder_refine_head"
    if name.startswith(
        (
            "encoder.down23.",
            "encoder.level3.",
            "encoder.down34.",
            "encoder.level4.",
        )
    ):
        return "encoder34"
    return None


def set_stage4_trainability(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for name, parameter in unwrap_model(model).named_parameters():
        role = stage4_parameter_role(name)
        parameter.requires_grad_(role is not None)
        counts[f"{role or 'frozen'}:{'trainable' if role else 'frozen'}"] += parameter.numel()
    return dict(counts)


def _is_norm_or_bias(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith(".bias") or ".norm" in name.lower()


def build_stage4_optimizer(
    model: nn.Module,
    *,
    planner_lr: float = 5.0e-5,
    skills_lr: float = 3.0e-5,
    decoder_lr: float = 1.0e-5,
    encoder34_lr: float = 2.0e-6,
    weight_decay: float = 1.0e-4,
    fused_if_supported: bool = True,
) -> torch.optim.AdamW:
    """Build a fresh, exhaustive, role-exclusive Stage4 AdamW."""

    learning_rates = {
        "planner": planner_lr,
        "skills_mixers": skills_lr,
        "decoder_refine_head": decoder_lr,
        "encoder34": encoder34_lr,
    }
    if min(learning_rates.values()) <= 0 or weight_decay < 0:
        raise ValueError("invalid Stage4 optimizer settings")
    set_stage4_trainability(model)
    grouped: dict[tuple[str, float], list[nn.Parameter]] = defaultdict(list)
    seen: set[int] = set()
    for name, parameter in unwrap_model(model).named_parameters():
        role = stage4_parameter_role(name)
        if role is None:
            if parameter.requires_grad:
                raise Stage4ContractError(f"unexpected Stage4 trainable parameter: {name}")
            continue
        if not parameter.requires_grad or id(parameter) in seen:
            raise Stage4ContractError(f"invalid Stage4 parameter assignment: {name}")
        decay = 0.0 if _is_norm_or_bias(name, parameter) else weight_decay
        grouped[(role, decay)].append(parameter)
        seen.add(id(parameter))
    expected = {
        id(parameter)
        for name, parameter in unwrap_model(model).named_parameters()
        if stage4_parameter_role(name) is not None
    }
    if seen != expected:
        raise Stage4ContractError("Stage4 optimizer does not cover each trainable tensor once")
    present_roles = {role for role, _ in grouped}
    if present_roles != set(learning_rates):
        raise Stage4ContractError(f"Stage4 optimizer roles incomplete: {present_roles}")
    groups = [
        {
            "params": parameters,
            "lr": float(learning_rates[role]),
            "initial_lr": float(learning_rates[role]),
            "weight_decay": decay,
            "role": role,
        }
        for (role, decay), parameters in sorted(grouped.items())
    ]
    kwargs: dict[str, Any] = {"betas": (0.9, 0.999)}
    if fused_if_supported and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def _require_tensor(batch: Mapping[str, Any], key: str, device: torch.device) -> Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise Stage4ContractError(f"Stage4 batch field {key!r} must be a tensor")
    return value.to(device=device, non_blocking=device.type == "cuda")


@dataclass(frozen=True)
class Stage4Batch:
    input: Tensor
    target: Tensor
    gt_clean: Tensor
    target_after_i: Tensor
    target_after_j: Tensor
    only_i: Tensor
    only_j: Tensor
    guard_targets: Tensor
    global_severity_targets: Tensor
    presence_target: Tensor
    dense_guard_mask: Tensor
    global_guard_mask: Tensor
    present_skill_ids: Tensor
    forced_skill_mask: Tensor
    use_teacher: Tensor
    relation_row: Tensor
    relation_label: Tensor
    relation_weight: Tensor
    relation_ambiguous: Tensor
    episode_types: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.input.shape[0])


def prepare_stage4_batch(batch: Mapping[str, Any], device: torch.device) -> Stage4Batch:
    float_fields = (
        "input",
        "target",
        "gt_clean",
        "target_after_i",
        "target_after_j",
        "only_i",
        "only_j",
        "guard_targets",
        "global_severity_targets",
        "presence_target",
    )
    tensors = {key: _require_tensor(batch, key, device).float() for key in float_fields}
    bool_fields = (
        "dense_guard_mask",
        "global_guard_mask",
        "forced_skill_mask",
        "use_teacher",
        "relation_ambiguous",
    )
    bools = {key: _require_tensor(batch, key, device).bool() for key in bool_fields}
    longs = {
        key: _require_tensor(batch, key, device).long()
        for key in ("present_skill_ids", "relation_row", "relation_label")
    }
    relation_weight = _require_tensor(batch, "relation_weight", device).float()
    raw_types = batch.get("stage4_episode_type")
    if not isinstance(raw_types, (list, tuple)):
        raise Stage4ContractError("collated Stage4 episode types must be a sequence")
    episode_types = tuple(str(value) for value in raw_types)
    batch_size = int(tensors["input"].shape[0])
    if len(episode_types) != batch_size or any(value not in EPISODE_TYPES for value in episode_types):
        raise Stage4ContractError("invalid collated Stage4 episode types")
    if tensors["input"].shape != tensors["gt_clean"].shape:
        raise Stage4ContractError("Stage4 input/clean shape mismatch")
    if tuple(bools["forced_skill_mask"].shape) != (batch_size, len(SKILLS)):
        raise Stage4ContractError("forced_skill_mask must be [B,8]")
    if tuple(tensors["presence_target"].shape) != (batch_size, len(SKILLS)):
        raise Stage4ContractError("presence_target must be [B,8]")
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"non-finite Stage4 batch tensor: {name}")
    return Stage4Batch(
        **tensors,
        **bools,
        **longs,
        relation_weight=relation_weight,
        episode_types=episode_types,
    )


def _teacher_relation_logits(batch: Stage4Batch, sample: int) -> Tensor:
    logits = batch.input.new_zeros((len(PAIR_INDICES), 3))
    row = int(batch.relation_row[sample])
    if row < 0:
        return logits
    label = int(batch.relation_label[sample])
    if bool(batch.relation_ambiguous[sample]):
        # Equal serial evidence is deliberately not a hard pseudo-direction.
        logits[row] = logits.new_tensor((10.0, 10.0, -10.0))
    elif 0 <= label < 3:
        logits[row].fill_(-20.0)
        logits[row, label] = 20.0
    else:
        raise Stage4ContractError("teacher pair lacks a legal distilled relation")
    return logits


def _compile_initial_graphs(
    model: GraphRestore,
    planner: PlannerOutput,
    batch: Stage4Batch,
    thresholds: Tensor,
) -> tuple[tuple[CompiledGraph, ...], tuple[bool, ...]]:
    graphs: list[CompiledGraph] = []
    teacher_flags: list[bool] = []
    probabilities = planner.presence_probabilities.detach()
    for sample, episode_type in enumerate(batch.episode_types):
        counterfactual = episode_type in COUNTERFACTUAL_TYPES
        use_teacher = bool(batch.use_teacher[sample]) and not counterfactual
        teacher_flags.append(use_teacher)
        if counterfactual:
            graphs.append(model.compiler.compile((), planner.relation_logits[sample]))
            continue
        if use_teacher:
            active = torch.nonzero(
                batch.presence_target[sample] > 0.5, as_tuple=False
            ).flatten().tolist()
            if not 1 <= len(active) <= 2:
                raise Stage4ContractError("ordinary teacher graph must have one or two skills")
            graphs.append(
                model.compiler.compile(active, _teacher_relation_logits(batch, sample))
            )
        else:
            active = model._select_active(probabilities[sample], thresholds)
            graphs.append(model.compiler.compile(active, planner.relation_logits[sample]))
    return tuple(graphs), tuple(teacher_flags)


def _remaining_target(batch: Stage4Batch, sample: int, remaining: Tensor) -> Tensor:
    if batch.episode_types[sample] in COUNTERFACTUAL_TYPES:
        return batch.input[sample]
    present = [int(value) for value in batch.present_skill_ids[sample].tolist() if int(value) >= 0]
    remaining_ids = [skill for skill in present if bool(remaining[skill])]
    if not remaining_ids:
        return batch.gt_clean[sample]
    if len(present) == 1:
        return batch.input[sample]
    if len(remaining_ids) == 2:
        return batch.input[sample]
    if remaining_ids[0] == present[0]:
        return batch.only_i[sample]
    if remaining_ids[0] == present[1]:
        return batch.only_j[sample]
    raise Stage4ContractError("remaining subset target cannot be resolved")


def _relation_supervision(batch: Stage4Batch) -> tuple[Tensor, Tensor, Tensor]:
    labels = torch.full(
        (batch.batch_size, len(PAIR_INDICES)),
        -2,
        device=batch.input.device,
        dtype=torch.long,
    )
    weights = torch.zeros_like(labels, dtype=torch.float32)
    ambiguous = torch.zeros_like(labels, dtype=torch.bool)
    for sample in range(batch.batch_size):
        row = int(batch.relation_row[sample])
        if row < 0:
            continue
        labels[sample, row] = batch.relation_label[sample]
        weights[sample, row] = batch.relation_weight[sample]
        ambiguous[sample, row] = batch.relation_ambiguous[sample]
    return labels, weights, ambiguous


def _planner_supervision(
    planner: PlannerOutput,
    batch: Stage4Batch,
    remaining: Tensor,
    *,
    include_relations: bool,
) -> PlannerLossBreakdown:
    target_guard = batch.guard_targets * remaining[:, :, None, None].to(
        batch.guard_targets
    )
    target_severity = batch.global_severity_targets * remaining.to(
        batch.global_severity_targets
    )
    dense_mask = batch.dense_guard_mask & remaining
    global_mask = batch.global_guard_mask & remaining
    absent = ~remaining
    guard = guard_supervision_loss(
        planner.guard_logits,
        target_guard,
        target_severity,
        dense_skill_mask=dense_mask,
        global_skill_mask=global_mask,
        absent_skill_mask=absent,
    )
    stop_target = (~remaining.any(dim=1)).to(planner.stop_logit.dtype)[:, None]
    if include_relations:
        labels, weights, ambiguous = _relation_supervision(batch)
        return planner_loss(
            presence_logits=planner.presence_logits,
            presence_targets=remaining.to(planner.presence_logits),
            # Preserve the final ambiguity ruling's mandatory FP32
            # log_softmax/logsumexp path under the outer BF16 autocast.
            relation_logits=planner.relation_logits.float(),
            relation_targets=labels,
            relation_weights=weights,
            relation_ambiguous_mask=ambiguous,
            stop_logits=planner.stop_logit,
            stop_targets=stop_target,
            guard=guard,
        )
    presence = focal_binary_cross_entropy(
        planner.presence_logits, remaining.to(planner.presence_logits), gamma=2.0
    )
    stop = F.binary_cross_entropy_with_logits(planner.stop_logit, stop_target)
    zero = planner.presence_logits.sum() * 0.0
    total = presence + 0.5 * guard.total + 0.25 * stop
    return PlannerLossBreakdown(total, presence, guard.total, zero, stop, zero)


@dataclass(frozen=True)
class Stage4RoundDiagnostics:
    round_index: int
    active_skill_counts: tuple[int, ...]
    active_sample_count: int
    skipped_node_count: int
    guard_mean_per_skill: tuple[float, ...]
    guard_max_per_skill: tuple[float, ...]
    union_guard_mean: float
    union_guard_std: float
    union_guard_high_fraction: float
    rgb_residual_norm: float
    identity_fraction: float


@dataclass(frozen=True)
class Stage4ProgramOutput:
    final: Tensor
    step_images: tuple[Tensor, ...]
    step_targets: tuple[Tensor, ...]
    step_valid_masks: tuple[Tensor, ...]
    planner_losses: tuple[PlannerLossBreakdown, ...]
    compiled_graphs: tuple[CompiledGraph, ...]
    graph_states: tuple[ProgramGraphState, ...]
    teacher_flags: tuple[bool, ...]
    executed_masks: tuple[Tensor, ...]
    round_diagnostics: tuple[Stage4RoundDiagnostics, ...]
    reentry_request_count: int
    unexpected_activation_count: int


def run_stage4_program(
    model: GraphRestore,
    batch: Stage4Batch,
    *,
    presence_thresholds: Tensor | Sequence[float] | None = None,
) -> Stage4ProgramOutput:
    """Run a two-round Stage4 trajectory with one t=0 compilation per sample."""

    core = unwrap_model(model)
    if not isinstance(core, GraphRestore):
        raise TypeError("Stage4 program requires GraphRestore")
    padded, original_shape = pad_to_multiple(batch.input, 8)
    current = padded
    x0 = padded
    features = core.encode(current)
    planner = core.plan_state(
        x0, current, features, round_value=0.0, compute_relations=True
    )
    thresholds = core._threshold_tensor(
        presence_thresholds, planner.presence_logits
    )
    compiled, teacher_flags = _compile_initial_graphs(
        core, planner, batch, thresholds
    )
    states = [ProgramGraphState.from_compiled(graph) for graph in compiled]
    remaining = batch.presence_target > 0.5
    initial_graph_mask = torch.zeros_like(remaining)
    for sample, graph in enumerate(compiled):
        for skill in graph.active_skills:
            initial_graph_mask[sample, SKILL_TO_INDEX[skill]] = True

    planner_losses: list[PlannerLossBreakdown] = []
    step_images: list[Tensor] = []
    step_targets: list[Tensor] = []
    step_valid: list[Tensor] = []
    executed_masks: list[Tensor] = []
    round_diagnostics: list[Stage4RoundDiagnostics] = []
    executed = torch.zeros_like(remaining)
    terminal = torch.zeros(batch.batch_size, device=batch.input.device, dtype=torch.bool)
    reentry_count = 0
    unexpected_count = 0

    for round_index in range(2):
        planner_losses.append(
            _planner_supervision(
                planner,
                batch,
                remaining,
                include_relations=round_index == 0,
            )
        )
        probabilities = planner.presence_probabilities.detach()
        above = probabilities >= thresholds[None, :]
        reentry_count += int((executed & above).sum().item())
        unexpected_count += int(((~initial_graph_mask) & above).sum().item())
        active = torch.zeros_like(remaining)
        forced_presence = torch.zeros_like(remaining)
        processed = torch.zeros(batch.batch_size, device=batch.input.device, dtype=torch.bool)
        skipped_node_count = 0

        for sample, episode_type in enumerate(batch.episode_types):
            if bool(terminal[sample]):
                continue
            if episode_type in COUNTERFACTUAL_TYPES:
                if round_index == 0:
                    active[sample] = batch.forced_skill_mask[sample]
                    forced_presence[sample] = batch.forced_skill_mask[sample]
                    processed[sample] = bool(active[sample].any())
                terminal[sample] = True
                continue
            state = states[sample]
            if state.complete:
                terminal[sample] = True
                continue
            level = state.current_level
            execute_names: list[str] = []
            skip_names: list[str] = []
            if teacher_flags[sample]:
                execute_names.extend(level)
            else:
                pending_ids = [SKILL_TO_INDEX[name] for name in state.pending]
                confident = bool(
                    pending_ids
                    and torch.any(above[sample, pending_ids]).item()
                )
                if float(planner.stop_logit[sample].sigmoid()) >= 0.5 and not confident:
                    state.skip_all_pending()
                    terminal[sample] = True
                    continue
                for skill in level:
                    skill_id = SKILL_TO_INDEX[skill]
                    if bool(above[sample, skill_id]):
                        execute_names.append(skill)
                    else:
                        skip_names.append(skill)
            skipped_node_count += len(skip_names)
            for skill in execute_names:
                skill_id = SKILL_TO_INDEX[skill]
                active[sample, skill_id] = True
                if teacher_flags[sample]:
                    forced_presence[sample, skill_id] = True
            state.finish_current_level(executed=execute_names, skipped=skip_names)
            processed[sample] = True
            if state.complete:
                terminal[sample] = True

        guards = planner.execution_guards(forced_presence)
        execution = None
        if bool(active.any()):
            execution = core.execute_level(
                current,
                features,
                guards=guards,
                active_mask=active,
                forced_presence_mask=forced_presence,
            )
            current = execution.next_image
        executed = executed | active
        # Only a true skill changes the recipe's remaining-degradation state.
        remaining = remaining & ~active
        targets = torch.stack(
            [_remaining_target(batch, sample, remaining[sample]) for sample in range(batch.batch_size)]
        )
        step_images.append(crop_to_shape(current, original_shape))
        step_targets.append(targets)
        step_valid.append(processed & active.any(dim=1))
        executed_masks.append(active)
        active_samples = active.any(dim=1)
        active_skill_counts = tuple(
            int(active[:, skill].sum().item()) for skill in range(len(SKILLS))
        )
        guard_means: list[float] = []
        guard_maxima: list[float] = []
        for skill in range(len(SKILLS)):
            selected = active[:, skill]
            if bool(selected.any()):
                values = guards[selected, skill].detach().float()
                guard_means.append(float(values.mean().item()))
                guard_maxima.append(float(values.amax().item()))
            else:
                guard_means.append(0.0)
                guard_maxima.append(0.0)
        if execution is not None and bool(active_samples.any()):
            union = execution.union_guard[active_samples].detach().float()
            residual = execution.residual_norm[active_samples].detach().float()
            identity = execution.identity_mask[active_samples].detach().float()
            union_mean = float(union.mean().item())
            union_std = float(union.std(unbiased=False).item())
            union_high = float((union > 0.9).float().mean().item())
            residual_norm = float(residual.mean().item())
            identity_fraction = float(identity.mean().item())
        else:
            union_mean = union_std = union_high = residual_norm = 0.0
            identity_fraction = 1.0
        round_diagnostics.append(
            Stage4RoundDiagnostics(
                round_index=round_index,
                active_skill_counts=active_skill_counts,
                active_sample_count=int(active_samples.sum().item()),
                skipped_node_count=skipped_node_count,
                guard_mean_per_skill=tuple(guard_means),
                guard_max_per_skill=tuple(guard_maxima),
                union_guard_mean=union_mean,
                union_guard_std=union_std,
                union_guard_high_fraction=union_high,
                rgb_residual_norm=residual_norm,
                identity_fraction=identity_fraction,
            )
        )

        if round_index == 0 and not bool(terminal.all()):
            features = core.encode(current)
            planner = core.plan_state(
                x0,
                current,
                features,
                round_value=0.5,
                compute_relations=False,
            )
        elif round_index == 0:
            break

    return Stage4ProgramOutput(
        final=crop_to_shape(current, original_shape),
        step_images=tuple(step_images),
        step_targets=tuple(step_targets),
        step_valid_masks=tuple(step_valid),
        planner_losses=tuple(planner_losses),
        compiled_graphs=compiled,
        graph_states=tuple(states),
        teacher_flags=teacher_flags,
        executed_masks=tuple(executed_masks),
        round_diagnostics=tuple(round_diagnostics),
        reentry_request_count=reentry_count,
        unexpected_activation_count=unexpected_count,
    )


def _per_image_charbonnier(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("per-image Charbonnier shape mismatch")
    return torch.sqrt((prediction - target).square() + 1.0e-6).mean((1, 2, 3))


@dataclass(frozen=True)
class Stage4ImageLoss:
    total: Tensor
    final_pixel: Tensor
    step_pixel: Tensor
    final_ssim: Tensor
    noop_pixel: Tensor
    noop_ssim: Tensor
    lambda_ssim: float


def stage4_image_loss(
    program: Stage4ProgramOutput,
    batch: Stage4Batch,
    *,
    step: int,
) -> Stage4ImageLoss:
    final_pix = _per_image_charbonnier(program.final, batch.gt_clean)
    noop_pix = _per_image_charbonnier(program.final, batch.input)
    counterfactual = torch.tensor(
        [value in COUNTERFACTUAL_TYPES for value in batch.episode_types],
        device=batch.input.device,
        dtype=torch.bool,
    )
    ordinary = ~counterfactual
    # Do not build two full SSIM graphs when each sample belongs to exactly one
    # branch.  This preserves the written loss and materially lowers Stage4's
    # two-round activation peak without changing crop/effective batch.
    final_ssim = torch.zeros(batch.batch_size, device=batch.input.device)
    noop_ssim = torch.zeros_like(final_ssim)
    if bool(ordinary.any()):
        final_ssim[ordinary] = 1.0 - train_ssim_y(
            program.final[ordinary], batch.gt_clean[ordinary]
        )
    if bool(counterfactual.any()):
        noop_ssim[counterfactual] = 1.0 - train_ssim_y(
            program.final[counterfactual], batch.input[counterfactual]
        )
    step_sum = torch.zeros(batch.batch_size, device=batch.input.device)
    step_count = torch.zeros_like(step_sum)
    for prediction, target, valid in zip(
        program.step_images,
        program.step_targets,
        program.step_valid_masks,
        strict=True,
    ):
        value = _per_image_charbonnier(prediction, target)
        step_sum = step_sum + value * valid.to(value)
        step_count = step_count + valid.to(step_count)
    step_pix = step_sum / step_count.clamp_min(1.0)
    lambda_ssim = stage4_ssim_weight(step)
    ordinary_loss = final_pix + 0.30 * step_pix + lambda_ssim * final_ssim
    noop_loss = noop_pix + 0.05 * noop_ssim
    per_image = torch.where(counterfactual, noop_loss, ordinary_loss)
    zero = program.final.new_zeros(())

    def selected_mean(value: Tensor, mask: Tensor) -> Tensor:
        return value[mask].mean() if bool(mask.any()) else zero

    return Stage4ImageLoss(
        total=per_image.mean(),
        final_pixel=selected_mean(final_pix, ordinary),
        step_pixel=selected_mean(step_pix, ordinary),
        final_ssim=selected_mean(final_ssim, ordinary),
        noop_pixel=selected_mean(noop_pix, counterfactual),
        noop_ssim=selected_mean(noop_ssim, counterfactual),
        lambda_ssim=lambda_ssim,
    )


@dataclass(frozen=True)
class Stage4StepResult:
    loss: float
    image_loss: float
    planner_loss: float
    final_pixel: float
    step_pixel: float
    final_ssim: float
    noop_pixel: float
    noop_ssim: float
    lambda_ssim: float
    teacher_fraction: float
    reentry_requests: int
    unexpected_activations: int
    round_diagnostics: tuple[Mapping[str, Any], ...]
    grad_norm: float
    samples: int
    seconds: float


def _autocast(device: torch.device, use_bf16: bool):
    if use_bf16:
        if device.type != "cuda":
            raise Stage4ContractError("formal Stage4 BF16 requires CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_stage4_optimizer_step(
    model: GraphRestore,
    micro_batches: Sequence[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler | None,
    ema: ExponentialMovingAverage | None,
    *,
    step: int,
    device: torch.device,
    use_bf16: bool = True,
) -> Stage4StepResult:
    if not micro_batches:
        raise ValueError("Stage4 requires at least one micro batch")
    model.train()
    set_stage4_trainability(model)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    reentry = unexpected = 0
    round_diagnostics: list[Mapping[str, Any]] = []

    for micro_index, raw in enumerate(micro_batches):
        batch = prepare_stage4_batch(raw, device)
        with _autocast(device, use_bf16):
            program = run_stage4_program(model, batch)
            image = stage4_image_loss(program, batch, step=step)
            planner_total = torch.stack(
                [item.total for item in program.planner_losses]
            ).mean()
            total = image.total + 0.05 * planner_total
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError("non-finite Stage4 total loss")
        (total / len(micro_batches)).backward()
        count = batch.batch_size
        samples += count
        for key, value in (
            ("loss", total),
            ("image_loss", image.total),
            ("planner_loss", planner_total),
            ("final_pixel", image.final_pixel),
            ("step_pixel", image.step_pixel),
            ("final_ssim", image.final_ssim),
            ("noop_pixel", image.noop_pixel),
            ("noop_ssim", image.noop_ssim),
        ):
            totals[key] += float(value.detach()) * count
        totals["teacher"] += sum(program.teacher_flags)
        reentry += program.reentry_request_count
        unexpected += program.unexpected_activation_count
        for diagnostic in program.round_diagnostics:
            round_diagnostics.append(
                {
                    "micro_batch_index": micro_index,
                    "round_index": diagnostic.round_index,
                    "active_skills": {
                        SKILLS[index]: count
                        for index, count in enumerate(diagnostic.active_skill_counts)
                        if count
                    },
                    "active_sample_count": diagnostic.active_sample_count,
                    "skipped_node_count": diagnostic.skipped_node_count,
                    "guard_mean_per_skill": {
                        SKILLS[index]: value
                        for index, value in enumerate(diagnostic.guard_mean_per_skill)
                    },
                    "guard_max_per_skill": {
                        SKILLS[index]: value
                        for index, value in enumerate(diagnostic.guard_max_per_skill)
                    },
                    "union_guard_mean": diagnostic.union_guard_mean,
                    "union_guard_std": diagnostic.union_guard_std,
                    "union_guard_high_fraction": diagnostic.union_guard_high_fraction,
                    "rgb_residual_norm": diagnostic.rgb_residual_norm,
                    "identity_fraction": diagnostic.identity_fraction,
                }
            )

    if samples != 4:
        raise Stage4ContractError(f"Stage4 effective batch must be four, got {samples}")
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise Stage4ContractError("Stage4 backward produced no gradients")
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=0.5, error_if_nonfinite=True
    )
    grad_norm = float(grad_norm_tensor.detach())
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return Stage4StepResult(
        loss=totals["loss"] / samples,
        image_loss=totals["image_loss"] / samples,
        planner_loss=totals["planner_loss"] / samples,
        final_pixel=totals["final_pixel"] / samples,
        step_pixel=totals["step_pixel"] / samples,
        final_ssim=totals["final_ssim"] / samples,
        noop_pixel=totals["noop_pixel"] / samples,
        noop_ssim=totals["noop_ssim"] / samples,
        lambda_ssim=stage4_ssim_weight(step),
        teacher_fraction=totals["teacher"] / samples,
        reentry_requests=reentry,
        unexpected_activations=unexpected,
        round_diagnostics=tuple(round_diagnostics),
        grad_norm=grad_norm,
        samples=samples,
        seconds=elapsed,
    )


@dataclass(frozen=True)
class Stage4MicroBatchTrial:
    micro_batch: int
    passed: bool
    images_per_second: float
    peak_reserved_bytes: int
    peak_reserved_fraction: float
    completed_forward_backward: int
    error: str | None = None


def _synthetic_probe_batch(micro_batch: int, crop_size: int, device: torch.device) -> Stage4Batch:
    image = torch.rand(micro_batch, 3, crop_size, crop_size, device=device)
    target = torch.rand_like(image)
    guard = torch.rand(
        micro_batch, len(SKILLS), crop_size // 4, crop_size // 4, device=device
    )
    presence = torch.zeros(micro_batch, len(SKILLS), device=device)
    presence[:, :2] = 1.0
    present = torch.tensor((0, 1), device=device).expand(micro_batch, -1).clone()
    dense = torch.tensor(
        [name in {"rain", "haze", "low_light"} for name in SKILLS],
        device=device,
    ).expand(micro_batch, -1)
    relation_row = torch.full((micro_batch,), PAIR_TO_ROW[(0, 1)], device=device)
    return Stage4Batch(
        input=image,
        target=target,
        gt_clean=target,
        target_after_i=target,
        target_after_j=target,
        only_i=target,
        only_j=target,
        guard_targets=guard,
        global_severity_targets=guard.mean((-2, -1)),
        presence_target=presence,
        dense_guard_mask=dense,
        global_guard_mask=~dense,
        present_skill_ids=present,
        forced_skill_mask=torch.zeros_like(presence, dtype=torch.bool),
        use_teacher=torch.ones(micro_batch, device=device, dtype=torch.bool),
        relation_row=relation_row,
        relation_label=torch.zeros(micro_batch, device=device, dtype=torch.long),
        relation_weight=torch.ones(micro_batch, device=device),
        relation_ambiguous=torch.zeros(micro_batch, device=device, dtype=torch.bool),
        episode_types=tuple("group_a_pair_restoration" for _ in range(micro_batch)),
    )


def choose_stage4_micro_batch(
    model: GraphRestore,
    *,
    device: torch.device,
    candidates: Sequence[int] = (2, 1),
    crop_size: int = 160,
    required_forward_backward: int = 10,
    maximum_reserved_fraction: float = 0.90,
) -> tuple[int, tuple[Stage4MicroBatchTrial, ...]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Stage4ContractError("Stage4 micro-batch selection requires CUDA")
    if (
        tuple(candidates) != (2, 1)
        or crop_size != 160
        or required_forward_backward != 10
        or maximum_reserved_fraction != 0.90
    ):
        raise Stage4ContractError("Stage4 VRAM-probe contract drifted")
    rng = capture_rng_state()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    trials: list[Stage4MicroBatchTrial] = []
    model.train()
    try:
        for micro_batch in candidates:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            completed = 0
            error: str | None = None
            throughput = 0.0
            peak = 0
            fraction = 1.0
            started = time.perf_counter()
            try:
                batch = _synthetic_probe_batch(micro_batch, crop_size, device)
                for _ in range(required_forward_backward):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        program = run_stage4_program(model, batch)
                        image_loss = stage4_image_loss(program, batch, step=12_000)
                        planner_total = torch.stack(
                            [value.total for value in program.planner_losses]
                        ).mean()
                        loss = image_loss.total + 0.05 * planner_total
                    loss.backward()
                    if not all(
                        bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ):
                        raise FloatingPointError("non-finite Stage4 probe gradient")
                    completed += 1
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                throughput = micro_batch * completed / max(elapsed, 1.0e-9)
                passed = completed == 10 and fraction <= 0.90
                if not passed:
                    error = f"peak reserved fraction {fraction:.4f} exceeds 0.90"
            except torch.OutOfMemoryError as exc:
                peak = int(torch.cuda.max_memory_reserved(device))
                fraction = peak / total_memory
                passed = False
                error = f"CUDA OOM: {exc}"
            finally:
                model.zero_grad(set_to_none=True)
                batch = program = image_loss = planner_total = loss = None
                torch.cuda.empty_cache()
            trials.append(
                Stage4MicroBatchTrial(
                    micro_batch=micro_batch,
                    passed=passed,
                    images_per_second=throughput,
                    peak_reserved_bytes=peak,
                    peak_reserved_fraction=fraction,
                    completed_forward_backward=completed,
                    error=error,
                )
            )
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng_state(rng)
        torch.cuda.empty_cache()
    passing = [trial for trial in trials if trial.passed]
    if not passing:
        raise Stage4ContractError("no crop160 Stage4 micro batch passed the 10-step VRAM gate")
    selected = max(passing, key=lambda item: (item.images_per_second, item.micro_batch))
    return selected.micro_batch, tuple(trials)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        raise Stage4ContractError("cannot aggregate an empty Stage4 metric bucket")
    result = math.fsum(collected) / len(collected)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite Stage4 validation aggregate")
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(prediction: Tensor, target: Tensor) -> float | None:
    x = prediction.detach().float().cpu().flatten().numpy().astype(np.float64)
    y = target.detach().float().cpu().flatten().numpy().astype(np.float64)
    if x.var() < 1.0e-8 or y.var() < 1.0e-8:
        return None
    result = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    return result if math.isfinite(result) else None


@torch.inference_mode()
def validate_stage4(
    model: GraphRestore,
    dataset: GraphRestoreEpisodeDataset,
    *,
    device: torch.device,
    relation_val_records: Mapping[str, Mapping[str, Any]],
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Full primary-val GraphRestore validation; no MiO100 path is accepted."""

    if dataset.training or dataset.crop_size is not None:
        raise Stage4ContractError("Stage4 validation must be full-resolution/no augmentation")
    if any(record.group not in {"single", "A"} for record in dataset.records):
        raise Stage4ContractError("Stage4 validation contains forbidden groups")
    relation_lookup = _relation_mapping(relation_val_records)
    model.eval()
    rows: list[dict[str, Any]] = []
    presence_tp = torch.zeros(len(SKILLS), dtype=torch.float64)
    presence_fp = torch.zeros_like(presence_tp)
    presence_fn = torch.zeros_like(presence_tp)
    relation_true: list[int] = []
    relation_pred: list[int] = []
    guard_values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("rain", "haze")
    }
    reentry = unexpected = dropped = proposed_edges = precycle_graphs = program_levels = 0
    clean_examples: dict[int, Mapping[str, Any]] = {}

    for index, record in enumerate(dataset.records):
        sample = dataset[
            EpisodeRequest(index=index, episode_type="restoration", absolute_step=0)
        ]
        image = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
        target = sample["gt_clean"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with _autocast(device, use_bf16):
            output = model(image, return_trace=True, max_rounds=3)
        from src.net.graphrestore import GraphRestoreOutput

        if not isinstance(output, GraphRestoreOutput):
            raise Stage4ContractError("Stage4 validation requires GraphRestore trace")
        metric = official_psnr_ssim(
            output.final.detach().float().cpu(), target.detach().float().cpu(), quantize=True
        )
        combination = "+".join(record.operator_order)
        rows.append(
            {
                "sample_id": record.sample_id,
                "group": record.group,
                "combination": combination,
                "psnr": float(metric.psnr.item()),
                "ssim": float(metric.ssim.item()),
            }
        )
        planner0 = output.planner_outputs[0]
        predicted = planner0.presence_probabilities[0] >= model.presence_thresholds
        truth = sample["presence_target"].bool().cpu()
        predicted_cpu = predicted.detach().cpu()
        presence_tp += (predicted_cpu & truth).to(torch.float64)
        presence_fp += (predicted_cpu & ~truth).to(torch.float64)
        presence_fn += (~predicted_cpu & truth).to(torch.float64)
        for skill_name in ("rain", "haze"):
            skill_id = SKILL_TO_INDEX[skill_name]
            if not bool(truth[skill_id]):
                continue
            pred_guard = planner0.spatial_guard_probabilities[0, skill_id]
            gt_guard = sample["guard_targets"][skill_id].to(pred_guard)
            correlation = _spearman(pred_guard, gt_guard)
            if correlation is not None:
                guard_values[skill_name]["spearman"].append(correlation)
            else:
                guard_values[skill_name]["spearman_skipped"].append(1.0)
            guard_values[skill_name]["mae"].append(
                float((pred_guard - gt_guard).abs().mean().cpu())
            )
            guard_values[skill_name]["std"].append(float(pred_guard.std(unbiased=False).cpu()))
            guard_values[skill_name]["high_frac"].append(
                float((pred_guard > 0.9).float().mean().cpu())
            )

        if record.is_pair and record.sample_id in relation_lookup:
            relation = relation_lookup[record.sample_id]
            label = str(relation.get("label", ""))
            if label != "ambiguous":
                if label not in {"i_before_j", "j_before_i", "parallel"}:
                    raise Stage4ContractError("invalid interaction_val label")
                ids = tuple(sorted(record.skill_ids))
                row = PAIR_TO_ROW[ids]
                relation_true.append(("i_before_j", "j_before_i", "parallel").index(label))
                relation_pred.append(int(planner0.relation_logits[0, row].argmax()))

        for trace in output.trace:
            reentry += int(trace.reentry_request_mask.sum().item())
            unexpected += int(trace.unexpected_activation_mask.sum().item())
            if trace.execution is not None:
                program_levels += int(trace.active_mask.any(dim=1).sum().item())
        for graph in output.compiled_graphs:
            precycle_graphs += int(bool(graph.dropped_edges))
            dropped += len(graph.dropped_edges)
            proposed_edges += len(graph.edges) + len(graph.dropped_edges)
        if not record.is_pair:
            skill_id = record.skill_ids[0]
            clean_examples.setdefault(skill_id, sample)

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "count": len(selected),
            "psnr": _mean(float(row["psnr"]) for row in selected),
            "ssim": _mean(float(row["ssim"]) for row in selected),
        }

    single_rows = [row for row in rows if row["group"] == "single"]
    pair_rows = [row for row in rows if row["group"] == "A"]
    pair_names = sorted({str(row["combination"]) for row in pair_rows})
    if len(pair_names) != 8:
        raise Stage4ContractError("primary_val lacks eight Group-A combinations")
    group_a_tasks = {
        name: aggregate([row for row in pair_rows if row["combination"] == name])
        for name in pair_names
    }
    group_a = {
        "count": len(pair_rows),
        "combination_count": 8,
        "psnr": _mean(float(row["psnr"]) for row in group_a_tasks.values()),
        "ssim": _mean(float(row["ssim"]) for row in group_a_tasks.values()),
    }
    single_tasks = {
        name: aggregate([row for row in single_rows if row["combination"] == name])
        for name in sorted({str(row["combination"]) for row in single_rows})
    }
    single_equal = {
        "count": len(single_rows),
        "task_count": len(single_tasks),
        "psnr": _mean(float(row["psnr"]) for row in single_tasks.values()),
        "ssim": _mean(float(row["ssim"]) for row in single_tasks.values()),
    }

    precision = presence_tp / (presence_tp + presence_fp).clamp_min(1.0)
    recall = presence_tp / (presence_tp + presence_fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    relation_accuracy = (
        sum(a == b for a, b in zip(relation_true, relation_pred, strict=True))
        / len(relation_true)
        if relation_true
        else 0.0
    )
    parallel_tp = sum(a == 2 and b == 2 for a, b in zip(relation_true, relation_pred, strict=True))
    parallel_fp = sum(a != 2 and b == 2 for a, b in zip(relation_true, relation_pred, strict=True))
    parallel_fn = sum(a == 2 and b != 2 for a, b in zip(relation_true, relation_pred, strict=True))

    # Fixed, bounded identity diagnostics: one clean and one wrong-skill call
    # per true single skill.  They use the same selected EMA snapshot.
    clean_metric_rows: list[tuple[float, float, float]] = []
    wrong_metric_rows: list[tuple[float, float, float]] = []
    for true_skill, sample in sorted(clean_examples.items()):
        clean = sample["gt_clean"].unsqueeze(0).to(device=device, dtype=torch.float32)
        degraded = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
        force_clean = torch.zeros(1, len(SKILLS), device=device, dtype=torch.bool)
        force_clean[0, true_skill] = True
        wrong_skill = (true_skill + 1) % len(SKILLS)
        force_wrong = torch.zeros_like(force_clean)
        force_wrong[0, wrong_skill] = True
        with _autocast(device, use_bf16):
            clean_out = model(clean, forced_counterfactual_mask=force_clean)
            wrong_out = model(degraded, forced_counterfactual_mask=force_wrong)
        if not torch.is_tensor(clean_out) or not torch.is_tensor(wrong_out):
            raise Stage4ContractError("counterfactual validation returned trace unexpectedly")
        for prediction, target_value, sink in (
            (clean_out, clean, clean_metric_rows),
            (wrong_out, degraded, wrong_metric_rows),
        ):
            metric = official_psnr_ssim(
                prediction.detach().float().cpu(), target_value.detach().float().cpu(), quantize=True
            )
            residual = float((prediction.float() - target_value).square().mean().sqrt().cpu())
            sink.append((float(metric.psnr.item()), float(metric.ssim.item()), residual))

    diagnostics: dict[str, Any] = {
        "planner_macro_f1": float(f1.mean()),
        "per_skill_f1": {name: float(f1[index]) for index, name in enumerate(SKILLS)},
        "relation_accuracy": relation_accuracy,
        "relation_n_nonambiguous": len(relation_true),
        "parallel_precision": parallel_tp / max(parallel_tp + parallel_fp, 1),
        "parallel_recall": parallel_tp / max(parallel_tp + parallel_fn, 1),
        "pre_cycle_rate": precycle_graphs / len(rows),
        "post_cycle_rate": 0.0,
        "dropped_edge_rate": dropped / proposed_edges if proposed_edges else 0.0,
        "reentry_request_rate": reentry / max(len(rows) * 3 * len(SKILLS), 1),
        "unexpected_skill_activation_rate": unexpected / max(len(rows) * 3 * len(SKILLS), 1),
        "mean_program_levels": program_levels / len(rows),
    }
    for name in ("rain", "haze"):
        values = guard_values[name]
        diagnostics.update(
            {
                f"guard_spearman_{name}": (
                    _mean(values["spearman"]) if values["spearman"] else None
                ),
                f"guard_mae_{name}": _mean(values["mae"]),
                f"guard_std_{name}": _mean(values["std"]),
                f"guard_high_frac_{name}": _mean(values["high_frac"]),
                f"valid_guard_images_{name}": len(values["spearman"]),
                f"skipped_guard_images_{name}": len(values["spearman_skipped"]),
            }
        )

    def identity_summary(values: Sequence[tuple[float, float, float]]) -> dict[str, float]:
        return {
            "psnr": _mean(row[0] for row in values),
            "ssim": _mean(row[1] for row in values),
            "residual_norm": _mean(row[2] for row in values),
        }

    diagnostics["clean_misuse"] = identity_summary(clean_metric_rows)
    diagnostics["wrong_skill_identity"] = identity_summary(wrong_metric_rows)
    return {
        "schema_version": "graphrestore-stage4-validation-v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now_iso(),
        "dataset": "primary_val_single_and_group_a_only",
        "relation_validation_source": "interaction_val_only",
        "output_quantization": "clamp_round_uint8",
        "single_equal_task_mean": single_equal,
        "single_tasks": single_tasks,
        "group_a_equal_combination_mean": group_a,
        "group_a_combinations": group_a_tasks,
        "diagnostics": diagnostics,
        "image_count": len(rows),
    }


def stage4_validation_score(summary: Mapping[str, Any], step: int) -> ValidationScore:
    group = summary["group_a_equal_combination_mean"]
    single = summary["single_equal_task_mean"]
    return ValidationScore(
        group_a_psnr=float(group["psnr"]),
        group_a_ssim=float(group["ssim"]),
        single_psnr=float(single["psnr"]),
        single_ssim=float(single["ssim"]),
        step=step,
    )


class _null_model_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


def save_stage4_checkpoint(
    destination: str | Path,
    *,
    step: int,
    model: GraphRestore,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: Stage4EpisodeSampler,
    provenance: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    model_as_ema: bool = False,
) -> None:
    context = ema.apply_to(model) if model_as_ema else _null_model_context()
    with context:
        payload = checkpoint_payload(
            stage=STAGE4_CHECKPOINT_STAGE,
            step=step,
            model=model,
            ema_state=ema.state_dict(),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            sampler_state=sampler.state_dict(consumed_optimizer_step=step),
            provenance=provenance,
            metrics=metrics,
        )
        payload["model_role"] = "ema_selection" if model_as_ema else "raw_training_state"
        payload["resumable"] = not model_as_ema
        atomic_torch_save(payload, destination)


def resume_stage4_checkpoint(
    checkpoint: str | Path,
    *,
    model: GraphRestore,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    sampler: Stage4EpisodeSampler,
    expected_provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    # Inspect role metadata before load_checkpoint is allowed to mutate the
    # model, optimizer, scheduler, or RNG.  A selected EMA is an evaluation
    # parent, never an exact continuation point for AdamW.
    header = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(header, Mapping):
        raise Stage4ContractError("Stage4 resume checkpoint must be a mapping")
    if (
        header.get("stage") != STAGE4_CHECKPOINT_STAGE
        or header.get("model_role") != "raw_training_state"
        or header.get("resumable") is not True
    ):
        raise Stage4ContractError(
            "Stage4 resume requires resumable raw last.pth, not best_ema.pth"
        )
    payload = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_provenance=expected_provenance,
        restore_rng=True,
        map_location="cpu",
    )
    if payload.get("stage") != STAGE4_CHECKPOINT_STAGE:
        raise Stage4ContractError("resume checkpoint is not Stage4")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Stage4ContractError("invalid Stage4 resume step")
    ema_state = payload.get("ema")
    if not isinstance(ema_state, Mapping):
        raise Stage4ContractError("Stage4 resume lacks EMA")
    ema.load_state_dict(ema_state)
    sampler_state = payload.get("sampler_state")
    if not isinstance(sampler_state, Mapping):
        raise Stage4ContractError("Stage4 resume lacks sampler state")
    sampler.load_state_dict(sampler_state)
    if sampler_state.get("consumed_optimizer_step") != step:
        raise Stage4ContractError("Stage4 checkpoint/sampler step mismatch")
    set_stage4_trainability(model)
    return payload


def dependency_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "basicsr",
        "numpy",
        "opencv-python",
        "pyiqa",
        "PyYAML",
        "torch",
        "torchvision",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "torch_runtime": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy_runtime": np.__version__,
        "opencv_runtime": cv2.__version__,
        "packages": packages,
    }


def build_stage4_provenance(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    resolved_path: Path,
    resolved: Mapping[str, Any],
    stage1_checkpoint: Path,
    stage3_checkpoint: Path,
    approval: Path,
    thresholds: Path,
    pair_prior: Path,
    global_priority: Path,
    relation_train: Path,
    relation_val: Path,
    micro_batch: int,
    max_steps: int,
) -> dict[str, Any]:
    train = Path(str(resolved[config["paths"]["train_manifest_key"]])).resolve()
    val = Path(str(resolved[config["paths"]["val_manifest_key"]])).resolve()
    expected = resolved.get("expected_identity")
    if not isinstance(expected, Mapping) or not isinstance(expected.get("manifests"), Mapping):
        raise Stage4ContractError("resolved paths lacks frozen identities")
    manifest_hashes = expected["manifests"]
    actual_train, actual_val = sha256_file(train), sha256_file(val)
    if actual_train != manifest_hashes.get("primary_train") or actual_val != manifest_hashes.get("primary_val"):
        raise Stage4ContractError("Stage4 primary manifest hash mismatch")
    agenticir_commit = git_commit(Path(str(resolved["agenticir_repo"])))
    mioir_commit = git_commit(Path(str(resolved["mioir_repo"])))
    if agenticir_commit != expected.get("agenticir_commit") or mioir_commit != expected.get("mioir_commit"):
        raise Stage4ContractError("Stage4 upstream commit mismatch")
    if micro_batch not in {1, 2} or 4 % micro_batch:
        raise Stage4ContractError("invalid frozen Stage4 micro batch")
    artifacts = {
        "stage1_checkpoint": stage1_checkpoint,
        "stage3_checkpoint": stage3_checkpoint,
        "stage3_approval": approval,
        "thresholds": thresholds,
        "pair_prior": pair_prior,
        "global_priority": global_priority,
        "relation_train": relation_train,
        "relation_val": relation_val,
    }
    return {
        "schema_version": STAGE4_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": sha256_file(config_path),
        "config_semantic_sha256": sha256_json(config),
        "resolved_paths_sha256": sha256_file(resolved_path),
        "semantic_source_sha256": semantic_source_hashes(
            config_path.resolve().parents[1],
            entrypoints=("scripts/train_stage4_e2e.py",),
        ),
        "manifests": {
            "primary_train": {"path": str(train), "sha256": actual_train},
            "primary_val": {"path": str(val), "sha256": actual_val},
        },
        "parents": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "repositories": {
            "agenticir_commit": agenticir_commit,
            "mioir_commit": mioir_commit,
        },
        "runtime": {
            "crop_size": 160,
            "micro_batch": micro_batch,
            "effective_batch_size": 4,
            "accumulation_steps": 4 // micro_batch,
            "max_steps": max_steps,
            "schedule_max_steps": 40_000,
            "kmax_train": 2,
            "kmax_test": 3,
            "gradient_checkpointing": True,
            "torch_compile": False,
            "amp_dtype": "bf16",
            "tf32": True,
        },
        "dependency_versions": dependency_versions(),
    }


def append_jsonl(handle: TextIO, value: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )
    handle.flush()


def lr_by_role(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    values: dict[str, float] = {}
    for group in optimizer.param_groups:
        role = str(group.get("role", "unknown"))
        lr = float(group["lr"])
        if role in values and not math.isclose(values[role], lr, rel_tol=0.0, abs_tol=0.0):
            raise Stage4ContractError(f"decay splits disagree on {role} LR")
        values[role] = lr
    return values


__all__ = [
    "COUNTERFACTUAL_TYPES",
    "EPISODE_TYPES",
    "FrozenStage3Snapshot",
    "PROTOCOL_ID",
    "STAGE4_SCHEMA",
    "Stage4Batch",
    "Stage4ContractError",
    "Stage4EpisodeDataset",
    "Stage4EpisodeSampler",
    "Stage4ImageLoss",
    "Stage4MicroBatchTrial",
    "Stage4ProgramOutput",
    "Stage4Request",
    "Stage4RoundDiagnostics",
    "Stage4StepResult",
    "append_jsonl",
    "build_stage4_optimizer",
    "build_stage4_provenance",
    "choose_stage4_micro_batch",
    "load_presence_thresholds",
    "load_relation_records",
    "load_stage3_best_ema",
    "lr_by_role",
    "prepare_stage4_batch",
    "resume_stage4_checkpoint",
    "run_stage4_program",
    "save_stage4_checkpoint",
    "set_stage4_trainability",
    "stage4_image_loss",
    "stage4_parameter_role",
    "stage4_ssim_weight",
    "stage4_validation_score",
    "teacher_forcing_probability",
    "train_stage4_optimizer_step",
    "validate_stage3_approval",
    "validate_stage4",
    "validate_stage4_config",
]
