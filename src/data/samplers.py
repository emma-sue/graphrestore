"""Deterministic, checkpointable task and Stage1 episode sampling."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .manifests import PrimaryRecipe, task_buckets

EpisodeType = Literal[
    "stage0_restoration",
    "single_skill",
    "pair_isolation",
    "pair_parallel",
    "restoration",
]


@dataclass(frozen=True)
class EpisodeRequest:
    """A sampler-selected recipe plus an unambiguous training target mode."""

    index: int
    episode_type: EpisodeType
    active_slot: int = -1
    absolute_step: int = 0
    sample_cursor: int = 0


class _EpisodeDatasetProtocol(Protocol):
    records: Sequence[PrimaryRecipe]

    def __len__(self) -> int: ...

    def set_worker_seed(self, seed: int) -> None: ...


def _normalize_stage(stage: str | int) -> str:
    value = str(stage).lower().replace("_", "").replace("-", "")
    aliases = {
        "0": "stage0",
        "stage0": "stage0",
        "1": "stage1",
        "stage1": "stage1",
        "2": "stage2",
        "stage2": "stage2",
        "3": "stage3",
        "stage3": "stage3",
        "4": "stage4",
        "stage4": "stage4",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"unsupported training stage: {stage!r}") from exc


def _cursor_rng(base_seed: int, stage: str, sample_cursor: int) -> random.Random:
    payload = f"graphrestore:{base_seed}:{stage}:{sample_cursor}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


class StatefulEpisodeSampler(Sampler[EpisodeRequest]):
    """Sample equal-probability tasks with stateless per-step randomness.

    The sequence at an absolute optimizer/sample step depends only on
    ``(base_seed, stage, sample_cursor)``.  ``set_step`` therefore resumes exactly even
    when DataLoader prefetch advanced an older iterator beyond the last
    consumed batch.
    """

    def __init__(
        self,
        dataset: _EpisodeDatasetProtocol,
        *,
        num_samples: int,
        stage: str | int,
        effective_batch_size: int,
        base_seed: int = 2027,
        start_step: int = 0,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if effective_batch_size <= 0:
            raise ValueError("effective_batch_size must be positive")
        if base_seed < 0 or start_step < 0:
            raise ValueError("base_seed and start_step must be non-negative")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.stage = _normalize_stage(stage)
        self.effective_batch_size = int(effective_batch_size)
        self.base_seed = int(base_seed)
        self._sample_cursor = int(start_step) * self.effective_batch_size
        # Only an explicitly acknowledged consumed optimizer step is serialized.
        # DataLoader prefetch may move _sample_cursor ahead of actual training.
        self._consumed_optimizer_step = int(start_step)
        buckets = task_buckets(dataset.records)
        self._single_tasks = tuple(sorted(key for key in buckets if len(key) == 1))
        self._pair_tasks = tuple(sorted(key for key in buckets if len(key) == 2))
        self._buckets = buckets
        if len(self._single_tasks) != 8 or len(self._pair_tasks) != 8:
            raise ValueError(
                "the frozen sampler requires exactly eight single and eight Group-A tasks"
            )

    @property
    def step(self) -> int:
        return self._sample_cursor // self.effective_batch_size

    @property
    def sample_cursor(self) -> int:
        return self._sample_cursor

    def set_step(self, step: int) -> None:
        """Resume at a consumed optimizer step, not a prefetched sample index."""

        if step < 0:
            raise ValueError("step must be non-negative")
        self._consumed_optimizer_step = int(step)
        self._sample_cursor = int(step) * self.effective_batch_size

    def mark_consumed_optimizer_step(self, step: int) -> None:
        """Acknowledge the last consumed step without rewinding prefetch."""

        if step < 0:
            raise ValueError("step must be non-negative")
        self._consumed_optimizer_step = int(step)

    def state_dict(
        self, *, consumed_optimizer_step: int | None = None
    ) -> dict[str, Any]:
        if consumed_optimizer_step is not None:
            self.mark_consumed_optimizer_step(consumed_optimizer_step)
        resume_cursor = self._consumed_optimizer_step * self.effective_batch_size
        return {
            "schema_version": 1,
            "stage": self.stage,
            "base_seed": self.base_seed,
            "num_samples": self.num_samples,
            "effective_batch_size": self.effective_batch_size,
            "consumed_optimizer_step": self._consumed_optimizer_step,
            "sample_cursor": resume_cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "stage": self.stage,
            "base_seed": self.base_seed,
            "num_samples": self.num_samples,
            "effective_batch_size": self.effective_batch_size,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"sampler state {key} mismatch: expected {value!r}, "
                    f"got {state.get(key)!r}"
                )
        step = state.get("consumed_optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("sampler state has an invalid consumed optimizer step")
        expected_cursor = step * self.effective_batch_size
        if state.get("sample_cursor") != expected_cursor:
            raise ValueError("sampler state cursor is not optimizer_step*effective_batch_size")
        self._consumed_optimizer_step = step
        self._sample_cursor = expected_cursor

    def _choose_index(
        self, rng: random.Random, tasks: tuple[tuple[str, ...], ...]
    ) -> tuple[int, tuple[str, ...]]:
        task = tasks[rng.randrange(len(tasks))]
        bucket = self._buckets[task]
        return bucket[rng.randrange(len(bucket))], task

    def _request(self, sample_cursor: int) -> EpisodeRequest:
        optimizer_step = sample_cursor // self.effective_batch_size
        rng = _cursor_rng(self.base_seed, self.stage, sample_cursor)
        if self.stage == "stage0":
            single_probability = 0.60 if optimizer_step < 10_000 else 0.30
            tasks = self._single_tasks if rng.random() < single_probability else self._pair_tasks
            index, _ = self._choose_index(rng, tasks)
            return EpisodeRequest(
                index=index,
                episode_type="stage0_restoration",
                absolute_step=optimizer_step,
                sample_cursor=sample_cursor,
            )
        if self.stage == "stage1":
            draw = rng.random()
            if draw < 0.50:
                index, _ = self._choose_index(rng, self._single_tasks)
                return EpisodeRequest(
                    index=index,
                    episode_type="single_skill",
                    active_slot=0,
                    absolute_step=optimizer_step,
                    sample_cursor=sample_cursor,
                )
            index, _ = self._choose_index(rng, self._pair_tasks)
            if draw < 0.75:
                return EpisodeRequest(
                    index=index,
                    episode_type="pair_isolation",
                    active_slot=rng.randrange(2),
                    absolute_step=optimizer_step,
                    sample_cursor=sample_cursor,
                )
            return EpisodeRequest(
                index=index,
                episode_type="pair_parallel",
                absolute_step=optimizer_step,
                sample_cursor=sample_cursor,
            )
        # Later-stage scripts can consume the same balanced restoration stream
        # and layer their planner/counterfactual request policy explicitly.
        tasks = self._single_tasks if rng.random() < 0.5 else self._pair_tasks
        index, _ = self._choose_index(rng, tasks)
        return EpisodeRequest(
            index=index,
            episode_type="restoration",
            absolute_step=optimizer_step,
            sample_cursor=sample_cursor,
        )

    def __iter__(self) -> Iterator[EpisodeRequest]:
        for _ in range(self.num_samples):
            sample_cursor = self._sample_cursor
            request = self._request(sample_cursor)
            self._sample_cursor += 1
            yield request

    def __len__(self) -> int:
        return self.num_samples


def _seed_worker(worker_id: int) -> None:
    worker = torch.utils.data.get_worker_info()
    if worker is None:
        return
    seed = int(torch.initial_seed() % 2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    dataset = worker.dataset
    setter = getattr(dataset, "set_worker_seed", None)
    if callable(setter):
        setter(seed)


def build_dataloader(
    dataset: _EpisodeDatasetProtocol,
    *,
    batch_size: int,
    effective_batch_size: int,
    num_samples: int | None = None,
    stage: str | int = "stage0",
    base_seed: int = 2027,
    start_step: int = 0,
    num_workers: int = 8,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
    training: bool = True,
) -> tuple[DataLoader, StatefulEpisodeSampler | None]:
    """Build the effective V7.1 DataLoader and return its resumable sampler."""

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    sampler: StatefulEpisodeSampler | None
    if training:
        if num_samples is None:
            num_samples = len(dataset)
        sampler = StatefulEpisodeSampler(
            dataset,
            num_samples=num_samples,
            stage=stage,
            effective_batch_size=effective_batch_size,
            base_seed=base_seed,
            start_step=start_step,
        )
    else:
        sampler = None
    generator = torch.Generator(device="cpu").manual_seed(base_seed)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": drop_last if training else False,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        arguments["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**arguments)
    return loader, sampler


# Backward-compatible descriptive alias for callers that prefer curriculum.
CurriculumTaskSampler = StatefulEpisodeSampler
