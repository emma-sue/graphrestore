from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.data.manifests import (
    ALLOWED_GROUP_A,
    ALLOWED_SINGLE,
    OperatorParameter,
    PrimaryRecipe,
)
from src.data.samplers import StatefulEpisodeSampler


class _FakeDataset:
    def __init__(self) -> None:
        records = []
        for task_index, order in enumerate((*ALLOWED_SINGLE, *ALLOWED_GROUP_A)):
            for sample_index in range(4):
                parameters = tuple(
                    OperatorParameter(name=name, seed=sample_index, actual={})
                    for name in order
                )
                records.append(
                    PrimaryRecipe(
                        sample_id=f"{task_index}-{sample_index}",
                        split="train",
                        clean_id=f"{task_index}-{sample_index}",
                        clean_path=Path("dummy.png"),
                        depth_path=None,
                        clean_sha256="0" * 64,
                        group="single" if len(order) == 1 else "A",
                        seed=sample_index,
                        operator_params=parameters,
                        raw={},
                    )
                )
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def set_worker_seed(self, seed: int) -> None:
        pass


def test_stage1_episode_probabilities_and_task_balance() -> None:
    dataset = _FakeDataset()
    sampler = StatefulEpisodeSampler(
        dataset,
        num_samples=12_000,
        stage="stage1",
        effective_batch_size=8,
        base_seed=2027,
    )
    requests = list(sampler)
    kinds = Counter(request.episode_type for request in requests)
    assert abs(kinds["single_skill"] / len(requests) - 0.50) < 0.03
    assert abs(kinds["pair_isolation"] / len(requests) - 0.25) < 0.03
    assert abs(kinds["pair_parallel"] / len(requests) - 0.25) < 0.03
    tasks = Counter(dataset.records[request.index].operator_order for request in requests)
    assert len(tasks) == 16
    assert min(tasks.values()) > 500
    assert max(tasks.values()) < 1000


def test_stage0_curriculum_switches_at_ten_thousand() -> None:
    dataset = _FakeDataset()
    early = StatefulEpisodeSampler(
        dataset,
        num_samples=5_000,
        stage="stage0",
        effective_batch_size=8,
        base_seed=88,
        start_step=0,
    )
    late = StatefulEpisodeSampler(
        dataset,
        num_samples=5_000,
        stage="stage0",
        effective_batch_size=8,
        base_seed=88,
        start_step=10_000,
    )
    early_single = sum(
        len(dataset.records[request.index].operator_order) == 1 for request in early
    )
    late_single = sum(
        len(dataset.records[request.index].operator_order) == 1 for request in late
    )
    assert abs(early_single / 5_000 - 0.60) < 0.04
    assert abs(late_single / 5_000 - 0.30) < 0.04


def test_stage0_boundary_is_optimizer_step_not_sample_cursor() -> None:
    dataset = _FakeDataset()
    sampler = StatefulEpisodeSampler(
        dataset,
        num_samples=80_001,
        stage="stage0",
        effective_batch_size=8,
        base_seed=17,
    )
    iterator = iter(sampler)
    for sample_cursor in range(80_000):
        request = next(iterator)
        assert request.sample_cursor == sample_cursor
        assert request.absolute_step == sample_cursor // 8
        assert request.absolute_step < 10_000
    boundary = next(iterator)
    assert boundary.sample_cursor == 80_000
    assert boundary.absolute_step == 10_000


def test_sampler_state_resume_uses_consumed_step_not_prefetch_cursor() -> None:
    dataset = _FakeDataset()
    original = StatefulEpisodeSampler(
        dataset,
        num_samples=100,
        stage="stage1",
        effective_batch_size=8,
        base_seed=123,
    )
    iterator = iter(original)
    # Simulate DataLoader prefetch advancing well beyond five batches while
    # only five optimizer steps have actually been consumed.
    prefetched = [next(iterator) for _ in range(67)]
    assert original.sample_cursor == 67
    state = original.state_dict(consumed_optimizer_step=5)
    assert state["sample_cursor"] == 40

    golden = StatefulEpisodeSampler(
        dataset,
        num_samples=100,
        stage="stage1",
        effective_batch_size=8,
        base_seed=123,
        start_step=5,
    )
    golden_iterator = iter(golden)
    expected = [next(golden_iterator) for _ in range(8)]

    resumed = StatefulEpisodeSampler(
        dataset,
        num_samples=100,
        stage="stage1",
        effective_batch_size=8,
        base_seed=123,
    )
    resumed.load_state_dict(state)
    actual_iterator = iter(resumed)
    actual = [next(actual_iterator) for _ in range(8)]
    assert prefetched
    assert actual == expected
