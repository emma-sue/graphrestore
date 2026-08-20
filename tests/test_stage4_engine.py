from __future__ import annotations

import argparse
import copy
import contextlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
import src.training.stage4_engine as stage4_engine

from src.data.manifests import (
    ALLOWED_GROUP_A,
    ALLOWED_SINGLE,
    OperatorParameter,
    PrimaryRecipe,
    SKILLS,
)
from src.data.samplers import EpisodeRequest
from src.net import GraphRestore
from src.net.graph_compiler import CompiledGraph, PAIR_TO_ROW
from src.net.graphrestore import GraphRestoreOutput, ProgramGraphState, RoundTrace
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import WarmupCosineScheduler
from src.training.stage3_engine import (
    CALIBRATION_COLUMNS,
    STAGE3_EMA_SCOPE,
    Stage3ContractError,
    append_calibration_history as append_stage3_calibration_history,
    stage3_ema_policy_metadata,
)
from src.training.stage4_engine import (
    STAGE4_ALLOCATOR_CONF,
    STAGE4_EMA_SCOPE,
    STAGE4_SCHEMA,
    Stage4Batch,
    Stage4ContractError,
    Stage4EpisodeDataset,
    Stage4EpisodeSampler,
    Stage4ExtensionEvidence,
    Stage4PhaseAwareEMA,
    Stage4ProgramOutput,
    Stage4Request,
    build_stage4_provenance,
    build_stage4_ema,
    build_stage4_optimizer,
    choose_stage4_micro_batch,
    is_stage4_cuda_oom_exception,
    load_presence_thresholds,
    load_stage3_best_ema,
    probe_stage4_validation_vram,
    require_stage4_allocator_conf,
    resume_stage4_checkpoint,
    run_stage4_program,
    run_stage4_zero_training_diagnostics,
    save_stage4_checkpoint,
    set_stage4_trainability,
    stage4_image_loss,
    stage4_ema_policy_metadata,
    stage4_parameter_role,
    stage4_probe_candidate_order,
    stage4_runtime_evidence_metadata,
    stage4_ssim_weight,
    teacher_forcing_probability,
    train_stage4_optimizer_step,
    validate_stage3_approval,
    validate_stage3_finalization_for_stage4,
    validate_stage4_config,
)
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_model() -> GraphRestore:
    return GraphRestore(
        dim=4,
        encoder_blocks=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        refinement=1,
        heads=(1, 2, 4, 8),
        skill_bottlenecks={
            "level3": 2,
            "level2": 2,
            "level1": 2,
            "refinement": 2,
        },
        planner_fpn_dim=8,
        planner_context_dim=16,
        effect_profile_dim=4,
    )


def _runtime_evidence_binding(
    crop_size: int = 160, micro_batch: int = 2
) -> dict[str, Any]:
    return {
        "schema_version": "graphrestore-stage4-runtime-evidence-v1",
        "selected_crop_size": crop_size,
        "selected_micro_batch": micro_batch,
        "micro_batch_trials_sha256": "a" * 64,
        "validation_vram_gate_sha256": "b" * 64,
    }


def _stage4_batch(
    *,
    batch_size: int = 4,
    size: int = 32,
    episode_types: tuple[str, ...] | None = None,
    teacher: bool = True,
) -> Stage4Batch:
    torch.manual_seed(31)
    image = torch.rand(batch_size, 3, size, size)
    target = torch.rand_like(image)
    guards = torch.rand(batch_size, len(SKILLS), size // 4, size // 4)
    presence = torch.zeros(batch_size, len(SKILLS))
    presence[:, :2] = 1.0
    dense = torch.tensor(
        [name in {"rain", "haze", "low_light"} for name in SKILLS]
    ).expand(batch_size, -1)
    if episode_types is None:
        episode_types = tuple("group_a_pair_restoration" for _ in range(batch_size))
    forced = torch.zeros(batch_size, len(SKILLS), dtype=torch.bool)
    for index, value in enumerate(episode_types):
        if value == "clean_misuse":
            presence[index].zero_()
            forced[index, 3] = True
            target[index] = image[index]
        elif value == "wrong_skill":
            presence[index].zero_()
            presence[index, 0] = 1.0
            forced[index, 3] = True
            target[index] = image[index]
    return Stage4Batch(
        input=image,
        target=target,
        gt_clean=target.clone(),
        target_after_i=target.clone(),
        target_after_j=target.clone(),
        only_i=target.clone(),
        only_j=target.clone(),
        guard_targets=guards,
        global_severity_targets=guards.mean((-2, -1)),
        presence_target=presence,
        dense_guard_mask=dense,
        global_guard_mask=~dense,
        present_skill_ids=torch.tensor((0, 1)).expand(batch_size, -1).clone(),
        forced_skill_mask=forced,
        use_teacher=torch.full((batch_size,), teacher, dtype=torch.bool),
        relation_row=torch.full((batch_size,), PAIR_TO_ROW[(0, 1)], dtype=torch.long),
        relation_label=torch.zeros(batch_size, dtype=torch.long),
        relation_weight=torch.ones(batch_size),
        relation_ambiguous=torch.zeros(batch_size, dtype=torch.bool),
        episode_types=episode_types,
    )


def _raw_batch(batch: Stage4Batch) -> dict[str, Any]:
    return {
        "input": batch.input,
        "target": batch.target,
        "gt_clean": batch.gt_clean,
        "target_after_i": batch.target_after_i,
        "target_after_j": batch.target_after_j,
        "only_i": batch.only_i,
        "only_j": batch.only_j,
        "guard_targets": batch.guard_targets,
        "global_severity_targets": batch.global_severity_targets,
        "presence_target": batch.presence_target,
        "dense_guard_mask": batch.dense_guard_mask,
        "global_guard_mask": batch.global_guard_mask,
        "present_skill_ids": batch.present_skill_ids,
        "forced_skill_mask": batch.forced_skill_mask,
        "use_teacher": batch.use_teacher,
        "relation_row": batch.relation_row,
        "relation_label": batch.relation_label,
        "relation_weight": batch.relation_weight,
        "relation_ambiguous": batch.relation_ambiguous,
        "stage4_episode_type": batch.episode_types,
    }


def _record(sample_id: str, operators: tuple[str, ...]) -> PrimaryRecipe:
    return PrimaryRecipe(
        sample_id=sample_id,
        split="train",
        clean_id=f"clean-{sample_id}",
        clean_path=Path("/does/not/matter.png"),
        depth_path=None,
        clean_sha256="0" * 64,
        group="single" if len(operators) == 1 else "A",
        seed=1,
        operator_params=tuple(
            OperatorParameter(name=name, seed=index + 1, actual={})
            for index, name in enumerate(operators)
        ),
        raw={},
    )


def _fake_sampling_dataset() -> Any:
    records = []
    relation = {}
    for index, operators in enumerate(ALLOWED_SINGLE):
        records.append(_record(f"s{index}", operators))
    for index, operators in enumerate(ALLOWED_GROUP_A):
        record = _record(f"p{index}", operators)
        records.append(record)
        relation[record.sample_id] = {
            "sample_id": record.sample_id,
            "label": "parallel",
            "relation_weight": 1.0,
        }
    return SimpleNamespace(records=tuple(records), relation_records=relation)


def test_stage4_full_resolution_guard_diagnostics_crop_only_4mod8_padding() -> None:
    sampling = _fake_sampling_dataset()

    class FullResolutionDataset:
        training = False
        crop_size = None
        records = sampling.records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, request: EpisodeRequest) -> dict[str, torch.Tensor]:
            record = self.records[request.index]
            image = torch.zeros(3, 12, 20)
            presence = torch.zeros(len(SKILLS))
            presence[list(record.skill_ids)] = 1.0
            guards = torch.zeros(len(SKILLS), 3, 5)
            pattern = torch.linspace(0.0, 1.0, 15).reshape(3, 5)
            for skill_id in record.skill_ids:
                if SKILLS[skill_id] in {"rain", "haze"}:
                    guards[skill_id] = pattern
            return {
                "input": image,
                "gt_clean": image.clone(),
                "presence_target": presence,
                "guard_targets": guards,
            }

    class PaddedGuardModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("presence_thresholds", torch.full((len(SKILLS),), 0.5))

        def forward(
            self,
            image: torch.Tensor,
            *,
            return_trace: bool = False,
            **_: Any,
        ) -> torch.Tensor | GraphRestoreOutput:
            if not return_trace:
                return image
            planner = stage4_engine.PlannerOutput(
                guard_logits=torch.zeros(1, len(SKILLS), 4, 6),
                presence_logits=torch.zeros(1, len(SKILLS)),
                stop_logit=torch.zeros(1, 1),
                relation_logits=torch.zeros(1, 28, 3),
                global_context=torch.zeros(1, 1),
            )
            graph = CompiledGraph((), (), (), (), ())
            return GraphRestoreOutput(
                final=image,
                steps=(),
                planner_outputs=(planner,),
                compiled_graphs=(graph,),
                graph_states=(ProgramGraphState((), (), ()),),
                trace=(),
            )

    summary = stage4_engine.validate_stage4(
        PaddedGuardModel(),  # type: ignore[arg-type]
        FullResolutionDataset(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        relation_val_records=sampling.relation_records,
        use_bf16=False,
    )
    assert summary["image_count"] == 16
    assert summary["diagnostics"]["skipped_guard_images_rain"] == 3
    assert summary["diagnostics"]["skipped_guard_images_haze"] == 3

    with pytest.raises(ValueError, match="guard map shape mismatch"):
        stage4_engine.align_guard_prediction_to_target(
            torch.zeros(8, 5, 5), torch.zeros(8, 3, 5)
        )


def test_stage4_config_and_written_schedules_are_locked() -> None:
    config = load_yaml(PROJECT_ROOT / "configs/stage4_graphrestore_e2e.yaml")
    validate_stage4_config(config)
    drifted = copy.deepcopy(config)
    drifted["program"]["kmax_train"] = 3
    with pytest.raises(Stage4ContractError):
        validate_stage4_config(drifted)

    assert teacher_forcing_probability(0) == 1.0
    assert teacher_forcing_probability(3999) == 1.0
    assert teacher_forcing_probability(4000) == 1.0
    assert teacher_forcing_probability(11_999) == pytest.approx(0.5000625)
    # The discontinuity is explicitly preserved by the contract.
    assert teacher_forcing_probability(12_000) == 0.25
    assert teacher_forcing_probability(40_000) == 0.25
    assert stage4_ssim_weight(0) == 0.0
    assert stage4_ssim_weight(4000) == pytest.approx(0.025)
    assert stage4_ssim_weight(8000) == 0.05


def test_stage4_sampler_ratios_uniformity_and_resume() -> None:
    dataset = _fake_sampling_dataset()
    sampler = Stage4EpisodeSampler(dataset, num_samples=40_000)
    requests = [sampler._request(cursor) for cursor in range(40_000)]
    counts = Counter(request.episode_type for request in requests)
    assert counts["single_restoration"] / len(requests) == pytest.approx(0.20, abs=0.01)
    assert counts["group_a_pair_restoration"] / len(requests) == pytest.approx(
        0.70, abs=0.01
    )
    assert counts["clean_misuse"] / len(requests) == pytest.approx(0.05, abs=0.006)
    assert counts["wrong_skill"] / len(requests) == pytest.approx(0.05, abs=0.006)

    pair_counts = Counter(
        dataset.records[request.index].operator_order
        for request in requests
        if request.episode_type == "group_a_pair_restoration"
    )
    assert len(pair_counts) == 8
    assert max(pair_counts.values()) / min(pair_counts.values()) < 1.10
    for request in requests:
        if request.episode_type == "wrong_skill":
            true_id = dataset.records[request.index].skill_ids[0]
            assert request.forced_skill_ids[0] != true_id
        if request.episode_type == "clean_misuse":
            assert len(request.forced_skill_ids) in {1, 2}

    sampler.mark_consumed_optimizer_step(71)
    state = sampler.state_dict()
    restored = Stage4EpisodeSampler(dataset, num_samples=40_000)
    restored.load_state_dict(state)
    assert restored.step == 71
    assert restored._request(71 * 4) == sampler._request(71 * 4)


class _FakeBase:
    def __init__(self, records: tuple[PrimaryRecipe, ...], sample: Mapping[str, Any]):
        self.training = True
        self.crop_size = (160, 160)
        self.records = records
        self.sample = sample

    def set_worker_seed(self, _: int) -> None:
        return None

    def __getitem__(self, _: object) -> Mapping[str, Any]:
        return copy.deepcopy(self.sample)

    def __len__(self) -> int:
        return len(self.records)


def test_counterfactual_dataset_views_are_identity_targets() -> None:
    record = _record("single", ALLOWED_SINGLE[0])
    image = torch.rand(3, 160, 160)
    clean = torch.rand_like(image)
    guards = torch.rand(8, 40, 40)
    sample = {
        "input": image,
        "x_both": image,
        "target": clean,
        "gt_clean": clean,
        "target_after_i": clean,
        "target_after_j": clean,
        "only_i": image,
        "only_j": clean,
        "guard_targets": guards,
        "global_severity_targets": guards.mean((-2, -1)),
        "presence_target": torch.tensor([1.0] + [0.0] * 7),
        "dense_guard_mask": torch.tensor([False] * 4 + [True] * 3 + [False]),
        "global_guard_mask": torch.tensor([True] * 4 + [False] * 3 + [True]),
        "present_skill_ids": torch.tensor((0, -1)),
    }
    dataset = Stage4EpisodeDataset(_FakeBase((record,), sample), {"unused": {}})
    clean_episode = dataset[Stage4Request(0, "clean_misuse", 0, 0, False, (2, 5))]
    assert torch.equal(clean_episode["input"], clean)
    assert torch.equal(clean_episode["target"], clean)
    assert not bool(clean_episode["presence_target"].any())
    assert not bool(clean_episode["guard_targets"].any())
    assert clean_episode["forced_skill_mask"].sum() == 2

    wrong = dataset[Stage4Request(0, "wrong_skill", 0, 1, False, (3,))]
    assert torch.equal(wrong["input"], image)
    assert torch.equal(wrong["target"], image)
    assert wrong["presence_target"][0] == 1
    assert wrong["forced_skill_mask"][3]


def test_stage4_optimizer_roles_include_deep_downsamples_and_freeze_early_encoder() -> (
    None
):
    model = _tiny_model()
    counts = set_stage4_trainability(model)
    assert counts
    for name, parameter in model.named_parameters():
        if name.startswith(
            (
                "encoder.patch.",
                "encoder.level1.",
                "encoder.down12.",
                "encoder.level2.",
            )
        ):
            assert stage4_parameter_role(name) is None
            assert not parameter.requires_grad
        if name.startswith(
            ("encoder.down23.", "encoder.level3.", "encoder.down34.", "encoder.level4.")
        ):
            assert stage4_parameter_role(name) == "encoder34"
            assert parameter.requires_grad

    optimizer = build_stage4_optimizer(model, fused_if_supported=False)
    role_lrs: dict[str, set[float]] = {}
    assigned: set[int] = set()
    for group in optimizer.param_groups:
        role_lrs.setdefault(str(group["role"]), set()).add(float(group["lr"]))
        for parameter in group["params"]:
            assert id(parameter) not in assigned
            assigned.add(id(parameter))
        if float(group["weight_decay"]) not in {0.0, 1.0e-4}:
            raise AssertionError("unexpected Stage4 decay")
    assert role_lrs == {
        "planner": {5.0e-5},
        "skills_mixers": {3.0e-5},
        "decoder_refine_head": {1.0e-5},
        "encoder34": {2.0e-6},
    }
    assert assigned == {
        id(value) for value in model.parameters() if value.requires_grad
    }


def test_stage4_phase_aware_ema_averages_only_trainable_and_copies_fixed() -> None:
    model = _tiny_model()
    set_stage4_trainability(model)
    ema = build_stage4_ema(model, decay=0.5)
    assert isinstance(ema, Stage4PhaseAwareEMA)
    trainable_name, trainable = next(
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    )
    frozen_name, frozen = next(
        (name, value)
        for name, value in model.named_parameters()
        if not value.requires_grad
    )
    trainable_before = ema.shadow[trainable_name].clone()
    with torch.no_grad():
        trainable.add_(2.0)
        frozen.add_(3.0)
        model.presence_thresholds.fill_(0.375)
    ema.update(model)
    assert torch.equal(
        ema.shadow[trainable_name],
        trainable_before.mul(0.5).add(trainable.detach().float(), alpha=0.5),
    )
    assert torch.equal(ema.shadow[frozen_name], frozen.detach().float())
    assert torch.equal(ema.shadow["presence_thresholds"], model.presence_thresholds)
    state = ema.state_dict()
    assert state["scope"] == STAGE4_EMA_SCOPE
    assert state["policy"] == stage4_ema_policy_metadata(0.5)


def test_stage4_allocator_and_probe_fallback_order_are_fail_closed() -> None:
    assert (
        require_stage4_allocator_conf(
            {"PYTORCH_CUDA_ALLOC_CONF": STAGE4_ALLOCATOR_CONF}
        )
        == STAGE4_ALLOCATOR_CONF
    )
    for environment in ({}, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}):
        with pytest.raises(Stage4ContractError, match="PYTORCH_CUDA_ALLOC_CONF"):
            require_stage4_allocator_conf(environment)
    assert stage4_probe_candidate_order() == (
        (160, 2),
        (160, 1),
        (128, 2),
        (128, 1),
    )
    assert is_stage4_cuda_oom_exception(torch.cuda.OutOfMemoryError("oom"))
    assert is_stage4_cuda_oom_exception(RuntimeError("CUDA out of memory. Tried"))
    assert not is_stage4_cuda_oom_exception(RuntimeError("CUDA launch failed"))
    assert not is_stage4_cuda_oom_exception(FloatingPointError("non-finite"))


@pytest.mark.parametrize(
    "failure",
    (RuntimeError("CUDA launch failed"), FloatingPointError("non-finite loss")),
)
def test_stage4_probe_non_oom_failure_never_tries_a_fallback(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    model = _tiny_model()
    calls = 0

    def fail_batch(*args: Any, **kwargs: Any) -> Stage4Batch:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(stage4_engine, "capture_rng_state", lambda: {"sentinel": True})
    monkeypatch.setattr(stage4_engine, "restore_rng_state", lambda _: None)
    monkeypatch.setattr(stage4_engine, "build_stage4_optimizer", lambda _: object())
    monkeypatch.setattr(stage4_engine, "_synthetic_probe_batch", fail_batch)
    with pytest.raises(type(failure), match=str(failure)):
        choose_stage4_micro_batch(model, device=torch.device("cuda"))
    assert calls == 1


def test_stage4_probe_refuses_crop128_after_non_oom_crop160_micro1_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model()
    attempted_crops: list[int] = []

    def fake_batch(micro_batch: int, crop_size: int, device: torch.device) -> object:
        del micro_batch, device
        attempted_crops.append(crop_size)
        return object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 950)
    monkeypatch.setattr(stage4_engine, "capture_rng_state", lambda: {"sentinel": True})
    monkeypatch.setattr(stage4_engine, "restore_rng_state", lambda _: None)
    monkeypatch.setattr(stage4_engine, "build_stage4_optimizer", lambda _: object())
    monkeypatch.setattr(
        stage4_engine, "build_stage4_ema", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(stage4_engine, "_synthetic_probe_batch", fake_batch)
    monkeypatch.setattr(stage4_engine, "_stage4_batch_as_mapping", lambda _: {})
    monkeypatch.setattr(
        stage4_engine, "train_stage4_optimizer_step", lambda *args, **kwargs: None
    )
    with pytest.raises(Stage4ContractError, match="crop128 fallback"):
        choose_stage4_micro_batch(model, device=torch.device("cuda"))
    assert len(attempted_crops) == 60
    assert set(attempted_crops) == {160}


def test_stage4_probe_allows_crop128_only_after_crop160_cuda_ooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model()
    attempted_crops: list[int] = []

    def fake_batch(micro_batch: int, crop_size: int, device: torch.device) -> int:
        del micro_batch, device
        attempted_crops.append(crop_size)
        return crop_size

    def fake_step(
        model: object, micro_batches: list[int], *args: Any, **kwargs: Any
    ) -> None:
        del model, args, kwargs
        if micro_batches[0] == 160:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 500)
    monkeypatch.setattr(stage4_engine, "capture_rng_state", lambda: {"sentinel": True})
    monkeypatch.setattr(stage4_engine, "restore_rng_state", lambda _: None)
    monkeypatch.setattr(stage4_engine, "build_stage4_optimizer", lambda _: object())
    monkeypatch.setattr(
        stage4_engine, "build_stage4_ema", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(stage4_engine, "_synthetic_probe_batch", fake_batch)
    monkeypatch.setattr(stage4_engine, "_stage4_batch_as_mapping", lambda value: value)
    monkeypatch.setattr(stage4_engine, "train_stage4_optimizer_step", fake_step)
    crop, micro, trials = choose_stage4_micro_batch(model, device=torch.device("cuda"))
    assert (crop, micro) == (128, 2)
    assert [(trial.crop_size, trial.micro_batch, trial.passed) for trial in trials] == [
        (160, 2, False),
        (160, 1, False),
        (128, 2, True),
    ]
    assert 128 in attempted_crops


def test_stage4_validation_vram_gate_covers_serial_and_parallel_topologies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGateModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.planner = torch.nn.Linear(1, 1)
            self.compiler = SimpleNamespace(mode="full_partial_order")
            self.called_modes: list[str] = []

        def eval(self) -> FakeGateModel:
            super().train(False)
            return self

        def train(self, mode: bool = True) -> FakeGateModel:
            super().train(mode)
            return self

        def __call__(self, image: torch.Tensor, **_: Any) -> GraphRestoreOutput:
            mode = self.compiler.mode
            self.called_modes.append(mode)
            active_skills = tuple(SKILLS[:3])
            if mode == "forced_total_order":
                levels = tuple((skill,) for skill in active_skills)
                masks = []
                for index in range(3):
                    mask = torch.zeros(1, len(SKILLS), dtype=torch.bool)
                    mask[0, index] = True
                    masks.append(mask)
            elif mode == "parallel_only":
                levels = (active_skills,)
                mask = torch.zeros(1, len(SKILLS), dtype=torch.bool)
                mask[0, :3] = True
                masks = [mask]
            else:
                raise AssertionError(mode)
            graph = CompiledGraph(active_skills, levels, (), (), ())
            traces = tuple(
                RoundTrace(
                    round_index=index,
                    active_mask=mask,
                    stopped_mask=torch.zeros(1, dtype=torch.bool),
                    skipped_mask=torch.zeros_like(mask),
                    stop_reasons=("gate",),
                    reentry_request_mask=torch.zeros_like(mask),
                    unexpected_activation_mask=torch.zeros_like(mask),
                    execution=None,
                )
                for index, mask in enumerate(masks)
            )
            return GraphRestoreOutput(
                final=image,
                steps=tuple(image for _ in traces),
                planner_outputs=(),
                compiled_graphs=(graph,),
                graph_states=(
                    ProgramGraphState(
                        nodes=active_skills,
                        edges=(),
                        levels=levels,
                    ),
                ),
                trace=traces,
            )

    real_rand = torch.rand
    peaks = [400, 600]
    model = FakeGateModel()
    optimizer = torch.optim.AdamW(model.parameters())
    ema = build_stage4_ema(model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: peaks.pop(0))
    monkeypatch.setattr(
        stage4_engine.torch,
        "rand",
        lambda *args, **kwargs: real_rand(1, 3, 16, 16),
    )
    monkeypatch.setattr(
        stage4_engine.torch,
        "autocast",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(stage4_engine, "capture_rng_state", lambda: {"sentinel": True})
    monkeypatch.setattr(stage4_engine, "restore_rng_state", lambda _: None)
    monkeypatch.setattr(
        stage4_engine,
        "official_psnr_ssim",
        lambda *args, **kwargs: SimpleNamespace(
            psnr=torch.tensor([1.0]), ssim=torch.tensor([0.5])
        ),
    )
    gate = probe_stage4_validation_vram(
        model,  # type: ignore[arg-type]
        optimizer=optimizer,
        ema=ema,
        device=torch.device("cuda"),
    )
    assert gate.peak_reserved_bytes == 600
    assert gate.peak_reserved_fraction == 0.6
    assert gate.passed is True
    assert gate.resident_optimizer_state_entries == 2
    assert gate.resident_optimizer_state_bytes > 0
    assert gate.resident_ema_bytes > 0
    assert gate.optimizer_state_empty_after is True
    assert not optimizer.state
    assert [trial.compiler_mode for trial in gate.topologies] == [
        "forced_total_order",
        "parallel_only",
    ]
    assert gate.topologies[0].active_skill_counts_by_round == (1, 1, 1)
    assert gate.topologies[1].active_skill_counts_by_round == (3,)
    assert model.called_modes == ["forced_total_order", "parallel_only"]
    assert model.compiler.mode == "full_partial_order"
    assert model.training is True


def test_stage4_runtime_evidence_is_strict_and_hash_bound() -> None:
    trials = [
        {
            "crop_size": 160,
            "micro_batch": 2,
            "passed": True,
            "images_per_second": 8.0,
            "peak_reserved_bytes": 600,
            "peak_reserved_fraction": 0.60,
            "completed_forward_backward": 10,
            "completed_optimizer_steps": 10,
            "error": None,
        }
    ]
    gate = {
        "image_size": 2040,
        "max_rounds": 3,
        "completed_rounds": 3,
        "topologies": [
            {
                "compiler_mode": "forced_total_order",
                "active_skill_count": 3,
                "completed_rounds": 3,
                "active_skill_counts_by_round": [1, 1, 1],
                "peak_reserved_bytes": 500,
                "peak_reserved_fraction": 0.50,
                "finite": True,
                "passed": True,
            },
            {
                "compiler_mode": "parallel_only",
                "active_skill_count": 3,
                "completed_rounds": 1,
                "active_skill_counts_by_round": [3],
                "peak_reserved_bytes": 600,
                "peak_reserved_fraction": 0.60,
                "finite": True,
                "passed": True,
            },
        ],
        "peak_reserved_bytes": 600,
        "peak_reserved_fraction": 0.60,
        "maximum_peak_reserved_fraction": 0.90,
        "resident_optimizer_state_entries": 4,
        "resident_optimizer_state_bytes": 1024,
        "resident_ema_bytes": 512,
        "optimizer_state_empty_after": True,
        "finite": True,
        "passed": True,
    }
    metadata = stage4_runtime_evidence_metadata(
        trials,
        gate,
        selected_crop_size=160,
        selected_micro_batch=2,
    )
    assert metadata["schema_version"] == "graphrestore-stage4-runtime-evidence-v1"
    assert len(metadata["micro_batch_trials_sha256"]) == 64
    assert len(metadata["validation_vram_gate_sha256"]) == 64

    missing_trial_field = copy.deepcopy(trials)
    del missing_trial_field[0]["completed_optimizer_steps"]
    with pytest.raises(Stage4ContractError, match="trial schema"):
        stage4_runtime_evidence_metadata(
            missing_trial_field,
            gate,
            selected_crop_size=160,
            selected_micro_batch=2,
        )
    forged_gate = copy.deepcopy(gate)
    forged_gate["topologies"][1]["peak_reserved_fraction"] = 0.91
    forged_gate["peak_reserved_fraction"] = 0.91
    with pytest.raises(Stage4ContractError, match="topology values"):
        stage4_runtime_evidence_metadata(
            trials,
            forged_gate,
            selected_crop_size=160,
            selected_micro_batch=2,
        )


def test_stage4_program_compiles_once_and_never_recomputes_relations() -> None:
    model = _tiny_model()
    batch = _stage4_batch(batch_size=4, teacher=True)
    compiler_calls = 0
    relation_flags: list[bool] = []
    original_compile = model.compiler.compile
    original_plan = model.plan_state

    def counted_compile(*args: Any, **kwargs: Any):
        nonlocal compiler_calls
        compiler_calls += 1
        return original_compile(*args, **kwargs)

    def counted_plan(*args: Any, **kwargs: Any):
        relation_flags.append(bool(kwargs["compute_relations"]))
        return original_plan(*args, **kwargs)

    model.compiler.compile = counted_compile  # type: ignore[method-assign]
    model.plan_state = counted_plan  # type: ignore[method-assign]
    output = run_stage4_program(model, batch)
    assert compiler_calls == batch.batch_size
    assert relation_flags == [True, False]
    assert len(output.compiled_graphs) == batch.batch_size
    for compiled, state in zip(
        output.compiled_graphs, output.graph_states, strict=True
    ):
        assert state.nodes == compiled.active_skills
        assert state.edges == compiled.edges
        assert state.levels == compiled.levels
        assert not state.pending
        assert state.executed == set(compiled.active_skills)
    assert all(mask[:, 2:].sum() == 0 for mask in output.executed_masks)
    assert len(output.round_diagnostics) == 2
    assert output.round_diagnostics[0].active_skill_counts[:2] == (4, 0)
    assert output.round_diagnostics[1].active_skill_counts[:2] == (0, 4)
    assert all(
        math.isfinite(value)
        for diagnostic in output.round_diagnostics
        for value in (
            diagnostic.union_guard_mean,
            diagnostic.union_guard_std,
            diagnostic.union_guard_high_fraction,
            diagnostic.rgb_residual_norm,
            diagnostic.identity_fraction,
        )
    )


def test_stage4_ambiguous_teacher_relation_never_chooses_a_pseudo_label() -> None:
    model = _tiny_model()
    batch = _stage4_batch(batch_size=4)
    batch = Stage4Batch(
        **{
            **batch.__dict__,
            "relation_label": torch.full((4,), -1, dtype=torch.long),
            "relation_weight": torch.full((4,), 0.25),
            "relation_ambiguous": torch.ones(4, dtype=torch.bool),
        }
    )
    output = run_stage4_program(model, batch)
    for graph in output.compiled_graphs:
        decision = graph.pair_decisions[0]
        assert decision.probabilities[0] == pytest.approx(decision.probabilities[1])
        assert decision.decision_source in {"pair_prior", "global_priority"}


def test_stage4_stops_after_one_parallel_level_without_replanning() -> None:
    model = _tiny_model()
    batch = _stage4_batch(batch_size=4)
    batch.relation_label.fill_(2)
    relation_flags: list[bool] = []
    original_plan = model.plan_state

    def counted_plan(*args: Any, **kwargs: Any):
        relation_flags.append(bool(kwargs["compute_relations"]))
        return original_plan(*args, **kwargs)

    model.plan_state = counted_plan  # type: ignore[method-assign]
    output = run_stage4_program(model, batch)
    assert relation_flags == [True]
    assert len(output.step_images) == 1
    assert len(output.planner_losses) == 1
    assert all(state.complete for state in output.graph_states)


def test_stage4_loss_and_cpu_optimizer_step_are_finite() -> None:
    model = _tiny_model()
    batch = _stage4_batch(
        batch_size=4,
        episode_types=(
            "single_restoration",
            "group_a_pair_restoration",
            "clean_misuse",
            "wrong_skill",
        ),
    )
    # Single teacher targets must contain one true skill, and its relation row
    # is unsupervised.  Keep the other three rows representative.
    batch.presence_target[0].zero_()
    batch.presence_target[0, 0] = 1.0
    batch.present_skill_ids[0] = torch.tensor((0, -1))
    batch.relation_row[0] = -1
    batch.relation_label[0] = -2
    batch.relation_weight[0] = 0.0
    batch.relation_row[2:] = -1
    batch.relation_label[2:] = -2
    batch.relation_weight[2:] = 0.0
    output = run_stage4_program(model, batch)
    image = stage4_image_loss(output, batch, step=4000)
    assert image.lambda_ssim == pytest.approx(0.025)
    assert torch.isfinite(image.total)
    assert image.noop_pixel > 0

    optimizer = build_stage4_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=800, max_steps=40_000, min_lr=5.0e-7
    )
    ema = build_stage4_ema(model, decay=0.9999)
    result = train_stage4_optimizer_step(
        model,
        [_raw_batch(batch)],
        optimizer,
        scheduler,
        ema,
        step=4000,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    assert result.samples == 4
    assert math.isfinite(result.loss)
    assert math.isfinite(result.grad_norm)
    assert result.lambda_ssim == pytest.approx(0.025)
    assert ema.num_updates == 1
    assert result.round_diagnostics
    assert "active_skills" in result.round_diagnostics[0]


def test_stage4_ssim_branches_stay_fp32_under_bf16_autocast() -> None:
    batch = _stage4_batch(
        batch_size=2,
        size=16,
        episode_types=("single_restoration", "clean_misuse"),
    )
    batch.input.fill_(0.5)
    batch.input[..., ::2, ::2] += 0.01
    batch.gt_clean.copy_(batch.input)

    def image_loss(prediction: torch.Tensor):
        program = Stage4ProgramOutput(
            final=prediction,
            step_images=(),
            step_targets=(),
            step_valid_masks=(),
            planner_losses=(),
            compiled_graphs=(),
            graph_states=(),
            teacher_flags=(),
            executed_masks=(),
            round_diagnostics=(),
            reentry_request_count=0,
            unexpected_activation_count=0,
        )
        return stage4_image_loss(program, batch, step=8000)

    prediction = torch.full(
        (2, 3, 16, 16), 0.5, dtype=torch.bfloat16, requires_grad=True
    )
    reference = image_loss(prediction.detach().float())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = image_loss(prediction)

    torch.testing.assert_close(
        result.final_ssim, reference.final_ssim, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result.noop_ssim, reference.noop_ssim, rtol=0.0, atol=0.0
    )
    assert result.final_ssim.dtype == torch.float32
    assert result.noop_ssim.dtype == torch.float32
    assert result.total.dtype == torch.float32
    assert bool(torch.isfinite(result.total))
    assert float(result.final_ssim) >= 0.0
    assert float(result.noop_ssim) >= 0.0
    result.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_stage4_finalization_preflight_wraps_shared_cpu_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = SimpleNamespace(
        path=tmp_path / "artifacts/approvals/STAGE3_EXTENSION_REVOKED.json",
        sha256="a" * 64,
        bindings={"historical_extension_authorization": {"sha256": "b" * 64}},
    )
    report = tmp_path / "STAGE3_PLANNER_GUARD.md"
    report.write_text(
        "\n".join(
            (
                "graphrestore-v7.1-agenticir-locked",
                authorization.sha256,
                "d" * 64,
                "step12000_finalize_only_no_training",
                "optimizer / scheduler / train loader created: false / false / false",
                "checkpoint written: false",
                "MiO100 / Group B / Group C rows read: 0 / 0 / 0",
                "learned raw relation accuracy",
                "always-parallel baseline accuracy",
                "per-pair majority-prior baseline accuracy",
                "STOP-rate definition",
            )
        ),
        encoding="utf-8",
    )
    expected = {
        "complete": {"path": "/complete.json", "sha256": "c" * 64},
        "report": {"path": str(report), "sha256": sha256_file(report)},
        "best_checkpoint": {"path": "/best.pth", "sha256": "d" * 64},
    }
    monkeypatch.setattr(
        stage4_engine,
        "validate_stage3_extension_revocation",
        lambda *_args, **_kwargs: authorization,
    )
    monkeypatch.setattr(
        stage4_engine,
        "validate_stage3_finalization_outputs",
        lambda *_args, **_kwargs: expected,
    )
    loaded_authorization, loaded_outputs = validate_stage3_finalization_for_stage4(
        tmp_path
    )
    assert loaded_authorization is authorization
    assert loaded_outputs is expected

    report.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(Stage4ContractError, match="scientific disclosures"):
        validate_stage3_finalization_for_stage4(tmp_path)

    monkeypatch.setattr(
        stage4_engine,
        "validate_stage3_finalization_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("stale")),
    )
    with pytest.raises(Stage4ContractError, match="incomplete or stale"):
        validate_stage3_finalization_for_stage4(tmp_path)


def test_stage4_provenance_rehashes_finalization_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    config_path = root / "configs/stage4_graphrestore_e2e.yaml"
    resolved_path = root / "configs/resolved_paths.yaml"
    train = root / "primary_train.jsonl"
    val = root / "primary_val.jsonl"
    artifacts = {
        name: root / f"{name}.bin"
        for name in (
            "stage1_checkpoint",
            "stage3_checkpoint",
            "approval",
            "thresholds",
            "pair_prior",
            "global_priority",
            "relation_train",
            "relation_val",
            "complete",
            "diagnostic",
        )
    }
    extension = root / "artifacts/approvals/STAGE3_EXTENSION_APPROVED.json"
    finalization = root / "artifacts/approvals/STAGE3_EXTENSION_REVOKED.json"
    for path in (
        config_path,
        resolved_path,
        train,
        val,
        extension,
        finalization,
        *artifacts.values(),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{path.name}\n".encode())
    extension_binding = {
        "path": str(extension.resolve()),
        "sha256": sha256_file(extension),
        "cycles": 3,
        "base_step": 12_000,
        "target_step": 18_000,
        "validation_every_steps": 2_000,
        "validation_steps": [14_000, 16_000, 18_000],
        "schedule_horizon_steps": 12_000,
        "min_lr": 2.0e-6,
        "lr_policy": "hold_original_cosine_floor_after_schedule_horizon",
    }
    finalization_binding = {
        "path": str(finalization.resolve()),
        "sha256": sha256_file(finalization),
    }
    monkeypatch.setattr(
        stage4_engine,
        "validate_stage3_extension_revocation",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha256=finalization_binding["sha256"]
        ),
    )
    monkeypatch.setattr(
        stage4_engine, "stage4_runtime_evidence_metadata", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(stage4_engine, "semantic_source_hashes", lambda *_a, **_k: {})
    monkeypatch.setattr(stage4_engine, "dependency_versions", lambda: {})
    monkeypatch.setattr(
        stage4_engine,
        "git_commit",
        lambda path: "agenticir" if path.name == "agenticir" else "mioir",
    )
    config = {
        "paths": {"train_manifest_key": "train", "val_manifest_key": "val"},
        "ema": {"decay": 0.9999},
    }
    resolved = {
        "train": str(train),
        "val": str(val),
        "agenticir_repo": str(root / "agenticir"),
        "mioir_repo": str(root / "mioir"),
        "expected_identity": {
            "manifests": {
                "primary_train": sha256_file(train),
                "primary_val": sha256_file(val),
            },
            "agenticir_commit": "agenticir",
            "mioir_commit": "mioir",
        },
    }
    kwargs = {
        "config_path": config_path,
        "config": config,
        "resolved_path": resolved_path,
        "resolved": resolved,
        "stage1_checkpoint": artifacts["stage1_checkpoint"],
        "stage3_checkpoint": artifacts["stage3_checkpoint"],
        "approval": artifacts["approval"],
        "thresholds": artifacts["thresholds"],
        "pair_prior": artifacts["pair_prior"],
        "global_priority": artifacts["global_priority"],
        "relation_train": artifacts["relation_train"],
        "relation_val": artifacts["relation_val"],
        "crop_size": 128,
        "micro_batch": 1,
        "max_steps": 40_000,
        "allocator_conf": STAGE4_ALLOCATOR_CONF,
        "frozen_parent_state_sha256": "f" * 64,
        "micro_batch_trials": {},
        "validation_vram_gate": {},
        "stage3_extension": extension_binding,
        "stage3_finalization": finalization_binding,
        "stage3_complete": artifacts["complete"],
        "stage3_calibrated_diagnostic": artifacts["diagnostic"],
        "stage3_complete_sha256": sha256_file(artifacts["complete"]),
        "stage3_calibrated_diagnostic_sha256": sha256_file(artifacts["diagnostic"]),
        "stage3_thresholds_sha256": sha256_file(artifacts["thresholds"]),
    }
    provenance = build_stage4_provenance(**kwargs)
    assert provenance["stage3_finalization"] == finalization_binding
    assert (
        provenance["parents"]["stage3_complete"]["sha256"]
        == kwargs["stage3_complete_sha256"]
    )

    conditional = (
        root / "artifacts/approvals/STAGE4_EXTENSION_CONDITIONAL_APPROVED.json"
    )
    gate = root / "artifacts/approvals/STAGE4_EXTENSION_GATE_RECEIPT.json"
    conditional.write_text("{}\n", encoding="utf-8")
    gate.write_text("{}\n", encoding="utf-8")
    stage4_extension = Stage4ExtensionEvidence(
        conditional_path=conditional.resolve(),
        conditional_sha256=sha256_file(conditional),
        gate_path=gate.resolve(),
        gate_sha256=sha256_file(gate),
    )
    extended_provenance = build_stage4_provenance(
        **(kwargs | {"max_steps": 48_000, "stage4_extension": stage4_extension})
    )
    assert extended_provenance["stage4_extension"] == (
        stage4_extension.provenance_binding()
    )
    assert set(extended_provenance["parents"]) == set(provenance["parents"])

    artifacts["diagnostic"].write_text("drift\n", encoding="utf-8")
    with pytest.raises(Stage4ContractError, match="diagnostic hash drifted"):
        build_stage4_provenance(**kwargs)


def test_approval_threshold_parent_ema_and_hash_binding(tmp_path: Path) -> None:
    decision = tmp_path / "stage2_decision.json"
    atomic_write_json(decision, {"approved": False, "schema": "stage2"})
    approval = tmp_path / "STAGE3_APPROVED.json"
    atomic_write_json(
        approval,
        {
            "schema_version": "graphrestore-stage3-approval-v1",
            "kind": "stage3_approval",
            "protocol_id": "graphrestore-v7.1-agenticir-locked",
            "approved": True,
            "stage2_decision_sha256": sha256_file(decision),
        },
    )
    validate_stage3_approval(approval, stage2_decision_path=decision)
    approval_sha = sha256_file(approval)

    model = _tiny_model()
    ema = ExponentialMovingAverage(model)
    ema_state = ema.state_dict()
    ema_state["scope"] = STAGE3_EMA_SCOPE
    ema_state["policy"] = stage3_ema_policy_metadata(0.9999)
    ema_state["num_updates"] = 12_000
    checkpoint = tmp_path / "best_ema.pth"
    valid_payload = {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage3",
        "step": 12_000,
        "model_role": "ema_selection",
        "resumable": False,
        "pending_validation_step": None,
        "optimizer_transaction_active": False,
        "scaler": None,
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "model": ema.shadow,
        "ema": ema_state,
        "optimizer": {"must_not_be_loaded": True},
        "provenance": {
            "stage3_approval": {"sha256": approval_sha},
            "ema_policy": stage3_ema_policy_metadata(0.9999),
        },
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
    }
    torch.save(valid_payload, checkpoint)
    target = _tiny_model()
    snapshot = load_stage3_best_ema(
        checkpoint, model=target, approval_sha256=approval_sha
    )
    assert snapshot.checkpoint_step == 12_000
    assert all(
        torch.equal(value, target.state_dict()[name])
        for name, value in model.state_dict().items()
    )

    corrupted_parents: list[tuple[str, Mapping[str, Any]]] = []
    bad_decay = copy.deepcopy(valid_payload)
    bad_decay["ema"]["decay"] = 0.99
    corrupted_parents.append(("decay", bad_decay))
    bad_updates = copy.deepcopy(valid_payload)
    bad_updates["ema"]["num_updates"] = 11_999
    corrupted_parents.append(("updates", bad_updates))
    bool_updates = copy.deepcopy(valid_payload)
    bool_updates["step"] = 1
    bool_updates["ema"]["num_updates"] = True
    corrupted_parents.append(("bool_updates", bool_updates))
    nonfinite_decay = copy.deepcopy(valid_payload)
    nonfinite_decay["ema"]["decay"] = math.nan
    corrupted_parents.append(("nonfinite_decay", nonfinite_decay))
    bad_policy = copy.deepcopy(valid_payload)
    bad_policy["ema"]["policy"]["scope"] = "generic_all_state_ema"
    corrupted_parents.append(("policy", bad_policy))
    bad_inf = copy.deepcopy(valid_payload)
    floating_name = next(
        name for name, value in bad_inf["model"].items() if value.is_floating_point()
    )
    bad_inf["model"][floating_name].view(-1)[0] = math.inf
    bad_inf["ema"]["shadow"][floating_name].view(-1)[0] = math.inf
    corrupted_parents.append(("nonfinite", bad_inf))
    for label, bad_payload in corrupted_parents:
        bad_dir = tmp_path / label
        bad_dir.mkdir()
        bad_checkpoint = bad_dir / "best_ema.pth"
        torch.save(bad_payload, bad_checkpoint)
        rejected_target = _tiny_model()
        before = {
            name: value.detach().clone()
            for name, value in rejected_target.state_dict().items()
        }
        with pytest.raises(Stage4ContractError):
            load_stage3_best_ema(
                bad_checkpoint,
                model=rejected_target,
                approval_sha256=approval_sha,
            )
        assert all(
            torch.equal(value, rejected_target.state_dict()[name])
            for name, value in before.items()
        )

    thresholds_path = tmp_path / "planner_thresholds.json"
    atomic_write_json(
        thresholds_path,
        {
            "schema_version": "graphrestore-presence-thresholds-v1",
            "protocol_id": "graphrestore-v7.1-agenticir-locked",
            "source": "primary_val_presence_f1_only",
            "frozen": True,
            "skills": list(SKILLS),
            "thresholds": {
                name: 0.20 + 0.02 * index for index, name in enumerate(SKILLS)
            },
            "checkpoint_sha256": snapshot.checkpoint_sha256,
            "selected_stage3_checkpoint": {"sha256": snapshot.checkpoint_sha256},
            "search_grid": [0.20 + 0.02 * index for index in range(31)],
            "tie_break": "lowest_threshold",
            "calibration_runs": 1,
            "mio100_rows_read": 0,
        },
    )
    thresholds, _ = load_presence_thresholds(
        thresholds_path, stage3_checkpoint_sha256=snapshot.checkpoint_sha256
    )
    assert thresholds.shape == (8,)
    target.set_presence_thresholds(thresholds)
    composite_parent = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    stage4_optimizer = build_stage4_optimizer(target, fused_if_supported=False)
    stage4_scheduler = WarmupCosineScheduler(
        stage4_optimizer, warmup_steps=800, max_steps=40_000, min_lr=5.0e-7
    )
    stage4_ema = build_stage4_ema(target)
    assert not stage4_optimizer.state
    assert stage4_scheduler.last_epoch == 0
    assert stage4_ema.num_updates == 0
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in composite_parent.items()
    )
    assert all(
        torch.equal(stage4_ema.shadow[name], value)
        for name, value in composite_parent.items()
    )
    stale = json.loads(thresholds_path.read_text())
    stale["checkpoint_sha256"] = "f" * 64
    atomic_write_json(thresholds_path, stale)
    with pytest.raises(Stage4ContractError):
        load_presence_thresholds(
            thresholds_path, stage3_checkpoint_sha256=snapshot.checkpoint_sha256
        )


def test_presence_thresholds_bind_stage3_extension_authorization(
    tmp_path: Path,
) -> None:
    checkpoint_sha = "a" * 64
    extension_sha = "b" * 64
    thresholds_path = tmp_path / "planner_thresholds.json"
    payload = {
        "schema_version": "graphrestore-presence-thresholds-v1",
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "source": "primary_val_presence_f1_only",
        "frozen": True,
        "skills": list(SKILLS),
        "thresholds": {name: 0.20 for name in SKILLS},
        "checkpoint_sha256": checkpoint_sha,
        "selected_stage3_checkpoint": {"sha256": checkpoint_sha},
        "stage3_extension_authorization_sha256": extension_sha,
        "search_grid": [0.20 + 0.02 * index for index in range(31)],
        "tie_break": "lowest_threshold",
        "calibration_runs": 1,
        "mio100_rows_read": 0,
    }
    atomic_write_json(thresholds_path, payload)
    loaded, _ = load_presence_thresholds(
        thresholds_path,
        stage3_checkpoint_sha256=checkpoint_sha,
        stage3_extension_authorization_sha256=extension_sha,
    )
    assert loaded.shape == (8,)

    with pytest.raises(Stage4ContractError):
        load_presence_thresholds(
            thresholds_path,
            stage3_checkpoint_sha256=checkpoint_sha,
            stage3_extension_authorization_sha256="c" * 64,
        )
    with pytest.raises(Stage4ContractError):
        load_presence_thresholds(
            thresholds_path,
            stage3_checkpoint_sha256=checkpoint_sha,
        )


def test_finalized_presence_thresholds_require_new_tie_break_and_nonregressing_f1(
    tmp_path: Path,
) -> None:
    checkpoint_sha = "a" * 64
    extension_sha = "b" * 64
    finalization_sha = "c" * 64
    thresholds_path = tmp_path / "planner_thresholds.json"
    per_skill = {
        name: {
            "baseline": {
                "threshold": 0.50,
                "precision": 0.7,
                "recall": 0.6,
                "f1": 0.64,
            },
            "calibrated": {
                "threshold": 0.50,
                "precision": 0.7,
                "recall": 0.6,
                "f1": 0.64,
            },
        }
        for name in SKILLS
    }
    payload = {
        "schema_version": "graphrestore-presence-thresholds-v1",
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "source": "primary_val_presence_f1_only",
        "frozen": True,
        "skills": list(SKILLS),
        "thresholds": {name: 0.50 for name in SKILLS},
        "checkpoint_sha256": checkpoint_sha,
        "selected_stage3_checkpoint": {"sha256": checkpoint_sha},
        "stage3_extension_authorization_sha256": extension_sha,
        "stage3_finalization_authorization_sha256": finalization_sha,
        "search_grid": [0.20 + 0.02 * index for index in range(31)],
        "tie_break": "nearest_0.50_then_higher_threshold",
        "numerical_tolerance": 1.0e-12,
        "per_skill_metrics": per_skill,
        "macro_f1_before": 0.64,
        "macro_f1_after": 0.64,
        "calibration_runs": 1,
        "mio100_rows_read": 0,
    }
    atomic_write_json(thresholds_path, payload)
    loaded, _ = load_presence_thresholds(
        thresholds_path,
        stage3_checkpoint_sha256=checkpoint_sha,
        stage3_extension_authorization_sha256=extension_sha,
        stage3_finalization_authorization_sha256=finalization_sha,
    )
    assert torch.equal(loaded, torch.full((len(SKILLS),), 0.50))

    regressed = copy.deepcopy(payload)
    regressed["per_skill_metrics"][SKILLS[0]]["calibrated"]["f1"] = 0.63
    atomic_write_json(thresholds_path, regressed)
    with pytest.raises(Stage4ContractError, match="calibrated F1 regressed"):
        load_presence_thresholds(
            thresholds_path,
            stage3_checkpoint_sha256=checkpoint_sha,
            stage3_extension_authorization_sha256=extension_sha,
            stage3_finalization_authorization_sha256=finalization_sha,
        )


def _stage3_extension_parent_case(
    tmp_path: Path,
    *,
    step: int,
    artifact_mutator: Any | None = None,
) -> tuple[Path, GraphRestore, str, dict[str, Any], Path]:
    root = tmp_path / "project"
    checkpoint_dir = root / "artifacts/checkpoints/stage3"
    approval_dir = root / "artifacts/approvals"
    backup_dir = root / "artifacts/migrations" / "stage3_extension_12000_to_18000_v1"
    config_dir = root / "configs"
    for directory in (checkpoint_dir, approval_dir, backup_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base_approval = approval_dir / "STAGE3_APPROVED.json"
    approval_required = approval_dir / "STAGE3_APPROVAL_REQUIRED.json"
    config = config_dir / "stage3_planner.yaml"
    atomic_write_json(base_approval, {"approved": True, "kind": "stage3_approval"})
    atomic_write_json(approval_required, {"approved": False})
    config.write_text("training:\n  max_steps: 12000\n", encoding="utf-8")

    backup_paths = {
        "pre_extension_run_contract": backup_dir / "run_contract.json",
        "pre_extension_last_checkpoint": backup_dir / "last.pth",
        "pre_extension_best_checkpoint": backup_dir / "best_ema.pth",
    }
    for index, path in enumerate(backup_paths.values()):
        path.write_bytes(f"immutable-pre-extension-{index}\n".encode())
        path.chmod(0o444)

    def binding(path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    extension_path = approval_dir / "STAGE3_EXTENSION_APPROVED.json"
    artifact: dict[str, Any] = {
        "schema_version": "graphrestore-stage3-extension-approval-v1",
        "kind": "stage3_extension_approval",
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "approved": True,
        "cycles": 3,
        "base_step": 12_000,
        "target_step": 18_000,
        "validation_every_steps": 2_000,
        "validation_steps": [14_000, 16_000, 18_000],
        "schedule_horizon_steps": 12_000,
        "min_lr": 2.0e-6,
        "lr_policy": "hold_original_cosine_floor_after_schedule_horizon",
        "formal_mio100_authorized": False,
        "authorized_pipeline": ["stage3_extension", "stage4"],
        "base_stage3_approval": binding(base_approval),
        "base_approval_required": binding(approval_required),
        "base_stage3_config": binding(config),
        **{name: binding(path) for name, path in backup_paths.items()},
    }
    if artifact_mutator is not None:
        artifact_mutator(artifact)
    atomic_write_json(extension_path, artifact)
    extension_sha = sha256_file(extension_path)
    extension = {
        "path": str(extension_path.resolve()),
        "sha256": extension_sha,
        **{
            key: artifact[key]
            for key in (
                "cycles",
                "base_step",
                "target_step",
                "validation_every_steps",
                "validation_steps",
                "schedule_horizon_steps",
                "min_lr",
                "lr_policy",
            )
        },
    }

    model = _tiny_model()
    ema = ExponentialMovingAverage(model)
    ema_state = ema.state_dict()
    ema_state["scope"] = STAGE3_EMA_SCOPE
    ema_state["policy"] = stage3_ema_policy_metadata(0.9999)
    ema_state["num_updates"] = step
    checkpoint = checkpoint_dir / "best_ema.pth"
    payload = {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage3",
        "step": step,
        "model_role": "ema_selection",
        "resumable": False,
        "pending_validation_step": None,
        "optimizer_transaction_active": False,
        "scaler": None,
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "model": ema.shadow,
        "ema": ema_state,
        "optimizer": {"must_not_be_loaded": True},
        "provenance": {
            "stage3_approval": {"sha256": sha256_file(base_approval)},
            "ema_policy": stage3_ema_policy_metadata(0.9999),
            "runtime": {"max_steps": 12_000, "training_target_step": 18_000},
            "stage3_extension": extension,
        },
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
    }
    torch.save(payload, checkpoint)
    return checkpoint, model, sha256_file(base_approval), payload, backup_dir


@pytest.mark.parametrize("step", (12_000, 14_000, 16_000, 18_000))
def test_stage4_accepts_exact_stage3_extension_parent_boundaries(
    tmp_path: Path, step: int
) -> None:
    checkpoint, _, approval_sha, payload, _ = _stage3_extension_parent_case(
        tmp_path, step=step
    )
    snapshot = load_stage3_best_ema(
        checkpoint,
        model=_tiny_model(),
        approval_sha256=approval_sha,
    )
    assert snapshot.checkpoint_step == step
    assert snapshot.stage3_extension == payload["provenance"]["stage3_extension"]


def test_stage4_rejects_step12000_parent_when_revocation_is_not_validated(
    tmp_path: Path,
) -> None:
    checkpoint, _, approval_sha, _, _ = _stage3_extension_parent_case(
        tmp_path, step=12_000
    )
    revocation = (
        checkpoint.parents[3] / "artifacts/approvals/STAGE3_EXTENSION_REVOKED.json"
    )
    atomic_write_json(revocation, {"revoked": True})
    with pytest.raises(Stage4ContractError, match="requires the validated"):
        load_stage3_best_ema(
            checkpoint,
            model=_tiny_model(),
            approval_sha256=approval_sha,
        )
    revocation.unlink()
    revocation.symlink_to(revocation.parent / "missing-revocation.json")
    with pytest.raises(Stage4ContractError, match="requires the validated"):
        load_stage3_best_ema(
            checkpoint,
            model=_tiny_model(),
            approval_sha256=approval_sha,
        )


def test_stage4_rejects_stage3_extension_off_boundary_and_contract_drift(
    tmp_path: Path,
) -> None:
    invalid_step, _, approval_sha, _, _ = _stage3_extension_parent_case(
        tmp_path / "step", step=13_000
    )
    with pytest.raises(Stage4ContractError, match="allowed 12k/14k/16k/18k"):
        load_stage3_best_ema(
            invalid_step, model=_tiny_model(), approval_sha256=approval_sha
        )

    invalid_lr, _, approval_sha, _, _ = _stage3_extension_parent_case(
        tmp_path / "lr",
        step=14_000,
        artifact_mutator=lambda value: value.__setitem__("min_lr", 3.0e-6),
    )
    with pytest.raises(Stage4ContractError, match="min_lr drifted"):
        load_stage3_best_ema(
            invalid_lr, model=_tiny_model(), approval_sha256=approval_sha
        )

    extra_key, _, approval_sha, payload, _ = _stage3_extension_parent_case(
        tmp_path / "extra", step=14_000
    )
    payload["provenance"]["stage3_extension"]["unexpected"] = True
    torch.save(payload, extra_key)
    with pytest.raises(Stage4ContractError, match="unknown/partial schema"):
        load_stage3_best_ema(
            extra_key, model=_tiny_model(), approval_sha256=approval_sha
        )


@pytest.mark.parametrize("runtime_mutation", ("missing", "target", "bool"))
def test_stage4_rejects_stage3_extension_runtime_drift(
    tmp_path: Path, runtime_mutation: str
) -> None:
    checkpoint, _, approval_sha, payload, _ = _stage3_extension_parent_case(
        tmp_path, step=14_000
    )
    runtime = payload["provenance"]["runtime"]
    if runtime_mutation == "missing":
        del payload["provenance"]["runtime"]
    elif runtime_mutation == "target":
        runtime["training_target_step"] = 12_000
    else:
        runtime["max_steps"] = True
    torch.save(payload, checkpoint)
    with pytest.raises(Stage4ContractError, match="12k schedule horizon"):
        load_stage3_best_ema(
            checkpoint, model=_tiny_model(), approval_sha256=approval_sha
        )


def test_stage4_rejects_writable_stage3_extension_backup(tmp_path: Path) -> None:
    checkpoint, _, approval_sha, _, backup_dir = _stage3_extension_parent_case(
        tmp_path, step=16_000
    )
    (backup_dir / "last.pth").chmod(0o644)
    with pytest.raises(Stage4ContractError, match="mode must be 0444"):
        load_stage3_best_ema(
            checkpoint, model=_tiny_model(), approval_sha256=approval_sha
        )


@pytest.mark.parametrize("field", ("path", "sha256"))
def test_stage4_rejects_stage3_extension_identity_drift(
    tmp_path: Path, field: str
) -> None:
    checkpoint, _, approval_sha, payload, _ = _stage3_extension_parent_case(
        tmp_path, step=18_000
    )
    extension = payload["provenance"]["stage3_extension"]
    extension[field] = (
        str(tmp_path / "noncanonical/STAGE3_EXTENSION_APPROVED.json")
        if field == "path"
        else "0" * 64
    )
    torch.save(payload, checkpoint)
    with pytest.raises(Stage4ContractError):
        load_stage3_best_ema(
            checkpoint, model=_tiny_model(), approval_sha256=approval_sha
        )


def test_stage4_rejects_stage3_extension_backup_hash_drift(tmp_path: Path) -> None:
    checkpoint, _, approval_sha, _, backup_dir = _stage3_extension_parent_case(
        tmp_path, step=18_000
    )
    backup = backup_dir / "run_contract.json"
    backup.chmod(0o644)
    backup.write_bytes(b"tampered-but-read-only-again\n")
    backup.chmod(0o444)
    with pytest.raises(Stage4ContractError, match="hash drifted"):
        load_stage3_best_ema(
            checkpoint, model=_tiny_model(), approval_sha256=approval_sha
        )


def test_stage4_rejects_unapproved_post_12k_stage3_parent(tmp_path: Path) -> None:
    checkpoint, _, approval_sha, payload, _ = _stage3_extension_parent_case(
        tmp_path, step=14_000
    )
    payload["provenance"].pop("stage3_extension")
    torch.save(payload, checkpoint)
    with pytest.raises(Stage4ContractError, match="original 2k..12k"):
        load_stage3_best_ema(
            checkpoint, model=_tiny_model(), approval_sha256=approval_sha
        )


def test_stage4_checkpoint_best_is_ema_and_resume_is_exact(tmp_path: Path) -> None:
    dataset = _fake_sampling_dataset()
    sampler = Stage4EpisodeSampler(dataset, num_samples=400)
    model = _tiny_model()
    optimizer = build_stage4_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=800, max_steps=40_000, min_lr=5.0e-7
    )
    ema = build_stage4_ema(model)
    for _ in range(7):
        optimizer.zero_grad(set_to_none=True)
        sum(
            parameter.square().mean()
            for parameter in model.parameters()
            if parameter.requires_grad
        ).backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
    provenance = {
        "schema": STAGE4_SCHEMA,
        "bound": "abc",
        "ema_policy": stage4_ema_policy_metadata(0.9999),
        "frozen_parent_state_sha256": stage4_engine.stage4_fixed_state_digest(model),
        "runtime_evidence": _runtime_evidence_binding(),
    }
    missing_evidence = copy.deepcopy(provenance)
    del missing_evidence["runtime_evidence"]
    with pytest.raises(Stage4ContractError, match="runtime gate evidence"):
        save_stage4_checkpoint(
            tmp_path / "missing_runtime_evidence.pth",
            step=7,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            provenance=missing_evidence,
        )
    selection_metrics = {
        "group_a_psnr": 2.0,
        "group_a_ssim": 0.2,
        "single_psnr": 1.0,
        "single_ssim": 0.1,
        "validation_step": 7.0,
        "best_group_a_psnr": 2.0,
        "best_group_a_ssim": 0.2,
        "best_single_psnr": 1.0,
        "best_single_ssim": 0.1,
        "best_step": 7.0,
    }
    with pytest.raises(Stage4ContractError, match="phase-aware EMA"):
        save_stage4_checkpoint(
            tmp_path / "generic.pth",
            step=0,
            model=model,
            ema=ExponentialMovingAverage(model),  # type: ignore[arg-type]
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            provenance=provenance,
        )
    frozen_parameter_name, frozen_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if stage4_parameter_role(name) is None and parameter.is_floating_point()
    )
    frozen_raw_before = frozen_parameter.detach().clone()
    frozen_ema_before = ema.shadow[frozen_parameter_name].detach().clone()
    with torch.no_grad():
        frozen_parameter.add_(1.0)
        ema.shadow[frozen_parameter_name].add_(1.0)
    with pytest.raises(Stage4ContractError, match="frozen model state"):
        save_stage4_checkpoint(
            tmp_path / "synchronized_frozen_save_tamper.pth",
            step=7,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            provenance=provenance,
        )
    with torch.no_grad():
        frozen_parameter.copy_(frozen_raw_before)
        ema.shadow[frozen_parameter_name].copy_(frozen_ema_before)
    checkpoint = tmp_path / "best_ema.pth"
    save_stage4_checkpoint(
        checkpoint,
        step=7,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        metrics=selection_metrics,
        model_as_ema=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["model_role"] == "ema_selection"
    assert payload["resumable"] is False
    assert all(
        torch.equal(payload["model"][name], payload["ema"]["shadow"][name])
        for name in payload["model"]
    )
    fixed_input = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(997))
    fixed_mask = torch.zeros(1, len(SKILLS), dtype=torch.bool)
    fixed_mask[:, 0] = True
    selected_reference = _tiny_model()
    selected_loaded = _tiny_model()
    reference_incompatible = selected_reference.load_state_dict(
        payload["model"], strict=True
    )
    loaded_incompatible = selected_loaded.load_state_dict(payload["model"], strict=True)
    assert not reference_incompatible.missing_keys
    assert not reference_incompatible.unexpected_keys
    assert not loaded_incompatible.missing_keys
    assert not loaded_incompatible.unexpected_keys
    selected_reference.eval()
    selected_loaded.eval()
    with torch.inference_mode():
        selected_reference_output = selected_reference(
            fixed_input,
            forced_counterfactual_mask=fixed_mask,
            max_rounds=1,
        )
        selected_loaded_output = selected_loaded(
            fixed_input,
            forced_counterfactual_mask=fixed_mask,
            max_rounds=1,
        )
    assert torch.is_tensor(selected_reference_output)
    assert torch.is_tensor(selected_loaded_output)
    assert torch.equal(selected_reference_output, selected_loaded_output)

    restored_model = _tiny_model()
    restored_optimizer = build_stage4_optimizer(
        restored_model, fused_if_supported=False
    )
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer, warmup_steps=800, max_steps=40_000, min_lr=5.0e-7
    )
    restored_ema = build_stage4_ema(restored_model)
    restored_sampler = Stage4EpisodeSampler(dataset, num_samples=400)
    before_rejected_resume = {
        name: value.detach().clone()
        for name, value in restored_model.state_dict().items()
    }
    with pytest.raises(Stage4ContractError):
        resume_stage4_checkpoint(
            checkpoint,
            model=restored_model,
            ema=restored_ema,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            sampler=restored_sampler,
            expected_provenance=provenance,
        )
    assert all(
        torch.equal(value, restored_model.state_dict()[name])
        for name, value in before_rejected_resume.items()
    )
    # Production resume first reconstructs the frozen Stage3-derived parent.
    # Mirror that precondition so synchronized frozen tampering is detectable.
    restored_model.load_state_dict(model.state_dict(), strict=True)

    last_checkpoint = tmp_path / "last.pth"
    raw_metrics = {
        **selection_metrics,
        "best_checkpoint_sha256": sha256_file(checkpoint),
    }
    save_stage4_checkpoint(
        last_checkpoint,
        step=7,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        metrics=raw_metrics,
        model_as_ema=False,
    )
    model.eval()
    with torch.inference_mode():
        raw_output_before_resume = model(
            fixed_input,
            forced_counterfactual_mask=fixed_mask,
            max_rounds=1,
        )
    last_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    assert last_payload["model_role"] == "raw_training_state"
    assert last_payload["resumable"] is True
    assert last_payload["pending_validation_step"] is None
    resumed = resume_stage4_checkpoint(
        last_checkpoint,
        model=restored_model,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=restored_sampler,
        expected_provenance=provenance,
    )
    assert resumed["step"] == 7
    assert restored_sampler.step == 7
    assert restored_ema.num_updates == ema.num_updates
    restored_model.eval()
    with torch.inference_mode():
        raw_output_after_resume = restored_model(
            fixed_input,
            forced_counterfactual_mask=fixed_mask,
            max_rounds=1,
        )
    assert torch.is_tensor(raw_output_before_resume)
    assert torch.is_tensor(raw_output_after_resume)
    assert torch.equal(raw_output_before_resume, raw_output_after_resume)

    dense_state_id = next(
        parameter_id
        for group in last_payload["optimizer"]["param_groups"]
        if group["role"] != "skills_mixers"
        for parameter_id in group["params"]
    )
    skill_state_ids = [
        parameter_id
        for group in last_payload["optimizer"]["param_groups"]
        if group["role"] == "skills_mixers"
        for parameter_id in group["params"]
    ]
    state_id = dense_state_id
    ledger_id = next(iter(last_payload["optimizer_state_name_ledger"]))
    corrupted_payloads: list[tuple[str, dict[str, Any]]] = []

    missing_ledger = copy.deepcopy(last_payload)
    del missing_ledger["optimizer_state_name_ledger"]
    corrupted_payloads.append(("missing_ledger", missing_ledger))

    deleted_state = copy.deepcopy(last_payload)
    del deleted_state["optimizer"]["state"][dense_state_id]
    corrupted_payloads.append(("deleted_dense_state", deleted_state))

    wrong_ledger_name = copy.deepcopy(last_payload)
    wrong_ledger_name["optimizer_state_name_ledger"][ledger_id]["name"] = "planner.bad"
    corrupted_payloads.append(("wrong_ledger_name", wrong_ledger_name))

    wrong_ledger_id = copy.deepcopy(last_payload)
    wrong_entry = wrong_ledger_id["optimizer_state_name_ledger"].pop(ledger_id)
    wrong_ledger_id["optimizer_state_name_ledger"][999_999] = wrong_entry
    corrupted_payloads.append(("wrong_ledger_id", wrong_ledger_id))

    wrong_role = copy.deepcopy(last_payload)
    wrong_role["optimizer"]["param_groups"][0]["role"] = "skills_mixers"
    corrupted_payloads.append(("wrong_role", wrong_role))

    wrong_adam_step = copy.deepcopy(last_payload)
    wrong_adam_step["optimizer"]["state"][state_id]["step"] = torch.tensor(8.0)
    corrupted_payloads.append(("wrong_adam_step", wrong_adam_step))

    stale_dense_adam_step = copy.deepcopy(last_payload)
    stale_dense_adam_step["optimizer"]["state"][dense_state_id]["step"] = torch.tensor(
        6.0
    )
    corrupted_payloads.append(("stale_dense_adam_step", stale_dense_adam_step))

    wrong_shape = copy.deepcopy(last_payload)
    wrong_shape["optimizer"]["state"][state_id]["exp_avg"] = torch.zeros(1)
    corrupted_payloads.append(("wrong_shape", wrong_shape))

    wrong_dtype = copy.deepcopy(last_payload)
    wrong_dtype["optimizer"]["state"][state_id]["exp_avg"] = wrong_dtype["optimizer"][
        "state"
    ][state_id]["exp_avg"].double()
    corrupted_payloads.append(("wrong_dtype", wrong_dtype))

    nonfinite_state = copy.deepcopy(last_payload)
    nonfinite_state["optimizer"]["state"][state_id]["exp_avg"].view(-1)[0] = math.inf
    corrupted_payloads.append(("nonfinite_state", nonfinite_state))

    wrong_scheduler = copy.deepcopy(last_payload)
    wrong_scheduler["scheduler"]["last_epoch"] = 6
    corrupted_payloads.append(("wrong_scheduler", wrong_scheduler))

    wrong_base_lr = copy.deepcopy(last_payload)
    wrong_base_lr["scheduler"]["base_lrs"][0] *= 2
    corrupted_payloads.append(("wrong_base_lr", wrong_base_lr))

    wrong_lr = copy.deepcopy(last_payload)
    wrong_lr["optimizer"]["param_groups"][0]["lr"] *= 2
    corrupted_payloads.append(("wrong_lr", wrong_lr))

    wrong_cursor = copy.deepcopy(last_payload)
    wrong_cursor["sampler_state"]["sample_cursor"] += 4
    corrupted_payloads.append(("wrong_cursor", wrong_cursor))

    missing_rng_field = copy.deepcopy(last_payload)
    del missing_rng_field["rng_states"]["torch_cpu"]
    corrupted_payloads.append(("missing_rng_field", missing_rng_field))

    missing_runtime_evidence = copy.deepcopy(last_payload)
    del missing_runtime_evidence["provenance"]["runtime_evidence"]
    corrupted_payloads.append(("missing_runtime_evidence", missing_runtime_evidence))

    forged_runtime_evidence = copy.deepcopy(last_payload)
    forged_runtime_evidence["provenance"]["runtime_evidence"][
        "validation_vram_gate_sha256"
    ] = "z" * 64
    corrupted_payloads.append(("forged_runtime_evidence", forged_runtime_evidence))

    trainable_name = next(
        name
        for name in last_payload["model"]
        if stage4_parameter_role(name) is not None
    )
    frozen_name = next(
        name
        for name, value in last_payload["model"].items()
        if stage4_parameter_role(name) is None and value.is_floating_point()
    )
    nonfinite_raw = copy.deepcopy(last_payload)
    nonfinite_raw["model"][trainable_name].view(-1)[0] = math.inf
    corrupted_payloads.append(("nonfinite_raw", nonfinite_raw))

    nonfinite_ema = copy.deepcopy(last_payload)
    nonfinite_ema["ema"]["shadow"][trainable_name].view(-1)[0] = math.inf
    corrupted_payloads.append(("nonfinite_ema", nonfinite_ema))

    synchronized_frozen_tamper = copy.deepcopy(last_payload)
    synchronized_frozen_tamper["model"][frozen_name].add_(1.0)
    synchronized_frozen_tamper["ema"]["shadow"][frozen_name].add_(1.0)
    corrupted_payloads.append(
        ("synchronized_frozen_tamper", synchronized_frozen_tamper)
    )

    nonfinite_metric = copy.deepcopy(last_payload)
    nonfinite_metric["metrics"]["group_a_psnr"] = math.nan
    corrupted_payloads.append(("nonfinite_metric", nonfinite_metric))

    wrong_metric_step = copy.deepcopy(last_payload)
    wrong_metric_step["metrics"]["validation_step"] = 8.0
    corrupted_payloads.append(("wrong_metric_step", wrong_metric_step))

    better_current_than_best = copy.deepcopy(last_payload)
    better_current_than_best["metrics"]["group_a_psnr"] = 3.0
    corrupted_payloads.append(("better_current_than_best", better_current_than_best))

    wrong_best_hash = copy.deepcopy(last_payload)
    wrong_best_hash["metrics"]["best_checkpoint_sha256"] = "0" * 64
    corrupted_payloads.append(("wrong_best_hash", wrong_best_hash))

    for label, bad_payload in corrupted_payloads:
        bad_path = tmp_path / f"{label}.pth"
        torch.save(bad_payload, bad_path)
        before_rejection = {
            name: value.detach().clone()
            for name, value in restored_model.state_dict().items()
        }
        before_optimizer_step = restored_optimizer.state_dict()["state"][state_id][
            "step"
        ].clone()
        before_scheduler_epoch = restored_scheduler.last_epoch
        before_ema = {
            name: value.detach().clone() for name, value in restored_ema.shadow.items()
        }
        before_sampler = copy.deepcopy(restored_sampler.state_dict())
        before_rng = stage4_engine._rng_state_digest(stage4_engine.capture_rng_state())
        with pytest.raises(Stage4ContractError):
            resume_stage4_checkpoint(
                bad_path,
                model=restored_model,
                ema=restored_ema,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                sampler=restored_sampler,
                expected_provenance=provenance,
            )
        assert all(
            torch.equal(value, restored_model.state_dict()[name])
            for name, value in before_rejection.items()
        )
        assert torch.equal(
            restored_optimizer.state_dict()["state"][state_id]["step"],
            before_optimizer_step,
        )
        assert restored_scheduler.last_epoch == before_scheduler_epoch
        assert all(
            torch.equal(value, restored_ema.shadow[name])
            for name, value in before_ema.items()
        )
        assert restored_sampler.state_dict() == before_sampler
        assert (
            stage4_engine._rng_state_digest(stage4_engine.capture_rng_state())
            == before_rng
        )

    sparse_skills = copy.deepcopy(last_payload)
    del sparse_skills["optimizer"]["state"][skill_state_ids[0]]
    sparse_skills["optimizer_state_name_ledger"][skill_state_ids[0]]["has_state"] = (
        False
    )
    sparse_skills["optimizer"]["state"][skill_state_ids[1]]["step"] = torch.tensor(6.0)
    sparse_path = tmp_path / "sparse_skills.pth"
    torch.save(sparse_skills, sparse_path)
    sparse_resume = resume_stage4_checkpoint(
        sparse_path,
        model=restored_model,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=restored_sampler,
        expected_provenance=provenance,
    )
    assert sparse_resume["step"] == 7
    assert skill_state_ids[0] not in restored_optimizer.state_dict()["state"]
    assert (
        int(restored_optimizer.state_dict()["state"][skill_state_ids[1]]["step"].item())
        == 6
    )

    bad_scope = copy.deepcopy(last_payload)
    bad_scope["ema"]["scope"] = "generic_all_state_ema"
    bad_scope_path = tmp_path / "bad_scope.pth"
    torch.save(bad_scope, bad_scope_path)
    before_bad_scope = {
        name: value.detach().clone()
        for name, value in restored_model.state_dict().items()
    }
    with pytest.raises(Stage4ContractError, match="scope"):
        resume_stage4_checkpoint(
            bad_scope_path,
            model=restored_model,
            ema=restored_ema,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            sampler=restored_sampler,
            expected_provenance=provenance,
        )
    assert all(
        torch.equal(value, restored_model.state_dict()[name])
        for name, value in before_bad_scope.items()
    )

    pending_path = tmp_path / "pending.pth"
    save_stage4_checkpoint(
        pending_path,
        step=7,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        pending_validation_step=7,
    )
    pending_payload = torch.load(pending_path, map_location="cpu", weights_only=False)
    assert pending_payload["pending_validation_step"] == 7
    pending_model = _tiny_model()
    pending_model.load_state_dict(model.state_dict(), strict=True)
    pending_optimizer = build_stage4_optimizer(pending_model, fused_if_supported=False)
    pending_scheduler = WarmupCosineScheduler(
        pending_optimizer, warmup_steps=800, max_steps=40_000, min_lr=5.0e-7
    )
    pending_ema = build_stage4_ema(pending_model)
    pending_sampler = Stage4EpisodeSampler(dataset, num_samples=400)
    replay = resume_stage4_checkpoint(
        pending_path,
        model=pending_model,
        ema=pending_ema,
        optimizer=pending_optimizer,
        scheduler=pending_scheduler,
        sampler=pending_sampler,
        expected_provenance=provenance,
        expected_validation_every=7,
        expected_max_steps=7,
    )
    assert replay["pending_validation_step"] == 7


def test_stage4_six_mode_diagnostics_are_zero_update_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _fake_sampling_dataset()
    sampler = Stage4EpisodeSampler(dataset, num_samples=400)
    model = _tiny_model()
    optimizer = build_stage4_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=800, max_steps=40_000, min_lr=5.0e-7
    )
    ema = build_stage4_ema(model)
    optimizer.zero_grad(set_to_none=True)
    sum(
        parameter.square().mean()
        for parameter in model.parameters()
        if parameter.requires_grad
    ).backward()
    optimizer.step()
    scheduler.step()
    ema.update(model)
    provenance = {
        "schema": STAGE4_SCHEMA,
        "ema_policy": stage4_ema_policy_metadata(0.9999),
        "frozen_parent_state_sha256": stage4_engine.stage4_fixed_state_digest(model),
        "runtime_evidence": _runtime_evidence_binding(),
    }
    best = tmp_path / "best_ema.pth"
    save_stage4_checkpoint(
        best,
        step=1,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        metrics={
            "group_a_psnr": 2.0,
            "group_a_ssim": 0.2,
            "single_psnr": 1.0,
            "single_ssim": 0.1,
            "validation_step": 1.0,
            "best_group_a_psnr": 2.0,
            "best_group_a_ssim": 0.2,
            "best_single_psnr": 1.0,
            "best_single_ssim": 0.1,
            "best_step": 1.0,
        },
        model_as_ema=True,
    )
    raw_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    ema_before = {name: value.detach().clone() for name, value in ema.shadow.items()}
    calls: list[tuple[str, bool]] = []

    def fake_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        current_model = args[0]
        calls.append(
            (
                current_model.compiler.mode,
                "execute_planned_level" in current_model.__dict__,
            )
        )
        return {
            "single_equal_task_mean": {"count": 1, "psnr": 1.0, "ssim": 0.1},
            "group_a_equal_combination_mean": {
                "count": 1,
                "psnr": 2.0,
                "ssim": 0.2,
            },
            "diagnostics": {"sentinel": 1.0},
            "image_count": 2,
        }

    monkeypatch.setattr(stage4_engine, "validate_stage4", fake_validation)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 100)
    json_path = tmp_path / "diagnostics.json"
    report_path = tmp_path / "diagnostics.md"
    result = run_stage4_zero_training_diagnostics(
        model,
        ema,
        dataset,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        relation_val_records={},
        selected_best_checkpoint=best,
        expected_provenance=provenance,
        json_path=json_path,
        report_path=report_path,
        use_bf16=False,
    )
    assert result["optimizer_updates"] == 0
    assert result["model_ema_rng_unchanged"] is True
    assert list(result["compiler_modes"]) == [
        "full_partial_order",
        "forced_total_order",
        "parallel_only",
    ]
    assert list(result["guard_modes"]) == [
        "predicted_spatial",
        "global_mean",
        "all_one",
    ]
    assert calls == [
        ("full_partial_order", False),
        ("forced_total_order", False),
        ("parallel_only", False),
        ("full_partial_order", False),
        ("full_partial_order", True),
        ("full_partial_order", True),
    ]
    assert json_path.is_file() and report_path.is_file()
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in raw_before.items()
    )
    assert all(
        torch.equal(value, ema.shadow[name]) for name, value in ema_before.items()
    )
    assert ema.num_updates == 1


def test_stage4_entrypoint_reads_real_finalization_payload_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = PROJECT_ROOT / "scripts/train_stage4_e2e.py"
    spec = importlib.util.spec_from_file_location(
        "stage4_cli_payload_regression", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frozen_history = tmp_path / "calibration_history.csv"
    frozen_history.write_text("frozen-stage3-history\n", encoding="utf-8")
    binding = {
        "path": str(frozen_history.resolve()),
        "sha256": sha256_file(frozen_history),
    }
    # validate_stage3_finalization_for_stage4 returns report/best convenience
    # bindings at the top level, while complete.json itself is under payload.
    real_return_shape = {
        "complete": {"path": "/complete.json", "sha256": "a" * 64},
        "report": {"path": "/report.md", "sha256": "b" * 64},
        "best_checkpoint": {"path": "/best.pth", "sha256": "c" * 64},
        "payload": {"bindings": {"calibration_history": binding}},
    }
    cuda_gate_calls: list[bool] = []
    monkeypatch.setattr(
        module.torch.cuda,
        "is_available",
        lambda: cuda_gate_calls.append(True) or False,
    )
    assert (
        module._stage3_frozen_calibration_binding(
            real_return_shape,
            expected_path=frozen_history,
        )
        == binding
    )
    assert cuda_gate_calls == []

    with pytest.raises(Stage4ContractError, match="payload must be a mapping"):
        module._stage3_frozen_calibration_binding(
            {"calibration_history": binding},
            expected_path=frozen_history,
        )

    extra_field = copy.deepcopy(real_return_shape)
    extra_field["payload"]["bindings"]["calibration_history"]["unexpected"] = True
    with pytest.raises(Stage4ContractError, match="schema drifted"):
        module._stage3_frozen_calibration_binding(
            extra_field,
            expected_path=frozen_history,
        )

    malformed_sha = copy.deepcopy(real_return_shape)
    malformed_sha["payload"]["bindings"]["calibration_history"]["sha256"] = "A" * 64
    with pytest.raises(Stage4ContractError, match="malformed"):
        module._stage3_frozen_calibration_binding(
            malformed_sha,
            expected_path=frozen_history,
        )

    wrong_path = copy.deepcopy(real_return_shape)
    wrong_path["payload"]["bindings"]["calibration_history"]["path"] = str(
        tmp_path / "other.csv"
    )
    with pytest.raises(Stage4ContractError, match="path drifted"):
        module._stage3_frozen_calibration_binding(
            wrong_path,
            expected_path=frozen_history,
        )

    wrong_hash = copy.deepcopy(real_return_shape)
    wrong_hash["payload"]["bindings"]["calibration_history"]["sha256"] = "0" * 64
    with pytest.raises(Stage4ContractError, match="hash drifted"):
        module._stage3_frozen_calibration_binding(
            wrong_hash,
            expected_path=frozen_history,
        )

    source = script_path.read_text(encoding="utf-8")
    run_start = source.index("def run(arguments: argparse.Namespace)")
    nested_binding_preflight = source.index(
        "frozen_calibration_binding = _stage3_frozen_calibration_binding(",
        run_start,
    )
    cuda_gate = source.index("if not torch.cuda.is_available():", run_start)
    assert nested_binding_preflight < cuda_gate


def test_stage4_formal_cli_argv_and_shared_calibration_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = PROJECT_ROOT / "scripts/train_stage4_e2e.py"
    spec = importlib.util.spec_from_file_location("stage4_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    arguments = parser.parse_args(["--config", "configs/stage4_graphrestore_e2e.yaml"])
    assert arguments.config == Path("configs/stage4_graphrestore_e2e.yaml")
    assert arguments.resume is None
    assert arguments.max_steps is None
    orchestration = (PROJECT_ROOT / "src/training/orchestration.py").read_text()
    assert '"scripts/train_stage4_e2e.py"' in orchestration
    assert '"configs/stage4_graphrestore_e2e.yaml"' in orchestration
    source = script_path.read_text(encoding="utf-8")
    for required_finalization_sha_argument in (
        "stage3_complete_sha256=stage3_finalization_outputs",
        "stage3_calibrated_diagnostic_sha256=(",
        "stage3_thresholds_sha256=stage3_finalization_outputs",
    ):
        assert required_finalization_sha_argument in source
    assert source.index("if pending_validation_step is not None:") < source.index(
        "iterator = iter(train_loader)"
    )
    assert source.index(
        "validation_in_progress_step: int | None = pending_validation_step"
    ) < source.index("if pending_validation_step is not None:")
    assert source.index('"event": "validation"') < source.index(
        "# Clearing pending is the final validation transaction commit"
    )
    assert source.index(
        "_update_stage4_running_status(",
        source.index('"event": "validation"'),
    ) < source.index("# Clearing pending is the final validation transaction commit")
    assert source.index("# The frozen CUDA gates are validated") < source.index(
        "if not torch.cuda.is_available():"
    )
    training_window = source.index("training_update_in_progress = True")
    peak_guard = source.index(
        "peak_reserved_fraction = peak_reserved / total_memory", training_window
    )
    train_log_commit = source.index(
        '"learning_rates": lr_by_role(optimizer)', training_window
    )
    training_commit = source.index(
        "# The optimizer transaction becomes signal-saveable", training_window
    )
    validate_due = source.index("validate_now = (", train_log_commit)
    validation_marker = source.index(
        "validation_in_progress_step = global_step", validate_due
    )
    marker_clear = source.index("training_update_in_progress = False", training_commit)
    assert training_window < peak_guard < train_log_commit < training_commit
    assert train_log_commit < validate_due < validation_marker < training_commit
    assert training_commit < marker_clear
    validation_handoff = source.index(
        "run_validation_boundary(replay_pending=False)", marker_clear
    )
    assert marker_clear < validation_handoff

    historical_log = tmp_path / "historical.jsonl"
    historical_log.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"step": 1, "peak_reserved_bytes": 100},
                {
                    "event": "validation",
                    "step": 4000,
                    "peak_reserved_bytes": 300,
                },
                {"step": 4001, "peak_reserved_bytes": 200},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert module._historical_stage4_peak_reserved(historical_log) == (200, 300)

    with pytest.raises(KeyboardInterrupt):
        module._sigterm_as_keyboard_interrupt(module.signal.SIGTERM, None)
    # Injection at the resume window immediately before pending replay: the
    # restored marker must prevent the interrupt path from clearing pending.
    restored_pending_marker = 4000
    with pytest.raises(KeyboardInterrupt):
        module._sigterm_as_keyboard_interrupt(module.signal.SIGTERM, None)
    assert not module._stage4_interrupt_can_checkpoint(
        mid_optimizer_update=False,
        pending_validation_step=restored_pending_marker,
    )
    assert module._stage4_interrupt_can_checkpoint(
        mid_optimizer_update=False, pending_validation_step=None
    )
    assert not module._stage4_interrupt_can_checkpoint(
        mid_optimizer_update=True, pending_validation_step=None
    )
    checkpoint_requests: list[str] = []
    try:
        module._sigterm_as_keyboard_interrupt(module.signal.SIGTERM, None)
    except KeyboardInterrupt:
        if module._stage4_interrupt_can_checkpoint(
            # Injected after optimizer/sampler advance but before peak/log
            # commit: the stable raw checkpoint must remain untouched.
            mid_optimizer_update=True,
            pending_validation_step=None,
        ):
            checkpoint_requests.append("save")
    assert checkpoint_requests == []
    assert not module._stage4_interrupt_can_checkpoint(
        mid_optimizer_update=False, pending_validation_step=4000
    )
    # Exact former race window: the optimizer marker has just been cleared,
    # but run_validation_boundary has not yet begun.  The already-armed local
    # validation marker must still prohibit a pending=None checkpoint.
    checkpoint_requests.clear()
    try:
        module._sigterm_as_keyboard_interrupt(module.signal.SIGTERM, None)
    except KeyboardInterrupt:
        if module._stage4_interrupt_can_checkpoint(
            mid_optimizer_update=False,
            pending_validation_step=4000,
        ):
            checkpoint_requests.append("save")
    assert checkpoint_requests == []
    signal_calls: list[tuple[object, object]] = []
    previous_handler = object()

    def fake_signal(signum: object, handler: object) -> object:
        signal_calls.append((signum, handler))
        return previous_handler

    def interrupt_run(arguments: argparse.Namespace) -> int:
        del arguments
        raise KeyboardInterrupt

    monkeypatch.setattr(module.signal, "signal", fake_signal)
    monkeypatch.setattr(module, "run", interrupt_run)
    assert module.main([]) == 130
    assert signal_calls == [
        (module.signal.SIGTERM, module._sigterm_as_keyboard_interrupt),
        (module.signal.SIGTERM, previous_handler),
    ]

    output_dir = tmp_path / "stage4"
    output_dir.mkdir()
    stable = output_dir / "last.pth"
    torch.save(
        {
            "step": 4000,
            "pending_validation_step": None,
            "model_role": "raw_training_state",
            "resumable": True,
        },
        stable,
    )
    stable_sha = sha256_file(stable)
    deviations = tmp_path / "DEVIATIONS.md"
    receipt = module._record_stage4_runtime_oom(
        output_dir=output_dir,
        error=torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate"),
        attempted_step=4123,
        mid_optimizer_update=True,
        pending_validation_step=None,
        crop_size=160,
        micro_batch=2,
        allocator_conf=STAGE4_ALLOCATOR_CONF,
        deviations_path=deviations,
    )
    assert receipt["checkpoint_advanced"] is False
    assert receipt["automatic_crop_or_micro_fallback"] is False
    assert receipt["same_process_continuation"] is False
    assert receipt["stable_checkpoint"]["sha256"] == stable_sha
    assert sha256_file(stable) == stable_sha
    assert (output_dir / "runtime_oom.json").is_file()
    assert "resume_post_approval_pipeline" in deviations.read_text()
    with pytest.raises(Stage4ContractError, match="non-OOM"):
        module._record_stage4_runtime_oom(
            output_dir=output_dir,
            error=RuntimeError("kernel launch failed"),
            attempted_step=4123,
            mid_optimizer_update=True,
            pending_validation_step=None,
            crop_size=160,
            micro_batch=2,
            allocator_conf=STAGE4_ALLOCATOR_CONF,
            deviations_path=deviations,
        )

    diagnostics = {
        key: 0.0
        for key in (
            "planner_macro_f1",
            "relation_accuracy",
            "parallel_precision",
            "parallel_recall",
            "pre_cycle_rate",
            "dropped_edge_rate",
            "guard_spearman_rain",
            "guard_spearman_haze",
            "guard_mae_rain",
            "guard_mae_haze",
            "guard_std_rain",
            "guard_std_haze",
            "guard_high_frac_rain",
            "guard_high_frac_haze",
            "reentry_request_rate",
            "unexpected_skill_activation_rate",
            "mean_program_levels",
        )
    }
    diagnostics["clean_misuse"] = {"psnr": 0.0, "ssim": 0.0, "residual_norm": 0.0}
    diagnostics["wrong_skill_identity"] = {
        "psnr": 0.0,
        "ssim": 0.0,
        "residual_norm": 0.0,
    }
    summary = {
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "single_equal_task_mean": {"psnr": 1.0, "ssim": 0.1},
        "group_a_equal_combination_mean": {"psnr": 2.0, "ssim": 0.2},
        "diagnostics": diagnostics,
    }
    history = tmp_path / "calibration_history.csv"
    stage0_row = {key: None for key in CALIBRATION_COLUMNS}
    stage0_row.update(
        {
            "step": 4000,
            "single_psnr": 26.0,
            "single_ssim": 0.82,
            "group_a_psnr": 20.0,
            "group_a_ssim": 0.66,
        }
    )
    stage3_row = {key: None for key in CALIBRATION_COLUMNS}
    stage3_row.update(
        {
            "step": 4000,
            "single_psnr": 22.0,
            "single_ssim": 0.74,
            "group_a_psnr": 20.1,
            "group_a_ssim": 0.62,
            "planner_macro_f1": 0.24,
        }
    )
    append_stage3_calibration_history(history, stage0_row)
    append_stage3_calibration_history(history, stage3_row)
    frozen_history_sha256 = sha256_file(history)
    stage4_history = module._stage4_calibration_history_path(history)
    routing = module._calibration_history_routing(
        frozen_stage3_history=history,
        frozen_stage3_sha256=frozen_history_sha256,
        stage4_history=stage4_history,
    )
    assert set(routing) == {
        "schema_version",
        "frozen_stage3_history",
        "stage4_history_path",
        "columns",
        "stage4_marker_columns",
        "validation_steps",
    }
    assert routing["stage4_history_path"] == str(stage4_history.resolve())
    assert routing["columns"] == list(CALIBRATION_COLUMNS)
    assert routing["validation_steps"] == list(range(4000, 40001, 4000))
    stage4_history.hardlink_to(history)
    with pytest.raises(Stage4ContractError, match="aliases the frozen Stage3"):
        module._require_calibration_history_boundary(
            frozen_stage3_history=history,
            frozen_stage3_sha256=frozen_history_sha256,
            stage4_history=stage4_history,
        )
    stage4_history.unlink()
    stage4_history.symlink_to(history)
    with pytest.raises(Stage4ContractError, match="cannot be a symlink"):
        module._require_calibration_history_boundary(
            frozen_stage3_history=history,
            frozen_stage3_sha256=frozen_history_sha256,
            stage4_history=stage4_history,
        )
    stage4_history.unlink()
    stage4_history.mkdir()
    with pytest.raises(Stage4ContractError, match="is not regular"):
        module._require_calibration_history_boundary(
            frozen_stage3_history=history,
            frozen_stage3_sha256=frozen_history_sha256,
            stage4_history=stage4_history,
        )
    stage4_history.rmdir()
    with pytest.raises(Stage4ContractError, match="sidecar path drifted"):
        module._calibration_history_routing(
            frozen_stage3_history=history,
            frozen_stage3_sha256=frozen_history_sha256,
            stage4_history=tmp_path / "wrong_stage4_history.csv",
        )
    symlink_parent = tmp_path / "symlink_parent"
    symlink_parent.mkdir()
    frozen_symlink = symlink_parent / "calibration_history.csv"
    frozen_symlink.symlink_to(history)
    with pytest.raises(Stage4ContractError, match="frozen Stage3.*not regular"):
        module._calibration_history_routing(
            frozen_stage3_history=frozen_symlink,
            frozen_stage3_sha256=frozen_history_sha256,
            stage4_history=symlink_parent / module.STAGE4_CALIBRATION_FILENAME,
        )
    module._append_calibration_history(stage4_history, step=4000, summary=summary)
    assert sha256_file(history) == frozen_history_sha256
    assert (
        tuple(stage4_history.read_text().splitlines()[0].split(","))
        == CALIBRATION_COLUMNS
    )
    assert len(stage4_history.read_text().splitlines()) == 2
    module._append_calibration_history(stage4_history, step=4000, summary=summary)
    assert len(stage4_history.read_text().splitlines()) == 2
    assert sha256_file(history) == frozen_history_sha256
    assert module._require_stage4_calibration_prefix(
        stage4_history,
        checkpoint_step=4000,
        pending_validation_step=4000,
    ) == (4000,)
    assert module._require_stage4_calibration_prefix(
        stage4_history,
        checkpoint_step=4000,
        pending_validation_step=None,
    ) == (4000,)
    absent_stage4_history = tmp_path / "absent_stage4_calibration_history.csv"
    assert (
        module._require_stage4_calibration_prefix(
            absent_stage4_history,
            checkpoint_step=4000,
            pending_validation_step=4000,
        )
        == ()
    )
    with pytest.raises(Stage4ContractError, match="checkpoint transaction"):
        module._require_stage4_calibration_prefix(
            absent_stage4_history,
            checkpoint_step=4000,
            pending_validation_step=None,
        )
    assert module._require_stage4_calibration_prefix(
        stage4_history,
        checkpoint_step=8000,
        pending_validation_step=8000,
    ) == (4000,)
    drifted_summary = copy.deepcopy(summary)
    drifted_summary["group_a_equal_combination_mean"]["psnr"] = 2.5
    with pytest.raises(Stage4ContractError, match="drifted during replay"):
        module._append_calibration_history(
            stage4_history, step=4000, summary=drifted_summary
        )

    stage4_line = stage4_history.read_text().splitlines()[-1]
    duplicate_history = tmp_path / "duplicate_stage4_calibration_history.csv"
    duplicate_history.write_text(
        stage4_history.read_text() + stage4_line + "\n", encoding="utf-8"
    )
    with pytest.raises(Stage4ContractError, match="multiple Stage4 calibration rows"):
        module._append_calibration_history(
            duplicate_history, step=4000, summary=summary
        )

    for marker_count in range(1, 6):
        partial_history = tmp_path / f"partial_{marker_count}_calibration_history.csv"
        partial_stage4_row = dict(stage3_row)
        for marker in module.STAGE4_CALIBRATION_MARKER_COLUMNS[:marker_count]:
            partial_stage4_row[marker] = 1.0
        append_stage3_calibration_history(partial_history, partial_stage4_row)
        with pytest.raises(Stage4ContractError, match="partial Stage4 calibration row"):
            module._append_calibration_history(
                partial_history, step=4000, summary=summary
            )

    foreign_history = tmp_path / "foreign_stage4_calibration_history.csv"
    append_stage3_calibration_history(foreign_history, stage3_row)
    with pytest.raises(Stage4ContractError, match="non-Stage4 row"):
        module._append_calibration_history(foreign_history, step=4000, summary=summary)
    with pytest.raises(Stage4ContractError, match="missing predecessor"):
        module._append_calibration_history(
            tmp_path / "missing_predecessor.csv", step=8000, summary=summary
        )
    with pytest.raises(Stage4ContractError, match="off the frozen schedule"):
        module._append_calibration_history(
            tmp_path / "off_schedule.csv", step=4001, summary=summary
        )

    module._append_calibration_history(stage4_history, step=8000, summary=summary)
    assert module._require_stage4_calibration_prefix(
        stage4_history,
        checkpoint_step=8000,
        pending_validation_step=8000,
    ) == (4000, 8000)
    for validation_step in range(12000, 40001, 4000):
        module._append_calibration_history(
            stage4_history, step=validation_step, summary=summary
        )
    completed_calibration_sha256 = sha256_file(stage4_history)
    history_lines = stage4_history.read_text(encoding="utf-8").splitlines()
    extra_column_history = tmp_path / "extra_column_history.csv"
    extra_column_history.write_text(
        history_lines[0] + "\n" + history_lines[1] + ",EXTRA\n",
        encoding="utf-8",
    )
    with pytest.raises(Stage4ContractError, match="exactly 28 columns"):
        module._load_stage4_calibration_rows(extra_column_history)
    missing_column_history = tmp_path / "missing_column_history.csv"
    missing_column_history.write_text(
        history_lines[0] + "\n" + ",".join(history_lines[1].split(",")[:-1]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Stage4ContractError, match="exactly 28 columns"):
        module._load_stage4_calibration_rows(missing_column_history)
    gap_history = tmp_path / "gap_stage4_history.csv"
    gap_history.write_text(
        "\n".join((history_lines[0], history_lines[1], history_lines[3])) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Stage4ContractError, match="checkpoint transaction"):
        module._require_stage4_calibration_prefix(
            gap_history,
            checkpoint_step=12000,
            pending_validation_step=12000,
        )
    ahead_history = tmp_path / "ahead_stage4_history.csv"
    ahead_history.write_text("\n".join(history_lines[:3]) + "\n", encoding="utf-8")
    with pytest.raises(Stage4ContractError, match="checkpoint transaction"):
        module._require_stage4_calibration_prefix(
            ahead_history,
            checkpoint_step=4000,
            pending_validation_step=4000,
        )

    best_path = output_dir / "best_ema.pth"
    best_score = stage4_engine.ValidationScore(3.0, 0.3, 2.0, 0.2, 3000)
    torch.save(
        {
            "schema_version": "graphrestore-checkpoint-v1",
            "stage": "stage4",
            "step": 3000,
            "model_role": "ema_selection",
            "resumable": False,
            "pending_validation_step": None,
            "metrics": {
                "best_group_a_psnr": 3.0,
                "best_group_a_ssim": 0.3,
                "best_single_psnr": 2.0,
                "best_single_ssim": 0.2,
                "best_step": 3000.0,
            },
        },
        best_path,
    )
    report_path = tmp_path / "STAGE4_E2E.md"
    final_latest_score = stage4_engine.ValidationScore(2.0, 0.2, 1.0, 0.1, 40000)
    report_path.write_text(
        module._render_report(
            summary,
            step=40000,
            best=best_score,
            checkpoint=best_path,
            calibration_history_routing=routing,
        ),
        encoding="utf-8",
    )
    binding = module._stage4_report_binding(
        report_path,
        selected_best_checkpoint=best_path,
        selected_best_score=best_score,
        latest_score=final_latest_score,
        calibration_history_routing=routing,
        completed_calibration_sha256=completed_calibration_sha256,
    )
    assert binding == {
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
    }
    assert sha256_file(best_path) in report_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    assert "Selected Single PSNR/SSIM: 2.000000 / 0.20000000" in report_text
    assert "Selected Group-A PSNR/SSIM: 3.000000 / 0.30000000" in report_text
    assert "Current Single PSNR/SSIM: 1.000000 / 0.10000000" in report_text
    assert "Current Group-A PSNR/SSIM: 2.000000 / 0.20000000" in report_text
    assert (
        f"Stage0 Group-A PSNR anchor: {module.STAGE0_GROUP_A_PSNR_ANCHOR!r}"
        in report_text
    )
    assert (
        f"Stage0 Group-A SSIM anchor: {module.STAGE0_GROUP_A_SSIM_ANCHOR!r}"
        in report_text
    )
    assert "SSIM_RETENTION_RISK: true" in report_text
    assert "risk is not offset by any average PSNR gain" in report_text
    assert str(stage4_history.resolve()) in report_text
    assert frozen_history_sha256 in report_text
    report_path.write_text(
        report_text.replace("Validation step: 40000", "Validation step: 36000"),
        encoding="utf-8",
    )
    with pytest.raises(Stage4ContractError, match="not bound"):
        module._stage4_report_binding(
            report_path,
            selected_best_checkpoint=best_path,
            selected_best_score=best_score,
            latest_score=final_latest_score,
            calibration_history_routing=routing,
            completed_calibration_sha256=completed_calibration_sha256,
        )
    report_path.write_text(report_text, encoding="utf-8")
    with pytest.raises(Stage4ContractError, match="history/final-step binding"):
        module._stage4_report_binding(
            report_path,
            selected_best_checkpoint=best_path,
            selected_best_score=best_score,
            latest_score=final_latest_score,
            calibration_history_routing=routing,
            completed_calibration_sha256="0" * 64,
        )

    running_status = tmp_path / "RUNNING_STATUS.md"
    running_status.write_text(
        "status: STAGE4_RUNNING\ncurrent_stage: STAGE4\ngpu: owned_by_child_process\n",
        encoding="utf-8",
    )
    latest_score = stage4_engine.ValidationScore(2.5, 0.25, 2.0, 0.2, 4000)
    module._update_stage4_running_status(
        running_status,
        latest=latest_score,
        selected=best_score,
    )
    first_status = running_status.read_text(encoding="utf-8")
    module._update_stage4_running_status(
        running_status,
        latest=latest_score,
        selected=best_score,
    )
    assert running_status.read_text(encoding="utf-8") == first_status
    assert first_status.splitlines()[0] == "status: STAGE4_RUNNING"
    assert "current_stage: STAGE4" in first_status
    assert first_status.count("latest_group_a_psnr:") == 1
    assert first_status.count("stage0_group_a_ssim_anchor:") == 1
    assert "SSIM_RETENTION_RISK: true" in first_status
    assert "SSIM_RETENTION_RISK_NOTE:" in first_status
    retained = stage4_engine.ValidationScore(
        26.0,
        module.STAGE0_GROUP_A_SSIM_ANCHOR,
        28.0,
        0.9,
        8000,
    )
    retained_lines = module._stage4_status_lines(retained, selected=retained)
    assert "SSIM_RETENTION_RISK: false" in retained_lines
    assert not any(
        line.startswith("SSIM_RETENTION_RISK_NOTE:") for line in retained_lines
    )

    decision_memo = tmp_path / "DECISION_MEMO.md"
    decision_memo.write_text(
        "# Decision Memo\n\n"
        "Status: Stage0–2 evidence verified; paused before Stage3 pending explicit "
        "user approval.\n\n"
        "The final baseline choice, complete 2–3 contribution assessment, "
        "ablations, formal MiO100 A/B/C table, paper claim strength and "
        "title/abstract package remain pending because the contract forbids "
        "Stage3/4 and formal B/C evaluation without further user approval.\n",
        encoding="utf-8",
    )
    module._update_stage4_decision_memo(decision_memo, best_score)
    memo_once = decision_memo.read_text(encoding="utf-8")
    module._update_stage4_decision_memo(decision_memo, best_score)
    assert decision_memo.read_text(encoding="utf-8") == memo_once
    assert "Status: Stage3–4 complete" in memo_once
    assert memo_once.count("<!-- STAGE4_SSIM_RETENTION_BEGIN -->") == 1
    assert "SSIM_RETENTION_RISK: `true`" in memo_once
    assert "does not offset the SSIM retention deficit" in memo_once
    resumed_payload = {
        "metrics": {
            "group_a_psnr": 2.0,
            "group_a_ssim": 0.2,
            "single_psnr": 1.0,
            "single_ssim": 0.1,
            "validation_step": 4000.0,
        }
    }
    resumed_current = module._checkpoint_current_score(resumed_payload)
    assert resumed_current == stage4_engine.ValidationScore(2.0, 0.2, 1.0, 0.1, 4000)
    rebound_metrics = module._checkpoint_metrics(
        resumed_current,
        best_score,
        best_checkpoint=best_path,
    )
    assert rebound_metrics["validation_step"] == 4000.0
    assert rebound_metrics["best_step"] == 3000.0
    assert rebound_metrics["best_checkpoint_sha256"] == sha256_file(best_path)
    assert '"report_sha256"' in source
    stage4_history.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(Stage3ContractError):
        module._append_calibration_history(stage4_history, step=8000, summary=summary)
