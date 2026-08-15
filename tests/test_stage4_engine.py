from __future__ import annotations

import copy
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from src.data.manifests import (
    ALLOWED_GROUP_A,
    ALLOWED_SINGLE,
    OperatorParameter,
    PrimaryRecipe,
    SKILLS,
)
from src.net import GraphRestore
from src.net.graph_compiler import PAIR_TO_ROW
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import WarmupCosineScheduler
from src.training.stage3_engine import CALIBRATION_COLUMNS, Stage3ContractError
from src.training.stage4_engine import (
    STAGE4_SCHEMA,
    Stage4Batch,
    Stage4ContractError,
    Stage4EpisodeDataset,
    Stage4EpisodeSampler,
    Stage4ProgramOutput,
    Stage4Request,
    build_stage4_optimizer,
    load_presence_thresholds,
    load_stage3_best_ema,
    resume_stage4_checkpoint,
    run_stage4_program,
    save_stage4_checkpoint,
    set_stage4_trainability,
    stage4_image_loss,
    stage4_parameter_role,
    stage4_ssim_weight,
    teacher_forcing_probability,
    train_stage4_optimizer_step,
    validate_stage3_approval,
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
        relation_row=torch.full(
            (batch_size,), PAIR_TO_ROW[(0, 1)], dtype=torch.long
        ),
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
    assert counts["group_a_pair_restoration"] / len(requests) == pytest.approx(0.70, abs=0.01)
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
    clean_episode = dataset[
        Stage4Request(0, "clean_misuse", 0, 0, False, (2, 5))
    ]
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


def test_stage4_optimizer_roles_include_deep_downsamples_and_freeze_early_encoder() -> None:
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
        if name.startswith(("encoder.down23.", "encoder.level3.", "encoder.down34.", "encoder.level4.")):
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
    assert assigned == {id(value) for value in model.parameters() if value.requires_grad}


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
    for compiled, state in zip(output.compiled_graphs, output.graph_states, strict=True):
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
    ema = ExponentialMovingAverage(model, decay=0.9999)
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
    ema_state["scope"] = "planner_parameters_only_executor_bitwise_frozen"
    checkpoint = tmp_path / "best_ema.pth"
    torch.save(
        {
            "schema_version": "graphrestore-checkpoint-v1",
            "stage": "stage3",
            "step": 12_000,
            "model_role": "ema_selection",
            "resumable": False,
            "model": ema.shadow,
            "ema": ema_state,
            "optimizer": {"must_not_be_loaded": True},
            "provenance": {"stage3_approval": {"sha256": approval_sha}},
            "executor_frozen": True,
            "trainable_prefixes": ["planner."],
        },
        checkpoint,
    )
    target = _tiny_model()
    snapshot = load_stage3_best_ema(
        checkpoint, model=target, approval_sha256=approval_sha
    )
    assert snapshot.checkpoint_step == 12_000
    assert all(
        torch.equal(value, target.state_dict()[name])
        for name, value in model.state_dict().items()
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
            "thresholds": {name: 0.20 + 0.02 * index for index, name in enumerate(SKILLS)},
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
    stale = json.loads(thresholds_path.read_text())
    stale["checkpoint_sha256"] = "f" * 64
    atomic_write_json(thresholds_path, stale)
    with pytest.raises(Stage4ContractError):
        load_presence_thresholds(
            thresholds_path, stage3_checkpoint_sha256=snapshot.checkpoint_sha256
        )


def test_stage4_checkpoint_best_is_ema_and_resume_is_exact(tmp_path: Path) -> None:
    dataset = _fake_sampling_dataset()
    sampler = Stage4EpisodeSampler(dataset, num_samples=400)
    model = _tiny_model()
    optimizer = build_stage4_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=800, max_steps=40_000, min_lr=5.0e-7
    )
    ema = ExponentialMovingAverage(model)
    for value in ema.shadow.values():
        if value.is_floating_point():
            value.add_(0.125)
    provenance = {"schema": STAGE4_SCHEMA, "bound": "abc"}
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
        model_as_ema=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["model_role"] == "ema_selection"
    assert payload["resumable"] is False
    assert all(
        torch.equal(payload["model"][name], payload["ema"]["shadow"][name])
        for name in payload["model"]
    )

    restored_model = _tiny_model()
    restored_optimizer = build_stage4_optimizer(restored_model, fused_if_supported=False)
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer, warmup_steps=800, max_steps=40_000, min_lr=5.0e-7
    )
    restored_ema = ExponentialMovingAverage(restored_model)
    restored_sampler = Stage4EpisodeSampler(dataset, num_samples=400)
    before_rejected_resume = {
        name: value.detach().clone() for name, value in restored_model.state_dict().items()
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

    last_checkpoint = tmp_path / "last.pth"
    save_stage4_checkpoint(
        last_checkpoint,
        step=7,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        model_as_ema=False,
    )
    last_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    assert last_payload["model_role"] == "raw_training_state"
    assert last_payload["resumable"] is True
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


def test_stage4_formal_cli_argv_and_shared_calibration_schema(tmp_path: Path) -> None:
    script_path = PROJECT_ROOT / "scripts/train_stage4_e2e.py"
    spec = importlib.util.spec_from_file_location("stage4_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    arguments = parser.parse_args(
        ["--config", "configs/stage4_graphrestore_e2e.yaml"]
    )
    assert arguments.config == Path("configs/stage4_graphrestore_e2e.yaml")
    assert arguments.resume is None
    assert arguments.max_steps is None
    orchestration = (PROJECT_ROOT / "src/training/orchestration.py").read_text()
    assert '"scripts/train_stage4_e2e.py"' in orchestration
    assert '"configs/stage4_graphrestore_e2e.yaml"' in orchestration

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
        "single_equal_task_mean": {"psnr": 1.0, "ssim": 0.1},
        "group_a_equal_combination_mean": {"psnr": 2.0, "ssim": 0.2},
        "diagnostics": diagnostics,
    }
    history = tmp_path / "calibration_history.csv"
    module._append_calibration_history(history, step=4000, summary=summary)
    assert tuple(history.read_text().splitlines()[0].split(",")) == CALIBRATION_COLUMNS
    history.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(Stage3ContractError):
        module._append_calibration_history(history, step=8000, summary=summary)
