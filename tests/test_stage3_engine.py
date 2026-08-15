from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import train_stage3_planner
from src.data.samplers import StatefulEpisodeSampler
from src.net import GraphRestore, PlannerOutput
from src.training.optimization import WarmupCosineScheduler
from src.training.stage1_engine import STAGE1_EMA_SCOPE, stage1_ema_policy_metadata
from src.training.stage3_engine import (
    Stage3ContractError,
    Stage3PlannerEMA,
    Stage3SupervisionBatch,
    build_stage3_optimizer,
    calibrate_presence_thresholds,
    load_stage1_ema_into_graphrestore,
    resume_stage3_checkpoint,
    save_stage3_checkpoint,
    set_stage3_trainability,
    stage3_supervision_loss,
    train_stage3_optimizer_step,
)
from src.utils.io import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_graphrestore() -> GraphRestore:
    return GraphRestore(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        refinement=1,
        heads=(1, 1, 1, 1),
        expansion=2.0,
        skill_bottlenecks={
            "level3": 2,
            "level2": 2,
            "level1": 2,
            "refinement": 2,
        },
        planner_fpn_dim=8,
        planner_context_dim=16,
        effect_profile_dim=40,
    )


def _supervision_batch(batch_size: int = 1) -> Stage3SupervisionBatch:
    generator = torch.Generator().manual_seed(71)
    image = torch.rand(batch_size, 3, 16, 16, generator=generator)
    presence = torch.zeros(batch_size, 8)
    presence[:, :2] = 1.0
    guards = torch.zeros(batch_size, 8, 4, 4)
    guards[:, 0] = 0.3
    guards[:, 1] = 0.6
    relation_targets = torch.full((batch_size, 28), -2, dtype=torch.long)
    relation_targets[:, 0] = -1
    relation_weights = torch.zeros(batch_size, 28)
    relation_weights[:, 0] = 0.25
    ambiguous = torch.zeros(batch_size, 28, dtype=torch.bool)
    ambiguous[:, 0] = True
    dense_ids = torch.tensor([False, False, False, False, True, True, True, False])
    return Stage3SupervisionBatch(
        x0=image,
        current=image.clone(),
        presence_targets=presence,
        guard_targets=guards,
        global_severity_targets=guards.mean(dim=(-2, -1)),
        dense_skill_mask=presence.bool() & dense_ids[None, :],
        global_skill_mask=presence.bool() & ~dense_ids[None, :],
        absent_skill_mask=~presence.bool(),
        stop_targets=torch.zeros(batch_size, 1),
        relation_targets=relation_targets,
        relation_weights=relation_weights,
        relation_ambiguous_mask=ambiguous,
        round_values=torch.zeros(batch_size),
        sample_ids=tuple(f"tiny-{index}" for index in range(batch_size)),
        state_kinds=tuple("group_a_pair" for _ in range(batch_size)),
        model_intermediate_count=0,
    )


def test_approval_failure_happens_before_any_cuda_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = dict(load_yaml(PROJECT_ROOT / "configs/stage3_planner.yaml"))
    config["paths"] = dict(config["paths"])
    config["paths"]["resolved_paths"] = str(
        PROJECT_ROOT / "configs/resolved_paths.yaml"
    )
    config["paths"]["required_approval"] = str(tmp_path / "never-approved.json")
    config_path = tmp_path / "stage3.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cuda_calls = 0

    def forbidden_cuda_probe() -> bool:
        nonlocal cuda_calls
        cuda_calls += 1
        raise AssertionError("CUDA was queried before approval")

    monkeypatch.setattr(train_stage3_planner.torch.cuda, "is_available", forbidden_cuda_probe)
    arguments = argparse.Namespace(
        config=config_path,
        resume=None,
        micro_batch=1,
        output_dir=tmp_path / "output",
    )
    with pytest.raises(Stage3ContractError, match="approval is missing"):
        train_stage3_planner.run(arguments)
    assert cuda_calls == 0
    assert not (tmp_path / "output").exists()


def test_only_planner_is_trainable_and_receives_gradients() -> None:
    torch.manual_seed(73)
    model = _tiny_graphrestore()
    counts = set_stage3_trainability(model)
    assert counts["planner"] > 0 and counts["frozen_executor"] > 0
    executor_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("planner.")
    }
    planner_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("planner.") and value.is_floating_point()
    }
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    ema = Stage3PlannerEMA(model, decay=0.9)
    result = train_stage3_optimizer_step(
        model,
        [_supervision_batch()],
        optimizer,
        scheduler=None,
        ema=ema,
        device=torch.device("cpu"),
        use_bf16=False,
        audit_gradients=True,
    )
    assert result.total > 0 and result.grad_norm > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is name.startswith("planner.")
        if not name.startswith("planner."):
            assert parameter.grad is None
    for name, value in executor_before.items():
        torch.testing.assert_close(model.state_dict()[name], value)
        assert torch.equal(ema.shadow[name], value)
    assert any(
        not torch.equal(model.state_dict()[name], value)
        for name, value in planner_before.items()
    )


def test_stage3_parent_loader_rejects_old_stage1_ema_contract(tmp_path: Path) -> None:
    stage1_source = {
        name: value.detach().clone()
        for name, value in _tiny_graphrestore().state_dict().items()
        if not name.startswith("planner.") and name != "presence_thresholds"
    }
    payload = {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage1",
        "model_role": "ema_selection",
        "resumable": False,
        "step": 30_000,
        "model": stage1_source,
        "provenance": {"ema_policy": stage1_ema_policy_metadata(0.9999)},
        "ema": {
            "decay": 0.9999,
            "num_updates": 30_000,
            "scope": STAGE1_EMA_SCOPE,
            "policy": stage1_ema_policy_metadata(0.9999),
            "shadow": {name: value.clone() for name, value in stage1_source.items()},
        },
    }
    valid_path = tmp_path / "best_ema.pth"
    torch.save(payload, valid_path)
    report = load_stage1_ema_into_graphrestore(_tiny_graphrestore(), valid_path)
    assert report.checkpoint_step == 30_000

    cases = (
        ("missing_scope", "scope", None, "scope"),
        ("wrong_scope", "scope", "generic_all_state_ema", "scope"),
        ("missing_policy", "policy", None, "policy"),
        ("wrong_policy", "policy", "wrong", "policy"),
        ("missing_updates", "num_updates", None, "update count"),
        ("wrong_updates", "num_updates", 29_999, "update count"),
        ("wrong_decay", "decay_bundle", 0.9, "decay"),
        ("missing_provenance", "provenance", None, "provenance"),
        ("wrong_provenance", "provenance", "wrong", "provenance EMA policy"),
        ("wrong_dtype", "dtype", torch.float64, "dtype"),
    )
    for label, field, bad_value, error_pattern in cases:
        bad_directory = tmp_path / label
        bad_directory.mkdir()
        bad_path = bad_directory / "best_ema.pth"
        bad_ema = dict(payload["ema"])
        bad_payload = {
            **payload,
            "ema": bad_ema,
            "provenance": dict(payload["provenance"]),
        }
        if field == "decay_bundle":
            bad_ema["decay"] = bad_value
            bad_ema["policy"] = stage1_ema_policy_metadata(float(bad_value))
            bad_payload["provenance"]["ema_policy"] = bad_ema["policy"]
        elif field == "provenance" and bad_value is None:
            bad_payload.pop("provenance")
        elif field == "provenance":
            bad_payload["provenance"]["ema_policy"] = {
                **bad_ema["policy"],
                "buffer_update": "standard_ema",
            }
        elif field == "dtype":
            bad_model = dict(payload["model"])
            bad_shadow = dict(bad_ema["shadow"])
            name = next(
                key for key, value in bad_model.items() if value.is_floating_point()
            )
            bad_model[name] = bad_model[name].to(dtype=bad_value)
            bad_shadow[name] = bad_shadow[name].to(dtype=bad_value)
            bad_payload["model"] = bad_model
            bad_ema["shadow"] = bad_shadow
        elif bad_value is None:
            bad_ema.pop(field)
        elif field == "policy":
            bad_ema[field] = {
                **bad_ema[field],
                "buffer_update": "standard_ema",
            }
        else:
            bad_ema[field] = bad_value
        torch.save(bad_payload, bad_path)
        with pytest.raises(Stage3ContractError, match=error_pattern):
            load_stage1_ema_into_graphrestore(_tiny_graphrestore(), bad_path)


def test_real_stage3_loss_wires_stable_three_class_ambiguous_partial_label() -> None:
    batch = _supervision_batch(batch_size=2)
    # Row 0 ambiguous; row 1 is an ordinary parallel one-hot target.
    batch.relation_targets[1, 0] = 2
    batch.relation_weights[1, 0] = 1.0
    batch.relation_ambiguous_mask[1, 0] = False
    relation = torch.zeros(2, 28, 3, requires_grad=True)
    relation.data[0, 0] = torch.tensor([1.2, -0.3, 0.7])
    relation.data[1, 0] = torch.tensor([-0.5, 0.2, 1.4])
    output = PlannerOutput(
        guard_logits=torch.zeros(2, 8, 4, 4, requires_grad=True),
        presence_logits=torch.zeros(2, 8, requires_grad=True),
        stop_logit=torch.zeros(2, 1, requires_grad=True),
        relation_logits=relation,
        global_context=torch.zeros(2, 4),
    )
    loss, _ = stage3_supervision_loss(output, batch)
    log_prob = torch.log_softmax(relation.float(), dim=-1)
    ambiguous_without_outer_weight = -torch.logsumexp(log_prob[0, 0, :2], dim=-1)
    ordinary = -log_prob[1, 0, 2]
    expected = (0.25 * ambiguous_without_outer_weight + ordinary) / 1.25
    torch.testing.assert_close(loss.relation, expected)
    loss.total.backward()
    assert relation.grad is not None
    assert torch.isfinite(relation.grad).all()
    assert relation.shape[-1] == 3


def test_threshold_ties_are_stable_and_choose_lowest_grid_value() -> None:
    probabilities = torch.full((6, 8), 0.9)
    targets = torch.ones(6, 8)
    first = calibrate_presence_thresholds(probabilities, targets)
    second = calibrate_presence_thresholds(probabilities, targets)
    assert first == second
    assert first.thresholds == (0.20,) * 8
    assert first.tie_break == "lowest_threshold"


@dataclass(frozen=True)
class _Record:
    operator_order: tuple[str, ...]


class _SamplerDataset:
    def __init__(self) -> None:
        singles = tuple((name,) for name in (
            "noise",
            "motion blur",
            "defocus blur",
            "jpeg compression artifact",
            "rain",
            "haze",
            "dark",
            "low resolution",
        ))
        pairs = (
            ("rain", "haze"),
            ("motion blur", "low resolution"),
            ("dark", "noise"),
            ("defocus blur", "jpeg compression artifact"),
            ("noise", "jpeg compression artifact"),
            ("rain", "low resolution"),
            ("motion blur", "dark"),
            ("defocus blur", "haze"),
        )
        self.records = tuple(_Record(value) for value in (*singles, *pairs))

    def __len__(self) -> int:
        return len(self.records)

    def set_worker_seed(self, seed: int) -> None:
        del seed


def _sampler() -> StatefulEpisodeSampler:
    return StatefulEpisodeSampler(
        _SamplerDataset(),
        num_samples=32,
        stage="stage3",
        effective_batch_size=8,
        base_seed=2027,
        start_step=0,
    )


def test_stage3_checkpoint_exact_resume(tmp_path: Path) -> None:
    random.seed(79)
    np.random.seed(79)
    torch.manual_seed(79)
    model = _tiny_graphrestore()
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    ema = Stage3PlannerEMA(model, decay=0.9)
    train_stage3_optimizer_step(
        model,
        [_supervision_batch()],
        optimizer,
        scheduler,
        ema,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    sampler = _sampler()
    provenance = {
        "stage3_approval": {"sha256": "a" * 64},
        "bindings": {"primary_train_manifest": {"sha256": "b" * 64}},
    }
    checkpoint = tmp_path / "last.pth"
    save_stage3_checkpoint(
        checkpoint,
        step=1,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
    )
    expected_random = torch.rand(4)

    restored = _tiny_graphrestore()
    restored_optimizer = build_stage3_optimizer(restored, fused_if_supported=False)
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    restored_ema = Stage3PlannerEMA(restored, decay=0.9)
    restored_sampler = _sampler()
    payload = resume_stage3_checkpoint(
        checkpoint,
        model=restored,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=restored_sampler,
        expected_provenance=provenance,
    )
    assert payload["step"] == 1
    assert payload["model_role"] == "raw_training_state"
    assert payload["resumable"] is True
    torch.testing.assert_close(torch.rand(4), expected_random)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert restored_sampler.state_dict()["consumed_optimizer_step"] == 1


def test_stage3_best_ema_is_selection_only_and_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    torch.manual_seed(83)
    model = _tiny_graphrestore()
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    ema = Stage3PlannerEMA(model, decay=0.9)
    sampler = _sampler()
    provenance = {"stage3_approval": {"sha256": "c" * 64}}
    best = tmp_path / "best_ema.pth"
    save_stage3_checkpoint(
        best,
        step=1,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        model_as_ema=True,
    )
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["model_role"] == "ema_selection"
    assert payload["resumable"] is False

    victim = _tiny_graphrestore()
    victim_optimizer = build_stage3_optimizer(victim, fused_if_supported=False)
    victim_scheduler = WarmupCosineScheduler(
        victim_optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    victim_ema = Stage3PlannerEMA(victim, decay=0.9)
    before = {name: value.detach().clone() for name, value in victim.state_dict().items()}
    with pytest.raises(Stage3ContractError, match="non-resumable"):
        resume_stage3_checkpoint(
            best,
            model=victim,
            ema=victim_ema,
            optimizer=victim_optimizer,
            scheduler=victim_scheduler,
            sampler=_sampler(),
            expected_provenance=provenance,
        )
    for name, expected in before.items():
        assert torch.equal(victim.state_dict()[name], expected)
