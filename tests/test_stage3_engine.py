from __future__ import annotations

import argparse
import contextlib
import copy
import inspect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch
import yaml

from scripts import finalize_stage3, train_stage3_planner
from src.training import stage3_engine
from src.data import GraphRestoreEpisodeDataset
from src.data.samplers import StatefulEpisodeSampler
from src.net import GraphRestore, PlannerOutput
from src.net import graphrestore as graphrestore_module
from src.net.graph_compiler import CompiledGraph
from src.net.graphrestore import GraphRestoreOutput, ProgramGraphState, RoundTrace
from src.net.restormer_blocks import crop_to_shape, pad_to_multiple
from src.training.optimization import WarmupCosineScheduler
from src.training.ema import ExponentialMovingAverage
from src.training.stage1_engine import STAGE1_EMA_SCOPE, stage1_ema_policy_metadata
from src.training.stage3_engine import (
    CALIBRATION_COLUMNS,
    STAGE3_ALLOCATOR_CONF,
    Stage3ContractError,
    Stage3OptimizerTransaction,
    Stage3ParentLoadReport,
    Stage3PlannerEMA,
    Stage3SupervisionBatch,
    append_calibration_history,
    build_stage3_optimizer,
    build_stage3_provenance,
    calibrate_presence_thresholds,
    enforce_stage3_peak_memory,
    load_stage1_ema_into_graphrestore,
    load_stage3_best_ema,
    probe_stage3_validation_vram,
    resume_stage3_checkpoint,
    reset_stage3_peak_memory,
    save_stage3_checkpoint,
    set_stage3_trainability,
    stage3_supervision_loss,
    train_stage3_optimizer_step,
    validate_stage3_allocator_conf,
    validate_stage3_pending_validation_step,
    validate_stage3_validation_vram_evidence,
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


def _resume_target(source: GraphRestore) -> GraphRestore:
    target = _tiny_graphrestore()
    state = target.state_dict()
    for name, value in source.state_dict().items():
        if not name.startswith("planner."):
            state[name] = value.detach().clone()
    target.load_state_dict(state, strict=True)
    return target


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


def test_stage3_full_resolution_guard_diagnostics_crop_only_4mod8_padding() -> None:
    single_tasks = tuple((name,) for name in stage3_engine.SKILLS)
    group_a_tasks = (
        ("rain", "haze"),
        ("motion_blur", "low_resolution"),
        ("low_light", "noise"),
        ("defocus_blur", "jpeg_artifact"),
        ("noise", "jpeg_artifact"),
        ("rain", "low_resolution"),
        ("motion_blur", "low_light"),
        ("defocus_blur", "haze"),
    )
    records = tuple(
        SimpleNamespace(
            sample_id=f"single-{index}",
            group="single",
            skill_names=names,
            skill_ids=tuple(stage3_engine.SKILL_TO_ID[name] for name in names),
        )
        for index, names in enumerate(single_tasks)
    ) + tuple(
        SimpleNamespace(
            sample_id=f"pair-{index}",
            group="A",
            skill_names=names,
            skill_ids=tuple(stage3_engine.SKILL_TO_ID[name] for name in names),
        )
        for index, names in enumerate(group_a_tasks)
    )

    class FullResolutionDataset:
        training = False
        crop_size = None

        def __init__(self) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            record = self.records[index]
            image = torch.zeros(3, 12, 20)
            presence = torch.zeros(len(stage3_engine.SKILLS))
            presence[list(record.skill_ids)] = 1.0
            guards = torch.zeros(len(stage3_engine.SKILLS), 3, 5)
            pattern = torch.linspace(0.0, 1.0, 15).reshape(3, 5)
            for skill_id in record.skill_ids:
                if stage3_engine.SKILLS[skill_id] in {"rain", "haze"}:
                    guards[skill_id] = pattern
            return {
                "input": image,
                "gt_clean": image.clone(),
                "presence_target": presence,
                "guard_targets": guards,
            }

    class PaddedGuardModel(torch.nn.Module):
        def forward(self, image: torch.Tensor, **_: Any) -> GraphRestoreOutput:
            # A 12x20 full-resolution input is 4mod8 on both axes.  The model
            # pads it to 16x24, so its H/4 planner map is 4x6 while the target
            # on the original image support is 3x5.
            planner = PlannerOutput(
                guard_logits=torch.zeros(1, len(stage3_engine.SKILLS), 4, 6),
                presence_logits=torch.zeros(1, len(stage3_engine.SKILLS)),
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

    relation_val = {
        record.sample_id: {
            "sample_id": record.sample_id,
            "split": "val",
            "skill_ids": tuple(sorted(record.skill_ids)),
            "label": "parallel",
            "relation_class_index": 2,
            "pair_id": "+".join(record.skill_names),
        }
        for record in records
        if record.group == "A"
    }
    summary = stage3_engine.validate_stage3(
        PaddedGuardModel(),
        FullResolutionDataset(),  # type: ignore[arg-type]
        relation_val,
        device=torch.device("cpu"),
        use_bf16=False,
        presence_threshold=(0.52, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50),
    )
    assert summary["graph"]["sample_count"] == 16
    assert summary["restoration"]["single"]["count"] == 8
    assert summary["restoration"]["group_a"]["count"] == 8
    assert summary["guard"]["present_guard_images_rain"] == 3
    assert summary["guard"]["present_guard_images_haze"] == 3
    assert summary["checkpoint_presence_threshold"] == [
        0.52,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    assert summary["planner"]["activation_rate"] == 0.875
    assert summary["graph"]["sample_stop_rate"] == 0.0
    assert math.isfinite(summary["relation"]["learned_raw"]["macro_f1"])
    assert math.isfinite(summary["relation"]["learned_raw"]["balanced_accuracy"])


@pytest.mark.parametrize(
    ("height_delta", "width_delta"),
    ((0, 0), (1, 0), (0, 1), (1, 1)),
)
def test_guard_diagnostic_alignment_crops_only_right_bottom_pattern(
    height_delta: int, width_delta: int
) -> None:
    target_height, target_width = 3, 5
    predicted = torch.arange(
        8 * (target_height + height_delta) * (target_width + width_delta),
        dtype=torch.float32,
    ).reshape(8, target_height + height_delta, target_width + width_delta)
    target = torch.zeros(8, target_height, target_width)

    aligned = stage3_engine.align_guard_prediction_to_target(predicted, target)

    assert torch.equal(
        aligned,
        predicted[..., :target_height, :target_width],
    )
    if (height_delta, width_delta) == (0, 0):
        assert torch.equal(aligned, predicted)
        assert aligned.data_ptr() == predicted.data_ptr()


@pytest.mark.parametrize(
    ("predicted_shape", "target_shape"),
    (
        ((8, 2, 5), (8, 3, 5)),
        ((8, 5, 5), (8, 3, 5)),
        ((7, 3, 5), (8, 3, 5)),
        ((1, 8, 3, 5), (8, 3, 5)),
    ),
)
def test_stage3_guard_diagnostic_alignment_rejects_nonpadding_shapes(
    predicted_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="guard map shape mismatch"):
        stage3_engine.align_guard_prediction_to_target(
            torch.zeros(predicted_shape), torch.zeros(target_shape)
        )


def test_approval_failure_happens_before_any_cuda_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_stage3_planner, "PROJECT_ROOT", tmp_path)
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

    monkeypatch.setattr(
        train_stage3_planner.torch.cuda, "is_available", forbidden_cuda_probe
    )
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


def test_threshold_ties_are_stable_and_choose_nearest_half() -> None:
    probabilities = torch.full((6, 8), 0.9)
    targets = torch.ones(6, 8)
    first = calibrate_presence_thresholds(probabilities, targets)
    second = calibrate_presence_thresholds(probabilities, targets)
    assert first == second
    assert first.thresholds == (0.50,) * 8
    assert first.tie_break == "nearest_0.50_then_higher_threshold"


def test_threshold_equal_distance_tie_chooses_higher_threshold() -> None:
    probabilities = torch.tensor([[0.49], [0.49], [0.50], [0.53]]).repeat(1, 8)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]]).repeat(1, 8)
    result = calibrate_presence_thresholds(probabilities, targets)
    assert result.thresholds == (0.52,) * 8
    assert result.tie_break == "nearest_0.50_then_higher_threshold"


def test_frozen_thresholds_bind_metrics_code_and_preserve_checkpoint(
    tmp_path: Path,
) -> None:
    probabilities = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
        ]
    )
    targets = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    manifest = tmp_path / "primary_val.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "best_ema.pth"
    checkpoint.write_bytes(b"immutable-selected-stage3")
    destination = tmp_path / "planner_thresholds.json"
    selected_sha = stage3_engine.sha256_file(checkpoint)
    calibration = calibrate_presence_thresholds(probabilities, targets)
    payload = stage3_engine.freeze_presence_thresholds(
        destination,
        calibration,
        primary_val_manifest=manifest,
        selected_checkpoint=checkpoint,
        approval_sha256="a" * 64,
        extension_authorization_sha256="b" * 64,
        finalization_authorization_sha256="c" * 64,
    )
    assert stage3_engine.sha256_file(checkpoint) == selected_sha
    assert payload["tie_break"] == "nearest_0.50_then_higher_threshold"
    assert payload["numerical_tolerance"] == 1.0e-15
    assert payload["calibration_code"]["sha256"] == stage3_engine.sha256_file(
        Path(stage3_engine.__file__)
    )
    assert payload["macro_f1_after"] >= payload["macro_f1_before"] - 1.0e-15
    for skill in stage3_engine.SKILLS:
        metrics = payload["per_skill_metrics"][skill]
        assert metrics["calibrated"]["f1"] >= metrics["baseline"]["f1"] - 1.0e-15


def test_relation_cpu_baselines_exclude_ambiguous_and_are_finite() -> None:
    rows = {
        "a": {"label": "i_before_j", "relation_class_index": 0, "pair_id": "x+y"},
        "b": {"label": "parallel", "relation_class_index": 2, "pair_id": "x+y"},
        "c": {"label": "ambiguous", "relation_class_index": None, "pair_id": "x+y"},
    }
    prior = {
        "pair_prior": {"x+y": {"i_before_j": 0.2, "j_before_i": 0.1, "parallel": 0.7}}
    }
    audit = stage3_engine.relation_baseline_audit(
        rows,
        prior,
        learned_raw_accuracy=0.5,
    )
    assert audit["n_total"] == 3
    assert audit["n_non_ambiguous"] == 2
    assert audit["n_ambiguous_excluded"] == 1
    assert audit["learned_raw_accuracy"] == 0.5
    assert audit["always_parallel"]["accuracy"] == 0.5
    assert audit["per_pair_majority_prior"]["accuracy"] == 0.5
    assert audit["mio100_rows_read"] == 0


def test_dedicated_stage3_finalizer_contains_no_training_or_checkpoint_writer() -> None:
    source = (PROJECT_ROOT / "scripts/finalize_stage3.py").read_text(encoding="utf-8")
    for forbidden in (
        "build_dataloader",
        "build_stage3_optimizer",
        "WarmupCosineScheduler",
        "save_stage3_checkpoint",
        "atomic_torch_save",
        "StatefulEpisodeSampler",
    ):
        assert forbidden not in source


def test_dedicated_finalizer_executes_production_validation_api() -> None:
    single_tasks = tuple((name,) for name in stage3_engine.SKILLS)
    pair_tasks = (
        ("rain", "haze"),
        ("motion_blur", "low_resolution"),
        ("low_light", "noise"),
        ("defocus_blur", "jpeg_artifact"),
        ("noise", "jpeg_artifact"),
        ("rain", "low_resolution"),
        ("motion_blur", "low_light"),
        ("defocus_blur", "haze"),
    )
    records = tuple(
        SimpleNamespace(
            sample_id=f"single-{index}",
            group="single",
            skill_names=names,
            skill_ids=tuple(stage3_engine.SKILL_TO_ID[name] for name in names),
        )
        for index, names in enumerate(single_tasks)
    ) + tuple(
        SimpleNamespace(
            sample_id=f"pair-{index}",
            group="A",
            skill_names=names,
            skill_ids=tuple(stage3_engine.SKILL_TO_ID[name] for name in names),
        )
        for index, names in enumerate(pair_tasks)
    )

    class Dataset:
        training = False
        crop_size = None

        def __len__(self) -> int:
            return len(records)

        def __init__(self) -> None:
            self.records = records

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            record = records[index]
            presence = torch.zeros(8)
            presence[list(record.skill_ids)] = 1.0
            return {
                "input": torch.zeros(3, 16, 16),
                "gt_clean": torch.zeros(3, 16, 16),
                "presence_target": presence,
                "guard_targets": torch.zeros(8, 4, 4),
            }

    relations = {
        record.sample_id: {
            "sample_id": record.sample_id,
            "split": "val",
            "skill_ids": tuple(sorted(record.skill_ids)),
            "label": "parallel",
            "relation_class_index": 2,
            "pair_id": "+".join(record.skill_names),
        }
        for record in records
        if record.group == "A"
    }
    summary = finalize_stage3._run_post_calibration_diagnostic(
        _tiny_graphrestore().eval(),
        Dataset(),  # type: ignore[arg-type]
        relations,
        device=torch.device("cpu"),
        threshold_values=(0.5,) * 8,
        use_bf16=False,
    )
    assert summary["graph"]["sample_count"] == 16
    assert summary["checkpoint_presence_threshold"] == [0.5] * len(stage3_engine.SKILLS)
    assert summary["presence_thresholds"] == {
        skill: 0.5 for skill in stage3_engine.SKILLS
    }
    assert all(
        row["threshold"] == 0.5 for row in summary["planner"]["per_skill"].values()
    )


def test_dedicated_finalizer_report_is_nonempty_and_records_zero_training(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_ema.pth"
    checkpoint.write_bytes(b"selected")
    thresholds = tmp_path / "planner_thresholds.json"
    thresholds.write_text(
        json.dumps({"macro_f1_before": 0.7, "macro_f1_after": 0.8}),
        encoding="utf-8",
    )
    summary = _report_summary()
    summary["planner"]["activation_rate"] = 0.25
    summary["planner"]["per_skill"] = {
        skill: {
            "threshold": 0.5,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.7466666667,
            "activation_rate": 0.25,
        }
        for skill in stage3_engine.SKILLS
    }
    summary["relation"]["learned_raw"] = {
        "accuracy": 0.6,
        "macro_f1": 0.5,
        "balanced_accuracy": 0.55,
    }
    summary["relation"]["cpu_baseline_audit"] = {
        "always_parallel": {
            "accuracy": 0.4,
            "macro_f1": 0.2,
            "balanced_accuracy": 1 / 3,
        },
        "per_pair_majority_prior": {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "balanced_accuracy": 0.45,
        },
    }
    summary["graph"].update({"mean_program_levels": 1.2, "sample_stop_rate": 0.1})
    score = stage3_engine.ValidationScore(25.0, 0.79, 28.0, 0.88, 12_000)
    report = finalize_stage3._render_report(
        summary,
        original=summary,
        score=score,
        checkpoint=checkpoint,
        thresholds=thresholds,
        authorization=SimpleNamespace(sha256="a" * 64),  # type: ignore[arg-type]
    )
    assert isinstance(report, str) and report
    assert "selected step: 12000" in report
    assert (
        "optimizer steps executed / checkpoint writes / sampler steps advanced: 0 / 0 / 0"
        in report
    )
    assert "STOP-rate definition" in report


def test_stage3_threshold_calibration_padding() -> None:
    # Calibration and formal GraphRestore inference resolve the same canonical
    # padding/cropping function objects; source checks bind those utilities to
    # the two production call sites rather than a test-only implementation.
    assert stage3_engine.pad_to_multiple is pad_to_multiple
    assert graphrestore_module.pad_to_multiple is pad_to_multiple
    assert graphrestore_module.crop_to_shape is crop_to_shape
    calibration_source = inspect.getsource(stage3_engine.collect_primary_val_presence)
    forward_source = inspect.getsource(GraphRestore.forward)
    assert "pad_to_multiple(image, 8)" in calibration_source
    assert "pad_to_multiple(x, 8)" in forward_source
    assert "crop_to_shape(current, original_shape)" in forward_source

    class FullResolutionDataset:
        training = False
        crop_size = None

        def __init__(self, image: torch.Tensor) -> None:
            self.image = image

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            assert index == 0
            return {
                "input": self.image,
                "presence_target": torch.tensor(
                    [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                ),
            }

    class PresenceProbe(torch.nn.Module):
        def encode(self, image: torch.Tensor) -> tuple[torch.Tensor]:
            return (image.mean(dim=(-2, -1), keepdim=True),)

        def plan_state(
            self,
            x0: torch.Tensor,
            current: torch.Tensor,
            features: tuple[torch.Tensor],
            *,
            round_value: float,
            compute_relations: bool,
        ) -> SimpleNamespace:
            assert torch.equal(x0, current)
            assert round_value == 0.0
            assert compute_relations is False
            base = features[0].mean(dim=(1, 2, 3), keepdim=False)
            probabilities = torch.sigmoid(
                base[:, None] + torch.arange(8, device=x0.device)[None, :] / 10.0
            )
            return SimpleNamespace(presence_probabilities=probabilities)

    model = PresenceProbe().eval()
    for height, width in ((16, 24), (12, 24), (16, 20), (12, 20)):
        image = torch.linspace(0.0, 1.0, 3 * height * width).reshape(3, height, width)
        dataset = FullResolutionDataset(image)
        probabilities, targets = stage3_engine.collect_primary_val_presence(
            model,  # type: ignore[arg-type]
            dataset,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            use_bf16=False,
        )
        padded, original_shape = stage3_engine.pad_to_multiple(image.unsqueeze(0), 8)
        cropped = crop_to_shape(padded, original_shape)
        expected = model.plan_state(
            padded,
            padded,
            model.encode(padded),
            round_value=0.0,
            compute_relations=False,
        ).presence_probabilities
        assert original_shape == (height, width)
        assert torch.equal(cropped, image.unsqueeze(0))
        assert tuple(probabilities.shape) == (1, 8)
        assert torch.isfinite(probabilities).all()
        assert torch.equal(probabilities, expected.float())
        assert torch.equal(targets, dataset[0]["presence_target"].unsqueeze(0))

    # The adjudication also requires one physical primary_val image whose
    # dimensions are not already divisible by eight.  This remains a read-only
    # test and guards the selected checkpoint hash across the data access.
    manifest = (
        PROJECT_ROOT.parent.parent / "graph/training_data/manifests/primary_val.jsonl"
    )
    training_root = PROJECT_ROOT.parent.parent / "graph/training_data"
    selected = PROJECT_ROOT / "artifacts/checkpoints/stage3/best_ema.pth"
    if not (manifest.is_file() and training_root.is_dir() and selected.is_file()):
        pytest.skip("formal primary_val/selected Stage3 artifacts are unavailable")
    best_sha_before = stage3_engine.sha256_file(selected)
    real_dataset = GraphRestoreEpisodeDataset(
        manifest,
        training_root,
        PROJECT_ROOT / "artifacts/cache/agenticir_depth_compat",
        crop_size=None,
        training=False,
        stage="stage3",
        base_seed=2027,
        agenticir_repo=PROJECT_ROOT.parent.parent / "graph/upstream/AgenticIR",
        mioir_repo=PROJECT_ROOT.parent.parent / "graph/upstream/MiOIR",
    )
    real_sample = real_dataset[0]
    real_image = real_sample["input"]
    assert real_image.shape[-2] % 8 or real_image.shape[-1] % 8
    real_wrapper = FullResolutionDataset(real_image)
    real_probabilities, _ = stage3_engine.collect_primary_val_presence(
        model,  # type: ignore[arg-type]
        real_wrapper,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        use_bf16=False,
    )
    real_padded, real_original_shape = stage3_engine.pad_to_multiple(
        real_image.unsqueeze(0), 8
    )
    assert torch.equal(
        crop_to_shape(real_padded, real_original_shape), real_image.unsqueeze(0)
    )
    real_expected = model.plan_state(
        real_padded,
        real_padded,
        model.encode(real_padded),
        round_value=0.0,
        compute_relations=False,
    ).presence_probabilities
    assert tuple(real_probabilities.shape) == (1, 8)
    assert torch.isfinite(real_probabilities).all()
    assert torch.equal(real_probabilities, real_expected.float())
    assert stage3_engine.sha256_file(selected) == best_sha_before


@dataclass(frozen=True)
class _Record:
    operator_order: tuple[str, ...]


class _SamplerDataset:
    def __init__(self) -> None:
        singles = tuple(
            (name,)
            for name in (
                "noise",
                "motion blur",
                "defocus blur",
                "jpeg compression artifact",
                "rain",
                "haze",
                "dark",
                "low resolution",
            )
        )
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


def _validation_vram_evidence() -> dict[str, object]:
    topologies = [
        {
            "compiler_mode": "forced_total_order",
            "active_skill_count": 3,
            "completed_rounds": 3,
            "active_skill_counts_by_round": [1, 1, 1],
            "metric_psnr": 10.0,
            "metric_ssim": 0.5,
            "peak_reserved_bytes": 600,
            "peak_reserved_fraction": 0.6,
            "finite": True,
            "passed": True,
        },
        {
            "compiler_mode": "parallel_only",
            "active_skill_count": 3,
            "completed_rounds": 1,
            "active_skill_counts_by_round": [3],
            "metric_psnr": 11.0,
            "metric_ssim": 0.6,
            "peak_reserved_bytes": 700,
            "peak_reserved_fraction": 0.7,
            "finite": True,
            "passed": True,
        },
    ]
    return {
        "schema_version": "graphrestore-stage3-validation-vram-gate-v1",
        "image_size": 2040,
        "max_rounds": 3,
        "completed_rounds": 3,
        "topologies": topologies,
        "peak_reserved_bytes": 700,
        "peak_reserved_fraction": 0.7,
        "maximum_peak_reserved_fraction": 0.9,
        "resident_optimizer_state_entries": 3,
        "resident_optimizer_state_bytes": 100,
        "resident_ema_bytes": 200,
        "optimizer_state_empty_after": True,
        "finite": True,
        "passed": True,
    }


@dataclass
class _CheckpointFixture:
    path: Path
    model: GraphRestore
    optimizer: torch.optim.Optimizer
    scheduler: WarmupCosineScheduler
    ema: Stage3PlannerEMA
    provenance: dict[str, object]


def _stage3_test_provenance(**extra: object) -> dict[str, object]:
    return {
        "stage3_approval": {"sha256": "a" * 64},
        "runtime": {"max_steps": 4, "training_target_step": 4},
        **extra,
    }


def _extension_authorization_fixture(
    tmp_path: Path,
) -> tuple[Path, SimpleNamespace]:
    root = tmp_path.resolve()
    approval_path = root / "artifacts/approvals/STAGE3_APPROVED.json"
    required_path = root / "artifacts/approvals/STAGE3_APPROVAL_REQUIRED.json"
    config_path = root / "configs/stage3_planner.yaml"
    migration = (
        root / "artifacts/migrations" / stage3_engine.STAGE3_EXTENSION_MIGRATION_NAME
    )
    for path, content in (
        (approval_path, b"approval"),
        (required_path, b"required"),
        (config_path, b"config"),
        (migration / "run_contract.json", b"contract-before"),
        (migration / "last.pth", b"last-before"),
        (migration / "best_ema.pth", b"best-before"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for name in ("run_contract.json", "last.pth", "best_ema.pth"):
        (migration / name).chmod(0o444)
    approval_sha = stage3_engine.sha256_file(approval_path)
    required_sha = stage3_engine.sha256_file(required_path)
    authorization = root / "artifacts/approvals/STAGE3_EXTENSION_APPROVED.json"
    payload = {
        "schema_version": stage3_engine.STAGE3_EXTENSION_SCHEMA,
        "kind": "stage3_extension_approval",
        "protocol_id": stage3_engine.PROTOCOL_ID,
        "approved": True,
        "cycles": 3,
        "base_step": 12_000,
        "target_step": 18_000,
        "validation_every_steps": 2_000,
        "validation_steps": [14_000, 16_000, 18_000],
        "schedule_horizon_steps": 12_000,
        "min_lr": 2.0e-6,
        "lr_policy": stage3_engine.STAGE3_EXTENSION_LR_POLICY,
        "authorized_pipeline": ["stage3_extension", "stage4"],
        "formal_mio100_authorized": False,
        "base_stage3_approval": {
            "path": str(approval_path),
            "sha256": approval_sha,
        },
        "base_approval_required": {
            "path": str(required_path),
            "sha256": required_sha,
        },
        "base_stage3_config": {
            "path": str(config_path),
            "sha256": stage3_engine.sha256_file(config_path),
        },
        "pre_extension_run_contract": {
            "path": str(migration / "run_contract.json"),
            "sha256": stage3_engine.sha256_file(migration / "run_contract.json"),
        },
        "pre_extension_last_checkpoint": {
            "path": str(migration / "last.pth"),
            "sha256": stage3_engine.sha256_file(migration / "last.pth"),
        },
        "pre_extension_best_checkpoint": {
            "path": str(migration / "best_ema.pth"),
            "sha256": stage3_engine.sha256_file(migration / "best_ema.pth"),
        },
    }
    authorization.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    paths = SimpleNamespace(
        project_root=root,
        config_path=config_path,
        approval=SimpleNamespace(
            approval_path=approval_path,
            approval_sha256=approval_sha,
            approval_required_path=required_path,
            approval_required_sha256=required_sha,
        ),
    )
    return authorization, paths


def _saved_checkpoint(
    tmp_path: Path,
    *,
    steps: int = 1,
    name: str = "last.pth",
    metrics: dict[str, float | int] | None = None,
    model_as_ema: bool = False,
    pending_validation_step: int | None = None,
) -> _CheckpointFixture:
    model = _tiny_graphrestore()
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    ema = Stage3PlannerEMA(model, decay=0.9)
    sampler = _sampler()
    for step in range(1, steps + 1):
        train_stage3_optimizer_step(
            model,
            [_supervision_batch()],
            optimizer,
            scheduler,
            ema,
            device=torch.device("cpu"),
            use_bf16=False,
        )
        sampler.mark_consumed_optimizer_step(step)
    provenance = _stage3_test_provenance()
    path = tmp_path / name
    save_stage3_checkpoint(
        path,
        step=steps,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        metrics=metrics,
        model_as_ema=model_as_ema,
        pending_validation_step=pending_validation_step,
        validation_every_steps=2,
    )
    return _CheckpointFixture(path, model, optimizer, scheduler, ema, provenance)


def _resume_fixture(
    source: GraphRestore,
) -> tuple[
    GraphRestore,
    torch.optim.Optimizer,
    WarmupCosineScheduler,
    Stage3PlannerEMA,
    StatefulEpisodeSampler,
]:
    model = _resume_target(source)
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    return model, optimizer, scheduler, Stage3PlannerEMA(model, decay=0.9), _sampler()


def test_stage3_exact_tensor_comparison_preserves_cpu_values_and_metadata() -> None:
    reference = torch.tensor([0.0, -0.0, 1.25], dtype=torch.float32)
    candidate = reference.clone()
    reference_before = reference.clone()
    candidate_before = candidate.clone()

    assert stage3_engine._stage3_tensors_equal_exact(reference, candidate)
    assert not stage3_engine._stage3_tensors_equal_exact(
        reference, candidate.to(dtype=torch.float64)
    )
    changed = candidate.clone()
    changed[-1] = torch.nextafter(
        changed[-1], torch.tensor(float("inf"), dtype=changed.dtype)
    )
    assert not stage3_engine._stage3_tensors_equal_exact(reference, changed)
    assert torch.equal(reference, reference_before)
    assert torch.equal(candidate, candidate_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stage3_cuda_resume_prevalidates_cpu_checkpoint_exactly(
    tmp_path: Path,
) -> None:
    fixture = _saved_checkpoint(tmp_path)
    device = torch.device("cuda", torch.cuda.current_device())

    target = _resume_target(fixture.model).to(device)
    optimizer = build_stage3_optimizer(target, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    ema = Stage3PlannerEMA(target, decay=0.9)
    payload = resume_stage3_checkpoint(
        fixture.path,
        model=target,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=_sampler(),
        expected_provenance=fixture.provenance,
        validation_every_steps=2,
    )
    assert all(
        torch.equal(value.detach().cpu(), payload["model"][name])
        for name, value in target.state_dict().items()
    )
    assert all(
        torch.equal(value.detach().cpu(), payload["ema"]["shadow"][name])
        for name, value in ema.shadow.items()
    )

    corrupted = copy.deepcopy(payload)
    frozen_name = next(
        name
        for name, value in corrupted["model"].items()
        if not name.startswith("planner.") and value.is_floating_point()
    )
    frozen = corrupted["model"][frozen_name].clone()
    frozen.flatten()[0] = torch.nextafter(
        frozen.flatten()[0],
        torch.tensor(float("inf"), dtype=frozen.dtype),
    )
    corrupted["model"][frozen_name] = frozen
    corrupted["ema"]["shadow"][frozen_name] = frozen.clone()
    bad_path = tmp_path / "cuda_frozen_parent_drift.pth"
    torch.save(corrupted, bad_path)

    victim = _resume_target(fixture.model).to(device)
    victim_optimizer = build_stage3_optimizer(victim, fused_if_supported=False)
    victim_scheduler = WarmupCosineScheduler(
        victim_optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    victim_ema = Stage3PlannerEMA(victim, decay=0.9)
    victim_sampler = _sampler()
    before_model = {
        name: value.detach().cpu().clone()
        for name, value in victim.state_dict().items()
    }
    before_ema = {
        name: value.detach().cpu().clone() for name, value in victim_ema.shadow.items()
    }
    before_optimizer = copy.deepcopy(victim_optimizer.state_dict())
    before_scheduler = copy.deepcopy(victim_scheduler.state_dict())
    before_sampler = copy.deepcopy(victim_sampler.state_dict())
    before_cpu_rng = torch.get_rng_state().clone()
    before_cuda_rng = torch.cuda.get_rng_state(device).clone()
    with pytest.raises(Stage3ContractError, match="live Stage1 parent"):
        resume_stage3_checkpoint(
            bad_path,
            model=victim,
            ema=victim_ema,
            optimizer=victim_optimizer,
            scheduler=victim_scheduler,
            sampler=victim_sampler,
            expected_provenance=fixture.provenance,
            validation_every_steps=2,
        )
    assert all(
        torch.equal(value.detach().cpu(), before_model[name])
        for name, value in victim.state_dict().items()
    )
    assert all(
        torch.equal(value.detach().cpu(), before_ema[name])
        for name, value in victim_ema.shadow.items()
    )
    assert victim_optimizer.state_dict() == before_optimizer
    assert victim_scheduler.state_dict() == before_scheduler
    assert victim_sampler.state_dict() == before_sampler
    assert torch.equal(torch.get_rng_state(), before_cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), before_cuda_rng)


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
    provenance = _stage3_test_provenance(
        bindings={"primary_train_manifest": {"sha256": "b" * 64}}
    )
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

    restored = _resume_target(model)
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


def test_stage3_resume_installs_the_single_prevalidated_payload_under_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _saved_checkpoint(tmp_path)
    real_load = torch.load
    valid = real_load(fixture.path, map_location="cpu", weights_only=False)
    planner_name = next(
        name
        for name, value in valid["model"].items()
        if name.startswith("planner.") and value.is_floating_point()
    )
    replacement = copy.deepcopy(valid)
    poisoned = replacement["model"][planner_name].clone()
    poisoned.flatten()[0] = float("nan")
    replacement["model"][planner_name] = poisoned
    target, optimizer, scheduler, ema, sampler = _resume_fixture(fixture.model)
    target_loads = 0

    def swap_after_first_read(
        path: str | Path, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal target_loads
        payload = real_load(path, *args, **kwargs)
        if Path(path).resolve() == fixture.path.resolve():
            target_loads += 1
            if target_loads == 1:
                # Model an atomic same-provenance replacement after validation.
                torch.save(replacement, fixture.path)
        return payload

    monkeypatch.setattr(stage3_engine.torch, "load", swap_after_first_read)
    resumed = resume_stage3_checkpoint(
        fixture.path,
        model=target,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        expected_provenance=fixture.provenance,
        validation_every_steps=2,
    )
    assert target_loads == 1
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in valid["model"].items()
    )
    assert all(
        torch.equal(resumed["model"][name], value)
        for name, value in valid["model"].items()
    )


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
    train_stage3_optimizer_step(
        model,
        [_supervision_batch()],
        optimizer,
        scheduler,
        ema,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    sampler.mark_consumed_optimizer_step(1)
    provenance = _stage3_test_provenance()
    provenance["stage3_approval"] = {"sha256": "c" * 64}
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
    before = {
        name: value.detach().clone() for name, value in victim.state_dict().items()
    }
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


def test_stage3_allocator_conf_is_exact_and_environment_only() -> None:
    assert (
        validate_stage3_allocator_conf(
            {"PYTORCH_CUDA_ALLOC_CONF": STAGE3_ALLOCATOR_CONF}
        )
        == STAGE3_ALLOCATOR_CONF
    )
    for environment in (
        {},
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        {
            "PYTORCH_CUDA_ALLOC_CONF": "backend:native,expandable_segments:True,max_split_size_mb:128"
        },
    ):
        with pytest.raises(Stage3ContractError, match="requires exact"):
            validate_stage3_allocator_conf(environment)


def test_formal_entry_rejects_allocator_before_cuda_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_stage3_planner, "PROJECT_ROOT", tmp_path)
    approval = SimpleNamespace(
        bindings={
            "stage1_checkpoint": {"sha256": "a" * 64},
            "interaction_train_manifest": {"sha256": "b" * 64},
            "interaction_val_manifest": {"sha256": "c" * 64},
        }
    )
    paths = SimpleNamespace(
        approval=approval,
        relation_train=tmp_path / "train.jsonl",
        relation_val=tmp_path / "val.jsonl",
    )
    monkeypatch.setattr(
        train_stage3_planner, "validate_stage3_approval", lambda *args, **kwargs: paths
    )
    monkeypatch.setattr(
        train_stage3_planner, "load_relation_records", lambda *args, **kwargs: ()
    )
    monkeypatch.setattr(
        train_stage3_planner, "assert_relation_clean_disjoint", lambda *args: None
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "wrong")
    cuda_calls = 0

    def forbidden_cuda_probe() -> bool:
        nonlocal cuda_calls
        cuda_calls += 1
        raise AssertionError("CUDA was queried before allocator validation")

    monkeypatch.setattr(
        train_stage3_planner.torch.cuda, "is_available", forbidden_cuda_probe
    )
    arguments = argparse.Namespace(
        config=tmp_path / "stage3.yaml",
        resume=None,
        micro_batch=1,
        output_dir=tmp_path / "output",
    )
    with pytest.raises(Stage3ContractError, match="requires exact"):
        train_stage3_planner.run(arguments)
    assert cuda_calls == 0


def test_stage3_provenance_records_exact_allocator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", STAGE3_ALLOCATOR_CONF)
    monkeypatch.setattr(stage3_engine, "git_commit", lambda unused: "1" * 40)
    monkeypatch.setattr(stage3_engine, "sha256_file", lambda unused: "2" * 64)
    monkeypatch.setattr(stage3_engine, "sha256_json", lambda unused: "3" * 64)
    monkeypatch.setattr(
        stage3_engine, "semantic_source_hashes", lambda *args, **kwargs: {"x": "4" * 64}
    )
    monkeypatch.setattr(stage3_engine, "stage3_dependency_versions", lambda: {})
    approval = SimpleNamespace(
        bindings={},
        approval_path=tmp_path / "approved.json",
        approval_sha256="5" * 64,
        stage2_decision_sha256="6" * 64,
        approval_required_sha256="7" * 64,
    )
    paths = SimpleNamespace(
        resolved={
            "expected_identity": {
                "agenticir_commit": "1" * 40,
                "mioir_commit": "1" * 40,
            },
            "agenticir_repo": tmp_path / "agenticir",
            "mioir_repo": tmp_path / "mioir",
        },
        approval=approval,
        config_path=tmp_path / "stage3.yaml",
        config={"stage": "stage3", "ema": {"decay": 0.9999}},
        resolved_path=tmp_path / "resolved.yaml",
        project_root=tmp_path,
        executor_checkpoint=tmp_path / "stage1.pth",
        relation_train=tmp_path / "relation_train.jsonl",
        relation_val=tmp_path / "relation_val.jsonl",
        effect_profiles=tmp_path / "effects.json",
        pair_prior=tmp_path / "prior.json",
        global_priority=tmp_path / "priority.json",
    )
    parent = Stage3ParentLoadReport(
        checkpoint_sha256="8" * 64,
        checkpoint_step=30_000,
        loaded_count=1,
        initialized_planner_keys=("planner.x",),
    )
    provenance = build_stage3_provenance(
        paths,
        parent,
        micro_batch=4,
        accumulation_steps=2,
        validation_vram_gate=_validation_vram_evidence(),
    )
    assert provenance["runtime"]["allocator_conf"] == STAGE3_ALLOCATOR_CONF
    assert provenance["runtime"]["validation_vram_gate"]["image_size"] == 2040


def test_stage3_extension_authorization_is_separate_exact_and_immutable(
    tmp_path: Path,
) -> None:
    authorization, paths = _extension_authorization_fixture(tmp_path)
    evidence = stage3_engine.validate_stage3_extension_authorization(
        authorization, paths
    )
    assert evidence.base_step == 12_000
    assert evidence.target_step == 18_000
    assert evidence.validation_steps == (14_000, 16_000, 18_000)
    binding = evidence.provenance_binding()
    assert set(binding) == {
        "path",
        "sha256",
        "cycles",
        "base_step",
        "target_step",
        "validation_every_steps",
        "validation_steps",
        "schedule_horizon_steps",
        "min_lr",
        "lr_policy",
    }
    provenance = {
        "runtime": {"max_steps": 12_000, "training_target_step": 18_000},
        "stage3_extension": binding,
    }
    assert (
        stage3_engine.stage3_training_target_step(
            provenance,
            schedule_horizon_steps=12_000,
            validation_every_steps=2_000,
        )
        == 18_000
    )

    payload = json.loads(authorization.read_text(encoding="utf-8"))
    backup = Path(payload["pre_extension_last_checkpoint"]["path"])
    backup.chmod(0o644)
    with pytest.raises(Stage3ContractError, match="authorization mismatch"):
        stage3_engine.validate_stage3_extension_authorization(authorization, paths)
    backup.chmod(0o444)

    payload["cycles"] = True
    authorization.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(Stage3ContractError, match="authorization mismatch"):
        stage3_engine.validate_stage3_extension_authorization(authorization, paths)

    payload["cycles"] = 3
    payload["min_lr"] = 3.0e-6
    authorization.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(Stage3ContractError, match="authorization mismatch"):
        stage3_engine.validate_stage3_extension_authorization(authorization, paths)


def test_stage3_extension_preserves_completed_cosine_floor_for_6000_steps() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2.0e-4)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=500,
        max_steps=12_000,
        min_lr=2.0e-6,
    )
    for _ in range(18_000):
        optimizer.step()
        scheduler.step()
        if scheduler.last_epoch >= 12_000:
            assert optimizer.param_groups[0]["lr"] == 2.0e-6
    assert scheduler.max_steps == 12_000
    assert scheduler.last_epoch == 18_000


def test_stage3_fresh_step0_anchor_is_raw_and_exactly_resumable(
    tmp_path: Path,
) -> None:
    model = _tiny_graphrestore()
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    ema = Stage3PlannerEMA(model, decay=0.9)
    sampler = _sampler()
    provenance = _stage3_test_provenance()
    provenance["stage3_approval"] = {"sha256": "d" * 64}
    anchor = tmp_path / "last.pth"
    save_stage3_checkpoint(
        anchor,
        step=0,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        validation_every_steps=2,
    )
    raw = torch.load(anchor, map_location="cpu", weights_only=False)
    assert raw["step"] == 0
    assert raw["model_role"] == "raw_training_state"
    assert raw["resumable"] is True
    assert raw["pending_validation_step"] is None
    assert raw["optimizer_transaction_active"] is False

    restored = _resume_target(model)
    restored_optimizer = build_stage3_optimizer(restored, fused_if_supported=False)
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    restored_ema = Stage3PlannerEMA(restored, decay=0.9)
    payload = resume_stage3_checkpoint(
        anchor,
        model=restored,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=_sampler(),
        expected_provenance=provenance,
        validation_every_steps=2,
    )
    assert payload["pending_validation_step"] is None


def test_stage3_pending_validation_checkpoint_replays_and_rejects_invalid_marker(
    tmp_path: Path,
) -> None:
    torch.manual_seed(89)
    model = _tiny_graphrestore()
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    ema = Stage3PlannerEMA(model, decay=0.9)
    sampler = _sampler()
    for step in (1, 2):
        train_stage3_optimizer_step(
            model,
            [_supervision_batch()],
            optimizer,
            scheduler,
            ema,
            device=torch.device("cpu"),
            use_bf16=False,
        )
        sampler.mark_consumed_optimizer_step(step)
    provenance = _stage3_test_provenance()
    provenance["stage3_approval"] = {"sha256": "e" * 64}
    checkpoint = tmp_path / "last.pth"
    save_stage3_checkpoint(
        checkpoint,
        step=2,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        pending_validation_step=2,
        validation_every_steps=2,
    )

    restored = _resume_target(model)
    restored_optimizer = build_stage3_optimizer(restored, fused_if_supported=False)
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    restored_ema = Stage3PlannerEMA(restored, decay=0.9)
    payload = resume_stage3_checkpoint(
        checkpoint,
        model=restored,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=_sampler(),
        expected_provenance=provenance,
        validation_every_steps=2,
    )
    assert payload["pending_validation_step"] == payload["step"] == 2

    corrupted = dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    corrupted["pending_validation_step"] = 1
    invalid = tmp_path / "invalid_pending.pth"
    torch.save(corrupted, invalid)
    victim = _tiny_graphrestore()
    before = {
        name: value.detach().clone() for name, value in victim.state_dict().items()
    }
    victim_optimizer = build_stage3_optimizer(victim, fused_if_supported=False)
    victim_scheduler = WarmupCosineScheduler(
        victim_optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    with pytest.raises(Stage3ContractError, match="must equal checkpoint step"):
        resume_stage3_checkpoint(
            invalid,
            model=victim,
            ema=Stage3PlannerEMA(victim, decay=0.9),
            optimizer=victim_optimizer,
            scheduler=victim_scheduler,
            sampler=_sampler(),
            expected_provenance=provenance,
            validation_every_steps=2,
        )
    for name, expected in before.items():
        assert torch.equal(victim.state_dict()[name], expected)

    with pytest.raises(Stage3ContractError, match="validation boundary"):
        validate_stage3_pending_validation_step(
            step=3,
            pending_validation_step=3,
            max_steps=4,
            validation_every_steps=2,
        )


def test_stage3_peak_gate_resets_and_rejects_over_90_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    resets: list[torch.device] = []

    class _Properties:
        total_memory = 1_000

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", resets.append)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda unused: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda unused: _Properties()
    )
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda unused: 900)
    reset_stage3_peak_memory(device)
    assert resets == [device]
    assert enforce_stage3_peak_memory(device, phase="validation") == (900, 0.9)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda unused: 901)
    with pytest.raises(Stage3ContractError, match="exceeds 0.90"):
        enforce_stage3_peak_memory(device, phase="validation")


def test_mid_optimizer_interrupt_cannot_be_checkpointed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _tiny_graphrestore()
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    ema = Stage3PlannerEMA(model, decay=0.9)
    transaction = Stage3OptimizerTransaction()

    def interrupt_step(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(optimizer, "step", interrupt_step)
    with pytest.raises(KeyboardInterrupt):
        train_stage3_optimizer_step(
            model,
            [_supervision_batch()],
            optimizer,
            scheduler,
            ema,
            device=torch.device("cpu"),
            use_bf16=False,
            optimizer_transaction=transaction,
        )
    assert transaction.active is True
    destination = tmp_path / "must_not_exist.pth"
    with pytest.raises(Stage3ContractError, match="mid-optimizer-update"):
        save_stage3_checkpoint(
            destination,
            step=0,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=_sampler(),
            provenance={"stage3_approval": {"sha256": "f" * 64}},
            optimizer_transaction=transaction,
        )
    assert not destination.exists()


def test_sigterm_handler_raises_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt):
        train_stage3_planner._sigterm_as_keyboard_interrupt(15, None)


def test_pending_validation_history_replay_is_idempotent(tmp_path: Path) -> None:
    history = tmp_path / "calibration.csv"
    row = {name: None for name in CALIBRATION_COLUMNS}
    row.update({"step": 2_000, "single_psnr": 30.0, "group_a_psnr": 25.0})
    append_calibration_history(history, row)
    first = history.read_bytes()
    append_calibration_history(history, row)
    assert history.read_bytes() == first
    assert history.read_text(encoding="utf-8").count("\n") == 2


def test_stage3_checkpoint_rejects_generic_ema_before_write(tmp_path: Path) -> None:
    model = _tiny_graphrestore()
    optimizer = build_stage3_optimizer(model, fused_if_supported=False)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    destination = tmp_path / "generic_ema.pth"
    with pytest.raises(Stage3ContractError, match="require Stage3PlannerEMA"):
        save_stage3_checkpoint(
            destination,
            step=0,
            model=model,
            ema=ExponentialMovingAverage(model, decay=0.9),
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=_sampler(),
            provenance={"stage3_approval": {"sha256": "9" * 64}},
        )
    assert not destination.exists()


def test_stage3_optimizer_state_ledger_is_fail_closed_before_mutation(
    tmp_path: Path,
) -> None:
    torch.manual_seed(97)
    source = _tiny_graphrestore()
    source_optimizer = build_stage3_optimizer(source, fused_if_supported=False)
    source_scheduler = WarmupCosineScheduler(
        source_optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
    )
    source_ema = Stage3PlannerEMA(source, decay=0.9)
    train_stage3_optimizer_step(
        source,
        [_supervision_batch()],
        source_optimizer,
        source_scheduler,
        source_ema,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    provenance = _stage3_test_provenance()
    valid_path = tmp_path / "valid.pth"
    save_stage3_checkpoint(
        valid_path,
        step=1,
        model=source,
        ema=source_ema,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        sampler=_sampler(),
        provenance=provenance,
        validation_every_steps=2,
    )
    valid = torch.load(valid_path, map_location="cpu", weights_only=False)
    ledger = valid["optimizer_state_name_ledger"]
    optimizer_state = valid["optimizer"]["state"]
    assert ledger
    assert set(ledger) == set(optimizer_state)
    assert len(ledger) == sum(
        len(group["params"]) for group in valid["optimizer"]["param_groups"]
    )

    first_id = next(iter(ledger))
    wrong_name = next(value for key, value in ledger.items() if key != first_id)
    cases = (
        ("missing_ledger", "lacks optimizer state-name ledger"),
        ("deleted_state", "does not cover every planner parameter"),
        ("cleared_state", "does not cover every planner parameter"),
        ("wrong_name", "ledger name drifted"),
        ("wrong_step", "Adam step differs"),
        ("wrong_shape", "Adam tensor state is invalid"),
        ("wrong_dtype", "Adam tensor state is invalid"),
        ("nonfinite", "Adam tensor state is invalid"),
    )
    for label, error_pattern in cases:
        bad = torch.load(valid_path, map_location="cpu", weights_only=False)
        bad_ledger = bad["optimizer_state_name_ledger"]
        bad_state = bad["optimizer"]["state"]
        if label == "missing_ledger":
            bad.pop("optimizer_state_name_ledger")
        elif label == "deleted_state":
            bad_state.pop(first_id)
        elif label == "cleared_state":
            bad["optimizer"]["state"] = {}
            bad["optimizer_state_name_ledger"] = {}
        elif label == "wrong_name":
            bad_ledger[first_id] = wrong_name
        elif label == "wrong_step":
            bad_state[first_id]["step"] = torch.tensor(2.0)
        elif label == "wrong_shape":
            bad_state[first_id]["exp_avg"] = bad_state[first_id]["exp_avg"].flatten()[
                :-1
            ]
        elif label == "wrong_dtype":
            bad_state[first_id]["exp_avg"] = bad_state[first_id]["exp_avg"].double()
        else:
            corrupted = bad_state[first_id]["exp_avg"].clone()
            corrupted.flatten()[0] = float("nan")
            bad_state[first_id]["exp_avg"] = corrupted
        bad_path = tmp_path / f"{label}.pth"
        torch.save(bad, bad_path)

        victim = _resume_target(source)
        victim_optimizer = build_stage3_optimizer(victim, fused_if_supported=False)
        victim_scheduler = WarmupCosineScheduler(
            victim_optimizer, warmup_steps=1, max_steps=4, min_lr=2e-6
        )
        victim_ema = Stage3PlannerEMA(victim, decay=0.9)
        victim_sampler = _sampler()
        before_model = {
            name: value.detach().clone() for name, value in victim.state_dict().items()
        }
        before_ema = {
            name: value.detach().clone() for name, value in victim_ema.shadow.items()
        }
        before_optimizer = victim_optimizer.state_dict()
        before_scheduler = victim_scheduler.state_dict()
        before_sampler = victim_sampler.state_dict()
        before_rng = torch.get_rng_state().clone()
        with pytest.raises(Stage3ContractError, match=error_pattern):
            resume_stage3_checkpoint(
                bad_path,
                model=victim,
                ema=victim_ema,
                optimizer=victim_optimizer,
                scheduler=victim_scheduler,
                sampler=victim_sampler,
                expected_provenance=provenance,
                validation_every_steps=2,
            )
        for name, expected in before_model.items():
            assert torch.equal(victim.state_dict()[name], expected)
        for name, expected in before_ema.items():
            assert torch.equal(victim_ema.shadow[name], expected)
        assert victim_optimizer.state_dict() == before_optimizer
        assert victim_scheduler.state_dict() == before_scheduler
        assert victim_sampler.state_dict() == before_sampler
        assert torch.equal(torch.get_rng_state(), before_rng)


def test_stage3_scheduler_state_is_exact_and_validated_before_mutation(
    tmp_path: Path,
) -> None:
    fixture = _saved_checkpoint(tmp_path)
    valid = torch.load(fixture.path, map_location="cpu", weights_only=False)
    cases = {
        "missing_field": lambda payload: payload["scheduler"].pop("verbose"),
        "last_epoch": lambda payload: payload["scheduler"].__setitem__("last_epoch", 0),
        "step_count": lambda payload: payload["scheduler"].__setitem__(
            "_step_count", 1
        ),
        "base_lrs": lambda payload: payload["scheduler"].__setitem__(
            "base_lrs", [1.0e-4]
        ),
        "warmup": lambda payload: payload["scheduler"].__setitem__("warmup_steps", 2),
        "maximum": lambda payload: payload["scheduler"].__setitem__("max_steps", 5),
        "minimum": lambda payload: payload["scheduler"].__setitem__("min_lr", 1.0e-6),
        "trajectory": lambda payload: (
            payload["scheduler"].__setitem__("_last_lr", [1.0e-4]),
            payload["optimizer"]["param_groups"][0].__setitem__("lr", 1.0e-4),
        ),
    }
    for label, corrupt in cases.items():
        payload = copy.deepcopy(valid)
        corrupt(payload)
        path = tmp_path / f"scheduler_{label}.pth"
        torch.save(payload, path)
        model, optimizer, scheduler, ema, sampler = _resume_fixture(fixture.model)
        before = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        with pytest.raises(Stage3ContractError, match="scheduler|trajectory"):
            resume_stage3_checkpoint(
                path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                expected_provenance=fixture.provenance,
                validation_every_steps=2,
            )
        assert all(
            torch.equal(model.state_dict()[name], expected)
            for name, expected in before.items()
        )

    fixture.scheduler._step_count = 99
    destination = tmp_path / "bad_scheduler_save.pth"
    with pytest.raises(Stage3ContractError, match=r"step \+ 1"):
        save_stage3_checkpoint(
            destination,
            step=1,
            model=fixture.model,
            ema=fixture.ema,
            optimizer=fixture.optimizer,
            scheduler=fixture.scheduler,
            sampler=_sampler(),
            provenance=fixture.provenance,
            validation_every_steps=2,
        )
    assert not destination.exists()


def test_stage3_rng_model_ema_and_live_frozen_parent_are_prevalidated(
    tmp_path: Path,
) -> None:
    fixture = _saved_checkpoint(tmp_path)
    valid = torch.load(fixture.path, map_location="cpu", weights_only=False)
    planner_name = next(
        name
        for name, value in valid["model"].items()
        if name.startswith("planner.") and value.is_floating_point()
    )
    frozen_name = next(
        name
        for name, value in valid["model"].items()
        if not name.startswith("planner.") and value.is_floating_point()
    )

    def corrupt_rng(payload: dict[str, Any]) -> None:
        payload["rng_states"]["torch_cpu"] = torch.zeros(2, dtype=torch.int64)

    def corrupt_model(payload: dict[str, Any]) -> None:
        tensor = payload["model"][planner_name].clone()
        tensor.flatten()[0] = float("nan")
        payload["model"][planner_name] = tensor

    def corrupt_ema(payload: dict[str, Any]) -> None:
        tensor = payload["ema"]["shadow"][planner_name].clone()
        tensor.flatten()[0] = float("inf")
        payload["ema"]["shadow"][planner_name] = tensor

    def corrupt_ema_updates_bool(payload: dict[str, Any]) -> None:
        # bool is an int subclass, so equality alone would accept True at step1.
        payload["ema"]["num_updates"] = True

    def corrupt_frozen_parent(payload: dict[str, Any]) -> None:
        tensor = payload["model"][frozen_name].clone()
        tensor.flatten()[0] += 1.0
        payload["model"][frozen_name] = tensor
        payload["ema"]["shadow"][frozen_name] = tensor.clone()

    for label, corrupt, pattern in (
        ("rng", corrupt_rng, "RNG state"),
        ("model", corrupt_model, "non-finite"),
        ("ema", corrupt_ema, "EMA tensor contract"),
        ("ema_updates_bool", corrupt_ema_updates_bool, "EMA metadata"),
        ("frozen_parent", corrupt_frozen_parent, "live Stage1 parent"),
    ):
        payload = copy.deepcopy(valid)
        corrupt(payload)
        path = tmp_path / f"corrupt_{label}.pth"
        torch.save(payload, path)
        model, optimizer, scheduler, ema, sampler = _resume_fixture(fixture.model)
        before_model = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        before_ema = {
            name: value.detach().clone() for name, value in ema.shadow.items()
        }
        with pytest.raises(Stage3ContractError, match=pattern):
            resume_stage3_checkpoint(
                path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                expected_provenance=fixture.provenance,
                validation_every_steps=2,
            )
        assert all(
            torch.equal(model.state_dict()[name], expected)
            for name, expected in before_model.items()
        )
        assert all(
            torch.equal(ema.shadow[name], expected)
            for name, expected in before_ema.items()
        )


def test_pending_marker_precedes_atomic_save_and_survives_clear_races() -> None:
    pending = train_stage3_planner._PendingValidationState()

    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pending.begin(2_000, interrupted)
    assert pending.step == 2_000
    with pytest.raises(KeyboardInterrupt):
        pending.clear(interrupted)
    assert pending.step == 2_000
    pending.clear(lambda: None)
    assert pending.step is None


def test_post_update_signal_window_is_not_checkpointable_and_log_replay_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "train.jsonl"
    log_path.touch()
    transaction = Stage3OptimizerTransaction()
    transaction.begin()
    pending = train_stage3_planner._PendingValidationState()
    row = {
        "event": "train_step",
        "utc": "first",
        "step": 2,
        "total": 1.0,
        "seconds": 0.5,
        "images_per_second": 16.0,
        "peak_reserved_bytes": 100,
        "peak_reserved_fraction": 0.5,
    }
    real_commit = transaction.commit

    def interrupt_before_checkpointable() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(transaction, "commit", interrupt_before_checkpointable)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        with pytest.raises(KeyboardInterrupt):
            train_stage3_planner._publish_stage3_train_boundary(
                log_path=log_path,
                log=log,
                row=row,
                optimizer_transaction=transaction,
                pending_validation=pending,
                validation_due=True,
            )
    assert transaction.active is True
    assert pending.step == 2
    assert (
        sum(
            row_value.get("event") == "train_step"
            for _, row_value in stage3_engine.iter_jsonl(log_path)
        )
        == 1
    )

    monkeypatch.setattr(transaction, "commit", real_commit)
    replay = dict(row, utc="replay", seconds=0.75, images_per_second=10.0)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        appended = train_stage3_planner._publish_stage3_train_boundary(
            log_path=log_path,
            log=log,
            row=replay,
            optimizer_transaction=transaction,
            pending_validation=pending,
            validation_due=True,
        )
    assert appended is False
    assert transaction.active is False
    assert pending.step == 2
    assert (
        sum(
            row_value.get("event") == "train_step"
            for _, row_value in stage3_engine.iter_jsonl(log_path)
        )
        == 1
    )

    source = (PROJECT_ROOT / "scripts/train_stage3_planner.py").read_text(
        encoding="utf-8"
    )
    loop = source[source.index("while step < training_target_step:") :]
    assert loop.index("enforce_stage3_peak_memory(") < loop.index(
        "_publish_stage3_train_boundary("
    )
    helper = source[
        source.index("def _publish_stage3_train_boundary(") : source.index(
            "def _validated_restoration("
        )
    ]
    assert helper.index("pending_validation.step = step") < helper.index(
        "optimizer_transaction.commit()"
    )


def _report_summary() -> dict[str, Any]:
    return {
        "protocol_id": stage3_engine.PROTOCOL_ID,
        "restoration": {
            "single": {"psnr": 28.0, "ssim": 0.88},
            "group_a": {"psnr": 25.0, "ssim": 0.79},
        },
        "planner": {"macro_f1": 0.7},
        "relation": {
            "relation_accuracy_non_ambiguous": 0.6,
            "parallel_precision_non_ambiguous": 0.5,
            "parallel_recall_non_ambiguous": 0.4,
            "n_ambiguous": 2,
            "ambiguous_fraction": 0.1,
        },
        "guard": {
            "guard_spearman_rain": 0.1,
            "guard_mae_rain": 0.2,
            "guard_std_rain": 0.3,
            "guard_high_frac_rain": 0.0,
            "guard_spearman_haze": 0.2,
            "guard_mae_haze": 0.3,
            "guard_std_haze": 0.4,
            "guard_high_frac_haze": 0.0,
            "valid_guard_images_rain": 1,
            "skipped_guard_images_rain": 0,
            "valid_guard_images_haze": 1,
            "skipped_guard_images_haze": 0,
        },
        "graph": {
            "pre_compiler_cycle_rate": 0.0,
            "post_compiler_cycle_rate": 0.0,
        },
    }


def test_stage3_reports_bind_protocol_best_sha_and_finite_restoration(
    tmp_path: Path,
) -> None:
    best_path = tmp_path / "best_ema.pth"
    best_path.write_bytes(b"selected")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text("{}", encoding="utf-8")
    score = stage3_engine.ValidationScore(25.0, 0.79, 28.0, 0.88, 2_000)
    summary = _report_summary()
    final = train_stage3_planner._render_report(
        summary,
        best=score,
        checkpoint=best_path,
        thresholds=thresholds,
    )
    interim = train_stage3_planner._render_validation_report(
        summary,
        current=score,
        best=score,
        checkpoint=best_path,
        peak_reserved_bytes=100,
        peak_reserved_fraction=0.5,
    )
    expected_sha = stage3_engine.sha256_file(best_path)
    for report in (final, interim):
        assert f"protocol: `{stage3_engine.PROTOCOL_ID}`" in report
        assert str(best_path.resolve()) in report
        assert expected_sha in report
    assert "Selected Single PSNR/SSIM" in final
    assert "Selected Group-A PSNR/SSIM" in final
    assert "single PSNR/SSIM" in interim
    assert "Group A PSNR/SSIM" in interim

    report_path = tmp_path / "STAGE3_PLANNER_GUARD.md"
    report_path.write_text(final, encoding="utf-8")
    binding = train_stage3_planner._report_binding(report_path)
    assert binding == {
        "report": str(report_path.resolve()),
        "report_sha256": stage3_engine.sha256_file(report_path),
    }
    bad = copy.deepcopy(summary)
    bad["restoration"]["group_a"]["psnr"] = float("nan")
    with pytest.raises(Stage3ContractError, match="non-finite"):
        train_stage3_planner._render_report(
            bad,
            best=score,
            checkpoint=best_path,
            thresholds=thresholds,
        )


def _selection_metrics(step: int = 2) -> dict[str, float | int]:
    return {
        "group_a_psnr": 25.0,
        "group_a_ssim": 0.79,
        "single_psnr": 28.0,
        "single_ssim": 0.88,
        "validation_step": step,
        "best_group_a_psnr": 25.0,
        "best_group_a_ssim": 0.79,
        "best_single_psnr": 28.0,
        "best_single_ssim": 0.88,
        "best_step": step,
    }


def test_stage3_metrics_and_best_incumbent_are_transactionally_consistent(
    tmp_path: Path,
) -> None:
    metrics = _selection_metrics()
    best = _saved_checkpoint(
        tmp_path,
        steps=2,
        name="best_ema.pth",
        metrics=metrics,
        model_as_ema=True,
    )
    sampler = _sampler()
    sampler.mark_consumed_optimizer_step(2)
    last = tmp_path / "last.pth"
    save_stage3_checkpoint(
        last,
        step=2,
        model=best.model,
        ema=best.ema,
        optimizer=best.optimizer,
        scheduler=best.scheduler,
        sampler=sampler,
        provenance=best.provenance,
        metrics=metrics,
        validation_every_steps=2,
    )
    model, optimizer, scheduler, ema, resume_sampler = _resume_fixture(best.model)
    resume_stage3_checkpoint(
        last,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=resume_sampler,
        expected_provenance=best.provenance,
        validation_every_steps=2,
    )

    valid = torch.load(last, map_location="cpu", weights_only=False)
    cases: dict[str, Callable[[dict[str, Any]], None]] = {
        "partial": lambda payload: payload["metrics"].pop("single_ssim"),
        "nonfinite": lambda payload: payload["metrics"].__setitem__(
            "group_a_psnr", float("nan")
        ),
        "best_after_current": lambda payload: payload["metrics"].__setitem__(
            "validation_step", 0
        ),
        "nonboundary": lambda payload: (
            payload["metrics"].__setitem__("validation_step", 1),
            payload["metrics"].__setitem__("best_step", 0),
        ),
        "unknown": lambda payload: payload["metrics"].__setitem__("extra", 1.0),
        "incumbent_mismatch": lambda payload: payload["metrics"].__setitem__(
            "best_group_a_psnr", 24.0
        ),
        "current_better_than_best": lambda payload: payload["metrics"].__setitem__(
            "group_a_psnr", 25.1
        ),
    }
    for label, corrupt in cases.items():
        payload = copy.deepcopy(valid)
        corrupt(payload)
        path = tmp_path / f"metrics_{label}.pth"
        torch.save(payload, path)
        # Incumbent lookup is sibling-relative.
        if path.parent / "best_ema.pth" != best.path:
            raise AssertionError("test fixture path drifted")
        victim, victim_optimizer, victim_scheduler, victim_ema, victim_sampler = (
            _resume_fixture(best.model)
        )
        with pytest.raises(Stage3ContractError, match="metrics|incumbent"):
            resume_stage3_checkpoint(
                path,
                model=victim,
                ema=victim_ema,
                optimizer=victim_optimizer,
                scheduler=victim_scheduler,
                sampler=victim_sampler,
                expected_provenance=best.provenance,
                validation_every_steps=2,
            )


def test_stage3_calibration_rejects_conflicting_same_step_row(tmp_path: Path) -> None:
    history = tmp_path / "calibration.csv"
    row = {name: None for name in CALIBRATION_COLUMNS}
    row.update(
        {
            "step": 2_000,
            "single_psnr": 28.0,
            "group_a_psnr": 25.0,
            "planner_macro_f1": 0.7,
        }
    )
    append_calibration_history(history, row)
    append_calibration_history(history, row)
    changed = dict(row)
    changed["planner_macro_f1"] = 0.71
    with pytest.raises(Stage3ContractError, match="conflicting Stage3 calibration"):
        append_calibration_history(history, changed)


def test_stage3_best_loader_strictly_validates_selection_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = _selection_metrics()
    fixture = _saved_checkpoint(
        tmp_path,
        steps=2,
        name="best_ema.pth",
        metrics=metrics,
        model_as_ema=True,
    )
    bindings = {
        "stage1_checkpoint": {"sha256": "1" * 64},
        "skill_effect_profiles": {"sha256": "2" * 64},
        "pair_prior": {"sha256": "3" * 64},
        "global_priority": {"sha256": "4" * 64},
        "relation_train": {"sha256": "5" * 64},
        "relation_val": {"sha256": "6" * 64},
    }
    policy = stage3_engine.stage3_ema_policy_metadata(0.9)
    valid = torch.load(fixture.path, map_location="cpu", weights_only=False)
    valid["provenance"] = {
        "stage3_approval": {"sha256": "7" * 64},
        "bindings": bindings,
        "ema_policy": policy,
        "parent_checkpoint": {"sha256": "1" * 64},
        "effect_profiles_sha256": "2" * 64,
        "pair_prior_sha256": "3" * 64,
        "global_priority_sha256": "4" * 64,
        "relation_supervision": {
            "train_sha256": "5" * 64,
            "validation_sha256": "6" * 64,
        },
        "runtime": {"max_steps": 4, "training_target_step": 4},
    }
    torch.save(valid, fixture.path)
    paths = SimpleNamespace(
        config={
            "training": {"max_steps": 4},
            "runtime": {"validation_every_steps": 2},
            "ema": {"decay": 0.9},
        },
        approval=SimpleNamespace(approval_sha256="7" * 64, bindings=bindings),
    )

    def fake_build(*args: Any, **kwargs: Any) -> tuple[GraphRestore, object]:
        del args, kwargs
        return _resume_target(fixture.model), object()

    monkeypatch.setattr(stage3_engine, "build_stage3_model", fake_build)
    loaded = load_stage3_best_ema(
        paths,
        fixture.path,
        device=torch.device("cpu"),
        load_frozen_thresholds=False,
    )
    assert loaded.training is False
    expected = _tiny_graphrestore()
    incompatible = expected.load_state_dict(valid["model"], strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    expected.eval()
    fixed_input = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(991))
    fixed_mask = torch.zeros(1, 8, dtype=torch.bool)
    fixed_mask[:, 0] = True
    with torch.inference_mode():
        expected_output = expected(
            fixed_input,
            forced_counterfactual_mask=fixed_mask,
            max_rounds=1,
        )
        fresh_output = loaded(
            fixed_input,
            forced_counterfactual_mask=fixed_mask,
            max_rounds=1,
        )
    assert torch.is_tensor(expected_output) and torch.is_tensor(fresh_output)
    assert torch.equal(expected_output, fresh_output)

    floating_name = next(
        name for name, value in valid["model"].items() if value.is_floating_point()
    )
    cases: dict[str, Callable[[dict[str, Any]], None]] = {
        "amp": lambda payload: payload.__setitem__("amp", {"dtype": "float32"}),
        "trainable": lambda payload: payload.__setitem__(
            "trainable_prefixes", ["encoder."]
        ),
        "role": lambda payload: payload.__setitem__("model_role", "raw_training_state"),
        "frozen": lambda payload: payload.__setitem__("executor_frozen", False),
        "decay": lambda payload: payload["ema"].__setitem__("decay", 0.8),
        "updates": lambda payload: payload["ema"].__setitem__("num_updates", 1),
        "bool_updates": lambda payload: payload["ema"].__setitem__("num_updates", True),
        "policy": lambda payload: payload["ema"]["policy"].__setitem__(
            "buffer_update", "standard_ema"
        ),
        "metric_step": lambda payload: payload["metrics"].__setitem__("best_step", 0),
    }

    def nonfinite(payload: dict[str, Any]) -> None:
        tensor = payload["model"][floating_name].clone()
        tensor.flatten()[0] = float("nan")
        payload["model"][floating_name] = tensor
        payload["ema"]["shadow"][floating_name] = tensor.clone()

    cases["nonfinite"] = nonfinite
    for label, corrupt in cases.items():
        payload = copy.deepcopy(valid)
        corrupt(payload)
        directory = tmp_path / label
        directory.mkdir()
        path = directory / "best_ema.pth"
        torch.save(payload, path)
        with pytest.raises(Stage3ContractError):
            load_stage3_best_ema(
                paths,
                path,
                device=torch.device("cpu"),
                load_frozen_thresholds=False,
            )


def test_stage3_validation_vram_evidence_is_complete_and_fail_closed() -> None:
    evidence = _validation_vram_evidence()
    normalized = validate_stage3_validation_vram_evidence(evidence)
    assert normalized["image_size"] == 2040
    assert normalized["topologies"][0]["active_skill_counts_by_round"] == [1, 1, 1]
    for label, corrupt in (
        ("missing", lambda value: value.pop("resident_ema_bytes")),
        (
            "topology",
            lambda value: value["topologies"][1].__setitem__("completed_rounds", 3),
        ),
        (
            "metric",
            lambda value: value["topologies"][0].__setitem__(
                "metric_psnr", float("nan")
            ),
        ),
        ("peak", lambda value: value.__setitem__("peak_reserved_fraction", 0.91)),
    ):
        bad = copy.deepcopy(evidence)
        corrupt(bad)
        with pytest.raises(Stage3ContractError, match="VRAM"):
            validate_stage3_validation_vram_evidence(bad)


def test_stage3_validation_vram_gate_covers_serial_parallel_and_full_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGateModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.planner = torch.nn.Linear(1, 1)
            self.compiler = SimpleNamespace(mode="full_partial_order")
            self.called_modes: list[str] = []

        def __call__(self, image: torch.Tensor, **_: Any) -> GraphRestoreOutput:
            mode = self.compiler.mode
            self.called_modes.append(mode)
            active_skills = tuple(stage3_engine.SKILLS[:3])
            if mode == "forced_total_order":
                levels = tuple((skill,) for skill in active_skills)
                masks = []
                for index in range(3):
                    mask = torch.zeros(1, 8, dtype=torch.bool)
                    mask[0, index] = True
                    masks.append(mask)
            elif mode == "parallel_only":
                levels = (active_skills,)
                mask = torch.zeros(1, 8, dtype=torch.bool)
                mask[0, :3] = True
                masks = [mask]
            else:
                raise AssertionError(mode)
            graph = CompiledGraph(active_skills, levels, (), (), ())
            trace = tuple(
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
                steps=tuple(image for _ in trace),
                planner_outputs=(),
                compiled_graphs=(graph,),
                graph_states=(
                    ProgramGraphState(nodes=active_skills, edges=(), levels=levels),
                ),
                trace=trace,
            )

    real_rand = torch.rand
    peaks = [500, 700]
    model = FakeGateModel()
    optimizer = torch.optim.AdamW(model.planner.parameters())
    ema = Stage3PlannerEMA(model, decay=0.9999)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1_000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: peaks.pop(0))
    monkeypatch.setattr(
        stage3_engine.torch,
        "rand",
        lambda *args, **kwargs: real_rand(1, 3, 16, 16),
    )
    monkeypatch.setattr(
        stage3_engine.torch,
        "autocast",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(stage3_engine, "capture_rng_state", lambda: {"sentinel": True})
    monkeypatch.setattr(stage3_engine, "restore_rng_state", lambda _: None)
    monkeypatch.setattr(
        stage3_engine,
        "official_psnr_ssim",
        lambda *args, **kwargs: SimpleNamespace(
            psnr=torch.tensor([10.0]), ssim=torch.tensor([0.5])
        ),
    )
    gate = probe_stage3_validation_vram(
        model,  # type: ignore[arg-type]
        optimizer=optimizer,
        ema=ema,
        device=torch.device("cuda"),
    )
    assert gate.peak_reserved_bytes == 700
    assert gate.peak_reserved_fraction == 0.7
    assert gate.resident_optimizer_state_entries == 2
    assert gate.resident_optimizer_state_bytes > 0
    assert gate.resident_ema_bytes > 0
    assert gate.optimizer_state_empty_after is True
    assert not optimizer.state
    assert [row.compiler_mode for row in gate.topologies] == [
        "forced_total_order",
        "parallel_only",
    ]
    assert gate.topologies[0].active_skill_counts_by_round == (1, 1, 1)
    assert gate.topologies[1].active_skill_counts_by_round == (3,)
    assert model.called_modes == ["forced_total_order", "parallel_only"]
    assert model.compiler.mode == "full_partial_order"


def test_stage3_micro_probe_runs_ten_full_effective_batch_optimizer_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_graphrestore()
    pristine = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    teacher_flags: list[bool] = []
    original_build_optimizer = stage3_engine.build_stage3_optimizer

    def fake_batch(
        batch: int,
        device: torch.device,
        *,
        model: GraphRestore | None = None,
        include_teacher_intermediate: bool = False,
    ) -> Stage3SupervisionBatch:
        del device, model
        teacher_flags.append(include_teacher_intermediate)
        result = _supervision_batch(batch)
        if include_teacher_intermediate:
            result.state_kinds = (
                "model_generated_intermediate",
                *result.state_kinds[1:],
            )
            result.model_intermediate_count = 1
        return result

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1_000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 600)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        stage3_engine.torch,
        "autocast",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(stage3_engine, "capture_rng_state", lambda: {"sentinel": True})
    monkeypatch.setattr(stage3_engine, "restore_rng_state", lambda _: None)
    monkeypatch.setattr(stage3_engine, "_synthetic_probe_batch", fake_batch)
    monkeypatch.setattr(
        stage3_engine,
        "build_stage3_optimizer",
        lambda candidate: original_build_optimizer(candidate, fused_if_supported=False),
    )
    selected, trials = stage3_engine.select_stage3_micro_batch(
        model,
        device=torch.device("cuda"),
        candidates=(8,),
    )
    assert selected == 8
    assert len(trials) == 1
    assert trials[0].completed_optimizer_steps == 10
    assert trials[0].images_per_second > 0
    assert teacher_flags == [True] + [False] * 9
    assert all(
        torch.equal(model.state_dict()[name], expected)
        for name, expected in pristine.items()
    )


def test_stage3_micro_probe_propagates_non_oom_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_graphrestore()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1_000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(stage3_engine, "capture_rng_state", lambda: {"sentinel": True})
    monkeypatch.setattr(stage3_engine, "restore_rng_state", lambda _: None)
    monkeypatch.setattr(
        stage3_engine,
        "build_stage3_optimizer",
        lambda candidate: (_ for _ in ()).throw(ValueError("non-OOM")),
    )
    with pytest.raises(ValueError, match="non-OOM"):
        stage3_engine.select_stage3_micro_batch(
            model,
            device=torch.device("cuda"),
            candidates=(8,),
        )
