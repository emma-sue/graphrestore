from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
import pytest

from src.data.manifests import SKILLS
from src.net.graphrestore import GuardedSkillRestormer
from src.training.stage2_distillation import (
    AtomicJsonlShardWriter,
    EFFECT_FIELDS,
    ProgramScore,
    Stage2Paths,
    aggregate_effect_profiles,
    assign_relation_label,
    build_pair_prior,
    build_stage2_decision,
    canonical_skill_pair,
    decision_warnings,
    enumerate_three_programs,
    fit_bradley_terry,
    load_frozen_stage1_ema,
    summarize_relations,
    summarize_split,
    train_val_consistency,
)


class TinyRecordingExecutor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, tuple[int, ...]]] = []

    def forward(
        self,
        x: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        guards: torch.Tensor,
        forced_presence_mask: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> torch.Tensor:
        del guards, forced_presence_mask, return_trace
        active = tuple(torch.nonzero(active_mask[0], as_tuple=False).flatten().tolist())
        self.calls.append((x.detach().clone(), active))
        # Non-commuting affine interventions make the two serial programs differ.
        if active == (0,):
            return x * 0.8 + 0.04
        if active == (1,):
            return x * 0.6 + 0.02
        if active == (0, 1):
            return x * 0.7 + 0.03
        raise AssertionError(active)


def _score(psnr: float, ssim: float) -> ProgramScore:
    return ProgramScore(psnr=psnr, ssim=ssim, residual_norm=0.1)


def _record(
    sample_id: str,
    label: str,
    gap: float,
    pair: str = "noise+motion_blur",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "pair_id": pair,
        "skills": pair.split("+"),
        "label": label,
        "relation_weight": 0.25 if label == "ambiguous" else 1.0,
        "margins": {"serial_gap_psnr": gap},
    }


def test_three_program_enumeration_uses_same_start_and_exact_call_shapes() -> None:
    model = TinyRecordingExecutor().eval()
    x = torch.linspace(0.05, 0.95, 3 * 16 * 16).reshape(3, 16, 16)
    gt = torch.full_like(x, 0.5)
    guards = torch.ones(8, 4, 4)
    scores = enumerate_three_programs(
        model,
        x,
        gt,
        guards,
        skill_i=0,
        skill_j=1,
        device=torch.device("cpu"),
    )
    assert tuple(scores) == ("i_before_j", "j_before_i", "parallel")
    assert [active for _, active in model.calls] == [(0,), (1,), (1,), (0,), (0, 1)]
    assert torch.equal(model.calls[0][0], x.unsqueeze(0))
    assert torch.equal(model.calls[2][0], x.unsqueeze(0))
    assert torch.equal(model.calls[4][0], x.unsqueeze(0))
    assert not torch.equal(model.calls[1][0], x.unsqueeze(0))
    assert not torch.equal(model.calls[3][0], x.unsqueeze(0))
    assert all(math.isfinite(value) for score in scores.values() for value in (score.psnr, score.ssim, score.residual_norm))


def test_label_boundaries_and_ambiguous_weight_are_exact() -> None:
    assert canonical_skill_pair((6, 0)) == (0, 6)
    # Both parallel inequalities are inclusive at exactly the locked boundary.
    parallel = assign_relation_label(
        _score(30.00, 0.900), _score(29.80, 0.890), _score(29.95, 0.899)
    )
    assert parallel.label == "parallel"
    assert parallel.relation_class_index == 2
    assert parallel.relation_weight == 1.0

    serial_psnr = assign_relation_label(
        _score(30.00, 0.900), _score(29.95, 0.901), _score(20.0, 0.5)
    )
    assert serial_psnr.label == "i_before_j"

    serial_ssim = assign_relation_label(
        _score(30.00, 0.904), _score(29.98, 0.902), _score(20.0, 0.5)
    )
    assert serial_ssim.label == "i_before_j"

    ambiguous = assign_relation_label(
        _score(30.00, 0.900), _score(29.98, 0.899), _score(20.0, 0.5)
    )
    assert ambiguous.label == "ambiguous"
    assert ambiguous.relation_class_index is None
    assert ambiguous.relation_weight == 0.25


def test_ambiguous_is_excluded_from_statistics_prior_and_priority() -> None:
    records = [
        _record("a", "i_before_j", 0.10),
        _record("b", "parallel", 0.01),
        _record("c", "ambiguous", 0.03),
        _record("d", "ambiguous", 0.02),
    ]
    summary = summarize_relations(records)
    assert summary["n_total"] == 4
    assert summary["n_ambiguous"] == 2
    assert summary["n_nonambiguous"] == 2
    assert summary["parallel_fraction_nonambiguous"] == 0.5
    assert summary["majority_label_share"] == 0.5
    assert summary["ambiguous_in_pair_prior"] == 0
    prior = build_pair_prior(records)
    assert prior["ambiguous_excluded"] == 2
    assert prior["pairs"]["noise+motion_blur"]["n_nonambiguous"] == 2
    assert sum(prior["pairs"]["noise+motion_blur"]["counts"].values()) == 2
    priority = fit_bradley_terry(records)
    assert priority["n_directional_observations"] == 1
    assert priority["n_ambiguous_excluded"] == 2
    assert priority["n_parallel_excluded"] == 1
    assert priority["priority"]["noise"] > priority["priority"]["motion_blur"]
    assert priority["manual_skill_order_used"] is False

    undefined = summarize_relations([_record("z", "ambiguous", 0.0)])
    assert undefined["parallel_fraction_nonambiguous"] is None
    assert undefined["majority_label_share"] is None
    # The JSON form remains standards-compliant (no non-standard NaN literal).
    assert "NaN" not in json.dumps(undefined, allow_nan=False)


def test_decision_uses_val_view_and_reports_train_val_consistency(tmp_path: Path) -> None:
    train_records = [
        _record("t1", "i_before_j", 0.10),
        _record("t2", "i_before_j", 0.08),
        _record("t3", "ambiguous", 0.01),
    ]
    val_records = [
        _record("v1", "parallel", 0.02),
        _record("v2", "parallel", 0.03),
        _record("v3", "ambiguous", 0.01),
    ]
    train_summary = summarize_split(train_records)
    val_summary = summarize_split(val_records)
    consistency = train_val_consistency(train_summary, val_summary)
    assert consistency["noise+motion_blur"]["consistent"] is False
    warnings, details = decision_warnings(train_summary, val_summary)
    assert "WARNING_LOW_LABEL_SUPPORT" in warnings
    assert details["train"]["warnings_trigger_automatic_model_changes"] is False

    config = tmp_path / "stage2.yaml"
    config.write_text("stage: stage2\n", encoding="utf-8")
    dummy = tmp_path / "unused"
    paths = Stage2Paths(
        project_root=tmp_path,
        config_path=config,
        config={},
        resolved={},
        training_data_root=dummy,
        primary_train=dummy,
        primary_val=dummy,
        checkpoint=dummy,
        effect_profiles=dummy,
        interaction_train_manifest=dummy,
        interaction_val_manifest=dummy,
        relation_train=dummy,
        relation_val=dummy,
        pair_prior=dummy,
        global_priority=dummy,
        decision=dummy,
        summary_csv=dummy,
        report=dummy,
    )
    decision = build_stage2_decision(
        paths=paths,
        checkpoint_sha256="1" * 64,
        train_manifest_sha256="2" * 64,
        val_manifest_sha256="3" * 64,
        relation_train_sha256="4" * 64,
        relation_val_sha256="5" * 64,
        train_summary=train_summary,
        val_summary=val_summary,
        pair_prior_sha256="6" * 64,
        global_priority_sha256="7" * 64,
    )
    assert decision["decision_view"] == "interaction_val"
    assert decision["overall"]["ambiguous_fraction"] == 1 / 3
    assert decision["approved"] is False
    assert decision["stage3_started"] is False
    assert decision["automatic_model_changes"] is False
    json.dumps(decision, allow_nan=False)


def test_effect_profile_grid_is_40d_and_never_uses_absent_zero_guard() -> None:
    records: list[dict[str, object]] = []
    for source_index, source in enumerate(SKILLS):
        for forced_index, forced in enumerate(SKILLS):
            records.append(
                {
                    "sample_id": f"{source}::{forced}",
                    "source_skill": source,
                    "forced_skill": forced,
                    "absent_zero_guard_used": False,
                    "effect": {
                        field: float(source_index + forced_index) / 100.0
                        for field in EFFECT_FIELDS
                    },
                }
            )
    profile = aggregate_effect_profiles(
        records,
        checkpoint_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        selection_manifest_sha256="c" * 64,
        selection_seed=2027,
    )
    assert profile["effect_vector_dim"] == 40
    assert all(len(vector) == 40 for vector in profile["effect_vectors"].values())
    assert profile["guard_policy"]["absent_zero_guard_used"] is False
    assert profile["mio100_rows_read"] == 0
    assert profile["group_b_or_c_rows_generated"] == 0


def test_atomic_shards_resume_and_consolidate_in_manifest_order(tmp_path: Path) -> None:
    output = tmp_path / "relations.jsonl"
    first = AtomicJsonlShardWriter(
        output, signature={"manifest": "a" * 64}, shard_size=1
    )
    first.append({"sample_id": "b", "value": 2})
    assert first.processed == {"b"}

    resumed = AtomicJsonlShardWriter(
        output, signature={"manifest": "a" * 64}, shard_size=1
    )
    assert resumed.processed == {"b"}
    resumed.append({"sample_id": "a", "value": 1})
    digest = resumed.consolidate(expected_ids=["a", "b"])
    assert len(digest) == 64
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in rows] == ["a", "b"]


def test_atomic_shards_refuse_config_or_code_binding_drift(tmp_path: Path) -> None:
    output = tmp_path / "bound.jsonl"
    first = AtomicJsonlShardWriter(
        output,
        signature={
            "manifest": "a" * 64,
            "resume_bindings": {
                "config_sha256": "b" * 64,
                "code_sha256": {"src/training/stage2_distillation.py": "c" * 64},
            },
        },
        shard_size=1,
    )
    first.append({"sample_id": "one"})
    with pytest.raises(Exception, match="signature mismatch"):
        AtomicJsonlShardWriter(
            output,
            signature={
                "manifest": "a" * 64,
                "resume_bindings": {
                    "config_sha256": "d" * 64,
                    "code_sha256": {
                        "src/training/stage2_distillation.py": "e" * 64
                    },
                },
            },
            shard_size=1,
        )


def test_stage1_best_ema_load_is_strict_and_frozen(tmp_path: Path) -> None:
    def factory() -> GuardedSkillRestormer:
        return GuardedSkillRestormer(
            dim=8,
            encoder_blocks=(1, 1, 1, 1),
            decoder_blocks=(1, 1, 1),
            refinement=1,
            heads=(1, 2, 4, 8),
            skill_bottlenecks={
                "level3": 4,
                "level2": 4,
                "level1": 4,
                "refinement": 4,
            },
        )

    source = factory()
    state = source.state_dict()
    checkpoint = tmp_path / "best_ema.pth"
    torch.save(
        {
            "schema_version": "graphrestore-checkpoint-v1",
            "stage": "stage1_skill_bank",
            "model_role": "ema_selection",
            "resumable": False,
            "step": 30_000,
            "model": state,
            "ema": {"shadow": {name: value.clone() for name, value in state.items()}},
        },
        checkpoint,
    )
    snapshot = load_frozen_stage1_ema(
        checkpoint, device=torch.device("cpu"), model_factory=factory
    )
    assert snapshot.checkpoint_step == 30_000
    assert len(snapshot.checkpoint_sha256) == 64
    assert snapshot.model.training is False
    assert not any(parameter.requires_grad for parameter in snapshot.model.parameters())
