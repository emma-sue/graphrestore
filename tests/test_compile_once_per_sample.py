"""SCIENTIFIC_DESIGN_LOCK: one relation plan with dynamic execution feedback."""

from __future__ import annotations

import torch
from torch import nn

from src.net.graph_compiler import GraphCompiler, PAIR_TO_ROW
from src.net.graphrestore import GraphRestore
from src.net.program_planner import PlannerOutput
from src.net.skill_adapter import SKILLS


class _ChangingRelationPlanner(nn.Module):
    """Return deliberately reversed later relations even when not requested."""

    def __init__(self, *, drop_second_after_first: bool = False) -> None:
        super().__init__()
        self.calls = 0
        self.drop_second_after_first = drop_second_after_first

    def forward(
        self,
        x0,
        xt,
        encoder_features,
        *,
        round_value,
        compute_relations=True,
    ):
        batch, _, height, width = xt.shape
        presence = torch.full(
            (batch, len(SKILLS)), -10.0, device=xt.device, dtype=xt.dtype
        )
        presence[:, :2] = 10.0
        if self.drop_second_after_first and self.calls > 0:
            presence[:, 1] = -10.0
        guards = torch.full(
            (batch, len(SKILLS), height // 4, width // 4),
            10.0,
            device=xt.device,
            dtype=xt.dtype,
        )
        relations = torch.zeros(
            batch, 28, 3, device=xt.device, dtype=xt.dtype
        )
        relation_class = 0 if self.calls == 0 else 1
        relations[:, PAIR_TO_ROW[(0, 1)], relation_class] = 10.0
        relations[:, PAIR_TO_ROW[(0, 1)], 1 - relation_class] = -10.0
        self.calls += 1
        return PlannerOutput(
            guard_logits=guards,
            presence_logits=presence,
            stop_logit=torch.full(
                (batch, 1), -10.0, device=xt.device, dtype=xt.dtype
            ),
            relation_logits=relations,
            global_context=torch.zeros(batch, 8, device=xt.device, dtype=xt.dtype),
        )


class _CountingCompiler:
    def __init__(self) -> None:
        self.delegate = GraphCompiler()
        self.calls = 0

    def compile(self, active_skills, relation_logits):
        self.calls += 1
        return self.delegate.compile(active_skills, relation_logits)


def _tiny_model() -> GraphRestore:
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
        planner_fpn_dim=16,
        planner_context_dim=32,
        effect_profile_dim=8,
    ).eval()


def test_compile_once_per_sample() -> None:
    model = _tiny_model()
    planner = _ChangingRelationPlanner()
    compiler = _CountingCompiler()
    model.planner = planner
    model.compiler = compiler
    images = torch.rand(2, 3, 16, 16)
    with torch.inference_mode():
        result = model(images, max_rounds=3, return_trace=True)

    assert compiler.calls == images.shape[0]
    assert len(result.planner_outputs) == 2
    first_relation = result.planner_outputs[0].relation_logits[
        0, PAIR_TO_ROW[(0, 1)]
    ]
    changed_later_relation = result.planner_outputs[1].relation_logits[
        0, PAIR_TO_ROW[(0, 1)]
    ]
    assert int(first_relation.argmax()) == 0
    assert int(changed_later_relation.argmax()) == 1

    for graph, state in zip(result.compiled_graphs, result.graph_states):
        initial_nodes = graph.active_skills
        initial_edges = graph.edges
        assert initial_nodes == ("noise", "motion_blur")
        assert tuple((edge.source, edge.target) for edge in initial_edges) == (
            ("noise", "motion_blur"),
        )
        assert state.nodes == initial_nodes
        assert state.edges == initial_edges
        assert state.levels == graph.levels
        assert state.executed == {"noise", "motion_blur"}
        assert state.skipped == set()
        assert state.pending == ()
        assert state.executed.isdisjoint(state.skipped)
        assert state.executed | state.skipped <= set(initial_nodes)
    assert result.trace[0].active_mask[:, 0].all()
    assert result.trace[1].active_mask[:, 1].all()
    assert not result.trace[1].active_mask[:, 0].any()


def test_dynamic_feedback_only_skips_initial_nodes_without_reentry() -> None:
    model = _tiny_model()
    planner = _ChangingRelationPlanner(drop_second_after_first=True)
    compiler = _CountingCompiler()
    model.planner = planner
    model.compiler = compiler
    with torch.inference_mode():
        result = model(torch.rand(1, 3, 16, 16), max_rounds=3, return_trace=True)

    assert compiler.calls == 1
    graph = result.compiled_graphs[0]
    state = result.graph_states[0]
    assert state.nodes == graph.active_skills == ("noise", "motion_blur")
    assert state.edges == graph.edges
    assert state.executed == {"noise"}
    assert state.skipped == {"motion_blur"}
    assert state.pending == ()
    assert bool(result.trace[1].skipped_mask[0, 1])
    assert result.trace[1].execution is None
