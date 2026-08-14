from __future__ import annotations

import torch

from src.net.graph_compiler import GraphCompiler, PAIR_TO_ROW


def _relations(default=(0.1, 0.1, 0.8)) -> torch.Tensor:
    probabilities = torch.tensor(default, dtype=torch.float64)
    return probabilities.log().repeat(28, 1)


def _set(
    logits: torch.Tensor,
    first: int,
    second: int,
    probabilities: tuple[float, float, float],
) -> None:
    logits[PAIR_TO_ROW[(first, second)]] = torch.tensor(
        probabilities,
        dtype=logits.dtype,
    ).log()


def test_pure_chain() -> None:
    logits = _relations()
    _set(logits, 0, 1, (0.90, 0.05, 0.05))
    _set(logits, 0, 2, (0.85, 0.05, 0.10))
    _set(logits, 1, 2, (0.90, 0.05, 0.05))
    graph = GraphCompiler().compile((0, 1, 2), logits)
    assert graph.levels == (
        ("noise",),
        ("motion_blur",),
        ("defocus_blur",),
    )
    assert graph.cycle_free
    assert not graph.dropped_edges


def test_all_parallel() -> None:
    graph = GraphCompiler().compile((0, 1, 2), _relations())
    assert graph.levels == (("noise", "motion_blur", "defocus_blur"),)
    assert graph.edges == ()
    assert graph.cycle_free


def test_v_shape() -> None:
    logits = _relations()
    _set(logits, 0, 2, (0.90, 0.05, 0.05))
    _set(logits, 1, 2, (0.90, 0.05, 0.05))
    graph = GraphCompiler().compile((0, 1, 2), logits)
    assert graph.levels == (
        ("noise", "motion_blur"),
        ("defocus_blur",),
    )
    assert graph.cycle_free


def test_three_cycle_drops_lowest_confidence_edge() -> None:
    logits = _relations()
    # noise -> motion (0.85 margin), motion -> defocus (0.70),
    # defocus -> noise (0.50).  The final edge must be rejected.
    _set(logits, 0, 1, (0.90, 0.05, 0.05))
    _set(logits, 1, 2, (0.80, 0.10, 0.10))
    _set(logits, 0, 2, (0.20, 0.70, 0.10))
    graph = GraphCompiler().compile((0, 1, 2), logits)
    assert graph.cycle_free
    assert len(graph.dropped_edges) == 1
    dropped = graph.dropped_edges[0].edge
    assert (dropped.source, dropped.target) == ("defocus_blur", "noise")
    assert graph.levels == (
        ("noise",),
        ("motion_blur",),
        ("defocus_blur",),
    )


def test_low_confidence_uses_priority_without_cycle() -> None:
    logits = torch.zeros(28, 3)
    compiler = GraphCompiler(
        global_priority={"haze": 2.0, "rain": 1.0},
    )
    graph = compiler.compile((4, 5), logits)
    assert graph.levels == (("haze",), ("rain",))
    assert graph.edges[0].decision_source == "global_priority"
