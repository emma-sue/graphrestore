"""Deterministic cycle-free compiler for partial-order skill programs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .program_planner import PAIR_INDICES, RELATION_CLASSES
from .skill_adapter import SKILLS, skill_indices


PAIR_TO_ROW = {pair: row for row, pair in enumerate(PAIR_INDICES)}


@dataclass(frozen=True)
class DirectedEdge:
    source: str
    target: str
    confidence: float
    decision_source: str
    probabilities: tuple[float, float, float]


@dataclass(frozen=True)
class PairDecision:
    first: str
    second: str
    relation: str
    confidence: float
    decision_source: str
    probabilities: tuple[float, float, float]


@dataclass(frozen=True)
class DroppedEdge:
    edge: DirectedEdge
    reason: str = "cycle"


@dataclass(frozen=True)
class CompiledGraph:
    active_skills: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]
    edges: tuple[DirectedEdge, ...]
    dropped_edges: tuple[DroppedEdge, ...]
    pair_decisions: tuple[PairDecision, ...]

    @property
    def cycle_free(self) -> bool:
        return _is_acyclic(
            tuple(range(len(self.active_skills))),
            tuple(
                (
                    self.active_skills.index(edge.source),
                    self.active_skills.index(edge.target),
                )
                for edge in self.edges
            ),
        )


def _is_acyclic(nodes: Sequence[int], edges: Sequence[tuple[int, int]]) -> bool:
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    ready = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited == len(nodes)


def _would_create_cycle(
    adjacency: Mapping[int, set[int]],
    source: int,
    target: int,
) -> bool:
    stack = [target]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, ()))
    return False


def _topological_levels(
    nodes: Sequence[int],
    edges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    adjacency = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
    ready = sorted(node for node in nodes if indegree[node] == 0)
    levels: list[tuple[int, ...]] = []
    visited = 0
    while ready:
        level = tuple(ready)
        levels.append(level)
        visited += len(level)
        next_ready: list[int] = []
        for source in level:
            for target in sorted(adjacency[source]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    next_ready.append(target)
        ready = sorted(next_ready)
    if visited != len(nodes):
        raise RuntimeError("compiler produced a cyclic graph")
    return tuple(levels)


class GraphCompiler:
    """Compile active skill relations once, dropping cycle-closing edges."""

    VALID_MODES = {"full_partial_order", "forced_total_order", "parallel_only"}

    def __init__(
        self,
        *,
        pair_prior: Mapping[Any, Any] | None = None,
        global_priority: Mapping[str, float] | Sequence[float] | None = None,
        mode: str = "full_partial_order",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown compiler mode: {mode}")
        self.pair_prior = dict(pair_prior or {})
        if global_priority is None:
            self.global_priority = {skill: 0.0 for skill in SKILLS}
        elif isinstance(global_priority, Mapping):
            self.global_priority = {
                skill: float(global_priority.get(skill, 0.0)) for skill in SKILLS
            }
        else:
            if len(global_priority) != len(SKILLS):
                raise ValueError("global_priority must contain eight scores")
            self.global_priority = {
                skill: float(global_priority[index])
                for index, skill in enumerate(SKILLS)
            }
        self.mode = mode

    @staticmethod
    def _probability_tuple(values: Any) -> tuple[float, float, float] | None:
        if isinstance(values, Mapping):
            try:
                result = tuple(float(values[name]) for name in RELATION_CLASSES)
            except (KeyError, TypeError, ValueError):
                return None
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if len(values) != 3:
                return None
            result = tuple(float(value) for value in values)
        else:
            return None
        if any(value < 0 for value in result) or sum(result) <= 0:
            return None
        total = sum(result)
        return tuple(value / total for value in result)  # type: ignore[return-value]

    def _prior_for(self, first: int, second: int) -> tuple[float, float, float] | None:
        candidates: tuple[Any, ...] = (
            (first, second),
            (SKILLS[first], SKILLS[second]),
            f"{SKILLS[first]}|{SKILLS[second]}",
            f"{SKILLS[first]}+{SKILLS[second]}",
        )
        for key in candidates:
            if key in self.pair_prior:
                return self._probability_tuple(self.pair_prior[key])
        return None

    def _priority_direction(self, first: int, second: int) -> tuple[int, int]:
        first_score = self.global_priority[SKILLS[first]]
        second_score = self.global_priority[SKILLS[second]]
        if first_score > second_score:
            return first, second
        if second_score > first_score:
            return second, first
        return (first, second) if first < second else (second, first)

    @staticmethod
    def _edge_from_class(
        first: int,
        second: int,
        relation_class: int,
    ) -> tuple[int, int] | None:
        if relation_class == 0:
            return first, second
        if relation_class == 1:
            return second, first
        if relation_class == 2:
            return None
        raise ValueError(f"invalid relation class: {relation_class}")

    def _full_decision(
        self,
        first: int,
        second: int,
        probabilities: tuple[float, float, float],
    ) -> tuple[str, float, str, tuple[int, int] | None]:
        p_ij, p_ji, p_parallel = probabilities
        if p_parallel >= 0.50 and p_parallel - max(p_ij, p_ji) >= 0.05:
            return "parallel", p_parallel - max(p_ij, p_ji), "predicted", None

        direction_class = 0 if p_ij >= p_ji else 1
        direction_probability = probabilities[direction_class]
        reverse_probability = probabilities[1 - direction_class]
        edge_confidence = direction_probability - max(
            reverse_probability, p_parallel
        )
        if max(probabilities) >= 0.45 and edge_confidence >= 0.08:
            edge = self._edge_from_class(first, second, direction_class)
            return RELATION_CLASSES[direction_class], edge_confidence, "predicted", edge

        prior = self._prior_for(first, second)
        if prior is not None and max(prior) >= 0.60:
            prior_class = max(range(3), key=prior.__getitem__)
            confidence = prior[prior_class] - max(
                prior[index] for index in range(3) if index != prior_class
            )
            edge = self._edge_from_class(first, second, prior_class)
            return RELATION_CLASSES[prior_class], confidence, "pair_prior", edge
        if p_parallel >= 0.40:
            return "parallel", p_parallel - max(p_ij, p_ji), "parallel_fallback", None

        source, target = self._priority_direction(first, second)
        relation = "i_before_j" if source == first else "j_before_i"
        return relation, 0.0, "global_priority", (source, target)

    def compile(
        self,
        active_skills: Sequence[str | int],
        relation_logits: torch.Tensor,
    ) -> CompiledGraph:
        active = tuple(sorted(skill_indices(active_skills)))
        active_names = tuple(SKILLS[index] for index in active)
        if not active:
            return CompiledGraph((), (), (), (), ())
        if tuple(relation_logits.shape) != (len(PAIR_INDICES), 3):
            raise ValueError(
                f"relation_logits must be [{len(PAIR_INDICES)},3], got "
                f"{tuple(relation_logits.shape)}"
            )

        if self.mode == "parallel_only":
            return CompiledGraph(
                active_names,
                (active_names,),
                (),
                (),
                (),
            )
        if self.mode == "forced_total_order":
            ordered = tuple(
                sorted(
                    active,
                    key=lambda index: (-self.global_priority[SKILLS[index]], index),
                )
            )
            edges = tuple(
                DirectedEdge(
                    SKILLS[source],
                    SKILLS[target],
                    0.0,
                    "forced_total_order",
                    (0.0, 0.0, 0.0),
                )
                for source, target in zip(ordered[:-1], ordered[1:])
            )
            return CompiledGraph(
                active_names,
                tuple((SKILLS[index],) for index in ordered),
                edges,
                (),
                (),
            )

        logits = relation_logits.detach().to(device="cpu", dtype=torch.float64)
        probabilities_all = logits.softmax(dim=-1)
        decisions: list[PairDecision] = []
        edge_candidates: list[tuple[int, int, DirectedEdge]] = []
        for position, first in enumerate(active):
            for second in active[position + 1 :]:
                row = PAIR_TO_ROW[(first, second)]
                probabilities = tuple(
                    float(value) for value in probabilities_all[row].tolist()
                )
                relation, confidence, source_kind, edge_pair = self._full_decision(
                    first,
                    second,
                    probabilities,  # type: ignore[arg-type]
                )
                decisions.append(
                    PairDecision(
                        first=SKILLS[first],
                        second=SKILLS[second],
                        relation=relation,
                        confidence=float(confidence),
                        decision_source=source_kind,
                        probabilities=probabilities,  # type: ignore[arg-type]
                    )
                )
                if edge_pair is not None:
                    source, target = edge_pair
                    edge_candidates.append(
                        (
                            source,
                            target,
                            DirectedEdge(
                                source=SKILLS[source],
                                target=SKILLS[target],
                                confidence=float(confidence),
                                decision_source=source_kind,
                                probabilities=probabilities,  # type: ignore[arg-type]
                            ),
                        )
                    )

        edge_candidates.sort(
            key=lambda item: (-item[2].confidence, item[0], item[1])
        )
        adjacency = {node: set() for node in active}
        kept: list[tuple[int, int, DirectedEdge]] = []
        dropped: list[DroppedEdge] = []
        for source, target, edge in edge_candidates:
            if _would_create_cycle(adjacency, source, target):
                dropped.append(DroppedEdge(edge=edge))
                continue
            adjacency[source].add(target)
            kept.append((source, target, edge))

        levels_index = _topological_levels(
            active,
            tuple((source, target) for source, target, _ in kept),
        )
        levels = tuple(
            tuple(SKILLS[index] for index in level) for level in levels_index
        )
        result = CompiledGraph(
            active_skills=active_names,
            levels=levels,
            edges=tuple(edge for _, _, edge in kept),
            dropped_edges=tuple(dropped),
            pair_decisions=tuple(decisions),
        )
        if not result.cycle_free:
            raise RuntimeError("post-compiler graph is cyclic")
        return result
