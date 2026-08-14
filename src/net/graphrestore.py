"""End-to-end guarded partial-order GraphRestore models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from .cooperative_executor import CooperativeExecutor, ExecutionResult
from .graph_compiler import CompiledGraph, DirectedEdge, GraphCompiler
from .latent_skill_bank import GuardedRestorationDecoder
from .mio_stagea import SharedEncoder
from .program_planner import PlannerOutput, ProgramPlanner
from .restormer_blocks import crop_to_shape, pad_to_multiple
from .skill_adapter import SKILLS, SKILL_TO_INDEX


@dataclass(frozen=True)
class SkillExecutionOutput:
    final: torch.Tensor
    execution: ExecutionResult


@dataclass(frozen=True)
class RoundTrace:
    round_index: int
    active_mask: torch.Tensor
    stopped_mask: torch.Tensor
    skipped_mask: torch.Tensor
    stop_reasons: tuple[str, ...]
    reentry_request_mask: torch.Tensor
    unexpected_activation_mask: torch.Tensor
    execution: ExecutionResult | None


@dataclass
class ProgramGraphState:
    """Fixed initial graph plus monotonic execution feedback.

    ``nodes``, ``edges`` and ``levels`` never change after t=0.  Runtime
    feedback may only move nodes from pending into ``executed`` or ``skipped``.
    """

    nodes: tuple[str, ...]
    edges: tuple[DirectedEdge, ...]
    levels: tuple[tuple[str, ...], ...]
    executed: set[str] = field(default_factory=set)
    skipped: set[str] = field(default_factory=set)
    level_cursor: int = 0

    @classmethod
    def from_compiled(cls, graph: CompiledGraph) -> "ProgramGraphState":
        return cls(
            nodes=graph.active_skills,
            edges=graph.edges,
            levels=graph.levels,
        )

    @property
    def complete(self) -> bool:
        return self.level_cursor >= len(self.levels)

    @property
    def current_level(self) -> tuple[str, ...]:
        return () if self.complete else self.levels[self.level_cursor]

    @property
    def pending(self) -> tuple[str, ...]:
        finished = self.executed | self.skipped
        return tuple(skill for skill in self.nodes if skill not in finished)

    def finish_current_level(
        self,
        *,
        executed: Sequence[str],
        skipped: Sequence[str],
    ) -> None:
        level = set(self.current_level)
        executed_set = set(executed)
        skipped_set = set(skipped)
        if executed_set & skipped_set:
            raise RuntimeError("a graph node cannot be both executed and skipped")
        if executed_set | skipped_set != level:
            raise RuntimeError("every current-level node must be executed or skipped")
        if (executed_set | skipped_set) & (self.executed | self.skipped):
            raise RuntimeError("a graph node was processed more than once")
        self.executed.update(executed_set)
        self.skipped.update(skipped_set)
        self.level_cursor += 1

    def skip_all_pending(self) -> None:
        self.skipped.update(self.pending)
        self.level_cursor = len(self.levels)


@dataclass(frozen=True)
class GraphRestoreOutput:
    final: torch.Tensor
    steps: tuple[torch.Tensor, ...]
    planner_outputs: tuple[PlannerOutput, ...]
    compiled_graphs: tuple[CompiledGraph, ...]
    graph_states: tuple[ProgramGraphState, ...]
    trace: tuple[RoundTrace, ...]


class GuardedSkillRestormer(nn.Module):
    """Stage1 host with teacher-forced active skills and spatial guards."""

    def __init__(
        self,
        dim: int = 48,
        encoder_blocks: Sequence[int] = (4, 6, 6, 8),
        decoder_blocks: Sequence[int] = (6, 6, 4),
        refinement: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        gradient_checkpointing: bool = False,
        skill_bottlenecks: Mapping[str, int] | None = None,
    ):
        super().__init__()
        self.encoder = SharedEncoder(
            dim,
            encoder_blocks,
            heads,
            expansion,
            bias,
            norm_type,
            gradient_checkpointing,
        )
        self.decoder = GuardedRestorationDecoder(
            dim,
            decoder_blocks,
            refinement,
            heads[:3],
            expansion,
            bias,
            norm_type,
            gradient_checkpointing,
            skill_bottlenecks,
        )
        # Parameter-free: retaining this as a module does not add state keys.
        self.executor = CooperativeExecutor()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.encoder(x)

    def decode_delta(
        self,
        features: Sequence[torch.Tensor],
        *,
        guards: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(features, guards=guards, active_mask=active_mask)

    def execute_level(
        self,
        current: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
        *,
        guards: torch.Tensor,
        active_mask: torch.Tensor,
        forced_presence_mask: torch.Tensor | None = None,
    ) -> ExecutionResult:
        return self.executor(
            current,
            encoder_features,
            self.decoder,
            guards=guards,
            active_mask=active_mask,
            forced_presence_mask=forced_presence_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        guards: torch.Tensor,
        forced_presence_mask: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> torch.Tensor | SkillExecutionOutput:
        """Run a single teacher-forced/cooperative skill level.

        ``guards`` are effective teacher/predicted spatial guards.  During a
        counterfactual forced call, the forced skill is ORed into ``active_mask``;
        callers using planner outputs should obtain guards with
        ``PlannerOutput.execution_guards(forced_presence_mask)``.
        """

        padded, original_shape = pad_to_multiple(x, 8)
        features = self.encode(padded)
        execution = self.execute_level(
            padded,
            features,
            guards=guards,
            active_mask=active_mask,
            forced_presence_mask=forced_presence_mask,
        )
        final = crop_to_shape(execution.next_image, original_shape)
        if return_trace:
            return SkillExecutionOutput(final=final, execution=execution)
        return final


class GraphRestore(GuardedSkillRestormer):
    """Full once-compiled partial-order program with per-round state updates."""

    def __init__(
        self,
        dim: int = 48,
        encoder_blocks: Sequence[int] = (4, 6, 6, 8),
        decoder_blocks: Sequence[int] = (6, 6, 4),
        refinement: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        gradient_checkpointing: bool = False,
        skill_bottlenecks: Mapping[str, int] | None = None,
        planner_fpn_dim: int = 96,
        planner_context_dim: int = 192,
        effect_profile_dim: int = 40,
        pair_prior: Mapping[object, object] | None = None,
        global_priority: Mapping[str, float] | Sequence[float] | None = None,
        compiler_mode: str = "full_partial_order",
        max_active_skills: int = 3,
        kmax_train: int = 2,
        kmax_test: int = 3,
        allow_skill_reentry: bool = False,
        max_calls_per_skill: int = 1,
    ):
        super().__init__(
            dim,
            encoder_blocks,
            decoder_blocks,
            refinement,
            heads,
            expansion,
            bias,
            norm_type,
            gradient_checkpointing,
            skill_bottlenecks,
        )
        if max_active_skills != 3:
            raise ValueError("main GraphRestore capacity is contract-bound to top-3")
        if kmax_train != 2 or kmax_test != 3:
            raise ValueError("scientific design lock requires Kmax_train=2/Kmax_test=3")
        if max_calls_per_skill <= 0:
            raise ValueError("max_calls_per_skill must be positive")
        if allow_skill_reentry or max_calls_per_skill != 1:
            raise ValueError(
                "scientific design lock requires no re-entry and max_calls_per_skill=1"
            )
        self.planner = ProgramPlanner(
            encoder_channels=tuple(dim * 2**level for level in range(4)),
            trace_channels=(dim, dim * 2, dim * 4),
            fpn_dim=planner_fpn_dim,
            context_dim=planner_context_dim,
            effect_profile_dim=effect_profile_dim,
        )
        self.compiler = GraphCompiler(
            pair_prior=pair_prior,
            global_priority=global_priority,
            mode=compiler_mode,
        )
        self.max_active_skills = max_active_skills
        self.kmax_train = kmax_train
        self.kmax_test = kmax_test
        self.allow_skill_reentry = allow_skill_reentry
        self.max_calls_per_skill = max_calls_per_skill
        self.register_buffer(
            "presence_thresholds",
            torch.full((len(SKILLS),), 0.5),
            persistent=True,
        )

    @torch.no_grad()
    def set_presence_thresholds(self, thresholds: torch.Tensor | Sequence[float]) -> None:
        values = torch.as_tensor(
            thresholds,
            device=self.presence_thresholds.device,
            dtype=self.presence_thresholds.dtype,
        )
        if tuple(values.shape) != (len(SKILLS),):
            raise ValueError("presence thresholds must contain eight values")
        if bool(torch.any((values < 0.0) | (values > 1.0)).item()):
            raise ValueError("presence thresholds must lie in [0,1]")
        self.presence_thresholds.copy_(values)

    def plan_state(
        self,
        x0: torch.Tensor,
        current: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
        *,
        round_value: float | torch.Tensor,
        compute_relations: bool = True,
    ) -> PlannerOutput:
        return self.planner(
            x0,
            current,
            encoder_features,
            round_value=round_value,
            compute_relations=compute_relations,
        )

    def _threshold_tensor(
        self,
        override: torch.Tensor | Sequence[float] | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        values = self.presence_thresholds if override is None else torch.as_tensor(override)
        values = values.to(device=reference.device, dtype=reference.dtype)
        if tuple(values.shape) != (len(SKILLS),):
            raise ValueError("presence_thresholds override must contain eight values")
        return values

    def _select_active(
        self,
        probabilities: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> tuple[int, ...]:
        passed = [
            index
            for index in range(len(SKILLS))
            if float(probabilities[index]) >= float(thresholds[index])
        ]
        passed.sort(key=lambda index: (-float(probabilities[index]), index))
        if passed:
            return tuple(sorted(passed[: self.max_active_skills]))
        maximum, index_tensor = probabilities.max(dim=0)
        if float(maximum) < 0.15:
            return ()
        return (int(index_tensor),)

    def compile_initial(
        self,
        planner_output: PlannerOutput,
        *,
        presence_thresholds: torch.Tensor | Sequence[float] | None = None,
    ) -> tuple[CompiledGraph, ...]:
        probabilities = planner_output.presence_probabilities.detach()
        thresholds = self._threshold_tensor(presence_thresholds, probabilities)
        return tuple(
            self.compiler.compile(
                self._select_active(probabilities[index], thresholds),
                planner_output.relation_logits[index],
            )
            for index in range(probabilities.shape[0])
        )

    def _round_active_masks(
        self,
        planner_output: PlannerOutput,
        compiled: Sequence[CompiledGraph],
        graph_states: Sequence[ProgramGraphState],
        executed: torch.Tensor,
        thresholds: torch.Tensor,
        forced_counterfactual_mask: torch.Tensor,
        round_index: int,
        terminal: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[str, ...],
        torch.Tensor,
        torch.Tensor,
        tuple[tuple[str, ...] | None, ...],
    ]:
        probabilities = planner_output.presence_probabilities.detach()
        batch = probabilities.shape[0]
        active = torch.zeros(
            batch,
            len(SKILLS),
            device=probabilities.device,
            dtype=torch.bool,
        )
        skipped = torch.zeros_like(active)
        stopped = torch.zeros(batch, device=probabilities.device, dtype=torch.bool)
        reasons: list[str] = []
        processed_levels: list[tuple[str, ...] | None] = []
        above_threshold = probabilities >= thresholds[None, :]
        reentry = executed & above_threshold
        initial = torch.zeros_like(executed)
        for sample, graph in enumerate(compiled):
            for skill in graph.active_skills:
                initial[sample, SKILL_TO_INDEX[skill]] = True
        unexpected = (~initial) & above_threshold

        for sample, (graph, state) in enumerate(zip(compiled, graph_states)):
            if bool(terminal[sample].item()):
                stopped[sample] = True
                reasons.append("already_terminal")
                processed_levels.append(None)
                continue
            forced = forced_counterfactual_mask[sample]
            if bool(torch.any(forced).item()):
                if round_index == 0:
                    active[sample] = forced
                    reasons.append("forced_counterfactual")
                else:
                    stopped[sample] = True
                    reasons.append("forced_counterfactual_complete")
                processed_levels.append(None)
                continue

            if state.complete:
                stopped[sample] = True
                reasons.append("program_complete")
                processed_levels.append(None)
                continue

            pending_indices = [SKILL_TO_INDEX[name] for name in state.pending]
            pending_probabilities = probabilities[sample, pending_indices]
            pending_thresholds = thresholds[pending_indices]
            confident_remaining = bool(
                torch.any(pending_probabilities >= pending_thresholds).item()
            )
            stop_probability = float(planner_output.stop_logit[sample].sigmoid())
            if stop_probability >= 0.5 and not confident_remaining:
                for skill in state.pending:
                    skipped[sample, SKILL_TO_INDEX[skill]] = True
                state.skip_all_pending()
                stopped[sample] = True
                reasons.append("stop_head_no_confident_remaining")
                processed_levels.append(None)
                continue

            level = state.current_level
            for skill in level:
                skill_index = SKILL_TO_INDEX[skill]
                if bool(above_threshold[sample, skill_index].item()):
                    active[sample, skill_index] = True
                else:
                    skipped[sample, skill_index] = True
            reasons.append(
                "compiled_level_execute"
                if bool(torch.any(active[sample]).item())
                else "compiled_level_all_skipped"
            )
            processed_levels.append(level)
        return (
            active,
            skipped,
            stopped,
            tuple(reasons),
            reentry,
            unexpected,
            tuple(processed_levels),
        )

    def execute_planned_level(
        self,
        current: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
        planner_output: PlannerOutput,
        *,
        active_mask: torch.Tensor,
        forced_presence_mask: torch.Tensor | None = None,
    ) -> ExecutionResult:
        guards = planner_output.execution_guards(forced_presence_mask)
        return self.execute_level(
            current,
            encoder_features,
            guards=guards,
            active_mask=active_mask,
            forced_presence_mask=forced_presence_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        presence_thresholds: torch.Tensor | Sequence[float] | None = None,
        forced_counterfactual_mask: torch.Tensor | None = None,
        max_rounds: int | None = None,
        return_trace: bool = False,
    ) -> torch.Tensor | GraphRestoreOutput:
        """Run one initially compiled DAG while replanning guards/stop per round.

        Relations are compiled exactly once at ``t=0``.  Subsequent rounds
        re-encode the current image and update guards, presence and stop, but
        never insert a newly detected skill into the frozen main-version DAG.
        ``forced_counterfactual_mask`` executes only those skills for one round
        with an execution presence override; planner targets remain untouched.
        """

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("GraphRestore input must be RGB BCHW")
        if max_rounds is None:
            rounds = self.kmax_train if self.training else self.kmax_test
        else:
            rounds = int(max_rounds)
        if rounds <= 0:
            raise ValueError("max_rounds must be positive")
        padded, original_shape = pad_to_multiple(x, 8)
        x0 = padded
        current = padded
        batch = x.shape[0]
        if forced_counterfactual_mask is None:
            forced = torch.zeros(
                batch,
                len(SKILLS),
                device=x.device,
                dtype=torch.bool,
            )
        else:
            if tuple(forced_counterfactual_mask.shape) != (batch, len(SKILLS)):
                raise ValueError("forced_counterfactual_mask must be [B,8]")
            forced = forced_counterfactual_mask.to(device=x.device, dtype=torch.bool)

        features = self.encode(current)
        planner_output = self.plan_state(
            x0,
            current,
            features,
            round_value=0.0,
            compute_relations=True,
        )
        thresholds = self._threshold_tensor(
            presence_thresholds,
            planner_output.presence_logits,
        )
        compiled = self.compile_initial(
            planner_output,
            presence_thresholds=thresholds,
        )
        graph_states = [ProgramGraphState.from_compiled(graph) for graph in compiled]
        executed = torch.zeros(
            batch,
            len(SKILLS),
            device=x.device,
            dtype=torch.bool,
        )
        terminal = torch.zeros(batch, device=x.device, dtype=torch.bool)
        planner_outputs: list[PlannerOutput] = []
        step_images: list[torch.Tensor] = []
        traces: list[RoundTrace] = []

        for round_index in range(rounds):
            if bool(torch.all(terminal).item()):
                break
            if round_index > 0:
                features = self.encode(current)
                planner_output = self.plan_state(
                    x0,
                    current,
                    features,
                    round_value=round_index / max(rounds, 1),
                    compute_relations=False,
                )
            planner_outputs.append(planner_output)
            (
                active,
                skipped,
                stopped,
                reasons,
                reentry,
                unexpected,
                processed_levels,
            ) = self._round_active_masks(
                planner_output,
                compiled,
                graph_states,
                executed,
                thresholds,
                forced,
                round_index,
                terminal,
            )
            has_active = bool(torch.any(active).item())
            has_skipped = bool(torch.any(skipped).item())
            if not has_active and not has_skipped:
                break

            execution: ExecutionResult | None = None
            if has_active:
                forced_this_round = (
                    forced if round_index == 0 else torch.zeros_like(forced)
                )
                execution = self.execute_planned_level(
                    current,
                    features,
                    planner_output,
                    active_mask=active,
                    forced_presence_mask=forced_this_round,
                )
                current = execution.next_image
                executed = executed | active

            for sample, level in enumerate(processed_levels):
                if level is None:
                    continue
                executed_names = tuple(
                    skill
                    for skill in level
                    if bool(active[sample, SKILL_TO_INDEX[skill]].item())
                )
                skipped_names = tuple(
                    skill
                    for skill in level
                    if bool(skipped[sample, SKILL_TO_INDEX[skill]].item())
                )
                graph_states[sample].finish_current_level(
                    executed=executed_names,
                    skipped=skipped_names,
                )

            terminal = terminal | stopped
            terminal = terminal | torch.any(forced, dim=1)
            terminal = terminal | torch.tensor(
                [state.complete for state in graph_states],
                device=terminal.device,
                dtype=torch.bool,
            )

            step_images.append(crop_to_shape(current, original_shape))
            traces.append(
                RoundTrace(
                    round_index=round_index,
                    active_mask=active,
                    stopped_mask=stopped,
                    skipped_mask=skipped,
                    stop_reasons=reasons,
                    reentry_request_mask=reentry,
                    unexpected_activation_mask=unexpected,
                    execution=execution,
                )
            )

        final = crop_to_shape(current, original_shape)
        if return_trace:
            return GraphRestoreOutput(
                final=final,
                steps=tuple(step_images),
                planner_outputs=tuple(planner_outputs),
                compiled_graphs=tuple(compiled),
                graph_states=tuple(graph_states),
                trace=tuple(traces),
            )
        return final
