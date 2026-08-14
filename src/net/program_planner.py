"""Interaction-aware partial-order planner with spatial skill guards."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .skill_adapter import SKILLS
from .trace_pyramid import TraceFeatures, TracePyramid


PAIR_INDICES: tuple[tuple[int, int], ...] = tuple(
    (first, second)
    for first in range(len(SKILLS))
    for second in range(first + 1, len(SKILLS))
)
RELATION_CLASSES = ("i_before_j", "j_before_i", "parallel")


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ProjectNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.project(x)))


class SmoothFeature(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
        )
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.project(self.depthwise(x))))


class ContinuousRoundEmbedding(nn.Module):
    """Continuous sinusoidal round encoding; no discrete round lookup table."""

    def __init__(self, dimension: int = 32):
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("round embedding dimension must be even and >=4")
        half = dimension // 2
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(1000.0), half)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )

    def forward(
        self,
        value: float | torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        round_value = torch.as_tensor(value, device=device, dtype=torch.float32)
        if round_value.ndim == 0:
            round_value = round_value.expand(batch_size)
        round_value = round_value.reshape(-1)
        if round_value.numel() != batch_size:
            raise ValueError("round value must be scalar or have one value per sample")
        phase = 2.0 * math.pi * round_value[:, None] * self.frequencies[None, :]
        encoded = torch.cat((phase.sin(), phase.cos()), dim=1).to(dtype=dtype)
        return self.mlp(encoded)


@dataclass(frozen=True)
class PlannerOutput:
    guard_logits: torch.Tensor
    presence_logits: torch.Tensor
    stop_logit: torch.Tensor
    relation_logits: torch.Tensor
    global_context: torch.Tensor

    @property
    def presence_probabilities(self) -> torch.Tensor:
        return self.presence_logits.sigmoid()

    @property
    def spatial_guard_probabilities(self) -> torch.Tensor:
        return self.guard_logits.sigmoid()

    def execution_guards(
        self,
        forced_presence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``presence * spatial_guard`` with explicit forced-call override.

        A forced counterfactual call overrides only the execution presence gate;
        it does not alter the planner logit or its absent-skill supervision.
        """

        presence = self.presence_probabilities
        if forced_presence_mask is not None:
            if forced_presence_mask.shape != presence.shape:
                raise ValueError("forced_presence_mask must match presence logits")
            forced = forced_presence_mask.to(device=presence.device, dtype=torch.bool)
            presence = torch.where(forced, torch.ones_like(presence), presence)
        return presence[:, :, None, None] * self.spatial_guard_probabilities


class ProgramPlanner(nn.Module):
    """Predict guards, skill presence, stop, and all 28 shared-head relations."""

    def __init__(
        self,
        *,
        encoder_channels: Sequence[int] = (48, 96, 192, 384),
        trace_channels: tuple[int, int, int] = (48, 96, 192),
        fpn_dim: int = 96,
        context_dim: int = 192,
        round_dim: int = 32,
        skill_embedding_dim: int = 32,
        effect_profile_dim: int = 40,
    ):
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("planner requires four encoder feature levels")
        if effect_profile_dim <= 0:
            raise ValueError("effect_profile_dim must be positive")
        self.num_skills = len(SKILLS)
        self.effect_profile_dim = effect_profile_dim
        self.trace = TracePyramid(12, trace_channels)
        self.encoder_projects = nn.ModuleList(
            [ProjectNorm(int(channels), fpn_dim) for channels in encoder_channels]
        )
        self.trace_projects = nn.ModuleList(
            [ProjectNorm(int(channels), fpn_dim) for channels in trace_channels]
        )
        self.smooth4 = SmoothFeature(fpn_dim)
        self.smooth3 = SmoothFeature(fpn_dim)
        self.smooth2 = SmoothFeature(fpn_dim)
        self.smooth1 = SmoothFeature(fpn_dim)
        self.guard_fusion = SmoothFeature(fpn_dim)

        self.round_embedding = ContinuousRoundEmbedding(round_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(fpn_dim * 4 + round_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
        )
        self.guard_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(fpn_dim, self.num_skills, 1),
        )
        self.guard_context_bias = nn.Linear(context_dim, self.num_skills)
        self.presence_head = nn.Linear(context_dim, self.num_skills)
        self.stop_head = nn.Linear(context_dim, 1)

        self.skill_embeddings = nn.Embedding(self.num_skills, skill_embedding_dim)
        self.register_buffer(
            "effect_profiles",
            torch.zeros(self.num_skills, effect_profile_dim),
            persistent=True,
        )
        self.effect_projection = nn.Sequential(
            nn.Linear(effect_profile_dim, skill_embedding_dim),
            nn.GELU(),
            nn.Linear(skill_embedding_dim, skill_embedding_dim),
        )
        # Per pair: two identities, two effects, image context, two presence
        # probabilities, six per-guard statistics, overlap and cosine.
        relation_input_dim = (
            4 * skill_embedding_dim + context_dim + 2 + 6 + 2
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(relation_input_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Linear(context_dim // 2, len(RELATION_CLASSES)),
        )

    @torch.no_grad()
    def set_effect_profiles(self, profiles: torch.Tensor) -> None:
        if tuple(profiles.shape) != tuple(self.effect_profiles.shape):
            raise ValueError(
                f"expected effect profiles {tuple(self.effect_profiles.shape)}, "
                f"got {tuple(profiles.shape)}"
            )
        self.effect_profiles.copy_(
            profiles.to(
                device=self.effect_profiles.device,
                dtype=self.effect_profiles.dtype,
            )
        )

    @staticmethod
    def _resize_add(base: torch.Tensor, addition: torch.Tensor) -> torch.Tensor:
        return base + F.interpolate(
            addition,
            size=base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _fpn(
        self,
        encoder_features: Sequence[torch.Tensor],
        trace: TraceFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(encoder_features) != 4:
            raise ValueError("planner expects four encoder feature tensors")
        e1, e2, e3, e4 = [
            projection(feature)
            for projection, feature in zip(self.encoder_projects, encoder_features)
        ]
        t2, t3, t4 = [
            projection(feature)
            for projection, feature in zip(
                self.trace_projects,
                (trace.half, trace.quarter, trace.eighth),
            )
        ]
        p4 = self.smooth4(e4 + t4)
        p3 = self.smooth3(self._resize_add(e3 + t3, p4))
        p2 = self.smooth2(self._resize_add(e2 + t2, p3))
        p1 = self.smooth1(self._resize_add(e1, p2))
        return p1, p2, p3, p4

    @staticmethod
    def _pool(feature: torch.Tensor) -> torch.Tensor:
        return feature.mean(dim=(-2, -1))

    def _relation_logits(
        self,
        context: torch.Tensor,
        presence: torch.Tensor,
        guards: torch.Tensor,
        effect_profiles: torch.Tensor,
    ) -> torch.Tensor:
        batch = context.shape[0]
        embeddings = self.skill_embeddings.weight
        effects = self.effect_projection(effect_profiles)
        means = guards.mean(dim=(-2, -1))
        maxima = guards.amax(dim=(-2, -1))
        deviations = guards.std(dim=(-2, -1), unbiased=False)
        flattened = guards.flatten(2)

        outputs: list[torch.Tensor] = []
        for first, second in PAIR_INDICES:
            first_embedding = embeddings[first].expand(batch, -1)
            second_embedding = embeddings[second].expand(batch, -1)
            first_effect = effects[first].expand(batch, -1)
            second_effect = effects[second].expand(batch, -1)
            overlap = (
                guards[:, first] * guards[:, second]
            ).mean(dim=(-2, -1), keepdim=False)[:, None]
            cosine = F.cosine_similarity(
                flattened[:, first],
                flattened[:, second],
                dim=1,
                eps=1e-8,
            )[:, None]
            statistics = torch.stack(
                (
                    means[:, first],
                    maxima[:, first],
                    deviations[:, first],
                    means[:, second],
                    maxima[:, second],
                    deviations[:, second],
                ),
                dim=1,
            )
            pair_input = torch.cat(
                (
                    first_embedding,
                    second_embedding,
                    first_effect,
                    second_effect,
                    context,
                    presence[:, (first, second)],
                    statistics,
                    overlap,
                    cosine,
                ),
                dim=1,
            )
            outputs.append(self.relation_mlp(pair_input))
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
        *,
        round_value: float | torch.Tensor,
        effect_profiles: torch.Tensor | None = None,
        compute_relations: bool = True,
    ) -> PlannerOutput:
        trace = self.trace(x0, xt)
        p1, p2, p3, p4 = self._fpn(encoder_features, trace)
        round_embedding = self.round_embedding(
            round_value,
            batch_size=xt.shape[0],
            device=xt.device,
            dtype=xt.dtype,
        )
        pooled = torch.cat(
            tuple(self._pool(feature) for feature in (p1, p2, p3, p4))
            + (round_embedding,),
            dim=1,
        )
        context = self.context_mlp(pooled)

        # H/4 dense guards include shallow F1/F2 evidence by deterministic
        # adaptive pooling, while p3 already carries the deep top-down path.
        guard_feature = p3
        guard_feature = guard_feature + F.adaptive_avg_pool2d(p2, p3.shape[-2:])
        guard_feature = guard_feature + F.adaptive_avg_pool2d(p1, p3.shape[-2:])
        guard_feature = self.guard_fusion(guard_feature)
        guard_logits = self.guard_head(guard_feature)
        guard_logits = guard_logits + self.guard_context_bias(context)[:, :, None, None]
        presence_logits = self.presence_head(context)
        stop_logit = self.stop_head(context)

        profiles = self.effect_profiles if effect_profiles is None else effect_profiles
        if tuple(profiles.shape) != (self.num_skills, self.effect_profile_dim):
            raise ValueError(
                f"effect_profiles must be [{self.num_skills},{self.effect_profile_dim}]"
            )
        profiles = profiles.to(device=context.device, dtype=context.dtype)
        if compute_relations:
            relation_logits = self._relation_logits(
                context,
                presence_logits.sigmoid(),
                guard_logits.sigmoid(),
                profiles,
            )
        else:
            # t>0 feedback updates presence/guards/stop only.  A correctly
            # shaped sentinel keeps tracing APIs stable without evaluating the
            # relation MLP or exposing a second compilation path.
            relation_logits = context.new_zeros(
                context.shape[0], len(PAIR_INDICES), len(RELATION_CLASSES)
            )
        return PlannerOutput(
            guard_logits=guard_logits,
            presence_logits=presence_logits,
            stop_logit=stop_logit,
            relation_logits=relation_logits,
            global_context=context,
        )
