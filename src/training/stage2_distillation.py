"""Inference-only Stage2 skill-effect and Group-A interaction distillation.

This module intentionally contains no optimizer or trainable operation.  It
binds every derived record to one frozen Stage1 EMA snapshot and only consumes
the already-audited ``primary_train``/``primary_val`` recipe manifests.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.episode_dataset import GraphRestoreEpisodeDataset
from src.data.manifests import ALLOWED_GROUP_A, SKILLS, PrimaryRecipe, load_primary_manifest
from src.metrics.agenticir_official import official_psnr_ssim, quantize_uint8_semantics
from src.net.graphrestore import GuardedSkillRestormer, SkillExecutionOutput
from src.training.stage1_engine import STAGE1_EMA_SCOPE, stage1_ema_policy_metadata
from src.utils.git import git_commit
from src.utils.hashing import sha256_file
from src.utils.io import (
    atomic_write_json,
    atomic_write_text,
    fsync_directory,
    iter_jsonl,
    load_json,
    load_yaml,
    utc_now_iso,
)


STAGE2_SCHEMA = "graphrestore-stage2-v1"
RELATION_LABELS = ("i_before_j", "j_before_i", "parallel")
AMBIGUOUS_LABEL = "ambiguous"
DEFAULT_SEED = 2027
DEFAULT_SHARD_SIZE = 16
_FORBIDDEN_PATH_TOKENS = ("mio100", "group_b", "group_c", "exploration")


class Stage2ContractError(RuntimeError):
    """Stage2 would violate a frozen data, checkpoint, or inference invariant."""


def stage2_resume_bindings(paths: "Stage2Paths") -> dict[str, Any]:
    """Bind every resumable shard to config, repositories, and executable code."""

    root = paths.project_root.resolve()
    code_paths = [
        root / "src/training/stage2_distillation.py",
        root / "src/training/stage1_engine.py",
        *sorted((root / "src/data").glob("*.py")),
        *sorted((root / "src/metrics").glob("*.py")),
        *sorted((root / "src/net").glob("*.py")),
    ]
    code_sha256: dict[str, str] = {}
    for path in code_paths:
        if not path.is_file():
            raise Stage2ContractError(f"missing Stage2 resume code binding: {path}")
        code_sha256[str(path.relative_to(root))] = sha256_file(path)
    config_paths = _mapping(paths.config.get("paths"), field="paths")
    resolved_path = _project_path(
        root,
        config_paths.get("resolved_paths"),
        field="paths.resolved_paths",
    )
    return {
        "config_sha256": sha256_file(paths.config_path),
        "resolved_paths_sha256": sha256_file(resolved_path),
        "code_sha256": code_sha256,
        "repositories": {
            "agenticir_commit": git_commit(str(paths.resolved["agenticir_repo"])),
            "mioir_commit": git_commit(str(paths.resolved["mioir_repo"])),
        },
    }


@dataclass(frozen=True)
class ProgramScore:
    psnr: float
    ssim: float
    residual_norm: float


@dataclass(frozen=True)
class LabelDecision:
    label: str
    relation_class_index: int | None
    relation_weight: float
    best_serial: str
    serial_gap_psnr: float
    serial_gap_ssim: float
    parallel_minus_best_serial_psnr: float
    parallel_minus_best_serial_ssim: float


@dataclass(frozen=True)
class FrozenStage1Snapshot:
    model: GuardedSkillRestormer
    checkpoint_sha256: str
    checkpoint_step: int


@dataclass(frozen=True)
class Stage2Paths:
    project_root: Path
    config_path: Path
    config: Mapping[str, Any]
    resolved: Mapping[str, Any]
    training_data_root: Path
    primary_train: Path
    primary_val: Path
    checkpoint: Path
    effect_profiles: Path
    interaction_train_manifest: Path
    interaction_val_manifest: Path
    relation_train: Path
    relation_val: Path
    pair_prior: Path
    global_priority: Path
    decision: Path
    summary_csv: Path
    report: Path


class Stage2Model(Protocol):
    training: bool

    def __call__(
        self,
        x: Tensor,
        *,
        active_mask: Tensor,
        guards: Tensor,
        forced_presence_mask: Tensor | None = None,
        return_trace: bool = False,
    ) -> Tensor | SkillExecutionOutput: ...


def _project_path(project_root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Stage2ContractError(f"{field} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage2ContractError(f"{field} must be a mapping")
    return value


def resolve_stage2_paths(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    smoke_root: str | Path | None = None,
) -> Stage2Paths:
    """Resolve and validate the locked Stage2 contract without opening images."""

    config_file = Path(config_path).resolve()
    root = Path(project_root or config_file.parents[1]).resolve()
    config = _mapping(load_yaml(config_file), field="Stage2 config")
    if config.get("protocol_id") != "graphrestore-v7.1-agenticir-locked":
        raise Stage2ContractError("Stage2 protocol_id drifted")
    if config.get("stage") != "stage2" or int(config.get("seed", -1)) != DEFAULT_SEED:
        raise Stage2ContractError("Stage2 name/seed drifted")
    paths = _mapping(config.get("paths"), field="paths")
    data = _mapping(config.get("data"), field="data")
    executor = _mapping(config.get("executor"), field="executor")
    if list(data.get("allowed_groups", [])) != ["single", "A"]:
        raise Stage2ContractError("Stage2 may consume only single and Group A")
    if set(data.get("forbidden_groups", [])) != {"B", "C"}:
        raise Stage2ContractError("Stage2 Group B/C hard guard is missing")
    if data.get("allow_mio100_exploration") is not False or data.get("allow_mio100_formal") is not False:
        raise Stage2ContractError("Stage2 must not read any MiO100 image split")
    if executor.get("checkpoint_snapshot") != "frozen_stage1_ema":
        raise Stage2ContractError("Stage2 executor must be the frozen Stage1 EMA")
    if executor.get("inference_mode") is not True or executor.get("amp_dtype") != "bf16":
        raise Stage2ContractError("Stage2 must use inference_mode plus BF16 autocast")
    if tuple(executor.get("programs", ())) != RELATION_LABELS:
        raise Stage2ContractError("Stage2 must enumerate exactly the three locked programs")

    resolved_path = _project_path(root, paths.get("resolved_paths"), field="paths.resolved_paths")
    resolved = _mapping(load_yaml(resolved_path), field="resolved paths")
    train_key = paths.get("primary_train_manifest_key")
    val_key = paths.get("primary_val_manifest_key")
    if train_key != "primary_train_manifest" or val_key != "primary_val_manifest":
        raise Stage2ContractError("Stage2 primary manifest keys drifted")

    output_root = Path(smoke_root).resolve() if smoke_root is not None else root
    output_paths = dict(paths)
    if smoke_root is not None:
        # A --limit smoke run is physically unable to overwrite formal outputs.
        output_paths.update(
            {
                "effect_profiles": "artifacts/interaction_labels/skill_effect_profiles.json",
                "interaction_train_manifest": "artifacts/interaction_labels/interaction_train_manifest.jsonl",
                "interaction_val_manifest": "artifacts/interaction_labels/interaction_val_manifest.jsonl",
                "relation_train": "artifacts/interaction_labels/group_a_relations_train.jsonl",
                "relation_val": "artifacts/interaction_labels/group_a_relations_val.jsonl",
                "pair_prior": "artifacts/interaction_labels/pair_prior.json",
                "global_priority": "artifacts/interaction_labels/global_priority.json",
                "decision": "artifacts/interaction_labels/stage2_decision.json",
                "summary_csv": "artifacts/metrics/stage2_interaction_summary.csv",
                "report": "reports/INTERACTION_DISTILLATION.md",
            }
        )

    def output(name: str) -> Path:
        return _project_path(output_root, output_paths.get(name), field=f"paths.{name}")

    training_root = _project_path(root, resolved.get("training_data_root"), field="training_data_root")
    train = _project_path(root, resolved.get(str(train_key)), field=str(train_key))
    val = _project_path(root, resolved.get(str(val_key)), field=str(val_key))
    for candidate in (train, val):
        lowered = str(candidate).lower()
        if any(token in lowered for token in _FORBIDDEN_PATH_TOKENS):
            raise Stage2ContractError(f"forbidden Stage2 source path: {candidate}")
        if not candidate.is_file():
            raise Stage2ContractError(f"missing Stage2 primary manifest: {candidate}")

    checkpoint = _project_path(root, paths.get("parent_checkpoint"), field="paths.parent_checkpoint")
    if checkpoint.name != "best_ema.pth":
        raise Stage2ContractError("Stage2 checkpoint must be Stage1 best_ema.pth")
    return Stage2Paths(
        project_root=root,
        config_path=config_file,
        config=config,
        resolved=resolved,
        training_data_root=training_root,
        primary_train=train,
        primary_val=val,
        checkpoint=checkpoint,
        effect_profiles=output("effect_profiles"),
        interaction_train_manifest=output("interaction_train_manifest"),
        interaction_val_manifest=output("interaction_val_manifest"),
        relation_train=output("relation_train"),
        relation_val=output("relation_val"),
        pair_prior=output("pair_prior"),
        global_priority=output("global_priority"),
        decision=output("decision"),
        summary_csv=output("summary_csv"),
        report=output("report"),
    )


def pair_id_from_recipe(recipe: PrimaryRecipe) -> str:
    if not recipe.is_pair or recipe.group != "A":
        raise Stage2ContractError(f"{recipe.sample_id}: interaction recipe is not Group A")
    if recipe.operator_order not in ALLOWED_GROUP_A:
        raise Stage2ContractError(f"{recipe.sample_id}: unapproved Group-A pair")
    first, second = canonical_skill_pair(recipe.skill_ids)
    return f"{SKILLS[first]}+{SKILLS[second]}"


def canonical_skill_pair(skill_ids: Sequence[int]) -> tuple[int, int]:
    """Use ProgramPlanner.PAIR_INDICES orientation (ascending normative ID)."""

    if len(skill_ids) != 2:
        raise Stage2ContractError("a relation pair must contain exactly two skills")
    first, second = sorted(int(value) for value in skill_ids)
    if first == second or first < 0 or second >= len(SKILLS):
        raise Stage2ContractError(f"invalid relation skill pair: {tuple(skill_ids)}")
    return first, second


def _seeded_rank(sample_id: str, pair_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"stage2:{seed}:{pair_id}:{sample_id}".encode("utf-8")).digest()


def select_recipes(
    recipes: Sequence[PrimaryRecipe],
    *,
    split: str,
    group: str,
    per_bucket_max: int,
    seed: int = DEFAULT_SEED,
) -> tuple[PrimaryRecipe, ...]:
    """Seeded deterministic selection followed by stable sample-id ordering."""

    if per_bucket_max <= 0:
        raise ValueError("per_bucket_max must be positive")
    buckets: dict[str, list[PrimaryRecipe]] = defaultdict(list)
    for recipe in recipes:
        if recipe.split != split:
            raise Stage2ContractError(f"{recipe.sample_id}: split drifted")
        if group == "A":
            if recipe.group != "A":
                continue
            bucket = pair_id_from_recipe(recipe)
        elif group == "single":
            if recipe.group != "single" or len(recipe.skill_names) != 1:
                continue
            bucket = recipe.skill_names[0]
        else:
            raise ValueError("group must be 'single' or 'A'")
        buckets[bucket].append(recipe)

    expected = (
        {
            "+".join(
                SKILLS[index]
                for index in canonical_skill_pair(
                    tuple(SKILLS.index(name) for name in _normalise_pair(pair))
                )
            )
            for pair in ALLOWED_GROUP_A
        }
        if group == "A"
        else set(SKILLS)
    )
    if set(buckets) != expected:
        raise Stage2ContractError(
            f"{split} {group} bucket mismatch: missing={sorted(expected-set(buckets))}, "
            f"unexpected={sorted(set(buckets)-expected)}"
        )
    selected: list[PrimaryRecipe] = []
    for bucket, values in sorted(buckets.items()):
        ranked = sorted(values, key=lambda item: (_seeded_rank(item.sample_id, bucket, seed), item.sample_id))
        selected.extend(ranked[:per_bucket_max])
    return tuple(sorted(selected, key=lambda item: (pair_id_from_recipe(item) if group == "A" else item.skill_names[0], item.sample_id)))


def _normalise_pair(pair: Sequence[str]) -> tuple[str, str]:
    aliases = {
        "motion blur": "motion_blur",
        "defocus blur": "defocus_blur",
        "jpeg compression artifact": "jpeg_artifact",
        "dark": "low_light",
        "low resolution": "low_resolution",
    }
    return tuple(aliases.get(value, value) for value in pair)  # type: ignore[return-value]


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"


def write_recipe_manifest(path: str | Path, recipes: Sequence[PrimaryRecipe]) -> str:
    payload = "".join(_json_line(dict(recipe.raw)) for recipe in recipes)
    atomic_write_text(path, payload)
    return sha256_file(path)


def build_interaction_manifests(
    paths: Stage2Paths,
    *,
    train_per_pair_max: int,
    val_per_pair_max: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    train_all = load_primary_manifest(paths.primary_train, paths.training_data_root, expected_split="train")
    val_all = load_primary_manifest(paths.primary_val, paths.training_data_root, expected_split="val")
    train = select_recipes(train_all, split="train", group="A", per_bucket_max=train_per_pair_max, seed=seed)
    val = select_recipes(val_all, split="val", group="A", per_bucket_max=val_per_pair_max, seed=seed)
    train_clean = {recipe.clean_id for recipe in train}
    val_clean = {recipe.clean_id for recipe in val}
    overlap = train_clean & val_clean
    if overlap:
        raise Stage2ContractError(f"interaction train/val clean IDs overlap: {sorted(overlap)[:8]}")
    train_sha = write_recipe_manifest(paths.interaction_train_manifest, train)
    val_sha = write_recipe_manifest(paths.interaction_val_manifest, val)
    return {
        "train": train,
        "val": val,
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "selection_seed": seed,
        "train_clean_count": len(train_clean),
        "val_clean_count": len(val_clean),
        "clean_overlap_count": 0,
    }


def assign_relation_label(
    i_before_j: ProgramScore,
    j_before_i: ProgramScore,
    parallel: ProgramScore,
    *,
    psnr_parallel_tolerance: float = 0.05,
    ssim_parallel_tolerance: float = 0.001,
    serial_psnr_threshold: float = 0.05,
    serial_ssim_threshold: float = 0.002,
) -> LabelDecision:
    """Apply the inclusive V7.1 label thresholds exactly once."""

    scores = (i_before_j, j_before_i, parallel)
    if not all(math.isfinite(value) for score in scores for value in asdict(score).values()):
        raise Stage2ContractError("non-finite program score")
    if i_before_j.psnr >= j_before_i.psnr:
        best_name, best = "i_before_j", i_before_j
    else:
        best_name, best = "j_before_i", j_before_i
    gap_psnr = abs(i_before_j.psnr - j_before_i.psnr)
    gap_ssim = abs(i_before_j.ssim - j_before_i.ssim)
    if (
        parallel.psnr >= best.psnr - psnr_parallel_tolerance
        and parallel.ssim >= best.ssim - ssim_parallel_tolerance
    ):
        label = "parallel"
    elif gap_psnr >= serial_psnr_threshold:
        label = "i_before_j" if i_before_j.psnr > j_before_i.psnr else "j_before_i"
    elif gap_ssim >= serial_ssim_threshold:
        label = "i_before_j" if i_before_j.ssim > j_before_i.ssim else "j_before_i"
    else:
        label = AMBIGUOUS_LABEL
    return LabelDecision(
        label=label,
        relation_class_index=RELATION_LABELS.index(label) if label in RELATION_LABELS else None,
        relation_weight=0.25 if label == AMBIGUOUS_LABEL else 1.0,
        best_serial=best_name,
        serial_gap_psnr=gap_psnr,
        serial_gap_ssim=gap_ssim,
        parallel_minus_best_serial_psnr=parallel.psnr - best.psnr,
        parallel_minus_best_serial_ssim=parallel.ssim - best.ssim,
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def summarize_relations(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize relation labels while excluding ambiguous rows everywhere required."""

    total = len(records)
    labels = [str(row["label"]) for row in records]
    unknown = sorted(set(labels) - set((*RELATION_LABELS, AMBIGUOUS_LABEL)))
    if unknown:
        raise Stage2ContractError(f"unknown relation labels: {unknown}")
    nonambiguous = [label for label in labels if label != AMBIGUOUS_LABEL]
    counts = Counter(nonambiguous)
    gaps = [float(row["margins"]["serial_gap_psnr"]) for row in records]
    if any(not math.isfinite(value) for value in gaps):
        raise Stage2ContractError("non-finite serial gap")
    n_ambiguous = labels.count(AMBIGUOUS_LABEL)
    n_nonambiguous = len(nonambiguous)
    majority_label: str | None = None
    majority_share: float | None = None
    if n_nonambiguous:
        maximum = max(counts.get(label, 0) for label in RELATION_LABELS)
        winners = [label for label in RELATION_LABELS if counts.get(label, 0) == maximum]
        majority_label = winners[0] if len(winners) == 1 else "tie:" + "|".join(winners)
        majority_share = maximum / n_nonambiguous
    return {
        "n_total": total,
        "n_ambiguous": n_ambiguous,
        "n_nonambiguous": n_nonambiguous,
        "ambiguous_fraction": n_ambiguous / total if total else None,
        # JSON null is the legal on-disk representation of undefined NaN semantics.
        "parallel_fraction_nonambiguous": counts.get("parallel", 0) / n_nonambiguous if n_nonambiguous else None,
        "label_counts_nonambiguous": {label: counts.get(label, 0) for label in RELATION_LABELS},
        "majority_label": majority_label,
        "majority_label_share": majority_share,
        "serial_gap_psnr_mean": float(np.mean(gaps)) if gaps else None,
        "serial_gap_psnr_median": _percentile(gaps, 50),
        "serial_gap_psnr_p25": _percentile(gaps, 25),
        "serial_gap_psnr_p75": _percentile(gaps, 75),
        "serial_gap_psnr_p90": _percentile(gaps, 90),
        "serial_gap_psnr_max": max(gaps) if gaps else None,
        "serial_gap_psnr_fraction_gte_0_02": sum(value >= 0.02 for value in gaps) / len(gaps) if gaps else None,
        "serial_gap_psnr_fraction_gte_0_05": sum(value >= 0.05 for value in gaps) / len(gaps) if gaps else None,
        "serial_gap_psnr_fraction_gte_0_10": sum(value >= 0.10 for value in gaps) / len(gaps) if gaps else None,
        "ambiguous_in_pair_prior": 0,
        "ambiguous_in_majority_label_share": 0,
        "ambiguous_in_parallel_fraction_nonambiguous": 0,
    }


def summarize_split(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[str(row["pair_id"])].append(row)
    per_pair = {pair: summarize_relations(rows) for pair, rows in sorted(buckets.items())}
    overall = summarize_relations(records)
    shares = [float(value["majority_label_share"]) for value in per_pair.values() if value["majority_label_share"] is not None]
    overall["median_majority_label_share"] = _percentile(shares, 50)
    return {"overall": overall, "per_pair": per_pair}


def build_pair_prior(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build Group-A priors from non-ambiguous train labels only."""

    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    ambiguous_excluded = 0
    for row in records:
        pair_id = str(row["pair_id"])
        buckets.setdefault(pair_id, Counter())
        label = str(row["label"])
        if label == AMBIGUOUS_LABEL:
            ambiguous_excluded += 1
            continue
        if label not in RELATION_LABELS:
            raise Stage2ContractError(f"unknown relation label {label!r}")
        buckets[pair_id][label] += 1
    priors: dict[str, Any] = {}
    for pair, counts in sorted(buckets.items()):
        total = sum(counts.values())
        priors[pair] = {
            "n_nonambiguous": total,
            "counts": {label: counts.get(label, 0) for label in RELATION_LABELS},
            "probabilities": {
                label: counts.get(label, 0) / total if total else None
                for label in RELATION_LABELS
            },
            "ambiguous_in_prior": 0,
        }
    compiler_prior = {
        pair: dict(value["probabilities"])
        for pair, value in priors.items()
        if int(value["n_nonambiguous"]) > 0
    }
    return {
        "schema_version": STAGE2_SCHEMA,
        "source": "interaction_train_nonambiguous_only",
        "relation_classes": list(RELATION_LABELS),
        "ambiguous_excluded": ambiguous_excluded,
        "pair_prior": compiler_prior,
        "pairs": priors,
    }


def fit_bradley_terry(
    records: Sequence[Mapping[str, Any]],
    *,
    skills: Sequence[str] = SKILLS,
    l2: float = 1.0e-4,
    max_iterations: int = 100,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Fit data-only Bradley--Terry scores to non-parallel directed labels."""

    index = {name: ordinal for ordinal, name in enumerate(skills)}
    observations: Counter[tuple[int, int, int]] = Counter()
    ambiguous_excluded = parallel_excluded = 0
    for row in records:
        label = str(row["label"])
        if label == AMBIGUOUS_LABEL:
            ambiguous_excluded += 1
            continue
        if label == "parallel":
            parallel_excluded += 1
            continue
        pair_skills = row.get("skills")
        if not isinstance(pair_skills, Sequence) or isinstance(pair_skills, (str, bytes)) or len(pair_skills) != 2:
            raise Stage2ContractError("relation record lacks two ordered skills")
        i, j = index[str(pair_skills[0])], index[str(pair_skills[1])]
        y = 1 if label == "i_before_j" else 0
        observations[(i, j, y)] += 1

    scores = np.zeros(len(skills), dtype=np.float64)
    converged = False
    for iteration in range(max_iterations):
        gradient = l2 * scores
        hessian = np.eye(len(skills), dtype=np.float64) * l2
        for (i, j, y), count in observations.items():
            difference = float(np.clip(scores[i] - scores[j], -40.0, 40.0))
            probability = 1.0 / (1.0 + math.exp(-difference))
            residual = count * (probability - y)
            curvature = count * probability * (1.0 - probability)
            gradient[i] += residual
            gradient[j] -= residual
            hessian[i, i] += curvature
            hessian[j, j] += curvature
            hessian[i, j] -= curvature
            hessian[j, i] -= curvature
        step = np.linalg.solve(hessian, gradient)
        maximum = float(np.max(np.abs(step)))
        if maximum > 1.0:
            step /= maximum
        scores -= step
        scores -= scores.mean()
        if maximum < tolerance:
            converged = True
            break
    priority = {name: float(scores[ordinal]) for ordinal, name in enumerate(skills)}
    # Tied scores receive the same rank; no hard-coded name order creates evidence.
    unique_scores = sorted(set(priority.values()), reverse=True)
    ranks = {name: unique_scores.index(value) + 1 for name, value in priority.items()}
    return {
        "schema_version": STAGE2_SCHEMA,
        "model": "bradley_terry_logistic",
        "source": "interaction_train_nonparallel_nonambiguous_only",
        "priority": priority,
        "rank": ranks,
        "n_directional_observations": int(sum(observations.values())),
        "n_ambiguous_excluded": ambiguous_excluded,
        "n_parallel_excluded": parallel_excluded,
        "l2": l2,
        "iterations": iteration + 1,
        "converged": converged,
        "manual_skill_order_used": False,
    }


def _strict_state_mapping(value: object, *, field: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise Stage2ContractError(f"{field} must be a non-empty tensor mapping")
    if any(not isinstance(key, str) or not torch.is_tensor(tensor) for key, tensor in value.items()):
        raise Stage2ContractError(f"{field} contains non-tensor state")
    return value  # type: ignore[return-value]


def load_frozen_stage1_ema(
    checkpoint: str | Path,
    *,
    device: torch.device,
    model_factory: Callable[[], GuardedSkillRestormer] = GuardedSkillRestormer,
) -> FrozenStage1Snapshot:
    """Strictly load one immutable Stage1 best-EMA snapshot.

    Best checkpoints are required to expose EMA weights under ``model``.  The
    embedded EMA shadow is compared tensor-for-tensor so an accidentally saved
    raw-training model cannot silently be used for distillation.
    """

    path = Path(checkpoint).resolve()
    if path.name != "best_ema.pth" or not path.is_file():
        raise Stage2ContractError(f"missing Stage1 best_ema.pth: {path}")
    checkpoint_sha = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "graphrestore-checkpoint-v1":
        raise Stage2ContractError("Stage1 checkpoint schema is not graphrestore-checkpoint-v1")
    stage = str(payload.get("stage", "")).lower().replace("-", "_")
    if stage != "stage1":
        raise Stage2ContractError(f"Stage2 parent checkpoint is not Stage1: {stage!r}")
    if payload.get("model_role") != "ema_selection" or payload.get("resumable") is not False:
        raise Stage2ContractError(
            "Stage2 parent must be a non-resumable Stage1 EMA selection checkpoint"
        )
    model_state = _strict_state_mapping(payload.get("model"), field="checkpoint.model")
    ema = _mapping(payload.get("ema"), field="checkpoint.ema")
    if ema.get("scope") != STAGE1_EMA_SCOPE:
        raise Stage2ContractError(
            "Stage1 best EMA phase-aware scope is missing or invalid"
        )
    decay = ema.get("decay")
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or float(decay) != 0.9999
    ):
        raise Stage2ContractError("Stage1 best EMA decay is missing or invalid")
    expected_policy = stage1_ema_policy_metadata(0.9999)
    if ema.get("policy") != expected_policy:
        raise Stage2ContractError(
            "Stage1 best EMA phase-aware policy is missing or invalid"
        )
    provenance = _mapping(payload.get("provenance"), field="checkpoint.provenance")
    if provenance.get("ema_policy") != expected_policy:
        raise Stage2ContractError(
            "Stage1 best provenance EMA policy is missing or invalid"
        )
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Stage2ContractError("Stage1 checkpoint lacks a valid step")
    if ema.get("num_updates") != step:
        raise Stage2ContractError(
            "Stage1 best EMA update count does not match checkpoint step"
        )
    shadow = _strict_state_mapping(ema.get("shadow"), field="checkpoint.ema.shadow")
    if set(model_state) != set(shadow):
        raise Stage2ContractError("best EMA model/EMA shadow keys differ")
    for name in model_state:
        if model_state[name].shape != shadow[name].shape or model_state[name].dtype != shadow[name].dtype:
            raise Stage2ContractError(f"best EMA state metadata differs at {name}")
        if not torch.equal(model_state[name], shadow[name]):
            raise Stage2ContractError(
                f"best_ema.pth exposes non-EMA model weights at {name}"
            )

    model = model_factory()
    target_state = model.state_dict()
    if model_state.keys() != target_state.keys():
        raise Stage2ContractError("Stage1 checkpoint model keys differ from executor")
    for name, tensor in model_state.items():
        target = target_state[name]
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise Stage2ContractError(
                f"Stage1 checkpoint model metadata differs from executor at {name}"
            )
    incompatible = model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Stage2ContractError("Stage1 checkpoint did not load strictly")
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise Stage2ContractError("Stage2 executor was not fully frozen")
    if sha256_file(path) != checkpoint_sha:
        raise Stage2ContractError("Stage1 checkpoint changed while Stage2 was loading it")
    return FrozenStage1Snapshot(model=model, checkpoint_sha256=checkpoint_sha, checkpoint_step=step)


def stage2_autocast(device: torch.device):
    """Locked BF16 autocast; CPU is supported only for bounded smoke tests."""

    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if not torch.cuda.is_available():
            raise Stage2ContractError("formal Stage2 requires CUDA; use --device cpu only for smoke tests")
        return torch.device("cuda")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Stage2ContractError("CUDA was requested but is unavailable")
    if device.type not in {"cuda", "cpu"}:
        raise Stage2ContractError("Stage2 supports only CUDA or bounded CPU smoke execution")
    return device


def _ensure_batch(tensor: Tensor, *, channels: int) -> Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != channels:
        raise Stage2ContractError(f"expected BCHW tensor with {channels} channels")
    return tensor


def _active_mask(batch: int, skill_ids: Sequence[int], device: torch.device) -> Tensor:
    mask = torch.zeros((batch, len(SKILLS)), dtype=torch.bool, device=device)
    for skill_id in skill_ids:
        if not 0 <= int(skill_id) < len(SKILLS):
            raise Stage2ContractError(f"invalid skill id {skill_id}")
        mask[:, int(skill_id)] = True
    return mask


def _model_image(value: Tensor | SkillExecutionOutput) -> Tensor:
    if isinstance(value, SkillExecutionOutput):
        return value.final
    if not torch.is_tensor(value):
        raise Stage2ContractError("Stage2 executor returned a non-tensor")
    return value


def _one_level(
    model: Stage2Model,
    current: Tensor,
    guards: Tensor,
    skill_ids: Sequence[int],
    *,
    forced: bool = False,
) -> Tensor:
    active = _active_mask(current.shape[0], () if forced else skill_ids, current.device)
    forced_mask = _active_mask(current.shape[0], skill_ids, current.device) if forced else None
    output = model(
        current,
        active_mask=active,
        guards=guards,
        forced_presence_mask=forced_mask,
        return_trace=False,
    )
    return _model_image(output)


def _score_output(output: Tensor, starting: Tensor, target: Tensor) -> ProgramScore:
    quantized_output = quantize_uint8_semantics(output.float())
    quantized_target = quantize_uint8_semantics(target.float())
    metric = official_psnr_ssim(quantized_output, quantized_target, quantize=False)
    residual = (output.float().clamp(0.0, 1.0) - starting.float()).square().mean(dim=(1, 2, 3)).sqrt()
    if output.shape[0] != 1:
        raise Stage2ContractError("record emission currently requires batch size one")
    return ProgramScore(
        psnr=float(metric.psnr[0].detach().cpu().item()),
        ssim=float(metric.ssim[0].detach().cpu().item()),
        residual_norm=float(residual[0].detach().cpu().item()),
    )


def enumerate_three_programs(
    model: Stage2Model,
    x_both: Tensor,
    target: Tensor,
    guards: Tensor,
    *,
    skill_i: int,
    skill_j: int,
    device: torch.device,
) -> dict[str, ProgramScore]:
    """Enumerate serial/serial/parallel from exactly the same starting tensor."""

    if skill_i == skill_j:
        raise Stage2ContractError("interaction pair contains the same skill twice")
    x = _ensure_batch(x_both, channels=3).to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    gt = _ensure_batch(target, channels=3).to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    guard_tensor = _ensure_batch(guards, channels=len(SKILLS)).to(
        device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
    )
    if x.shape != gt.shape:
        raise Stage2ContractError("Stage2 starting image and GT shapes differ")
    if model.training:
        raise Stage2ContractError("Stage2 model must be in eval mode")

    with torch.inference_mode(), stage2_autocast(device):
        first_i = _one_level(model, x, guard_tensor, (skill_i,))
        final_ij = _one_level(model, first_i, guard_tensor, (skill_j,))
        first_j = _one_level(model, x, guard_tensor, (skill_j,))
        final_ji = _one_level(model, first_j, guard_tensor, (skill_i,))
        final_parallel = _one_level(model, x, guard_tensor, (skill_i, skill_j))
    return {
        "i_before_j": _score_output(final_ij, x, gt),
        "j_before_i": _score_output(final_ji, x, gt),
        "parallel": _score_output(final_parallel, x, gt),
    }


class AtomicJsonlShardWriter:
    """Durable, bounded JSONL shards with a signature-checked resume path."""

    def __init__(
        self,
        final_path: str | Path,
        *,
        signature: Mapping[str, Any],
        shard_size: int = DEFAULT_SHARD_SIZE,
        resume: bool = True,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.final_path = Path(final_path)
        self.parts_dir = self.final_path.parent / f".{self.final_path.name}.parts"
        self.state_path = self.parts_dir / "state.json"
        self.signature = dict(signature)
        self.signature_sha256 = hashlib.sha256(
            json.dumps(self.signature, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.shard_size = int(shard_size)
        self.buffer: list[Mapping[str, Any]] = []
        self.processed: set[str] = set()
        self.next_part = 0
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        existing_parts = sorted(self.parts_dir.glob("part-*.jsonl"))
        if self.final_path.exists() and not self.state_path.exists() and not existing_parts:
            raise Stage2ContractError(
                f"existing Stage2 output has no resumable signature evidence: {self.final_path}"
            )
        if self.state_path.exists() or existing_parts:
            if not resume:
                raise Stage2ContractError(
                    f"partial Stage2 output exists and --no-resume was selected: {self.parts_dir}"
                )
            state = _mapping(load_json(self.state_path), field="Stage2 shard state")
            if state.get("signature_sha256") != self.signature_sha256:
                raise Stage2ContractError(
                    f"Stage2 shard signature mismatch; refusing stale resume: {self.parts_dir}"
                )
            for part in existing_parts:
                for _, row in iter_jsonl(part):
                    sample_id = str(row.get("sample_id", ""))
                    if not sample_id or sample_id in self.processed:
                        raise Stage2ContractError(f"duplicate/invalid resumed sample_id in {part}")
                    self.processed.add(sample_id)
            self.next_part = len(existing_parts)
        else:
            self._write_state()

    def _write_state(self) -> None:
        atomic_write_json(
            self.state_path,
            {
                "schema_version": STAGE2_SCHEMA,
                "signature": self.signature,
                "signature_sha256": self.signature_sha256,
                "completed_records": len(self.processed),
                "completed_parts": self.next_part,
                "updated_utc": utc_now_iso(),
            },
        )

    def append(self, record: Mapping[str, Any]) -> None:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise Stage2ContractError("sharded record lacks sample_id")
        if sample_id in self.processed or any(str(row["sample_id"]) == sample_id for row in self.buffer):
            raise Stage2ContractError(f"duplicate Stage2 sample_id: {sample_id}")
        self.buffer.append(dict(record))
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.parts_dir / f"part-{self.next_part:06d}.jsonl"
        if path.exists():
            raise Stage2ContractError(f"refusing to overwrite Stage2 shard: {path}")
        atomic_write_text(path, "".join(_json_line(row) for row in self.buffer))
        self.processed.update(str(row["sample_id"]) for row in self.buffer)
        self.buffer.clear()
        self.next_part += 1
        self._write_state()

    def consolidate(self, *, expected_ids: Sequence[str]) -> str:
        self.flush()
        expected = tuple(expected_ids)
        if len(expected) != len(set(expected)) or set(expected) != self.processed:
            missing = sorted(set(expected) - self.processed)
            unexpected = sorted(self.processed - set(expected))
            raise Stage2ContractError(
                f"Stage2 shard coverage mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        rows: dict[str, Mapping[str, Any]] = {}
        for part in sorted(self.parts_dir.glob("part-*.jsonl")):
            for _, row in iter_jsonl(part):
                rows[str(row["sample_id"])] = row
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.final_path.name}.", suffix=".tmp", dir=self.final_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                for sample_id in expected:
                    handle.write(_json_line(rows[sample_id]))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.final_path)
            fsync_directory(self.final_path.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        output_sha = sha256_file(self.final_path)
        atomic_write_json(
            self.parts_dir / "complete.json",
            {
                "schema_version": STAGE2_SCHEMA,
                "signature_sha256": self.signature_sha256,
                "record_count": len(expected),
                "output_path": str(self.final_path.resolve(strict=False)),
                "output_sha256": output_sha,
                "completed_utc": utc_now_iso(),
            },
        )
        return output_sha


def release_stage2_gpu(model: object | None = None) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _sample_value(sample: Mapping[str, Any], key: str) -> Any:
    if key not in sample:
        raise Stage2ContractError(f"Stage2 sample lacks {key!r}")
    value = sample[key]
    if torch.is_tensor(value) and value.numel() == 1:
        return value.item()
    return value


def _prefetched_samples(
    dataset: GraphRestoreEpisodeDataset,
    indices: Sequence[int],
    *,
    num_workers: int,
) -> Iterable[Mapping[str, Any]]:
    """Prefetch official CPU synthesis while retaining batch-size-one numerics."""

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not indices:
        return ()
    loader = DataLoader(
        dataset,
        batch_size=None,
        sampler=list(indices),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader


def interaction_record(
    sample: Mapping[str, Any],
    scores: Mapping[str, ProgramScore],
    *,
    split: str,
    checkpoint_sha256: str,
    checkpoint_step: int,
    manifest_sha256: str,
) -> dict[str, Any]:
    if tuple(scores) != RELATION_LABELS:
        raise Stage2ContractError("three-program score ordering drifted")
    skill_ids_raw = _sample_value(sample, "present_skill_ids")
    if not torch.is_tensor(skill_ids_raw) or skill_ids_raw.numel() != 2:
        raise Stage2ContractError("interaction sample lacks two present skill IDs")
    skill_ids = list(canonical_skill_pair([int(value) for value in skill_ids_raw.flatten().tolist()]))
    if any(value < 0 for value in skill_ids):
        raise Stage2ContractError("interaction sample is not a pair")
    skills = [SKILLS[value] for value in skill_ids]
    pair_id = "+".join(skills)
    decision = assign_relation_label(
        scores["i_before_j"], scores["j_before_i"], scores["parallel"]
    )
    crop_box = _sample_value(sample, "crop_box")
    augmentation = _sample_value(sample, "augmentation")
    return {
        "schema_version": STAGE2_SCHEMA,
        "split": split,
        "sample_id": str(_sample_value(sample, "sample_id")),
        "clean_id": str(_sample_value(sample, "clean_id")) if "clean_id" in sample else None,
        "pair_id": pair_id,
        "skills": skills,
        "skill_ids": skill_ids,
        "programs": {
            name: {
                **asdict(scores[name]),
                "sequence": skills if name == "i_before_j" else (
                    list(reversed(skills)) if name == "j_before_i" else [skills]
                ),
                "skill_calls": 2 if name != "parallel" else 1,
            }
            for name in RELATION_LABELS
        },
        "margins": {
            "serial_gap_psnr": decision.serial_gap_psnr,
            "serial_gap_ssim": decision.serial_gap_ssim,
            "parallel_minus_best_serial_psnr": decision.parallel_minus_best_serial_psnr,
            "parallel_minus_best_serial_ssim": decision.parallel_minus_best_serial_ssim,
        },
        "best_serial": decision.best_serial,
        "label": decision.label,
        "relation_class_index": decision.relation_class_index,
        "relation_weight": decision.relation_weight,
        "stage1_checkpoint_sha256": checkpoint_sha256,
        "stage1_checkpoint_step": checkpoint_step,
        "interaction_manifest_sha256": manifest_sha256,
        "metric_protocol": "agenticir_official_parity_crop_clamp_round_uint8",
        "same_starting_x_both": True,
        "same_recipe_canonicalization_crop_pad": True,
        "pair_orientation": "ProgramPlanner.PAIR_INDICES_ascending_normative_skill_id",
        "crop_box": crop_box.tolist() if torch.is_tensor(crop_box) else crop_box,
        "augmentation": augmentation.tolist() if torch.is_tensor(augmentation) else augmentation,
        "contains_low_resolution": bool(_sample_value(sample, "contains_low_resolution")),
    }


def run_interaction_split(
    *,
    model: Stage2Model,
    manifest_path: str | Path,
    manifest_sha256: str,
    training_data_root: str | Path,
    depth_compat_root: str | Path,
    split: str,
    checkpoint_sha256: str,
    checkpoint_step: int,
    output_path: str | Path,
    device: torch.device,
    seed: int = DEFAULT_SEED,
    shard_size: int = DEFAULT_SHARD_SIZE,
    resume: bool = True,
    agenticir_repo: str | Path | None = None,
    mioir_repo: str | Path | None = None,
    num_workers: int = 8,
    resume_bindings: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run a selected explicit interaction manifest one sample at a time."""

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if not isinstance(resume_bindings, Mapping) or not resume_bindings:
        raise Stage2ContractError("interaction shards require code/config resume bindings")
    kwargs: dict[str, Any] = {}
    if agenticir_repo is not None:
        kwargs["agenticir_repo"] = agenticir_repo
    if mioir_repo is not None:
        kwargs["mioir_repo"] = mioir_repo
    dataset = GraphRestoreEpisodeDataset(
        manifest_path,
        training_data_root,
        depth_compat_root,
        192 if split == "train" else None,
        split == "train",
        "stage2",
        seed,
        **kwargs,
    )
    ordered_ids = [recipe.sample_id for recipe in dataset.records]
    # Derived manifests are expected to preserve the stable writer order.
    ordered_pairs = [(pair_id_from_recipe(recipe), recipe.sample_id) for recipe in dataset.records]
    if ordered_pairs != sorted(ordered_pairs):
        raise Stage2ContractError("interaction manifest is not stably ordered")
    writer = AtomicJsonlShardWriter(
        output_path,
        signature={
            "kind": f"interaction_{split}",
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "seed": seed,
            "metric_protocol": "agenticir_official_parity_crop_clamp_round_uint8",
            "resume_bindings": dict(resume_bindings),
        },
        shard_size=shard_size,
        resume=resume,
    )
    pending_indices = [
        index
        for index, recipe in enumerate(dataset.records)
        if recipe.sample_id not in writer.processed
    ]
    for sample in _prefetched_samples(dataset, pending_indices, num_workers=num_workers):
        index = int(_sample_value(sample, "sample_index"))
        recipe = dataset.records[index]
        # Dataset intentionally returns no clean_id metadata; bind it from the
        # same parsed recipe rather than scanning any alternate source.
        sample = dict(sample)
        sample["clean_id"] = recipe.clean_id
        present = sample["present_skill_ids"]
        if not torch.is_tensor(present):
            raise Stage2ContractError("present_skill_ids is not a tensor")
        ids = list(canonical_skill_pair([int(value) for value in present.tolist()]))
        scores = enumerate_three_programs(
            model,
            sample["x_both"],
            sample["gt_clean"],
            sample["guard_targets"],
            skill_i=ids[0],
            skill_j=ids[1],
            device=device,
        )
        writer.append(
            interaction_record(
                sample,
                scores,
                split=split,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_step=checkpoint_step,
                manifest_sha256=manifest_sha256,
            )
        )
    output_sha = writer.consolidate(expected_ids=ordered_ids)
    records = [row for _, row in iter_jsonl(output_path)]
    return records, output_sha


def train_val_consistency(
    train_summary: Mapping[str, Any], val_summary: Mapping[str, Any]
) -> dict[str, Any]:
    train_pairs = _mapping(train_summary.get("per_pair"), field="train per_pair")
    val_pairs = _mapping(val_summary.get("per_pair"), field="val per_pair")
    result: dict[str, Any] = {}
    for pair in sorted(set(train_pairs) | set(val_pairs)):
        train_label = _mapping(train_pairs.get(pair, {}), field=f"train {pair}").get("majority_label")
        val_label = _mapping(val_pairs.get(pair, {}), field=f"val {pair}").get("majority_label")
        result[pair] = {
            "train_majority_label": train_label,
            "val_majority_label": val_label,
            "consistent": train_label == val_label if train_label is not None and val_label is not None else None,
        }
    return result


def decision_warnings(
    train_summary: Mapping[str, Any], val_summary: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    warnings: set[str] = set()
    details: dict[str, Any] = {}
    for split, summary in (("train", train_summary), ("val", val_summary)):
        overall = _mapping(summary.get("overall"), field=f"{split}.overall")
        per_pair = _mapping(summary.get("per_pair"), field=f"{split}.per_pair")
        split_codes: list[str] = []
        nonamb_fraction = (
            float(overall["n_nonambiguous"]) / float(overall["n_total"])
            if int(overall["n_total"]) else 0.0
        )
        if nonamb_fraction < 0.30 or any(int(_mapping(value, field="pair summary")["n_nonambiguous"]) < 64 for value in per_pair.values()):
            split_codes.append("WARNING_LOW_LABEL_SUPPORT")
        median = overall.get("serial_gap_psnr_median")
        p75 = overall.get("serial_gap_psnr_p75")
        if median is not None and p75 is not None and float(median) < 0.02 and float(p75) < 0.05:
            split_codes.append("WARNING_ORDER_SIGNAL_WEAK")
        parallel = overall.get("parallel_fraction_nonambiguous")
        if parallel is not None and float(parallel) < 0.05:
            split_codes.append("WARNING_COLLAPSE_TO_TOTAL_ORDER")
        if parallel is not None and float(parallel) > 0.95:
            split_codes.append("WARNING_COLLAPSE_TO_PARALLEL_FUSION")
        info: dict[str, list[str]] = {
            "INFO_CONTEXT_DEPENDENT_RELATION": [],
            "INFO_STABLE_PAIR_RULE": [],
        }
        for pair, value in per_pair.items():
            share = _mapping(value, field="pair summary").get("majority_label_share")
            if share is None:
                continue
            if 0.45 <= float(share) < 0.70:
                info["INFO_CONTEXT_DEPENDENT_RELATION"].append(pair)
            elif float(share) >= 0.70:
                info["INFO_STABLE_PAIR_RULE"].append(pair)
        split_codes.extend(code for code, pairs in info.items() if pairs)
        warnings.update(split_codes)
        details[split] = {
            "codes": split_codes,
            "info_pairs": info,
            "warnings_trigger_automatic_model_changes": False,
        }
    ordered = [
        code
        for code in (
            "WARNING_LOW_LABEL_SUPPORT",
            "WARNING_ORDER_SIGNAL_WEAK",
            "WARNING_COLLAPSE_TO_TOTAL_ORDER",
            "WARNING_COLLAPSE_TO_PARALLEL_FUSION",
            "INFO_CONTEXT_DEPENDENT_RELATION",
            "INFO_STABLE_PAIR_RULE",
        )
        if code in warnings
    ]
    return ordered, details


def _recommended_interpretation(warnings: Sequence[str]) -> str:
    severe = [value for value in warnings if value.startswith("WARNING_")]
    if severe:
        return (
            "Stage2 evidence requires user review before Stage3; warnings are descriptive "
            "and did not alter thresholds, skills, labels, or model structure."
        )
    return (
        "Stage2 interaction evidence passed the configured descriptive warning checks; "
        "explicit user approval is still mandatory before Stage3."
    )


def _csv_rows(split: str, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [{"split": split, "scope": "overall", "pair_id": "__overall__", **dict(_mapping(summary["overall"], field="overall"))}]
    for pair, value in _mapping(summary["per_pair"], field="per_pair").items():
        rows.append({"split": split, "scope": "pair", "pair_id": pair, **dict(_mapping(value, field="pair"))})
    return rows


def write_summary_csv(
    path: str | Path,
    train: Mapping[str, Any],
    val: Mapping[str, Any],
    consistency: Mapping[str, Any],
) -> None:
    rows = _csv_rows("train", train) + _csv_rows("val", val)
    for row in rows:
        relation = consistency.get(str(row["pair_id"]))
        if isinstance(relation, Mapping):
            row.update(
                {
                    "train_majority_label": relation.get("train_majority_label"),
                    "val_majority_label": relation.get("val_majority_label"),
                    "train_val_majority_consistent": relation.get("consistent"),
                }
            )
    columns = (
        "split", "scope", "pair_id", "n_total", "n_ambiguous", "n_nonambiguous",
        "ambiguous_fraction", "parallel_fraction_nonambiguous", "majority_label",
        "majority_label_share", "serial_gap_psnr_mean", "serial_gap_psnr_median",
        "serial_gap_psnr_p25", "serial_gap_psnr_p75", "serial_gap_psnr_p90",
        "serial_gap_psnr_max", "serial_gap_psnr_fraction_gte_0_02",
        "serial_gap_psnr_fraction_gte_0_05", "serial_gap_psnr_fraction_gte_0_10",
        "label_counts_nonambiguous", "train_majority_label", "val_majority_label",
        "train_val_majority_consistent",
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                serialised = dict(row)
                if serialised.get("parallel_fraction_nonambiguous") is None:
                    serialised["parallel_fraction_nonambiguous"] = "NaN"
                serialised["label_counts_nonambiguous"] = json.dumps(
                    serialised.get("label_counts_nonambiguous", {}), sort_keys=True, separators=(",", ":")
                )
                writer.writerow(serialised)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_interaction_report(
    path: str | Path,
    *,
    checkpoint_sha256: str,
    train_manifest_sha256: str,
    val_manifest_sha256: str,
    train_summary: Mapping[str, Any],
    val_summary: Mapping[str, Any],
    consistency: Mapping[str, Any],
    warnings: Sequence[str],
) -> None:
    lines = [
        "# Stage2 Interaction Distillation",
        "",
        f"- Stage1 EMA SHA256: `{checkpoint_sha256}`",
        f"- interaction_train manifest SHA256: `{train_manifest_sha256}`",
        f"- interaction_val manifest SHA256: `{val_manifest_sha256}`",
        "- Executor: `torch.inference_mode()` + BF16 autocast; no optimizer",
        "- Data exposure: primary single/Group-A recipes only; MiO100 and Group B/C were not read or generated",
        "- Ambiguous policy: retained with weight 0.25 for serial-mass partial-label supervision; excluded from priors and descriptive non-ambiguous metrics",
        "",
        "## Decision summaries",
        "",
        "```json",
        json.dumps({"train": train_summary, "val": val_summary, "train_val_consistency": consistency}, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        "```",
        "",
        "## Warnings / information",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Warnings are descriptive only. No threshold, label, skill set, or architecture was changed automatically.",
            "Stage3 remains NOT STARTED pending explicit user approval.",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def build_stage2_decision(
    *,
    paths: Stage2Paths,
    checkpoint_sha256: str,
    train_manifest_sha256: str,
    val_manifest_sha256: str,
    relation_train_sha256: str,
    relation_val_sha256: str,
    train_summary: Mapping[str, Any],
    val_summary: Mapping[str, Any],
    pair_prior_sha256: str,
    global_priority_sha256: str,
) -> dict[str, Any]:
    consistency = train_val_consistency(train_summary, val_summary)
    warnings, warning_details = decision_warnings(train_summary, val_summary)
    # The unbiased interaction_val audit is the one compact decision view.
    val_overall = dict(_mapping(val_summary["overall"], field="val overall"))
    decision = {
        "schema_version": STAGE2_SCHEMA,
        "protocol_id": "graphrestore-v7.1-agenticir-locked",
        "created_utc": utc_now_iso(),
        "stage1_checkpoint_sha256": checkpoint_sha256,
        "interaction_train_manifest_sha256": train_manifest_sha256,
        "interaction_val_manifest_sha256": val_manifest_sha256,
        "relation_train_sha256": relation_train_sha256,
        "relation_val_sha256": relation_val_sha256,
        "pair_prior_sha256": pair_prior_sha256,
        "global_priority_sha256": global_priority_sha256,
        "config_sha256": sha256_file(paths.config_path),
        "decision_view": "interaction_val",
        "overall": {
            "ambiguous_fraction": val_overall["ambiguous_fraction"],
            "parallel_fraction_nonambiguous": val_overall["parallel_fraction_nonambiguous"],
            "serial_gap_psnr_median": val_overall["serial_gap_psnr_median"],
            "serial_gap_psnr_p75": val_overall["serial_gap_psnr_p75"],
            "median_majority_label_share": val_overall["median_majority_label_share"],
        },
        "splits": {"train": train_summary, "val": val_summary},
        "train_val_majority_consistency": consistency,
        "warnings": warnings,
        "warning_details": warning_details,
        "recommended_interpretation": _recommended_interpretation(warnings),
        "ambiguous_policy": {
            "label": AMBIGUOUS_LABEL,
            "relation_head_classes": list(RELATION_LABELS),
            "weight": 0.25,
            "supervision": "stable_log_softmax_logsumexp_serial_mass",
            "excluded_from_pair_prior_majority_parallel_fraction": True,
        },
        "automatic_model_changes": False,
        "approved": False,
        "stage3_started": False,
        "gpu_release_required": True,
    }
    return decision


def _gradient_magnitude(image: Tensor) -> Tensor:
    gray = (image * image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    sobel_x = image.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    dx = F.conv2d(gray, sobel_x, padding=1)
    dy = F.conv2d(gray, sobel_y, padding=1)
    return (dx.square() + dy.square() + 1.0e-12).sqrt()


def compute_effect_measurements(
    starting: Tensor,
    output: Tensor,
    target: Tensor,
    source_guard: Tensor,
    *,
    source_is_dense: bool,
) -> dict[str, float]:
    """Compute the five locked profile fields with explicit sign conventions."""

    x = _ensure_batch(starting.float(), channels=3)
    restored = _ensure_batch(output.float(), channels=3).clamp(0.0, 1.0)
    gt = _ensure_batch(target.float(), channels=3)
    guard = _ensure_batch(source_guard.float(), channels=1)
    guard = F.interpolate(guard, size=x.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    before_metric = official_psnr_ssim(
        quantize_uint8_semantics(x), quantize_uint8_semantics(gt), quantize=False
    )
    after_metric = official_psnr_ssim(
        quantize_uint8_semantics(restored), quantize_uint8_semantics(gt), quantize=False
    )
    support_denominator = guard.sum(dim=(1, 2, 3)).clamp_min(1.0e-8) * 3.0
    before_source = ((x - gt).abs() * guard).sum(dim=(1, 2, 3)) / support_denominator
    after_source = ((restored - gt).abs() * guard).sum(dim=(1, 2, 3)) / support_denominator
    structure_weight = 1.0 - guard if source_is_dense else torch.ones_like(guard)
    before_structure = (_gradient_magnitude(x) - _gradient_magnitude(gt)).abs()
    after_structure = (_gradient_magnitude(restored) - _gradient_magnitude(gt)).abs()
    structure_denominator = structure_weight.sum(dim=(1, 2, 3)).clamp_min(1.0e-8)
    before_structure_error = (before_structure * structure_weight).sum(dim=(1, 2, 3)) / structure_denominator
    after_structure_error = (after_structure * structure_weight).sum(dim=(1, 2, 3)) / structure_denominator
    residual = (restored - x).square().mean(dim=(1, 2, 3)).sqrt()
    if x.shape[0] != 1:
        raise Stage2ContractError("effect record emission requires batch size one")
    return {
        "delta_psnr": float((after_metric.psnr - before_metric.psnr)[0].detach().cpu().item()),
        "delta_ssim": float((after_metric.ssim - before_metric.ssim)[0].detach().cpu().item()),
        "output_residual_norm": float(residual[0].detach().cpu().item()),
        # Positive means the source-supported RGB error decreased.
        "source_severity_change": float((before_source - after_source)[0].detach().cpu().item()),
        # Positive means non-target structural error increased (collateral damage).
        "non_target_structure_error_change": float(
            (after_structure_error - before_structure_error)[0].detach().cpu().item()
        ),
    }


def effect_record_for_skill(
    model: Stage2Model,
    sample: Mapping[str, Any],
    *,
    source_skill_id: int,
    forced_skill_id: int,
    device: torch.device,
    checkpoint_sha256: str,
    checkpoint_step: int,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    x = _ensure_batch(sample["x_both"], channels=3).to(device=device, dtype=torch.float32)
    gt = _ensure_batch(sample["gt_clean"], channels=3).to(device=device, dtype=torch.float32)
    original_guards = _ensure_batch(sample["guard_targets"], channels=len(SKILLS)).to(
        device=device, dtype=torch.float32
    )
    execution_guards = original_guards.clone()
    if forced_skill_id != source_skill_id:
        # There is no ground-truth absent-skill support.  A full-support unit
        # guard makes the intervention real and prevents a fake zero-guard
        # identity profile; this policy is persisted in every record/profile.
        execution_guards[:, forced_skill_id].fill_(1.0)
        guard_policy = "counterfactual_absent_skill_full_support_unit_guard"
    else:
        guard_policy = "present_source_continuous_gt_guard"
    with torch.inference_mode(), stage2_autocast(device):
        output = _one_level(
            model,
            x,
            execution_guards,
            (forced_skill_id,),
            forced=True,
        )
    source_guard = original_guards[:, source_skill_id : source_skill_id + 1]
    fields = compute_effect_measurements(
        x,
        output,
        gt,
        source_guard,
        source_is_dense=SKILLS[source_skill_id] in {"rain", "haze", "low_light"},
    )
    sample_id = str(sample["sample_id"])
    return {
        "schema_version": STAGE2_SCHEMA,
        "sample_id": f"{sample_id}::forced::{SKILLS[forced_skill_id]}",
        "source_sample_id": sample_id,
        "source_skill": SKILLS[source_skill_id],
        "source_skill_id": source_skill_id,
        "forced_skill": SKILLS[forced_skill_id],
        "forced_skill_id": forced_skill_id,
        "effect": fields,
        "guard_policy": guard_policy,
        "forced_presence_override": True,
        "absent_zero_guard_used": False,
        "stage1_checkpoint_sha256": checkpoint_sha256,
        "stage1_checkpoint_step": checkpoint_step,
        "source_manifest_sha256": source_manifest_sha256,
        "metric_protocol": "agenticir_official_parity_crop_clamp_round_uint8",
    }


EFFECT_FIELDS = (
    "delta_psnr",
    "delta_ssim",
    "output_residual_norm",
    "source_severity_change",
    "non_target_structure_error_change",
)


def aggregate_effect_profiles(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
    source_manifest_sha256: str,
    selection_manifest_sha256: str,
    selection_seed: int,
) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        source = str(row["source_skill"])
        forced = str(row["forced_skill"])
        if source not in SKILLS or forced not in SKILLS:
            raise Stage2ContractError("effect record contains unknown skill")
        effect = _mapping(row.get("effect"), field="effect")
        if set(effect) != set(EFFECT_FIELDS):
            raise Stage2ContractError("effect field schema drifted")
        if any(not math.isfinite(float(effect[field])) for field in EFFECT_FIELDS):
            raise Stage2ContractError("effect field is non-finite")
        if bool(row.get("absent_zero_guard_used")):
            raise Stage2ContractError("zero absent-skill guard leaked into effect profiles")
        buckets[(source, forced)].append(effect)
    expected = {(source, forced) for source in SKILLS for forced in SKILLS}
    if set(buckets) != expected:
        raise Stage2ContractError(
            f"incomplete effect profile grid: missing={sorted(expected-set(buckets))[:8]}"
        )
    matrix: dict[str, dict[str, Any]] = {}
    for source in SKILLS:
        matrix[source] = {}
        for forced in SKILLS:
            values = buckets[(source, forced)]
            matrix[source][forced] = {
                "n": len(values),
                "mean": {
                    field: float(np.mean([float(value[field]) for value in values]))
                    for field in EFFECT_FIELDS
                },
                "median": {
                    field: float(np.median([float(value[field]) for value in values]))
                    for field in EFFECT_FIELDS
                },
            }
    vectors: dict[str, list[float]] = {}
    vector_layout = [f"source={source}/{field}" for source in SKILLS for field in EFFECT_FIELDS]
    for forced in SKILLS:
        vectors[forced] = [
            float(matrix[source][forced]["mean"][field])
            for source in SKILLS
            for field in EFFECT_FIELDS
        ]
        if len(vectors[forced]) != 40:
            raise Stage2ContractError("effect vector must have 8x5=40 entries")
    return {
        "schema_version": STAGE2_SCHEMA,
        "kind": "single_degradation_skill_effect_profiles",
        "created_utc": utc_now_iso(),
        "stage1_checkpoint_sha256": checkpoint_sha256,
        "source_primary_val_manifest_sha256": source_manifest_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "selection_seed": selection_seed,
        "source_skills": list(SKILLS),
        "forced_skills": list(SKILLS),
        "effect_fields": list(EFFECT_FIELDS),
        "sign_conventions": {
            "delta_psnr": "after_minus_before_higher_is_better",
            "delta_ssim": "after_minus_before_higher_is_better",
            "output_residual_norm": "rms_clamped_output_minus_input",
            "source_severity_change": "guard_weighted_rgb_error_before_minus_after_higher_is_better",
            "non_target_structure_error_change": "sobel_error_after_minus_before_lower_is_better",
        },
        "guard_policy": {
            "present_source": "continuous_ground_truth_guard",
            "absent_counterfactual": "full_support_unit_guard_plus_forced_presence_override",
            "rationale": "an absent zero guard would create a false identity effect",
            "absent_zero_guard_used": False,
        },
        "matrix": matrix,
        "vector_layout": vector_layout,
        "effect_vectors": vectors,
        "effect_vector_dim": 40,
        "record_count": len(records),
        "optimizer_created": False,
        "inference_mode": True,
        "amp_dtype": "bf16",
        "mio100_rows_read": 0,
        "group_b_or_c_rows_generated": 0,
    }


def build_single_effect_manifest(
    paths: Stage2Paths,
    *,
    per_source_max: int,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, tuple[PrimaryRecipe, ...], str]:
    all_val = load_primary_manifest(paths.primary_val, paths.training_data_root, expected_split="val")
    selected = select_recipes(
        all_val, split="val", group="single", per_bucket_max=per_source_max, seed=seed
    )
    destination = paths.effect_profiles.parent / "single_val_effect_manifest.jsonl"
    sha = write_recipe_manifest(destination, selected)
    return destination, selected, sha


def run_effect_profiles(
    *,
    model: Stage2Model,
    paths: Stage2Paths,
    checkpoint_sha256: str,
    checkpoint_step: int,
    device: torch.device,
    per_source_max: int,
    seed: int = DEFAULT_SEED,
    shard_size: int = DEFAULT_SHARD_SIZE,
    resume: bool = True,
    num_workers: int = 8,
) -> dict[str, Any]:
    selection_path, selected, selection_sha = build_single_effect_manifest(
        paths, per_source_max=per_source_max, seed=seed
    )
    dataset = GraphRestoreEpisodeDataset(
        selection_path,
        paths.training_data_root,
        paths.project_root / "artifacts/cache/depth_compat",
        None,
        False,
        "stage2",
        seed,
        agenticir_repo=paths.resolved["agenticir_repo"],
        mioir_repo=paths.resolved["mioir_repo"],
    )
    if [recipe.sample_id for recipe in dataset.records] != [recipe.sample_id for recipe in selected]:
        raise Stage2ContractError("single-effect selection manifest order drifted")
    records_path = paths.effect_profiles.parent / "skill_effect_records.jsonl"
    primary_val_sha = sha256_file(paths.primary_val)
    writer = AtomicJsonlShardWriter(
        records_path,
        signature={
            "kind": "skill_effect_profiles",
            "selection_manifest_sha256": selection_sha,
            "checkpoint_sha256": checkpoint_sha256,
            "seed": seed,
            "guard_policy": "gt_if_present_else_unit_full_support",
            "resume_bindings": stage2_resume_bindings(paths),
        },
        shard_size=shard_size,
        resume=resume,
    )
    expected_ids = [
        f"{recipe.sample_id}::forced::{skill}"
        for recipe in dataset.records
        for skill in SKILLS
    ]
    pending_indices = [
        index
        for index, recipe in enumerate(dataset.records)
        if any(
            f"{recipe.sample_id}::forced::{skill}" not in writer.processed
            for skill in SKILLS
        )
    ]
    for sample in _prefetched_samples(dataset, pending_indices, num_workers=num_workers):
        index = int(_sample_value(sample, "sample_index"))
        recipe = dataset.records[index]
        pending = [skill for skill in SKILLS if f"{recipe.sample_id}::forced::{skill}" not in writer.processed]
        source_skill_id = recipe.skill_ids[0]
        for forced_skill in pending:
            forced_skill_id = SKILLS.index(forced_skill)
            writer.append(
                effect_record_for_skill(
                    model,
                    sample,
                    source_skill_id=source_skill_id,
                    forced_skill_id=forced_skill_id,
                    device=device,
                    checkpoint_sha256=checkpoint_sha256,
                    checkpoint_step=checkpoint_step,
                    source_manifest_sha256=primary_val_sha,
                )
            )
    writer.consolidate(expected_ids=expected_ids)
    records = [row for _, row in iter_jsonl(records_path)]
    profile = aggregate_effect_profiles(
        records,
        checkpoint_sha256=checkpoint_sha256,
        source_manifest_sha256=primary_val_sha,
        selection_manifest_sha256=selection_sha,
        selection_seed=seed,
    )
    atomic_write_json(paths.effect_profiles, profile)
    return profile


def finalize_interaction_outputs(
    *,
    paths: Stage2Paths,
    checkpoint_sha256: str,
    train_manifest_sha256: str,
    val_manifest_sha256: str,
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    relation_train_sha256: str,
    relation_val_sha256: str,
) -> dict[str, Any]:
    effect_profiles = _mapping(load_json(paths.effect_profiles), field="effect profiles")
    if effect_profiles.get("stage1_checkpoint_sha256") != checkpoint_sha256:
        raise Stage2ContractError("effect profiles and interaction labels use different Stage1 snapshots")
    train_summary = summarize_split(train_records)
    val_summary = summarize_split(val_records)
    prior = build_pair_prior(train_records)
    prior.update(
        {
            "created_utc": utc_now_iso(),
            "stage1_checkpoint_sha256": checkpoint_sha256,
            "relation_train_sha256": relation_train_sha256,
        }
    )
    atomic_write_json(paths.pair_prior, prior)
    priority = fit_bradley_terry(train_records)
    priority.update(
        {
            "created_utc": utc_now_iso(),
            "stage1_checkpoint_sha256": checkpoint_sha256,
            "relation_train_sha256": relation_train_sha256,
        }
    )
    atomic_write_json(paths.global_priority, priority)
    prior_sha = sha256_file(paths.pair_prior)
    priority_sha = sha256_file(paths.global_priority)
    decision = build_stage2_decision(
        paths=paths,
        checkpoint_sha256=checkpoint_sha256,
        train_manifest_sha256=train_manifest_sha256,
        val_manifest_sha256=val_manifest_sha256,
        relation_train_sha256=relation_train_sha256,
        relation_val_sha256=relation_val_sha256,
        train_summary=train_summary,
        val_summary=val_summary,
        pair_prior_sha256=prior_sha,
        global_priority_sha256=priority_sha,
    )
    atomic_write_json(paths.decision, decision)
    write_summary_csv(
        paths.summary_csv,
        train_summary,
        val_summary,
        decision["train_val_majority_consistency"],
    )
    write_interaction_report(
        paths.report,
        checkpoint_sha256=checkpoint_sha256,
        train_manifest_sha256=train_manifest_sha256,
        val_manifest_sha256=val_manifest_sha256,
        train_summary=train_summary,
        val_summary=val_summary,
        consistency=decision["train_val_majority_consistency"],
        warnings=decision["warnings"],
    )
    return decision
