#!/usr/bin/env python3
"""Fail-closed V7.1 identity, split, and data-boundary audit.

The MiO100 portion reads JSONL metadata and performs path ``stat`` checks only.
It never opens, decodes, or hashes MiO100 image content.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (  # noqa: E402
    AuditTrail,
    atomic_write_json,
    atomic_write_text,
    canonical_git_remote,
    ensure_within,
    git_commit,
    git_remote_url,
    git_status_porcelain,
    is_sha256,
    iter_jsonl,
    load_json,
    load_resolved_paths,
    resolve_config_path,
    sha256_file,
)

PROTOCOL = "graphrestore-v7.1-agenticir-locked"

EXPECTED_AGENTICIR_COMMIT = "9640a291480dee3ba8f2974125d4ee9e3440f3d6"
EXPECTED_MIOIR_COMMIT = "4d5f6ca0235cf2c307319673242d5722ee35d73f"
EXPECTED_AGENTICIR_REMOTE = "https://github.com/Kaiwen-Zhu/AgenticIR.git"
EXPECTED_MIOIR_REMOTE = "https://github.com/Xiangtaokong/MiOIR.git"
EXPECTED_STAGE_A_PARENT_SHA256 = (
    "66e056ff3537ea99416aeb119173e90fbcafc9e9f809db169ef7381cc93f77b8"
)

CONTRACT_MANIFEST_SHA256 = {
    "clean_train": "00247444a3b7304fe83a4783cae694181e6796253c6915d2491009def03df257",
    "clean_val": "88276445c7cc1166ace77904276dbeb61f3a049572e3b23fd1aad2b5f831947d",
    "primary_train": "83da30d0b8445d5bb427c336b125214ee62f2a0ec3a5bab61ca7119703044071",
    "primary_val": "af89bb22896a3744eab5e4b6414f5ee1b19770ce11e372e27b798afd9583a21b",
    "primary_all": "f4080efc2572ce2377646a8acabcbebe092e4a3feeabafc4200984b716c8e8eb",
    "mio100_test_1440": "5a53c28ad93d49a70d3632bfbff008a78309543bb6710921ab2a01b9bdb10950",
}

LOCKED_AGENTICIR_FILE_SHA256 = {
    "add_single_degradation": "c97450a05acb805e59291a1335a743c77eca3db36f26a444b4033c7f6fe6369c",
    "degradations_txt": "1a9bae77190579efe9ec17e8f31e09810cb2361b862c33ce4be25a5e3a04d54d",
    "scorer": "b6eee989575ee17d2cbf9e38fbab0a996b54a5260ae205246c718c08facab830",
    "compute_scores": "ce1a35f9f110a67c4581885f631dae6c283e438bcaf2749199fb9d19fa440548",
    "compare_methods": "a246b8656744649ed5adfd5f482491f89006ef7bec1ce9923b5971a1da3d856a",
    "requirements": "3e76d9e7c658ce7df907dc39ea7af8aa36aa2d5fcf5bd6ec91d34c109a9b45e2",
}

CLEAN_SCHEMA = frozenset(
    {
        "clean_id",
        "clean_path",
        "clean_sha256",
        "depth_dtype",
        "depth_height",
        "depth_path",
        "depth_width",
        "height",
        "source",
        "split",
        "split_seed",
        "width",
    }
)
PRIMARY_SCHEMA = frozenset(
    {
        "canonical_resize",
        "clean_id",
        "clean_path",
        "clean_sha256",
        "degradations",
        "depth_path",
        "group",
        "lq_model_path",
        "lq_native_path",
        "native_scale",
        "operator_order",
        "operator_params",
        "sample_id",
        "seed",
        "source",
        "split",
    }
)
MIO100_SCHEMA = frozenset(
    {
        "canonical_lq_path",
        "clean_id",
        "degradations",
        "depth_path",
        "group",
        "gt_path",
        "native_lq_path",
        "sample_id",
        "scale_factor",
        "source",
        "split",
    }
)

SINGLE_TASKS = (
    ("rain",),
    ("haze",),
    ("motion blur",),
    ("low resolution",),
    ("dark",),
    ("noise",),
    ("defocus blur",),
    ("jpeg compression artifact",),
)
GROUP_A_TASKS = (
    ("rain", "haze"),
    ("motion blur", "low resolution"),
    ("dark", "noise"),
    ("defocus blur", "jpeg compression artifact"),
    ("noise", "jpeg compression artifact"),
    ("rain", "low resolution"),
    ("motion blur", "dark"),
    ("defocus blur", "haze"),
)
ALLOWED_TASKS = frozenset(SINGLE_TASKS + GROUP_A_TASKS)

FORBIDDEN_DATA_MARKERS = (
    "rar/",
    "pir_tar",
    "sidd",
    "gopro",
    "lol",
    "reside",
    "rain200l",
    "div2k",
    "flickr2k",
    "mio100_exploration",
)

MANIFEST_SPECS: dict[str, dict[str, Any]] = {
    "clean_train": {
        "path_key": "clean_train_manifest",
        "rows": 3105,
        "schema": CLEAN_SCHEMA,
        "splits": {"train": 3105},
    },
    "clean_val": {
        "path_key": "clean_val_manifest",
        "rows": 345,
        "schema": CLEAN_SCHEMA,
        "splits": {"val": 345},
    },
    "primary_train": {
        "path_key": "primary_train_manifest",
        "rows": 14400,
        "schema": PRIMARY_SCHEMA,
        "splits": {"train": 14400},
    },
    "primary_val": {
        "path_key": "primary_val_manifest",
        "rows": 1600,
        "schema": PRIMARY_SCHEMA,
        "splits": {"val": 1600},
    },
    "primary_all": {
        "path_key": "primary_all_manifest",
        "rows": 16000,
        "schema": PRIMARY_SCHEMA,
        "splits": {"train": 14400, "val": 1600},
    },
    "mio100_test_1440": {
        "path_key": "mio100_test_1440_manifest",
        "rows": 1440,
        "schema": MIO100_SCHEMA,
        "splits": {"test": 1440},
    },
}

MIO100_SUBSET_SPECS: dict[str, tuple[str, int, str]] = {
    "mio100_group_a_test": ("mio100_group_a_test_manifest", 640, "A"),
    "mio100_group_b_test": ("mio100_group_b_test_manifest", 400, "B"),
    "mio100_group_c_test": ("mio100_group_c_test_manifest", 400, "C"),
}

AGENTICIR_FILE_KEYS = {
    "add_single_degradation": "agenticir_add_single_degradation",
    "degradations_txt": "agenticir_degradations_txt",
    "scorer": "agenticir_scorer",
    "compute_scores": "agenticir_compute_scores",
    "compare_methods": "agenticir_compare_methods",
    "requirements": "agenticir_requirements",
}


def _format_examples(values: Sequence[str], *, limit: int = 3) -> str:
    if not values:
        return "none"
    shown = list(values[:limit])
    suffix = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
    return "; ".join(shown) + suffix


def _load_rows(path: Path, trail: AuditTrail, name: str) -> list[dict[str, Any]]:
    try:
        rows = [record for _, record in iter_jsonl(path)]
    except Exception as exc:
        trail.record(f"manifest.{name}.jsonl", "FAIL", str(exc))
        return []
    trail.record(
        f"manifest.{name}.jsonl",
        "PASS",
        f"decoded {len(rows)} strict JSON objects",
    )
    return rows


def _audit_repo(
    *,
    name: str,
    repo: Path,
    expected_commit: str,
    expected_remote: str,
    trail: AuditTrail,
) -> None:
    trail.require(
        repo.is_dir(),
        f"repo.{name}.exists",
        f"repository exists: {repo}",
        f"missing repository: {repo}",
    )
    if not repo.is_dir():
        return
    try:
        actual_commit = git_commit(repo)
        actual_remote = git_remote_url(repo)
        tracked_changes = git_status_porcelain(repo)
    except Exception as exc:
        trail.record(f"repo.{name}.identity", "FAIL", str(exc))
        return

    trail.require(
        actual_commit == expected_commit,
        f"repo.{name}.commit",
        f"commit={actual_commit}",
        f"expected {expected_commit}, got {actual_commit}",
    )
    trail.require(
        canonical_git_remote(actual_remote) == canonical_git_remote(expected_remote),
        f"repo.{name}.remote",
        f"remote={actual_remote}",
        f"expected repository {expected_remote}, got {actual_remote}",
    )
    trail.require(
        not tracked_changes,
        f"repo.{name}.tracked_clean",
        "no tracked worktree changes",
        f"tracked worktree changes: {_format_examples(tracked_changes)}",
    )
    trail.facts.setdefault("repositories", {})[name] = {
        "path": str(repo),
        "commit": actual_commit,
        "remote": actual_remote,
        "tracked_changes": list(tracked_changes),
    }


def _audit_agenticir_files(
    config: Mapping[str, Any], expected: Mapping[str, Any], trail: AuditTrail
) -> None:
    repo = Path(config["agenticir_repo"]).resolve()
    expected_files = expected.get("agenticir_files")
    if not isinstance(expected_files, Mapping):
        trail.record(
            "agenticir.files.expected",
            "FAIL",
            "expected_identity.agenticir_files must be a mapping",
        )
        return

    facts: dict[str, Any] = {}
    for identity_key, config_key in AGENTICIR_FILE_KEYS.items():
        path = Path(config[config_key]).resolve(strict=False)
        expected_sha = expected_files.get(identity_key)
        try:
            ensure_within(path, repo)
            inside_repo = True
        except Exception:
            inside_repo = False
        trail.require(
            inside_repo,
            f"agenticir.file.{identity_key}.boundary",
            f"inside locked repo: {path}",
            f"critical file escapes AgenticIR repo: {path}",
        )
        if not path.is_file():
            trail.record(
                f"agenticir.file.{identity_key}.sha256",
                "FAIL",
                f"missing critical file: {path}",
            )
            continue
        actual_sha = sha256_file(path)
        trail.require(
            is_sha256(expected_sha) and actual_sha == expected_sha,
            f"agenticir.file.{identity_key}.sha256",
            f"sha256={actual_sha}",
            f"expected {expected_sha}, got {actual_sha}",
        )
        facts[identity_key] = {"path": str(path), "sha256": actual_sha}
    trail.facts["agenticir_files"] = facts


def _audit_manifest_identity(
    *,
    name: str,
    path: Path,
    expected_sha: object,
    expected_rows: int,
    expected_schema: frozenset[str],
    expected_splits: Mapping[str, int],
    trail: AuditTrail,
) -> list[dict[str, Any]]:
    if not path.is_file():
        trail.record(f"manifest.{name}.exists", "FAIL", f"missing: {path}")
        return []
    trail.record(f"manifest.{name}.exists", "PASS", str(path))
    actual_sha = sha256_file(path)
    trail.require(
        is_sha256(expected_sha) and actual_sha == expected_sha,
        f"manifest.{name}.sha256",
        f"sha256={actual_sha}",
        f"expected {expected_sha}, got {actual_sha}",
    )
    rows = _load_rows(path, trail, name)
    trail.require(
        len(rows) == expected_rows,
        f"manifest.{name}.row_count",
        f"rows={len(rows)}",
        f"expected {expected_rows}, got {len(rows)}",
    )
    schema_counts = Counter(tuple(sorted(record)) for record in rows)
    schema_ok = len(schema_counts) == 1 and frozenset(next(iter(schema_counts), ())) == expected_schema
    schema_detail = (
        f"exact schema with {len(expected_schema)} fields"
        if schema_ok
        else f"observed schemas={dict(schema_counts)}"
    )
    trail.require(
        schema_ok,
        f"manifest.{name}.schema",
        schema_detail,
        schema_detail,
    )
    split_counts = Counter(record.get("split") for record in rows)
    trail.require(
        dict(split_counts) == dict(expected_splits),
        f"manifest.{name}.split",
        f"split_counts={dict(split_counts)}",
        f"expected {dict(expected_splits)}, got {dict(split_counts)}",
    )
    trail.facts.setdefault("manifests", {})[name] = {
        "path": str(path),
        "sha256": actual_sha,
        "rows": len(rows),
        "splits": dict(split_counts),
        "schema": sorted(next(iter(schema_counts), ())),
    }
    return rows


def _resolve_training_reference(value: object, training_root: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid path value: {value!r}")
    candidate = Path(value)
    if candidate.is_absolute():
        return ensure_within(candidate, training_root)
    return ensure_within(training_root / candidate, training_root)


def _audit_clean_rows(
    rows: Sequence[dict[str, Any]],
    *,
    name: str,
    expected_split: str,
    training_root: Path,
    trail: AuditTrail,
) -> dict[str, dict[str, Any]]:
    errors: list[str] = []
    ids: set[str] = set()
    index: dict[str, dict[str, Any]] = {}
    referenced_paths: set[Path] = set()
    for row_index, row in enumerate(rows, start=1):
        clean_id = row.get("clean_id")
        if not isinstance(clean_id, str) or not clean_id:
            errors.append(f"row {row_index}: invalid clean_id")
            continue
        if clean_id in ids:
            errors.append(f"row {row_index}: duplicate clean_id={clean_id}")
        ids.add(clean_id)
        index[clean_id] = row
        if row.get("split") != expected_split:
            errors.append(f"row {row_index}: split={row.get('split')!r}")
        if row.get("split_seed") != 2027:
            errors.append(f"row {row_index}: split_seed={row.get('split_seed')!r}")
        if row.get("source") != "mioir_official_gt_depth":
            errors.append(f"row {row_index}: source={row.get('source')!r}")
        if row.get("depth_dtype") != "single":
            errors.append(f"row {row_index}: depth_dtype={row.get('depth_dtype')!r}")
        if not is_sha256(row.get("clean_sha256")):
            errors.append(f"row {row_index}: invalid clean SHA")
        numeric = ("height", "width", "depth_height", "depth_width")
        if any(not isinstance(row.get(key), int) or row[key] <= 0 for key in numeric):
            errors.append(f"row {row_index}: invalid dimensions")
        elif row["height"] != 4 * row["depth_height"] or row["width"] != 4 * row["depth_width"]:
            errors.append(f"row {row_index}: depth is not exact quarter-size")
        for key in ("clean_path", "depth_path"):
            try:
                resolved = _resolve_training_reference(row.get(key), training_root)
            except Exception as exc:
                errors.append(f"row {row_index}: {key}: {exc}")
                continue
            if resolved is not None:
                referenced_paths.add(resolved)

    missing = [str(path) for path in sorted(referenced_paths) if not path.is_file()]
    if missing:
        errors.append(f"missing referenced files: {_format_examples(missing)}")
    trail.require(
        not errors,
        f"clean.{name}.content",
        f"{len(rows)} rows, {len(ids)} unique IDs, {len(referenced_paths)} existing GT/depth paths",
        _format_examples(errors),
    )
    return index


def _audit_primary_rows(
    rows: Sequence[dict[str, Any]],
    *,
    name: str,
    clean_by_split: Mapping[str, Mapping[str, dict[str, Any]]],
    training_root: Path,
    expected_task_count_by_split: Mapping[str, int],
    trail: AuditTrail,
) -> None:
    errors: list[str] = []
    sample_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    task_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    forbidden_hits: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"row {row_index}: invalid sample_id")
        elif sample_id in sample_ids:
            errors.append(f"row {row_index}: duplicate sample_id={sample_id}")
        else:
            sample_ids.add(sample_id)

        split = row.get("split")
        group = row.get("group")
        group_counts[str(group)] += 1
        degradations_raw = row.get("degradations")
        operator_order_raw = row.get("operator_order")
        degradations = (
            tuple(degradations_raw)
            if isinstance(degradations_raw, list)
            and all(isinstance(item, str) for item in degradations_raw)
            else ()
        )
        operator_order = (
            tuple(operator_order_raw)
            if isinstance(operator_order_raw, list)
            and all(isinstance(item, str) for item in operator_order_raw)
            else ()
        )
        if degradations not in ALLOWED_TASKS:
            errors.append(f"row {row_index}: forbidden task/order={degradations!r}")
        if operator_order != degradations:
            errors.append(f"row {row_index}: operator_order differs from degradations")
        if group == "single" and degradations not in SINGLE_TASKS:
            errors.append(f"row {row_index}: invalid single task={degradations!r}")
        if group == "A" and degradations not in GROUP_A_TASKS:
            errors.append(f"row {row_index}: invalid Group-A task={degradations!r}")
        if group not in {"single", "A"}:
            errors.append(f"row {row_index}: forbidden group={group!r}")
        if isinstance(split, str):
            task_counts[(split, degradations)] += 1
        if row.get("source") != "agenticir_official":
            errors.append(f"row {row_index}: source={row.get('source')!r}")
        if not isinstance(row.get("seed"), int):
            errors.append(f"row {row_index}: non-integer recipe seed")

        clean_index = clean_by_split.get(str(split), {})
        clean_id = row.get("clean_id")
        clean_row = clean_index.get(clean_id) if isinstance(clean_id, str) else None
        if clean_row is None:
            errors.append(f"row {row_index}: clean_id={clean_id!r} not in {split} clean split")
        else:
            for key in ("clean_path", "clean_sha256"):
                if row.get(key) != clean_row.get(key):
                    errors.append(f"row {row_index}: {key} differs from clean manifest")

        params = row.get("operator_params")
        if not isinstance(params, list) or len(params) != len(degradations):
            errors.append(f"row {row_index}: invalid operator_params length")
        else:
            names: list[object] = []
            for parameter in params:
                if not isinstance(parameter, dict) or set(parameter) != {"actual", "name", "seed"}:
                    errors.append(f"row {row_index}: invalid operator parameter schema")
                    continue
                names.append(parameter.get("name"))
                if not isinstance(parameter.get("seed"), int):
                    errors.append(f"row {row_index}: non-integer operator seed")
                if not isinstance(parameter.get("actual"), dict):
                    errors.append(f"row {row_index}: operator actual is not an object")
            if tuple(names) != degradations:
                errors.append(f"row {row_index}: operator parameter names differ")

        has_haze = "haze" in degradations
        if has_haze != (row.get("depth_path") is not None):
            errors.append(f"row {row_index}: haze/depth presence mismatch")
        if clean_row is not None and has_haze and row.get("depth_path") != clean_row.get("depth_path"):
            errors.append(f"row {row_index}: haze depth path differs from clean manifest")

        has_low_resolution = "low resolution" in degradations
        if has_low_resolution:
            if row.get("native_scale") != 0.25:
                errors.append(f"row {row_index}: low-resolution native_scale is not 0.25")
            if row.get("canonical_resize") != "AgenticIR/BasicSR imresize x4":
                errors.append(f"row {row_index}: wrong low-resolution canonicalizer")
        elif row.get("native_scale") is not None or row.get("canonical_resize") is not None:
            errors.append(f"row {row_index}: non-LR row has scale metadata")
        if row.get("lq_model_path") is not None or row.get("lq_native_path") is not None:
            errors.append(f"row {row_index}: primary recipe unexpectedly materializes LQ")

        for field in ("clean_path", "depth_path", "lq_model_path", "lq_native_path"):
            value = row.get(field)
            if isinstance(value, str):
                lowered = value.replace("\\", "/").lower()
                for marker in FORBIDDEN_DATA_MARKERS:
                    if marker in lowered:
                        forbidden_hits.append(f"row {row_index} {field} contains {marker}")
                try:
                    _resolve_training_reference(value, training_root)
                except Exception as exc:
                    errors.append(f"row {row_index}: {field}: {exc}")

    expected_groups: Counter[str] = Counter()
    for split, per_task_count in expected_task_count_by_split.items():
        expected_groups["single"] += len(SINGLE_TASKS) * per_task_count
        expected_groups["A"] += len(GROUP_A_TASKS) * per_task_count
        for task in SINGLE_TASKS + GROUP_A_TASKS:
            actual = task_counts[(split, task)]
            if actual != per_task_count:
                errors.append(
                    f"split={split} task={task!r}: expected {per_task_count}, got {actual}"
                )
    if group_counts != expected_groups:
        errors.append(f"expected groups {dict(expected_groups)}, got {dict(group_counts)}")
    if forbidden_hits:
        errors.extend(forbidden_hits)

    trail.require(
        not errors,
        f"primary.{name}.boundary",
        (
            f"{len(rows)} recipes; groups={dict(group_counts)}; only 8 single + "
            "8 ordered Group-A tasks; no forbidden source"
        ),
        _format_examples(errors),
    )
    trail.facts.setdefault("primary", {})[name] = {
        "rows": len(rows),
        "unique_sample_ids": len(sample_ids),
        "groups": dict(group_counts),
        "task_counts": {
            f"{split}:{' + '.join(task)}": count
            for (split, task), count in sorted(task_counts.items())
        },
        "forbidden_reference_count": len(forbidden_hits),
    }


def _path_stat_only(path: Path, cache: dict[Path, bool]) -> bool:
    """Return file existence without opening or decoding it."""

    if path not in cache:
        cache[path] = path.is_file()
    return cache[path]


def _audit_mio100_rows(
    rows: Sequence[dict[str, Any]],
    *,
    name: str,
    data_root: Path,
    expected_group: str | None,
    expected_count: int,
    trail: AuditTrail,
) -> None:
    errors: list[str] = []
    sample_ids: set[str] = set()
    path_cache: dict[Path, bool] = {}
    group_counts: Counter[str] = Counter()
    low_resolution_count = 0
    for row_index, row in enumerate(rows, start=1):
        if frozenset(row) != MIO100_SCHEMA:
            errors.append(f"row {row_index}: schema mismatch")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"row {row_index}: invalid sample_id")
        elif sample_id in sample_ids:
            errors.append(f"row {row_index}: duplicate sample_id={sample_id}")
        else:
            sample_ids.add(sample_id)
        group = row.get("group")
        group_counts[str(group)] += 1
        if expected_group is not None and group != expected_group:
            errors.append(f"row {row_index}: expected group {expected_group}, got {group!r}")
        if group not in {"A", "B", "C"}:
            errors.append(f"row {row_index}: invalid group={group!r}")
        if row.get("split") != "test" or row.get("source") != "AgenticIR":
            errors.append(f"row {row_index}: invalid split/source")
        degradations = row.get("degradations")
        has_lr = isinstance(degradations, list) and "low resolution" in degradations
        low_resolution_count += int(has_lr)
        if row.get("scale_factor") != (4 if has_lr else 1):
            errors.append(f"row {row_index}: scale_factor/degradation mismatch")

        for key in ("gt_path", "native_lq_path", "canonical_lq_path"):
            value = row.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute():
                errors.append(f"row {row_index}: {key} must be an absolute path")
                continue
            try:
                resolved = ensure_within(value, data_root)
            except Exception as exc:
                errors.append(f"row {row_index}: {key}: {exc}")
                continue
            if not _path_stat_only(resolved, path_cache):
                errors.append(f"row {row_index}: missing {key}={resolved}")
        if not has_lr and row.get("canonical_lq_path") != row.get("native_lq_path"):
            errors.append(f"row {row_index}: non-LR canonical/native paths differ")

    if len(rows) != expected_count:
        errors.append(f"expected {expected_count} rows, got {len(rows)}")
    trail.require(
        not errors,
        f"mio100.{name}.metadata_boundary",
        (
            f"{len(rows)} metadata rows; groups={dict(group_counts)}; "
            f"{len(path_cache)} referenced paths stat-only; image files opened=0"
        ),
        _format_examples(errors),
    )
    trail.facts.setdefault("mio100", {})[name] = {
        "rows": len(rows),
        "groups": dict(group_counts),
        "unique_sample_ids": len(sample_ids),
        "low_resolution_rows": low_resolution_count,
        "referenced_paths_stat_only": len(path_cache),
        "image_files_opened": 0,
    }


def _audit_mio100_subsets(
    config: Mapping[str, Any],
    expected_manifests: Mapping[str, Any],
    full_rows: Sequence[dict[str, Any]],
    data_root: Path,
    trail: AuditTrail,
) -> None:
    full_by_group: dict[str, dict[str, dict[str, Any]]] = {"A": {}, "B": {}, "C": {}}
    for row in full_rows:
        group = row.get("group")
        sample_id = row.get("sample_id")
        if group in full_by_group and isinstance(sample_id, str):
            full_by_group[group][sample_id] = row

    for name, (path_key, count, group) in MIO100_SUBSET_SPECS.items():
        path = Path(config[path_key]).resolve(strict=False)
        expected_sha = expected_manifests.get(name)
        rows = _audit_manifest_identity(
            name=name,
            path=path,
            expected_sha=expected_sha,
            expected_rows=count,
            expected_schema=MIO100_SCHEMA,
            expected_splits={"test": count},
            trail=trail,
        )
        _audit_mio100_rows(
            rows,
            name=name,
            data_root=data_root,
            expected_group=group,
            expected_count=count,
            trail=trail,
        )
        mismatches = []
        expected_subset = full_by_group[group]
        if len(expected_subset) != count:
            mismatches.append(f"full manifest group {group} has {len(expected_subset)} rows")
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or expected_subset.get(sample_id) != row:
                mismatches.append(f"row not identical to full manifest: {sample_id!r}")
        trail.require(
            not mismatches,
            f"mio100.{name}.subset_identity",
            f"exact metadata subset of formal-1440 Group {group}",
            _format_examples(mismatches),
        )

    exploration_path = Path(config["mio100_exploration_manifest"]).resolve(strict=False)
    expected_exploration_sha = expected_manifests.get("mio100_exploration")
    if not exploration_path.is_file():
        trail.record(
            "mio100.exploration.boundary",
            "FAIL",
            f"missing read-only archive manifest: {exploration_path}",
        )
    else:
        actual_sha = sha256_file(exploration_path)
        trail.require(
            is_sha256(expected_exploration_sha) and actual_sha == expected_exploration_sha,
            "mio100.exploration.boundary",
            f"manifest path/hash archived only; rows not read; sha256={actual_sha}",
            f"expected {expected_exploration_sha}, got {actual_sha}",
        )
        trail.facts.setdefault("mio100", {})["exploration_archive"] = {
            "path": str(exploration_path),
            "sha256": actual_sha,
            "rows_read": 0,
            "image_files_opened": 0,
            "allowed_uses": ["read_only_protocol_archive"],
        }


def _audit_parent_checkpoint(
    config: Mapping[str, Any], expected: Mapping[str, Any], trail: AuditTrail
) -> None:
    manifest_path = Path(config["stage_a_parent_manifest"]).resolve(strict=False)
    checkpoint_path = Path(config["stage_a_parent_checkpoint"]).resolve(strict=False)
    expected_manifest_sha = expected.get("stage_a_parent_manifest_sha256")
    expected_checkpoint_sha = expected.get("stage_a_parent_sha256")

    if not manifest_path.is_file():
        trail.record("parent.manifest", "FAIL", f"missing: {manifest_path}")
        return
    actual_manifest_sha = sha256_file(manifest_path)
    trail.require(
        is_sha256(expected_manifest_sha) and actual_manifest_sha == expected_manifest_sha,
        "parent.manifest.sha256",
        f"sha256={actual_manifest_sha}",
        f"expected {expected_manifest_sha}, got {actual_manifest_sha}",
    )
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        trail.record("parent.manifest.json", "FAIL", str(exc))
        return
    if not isinstance(manifest, dict):
        trail.record("parent.manifest.json", "FAIL", "manifest is not a JSON object")
        return

    selected_path_raw = manifest.get("selected_checkpoint")
    selected_path = (
        Path(selected_path_raw).resolve(strict=False)
        if isinstance(selected_path_raw, str)
        else None
    )
    selected_sha = manifest.get("selected_checkpoint_sha256")
    trail.require(
        selected_path == checkpoint_path,
        "parent.manifest.selected_path",
        f"selected checkpoint={checkpoint_path}",
        f"config={checkpoint_path}, manifest={selected_path_raw!r}",
    )
    trail.require(
        selected_sha == expected_checkpoint_sha,
        "parent.manifest.selected_sha",
        f"selected sha256={selected_sha}",
        f"expected {expected_checkpoint_sha}, manifest={selected_sha}",
    )
    trail.require(
        manifest.get("official_test_used") is False,
        "parent.manifest.no_official_test_selection",
        "official_test_used=false",
        f"official_test_used={manifest.get('official_test_used')!r}",
    )
    if not checkpoint_path.is_file():
        trail.record("parent.checkpoint.exists", "FAIL", f"missing: {checkpoint_path}")
        return
    actual_checkpoint_sha = sha256_file(checkpoint_path)
    trail.require(
        is_sha256(expected_checkpoint_sha) and actual_checkpoint_sha == expected_checkpoint_sha,
        "parent.checkpoint.sha256",
        f"sha256={actual_checkpoint_sha}",
        f"expected {expected_checkpoint_sha}, got {actual_checkpoint_sha}",
    )
    trail.facts["stage_a_parent"] = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_checkpoint_sha,
        "selected": manifest.get("selected"),
        "official_test_used": manifest.get("official_test_used"),
    }


def run_audit(config_path: str | Path) -> AuditTrail:
    """Run the complete CPU-only pre-Stage0 data/identity audit."""

    config_path = Path(config_path).resolve()
    trail = AuditTrail(protocol=PROTOCOL)
    try:
        config = load_resolved_paths(config_path)
    except Exception as exc:
        trail.record("config.resolved_paths", "FAIL", str(exc))
        return trail
    trail.record("config.resolved_paths", "PASS", str(config_path))

    expected = config["expected_identity"]
    if not isinstance(expected, Mapping):
        trail.record("config.expected_identity", "FAIL", "expected_identity is not a mapping")
        return trail
    expected_manifests = expected.get("manifests")
    if not isinstance(expected_manifests, Mapping):
        trail.record(
            "config.expected_manifests", "FAIL", "expected_identity.manifests is missing"
        )
        expected_manifests = {}

    agenticir_repo = Path(config["agenticir_repo"]).resolve(strict=False)
    mioir_repo = Path(config["mioir_repo"]).resolve(strict=False)
    _audit_repo(
        name="agenticir",
        repo=agenticir_repo,
        expected_commit=str(expected.get("agenticir_commit", "")),
        expected_remote=str(expected.get("agenticir_remote", "")),
        trail=trail,
    )
    _audit_repo(
        name="mioir",
        repo=mioir_repo,
        expected_commit=str(expected.get("mioir_commit", "")),
        expected_remote=str(expected.get("mioir_remote", "")),
        trail=trail,
    )
    trail.require(
        expected.get("agenticir_commit") == EXPECTED_AGENTICIR_COMMIT,
        "contract.agenticir_commit",
        f"contract commit={EXPECTED_AGENTICIR_COMMIT}",
        f"resolved config changed contract commit to {expected.get('agenticir_commit')}",
    )
    trail.require(
        expected.get("mioir_commit") == EXPECTED_MIOIR_COMMIT,
        "contract.mioir_commit",
        f"contract commit={EXPECTED_MIOIR_COMMIT}",
        f"resolved config changed contract commit to {expected.get('mioir_commit')}",
    )
    trail.require(
        canonical_git_remote(str(expected.get("agenticir_remote", "")))
        == canonical_git_remote(EXPECTED_AGENTICIR_REMOTE),
        "contract.agenticir_remote",
        f"contract remote={EXPECTED_AGENTICIR_REMOTE}",
        f"resolved config changed AgenticIR remote to {expected.get('agenticir_remote')}",
    )
    trail.require(
        canonical_git_remote(str(expected.get("mioir_remote", "")))
        == canonical_git_remote(EXPECTED_MIOIR_REMOTE),
        "contract.mioir_remote",
        f"contract remote={EXPECTED_MIOIR_REMOTE}",
        f"resolved config changed MiOIR remote to {expected.get('mioir_remote')}",
    )
    trail.require(
        expected.get("stage_a_parent_sha256") == EXPECTED_STAGE_A_PARENT_SHA256,
        "contract.stage_a_parent_sha256",
        f"contract sha256={EXPECTED_STAGE_A_PARENT_SHA256}",
        f"resolved config changed parent SHA to {expected.get('stage_a_parent_sha256')}",
    )
    for manifest_name, contract_sha in CONTRACT_MANIFEST_SHA256.items():
        trail.require(
            expected_manifests.get(manifest_name) == contract_sha,
            f"contract.manifest.{manifest_name}.sha256",
            f"contract sha256={contract_sha}",
            f"resolved config changed SHA to {expected_manifests.get(manifest_name)}",
        )
    configured_agenticir_files = expected.get("agenticir_files")
    for file_name, locked_sha in LOCKED_AGENTICIR_FILE_SHA256.items():
        actual_configured = (
            configured_agenticir_files.get(file_name)
            if isinstance(configured_agenticir_files, Mapping)
            else None
        )
        trail.require(
            actual_configured == locked_sha,
            f"contract.agenticir_file.{file_name}.sha256",
            f"locked sha256={locked_sha}",
            f"resolved config changed SHA to {actual_configured}",
        )
    _audit_agenticir_files(config, expected, trail)

    training_root = Path(config["training_data_root"]).resolve(strict=False)
    data_root = Path(config["data_root"]).resolve(strict=False)
    trail.require(
        training_root.is_dir(),
        "root.training_data",
        str(training_root),
        f"missing training data root: {training_root}",
    )
    trail.require(
        data_root.is_dir(),
        "root.mio100_data",
        str(data_root),
        f"missing MiO100 data root: {data_root}",
    )

    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, spec in MANIFEST_SPECS.items():
        path = resolve_config_path(config_path, config[spec["path_key"]])
        rows_by_name[name] = _audit_manifest_identity(
            name=name,
            path=path,
            expected_sha=expected_manifests.get(name),
            expected_rows=spec["rows"],
            expected_schema=spec["schema"],
            expected_splits=spec["splits"],
            trail=trail,
        )

    clean_train_index = _audit_clean_rows(
        rows_by_name.get("clean_train", []),
        name="train",
        expected_split="train",
        training_root=training_root,
        trail=trail,
    )
    clean_val_index = _audit_clean_rows(
        rows_by_name.get("clean_val", []),
        name="val",
        expected_split="val",
        training_root=training_root,
        trail=trail,
    )
    overlap = sorted(set(clean_train_index).intersection(clean_val_index))
    trail.require(
        not overlap,
        "clean.split_disjoint",
        f"train={len(clean_train_index)}, val={len(clean_val_index)}, overlap=0",
        f"overlapping clean IDs: {_format_examples(overlap)}",
    )
    clean_by_split = {"train": clean_train_index, "val": clean_val_index}

    _audit_primary_rows(
        rows_by_name.get("primary_train", []),
        name="train",
        clean_by_split=clean_by_split,
        training_root=training_root,
        expected_task_count_by_split={"train": 900},
        trail=trail,
    )
    _audit_primary_rows(
        rows_by_name.get("primary_val", []),
        name="val",
        clean_by_split=clean_by_split,
        training_root=training_root,
        expected_task_count_by_split={"val": 100},
        trail=trail,
    )
    _audit_primary_rows(
        rows_by_name.get("primary_all", []),
        name="all",
        clean_by_split=clean_by_split,
        training_root=training_root,
        expected_task_count_by_split={"train": 900, "val": 100},
        trail=trail,
    )
    expected_all = rows_by_name.get("primary_train", []) + rows_by_name.get(
        "primary_val", []
    )
    actual_all = rows_by_name.get("primary_all", [])
    expected_all_by_id = {
        row.get("sample_id"): row
        for row in expected_all
        if isinstance(row.get("sample_id"), str)
    }
    actual_all_by_id = {
        row.get("sample_id"): row
        for row in actual_all
        if isinstance(row.get("sample_id"), str)
    }
    trail.require(
        actual_all_by_id == expected_all_by_id
        and len(actual_all_by_id) == len(actual_all) == len(expected_all),
        "primary.all_exact_union",
        "primary_all is the exact record union of primary_train and primary_val",
        "primary_all record content differs from primary_train union primary_val",
    )

    full_mio100 = rows_by_name.get("mio100_test_1440", [])
    _audit_mio100_rows(
        full_mio100,
        name="formal_1440",
        data_root=data_root,
        expected_group=None,
        expected_count=1440,
        trail=trail,
    )
    full_group_counts = Counter(row.get("group") for row in full_mio100)
    trail.require(
        full_group_counts == Counter({"A": 640, "B": 400, "C": 400}),
        "mio100.formal_1440.group_counts",
        f"groups={dict(full_group_counts)}",
        f"expected A=640/B=400/C=400, got {dict(full_group_counts)}",
    )
    _audit_mio100_subsets(
        config, expected_manifests, full_mio100, data_root, trail
    )
    _audit_parent_checkpoint(config, expected, trail)

    trail.facts["data_boundary"] = {
        "training_sources": ["MiOIR-Train clean/depth", "AgenticIR official operators"],
        "training_groups": ["single", "A"],
        "group_b_or_c_training_rows": 0,
        "mio100_formal_image_files_opened": 0,
        "mio100_exploration_rows_read": 0,
        "mio100_allowed_use_at_this_stage": "manifest path/hash boundary audit only",
    }
    return trail


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "resolved_paths.yaml",
        help="resolved path/identity YAML",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "DATA_AUDIT.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "audits" / "data_audit.json",
        help="machine-readable report path",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run all checks but do not write report files",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress Markdown output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    trail = run_audit(args.config)
    markdown = trail.to_markdown(title="GraphRestore V7.1 Data and Identity Audit")
    if not args.check_only:
        atomic_write_text(args.report, markdown)
        atomic_write_json(args.json_output, trail.to_dict())
    if not args.quiet:
        print(markdown, end="")
    return 0 if trail.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
