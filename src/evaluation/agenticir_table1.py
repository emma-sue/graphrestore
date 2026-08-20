"""Fail-closed AgenticIR Table-1 six-metric scoring.

This module deliberately keeps heavyweight ``pyiqa`` imports in a separately
launched, version-locked interpreter.  The public controller only accepts the
immutable 1,440-row mapping published by formal MiO100 inference, verifies all
PNG hashes, and consumes/creates a contiguous prefix of immutable score
shards.  A valid shard is never overwritten or selectively recomputed.

The metric implementation itself is the pinned AgenticIR ``Scorer`` class
with only its eager module-level singleton removed.  Device placement and the
CLIP download root are supplied explicitly; its image loading, x4 handling,
metric defaults, and calls are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import is_dataclass
from importlib import metadata, util
import io
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import sys
import tempfile
import types
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.metrics.agenticir_official import OFFICIAL_GROUPS
from src.utils.hashing import is_sha256, sha256_file, sha256_json
from src.utils.io import fsync_directory, utc_now_iso


INPUT_SCHEMA = "graphrestore.agenticir_table1_input.v1"
WEIGHTS_LOCK_SCHEMA = "graphrestore.agenticir_table1_weights_lock.v1"
INPUT_LOCK_SCHEMA = "graphrestore.agenticir_table1_input_lock.v1"
RUN_CONTRACT_SCHEMA = "graphrestore.agenticir_table1_run_contract.v1"
SHARD_SCHEMA = "graphrestore.agenticir_table1_score_shard.v1"
SUMMARY_SCHEMA = "graphrestore.agenticir_table1_summary.v1"
COMPLETE_SCHEMA = "graphrestore.agenticir_table1_complete.v1"
WORKER_REQUEST_SCHEMA = "graphrestore.agenticir_table1_worker_request.v1"
WORKER_RESULT_SCHEMA = "graphrestore.agenticir_table1_worker_result.v1"

METRICS = ("psnr", "ssim", "lpips", "maniqa", "clipiqa", "musiq")
METRIC_DIRECTIONS = {
    metric: ("lower" if metric == "lpips" else "higher") for metric in METRICS
}
FULL_REFERENCE_METRICS = ("psnr", "ssim", "lpips")
NO_REFERENCE_METRICS = ("maniqa", "clipiqa", "musiq")
EXPECTED_METRIC_RUNTIME = [
    {
        "name": metric,
        "mode": "FR" if metric in FULL_REFERENCE_METRICS else "NR",
        "lower_better": metric == "lpips",
    }
    for metric in METRICS
]
INPUT_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "group",
        "combination",
        "prediction_png",
        "prediction_sha256",
        "target_png",
        "target_sha256",
    }
)
SCORE_ROW_KEYS = INPUT_KEYS | frozenset(METRICS)

EXPECTED_COUNTS: dict[str, int] = {
    combination: (80 if group == "A" else 100)
    for group, combinations in OFFICIAL_GROUPS.items()
    for combination in combinations
}
EXPECTED_IMAGE_COUNT = sum(EXPECTED_COUNTS.values())

PINNED_AGENTICIR_COMMIT = "9640a291480dee3ba8f2974125d4ee9e3440f3d6"
PINNED_SOURCE_SHA256 = {
    "official_scorer": "b6eee989575ee17d2cbf9e38fbab0a996b54a5260ae205246c718c08facab830",
    "official_compute_scores": "ce1a35f9f110a67c4581885f631dae6c283e438bcaf2749199fb9d19fa440548",
    "official_compare_methods": "a246b8656744649ed5adfd5f482491f89006ef7bec1ce9923b5971a1da3d856a",
    "official_requirements": "3e76d9e7c658ce7df907dc39ea7af8aa36aa2d5fcf5bd6ec91d34c109a9b45e2",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DISK_ROOT = Path("/root/autodl-tmp")
DEFAULT_REFERENCE_PYTHON = PROJECT_ROOT / ".venv-reference" / "bin" / "python"
TRUSTED_REFERENCE_BASE_PYTHON = Path("/root/miniconda3/bin/python3.12")
DEFAULT_UPSTREAM_ROOT = Path("/root/autodl-tmp/graph/upstream/AgenticIR")
DEFAULT_CACHE_ROOT = Path(
    "/root/autodl-tmp/aaa/graphrestore/artifacts/formal_mio100/cache"
)
DEFAULT_CLI_PATH = PROJECT_ROOT / "scripts" / "score_agenticir_table1.py"
FORMAL_AUTHORIZATION_PATH = (
    PROJECT_ROOT / "artifacts" / "approvals" / "FORMAL_MIO100_APPROVED.json"
)
FORMAL_EVALUATOR_ROOT = (
    PROJECT_ROOT / "artifacts" / "formal_mio100" / "graphrestore_v7_1_stage4_step040000"
)
FORMAL_EVALUATOR_COMPLETE_PATH = FORMAL_EVALUATOR_ROOT / "complete.json"
FORMAL_TABLE1_INPUT_PATH = FORMAL_EVALUATOR_ROOT / "table1_input.jsonl"
FORMAL_SCORE_ROOT = FORMAL_EVALUATOR_ROOT / "table1_scores"
FORMAL_WORK_ROOT = PROJECT_ROOT / "artifacts" / "work" / "agenticir_table1"
FORMAL_DEVICE = "cuda:0"
FORMAL_SHARD_SIZE = 10
FORMAL_ALLOCATOR = "backend:native,expandable_segments:True"
MAXIMUM_VRAM_RESERVED_FRACTION = 0.90

_SCORE_ROOT_ALLOWED = {
    "input_lock.json",
    "run_contract.json",
    "shards",
    ".worker",
    "per_image.csv",
    "summary.json",
    "complete.json",
}
_SHARD_NAME_PATTERN = re.compile(r"^shard-[0-9]{5}\.json$")
_WORKER_REQUEST_NAME_PATTERN = re.compile(r"^request-[0-9]{5}\.json$")
_FORMAL_EVIDENCE_KEYS = frozenset(
    {
        "authorization",
        "evaluator_complete",
        "run_contract",
        "summary",
        "per_image",
        "table1_input",
        "checkpoint",
        "manifest",
        "formal_data_inventory",
        "metric_parity_summary",
        "predictions_digest",
    }
)

_CUDA_DEVICE_PATTERN = re.compile(r"^cuda(?::([0-9]+))?$")


class Table1ContractError(RuntimeError):
    """A frozen-input, cache, shard, or publication contract was violated."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Table1ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_strict(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise Table1ContractError(f"cannot read strict JSON {path}: {exc}") from exc


def _load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise Table1ContractError(
                        f"{path}:{line_number}: blank JSONL rows are forbidden"
                    )
                try:
                    value = json.loads(raw_line, object_pairs_hook=_strict_object)
                except json.JSONDecodeError as exc:
                    raise Table1ContractError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise Table1ContractError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                rows.append(value)
    except OSError as exc:
        raise Table1ContractError(f"cannot read input manifest {path}: {exc}") from exc
    return rows


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _assert_no_symlink_chain(path: Path, *, data_root: Path = DATA_DISK_ROOT) -> Path:
    """Require an absolute canonical path below data disk with no symlink component."""

    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise Table1ContractError(f"path must be absolute and normalized: {candidate}")
    configured_root = Path(data_root)
    if not configured_root.is_absolute() or configured_root != Path(
        os.path.abspath(configured_root)
    ):
        raise Table1ContractError(f"data root must be absolute: {configured_root}")
    current_ancestor = Path(configured_root.anchor)
    for part in configured_root.parts[1:]:
        current_ancestor = current_ancestor / part
        info = current_ancestor.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise Table1ContractError(
                f"data-root ancestor is a symlink: {current_ancestor}"
            )
    root = configured_root.resolve(strict=True)
    if root != configured_root:
        raise Table1ContractError(f"data root is not canonical: {configured_root}")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise Table1ContractError(
            f"path escapes data disk {root}: {candidate}"
        ) from exc
    current = root
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise Table1ContractError(f"data root is not a real directory: {root}")
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise Table1ContractError(f"path crosses a symlink: {current}")
    return candidate


def _ensure_secure_directory(path: Path, *, data_root: Path = DATA_DISK_ROOT) -> Path:
    candidate = _assert_no_symlink_chain(path, data_root=data_root)
    if candidate.exists():
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise Table1ContractError(f"expected a real directory: {candidate}")
    else:
        parent = _ensure_secure_directory(candidate.parent, data_root=data_root)
        candidate.mkdir(mode=0o755)
        fsync_directory(parent)
    _assert_no_symlink_chain(candidate, data_root=data_root)
    if not candidate.is_dir() or candidate.is_symlink():
        raise Table1ContractError(f"directory changed during creation: {candidate}")
    return candidate


def _assert_confined_write_path(path: Path, confinement_root: Path) -> None:
    root = _assert_no_symlink_chain(confinement_root)
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise Table1ContractError(f"confinement root is not a real directory: {root}")
    candidate = _assert_no_symlink_chain(path)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Table1ContractError(
            f"write target escapes confinement root {root}: {candidate}"
        ) from exc
    parent = _assert_no_symlink_chain(candidate.parent)
    if not parent.is_dir() or parent.is_symlink():
        raise Table1ContractError(f"write parent is not a real directory: {parent}")
    if candidate.is_symlink():
        raise Table1ContractError(f"write target is a symlink: {candidate}")


def _assert_score_tree_shape(
    root: Path,
    *,
    data_root: Path = DATA_DISK_ROOT,
) -> None:
    """Reject any unrecognized, writable, or symlinked score-root artifact."""

    root = _assert_no_symlink_chain(root, data_root=data_root)
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise Table1ContractError(f"score root is not a real directory: {root}")
    names = {entry.name for entry in root.iterdir()}
    unexpected = names - _SCORE_ROOT_ALLOWED
    if unexpected:
        raise Table1ContractError(
            f"unauthorized score-root entries: {sorted(unexpected)}"
        )
    if names and "input_lock.json" not in names:
        raise Table1ContractError(
            "non-empty score root is not an exact resume: input_lock.json is absent"
        )
    if "run_contract.json" not in names:
        allowed_prefix = {"input_lock.json"}
        if names - allowed_prefix:
            raise Table1ContractError(
                "pre-contract score root contains non-prefix artifacts"
            )
    for filename in (
        "input_lock.json",
        "run_contract.json",
        "per_image.csv",
        "summary.json",
        "complete.json",
    ):
        path = root / filename
        if path.is_symlink():
            raise Table1ContractError(f"score artifact is a symlink: {path}")
        if path.exists():
            _require_regular_file(path, immutable=True)

    nested_specs = {
        "shards": _SHARD_NAME_PATTERN,
        ".worker": _WORKER_REQUEST_NAME_PATTERN,
    }
    for dirname, pattern in nested_specs.items():
        directory = root / dirname
        if directory.is_symlink():
            raise Table1ContractError(f"score directory is a symlink: {directory}")
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise Table1ContractError(f"score path is not a directory: {directory}")
        for entry in directory.iterdir():
            if entry.is_symlink():
                raise Table1ContractError(f"score-tree entry is a symlink: {entry}")
            if not pattern.fullmatch(entry.name):
                raise Table1ContractError(f"unauthorized {dirname} entry: {entry.name}")
            _require_regular_file(entry, immutable=True)


def _prepare_score_root(
    root: Path,
    *,
    data_root: Path = DATA_DISK_ROOT,
    expected_root: Path | None = None,
) -> Path:
    candidate = Path(root)
    if expected_root is not None and candidate != expected_root:
        raise Table1ContractError(
            f"formal score root is fixed at {expected_root}, got {candidate}"
        )
    candidate = _ensure_secure_directory(candidate, data_root=data_root)
    _assert_score_tree_shape(candidate, data_root=data_root)
    return candidate


def _atomic_create_text(
    path: Path,
    payload: str,
    *,
    mode: int = 0o444,
    confinement_root: Path | None = None,
) -> None:
    """Publish *path* atomically and fail if it already exists.

    A same-directory temporary file is fully fsynced and chmodded before a
    hard-link publication.  ``os.link`` supplies the no-overwrite guarantee.
    """

    if confinement_root is not None:
        _assert_confined_write_path(path, confinement_root)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if confinement_root is not None:
            _assert_confined_write_path(path, confinement_root)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Table1ContractError(
                f"refusing to overwrite published artifact: {path}"
            ) from exc
        fsync_directory(path.parent)
        if confinement_root is not None:
            _assert_confined_write_path(path, confinement_root)
    finally:
        temporary.unlink(missing_ok=True)
        fsync_directory(path.parent)


def _atomic_create_json(
    path: Path, value: Any, *, confinement_root: Path | None = None
) -> None:
    _atomic_create_text(path, _canonical_json(value), confinement_root=confinement_root)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _require_regular_file(path: Path, *, immutable: bool = False) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise Table1ContractError(f"missing required file {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise Table1ContractError(f"expected a non-symlink regular file: {path}")
    if immutable and stat.S_IMODE(info.st_mode) != 0o444:
        raise Table1ContractError(
            f"expected mode 0444 for immutable file {path}, got {stat.S_IMODE(info.st_mode):04o}"
        )
    return info


def _require_data_disk(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    root = DATA_DISK_ROOT.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Table1ContractError(
            f"{label} must be on the data disk below {root}, got {resolved}"
        ) from exc
    return resolved


def _require_exact_keys(
    value: Mapping[str, Any], keys: Iterable[str], *, label: str
) -> None:
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise Table1ContractError(
            f"{label} key mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise Table1ContractError(f"{label} must be a lowercase SHA256")
    return str(value)


def _file_binding(path: Path, *, immutable: bool = False) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    info = _require_regular_file(resolved, immutable=immutable)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "mode": stat.S_IMODE(info.st_mode),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size": int(info.st_size),
    }


def _reference_launcher_binding(
    reference_python: Path,
    *,
    trusted_base_python: Path = TRUSTED_REFERENCE_BASE_PYTHON,
) -> dict[str, Any]:
    """Bind a venv launcher without replacing it by its base interpreter.

    CPython discovers ``pyvenv.cfg`` from the path used to launch it.  Resolving
    ``.venv-reference/bin/python`` before execution therefore silently disables
    the pinned reference environment.  We instead bind both the launcher link
    and its trusted final target, while retaining the unresolved launcher path
    for every subprocess call.
    """

    launcher = Path(reference_python)
    if not launcher.is_absolute() or launcher != Path(os.path.abspath(launcher)):
        raise Table1ContractError(
            f"reference launcher must be absolute and normalized: {launcher}"
        )
    try:
        launcher_info = launcher.lstat()
    except OSError as exc:
        raise Table1ContractError(
            f"missing reference launcher {launcher}: {exc}"
        ) from exc
    if not stat.S_ISLNK(launcher_info.st_mode):
        raise Table1ContractError(
            f"reference launcher must be a venv symlink: {launcher}"
        )

    # No ancestor of the venv launcher may itself redirect path discovery.
    current = Path(launcher.anchor)
    for part in launcher.parts[1:-1]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise Table1ContractError(
                f"cannot inspect reference-launcher ancestor {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise Table1ContractError(
                f"reference-launcher ancestor is a symlink: {current}"
            )

    prefix = launcher.parent.parent
    if launcher.parent.name != "bin":
        raise Table1ContractError(
            f"reference launcher is not under a venv bin directory: {launcher}"
        )
    prefix_info = prefix.lstat()
    if stat.S_ISLNK(prefix_info.st_mode) or not stat.S_ISDIR(prefix_info.st_mode):
        raise Table1ContractError(f"reference prefix is not a real directory: {prefix}")
    config_path = prefix / "pyvenv.cfg"
    config_info = _require_regular_file(config_path)

    resolved_target = launcher.resolve(strict=True)
    trusted_target = Path(trusted_base_python).resolve(strict=True)
    if resolved_target != trusted_target:
        raise Table1ContractError(
            "reference launcher target mismatch: "
            f"expected {trusted_target}, got {resolved_target}"
        )
    target_info = _require_regular_file(resolved_target)
    if not os.access(resolved_target, os.X_OK):
        raise Table1ContractError(
            f"reference launcher target is not executable: {resolved_target}"
        )

    config: dict[str, str] = {}
    try:
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            key, separator, value = raw_line.partition("=")
            if not separator or not key.strip() or key.strip() in config:
                raise Table1ContractError(
                    f"malformed/duplicate pyvenv.cfg line: {raw_line!r}"
                )
            config[key.strip()] = value.strip()
    except OSError as exc:
        raise Table1ContractError(f"cannot read {config_path}: {exc}") from exc
    configured_executable = config.get("executable")
    if (
        configured_executable is None
        or Path(configured_executable).resolve(strict=True) != trusted_target
    ):
        raise Table1ContractError(
            "pyvenv.cfg executable does not bind the trusted base interpreter"
        )

    return {
        "launcher": {
            "path": str(launcher),
            "mode": stat.S_IMODE(launcher_info.st_mode),
            "device": int(launcher_info.st_dev),
            "inode": int(launcher_info.st_ino),
            "size": int(launcher_info.st_size),
            "symlink_target": os.readlink(launcher),
        },
        "resolved_target": {
            "path": str(resolved_target),
            "sha256": sha256_file(resolved_target),
            "mode": stat.S_IMODE(target_info.st_mode),
            "device": int(target_info.st_dev),
            "inode": int(target_info.st_ino),
            "size": int(target_info.st_size),
        },
        "reference_prefix": {
            "path": str(prefix),
            "mode": stat.S_IMODE(prefix_info.st_mode),
            "device": int(prefix_info.st_dev),
            "inode": int(prefix_info.st_ino),
        },
        "pyvenv_cfg": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "mode": stat.S_IMODE(config_info.st_mode),
            "device": int(config_info.st_dev),
            "inode": int(config_info.st_ino),
            "size": int(config_info.st_size),
        },
        "expected_sys_base_prefix": str(trusted_target.parent.parent),
    }


def _validate_reference_environment_for_launcher(
    environment: Mapping[str, Any], launcher_binding: Mapping[str, Any]
) -> None:
    invocation = environment.get("invocation")
    if not isinstance(invocation, Mapping):
        raise Table1ContractError("reference environment has no invocation binding")
    _require_exact_keys(
        invocation,
        {
            "sys_executable",
            "sys_executable_realpath",
            "sys_prefix",
            "sys_base_prefix",
        },
        label="reference invocation",
    )
    launcher = launcher_binding["launcher"]
    target = launcher_binding["resolved_target"]
    prefix = launcher_binding["reference_prefix"]
    expected = {
        "sys_executable": launcher["path"],
        "sys_executable_realpath": target["path"],
        "sys_prefix": prefix["path"],
        "sys_base_prefix": launcher_binding["expected_sys_base_prefix"],
    }
    if dict(invocation) != expected:
        raise Table1ContractError(
            f"reference interpreter did not activate the bound venv: "
            f"expected={expected}, actual={dict(invocation)}"
        )
    if environment.get("executable") != target:
        raise Table1ContractError(
            "reference runtime executable differs from the trusted launcher target"
        )


def default_source_paths() -> dict[str, Path]:
    return {
        "official_scorer": DEFAULT_UPSTREAM_ROOT / "utils" / "scorer.py",
        "official_compute_scores": DEFAULT_UPSTREAM_ROOT / "eval" / "compute_scores.py",
        "official_compare_methods": DEFAULT_UPSTREAM_ROOT
        / "eval"
        / "compare_methods.py",
        "official_requirements": DEFAULT_UPSTREAM_ROOT
        / "installation"
        / "requirements.txt",
    }


def validate_pinned_sources(
    source_paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    if set(source_paths) != set(PINNED_SOURCE_SHA256):
        raise Table1ContractError("the four pinned AgenticIR source paths are required")
    bindings: dict[str, dict[str, Any]] = {}
    for label, expected_sha in PINNED_SOURCE_SHA256.items():
        binding = _file_binding(Path(source_paths[label]))
        if binding["sha256"] != expected_sha:
            raise Table1ContractError(
                f"{label} is not pinned AgenticIR {PINNED_AGENTICIR_COMMIT}: "
                f"expected {expected_sha}, got {binding['sha256']}"
            )
        bindings[label] = binding
    return bindings


def _validate_expected_counts(counts: Mapping[str, int]) -> None:
    official = {
        combination
        for combinations in OFFICIAL_GROUPS.values()
        for combination in combinations
    }
    if set(counts) != official:
        raise Table1ContractError(
            "expected-count combinations must exactly match the 16 official tasks"
        )
    for combination, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise Table1ContractError(
                f"{combination}: expected count must be a positive integer"
            )


def _expected_order(counts: Mapping[str, int]) -> list[tuple[str, str]]:
    _validate_expected_counts(counts)
    order: list[tuple[str, str]] = []
    for group, combinations in OFFICIAL_GROUPS.items():
        for combination in combinations:
            order.extend((group, combination) for _ in range(int(counts[combination])))
    return order


def validate_manifest_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
    verify_files: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate exact schema, canonical ordering, counts, hashes, and modes."""

    expected_order = _expected_order(expected_counts)
    if len(rows) != len(expected_order):
        raise Table1ContractError(
            f"expected {len(expected_order)} Table-1 rows, got {len(rows)}"
        )

    canonical_rows: list[dict[str, Any]] = []
    locked_rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_predictions: set[str] = set()
    previous_sample_by_combination: dict[str, str] = {}

    for index, (raw, (expected_group, expected_combination)) in enumerate(
        zip(rows, expected_order, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise Table1ContractError(f"manifest row {index} is not an object")
        _require_exact_keys(raw, INPUT_KEYS, label=f"manifest row {index}")
        if raw["schema_version"] != INPUT_SCHEMA:
            raise Table1ContractError(f"manifest row {index}: bad schema_version")
        for key in (
            "sample_id",
            "group",
            "combination",
            "prediction_png",
            "target_png",
        ):
            if not isinstance(raw[key], str) or not raw[key]:
                raise Table1ContractError(
                    f"manifest row {index}: {key} must be non-empty text"
                )
        if (raw["group"], raw["combination"]) != (
            expected_group,
            expected_combination,
        ):
            raise Table1ContractError(
                f"manifest row {index}: expected {expected_group}/{expected_combination}, "
                f"got {raw['group']}/{raw['combination']}"
            )
        sample_id = str(raw["sample_id"])
        previous = previous_sample_by_combination.get(expected_combination)
        if previous is not None and sample_id <= previous:
            raise Table1ContractError(
                f"{expected_combination}: sample_id order is not strictly increasing at {sample_id!r}"
            )
        previous_sample_by_combination[expected_combination] = sample_id
        if sample_id in seen_sample_ids:
            raise Table1ContractError(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        prediction_sha = _require_sha256(
            raw["prediction_sha256"], label=f"manifest row {index} prediction_sha256"
        )
        target_sha = _require_sha256(
            raw["target_sha256"], label=f"manifest row {index} target_sha256"
        )
        prediction = Path(str(raw["prediction_png"]))
        target = Path(str(raw["target_png"]))
        if not prediction.is_absolute() or not target.is_absolute():
            raise Table1ContractError(
                f"manifest row {index}: PNG paths must be absolute"
            )
        if prediction.suffix.lower() != ".png" or target.suffix.lower() != ".png":
            raise Table1ContractError(
                f"manifest row {index}: both inputs must be PNG files"
            )
        prediction_text = str(prediction)
        if prediction_text in seen_predictions:
            raise Table1ContractError(f"duplicate prediction path: {prediction}")
        seen_predictions.add(prediction_text)
        if prediction == target:
            raise Table1ContractError(
                f"manifest row {index}: prediction and target are identical paths"
            )

        canonical = {
            "schema_version": INPUT_SCHEMA,
            "sample_id": sample_id,
            "group": expected_group,
            "combination": expected_combination,
            "prediction_png": prediction_text,
            "prediction_sha256": prediction_sha,
            "target_png": str(target),
            "target_sha256": target_sha,
        }
        locked = dict(canonical)
        if verify_files:
            try:
                prediction_resolved = prediction.resolve(strict=True)
                target_resolved = target.resolve(strict=True)
            except OSError as exc:
                raise Table1ContractError(
                    f"manifest row {index}: cannot resolve PNG path: {exc}"
                ) from exc
            if str(prediction_resolved) != prediction_text or str(
                target_resolved
            ) != str(target):
                raise Table1ContractError(
                    f"manifest row {index}: PNG paths must be canonical and non-symlinked"
                )
            prediction_info = _require_regular_file(prediction, immutable=True)
            target_info = _require_regular_file(target)
            if sha256_file(prediction) != prediction_sha:
                raise Table1ContractError(
                    f"prediction hash mismatch at row {index}: {prediction}"
                )
            if sha256_file(target) != target_sha:
                raise Table1ContractError(
                    f"target hash mismatch at row {index}: {target}"
                )
            locked.update(
                {
                    "prediction_mode": stat.S_IMODE(prediction_info.st_mode),
                    "prediction_device": int(prediction_info.st_dev),
                    "prediction_inode": int(prediction_info.st_ino),
                    "prediction_size": int(prediction_info.st_size),
                    "target_mode": stat.S_IMODE(target_info.st_mode),
                    "target_device": int(target_info.st_dev),
                    "target_inode": int(target_info.st_ino),
                    "target_size": int(target_info.st_size),
                }
            )
        canonical_rows.append(canonical)
        locked_rows.append(locked)
    return canonical_rows, locked_rows


def validate_input_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    manifest_info = _require_regular_file(resolved, immutable=True)
    raw_rows = _load_jsonl_strict(resolved)
    rows, locked_rows = validate_manifest_records(raw_rows)
    lock = {
        "schema_version": INPUT_LOCK_SCHEMA,
        "manifest": {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "mode": stat.S_IMODE(manifest_info.st_mode),
            "device": int(manifest_info.st_dev),
            "inode": int(manifest_info.st_ino),
            "size": int(manifest_info.st_size),
        },
        "image_count": len(rows),
        "expected_counts": dict(EXPECTED_COUNTS),
        "ordering": "OFFICIAL_GROUPS order, then strictly increasing sample_id",
        "rows": locked_rows,
    }
    return rows, lock


def _source_tree(import_name: str) -> dict[str, Any]:
    spec = util.find_spec(import_name)
    if spec is None or not spec.submodule_search_locations:
        raise Table1ContractError(
            f"cannot locate installed package source: {import_name}"
        )
    roots = [Path(item).resolve() for item in spec.submodule_search_locations]
    if len(roots) != 1:
        raise Table1ContractError(f"ambiguous package roots for {import_name}: {roots}")
    root = roots[0]
    files = [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*.py"))
        if path.is_file() and not path.is_symlink()
    ]
    if not files:
        raise Table1ContractError(f"no Python source files found for {import_name}")
    return {
        "root": str(root),
        "file_count": len(files),
        "sha256": sha256_json(files),
        "files": files,
    }


def _install_torchvision_compatibility() -> None:
    from torchvision.transforms.functional import rgb_to_grayscale

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = rgb_to_grayscale
    sys.modules.setdefault("torchvision.transforms.functional_tensor", shim)


def _reference_environment() -> dict[str, Any]:
    _install_torchvision_compatibility()
    import cv2
    import numpy as np
    import torch
    from pyiqa.default_model_configs import DEFAULT_CONFIGS

    dependency_names = (
        "pyiqa",
        "basicsr",
        "opencv-python",
        "numpy",
        "torch",
        "torchvision",
        "timm",
        "scipy",
    )
    versions: dict[str, str] = {}
    for name in dependency_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise Table1ContractError(f"missing reference dependency: {name}") from exc
    versions["python"] = sys.version.split()[0]
    versions["opencv_runtime"] = cv2.__version__
    versions["numpy_runtime"] = np.__version__
    versions["torch_runtime"] = torch.__version__
    if versions["pyiqa"] != "0.1.10":
        raise Table1ContractError(f"pyiqa must be 0.1.10, got {versions['pyiqa']}")
    if versions["basicsr"] != "1.4.2":
        raise Table1ContractError(f"basicsr must be 1.4.2, got {versions['basicsr']}")
    if versions["opencv_runtime"] != "4.9.0":
        raise Table1ContractError(
            f"OpenCV runtime must be 4.9.0, got {versions['opencv_runtime']}"
        )

    metric_configs = {
        name: json.loads(json.dumps(DEFAULT_CONFIGS[name])) for name in METRICS
    }
    executable = Path(sys.executable)
    return {
        "executable": _file_binding(executable.resolve(strict=True)),
        "invocation": {
            "sys_executable": str(executable),
            "sys_executable_realpath": str(executable.resolve(strict=True)),
            "sys_prefix": str(Path(sys.prefix)),
            "sys_base_prefix": str(Path(sys.base_prefix)),
        },
        "dependencies": versions,
        "metric_default_configs": metric_configs,
        "source_trees": {
            "pyiqa": _source_tree("pyiqa"),
            "basicsr": _source_tree("basicsr"),
            "timm": _source_tree("timm"),
            "torchvision": _source_tree("torchvision"),
        },
    }


def _patch_clip_download_root(cache_root: Path) -> None:
    """Route pyiqa's one hard-coded ``~/.cache/clip`` call to data disk."""

    from pyiqa.archs import clip_model, clipiqa_arch

    original = clip_model.load
    clip_root = cache_root / "clip"
    clip_root.mkdir(parents=True, exist_ok=True)

    def load_with_data_cache(
        name: str,
        device: Any = "cpu",
        jit: bool = False,
        download_root: str | None = None,
    ) -> Any:
        selected_root = str(clip_root) if download_root is None else download_root
        return original(name, device=device, jit=jit, download_root=selected_root)

    clipiqa_arch.load = load_with_data_cache


def _disable_network() -> None:
    """Deny Python-level network access in the formal score worker."""

    import socket

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise Table1ContractError(
            "network access is forbidden after the explicit weight-prefetch phase"
        )

    socket.create_connection = denied
    socket.getaddrinfo = denied
    socket.socket.connect = denied


def _load_official_scorer_module(source_path: Path) -> types.ModuleType:
    if sha256_file(source_path) != PINNED_SOURCE_SHA256["official_scorer"]:
        raise Table1ContractError(
            "worker scorer source no longer matches the pinned AgenticIR source"
        )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "scorer"
                for target in node.targets
            )
        )
    ]
    module = types.ModuleType("graphrestore_pinned_agenticir_scorer")
    module.__file__ = str(source_path)
    exec(compile(tree, str(source_path), "exec"), module.__dict__)
    return module


def _build_official_scorer(source_path: Path, *, device: str) -> Any:
    import pyiqa

    module = _load_official_scorer_module(source_path)
    scorer = module.Scorer.__new__(module.Scorer)
    scorer.fr_metric_name_lst = list(FULL_REFERENCE_METRICS)
    scorer.nr_metric_name_lst = list(NO_REFERENCE_METRICS)
    scorer.metric_name_lst = scorer.fr_metric_name_lst + scorer.nr_metric_name_lst
    scorer.fr_metrics = [
        pyiqa.create_metric(metric_name, device=device)
        for metric_name in scorer.fr_metric_name_lst
    ]
    scorer.nr_metrics = [
        pyiqa.create_metric(metric_name, device=device)
        for metric_name in scorer.nr_metric_name_lst
    ]
    scorer.metrics = scorer.fr_metrics + scorer.nr_metrics
    scorer.lower_better_dict = {
        metric.metric_name: metric.lower_better for metric in scorer.metrics
    }
    actual_names = tuple(metric.metric_name for metric in scorer.metrics)
    if actual_names != METRICS:
        raise Table1ContractError(f"unexpected pyiqa metric order: {actual_names}")
    return scorer


def _rng_state(*, include_cuda: bool) -> dict[str, Any]:
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": {
            "version": int(python_state[0]),
            "state": [int(item) for item in python_state[1]],
            "gauss_next": python_state[2],
        },
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "state": [int(item) for item in numpy_state[1].tolist()],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": [int(item) for item in torch.get_rng_state().tolist()],
    }
    if include_cuda:
        state["torch_cuda"] = [
            [int(item) for item in rng.tolist()]
            for rng in torch.cuda.get_rng_state_all()
        ]
    return state


def _rng_core(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: state[key] for key in ("python", "numpy", "torch_cpu")}


def _restore_rng_state(state: Mapping[str, Any], *, include_cuda: bool) -> None:
    import numpy as np
    import torch

    expected_keys = {"python", "numpy", "torch_cpu"}
    if include_cuda:
        expected_keys.add("torch_cuda")
    _require_exact_keys(state, expected_keys, label="RNG state")
    python_state = state["python"]
    numpy_state = state["numpy"]
    if not isinstance(python_state, Mapping) or not isinstance(numpy_state, Mapping):
        raise Table1ContractError("malformed RNG mapping")
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(item) for item in python_state["state"]),
            python_state["gauss_next"],
        )
    )
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    if include_cuda:
        torch.cuda.set_rng_state_all(
            [torch.tensor(item, dtype=torch.uint8) for item in state["torch_cuda"]]
        )


def _runtime_descriptor(device: str) -> dict[str, Any]:
    import torch

    backend_flags = {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    if device == "cpu":
        return {
            "device": "cpu",
            "torch_cuda": torch.version.cuda,
            "cuda_device": None,
            "visible_cuda_devices": [],
            "backend_flags": backend_flags,
            "network_access": False,
        }
    match = _CUDA_DEVICE_PATTERN.fullmatch(device)
    if match is None or not torch.cuda.is_available():
        raise Table1ContractError(f"requested CUDA device is unavailable: {device}")
    index = int(match.group(1) or 0)
    visible_devices = []
    for visible_index in range(torch.cuda.device_count()):
        visible_properties = torch.cuda.get_device_properties(visible_index)
        visible_devices.append(
            {
                "index": visible_index,
                "name": visible_properties.name,
                "total_memory": int(visible_properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(visible_index)),
            }
        )
    if index >= len(visible_devices):
        raise Table1ContractError(f"requested CUDA index is unavailable: {device}")
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": f"cuda:{index}",
        "torch_cuda": torch.version.cuda,
        "cuda_device": {
            "index": index,
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(index)),
        },
        "visible_cuda_devices": visible_devices,
        "backend_flags": backend_flags,
        "network_access": False,
    }


def _cache_environment(
    cache_root: Path,
    *,
    offline: bool,
    temporary_root: Path,
    cpu_only: bool,
    allocator: str | None = None,
) -> dict[str, str]:
    """Return a subprocess environment with every known cache on data disk."""

    environment = dict(os.environ)
    mappings = {
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch_extensions",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "TMPDIR": temporary_root,
    }
    for key, value in mappings.items():
        environment[key] = str(value)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONHASHSEED"] = "123"
    environment["CUDA_CACHE_DISABLE"] = "1"
    if allocator is not None:
        environment["PYTORCH_CUDA_ALLOC_CONF"] = allocator
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    else:
        environment.pop("HF_HUB_OFFLINE", None)
        environment.pop("TRANSFORMERS_OFFLINE", None)
    if cpu_only:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def _prepare_cache_directories(cache_root: Path) -> None:
    for relative in (
        "torch",
        "xdg",
        "huggingface/hub",
        "huggingface/datasets",
        "transformers",
        "matplotlib",
        "numba",
        "torch_extensions",
        "triton",
        "cuda",
        "clip",
    ):
        path = cache_root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o755)


def _cache_content_inventory(cache_root: Path) -> list[dict[str, Any]]:
    """Hash an unlocked cache without treating permission bits as content.

    This inventory exists solely for no-loss recovery after an interrupted
    prefetch.  Frozen modes may need to be reopened, but every directory,
    regular-file byte sequence, and internal symlink must remain identical.
    """

    root = cache_root.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for directory, dir_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        if directory_path != root:
            entries.append(
                {
                    "path": str(directory_path.relative_to(root)),
                    "type": "directory",
                }
            )
        retained_dirs: list[str] = []
        for name in sorted(dir_names):
            path = directory_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise Table1ContractError(
                        f"partial-cache symlink escapes cache root: {path}"
                    ) from exc
                if not resolved.is_file():
                    raise Table1ContractError(
                        f"partial-cache symlink does not resolve to a file: {path}"
                    )
                entries.append(
                    {
                        "path": str(path.relative_to(root)),
                        "type": "symlink",
                        "target": os.readlink(path),
                        "resolved_path": str(resolved.relative_to(root)),
                        "resolved_sha256": sha256_file(resolved),
                        "resolved_size": int(resolved.stat().st_size),
                    }
                )
            elif stat.S_ISDIR(info.st_mode):
                retained_dirs.append(name)
            else:
                raise Table1ContractError(
                    f"unsupported partial-cache entry type: {path}"
                )
        dir_names[:] = retained_dirs
        for name in sorted(file_names):
            path = directory_path / name
            relative = str(path.relative_to(root))
            if relative == "weights_lock.json" or name.startswith(
                ".weights_lock.json."
            ):
                raise Table1ContractError(
                    f"unexpected partial/final weights-lock artifact: {path}"
                )
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise Table1ContractError(
                        f"partial-cache symlink escapes cache root: {path}"
                    ) from exc
                if not resolved.is_file():
                    raise Table1ContractError(
                        f"partial-cache symlink does not resolve to a file: {path}"
                    )
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                        "resolved_path": str(resolved.relative_to(root)),
                        "resolved_sha256": sha256_file(resolved),
                        "resolved_size": int(resolved.stat().st_size),
                    }
                )
            elif stat.S_ISREG(info.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "sha256": sha256_file(path),
                        "size": int(info.st_size),
                    }
                )
            else:
                raise Table1ContractError(
                    f"unsupported partial-cache entry type: {path}"
                )
    entries.sort(key=lambda item: (str(item["path"]), str(item["type"])))
    return entries


def _recover_unlocked_cache_for_prefetch(
    cache_root: Path, *, data_root: Path = DATA_DISK_ROOT
) -> dict[str, Any]:
    """Reopen an interrupted, lock-free cache without deleting any bytes."""

    cache_preexisting = cache_root.exists()
    cache_root = _ensure_secure_directory(cache_root, data_root=data_root)
    lock_path = cache_root / "weights_lock.json"
    if lock_path.exists() or lock_path.is_symlink():
        raise Table1ContractError(
            "unlocked-cache recovery refuses a present weights_lock.json"
        )
    before = _cache_content_inventory(cache_root)
    modes_reopened = 0
    for directory, dir_names, file_names in os.walk(
        cache_root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        if _mode(directory_path) != 0o755:
            os.chmod(directory_path, 0o755)
            modes_reopened += 1
        retained_dirs: list[str] = []
        for name in sorted(dir_names):
            path = directory_path / name
            if path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise Table1ContractError(
                    f"unsupported partial-cache entry type: {path}"
                )
            if stat.S_IMODE(info.st_mode) != 0o755:
                os.chmod(path, 0o755)
                modes_reopened += 1
            retained_dirs.append(name)
        dir_names[:] = retained_dirs
        for name in sorted(file_names):
            path = directory_path / name
            if path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise Table1ContractError(
                    f"unsupported partial-cache entry type: {path}"
                )
            if stat.S_IMODE(info.st_mode) != 0o644:
                os.chmod(path, 0o644)
                modes_reopened += 1
    after = _cache_content_inventory(cache_root)
    if after != before:
        raise Table1ContractError(
            "partial-cache mode recovery changed cache paths or bytes"
        )
    regular_files = [entry for entry in before if entry["type"] == "file"]
    return {
        "cache_preexisting": cache_preexisting,
        "entry_count": len(before),
        "regular_file_count": len(regular_files),
        "symlink_count": sum(entry["type"] == "symlink" for entry in before),
        "total_regular_bytes": sum(int(entry["size"]) for entry in regular_files),
        "content_sha256_before": sha256_json(before),
        "content_sha256_after": sha256_json(after),
        "modes_reopened": modes_reopened,
        "no_paths_or_bytes_removed_or_modified": True,
    }


def _validate_partial_cache_recovery(value: object) -> None:
    if not isinstance(value, Mapping):
        raise Table1ContractError("partial-cache recovery receipt is not an object")
    _require_exact_keys(
        value,
        {
            "cache_preexisting",
            "entry_count",
            "regular_file_count",
            "symlink_count",
            "total_regular_bytes",
            "content_sha256_before",
            "content_sha256_after",
            "modes_reopened",
            "no_paths_or_bytes_removed_or_modified",
        },
        label="partial-cache recovery receipt",
    )
    for key in (
        "entry_count",
        "regular_file_count",
        "symlink_count",
        "total_regular_bytes",
        "modes_reopened",
    ):
        if type(value[key]) is not int or value[key] < 0:
            raise Table1ContractError(
                f"partial-cache recovery {key} must be a nonnegative integer"
            )
    if type(value["cache_preexisting"]) is not bool:
        raise Table1ContractError(
            "partial-cache recovery cache_preexisting must be boolean"
        )
    before = _require_sha256(
        value["content_sha256_before"],
        label="partial-cache recovery before hash",
    )
    after = _require_sha256(
        value["content_sha256_after"],
        label="partial-cache recovery after hash",
    )
    if before != after or value["no_paths_or_bytes_removed_or_modified"] is not True:
        raise Table1ContractError(
            "partial-cache recovery does not prove path/byte preservation"
        )


def _cache_inventory(cache_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    root = cache_root.resolve(strict=True)
    for directory, dir_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        retained_dirs: list[str] = []
        for name in sorted(dir_names):
            path = directory_path / name
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise Table1ContractError(
                        f"cache symlink escapes cache root: {path}"
                    ) from exc
                entries.append(
                    {
                        "path": str(path.relative_to(root)),
                        "type": "symlink",
                        "target": os.readlink(path),
                        "resolved_sha256": sha256_file(resolved),
                        "resolved_size": resolved.stat().st_size,
                    }
                )
            else:
                info = path.lstat()
                entries.append(
                    {
                        "path": str(path.relative_to(root)),
                        "type": "directory",
                        "mode": stat.S_IMODE(info.st_mode),
                    }
                )
                retained_dirs.append(name)
        dir_names[:] = retained_dirs
        for name in sorted(file_names):
            path = directory_path / name
            relative = str(path.relative_to(root))
            if relative == "weights_lock.json":
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise Table1ContractError(
                        f"cache symlink escapes cache root: {path}"
                    ) from exc
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                        "resolved_sha256": sha256_file(resolved),
                        "resolved_size": resolved.stat().st_size,
                    }
                )
            elif stat.S_ISREG(info.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "sha256": sha256_file(path),
                        "size": int(info.st_size),
                        "mode": stat.S_IMODE(info.st_mode),
                    }
                )
            else:
                raise Table1ContractError(f"unsupported cache entry type: {path}")
    entries.sort(key=lambda item: (str(item["path"]), str(item["type"])))
    return entries


def _validate_cache_inventory_contract(entries: object) -> None:
    if not isinstance(entries, list) or not entries:
        raise Table1ContractError("weight cache inventory must be a non-empty list")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise Table1ContractError(f"cache inventory entry {index} is not an object")
        entry_type = entry.get("type")
        path_text = entry.get("path")
        if not isinstance(path_text, str) or not path_text:
            raise Table1ContractError(f"cache inventory entry {index} has no path")
        relative = Path(path_text)
        if relative.is_absolute() or ".." in relative.parts or path_text in seen:
            raise Table1ContractError(
                f"unsafe/duplicate cache inventory path: {path_text}"
            )
        seen.add(path_text)
        if entry_type == "file":
            _require_exact_keys(
                entry,
                {"path", "type", "sha256", "size", "mode"},
                label=f"cache file entry {path_text}",
            )
            _require_sha256(entry["sha256"], label=f"cache file {path_text}")
            if entry["mode"] != 0o444 or not isinstance(entry["size"], int):
                raise Table1ContractError(
                    f"cache file is not frozen/valid: {path_text}"
                )
        elif entry_type == "directory":
            _require_exact_keys(
                entry,
                {"path", "type", "mode"},
                label=f"cache directory entry {path_text}",
            )
            if entry["mode"] != 0o555:
                raise Table1ContractError(f"cache directory is not frozen: {path_text}")
        elif entry_type == "symlink":
            _require_exact_keys(
                entry,
                {"path", "type", "target", "resolved_sha256", "resolved_size"},
                label=f"cache symlink entry {path_text}",
            )
            _require_sha256(
                entry["resolved_sha256"], label=f"cache symlink {path_text}"
            )
            if not isinstance(entry["target"], str) or not isinstance(
                entry["resolved_size"], int
            ):
                raise Table1ContractError(f"invalid cache symlink: {path_text}")
        else:
            raise Table1ContractError(
                f"unsupported cache inventory type at {path_text}: {entry_type!r}"
            )


def _freeze_cache(cache_root: Path) -> None:
    root = cache_root.resolve(strict=True)
    for directory, dir_names, file_names in os.walk(
        root, topdown=False, followlinks=False
    ):
        directory_path = Path(directory)
        for name in file_names:
            path = directory_path / name
            if not path.is_symlink():
                os.chmod(path, 0o444)
        for name in dir_names:
            path = directory_path / name
            if not path.is_symlink():
                os.chmod(path, 0o555)
        os.chmod(directory_path, 0o555)


def _verify_frozen_cache_directories(cache_root: Path) -> None:
    for directory, dir_names, _file_names in os.walk(
        cache_root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        if _mode(directory_path) != 0o555:
            raise Table1ContractError(
                f"cache directory is not frozen 0555: {directory_path} ({_mode(directory_path):04o})"
            )
        dir_names[:] = [
            name for name in dir_names if not (directory_path / name).is_symlink()
        ]


def _run_json_worker(
    reference_python: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
    work_parent: Path,
) -> dict[str, Any]:
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="agenticir-table1-", dir=work_parent
    ) as temporary:
        result_path = Path(temporary) / "result.json"
        command = [
            str(reference_python),
            str(DEFAULT_CLI_PATH),
            *arguments,
            "--worker-result",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            stdout_tail = completed.stdout[-4000:]
            stderr_tail = completed.stderr[-8000:]
            raise Table1ContractError(
                f"reference worker exited {completed.returncode}\n"
                f"stdout tail:\n{stdout_tail}\nstderr tail:\n{stderr_tail}"
            )
        result = _load_json_strict(result_path)
        if not isinstance(result, dict):
            raise Table1ContractError("reference worker result is not an object")
        return result


def _inspect_reference(
    reference_python: Path,
    cache_root: Path,
    *,
    work_parent: Path,
) -> dict[str, Any]:
    environment = _cache_environment(
        cache_root,
        offline=True,
        temporary_root=work_parent,
        cpu_only=True,
    )
    return _run_json_worker(
        reference_python,
        ["_worker-inspect"],
        environment=environment,
        work_parent=work_parent,
    )


def prefetch_weights(
    *,
    reference_python: Path,
    cache_root: Path,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    cache_root = _require_data_disk(cache_root, label="cache root")
    launcher_binding = _reference_launcher_binding(reference_python)
    source_bindings = validate_pinned_sources(source_paths)
    lock_path = cache_root / "weights_lock.json"
    if lock_path.exists():
        return check_cache(
            reference_python=reference_python,
            cache_root=cache_root,
            source_paths=source_paths,
        )

    recovery = _recover_unlocked_cache_for_prefetch(cache_root)
    _prepare_cache_directories(cache_root)
    work_parent = cache_root.parent / ".agenticir_table1_prefetch_work"
    work_parent.mkdir(parents=True, exist_ok=True)
    environment = _cache_environment(
        cache_root,
        offline=False,
        temporary_root=work_parent,
        cpu_only=True,
    )
    result = _run_json_worker(
        reference_python,
        [
            "_worker-prefetch",
            "--cache-root",
            str(cache_root),
            "--official-scorer",
            str(Path(source_paths["official_scorer"]).resolve(strict=True)),
        ],
        environment=environment,
        work_parent=work_parent,
    )
    _require_exact_keys(
        result,
        {
            "schema_version",
            "reference_environment",
            "metric_runtime",
            "initial_rng_core",
            "initial_rng_core_sha256",
            "cuda_initialized",
        },
        label="prefetch worker result",
    )
    if (
        result["schema_version"] != WORKER_RESULT_SCHEMA
        or result["cuda_initialized"] is not False
    ):
        raise Table1ContractError("prefetch worker violated its CPU-only contract")
    if result["metric_runtime"] != EXPECTED_METRIC_RUNTIME:
        raise Table1ContractError("prefetch worker metric runtime mismatch")
    if sha256_json(result["initial_rng_core"]) != result["initial_rng_core_sha256"]:
        raise Table1ContractError("prefetch worker RNG binding mismatch")
    _validate_reference_environment_for_launcher(
        result["reference_environment"], launcher_binding
    )
    if _reference_launcher_binding(reference_python) != launcher_binding:
        raise Table1ContractError("reference launcher changed during prefetch")

    # Freeze cache contents before hashing them into the lock.  Temporarily
    # reopen only the root directory so the excluded lock can be published;
    # every nested directory stays 0555 and every regular file stays 0444.
    _freeze_cache(cache_root)
    os.chmod(cache_root, 0o755)
    inventory = _cache_inventory(cache_root)
    _validate_cache_inventory_contract(inventory)
    lock = {
        "schema_version": WEIGHTS_LOCK_SCHEMA,
        "created_utc": utc_now_iso(),
        "cache_root": str(cache_root),
        "agenticir_commit": PINNED_AGENTICIR_COMMIT,
        "agenticir_sources": source_bindings,
        "reference_launcher": launcher_binding,
        "reference_environment": result["reference_environment"],
        "metric_runtime": result["metric_runtime"],
        "initial_rng_core": result["initial_rng_core"],
        "initial_rng_core_sha256": result["initial_rng_core_sha256"],
        "cache_policy": {
            "system_cache_writes": False,
            "score_network_access": False,
            "regular_file_mode": 292,
            "directory_mode": 365,
            "clip_download_root": str(cache_root / "clip"),
        },
        "partial_cache_recovery": recovery,
        "weights": inventory,
    }
    _atomic_create_json(lock_path, lock)
    os.chmod(cache_root, 0o555)
    fsync_directory(cache_root.parent)
    return check_cache(
        reference_python=reference_python,
        cache_root=cache_root,
        source_paths=source_paths,
    )


def check_cache(
    *,
    reference_python: Path,
    cache_root: Path,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    cache_root = _require_data_disk(cache_root, label="cache root")
    launcher_binding = _reference_launcher_binding(reference_python)
    source_bindings = validate_pinned_sources(source_paths)
    lock_path = cache_root / "weights_lock.json"
    _require_regular_file(lock_path, immutable=True)
    lock = _load_json_strict(lock_path)
    if not isinstance(lock, dict):
        raise Table1ContractError("weights lock is not an object")
    _require_exact_keys(
        lock,
        {
            "schema_version",
            "created_utc",
            "cache_root",
            "agenticir_commit",
            "agenticir_sources",
            "reference_launcher",
            "reference_environment",
            "metric_runtime",
            "initial_rng_core",
            "initial_rng_core_sha256",
            "cache_policy",
            "partial_cache_recovery",
            "weights",
        },
        label="weights lock",
    )
    if lock["schema_version"] != WEIGHTS_LOCK_SCHEMA:
        raise Table1ContractError("bad weights lock schema")
    if lock["cache_root"] != str(cache_root):
        raise Table1ContractError("weights lock cache_root mismatch")
    if lock["agenticir_commit"] != PINNED_AGENTICIR_COMMIT:
        raise Table1ContractError("weights lock AgenticIR commit mismatch")
    if lock["agenticir_sources"] != source_bindings:
        raise Table1ContractError("weights lock pinned-source binding mismatch")
    if lock["reference_launcher"] != launcher_binding:
        raise Table1ContractError("weights lock reference-launcher binding mismatch")
    if sha256_json(lock["initial_rng_core"]) != lock["initial_rng_core_sha256"]:
        raise Table1ContractError("weights lock initial RNG hash mismatch")
    expected_policy = {
        "system_cache_writes": False,
        "score_network_access": False,
        "regular_file_mode": 292,
        "directory_mode": 365,
        "clip_download_root": str(cache_root / "clip"),
    }
    if lock["cache_policy"] != expected_policy:
        raise Table1ContractError("weights lock cache policy mismatch")
    if lock["metric_runtime"] != EXPECTED_METRIC_RUNTIME:
        raise Table1ContractError("weights lock metric runtime mismatch")
    _validate_partial_cache_recovery(lock["partial_cache_recovery"])
    _validate_cache_inventory_contract(lock["weights"])
    current_inventory = _cache_inventory(cache_root)
    if current_inventory != lock["weights"]:
        raise Table1ContractError("weight cache inventory/hash/mode mismatch")
    _verify_frozen_cache_directories(cache_root)

    work_parent = cache_root.parent / ".agenticir_table1_check_work"
    environment = _inspect_reference(
        reference_python,
        cache_root,
        work_parent=work_parent,
    )
    if environment != lock["reference_environment"]:
        raise Table1ContractError(
            "reference dependency/source environment differs from weights lock"
        )
    _validate_reference_environment_for_launcher(environment, launcher_binding)
    if _reference_launcher_binding(reference_python) != launcher_binding:
        raise Table1ContractError("reference launcher changed during cache check")
    if _cache_inventory(cache_root) != lock["weights"]:
        raise Table1ContractError(
            "reference inspection mutated the frozen weight cache"
        )
    return {
        "path": str(lock_path.resolve(strict=True)),
        "sha256": sha256_file(lock_path),
        "mode": _mode(lock_path),
        "lock": lock,
    }


def _ensure_semantic_json(
    path: Path,
    semantic: Mapping[str, Any],
    *,
    confinement_root: Path | None = None,
) -> dict[str, Any]:
    """Create a JSON artifact with one volatile ``created_utc`` field or verify it."""

    if path.exists():
        _require_regular_file(path, immutable=True)
        actual = _load_json_strict(path)
        if not isinstance(actual, dict) or set(actual) != {*semantic, "created_utc"}:
            raise Table1ContractError(f"published artifact schema mismatch: {path}")
        comparable = dict(actual)
        created = comparable.pop("created_utc")
        if (
            not isinstance(created, str)
            or not created.endswith("Z")
            or comparable != dict(semantic)
        ):
            raise Table1ContractError(f"published artifact content mismatch: {path}")
        return actual
    value = {**semantic, "created_utc": utc_now_iso()}
    _atomic_create_json(path, value, confinement_root=confinement_root)
    return value


def _implementation_bindings() -> dict[str, dict[str, Any]]:
    return {
        "table1_scorer_module": _file_binding(Path(__file__).resolve()),
        "table1_scorer_cli": _file_binding(DEFAULT_CLI_PATH.resolve(strict=True)),
    }


def _score_row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in INPUT_KEYS}


def _validate_score_value(value: object, *, metric: str, sample_id: str) -> float:
    if isinstance(value, bool):
        raise Table1ContractError(f"{sample_id}/{metric}: boolean is not a metric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise Table1ContractError(
            f"{sample_id}/{metric}: invalid metric {value!r}"
        ) from exc
    if not math.isfinite(numeric):
        raise Table1ContractError(f"{sample_id}/{metric}: non-finite metric")
    return numeric


def _shard_path(shards_dir: Path, index: int) -> Path:
    return shards_dir / f"shard-{index:05d}.json"


def scan_score_shards(
    *,
    shards_dir: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    shard_size: int,
    run_contract_sha256: str,
    input_lock_sha256: str,
    initial_rng_core: Mapping[str, Any],
    confinement_root: Path | None = None,
) -> dict[str, Any]:
    if shard_size <= 0:
        raise Table1ContractError("shard_size must be positive")
    if confinement_root is None:
        shards_dir.mkdir(parents=True, exist_ok=True)
    else:
        _ensure_secure_directory(shards_dir)
        _assert_confined_write_path(shards_dir / "probe", confinement_root)
    expected_shards = (len(expected_rows) + shard_size - 1) // shard_size
    visible = {
        path.name for path in shards_dir.iterdir() if not path.name.startswith(".")
    }
    allowed = {_shard_path(shards_dir, index).name for index in range(expected_shards)}
    unexpected = visible - allowed
    if unexpected:
        raise Table1ContractError(
            f"unexpected score shard artifacts: {sorted(unexpected)}"
        )

    all_rows: list[dict[str, Any]] = []
    previous_rng: Mapping[str, Any] | None = None
    runtime: Mapping[str, Any] | None = None
    shard_peaks: list[dict[str, Any]] = []
    first_missing: int | None = None
    for shard_index in range(expected_shards):
        path = _shard_path(shards_dir, shard_index)
        if not path.exists():
            if first_missing is None:
                first_missing = shard_index
            continue
        if first_missing is not None:
            raise Table1ContractError(
                f"non-contiguous shard set: shard {shard_index} exists after missing shard {first_missing}"
            )
        _require_regular_file(path, immutable=True)
        payload = _load_json_strict(path)
        if not isinstance(payload, dict):
            raise Table1ContractError(f"shard is not an object: {path}")
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "shard_index",
                "start_index",
                "end_index",
                "run_contract_sha256",
                "input_lock_sha256",
                "runtime",
                "rng_before",
                "rng_before_sha256",
                "rng_after",
                "rng_after_sha256",
                "peak_reserved_bytes",
                "total_memory_bytes",
                "peak_reserved_fraction",
                "rows",
            },
            label=f"shard {shard_index}",
        )
        start = shard_index * shard_size
        end = min(start + shard_size, len(expected_rows))
        if (
            payload["schema_version"] != SHARD_SCHEMA
            or payload["shard_index"] != shard_index
            or payload["start_index"] != start
            or payload["end_index"] != end
            or payload["run_contract_sha256"] != run_contract_sha256
            or payload["input_lock_sha256"] != input_lock_sha256
        ):
            raise Table1ContractError(f"shard metadata mismatch: {path}")
        if sha256_json(payload["rng_before"]) != payload["rng_before_sha256"]:
            raise Table1ContractError(f"shard rng_before hash mismatch: {path}")
        if sha256_json(payload["rng_after"]) != payload["rng_after_sha256"]:
            raise Table1ContractError(f"shard rng_after hash mismatch: {path}")
        peak = _validate_peak_reserved(
            payload["peak_reserved_bytes"], payload["total_memory_bytes"]
        )
        stored_peak_fraction = payload["peak_reserved_fraction"]
        if (
            isinstance(stored_peak_fraction, bool)
            or not isinstance(stored_peak_fraction, (int, float))
            or not math.isfinite(float(stored_peak_fraction))
            or float(stored_peak_fraction) != peak["peak_reserved_fraction"]
        ):
            raise Table1ContractError(f"shard CUDA peak fraction mismatch: {path}")
        shard_peaks.append({"shard_index": shard_index, **peak})
        if shard_index == 0:
            if _rng_core(payload["rng_before"]) != dict(initial_rng_core):
                raise Table1ContractError(
                    "first shard did not start from the locked seed-123 state"
                )
        elif payload["rng_before"] != previous_rng:
            raise Table1ContractError(f"RNG chain break before shard {shard_index}")
        previous_rng = payload["rng_after"]
        if not isinstance(payload["runtime"], Mapping):
            raise Table1ContractError(f"shard runtime is not an object: {path}")
        if runtime is None:
            runtime = payload["runtime"]
        elif payload["runtime"] != runtime:
            raise Table1ContractError(f"runtime changed at shard {shard_index}")

        shard_rows = payload["rows"]
        if not isinstance(shard_rows, list) or len(shard_rows) != end - start:
            raise Table1ContractError(f"shard row count mismatch: {path}")
        for offset, score_row in enumerate(shard_rows):
            absolute_index = start + offset
            if not isinstance(score_row, dict):
                raise Table1ContractError(
                    f"shard row {absolute_index} is not an object"
                )
            _require_exact_keys(
                score_row, SCORE_ROW_KEYS, label=f"score row {absolute_index}"
            )
            if _score_row_identity(score_row) != _score_row_identity(
                expected_rows[absolute_index]
            ):
                raise Table1ContractError(
                    f"score row identity mismatch at {absolute_index}"
                )
            canonical_score = dict(_score_row_identity(score_row))
            for metric in METRICS:
                canonical_score[metric] = _validate_score_value(
                    score_row[metric],
                    metric=metric,
                    sample_id=str(score_row["sample_id"]),
                )
            all_rows.append(canonical_score)
    completed = first_missing if first_missing is not None else expected_shards
    return {
        "records": all_rows,
        "completed_shards": completed,
        "expected_shards": expected_shards,
        "next_index": completed * shard_size,
        "previous_rng": previous_rng,
        "runtime": runtime,
        "shard_peaks": shard_peaks,
        "maximum_peak_reserved_fraction": max(
            (item["peak_reserved_fraction"] for item in shard_peaks), default=0.0
        ),
    }


def aggregate_table1_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> dict[str, Any]:
    _validate_expected_counts(expected_counts)
    expected_total = sum(int(value) for value in expected_counts.values())
    if len(records) != expected_total:
        raise Table1ContractError(
            f"expected {expected_total} score records, got {len(records)}"
        )
    buckets: dict[str, dict[str, list[float]]] = {
        combination: {metric: [] for metric in METRICS}
        for combinations in OFFICIAL_GROUPS.values()
        for combination in combinations
        if combination in expected_counts
    }
    for index, row in enumerate(records):
        combination = str(row.get("combination", ""))
        if combination not in buckets:
            raise Table1ContractError(
                f"score row {index}: unexpected combination {combination!r}"
            )
        for metric in METRICS:
            buckets[combination][metric].append(
                _validate_score_value(
                    row.get(metric),
                    metric=metric,
                    sample_id=str(row.get("sample_id", index)),
                )
            )

    combinations_result: dict[str, dict[str, Any]] = {}
    for group, combinations in OFFICIAL_GROUPS.items():
        for combination in combinations:
            if combination not in expected_counts:
                continue
            actual_count = len(buckets[combination][METRICS[0]])
            wanted_count = int(expected_counts[combination])
            if actual_count != wanted_count:
                raise Table1ContractError(
                    f"{combination}: expected {wanted_count} scores, got {actual_count}"
                )
            combinations_result[combination] = {
                "group": group,
                "count": actual_count,
                **{
                    metric: math.fsum(buckets[combination][metric]) / actual_count
                    for metric in METRICS
                },
            }

    groups_result: dict[str, dict[str, Any]] = {}
    for group, combinations in OFFICIAL_GROUPS.items():
        selected = [name for name in combinations if name in expected_counts]
        if not selected:
            continue
        groups_result[group] = {
            "combination_count": len(selected),
            "image_count": sum(
                int(combinations_result[name]["count"]) for name in selected
            ),
            **{
                metric: math.fsum(
                    float(combinations_result[name][metric]) for name in selected
                )
                / len(selected)
                for metric in METRICS
            },
        }
    return {
        "image_count": len(records),
        "combinations": combinations_result,
        "groups": groups_result,
        "aggregation": (
            "per-image score -> arithmetic mean within each combination -> "
            "equal arithmetic mean of combination means within each group"
        ),
    }


def _per_image_csv(records: Sequence[Mapping[str, Any]]) -> str:
    destination = io.StringIO(newline="")
    fieldnames = [
        "sample_id",
        "group",
        "combination",
        "prediction_png",
        "prediction_sha256",
        "target_png",
        "target_sha256",
        *METRICS,
    ]
    writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in records:
        writer.writerow({field: row[field] for field in fieldnames})
    return destination.getvalue()


def _publish_or_verify_text(
    path: Path,
    payload: str,
    *,
    confinement_root: Path | None = None,
) -> None:
    if confinement_root is not None:
        _assert_confined_write_path(path, confinement_root)
    if path.exists():
        _require_regular_file(path, immutable=True)
        if path.read_text(encoding="utf-8") != payload:
            raise Table1ContractError(f"published artifact content mismatch: {path}")
        return
    _atomic_create_text(path, payload, confinement_root=confinement_root)


def _publish_or_verify_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    confinement_root: Path | None = None,
) -> None:
    payload = _canonical_json(value)
    _publish_or_verify_text(path, payload, confinement_root=confinement_root)


def _write_worker_request(
    path: Path,
    value: Mapping[str, Any],
    *,
    confinement_root: Path | None = None,
) -> None:
    _publish_or_verify_json(path, value, confinement_root=confinement_root)


def _formal_expected_authorization_bindings() -> dict[str, Path]:
    return {
        "table1_scorer_module": Path(__file__).resolve(),
        "table1_scorer_cli": DEFAULT_CLI_PATH.resolve(strict=True),
        "metric_weight_inventory": DEFAULT_CACHE_ROOT / "weights_lock.json",
    }


def _binding_from_payload(
    raw: object,
    *,
    label: str,
    expected_path: Path | None = None,
    require_read_only: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
        raise Table1ContractError(f"{label} binding must contain only path/sha256")
    path_value = raw.get("path")
    digest = _require_sha256(raw.get("sha256"), label=f"{label} sha256")
    if not isinstance(path_value, str):
        raise Table1ContractError(f"{label} path must be text")
    path = Path(path_value)
    if expected_path is not None and path != expected_path:
        raise Table1ContractError(
            f"{label} path drifted: expected {expected_path}, got {path}"
        )
    binding = _file_binding(path)
    if binding["sha256"] != digest:
        raise Table1ContractError(f"{label} hash drifted")
    if require_read_only and binding["mode"] & 0o222:
        raise Table1ContractError(f"{label} must be read-only")
    return binding


def _load_formal_evidence() -> dict[str, Any]:
    """Validate and serialize the whole formal-evaluator evidence chain.

    The evaluator owns the semantic validation helper.  This adapter only
    converts its accepted immutable artifacts into an exact, JSON-stable
    scorer binding; it deliberately does not trust worker-request fields.
    """

    try:
        from src.evaluation import mio100

        validate_complete = getattr(mio100, "validate_formal_evaluator_complete")
    except (ImportError, AttributeError) as exc:
        raise Table1ContractError(
            "formal evaluator completion validator is unavailable"
        ) from exc
    expected_bindings = _formal_expected_authorization_bindings()
    try:
        authorization = mio100.validate_formal_authorization(
            FORMAL_AUTHORIZATION_PATH,
            expected_bindings=expected_bindings,
        )
        completion = validate_complete(
            FORMAL_EVALUATOR_COMPLETE_PATH,
            authorization_path=FORMAL_AUTHORIZATION_PATH,
            expected_bindings=expected_bindings,
            # The scorer is authorized to consume only the evaluator's frozen
            # prediction/target mapping.  Native MiO100 LQ bytes are not score
            # inputs; the bound inventory/digests are still cross-validated.
            verify_data_files=False,
        )
    except Exception as exc:
        if isinstance(exc, Table1ContractError):
            raise
        raise Table1ContractError(
            f"formal evaluator evidence validation failed: {exc}"
        ) from exc
    if completion is None or not (
        is_dataclass(completion) or isinstance(completion, Mapping)
    ):
        raise Table1ContractError(
            "formal evaluator completion helper returned no evidence"
        )
    helper_evidence = (
        completion.get("evidence")
        if isinstance(completion, Mapping)
        else getattr(completion, "evidence", None)
    )
    helper_keys = {
        "authorization",
        "evaluator_complete",
        "run_contract",
        "summary",
        "per_image",
        "table1_input",
        "checkpoint",
        "manifest",
        "formal_data_inventory",
        "predictions_digest",
    }
    if not isinstance(helper_evidence, Mapping):
        raise Table1ContractError("formal evaluator helper lacks stable evidence")
    _require_exact_keys(helper_evidence, helper_keys, label="evaluator helper evidence")
    if not hasattr(authorization, "bindings"):
        raise Table1ContractError("formal authorization helper returned no bindings")
    auth_bindings = authorization.bindings

    def authorization_binding(name: str) -> dict[str, str]:
        value = auth_bindings.get(name)
        if value is None or not hasattr(value, "path") or not hasattr(value, "sha256"):
            raise Table1ContractError(f"formal authorization lacks binding {name}")
        binding = _file_binding(Path(value.path))
        if binding["sha256"] != value.sha256:
            raise Table1ContractError(f"formal authorization binding drifted: {name}")
        return {"path": binding["path"], "sha256": binding["sha256"]}

    parity = authorization_binding("metric_parity_summary")
    expected_paths = {
        "authorization": FORMAL_AUTHORIZATION_PATH,
        "evaluator_complete": FORMAL_EVALUATOR_COMPLETE_PATH,
        "run_contract": FORMAL_EVALUATOR_ROOT / "run_contract.json",
        "summary": FORMAL_EVALUATOR_ROOT / "summary.json",
        "per_image": FORMAL_EVALUATOR_ROOT / "per_image.csv",
        "table1_input": FORMAL_TABLE1_INPUT_PATH,
        "checkpoint": auth_bindings["stage4_checkpoint"].path,
        "manifest": auth_bindings["formal_manifest"].path,
        "formal_data_inventory": auth_bindings["formal_data_inventory"].path,
    }
    evidence: dict[str, Any] = {}
    for label, expected_path in expected_paths.items():
        verified = _binding_from_payload(
            helper_evidence[label],
            label=f"evaluator helper {label}",
            expected_path=Path(expected_path),
            require_read_only=label
            in {
                "authorization",
                "evaluator_complete",
                "run_contract",
                "summary",
                "per_image",
                "table1_input",
                "formal_data_inventory",
            },
        )
        evidence[label] = {
            "path": verified["path"],
            "sha256": verified["sha256"],
        }
    evidence["metric_parity_summary"] = parity
    evidence["predictions_digest"] = _require_sha256(
        helper_evidence["predictions_digest"], label="formal predictions digest"
    )
    _require_exact_keys(evidence, _FORMAL_EVIDENCE_KEYS, label="formal evidence")
    return evidence


def _revalidate_formal_evidence(
    expected: Mapping[str, Any],
    *,
    loader: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _require_exact_keys(expected, _FORMAL_EVIDENCE_KEYS, label="formal evidence")
    if loader is None:
        loader = _load_formal_evidence
    current = dict(loader())
    _require_exact_keys(current, _FORMAL_EVIDENCE_KEYS, label="current formal evidence")
    if current != dict(expected):
        raise Table1ContractError("formal evidence binding changed")
    return current


def _assert_cuda_uninitialized(*, label: str) -> None:
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        raise Table1ContractError(f"CUDA initialized before {label}")


def _query_gpu_compute_pids(
    *,
    device_index: int = 0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> set[int]:
    completed = runner(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise Table1ContractError(
            f"cannot audit GPU ownership: {completed.stderr.strip()}"
        )
    pids: set[int] = set()
    for raw in completed.stdout.splitlines():
        value = raw.strip()
        if not value:
            continue
        if not value.isdecimal() or int(value) <= 0:
            raise Table1ContractError(f"malformed nvidia-smi compute PID: {value!r}")
        pids.add(int(value))
    return pids


def _assert_gpu_ownership(
    expected_pids: set[int],
    *,
    device_index: int = 0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    actual = _query_gpu_compute_pids(device_index=device_index, runner=runner)
    if actual != expected_pids:
        raise Table1ContractError(
            f"GPU {device_index} ownership mismatch: expected={sorted(expected_pids)}, "
            f"actual={sorted(actual)}"
        )


def _validate_peak_reserved(
    peak_reserved_bytes: object,
    total_memory_bytes: object,
) -> dict[str, Any]:
    if (
        isinstance(peak_reserved_bytes, bool)
        or not isinstance(peak_reserved_bytes, int)
        or peak_reserved_bytes < 0
        or isinstance(total_memory_bytes, bool)
        or not isinstance(total_memory_bytes, int)
        or total_memory_bytes <= 0
    ):
        raise Table1ContractError("invalid CUDA peak/total memory counters")
    fraction = peak_reserved_bytes / total_memory_bytes
    if not math.isfinite(fraction) or fraction >= MAXIMUM_VRAM_RESERVED_FRACTION:
        raise Table1ContractError(
            f"score-shard peak reserved fraction {fraction:.12f} is not below "
            f"{MAXIMUM_VRAM_RESERVED_FRACTION:.2f}"
        )
    return {
        "peak_reserved_bytes": peak_reserved_bytes,
        "total_memory_bytes": total_memory_bytes,
        "peak_reserved_fraction": fraction,
    }


_EVALUATOR_PER_IMAGE_FIELDS = (
    "sample_id",
    "group",
    "combination",
    "clean_id",
    "prediction_png",
    "prediction_sha256",
    "target_png",
    "target_sha256",
    "psnr",
    "ssim",
    "latency_ms",
    "program_levels",
    "parallel_levels",
    "active_skill_calls",
    "reentry_requests",
    "unexpected_activations",
    "precycle_graphs",
    "dropped_edges",
    "peak_reserved_fraction",
)


def crosscheck_evaluator_psnr_ssim(
    records: Sequence[Mapping[str, Any]],
    *,
    evaluator_csv: Path,
    metric_parity_summary: Path,
    predictions_digest: str,
    expected_count: int = EXPECTED_IMAGE_COUNT,
) -> dict[str, Any]:
    """Cross-check scorer FR metrics against frozen inference CSV per image."""

    _require_regular_file(evaluator_csv, immutable=True)
    _require_regular_file(metric_parity_summary)
    scorer_by_id: dict[str, Mapping[str, Any]] = {}
    for row in records:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in scorer_by_id:
            raise Table1ContractError(
                f"duplicate/empty scorer sample_id: {sample_id!r}"
            )
        scorer_by_id[sample_id] = row
    try:
        with evaluator_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _EVALUATOR_PER_IMAGE_FIELDS:
                raise Table1ContractError(
                    "formal evaluator per-image CSV header drifted"
                )
            evaluator_rows = list(reader)
    except OSError as exc:
        raise Table1ContractError(
            f"cannot read evaluator per-image CSV: {exc}"
        ) from exc
    if len(records) != expected_count or len(evaluator_rows) != expected_count:
        raise Table1ContractError(
            f"FR cross-check requires exactly {expected_count} rows"
        )
    parity = _load_json_strict(metric_parity_summary)
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise Table1ContractError("metric parity summary is not a passing object")
    facts = parity.get("facts")
    if not isinstance(facts, Mapping):
        raise Table1ContractError("metric parity summary facts are malformed")
    psnr_tolerance = (
        _validate_score_value(
            facts.get("max_psnr_abs_diff"), metric="psnr", sample_id="parity"
        )
        + 1e-12
    )
    ssim_tolerance = (
        _validate_score_value(
            facts.get("max_ssim_abs_diff"), metric="ssim", sample_id="parity"
        )
        + 1e-12
    )
    if psnr_tolerance < 0 or ssim_tolerance < 0:
        raise Table1ContractError("metric parity tolerances must be nonnegative")

    evaluator_seen: set[str] = set()
    digest_rows: list[dict[str, str]] = []
    max_differences = {"psnr": 0.0, "ssim": 0.0}
    identity_fields = (
        "group",
        "combination",
        "prediction_png",
        "prediction_sha256",
        "target_png",
        "target_sha256",
    )
    for index, evaluator_row in enumerate(evaluator_rows):
        sample_id = evaluator_row["sample_id"]
        if (
            not sample_id
            or sample_id in evaluator_seen
            or sample_id not in scorer_by_id
        ):
            raise Table1ContractError(
                f"evaluator sample identity mismatch at row {index}: {sample_id!r}"
            )
        evaluator_seen.add(sample_id)
        scorer_row = scorer_by_id[sample_id]
        for field in identity_fields:
            if str(scorer_row[field]) != evaluator_row[field]:
                raise Table1ContractError(
                    f"evaluator/scorer identity drift for {sample_id}/{field}"
                )
        for metric, tolerance in (
            ("psnr", psnr_tolerance),
            ("ssim", ssim_tolerance),
        ):
            scorer_value = _validate_score_value(
                scorer_row[metric], metric=metric, sample_id=sample_id
            )
            evaluator_value = _validate_score_value(
                evaluator_row[metric], metric=metric, sample_id=sample_id
            )
            difference = abs(scorer_value - evaluator_value)
            max_differences[metric] = max(max_differences[metric], difference)
            if difference > tolerance:
                raise Table1ContractError(
                    f"evaluator/scorer {metric} drift for {sample_id}: "
                    f"{difference:.12g} > {tolerance:.12g}"
                )
        digest_rows.append(
            {
                "sample_id": sample_id,
                "prediction_sha256": evaluator_row["prediction_sha256"],
                "target_sha256": evaluator_row["target_sha256"],
            }
        )
    if evaluator_seen != set(scorer_by_id):
        raise Table1ContractError("evaluator/scorer sample sets differ")
    expected_digest = _require_sha256(
        predictions_digest, label="formal predictions digest"
    )
    if sha256_json(digest_rows) != expected_digest:
        raise Table1ContractError("formal evaluator predictions digest drifted")
    return {
        "image_count": expected_count,
        "prediction_digest": expected_digest,
        "psnr_max_abs_difference": max_differences["psnr"],
        "psnr_tolerance": psnr_tolerance,
        "ssim_max_abs_difference": max_differences["ssim"],
        "ssim_tolerance": ssim_tolerance,
        "passed": True,
    }


def score_table1(
    *,
    input_manifest: Path,
    output_root: Path,
    cache_root: Path,
    reference_python: Path,
    source_paths: Mapping[str, Path],
    device: str,
    shard_size: int = 10,
    worker_launcher: Callable[[Path, Mapping[str, str], Path], None] | None = None,
    enforce_data_disk: bool = True,
    formal_evidence: Mapping[str, Any] | None = None,
    enforce_formal: bool = False,
) -> dict[str, Any]:
    """Validate/resume score shards and atomically publish Table-1 outputs.

    ``worker_launcher`` is an internal CPU-test seam.  Production callers must
    leave it ``None``; the locked reference interpreter is then launched.
    """

    if device != "cpu" and _CUDA_DEVICE_PATTERN.fullmatch(device) is None:
        raise Table1ContractError("device must be 'cpu' or 'cuda[:index]'")
    if shard_size <= 0 or shard_size > EXPECTED_IMAGE_COUNT:
        raise Table1ContractError("shard_size is outside the allowed range")
    if enforce_formal:
        if worker_launcher is not None:
            raise Table1ContractError(
                "formal scoring forbids the internal worker-launcher test seam"
            )
        fixed_sources = default_source_paths()
        fixed_values = {
            "input_manifest": (input_manifest, FORMAL_TABLE1_INPUT_PATH),
            "output_root": (output_root, FORMAL_SCORE_ROOT),
            "cache_root": (cache_root, DEFAULT_CACHE_ROOT),
            "reference_python": (reference_python, DEFAULT_REFERENCE_PYTHON),
        }
        for label, (actual, expected) in fixed_values.items():
            if Path(actual) != expected:
                raise Table1ContractError(
                    f"formal {label} is fixed at {expected}, got {actual}"
                )
        if {key: Path(value) for key, value in source_paths.items()} != fixed_sources:
            raise Table1ContractError("formal AgenticIR source paths are fixed")
        if device != FORMAL_DEVICE or shard_size != FORMAL_SHARD_SIZE:
            raise Table1ContractError("formal device/shard size are fixed")
        current_evidence = _load_formal_evidence()
        _assert_cuda_uninitialized(label="formal scorer preflight completed")
        if formal_evidence is None:
            formal_evidence = current_evidence
        else:
            _revalidate_formal_evidence(formal_evidence)
        _require_exact_keys(
            formal_evidence, _FORMAL_EVIDENCE_KEYS, label="formal evidence"
        )
        output_root = _prepare_score_root(
            output_root,
            expected_root=FORMAL_SCORE_ROOT,
        )
        confinement_root: Path | None = output_root
    else:
        output_root = output_root.resolve(strict=False)
        if enforce_data_disk:
            output_root = _require_data_disk(output_root, label="score output root")
        output_root.mkdir(parents=True, exist_ok=True)
        confinement_root = None
    source_bindings = validate_pinned_sources(source_paths)
    cache_binding = check_cache(
        reference_python=reference_python,
        cache_root=cache_root,
        source_paths=source_paths,
    )
    rows, input_lock_semantic = validate_input_manifest(input_manifest)
    _ensure_semantic_json(
        output_root / "input_lock.json",
        input_lock_semantic,
        confinement_root=confinement_root,
    )
    input_lock_path = output_root / "input_lock.json"
    input_lock_sha = sha256_file(input_lock_path)

    run_semantic = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "agenticir_commit": PINNED_AGENTICIR_COMMIT,
        "metrics": list(METRICS),
        "metric_directions": METRIC_DIRECTIONS,
        "device": device,
        "shard_size": shard_size,
        "image_count": EXPECTED_IMAGE_COUNT,
        "expected_counts": dict(EXPECTED_COUNTS),
        "input_lock": {
            "path": str(input_lock_path.resolve(strict=True)),
            "sha256": input_lock_sha,
        },
        "weights_lock": {
            "path": cache_binding["path"],
            "sha256": cache_binding["sha256"],
        },
        "agenticir_sources": source_bindings,
        "implementation": _implementation_bindings(),
        "formal_evidence": dict(formal_evidence)
        if formal_evidence is not None
        else None,
        "allocator": FORMAL_ALLOCATOR if enforce_formal else None,
        "resume_policy": {
            "contiguous_prefix_only": True,
            "overwrite_existing_shard": False,
            "selective_rerun": False,
            "rng_chain_persisted": True,
        },
        "aggregation": (
            "per-image -> combination arithmetic mean -> group equal-combination mean"
        ),
        "formal_mio100_only": True,
    }
    _ensure_semantic_json(
        output_root / "run_contract.json",
        run_semantic,
        confinement_root=confinement_root,
    )
    run_contract_path = output_root / "run_contract.json"
    run_contract_sha = sha256_file(run_contract_path)
    shards_dir = output_root / "shards"
    scan = scan_score_shards(
        shards_dir=shards_dir,
        expected_rows=rows,
        shard_size=shard_size,
        run_contract_sha256=run_contract_sha,
        input_lock_sha256=input_lock_sha,
        initial_rng_core=cache_binding["lock"]["initial_rng_core"],
        confinement_root=confinement_root,
    )

    if scan["completed_shards"] < scan["expected_shards"]:
        request = {
            "schema_version": WORKER_REQUEST_SCHEMA,
            "rows": rows,
            "shards_dir": str(shards_dir.resolve()),
            "shard_size": shard_size,
            "start_shard": scan["completed_shards"],
            "run_contract_sha256": run_contract_sha,
            "input_lock_sha256": input_lock_sha,
            "initial_rng_core": cache_binding["lock"]["initial_rng_core"],
            "expected_metric_runtime": cache_binding["lock"]["metric_runtime"],
            "previous_rng": scan["previous_rng"],
            "expected_runtime": scan["runtime"],
            "device": device,
            "implementation": run_semantic["implementation"],
            "formal_evidence": run_semantic["formal_evidence"],
            "score_root": str(output_root),
            "run_contract": {
                "path": str(run_contract_path),
                "sha256": run_contract_sha,
            },
            "input_lock": {
                "path": str(input_lock_path),
                "sha256": input_lock_sha,
            },
            "weights_lock": {
                "path": cache_binding["path"],
                "sha256": cache_binding["sha256"],
            },
        }
        request_root = output_root / ".worker"
        if enforce_formal:
            _ensure_secure_directory(request_root)
            launch_work_root = _ensure_secure_directory(FORMAL_WORK_ROOT)
        else:
            request_root.mkdir(parents=True, exist_ok=True)
            launch_work_root = request_root
        request_path = request_root / f"request-{scan['completed_shards']:05d}.json"
        _write_worker_request(request_path, request, confinement_root=confinement_root)
        if enforce_formal:
            _assert_score_tree_shape(output_root)
        environment = _cache_environment(
            cache_root.resolve(strict=True),
            offline=True,
            temporary_root=launch_work_root,
            cpu_only=device == "cpu",
            allocator=FORMAL_ALLOCATOR if enforce_formal else None,
        )
        if enforce_formal:
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        if worker_launcher is None:
            if enforce_formal:
                _assert_cuda_uninitialized(label="formal score worker launch")
            result = _run_json_worker(
                reference_python,
                [
                    "_worker-score",
                    "--request",
                    str(request_path),
                ],
                environment=environment,
                work_parent=launch_work_root,
            )
            if result.get("schema_version") != WORKER_RESULT_SCHEMA:
                raise Table1ContractError("score worker returned a bad schema")
            _require_exact_keys(
                result,
                {
                    "schema_version",
                    "completed_shards",
                    "completed_rows",
                    "runtime",
                    "cuda_initialized",
                },
                label="score worker result",
            )
            if (
                result["completed_shards"]
                != scan["expected_shards"] - scan["completed_shards"]
                or result["completed_rows"] != len(rows) - scan["next_index"]
                or result["cuda_initialized"] is not True
            ):
                raise Table1ContractError(
                    "score worker completion counts/state drifted"
                )
        else:
            worker_launcher(request_path, environment, launch_work_root)

        # Catch downloads, cache mutation, input mutation, or source mutation
        # that occurred during the potentially long score process.
        check_cache(
            reference_python=reference_python,
            cache_root=cache_root,
            source_paths=source_paths,
        )
        rows_after, input_lock_after = validate_input_manifest(input_manifest)
        if rows_after != rows or input_lock_after != input_lock_semantic:
            raise Table1ContractError(
                "formal input mapping/files changed during scoring"
            )
        if validate_pinned_sources(source_paths) != source_bindings:
            raise Table1ContractError("pinned AgenticIR source changed during scoring")
        if _implementation_bindings() != run_semantic["implementation"]:
            raise Table1ContractError(
                "Table-1 scorer implementation changed during scoring"
            )
        if enforce_formal:
            _revalidate_formal_evidence(formal_evidence)
        if enforce_formal:
            _assert_score_tree_shape(output_root)
        scan = scan_score_shards(
            shards_dir=shards_dir,
            expected_rows=rows,
            shard_size=shard_size,
            run_contract_sha256=run_contract_sha,
            input_lock_sha256=input_lock_sha,
            initial_rng_core=cache_binding["lock"]["initial_rng_core"],
            confinement_root=confinement_root,
        )

    if scan["completed_shards"] != scan["expected_shards"]:
        raise Table1ContractError(
            "score worker returned without completing all remaining shards"
        )
    records = scan["records"]
    if enforce_formal:
        _revalidate_formal_evidence(formal_evidence)
    aggregate = aggregate_table1_records(records)
    crosscheck = None
    if formal_evidence is not None:
        crosscheck = crosscheck_evaluator_psnr_ssim(
            records,
            evaluator_csv=Path(formal_evidence["per_image"]["path"]),
            metric_parity_summary=Path(
                formal_evidence["metric_parity_summary"]["path"]
            ),
            predictions_digest=str(formal_evidence["predictions_digest"]),
        )
    per_image_path = output_root / "per_image.csv"
    summary_path = output_root / "summary.json"
    complete_path = output_root / "complete.json"
    _publish_or_verify_text(
        per_image_path,
        _per_image_csv(records),
        confinement_root=confinement_root,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "run_contract_sha256": run_contract_sha,
        "input_lock_sha256": input_lock_sha,
        "weights_lock_sha256": cache_binding["sha256"],
        "runtime": scan["runtime"],
        "metrics": list(METRICS),
        "metric_directions": METRIC_DIRECTIONS,
        "formal_evidence": dict(formal_evidence)
        if formal_evidence is not None
        else None,
        "evaluator_psnr_ssim_crosscheck": crosscheck,
        "shard_vram": {
            "ceiling": MAXIMUM_VRAM_RESERVED_FRACTION,
            "maximum_peak_reserved_fraction": scan["maximum_peak_reserved_fraction"],
            "shards": scan["shard_peaks"],
        },
        **aggregate,
    }
    _publish_or_verify_json(summary_path, summary, confinement_root=confinement_root)
    complete_semantic = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "COMPLETE",
        "image_count": len(records),
        "shard_count": scan["expected_shards"],
        "run_contract": {
            "path": str(run_contract_path.resolve(strict=True)),
            "sha256": run_contract_sha,
        },
        "input_lock": {
            "path": str(input_lock_path.resolve(strict=True)),
            "sha256": input_lock_sha,
        },
        "weights_lock": {
            "path": cache_binding["path"],
            "sha256": cache_binding["sha256"],
        },
        "per_image": {
            "path": str(per_image_path.resolve(strict=True)),
            "sha256": sha256_file(per_image_path),
        },
        "summary": {
            "path": str(summary_path.resolve(strict=True)),
            "sha256": sha256_file(summary_path),
        },
        "no_selective_rerun": True,
        "all_values_finite": True,
        "formal_evidence": dict(formal_evidence)
        if formal_evidence is not None
        else None,
        "evaluator_psnr_ssim_crosscheck": crosscheck,
        "maximum_peak_reserved_fraction": scan["maximum_peak_reserved_fraction"],
        "vram_ceiling": MAXIMUM_VRAM_RESERVED_FRACTION,
    }
    complete = _ensure_semantic_json(
        complete_path,
        complete_semantic,
        confinement_root=confinement_root,
    )
    if enforce_formal:
        _assert_score_tree_shape(output_root)
    return complete


def score_formal_table1() -> dict[str, Any]:
    """Run or exactly resume the one canonical formal Table-1 score job."""

    return score_table1(
        input_manifest=FORMAL_TABLE1_INPUT_PATH,
        output_root=FORMAL_SCORE_ROOT,
        cache_root=DEFAULT_CACHE_ROOT,
        reference_python=DEFAULT_REFERENCE_PYTHON,
        source_paths=default_source_paths(),
        device=FORMAL_DEVICE,
        shard_size=FORMAL_SHARD_SIZE,
        enforce_data_disk=True,
        formal_evidence=None,
        enforce_formal=True,
    )


def _worker_inspect(result_path: Path) -> None:
    import torch

    if torch.cuda.is_initialized():
        raise Table1ContractError("inspect worker unexpectedly initialized CUDA")
    result = _reference_environment()
    if torch.cuda.is_initialized():
        raise Table1ContractError("inspect worker initialized CUDA")
    from src.utils.io import atomic_write_json

    atomic_write_json(result_path, result)


def _worker_prefetch(cache_root: Path, scorer_path: Path, result_path: Path) -> None:
    import torch

    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise Table1ContractError("prefetch worker must run with CUDA hidden")
    environment = _reference_environment()
    _patch_clip_download_root(cache_root)
    scorer = _build_official_scorer(scorer_path, device="cpu")
    initial_state = _rng_state(include_cuda=False)
    metric_runtime = [
        {
            "name": metric.metric_name,
            "mode": metric.metric_mode,
            "lower_better": bool(metric.lower_better),
        }
        for metric in scorer.metrics
    ]
    if tuple(item["name"] for item in metric_runtime) != METRICS:
        raise Table1ContractError("prefetch metric runtime order mismatch")
    if torch.cuda.is_initialized():
        raise Table1ContractError("prefetch worker initialized CUDA")
    from src.utils.io import atomic_write_json

    atomic_write_json(
        result_path,
        {
            "schema_version": WORKER_RESULT_SCHEMA,
            "reference_environment": environment,
            "metric_runtime": metric_runtime,
            "initial_rng_core": _rng_core(initial_state),
            "initial_rng_core_sha256": sha256_json(_rng_core(initial_state)),
            "cuda_initialized": False,
        },
    )


def _worker_score(request_path: Path, result_path: Path) -> None:
    request = _load_json_strict(request_path)
    if not isinstance(request, dict):
        raise Table1ContractError("worker request is not an object")
    _require_exact_keys(
        request,
        {
            "schema_version",
            "rows",
            "shards_dir",
            "shard_size",
            "start_shard",
            "run_contract_sha256",
            "input_lock_sha256",
            "initial_rng_core",
            "expected_metric_runtime",
            "previous_rng",
            "expected_runtime",
            "device",
            "implementation",
            "formal_evidence",
            "score_root",
            "run_contract",
            "input_lock",
            "weights_lock",
        },
        label="worker request",
    )
    if request["schema_version"] != WORKER_REQUEST_SCHEMA:
        raise Table1ContractError("bad worker request schema")
    if (
        isinstance(request["start_shard"], bool)
        or not isinstance(request["start_shard"], int)
        or request["start_shard"] < 0
        or not isinstance(request["rows"], list)
    ):
        raise Table1ContractError("worker start_shard/rows are malformed")
    request_path = _assert_no_symlink_chain(request_path)
    expected_request_root = FORMAL_SCORE_ROOT / ".worker"
    if (
        request_path.parent != expected_request_root
        or not _WORKER_REQUEST_NAME_PATTERN.fullmatch(request_path.name)
    ):
        raise Table1ContractError(
            "hidden worker accepts only a canonical formal request"
        )
    _require_regular_file(request_path, immutable=True)
    _assert_confined_write_path(result_path, FORMAL_WORK_ROOT)
    _assert_score_tree_shape(FORMAL_SCORE_ROOT)
    if request["score_root"] != str(FORMAL_SCORE_ROOT):
        raise Table1ContractError("worker score root is not the fixed formal root")
    if request["shards_dir"] != str(FORMAL_SCORE_ROOT / "shards"):
        raise Table1ContractError("worker shards directory is not fixed")
    if request["device"] != FORMAL_DEVICE or request["shard_size"] != FORMAL_SHARD_SIZE:
        raise Table1ContractError("worker device/shard size are not fixed")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != FORMAL_ALLOCATOR:
        raise Table1ContractError("worker CUDA allocator contract is not exact")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise Table1ContractError("worker must expose exactly physical GPU 0")

    if not isinstance(request["formal_evidence"], Mapping):
        raise Table1ContractError("worker formal evidence is malformed")
    formal_evidence = _revalidate_formal_evidence(request["formal_evidence"])
    if request["implementation"] != _implementation_bindings():
        raise Table1ContractError("worker implementation binding mismatch")
    verified_request_bindings: dict[str, dict[str, Any]] = {}
    for label, raw, expected_path in (
        (
            "run contract",
            request["run_contract"],
            FORMAL_SCORE_ROOT / "run_contract.json",
        ),
        ("input lock", request["input_lock"], FORMAL_SCORE_ROOT / "input_lock.json"),
        (
            "weights lock",
            request["weights_lock"],
            DEFAULT_CACHE_ROOT / "weights_lock.json",
        ),
    ):
        verified_request_bindings[label] = _binding_from_payload(
            raw, label=label, expected_path=expected_path
        )
    if (
        request["run_contract_sha256"]
        != verified_request_bindings["run contract"]["sha256"]
        or request["input_lock_sha256"]
        != verified_request_bindings["input lock"]["sha256"]
    ):
        raise Table1ContractError("worker request top-level binding hash drifted")
    run_contract = _load_json_strict(FORMAL_SCORE_ROOT / "run_contract.json")
    if not isinstance(run_contract, Mapping):
        raise Table1ContractError("worker run contract is malformed")
    _require_exact_keys(
        run_contract,
        {
            "schema_version",
            "agenticir_commit",
            "metrics",
            "metric_directions",
            "device",
            "shard_size",
            "image_count",
            "expected_counts",
            "input_lock",
            "weights_lock",
            "agenticir_sources",
            "implementation",
            "formal_evidence",
            "allocator",
            "resume_policy",
            "aggregation",
            "formal_mio100_only",
            "created_utc",
        },
        label="worker run contract",
    )
    fixed_contract_values = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "agenticir_commit": PINNED_AGENTICIR_COMMIT,
        "metrics": list(METRICS),
        "metric_directions": METRIC_DIRECTIONS,
        "device": FORMAL_DEVICE,
        "shard_size": FORMAL_SHARD_SIZE,
        "image_count": EXPECTED_IMAGE_COUNT,
        "expected_counts": dict(EXPECTED_COUNTS),
        "input_lock": request["input_lock"],
        "weights_lock": request["weights_lock"],
        "implementation": request["implementation"],
        "formal_evidence": formal_evidence,
        "allocator": FORMAL_ALLOCATOR,
        "resume_policy": {
            "contiguous_prefix_only": True,
            "overwrite_existing_shard": False,
            "selective_rerun": False,
            "rng_chain_persisted": True,
        },
        "aggregation": (
            "per-image -> combination arithmetic mean -> group equal-combination mean"
        ),
        "formal_mio100_only": True,
    }
    drift = {
        key: {"expected": value, "actual": run_contract.get(key)}
        for key, value in fixed_contract_values.items()
        if run_contract.get(key) != value
    }
    if drift:
        raise Table1ContractError(f"worker run contract drifted: {drift}")
    created_utc = run_contract.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc.endswith("Z"):
        raise Table1ContractError("worker run contract timestamp is malformed")

    rows, input_lock_semantic = validate_input_manifest(FORMAL_TABLE1_INPUT_PATH)
    if request["rows"] != rows:
        raise Table1ContractError("worker rows differ from fixed formal Table-1 input")
    stored_input_lock = _load_json_strict(FORMAL_SCORE_ROOT / "input_lock.json")
    if not isinstance(stored_input_lock, Mapping):
        raise Table1ContractError("worker input lock is malformed")
    comparable_input_lock = dict(stored_input_lock)
    created = comparable_input_lock.pop("created_utc", None)
    if not isinstance(created, str) or not created.endswith("Z"):
        raise Table1ContractError("worker input lock timestamp is malformed")
    if comparable_input_lock != input_lock_semantic:
        raise Table1ContractError("worker input lock content drifted")

    source_bindings = validate_pinned_sources(default_source_paths())
    if run_contract.get("agenticir_sources") != source_bindings:
        raise Table1ContractError("worker AgenticIR source binding drifted")
    weights_path = DEFAULT_CACHE_ROOT / "weights_lock.json"
    weights_lock = _load_json_strict(weights_path)
    if not isinstance(weights_lock, Mapping):
        raise Table1ContractError("worker weights lock is malformed")
    if weights_lock.get("schema_version") != WEIGHTS_LOCK_SCHEMA:
        raise Table1ContractError("worker weights lock schema drifted")
    launcher_binding = _reference_launcher_binding(DEFAULT_REFERENCE_PYTHON)
    if weights_lock.get("reference_launcher") != launcher_binding:
        raise Table1ContractError("worker reference-launcher binding drifted")
    reference_environment = _reference_environment()
    _validate_reference_environment_for_launcher(
        reference_environment, launcher_binding
    )
    if weights_lock.get("reference_environment") != reference_environment:
        raise Table1ContractError("worker reference environment drifted")
    if _cache_inventory(DEFAULT_CACHE_ROOT) != weights_lock.get("weights"):
        raise Table1ContractError("worker frozen weight inventory drifted")
    _verify_frozen_cache_directories(DEFAULT_CACHE_ROOT)
    if request["expected_metric_runtime"] != weights_lock.get("metric_runtime"):
        raise Table1ContractError("worker metric runtime request drifted")
    if request["initial_rng_core"] != weights_lock.get("initial_rng_core"):
        raise Table1ContractError("worker initial RNG differs from weights lock")

    scan = scan_score_shards(
        shards_dir=FORMAL_SCORE_ROOT / "shards",
        expected_rows=rows,
        shard_size=FORMAL_SHARD_SIZE,
        run_contract_sha256=str(request["run_contract_sha256"]),
        input_lock_sha256=str(request["input_lock_sha256"]),
        initial_rng_core=request["initial_rng_core"],
        confinement_root=FORMAL_SCORE_ROOT,
    )
    if request["start_shard"] != scan["completed_shards"]:
        raise Table1ContractError("worker request is not the exact contiguous resume")
    if request["previous_rng"] != scan["previous_rng"]:
        raise Table1ContractError(
            "worker previous RNG differs from durable shard chain"
        )
    if request["expected_runtime"] != scan["runtime"]:
        raise Table1ContractError("worker expected runtime differs from durable shards")

    import torch

    if torch.cuda.is_initialized():
        raise Table1ContractError(
            "worker initialized CUDA before formal gate validation"
        )
    device = FORMAL_DEVICE
    device_index = 0
    _assert_gpu_ownership(set(), device_index=device_index)
    _disable_network()
    _patch_clip_download_root(DEFAULT_CACHE_ROOT)
    scorer = _build_official_scorer(
        default_source_paths()["official_scorer"], device=device
    )
    torch.cuda.synchronize(device_index)
    _assert_gpu_ownership({os.getpid()}, device_index=device_index)
    metric_runtime = [
        {
            "name": metric.metric_name,
            "mode": metric.metric_mode,
            "lower_better": bool(metric.lower_better),
        }
        for metric in scorer.metrics
    ]
    if metric_runtime != request["expected_metric_runtime"]:
        raise Table1ContractError("worker metric runtime differs from weights lock")
    runtime = _runtime_descriptor(device)
    if (
        request["expected_runtime"] is not None
        and request["expected_runtime"] != runtime
    ):
        raise Table1ContractError("worker runtime differs from prior shard runtime")
    include_cuda = True
    state = _rng_state(include_cuda=include_cuda)
    if _rng_core(state) != request["initial_rng_core"]:
        raise Table1ContractError(
            "metric construction did not reproduce locked seed-123 RNG core"
        )
    if request["previous_rng"] is not None:
        _restore_rng_state(request["previous_rng"], include_cuda=include_cuda)
        state = _rng_state(include_cuda=include_cuda)
        if state != request["previous_rng"]:
            raise Table1ContractError("failed to restore exact RNG continuation state")

    shard_size = FORMAL_SHARD_SIZE
    start_shard = int(request["start_shard"])
    shards_dir = FORMAL_SCORE_ROOT / "shards"
    expected_shards = (len(rows) + shard_size - 1) // shard_size
    completed_rows = 0
    for shard_index in range(start_shard, expected_shards):
        start = shard_index * shard_size
        end = min(start + shard_size, len(rows))
        _assert_score_tree_shape(FORMAL_SCORE_ROOT)
        _assert_gpu_ownership({os.getpid()}, device_index=device_index)
        torch.cuda.synchronize(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)
        rng_before = _rng_state(include_cuda=include_cuda)
        scored_rows: list[dict[str, Any]] = []
        for index in range(start, end):
            row = rows[index]
            if not isinstance(row, dict):
                raise Table1ContractError(f"worker row {index} is not an object")
            _require_exact_keys(row, INPUT_KEYS, label=f"worker row {index}")
            prediction = Path(row["prediction_png"])
            target = Path(row["target_png"])
            _assert_gpu_ownership({os.getpid()}, device_index=device_index)
            _require_regular_file(prediction, immutable=True)
            _require_regular_file(target)
            if sha256_file(prediction) != row["prediction_sha256"]:
                raise Table1ContractError(
                    f"worker prediction hash mismatch at row {index}"
                )
            if sha256_file(target) != row["target_sha256"]:
                raise Table1ContractError(f"worker target hash mismatch at row {index}")
            raw_scores = scorer(prediction, target)
            torch.cuda.synchronize(device_index)
            _assert_gpu_ownership({os.getpid()}, device_index=device_index)
            if tuple(item[0] for item in raw_scores) != METRICS:
                raise Table1ContractError(
                    f"worker metric order mismatch at row {index}"
                )
            score_row = dict(row)
            for metric, _lower_better, value in raw_scores:
                score_row[metric] = _validate_score_value(
                    value, metric=metric, sample_id=str(row["sample_id"])
                )
            scored_rows.append(score_row)
        torch.cuda.synchronize(device_index)
        memory = _validate_peak_reserved(
            int(torch.cuda.max_memory_reserved(device_index)),
            int(torch.cuda.get_device_properties(device_index).total_memory),
        )
        _assert_gpu_ownership({os.getpid()}, device_index=device_index)
        rng_after = _rng_state(include_cuda=include_cuda)
        shard = {
            "schema_version": SHARD_SCHEMA,
            "shard_index": shard_index,
            "start_index": start,
            "end_index": end,
            "run_contract_sha256": request["run_contract_sha256"],
            "input_lock_sha256": request["input_lock_sha256"],
            "runtime": runtime,
            "rng_before": rng_before,
            "rng_before_sha256": sha256_json(rng_before),
            "rng_after": rng_after,
            "rng_after_sha256": sha256_json(rng_after),
            **memory,
            "rows": scored_rows,
        }
        _assert_score_tree_shape(FORMAL_SCORE_ROOT)
        _atomic_create_json(
            _shard_path(shards_dir, shard_index),
            shard,
            confinement_root=FORMAL_SCORE_ROOT,
        )
        _assert_score_tree_shape(FORMAL_SCORE_ROOT)
        completed_rows += len(scored_rows)
    _atomic_create_json(
        result_path,
        {
            "schema_version": WORKER_RESULT_SCHEMA,
            "completed_shards": expected_shards - start_shard,
            "completed_rows": completed_rows,
            "runtime": runtime,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
        },
        confinement_root=result_path.parent,
    )


def _source_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "official_scorer": args.official_scorer,
        "official_compute_scores": args.official_compute_scores,
        "official_compare_methods": args.official_compare_methods,
        "official_requirements": args.official_requirements,
    }


def _add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = default_source_paths()
    parser.add_argument(
        "--reference-python", type=Path, default=DEFAULT_REFERENCE_PYTHON
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--official-scorer", type=Path, default=defaults["official_scorer"]
    )
    parser.add_argument(
        "--official-compute-scores",
        type=Path,
        default=defaults["official_compute_scores"],
    )
    parser.add_argument(
        "--official-compare-methods",
        type=Path,
        default=defaults["official_compare_methods"],
    )
    parser.add_argument(
        "--official-requirements",
        type=Path,
        default=defaults["official_requirements"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen AgenticIR Table-1 six-metric scorer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prefetch = subparsers.add_parser(
        "prefetch",
        help="explicitly download all pyiqa 0.1.10 weights to data disk and freeze them",
    )
    _add_reference_arguments(prefetch)

    check = subparsers.add_parser(
        "check-cache",
        help="CPU-only/offline verification of dependency, source, and weight hashes",
    )
    _add_reference_arguments(check)

    subparsers.add_parser(
        "score",
        help=(
            "score/resume the one canonical authorized 1,440-row formal mapping; "
            "all paths/device/cache/sharding are fixed"
        ),
    )

    inspect_worker = subparsers.add_parser("_worker-inspect", help=argparse.SUPPRESS)
    inspect_worker.add_argument("--worker-result", type=Path, required=True)

    prefetch_worker = subparsers.add_parser("_worker-prefetch", help=argparse.SUPPRESS)
    prefetch_worker.add_argument("--cache-root", type=Path, required=True)
    prefetch_worker.add_argument("--official-scorer", type=Path, required=True)
    prefetch_worker.add_argument("--worker-result", type=Path, required=True)

    score_worker = subparsers.add_parser("_worker-score", help=argparse.SUPPRESS)
    score_worker.add_argument("--request", type=Path, required=True)
    score_worker.add_argument("--worker-result", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "prefetch":
            binding = prefetch_weights(
                reference_python=args.reference_python,
                cache_root=args.cache_root,
                source_paths=_source_paths_from_args(args),
            )
            print(json.dumps({key: binding[key] for key in ("path", "sha256", "mode")}))
        elif args.command == "check-cache":
            binding = check_cache(
                reference_python=args.reference_python,
                cache_root=args.cache_root,
                source_paths=_source_paths_from_args(args),
            )
            print(json.dumps({key: binding[key] for key in ("path", "sha256", "mode")}))
        elif args.command == "score":
            complete = score_formal_table1()
            print(json.dumps(complete, sort_keys=True))
        elif args.command == "_worker-inspect":
            _worker_inspect(args.worker_result)
        elif args.command == "_worker-prefetch":
            _worker_prefetch(args.cache_root, args.official_scorer, args.worker_result)
        elif args.command == "_worker-score":
            _worker_score(args.request, args.worker_result)
        else:  # pragma: no cover - argparse enforces choices
            raise Table1ContractError(f"unsupported command: {args.command}")
    except Table1ContractError as exc:
        print(f"AgenticIR Table-1 contract error: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
