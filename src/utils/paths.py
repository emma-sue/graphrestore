"""Path and repository-identity helpers for fail-closed configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import load_yaml


class ResolvedPathsError(ValueError):
    """The resolved-path configuration is malformed or incomplete."""


REQUIRED_PATH_KEYS = (
    "data_root",
    "training_data_root",
    "agenticir_repo",
    "mioir_repo",
    "agenticir_add_single_degradation",
    "agenticir_degradations_txt",
    "agenticir_scorer",
    "agenticir_compute_scores",
    "agenticir_compare_methods",
    "agenticir_requirements",
    "clean_train_manifest",
    "clean_val_manifest",
    "primary_train_manifest",
    "primary_val_manifest",
    "primary_all_manifest",
    "mio100_test_1440_manifest",
    "mio100_group_a_test_manifest",
    "mio100_group_b_test_manifest",
    "mio100_group_c_test_manifest",
    "mio100_exploration_manifest",
    "stage_a_parent_manifest",
    "stage_a_parent_checkpoint",
)


def load_resolved_paths(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate ``configs/resolved_paths.yaml``."""

    config_path = Path(path).resolve()
    value = load_yaml(config_path)
    if not isinstance(value, Mapping):
        raise ResolvedPathsError(f"expected YAML mapping: {config_path}")
    config = dict(value)

    missing = [key for key in REQUIRED_PATH_KEYS if not config.get(key)]
    if missing:
        raise ResolvedPathsError(f"missing resolved path keys: {', '.join(missing)}")
    if not isinstance(config.get("expected_identity"), Mapping):
        raise ResolvedPathsError("expected_identity must be a mapping")
    return config


def resolve_config_path(
    config_path: str | Path,
    value: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve an absolute path, or a relative path against the project root."""

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if project_root is None:
        project_root = Path(config_path).resolve().parent.parent
    return (Path(project_root) / candidate).resolve(strict=False)


def ensure_within(path: str | Path, root: str | Path) -> Path:
    """Resolve *path* and require it to remain inside *root*."""

    resolved_path = Path(path).resolve(strict=False)
    resolved_root = Path(root).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ResolvedPathsError(
            f"path escapes allowed root: {resolved_path} (root={resolved_root})"
        ) from exc
    return resolved_path


def canonical_git_remote(value: str) -> str:
    """Canonicalize common GitHub HTTPS/SSH spellings for identity comparison."""

    remote = value.strip()
    scp_match = re.fullmatch(r"git@github\.com:(.+)", remote)
    if scp_match:
        remote = f"https://github.com/{scp_match.group(1)}"
    elif remote.startswith("ssh://git@github.com/"):
        remote = f"https://github.com/{remote.removeprefix('ssh://git@github.com/')}"
    if remote.startswith("http://github.com/"):
        remote = f"https://github.com/{remote.removeprefix('http://github.com/')}"
    return remote.removesuffix("/").removesuffix(".git")
