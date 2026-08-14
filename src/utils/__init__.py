"""Small, dependency-light utilities shared by GraphRestore scripts."""

from .audit import AuditCheck, AuditTrail
from .git import GitCommandError, git_commit, git_remote_url, git_status_porcelain
from .hashing import is_sha256, sha256_file, sha256_json
from .io import (
    JsonlDecodeError,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    iter_jsonl,
    load_json,
    load_yaml,
    utc_now_iso,
)
from .paths import (
    ResolvedPathsError,
    canonical_git_remote,
    ensure_within,
    load_resolved_paths,
    resolve_config_path,
)

__all__ = [
    "AuditCheck",
    "AuditTrail",
    "GitCommandError",
    "JsonlDecodeError",
    "ResolvedPathsError",
    "atomic_write_json",
    "atomic_write_text",
    "atomic_write_yaml",
    "canonical_git_remote",
    "ensure_within",
    "git_commit",
    "git_remote_url",
    "git_status_porcelain",
    "is_sha256",
    "iter_jsonl",
    "load_json",
    "load_resolved_paths",
    "load_yaml",
    "resolve_config_path",
    "sha256_file",
    "sha256_json",
    "utc_now_iso",
]
