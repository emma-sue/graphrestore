"""Deterministic hashing helpers used by protocol audits and checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 of *path* without loading the whole file in memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"not a regular file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using a canonical UTF-8 representation."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether *value* is a lowercase 64-character SHA256 string."""

    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
