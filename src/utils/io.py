"""Strict YAML/JSON loading plus UTC and atomic-output helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class JsonlDecodeError(ValueError):
    """A JSONL record could not be decoded or was not an object."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def fsync_directory(path: str | Path) -> None:
    """Durably commit a rename or directory-entry update on POSIX filesystems."""

    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def utc_now_iso() -> str:
    """Return an RFC3339 UTC timestamp with second precision."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def load_yaml(path: str | Path) -> Any:
    """Load YAML safely and reject duplicate keys."""

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_UniqueKeyLoader)


def load_json(path: str | Path) -> Any:
    """Load a UTF-8 JSON document."""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)`` pairs from a strict JSONL file."""

    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise JsonlDecodeError(
                    f"{jsonl_path}:{line_number}: blank JSONL records are forbidden"
                )
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise JsonlDecodeError(
                    f"{jsonl_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise JsonlDecodeError(
                    f"{jsonl_path}:{line_number}: expected a JSON object"
                )
            yield line_number, value


def _atomic_replace(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace a UTF-8 text file on the same filesystem."""

    _atomic_replace(Path(path), text)


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically write deterministic, human-readable JSON."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(path, f"{payload}\n")


def atomic_write_yaml(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write a YAML mapping without sorting user-declared keys."""

    if not isinstance(value, Mapping):
        raise TypeError("atomic_write_yaml expects a mapping")
    payload = yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    atomic_write_text(path, payload)
