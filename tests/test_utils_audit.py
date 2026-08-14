from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.utils import (
    AuditTrail,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    canonical_git_remote,
    ensure_within,
    is_sha256,
    load_json,
    load_yaml,
    sha256_file,
    sha256_json,
    utc_now_iso,
)
from src.utils.paths import ResolvedPathsError


def test_sha256_helpers_are_streaming_and_canonical(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    assert sha256_file(payload) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert is_sha256(sha256_file(payload))
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        sha256_file(payload, chunk_size=0)


def test_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("a: 1\na: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_yaml(path)


def test_atomic_writers_and_utc(tmp_path: Path) -> None:
    text_path = tmp_path / "nested" / "value.txt"
    json_path = tmp_path / "value.json"
    yaml_path = tmp_path / "value.yaml"
    atomic_write_text(text_path, "hello\n")
    atomic_write_json(json_path, {"b": 2, "a": 1})
    atomic_write_yaml(yaml_path, {"first": 1, "second": [2, 3]})
    assert text_path.read_text(encoding="utf-8") == "hello\n"
    assert load_json(json_path) == {"a": 1, "b": 2}
    assert load_yaml(yaml_path) == {"first": 1, "second": [2, 3]}
    assert not list(tmp_path.rglob("*.tmp"))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now_iso())


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        ("https://github.com/Kaiwen-Zhu/AgenticIR.git", "https://github.com/Kaiwen-Zhu/AgenticIR"),
        ("git@github.com:Kaiwen-Zhu/AgenticIR.git", "https://github.com/Kaiwen-Zhu/AgenticIR"),
        ("ssh://git@github.com/Kaiwen-Zhu/AgenticIR.git", "https://github.com/Kaiwen-Zhu/AgenticIR"),
    ],
)
def test_git_remote_canonicalization(remote: str, expected: str) -> None:
    assert canonical_git_remote(remote) == expected


def test_ensure_within_fails_closed(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    assert ensure_within(allowed / "child", allowed) == allowed / "child"
    with pytest.raises(ResolvedPathsError, match="escapes"):
        ensure_within(tmp_path / "outside", allowed)


def test_audit_trail_renders_machine_and_human_results() -> None:
    trail = AuditTrail(protocol="test")
    trail.require(True, "ok", "yes", "no")
    trail.warn(False, "warning", "yes", "caution")
    trail.facts["count"] = 2
    assert trail.passed
    assert trail.warning_count == 1
    payload = trail.to_dict()
    assert payload["passed"] is True
    assert json.loads(json.dumps(payload))["facts"] == {"count": 2}
    markdown = trail.to_markdown(title="Audit")
    assert "**PASS** `ok`" in markdown
    assert "**WARN** `warning`" in markdown
