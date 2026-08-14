"""Shared exact-source bindings for resumable GraphRestore stages."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.utils.hashing import sha256_file


class SourceBindingError(RuntimeError):
    """A semantic source path is missing or escaped the project root."""


def semantic_source_hashes(
    project_root: str | Path,
    *,
    entrypoints: Sequence[str],
) -> dict[str, str]:
    """Hash all importable project sources plus the stage entry script(s)."""

    root = Path(project_root).resolve()
    candidates = [*sorted((root / "src").rglob("*.py"))]
    candidates.extend(root / relative for relative in entrypoints)
    result: dict[str, str] = {}
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise SourceBindingError(f"semantic source escaped project root: {resolved}") from exc
        if not resolved.is_file():
            raise SourceBindingError(f"semantic source is missing: {resolved}")
        result[str(relative)] = sha256_file(resolved)
    if not result:
        raise SourceBindingError("semantic source binding set is empty")
    return dict(sorted(result.items()))


__all__ = ["SourceBindingError", "semantic_source_hashes"]
