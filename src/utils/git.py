"""Read-only Git identity helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitCommandError(RuntimeError):
    """A read-only Git identity query failed."""


def _run_git(repo: str | Path, *arguments: str) -> str:
    repository = Path(repo)
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise GitCommandError(
            f"git {' '.join(arguments)} failed in {repository}: {detail}"
        )
    return process.stdout.strip()


def git_commit(repo: str | Path) -> str:
    """Return the checked-out commit SHA."""

    return _run_git(repo, "rev-parse", "HEAD")


def git_remote_url(repo: str | Path, remote: str = "origin") -> str:
    """Return the configured URL for a named remote."""

    return _run_git(repo, "remote", "get-url", remote)


def git_status_porcelain(repo: str | Path) -> tuple[str, ...]:
    """Return tracked changes without scanning untracked data files."""

    output = _run_git(repo, "status", "--porcelain", "--untracked-files=no")
    return tuple(line for line in output.splitlines() if line)
