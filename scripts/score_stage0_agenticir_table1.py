#!/usr/bin/env python3
"""Fixed Stage0 formal AgenticIR six-metric scorer and paired comparison.

The legacy scorer is loaded into a private module namespace because its source
SHA is already frozen by Stage4.  This entry point deterministically installs
only the preregistered Stage0 authorization/evaluator/score/work roots.  The
public ``score`` and ``verify`` commands accept no path, device, worker, or
evidence override.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCORER_PATH = PROJECT_ROOT / "src/evaluation/agenticir_table1.py"
LEGACY_SCORER_SHA256 = (
    "22c8f48607b631ab9ddf2e0565012be2be5f52674eae18e2b2f09ad02faa8d73"
)


def _verified_legacy_scorer_path(
    path: Path = LEGACY_SCORER_PATH,
    *,
    expected_sha256: str = LEGACY_SCORER_SHA256,
) -> Path:
    """Hash the frozen legacy source before importing any project module."""

    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError("frozen Table-1 scorer path must be canonical and absolute")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if resolved != path or not stat.S_ISREG(before.st_mode):
        raise RuntimeError("frozen Table-1 scorer is not a canonical regular file")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or digest.hexdigest() != expected_sha256:
        raise RuntimeError("frozen Table-1 scorer SHA256 drifted")
    return resolved


# Fail before importing torch-bearing project modules or executing the legacy
# module if its Stage4-bound bytes changed.
_verified_legacy_scorer_path()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stage0_formal import (  # noqa: E402
    configure_stage0_table1_module,
    default_stage0_authorization_paths,
    publish_stage0_vs_stage4_comparison,
    validate_stage0_formal_authorization,
    validate_stage0_vs_stage4_comparison_complete,
)
from src.evaluation.stage0_formal_inventory import (  # noqa: E402
    STAGE0_APPROVAL_PATH,
    STAGE0_SCORE_ROOT,
)
from src.evaluation.mio100 import MiO100EvaluationError  # noqa: E402


def _load_fixed_scorer(
    path: Path = LEGACY_SCORER_PATH,
    *,
    expected_sha256: str = LEGACY_SCORER_SHA256,
) -> ModuleType:
    path = _verified_legacy_scorer_path(path, expected_sha256=expected_sha256)
    spec = importlib.util.spec_from_file_location(
        "graphrestore_standalone_stage0_agenticir_table1", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen Table-1 scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return configure_stage0_table1_module(module, cli_path=Path(__file__))


SCORER = _load_fixed_scorer()


def _canonical_worker_result(path: str, *, work_parent: Path) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.name != "result.json":
        return False
    try:
        parent = candidate.parent.resolve(strict=True)
        root = work_parent.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return False
    return (
        candidate.parent == parent
        and parent.parent == root
        and parent.name.startswith("agenticir-table1-")
        and not candidate.exists()
        and not candidate.is_symlink()
    )


def _is_canonical_internal_worker(arguments: Sequence[str]) -> bool:
    if (
        len(arguments) == 3
        and arguments[0] == "_worker-inspect"
        and arguments[1] == "--worker-result"
    ):
        return _canonical_worker_result(
            arguments[2],
            work_parent=(
                SCORER.DEFAULT_CACHE_ROOT.parent / ".agenticir_table1_check_work"
            ),
        )
    if (
        len(arguments) == 5
        and arguments[0] == "_worker-score"
        and arguments[1] == "--request"
        and arguments[3] == "--worker-result"
    ):
        request = Path(arguments[2])
        try:
            request_file = request.resolve(strict=True)
            request_root = (SCORER.FORMAL_SCORE_ROOT / ".worker").resolve(strict=True)
        except (FileNotFoundError, OSError):
            return False
        return (
            request == request_file
            and request.parent == request_root
            and SCORER._WORKER_REQUEST_NAME_PATTERN.fullmatch(request.name)  # noqa: SLF001
            is not None
            and _canonical_worker_result(
                arguments[4], work_parent=SCORER.FORMAL_WORK_ROOT
            )
        )
    return False


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _is_canonical_internal_worker(arguments):
        # These are the two subprocesses launched by the frozen controller
        # during formal scoring.  Their exact paths are prefiltered here and
        # the legacy worker then revalidates request/evidence/root/GPU state.
        return int(SCORER.main(arguments))
    if arguments in (["score"], ["verify"]):
        try:
            expected = default_stage0_authorization_paths()
            authorization = validate_stage0_formal_authorization(
                STAGE0_APPROVAL_PATH,
                expected_bindings=expected,
            )
            if arguments == ["score"]:
                score_complete = SCORER.score_formal_table1()
                comparison = publish_stage0_vs_stage4_comparison(
                    stage0_per_image=STAGE0_SCORE_ROOT / "per_image.csv",
                    stage4_per_image=authorization.bindings[
                        "stage4_table1_per_image"
                    ].path,
                    authorization=authorization,
                )
            else:
                comparison = validate_stage0_vs_stage4_comparison_complete(
                    authorization=authorization
                )
                score_path = STAGE0_SCORE_ROOT / "complete.json"
                score_complete = {
                    "status": "VERIFIED",
                    "path": str(score_path),
                    "sha256": SCORER.sha256_file(score_path),
                }
            print(
                json.dumps(
                    {
                        "stage0_table1_complete": score_complete,
                        "stage0_vs_stage4_complete": comparison,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        except (SCORER.Table1ContractError, MiO100EvaluationError) as exc:
            print(f"Stage0 AgenticIR Table-1 contract error: {exc}", file=sys.stderr)
            return 3
    print(
        "Stage0 AgenticIR Table-1 accepts exactly one command: score or verify",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
