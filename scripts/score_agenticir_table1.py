#!/usr/bin/env python3
"""CLI entry point for immutable AgenticIR Table-1 six-metric scoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_standalone_module() -> ModuleType:
    """Bypass ``src.evaluation.__init__`` and its heavyweight model imports."""

    path = PROJECT_ROOT / "src" / "evaluation" / "agenticir_table1.py"
    spec = importlib.util.spec_from_file_location(
        "graphrestore_standalone_agenticir_table1", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Table-1 scorer module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = _load_standalone_module().main


if __name__ == "__main__":
    raise SystemExit(main())
