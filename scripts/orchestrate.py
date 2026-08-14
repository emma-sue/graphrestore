#!/usr/bin/env python3
"""CLI for the fail-closed GraphRestore V7.1 pipeline state machine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.orchestration import (  # noqa: E402
    ApprovalError,
    ChildCommandError,
    CommandRunner,
    GraphRestoreOrchestrator,
    OrchestrationError,
    PipelineState,
    command_plan,
    recommended_tmux_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one explicit GraphRestore V7.1 orchestration action. Stage3 "
            "is fail-closed behind two approval flags and frozen artifact hashes."
        )
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        help="GraphRestore repository root (defaults to this script's parent)",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="run only the locked audits and mandatory CPU tests",
    )
    parser.add_argument(
        "--integration_steps",
        type=int,
        metavar="N",
        help="run preflight if needed, then the integration run (N must be 100)",
    )
    parser.add_argument(
        "--run_main_pipeline",
        action="store_true",
        help="run Stage0, Stage1, effect profiles, and Stage2, then pause",
    )
    parser.add_argument(
        "--resume_main_pipeline",
        action="store_true",
        help="explicitly resume a FAILED Stage0/1/2 pipeline from bound artifacts",
    )
    parser.add_argument(
        "--approve_stage3",
        action="store_true",
        help="explicitly approve frozen Stage2 artifacts for Stage3",
    )
    parser.add_argument(
        "--resume_from_stage3",
        action="store_true",
        help="resume only at Stage3 (must accompany --approve_stage3)",
    )
    parser.add_argument(
        "--show_state",
        action="store_true",
        help="print durable orchestration state without changing it",
    )
    parser.add_argument(
        "--print_plan",
        action="store_true",
        help="print the command plan without executing it",
    )
    parser.add_argument(
        "--print_tmux_command",
        action="store_true",
        help="print the pipefail-safe main-pipeline tmux command",
    )
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _print_state(state: PipelineState) -> None:
    _print_json(state.to_dict())


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: CommandRunner | None = None,
    project_root: str | Path | None = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    approval_action = arguments.approve_stage3 or arguments.resume_from_stage3
    action_count = sum(
        bool(action)
        for action in (
            arguments.preflight_only,
            arguments.integration_steps is not None,
            arguments.run_main_pipeline,
            arguments.resume_main_pipeline,
            approval_action,
            arguments.show_state,
            arguments.print_plan,
            arguments.print_tmux_command,
        )
    )
    if action_count != 1:
        parser.error("select exactly one orchestration action")

    root = Path(project_root or arguments.project_root or PROJECT_ROOT).resolve()
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)

    try:
        if arguments.preflight_only:
            _print_state(orchestrator.run_preflight())
        elif arguments.integration_steps is not None:
            _print_state(orchestrator.run_integration(arguments.integration_steps))
        elif arguments.run_main_pipeline:
            _print_state(orchestrator.run_main_pipeline())
        elif arguments.resume_main_pipeline:
            _print_state(orchestrator.resume_main_pipeline())
        elif approval_action:
            _print_state(
                orchestrator.approve_and_resume_stage3(
                    approve_stage3=arguments.approve_stage3,
                    resume_from_stage3=arguments.resume_from_stage3,
                )
            )
        elif arguments.show_state:
            _print_state(orchestrator.load_state())
        elif arguments.print_plan:
            _print_json(command_plan(orchestrator))
        else:
            print(recommended_tmux_command(root, orchestrator.python))
    except KeyboardInterrupt:
        print("orchestration interrupted", file=sys.stderr)
        return 130
    except ChildCommandError as exc:
        print(f"orchestration child failure: {exc}", file=sys.stderr)
        return exc.exit_code
    except ApprovalError as exc:
        print(f"Stage3 approval refused: {exc}", file=sys.stderr)
        return exc.exit_code
    except OrchestrationError as exc:
        print(f"orchestration refused: {exc}", file=sys.stderr)
        return exc.exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
