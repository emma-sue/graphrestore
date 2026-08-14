"""Fail-closed GraphRestore V7.1 pipeline orchestration.

This module owns process sequencing, durable state, and the Stage2 approval
barrier.  It deliberately contains no training or evaluation algorithms.
Every child is invoked with an argv sequence and ``shell=False`` so its exit
status cannot be hidden by a shell pipeline.
"""

from __future__ import annotations

import os
import math
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TextIO

from src.utils.hashing import is_sha256, sha256_file
from src.utils.io import (
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_yaml,
    utc_now_iso,
)

STATE_SCHEMA = "graphrestore-orchestration-v1"
APPROVAL_SCHEMA = "graphrestore-stage3-approval-v1"
PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"


class PipelineStatus(str, Enum):
    PRE_STAGE0 = "PRE_STAGE0"
    PREFLIGHT_RUNNING = "PREFLIGHT_RUNNING"
    PREFLIGHT_COMPLETE = "PREFLIGHT_COMPLETE"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    INTEGRATION_RUNNING = "INTEGRATION_RUNNING"
    INTEGRATION_FAILED = "INTEGRATION_FAILED"
    READY_FOR_MAIN = "READY_FOR_MAIN"
    STAGE0_RUNNING = "STAGE0_RUNNING"
    STAGE1_RUNNING = "STAGE1_RUNNING"
    EFFECT_PROFILES_RUNNING = "EFFECT_PROFILES_RUNNING"
    STAGE2_DISTILL_RUNNING = "STAGE2_DISTILL_RUNNING"
    PAUSED_AFTER_STAGE2 = "PAUSED_AFTER_STAGE2"
    STAGE3_APPROVED = "STAGE3_APPROVED"
    STAGE3_RUNNING = "STAGE3_RUNNING"
    STAGE4_RUNNING = "STAGE4_RUNNING"
    STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION = (
        "STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION"
    )
    FAILED = "FAILED"


class OrchestrationError(RuntimeError):
    """A fail-closed orchestration invariant was violated."""

    exit_code = 2


class ApprovalError(OrchestrationError):
    """Stage3 approval is absent, stale, or does not match frozen artifacts."""

    exit_code = 3


class ChildCommandError(OrchestrationError):
    """A child command failed and its exit code must propagate."""

    def __init__(self, command: "CommandSpec", returncode: int) -> None:
        self.command = command
        self.returncode = _normalise_returncode(returncode)
        super().__init__(
            f"child command {command.name!r} failed with exit code {self.returncode}: "
            f"{shlex.join(command.argv)}"
        )

    @property
    def exit_code(self) -> int:  # type: ignore[override]
        return self.returncode


@dataclass(frozen=True)
class CommandSpec:
    """One named child process invocation."""

    name: str
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("command name must not be empty")
        if not self.argv or any(not isinstance(argument, str) or not argument for argument in self.argv):
            raise ValueError("command argv must contain non-empty strings")


class CommandRunner(Protocol):
    """Injectable child runner used by production and CPU-only tests."""

    def run(
        self,
        command: CommandSpec,
        *,
        cwd: Path,
        log_path: Path,
    ) -> int: ...


class SubprocessCommandRunner:
    """Stream a child process to stdout and an append-only log."""

    def __init__(self, *, output: TextIO | None = None) -> None:
        self.output = output if output is not None else sys.stdout

    def run(
        self,
        command: CommandSpec,
        *,
        cwd: Path,
        log_path: Path,
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{cwd}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(cwd)
        )
        environment["PYTHONUNBUFFERED"] = "1"

        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            header = f"[{utc_now_iso()}] START {command.name}: {shlex.join(command.argv)}\n"
            log.write(header)
            log.flush()
            self.output.write(header)
            self.output.flush()
            try:
                process = subprocess.Popen(
                    list(command.argv),
                    cwd=cwd,
                    env=environment,
                    stdin=None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                )
            except FileNotFoundError as exc:
                line = f"[{utc_now_iso()}] EXEC_ERROR {command.name}: {exc}\n"
                log.write(line)
                log.flush()
                self.output.write(line)
                self.output.flush()
                return 127

            if process.stdout is None:  # pragma: no cover - Popen contract guard
                process.kill()
                process.wait()
                return 126
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self.output.write(line)
                    self.output.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()

            returncode = _normalise_returncode(returncode)
            footer = f"[{utc_now_iso()}] END {command.name}: exit={returncode}\n"
            log.write(footer)
            log.flush()
            self.output.write(footer)
            self.output.flush()
            return returncode


def _normalise_returncode(returncode: int) -> int:
    """Convert a signal-style negative code to conventional shell status."""

    if returncode < 0:
        return min(255, 128 + abs(returncode))
    return min(255, returncode)


@dataclass
class PipelineState:
    """Small durable state; training details remain in stage checkpoints/logs."""

    schema_version: str = STATE_SCHEMA
    protocol_id: str = PROTOCOL_ID
    status: str = PipelineStatus.PRE_STAGE0.value
    current_stage: str = "PRE_STAGE0"
    current_step: int = 0
    completed: list[str] = field(default_factory=list)
    integration_steps: int | None = None
    last_command: list[str] | None = None
    last_exit_code: int | None = None
    last_checkpoint: str | None = None
    stage2_decision_sha256: str | None = None
    stage3_approval_sha256: str | None = None
    gpu: str = "released"
    recent_validation: str = "none"
    peak_vram: str = "not_measured"
    throughput: str = "not_measured"
    next_command: str = "python scripts/orchestrate.py --integration_steps 100"
    last_error: str | None = None
    created_utc: str = field(default_factory=utc_now_iso)
    updated_utc: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineState":
        if value.get("schema_version") != STATE_SCHEMA:
            raise OrchestrationError(
                f"unsupported orchestration state schema: {value.get('schema_version')!r}"
            )
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise OrchestrationError(f"unknown orchestration state fields: {unknown}")
        state = cls(**dict(value))
        try:
            PipelineStatus(state.status)
        except ValueError as exc:
            raise OrchestrationError(f"unknown pipeline status: {state.status!r}") from exc
        if not isinstance(state.completed, list) or not all(
            isinstance(item, str) for item in state.completed
        ):
            raise OrchestrationError("state.completed must be a list of strings")
        return state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrchestrationPaths:
    project_root: Path

    @property
    def state(self) -> Path:
        return self.project_root / "artifacts" / "orchestration" / "state.json"

    @property
    def status(self) -> Path:
        return self.project_root / "RUNNING_STATUS.md"

    @property
    def integration_dir(self) -> Path:
        return self.project_root / "artifacts/integration/stage0_100_steps"

    @property
    def log(self) -> Path:
        return self.project_root / "artifacts" / "logs" / "main_pipeline.log"

    @property
    def approval_required(self) -> Path:
        return self.project_root / "artifacts" / "approvals" / "STAGE3_APPROVAL_REQUIRED.json"

    @property
    def approval_granted(self) -> Path:
        return self.project_root / "artifacts" / "approvals" / "STAGE3_APPROVED.json"

    @property
    def stage2_decision(self) -> Path:
        return self.project_root / "artifacts" / "interaction_labels" / "stage2_decision.json"


class GraphRestoreOrchestrator:
    """Run the frozen V7.1 state machine without embedding stage algorithms."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        runner: CommandRunner | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        self.paths = OrchestrationPaths(Path(project_root).resolve())
        self.runner = runner if runner is not None else SubprocessCommandRunner()
        self.python = str(python_executable or sys.executable)

    def _python_command(self, name: str, script: str, *arguments: str) -> CommandSpec:
        return CommandSpec(name=name, argv=(self.python, script, *arguments))

    def _required_preflight_completions(self) -> set[str]:
        return {command.name for command in self.preflight_commands()}

    def _require_completion_evidence(
        self,
        state: PipelineState,
        required: Iterable[str],
        *,
        context: str,
    ) -> None:
        missing = sorted(set(required).difference(state.completed))
        if missing:
            raise OrchestrationError(
                f"{context} lacks successful command evidence: {missing}"
            )

    def preflight_commands(self) -> tuple[CommandSpec, ...]:
        return (
            self._python_command("audit_data", "scripts/audit_data.py"),
            self._python_command(
                "build_online_canonical_manifests",
                "scripts/build_agenticir_online_canonical_manifests.py",
            ),
            self._python_command("audit_metric_parity", "scripts/audit_metric_parity.py"),
            self._python_command(
                "audit_degradation_parity",
                "scripts/audit_degradation_parity.py",
            ),
            CommandSpec(
                name="mandatory_pytests",
                # Run the complete CPU suite so newly added cross-stage resume,
                # EMA, sampler and adjudication regressions cannot sit outside
                # the formal gate. CUDA one-batch cases remain explicit below.
                argv=(self.python, "-m", "pytest", "-q", "tests"),
            ),
            self._python_command(
                "probe_validation_vram",
                "scripts/probe_validation_vram.py",
            ),
            self._python_command(
                "one_batch_single", "tests/test_one_batch.py", "--case", "single"
            ),
            self._python_command(
                "one_batch_group_a_low_resolution",
                "tests/test_one_batch.py",
                "--case",
                "group_a_low_resolution",
            ),
            self._python_command(
                "profile_stage0_compile",
                "scripts/profile_stage0_compile.py",
                "--config",
                "configs/stage0_mio_stagea.yaml",
            ),
        )

    def integration_command(self, steps: int) -> CommandSpec:
        return self._python_command(
            "integration_100_steps",
            "scripts/train_stage0.py",
            "--config",
            "configs/stage0_mio_stagea.yaml",
            "--integration_steps",
            str(steps),
            "--output_dir",
            "artifacts/integration/stage0_100_steps",
        )

    def main_stage_commands(self) -> tuple[tuple[PipelineStatus, CommandSpec], ...]:
        return (
            (
                PipelineStatus.STAGE0_RUNNING,
                self._python_command(
                    "stage0", "scripts/train_stage0.py", "--config", "configs/stage0_mio_stagea.yaml"
                ),
            ),
            (
                PipelineStatus.STAGE1_RUNNING,
                self._python_command(
                    "stage1",
                    "scripts/train_stage1_skills.py",
                    "--config",
                    "configs/stage1_skill_bank.yaml",
                ),
            ),
            (
                PipelineStatus.EFFECT_PROFILES_RUNNING,
                self._python_command(
                    "effect_profiles",
                    "scripts/build_skill_effect_profiles.py",
                    "--config",
                    "configs/stage2_interaction_distill.yaml",
                ),
            ),
            (
                PipelineStatus.STAGE2_DISTILL_RUNNING,
                self._python_command(
                    "stage2_distill",
                    "scripts/distill_interactions.py",
                    "--config",
                    "configs/stage2_interaction_distill.yaml",
                ),
            ),
        )

    def post_approval_commands(self) -> tuple[tuple[PipelineStatus, CommandSpec], ...]:
        return (
            (
                PipelineStatus.STAGE3_RUNNING,
                self._python_command(
                    "stage3",
                    "scripts/train_stage3_planner.py",
                    "--config",
                    "configs/stage3_planner.yaml",
                ),
            ),
            (
                PipelineStatus.STAGE4_RUNNING,
                self._python_command(
                    "stage4",
                    "scripts/train_stage4_e2e.py",
                    "--config",
                    "configs/stage4_graphrestore_e2e.yaml",
                ),
            ),
        )

    def load_state(self) -> PipelineState:
        if not self.paths.state.exists():
            return PipelineState()
        value = load_json(self.paths.state)
        if not isinstance(value, Mapping):
            raise OrchestrationError("orchestration state must be a JSON object")
        return PipelineState.from_dict(value)

    def _persist(self, state: PipelineState) -> None:
        state.updated_utc = utc_now_iso()
        atomic_write_json(self.paths.state, state.to_dict())
        atomic_write_text(self.paths.status, self._render_running_status(state))

    def _render_running_status(self, state: PipelineState) -> str:
        error = state.last_error if state.last_error is not None else "none"
        checkpoint = state.last_checkpoint if state.last_checkpoint is not None else "none"
        decision_sha = (
            state.stage2_decision_sha256
            if state.stage2_decision_sha256 is not None
            else "none"
        )
        approval_sha = (
            state.stage3_approval_sha256
            if state.stage3_approval_sha256 is not None
            else "none"
        )
        return (
            f"status: {state.status}\n"
            f"current_stage: {state.current_stage}\n"
            f"current_step: {state.current_step}\n"
            f"recent_validation: {state.recent_validation}\n"
            f"gpu: {state.gpu}\n"
            f"peak_vram: {state.peak_vram}\n"
            f"throughput: {state.throughput}\n"
            f"last_checkpoint: {checkpoint}\n"
            f"stage2_decision_sha256: {decision_sha}\n"
            f"stage3_approval_sha256: {approval_sha}\n"
            f"last_error: {error}\n"
            f"next_command: {state.next_command}\n"
            f"orchestration_state: {self.paths.state}\n"
            f"updated_utc: {state.updated_utc}\n"
        )

    def _run_child(self, state: PipelineState, status: PipelineStatus, command: CommandSpec) -> None:
        state.status = status.value
        state.current_stage = status.value.removesuffix("_RUNNING")
        state.current_step = 0
        state.last_command = list(command.argv)
        state.last_exit_code = None
        state.last_error = None
        state.gpu = "owned_by_child_process"
        state.next_command = "wait_for_current_child"
        self._persist(state)
        try:
            returncode = self.runner.run(
                command,
                cwd=self.paths.project_root,
                log_path=self.paths.log,
            )
        except KeyboardInterrupt:
            state.status = PipelineStatus.FAILED.value
            state.gpu = "released"
            state.last_exit_code = 130
            state.last_error = f"interrupted while running {command.name}"
            state.next_command = "inspect artifacts/logs/main_pipeline.log"
            self._persist(state)
            raise
        returncode = _normalise_returncode(returncode)
        state.last_exit_code = returncode
        if returncode != 0:
            state.status = PipelineStatus.FAILED.value
            state.gpu = "released"
            state.last_error = f"{command.name} exited with {returncode}"
            state.next_command = "inspect artifacts/logs/main_pipeline.log"
            self._persist(state)
            raise ChildCommandError(command, returncode)
        if command.name not in state.completed:
            state.completed.append(command.name)
        state.gpu = "released"
        self._persist(state)

    def _mark_failed(self, state: PipelineState, error: BaseException) -> None:
        """Durably release ownership and record a non-child invariant failure."""

        state.status = PipelineStatus.FAILED.value
        state.current_stage = "FAILED"
        state.gpu = "released"
        state.last_error = str(error)
        state.next_command = "inspect artifacts/logs/main_pipeline.log and state.json"
        self._persist(state)

    def _matching_live_child_pids(self, state: PipelineState) -> tuple[int, ...]:
        """Find an exact still-running child for a durable *_RUNNING state."""

        if not state.last_command:
            raise OrchestrationError(
                f"{state.status} state has no durable last_command; cannot prove it stale"
            )
        expected = tuple(state.last_command)
        matches: list[int] = []
        proc_root = Path("/proc")
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
                command = tuple(
                    item.decode("utf-8", errors="surrogateescape")
                    for item in raw.split(b"\0")
                    if item
                )
                process_cwd = (entry / "cwd").resolve(strict=True)
                stat_fields = (entry / "stat").read_text(encoding="utf-8").split()
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            if len(stat_fields) >= 3 and stat_fields[2] == "Z":
                continue
            if command == expected and process_cwd == self.paths.project_root:
                matches.append(pid)
        return tuple(sorted(matches))

    def _recover_stale_running(
        self,
        state: PipelineState,
        *,
        allowed_running: set[str],
        recovered_status: PipelineStatus,
    ) -> PipelineState:
        """Explicitly convert a proven-stale durable RUNNING state."""

        if state.status not in allowed_running:
            return state
        live = self._matching_live_child_pids(state)
        if live:
            raise OrchestrationError(
                f"refusing stale-state recovery: child process is still live: pids={live}"
            )
        previous = state.status
        state.status = recovered_status.value
        state.current_stage = recovered_status.value
        state.gpu = "released"
        state.last_exit_code = None
        state.last_error = (
            f"recovered proven-stale durable state {previous}; no exact live child found"
        )
        state.next_command = (
            "resume the interrupted pipeline from durable checkpoints/shards"
        )
        self._persist(state)
        return state

    def run_preflight(self) -> PipelineState:
        state = self.load_state()
        state = self._recover_stale_running(
            state,
            allowed_running={PipelineStatus.PREFLIGHT_RUNNING.value},
            recovered_status=PipelineStatus.PREFLIGHT_FAILED,
        )
        allowed = {
            PipelineStatus.PRE_STAGE0.value,
            PipelineStatus.PREFLIGHT_FAILED.value,
        }
        if state.status == PipelineStatus.PREFLIGHT_COMPLETE.value:
            self._require_completion_evidence(
                state,
                self._required_preflight_completions(),
                context="PREFLIGHT_COMPLETE state",
            )
            return state
        if state.status not in allowed:
            raise OrchestrationError(
                f"preflight cannot start from state {state.status}; refusing implicit rewind"
            )
        for command in self.preflight_commands():
            try:
                self._run_child(state, PipelineStatus.PREFLIGHT_RUNNING, command)
            except ChildCommandError:
                state.status = PipelineStatus.PREFLIGHT_FAILED.value
                self._persist(state)
                raise
        state.status = PipelineStatus.PREFLIGHT_COMPLETE.value
        state.current_stage = "PREFLIGHT_COMPLETE"
        state.gpu = "released"
        state.next_command = "python scripts/orchestrate.py --integration_steps 100"
        self._persist(state)
        return state

    def run_integration(self, steps: int) -> PipelineState:
        if steps != 100:
            raise OrchestrationError(
                f"V7.1 requires exactly 100 optimizer steps for integration, got {steps}"
            )
        state = self.load_state()
        state = self._recover_stale_running(
            state,
            allowed_running={PipelineStatus.INTEGRATION_RUNNING.value},
            recovered_status=PipelineStatus.INTEGRATION_FAILED,
        )
        if state.status in {
            PipelineStatus.PRE_STAGE0.value,
            PipelineStatus.PREFLIGHT_FAILED.value,
        }:
            state = self.run_preflight()
        if state.status == PipelineStatus.READY_FOR_MAIN.value:
            if state.integration_steps != 100:
                raise OrchestrationError("READY_FOR_MAIN state lacks the frozen 100-step proof")
            self._require_completion_evidence(
                state,
                {*self._required_preflight_completions(), "integration_100_steps"},
                context="READY_FOR_MAIN state",
            )
            self._verify_integration_outputs()
            return state
        if state.status not in {
            PipelineStatus.PREFLIGHT_COMPLETE.value,
            PipelineStatus.INTEGRATION_FAILED.value,
        }:
            raise OrchestrationError(
                f"integration cannot start from state {state.status}; refusing implicit rewind"
            )
        command = self.integration_command(steps)
        integration_last = self.paths.integration_dir / "last.pth"
        if (
            state.status == PipelineStatus.INTEGRATION_FAILED.value
            and integration_last.is_file()
        ):
            command = CommandSpec(
                name=command.name,
                argv=(*command.argv, "--resume", str(integration_last)),
            )
        try:
            self._run_child(state, PipelineStatus.INTEGRATION_RUNNING, command)
        except ChildCommandError:
            state.status = PipelineStatus.INTEGRATION_FAILED.value
            self._persist(state)
            raise
        try:
            self._verify_integration_outputs()
        except OrchestrationError as exc:
            if command.name in state.completed:
                state.completed.remove(command.name)
            state.status = PipelineStatus.INTEGRATION_FAILED.value
            state.current_stage = "INTEGRATION_FAILED"
            state.gpu = "released"
            state.last_error = str(exc)
            state.next_command = "inspect artifacts/integration/stage0_100_steps"
            self._persist(state)
            raise
        state.integration_steps = steps
        state.status = PipelineStatus.READY_FOR_MAIN.value
        state.current_stage = "READY_FOR_MAIN"
        state.gpu = "released"
        state.next_command = recommended_tmux_command(self.paths.project_root, self.python)
        self._persist(state)
        return state

    def _verify_integration_outputs(self) -> None:
        directory = self.paths.integration_dir
        summary_path = directory / "summary.json"
        required = (
            summary_path,
            directory / "last.pth",
            directory / "INTEGRATION_REPORT.md",
            directory / "micro_batch_probe.json",
        )
        self._require_files(required, context="100-step integration")
        value = load_json(summary_path)
        if not isinstance(value, Mapping):
            raise OrchestrationError("integration summary must be a mapping")
        expected = {
            "schema_version": "graphrestore-stage0-run-v1",
            "protocol_id": PROTOCOL_ID,
            "integration": True,
            "completed_step": 100,
            "target_step": 100,
            "finite": True,
        }
        mismatches = {
            key: {"expected": expected_value, "actual": value.get(key)}
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        }
        runtime = value.get("runtime")
        if not isinstance(runtime, Mapping):
            mismatches["runtime"] = {"expected": "mapping", "actual": runtime}
        else:
            for key, expected_value in (
                ("crop_size", 192),
                ("effective_batch", 8),
                ("target_step", 100),
                ("integration", True),
            ):
                if runtime.get(key) != expected_value:
                    mismatches[f"runtime.{key}"] = {
                        "expected": expected_value,
                        "actual": runtime.get(key),
                    }
        peak = value.get("peak_reserved_fraction")
        if (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(float(peak))
            or float(peak) > 0.90
        ):
            mismatches["peak_reserved_fraction"] = {
                "expected": "finite <= 0.90",
                "actual": peak,
            }
        checkpoint = value.get("last_checkpoint")
        expected_checkpoint = str((directory / "last.pth").resolve())
        if checkpoint != expected_checkpoint:
            mismatches["last_checkpoint"] = {
                "expected": expected_checkpoint,
                "actual": checkpoint,
            }
        if mismatches:
            raise OrchestrationError(f"100-step integration evidence mismatch: {mismatches}")

    def run_main_pipeline(self) -> PipelineState:
        state = self.load_state()
        if state.status != PipelineStatus.READY_FOR_MAIN.value or state.integration_steps != 100:
            raise OrchestrationError(
                "main pipeline requires a successful preflight and exact 100-step integration"
            )
        self._require_completion_evidence(
            state,
            {*self._required_preflight_completions(), "integration_100_steps"},
            context="Stage0 hard gate",
        )
        if self.paths.approval_granted.exists():
            raise ApprovalError(
                f"stale Stage3 approval exists before Stage2 pause: {self.paths.approval_granted}"
            )

        return self._continue_main_pipeline(state, resume_failed=False)

    def resume_main_pipeline(self) -> PipelineState:
        """Explicitly resume the pre-approval pipeline from durable stage evidence."""

        state = self.load_state()
        state = self._recover_stale_running(
            state,
            allowed_running={
                PipelineStatus.STAGE0_RUNNING.value,
                PipelineStatus.STAGE1_RUNNING.value,
                PipelineStatus.EFFECT_PROFILES_RUNNING.value,
                PipelineStatus.STAGE2_DISTILL_RUNNING.value,
            },
            recovered_status=PipelineStatus.FAILED,
        )
        if state.status != PipelineStatus.FAILED.value:
            raise OrchestrationError(
                "--resume_main_pipeline is allowed only from FAILED state"
            )
        if state.integration_steps != 100:
            raise OrchestrationError("failed main pipeline lacks exact 100-step integration")
        self._require_completion_evidence(
            state,
            {*self._required_preflight_completions(), "integration_100_steps"},
            context="main-pipeline resume gate",
        )
        if self.paths.approval_granted.exists():
            raise ApprovalError("pre-Stage3 main resume refuses an approval artifact")
        return self._continue_main_pipeline(state, resume_failed=True)

    def _main_expected_outputs(self) -> dict[str, tuple[Path, ...]]:
        return {
            "stage0": (
                self.paths.project_root / "artifacts/checkpoints/stage0/best_ema.pth",
            ),
            "stage1": (
                self.paths.project_root / "artifacts/checkpoints/stage1/best_ema.pth",
            ),
            "effect_profiles": (
                self.paths.project_root
                / "artifacts/interaction_labels/skill_effect_profiles.json",
            ),
            "stage2_distill": tuple(self._stage2_artifact_paths().values()),
        }

    def _resumable_main_command(self, command: CommandSpec) -> CommandSpec:
        checkpoint_paths = {
            "stage0": self.paths.project_root
            / "artifacts/checkpoints/stage0/last.pth",
            "stage1": self.paths.project_root
            / "artifacts/checkpoints/stage1/last.pth",
        }
        checkpoint = checkpoint_paths.get(command.name)
        if checkpoint is None or not checkpoint.is_file():
            return command
        return CommandSpec(
            name=command.name,
            argv=(*command.argv, "--resume", str(checkpoint)),
        )

    def _continue_main_pipeline(
        self,
        state: PipelineState,
        *,
        resume_failed: bool,
    ) -> PipelineState:
        """Run or explicitly resume Stage0→Stage2 while preserving evidence."""

        expected_outputs = self._main_expected_outputs()
        for status, base_command in self.main_stage_commands():
            if base_command.name in state.completed:
                # Successful child evidence is immutable: never silently rerun it.
                try:
                    self._require_files(
                        expected_outputs[base_command.name],
                        context=f"completed {base_command.name}",
                    )
                except OrchestrationError as exc:
                    self._mark_failed(state, exc)
                    raise
                if base_command.name in {"stage0", "stage1"}:
                    state.last_checkpoint = str(
                        expected_outputs[base_command.name][0]
                    )
                continue
            command = (
                self._resumable_main_command(base_command)
                if resume_failed
                else base_command
            )
            self._run_child(state, status, command)
            try:
                self._require_files(expected_outputs[command.name], context=command.name)
            except OrchestrationError as exc:
                self._mark_failed(state, exc)
                raise
            if command.name == "stage0":
                state.last_checkpoint = str(expected_outputs[command.name][0])
            elif command.name == "stage1":
                state.last_checkpoint = str(expected_outputs[command.name][0])
            self._persist(state)

        try:
            required = self._create_approval_required()
        except Exception as exc:
            self._mark_failed(state, exc)
            raise
        state.stage2_decision_sha256 = required["stage2_decision"]["sha256"]
        state.status = PipelineStatus.PAUSED_AFTER_STAGE2.value
        state.current_stage = "PAUSED_AFTER_STAGE2"
        state.current_step = 0
        state.gpu = "released"
        state.last_error = None
        state.next_command = (
            "python scripts/orchestrate.py --approve_stage3 --resume_from_stage3"
        )
        self._persist(state)
        return state

    def approve_and_resume_stage3(
        self,
        *,
        approve_stage3: bool,
        resume_from_stage3: bool,
    ) -> PipelineState:
        if not approve_stage3 or not resume_from_stage3:
            raise ApprovalError(
                "Stage3 requires both explicit flags: --approve_stage3 --resume_from_stage3"
            )
        state = self.load_state()
        if state.status != PipelineStatus.PAUSED_AFTER_STAGE2.value:
            raise ApprovalError(
                f"Stage3 can start only from PAUSED_AFTER_STAGE2, got {state.status}"
            )
        try:
            required = self._load_and_verify_approval_required(state)
        except Exception as exc:
            approval_error = (
                exc
                if isinstance(exc, ApprovalError)
                else ApprovalError(f"could not verify frozen Stage2 artifacts: {exc}")
            )
            state.status = PipelineStatus.PAUSED_AFTER_STAGE2.value
            state.current_stage = "PAUSED_AFTER_STAGE2"
            state.gpu = "released"
            state.last_error = str(approval_error)
            state.next_command = (
                "fix hash mismatch; then rerun python scripts/orchestrate.py "
                "--approve_stage3 --resume_from_stage3"
            )
            self._persist(state)
            if approval_error is exc:
                raise
            raise approval_error from exc

        approval_payload = {
            "schema_version": APPROVAL_SCHEMA,
            "kind": "stage3_approval",
            "protocol_id": PROTOCOL_ID,
            "approved": True,
            "approved_utc": utc_now_iso(),
            "stage2_decision_path": required["stage2_decision"]["path"],
            "stage2_decision_sha256": required["stage2_decision"]["sha256"],
            "approval_required_path": str(self.paths.approval_required),
            "approval_required_sha256": sha256_file(self.paths.approval_required),
            "bindings": required["bindings"],
            "approval_command": [
                self.python,
                "scripts/orchestrate.py",
                "--approve_stage3",
                "--resume_from_stage3",
            ],
        }
        atomic_write_json(self.paths.approval_granted, approval_payload)
        state.stage3_approval_sha256 = sha256_file(self.paths.approval_granted)
        state.status = PipelineStatus.STAGE3_APPROVED.value
        state.current_stage = "STAGE3_APPROVED"
        state.gpu = "released"
        state.last_error = None
        state.next_command = "run_stage3_then_stage4_only"
        self._persist(state)

        expected_outputs = {
            "stage3": self.paths.project_root
            / "artifacts/checkpoints/stage3/best_ema.pth",
            "stage4": self.paths.project_root
            / "artifacts/checkpoints/stage4/best_ema.pth",
        }
        for status, command in self.post_approval_commands():
            self._run_child(state, status, command)
            try:
                self._require_files(
                    (expected_outputs[command.name],), context=command.name
                )
            except OrchestrationError as exc:
                self._mark_failed(state, exc)
                raise
            state.last_checkpoint = str(expected_outputs[command.name])
            self._persist(state)

        state.status = (
            PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
        )
        state.current_stage = "STAGE4_COMPLETE"
        state.current_step = 0
        state.gpu = "released"
        state.last_error = None
        state.next_command = "await_explicit_user_authorization_for_formal_mio100"
        self._persist(state)
        return state

    def _require_files(self, paths: Iterable[Path], *, context: str) -> None:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise OrchestrationError(
                f"{context} returned success but required outputs are missing: {missing}"
            )

    def _stage2_artifact_paths(self) -> dict[str, Path]:
        root = self.paths.project_root
        return {
            "stage1_checkpoint": root / "artifacts/checkpoints/stage1/best_ema.pth",
            "skill_effect_profiles": root
            / "artifacts/interaction_labels/skill_effect_profiles.json",
            "interaction_train_manifest": root
            / "artifacts/interaction_labels/interaction_train_manifest.jsonl",
            "interaction_val_manifest": root
            / "artifacts/interaction_labels/interaction_val_manifest.jsonl",
            "relation_train": root
            / "artifacts/interaction_labels/group_a_relations_train.jsonl",
            "relation_val": root
            / "artifacts/interaction_labels/group_a_relations_val.jsonl",
            "pair_prior": root / "artifacts/interaction_labels/pair_prior.json",
            "global_priority": root / "artifacts/interaction_labels/global_priority.json",
            "interaction_summary_csv": root
            / "artifacts/metrics/stage2_interaction_summary.csv",
            "interaction_report": root / "reports/INTERACTION_DISTILLATION.md",
            "stage2_decision": self.paths.stage2_decision,
        }

    def _binding_paths(self) -> dict[str, Path]:
        root = self.paths.project_root
        bindings = self._stage2_artifact_paths()
        bindings.update(
            {
                "config_resolved_paths": root / "configs/resolved_paths.yaml",
                "config_stage0": root / "configs/stage0_mio_stagea.yaml",
                "config_stage1": root / "configs/stage1_skill_bank.yaml",
                "config_stage2": root / "configs/stage2_interaction_distill.yaml",
                "config_stage3": root / "configs/stage3_planner.yaml",
                "config_stage4": root / "configs/stage4_graphrestore_e2e.yaml",
            }
        )
        resolved_path = bindings["config_resolved_paths"]
        if not resolved_path.is_file():
            raise OrchestrationError(f"missing resolved-path config: {resolved_path}")
        resolved = load_yaml(resolved_path)
        if not isinstance(resolved, Mapping):
            raise OrchestrationError("configs/resolved_paths.yaml must be a mapping")
        for logical, key in (
            ("clean_train_manifest", "clean_train_manifest"),
            ("clean_val_manifest", "clean_val_manifest"),
            ("primary_train_manifest", "primary_train_manifest"),
            ("primary_val_manifest", "primary_val_manifest"),
            ("primary_all_manifest", "primary_all_manifest"),
        ):
            raw_path = resolved.get(key)
            if not isinstance(raw_path, str) or not raw_path:
                raise OrchestrationError(f"resolved path {key!r} is missing")
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            bindings[logical] = path.resolve(strict=False)
        return bindings

    def _hash_bindings(self) -> dict[str, dict[str, str]]:
        bindings: dict[str, dict[str, str]] = {}
        for logical, path in sorted(self._binding_paths().items()):
            if not path.is_file():
                raise OrchestrationError(f"cannot bind missing artifact {logical}: {path}")
            bindings[logical] = {
                "path": str(path.resolve(strict=False)),
                "sha256": sha256_file(path),
            }
        return bindings

    def _load_stage2_decision(self) -> dict[str, Any]:
        if not self.paths.stage2_decision.is_file():
            raise OrchestrationError(
                f"Stage2 did not write decision file: {self.paths.stage2_decision}"
            )
        decision = load_json(self.paths.stage2_decision)
        if not isinstance(decision, dict):
            raise OrchestrationError("stage2_decision.json must be a JSON object")
        return decision

    def _verify_decision_against_bindings(
        self,
        decision: Mapping[str, Any],
        bindings: Mapping[str, Mapping[str, str]],
        *,
        error_type: type[OrchestrationError] = OrchestrationError,
    ) -> None:
        expected_fields = {
            "stage1_checkpoint_sha256": bindings["stage1_checkpoint"]["sha256"],
            "interaction_train_manifest_sha256": bindings["interaction_train_manifest"][
                "sha256"
            ],
            "interaction_val_manifest_sha256": bindings["interaction_val_manifest"][
                "sha256"
            ],
            "relation_train_sha256": bindings["relation_train"]["sha256"],
            "relation_val_sha256": bindings["relation_val"]["sha256"],
            "pair_prior_sha256": bindings["pair_prior"]["sha256"],
            "global_priority_sha256": bindings["global_priority"]["sha256"],
            "config_sha256": bindings["config_stage2"]["sha256"],
        }
        mismatches = {
            key: {"expected": expected_sha, "actual": decision.get(key)}
            for key, expected_sha in expected_fields.items()
            if decision.get(key) != expected_sha or not is_sha256(decision.get(key))
        }
        if decision.get("approved") is not False:
            mismatches["approved"] = {"expected": False, "actual": decision.get("approved")}
        if not isinstance(decision.get("overall"), Mapping):
            mismatches["overall"] = {"expected": "mapping", "actual": type(decision.get("overall")).__name__}
        if not isinstance(decision.get("warnings"), list):
            mismatches["warnings"] = {"expected": "list", "actual": type(decision.get("warnings")).__name__}
        if mismatches:
            raise error_type(f"Stage2 decision binding mismatch: {mismatches}")

    def _create_approval_required(self) -> dict[str, Any]:
        bindings = self._hash_bindings()
        decision = self._load_stage2_decision()
        self._verify_decision_against_bindings(decision, bindings)
        decision_binding = bindings["stage2_decision"]
        payload: dict[str, Any] = {
            "schema_version": APPROVAL_SCHEMA,
            "kind": "stage3_approval_required",
            "protocol_id": PROTOCOL_ID,
            "approved": False,
            "created_utc": utc_now_iso(),
            "stage2_decision": decision_binding,
            "bindings": bindings,
            "warnings": list(decision["warnings"]),
            "resume_command": (
                "python scripts/orchestrate.py --approve_stage3 --resume_from_stage3"
            ),
        }
        atomic_write_json(self.paths.approval_required, payload)
        return payload

    def _load_and_verify_approval_required(
        self, state: PipelineState
    ) -> dict[str, Any]:
        if not self.paths.approval_required.is_file():
            raise ApprovalError(
                f"missing Stage3 approval requirement: {self.paths.approval_required}"
            )
        payload = load_json(self.paths.approval_required)
        if not isinstance(payload, dict):
            raise ApprovalError("STAGE3_APPROVAL_REQUIRED.json must be a JSON object")
        if payload.get("schema_version") != APPROVAL_SCHEMA:
            raise ApprovalError("unsupported Stage3 approval schema")
        if payload.get("kind") != "stage3_approval_required" or payload.get("approved") is not False:
            raise ApprovalError("invalid Stage3 approval-required marker")
        recorded_bindings = payload.get("bindings")
        if not isinstance(recorded_bindings, Mapping):
            raise ApprovalError("approval-required bindings are missing")
        current_bindings = self._hash_bindings()
        if dict(recorded_bindings) != current_bindings:
            differences = _mapping_differences(recorded_bindings, current_bindings)
            raise ApprovalError(f"frozen Stage2/config/manifest hashes changed: {differences}")
        stage2_binding = payload.get("stage2_decision")
        expected_decision_binding = current_bindings["stage2_decision"]
        if stage2_binding != expected_decision_binding:
            raise ApprovalError(
                "approval-required stage2_decision binding differs from current artifact"
            )
        if state.stage2_decision_sha256 != expected_decision_binding["sha256"]:
            raise ApprovalError(
                "durable state stage2_decision SHA differs from approval-required artifact"
            )
        decision = self._load_stage2_decision()
        self._verify_decision_against_bindings(
            decision, current_bindings, error_type=ApprovalError
        )
        return payload


def _mapping_differences(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    keys = sorted(set(recorded).union(current))
    return {
        key: {"recorded": recorded.get(key), "current": current.get(key)}
        for key in keys
        if recorded.get(key) != current.get(key)
    }


def recommended_tmux_argv(
    project_root: str | Path,
    python_executable: str | Path | None = None,
) -> tuple[str, ...]:
    """Return the non-executing, pipefail-safe tmux launch argv."""

    root = Path(project_root).resolve()
    python = str(python_executable or sys.executable)
    pipeline = (
        f"cd {shlex.quote(str(root))} && "
        f"{shlex.quote(python)} scripts/orchestrate.py --run_main_pipeline"
    )
    return (
        "tmux",
        "new-session",
        "-d",
        "-s",
        "graphrestore",
        "bash",
        "-o",
        "pipefail",
        "-c",
        pipeline,
    )


def recommended_tmux_command(
    project_root: str | Path,
    python_executable: str | Path | None = None,
) -> str:
    """Render the safe tmux argv for status/report display only."""

    return shlex.join(recommended_tmux_argv(project_root, python_executable))


def command_plan(orchestrator: GraphRestoreOrchestrator) -> dict[str, Any]:
    """Return a read-only plan suitable for ``--print_plan``."""

    return {
        "preflight": [asdict(command) for command in orchestrator.preflight_commands()],
        "integration": asdict(orchestrator.integration_command(100)),
        "main_before_pause": [
            {"status": status.value, **asdict(command)}
            for status, command in orchestrator.main_stage_commands()
        ],
        "mandatory_pause": PipelineStatus.PAUSED_AFTER_STAGE2.value,
        "approval_command": (
            "python scripts/orchestrate.py --approve_stage3 --resume_from_stage3"
        ),
        "after_approval": [
            {"status": status.value, **asdict(command)}
            for status, command in orchestrator.post_approval_commands()
        ],
        "formal_mio100_in_automatic_pipeline": False,
        "tmux_argv": list(
            recommended_tmux_argv(orchestrator.paths.project_root, orchestrator.python)
        ),
    }
