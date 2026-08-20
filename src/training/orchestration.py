"""Fail-closed GraphRestore V7.1 pipeline orchestration.

This module owns process sequencing, durable state, and the Stage2 approval
barrier.  It deliberately contains no training or evaluation algorithms.
Every child is invoked with an argv sequence and ``shell=False`` so its exit
status cannot be hidden by a shell pipeline.
"""

from __future__ import annotations

import math
import os
import shlex
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TextIO

from src.training.stage3_finalization import (
    Stage3RevocationAuthorization,
    validate_stage3_extension_revocation,
)
from src.utils.hashing import is_sha256, sha256_file
from src.utils.io import (
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_json,
    load_yaml,
    utc_now_iso,
)

STATE_SCHEMA = "graphrestore-orchestration-v1"
APPROVAL_SCHEMA = "graphrestore-stage3-approval-v1"
STAGE3_EXTENSION_APPROVAL_SCHEMA = "graphrestore-stage3-extension-approval-v1"
PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
CUDA_ALLOCATOR_CONF = "backend:native,expandable_segments:True"
POST_APPROVAL_RESUME_COMMAND = (
    "python scripts/orchestrate.py --resume_post_approval_pipeline"
)
STAGE3_EXTENSION_BASE_STEP = 12_000
STAGE3_EXTENSION_TARGET_STEP = 18_000
STAGE3_EXTENSION_VALIDATION_EVERY_STEPS = 2_000
STAGE3_EXTENSION_VALIDATION_STEPS = (14_000, 16_000, 18_000)
STAGE3_EXTENSION_CYCLES = 3
STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS = 12_000
STAGE3_EXTENSION_MIN_LR = 2.0e-6
STAGE3_EXTENSION_LR_POLICY = "hold_original_cosine_floor_after_schedule_horizon"
STAGE3_EXTENSION_BACKUP_DIRECTORY = (
    "artifacts/migrations/stage3_extension_12000_to_18000_v1"
)
STAGE4_EXTENSION_CONDITIONAL_SCHEMA = (
    "graphrestore-stage4-extension-conditional-approval-v1"
)
STAGE4_EXTENSION_GATE_SCHEMA = "graphrestore-stage4-extension-gate-receipt-v1"
STAGE4_EXTENSION_BASE_STEP = 40_000
STAGE4_EXTENSION_TARGET_STEP = 48_000
STAGE4_EXTENSION_VALIDATION_STEPS = (44_000, 48_000)
STAGE4_EXTENSION_VALIDATION_EVERY_STEPS = 4_000
STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS = 40_000
STAGE4_EXTENSION_MIN_LR = 5.0e-7
STAGE4_EXTENSION_LR_POLICY = "hold_original_cosine_floor_after_schedule_horizon"
STAGE4_EXTENSION_BACKUP_DIRECTORY = (
    "artifacts/migrations/stage4_extension_40000_to_48000_v1"
)
STAGE0_GROUP_A_PSNR_ANCHOR = 24.809721372127534
STAGE0_GROUP_A_SSIM_ANCHOR = 0.785909488574689
D017_ACCEPTANCE = {
    "accepted": True,
    "guard_severity": "severity / 2",
    "severity_domain": [0, 1, 2],
    "target_values": [0.0, 0.5, 1.0],
    "interpretation": (
        "ordinal normalization of the official three-level severity index, "
        "where each level selects a locked joint parameter tuple"
    ),
    "parameter_wise_minmax_claimed": False,
}


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
        if not self.argv or any(
            not isinstance(argument, str) or not argument for argument in self.argv
        ):
            raise ValueError("command argv must contain non-empty strings")


@dataclass(frozen=True)
class Stage3ExtensionAuthorization:
    """Verified, immutable authorization for the exact 12k -> 18k extension."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]

    def provenance_binding(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "cycles": STAGE3_EXTENSION_CYCLES,
            "base_step": STAGE3_EXTENSION_BASE_STEP,
            "target_step": STAGE3_EXTENSION_TARGET_STEP,
            "validation_every_steps": STAGE3_EXTENSION_VALIDATION_EVERY_STEPS,
            "validation_steps": list(STAGE3_EXTENSION_VALIDATION_STEPS),
            "schedule_horizon_steps": STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS,
            "min_lr": STAGE3_EXTENSION_MIN_LR,
            "lr_policy": STAGE3_EXTENSION_LR_POLICY,
        }


@dataclass(frozen=True)
class Stage4ExtensionAuthorization:
    """Verified activated authorization for the exact 40k -> 48k extension."""

    conditional_path: Path
    conditional_sha256: str
    gate_path: Path
    gate_sha256: str
    payload: Mapping[str, Any]

    def provenance_binding(self) -> dict[str, Any]:
        return {
            "conditional_authorization": {
                "path": str(self.conditional_path),
                "sha256": self.conditional_sha256,
            },
            "gate_receipt": {
                "path": str(self.gate_path),
                "sha256": self.gate_sha256,
            },
            "cycles": 2,
            "additional_optimizer_steps": (
                STAGE4_EXTENSION_TARGET_STEP - STAGE4_EXTENSION_BASE_STEP
            ),
            "base_step": STAGE4_EXTENSION_BASE_STEP,
            "target_step": STAGE4_EXTENSION_TARGET_STEP,
            "hard_terminal_step": STAGE4_EXTENSION_TARGET_STEP,
            "validation_every_steps": STAGE4_EXTENSION_VALIDATION_EVERY_STEPS,
            "validation_steps": list(STAGE4_EXTENSION_VALIDATION_STEPS),
            "schedule_horizon_steps": STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS,
            "min_lr": STAGE4_EXTENSION_MIN_LR,
            "lr_policy": STAGE4_EXTENSION_LR_POLICY,
            "exact_resume": True,
            "reset_optimizer": False,
            "reset_ema": False,
            "reset_scheduler": False,
            "reset_rng": False,
            "reset_sampler": False,
            "further_extension_authorized": False,
        }


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
            f"{cwd}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(cwd)
        )
        environment["PYTHONUNBUFFERED"] = "1"
        # Every formal child shares the allocator configuration that passed
        # the Stage0 numerical-equivalence/VRAM audit.  Override a conflicting
        # parent value so resumed stages cannot silently return to the failed
        # high-reserved-memory path.
        environment["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOCATOR_CONF

        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            header = (
                f"[{utc_now_iso()}] START {command.name}: {shlex.join(command.argv)}\n"
            )
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
            raise OrchestrationError(
                f"unknown pipeline status: {state.status!r}"
            ) from exc
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
        return (
            self.project_root
            / "artifacts"
            / "approvals"
            / "STAGE3_APPROVAL_REQUIRED.json"
        )

    @property
    def approval_granted(self) -> Path:
        return self.project_root / "artifacts" / "approvals" / "STAGE3_APPROVED.json"

    @property
    def stage3_extension_approval(self) -> Path:
        return (
            self.project_root
            / "artifacts"
            / "approvals"
            / "STAGE3_EXTENSION_APPROVED.json"
        )

    @property
    def stage3_extension_revocation(self) -> Path:
        return (
            self.project_root
            / "artifacts"
            / "approvals"
            / "STAGE3_EXTENSION_REVOKED.json"
        )

    @property
    def stage4_extension_conditional_approval(self) -> Path:
        return (
            self.project_root
            / "artifacts"
            / "approvals"
            / "STAGE4_EXTENSION_CONDITIONAL_APPROVED.json"
        )

    @property
    def stage4_extension_gate_receipt(self) -> Path:
        return (
            self.project_root
            / "artifacts"
            / "approvals"
            / "STAGE4_EXTENSION_GATE_RECEIPT.json"
        )

    @property
    def stage2_decision(self) -> Path:
        return (
            self.project_root
            / "artifacts"
            / "interaction_labels"
            / "stage2_decision.json"
        )


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
            self._python_command(
                "audit_metric_parity", "scripts/audit_metric_parity.py"
            ),
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
                    "stage0",
                    "scripts/train_stage0.py",
                    "--config",
                    "configs/stage0_mio_stagea.yaml",
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
        checkpoint = (
            state.last_checkpoint if state.last_checkpoint is not None else "none"
        )
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
        authoritative_pause = ""
        if state.status == PipelineStatus.PAUSED_AFTER_STAGE2.value:
            authoritative_pause = (
                "status: PAUSED_AFTER_STAGE2\n"
                "GPU: released\n"
                "Stage3: NOT STARTED\n"
                "waiting_for: user approval\n"
                "resume_command: python scripts/orchestrate.py "
                "--approve_stage3 --resume_from_stage3\n"
            )
        generic_status = "" if authoritative_pause else f"status: {state.status}\n"
        stage4_metrics = self._render_stage4_metric_status()
        return (
            authoritative_pause
            + generic_status
            + stage4_metrics
            + f"current_stage: {state.current_stage}\n"
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

    def _render_stage4_metric_status(self) -> str:
        """Render the approved Stage0-retention comparison when available."""

        stage4_dir = self.paths.project_root / "artifacts/checkpoints/stage4"
        validation_path = stage4_dir / "validation_latest.json"
        if not validation_path.is_file():
            return ""
        validation = self._load_json_mapping(
            validation_path, context="Stage4 latest validation"
        )
        group = validation.get("group_a_equal_combination_mean")
        if not isinstance(group, Mapping):
            raise OrchestrationError(
                "Stage4 latest validation lacks Group-A equal-combination metrics"
            )
        latest_psnr = group.get("psnr")
        latest_ssim = group.get("ssim")
        if not self._is_finite_number(latest_psnr) or not self._is_finite_number(
            latest_ssim
        ):
            raise OrchestrationError("Stage4 latest Group-A PSNR/SSIM must be finite")
        latest_psnr = float(latest_psnr)
        latest_ssim = float(latest_ssim)
        lines = [
            f"latest_group_a_psnr: {latest_psnr!r}",
            "delta_group_a_psnr_vs_stage0: "
            f"{latest_psnr - STAGE0_GROUP_A_PSNR_ANCHOR!r}",
            f"latest_group_a_ssim: {latest_ssim!r}",
            "delta_group_a_ssim_vs_stage0: "
            f"{latest_ssim - STAGE0_GROUP_A_SSIM_ANCHOR!r}",
            f"stage0_group_a_psnr_anchor: {STAGE0_GROUP_A_PSNR_ANCHOR!r}",
            f"stage0_group_a_ssim_anchor: {STAGE0_GROUP_A_SSIM_ANCHOR!r}",
        ]
        complete_path = stage4_dir / "complete.json"
        if complete_path.is_file():
            complete = self._load_json_mapping(
                complete_path, context="Stage4 completion"
            )
            best = complete.get("best_score")
            if not isinstance(best, Mapping):
                raise OrchestrationError("Stage4 completion lacks selected best score")
            selected_psnr = best.get("group_a_psnr")
            selected_ssim = best.get("group_a_ssim")
            if not self._is_finite_number(selected_psnr) or not self._is_finite_number(
                selected_ssim
            ):
                raise OrchestrationError(
                    "Stage4 selected Group-A PSNR/SSIM must be finite"
                )
            selected_psnr = float(selected_psnr)
            selected_ssim = float(selected_ssim)
            lines.extend(
                (
                    f"selected_group_a_psnr: {selected_psnr!r}",
                    "selected_delta_group_a_psnr_vs_stage0: "
                    f"{selected_psnr - STAGE0_GROUP_A_PSNR_ANCHOR!r}",
                    f"selected_group_a_ssim: {selected_ssim!r}",
                    "selected_delta_group_a_ssim_vs_stage0: "
                    f"{selected_ssim - STAGE0_GROUP_A_SSIM_ANCHOR!r}",
                    "SSIM_RETENTION_RISK: "
                    f"{str(selected_ssim < STAGE0_GROUP_A_SSIM_ANCHOR).lower()}",
                )
            )
            if selected_ssim < STAGE0_GROUP_A_SSIM_ANCHOR:
                lines.append(
                    "SSIM_RETENTION_RISK_NOTE: selected Group-A PSNR does not "
                    "offset the SSIM retention deficit"
                )
        return "\n".join(lines) + "\n"

    def _run_child(
        self, state: PipelineState, status: PipelineStatus, command: CommandSpec
    ) -> None:
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
        state.last_error = f"recovered proven-stale durable state {previous}; no exact live child found"
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
                raise OrchestrationError(
                    "READY_FOR_MAIN state lacks the frozen 100-step proof"
                )
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
        state.next_command = recommended_tmux_command(
            self.paths.project_root, self.python
        )
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
            raise OrchestrationError(
                f"100-step integration evidence mismatch: {mismatches}"
            )

    def run_main_pipeline(self) -> PipelineState:
        state = self.load_state()
        if (
            state.status != PipelineStatus.READY_FOR_MAIN.value
            or state.integration_steps != 100
        ):
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
            raise OrchestrationError(
                "failed main pipeline lacks exact 100-step integration"
            )
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
            "stage0": self.paths.project_root / "artifacts/checkpoints/stage0/last.pth",
            "stage1": self.paths.project_root / "artifacts/checkpoints/stage1/last.pth",
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
                    state.last_checkpoint = str(expected_outputs[base_command.name][0])
                continue
            command = (
                self._resumable_main_command(base_command)
                if resume_failed
                else base_command
            )
            self._run_child(state, status, command)
            try:
                self._require_files(
                    expected_outputs[command.name], context=command.name
                )
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

    def _post_approval_outputs(self) -> dict[str, dict[str, Path]]:
        root = self.paths.project_root
        return {
            "stage3": {
                "best": root / "artifacts/checkpoints/stage3/best_ema.pth",
                "last": root / "artifacts/checkpoints/stage3/last.pth",
                "complete": root / "artifacts/checkpoints/stage3/complete.json",
                "thresholds": root / "artifacts/planner_thresholds.json",
                "report": root / "reports/STAGE3_PLANNER_GUARD.md",
            },
            "stage4": {
                "best": root / "artifacts/checkpoints/stage4/best_ema.pth",
                "last": root / "artifacts/checkpoints/stage4/last.pth",
                "complete": root / "artifacts/checkpoints/stage4/complete.json",
                "validation": (
                    root / "artifacts/checkpoints/stage4/validation_latest.json"
                ),
                "report": root / "reports/STAGE4_E2E.md",
                "diagnostics": root / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.md",
                "diagnostics_json": root / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.json",
            },
        }

    def _load_checkpoint_header(
        self,
        path: Path,
        *,
        stage: str,
        model_role: str,
        resumable: bool,
        approval_sha256: str,
        extension_authorization: Stage3ExtensionAuthorization | None = None,
        stage4_extension_authorization: Stage4ExtensionAuthorization | None = None,
    ) -> Mapping[str, Any]:
        """CPU-load and validate the resume/selection role before a child starts."""

        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise OrchestrationError(
                f"could not inspect {stage} checkpoint {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise OrchestrationError(f"{stage} checkpoint is not a mapping: {path}")
        expected = {
            "schema_version": "graphrestore-checkpoint-v1",
            "stage": stage,
            "model_role": model_role,
            "resumable": resumable,
        }
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        step = payload.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            mismatches["step"] = {"expected": "non-negative integer", "actual": step}
        for key in ("model", "ema", "provenance"):
            if not isinstance(payload.get(key), Mapping):
                mismatches[key] = {
                    "expected": "mapping",
                    "actual": type(payload.get(key)).__name__,
                }
        if resumable:
            for key in ("optimizer", "scheduler", "rng_states", "sampler_state"):
                if not isinstance(payload.get(key), Mapping):
                    mismatches[key] = {
                        "expected": "mapping in raw resumable last.pth",
                        "actual": type(payload.get(key)).__name__,
                    }
        provenance = payload.get("provenance")
        recorded_approval: object = None
        if isinstance(provenance, Mapping):
            if stage == "stage3":
                approval = provenance.get("stage3_approval")
                if isinstance(approval, Mapping):
                    recorded_approval = approval.get("sha256")
            else:
                parents = provenance.get("parents")
                if isinstance(parents, Mapping):
                    approval = parents.get("stage3_approval")
                    if isinstance(approval, Mapping):
                        recorded_approval = approval.get("sha256")
        if recorded_approval != approval_sha256:
            mismatches["provenance.stage3_approval.sha256"] = {
                "expected": approval_sha256,
                "actual": recorded_approval,
            }
        if extension_authorization is not None:
            expected_extension = extension_authorization.provenance_binding()
            recorded_extension: object = None
            if isinstance(provenance, Mapping):
                if stage == "stage3":
                    recorded_extension = provenance.get("stage3_extension")
                    runtime = provenance.get("runtime")
                    if (
                        not isinstance(runtime, Mapping)
                        or runtime.get("max_steps")
                        != STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS
                        or runtime.get("training_target_step")
                        != STAGE3_EXTENSION_TARGET_STEP
                    ):
                        mismatches["provenance.runtime.stage3_extension"] = {
                            "expected": {
                                "max_steps": (STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS),
                                "training_target_step": (STAGE3_EXTENSION_TARGET_STEP),
                            },
                            "actual": runtime,
                        }
                else:
                    recorded_extension = provenance.get("stage3_extension")
                    expected_extension = {
                        "path": str(extension_authorization.path),
                        "sha256": extension_authorization.sha256,
                    }
            if recorded_extension != expected_extension:
                mismatches["provenance.stage3_extension"] = {
                    "expected": expected_extension,
                    "actual": recorded_extension,
                }
        if stage4_extension_authorization is not None:
            expected_stage4_extension = (
                stage4_extension_authorization.provenance_binding()
            )
            recorded_stage4_extension: object = None
            runtime: object = None
            if isinstance(provenance, Mapping):
                recorded_stage4_extension = provenance.get("stage4_extension")
                runtime = provenance.get("runtime")
            if stage != "stage4":
                mismatches["stage4_extension.stage"] = {
                    "expected": "stage4",
                    "actual": stage,
                }
            if recorded_stage4_extension != expected_stage4_extension:
                mismatches["provenance.stage4_extension"] = {
                    "expected": expected_stage4_extension,
                    "actual": recorded_stage4_extension,
                }
            if (
                not isinstance(runtime, Mapping)
                or runtime.get("max_steps") != STAGE4_EXTENSION_TARGET_STEP
                or runtime.get("schedule_max_steps")
                != STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS
            ):
                mismatches["provenance.runtime.stage4_extension"] = {
                    "expected": {
                        "max_steps": STAGE4_EXTENSION_TARGET_STEP,
                        "schedule_max_steps": (STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS),
                    },
                    "actual": runtime,
                }
        if mismatches:
            raise OrchestrationError(
                f"{stage} checkpoint role/provenance mismatch: {mismatches}"
            )
        return payload

    def _load_json_mapping(self, path: Path, *, context: str) -> Mapping[str, Any]:
        try:
            value = load_json(path)
        except Exception as exc:
            raise OrchestrationError(f"could not load {context} {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise OrchestrationError(f"{context} must be a JSON mapping: {path}")
        return value

    def _load_text(self, path: Path, *, context: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            raise OrchestrationError(f"could not load {context} {path}: {exc}") from exc

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False

    def _verified_binding(
        self,
        value: object,
        *,
        label: str,
        expected_path: Path | None = None,
    ) -> tuple[Path, str]:
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise OrchestrationError(f"{label} must contain only path/sha256")
        raw_path = value.get("path")
        digest = value.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or not isinstance(digest, str)
            or not is_sha256(digest)
        ):
            raise OrchestrationError(f"{label} has an invalid path/hash binding")
        path = Path(raw_path)
        if path.resolve(strict=False) != path or (
            expected_path is not None and path != expected_path.resolve(strict=False)
        ):
            raise OrchestrationError(f"{label} path drifted")
        if not path.is_file() or sha256_file(path) != digest:
            raise OrchestrationError(f"{label} physical file/hash drifted")
        return path, digest

    def _verify_stage3_finalization_completion(
        self,
        *,
        approval_sha256: str,
        authorization: Stage3RevocationAuthorization,
    ) -> Path:
        """Gate Stage4 on the one authorized zero-training Stage3 finalizer."""

        # Revalidate the tombstone at the transition boundary; the child must
        # not be able to replace it after the pre-run authorization check.
        current = validate_stage3_extension_revocation(
            authorization.path,
            project_root=self.paths.project_root,
            require_present=True,
        )
        if current.sha256 != authorization.sha256:
            raise OrchestrationError(
                "Stage3 finalization authorization changed during execution"
            )
        root = self.paths.project_root
        stage3_dir = root / "artifacts/checkpoints/stage3"
        outputs = {
            "best": stage3_dir / "best_ema.pth",
            "last": stage3_dir / "last.pth",
            "selected_validation": stage3_dir / "selected_validation.json",
            "calibration_history": root / "artifacts/metrics/calibration_history.csv",
            "thresholds": root / "artifacts/planner_thresholds.json",
            "calibrated_diagnostic": (
                stage3_dir / "selected_validation_calibrated.json"
            ),
            "complete": stage3_dir / "complete.json",
            "report": root / "reports/STAGE3_PLANNER_GUARD.md",
        }
        self._require_files(outputs.values(), context="finalized Stage3")
        bindings = authorization.bindings
        anchor_names = {
            "selected_checkpoint": "best",
            "abandoned_last_checkpoint": "last",
            "selected_validation": "selected_validation",
            "calibration_history": "calibration_history",
        }
        anchor_hashes: dict[str, str] = {}
        for binding_name, output_name in anchor_names.items():
            expected_path = (
                None
                if binding_name == "abandoned_last_checkpoint"
                else outputs[output_name]
            )
            path, digest = self._verified_binding(
                bindings.get(binding_name),
                label=f"Stage3 revocation {binding_name}",
                expected_path=expected_path,
            )
            if (
                binding_name != "abandoned_last_checkpoint"
                and path != outputs[output_name].resolve()
            ):
                raise OrchestrationError(
                    f"Stage3 revocation {binding_name} path drifted"
                )
            if sha256_file(outputs[output_name]) != digest:
                raise OrchestrationError(
                    f"Stage3 live {binding_name} differs from its frozen anchor"
                )
            anchor_hashes[binding_name] = digest

        historical_extension_path, historical_extension_sha = self._verified_binding(
            bindings.get("historical_extension_authorization"),
            label="Stage3 revocation historical extension",
            expected_path=self.paths.stage3_extension_approval,
        )
        historical_extension_payload = self._load_json_mapping(
            historical_extension_path,
            context="historical Stage3 extension authorization",
        )
        extension = Stage3ExtensionAuthorization(
            path=historical_extension_path,
            sha256=historical_extension_sha,
            payload=historical_extension_payload,
        )
        best = self._load_checkpoint_header(
            outputs["best"],
            stage="stage3",
            model_role="ema_selection",
            resumable=False,
            approval_sha256=approval_sha256,
            extension_authorization=extension,
        )
        last = self._load_checkpoint_header(
            outputs["last"],
            stage="stage3",
            model_role="raw_training_state",
            resumable=True,
            approval_sha256=approval_sha256,
            extension_authorization=extension,
        )
        checkpoint_mismatches: dict[str, object] = {}
        expected_headers = {
            "best.step": (best.get("step"), 12_000),
            "best.pending_validation_step": (
                best.get("pending_validation_step"),
                None,
            ),
            "best.optimizer_transaction_active": (
                best.get("optimizer_transaction_active"),
                False,
            ),
            "last.step": (last.get("step"), 14_000),
            "last.pending_validation_step": (
                last.get("pending_validation_step"),
                14_000,
            ),
            "last.optimizer_transaction_active": (
                last.get("optimizer_transaction_active"),
                False,
            ),
        }
        for key, (actual, expected) in expected_headers.items():
            if actual != expected:
                checkpoint_mismatches[key] = {
                    "expected": expected,
                    "actual": actual,
                }
        if checkpoint_mismatches:
            raise OrchestrationError(
                "Stage3 finalize-only checkpoint anchors drifted: "
                f"{checkpoint_mismatches}"
            )

        thresholds = self._load_json_mapping(
            outputs["thresholds"], context="finalized Stage3 thresholds"
        )
        threshold_expected = {
            "schema_version": "graphrestore-presence-thresholds-v1",
            "protocol_id": PROTOCOL_ID,
            "frozen": True,
            "source": "primary_val_presence_f1_only",
            "calibration_runs": 1,
            "mio100_rows_read": 0,
            "checkpoint_sha256": anchor_hashes["selected_checkpoint"],
            "stage3_finalization_authorization_sha256": authorization.sha256,
            "stage3_extension_authorization_sha256": historical_extension_sha,
            "tie_break": "nearest_0.50_then_higher_threshold",
        }
        threshold_mismatches = {
            key: {"expected": expected, "actual": thresholds.get(key)}
            for key, expected in threshold_expected.items()
            if thresholds.get(key) != expected
        }
        selected_checkpoint = thresholds.get("selected_stage3_checkpoint")
        expected_selected = {
            "path": str(outputs["best"].resolve()),
            "sha256": anchor_hashes["selected_checkpoint"],
        }
        if selected_checkpoint != expected_selected:
            threshold_mismatches["selected_stage3_checkpoint"] = {
                "expected": expected_selected,
                "actual": selected_checkpoint,
            }
        raw_thresholds = thresholds.get("thresholds")
        per_skill = thresholds.get("per_skill_metrics")
        skills = thresholds.get("skills")
        if (
            not isinstance(skills, list)
            or len(skills) != 8
            or len(set(skills)) != 8
            or not isinstance(raw_thresholds, Mapping)
            or set(raw_thresholds) != set(skills)
            or not isinstance(per_skill, Mapping)
            or set(per_skill) != set(skills)
        ):
            threshold_mismatches["skills"] = {
                "expected": "eight matching ordered skill threshold/metric entries",
                "actual": skills,
            }
        else:
            tolerance = thresholds.get("numerical_tolerance", 1.0e-12)
            if not self._is_finite_number(tolerance) or float(tolerance) < 0.0:
                threshold_mismatches["numerical_tolerance"] = {
                    "expected": "finite non-negative number",
                    "actual": tolerance,
                }
                tolerance = 0.0
            for skill in skills:
                threshold = raw_thresholds.get(skill)
                metrics = per_skill.get(skill)
                if (
                    not self._is_finite_number(threshold)
                    or not (0.20 <= float(threshold) <= 0.80)
                    or not math.isclose(
                        (float(threshold) - 0.20) / 0.02,
                        round((float(threshold) - 0.20) / 0.02),
                        rel_tol=0.0,
                        abs_tol=2.0e-5,
                    )
                    or not isinstance(metrics, Mapping)
                    or set(metrics) != {"baseline", "calibrated"}
                ):
                    threshold_mismatches[f"per_skill_metrics.{skill}"] = {
                        "expected": "finite grid threshold and baseline/calibrated metrics",
                        "actual": metrics,
                    }
                    continue
                baseline = metrics.get("baseline")
                calibrated = metrics.get("calibrated")
                for label, metric, expected_threshold in (
                    ("baseline", baseline, 0.50),
                    ("calibrated", calibrated, float(threshold)),
                ):
                    if (
                        not isinstance(metric, Mapping)
                        or set(metric) != {"threshold", "precision", "recall", "f1"}
                        or any(
                            not self._is_finite_number(metric.get(name))
                            for name in ("threshold", "precision", "recall", "f1")
                        )
                        or not math.isclose(
                            float(metric.get("threshold", -1.0)),
                            expected_threshold,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    ):
                        threshold_mismatches[f"per_skill_metrics.{skill}.{label}"] = {
                            "expected": "finite threshold/precision/recall/f1",
                            "actual": metric,
                        }
                if (
                    isinstance(baseline, Mapping)
                    and isinstance(calibrated, Mapping)
                    and self._is_finite_number(baseline.get("f1"))
                    and self._is_finite_number(calibrated.get("f1"))
                    and float(calibrated["f1"])
                    < float(baseline["f1"]) - float(tolerance)
                ):
                    threshold_mismatches[f"per_skill_metrics.{skill}.f1_gate"] = {
                        "expected": f">= baseline - {tolerance}",
                        "actual": calibrated.get("f1"),
                    }
        for key in ("macro_f1_before", "macro_f1_after"):
            if not self._is_finite_number(thresholds.get(key)):
                threshold_mismatches[key] = {
                    "expected": "finite number",
                    "actual": thresholds.get(key),
                }
        if threshold_mismatches:
            raise OrchestrationError(
                f"Stage3 finalize-only threshold contract drifted: {threshold_mismatches}"
            )

        diagnostic = self._load_json_mapping(
            outputs["calibrated_diagnostic"],
            context="Stage3 calibrated selected diagnostic",
        )
        complete = self._load_json_mapping(
            outputs["complete"], context="Stage3 finalize-only completion evidence"
        )
        try:
            from src.training.stage3_engine import (
                validate_stage3_finalization_outputs,
            )

            shared_evidence = validate_stage3_finalization_outputs(
                root,
                finalization_authorization_sha256=authorization.sha256,
                historical_extension_authorization_sha256=(historical_extension_sha),
            )
        except Exception as exc:
            raise OrchestrationError(
                f"shared Stage3 finalization output validation failed: {exc}"
            ) from exc
        for logical, output_name in (
            ("best_checkpoint", "best"),
            ("thresholds", "thresholds"),
            ("selected_validation_calibrated", "calibrated_diagnostic"),
            ("report", "report"),
        ):
            shared = shared_evidence.get(logical)
            if (
                not isinstance(shared, Mapping)
                or shared.get("path") != str(outputs[output_name].resolve())
                or shared.get("sha256") != sha256_file(outputs[output_name])
            ):
                raise OrchestrationError(
                    f"shared Stage3 finalization evidence drifted for {logical}"
                )
        report_text = self._load_text(
            outputs["report"], context="Stage3 finalize-only report"
        )
        required_report_fragments = (
            PROTOCOL_ID,
            authorization.sha256,
            anchor_hashes["selected_checkpoint"],
            "step12000_finalize_only_no_training",
            "optimizer / scheduler / train loader created: false / false / false",
            "checkpoint written: false",
            "MiO100 / Group B / Group C rows read: 0 / 0 / 0",
            "learned raw relation accuracy",
            "always-parallel baseline accuracy",
            "per-pair majority-prior baseline accuracy",
            "STOP-rate definition",
        )
        missing_report_fragments = [
            fragment
            for fragment in required_report_fragments
            if fragment not in report_text
        ]
        if missing_report_fragments:
            raise OrchestrationError(
                "Stage3 finalize-only report lacks required disclosures: "
                f"{missing_report_fragments}"
            )
        # Finalizer-generated evidence binds every mutable output while the
        # revocation itself permanently binds every protected training anchor.
        expected_complete_scalars = {
            "schema_version": "graphrestore-stage3-runtime-v1",
            "kind": "stage3_finalize_only",
            "protocol_id": PROTOCOL_ID,
            "step": 12_000,
            "optimizer_created": False,
            "scheduler_created": False,
            "train_loader_created": False,
            "checkpoint_written": False,
            "optimizer_steps_executed": 0,
            "checkpoint_writes": 0,
            "sampler_steps_advanced": 0,
            "threshold_calibration_runs": 1,
            "post_calibration_diagnostic_runs": 1,
            "mio100_rows_read": 0,
            "group_b_rows_read": 0,
            "group_c_rows_read": 0,
        }
        completion_mismatches = {
            key: {"expected": expected, "actual": complete.get(key)}
            for key, expected in expected_complete_scalars.items()
            if complete.get(key) != expected
        }
        complete_bindings = complete.get("bindings")
        required_complete_bindings = {
            "best_checkpoint": outputs["best"],
            "abandoned_last_checkpoint": Path(
                str(bindings["abandoned_last_checkpoint"]["path"])
            ),
            "selected_validation": outputs["selected_validation"],
            "calibration_history": outputs["calibration_history"],
            "thresholds": outputs["thresholds"],
            "selected_validation_calibrated": outputs["calibrated_diagnostic"],
            "report": outputs["report"],
            "finalization_authorization": authorization.path,
            "historical_extension_authorization": historical_extension_path,
        }
        if not isinstance(complete_bindings, Mapping):
            completion_mismatches["bindings"] = {
                "expected": "mapping",
                "actual": type(complete_bindings).__name__,
            }
        else:
            for name, expected_path in required_complete_bindings.items():
                try:
                    self._verified_binding(
                        complete_bindings.get(name),
                        label=f"Stage3 completion {name}",
                        expected_path=expected_path,
                    )
                except OrchestrationError as exc:
                    completion_mismatches[f"bindings.{name}"] = {
                        "expected": str(expected_path.resolve()),
                        "actual": str(exc),
                    }
        if diagnostic.get("protocol_id") != PROTOCOL_ID:
            completion_mismatches["diagnostic.protocol_id"] = {
                "expected": PROTOCOL_ID,
                "actual": diagnostic.get("protocol_id"),
            }
        diagnostic_planner = diagnostic.get("planner")
        diagnostic_graph = diagnostic.get("graph")
        if (
            not isinstance(diagnostic_planner, Mapping)
            or diagnostic_planner.get("sample_count") != 1_600
            or not isinstance(diagnostic_graph, Mapping)
            or diagnostic_graph.get("sample_count") != 1_600
        ):
            completion_mismatches["diagnostic.sample_count"] = {
                "expected": "planner and graph sample_count=1600",
                "actual": {
                    "planner": (
                        diagnostic_planner.get("sample_count")
                        if isinstance(diagnostic_planner, Mapping)
                        else None
                    ),
                    "graph": (
                        diagnostic_graph.get("sample_count")
                        if isinstance(diagnostic_graph, Mapping)
                        else None
                    ),
                },
            }
        diagnostic_sources = diagnostic.get("sources")
        if (
            not isinstance(diagnostic_sources, Mapping)
            or diagnostic_sources.get("mio100_rows_read") != 0
            or diagnostic.get("group_b_rows_read") != 0
            or diagnostic.get("group_c_rows_read") != 0
        ):
            completion_mismatches["diagnostic.sources"] = {
                "expected": "MiO100/Group-B/Group-C counters all zero",
                "actual": diagnostic_sources,
            }
        if completion_mismatches:
            raise OrchestrationError(
                "completed Stage3 finalize-only evidence/hash mismatch: "
                f"{completion_mismatches}"
            )
        # Detect any protected-file mutation between validation and transition.
        for binding_name, output_name in anchor_names.items():
            if sha256_file(outputs[output_name]) != anchor_hashes[binding_name]:
                raise OrchestrationError(
                    f"Stage3 protected {binding_name} changed during finalization gate"
                )
        return outputs["best"]

    def _verify_post_approval_completion(
        self,
        stage: str,
        *,
        approval_sha256: str,
        extension_authorization: Stage3ExtensionAuthorization | None = None,
        finalization_authorization: Stage3RevocationAuthorization | None = None,
        stage4_extension_authorization: Stage4ExtensionAuthorization | None = None,
    ) -> Path:
        if stage == "stage3" and finalization_authorization is not None:
            return self._verify_stage3_finalization_completion(
                approval_sha256=approval_sha256,
                authorization=finalization_authorization,
            )
        outputs = dict(self._post_approval_outputs()[stage])
        if stage == "stage3" and extension_authorization is not None:
            stage3_dir = self.paths.project_root / "artifacts/checkpoints/stage3"
            outputs.update(
                {
                    "run_contract": stage3_dir / "run_contract.json",
                    "selected_validation": stage3_dir / "selected_validation.json",
                    "validation": stage3_dir / "validation_latest.json",
                    "train_log": stage3_dir / "train.jsonl",
                }
            )
        if stage == "stage4" and stage4_extension_authorization is not None:
            stage4_dir = self.paths.project_root / "artifacts/checkpoints/stage4"
            outputs["run_contract"] = stage4_dir / "run_contract.json"
        self._require_files(outputs.values(), context=f"completed {stage}")
        empty = [str(path) for path in outputs.values() if path.stat().st_size <= 0]
        if empty:
            raise OrchestrationError(
                f"completed {stage} has empty required outputs: {empty}"
            )
        best = self._load_checkpoint_header(
            outputs["best"],
            stage=stage,
            model_role="ema_selection",
            resumable=False,
            approval_sha256=approval_sha256,
            extension_authorization=extension_authorization,
            stage4_extension_authorization=(
                stage4_extension_authorization if stage == "stage4" else None
            ),
        )
        last = self._load_checkpoint_header(
            outputs["last"],
            stage=stage,
            model_role="raw_training_state",
            resumable=True,
            approval_sha256=approval_sha256,
            extension_authorization=extension_authorization,
            stage4_extension_authorization=(
                stage4_extension_authorization if stage == "stage4" else None
            ),
        )
        if stage == "stage3" and extension_authorization is not None:
            run_contract = self._load_json_mapping(
                outputs["run_contract"], context="extended Stage3 run contract"
            )
            contract_provenance = run_contract.get("provenance")
            if (
                run_contract.get("schema_version") != "graphrestore-stage3-runtime-v1"
                or not isinstance(contract_provenance, Mapping)
                or contract_provenance != last.get("provenance")
                or contract_provenance != best.get("provenance")
            ):
                raise OrchestrationError(
                    "extended Stage3 run contract/last/best provenance drifted"
                )
        if stage == "stage4" and stage4_extension_authorization is not None:
            run_contract = self._load_json_mapping(
                outputs["run_contract"], context="extended Stage4 run contract"
            )
            contract_provenance = run_contract.get("provenance")
            if (
                run_contract.get("schema_version") != "graphrestore-stage4-runtime-v1"
                or not isinstance(contract_provenance, Mapping)
                or contract_provenance != last.get("provenance")
                or contract_provenance != best.get("provenance")
            ):
                raise OrchestrationError(
                    "extended Stage4 run contract/last/best provenance drifted"
                )
        complete = self._load_json_mapping(
            outputs["complete"], context=f"{stage} completion evidence"
        )
        mismatches: dict[str, dict[str, Any]] = {}
        expected_schema = (
            "graphrestore-stage3-runtime-v1"
            if stage == "stage3"
            else "graphrestore-stage4-runtime-v1"
        )
        expected_step = (
            STAGE3_EXTENSION_TARGET_STEP
            if stage == "stage3" and extension_authorization is not None
            else 12_000
            if stage == "stage3"
            else STAGE4_EXTENSION_TARGET_STEP
            if stage4_extension_authorization is not None
            else 40_000
        )
        expected_values: dict[str, Any] = {
            "schema_version": expected_schema,
            "step": expected_step,
        }
        if stage == "stage4":
            expected_values.update(
                {
                    "protocol_id": PROTOCOL_ID,
                    "formal_mio100_started": False,
                    "waiting_for": "new_user_authorization_for_formal_mio100",
                }
            )
        for key, expected in expected_values.items():
            if complete.get(key) != expected:
                mismatches[key] = {"expected": expected, "actual": complete.get(key)}
        if last["step"] != expected_step:
            mismatches["last.step"] = {
                "expected": expected_step,
                "actual": last["step"],
            }
        best_sha = sha256_file(outputs["best"])
        best_hash_key = (
            "best_checkpoint_sha256" if stage == "stage3" else "best_ema_sha256"
        )
        if complete.get(best_hash_key) != best_sha:
            mismatches[best_hash_key] = {
                "expected": best_sha,
                "actual": complete.get(best_hash_key),
            }
        best_path_key = "best_checkpoint" if stage == "stage3" else "best_ema_path"
        recorded_best_path = complete.get(best_path_key)
        if (
            not isinstance(recorded_best_path, str)
            or Path(recorded_best_path).resolve() != outputs["best"].resolve()
        ):
            mismatches[best_path_key] = {
                "expected": str(outputs["best"].resolve()),
                "actual": recorded_best_path,
            }
        if int(best["step"]) > int(last["step"]):
            mismatches["best.step"] = {
                "expected": f"<= {last['step']}",
                "actual": best["step"],
            }
        report_sha = sha256_file(outputs["report"])
        completion_report = {
            "report": str(outputs["report"].resolve()),
            "report_sha256": report_sha,
        }
        for key, expected in completion_report.items():
            if complete.get(key) != expected:
                mismatches[key] = {
                    "expected": expected,
                    "actual": complete.get(key),
                }
        report = self._load_text(outputs["report"], context=f"{stage} main report")
        if PROTOCOL_ID not in report:
            mismatches["report.protocol_id"] = {
                "expected": PROTOCOL_ID,
                "actual": "missing",
            }
        if best_sha not in report:
            mismatches["report.selected_checkpoint_sha256"] = {
                "expected": best_sha,
                "actual": "missing",
            }
        best_score = complete.get("best_score")
        score_values: dict[str, float] = {}
        score_step: int | None = None
        if not isinstance(best_score, Mapping):
            mismatches["best_score"] = {
                "expected": "mapping with selected checkpoint metrics and step",
                "actual": type(best_score).__name__,
            }
        else:
            for score_name in (
                "group_a_psnr",
                "group_a_ssim",
                "single_psnr",
                "single_ssim",
            ):
                score_value = best_score.get(score_name)
                if not self._is_finite_number(score_value):
                    mismatches[f"best_score.{score_name}"] = {
                        "expected": "finite number",
                        "actual": score_value,
                    }
                else:
                    score_values[score_name] = float(score_value)
            raw_score_step = best_score.get("step")
            if (
                isinstance(raw_score_step, bool)
                or not isinstance(raw_score_step, int)
                or raw_score_step < 0
            ):
                mismatches["best_score.step"] = {
                    "expected": "non-negative integer",
                    "actual": raw_score_step,
                }
            else:
                score_step = raw_score_step
                if score_step != best["step"]:
                    mismatches["best_score.step"] = {
                        "expected": best["step"],
                        "actual": score_step,
                    }
        if len(score_values) == 4 and score_step is not None:
            precision = (10, 10) if stage == "stage3" else (6, 8)
            expected_metric_lines = {
                "single_psnr_ssim": (
                    "- Selected Single PSNR/SSIM: "
                    f"{score_values['single_psnr']:.{precision[0]}f} / "
                    f"{score_values['single_ssim']:.{precision[1]}f}"
                ),
                "group_a_psnr_ssim": (
                    "- Selected Group-A PSNR/SSIM: "
                    f"{score_values['group_a_psnr']:.{precision[0]}f} / "
                    f"{score_values['group_a_ssim']:.{precision[1]}f}"
                ),
            }
            report_lines = set(report.splitlines())
            for metric_name, expected_line in expected_metric_lines.items():
                if expected_line not in report_lines:
                    mismatches[f"report.selected_{metric_name}"] = {
                        "expected": expected_line,
                        "actual": "missing or different",
                    }
        if stage == "stage3":
            if extension_authorization is not None:
                expected_extension_paths = {
                    "extension_authorization": str(extension_authorization.path),
                    "selected_validation": str(
                        outputs["selected_validation"].resolve()
                    ),
                    "validation": str(outputs["validation"].resolve()),
                }
                expected_extension_hashes = {
                    "extension_authorization_sha256": (extension_authorization.sha256),
                    "selected_validation_sha256": sha256_file(
                        outputs["selected_validation"]
                    ),
                    "validation_sha256": sha256_file(outputs["validation"]),
                }
                for key, expected in {
                    **expected_extension_paths,
                    **expected_extension_hashes,
                    "extension_validation_steps": list(
                        STAGE3_EXTENSION_VALIDATION_STEPS
                    ),
                    "schedule_horizon_steps": (STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS),
                    "training_target_step": STAGE3_EXTENSION_TARGET_STEP,
                }.items():
                    actual = complete.get(key)
                    if key in expected_extension_paths and isinstance(actual, str):
                        matches = Path(actual).resolve() == Path(expected).resolve()
                    else:
                        matches = actual == expected
                    if not matches:
                        mismatches[key] = {"expected": expected, "actual": actual}

                extension_report_lines = {
                    f"- completed training target step: {STAGE3_EXTENSION_TARGET_STEP}",
                    "- cosine schedule horizon step: "
                    f"{STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS}",
                    "- Stage3 extension authorization SHA256: "
                    f"`{extension_authorization.sha256}`",
                }
                missing_extension_report_lines = sorted(
                    extension_report_lines - set(report.splitlines())
                )
                if missing_extension_report_lines:
                    mismatches["report.stage3_extension"] = {
                        "expected": sorted(extension_report_lines),
                        "actual": missing_extension_report_lines,
                    }

                allowed_best_steps = {
                    STAGE3_EXTENSION_BASE_STEP,
                    *STAGE3_EXTENSION_VALIDATION_STEPS,
                }
                if best.get("step") not in allowed_best_steps:
                    mismatches["best.step.extension_boundary"] = {
                        "expected": sorted(allowed_best_steps),
                        "actual": best.get("step"),
                    }
                for label, checkpoint in (("last", last), ("best", best)):
                    if checkpoint.get("pending_validation_step") is not None:
                        mismatches[f"{label}.pending_validation_step"] = {
                            "expected": None,
                            "actual": checkpoint.get("pending_validation_step"),
                        }
                    if checkpoint.get("optimizer_transaction_active") is not False:
                        mismatches[f"{label}.optimizer_transaction_active"] = {
                            "expected": False,
                            "actual": checkpoint.get("optimizer_transaction_active"),
                        }
                last_metrics = last.get("metrics")
                best_metrics = best.get("metrics")
                if not isinstance(last_metrics, Mapping):
                    mismatches["last.metrics"] = {
                        "expected": "mapping bound to final validation",
                        "actual": type(last_metrics).__name__,
                    }
                else:
                    for key, expected in (
                        ("validation_step", STAGE3_EXTENSION_TARGET_STEP),
                        ("best_step", best.get("step")),
                    ):
                        if last_metrics.get(key) != expected:
                            mismatches[f"last.metrics.{key}"] = {
                                "expected": expected,
                                "actual": last_metrics.get(key),
                            }
                if not isinstance(best_metrics, Mapping):
                    mismatches["best.metrics"] = {
                        "expected": "mapping bound to selected checkpoint",
                        "actual": type(best_metrics).__name__,
                    }
                else:
                    for key in ("validation_step", "best_step"):
                        if best_metrics.get(key) != best.get("step"):
                            mismatches[f"best.metrics.{key}"] = {
                                "expected": best.get("step"),
                                "actual": best_metrics.get(key),
                            }

                try:
                    extension_validation_rows = [
                        row
                        for _, row in iter_jsonl(outputs["train_log"])
                        if row.get("event") == "validation"
                        and isinstance(row.get("step"), int)
                        and not isinstance(row.get("step"), bool)
                        and int(row["step"]) > STAGE3_EXTENSION_BASE_STEP
                    ]
                except (OSError, ValueError) as exc:
                    raise OrchestrationError(
                        f"could not verify Stage3 extension validation log: {exc}"
                    ) from exc
                recorded_validation_steps = sorted(
                    {int(row["step"]) for row in extension_validation_rows}
                )
                if recorded_validation_steps != list(STAGE3_EXTENSION_VALIDATION_STEPS):
                    mismatches["train_log.extension_validation_steps"] = {
                        "expected": list(STAGE3_EXTENSION_VALIDATION_STEPS),
                        "actual": recorded_validation_steps,
                    }

            thresholds_sha = sha256_file(outputs["thresholds"])
            if complete.get("thresholds_sha256") != thresholds_sha:
                mismatches["thresholds_sha256"] = {
                    "expected": thresholds_sha,
                    "actual": complete.get("thresholds_sha256"),
                }
            recorded_thresholds = complete.get("thresholds")
            if (
                not isinstance(recorded_thresholds, str)
                or Path(recorded_thresholds).resolve()
                != outputs["thresholds"].resolve()
            ):
                mismatches["thresholds"] = {
                    "expected": str(outputs["thresholds"].resolve()),
                    "actual": recorded_thresholds,
                }
            for key, expected in (
                ("threshold_calibration_runs", 1),
                ("mio100_rows_read", 0),
            ):
                if complete.get(key) != expected:
                    mismatches[key] = {
                        "expected": expected,
                        "actual": complete.get(key),
                    }
            thresholds = self._load_json_mapping(
                outputs["thresholds"], context="frozen Stage3 thresholds"
            )
            threshold_expected = {
                "schema_version": "graphrestore-presence-thresholds-v1",
                "protocol_id": PROTOCOL_ID,
                "frozen": True,
                "checkpoint_sha256": best_sha,
                "stage3_approval_sha256": approval_sha256,
                "calibration_runs": 1,
                "mio100_rows_read": 0,
            }
            if extension_authorization is not None:
                threshold_expected["stage3_extension_authorization_sha256"] = (
                    extension_authorization.sha256
                )
                selected_checkpoint = thresholds.get("selected_stage3_checkpoint")
                expected_selected_checkpoint = {
                    "path": str(outputs["best"].resolve()),
                    "sha256": best_sha,
                }
                if selected_checkpoint != expected_selected_checkpoint:
                    mismatches["thresholds.selected_stage3_checkpoint"] = {
                        "expected": expected_selected_checkpoint,
                        "actual": selected_checkpoint,
                    }
            for key, expected in threshold_expected.items():
                if thresholds.get(key) != expected:
                    mismatches[f"thresholds.{key}"] = {
                        "expected": expected,
                        "actual": thresholds.get(key),
                    }
        else:
            if stage4_extension_authorization is not None:
                extension_paths = {
                    "stage4_extension_conditional_authorization": str(
                        stage4_extension_authorization.conditional_path
                    ),
                    "stage4_extension_gate_receipt": str(
                        stage4_extension_authorization.gate_path
                    ),
                }
                extension_values: dict[str, Any] = {
                    "stage4_extension_conditional_authorization_sha256": (
                        stage4_extension_authorization.conditional_sha256
                    ),
                    "stage4_extension_gate_receipt_sha256": (
                        stage4_extension_authorization.gate_sha256
                    ),
                    "stage4_extension_validation_steps": list(
                        STAGE4_EXTENSION_VALIDATION_STEPS
                    ),
                    "schedule_horizon_steps": (STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS),
                    "training_target_step": STAGE4_EXTENSION_TARGET_STEP,
                    "additional_optimizer_steps": (
                        STAGE4_EXTENSION_TARGET_STEP - STAGE4_EXTENSION_BASE_STEP
                    ),
                    "further_extension_authorized": False,
                    "calibration_history_steps": list(
                        range(
                            STAGE4_EXTENSION_VALIDATION_EVERY_STEPS,
                            STAGE4_EXTENSION_TARGET_STEP + 1,
                            STAGE4_EXTENSION_VALIDATION_EVERY_STEPS,
                        )
                    ),
                }
                for key, expected in {**extension_paths, **extension_values}.items():
                    actual = complete.get(key)
                    if key in extension_paths and isinstance(actual, str):
                        matches = Path(actual).resolve() == Path(expected).resolve()
                    else:
                        matches = actual == expected
                    if not matches:
                        mismatches[key] = {"expected": expected, "actual": actual}

                expected_extension_report_lines = {
                    "- Conditional Stage4 extension: activated",
                    "- Conditional authorization SHA256: "
                    f"`{stage4_extension_authorization.conditional_sha256}`",
                    "- Extension gate receipt SHA256: "
                    f"`{stage4_extension_authorization.gate_sha256}`",
                    f"- Completed training target step: {STAGE4_EXTENSION_TARGET_STEP}",
                    "- Original cosine schedule horizon step: "
                    f"{STAGE4_EXTENSION_SCHEDULE_HORIZON_STEPS}",
                }
                missing_extension_report_lines = sorted(
                    expected_extension_report_lines - set(report.splitlines())
                )
                if missing_extension_report_lines:
                    mismatches["report.stage4_extension"] = {
                        "expected": sorted(expected_extension_report_lines),
                        "actual": missing_extension_report_lines,
                    }

                for label, checkpoint in (("last", last), ("best", best)):
                    if checkpoint.get("pending_validation_step") is not None:
                        mismatches[f"{label}.pending_validation_step"] = {
                            "expected": None,
                            "actual": checkpoint.get("pending_validation_step"),
                        }
                if last.get("optimizer_transaction_active") is not False:
                    mismatches["last.optimizer_transaction_active"] = {
                        "expected": False,
                        "actual": last.get("optimizer_transaction_active"),
                    }
            stage0_psnr = STAGE0_GROUP_A_PSNR_ANCHOR
            stage0_ssim = STAGE0_GROUP_A_SSIM_ANCHOR
            if len(score_values) == 4:
                selected_psnr = score_values["group_a_psnr"]
                selected_ssim = score_values["group_a_ssim"]
                retention_expected = {
                    "stage0_group_a_psnr_anchor": stage0_psnr,
                    "stage0_group_a_ssim_anchor": stage0_ssim,
                    "selected_group_a_psnr": selected_psnr,
                    "selected_delta_group_a_psnr_vs_stage0": (
                        selected_psnr - stage0_psnr
                    ),
                    "selected_group_a_ssim": selected_ssim,
                    "selected_delta_group_a_ssim_vs_stage0": (
                        selected_ssim - stage0_ssim
                    ),
                    "SSIM_RETENTION_RISK": selected_ssim < stage0_ssim,
                }
                for key, expected in retention_expected.items():
                    actual = complete.get(key)
                    if isinstance(expected, bool):
                        matches = isinstance(actual, bool) and actual is expected
                    else:
                        matches = self._is_finite_number(actual) and actual == expected
                    if not matches:
                        mismatches[key] = {"expected": expected, "actual": actual}
                report_retention_lines = (
                    f"- Stage0 Group-A PSNR anchor: {stage0_psnr!r}",
                    f"- Stage0 Group-A SSIM anchor: {stage0_ssim!r}",
                    "- Selected Group-A PSNR delta vs Stage0: "
                    f"{selected_psnr - stage0_psnr!r}",
                    "- Selected Group-A SSIM delta vs Stage0: "
                    f"{selected_ssim - stage0_ssim!r}",
                    "- SSIM_RETENTION_RISK: "
                    f"{str(selected_ssim < stage0_ssim).lower()}",
                )
                report_lines = set(report.splitlines())
                for line in report_retention_lines:
                    if line not in report_lines:
                        mismatches[f"report.retention.{line.partition(':')[0]}"] = {
                            "expected": line,
                            "actual": "missing or different",
                        }
                if (
                    selected_ssim < stage0_ssim
                    and "risk is not offset by any average PSNR gain" not in report
                ):
                    mismatches["report.retention.psnr_non_offset"] = {
                        "expected": "explicit PSNR non-offset statement",
                        "actual": "missing",
                    }
                decision_memo = self.paths.project_root / "DECISION_MEMO.md"
                if decision_memo.is_file():
                    decision_text = self._load_text(
                        decision_memo, context="final model decision memo"
                    )
                    required_decision_fragments = (
                        "SSIM_RETENTION_RISK",
                        repr(stage0_ssim),
                        repr(selected_ssim),
                        repr(selected_ssim - stage0_ssim),
                        repr(selected_psnr - stage0_psnr),
                    )
                    if selected_ssim < stage0_ssim:
                        required_decision_fragments += (
                            "does not offset the SSIM retention deficit",
                        )
                    for fragment in required_decision_fragments:
                        if fragment not in decision_text:
                            mismatches[f"decision_memo.retention.{fragment}"] = {
                                "expected": "present",
                                "actual": "missing",
                            }
            validation = self._load_json_mapping(
                outputs["validation"], context="Stage4 latest validation"
            )
            validation_expected = {
                "validation": str(outputs["validation"].resolve()),
                "validation_sha256": sha256_file(outputs["validation"]),
            }
            for key, expected in validation_expected.items():
                if complete.get(key) != expected:
                    mismatches[key] = {
                        "expected": expected,
                        "actual": complete.get(key),
                    }
            validation_group = validation.get("group_a_equal_combination_mean")
            if not isinstance(validation_group, Mapping):
                mismatches["validation.group_a_equal_combination_mean"] = {
                    "expected": "mapping",
                    "actual": type(validation_group).__name__,
                }
            else:
                for metric in ("psnr", "ssim"):
                    if not self._is_finite_number(validation_group.get(metric)):
                        mismatches[f"validation.group_a.{metric}"] = {
                            "expected": "finite number",
                            "actual": validation_group.get(metric),
                        }
            latest_score = complete.get("latest_score")
            if not isinstance(latest_score, Mapping):
                mismatches["latest_score"] = {
                    "expected": "mapping with final validation metrics",
                    "actual": type(latest_score).__name__,
                }
            else:
                latest_expected: dict[str, object] = {}
                if isinstance(validation_group, Mapping):
                    latest_expected.update(
                        {
                            "group_a_psnr": validation_group.get("psnr"),
                            "group_a_ssim": validation_group.get("ssim"),
                        }
                    )
                validation_single = validation.get("single_equal_task_mean")
                if isinstance(validation_single, Mapping):
                    latest_expected.update(
                        {
                            "single_psnr": validation_single.get("psnr"),
                            "single_ssim": validation_single.get("ssim"),
                        }
                    )
                else:
                    mismatches["validation.single_equal_task_mean"] = {
                        "expected": "mapping",
                        "actual": type(validation_single).__name__,
                    }
                latest_expected["step"] = expected_step
                for key, expected in latest_expected.items():
                    actual = latest_score.get(key)
                    if key == "step":
                        matches = (
                            not isinstance(actual, bool)
                            and isinstance(actual, int)
                            and actual == expected
                        )
                    else:
                        matches = (
                            self._is_finite_number(expected)
                            and self._is_finite_number(actual)
                            and actual == expected
                        )
                    if not matches:
                        mismatches[f"latest_score.{key}"] = {
                            "expected": expected,
                            "actual": actual,
                        }
            diagnostics_json = outputs["diagnostics_json"]
            diagnostics_report = outputs["diagnostics"]
            diagnostics_json_sha = sha256_file(diagnostics_json)
            diagnostics_report_sha = sha256_file(diagnostics_report)
            completion_diagnostics = {
                "diagnostics_json": str(diagnostics_json.resolve()),
                "diagnostics_json_sha256": diagnostics_json_sha,
                "diagnostics_report": str(diagnostics_report.resolve()),
                "diagnostics_report_sha256": diagnostics_report_sha,
                "diagnostics_selected_best_ema_sha256": best_sha,
            }
            for key, expected in completion_diagnostics.items():
                if complete.get(key) != expected:
                    mismatches[key] = {
                        "expected": expected,
                        "actual": complete.get(key),
                    }
            diagnostics = self._load_json_mapping(
                diagnostics_json, context="Stage4 zero-training diagnostics"
            )
            diagnostics_expected = {
                "schema_version": ("graphrestore-stage4-zero-training-diagnostics-v1"),
                "protocol_id": PROTOCOL_ID,
                "selected_best_ema_path": str(outputs["best"].resolve()),
                "selected_best_ema_sha256": best_sha,
                "optimizer_updates": 0,
                "model_ema_rng_unchanged": True,
            }
            for key, expected in diagnostics_expected.items():
                if diagnostics.get(key) != expected:
                    mismatches[f"diagnostics.{key}"] = {
                        "expected": expected,
                        "actual": diagnostics.get(key),
                    }
            required_modes = {
                "compiler_modes": (
                    "full_partial_order",
                    "forced_total_order",
                    "parallel_only",
                ),
                "guard_modes": (
                    "predicted_spatial",
                    "global_mean",
                    "all_one",
                ),
            }
            for family, expected_modes in required_modes.items():
                actual_modes = diagnostics.get(family)
                if not isinstance(actual_modes, Mapping) or set(actual_modes) != set(
                    expected_modes
                ):
                    mismatches[f"diagnostics.{family}"] = {
                        "expected": list(expected_modes),
                        "actual": (
                            list(actual_modes)
                            if isinstance(actual_modes, Mapping)
                            else actual_modes
                        ),
                    }
                else:
                    for mode in expected_modes:
                        mode_value = actual_modes[mode]
                        mode_prefix = f"diagnostics.{family}.{mode}"
                        if not isinstance(mode_value, Mapping):
                            mismatches[mode_prefix] = {
                                "expected": "mapping",
                                "actual": type(mode_value).__name__,
                            }
                            continue
                        for aggregate_name in (
                            "single_equal_task_mean",
                            "group_a_equal_combination_mean",
                        ):
                            aggregate = mode_value.get(aggregate_name)
                            aggregate_prefix = f"{mode_prefix}.{aggregate_name}"
                            if not isinstance(aggregate, Mapping):
                                mismatches[aggregate_prefix] = {
                                    "expected": "mapping with finite psnr/ssim",
                                    "actual": type(aggregate).__name__,
                                }
                                continue
                            for metric_name in ("psnr", "ssim"):
                                metric = aggregate.get(metric_name)
                                if not self._is_finite_number(metric):
                                    mismatches[f"{aggregate_prefix}.{metric_name}"] = {
                                        "expected": "finite number",
                                        "actual": metric,
                                    }
                        mode_diagnostics = mode_value.get("diagnostics")
                        if (
                            not isinstance(mode_diagnostics, Mapping)
                            or not mode_diagnostics
                        ):
                            mismatches[f"{mode_prefix}.diagnostics"] = {
                                "expected": "non-empty mapping",
                                "actual": mode_diagnostics,
                            }
                        image_count = mode_value.get("image_count")
                        if (
                            isinstance(image_count, bool)
                            or not isinstance(image_count, int)
                            or image_count <= 0
                        ):
                            mismatches[f"{mode_prefix}.image_count"] = {
                                "expected": "positive integer",
                                "actual": image_count,
                            }
                        peak_bytes = mode_value.get("peak_reserved_bytes")
                        if (
                            isinstance(peak_bytes, bool)
                            or not isinstance(peak_bytes, int)
                            or peak_bytes < 0
                        ):
                            mismatches[f"{mode_prefix}.peak_reserved_bytes"] = {
                                "expected": "non-negative integer",
                                "actual": peak_bytes,
                            }
                        peak_fraction = mode_value.get("peak_reserved_fraction")
                        if not self._is_finite_number(peak_fraction) or not (
                            0.0 <= float(peak_fraction) <= 0.90
                        ):
                            mismatches[f"{mode_prefix}.peak_reserved_fraction"] = {
                                "expected": "finite number in [0, 0.90]",
                                "actual": peak_fraction,
                            }
            diagnostics_report_text = self._load_text(
                diagnostics_report,
                context="Stage4 guard and misuse diagnostics report",
            )
            if best_sha not in diagnostics_report_text:
                mismatches["diagnostics_report.selected_best_ema_sha256"] = {
                    "expected": best_sha,
                    "actual": "missing",
                }
            for expected_modes in required_modes.values():
                for mode in expected_modes:
                    if mode not in diagnostics_report_text:
                        mismatches[f"diagnostics_report.mode.{mode}"] = {
                            "expected": "present",
                            "actual": "missing",
                        }
        if mismatches:
            raise OrchestrationError(
                f"completed {stage} evidence/hash mismatch: {mismatches}"
            )
        return outputs["best"]

    def _resumable_post_approval_command(
        self,
        command: CommandSpec,
        *,
        approval_sha256: str,
        extension_authorization: Stage3ExtensionAuthorization | None = None,
        finalization_authorization: Stage3RevocationAuthorization | None = None,
        stage4_extension_authorization: Stage4ExtensionAuthorization | None = None,
    ) -> CommandSpec:
        if command.name == "stage3" and finalization_authorization is not None:
            return self._python_command(
                "stage3",
                "scripts/finalize_stage3.py",
                "--config",
                "configs/stage3_planner.yaml",
                "--finalization_authorization",
                str(finalization_authorization.path),
            )
        last = self._post_approval_outputs()[command.name]["last"]
        if not last.is_file():
            raise OrchestrationError(
                f"cannot resume incomplete {command.name}: missing raw last.pth: {last}"
            )
        self._load_checkpoint_header(
            last,
            stage=command.name,
            model_role="raw_training_state",
            resumable=True,
            approval_sha256=approval_sha256,
            extension_authorization=extension_authorization,
            stage4_extension_authorization=(
                stage4_extension_authorization if command.name == "stage4" else None
            ),
        )
        extension_arguments: tuple[str, ...] = ()
        if command.name == "stage3" and extension_authorization is not None:
            extension_arguments = (
                "--extension_authorization",
                str(extension_authorization.path),
            )
        if command.name == "stage4" and stage4_extension_authorization is not None:
            extension_arguments = (
                "--extension_authorization",
                str(stage4_extension_authorization.gate_path),
            )
        return CommandSpec(
            name=command.name,
            argv=(
                *command.argv,
                "--resume",
                str(last),
                *extension_arguments,
            ),
        )

    def _post_approval_stage_from_last_command(
        self,
        state: PipelineState,
        *,
        extension_authorization: Stage3ExtensionAuthorization | None = None,
        finalization_authorization: Stage3RevocationAuthorization | None = None,
        stage4_extension_authorization: Stage4ExtensionAuthorization | None = None,
    ) -> str:
        if not state.last_command:
            raise OrchestrationError(
                "post-approval recovery lacks a durable last_command"
            )
        actual = tuple(state.last_command)
        for _, base in self.post_approval_commands():
            last = self._post_approval_outputs()[base.name]["last"]
            accepted = [base.argv, (*base.argv, "--resume", str(last))]
            if base.name == "stage3" and extension_authorization is not None:
                accepted.append(
                    (
                        *base.argv,
                        "--resume",
                        str(last),
                        "--extension_authorization",
                        str(extension_authorization.path),
                    )
                )
            if base.name == "stage3" and finalization_authorization is not None:
                # The permanent revocation supersedes either the interrupted
                # extension child or a prior finalize-only attempt.  These are
                # the only Stage3 commands accepted on this path; a plain
                # trainer command can never bypass the tombstone.
                canonical_extension = self.paths.stage3_extension_approval.resolve()
                accepted.extend(
                    (
                        (
                            *base.argv,
                            "--resume",
                            str(last),
                            "--extension_authorization",
                            str(canonical_extension),
                        ),
                        self._python_command(
                            "stage3",
                            "scripts/finalize_stage3.py",
                            "--config",
                            "configs/stage3_planner.yaml",
                            "--finalization_authorization",
                            str(finalization_authorization.path),
                        ).argv,
                    )
                )
            if base.name == "stage4" and stage4_extension_authorization is not None:
                accepted.append(
                    (
                        *base.argv,
                        "--resume",
                        str(last),
                        "--extension_authorization",
                        str(stage4_extension_authorization.gate_path),
                    )
                )
            if actual in accepted:
                return base.name
        raise OrchestrationError(
            "post-approval recovery last_command is not an exact Stage3/Stage4 child"
        )

    def _mark_post_approval_failed(
        self,
        state: PipelineState,
        error: BaseException,
        *,
        command_name: str | None = None,
        extension_authorization: Stage3ExtensionAuthorization | Path | None = None,
        finalization_authorization: Stage3RevocationAuthorization | Path | None = None,
        stage4_extension_authorization: (
            Stage4ExtensionAuthorization | Path | None
        ) = None,
    ) -> None:
        if command_name is not None and command_name in state.completed:
            state.completed.remove(command_name)
        state.status = PipelineStatus.FAILED.value
        state.current_stage = "FAILED"
        state.gpu = "released"
        state.last_error = str(error)
        resume_arguments = [
            "python",
            "scripts/orchestrate.py",
            "--resume_post_approval_pipeline",
        ]
        if finalization_authorization is not None:
            finalization_path = (
                finalization_authorization.path
                if isinstance(finalization_authorization, Stage3RevocationAuthorization)
                else finalization_authorization
            )
            resume_arguments.extend(
                ("--stage3_finalization_authorization", str(finalization_path))
            )
        elif extension_authorization is not None:
            extension_path = (
                extension_authorization.path
                if isinstance(extension_authorization, Stage3ExtensionAuthorization)
                else extension_authorization
            )
            resume_arguments.extend(
                ("--stage3_extension_authorization", str(extension_path))
            )
        if stage4_extension_authorization is not None:
            stage4_extension_path = (
                stage4_extension_authorization.gate_path
                if isinstance(
                    stage4_extension_authorization, Stage4ExtensionAuthorization
                )
                else stage4_extension_authorization
            )
            resume_arguments.extend(
                ("--stage4_extension_authorization", str(stage4_extension_path))
            )
        state.next_command = (
            POST_APPROVAL_RESUME_COMMAND
            if len(resume_arguments) == 3
            else shlex.join(resume_arguments)
        )
        self._persist(state)

    def _load_and_verify_approval_granted(
        self,
        state: PipelineState,
    ) -> Mapping[str, Any]:
        required = self._load_and_verify_approval_required(state)
        if not self.paths.approval_granted.is_file():
            raise ApprovalError(
                f"missing persisted Stage3 approval: {self.paths.approval_granted}"
            )
        approval = self._load_json_mapping(
            self.paths.approval_granted, context="persisted Stage3 approval"
        )
        approval_sha = sha256_file(self.paths.approval_granted)
        expected = {
            "schema_version": APPROVAL_SCHEMA,
            "kind": "stage3_approval",
            "protocol_id": PROTOCOL_ID,
            "approved": True,
            "stage2_decision_path": required["stage2_decision"]["path"],
            "stage2_decision_sha256": required["stage2_decision"]["sha256"],
            "approval_required_path": str(self.paths.approval_required),
            "approval_required_sha256": sha256_file(self.paths.approval_required),
            "bindings": required["bindings"],
            "scientific_adjudications": {"D-017": dict(D017_ACCEPTANCE)},
            "authorized_pipeline": ["stage3", "stage4"],
            "formal_mio100_authorized": False,
        }
        mismatches = {
            key: {"expected": value, "actual": approval.get(key)}
            for key, value in expected.items()
            if approval.get(key) != value
        }
        approved_utc = approval.get("approved_utc")
        if not isinstance(approved_utc, str) or not approved_utc.endswith("Z"):
            mismatches["approved_utc"] = {
                "expected": "UTC timestamp ending in Z",
                "actual": approved_utc,
            }
        if state.stage3_approval_sha256 != approval_sha:
            mismatches["durable_state.stage3_approval_sha256"] = {
                "expected": approval_sha,
                "actual": state.stage3_approval_sha256,
            }
        if mismatches:
            raise ApprovalError(
                f"persisted Stage3 approval/hash binding mismatch: {mismatches}"
            )
        return approval

    def _verify_extension_file_binding(
        self,
        value: object,
        *,
        field: str,
        expected_path: Path,
        expected_sha256: str | None = None,
        immutable_backup: bool = False,
    ) -> Path:
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ApprovalError(
                f"Stage3 extension {field} must contain exactly path and sha256"
            )
        raw_path = value.get("path")
        digest = value.get("sha256")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ApprovalError(
                f"Stage3 extension {field}.path must be an absolute path"
            )
        path = Path(raw_path)
        absolute = Path(os.path.abspath(os.fspath(path)))
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ApprovalError(
                    f"Stage3 extension {field}.path contains a symlink: {current}"
                )
        if str(path.resolve()) != raw_path or path.resolve() != expected_path.resolve():
            raise ApprovalError(f"Stage3 extension {field}.path drifted")
        if not is_sha256(digest):
            raise ApprovalError(f"Stage3 extension {field}.sha256 is invalid")
        if expected_sha256 is not None and digest != expected_sha256:
            raise ApprovalError(f"Stage3 extension {field}.sha256 drifted")
        if not path.is_file() or sha256_file(path) != digest:
            raise ApprovalError(
                f"Stage3 extension {field} physical file/hash is missing or stale"
            )
        if immutable_backup and stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise ApprovalError(
                f"Stage3 extension {field} backup mode must be exactly 0444"
            )
        return path.resolve()

    def _load_and_verify_stage3_extension_authorization(
        self,
        state: PipelineState,
        requested_path: str | Path,
    ) -> Stage3ExtensionAuthorization:
        """Verify the one approved three-cycle extension without mutating it."""

        approval = self._load_and_verify_approval_granted(state)
        requested = Path(requested_path)
        expected_path = self.paths.stage3_extension_approval
        absolute = Path(os.path.abspath(os.fspath(requested)))
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ApprovalError(
                    f"Stage3 extension authorization path contains a symlink: {current}"
                )
        canonical = expected_path.resolve()
        if (
            not requested.is_absolute()
            or str(requested.resolve()) != str(requested)
            or requested.resolve() != canonical
            or not requested.is_file()
        ):
            raise ApprovalError(
                "Stage3 extension authorization must be the canonical regular file: "
                f"{canonical}"
            )
        authorization_sha256 = sha256_file(canonical)
        payload = self._load_json_mapping(
            canonical, context="Stage3 extension authorization"
        )
        if sha256_file(canonical) != authorization_sha256:
            raise ApprovalError("Stage3 extension authorization changed while loading")
        exact_keys = {
            "schema_version",
            "kind",
            "protocol_id",
            "approved",
            "cycles",
            "base_step",
            "target_step",
            "validation_every_steps",
            "validation_steps",
            "schedule_horizon_steps",
            "min_lr",
            "lr_policy",
            "formal_mio100_authorized",
            "authorized_pipeline",
            "base_stage3_approval",
            "base_approval_required",
            "base_stage3_config",
            "pre_extension_run_contract",
            "pre_extension_last_checkpoint",
            "pre_extension_best_checkpoint",
        }
        if set(payload) != exact_keys:
            raise ApprovalError(
                "Stage3 extension authorization fields drifted: "
                f"expected={sorted(exact_keys)}, actual={sorted(payload)}"
            )
        expected_scalars: dict[str, Any] = {
            "schema_version": STAGE3_EXTENSION_APPROVAL_SCHEMA,
            "kind": "stage3_extension_approval",
            "protocol_id": PROTOCOL_ID,
            "approved": True,
            "cycles": STAGE3_EXTENSION_CYCLES,
            "base_step": STAGE3_EXTENSION_BASE_STEP,
            "target_step": STAGE3_EXTENSION_TARGET_STEP,
            "validation_every_steps": STAGE3_EXTENSION_VALIDATION_EVERY_STEPS,
            "validation_steps": list(STAGE3_EXTENSION_VALIDATION_STEPS),
            "schedule_horizon_steps": STAGE3_EXTENSION_SCHEDULE_HORIZON_STEPS,
            "min_lr": STAGE3_EXTENSION_MIN_LR,
            "lr_policy": STAGE3_EXTENSION_LR_POLICY,
            "formal_mio100_authorized": False,
            "authorized_pipeline": ["stage3_extension", "stage4"],
        }
        scalar_mismatches: dict[str, dict[str, Any]] = {}
        for key, expected in expected_scalars.items():
            actual = payload.get(key)
            if isinstance(expected, bool):
                matches = isinstance(actual, bool) and actual is expected
            elif isinstance(expected, int):
                matches = (
                    isinstance(actual, int)
                    and not isinstance(actual, bool)
                    and actual == expected
                )
            elif isinstance(expected, float):
                matches = (
                    isinstance(actual, (int, float))
                    and not isinstance(actual, bool)
                    and float(actual) == expected
                )
            else:
                matches = actual == expected
            if not matches:
                scalar_mismatches[key] = {
                    "expected": expected,
                    "actual": actual,
                }
        if scalar_mismatches:
            raise ApprovalError(
                f"Stage3 extension authorization contract drifted: {scalar_mismatches}"
            )

        approval_sha = sha256_file(self.paths.approval_granted)
        required_sha = sha256_file(self.paths.approval_required)
        bindings = approval.get("bindings")
        if not isinstance(bindings, Mapping) or not isinstance(
            bindings.get("config_stage3"), Mapping
        ):
            raise ApprovalError("Stage3 approval lacks the config_stage3 binding")
        config_binding = bindings["config_stage3"]
        config_path = Path(str(config_binding.get("path"))).resolve()
        config_sha = config_binding.get("sha256")
        if not isinstance(config_sha, str):
            raise ApprovalError("Stage3 approval config binding is invalid")
        self._verify_extension_file_binding(
            payload["base_stage3_approval"],
            field="base_stage3_approval",
            expected_path=self.paths.approval_granted,
            expected_sha256=approval_sha,
        )
        self._verify_extension_file_binding(
            payload["base_approval_required"],
            field="base_approval_required",
            expected_path=self.paths.approval_required,
            expected_sha256=required_sha,
        )
        self._verify_extension_file_binding(
            payload["base_stage3_config"],
            field="base_stage3_config",
            expected_path=config_path,
            expected_sha256=config_sha,
        )

        backup_root = (
            self.paths.project_root / STAGE3_EXTENSION_BACKUP_DIRECTORY
        ).resolve()
        run_contract_backup = self._verify_extension_file_binding(
            payload["pre_extension_run_contract"],
            field="pre_extension_run_contract",
            expected_path=backup_root / "run_contract.json",
            immutable_backup=True,
        )
        last_backup = self._verify_extension_file_binding(
            payload["pre_extension_last_checkpoint"],
            field="pre_extension_last_checkpoint",
            expected_path=backup_root / "last.pth",
            immutable_backup=True,
        )
        best_backup = self._verify_extension_file_binding(
            payload["pre_extension_best_checkpoint"],
            field="pre_extension_best_checkpoint",
            expected_path=backup_root / "best_ema.pth",
            immutable_backup=True,
        )

        run_contract = self._load_json_mapping(
            run_contract_backup, context="pre-extension Stage3 run contract"
        )
        provenance = run_contract.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ApprovalError("pre-extension Stage3 run contract lacks provenance")
        runtime = provenance.get("runtime")
        recorded_approval = provenance.get("stage3_approval")
        if (
            run_contract.get("schema_version") != "graphrestore-stage3-runtime-v1"
            or not isinstance(runtime, Mapping)
            or runtime.get("max_steps") != STAGE3_EXTENSION_BASE_STEP
            or provenance.get("config_sha256") != config_sha
            or not isinstance(recorded_approval, Mapping)
            or recorded_approval.get("sha256") != approval_sha
            or "stage3_extension" in provenance
        ):
            raise ApprovalError("pre-extension Stage3 run contract anchor drifted")

        last = self._load_checkpoint_header(
            last_backup,
            stage="stage3",
            model_role="raw_training_state",
            resumable=True,
            approval_sha256=approval_sha,
        )
        best = self._load_checkpoint_header(
            best_backup,
            stage="stage3",
            model_role="ema_selection",
            resumable=False,
            approval_sha256=approval_sha,
        )
        for label, checkpoint in (("last", last), ("best", best)):
            checkpoint_provenance = checkpoint.get("provenance")
            if (
                checkpoint.get("step") != STAGE3_EXTENSION_BASE_STEP
                or checkpoint.get("pending_validation_step") is not None
                or not isinstance(checkpoint_provenance, Mapping)
                or "stage3_extension" in checkpoint_provenance
            ):
                raise ApprovalError(
                    f"pre-extension Stage3 {label} checkpoint anchor drifted"
                )
        last_metrics = last.get("metrics")
        best_metrics = best.get("metrics")
        if (
            not isinstance(last_metrics, Mapping)
            or last_metrics.get("validation_step") != STAGE3_EXTENSION_BASE_STEP
            or last_metrics.get("best_step") != STAGE3_EXTENSION_BASE_STEP
            or not isinstance(best_metrics, Mapping)
            or best_metrics.get("validation_step") != STAGE3_EXTENSION_BASE_STEP
            or best_metrics.get("best_step") != STAGE3_EXTENSION_BASE_STEP
        ):
            raise ApprovalError("pre-extension Stage3 12k metric anchors drifted")

        return Stage3ExtensionAuthorization(
            path=canonical,
            sha256=authorization_sha256,
            payload=payload,
        )

    def _load_and_verify_stage3_finalization_authorization(
        self,
        state: PipelineState,
        requested_path: str | Path,
    ) -> Stage3RevocationAuthorization:
        """Verify the canonical permanent extension tombstone without mutation."""

        self._load_and_verify_approval_granted(state)
        requested = Path(requested_path)
        canonical = self.paths.stage3_extension_revocation
        absolute = Path(os.path.abspath(os.fspath(requested)))
        if (
            not requested.is_absolute()
            or requested != absolute
            or requested != canonical
        ):
            raise ApprovalError(
                "Stage3 finalization authorization must be the canonical path: "
                f"{canonical}"
            )
        try:
            authorization = validate_stage3_extension_revocation(
                requested,
                project_root=self.paths.project_root,
                require_present=True,
            )
        except Exception as exc:
            if isinstance(exc, ApprovalError):
                raise
            raise ApprovalError(
                f"Stage3 finalization authorization refused: {exc}"
            ) from exc
        if authorization.path != canonical:
            raise ApprovalError(
                "Stage3 finalization validator returned a non-canonical artifact"
            )
        return authorization

    def _load_and_verify_stage4_extension_authorization(
        self,
        state: PipelineState,
        requested_path: str | Path,
    ) -> Stage4ExtensionAuthorization:
        """Verify the activated 40k gate before any resumed child/CUDA setup."""

        self._load_and_verify_approval_granted(state)
        requested = Path(requested_path)
        canonical_gate = self.paths.stage4_extension_gate_receipt
        canonical_conditional = self.paths.stage4_extension_conditional_approval
        absolute = Path(os.path.abspath(os.fspath(requested)))
        if (
            not requested.is_absolute()
            or requested != absolute
            or requested != canonical_gate
        ):
            raise ApprovalError(
                "Stage4 extension authorization must be the canonical activated "
                f"gate receipt: {canonical_gate}"
            )
        try:
            # Kept local so ordinary Stage0/1/2 orchestration does not import
            # the Stage4 model stack.  The validator is CPU-only and rejects
            # malformed/non-activated gate evidence before a child is started.
            from src.training.stage4_engine import (
                validate_stage4_extension_authorization,
            )

            evidence = validate_stage4_extension_authorization(
                requested,
                project_root=self.paths.project_root,
                config_path=(
                    self.paths.project_root / "configs/stage4_graphrestore_e2e.yaml"
                ),
            )
        except Exception as exc:
            if isinstance(exc, ApprovalError):
                raise
            raise ApprovalError(
                f"Stage4 extension authorization refused: {exc}"
            ) from exc
        if (
            evidence.gate_path != canonical_gate
            or evidence.conditional_path != canonical_conditional
            or sha256_file(canonical_gate) != evidence.gate_sha256
            or sha256_file(canonical_conditional) != evidence.conditional_sha256
        ):
            raise ApprovalError(
                "Stage4 extension validator returned a stale or non-canonical binding"
            )
        return Stage4ExtensionAuthorization(
            conditional_path=evidence.conditional_path,
            conditional_sha256=evidence.conditional_sha256,
            gate_path=evidence.gate_path,
            gate_sha256=evidence.gate_sha256,
            payload=self._load_json_mapping(
                canonical_gate, context="Stage4 extension gate receipt"
            ),
        )

    def _continue_post_approval_pipeline(
        self,
        state: PipelineState,
        *,
        resume_stage: str | None,
        extension_authorization: Stage3ExtensionAuthorization | None = None,
        finalization_authorization: Stage3RevocationAuthorization | None = None,
        stage4_extension_authorization: Stage4ExtensionAuthorization | None = None,
    ) -> PipelineState:
        """Shared Stage3→Stage4 continuation for first approval and recovery."""

        try:
            self._load_and_verify_approval_granted(state)
        except OrchestrationError as exc:
            self._mark_post_approval_failed(
                state,
                exc,
                extension_authorization=extension_authorization,
                finalization_authorization=finalization_authorization,
                stage4_extension_authorization=stage4_extension_authorization,
            )
            raise
        approval_sha = sha256_file(self.paths.approval_granted)
        resume_reached = resume_stage is None
        for status, base_command in self.post_approval_commands():
            if base_command.name in state.completed:
                try:
                    best = self._verify_post_approval_completion(
                        base_command.name,
                        approval_sha256=approval_sha,
                        extension_authorization=extension_authorization,
                        finalization_authorization=finalization_authorization,
                        stage4_extension_authorization=(stage4_extension_authorization),
                    )
                except OrchestrationError as exc:
                    self._mark_post_approval_failed(
                        state,
                        exc,
                        command_name=base_command.name,
                        extension_authorization=extension_authorization,
                        finalization_authorization=finalization_authorization,
                        stage4_extension_authorization=(stage4_extension_authorization),
                    )
                    raise
                if base_command.name == resume_stage:
                    resume_reached = True
                state.last_checkpoint = str(best)
                self._persist(state)
                continue
            if resume_stage is not None and not resume_reached:
                if base_command.name != resume_stage:
                    exc = OrchestrationError(
                        f"cannot recover {resume_stage}: prior {base_command.name} "
                        "lacks completed full-output evidence"
                    )
                    self._mark_post_approval_failed(
                        state,
                        exc,
                        extension_authorization=extension_authorization,
                        finalization_authorization=finalization_authorization,
                        stage4_extension_authorization=(stage4_extension_authorization),
                    )
                    raise exc
                try:
                    command = self._resumable_post_approval_command(
                        base_command,
                        approval_sha256=approval_sha,
                        extension_authorization=extension_authorization,
                        finalization_authorization=finalization_authorization,
                        stage4_extension_authorization=(stage4_extension_authorization),
                    )
                except OrchestrationError as exc:
                    self._mark_post_approval_failed(
                        state,
                        exc,
                        extension_authorization=extension_authorization,
                        finalization_authorization=finalization_authorization,
                        stage4_extension_authorization=(stage4_extension_authorization),
                    )
                    raise
                resume_reached = True
            else:
                command = base_command
            try:
                self._run_child(state, status, command)
            except (ChildCommandError, KeyboardInterrupt) as exc:
                self._mark_post_approval_failed(
                    state,
                    exc,
                    extension_authorization=extension_authorization,
                    finalization_authorization=finalization_authorization,
                    stage4_extension_authorization=stage4_extension_authorization,
                )
                raise
            try:
                best = self._verify_post_approval_completion(
                    command.name,
                    approval_sha256=approval_sha,
                    extension_authorization=extension_authorization,
                    finalization_authorization=finalization_authorization,
                    stage4_extension_authorization=stage4_extension_authorization,
                )
            except OrchestrationError as exc:
                self._mark_post_approval_failed(
                    state,
                    exc,
                    command_name=command.name,
                    extension_authorization=extension_authorization,
                    finalization_authorization=finalization_authorization,
                    stage4_extension_authorization=stage4_extension_authorization,
                )
                raise
            state.last_checkpoint = str(best)
            self._persist(state)

        if resume_stage is not None and not resume_reached:
            exc = OrchestrationError(
                f"post-approval resume target was not reached: {resume_stage}"
            )
            self._mark_post_approval_failed(
                state,
                exc,
                extension_authorization=extension_authorization,
                finalization_authorization=finalization_authorization,
                stage4_extension_authorization=stage4_extension_authorization,
            )
            raise exc
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
        if self.paths.approval_granted.exists():
            raise ApprovalError(
                "refusing to overwrite an existing STAGE3_APPROVED.json; use "
                "--resume_post_approval_pipeline for an interrupted approved run"
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
            "scientific_adjudications": {"D-017": dict(D017_ACCEPTANCE)},
            "authorized_pipeline": ["stage3", "stage4"],
            "formal_mio100_authorized": False,
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
        return self._continue_post_approval_pipeline(state, resume_stage=None)

    def resume_post_approval_pipeline(
        self,
        *,
        stage3_extension_authorization: str | Path | None = None,
        stage3_finalization_authorization: str | Path | None = None,
        stage4_extension_authorization: str | Path | None = None,
    ) -> PipelineState:
        """Resume an already-approved Stage3/4 run without rewriting approval."""

        state = self.load_state()
        state = self._recover_stale_running(
            state,
            allowed_running={
                PipelineStatus.STAGE3_RUNNING.value,
                PipelineStatus.STAGE4_RUNNING.value,
            },
            recovered_status=PipelineStatus.FAILED,
        )
        if state.status != PipelineStatus.FAILED.value:
            raise OrchestrationError(
                "--resume_post_approval_pipeline is allowed only from FAILED or "
                "a proven-stale STAGE3_RUNNING/STAGE4_RUNNING state"
            )
        extension: Stage3ExtensionAuthorization | None = None
        finalization: Stage3RevocationAuthorization | None = None
        stage4_extension: Stage4ExtensionAuthorization | None = None
        canonical_extension_path = self.paths.stage3_extension_approval.resolve()
        canonical_finalization_path = self.paths.stage3_extension_revocation
        canonical_stage4_extension_path = self.paths.stage4_extension_gate_receipt
        try:
            self._load_and_verify_approval_granted(state)
            if (
                stage3_extension_authorization is not None
                and stage3_finalization_authorization is not None
            ):
                raise ApprovalError(
                    "Stage3 extension and finalize-only authorizations are mutually "
                    "exclusive"
                )
            if (
                os.path.lexists(self.paths.stage3_extension_revocation)
                and stage3_finalization_authorization is None
            ):
                raise ApprovalError(
                    "STAGE3_EXTENSION_REVOKED.json permanently disables plain/"
                    "extension Stage3 recovery; pass the canonical "
                    "--stage3_finalization_authorization"
                )
            if stage3_extension_authorization is not None:
                extension = self._load_and_verify_stage3_extension_authorization(
                    state, stage3_extension_authorization
                )
            if stage3_finalization_authorization is not None:
                finalization = self._load_and_verify_stage3_finalization_authorization(
                    state, stage3_finalization_authorization
                )
            if stage4_extension_authorization is not None:
                stage4_extension = self._load_and_verify_stage4_extension_authorization(
                    state, stage4_extension_authorization
                )
            resume_stage = self._post_approval_stage_from_last_command(
                state,
                extension_authorization=extension,
                finalization_authorization=finalization,
                stage4_extension_authorization=stage4_extension,
            )
            if (
                extension is not None
                and resume_stage == "stage3"
                and "stage3" in state.completed
            ):
                raise OrchestrationError(
                    "Stage3 extension cannot restart after Stage3 was marked complete"
                )
            if stage4_extension is not None and resume_stage != "stage4":
                raise OrchestrationError(
                    "Stage4 extension authorization can resume only the exact "
                    "interrupted Stage4 child"
                )
            if stage4_extension is not None and "stage4" in state.completed:
                raise OrchestrationError(
                    "Stage4 extension cannot restart after Stage4 was marked complete"
                )
        except (ApprovalError, OrchestrationError) as exc:
            preserve_extension_path: Path | None = None
            preserve_finalization_path: Path | None = None
            preserve_stage4_extension_path: Path | None = None
            if stage3_extension_authorization is not None:
                preserve_extension_path = canonical_extension_path
            if stage3_finalization_authorization is not None:
                preserve_finalization_path = canonical_finalization_path
            elif os.path.lexists(self.paths.stage3_extension_revocation):
                preserve_finalization_path = canonical_finalization_path
            elif state.last_command:
                expected_finalization_suffix = (
                    "--finalization_authorization",
                    str(canonical_finalization_path),
                )
                if tuple(state.last_command[-2:]) == expected_finalization_suffix:
                    preserve_finalization_path = canonical_finalization_path
                expected_suffix = (
                    "--extension_authorization",
                    str(canonical_extension_path),
                )
                if (
                    preserve_finalization_path is None
                    and tuple(state.last_command[-2:]) == expected_suffix
                ):
                    preserve_extension_path = canonical_extension_path
            if stage4_extension_authorization is not None:
                preserve_stage4_extension_path = canonical_stage4_extension_path
            elif state.last_command:
                expected_stage4_extension_suffix = (
                    "--extension_authorization",
                    str(canonical_stage4_extension_path),
                )
                if tuple(state.last_command[-2:]) == expected_stage4_extension_suffix:
                    preserve_stage4_extension_path = canonical_stage4_extension_path
            self._mark_post_approval_failed(
                state,
                exc,
                extension_authorization=preserve_extension_path,
                finalization_authorization=preserve_finalization_path,
                stage4_extension_authorization=preserve_stage4_extension_path,
            )
            raise
        return self._continue_post_approval_pipeline(
            state,
            resume_stage=resume_stage,
            extension_authorization=extension,
            finalization_authorization=finalization,
            stage4_extension_authorization=stage4_extension,
        )

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
            "global_priority": root
            / "artifacts/interaction_labels/global_priority.json",
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
                raise OrchestrationError(
                    f"cannot bind missing artifact {logical}: {path}"
                )
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
            mismatches["approved"] = {
                "expected": False,
                "actual": decision.get("approved"),
            }
        if not isinstance(decision.get("overall"), Mapping):
            mismatches["overall"] = {
                "expected": "mapping",
                "actual": type(decision.get("overall")).__name__,
            }
        if not isinstance(decision.get("warnings"), list):
            mismatches["warnings"] = {
                "expected": "list",
                "actual": type(decision.get("warnings")).__name__,
            }
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
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ApprovalError("Stage3 approval-required protocol_id mismatch")
        if (
            payload.get("kind") != "stage3_approval_required"
            or payload.get("approved") is not False
        ):
            raise ApprovalError("invalid Stage3 approval-required marker")
        recorded_bindings = payload.get("bindings")
        if not isinstance(recorded_bindings, Mapping):
            raise ApprovalError("approval-required bindings are missing")
        current_bindings = self._hash_bindings()
        if len(recorded_bindings) != 22 or len(current_bindings) != 22:
            raise ApprovalError(
                "Stage3 approval requires exactly 22 frozen bindings: "
                f"recorded={len(recorded_bindings)}, current={len(current_bindings)}"
            )
        if dict(recorded_bindings) != current_bindings:
            differences = _mapping_differences(recorded_bindings, current_bindings)
            raise ApprovalError(
                f"frozen Stage2/config/manifest hashes changed: {differences}"
            )
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
