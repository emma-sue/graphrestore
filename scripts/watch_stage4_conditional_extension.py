#!/usr/bin/env python3
"""CPU-only watcher for the one-shot Stage4 40k conditional extension.

The watcher never acts on the step-36000 boundary.  It waits for the atomic
step-40000 raw checkpoint commit, recomputes the adjacent Group-A PSNR delta
from the original CSV lexemes with Decimal precision 80, and then follows one
of two fail-closed paths:

* ``delta < 0.20``: publish an immutable ``DO_NOT_EXTEND`` gate receipt with
  no snapshots and leave the original Stage4 diagnostics process untouched.
* ``delta >= 0.20``: terminate only the exact direct Stage4 trainer, prove the
  process/GPU/orchestration stop, run the migration tool's gate and provenance
  dry-run/execute pairs, and start the exact authorized resume in one tmux
  session.

Without ``--execute`` this module is read-only and never signals a process.
It is intended to be launched by an external detached supervisor whose stdout
and stderr live on the data disk; it does not daemonize itself or write logs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, NoReturn

# This must precede imports that can transitively import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts import migrate_stage4_extension_provenance as migration  # noqa: E402
from src.training.provenance import semantic_source_hashes  # noqa: E402
from src.utils.hashing import is_sha256, sha256_file  # noqa: E402
from src.utils.io import fsync_directory, load_json  # noqa: E402


WATCHER_CONFIRMATION_TOKEN = "EXECUTE_STAGE4_CONDITIONAL_EXTENSION_WATCHER"
DEFAULT_POLL_SECONDS = 30.0
MAXIMUM_POLL_SECONDS = 30.0
STOP_CONFIRMATION_TIMEOUT_SECONDS = 300.0
RESUME_CONFIRMATION_TIMEOUT_SECONDS = 180.0
SUBPROCESS_TIMEOUT_SECONDS = 1_800.0
TMUX_SESSION = "graphrestore"
CUDA_ALLOCATOR_CONF = "backend:native,expandable_segments:True"
FINALIZATION_NAME = "STAGE3_EXTENSION_REVOKED.json"
OLD_SOURCE_MAP_NAME = "old_semantic_source_sha256.json"
NEW_SOURCE_MAP_NAME = "new_semantic_source_sha256.json"


class Stage4ExtensionWatcherError(RuntimeError):
    """The watcher refused to authorize additional optimizer steps."""


class Stage4BoundaryNotReady(Stage4ExtensionWatcherError):
    """The atomically committed step-40000 boundary is not available yet."""


def _fail(message: str) -> NoReturn:
    raise Stage4ExtensionWatcherError(message)


@dataclass(frozen=True)
class DecimalGate:
    """Exact threshold evidence copied from the canonical CSV strings."""

    lhs_decimal: str
    rhs_decimal: str
    delta_decimal: str
    decision: str

    def migration_evidence(self) -> dict[str, Any]:
        return {
            "row_count": len(migration.PRE_EXTENSION_VALIDATION_STEPS),
            "steps": list(migration.PRE_EXTENSION_VALIDATION_STEPS),
            "observed_lhs_decimal": self.lhs_decimal,
            "observed_rhs_decimal": self.rhs_decimal,
            "observed_delta_decimal": self.delta_decimal,
            "threshold_decimal": migration.TRIGGER_THRESHOLD_DECIMAL,
            "decision": self.decision,
        }


@dataclass(frozen=True)
class CommittedBoundary:
    """Read-only evidence for the complete, non-pending step-40000 commit."""

    gate: DecimalGate
    conditional_sha256: str
    stable_hashes: Mapping[str, str]


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    starttime: int
    state: str
    command: tuple[str, ...]
    cwd: Path


@dataclass(frozen=True)
class LiveProcessPair:
    orchestrator: ProcessRecord
    trainer: ProcessRecord
    trainer_command: tuple[str, ...]
    orchestrator_command: tuple[str, ...]
    python_executable: str


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
SignalSender = Callable[[int, int], None]


def evaluate_decimal_gate(lhs_raw: str, rhs_raw: str) -> DecimalGate:
    """Compute the exact registered gate without binary-float conversion."""

    values: list[Decimal] = []
    for label, raw in (("step-40000", lhs_raw), ("step-36000", rhs_raw)):
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            _fail(f"{label} Group-A PSNR is not a canonical CSV string")
        try:
            value = Decimal(raw)
        except InvalidOperation as exc:
            raise Stage4ExtensionWatcherError(
                f"{label} Group-A PSNR is not Decimal"
            ) from exc
        if not value.is_finite():
            _fail(f"{label} Group-A PSNR is not finite")
        values.append(value)
    with localcontext() as context:
        context.prec = 80
        delta = values[0] - values[1]
    decision = (
        migration.DECISION_ACTIVATE
        if delta >= Decimal(migration.TRIGGER_THRESHOLD_DECIMAL)
        else migration.DECISION_DO_NOT_EXTEND
    )
    return DecimalGate(
        lhs_decimal=lhs_raw,
        rhs_decimal=rhs_raw,
        delta_decimal=str(delta),
        decision=decision,
    )


def _assert_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        _fail("Stage4 extension watcher must remain CPU-only")


def _require_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not is_sha256(value):
        _fail(f"{label} is not a lowercase SHA256")
    return value


def _atomic_create_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    """Publish a new JSON inode that is already mode 0444 at rename time."""

    if path.exists() or path.is_symlink():
        _fail(f"refusing to replace existing immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            _fail(f"immutable artifact appeared during publication: {path}")
        try:
            # The hard-link publication is same-filesystem and no-clobber: an
            # unexpected competing writer yields EEXIST instead of replacing
            # its canonical receipt or source map.
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise Stage4ExtensionWatcherError(
                f"immutable artifact appeared during publication: {path}"
            ) from exc
        temporary.unlink()
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o444
        or dict(load_json(path)) != dict(value)
    ):
        _fail(f"immutable JSON publication verification failed: {path}")
    return sha256_file(path)


def _read_proc_record(path: Path) -> ProcessRecord | None:
    try:
        command = tuple(
            item.decode("utf-8", errors="surrogateescape")
            for item in (path / "cmdline").read_bytes().split(b"\0")
            if item
        )
        raw_stat = (path / "stat").read_text(encoding="utf-8")
        closing = raw_stat.rfind(")")
        if closing < 0:
            return None
        fields = raw_stat[closing + 2 :].split()
        if len(fields) <= 19:
            return None
        state, ppid, starttime = fields[0], int(fields[1]), int(fields[19])
        cwd = (path / "cwd").resolve(strict=True)
        pid = int(path.name)
    except (
        FileNotFoundError,
        PermissionError,
        ProcessLookupError,
        OSError,
        ValueError,
    ):
        return None
    if not command or state == "Z":
        return None
    return ProcessRecord(
        pid=pid,
        ppid=ppid,
        starttime=starttime,
        state=state,
        command=command,
        cwd=cwd,
    )


def _process_records(proc_root: Path) -> tuple[ProcessRecord, ...]:
    records: list[ProcessRecord] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise Stage4ExtensionWatcherError(
            f"cannot enumerate {proc_root}: {exc}"
        ) from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        record = _read_proc_record(entry)
        if record is not None:
            records.append(record)
    return tuple(sorted(records, key=lambda item: item.pid))


def _require_step_40000_doorbell(path: Path) -> None:
    """Read the small CSV doorbell before any multi-GB checkpoint load."""

    if path.is_symlink():
        _fail("Stage4 calibration doorbell is symlinked")
    if not path.is_file():
        raise Stage4BoundaryNotReady(
            "waiting for the Stage4 calibration sidecar doorbell"
        )
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if tuple(header or ()) != migration.CALIBRATION_COLUMNS:
                _fail("Stage4 calibration doorbell has a non-canonical header")
            step_40000_rows = 0
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(migration.CALIBRATION_COLUMNS):
                    _fail(
                        "Stage4 calibration doorbell has a malformed row at "
                        f"line {line_number}"
                    )
                raw_step = row[0]
                try:
                    step = int(raw_step)
                except ValueError as exc:
                    raise Stage4ExtensionWatcherError(
                        "Stage4 calibration doorbell has a non-integer step at "
                        f"line {line_number}"
                    ) from exc
                if raw_step != str(step):
                    _fail(
                        "Stage4 calibration doorbell has a non-canonical step at "
                        f"line {line_number}"
                    )
                if step > migration.BASE_STEP:
                    _fail("Stage4 calibration doorbell advanced past step 40000")
                if step == migration.BASE_STEP:
                    step_40000_rows += 1
    except FileNotFoundError as exc:
        raise Stage4BoundaryNotReady(
            "waiting for the Stage4 calibration sidecar doorbell"
        ) from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Stage4ExtensionWatcherError(
            f"cannot read the Stage4 calibration doorbell: {exc}"
        ) from exc
    if step_40000_rows == 0:
        raise Stage4BoundaryNotReady(
            "Stage4 calibration sidecar has no canonical step-40000 row"
        )
    if step_40000_rows != 1:
        _fail("Stage4 calibration doorbell has duplicate step-40000 rows")


class Stage4ConditionalExtensionWatcher:
    """Read, decide, and optionally execute the exact conditional handoff."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        proc_root: str | Path = "/proc",
        signal_sender: SignalSender = os.kill,
        command_runner: CommandRunner | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < poll_seconds <= MAXIMUM_POLL_SECONDS:
            _fail("--poll-seconds must be > 0 and <= 30")
        self.project_root = migration._canonical_path(
            project_root, label="watcher project root"
        )
        self.paths = migration._resolve_paths(self.project_root)
        self.poll_seconds = float(poll_seconds)
        self.proc_root = Path(proc_root)
        self.signal_sender = signal_sender
        self.command_runner = command_runner or self._default_command_runner
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.migration_script = (
            self.project_root / "scripts/migrate_stage4_extension_provenance.py"
        )
        self.finalization_authorization = (
            self.project_root / "artifacts/approvals" / FINALIZATION_NAME
        )

    def _default_command_runner(
        self, argv: Sequence[str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            list(argv),
            cwd=self.project_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    def _run(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self.command_runner(tuple(argv), timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            raise Stage4ExtensionWatcherError(
                f"command execution failed: {shlex.join(argv)}: {exc}"
            ) from exc

    def _invoke_json(self, argv: Sequence[str]) -> Mapping[str, Any]:
        result = self._run(argv, timeout=SUBPROCESS_TIMEOUT_SECONDS)
        if result.returncode != 0:
            _fail(
                "command failed closed: "
                f"{shlex.join(argv)}; exit={result.returncode}; "
                f"stdout={result.stdout[-2000:]!r}; stderr={result.stderr[-2000:]!r}"
            )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise Stage4ExtensionWatcherError(
                f"command did not return one JSON document: {shlex.join(argv)}"
            ) from exc
        if not isinstance(value, Mapping):
            _fail(f"command returned non-object JSON: {shlex.join(argv)}")
        return value

    def inspect_committed_boundary(self) -> CommittedBoundary:
        """Return only after the complete step-40000 validation commit exists."""

        _assert_cpu_only()
        phase_artifacts = {
            "gate receipt": self.paths["gate"],
            "migration snapshot directory": self.paths["backup_dir"],
            "migration receipt": self.paths["backup_dir"] / migration.RECEIPT_NAME,
        }
        discovered = {
            label: str(path)
            for label, path in phase_artifacts.items()
            if path.exists() or path.is_symlink()
        }
        if discovered:
            _fail(
                "pre-existing conditional-extension phase artifacts require "
                f"manual audit; watcher will not rerun or delete them: {discovered}"
            )

        # The sidecar is a tiny, atomically published doorbell.  Until its
        # canonical 40k row exists, polling must not deserialize last.pth.
        _require_step_40000_doorbell(self.paths["calibration_history"])
        last_path = self.paths["last_checkpoint"]
        if last_path.is_symlink() or not last_path.is_file():
            raise Stage4BoundaryNotReady("Stage4 raw last.pth is not present")
        last = migration._load_checkpoint(last_path)
        step = last.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            _fail("Stage4 raw checkpoint has an invalid step")
        if step < migration.BASE_STEP:
            raise Stage4BoundaryNotReady(
                f"waiting for step 40000; raw checkpoint is step {step}"
            )
        if step > migration.BASE_STEP:
            _fail(f"Stage4 advanced past the authorized gate boundary: step={step}")
        if last.get("pending_validation_step") is not None:
            raise Stage4BoundaryNotReady(
                "step-40000 validation is still pending; no action is permitted"
            )
        if self.paths["gate"].exists() or self.paths["gate"].is_symlink():
            _fail("canonical Stage4 gate receipt appeared during boundary inspection")
        if self.paths["complete"].exists() or self.paths["complete"].is_symlink():
            _fail("original Stage4 already completed before conditional handoff")

        labels = (
            "run_contract",
            "last_checkpoint",
            "best_checkpoint",
            "calibration_history",
            "validation_latest",
            "report",
            "train_log",
            "config",
            "conditional",
            "instruction_protocol",
        )
        hashes = {
            label: migration._require_file(self.paths[label], label=label)
            for label in labels
        }
        conditional_sha = hashes["conditional"]
        migration._validate_conditional(self.paths, conditional_sha)
        gate_evidence = migration._validate_calibration_history(
            self.paths["calibration_history"],
            expected_sha256=hashes["calibration_history"],
        )
        exact_gate = evaluate_decimal_gate(
            str(gate_evidence["observed_lhs_decimal"]),
            str(gate_evidence["observed_rhs_decimal"]),
        )
        if gate_evidence != exact_gate.migration_evidence():
            _fail("watcher Decimal gate differs from migration gate evidence")
        migration._validate_validation_latest(
            self.paths["validation_latest"],
            expected_sha256=hashes["validation_latest"],
            gate_evidence=gate_evidence,
        )
        migration._validate_train_log(
            self.paths["train_log"], expected_sha256=hashes["train_log"]
        )

        best = migration._load_checkpoint(self.paths["best_checkpoint"])
        migration._validate_checkpoint_pair(
            last, best, expected_best_sha256=hashes["best_checkpoint"]
        )
        run = migration._mapping(load_json(self.paths["run_contract"]), field="run")
        provenance = migration._mapping(
            run.get("provenance"), field="Stage4 run provenance"
        )
        if (
            run.get("schema_version") != migration.STAGE4_RUNTIME_SCHEMA
            or provenance != last.get("provenance")
            or provenance != best.get("provenance")
        ):
            _fail("committed run/last/best provenance identity differs")
        old_sources = migration._mapping(
            provenance.get("semantic_source_sha256"), field="old semantic sources"
        )
        migration._validate_provenance_anchor(
            provenance, old_source_map=dict(old_sources)
        )
        report_text = self.paths["report"].read_text(encoding="utf-8")
        if "Validation step: 40000" not in report_text:
            _fail("Stage4 report is not the committed step-40000 report")
        for label, expected in hashes.items():
            migration._require_file(
                self.paths[label], label=label, expected_sha256=expected
            )
        _assert_cpu_only()
        return CommittedBoundary(
            gate=exact_gate,
            conditional_sha256=conditional_sha,
            stable_hashes=dict(hashes),
        )

    def _publish_do_not_extend(self, boundary: CommittedBoundary) -> Mapping[str, Any]:
        if boundary.gate.decision != migration.DECISION_DO_NOT_EXTEND:
            _fail("DO_NOT_EXTEND publication received an activated gate")
        conditional = self.paths["conditional"]
        migration._require_file(
            conditional,
            label="conditional authorization",
            expected_sha256=boundary.conditional_sha256,
        )
        if stat.S_IMODE(conditional.stat().st_mode) != 0o444:
            _fail("conditional authorization is no longer immutable")
        self._verify_boundary_stable(boundary)
        migrations = self.paths["backup_dir"].parent
        if migrations.is_symlink() or not migrations.is_dir():
            _fail("canonical migrations directory is missing or symlinked")
        with migration._single_writer_lock(migrations):
            if (
                self.paths["backup_dir"].exists()
                or self.paths["backup_dir"].is_symlink()
            ):
                _fail("snapshot directory exists for a DO_NOT_EXTEND decision")
            receipt = migration._build_gate_receipt(
                paths=self.paths,
                conditional_sha256=boundary.conditional_sha256,
                gate_evidence=boundary.gate.migration_evidence(),
                snapshots={},
            )
            if (
                receipt.get("decision") != migration.DECISION_DO_NOT_EXTEND
                or receipt.get("snapshots") != {}
            ):
                _fail("DO_NOT_EXTEND receipt construction drifted")
            digest = _atomic_create_immutable_json(self.paths["gate"], receipt)
        return dict(receipt) | {
            "sha256": digest,
            "resume_authorized": False,
        }

    def _verify_boundary_stable(self, boundary: CommittedBoundary) -> None:
        for label, expected in boundary.stable_hashes.items():
            migration._require_file(
                self.paths[label], label=label, expected_sha256=expected
            )

    def _tmux_sessions(self) -> tuple[str, ...]:
        result = self._run(
            ("tmux", "list-sessions", "-F", "#{session_name}"), timeout=15.0
        )
        if result.returncode != 0:
            if not result.stdout.strip():
                return ()
            _fail(f"cannot enumerate tmux sessions: {result.stderr.strip()}")
        return tuple(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )

    def _load_state(self) -> Mapping[str, Any]:
        value = load_json(self.paths["orchestration_state"])
        if not isinstance(value, Mapping):
            _fail("orchestration state is not a mapping")
        return value

    def _resolve_live_pair(self) -> LiveProcessPair:
        if self.paths["complete"].exists() or self.paths["complete"].is_symlink():
            _fail("Stage4 completed before the exact trainer could be stopped")
        state = self._load_state()
        command = state.get("last_command")
        if (
            state.get("status") != "STAGE4_RUNNING"
            or state.get("current_stage") != "STAGE4"
            or state.get("gpu") != "owned_by_child_process"
            or state.get("last_exit_code") is not None
            or not isinstance(command, list)
            or not all(isinstance(item, str) for item in command)
        ):
            _fail("orchestration is not the exact live Stage4 state")
        expected_trainer = (
            command[0],
            "scripts/train_stage4_e2e.py",
            "--config",
            "configs/stage4_graphrestore_e2e.yaml",
            "--resume",
            str(self.paths["last_checkpoint"]),
        )
        if tuple(command) != expected_trainer or not Path(command[0]).is_absolute():
            _fail("durable Stage4 trainer command is not exact")
        expected_orchestrator = (
            command[0],
            "scripts/orchestrate.py",
            "--resume_post_approval_pipeline",
            "--stage3_finalization_authorization",
            str(self.finalization_authorization),
        )
        records = _process_records(self.proc_root)
        orchestrators = tuple(
            row
            for row in records
            if row.command == expected_orchestrator and row.cwd == self.project_root
        )
        if len(orchestrators) != 1:
            _fail(
                "expected exactly one live orchestrator, found "
                f"{[row.pid for row in orchestrators]}"
            )
        direct_trainers = tuple(
            row
            for row in records
            if row.command == expected_trainer
            and row.cwd == self.project_root
            and row.ppid == orchestrators[0].pid
        )
        if len(direct_trainers) != 1:
            _fail(
                "expected exactly one direct Stage4 trainer, found "
                f"{[row.pid for row in direct_trainers]}"
            )
        if self._tmux_sessions().count(TMUX_SESSION) != 1:
            _fail("the original graphrestore tmux session is not unique")
        return LiveProcessPair(
            orchestrator=orchestrators[0],
            trainer=direct_trainers[0],
            trainer_command=expected_trainer,
            orchestrator_command=expected_orchestrator,
            python_executable=command[0],
        )

    def _gpu_pids(self) -> set[int]:
        result = self._run(
            (
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ),
            timeout=15.0,
        )
        if result.returncode != 0:
            _fail(f"cannot prove GPU release: {result.stderr.strip()}")
        pids: set[int] = set()
        for raw in result.stdout.splitlines():
            value = raw.strip()
            if not value or "No running processes" in value:
                continue
            if not value.isdigit():
                _fail(f"unexpected nvidia-smi PID row: {value!r}")
            pids.add(int(value))
        return pids

    def _stopped(self, pair: LiveProcessPair) -> bool:
        records = _process_records(self.proc_root)
        if any(
            row.command in {pair.trainer_command, pair.orchestrator_command}
            and row.cwd == self.project_root
            for row in records
        ):
            return False
        if self._gpu_pids():
            return False
        try:
            state = self._load_state()
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if not (
            state.get("status") == "FAILED"
            and state.get("current_stage") == "FAILED"
            and state.get("gpu") == "released"
            and state.get("last_exit_code") == 130
            and state.get("last_command") == list(pair.trainer_command)
        ):
            return False
        return TMUX_SESSION not in self._tmux_sessions()

    @staticmethod
    def _same_process_identity(left: ProcessRecord, right: ProcessRecord) -> bool:
        return (
            left.pid,
            left.ppid,
            left.starttime,
            left.command,
            left.cwd,
        ) == (
            right.pid,
            right.ppid,
            right.starttime,
            right.command,
            right.cwd,
        )

    def _wait_for_stopped(self, pair: LiveProcessPair) -> None:
        deadline = self.monotonic() + STOP_CONFIRMATION_TIMEOUT_SECONDS
        while True:
            if self._stopped(pair):
                return
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                _fail(
                    "timed out proving trainer/workers/GPU/orchestration release; "
                    "no migration or restart was attempted"
                )
            self.sleeper(min(self.poll_seconds, 5.0, remaining))

    def _gate_hashes(self, boundary: CommittedBoundary) -> dict[str, str]:
        labels = {
            "conditional": "conditional",
            "run_contract": "run_contract",
            "last_checkpoint": "last_checkpoint",
            "best_checkpoint": "best_checkpoint",
            "calibration_history": "calibration_history",
            "validation_latest": "validation_latest",
            "report": "report",
            "train_log": "train_log",
            "state": "orchestration_state",
            "pipeline_log": "pipeline_log",
            "config": "config",
        }
        result = {
            name: migration._require_file(self.paths[path_key], label=name)
            for name, path_key in labels.items()
        }
        if result["conditional"] != boundary.conditional_sha256:
            _fail("conditional authorization changed before gate execution")
        repeated = {
            name: migration._require_file(self.paths[path_key], label=name)
            for name, path_key in labels.items()
        }
        if repeated != result:
            _fail("one of the 11 exact gate SHA256 values changed while hashing")
        return result

    def _gate_command(
        self, *, python_executable: str, hashes: Mapping[str, str], execute: bool
    ) -> tuple[str, ...]:
        argv = [
            python_executable,
            str(self.migration_script),
            "--project-root",
            str(self.project_root),
            "gate",
        ]
        for option, key in (
            ("conditional", "conditional"),
            ("run-contract", "run_contract"),
            ("last-checkpoint", "last_checkpoint"),
            ("best-checkpoint", "best_checkpoint"),
            ("calibration-history", "calibration_history"),
            ("validation-latest", "validation_latest"),
            ("report", "report"),
            ("train-log", "train_log"),
            ("state", "state"),
            ("pipeline-log", "pipeline_log"),
            ("config", "config"),
        ):
            argv.extend((f"--expected-{option}-sha256", hashes[key]))
        if execute:
            argv.extend(
                (
                    "--execute",
                    "--confirmation-token",
                    migration.GATE_CONFIRMATION_TOKEN,
                )
            )
        return tuple(argv)

    def _publish_source_maps(self) -> tuple[Path, Path]:
        snapshot_run = (
            self.paths["backup_dir"] / migration.SNAPSHOT_FILENAMES["run_contract"]
        )
        run = migration._mapping(
            load_json(snapshot_run), field="pre-extension run-contract snapshot"
        )
        provenance = migration._mapping(
            run.get("provenance"), field="pre-migration provenance"
        )
        old_raw = migration._mapping(
            provenance.get("semantic_source_sha256"), field="old source map"
        )
        old_map = migration._validate_source_map(old_raw, field="old source map")
        new_map = migration._validate_source_map(
            semantic_source_hashes(
                self.project_root, entrypoints=migration.ENTRYPOINTS
            ),
            field="new physical source map",
        )
        old_map, new_map = migration._validate_source_transition(old_map, new_map)
        directory = self.paths["backup_dir"]
        if directory.is_symlink() or not directory.is_dir():
            _fail("gate execution did not create the canonical migration directory")
        old_path = directory / OLD_SOURCE_MAP_NAME
        new_path = directory / NEW_SOURCE_MAP_NAME
        _atomic_create_immutable_json(old_path, old_map)
        _atomic_create_immutable_json(new_path, new_map)
        return old_path, new_path

    def _migration_command(
        self,
        *,
        python_executable: str,
        conditional_sha256: str,
        gate_sha256: str,
        old_source_map: Path,
        new_source_map: Path,
        execute: bool,
    ) -> tuple[str, ...]:
        argv = [
            python_executable,
            str(self.migration_script),
            "--project-root",
            str(self.project_root),
            "migrate",
            "--expected-conditional-sha256",
            conditional_sha256,
            "--expected-gate-sha256",
            gate_sha256,
            "--old-source-map-json",
            str(old_source_map),
            "--new-source-map-json",
            str(new_source_map),
        ]
        if execute:
            argv.extend(
                (
                    "--execute",
                    "--confirmation-token",
                    migration.MIGRATION_CONFIRMATION_TOKEN,
                )
            )
        return tuple(argv)

    def _resume_command(self, *, python_executable: str) -> tuple[str, ...]:
        return (
            python_executable,
            "scripts/orchestrate.py",
            "--project_root",
            str(self.project_root),
            "--resume_post_approval_pipeline",
            "--stage3_finalization_authorization",
            str(self.finalization_authorization),
            "--stage4_extension_authorization",
            str(self.paths["gate"]),
        )

    def _launch_resume(self, *, python_executable: str) -> Mapping[str, Any]:
        if TMUX_SESSION in self._tmux_sessions():
            _fail("refusing to replace an existing graphrestore tmux session")
        records = _process_records(self.proc_root)
        if any(
            row.cwd == self.project_root
            and row.command
            and row.command[1:2]
            in (("scripts/orchestrate.py",), ("scripts/train_stage4_e2e.py",))
            for row in records
        ):
            _fail(
                "refusing extension launch while a project trainer/orchestrator lives"
            )
        resume = self._resume_command(python_executable=python_executable)
        shell_command = (
            "exec env -u CUDA_VISIBLE_DEVICES "
            f"PYTORCH_CUDA_ALLOC_CONF={shlex.quote(CUDA_ALLOCATOR_CONF)} "
            + shlex.join(resume)
        )
        launch = (
            "tmux",
            "new-session",
            "-d",
            "-s",
            TMUX_SESSION,
            "-c",
            str(self.project_root),
            shell_command,
        )
        result = self._run(launch, timeout=30.0)
        if result.returncode != 0:
            _fail(
                f"tmux extension launch failed: exit={result.returncode}; "
                f"stdout={result.stdout!r}; stderr={result.stderr!r}"
            )
        deadline = self.monotonic() + RESUME_CONFIRMATION_TIMEOUT_SECONDS
        expected_trainer = (
            python_executable,
            "scripts/train_stage4_e2e.py",
            "--config",
            "configs/stage4_graphrestore_e2e.yaml",
            "--resume",
            str(self.paths["last_checkpoint"]),
            "--extension_authorization",
            str(self.paths["gate"]),
        )
        while True:
            sessions = self._tmux_sessions()
            try:
                state = self._load_state()
            except (OSError, ValueError, json.JSONDecodeError):
                state = {}
            records = _process_records(self.proc_root)
            orchestrators = tuple(
                row
                for row in records
                if row.cwd == self.project_root and row.command == resume
            )
            trainers = tuple(
                row
                for row in records
                if row.cwd == self.project_root
                and row.command == expected_trainer
                and len(orchestrators) == 1
                and row.ppid == orchestrators[0].pid
            )
            if (
                sessions.count(TMUX_SESSION) == 1
                and len(orchestrators) == 1
                and len(trainers) == 1
                and state.get("status") == "STAGE4_RUNNING"
                and state.get("current_stage") == "STAGE4"
                and state.get("gpu") == "owned_by_child_process"
                and state.get("last_command") == list(expected_trainer)
            ):
                return {
                    "tmux_session": TMUX_SESSION,
                    "orchestrator_pid": orchestrators[0].pid,
                    "trainer_pid": trainers[0].pid,
                    "resume_command": list(resume),
                }
            if state.get("status") == "FAILED" and TMUX_SESSION not in sessions:
                _fail("authorized extension resume exited before becoming healthy")
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                _fail("timed out proving the authorized Stage4 extension launch")
            self.sleeper(min(self.poll_seconds, 5.0, remaining))

    def _activate_extension(self, boundary: CommittedBoundary) -> Mapping[str, Any]:
        self._verify_boundary_stable(boundary)
        pair = self._resolve_live_pair()
        if self.paths["complete"].exists() or self.paths["complete"].is_symlink():
            _fail("Stage4 completed during process resolution; no signal was sent")
        current = _read_proc_record(self.proc_root / str(pair.trainer.pid))
        if current is None or not self._same_process_identity(current, pair.trainer):
            _fail("trainer PID identity changed before SIGTERM; no signal was sent")
        # Deliberately signal only the exact direct trainer.  The orchestrator
        # owns fail-state persistence, and the trainer owns worker shutdown.
        self.signal_sender(pair.trainer.pid, signal.SIGTERM)
        self._wait_for_stopped(pair)

        hashes = self._gate_hashes(boundary)
        dry_gate = self._invoke_json(
            self._gate_command(
                python_executable=pair.python_executable,
                hashes=hashes,
                execute=False,
            )
        )
        if (
            dry_gate.get("status") != "DRY_RUN"
            or dry_gate.get("decision") != migration.DECISION_ACTIVATE
            or dry_gate.get("gate_evidence") != boundary.gate.migration_evidence()
        ):
            _fail("gate dry-run did not reproduce the activated watcher decision")
        gate = self._invoke_json(
            self._gate_command(
                python_executable=pair.python_executable,
                hashes=hashes,
                execute=True,
            )
        )
        if gate.get("decision") != migration.DECISION_ACTIVATE:
            _fail("published gate receipt did not activate the extension")
        gate_sha = _require_sha(gate.get("sha256"), label="published gate receipt")
        if sha256_file(self.paths["gate"]) != gate_sha:
            _fail("published gate receipt SHA256 differs from the physical file")

        old_map, new_map = self._publish_source_maps()
        dry_migration = self._invoke_json(
            self._migration_command(
                python_executable=pair.python_executable,
                conditional_sha256=boundary.conditional_sha256,
                gate_sha256=gate_sha,
                old_source_map=old_map,
                new_source_map=new_map,
                execute=False,
            )
        )
        if dry_migration.get("status") != "DRY_RUN":
            _fail("provenance migration dry-run did not return DRY_RUN")
        migrated = self._invoke_json(
            self._migration_command(
                python_executable=pair.python_executable,
                conditional_sha256=boundary.conditional_sha256,
                gate_sha256=gate_sha,
                old_source_map=old_map,
                new_source_map=new_map,
                execute=True,
            )
        )
        if migrated.get("status") != "COMPLETE":
            _fail("provenance migration did not complete")
        migration_receipt = self.paths["backup_dir"] / migration.RECEIPT_NAME
        receipt = load_json(migration_receipt)
        if not isinstance(receipt, Mapping) or receipt.get("status") != "COMPLETE":
            _fail("physical provenance migration receipt is not COMPLETE")
        launch = self._launch_resume(python_executable=pair.python_executable)
        _assert_cpu_only()
        return {
            "status": "EXTENSION_STARTED",
            "decision": migration.DECISION_ACTIVATE,
            "delta_decimal": boundary.gate.delta_decimal,
            "gate_receipt_sha256": gate_sha,
            "migration_receipt": str(migration_receipt),
            "migration_receipt_sha256": sha256_file(migration_receipt),
            "old_source_map": str(old_map),
            "new_source_map": str(new_map),
            **launch,
        }

    def handle_boundary(
        self,
        boundary: CommittedBoundary,
        *,
        execute: bool,
        confirmation_token: str | None,
    ) -> Mapping[str, Any]:
        """Apply no mutation unless both execution controls are exact."""

        _assert_cpu_only()
        if not execute:
            return {
                "status": "DRY_RUN",
                "decision": boundary.gate.decision,
                "lhs_decimal": boundary.gate.lhs_decimal,
                "rhs_decimal": boundary.gate.rhs_decimal,
                "delta_decimal": boundary.gate.delta_decimal,
                "would_signal_trainer": (
                    boundary.gate.decision == migration.DECISION_ACTIVATE
                ),
                "would_publish_do_not_extend": (
                    boundary.gate.decision == migration.DECISION_DO_NOT_EXTEND
                ),
            }
        if confirmation_token != WATCHER_CONFIRMATION_TOKEN:
            _fail("--execute requires the exact watcher confirmation token")
        if boundary.gate.decision == migration.DECISION_DO_NOT_EXTEND:
            receipt = self._publish_do_not_extend(boundary)
            return {
                "status": "DO_NOT_EXTEND_PUBLISHED",
                "decision": migration.DECISION_DO_NOT_EXTEND,
                "delta_decimal": boundary.gate.delta_decimal,
                "gate_receipt": str(self.paths["gate"]),
                "gate_receipt_sha256": receipt["sha256"],
                "snapshots": {},
                "trainer_signalled": False,
                "original_pipeline_left_running": True,
                "resume_authorized": False,
            }
        if boundary.gate.decision != migration.DECISION_ACTIVATE:
            _fail(f"unknown Stage4 extension decision: {boundary.gate.decision!r}")
        return self._activate_extension(boundary)

    def run(
        self,
        *,
        once: bool,
        execute: bool,
        confirmation_token: str | None,
    ) -> Mapping[str, Any]:
        while True:
            try:
                boundary = self.inspect_committed_boundary()
            except Stage4BoundaryNotReady as exc:
                waiting = {"status": "WAITING", "reason": str(exc)}
                if once:
                    return waiting
                print(
                    json.dumps(waiting, ensure_ascii=False, sort_keys=True),
                    flush=True,
                )
                self.sleeper(self.poll_seconds)
                continue
            return self.handle_boundary(
                boundary,
                execute=execute,
                confirmation_token=confirmation_token,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--once",
        action="store_true",
        help="inspect once and exit instead of polling until the 40k commit",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="poll interval in (0, 30], default 30 seconds",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="permit the registered receipt/signal/migration/resume actions",
    )
    parser.add_argument(
        "--confirmation-token",
        help=f"required with --execute: {WATCHER_CONFIRMATION_TOKEN}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if (
            arguments.execute
            and arguments.confirmation_token != WATCHER_CONFIRMATION_TOKEN
        ):
            _fail("--execute requires the exact watcher confirmation token")
        watcher = Stage4ConditionalExtensionWatcher(
            arguments.project_root, poll_seconds=arguments.poll_seconds
        )
        result = watcher.run(
            once=arguments.once,
            execute=arguments.execute,
            confirmation_token=arguments.confirmation_token,
        )
    except (
        Stage4ExtensionWatcherError,
        migration.Stage4ExtensionMigrationError,
    ) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
