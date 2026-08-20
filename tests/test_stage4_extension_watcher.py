from __future__ import annotations

import hashlib
from pathlib import Path
import stat
from typing import Any

import pytest
import torch

from scripts import migrate_stage4_extension_provenance as migration
from scripts import watch_stage4_conditional_extension as watcher_module
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, load_json


def _root(tmp_path: Path) -> Path:
    root = (tmp_path / "project").resolve()
    for relative in (
        "artifacts/approvals",
        "artifacts/checkpoints/stage4",
        "artifacts/metrics",
        "artifacts/migrations",
        "artifacts/orchestration",
        "artifacts/logs",
        "configs",
        "reports",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _boundary(
    *,
    decision: str,
    delta: str,
    conditional_sha256: str = "a" * 64,
    stable_hashes: dict[str, str] | None = None,
) -> watcher_module.CommittedBoundary:
    return watcher_module.CommittedBoundary(
        gate=watcher_module.DecimalGate(
            lhs_decimal="24.20",
            rhs_decimal="24.00",
            delta_decimal=delta,
            decision=decision,
        ),
        conditional_sha256=conditional_sha256,
        stable_hashes={} if stable_hashes is None else stable_hashes,
    )


def _write_calibration_doorbell(root: Path, *, steps: tuple[int, ...]) -> Path:
    path = root / "artifacts/metrics/stage4_calibration_history.csv"
    header = ",".join(migration.CALIBRATION_COLUMNS)
    rows = [header]
    for step in steps:
        rows.append(",".join((str(step), *("0" for _ in header.split(",")[1:]))))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_decimal_gate_uses_exact_80_digit_boundary() -> None:
    equal = watcher_module.evaluate_decimal_gate("24.20", "24.00")
    assert equal.delta_decimal == "0.20"
    assert equal.decision == migration.DECISION_ACTIVATE

    below = watcher_module.evaluate_decimal_gate(
        "24.19999999999999999999999999999999999999999999999999999999999999999999999999",
        "24.00",
    )
    assert below.decision == migration.DECISION_DO_NOT_EXTEND
    assert below.delta_decimal.startswith("0.199999999999999999999999999")


def test_do_not_extend_publishes_immutable_empty_snapshot_receipt_without_signal(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    conditional = root / "artifacts/approvals" / migration.CONDITIONAL_NAME
    atomic_write_json(conditional, {"fixture": "conditional"})
    conditional.chmod(0o444)
    conditional_sha = sha256_file(conditional)
    signals: list[tuple[int, int]] = []

    watcher = watcher_module.Stage4ConditionalExtensionWatcher(
        root,
        signal_sender=lambda pid, sig: signals.append((pid, sig)),
    )
    boundary = _boundary(
        decision=migration.DECISION_DO_NOT_EXTEND,
        delta="0.19",
        conditional_sha256=conditional_sha,
        stable_hashes={"conditional": conditional_sha},
    )
    result = watcher.handle_boundary(
        boundary,
        execute=True,
        confirmation_token=watcher_module.WATCHER_CONFIRMATION_TOKEN,
    )

    gate_path = root / "artifacts/approvals" / migration.GATE_NAME
    gate = load_json(gate_path)
    assert result["status"] == "DO_NOT_EXTEND_PUBLISHED"
    assert result["trainer_signalled"] is False
    assert result["original_pipeline_left_running"] is True
    assert result["resume_authorized"] is False
    assert signals == []
    assert gate["decision"] == migration.DECISION_DO_NOT_EXTEND
    assert gate["observed_delta_decimal"] == "0.19"
    assert gate["snapshots"] == {}
    assert stat.S_IMODE(gate_path.stat().st_mode) == 0o444
    assert result["gate_receipt_sha256"] == sha256_file(gate_path)
    assert not (root / "artifacts/migrations" / migration.BACKUP_DIR_NAME).exists()


def test_activate_requires_execute_and_exact_confirmation_before_signal_or_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    signals: list[tuple[int, int]] = []
    activations: list[watcher_module.CommittedBoundary] = []
    watcher = watcher_module.Stage4ConditionalExtensionWatcher(
        root,
        signal_sender=lambda pid, sig: signals.append((pid, sig)),
    )
    boundary = _boundary(decision=migration.DECISION_ACTIVATE, delta="0.20")

    def fake_activate(
        value: watcher_module.CommittedBoundary,
    ) -> dict[str, Any]:
        activations.append(value)
        return {"status": "EXTENSION_STARTED"}

    monkeypatch.setattr(watcher, "_activate_extension", fake_activate)
    dry = watcher.handle_boundary(boundary, execute=False, confirmation_token=None)
    assert dry["status"] == "DRY_RUN"
    assert dry["would_signal_trainer"] is True
    assert activations == [] and signals == []

    with pytest.raises(
        watcher_module.Stage4ExtensionWatcherError,
        match="exact watcher confirmation token",
    ):
        watcher.handle_boundary(
            boundary, execute=True, confirmation_token="wrong-token"
        )
    assert activations == [] and signals == []

    result = watcher.handle_boundary(
        boundary,
        execute=True,
        confirmation_token=watcher_module.WATCHER_CONFIRMATION_TOKEN,
    )
    assert result == {"status": "EXTENSION_STARTED"}
    assert activations == [boundary]
    assert signals == []


def test_pending_step_40000_is_not_a_commit_and_never_reaches_gate_or_signal(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    _write_calibration_doorbell(root, steps=(migration.BASE_STEP,))
    last = root / "artifacts/checkpoints/stage4/last.pth"
    torch.save(
        {
            "step": migration.BASE_STEP,
            "pending_validation_step": migration.BASE_STEP,
        },
        last,
    )
    signals: list[tuple[int, int]] = []
    watcher = watcher_module.Stage4ConditionalExtensionWatcher(
        root,
        signal_sender=lambda pid, sig: signals.append((pid, sig)),
    )

    with pytest.raises(
        watcher_module.Stage4BoundaryNotReady,
        match="still pending",
    ):
        watcher.inspect_committed_boundary()
    assert signals == []
    assert not (root / "artifacts/approvals" / migration.GATE_NAME).exists()


def test_pre_40000_sidecar_doorbell_never_loads_the_large_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    _write_calibration_doorbell(root, steps=(32_000, 36_000))
    loads: list[Path] = []

    def forbidden_checkpoint_load(path: Path) -> dict[str, Any]:
        loads.append(path)
        pytest.fail("pre-40k polling must not deserialize last.pth")

    monkeypatch.setattr(migration, "_load_checkpoint", forbidden_checkpoint_load)
    watcher = watcher_module.Stage4ConditionalExtensionWatcher(root)
    with pytest.raises(
        watcher_module.Stage4BoundaryNotReady,
        match="no canonical step-40000 row",
    ):
        watcher.inspect_committed_boundary()
    assert loads == []


def test_preexisting_gate_phase_is_fail_closed_without_load_signal_or_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    gate = root / "artifacts/approvals" / migration.GATE_NAME
    atomic_write_json(gate, {"decision": migration.DECISION_ACTIVATE})
    original_sha = sha256_file(gate)
    signals: list[tuple[int, int]] = []

    def forbidden_checkpoint_load(path: Path) -> dict[str, Any]:
        pytest.fail(f"phase discovery must precede checkpoint loading: {path}")

    monkeypatch.setattr(migration, "_load_checkpoint", forbidden_checkpoint_load)
    watcher = watcher_module.Stage4ConditionalExtensionWatcher(
        root,
        signal_sender=lambda pid, sig: signals.append((pid, sig)),
    )
    with pytest.raises(
        watcher_module.Stage4ExtensionWatcherError,
        match="pre-existing conditional-extension phase artifacts",
    ):
        watcher.inspect_committed_boundary()
    assert signals == []
    assert gate.is_file()
    assert sha256_file(gate) == original_sha


def test_gate_cli_builder_carries_the_same_eleven_exact_sha256_values(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    watcher = watcher_module.Stage4ConditionalExtensionWatcher(root)
    keys = (
        "conditional",
        "run_contract",
        "last_checkpoint",
        "best_checkpoint",
        "calibration_history",
        "validation_latest",
        "report",
        "train_log",
        "state",
        "pipeline_log",
        "config",
    )
    hashes = {
        key: hashlib.sha256(f"watcher:{key}".encode()).hexdigest() for key in keys
    }
    dry = watcher._gate_command(
        python_executable="/root/miniconda3/bin/python",
        hashes=hashes,
        execute=False,
    )
    execute = watcher._gate_command(
        python_executable="/root/miniconda3/bin/python",
        hashes=hashes,
        execute=True,
    )
    for digest in hashes.values():
        assert dry.count(digest) == 1
        assert execute.count(digest) == 1
    assert "--execute" not in dry
    assert execute[-3:] == (
        "--execute",
        "--confirmation-token",
        migration.GATE_CONFIRMATION_TOKEN,
    )


def test_execute_token_is_rejected_by_cli_before_any_boundary_read(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _root(tmp_path)
    code = watcher_module.main(
        [
            "--project-root",
            str(root),
            "--once",
            "--execute",
            "--confirmation-token",
            "wrong-token",
        ]
    )
    assert code == 2
    assert "exact watcher confirmation token" in capsys.readouterr().err
