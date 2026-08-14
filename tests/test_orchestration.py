from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from src.training.orchestration import (
    ApprovalError,
    ChildCommandError,
    CommandSpec,
    GraphRestoreOrchestrator,
    OrchestrationError,
    PipelineStatus,
    SubprocessCommandRunner,
    command_plan,
    recommended_tmux_argv,
)
from src.utils.hashing import sha256_file
from src.utils.io import load_json


class RecordingRunner:
    def __init__(
        self,
        callback: Callable[[CommandSpec, Path], None] | None = None,
        returncodes: dict[str, int] | None = None,
    ) -> None:
        self.commands: list[CommandSpec] = []
        self.callback = callback
        self.returncodes = returncodes or {}

    def run(self, command: CommandSpec, *, cwd: Path, log_path: Path) -> int:
        self.commands.append(command)
        if self.callback is not None:
            self.callback(command, cwd)
        return self.returncodes.get(command.name, 0)


def _write(path: Path, value: str | bytes = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "graphrestore"
    manifests = root / "fixture_manifests"
    manifest_paths: dict[str, Path] = {}
    for name in (
        "clean_train_manifest",
        "clean_val_manifest",
        "primary_train_manifest",
        "primary_val_manifest",
        "primary_all_manifest",
    ):
        path = manifests / f"{name}.jsonl"
        _write(path, '{"sample_id":"fixture"}\n')
        manifest_paths[name] = path

    resolved = "\n".join(
        f"{name}: {path}" for name, path in manifest_paths.items()
    )
    _write(root / "configs/resolved_paths.yaml", f"{resolved}\n")
    for name in (
        "stage0_mio_stagea.yaml",
        "stage1_skill_bank.yaml",
        "stage2_interaction_distill.yaml",
        "stage3_planner.yaml",
        "stage4_graphrestore_e2e.yaml",
    ):
        _write(root / "configs" / name, f"fixture: {name}\n")
    return root


def _stage_callback(command: CommandSpec, root: Path) -> None:
    if command.name == "integration_100_steps":
        directory = root / "artifacts/integration/stage0_100_steps"
        last = directory / "last.pth"
        _write(last, b"integration")
        _write(directory / "INTEGRATION_REPORT.md", "# pass\n")
        _write(directory / "micro_batch_probe.json", "{}\n")
        summary = {
            "schema_version": "graphrestore-stage0-run-v1",
            "protocol_id": "graphrestore-v7.1-agenticir-locked",
            "integration": True,
            "completed_step": 100,
            "target_step": 100,
            "finite": True,
            "runtime": {
                "crop_size": 192,
                "effective_batch": 8,
                "target_step": 100,
                "integration": True,
            },
            "peak_reserved_fraction": 0.8,
            "last_checkpoint": str(last.resolve()),
        }
        _write(directory / "summary.json", json.dumps(summary) + "\n")
    elif command.name == "stage0":
        _write(root / "artifacts/checkpoints/stage0/best_ema.pth", b"stage0")
    elif command.name == "stage1":
        _write(root / "artifacts/checkpoints/stage1/best_ema.pth", b"stage1")
    elif command.name == "effect_profiles":
        _write(
            root / "artifacts/interaction_labels/skill_effect_profiles.json",
            "{}\n",
        )
    elif command.name == "stage2_distill":
        artifact_root = root / "artifacts/interaction_labels"
        for name in (
            "interaction_train_manifest.jsonl",
            "interaction_val_manifest.jsonl",
            "group_a_relations_train.jsonl",
            "group_a_relations_val.jsonl",
        ):
            _write(artifact_root / name, '{"fixture":true}\n')
        _write(artifact_root / "pair_prior.json", "{}\n")
        _write(artifact_root / "global_priority.json", "{}\n")
        _write(root / "artifacts/metrics/stage2_interaction_summary.csv", "split,count\n")
        _write(root / "reports/INTERACTION_DISTILLATION.md", "# fixture\n")
        decision = {
            "approved": False,
            "stage1_checkpoint_sha256": sha256_file(
                root / "artifacts/checkpoints/stage1/best_ema.pth"
            ),
            "interaction_train_manifest_sha256": sha256_file(
                artifact_root / "interaction_train_manifest.jsonl"
            ),
            "interaction_val_manifest_sha256": sha256_file(
                artifact_root / "interaction_val_manifest.jsonl"
            ),
            "relation_train_sha256": sha256_file(
                artifact_root / "group_a_relations_train.jsonl"
            ),
            "relation_val_sha256": sha256_file(
                artifact_root / "group_a_relations_val.jsonl"
            ),
            "pair_prior_sha256": sha256_file(artifact_root / "pair_prior.json"),
            "global_priority_sha256": sha256_file(
                artifact_root / "global_priority.json"
            ),
            "config_sha256": sha256_file(root / "configs/stage2_interaction_distill.yaml"),
            "overall": {"non_ambiguous": 1},
            "warnings": [],
        }
        _write(
            artifact_root / "stage2_decision.json",
            json.dumps(decision, sort_keys=True) + "\n",
        )
    elif command.name == "stage3":
        _write(root / "artifacts/checkpoints/stage3/best_ema.pth", b"stage3")
    elif command.name == "stage4":
        _write(root / "artifacts/checkpoints/stage4/best_ema.pth", b"stage4")


def _pause_after_stage2(root: Path) -> tuple[GraphRestoreOrchestrator, RecordingRunner]:
    runner = RecordingRunner(_stage_callback)
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)
    orchestrator.run_integration(100)
    state = orchestrator.run_main_pipeline()
    assert state.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    return orchestrator, runner


def test_preflight_integration_pause_approval_and_stage3_stage4(tmp_path: Path) -> None:
    root = _project(tmp_path)
    runner = RecordingRunner(_stage_callback)
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)

    ready = orchestrator.run_integration(100)
    assert ready.status == PipelineStatus.READY_FOR_MAIN.value
    assert ready.integration_steps == 100
    assert runner.commands[-1].name == "integration_100_steps"
    assert "--integration_steps" in runner.commands[-1].argv
    assert runner.commands[-1].argv[-2:] == (
        "--output_dir",
        "artifacts/integration/stage0_100_steps",
    )

    paused = orchestrator.run_main_pipeline()
    assert paused.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    assert paused.gpu == "released"
    assert [item.name for item in runner.commands[-4:]] == [
        "stage0",
        "stage1",
        "effect_profiles",
        "stage2_distill",
    ]
    assert orchestrator.paths.approval_required.is_file()
    assert not orchestrator.paths.approval_granted.exists()
    required = load_json(orchestrator.paths.approval_required)
    assert required["approved"] is False
    assert required["stage2_decision"]["sha256"] == sha256_file(
        orchestrator.paths.stage2_decision
    )

    command_count_at_pause = len(runner.commands)
    complete = orchestrator.approve_and_resume_stage3(
        approve_stage3=True,
        resume_from_stage3=True,
    )
    assert [item.name for item in runner.commands[command_count_at_pause:]] == [
        "stage3",
        "stage4",
    ]
    assert complete.status == (
        PipelineStatus.STAGE4_COMPLETE_AWAITING_FORMAL_TEST_AUTHORIZATION.value
    )
    assert complete.gpu == "released"
    assert complete.next_command == "await_explicit_user_authorization_for_formal_mio100"
    approval = load_json(orchestrator.paths.approval_granted)
    assert approval["approved"] is True
    assert approval["approved_utc"].endswith("Z")
    assert approval["stage2_decision_sha256"] == paused.stage2_decision_sha256
    assert approval["approval_required_sha256"] == sha256_file(
        orchestrator.paths.approval_required
    )


def test_stage3_refuses_missing_flag_or_changed_hash(tmp_path: Path) -> None:
    root = _project(tmp_path)
    orchestrator, runner = _pause_after_stage2(root)

    with pytest.raises(ApprovalError, match="both explicit flags"):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=False,
        )

    _write(root / "configs/stage2_interaction_distill.yaml", "fixture: changed\n")
    command_count = len(runner.commands)
    with pytest.raises(ApprovalError, match="hashes changed"):
        orchestrator.approve_and_resume_stage3(
            approve_stage3=True,
            resume_from_stage3=True,
        )
    assert len(runner.commands) == command_count
    assert not orchestrator.paths.approval_granted.exists()
    state = orchestrator.load_state()
    assert state.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    assert state.gpu == "released"
    assert "hashes changed" in (state.last_error or "")


@pytest.mark.parametrize(
    "field",
    (
        "relation_train_sha256",
        "relation_val_sha256",
        "pair_prior_sha256",
        "global_priority_sha256",
        "config_sha256",
    ),
)
def test_stage2_pause_rejects_internal_decision_hash_drift(
    tmp_path: Path,
    field: str,
) -> None:
    root = _project(tmp_path)
    orchestrator, _ = _pause_after_stage2(root)
    decision = load_json(orchestrator.paths.stage2_decision)
    assert isinstance(decision, dict)
    decision[field] = "0" * 64
    _write(
        orchestrator.paths.stage2_decision,
        json.dumps(decision, sort_keys=True) + "\n",
    )
    with pytest.raises(OrchestrationError, match=field):
        orchestrator._create_approval_required()


def test_missing_success_artifact_marks_pipeline_failed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    def integration_only(command: CommandSpec, cwd: Path) -> None:
        if command.name == "integration_100_steps":
            _stage_callback(command, cwd)

    orchestrator = GraphRestoreOrchestrator(
        root, runner=RecordingRunner(integration_only)
    )
    orchestrator.run_integration(100)

    with pytest.raises(OrchestrationError, match="required outputs are missing"):
        orchestrator.run_main_pipeline()
    state = orchestrator.load_state()
    assert state.status == PipelineStatus.FAILED.value
    assert state.gpu == "released"


def test_preflight_hard_gates_and_plan_exclude_formal_mio100(tmp_path: Path) -> None:
    orchestrator = GraphRestoreOrchestrator(
        _project(tmp_path), runner=RecordingRunner()
    )
    mandatory = next(
        command for command in orchestrator.preflight_commands()
        if command.name == "mandatory_pytests"
    )
    assert mandatory.argv[-1] == "tests"
    for required_test in (
        "tests/test_online_canonical_manifest_semantics.py",
        "tests/test_ambiguous_relation_partial_label.py",
        "tests/test_compile_once_per_sample.py",
        "tests/test_checkpoint_resume.py",
    ):
        assert (Path(__file__).resolve().parents[1] / required_test).is_file()
    preflight_names = {
        command.name for command in orchestrator.preflight_commands()
    }
    assert "audit_degradation_parity" in preflight_names
    assert "probe_validation_vram" in preflight_names
    assert "profile_stage0_compile" in preflight_names

    plan = command_plan(orchestrator)
    assert plan["formal_mio100_in_automatic_pipeline"] is False
    all_argv = json.dumps(plan)
    assert "eval_mio100" not in all_argv
    assert "mio100_test" not in all_argv


def test_main_requires_exact_integration_and_integration_requires_100(tmp_path: Path) -> None:
    orchestrator = GraphRestoreOrchestrator(
        _project(tmp_path), runner=RecordingRunner()
    )
    with pytest.raises(OrchestrationError, match="exactly 100"):
        orchestrator.run_integration(99)
    with pytest.raises(OrchestrationError, match="100-step integration"):
        orchestrator.run_main_pipeline()


def test_explicit_main_resume_uses_last_checkpoint_and_skips_completed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    def integration_only(command: CommandSpec, cwd: Path) -> None:
        if command.name == "integration_100_steps":
            _stage_callback(command, cwd)

    runner = RecordingRunner(
        callback=integration_only,
        returncodes={"stage0": 9},
    )
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)
    orchestrator.run_integration(100)
    with pytest.raises(ChildCommandError):
        orchestrator.run_main_pipeline()
    assert orchestrator.load_state().status == PipelineStatus.FAILED.value
    _write(root / "artifacts/checkpoints/stage0/last.pth", b"last")
    runner.returncodes["stage0"] = 0
    runner.callback = _stage_callback
    paused = orchestrator.resume_main_pipeline()
    assert paused.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    resumed_stage0 = next(
        command for command in runner.commands if "--resume" in command.argv
    )
    assert resumed_stage0.name == "stage0"
    assert resumed_stage0.argv[-2:] == (
        "--resume",
        str(root / "artifacts/checkpoints/stage0/last.pth"),
    )


def test_explicit_main_resume_recovers_proven_stale_running_state(tmp_path: Path) -> None:
    root = _project(tmp_path)
    runner = RecordingRunner(_stage_callback)
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)
    orchestrator.run_integration(100)
    state = orchestrator.load_state()
    stage0 = orchestrator.main_stage_commands()[0][1]
    state.status = PipelineStatus.STAGE0_RUNNING.value
    state.current_stage = "STAGE0"
    state.last_command = list(stage0.argv)
    state.gpu = "owned_by_child_process"
    orchestrator._persist(state)
    _write(root / "artifacts/checkpoints/stage0/last.pth", b"last")

    paused = orchestrator.resume_main_pipeline()
    assert paused.status == PipelineStatus.PAUSED_AFTER_STAGE2.value
    resumed = next(command for command in runner.commands if "--resume" in command.argv)
    assert resumed.name == "stage0"


def test_stale_running_recovery_refuses_an_exact_live_child(tmp_path: Path) -> None:
    root = _project(tmp_path)
    orchestrator = GraphRestoreOrchestrator(
        root,
        runner=RecordingRunner(_stage_callback),
    )
    orchestrator.run_integration(100)
    command = (sys.executable, "-c", "import time; time.sleep(60)")
    process = subprocess.Popen(command, cwd=root)
    try:
        state = orchestrator.load_state()
        state.status = PipelineStatus.STAGE0_RUNNING.value
        state.current_stage = "STAGE0"
        state.last_command = list(command)
        state.gpu = "owned_by_child_process"
        orchestrator._persist(state)
        with pytest.raises(OrchestrationError, match="still live"):
            orchestrator.resume_main_pipeline()
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_subprocess_runner_preserves_child_exit_and_tmux_avoids_double_log(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    runner = SubprocessCommandRunner(output=output)
    command = CommandSpec(
        name="exit_seven",
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
    )
    assert runner.run(command, cwd=tmp_path, log_path=tmp_path / "child.log") == 7
    assert "exit=7" in (tmp_path / "child.log").read_text(encoding="utf-8")

    tmux = recommended_tmux_argv(tmp_path, sys.executable)
    assert tmux[5:9] == ("bash", "-o", "pipefail", "-c")
    assert "--run_main_pipeline" in tmux[-1]
    assert "tee" not in tmux[-1]


def test_cli_partial_approval_flag_returns_distinct_refusal_code(tmp_path: Path) -> None:
    root = _project(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/orchestrate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project_root",
            str(root),
            "--approve_stage3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == ApprovalError.exit_code
    assert "both explicit flags" in result.stderr


def test_child_failure_is_recorded_with_exact_exit_code(tmp_path: Path) -> None:
    root = _project(tmp_path)
    runner = RecordingRunner(returncodes={"audit_data": 17})
    orchestrator = GraphRestoreOrchestrator(root, runner=runner)
    with pytest.raises(ChildCommandError) as caught:
        orchestrator.run_preflight()
    assert caught.value.exit_code == 17
    state = orchestrator.load_state()
    assert state.status == PipelineStatus.PREFLIGHT_FAILED.value
    assert state.last_exit_code == 17
    assert state.gpu == "released"
