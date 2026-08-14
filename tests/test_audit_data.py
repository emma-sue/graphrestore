from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_data import run_audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frozen_audit():
    return run_audit(ROOT / "configs" / "resolved_paths.yaml")


def test_frozen_v71_data_identity_audit_passes(frozen_audit) -> None:
    failures = [
        f"{check.name}: {check.detail}"
        for check in frozen_audit.checks
        if check.status == "FAIL"
    ]
    assert not failures, failures
    assert frozen_audit.passed


def test_audit_proves_training_split_and_source_boundary(frozen_audit) -> None:
    boundary = frozen_audit.facts["data_boundary"]
    assert boundary["training_groups"] == ["single", "A"]
    assert boundary["group_b_or_c_training_rows"] == 0
    assert frozen_audit.facts["primary"]["all"]["rows"] == 16000
    assert frozen_audit.facts["primary"]["all"]["forbidden_reference_count"] == 0


def test_mio100_audit_never_opens_image_content(frozen_audit) -> None:
    boundary = frozen_audit.facts["data_boundary"]
    assert boundary["mio100_formal_image_files_opened"] == 0
    assert boundary["mio100_exploration_rows_read"] == 0
    for name, facts in frozen_audit.facts["mio100"].items():
        assert facts["image_files_opened"] == 0, name


def test_parent_checkpoint_and_upstream_commits_are_locked(frozen_audit) -> None:
    repositories = frozen_audit.facts["repositories"]
    assert repositories["agenticir"]["commit"] == "9640a291480dee3ba8f2974125d4ee9e3440f3d6"
    assert repositories["mioir"]["commit"] == "4d5f6ca0235cf2c307319673242d5722ee35d73f"
    assert frozen_audit.facts["stage_a_parent"]["checkpoint_sha256"] == (
        "66e056ff3537ea99416aeb119173e90fbcafc9e9f809db169ef7381cc93f77b8"
    )
