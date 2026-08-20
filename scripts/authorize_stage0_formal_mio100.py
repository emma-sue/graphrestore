#!/usr/bin/env python3
"""Publish the independent one-shot Stage0 MiO100 control approval.

The command is standard-library-only and revalidates the already immutable
Stage4-era data inventory.  It never builds a second inventory, decodes an
image, imports torch/OpenCV, initializes CUDA, or writes into the Stage4
formal result tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.formal_inventory import (  # noqa: E402
    FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    FORMAL_DATA_INVENTORY_PATH,
    FormalInventoryError,
    assert_no_gpu_compute_processes,
    assert_standard_library_only,
    load_formal_data_inventory,
    sha256_file,
    write_new_read_only_json,
)
from src.evaluation.stage0_formal_inventory import (  # noqa: E402
    STAGE0_APPROVAL_PATH,
    STAGE0_AUTHORIZATION_PROTOCOL_PATH,
    STAGE0_OUTPUT_ROOT,
    STAGE0_USER_AUTHORIZATION_PROTOCOL_PATH,
    build_stage0_authorization_payload,
    stage0_authorization_binding_paths,
    validate_stage0_lightweight_authorization,
    validate_stage0_readiness_receipt_without_torch,
    validate_stage0_ready_without_torch,
    validate_stage0_user_authorization_protocol,
)


APPROVAL_EXECUTE_TOKEN = "PUBLISH_FORMAL_MIO100_STAGE0_CONTROL_APPROVAL"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "manifests/mio100_test_1440_agenticir_online_canonical.jsonl"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/stage0/best_ema.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage0_mio_stagea.yaml"
DEFAULT_SUMMARY = PROJECT_ROOT / "artifacts/checkpoints/stage0/summary.json"
DEFAULT_PRIMARY_VALIDATION = (
    PROJECT_ROOT / "artifacts/metrics/stage0_primary_val_step_060000.json"
)
DEFAULT_CALIBRATION_HISTORY = PROJECT_ROOT / "artifacts/metrics/calibration_history.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports/STAGE0_MIO_STAGEA.md"
DEFAULT_READINESS = PROJECT_ROOT / "artifacts/audits/stage0_formal_readiness.json"
DEFAULT_STAGE1_RUN_CONTRACT = (
    PROJECT_ROOT / "artifacts/checkpoints/stage1/run_contract.json"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _canonical_argument(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (PROJECT_ROOT / candidate).resolve(strict=False)
    )


def _assert_prepublication_state(*, approval_path: Path, output_root: Path) -> None:
    if approval_path.exists() or approval_path.is_symlink():
        raise FormalInventoryError(
            f"Stage0 formal approval already exists; refusing replacement: {approval_path}"
        )
    if output_root.exists() or output_root.is_symlink():
        raise FormalInventoryError(
            "Stage0 formal output already exists; refusing post-result approval: "
            f"{output_root}"
        )


def run_approval_phase(
    *,
    execute_token: str,
    manifest: Path,
    inventory_path: Path,
    inventory_protocol: Path,
    authorization_protocol: Path,
    user_authorization_protocol: Path,
    summary: Path,
    checkpoint: Path,
    config: Path,
    primary_validation: Path,
    calibration_history: Path,
    report: Path,
    readiness: Path,
    stage1_run_contract: Path = DEFAULT_STAGE1_RUN_CONTRACT,
    approval_path: Path = STAGE0_APPROVAL_PATH,
    output_root: Path = STAGE0_OUTPUT_ROOT,
    gpu_runner: Callable[..., Any] | None = None,
    binding_paths_override: Mapping[str, str | Path] | None = None,
    inventory_validation_kwargs: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if execute_token != APPROVAL_EXECUTE_TOKEN:
        raise FormalInventoryError(
            f"Stage0 approval requires --execute {APPROVAL_EXECUTE_TOKEN}"
        )
    assert_standard_library_only()
    # This evidence is authored only after an explicit user instruction.  The
    # authorizer verifies it before inspecting readiness, inventory bytes, or
    # any future output and never creates it itself.
    validate_stage0_user_authorization_protocol(user_authorization_protocol)
    if gpu_runner is None:
        assert_no_gpu_compute_processes()
    else:
        assert_no_gpu_compute_processes(runner=gpu_runner)
    validate_stage0_ready_without_torch(
        summary,
        checkpoint_path=checkpoint,
        primary_validation_path=primary_validation,
    )
    validate_stage0_readiness_receipt_without_torch(
        readiness,
        checkpoint_path=checkpoint,
        config_path=config,
        summary_path=summary,
        primary_validation_path=primary_validation,
        calibration_history_path=calibration_history,
        report_path=report,
        stage1_run_contract_path=stage1_run_contract,
    )
    _assert_prepublication_state(
        approval_path=approval_path,
        output_root=output_root,
    )
    # Reuse the exact prior inventory and its original generation protocol.
    # Prepublication checks stop at the immutable inventory/manifest metadata;
    # no formal image byte is read until after this approval exists.
    inventory = load_formal_data_inventory(
        inventory_path,
        expected_manifest_path=manifest,
        expected_authorization_protocol_path=inventory_protocol,
        # Approval must exist before the first MiO100 data-file byte read.
        # The immutable inventory JSON, manifest metadata and internal
        # row/files digests are checked here; the evaluator performs full
        # per-file byte verification only after this approval is published.
        verify_file_bytes=False,
        **dict(inventory_validation_kwargs or {}),
    )
    paths = dict(
        binding_paths_override
        or stage0_authorization_binding_paths(
            PROJECT_ROOT,
            manifest=manifest,
            formal_data_inventory=inventory.path,
            checkpoint=checkpoint,
            config=config,
            summary=summary,
            primary_validation=primary_validation,
            calibration_history=calibration_history,
            report=report,
            readiness=readiness,
            stage1_run_contract=stage1_run_contract,
            authorization_protocol=authorization_protocol,
            user_authorization_protocol=user_authorization_protocol,
        )
    )
    if Path(paths.get("formal_data_inventory", "")) != inventory.path:
        raise FormalInventoryError(
            "Stage0 approval does not bind the validated shared inventory"
        )
    if Path(paths.get("inventory_origin_protocol", "")) != inventory_protocol:
        raise FormalInventoryError(
            "Stage0 approval does not bind the shared inventory protocol"
        )
    if Path(paths.get("stage0_control_protocol", "")) != (authorization_protocol):
        raise FormalInventoryError("Stage0 control protocol path drifted")
    if Path(paths.get("stage0_user_authorization_protocol", "")) != (
        user_authorization_protocol
    ):
        raise FormalInventoryError("Stage0 user authorization path drifted")
    if Path(paths.get("stage1_run_contract", "")) != stage1_run_contract:
        raise FormalInventoryError("Stage1 parent receipt path drifted")
    payload = build_stage0_authorization_payload(
        paths,
        approved_utc=_utc_now(),
    )
    write_new_read_only_json(approval_path, payload)
    validated = validate_stage0_lightweight_authorization(
        approval_path,
        expected_binding_paths=paths,
    )
    if validated["bindings"]["formal_data_inventory"]["sha256"] != inventory.sha256:
        raise FormalInventoryError("published Stage0 approval inventory SHA drifted")
    assert_standard_library_only()
    return {
        "status": "FORMAL_MIO100_STAGE0_CONTROL_APPROVED",
        "path": str(approval_path),
        "sha256": sha256_file(approval_path),
        "binding_count": len(validated["bindings"]),
        "formal_data_inventory_sha256": inventory.sha256,
        "output_root": str(output_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash-only publisher for the independent Stage0 MiO100 control approval"
        )
    )
    parser.add_argument("--execute", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--formal-data-inventory", type=Path, default=FORMAL_DATA_INVENTORY_PATH
    )
    parser.add_argument(
        "--inventory-protocol",
        type=Path,
        default=FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--authorization-protocol",
        type=Path,
        default=STAGE0_AUTHORIZATION_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--user-authorization-protocol",
        type=Path,
        default=STAGE0_USER_AUTHORIZATION_PROTOCOL_PATH,
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--primary-validation", type=Path, default=DEFAULT_PRIMARY_VALIDATION
    )
    parser.add_argument(
        "--calibration-history", type=Path, default=DEFAULT_CALIBRATION_HISTORY
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument(
        "--stage1-run-contract", type=Path, default=DEFAULT_STAGE1_RUN_CONTRACT
    )
    parser.add_argument("--approval", type=Path, default=STAGE0_APPROVAL_PATH)
    parser.add_argument("--output-root", type=Path, default=STAGE0_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_standard_library_only()
    values = {
        name: _canonical_argument(getattr(args, name))
        for name in (
            "manifest",
            "formal_data_inventory",
            "inventory_protocol",
            "authorization_protocol",
            "user_authorization_protocol",
            "summary",
            "checkpoint",
            "config",
            "primary_validation",
            "calibration_history",
            "report",
            "readiness",
            "stage1_run_contract",
            "approval",
            "output_root",
        )
    }
    frozen = {
        "manifest": DEFAULT_MANIFEST,
        "formal_data_inventory": FORMAL_DATA_INVENTORY_PATH,
        "inventory_protocol": FORMAL_AUTHORIZATION_PROTOCOL_PATH,
        "authorization_protocol": STAGE0_AUTHORIZATION_PROTOCOL_PATH,
        "user_authorization_protocol": STAGE0_USER_AUTHORIZATION_PROTOCOL_PATH,
        "summary": DEFAULT_SUMMARY,
        "checkpoint": DEFAULT_CHECKPOINT,
        "config": DEFAULT_CONFIG,
        "primary_validation": DEFAULT_PRIMARY_VALIDATION,
        "calibration_history": DEFAULT_CALIBRATION_HISTORY,
        "report": DEFAULT_REPORT,
        "readiness": DEFAULT_READINESS,
        "stage1_run_contract": DEFAULT_STAGE1_RUN_CONTRACT,
        "approval": STAGE0_APPROVAL_PATH,
        "output_root": STAGE0_OUTPUT_ROOT,
    }
    for name, expected in frozen.items():
        if values[name] != expected:
            raise FormalInventoryError(f"Stage0 formal {name} is frozen to {expected}")
    receipt = run_approval_phase(
        execute_token=args.execute,
        manifest=values["manifest"],
        inventory_path=values["formal_data_inventory"],
        inventory_protocol=values["inventory_protocol"],
        authorization_protocol=values["authorization_protocol"],
        user_authorization_protocol=values["user_authorization_protocol"],
        summary=values["summary"],
        checkpoint=values["checkpoint"],
        config=values["config"],
        primary_validation=values["primary_validation"],
        calibration_history=values["calibration_history"],
        report=values["report"],
        readiness=values["readiness"],
        stage1_run_contract=values["stage1_run_contract"],
        approval_path=values["approval"],
        output_root=values["output_root"],
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalInventoryError as exc:
        print(f"STAGE0_FORMAL_MIO100_AUTHORIZATION_REJECTED: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
