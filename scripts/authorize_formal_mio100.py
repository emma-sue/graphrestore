#!/usr/bin/env python3
"""Publish the two immutable gates for one formal MiO100 evaluation.

This controller is deliberately standard-library-only.  Phase ``inventory``
streams the frozen manifest's referenced files for SHA256/stat identity without
decoding them.  Phase ``approval`` revalidates that inventory and publishes the
exact 28-binding one-shot authorization.  Neither phase may run while a GPU
compute process exists, and neither phase overwrites an artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.formal_inventory import (  # noqa: E402
    FORMAL_APPROVAL_PATH,
    FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    FORMAL_AUTHORIZATION_PROTOCOL_SHA256,
    FORMAL_DATA_INVENTORY_PATH,
    FORMAL_OUTPUT_ROOT,
    FormalInventoryError,
    assert_no_gpu_compute_processes,
    assert_standard_library_only,
    authorization_binding_paths,
    build_formal_authorization_payload,
    build_formal_data_inventory,
    load_formal_data_inventory,
    sha256_file,
    validate_lightweight_authorization,
    validate_stage4_ready_without_torch,
    write_new_read_only_json,
)


INVENTORY_EXECUTE_TOKEN = "BUILD_FORMAL_MIO100_DATA_INVENTORY"
APPROVAL_EXECUTE_TOKEN = "PUBLISH_FORMAL_MIO100_APPROVAL"

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "manifests/mio100_test_1440_agenticir_online_canonical.jsonl"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/stage4/best_ema.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage4_graphrestore_e2e.yaml"
DEFAULT_COMPLETE = PROJECT_ROOT / "artifacts/checkpoints/stage4/complete.json"
DEFAULT_DIAGNOSTICS = PROJECT_ROOT / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.json"
DEFAULT_THRESHOLDS = PROJECT_ROOT / "artifacts/planner_thresholds.json"
DEFAULT_PAIR_PRIOR = PROJECT_ROOT / "artifacts/interaction_labels/pair_prior.json"
DEFAULT_GLOBAL_PRIORITY = (
    PROJECT_ROOT / "artifacts/interaction_labels/global_priority.json"
)


def _canonical_argument(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (PROJECT_ROOT / candidate).resolve(strict=False)
    )


def _assert_prepublication_state(
    *,
    approval_path: Path,
    output_root: Path,
) -> None:
    if approval_path.exists() or approval_path.is_symlink():
        raise FormalInventoryError(
            f"formal approval already exists; refusing replacement: {approval_path}"
        )
    if output_root.exists() or output_root.is_symlink():
        raise FormalInventoryError(
            "formal evaluator output already exists; refusing a post-result approval: "
            f"{output_root}"
        )


def run_inventory_phase(
    *,
    execute_token: str,
    manifest: Path,
    inventory_path: Path,
    authorization_protocol: Path,
    stage4_complete: Path,
    checkpoint: Path,
    diagnostics: Path,
    approval_path: Path = FORMAL_APPROVAL_PATH,
    output_root: Path = FORMAL_OUTPUT_ROOT,
    gpu_runner: Callable[..., Any] | None = None,
    inventory_builder_kwargs: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if execute_token != INVENTORY_EXECUTE_TOKEN:
        raise FormalInventoryError(
            f"inventory phase requires --execute {INVENTORY_EXECUTE_TOKEN}"
        )
    assert_standard_library_only()
    if gpu_runner is None:
        assert_no_gpu_compute_processes()
    else:
        assert_no_gpu_compute_processes(runner=gpu_runner)
    validate_stage4_ready_without_torch(
        stage4_complete,
        checkpoint_path=checkpoint,
        diagnostics_path=diagnostics,
    )
    _assert_prepublication_state(
        approval_path=approval_path,
        output_root=output_root,
    )
    if inventory_path.exists() or inventory_path.is_symlink():
        raise FormalInventoryError(
            f"formal data inventory already exists; refusing overwrite: {inventory_path}"
        )
    payload = build_formal_data_inventory(
        manifest,
        authorization_protocol=authorization_protocol,
        **dict(inventory_builder_kwargs or {}),
    )
    write_new_read_only_json(inventory_path, payload)
    validation_kwargs = dict(inventory_builder_kwargs or {})
    validation_kwargs.pop("expected_manifest_sha256", None)
    validation_kwargs.pop("expected_authorization_protocol_sha256", None)
    validation_kwargs.pop("generator_source", None)
    validated = load_formal_data_inventory(
        inventory_path,
        expected_manifest_path=manifest,
        expected_manifest_sha256=str(payload["manifest"]["sha256"]),
        expected_authorization_protocol_path=authorization_protocol,
        expected_authorization_protocol_sha256=str(
            payload["authorization_protocol"]["sha256"]
        ),
        expected_generator_source=Path(str(payload["generator_source"]["path"])),
        verify_file_bytes=True,
        **validation_kwargs,
    )
    assert_standard_library_only()
    return {
        "status": "FORMAL_DATA_INVENTORY_COMPLETE",
        "path": str(validated.path),
        "sha256": validated.sha256,
        "row_count": len(validated.rows),
        "unique_file_count": len(validated.files),
        "rows_digest": validated.rows_digest,
        "files_digest": validated.files_digest,
    }


def run_approval_phase(
    *,
    execute_token: str,
    manifest: Path,
    inventory_path: Path,
    authorization_protocol: Path,
    stage4_complete: Path,
    checkpoint: Path,
    config: Path,
    diagnostics: Path,
    thresholds: Path,
    pair_prior: Path,
    global_priority: Path,
    approval_path: Path = FORMAL_APPROVAL_PATH,
    output_root: Path = FORMAL_OUTPUT_ROOT,
    gpu_runner: Callable[..., Any] | None = None,
    binding_paths_override: Mapping[str, str | Path] | None = None,
    inventory_validation_kwargs: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if execute_token != APPROVAL_EXECUTE_TOKEN:
        raise FormalInventoryError(
            f"approval phase requires --execute {APPROVAL_EXECUTE_TOKEN}"
        )
    assert_standard_library_only()
    if gpu_runner is None:
        assert_no_gpu_compute_processes()
    else:
        assert_no_gpu_compute_processes(runner=gpu_runner)
    validate_stage4_ready_without_torch(
        stage4_complete,
        checkpoint_path=checkpoint,
        diagnostics_path=diagnostics,
    )
    _assert_prepublication_state(
        approval_path=approval_path,
        output_root=output_root,
    )
    inventory = load_formal_data_inventory(
        inventory_path,
        expected_manifest_path=manifest,
        expected_authorization_protocol_path=authorization_protocol,
        verify_file_bytes=True,
        **dict(inventory_validation_kwargs or {}),
    )
    paths = dict(
        binding_paths_override
        or authorization_binding_paths(
            PROJECT_ROOT,
            manifest=manifest,
            formal_data_inventory=inventory.path,
            checkpoint=checkpoint,
            config=config,
            stage4_complete=stage4_complete,
            thresholds=thresholds,
            pair_prior=pair_prior,
            global_priority=global_priority,
        )
    )
    if Path(paths.get("formal_data_inventory", "")) != inventory.path:
        raise FormalInventoryError(
            "approval does not bind the validated data inventory"
        )
    if Path(paths.get("formal_authorization_protocol", "")) != authorization_protocol:
        raise FormalInventoryError(
            "approval does not bind the final authorization protocol"
        )
    payload = build_formal_authorization_payload(paths)
    write_new_read_only_json(approval_path, payload)
    validated = validate_lightweight_authorization(
        approval_path,
        expected_binding_paths=paths,
    )
    if validated["bindings"]["formal_data_inventory"]["sha256"] != inventory.sha256:
        raise FormalInventoryError("published approval inventory SHA256 drifted")
    if (
        validated["bindings"]["formal_authorization_protocol"]["sha256"]
        != FORMAL_AUTHORIZATION_PROTOCOL_SHA256
    ):
        raise FormalInventoryError("published approval protocol SHA256 drifted")
    assert_standard_library_only()
    return {
        "status": "FORMAL_MIO100_APPROVED",
        "path": str(approval_path),
        "sha256": sha256_file(approval_path, field="formal approval"),
        "binding_count": len(validated["bindings"]),
        "formal_data_inventory_sha256": inventory.sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hash-only two-phase formal MiO100 authorization publisher."
    )
    parser.add_argument("--phase", choices=("inventory", "approval"), required=True)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--formal-data-inventory", type=Path, default=FORMAL_DATA_INVENTORY_PATH
    )
    parser.add_argument(
        "--authorization-protocol",
        type=Path,
        default=FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    )
    parser.add_argument("--stage4-complete", type=Path, default=DEFAULT_COMPLETE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--pair-prior", type=Path, default=DEFAULT_PAIR_PRIOR)
    parser.add_argument("--global-priority", type=Path, default=DEFAULT_GLOBAL_PRIORITY)
    parser.add_argument("--approval", type=Path, default=FORMAL_APPROVAL_PATH)
    parser.add_argument("--output-root", type=Path, default=FORMAL_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_standard_library_only()
    values = {
        name: _canonical_argument(getattr(args, name))
        for name in (
            "manifest",
            "formal_data_inventory",
            "authorization_protocol",
            "stage4_complete",
            "checkpoint",
            "config",
            "diagnostics",
            "thresholds",
            "pair_prior",
            "global_priority",
            "approval",
            "output_root",
        )
    }
    if values["formal_data_inventory"] != FORMAL_DATA_INVENTORY_PATH:
        raise FormalInventoryError(
            f"formal data inventory path is frozen to {FORMAL_DATA_INVENTORY_PATH}"
        )
    if values["authorization_protocol"] != FORMAL_AUTHORIZATION_PROTOCOL_PATH:
        raise FormalInventoryError(
            "formal authorization protocol path is frozen to "
            f"{FORMAL_AUTHORIZATION_PROTOCOL_PATH}"
        )
    if values["approval"] != FORMAL_APPROVAL_PATH:
        raise FormalInventoryError(
            f"formal approval path is frozen to {FORMAL_APPROVAL_PATH}"
        )
    if values["output_root"] != FORMAL_OUTPUT_ROOT:
        raise FormalInventoryError(
            f"formal output path is frozen to {FORMAL_OUTPUT_ROOT}"
        )
    common = {
        "execute_token": args.execute,
        "manifest": values["manifest"],
        "inventory_path": values["formal_data_inventory"],
        "authorization_protocol": values["authorization_protocol"],
        "stage4_complete": values["stage4_complete"],
        "checkpoint": values["checkpoint"],
        "diagnostics": values["diagnostics"],
        "approval_path": values["approval"],
        "output_root": values["output_root"],
    }
    if args.phase == "inventory":
        receipt = run_inventory_phase(**common)
    else:
        receipt = run_approval_phase(
            **common,
            config=values["config"],
            thresholds=values["thresholds"],
            pair_prior=values["pair_prior"],
            global_priority=values["global_priority"],
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalInventoryError as exc:
        print(f"FORMAL_MIO100_AUTHORIZATION_REJECTED: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
