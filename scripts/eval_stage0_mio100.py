#!/usr/bin/env python3
"""Authorize-gated formal MiO100 inference for frozen Stage0 step-60000 EMA."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import mio100 as shared  # noqa: E402
from src.evaluation.formal_inventory import (  # noqa: E402
    FORMAL_AUTHORIZATION_PROTOCOL_PATH,
    FORMAL_DATA_INVENTORY_PATH,
    FormalInventoryError,
)
from src.evaluation.stage0_formal import (  # noqa: E402
    bind_default_stage0_authorization_paths,
    build_formal_stage0,
    load_stage0_best_ema,
    prepare_stage0_run_contract,
    publish_stage0_readiness,
    stage0_formal_inference,
    validate_stage0_evaluator_complete,
    validate_stage0_formal_authorization,
    validate_stage0_protocol_bindings,
)
from src.evaluation.stage0_formal_inventory import (  # noqa: E402
    STAGE0_APPROVAL_PATH,
    STAGE0_METHOD_NAME,
    STAGE0_OUTPUT_ROOT,
    validate_stage0_readiness_receipt_without_torch,
)
from src.utils.hashing import sha256_file  # noqa: E402


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
DEFAULT_STAGE1_RUN_CONTRACT = (
    PROJECT_ROOT / "artifacts/checkpoints/stage1/run_contract.json"
)
DEFAULT_READINESS = PROJECT_ROOT / "artifacts/audits/stage0_formal_readiness.json"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve(strict=False)
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve(strict=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent Stage0 step-60000 formal MiO100 control"
    )
    parser.add_argument("--authorization", type=Path, default=STAGE0_APPROVAL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--formal-data-inventory", type=Path, default=FORMAL_DATA_INVENTORY_PATH
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--primary-validation", type=Path, default=DEFAULT_PRIMARY_VALIDATION
    )
    parser.add_argument(
        "--calibration-history", type=Path, default=DEFAULT_CALIBRATION_HISTORY
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--stage1-run-contract", type=Path, default=DEFAULT_STAGE1_RUN_CONTRACT
    )
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-root", type=Path, default=STAGE0_OUTPUT_ROOT)
    parser.add_argument("--method-name", default=STAGE0_METHOD_NAME)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--prepare-readiness",
        action="store_true",
        help="CPU-audit and publish only the immutable readiness receipt",
    )
    parser.add_argument(
        "--verify-readiness",
        action="store_true",
        help="verify an existing readiness receipt without CUDA",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify/finalize an already completed authorized inference",
    )
    return parser


def _readiness_kwargs(values: dict[str, Path]) -> dict[str, Path]:
    return {
        "checkpoint_path": values["checkpoint"],
        "config_path": values["config"],
        "summary_path": values["summary"],
        "primary_validation_path": values["primary_validation"],
        "calibration_history_path": values["calibration_history"],
        "report_path": values["report"],
        "stage1_run_contract_path": values["stage1_run_contract"],
    }


def _expected_bindings(values: dict[str, Path]) -> Mapping[str, Path]:
    return bind_default_stage0_authorization_paths(
        PROJECT_ROOT,
        manifest=values["manifest"],
        formal_data_inventory=values["formal_data_inventory"],
        checkpoint=values["checkpoint"],
        config=values["config"],
        summary=values["summary"],
        primary_validation=values["primary_validation"],
        calibration_history=values["calibration_history"],
        report=values["report"],
        readiness=values["readiness"],
        stage1_run_contract=values["stage1_run_contract"],
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = {
        name: _project_path(getattr(args, name))
        for name in (
            "authorization",
            "manifest",
            "formal_data_inventory",
            "checkpoint",
            "config",
            "summary",
            "primary_validation",
            "calibration_history",
            "report",
            "stage1_run_contract",
            "readiness",
            "output_root",
        )
    }
    frozen = {
        "authorization": STAGE0_APPROVAL_PATH,
        "manifest": DEFAULT_MANIFEST,
        "formal_data_inventory": FORMAL_DATA_INVENTORY_PATH,
        "checkpoint": DEFAULT_CHECKPOINT,
        "config": DEFAULT_CONFIG,
        "summary": DEFAULT_SUMMARY,
        "primary_validation": DEFAULT_PRIMARY_VALIDATION,
        "calibration_history": DEFAULT_CALIBRATION_HISTORY,
        "report": DEFAULT_REPORT,
        "stage1_run_contract": DEFAULT_STAGE1_RUN_CONTRACT,
        "readiness": DEFAULT_READINESS,
        "output_root": STAGE0_OUTPUT_ROOT,
    }
    for name, expected in frozen.items():
        if values[name] != expected:
            raise shared.MiO100EvaluationError(
                f"Stage0 formal {name} is frozen to {expected}"
            )
    if args.method_name != STAGE0_METHOD_NAME:
        raise shared.MiO100EvaluationError("Stage0 formal method name is frozen")
    if args.shard_index != 0 or args.shard_count != 1:
        raise shared.MiO100EvaluationError("Stage0 formal run is shard 0/1")
    if sum((args.prepare_readiness, args.verify_readiness, args.verify_only)) > 1:
        raise shared.MiO100EvaluationError("choose only one readiness/verify mode")
    if args.prepare_readiness:
        if torch.cuda.is_initialized():
            raise shared.MiO100EvaluationError("CUDA initialized before readiness")
        receipt = publish_stage0_readiness(
            values["readiness"], **_readiness_kwargs(values)
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.verify_readiness:
        try:
            receipt = validate_stage0_readiness_receipt_without_torch(
                values["readiness"], **_readiness_kwargs(values)
            )
        except FormalInventoryError as exc:
            raise shared.MiO100EvaluationError(str(exc)) from exc
        if torch.cuda.is_initialized():
            raise shared.MiO100EvaluationError(
                "readiness verification initialized CUDA"
            )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    expected_bindings = _expected_bindings(values)

    # Authorization and all source/checkpoint/result hashes precede manifest
    # parsing, image reads, torch checkpoint loading, or CUDA initialization.
    authorization = validate_stage0_formal_authorization(
        values["authorization"], expected_bindings=expected_bindings
    )
    validate_stage0_protocol_bindings(authorization)
    try:
        validate_stage0_readiness_receipt_without_torch(
            values["readiness"], **_readiness_kwargs(values)
        )
    except FormalInventoryError as exc:
        raise shared.MiO100EvaluationError(str(exc)) from exc
    if args.verify_only and (STAGE0_OUTPUT_ROOT / "complete.json").is_file():
        completion = validate_stage0_evaluator_complete(
            authorization_path=values["authorization"],
            expected_bindings=expected_bindings,
            verify_data_files=True,
        )
        print(
            json.dumps(
                completion.evidence, ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        return 0
    if not args.verify_only:
        shared.assert_exclusive_gpu_process(expected_pid=None)
    records = shared.load_formal_manifest(
        values["manifest"],
        expected_sha256=authorization.bindings["formal_manifest"].sha256,
    )
    records, inventory = shared.load_and_bind_formal_data_inventory(
        values["formal_data_inventory"],
        records,
        expected_sha256=authorization.bindings["formal_data_inventory"].sha256,
        manifest_path=authorization.bindings["formal_manifest"].path,
        manifest_sha256=authorization.bindings["formal_manifest"].sha256,
        authorization_protocol_path=FORMAL_AUTHORIZATION_PROTOCOL_PATH,
        authorization_protocol_sha256=authorization.bindings[
            "inventory_origin_protocol"
        ].sha256,
        verify_file_bytes=True,
    )
    checkpoint = load_stage0_best_ema(
        values["checkpoint"],
        expected_sha256=authorization.bindings["stage0_checkpoint"].sha256,
        expected_config_sha256=authorization.bindings["stage0_config"].sha256,
        expected_config_path=values["config"],
    )
    if sha256_file(values["config"]) != authorization.bindings["stage0_config"].sha256:
        raise shared.MiO100EvaluationError("Stage0 config changed after authorization")
    run = prepare_stage0_run_contract(
        authorization,
        manifest_sha256=authorization.bindings["formal_manifest"].sha256,
        data_inventory_sha256=inventory.sha256,
        data_inventory_rows_digest=inventory.rows_digest,
        data_inventory_files_digest=inventory.files_digest,
        checkpoint_sha256=checkpoint.sha256,
        config_sha256=authorization.bindings["stage0_config"].sha256,
        shard_count=1,
    )

    def finalize_and_verify() -> dict[str, object]:
        finalized = shared.finalize_evaluation(
            run, records, authorization=authorization
        )
        completion = validate_stage0_evaluator_complete(
            authorization_path=values["authorization"],
            expected_bindings=expected_bindings,
            verify_data_files=True,
        )
        return {
            "finalizer": dict(finalized),
            "verified_evidence": dict(completion.evidence),
        }

    if args.verify_only:
        result = finalize_and_verify()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    shard_path = run.root / "shards/shard-0000-of-0001.json"
    if shard_path.exists() or shard_path.is_symlink():
        result = finalize_and_verify()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    model = build_formal_stage0(checkpoint)
    before_cuda = validate_stage0_formal_authorization(
        values["authorization"], expected_bindings=expected_bindings
    )
    if before_cuda.sha256 != authorization.sha256:
        raise shared.MiO100EvaluationError("Stage0 authorization changed before CUDA")
    shared.assert_exclusive_gpu_process(expected_pid=None)
    shared.configure_formal_runtime()
    if not torch.cuda.is_available():
        raise shared.MiO100EvaluationError("formal Stage0 evaluation requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    shared.assert_exclusive_gpu_process(expected_pid=os.getpid())

    def infer(image: torch.Tensor) -> shared.InferenceResult:
        shared.assert_exclusive_gpu_process(expected_pid=os.getpid())
        result = stage0_formal_inference(model, image, device=device)
        shared.assert_exclusive_gpu_process(expected_pid=os.getpid())
        return result

    shared.run_shard(
        run,
        records,
        shard_index=0,
        shard_count=1,
        infer=infer,
        device=device,
    )
    shared.assert_exclusive_gpu_process(expected_pid=os.getpid())
    after = validate_stage0_formal_authorization(
        values["authorization"], expected_bindings=expected_bindings
    )
    if after.sha256 != authorization.sha256:
        raise shared.MiO100EvaluationError("Stage0 authorization changed during run")
    result = finalize_and_verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except shared.MiO100EvaluationError as exc:
        print(f"STAGE0_FORMAL_MIO100_REJECTED: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
