#!/usr/bin/env python3
"""Run the one-shot, authorization-bound formal MiO100 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.mio100 import (  # noqa: E402
    FORMAL_DATA_INVENTORY_PATH,
    FORMAL_MANIFEST_SHA256,
    FORMAL_METHOD_NAME,
    FORMAL_OUTPUT_ROOT,
    FORMAL_SHARD_COUNT,
    MIOIR_MATLAB_FUNCTIONS_SHA256,
    MiO100EvaluationError,
    assert_exclusive_gpu_process,
    autonomous_graphrestore_inference,
    bind_default_authorization_paths,
    build_formal_graphrestore,
    configure_formal_runtime,
    finalize_evaluation,
    load_and_bind_formal_data_inventory,
    load_formal_manifest,
    load_stage4_best_ema,
    prepare_run_contract,
    run_shard,
    validate_formal_authorization,
    validate_protocol_bindings,
    validate_stage4_completion,
)
from src.utils.hashing import sha256_file  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve(strict=False)
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve(strict=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Stage4 step-40000 EMA once on the exact "
            "online-canonical MiO100 1440 manifest. The independent immutable "
            "formal authorization is mandatory."
        )
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        required=True,
        help="immutable artifacts/approvals/FORMAL_MIO100_APPROVED.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "manifests/mio100_test_1440_agenticir_online_canonical.jsonl",
    )
    parser.add_argument(
        "--formal-data-inventory",
        type=Path,
        default=FORMAL_DATA_INVENTORY_PATH,
        help="immutable pre-registered native-LQ/GT byte inventory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/stage4/best_ema.pth",
    )
    parser.add_argument(
        "--stage4-complete",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/stage4/complete.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/stage4_graphrestore_e2e.yaml",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=PROJECT_ROOT / "artifacts/planner_thresholds.json",
    )
    parser.add_argument(
        "--pair-prior",
        type=Path,
        default=PROJECT_ROOT / "artifacts/interaction_labels/pair_prior.json",
    )
    parser.add_argument(
        "--global-priority",
        type=Path,
        default=PROJECT_ROOT / "artifacts/interaction_labels/global_priority.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FORMAL_OUTPUT_ROOT,
    )
    parser.add_argument("--method-name", default=FORMAL_METHOD_NAME)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=FORMAL_SHARD_COUNT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an already complete run without CUDA or inference",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _project_path(args.manifest)
    formal_data_inventory = _project_path(args.formal_data_inventory)
    checkpoint_path = _project_path(args.checkpoint)
    complete_path = _project_path(args.stage4_complete)
    config_path = _project_path(args.config)
    thresholds_path = _project_path(args.thresholds)
    pair_prior_path = _project_path(args.pair_prior)
    priority_path = _project_path(args.global_priority)
    authorization_path = _project_path(args.authorization)
    output_root = _project_path(args.output_root)
    if output_root != FORMAL_OUTPUT_ROOT.resolve():
        raise MiO100EvaluationError(
            f"formal output_root is frozen to {FORMAL_OUTPUT_ROOT}"
        )
    if args.method_name != FORMAL_METHOD_NAME:
        raise MiO100EvaluationError(
            f"formal method_name is frozen to {FORMAL_METHOD_NAME}"
        )
    if args.shard_count != FORMAL_SHARD_COUNT or args.shard_index != 0:
        raise MiO100EvaluationError("this authorized formal run is frozen to shard 0/1")

    # Authorization and every bound hash are checked before manifest parsing,
    # image decoding, torch checkpoint loading, or CUDA initialization.
    expected_bindings = dict(
        bind_default_authorization_paths(
            PROJECT_ROOT,
            manifest=manifest,
            formal_data_inventory=formal_data_inventory,
            checkpoint=checkpoint_path,
            config=config_path,
            stage4_complete=complete_path,
            thresholds=thresholds_path,
            pair_prior=pair_prior_path,
            global_priority=priority_path,
        )
    )
    expected_bindings.update(
        {
            "table1_scorer_module": (
                PROJECT_ROOT / "src/evaluation/agenticir_table1.py"
            ),
            "table1_scorer_cli": PROJECT_ROOT / "scripts/score_agenticir_table1.py",
        }
    )
    authorization = validate_formal_authorization(
        authorization_path,
        expected_bindings=expected_bindings,
        expected_output_root=output_root,
        expected_method_name=args.method_name,
        expected_shard_count=args.shard_count,
    )
    checkpoint_binding = authorization.bindings["stage4_checkpoint"]
    config_binding = authorization.bindings["stage4_config"]
    manifest_binding = authorization.bindings["formal_manifest"]
    if manifest_binding.sha256 != FORMAL_MANIFEST_SHA256:
        raise MiO100EvaluationError(
            "authorization does not bind the frozen full MiO100 manifest SHA"
        )
    if (
        authorization.bindings["mioir_matlab_functions"].sha256
        != MIOIR_MATLAB_FUNCTIONS_SHA256
    ):
        raise MiO100EvaluationError(
            "authorization does not bind the manifest-declared MiOIR canonicalizer"
        )
    validate_protocol_bindings(authorization)
    validate_stage4_completion(
        complete_path,
        checkpoint_sha256=checkpoint_binding.sha256,
        authorization=authorization,
    )
    if not args.verify_only:
        assert_exclusive_gpu_process(expected_pid=None)
    records = load_formal_manifest(
        manifest,
        expected_sha256=manifest_binding.sha256,
    )
    records, data_inventory = load_and_bind_formal_data_inventory(
        authorization.bindings["formal_data_inventory"].path,
        records,
        expected_sha256=authorization.bindings["formal_data_inventory"].sha256,
        manifest_path=manifest_binding.path,
        manifest_sha256=manifest_binding.sha256,
        authorization_protocol_path=authorization.bindings[
            "formal_authorization_protocol"
        ].path,
        authorization_protocol_sha256=authorization.bindings[
            "formal_authorization_protocol"
        ].sha256,
        verify_file_bytes=True,
    )
    checkpoint = load_stage4_best_ema(
        checkpoint_path,
        expected_sha256=checkpoint_binding.sha256,
        expected_config_sha256=config_binding.sha256,
    )
    if sha256_file(config_path) != config_binding.sha256:
        raise MiO100EvaluationError("Stage4 config changed after authorization")
    run = prepare_run_contract(
        authorization,
        manifest_sha256=manifest_binding.sha256,
        data_inventory_sha256=data_inventory.sha256,
        data_inventory_rows_digest=data_inventory.rows_digest,
        data_inventory_files_digest=data_inventory.files_digest,
        checkpoint_sha256=checkpoint.sha256,
        config_sha256=config_binding.sha256,
        shard_count=args.shard_count,
    )
    if args.verify_only:
        summary = finalize_evaluation(run, records, authorization=authorization)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    shard_path = run.root / "shards/shard-0000-of-0001.json"
    if shard_path.exists() or shard_path.is_symlink():
        summary = finalize_evaluation(run, records, authorization=authorization)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    model = build_formal_graphrestore(
        checkpoint,
        config_path=config_path,
        thresholds_path=thresholds_path,
        pair_prior_path=pair_prior_path,
        global_priority_path=priority_path,
    )
    authorization_before_cuda = validate_formal_authorization(
        authorization_path,
        expected_bindings=expected_bindings,
        expected_output_root=output_root,
        expected_method_name=args.method_name,
        expected_shard_count=args.shard_count,
    )
    if authorization_before_cuda.sha256 != authorization.sha256:
        raise MiO100EvaluationError("formal authorization changed before CUDA")
    assert_exclusive_gpu_process(expected_pid=None)
    configure_formal_runtime()
    if not torch.cuda.is_available():
        raise MiO100EvaluationError("formal MiO100 evaluation requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    model.eval()
    torch.cuda.synchronize(device)
    assert_exclusive_gpu_process(expected_pid=os.getpid())

    def infer(image: torch.Tensor):
        assert_exclusive_gpu_process(expected_pid=os.getpid())
        result = autonomous_graphrestore_inference(model, image, device=device)
        assert_exclusive_gpu_process(expected_pid=os.getpid())
        return result

    run_shard(
        run,
        records,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        infer=infer,
        device=device,
    )
    assert_exclusive_gpu_process(expected_pid=os.getpid())
    authorization_before_finalize = validate_formal_authorization(
        authorization_path,
        expected_bindings=expected_bindings,
        expected_output_root=output_root,
        expected_method_name=args.method_name,
        expected_shard_count=args.shard_count,
    )
    if authorization_before_finalize.sha256 != authorization.sha256:
        raise MiO100EvaluationError("formal authorization changed during evaluation")
    summary = finalize_evaluation(run, records, authorization=authorization)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MiO100EvaluationError as exc:
        print(f"FORMAL_MIO100_REJECTED: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
