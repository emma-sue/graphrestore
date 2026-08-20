"""CPU-only recovery finalizer for the completed AgenticIR Table-1 shards.

The original scorer correctly stopped before publication when a CPU proxy was
compared to the official CUDA pyiqa values with a CPU-only, small-sample
observed maximum.  This module never imports the scorer, torch, OpenCV, pyiqa,
or an image decoder.  It validates the already immutable score transaction,
keeps the CUDA shard values authoritative, and records (without turning it
into a scientific acceptance tolerance) whether the CPU proxy and official
scorer agree at AgenticIR's frozen ``.4`` table display precision.

Production recovery is deliberately two phase.  An immutable approval binds
all pre-existing evidence, every shard, the legacy implementation, and this
recovery implementation.  A different execute token then publishes the three
standard artifacts and a terminal remediation receipt.  Existing files are
never replaced; an interrupted publication may only resume when every file
already present is byte-exact to the deterministic candidate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence


APPROVAL_SCHEMA = "graphrestore.agenticir_table1_backend_recovery_approval.v1"
RECEIPT_SCHEMA = "graphrestore.agenticir_table1_backend_recovery_receipt.v1"
SUMMARY_SCHEMA = "graphrestore.agenticir_table1_summary.v1"
COMPLETE_SCHEMA = "graphrestore.agenticir_table1_complete.v1"
INPUT_SCHEMA = "graphrestore.agenticir_table1_input.v1"
INPUT_LOCK_SCHEMA = "graphrestore.agenticir_table1_input_lock.v1"
RUN_CONTRACT_SCHEMA = "graphrestore.agenticir_table1_run_contract.v1"
SHARD_SCHEMA = "graphrestore.agenticir_table1_score_shard.v1"

APPROVAL_EXECUTE_TOKEN = "PUBLISH_TABLE1_BACKEND_RECOVERY_APPROVAL"
FINALIZE_EXECUTE_TOKEN = "FINALIZE_TABLE1_BACKEND_RECOVERY"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DISK_ROOT = Path("/root/autodl-tmp")
FORMAL_ROOT = (
    PROJECT_ROOT / "artifacts/formal_mio100/graphrestore_v7_1_stage4_step040000"
)
SCORE_ROOT = FORMAL_ROOT / "table1_scores"
APPROVAL_PATH = (
    PROJECT_ROOT / "artifacts/approvals/AGENTICIR_TABLE1_BACKEND_RECOVERY_APPROVED.json"
)
REMEDIATION_RECEIPT_PATH = (
    PROJECT_ROOT
    / "artifacts/migrations/agenticir_table1_backend_recovery/COMPLETE.json"
)

METRICS = ("psnr", "ssim", "lpips", "maniqa", "clipiqa", "musiq")
METRIC_DIRECTIONS = {
    metric: ("lower" if metric == "lpips" else "higher") for metric in METRICS
}
IDENTITY_FIELDS = (
    "sample_id",
    "group",
    "combination",
    "prediction_png",
    "prediction_sha256",
    "target_png",
    "target_sha256",
)
INPUT_KEYS = frozenset(("schema_version", *IDENTITY_FIELDS))
SCORE_KEYS = INPUT_KEYS | frozenset(METRICS)
EVALUATOR_FIELDS = (
    "sample_id",
    "group",
    "combination",
    "clean_id",
    "prediction_png",
    "prediction_sha256",
    "target_png",
    "target_sha256",
    "psnr",
    "ssim",
    "latency_ms",
    "program_levels",
    "parallel_levels",
    "active_skill_calls",
    "reentry_requests",
    "unexpected_activations",
    "precycle_graphs",
    "dropped_edges",
    "peak_reserved_fraction",
)

OFFICIAL_GROUPS: dict[str, tuple[str, ...]] = {
    "A": (
        "rain+haze",
        "motion blur+low resolution",
        "dark+noise",
        "defocus blur+jpeg compression artifact",
        "noise+jpeg compression artifact",
        "rain+low resolution",
        "motion blur+dark",
        "defocus blur+haze",
    ),
    "B": (
        "motion blur+jpeg compression artifact",
        "haze+noise",
        "defocus blur+low resolution",
        "rain+dark",
    ),
    "C": (
        "haze+motion blur+low resolution",
        "rain+noise+low resolution",
        "dark+defocus blur+jpeg compression artifact",
        "motion blur+defocus blur+noise",
    ),
}
EXPECTED_COUNTS = {
    combination: (80 if group == "A" else 100)
    for group, combinations in OFFICIAL_GROUPS.items()
    for combination in combinations
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD_NAME = re.compile(r"^shard-[0-9]{5}\.json$")
_ALLOWED_SCORE_ROOT = {
    ".worker",
    "shards",
    "input_lock.json",
    "run_contract.json",
    "per_image.csv",
    "summary.json",
    "complete.json",
}
_FORMAL_EVIDENCE_KEYS = frozenset(
    {
        "authorization",
        "evaluator_complete",
        "run_contract",
        "summary",
        "per_image",
        "table1_input",
        "checkpoint",
        "manifest",
        "formal_data_inventory",
        "metric_parity_summary",
        "predictions_digest",
    }
)


class Table1RecoveryError(RuntimeError):
    """A recovery approval, evidence, or publication contract was violated."""


@dataclass(frozen=True)
class RecoverySpec:
    groups: Mapping[str, tuple[str, ...]]
    expected_counts: Mapping[str, int]
    shard_size: int
    maximum_vram_fraction: float = 0.90

    @property
    def image_count(self) -> int:
        return sum(int(value) for value in self.expected_counts.values())

    @property
    def shard_count(self) -> int:
        return (self.image_count + self.shard_size - 1) // self.shard_size


PRODUCTION_SPEC = RecoverySpec(
    groups=OFFICIAL_GROUPS,
    expected_counts=EXPECTED_COUNTS,
    shard_size=10,
)


@dataclass(frozen=True)
class RecoveryPaths:
    confinement_root: Path
    formal_authorization: Path
    evaluator_complete: Path
    evaluator_per_image: Path
    table1_input: Path
    weights_lock: Path
    metric_parity_summary: Path
    run_contract: Path
    input_lock: Path
    worker_request: Path
    shards_dir: Path
    score_root: Path
    legacy_module: Path
    legacy_cli: Path
    failure_log: Path
    official_compare_methods: Path
    recovery_module: Path
    recovery_cli: Path
    approval: Path
    remediation_receipt: Path
    output_per_image: Path
    output_summary: Path
    output_complete: Path


def production_paths() -> RecoveryPaths:
    """Return the one canonical production recovery path set."""

    return RecoveryPaths(
        confinement_root=DATA_DISK_ROOT,
        formal_authorization=PROJECT_ROOT
        / "artifacts/approvals/FORMAL_MIO100_APPROVED.json",
        evaluator_complete=FORMAL_ROOT / "complete.json",
        evaluator_per_image=FORMAL_ROOT / "per_image.csv",
        table1_input=FORMAL_ROOT / "table1_input.jsonl",
        weights_lock=PROJECT_ROOT / "artifacts/formal_mio100/cache/weights_lock.json",
        metric_parity_summary=PROJECT_ROOT
        / "artifacts/metrics/metric_parity_summary.json",
        run_contract=SCORE_ROOT / "run_contract.json",
        input_lock=SCORE_ROOT / "input_lock.json",
        worker_request=SCORE_ROOT / ".worker/request-00000.json",
        shards_dir=SCORE_ROOT / "shards",
        score_root=SCORE_ROOT,
        legacy_module=PROJECT_ROOT / "src/evaluation/agenticir_table1.py",
        legacy_cli=PROJECT_ROOT / "scripts/score_agenticir_table1.py",
        failure_log=PROJECT_ROOT / "artifacts/formal_mio100_table1_score.log",
        official_compare_methods=Path(
            "/root/autodl-tmp/graph/upstream/AgenticIR/eval/compare_methods.py"
        ),
        recovery_module=Path(__file__).resolve(),
        recovery_cli=PROJECT_ROOT
        / "scripts/recover_agenticir_table1_backend_parity.py",
        approval=APPROVAL_PATH,
        remediation_receipt=REMEDIATION_RECEIPT_PATH,
        output_per_image=SCORE_ROOT / "per_image.csv",
        output_summary=SCORE_ROOT / "summary.json",
        output_complete=SCORE_ROOT / "complete.json",
    )


# These are identity anchors, not post-hoc numerical tolerances.  They freeze
# the exact fail-closed transaction that the user authorized us to recover.
PRODUCTION_ANCHORS: dict[str, str] = {
    "formal_authorization": "38647a598503ff0c1776e618a559c6d25d1a8b552fbf1bace765b08ddad9b474",
    "evaluator_complete": "19efefd629a03bbbf85005dc728b93d8f6d8edd04bbfcb8bd8b8c19ae958e000",
    "evaluator_per_image": "83b1f1caeb3c72e4d0bac5125935721e78cb69ff7c7f158f6d185e9a3262d745",
    "table1_input": "a7bf90ee91251be68a6420799b8b343b61d4c0fad4ac4372cd928511ca6ffe07",
    "weights_lock": "796e39eddc51c28e57b9c40b393f99fd73bd14fde2bab138987c2ddcde746e7d",
    "metric_parity_summary": "554f34cb53639d2faa484bc2f2d2273c657f863edbe8258bcbfc1985abd6c35c",
    "run_contract": "d26068b057dfac14acd7c1bc634a34bf335c145161555cf4c50eb13eb04bc84f",
    "input_lock": "a4f84f17ece8e556f8ac4d085847db241314e282e3458465184eb2e803444f75",
    "legacy_module": "22c8f48607b631ab9ddf2e0565012be2be5f52674eae18e2b2f09ad02faa8d73",
    "legacy_cli": "a0578350a55b1740e6ad9a1096062096ca634f9e459bb44852eaaf75a60b5d34",
    "failure_log": "62fcef37816b552422e6099e13b2af965bc31d124ae72991efb6fefd3e303868",
    "worker_request": "28a6bd881000233c43928720e2f6a8d98f4338fa81201ec106924984dcbd8f80",
    "official_compare_methods": "a246b8656744649ed5adfd5f482491f89006ef7bec1ce9923b5971a1da3d856a",
    "shard_sha_list": "26f2d746da1132e99c3e92e4b298de606496ab1cc59e036ff94ff80e5eb55256",
}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Table1RecoveryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Table1RecoveryError(f"{label} is not a lowercase SHA256")
    return value


def _assert_confined(path: Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise Table1RecoveryError(f"path must be absolute and normalized: {candidate}")
    canonical_root = root.resolve(strict=True)
    try:
        candidate.relative_to(canonical_root)
    except ValueError as exc:
        raise Table1RecoveryError(
            f"path escapes recovery confinement root {canonical_root}: {candidate}"
        ) from exc
    current = canonical_root
    relative = candidate.relative_to(canonical_root)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise Table1RecoveryError(f"recovery path crosses symlink: {current}")
        if not current.exists():
            break
    return candidate


def _require_regular(
    path: Path,
    *,
    confinement_root: Path,
    immutable: bool = False,
) -> os.stat_result:
    _assert_confined(path, confinement_root)
    try:
        info = path.lstat()
    except OSError as exc:
        raise Table1RecoveryError(f"missing recovery evidence {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise Table1RecoveryError(f"expected non-symlink regular file: {path}")
    if immutable and stat.S_IMODE(info.st_mode) != 0o444:
        raise Table1RecoveryError(
            f"immutable evidence mode is not 0444: {path} "
            f"({stat.S_IMODE(info.st_mode):04o})"
        )
    return info


def _sha256_file(path: Path, *, confinement_root: Path) -> str:
    _require_regular(path, confinement_root=confinement_root)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(
    path: Path,
    *,
    confinement_root: Path,
    immutable: bool = False,
) -> dict[str, Any]:
    info = _require_regular(
        path, confinement_root=confinement_root, immutable=immutable
    )
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_file(path, confinement_root=confinement_root),
        "size": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _load_json(path: Path, *, confinement_root: Path) -> Any:
    _require_regular(path, confinement_root=confinement_root)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_strict_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise Table1RecoveryError(f"cannot read strict JSON {path}: {exc}") from exc


def _load_jsonl(path: Path, *, confinement_root: Path) -> list[dict[str, Any]]:
    _require_regular(path, confinement_root=confinement_root)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Table1RecoveryError(
                        f"blank JSONL row at {path}:{line_number}"
                    )
                value = json.loads(line, object_pairs_hook=_strict_pairs)
                if not isinstance(value, dict):
                    raise Table1RecoveryError(
                        f"JSONL row is not an object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise Table1RecoveryError(f"cannot read strict JSONL {path}: {exc}") from exc
    return rows


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise Table1RecoveryError(f"{label} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Table1RecoveryError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise Table1RecoveryError(f"{label} is non-finite")
    return result


def _verify_binding_payload(
    payload: object,
    *,
    expected_path: Path,
    confinement_root: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) < {"path", "sha256"}:
        raise Table1RecoveryError(f"{label} binding is malformed")
    if Path(str(payload["path"])) != expected_path.resolve(strict=True):
        raise Table1RecoveryError(f"{label} binding path drifted")
    actual = _binding(expected_path, confinement_root=confinement_root)
    if actual["sha256"] != payload["sha256"]:
        raise Table1RecoveryError(f"{label} binding SHA256 drifted")
    return actual


def _identity(row: Mapping[str, Any]) -> dict[str, str]:
    return {field: str(row[field]) for field in IDENTITY_FIELDS}


def _validate_input_rows(rows: Sequence[Mapping[str, Any]], spec: RecoverySpec) -> None:
    if len(rows) != spec.image_count:
        raise Table1RecoveryError(
            f"expected {spec.image_count} Table-1 inputs, got {len(rows)}"
        )
    seen: set[str] = set()
    counts = {name: 0 for name in spec.expected_counts}
    expected_order = [
        name
        for combinations in spec.groups.values()
        for name in combinations
        if name in spec.expected_counts
    ]
    combination_order = {name: index for index, name in enumerate(expected_order)}
    prior_key: tuple[int, str] | None = None
    for index, row in enumerate(rows):
        if set(row) != INPUT_KEYS or row.get("schema_version") != INPUT_SCHEMA:
            raise Table1RecoveryError(f"input row {index} schema/key drifted")
        identity = _identity(row)
        sample_id = identity["sample_id"]
        combination = identity["combination"]
        group = identity["group"]
        if sample_id in seen:
            raise Table1RecoveryError(f"duplicate input sample_id: {sample_id}")
        seen.add(sample_id)
        if combination not in counts:
            raise Table1RecoveryError(f"unexpected input combination: {combination}")
        expected_group = next(
            (
                name
                for name, combinations in spec.groups.items()
                if combination in combinations
            ),
            None,
        )
        if group != expected_group:
            raise Table1RecoveryError(
                f"input group mismatch for {sample_id}: {group} != {expected_group}"
            )
        for field in ("prediction_sha256", "target_sha256"):
            _require_sha256(identity[field], label=f"{sample_id}/{field}")
        key = (combination_order[combination], sample_id)
        if prior_key is not None and key <= prior_key:
            raise Table1RecoveryError("Table-1 input ordering drifted")
        prior_key = key
        counts[combination] += 1
    if counts != {name: int(value) for name, value in spec.expected_counts.items()}:
        raise Table1RecoveryError(f"Table-1 input counts drifted: {counts}")


def _validate_score_tree(paths: RecoveryPaths, *, allow_outputs: bool) -> None:
    root = paths.score_root
    _assert_confined(root, paths.confinement_root)
    if root.is_symlink() or not root.is_dir():
        raise Table1RecoveryError(f"invalid score root: {root}")
    names = {entry.name for entry in root.iterdir()}
    unexpected = names - _ALLOWED_SCORE_ROOT
    if unexpected:
        raise Table1RecoveryError(
            f"unauthorized score-root entries: {sorted(unexpected)}"
        )
    for directory in (paths.shards_dir, root / ".worker"):
        if directory.is_symlink() or not directory.is_dir():
            raise Table1RecoveryError(f"invalid score subdirectory: {directory}")
        for entry in directory.iterdir():
            if entry.is_symlink() or not entry.is_file():
                raise Table1RecoveryError(f"invalid score-tree entry: {entry}")
    worker_names = {entry.name for entry in (root / ".worker").iterdir()}
    if worker_names != {paths.worker_request.name}:
        raise Table1RecoveryError(
            f"frozen worker-request set drifted: {sorted(worker_names)}"
        )
    for output in (
        paths.output_per_image,
        paths.output_summary,
        paths.output_complete,
    ):
        if output.exists() or output.is_symlink():
            if not allow_outputs:
                raise Table1RecoveryError(
                    f"recovery approval must precede result publication: {output}"
                )
            _require_regular(
                output, confinement_root=paths.confinement_root, immutable=True
            )


def _aggregate(
    records: Sequence[Mapping[str, Any]], spec: RecoverySpec
) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[float]]] = {
        combination: {metric: [] for metric in METRICS}
        for combination in spec.expected_counts
    }
    for row in records:
        combination = str(row["combination"])
        for metric in METRICS:
            buckets[combination][metric].append(
                _finite(row[metric], label=f"{row['sample_id']}/{metric}")
            )
    combinations_result: dict[str, dict[str, Any]] = {}
    for group, combinations in spec.groups.items():
        for combination in combinations:
            if combination not in spec.expected_counts:
                continue
            count = int(spec.expected_counts[combination])
            if any(len(values) != count for values in buckets[combination].values()):
                raise Table1RecoveryError(f"score count drifted: {combination}")
            combinations_result[combination] = {
                "group": group,
                "count": count,
                **{
                    metric: math.fsum(buckets[combination][metric]) / count
                    for metric in METRICS
                },
            }
    groups_result: dict[str, dict[str, Any]] = {}
    for group, combinations in spec.groups.items():
        selected = [name for name in combinations if name in spec.expected_counts]
        groups_result[group] = {
            "combination_count": len(selected),
            "image_count": sum(
                int(combinations_result[name]["count"]) for name in selected
            ),
            **{
                metric: math.fsum(
                    float(combinations_result[name][metric]) for name in selected
                )
                / len(selected)
                for metric in METRICS
            },
        }
    return {
        "image_count": len(records),
        "combinations": combinations_result,
        "groups": groups_result,
        "aggregation": (
            "per-image score -> arithmetic mean within each combination -> "
            "equal arithmetic mean of combination means within each group"
        ),
    }


def _aggregate_proxy(
    rows: Sequence[Mapping[str, Any]], spec: RecoverySpec
) -> dict[str, Any]:
    converted = []
    for row in rows:
        converted.append(
            {
                "sample_id": row["sample_id"],
                "group": row["group"],
                "combination": row["combination"],
                "psnr": _finite(row["psnr"], label=f"{row['sample_id']}/psnr"),
                "ssim": _finite(row["ssim"], label=f"{row['sample_id']}/ssim"),
                "lpips": 0.0,
                "maniqa": 0.0,
                "clipiqa": 0.0,
                "musiq": 0.0,
            }
        )
    return _aggregate(converted, spec)


def _drift_stats(
    rows: Sequence[Mapping[str, Any]], metric: str, observed_cpu_max: float
) -> dict[str, Any]:
    signed = [float(row[f"{metric}_signed_difference"]) for row in rows]
    absolute = [abs(value) for value in signed]
    ordered = sorted(absolute)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    maximum = max(range(len(rows)), key=lambda index: absolute[index])
    return {
        "count": len(rows),
        "minimum_signed_difference": min(signed),
        "maximum_signed_difference": max(signed),
        "maximum_absolute_difference": absolute[maximum],
        "maximum_absolute_sample_id": rows[maximum]["sample_id"],
        "mean_signed_difference": math.fsum(signed) / len(signed),
        "mean_absolute_difference": math.fsum(absolute) / len(absolute),
        "median_absolute_difference": percentile(0.50),
        "q90_absolute_difference": percentile(0.90),
        "q95_absolute_difference": percentile(0.95),
        "q99_absolute_difference": percentile(0.99),
        "zero_difference_count": sum(value == 0.0 for value in signed),
        "positive_difference_count": sum(value > 0.0 for value in signed),
        "negative_difference_count": sum(value < 0.0 for value in signed),
        "cpu_preflight_observed_max": observed_cpu_max,
        "exceeds_cpu_preflight_observed_max_count": sum(
            value > observed_cpu_max for value in absolute
        ),
    }


def _crosscheck(
    records: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, str]],
    *,
    predictions_digest: str,
    parity: Mapping[str, Any],
    compare_binding: Mapping[str, Any],
    spec: RecoverySpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if parity.get("passed") is not True or not isinstance(parity.get("facts"), Mapping):
        raise Table1RecoveryError("metric parity artifact is not passing")
    facts = parity["facts"]
    cpu_psnr_max = _finite(
        facts.get("max_psnr_abs_diff"), label="CPU preflight PSNR observed max"
    )
    cpu_ssim_max = _finite(
        facts.get("max_ssim_abs_diff"), label="CPU preflight SSIM observed max"
    )
    evaluator_by_id: dict[str, Mapping[str, str]] = {}
    for row in evaluator_rows:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in evaluator_by_id:
            raise Table1RecoveryError(
                f"duplicate/empty evaluator sample_id: {sample_id!r}"
            )
        evaluator_by_id[sample_id] = row
    if len(evaluator_by_id) != spec.image_count or len(records) != spec.image_count:
        raise Table1RecoveryError("crosscheck row count drifted")

    digest_rows = [
        {
            "sample_id": row["sample_id"],
            "prediction_sha256": row["prediction_sha256"],
            "target_sha256": row["target_sha256"],
        }
        for row in evaluator_rows
    ]
    per_image: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        evaluator = evaluator_by_id.get(sample_id)
        if evaluator is None or sample_id in seen:
            raise Table1RecoveryError(f"crosscheck sample set drifted: {sample_id}")
        seen.add(sample_id)
        for field in IDENTITY_FIELDS[1:]:
            if str(record[field]) != evaluator[field]:
                raise Table1RecoveryError(
                    f"evaluator/scorer identity drift: {sample_id}/{field}"
                )
        scorer_psnr = _finite(record["psnr"], label=f"{sample_id}/scorer psnr")
        scorer_ssim = _finite(record["ssim"], label=f"{sample_id}/scorer ssim")
        evaluator_psnr = _finite(evaluator["psnr"], label=f"{sample_id}/evaluator psnr")
        evaluator_ssim = _finite(evaluator["ssim"], label=f"{sample_id}/evaluator ssim")
        per_image.append(
            {
                **_identity(record),
                "evaluator_cpu_psnr": evaluator_psnr,
                "official_cuda_psnr": scorer_psnr,
                "psnr_signed_difference": scorer_psnr - evaluator_psnr,
                "evaluator_cpu_ssim": evaluator_ssim,
                "official_cuda_ssim": scorer_ssim,
                "ssim_signed_difference": scorer_ssim - evaluator_ssim,
            }
        )
    if seen != set(evaluator_by_id):
        raise Table1RecoveryError("evaluator/scorer sample sets differ")
    expected_digest = _require_sha256(
        predictions_digest, label="formal predictions digest"
    )
    if _sha256_json(digest_rows) != expected_digest:
        raise Table1RecoveryError("formal predictions digest drifted")

    scorer_aggregate = _aggregate(records, spec)
    evaluator_aggregate = _aggregate_proxy(evaluator_rows, spec)
    display_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for level, names in (
        ("combination", [name for values in spec.groups.values() for name in values]),
        ("group", list(spec.groups)),
    ):
        scorer_values = (
            scorer_aggregate["combinations"]
            if level == "combination"
            else scorer_aggregate["groups"]
        )
        evaluator_values = (
            evaluator_aggregate["combinations"]
            if level == "combination"
            else evaluator_aggregate["groups"]
        )
        for name in names:
            if name not in scorer_values:
                continue
            for metric in ("psnr", "ssim"):
                scorer_value = float(scorer_values[name][metric])
                evaluator_value = float(evaluator_values[name][metric])
                row = {
                    "level": level,
                    "name": name,
                    "metric": metric,
                    "format_spec": ".4",
                    "evaluator_cpu_value": evaluator_value,
                    "official_cuda_value": scorer_value,
                    "evaluator_cpu_display": format(evaluator_value, ".4"),
                    "official_cuda_display": format(scorer_value, ".4"),
                }
                row["equal"] = (
                    row["evaluator_cpu_display"] == row["official_cuda_display"]
                )
                display_rows.append(row)
                if not row["equal"]:
                    mismatches.append(row)
    crosscheck = {
        "status": "IDENTITY_EXACT_OFFICIAL_SCORER_AUTHORITATIVE",
        "identity_passed": True,
        "numeric_parity_claim": False,
        "numeric_gate_applied": False,
        "tolerance_changed": False,
        "mode": "exact_per_image_identity_with_numeric_drift_diagnostic",
        "canonical_metric_source": (
            "immutable pinned-AgenticIR pyiqa-0.1.10 CUDA score shards"
        ),
        "evaluator_metric_role": "CPU fast-parity diagnostic only",
        "metric_parity_artifact_role": (
            "CPU-to-CPU finite preflight sample; observed maxima are not "
            "cross-device tolerances"
        ),
        "image_count": spec.image_count,
        "identity_fields": list(IDENTITY_FIELDS),
        "prediction_digest": expected_digest,
        "all_six_scorer_metrics_finite": True,
        "per_image_numeric_comparison_is_diagnostic": True,
        "numeric_drift": {
            "psnr": _drift_stats(per_image, "psnr", cpu_psnr_max),
            "ssim": _drift_stats(per_image, "ssim", cpu_ssim_max),
        },
        "table_display_diagnostic": {
            "source": dict(compare_binding),
            "format_spec": ".4",
            "meaning": "four significant digits, as frozen by AgenticIR compare_methods.py",
            "combination_count": len(spec.expected_counts),
            "group_count": len(spec.groups),
            "rows": display_rows,
            "mismatch_count": len(mismatches),
            "all_equal": not mismatches,
            "scientific_acceptance_gate": False,
        },
    }
    return crosscheck, per_image


def _scan_shards(
    paths: RecoveryPaths,
    *,
    rows: Sequence[Mapping[str, Any]],
    run_contract_sha256: str,
    input_lock_sha256: str,
    initial_rng_core: Mapping[str, Any],
    spec: RecoverySpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    entries = list(paths.shards_dir.iterdir())
    unexpected = [
        entry.name for entry in entries if not _SHARD_NAME.fullmatch(entry.name)
    ]
    if unexpected:
        raise Table1RecoveryError(f"unexpected shard entries: {sorted(unexpected)}")
    expected_names = [f"shard-{index:05d}.json" for index in range(spec.shard_count)]
    if sorted(entry.name for entry in entries) != expected_names:
        raise Table1RecoveryError("score shard set is not exact and complete")

    score_rows: list[dict[str, Any]] = []
    shard_bindings: list[dict[str, Any]] = []
    previous_rng: object = None
    runtime: Mapping[str, Any] | None = None
    for shard_index, name in enumerate(expected_names):
        path = paths.shards_dir / name
        start = shard_index * spec.shard_size
        end = min(start + spec.shard_size, spec.image_count)
        binding = _binding(
            path, confinement_root=paths.confinement_root, immutable=True
        )
        shard_bindings.append(
            {
                "shard_index": shard_index,
                "start_index": start,
                "end_index": end,
                "name": name,
                **binding,
            }
        )
        payload = _load_json(path, confinement_root=paths.confinement_root)
        if not isinstance(payload, Mapping):
            raise Table1RecoveryError(f"shard is not an object: {name}")
        exact_keys = {
            "schema_version",
            "shard_index",
            "start_index",
            "end_index",
            "run_contract_sha256",
            "input_lock_sha256",
            "runtime",
            "rng_before",
            "rng_before_sha256",
            "rng_after",
            "rng_after_sha256",
            "peak_reserved_bytes",
            "total_memory_bytes",
            "peak_reserved_fraction",
            "rows",
        }
        if set(payload) != exact_keys:
            raise Table1RecoveryError(f"shard keys drifted: {name}")
        if (
            payload["schema_version"] != SHARD_SCHEMA
            or payload["shard_index"] != shard_index
            or payload["start_index"] != start
            or payload["end_index"] != end
            or payload["run_contract_sha256"] != run_contract_sha256
            or payload["input_lock_sha256"] != input_lock_sha256
        ):
            raise Table1RecoveryError(f"shard metadata drifted: {name}")
        if _sha256_json(payload["rng_before"]) != payload["rng_before_sha256"]:
            raise Table1RecoveryError(f"shard rng_before hash drifted: {name}")
        if _sha256_json(payload["rng_after"]) != payload["rng_after_sha256"]:
            raise Table1RecoveryError(f"shard rng_after hash drifted: {name}")
        if shard_index == 0:
            core = {
                key: payload["rng_before"][key]
                for key in ("python", "numpy", "torch_cpu")
            }
            if core != dict(initial_rng_core):
                raise Table1RecoveryError("first shard RNG differs from weights lock")
        elif payload["rng_before"] != previous_rng:
            raise Table1RecoveryError(f"RNG chain break before shard {shard_index}")
        previous_rng = payload["rng_after"]
        if runtime is None:
            runtime = payload["runtime"]
        elif payload["runtime"] != runtime:
            raise Table1RecoveryError(f"runtime drifted at shard {shard_index}")
        if not isinstance(runtime, Mapping) or runtime.get("device") != "cuda:0":
            raise Table1RecoveryError("official shard runtime is not cuda:0")
        reserved = _finite(
            payload["peak_reserved_bytes"], label=f"{name}/peak_reserved_bytes"
        )
        total = _finite(payload["total_memory_bytes"], label=f"{name}/total_memory")
        stored_fraction = _finite(
            payload["peak_reserved_fraction"], label=f"{name}/peak fraction"
        )
        if total <= 0.0 or stored_fraction != reserved / total:
            raise Table1RecoveryError(f"shard VRAM fraction drifted: {name}")
        if stored_fraction >= spec.maximum_vram_fraction:
            raise Table1RecoveryError(f"shard VRAM ceiling exceeded: {name}")
        shard_rows = payload["rows"]
        if not isinstance(shard_rows, list) or len(shard_rows) != end - start:
            raise Table1RecoveryError(f"shard row count drifted: {name}")
        for offset, score_row in enumerate(shard_rows):
            absolute_index = start + offset
            if not isinstance(score_row, Mapping) or set(score_row) != SCORE_KEYS:
                raise Table1RecoveryError(f"score row keys drifted: {absolute_index}")
            if score_row.get("schema_version") != INPUT_SCHEMA:
                raise Table1RecoveryError(f"score row schema drifted: {absolute_index}")
            if _identity(score_row) != _identity(rows[absolute_index]):
                raise Table1RecoveryError(f"score identity drifted: {absolute_index}")
            canonical = dict(score_row)
            for metric in METRICS:
                canonical[metric] = _finite(
                    score_row[metric], label=f"{score_row['sample_id']}/{metric}"
                )
            score_rows.append(canonical)
    if runtime is None:
        raise Table1RecoveryError("no shard runtime found")
    return score_rows, shard_bindings, runtime


def _read_evaluator_csv(path: Path, *, confinement_root: Path) -> list[dict[str, str]]:
    _require_regular(path, confinement_root=confinement_root, immutable=True)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EVALUATOR_FIELDS:
                raise Table1RecoveryError("evaluator CSV header drifted")
            return list(reader)
    except OSError as exc:
        raise Table1RecoveryError(f"cannot read evaluator CSV: {exc}") from exc


def _validate_hash_anchors(
    bindings: Mapping[str, Mapping[str, Any]], expected: Mapping[str, str]
) -> None:
    for label, wanted in expected.items():
        if label == "shard_sha_list":
            continue
        actual = bindings.get(label, {}).get("sha256")
        if actual != wanted:
            raise Table1RecoveryError(
                f"recovery identity anchor drifted for {label}: {actual} != {wanted}"
            )


def audit_recovery_state(
    paths: RecoveryPaths,
    *,
    spec: RecoverySpec = PRODUCTION_SPEC,
    expected_anchors: Mapping[str, str] = PRODUCTION_ANCHORS,
    allow_outputs: bool,
) -> dict[str, Any]:
    """Validate the complete failed transaction without opening any image."""

    _validate_score_tree(paths, allow_outputs=allow_outputs)
    binding_paths = {
        "formal_authorization": paths.formal_authorization,
        "evaluator_complete": paths.evaluator_complete,
        "evaluator_per_image": paths.evaluator_per_image,
        "table1_input": paths.table1_input,
        "weights_lock": paths.weights_lock,
        "metric_parity_summary": paths.metric_parity_summary,
        "run_contract": paths.run_contract,
        "input_lock": paths.input_lock,
        "worker_request": paths.worker_request,
        "legacy_module": paths.legacy_module,
        "legacy_cli": paths.legacy_cli,
        "failure_log": paths.failure_log,
        "official_compare_methods": paths.official_compare_methods,
        "recovery_module": paths.recovery_module,
        "recovery_cli": paths.recovery_cli,
    }
    bindings = {
        label: _binding(
            path,
            confinement_root=paths.confinement_root,
            immutable=label
            in {
                "formal_authorization",
                "evaluator_complete",
                "evaluator_per_image",
                "table1_input",
                "weights_lock",
                "run_contract",
                "input_lock",
                "worker_request",
            },
        )
        for label, path in binding_paths.items()
    }
    _validate_hash_anchors(bindings, expected_anchors)

    failure_text = paths.failure_log.read_text(encoding="utf-8")
    if (
        "AgenticIR Table-1 contract error: evaluator/scorer" not in failure_text
        or "drift" not in failure_text
    ):
        raise Table1RecoveryError("failure log is not the audited FR drift failure")

    rows = _load_jsonl(paths.table1_input, confinement_root=paths.confinement_root)
    _validate_input_rows(rows, spec)
    input_lock = _load_json(paths.input_lock, confinement_root=paths.confinement_root)
    input_lock_keys = {
        "schema_version",
        "created_utc",
        "manifest",
        "image_count",
        "expected_counts",
        "ordering",
        "rows",
    }
    if (
        not isinstance(input_lock, Mapping)
        or set(input_lock) != input_lock_keys
        or input_lock.get("schema_version") != INPUT_LOCK_SCHEMA
        or input_lock.get("image_count") != spec.image_count
        or input_lock.get("expected_counts")
        != {name: int(value) for name, value in spec.expected_counts.items()}
        or input_lock.get("ordering")
        != "OFFICIAL_GROUPS order, then strictly increasing sample_id"
    ):
        raise Table1RecoveryError("input lock semantic content drifted")
    if input_lock.get("manifest") != bindings["table1_input"]:
        raise Table1RecoveryError("input lock manifest binding drifted")
    locked_rows = input_lock.get("rows")
    locked_stat_fields = {
        "prediction_mode",
        "prediction_device",
        "prediction_inode",
        "prediction_size",
        "target_mode",
        "target_device",
        "target_inode",
        "target_size",
    }
    if not isinstance(locked_rows, list) or len(locked_rows) != len(rows):
        raise Table1RecoveryError("input lock row count drifted")
    for index, (locked, canonical) in enumerate(zip(locked_rows, rows, strict=True)):
        if (
            not isinstance(locked, Mapping)
            or set(locked) != INPUT_KEYS | locked_stat_fields
            or {key: locked[key] for key in INPUT_KEYS} != canonical
        ):
            raise Table1RecoveryError(f"input lock row drifted: {index}")
        for key in locked_stat_fields:
            if isinstance(locked[key], bool) or not isinstance(locked[key], int):
                raise Table1RecoveryError(
                    f"input lock stat metadata drifted: {index}/{key}"
                )

    run_contract = _load_json(
        paths.run_contract, confinement_root=paths.confinement_root
    )
    if not isinstance(run_contract, Mapping):
        raise Table1RecoveryError("run contract is not an object")
    if (
        run_contract.get("schema_version") != RUN_CONTRACT_SCHEMA
        or run_contract.get("metrics") != list(METRICS)
        or run_contract.get("metric_directions") != METRIC_DIRECTIONS
        or run_contract.get("device") != "cuda:0"
        or run_contract.get("shard_size") != spec.shard_size
        or run_contract.get("image_count") != spec.image_count
        or run_contract.get("expected_counts")
        != {name: int(value) for name, value in spec.expected_counts.items()}
        or run_contract.get("formal_mio100_only") is not True
    ):
        raise Table1RecoveryError("run contract scientific scope drifted")
    _verify_binding_payload(
        run_contract.get("input_lock"),
        expected_path=paths.input_lock,
        confinement_root=paths.confinement_root,
        label="run contract input lock",
    )
    _verify_binding_payload(
        run_contract.get("weights_lock"),
        expected_path=paths.weights_lock,
        confinement_root=paths.confinement_root,
        label="run contract weights lock",
    )
    implementation = run_contract.get("implementation")
    if not isinstance(implementation, Mapping):
        raise Table1RecoveryError("run contract implementation binding is malformed")
    expected_implementation = {
        "table1_scorer_module": bindings["legacy_module"],
        "table1_scorer_cli": bindings["legacy_cli"],
    }
    if implementation != expected_implementation:
        raise Table1RecoveryError("legacy scorer implementation binding drifted")

    sources = run_contract.get("agenticir_sources")
    if not isinstance(sources, Mapping):
        raise Table1RecoveryError("AgenticIR source bindings are malformed")
    compare_source = sources.get("official_compare_methods")
    if compare_source != bindings["official_compare_methods"]:
        raise Table1RecoveryError("official compare_methods binding drifted")
    for label, raw in sources.items():
        if not isinstance(raw, Mapping) or set(raw) < {"path", "sha256"}:
            raise Table1RecoveryError(f"AgenticIR source binding malformed: {label}")
        source_path = Path(str(raw["path"]))
        actual = _binding(source_path, confinement_root=paths.confinement_root)
        if actual != raw:
            raise Table1RecoveryError(f"AgenticIR source drifted: {label}")

    formal_evidence = run_contract.get("formal_evidence")
    if (
        not isinstance(formal_evidence, Mapping)
        or set(formal_evidence) != _FORMAL_EVIDENCE_KEYS
    ):
        raise Table1RecoveryError("run contract formal evidence keys drifted")
    expected_formal_paths = {
        "authorization": paths.formal_authorization,
        "evaluator_complete": paths.evaluator_complete,
        "per_image": paths.evaluator_per_image,
        "table1_input": paths.table1_input,
        "metric_parity_summary": paths.metric_parity_summary,
    }
    verified_formal: dict[str, Any] = {}
    for label, raw in formal_evidence.items():
        if label == "predictions_digest":
            verified_formal[label] = _require_sha256(
                raw, label="formal predictions digest"
            )
            continue
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise Table1RecoveryError(f"formal evidence binding malformed: {label}")
        evidence_path = Path(str(raw["path"]))
        if (
            label in expected_formal_paths
            and evidence_path != expected_formal_paths[label]
        ):
            raise Table1RecoveryError(f"formal evidence path drifted: {label}")
        actual = _binding(evidence_path, confinement_root=paths.confinement_root)
        if actual["sha256"] != raw["sha256"]:
            raise Table1RecoveryError(f"formal evidence SHA256 drifted: {label}")
        verified_formal[label] = actual

    evaluator_complete = _load_json(
        paths.evaluator_complete, confinement_root=paths.confinement_root
    )
    if (
        not isinstance(evaluator_complete, Mapping)
        or evaluator_complete.get("schema_version")
        != "graphrestore-formal-mio100-complete-v1"
        or evaluator_complete.get("status") != "COMPLETE"
        or evaluator_complete.get("image_count") != spec.image_count
        or evaluator_complete.get("predictions_digest")
        != formal_evidence["predictions_digest"]
        or evaluator_complete.get("authorization_sha256")
        != bindings["formal_authorization"]["sha256"]
    ):
        raise Table1RecoveryError("formal evaluator completion drifted")
    evaluator_bindings = evaluator_complete.get("bindings")
    if not isinstance(evaluator_bindings, Mapping):
        raise Table1RecoveryError("formal evaluator completion bindings malformed")
    _verify_binding_payload(
        evaluator_bindings.get("per_image_csv"),
        expected_path=paths.evaluator_per_image,
        confinement_root=paths.confinement_root,
        label="evaluator per-image",
    )
    _verify_binding_payload(
        evaluator_bindings.get("table1_input_jsonl"),
        expected_path=paths.table1_input,
        confinement_root=paths.confinement_root,
        label="evaluator Table-1 input",
    )

    weights = _load_json(paths.weights_lock, confinement_root=paths.confinement_root)
    if not isinstance(weights, Mapping) or not isinstance(
        weights.get("initial_rng_core"), Mapping
    ):
        raise Table1RecoveryError("weights lock is malformed")
    worker_request = _load_json(
        paths.worker_request, confinement_root=paths.confinement_root
    )
    worker_keys = {
        "schema_version",
        "device",
        "expected_metric_runtime",
        "expected_runtime",
        "formal_evidence",
        "implementation",
        "initial_rng_core",
        "input_lock",
        "input_lock_sha256",
        "previous_rng",
        "rows",
        "run_contract",
        "run_contract_sha256",
        "score_root",
        "shard_size",
        "shards_dir",
        "start_shard",
        "weights_lock",
    }
    if (
        not isinstance(worker_request, Mapping)
        or set(worker_request) != worker_keys
        or worker_request.get("schema_version")
        != "graphrestore.agenticir_table1_worker_request.v1"
        or worker_request.get("device") != "cuda:0"
        or worker_request.get("expected_runtime") is not None
        or worker_request.get("formal_evidence") != formal_evidence
        or worker_request.get("implementation") != implementation
        or worker_request.get("initial_rng_core") != weights["initial_rng_core"]
        or worker_request.get("input_lock") != run_contract["input_lock"]
        or worker_request.get("input_lock_sha256") != bindings["input_lock"]["sha256"]
        or worker_request.get("previous_rng") is not None
        or worker_request.get("rows") != rows
        or worker_request.get("run_contract")
        != {
            "path": bindings["run_contract"]["path"],
            "sha256": bindings["run_contract"]["sha256"],
        }
        or worker_request.get("run_contract_sha256")
        != bindings["run_contract"]["sha256"]
        or worker_request.get("score_root") != str(paths.score_root)
        or worker_request.get("shard_size") != spec.shard_size
        or worker_request.get("shards_dir") != str(paths.shards_dir)
        or worker_request.get("start_shard") != 0
        or worker_request.get("weights_lock") != run_contract["weights_lock"]
        or worker_request.get("expected_metric_runtime")
        != weights.get("metric_runtime")
    ):
        raise Table1RecoveryError("frozen worker request semantic content drifted")
    records, shard_bindings, runtime = _scan_shards(
        paths,
        rows=rows,
        run_contract_sha256=bindings["run_contract"]["sha256"],
        input_lock_sha256=bindings["input_lock"]["sha256"],
        initial_rng_core=weights["initial_rng_core"],
        spec=spec,
    )
    simple_shard_sha_list = [
        {"name": item["name"], "sha256": item["sha256"]} for item in shard_bindings
    ]
    shard_sha_list_digest = _sha256_json(simple_shard_sha_list)
    shard_inventory_digest = _sha256_json(shard_bindings)
    wanted_shards = expected_anchors.get("shard_sha_list")
    if wanted_shards is not None and shard_sha_list_digest != wanted_shards:
        raise Table1RecoveryError("immutable shard SHA list drifted")

    evaluator_rows = _read_evaluator_csv(
        paths.evaluator_per_image, confinement_root=paths.confinement_root
    )
    if len(evaluator_rows) != spec.image_count:
        raise Table1RecoveryError("evaluator per-image row count drifted")
    parity = _load_json(
        paths.metric_parity_summary, confinement_root=paths.confinement_root
    )
    if not isinstance(parity, Mapping):
        raise Table1RecoveryError("metric parity summary is malformed")
    crosscheck, per_image_drift = _crosscheck(
        records,
        evaluator_rows,
        predictions_digest=str(formal_evidence["predictions_digest"]),
        parity=parity,
        compare_binding=bindings["official_compare_methods"],
        spec=spec,
    )
    aggregate = _aggregate(records, spec)
    return {
        "bindings": bindings,
        "formal_evidence": verified_formal,
        "rows": rows,
        "records": records,
        "runtime": runtime,
        "shards": shard_bindings,
        "shard_sha_list_digest": shard_sha_list_digest,
        "shard_inventory_digest": shard_inventory_digest,
        "aggregate": aggregate,
        "crosscheck": crosscheck,
        "per_image_drift": per_image_drift,
        "state_digest": _sha256_json(
            {
                "bindings": bindings,
                "formal_evidence": verified_formal,
                "shards": shard_bindings,
                "aggregate": aggregate,
                "crosscheck": crosscheck,
                "per_image_drift": per_image_drift,
            }
        ),
    }


def _policy() -> dict[str, Any]:
    return {
        "reason": "CPU fast-proxy versus official CUDA pyiqa backend divergence",
        "official_cuda_shard_six_metrics_are_sole_authority": True,
        "evaluator_cpu_psnr_ssim_are_diagnostic_only": True,
        "per_image_identity_and_hashes_must_be_exact": True,
        "publication_gate": (
            "exact identity, source/evidence bindings, and immutable shard closure"
        ),
        "table_display_comparison_role": (
            "diagnostic only; .4 display agreement is recorded but is not a "
            "numeric scientific acceptance gate"
        ),
        "numeric_parity_claim": False,
        "numeric_gate_applied": False,
        "tolerance_changed": False,
        "posthoc_raw_tolerance_authorized": False,
        "metric_recomputation_authorized": False,
        "image_read_authorized": False,
        "cuda_authorized": False,
        "worker_authorized": False,
        "network_authorized": False,
        "selective_rerun_authorized": False,
        "shard_mutation_authorized": False,
        "legacy_contract_mutation_authorized": False,
    }


def _approval_semantic(
    state: Mapping[str, Any], *, approved_utc: str
) -> dict[str, Any]:
    return {
        "schema_version": APPROVAL_SCHEMA,
        "status": "APPROVED",
        "kind": "agenticir_table1_cpu_cuda_backend_recovery",
        "approved": True,
        "approved_utc": approved_utc,
        "scientific_policy": _policy(),
        "bindings": state["bindings"],
        "formal_evidence": state["formal_evidence"],
        "shard_count": len(state["shards"]),
        "shards": state["shards"],
        "shard_sha_list_digest": state["shard_sha_list_digest"],
        "shard_inventory_digest": state["shard_inventory_digest"],
        "image_count": len(state["records"]),
        "state_digest": state["state_digest"],
        "aggregate_digest": _sha256_json(state["aggregate"]),
        "crosscheck_digest": _sha256_json(state["crosscheck"]),
        "per_image_drift_digest": _sha256_json(state["per_image_drift"]),
        "standard_outputs_absent_at_approval": True,
        "finalize_execute_token_sha256": hashlib.sha256(
            FINALIZE_EXECUTE_TOKEN.encode("utf-8")
        ).hexdigest(),
    }


def _ensure_parent(path: Path, *, confinement_root: Path) -> None:
    _assert_confined(path, confinement_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_confined(path.parent, confinement_root)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise Table1RecoveryError(f"invalid publication directory: {path.parent}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_or_verify(
    path: Path,
    payload: str,
    *,
    confinement_root: Path,
) -> None:
    _ensure_parent(path, confinement_root=confinement_root)
    if path.exists() or path.is_symlink():
        _require_regular(path, confinement_root=confinement_root, immutable=True)
        if path.read_text(encoding="utf-8") != payload:
            raise Table1RecoveryError(f"refusing to clobber published artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        _assert_confined(path, confinement_root)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Table1RecoveryError(
                f"publication race/refusing overwrite: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def publish_approval(
    paths: RecoveryPaths,
    *,
    execute_token: str,
    spec: RecoverySpec = PRODUCTION_SPEC,
    expected_anchors: Mapping[str, str] = PRODUCTION_ANCHORS,
    approved_utc: str | None = None,
) -> dict[str, Any]:
    assert_cpu_only_entrypoint()
    if execute_token != APPROVAL_EXECUTE_TOKEN:
        raise Table1RecoveryError(
            f"approval requires --execute {APPROVAL_EXECUTE_TOKEN}"
        )
    if paths.approval.exists() or paths.approval.is_symlink():
        raise Table1RecoveryError(f"recovery approval already exists: {paths.approval}")
    if paths.remediation_receipt.exists() or paths.remediation_receipt.is_symlink():
        raise Table1RecoveryError("remediation receipt already exists before approval")
    state = audit_recovery_state(
        paths,
        spec=spec,
        expected_anchors=expected_anchors,
        allow_outputs=False,
    )
    approval = _approval_semantic(state, approved_utc=approved_utc or _utc_now())
    _publish_or_verify(
        paths.approval,
        _canonical_json(approval),
        confinement_root=paths.confinement_root,
    )
    validate_approval(
        paths, spec=spec, expected_anchors=expected_anchors, allow_outputs=False
    )
    return approval


def approval_candidate(
    paths: RecoveryPaths,
    *,
    spec: RecoverySpec = PRODUCTION_SPEC,
    expected_anchors: Mapping[str, str] = PRODUCTION_ANCHORS,
    approved_utc: str = "DRY_RUN_UTC",
) -> dict[str, Any]:
    state = audit_recovery_state(
        paths,
        spec=spec,
        expected_anchors=expected_anchors,
        allow_outputs=False,
    )
    return _approval_semantic(state, approved_utc=approved_utc)


def validate_approval(
    paths: RecoveryPaths,
    *,
    spec: RecoverySpec = PRODUCTION_SPEC,
    expected_anchors: Mapping[str, str] = PRODUCTION_ANCHORS,
    allow_outputs: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_regular(
        paths.approval, confinement_root=paths.confinement_root, immutable=True
    )
    approval = _load_json(paths.approval, confinement_root=paths.confinement_root)
    if not isinstance(approval, dict):
        raise Table1RecoveryError("recovery approval is not an object")
    approved_utc = approval.get("approved_utc")
    if not isinstance(approved_utc, str) or not approved_utc.endswith("Z"):
        raise Table1RecoveryError("recovery approval timestamp is malformed")
    state = audit_recovery_state(
        paths,
        spec=spec,
        expected_anchors=expected_anchors,
        allow_outputs=allow_outputs,
    )
    expected = _approval_semantic(state, approved_utc=approved_utc)
    if approval != expected:
        raise Table1RecoveryError("recovery approval content/evidence drifted")
    return approval, state


def _per_image_csv(records: Sequence[Mapping[str, Any]]) -> str:
    destination = io.StringIO(newline="")
    fields = [*IDENTITY_FIELDS, *METRICS]
    writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in records:
        writer.writerow({field: row[field] for field in fields})
    return destination.getvalue()


def _standard_candidates(
    paths: RecoveryPaths,
    *,
    approval: Mapping[str, Any],
    state: Mapping[str, Any],
    spec: RecoverySpec,
) -> tuple[str, str, str, dict[str, Any]]:
    approval_binding = _binding(
        paths.approval,
        confinement_root=paths.confinement_root,
        immutable=True,
    )
    per_image_text = _per_image_csv(state["records"])
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "run_contract_sha256": state["bindings"]["run_contract"]["sha256"],
        "input_lock_sha256": state["bindings"]["input_lock"]["sha256"],
        "weights_lock_sha256": state["bindings"]["weights_lock"]["sha256"],
        "runtime": state["runtime"],
        "metrics": list(METRICS),
        "metric_directions": METRIC_DIRECTIONS,
        "formal_evidence": {
            key: (
                value
                if key == "predictions_digest"
                else {"path": value["path"], "sha256": value["sha256"]}
            )
            for key, value in state["formal_evidence"].items()
        },
        "evaluator_psnr_ssim_crosscheck": {
            **state["crosscheck"],
            "recovery_approval": {
                "path": approval_binding["path"],
                "sha256": approval_binding["sha256"],
            },
        },
        "shard_vram": {
            "ceiling": spec.maximum_vram_fraction,
            "maximum_peak_reserved_fraction": max(
                _finite(
                    _load_json(
                        Path(item["path"]),
                        confinement_root=paths.confinement_root,
                    )["peak_reserved_fraction"],
                    label="shard peak fraction",
                )
                for item in state["shards"]
            ),
            "shards": [
                {
                    "shard_index": index,
                    "peak_reserved_bytes": _load_json(
                        Path(item["path"]),
                        confinement_root=paths.confinement_root,
                    )["peak_reserved_bytes"],
                    "total_memory_bytes": _load_json(
                        Path(item["path"]),
                        confinement_root=paths.confinement_root,
                    )["total_memory_bytes"],
                    "peak_reserved_fraction": _load_json(
                        Path(item["path"]),
                        confinement_root=paths.confinement_root,
                    )["peak_reserved_fraction"],
                }
                for index, item in enumerate(state["shards"])
            ],
        },
        **state["aggregate"],
    }
    summary_text = _canonical_json(summary)
    per_image_sha = hashlib.sha256(per_image_text.encode("utf-8")).hexdigest()
    summary_sha = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "COMPLETE",
        "created_utc": approval["approved_utc"],
        "image_count": len(state["records"]),
        "shard_count": len(state["shards"]),
        "run_contract": {
            "path": state["bindings"]["run_contract"]["path"],
            "sha256": state["bindings"]["run_contract"]["sha256"],
        },
        "input_lock": {
            "path": state["bindings"]["input_lock"]["path"],
            "sha256": state["bindings"]["input_lock"]["sha256"],
        },
        "weights_lock": {
            "path": state["bindings"]["weights_lock"]["path"],
            "sha256": state["bindings"]["weights_lock"]["sha256"],
        },
        "per_image": {
            "path": str(paths.output_per_image),
            "sha256": per_image_sha,
        },
        "summary": {"path": str(paths.output_summary), "sha256": summary_sha},
        "no_selective_rerun": True,
        "all_values_finite": True,
        "formal_evidence": summary["formal_evidence"],
        "evaluator_psnr_ssim_crosscheck": summary["evaluator_psnr_ssim_crosscheck"],
        "maximum_peak_reserved_fraction": summary["shard_vram"][
            "maximum_peak_reserved_fraction"
        ],
        "vram_ceiling": spec.maximum_vram_fraction,
    }
    complete_text = _canonical_json(complete)
    return per_image_text, summary_text, complete_text, summary


def _receipt(
    paths: RecoveryPaths,
    *,
    approval: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = {
        "per_image": _binding(
            paths.output_per_image,
            confinement_root=paths.confinement_root,
            immutable=True,
        ),
        "summary": _binding(
            paths.output_summary,
            confinement_root=paths.confinement_root,
            immutable=True,
        ),
        "complete": _binding(
            paths.output_complete,
            confinement_root=paths.confinement_root,
            immutable=True,
        ),
    }
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "kind": "agenticir_table1_cpu_cuda_backend_recovery",
        "created_utc": approval["approved_utc"],
        "scientific_policy": _policy(),
        "approval": _binding(
            paths.approval,
            confinement_root=paths.confinement_root,
            immutable=True,
        ),
        "inputs": state["bindings"],
        "formal_evidence": state["formal_evidence"],
        "shard_count": len(state["shards"]),
        "shards": state["shards"],
        "shard_sha_list_digest": state["shard_sha_list_digest"],
        "shard_inventory_digest_before": approval["shard_inventory_digest"],
        "shard_inventory_digest_after": state["shard_inventory_digest"],
        "shard_inventory_unchanged": (
            approval["shard_inventory_digest"] == state["shard_inventory_digest"]
        ),
        "outputs": outputs,
        "state_digest": state["state_digest"],
        "crosscheck": state["crosscheck"],
        "per_image_drift": state["per_image_drift"],
        "per_image_drift_digest": _sha256_json(state["per_image_drift"]),
        "official_cuda_aggregate": state["aggregate"],
        "official_cuda_aggregate_digest": _sha256_json(state["aggregate"]),
        "no_metric_recomputation": True,
        "no_image_read": True,
        "no_cuda_initialization": True,
        "no_worker_launch": True,
        "no_network_access": True,
        "legacy_artifacts_mutated": False,
    }


def finalize_recovery(
    paths: RecoveryPaths,
    *,
    execute_token: str,
    spec: RecoverySpec = PRODUCTION_SPEC,
    expected_anchors: Mapping[str, str] = PRODUCTION_ANCHORS,
) -> dict[str, Any]:
    assert_cpu_only_entrypoint()
    if execute_token != FINALIZE_EXECUTE_TOKEN:
        raise Table1RecoveryError(
            f"finalization requires --execute {FINALIZE_EXECUTE_TOKEN}"
        )
    approval, state = validate_approval(
        paths, spec=spec, expected_anchors=expected_anchors, allow_outputs=True
    )
    per_image_text, summary_text, complete_text, _summary = _standard_candidates(
        paths, approval=approval, state=state, spec=spec
    )
    for path, payload in (
        (paths.output_per_image, per_image_text),
        (paths.output_summary, summary_text),
        (paths.output_complete, complete_text),
    ):
        _publish_or_verify(path, payload, confinement_root=paths.confinement_root)
    # Rehash every protected input after the standard artifacts are durable.
    approval_after, state_after = validate_approval(
        paths, spec=spec, expected_anchors=expected_anchors, allow_outputs=True
    )
    if (
        approval_after != approval
        or state_after["state_digest"] != state["state_digest"]
    ):
        raise Table1RecoveryError("protected recovery evidence changed during publish")
    receipt = _receipt(paths, approval=approval, state=state_after)
    _publish_or_verify(
        paths.remediation_receipt,
        _canonical_json(receipt),
        confinement_root=paths.confinement_root,
    )
    verify_recovery(
        paths, spec=spec, expected_anchors=expected_anchors, require_complete=True
    )
    return receipt


def verify_recovery(
    paths: RecoveryPaths,
    *,
    spec: RecoverySpec = PRODUCTION_SPEC,
    expected_anchors: Mapping[str, str] = PRODUCTION_ANCHORS,
    require_complete: bool = False,
) -> dict[str, Any]:
    approval, state = validate_approval(
        paths, spec=spec, expected_anchors=expected_anchors, allow_outputs=True
    )
    per_image_text, summary_text, complete_text, _summary = _standard_candidates(
        paths, approval=approval, state=state, spec=spec
    )
    output_payloads = {
        paths.output_per_image: per_image_text,
        paths.output_summary: summary_text,
        paths.output_complete: complete_text,
    }
    present = [path.exists() or path.is_symlink() for path in output_payloads]
    if any(present):
        for path, payload in output_payloads.items():
            if not path.exists():
                if require_complete:
                    raise Table1RecoveryError(f"missing standard output: {path}")
                continue
            _require_regular(
                path, confinement_root=paths.confinement_root, immutable=True
            )
            if path.read_text(encoding="utf-8") != payload:
                raise Table1RecoveryError(f"standard output drifted: {path}")
    receipt_present = (
        paths.remediation_receipt.exists() or paths.remediation_receipt.is_symlink()
    )
    if receipt_present:
        _require_regular(
            paths.remediation_receipt,
            confinement_root=paths.confinement_root,
            immutable=True,
        )
        receipt = _load_json(
            paths.remediation_receipt, confinement_root=paths.confinement_root
        )
        expected_receipt = _receipt(paths, approval=approval, state=state)
        if receipt != expected_receipt:
            raise Table1RecoveryError("remediation receipt drifted")
        status = "COMPLETE"
    else:
        if require_complete:
            raise Table1RecoveryError("terminal remediation receipt is missing")
        status = "READY" if not any(present) else "PARTIAL_EXACT"
    return {
        "status": status,
        "approval_sha256": _sha256_file(
            paths.approval, confinement_root=paths.confinement_root
        ),
        "image_count": len(state["records"]),
        "shard_count": len(state["shards"]),
        "state_digest": state["state_digest"],
        "candidate_outputs": {
            path.name: hashlib.sha256(payload.encode("utf-8")).hexdigest()
            for path, payload in output_payloads.items()
        },
    }


def approval_verify_only(
    paths: RecoveryPaths,
    *,
    spec: RecoverySpec = PRODUCTION_SPEC,
    expected_anchors: Mapping[str, str] = PRODUCTION_ANCHORS,
) -> dict[str, Any]:
    if paths.approval.exists() or paths.approval.is_symlink():
        approval, state = validate_approval(
            paths,
            spec=spec,
            expected_anchors=expected_anchors,
            allow_outputs=True,
        )
        return {
            "status": "APPROVED",
            "approval_sha256": _sha256_file(
                paths.approval, confinement_root=paths.confinement_root
            ),
            "state_digest": state["state_digest"],
            "approved_utc": approval["approved_utc"],
        }
    candidate = approval_candidate(paths, spec=spec, expected_anchors=expected_anchors)
    return {
        "status": "READY_FOR_APPROVAL",
        "candidate_sha256_with_dry_run_timestamp": hashlib.sha256(
            _canonical_json(candidate).encode("utf-8")
        ).hexdigest(),
        "state_digest": candidate["state_digest"],
    }


def assert_cpu_only_entrypoint() -> None:
    """Fail if a heavyweight image/metric module was imported by this process."""

    forbidden = {"torch", "cv2", "pyiqa", "PIL", "torchvision"}
    loaded = sorted(name for name in forbidden if name in os.sys.modules)
    if loaded:
        raise Table1RecoveryError(
            f"CPU-only recovery process imported forbidden modules: {loaded}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"


__all__ = [
    "APPROVAL_EXECUTE_TOKEN",
    "FINALIZE_EXECUTE_TOKEN",
    "PRODUCTION_ANCHORS",
    "PRODUCTION_SPEC",
    "RecoveryPaths",
    "RecoverySpec",
    "Table1RecoveryError",
    "approval_candidate",
    "approval_verify_only",
    "assert_cpu_only_entrypoint",
    "audit_recovery_state",
    "finalize_recovery",
    "production_paths",
    "publish_approval",
    "validate_approval",
    "verify_recovery",
]
