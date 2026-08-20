"""Standard-library-only provenance for the formal MiO100 evaluation.

This module intentionally imports neither torch nor OpenCV.  The inventory
phase is allowed to stream bytes for SHA256/stat identity only; it must never
decode an image or initialize CUDA.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
AUTHORIZATION_SCHEMA = "graphrestore-formal-mio100-approval-v1"
FORMAL_DATA_INVENTORY_SCHEMA = "graphrestore-formal-mio100-data-inventory-v1"
FORMAL_MANIFEST_FILENAME = "mio100_test_1440_agenticir_online_canonical.jsonl"
FORMAL_MANIFEST_SHA256 = (
    "83fb90dfa121681123f55e73df32eb6c1bc37e685c0e27ae07ad7e59a687a7f5"
)
FORMAL_METHOD_NAME = "graphrestore_v7_1_stage4_step040000"
FORMAL_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/aaa/graphrestore/artifacts/formal_mio100/"
    "graphrestore_v7_1_stage4_step040000"
)
FORMAL_DATA_INVENTORY_PATH = Path(
    "/root/autodl-tmp/aaa/graphrestore/artifacts/formal_mio100/"
    "formal_data_inventory.json"
)
FORMAL_APPROVAL_PATH = Path(
    "/root/autodl-tmp/aaa/graphrestore/artifacts/approvals/FORMAL_MIO100_APPROVED.json"
)
FORMAL_AUTHORIZATION_PROTOCOL_PATH = Path(
    "/root/autodl-tmp/aaa/graphrestore/reports/FORMAL_MIO100_AUTHORIZATION_PROTOCOL.md"
)
FORMAL_AUTHORIZATION_PROTOCOL_SHA256 = (
    "3bb7bea0e26709d284b3efc817f6f79ce0114eb413b7aa9463d49a9528475eb7"
)
FORMAL_ROW_COUNT = 1_440
FORMAL_NATIVE_REFERENCE_COUNT = 1_440
FORMAL_TARGET_REFERENCE_COUNT = 1_440
FORMAL_UNIQUE_NATIVE_COUNT = 1_440
FORMAL_UNIQUE_TARGET_COUNT = 100
FORMAL_UNIQUE_FILE_COUNT = 1_540
FORMAL_GROUP_COUNTS: Mapping[str, int] = {"A": 640, "B": 400, "C": 400}

OFFICIAL_GROUPS: Mapping[str, tuple[str, ...]] = {
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
FORMAL_COMBINATION_COUNTS: Mapping[str, int] = {
    combination: (80 if group == "A" else 100)
    for group, combinations in OFFICIAL_GROUPS.items()
    for combination in combinations
}

REQUIRED_AUTHORIZATION_BINDINGS = (
    "stage4_complete",
    "stage4_checkpoint",
    "stage4_config",
    "stage4_run_contract",
    "stage4_validation",
    "stage4_calibration_history",
    "stage4_report",
    "stage4_diagnostics_json",
    "stage4_diagnostics_report",
    "thresholds",
    "pair_prior",
    "global_priority",
    "formal_manifest",
    "manifest_inventory",
    "formal_data_inventory",
    "metric_parity_summary",
    "metric_protocol",
    "evaluator_module",
    "evaluator_cli",
    "formal_authorization_protocol",
    "canonicalizer_source",
    "mioir_matlab_functions",
    "agenticir_scorer",
    "agenticir_compute_scores",
    "agenticir_compare_methods",
    "table1_scorer_module",
    "table1_scorer_cli",
    "metric_weight_inventory",
)

AUTHORIZATION_KEYS = {
    "schema_version",
    "kind",
    "protocol_id",
    "approved",
    "formal_mio100_authorized",
    "one_shot",
    "inference_only",
    "authorized_groups",
    "manifest_row_count",
    "method_name",
    "shard_count",
    "output_root",
    "approved_utc",
    "restrictions",
    "bindings",
}
AUTHORIZATION_RESTRICTIONS = {
    "task_label_routing": False,
    "tta": False,
    "model_soup": False,
    "threshold_tuning": False,
    "result_driven_rerun": False,
    "overwrite": False,
}
INVENTORY_KEYS = {
    "schema_version",
    "protocol_id",
    "created_utc",
    "manifest",
    "authorization_protocol",
    "generator_source",
    "native_reference_count",
    "target_reference_count",
    "unique_native_count",
    "unique_target_count",
    "unique_file_count",
    "group_counts",
    "combination_counts",
    "rows_digest",
    "files_digest",
    "rows",
    "files",
}
INVENTORY_ROW_KEYS = {
    "index",
    "sample_id",
    "row_sha256",
    "native_lq_path",
    "native_lq_sha256",
    "target_path",
    "target_sha256",
}
INVENTORY_FILE_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "mode",
    "device",
    "inode",
    "roles",
    "reference_count",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FormalInventoryError(RuntimeError):
    """A formal inventory/authorization invariant was violated."""


@dataclass(frozen=True)
class InventoryFileIdentity:
    path: Path
    sha256: str
    size_bytes: int
    mode: int
    device: int
    inode: int
    roles: tuple[str, ...]
    reference_count: int


@dataclass(frozen=True)
class InventoryRowIdentity:
    index: int
    sample_id: str
    row_sha256: str
    native_lq_path: Path
    native_lq_sha256: str
    target_path: Path
    target_sha256: str


@dataclass(frozen=True)
class FormalDataInventory:
    path: Path
    sha256: str
    manifest_sha256: str
    rows_digest: str
    files_digest: str
    rows: tuple[InventoryRowIdentity, ...]
    files: Mapping[Path, InventoryFileIdentity]


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _validate_utc(value: object, *, field: str) -> str:
    if not isinstance(value, str) or "T" not in value or not value.endswith("Z"):
        raise FormalInventoryError(f"{field} must be an RFC3339 UTC timestamp")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalInventoryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalInventoryError(f"could not read strict JSON {path}: {exc}") from exc


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalInventoryError(f"{field} must be a mapping")
    return value


def canonical_regular_file(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise FormalInventoryError(f"{field} must be absolute")
    if candidate.is_symlink():
        raise FormalInventoryError(f"{field} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FormalInventoryError(f"missing {field}: {candidate}") from exc
    if resolved != candidate or not resolved.is_file():
        raise FormalInventoryError(
            f"{field} must be a canonical regular file: {candidate}"
        )
    return resolved


def require_read_only(path: Path, *, field: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise FormalInventoryError(f"{field} must be read-only: {path}")


def require_mode_0444(path: Path, *, field: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o444:
        raise FormalInventoryError(f"{field} must have exact mode 0444, got {mode:04o}")


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_mode,
        value.st_size,
        value.st_dev,
        value.st_ino,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stream_file_identity(
    path: str | Path,
    *,
    field: str,
    opener: Callable[..., int] = os.open,
) -> dict[str, int | str]:
    """Hash a canonical file using only streaming bytes and stable fstat data."""

    canonical = canonical_regular_file(path, field=field)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = opener(canonical, flags)
    except OSError as exc:
        raise FormalInventoryError(
            f"could not open {field}: {canonical}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FormalInventoryError(f"{field} is not regular: {canonical}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = canonical.stat()
    if _stat_signature(before) != _stat_signature(after) or (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
    ) != (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mode,
    ):
        raise FormalInventoryError(f"{field} changed while hashing: {canonical}")
    return {
        "path": str(canonical),
        "sha256": digest.hexdigest(),
        "size_bytes": int(after.st_size),
        "mode": int(stat.S_IMODE(after.st_mode)),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
    }


def sha256_file(path: str | Path, *, field: str = "file") -> str:
    return str(stream_file_identity(path, field=field)["sha256"])


def _manifest_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise FormalInventoryError(
                        f"{path}:{line_number}: blank JSONL record"
                    )
                try:
                    row = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
                except json.JSONDecodeError as exc:
                    raise FormalInventoryError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                rows.append(_mapping(row, field=f"{path}:{line_number}"))
    except (OSError, UnicodeError) as exc:
        raise FormalInventoryError(f"could not stream manifest {path}: {exc}") from exc
    return tuple(rows)


def _validated_manifest_metadata(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_row_count: int,
    expected_group_counts: Mapping[str, int],
    expected_combination_counts: Mapping[str, int],
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    if manifest_path.name != FORMAL_MANIFEST_FILENAME:
        raise FormalInventoryError("formal inventory requires the frozen manifest name")
    identity = stream_file_identity(manifest_path, field="formal manifest")
    actual_sha = str(identity["sha256"])
    if actual_sha != expected_manifest_sha256:
        raise FormalInventoryError("formal manifest SHA256 drifted")
    rows = _manifest_rows(manifest_path)
    if len(rows) != expected_row_count:
        raise FormalInventoryError(
            f"formal manifest row count drifted: {len(rows)} != {expected_row_count}"
        )
    sample_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        context = f"manifest row {index}"
        if (
            row.get("schema_version") != "graphrestore.agenticir_online_canonical.v1"
            or row.get("input_mode") != "agenticir_online_canonical"
            or row.get("source") != "AgenticIR"
            or row.get("split") != "test"
        ):
            raise FormalInventoryError(f"{context}: protocol metadata drifted")
        sample_id = row.get("sample_id")
        group = row.get("group")
        degradations = row.get("degradations")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise FormalInventoryError(f"{context}: invalid/duplicate sample_id")
        if group not in expected_group_counts:
            raise FormalInventoryError(f"{context}: invalid group")
        if not isinstance(degradations, list) or not all(
            isinstance(item, str) and item for item in degradations
        ):
            raise FormalInventoryError(f"{context}: invalid degradations")
        combination = "+".join(degradations)
        if combination not in expected_combination_counts:
            raise FormalInventoryError(f"{context}: invalid combination")
        native = row.get("native_lq_path")
        target = row.get("gt_path")
        if (
            not isinstance(native, str)
            or not isinstance(target, str)
            or not Path(native).is_absolute()
            or not Path(target).is_absolute()
            or row.get("input_path") != native
        ):
            raise FormalInventoryError(f"{context}: invalid native/target path")
        sample_ids.add(sample_id)
        group_counts[str(group)] += 1
        combination_counts[combination] += 1
    if dict(group_counts) != dict(expected_group_counts):
        raise FormalInventoryError(f"manifest group counts drifted: {group_counts}")
    if dict(combination_counts) != dict(expected_combination_counts):
        raise FormalInventoryError(
            f"manifest combination counts drifted: {combination_counts}"
        )
    return rows, actual_sha


def build_formal_data_inventory(
    manifest: str | Path,
    *,
    authorization_protocol: str | Path,
    generator_source: str | Path = Path(__file__).resolve(),
    expected_manifest_sha256: str = FORMAL_MANIFEST_SHA256,
    expected_authorization_protocol_sha256: str = (
        FORMAL_AUTHORIZATION_PROTOCOL_SHA256
    ),
    expected_row_count: int = FORMAL_ROW_COUNT,
    expected_group_counts: Mapping[str, int] = FORMAL_GROUP_COUNTS,
    expected_combination_counts: Mapping[str, int] = FORMAL_COMBINATION_COUNTS,
    expected_unique_native_count: int = FORMAL_UNIQUE_NATIVE_COUNT,
    expected_unique_target_count: int = FORMAL_UNIQUE_TARGET_COUNT,
    expected_unique_file_count: int = FORMAL_UNIQUE_FILE_COUNT,
) -> Mapping[str, Any]:
    """Build the immutable inventory payload without decoding any image."""

    manifest_path = canonical_regular_file(manifest, field="formal manifest")
    protocol_path = canonical_regular_file(
        authorization_protocol, field="formal authorization protocol"
    )
    generator_path = canonical_regular_file(
        generator_source, field="formal inventory generator"
    )
    protocol_sha = sha256_file(protocol_path, field="authorization protocol")
    if protocol_sha != expected_authorization_protocol_sha256:
        raise FormalInventoryError("formal authorization protocol SHA256 drifted")
    generator_sha = sha256_file(generator_path, field="inventory generator")
    manifest_rows, manifest_sha = _validated_manifest_metadata(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_row_count=expected_row_count,
        expected_group_counts=expected_group_counts,
        expected_combination_counts=expected_combination_counts,
    )

    native_paths = [Path(str(row["native_lq_path"])) for row in manifest_rows]
    target_paths = [Path(str(row["gt_path"])) for row in manifest_rows]
    unique_native = set(native_paths)
    unique_target = set(target_paths)
    if len(unique_native) != expected_unique_native_count:
        raise FormalInventoryError("formal native LQ unique count drifted")
    if len(unique_target) != expected_unique_target_count:
        raise FormalInventoryError("formal GT unique count drifted")
    if unique_native & unique_target:
        raise FormalInventoryError("native LQ and GT inventories overlap")
    all_paths = unique_native | unique_target
    if len(all_paths) != expected_unique_file_count:
        raise FormalInventoryError("formal total unique file count drifted")

    native_references = Counter(native_paths)
    target_references = Counter(target_paths)
    file_rows: list[dict[str, Any]] = []
    file_identity_by_path: dict[Path, Mapping[str, int | str]] = {}
    for file_path in sorted(all_paths, key=str):
        identity = stream_file_identity(file_path, field="formal MiO100 data file")
        file_identity_by_path[file_path] = identity
        roles = []
        reference_count = 0
        if file_path in native_references:
            roles.append("native_lq")
            reference_count += native_references[file_path]
        if file_path in target_references:
            roles.append("target")
            reference_count += target_references[file_path]
        file_rows.append(
            {
                **identity,
                "roles": roles,
                "reference_count": reference_count,
            }
        )

    row_rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows):
        native_path = Path(str(row["native_lq_path"]))
        target_path = Path(str(row["gt_path"]))
        row_rows.append(
            {
                "index": index,
                "sample_id": str(row["sample_id"]),
                "row_sha256": hashlib.sha256(_canonical_json_bytes(row)).hexdigest(),
                "native_lq_path": str(native_path),
                "native_lq_sha256": str(file_identity_by_path[native_path]["sha256"]),
                "target_path": str(target_path),
                "target_sha256": str(file_identity_by_path[target_path]["sha256"]),
            }
        )
    return {
        "schema_version": FORMAL_DATA_INVENTORY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": _utc_now(),
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "row_count": len(row_rows),
        },
        "authorization_protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha,
        },
        "generator_source": {
            "path": str(generator_path),
            "sha256": generator_sha,
        },
        "native_reference_count": len(native_paths),
        "target_reference_count": len(target_paths),
        "unique_native_count": len(unique_native),
        "unique_target_count": len(unique_target),
        "unique_file_count": len(file_rows),
        "group_counts": dict(expected_group_counts),
        "combination_counts": dict(expected_combination_counts),
        "rows_digest": _sha256_json(row_rows),
        "files_digest": _sha256_json(file_rows),
        "rows": row_rows,
        "files": file_rows,
    }


def write_new_read_only_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    if not destination.is_absolute():
        raise FormalInventoryError("immutable output path must be absolute")
    for ancestor in (destination, *destination.parents):
        if ancestor.is_symlink():
            raise FormalInventoryError(f"output path crosses symlink: {ancestor}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise FormalInventoryError(
            f"refusing to overwrite immutable output: {destination}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_pretty_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o444)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _validate_file_row(
    raw: object,
    *,
    verify_file_bytes: bool,
) -> InventoryFileIdentity:
    row = _mapping(raw, field="formal inventory file row")
    if set(row) != INVENTORY_FILE_KEYS:
        raise FormalInventoryError("formal inventory file fields drifted")
    path_value = row.get("path")
    digest = row.get("sha256")
    roles = row.get("roles")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not _is_sha256(digest)
        or not isinstance(roles, list)
        or roles not in (["native_lq"], ["target"])
    ):
        raise FormalInventoryError("formal inventory file row is malformed")
    numeric = {}
    for key in ("size_bytes", "mode", "device", "inode", "reference_count"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FormalInventoryError(f"formal inventory file has invalid {key}")
        numeric[key] = value
    path = Path(path_value)
    if verify_file_bytes:
        actual = stream_file_identity(path, field="inventoried MiO100 data file")
        for key in ("sha256", "size_bytes", "mode", "device", "inode"):
            if actual[key] != row[key]:
                raise FormalInventoryError(
                    f"inventoried MiO100 file identity drifted at {path}: {key}"
                )
    return InventoryFileIdentity(
        path=path,
        sha256=str(digest),
        size_bytes=numeric["size_bytes"],
        mode=numeric["mode"],
        device=numeric["device"],
        inode=numeric["inode"],
        roles=tuple(roles),
        reference_count=numeric["reference_count"],
    )


def load_formal_data_inventory(
    path: str | Path,
    *,
    expected_manifest_path: str | Path,
    expected_manifest_sha256: str = FORMAL_MANIFEST_SHA256,
    expected_authorization_protocol_path: str | Path = (
        FORMAL_AUTHORIZATION_PROTOCOL_PATH
    ),
    expected_authorization_protocol_sha256: str = (
        FORMAL_AUTHORIZATION_PROTOCOL_SHA256
    ),
    expected_generator_source: str | Path = Path(__file__).resolve(),
    expected_row_count: int = FORMAL_ROW_COUNT,
    expected_group_counts: Mapping[str, int] = FORMAL_GROUP_COUNTS,
    expected_combination_counts: Mapping[str, int] = FORMAL_COMBINATION_COUNTS,
    expected_unique_native_count: int = FORMAL_UNIQUE_NATIVE_COUNT,
    expected_unique_target_count: int = FORMAL_UNIQUE_TARGET_COUNT,
    expected_unique_file_count: int = FORMAL_UNIQUE_FILE_COUNT,
    verify_file_bytes: bool = True,
) -> FormalDataInventory:
    inventory_path = canonical_regular_file(path, field="formal data inventory")
    require_mode_0444(inventory_path, field="formal data inventory")
    inventory_sha = sha256_file(inventory_path, field="formal data inventory")
    payload = _mapping(_load_json(inventory_path), field="formal data inventory")
    if set(payload) != INVENTORY_KEYS:
        raise FormalInventoryError("formal data inventory fields drifted")
    if (
        payload.get("schema_version") != FORMAL_DATA_INVENTORY_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
    ):
        raise FormalInventoryError("formal data inventory protocol drifted")
    _validate_utc(payload.get("created_utc"), field="inventory.created_utc")
    manifest_path = canonical_regular_file(
        expected_manifest_path, field="expected formal manifest"
    )
    protocol_path = canonical_regular_file(
        expected_authorization_protocol_path,
        field="expected authorization protocol",
    )
    generator_path = canonical_regular_file(
        expected_generator_source, field="expected inventory generator"
    )
    expected_bound = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": expected_manifest_sha256,
            "row_count": expected_row_count,
        },
        "authorization_protocol": {
            "path": str(protocol_path),
            "sha256": expected_authorization_protocol_sha256,
        },
        "generator_source": {
            "path": str(generator_path),
            "sha256": sha256_file(generator_path, field="inventory generator"),
        },
        "native_reference_count": expected_row_count,
        "target_reference_count": expected_row_count,
        "unique_native_count": expected_unique_native_count,
        "unique_target_count": expected_unique_target_count,
        "unique_file_count": expected_unique_file_count,
        "group_counts": dict(expected_group_counts),
        "combination_counts": dict(expected_combination_counts),
    }
    for key, value in expected_bound.items():
        if payload.get(key) != value:
            raise FormalInventoryError(f"formal data inventory drifted at {key}")
    if sha256_file(protocol_path, field="authorization protocol") != (
        expected_authorization_protocol_sha256
    ):
        raise FormalInventoryError("authorization protocol changed after inventory")
    manifest_rows, actual_manifest_sha = _validated_manifest_metadata(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_row_count=expected_row_count,
        expected_group_counts=expected_group_counts,
        expected_combination_counts=expected_combination_counts,
    )
    if actual_manifest_sha != expected_manifest_sha256:
        raise FormalInventoryError("formal manifest changed after inventory")

    raw_files = payload.get("files")
    raw_rows = payload.get("rows")
    if not isinstance(raw_files, list) or not isinstance(raw_rows, list):
        raise FormalInventoryError("formal inventory files/rows must be lists")
    if payload.get("files_digest") != _sha256_json(raw_files):
        raise FormalInventoryError("formal inventory files digest drifted")
    if payload.get("rows_digest") != _sha256_json(raw_rows):
        raise FormalInventoryError("formal inventory rows digest drifted")
    if (
        len(raw_files) != expected_unique_file_count
        or len(raw_rows) != expected_row_count
    ):
        raise FormalInventoryError("formal inventory files/rows count drifted")

    files: dict[Path, InventoryFileIdentity] = {}
    file_path_order = []
    for raw in raw_files:
        raw_file = _mapping(raw, field="formal inventory file row")
        path_value = raw_file.get("path")
        if not isinstance(path_value, str):
            raise FormalInventoryError("formal inventory file path is malformed")
        file_path_order.append(path_value)
    if file_path_order != sorted(file_path_order):
        raise FormalInventoryError("formal inventory files are not path-sorted")
    for raw in raw_files:
        identity = _validate_file_row(raw, verify_file_bytes=verify_file_bytes)
        if identity.path in files:
            raise FormalInventoryError("duplicate formal inventory file path")
        files[identity.path] = identity

    rows: list[InventoryRowIdentity] = []
    sample_ids: set[str] = set()
    native_paths: set[Path] = set()
    target_paths: set[Path] = set()
    native_references: Counter[Path] = Counter()
    target_references: Counter[Path] = Counter()
    for expected_index, (raw, manifest_row) in enumerate(
        zip(raw_rows, manifest_rows, strict=True)
    ):
        row = _mapping(raw, field="formal inventory row")
        if set(row) != INVENTORY_ROW_KEYS:
            raise FormalInventoryError("formal inventory row fields drifted")
        if row.get("index") != expected_index:
            raise FormalInventoryError(
                "formal inventory row indices are not contiguous"
            )
        sample_id = row.get("sample_id")
        native_text = row.get("native_lq_path")
        target_text = row.get("target_path")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not isinstance(native_text, str)
            or not isinstance(target_text, str)
            or not _is_sha256(row.get("row_sha256"))
            or not _is_sha256(row.get("native_lq_sha256"))
            or not _is_sha256(row.get("target_sha256"))
        ):
            raise FormalInventoryError("formal inventory row is malformed")
        native_path = Path(native_text)
        target_path = Path(target_text)
        expected_row_sha = hashlib.sha256(
            _canonical_json_bytes(manifest_row)
        ).hexdigest()
        if (
            sample_id != manifest_row.get("sample_id")
            or native_text != manifest_row.get("native_lq_path")
            or target_text != manifest_row.get("gt_path")
            or row.get("row_sha256") != expected_row_sha
        ):
            raise FormalInventoryError(
                f"formal inventory/manifest row binding drifted at {expected_index}"
            )
        native_file = files.get(native_path)
        target_file = files.get(target_path)
        if (
            native_file is None
            or target_file is None
            or native_file.roles != ("native_lq",)
            or target_file.roles != ("target",)
            or native_file.sha256 != row["native_lq_sha256"]
            or target_file.sha256 != row["target_sha256"]
        ):
            raise FormalInventoryError("formal inventory row/file binding drifted")
        sample_ids.add(sample_id)
        native_paths.add(native_path)
        target_paths.add(target_path)
        native_references[native_path] += 1
        target_references[target_path] += 1
        rows.append(
            InventoryRowIdentity(
                index=expected_index,
                sample_id=sample_id,
                row_sha256=str(row["row_sha256"]),
                native_lq_path=native_path,
                native_lq_sha256=str(row["native_lq_sha256"]),
                target_path=target_path,
                target_sha256=str(row["target_sha256"]),
            )
        )
    if (
        len(native_paths) != expected_unique_native_count
        or len(target_paths) != expected_unique_target_count
        or native_paths & target_paths
        or sum(native_references.values()) != expected_row_count
        or sum(target_references.values()) != expected_row_count
    ):
        raise FormalInventoryError("formal inventory native/target partition drifted")
    for file_path, identity in files.items():
        expected_references = (
            native_references[file_path]
            if identity.roles == ("native_lq",)
            else target_references[file_path]
        )
        if identity.reference_count != expected_references:
            raise FormalInventoryError("formal inventory reference count drifted")
    return FormalDataInventory(
        path=inventory_path,
        sha256=inventory_sha,
        manifest_sha256=expected_manifest_sha256,
        rows_digest=str(payload["rows_digest"]),
        files_digest=str(payload["files_digest"]),
        rows=tuple(rows),
        files=files,
    )


def assert_no_gpu_compute_processes(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FormalInventoryError(f"could not prove GPU release: {exc}") from exc
    if result.returncode != 0:
        raise FormalInventoryError(
            f"could not prove GPU release: nvidia-smi exit={result.returncode}"
        )
    pids = set()
    for raw in result.stdout.splitlines():
        value = raw.strip()
        if not value or "No running processes" in value:
            continue
        if not value.isdigit():
            raise FormalInventoryError(f"unexpected nvidia-smi PID row: {value!r}")
        pids.add(int(value))
    if pids:
        raise FormalInventoryError(f"GPU is not released: compute PIDs={sorted(pids)}")


def authorization_binding_paths(
    project_root: str | Path,
    *,
    manifest: str | Path,
    formal_data_inventory: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    stage4_complete: str | Path,
    thresholds: str | Path,
    pair_prior: str | Path,
    global_priority: str | Path,
) -> Mapping[str, Path]:
    root = Path(project_root).resolve(strict=True)
    return {
        "stage4_complete": Path(stage4_complete).resolve(strict=False),
        "stage4_checkpoint": Path(checkpoint).resolve(strict=False),
        "stage4_config": Path(config).resolve(strict=False),
        "stage4_run_contract": root / "artifacts/checkpoints/stage4/run_contract.json",
        "stage4_validation": root
        / "artifacts/checkpoints/stage4/validation_latest.json",
        "stage4_calibration_history": root
        / "artifacts/metrics/stage4_calibration_history.csv",
        "stage4_report": root / "reports/STAGE4_E2E.md",
        "stage4_diagnostics_json": root / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.json",
        "stage4_diagnostics_report": root / "reports/GUARD_AND_MISUSE_DIAGNOSTICS.md",
        "thresholds": Path(thresholds).resolve(strict=False),
        "pair_prior": Path(pair_prior).resolve(strict=False),
        "global_priority": Path(global_priority).resolve(strict=False),
        "formal_manifest": Path(manifest).resolve(strict=False),
        "manifest_inventory": root
        / "manifests/agenticir_online_canonical_inventory.json",
        "formal_data_inventory": Path(formal_data_inventory).resolve(strict=False),
        "metric_parity_summary": root / "artifacts/metrics/metric_parity_summary.json",
        "metric_protocol": root / "reports/METRIC_PROTOCOL.md",
        "evaluator_module": root / "src/evaluation/mio100.py",
        "evaluator_cli": root / "scripts/eval_mio100.py",
        "formal_authorization_protocol": FORMAL_AUTHORIZATION_PROTOCOL_PATH,
        "canonicalizer_source": root / "src/data/scale_canonicalizer.py",
        "mioir_matlab_functions": Path(
            "/root/autodl-tmp/graph/upstream/MiOIR/basicsr/utils/matlab_functions.py"
        ),
        "agenticir_scorer": Path(
            "/root/autodl-tmp/graph/upstream/AgenticIR/utils/scorer.py"
        ),
        "agenticir_compute_scores": Path(
            "/root/autodl-tmp/graph/upstream/AgenticIR/eval/compute_scores.py"
        ),
        "agenticir_compare_methods": Path(
            "/root/autodl-tmp/graph/upstream/AgenticIR/eval/compare_methods.py"
        ),
        "table1_scorer_module": root / "src/evaluation/agenticir_table1.py",
        "table1_scorer_cli": root / "scripts/score_agenticir_table1.py",
        "metric_weight_inventory": root
        / "artifacts/formal_mio100/cache/weights_lock.json",
    }


def validate_stage4_ready_without_torch(
    complete_path: str | Path,
    *,
    checkpoint_path: str | Path,
    diagnostics_path: str | Path,
) -> Mapping[str, Any]:
    complete_file = canonical_regular_file(complete_path, field="Stage4 completion")
    checkpoint = canonical_regular_file(checkpoint_path, field="Stage4 best EMA")
    diagnostics_file = canonical_regular_file(
        diagnostics_path, field="Stage4 diagnostics"
    )
    payload = _mapping(_load_json(complete_file), field="Stage4 completion")
    checkpoint_sha = sha256_file(checkpoint, field="Stage4 best EMA")
    diagnostics_sha = sha256_file(diagnostics_file, field="Stage4 diagnostics")
    best = _mapping(payload.get("best_score"), field="Stage4 best score")
    latest = _mapping(payload.get("latest_score"), field="Stage4 latest score")
    if (
        payload.get("schema_version") != "graphrestore-stage4-runtime-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("step") != 40_000
        or best.get("step") != 40_000
        or latest.get("step") != 40_000
        or payload.get("formal_mio100_started") is not False
        or payload.get("waiting_for") != "new_user_authorization_for_formal_mio100"
        or payload.get("best_ema_path") != str(checkpoint)
        or payload.get("best_ema_sha256") != checkpoint_sha
        or payload.get("diagnostics_json") != str(diagnostics_file)
        or payload.get("diagnostics_json_sha256") != diagnostics_sha
        or payload.get("diagnostics_selected_best_ema_sha256") != checkpoint_sha
    ):
        raise FormalInventoryError("Stage4 is not ready for formal authorization")
    diagnostics = _mapping(
        _load_json(diagnostics_file), field="Stage4 zero-training diagnostics"
    )
    compiler_modes = _mapping(
        diagnostics.get("compiler_modes"), field="compiler diagnostics"
    )
    guard_modes = _mapping(diagnostics.get("guard_modes"), field="guard diagnostics")
    if (
        diagnostics.get("schema_version")
        != "graphrestore-stage4-zero-training-diagnostics-v1"
        or diagnostics.get("protocol_id") != PROTOCOL_ID
        or diagnostics.get("selected_best_ema_path") != str(checkpoint)
        or diagnostics.get("selected_best_ema_sha256") != checkpoint_sha
        or diagnostics.get("optimizer_updates") != 0
        or diagnostics.get("model_ema_rng_unchanged") is not True
        or set(compiler_modes)
        != {"full_partial_order", "forced_total_order", "parallel_only"}
        or set(guard_modes) != {"predicted_spatial", "global_mean", "all_one"}
    ):
        raise FormalInventoryError("Stage4 diagnostics are not authorization-ready")
    for name, raw in (*compiler_modes.items(), *guard_modes.items()):
        mode = _mapping(raw, field=f"diagnostic mode {name}")
        peak = mode.get("peak_reserved_fraction")
        if (
            mode.get("image_count") != 1_600
            or isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(float(peak))
            or not 0.0 <= float(peak) < 0.90
        ):
            raise FormalInventoryError(f"Stage4 diagnostic mode {name} is invalid")
    return payload


def build_formal_authorization_payload(
    binding_paths: Mapping[str, str | Path],
    *,
    approved_utc: str | None = None,
) -> Mapping[str, Any]:
    if set(binding_paths) != set(REQUIRED_AUTHORIZATION_BINDINGS):
        raise FormalInventoryError("formal authorization binding keys drifted")
    bindings = {}
    for name in REQUIRED_AUTHORIZATION_BINDINGS:
        path = canonical_regular_file(binding_paths[name], field=f"binding {name}")
        bindings[name] = {
            "path": str(path),
            "sha256": sha256_file(path, field=f"binding {name}"),
        }
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "kind": "formal_mio100_approval",
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "formal_mio100_authorized": True,
        "one_shot": True,
        "inference_only": True,
        "authorized_groups": ["A", "B", "C"],
        "manifest_row_count": FORMAL_ROW_COUNT,
        "method_name": FORMAL_METHOD_NAME,
        "shard_count": 1,
        "output_root": str(FORMAL_OUTPUT_ROOT),
        "approved_utc": approved_utc or _utc_now(),
        "restrictions": dict(AUTHORIZATION_RESTRICTIONS),
        "bindings": bindings,
    }


def validate_lightweight_authorization(
    path: str | Path,
    *,
    expected_binding_paths: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    authorization = canonical_regular_file(path, field="formal authorization")
    require_mode_0444(authorization, field="formal authorization")
    payload = _mapping(_load_json(authorization), field="formal authorization")
    if set(payload) != AUTHORIZATION_KEYS:
        raise FormalInventoryError("formal authorization fields drifted")
    expected = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "kind": "formal_mio100_approval",
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "formal_mio100_authorized": True,
        "one_shot": True,
        "inference_only": True,
        "authorized_groups": ["A", "B", "C"],
        "manifest_row_count": FORMAL_ROW_COUNT,
        "method_name": FORMAL_METHOD_NAME,
        "shard_count": 1,
        "output_root": str(FORMAL_OUTPUT_ROOT),
        "restrictions": dict(AUTHORIZATION_RESTRICTIONS),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise FormalInventoryError("formal authorization scope drifted")
    _validate_utc(payload.get("approved_utc"), field="authorization.approved_utc")
    raw_bindings = _mapping(payload.get("bindings"), field="authorization bindings")
    if set(raw_bindings) != set(REQUIRED_AUTHORIZATION_BINDINGS):
        raise FormalInventoryError("formal authorization binding keys drifted")
    expected_paths = {
        name: Path(value).resolve(strict=False)
        for name, value in expected_binding_paths.items()
    }
    if set(expected_paths) != set(REQUIRED_AUTHORIZATION_BINDINGS):
        raise FormalInventoryError("expected binding path keys drifted")
    for name in REQUIRED_AUTHORIZATION_BINDINGS:
        raw = _mapping(raw_bindings[name], field=f"binding {name}")
        if set(raw) != {"path", "sha256"}:
            raise FormalInventoryError(f"binding {name} fields drifted")
        bound = canonical_regular_file(raw.get("path"), field=f"binding {name}")
        if bound != expected_paths[name] or not _is_sha256(raw.get("sha256")):
            raise FormalInventoryError(f"binding {name} path/hash is malformed")
        if sha256_file(bound, field=f"binding {name}") != raw["sha256"]:
            raise FormalInventoryError(f"binding {name} SHA256 drifted")
    return payload


def assert_standard_library_only() -> None:
    import sys

    forbidden = sorted(name for name in ("cv2", "torch") if name in sys.modules)
    if forbidden:
        raise FormalInventoryError(
            f"hash-only authorization process imported forbidden modules: {forbidden}"
        )


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "FORMAL_APPROVAL_PATH",
    "FORMAL_AUTHORIZATION_PROTOCOL_PATH",
    "FORMAL_AUTHORIZATION_PROTOCOL_SHA256",
    "FORMAL_DATA_INVENTORY_PATH",
    "FORMAL_DATA_INVENTORY_SCHEMA",
    "FORMAL_MANIFEST_FILENAME",
    "FORMAL_MANIFEST_SHA256",
    "FORMAL_METHOD_NAME",
    "FORMAL_OUTPUT_ROOT",
    "FormalDataInventory",
    "FormalInventoryError",
    "InventoryFileIdentity",
    "InventoryRowIdentity",
    "REQUIRED_AUTHORIZATION_BINDINGS",
    "assert_no_gpu_compute_processes",
    "assert_standard_library_only",
    "authorization_binding_paths",
    "build_formal_authorization_payload",
    "build_formal_data_inventory",
    "canonical_regular_file",
    "load_formal_data_inventory",
    "require_mode_0444",
    "sha256_file",
    "stream_file_identity",
    "validate_lightweight_authorization",
    "validate_stage4_ready_without_torch",
    "write_new_read_only_json",
]
