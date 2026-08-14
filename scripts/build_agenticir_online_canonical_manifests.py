#!/usr/bin/env python3
"""Build metadata-only MiO100 manifests for online BasicSR canonicalization.

No MiO100 image is opened and no image is copied.  The legacy OpenCV canonical
path is validated only as source-protocol metadata and is never emitted into a
derived row.  Consumers must read the official native LQ and canonicalize it
in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scale_canonicalizer import canonicalizer_identity  # noqa: E402
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.io import atomic_write_json, atomic_write_text, iter_jsonl  # noqa: E402
from src.utils.paths import ensure_within, load_resolved_paths  # noqa: E402

SOURCE_TO_OUTPUT = {
    "mio100_test_1440_manifest": "mio100_test_1440_agenticir_online_canonical.jsonl",
    "mio100_group_a_test_manifest": "mio100_group_a_test_640_agenticir_online_canonical.jsonl",
    "mio100_group_b_test_manifest": "mio100_group_b_test_400_agenticir_online_canonical.jsonl",
    "mio100_group_c_test_manifest": "mio100_group_c_test_400_agenticir_online_canonical.jsonl",
}
EXPECTED_ROWS = {
    "mio100_test_1440_manifest": 1440,
    "mio100_group_a_test_manifest": 640,
    "mio100_group_b_test_manifest": 400,
    "mio100_group_c_test_manifest": 400,
}
EXPECTED_GROUP = {
    "mio100_group_a_test_manifest": "A",
    "mio100_group_b_test_manifest": "B",
    "mio100_group_c_test_manifest": "C",
}


class OnlineManifestError(RuntimeError):
    """A source row could accidentally violate the formal native-LQ path."""


def _expected_manifest_hash(config: Mapping[str, Any], source_key: str) -> str:
    identities = config.get("expected_identity", {})
    manifests = identities.get("manifests", {}) if isinstance(identities, Mapping) else {}
    mapping = {
        "mio100_test_1440_manifest": "mio100_test_1440",
        "mio100_group_a_test_manifest": "mio100_group_a_test",
        "mio100_group_b_test_manifest": "mio100_group_b_test",
        "mio100_group_c_test_manifest": "mio100_group_c_test",
    }
    value = manifests.get(mapping[source_key]) if isinstance(manifests, Mapping) else None
    if not isinstance(value, str):
        raise OnlineManifestError(f"missing expected SHA for {source_key}")
    return value


def _transform_row(
    row: dict[str, Any],
    *,
    context: str,
    source_manifest: Path,
    source_sha256: str,
    data_root: Path,
    canonicalizer_sha256: str,
    expected_group: str | None,
) -> dict[str, Any]:
    if row.get("split") != "test":
        raise OnlineManifestError(f"{context}: only formal test rows are allowed")
    group = row.get("group")
    if group not in {"A", "B", "C"}:
        raise OnlineManifestError(f"{context}: invalid MiO100 group {group!r}")
    if expected_group is not None and group != expected_group:
        raise OnlineManifestError(
            f"{context}: expected Group {expected_group}, got {group}"
        )
    degradations = row.get("degradations")
    if not isinstance(degradations, list) or not all(
        isinstance(item, str) for item in degradations
    ):
        raise OnlineManifestError(f"{context}: degradations must be a string list")
    native_text = row.get("native_lq_path")
    gt_text = row.get("gt_path")
    legacy_text = row.get("canonical_lq_path")
    if not all(isinstance(value, str) and value for value in (native_text, gt_text, legacy_text)):
        raise OnlineManifestError(f"{context}: source path fields are malformed")
    native = ensure_within(native_text, data_root / "raw" / "agenticir" / "extracted" / "test")
    gt = ensure_within(gt_text, data_root / "raw" / "agenticir" / "extracted" / "HQ")
    legacy = ensure_within(legacy_text, data_root)
    # Metadata-only existence checks are permitted; no image bytes are opened.
    for label, path in (("native_lq_path", native), ("gt_path", gt)):
        if not path.is_file():
            raise OnlineManifestError(f"{context}: missing {label}: {path}")
    contains_low_resolution = "low resolution" in degradations
    expected_scale = 4 if contains_low_resolution else 1
    if row.get("scale_factor") != expected_scale:
        raise OnlineManifestError(
            f"{context}: scale_factor disagrees with degradation list"
        )
    if contains_low_resolution:
        ensure_within(
            legacy,
            data_root / "processed" / "test" / "canonical_lq",
        )
        if legacy == native:
            raise OnlineManifestError(
                f"{context}: low-resolution legacy/native paths unexpectedly match"
            )
    elif legacy != native:
        raise OnlineManifestError(
            f"{context}: non-low-resolution canonical path must equal native"
        )

    transformed = dict(row)
    transformed.pop("canonical_lq_path", None)
    transformed.update(
        {
            "schema_version": "graphrestore.agenticir_online_canonical.v1",
            "input_path": str(native),
            "input_mode": "agenticir_online_canonical",
            "contains_low_resolution": contains_low_resolution,
            "native_scale": 0.25 if contains_low_resolution else 1.0,
            "online_scale_factor": expected_scale,
            "online_canonicalization": (
                "mioir_basicsr_native_uint8_to_rgb_float_x4"
                if contains_low_resolution
                else "native_uint8_to_rgb_float_identity"
            ),
            "requantize_after_online_resize": False,
            "input_storage_color_order": "BGR",
            "model_input_color_order": "RGB",
            "model_input_dtype": "float32",
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": source_sha256,
            "mioir_matlab_functions_sha256": canonicalizer_sha256,
        }
    )
    return transformed


def _jsonl_payload(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )


def _write_idempotent(path: Path, payload: str, *, force: bool) -> str:
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == payload:
            return "unchanged"
        if not force:
            raise OnlineManifestError(
                f"refusing to replace a different derived manifest: {path}; "
                "review it and pass --force"
            )
    atomic_write_text(path, payload)
    return "written"


def build_manifests(
    config_path: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config = load_resolved_paths(config_path)
    data_root = Path(config["data_root"]).resolve()
    mioir_repo = Path(config["mioir_repo"]).resolve()
    identity = canonicalizer_identity(mioir_repo)
    canonicalizer_sha = identity["sha256"]
    output_directory = Path(output_root).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, Any] = {
        "schema_version": "graphrestore.agenticir_online_canonical.inventory.v1",
        "canonicalizer": identity,
        "manifests": {},
    }
    for source_key, output_name in SOURCE_TO_OUTPUT.items():
        source = Path(config[source_key]).resolve()
        expected_sha = _expected_manifest_hash(config, source_key)
        source_sha = sha256_file(source)
        if source_sha != expected_sha:
            raise OnlineManifestError(
                f"{source_key} SHA mismatch: expected {expected_sha}, got {source_sha}"
            )
        rows: list[dict[str, Any]] = []
        for line_number, row in iter_jsonl(source):
            rows.append(
                _transform_row(
                    row,
                    context=f"{source}:{line_number}",
                    source_manifest=source,
                    source_sha256=source_sha,
                    data_root=data_root,
                    canonicalizer_sha256=canonicalizer_sha,
                    expected_group=EXPECTED_GROUP.get(source_key),
                )
            )
        if len(rows) != EXPECTED_ROWS[source_key]:
            raise OnlineManifestError(
                f"{source_key}: expected {EXPECTED_ROWS[source_key]} rows, got {len(rows)}"
            )
        output = output_directory / output_name
        action = _write_idempotent(output, _jsonl_payload(rows), force=force)
        inventory["manifests"][output_name] = {
            "path": str(output),
            "rows": len(rows),
            "sha256": sha256_file(output),
            "source_path": str(source),
            "source_sha256": source_sha,
            "action": action,
        }
    inventory_path = output_directory / "agenticir_online_canonical_inventory.json"
    atomic_write_json(inventory_path, inventory)
    inventory["inventory_path"] = str(inventory_path)
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "resolved_paths.yaml",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "manifests"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_manifests(args.config, args.output_root, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
