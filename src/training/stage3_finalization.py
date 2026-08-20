"""Permanent Stage3-extension revocation and finalize-only authorization.

The revocation marker is deliberately both a tombstone and an authorization
root.  The Stage3 trainer treats *any* directory entry at its canonical path as
an unconditional stop signal.  Consumers that need the finalize-only grant use
the stricter validator below, which re-hashes every bound file and every
semantic source before returning evidence.
"""

from __future__ import annotations

import csv
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import torch

from src.training.provenance import semantic_source_hashes
from src.utils.hashing import is_sha256, sha256_file
from src.utils.io import load_json, utc_now_iso


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
REVOCATION_SCHEMA = "graphrestore-stage3-extension-revocation-v1"
REVOCATION_KIND = "stage3_extension_revocation"
REVOCATION_RELATIVE_PATH = Path("artifacts/approvals/STAGE3_EXTENSION_REVOKED.json")
FINALIZER_ENTRYPOINT = "scripts/finalize_stage3.py"
HISTORICAL_SEMANTIC_SOURCE_COUNT = 47
ALLOWED_SEMANTIC_SOURCE_DRIFT = (
    "scripts/train_stage3_planner.py",
    "src/training/orchestration.py",
    "src/training/stage3_engine.py",
    "src/training/stage4_engine.py",
)

SELECTED_STEP = 12_000
ABANDONED_STEP = 14_000

REVOCATION_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_id",
        "created_utc",
        "revoked",
        "stage3_training_authorized",
        "extension_training_permanently_disabled",
        "optimizer_steps_authorized",
        "authorized_pipeline",
        "formal_mio100_authorized",
        "stage3_checkpoint_write_authorized",
        "stage3_optimizer_authorized",
        "stage3_scheduler_authorized",
        "stage3_train_loader_authorized",
        "stage3_finalize_authorized",
        "stage4_authorized",
        "selected_step",
        "abandoned_step",
        "abandoned_pending_validation_step",
        "threshold_calibration_runs_authorized",
        "post_calibration_diagnostic_runs_authorized",
        "selected_checkpoint_byte_exact_required",
        "selected_validation_byte_exact_required",
        "calibration_history_byte_exact_required",
        "bindings",
        "historical_semantic_source_sha256",
        "current_semantic_source_sha256",
        "finalizer_semantic_source_sha256",
        "allowed_semantic_source_drift",
    }
)

BINDING_KEYS = frozenset(
    {
        "user_instruction",
        "target_contract",
        "stage3_approval",
        "approval_required",
        "historical_extension_authorization",
        "historical_extension_migration_receipt",
        "run_contract",
        "abandoned_last_checkpoint",
        "selected_checkpoint",
        "selected_validation",
        "calibration_history",
        "stage3_config",
        "primary_val_manifest",
        "relation_val",
        "pair_prior",
        "global_priority",
        "stage1_checkpoint",
        "pre_extension_run_contract",
        "pre_extension_last_checkpoint",
        "pre_extension_best_checkpoint",
    }
)

_FIXED_VALUES: Mapping[str, Any] = MappingProxyType(
    {
        "schema_version": REVOCATION_SCHEMA,
        "kind": REVOCATION_KIND,
        "protocol_id": PROTOCOL_ID,
        "revoked": True,
        "stage3_training_authorized": False,
        "extension_training_permanently_disabled": True,
        "optimizer_steps_authorized": 0,
        "authorized_pipeline": ["stage3_finalize_only", "stage4"],
        "formal_mio100_authorized": False,
        "stage3_checkpoint_write_authorized": False,
        "stage3_optimizer_authorized": False,
        "stage3_scheduler_authorized": False,
        "stage3_train_loader_authorized": False,
        "stage3_finalize_authorized": True,
        "stage4_authorized": True,
        "selected_step": SELECTED_STEP,
        "abandoned_step": ABANDONED_STEP,
        "abandoned_pending_validation_step": ABANDONED_STEP,
        "threshold_calibration_runs_authorized": 1,
        "post_calibration_diagnostic_runs_authorized": 1,
        "selected_checkpoint_byte_exact_required": True,
        "selected_validation_byte_exact_required": True,
        "calibration_history_byte_exact_required": True,
    }
)

# These physical anchors are the independently audited values immediately
# before the permanent-revocation transaction.  Tests replace this mapping with
# fixture hashes; production generation and validation may not learn hashes from
# the files they are supposed to authenticate.
AUDITED_BINDING_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "user_instruction": (
            "94c0bf820f9a101c51389a27ee7ac151938edd8de228d43ab12338fb9b4632e4"
        ),
        "target_contract": (
            "08144e29c275d2f2a962e9139b1458eebe8ab0d50e4807d0a1e420ecf68ad702"
        ),
        "stage3_approval": (
            "7b351c0958aa681dc1f65114e801c58e3a5bc4bb7cc73c06507c0b647e51a08b"
        ),
        "approval_required": (
            "33be4aba2c4229175ac33edef7a5914a48a249b8c733d86338c64a8662072825"
        ),
        "historical_extension_authorization": (
            "43e010f9c66301415b8bd2d3ac7e48aa7653283671a2756dc744926a4a4724fd"
        ),
        "historical_extension_migration_receipt": (
            "a5fe047b065542e825ba39a1729d94ffe76eba0aa268d5cfaa81a3213867a9a1"
        ),
        "run_contract": (
            "0f38784922de0670a2974f234de22b1c464495fd3c268d08128e29205ed26311"
        ),
        "abandoned_last_checkpoint": (
            "c5455bea98322a23923c9d7173ead50abb818d8ca37e2498fcea4faf9283cc67"
        ),
        "selected_checkpoint": (
            "9114974f68f202119d4241077d0c46333315204959d58b7eabecaf68a3e32ff3"
        ),
        "selected_validation": (
            "85355a5d6b183e688170c846b121fece3735b44bdef9a5b000ce546478c17bec"
        ),
        "calibration_history": (
            "b282987c3f77034f76788a412e91823cd4570ce8c6c10cd93030ee181612e034"
        ),
        "stage3_config": (
            "9ccf41bb3ce6ee859ec553c7b805250020445a8947019e0158aa7f6f693fa01e"
        ),
        "primary_val_manifest": (
            "af89bb22896a3744eab5e4b6414f5ee1b19770ce11e372e27b798afd9583a21b"
        ),
        "relation_val": (
            "6c641406fc50e26a5e1af30b4d113a00439576341b5620022e5ab8514c189f30"
        ),
        "pair_prior": (
            "4116725bce4ecfaceaa1429183e86738ee7ec38835e25886cacd9aa3aec38d82"
        ),
        "global_priority": (
            "80504122f5ce8e8beedf630426bd8e15485efc9db0f9b3d0adf39ca6dd54b0d8"
        ),
        "stage1_checkpoint": (
            "433bcab29f21c98f42107ad6d1c3f8214848254a7ef4d6ca7d6a2141da5bfcaa"
        ),
        "pre_extension_run_contract": (
            "d98b7493b41a0ace9fcb228c50b3acbdf855f092bb2ddc9c9f479730cecf053f"
        ),
        "pre_extension_last_checkpoint": (
            "39733371064c282e46e858aaf50df7b0d4a9fdf3c49c5bc8838798b4958e2438"
        ),
        "pre_extension_best_checkpoint": (
            "b26ebca987fae140bbaff8a7b530692f7a4e0113bdeea863547b6aaec8958b20"
        ),
    }
)

EXPECTED_RELATIVE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "stage3_approval": "artifacts/approvals/STAGE3_APPROVED.json",
        "approval_required": "artifacts/approvals/STAGE3_APPROVAL_REQUIRED.json",
        "historical_extension_authorization": (
            "artifacts/approvals/STAGE3_EXTENSION_APPROVED.json"
        ),
        "historical_extension_migration_receipt": (
            "artifacts/migrations/stage3_extension_12000_to_18000_v1/"
            "MIGRATION_RECEIPT.json"
        ),
        "run_contract": (
            "artifacts/migrations/stage3_extension_revoked_after_14000_v1/"
            "run_contract.json"
        ),
        "abandoned_last_checkpoint": (
            "artifacts/migrations/stage3_extension_revoked_after_14000_v1/"
            "abandoned_last_step14000_pending14000.pth"
        ),
        "selected_checkpoint": "artifacts/checkpoints/stage3/best_ema.pth",
        "selected_validation": (
            "artifacts/checkpoints/stage3/selected_validation.json"
        ),
        "calibration_history": "artifacts/metrics/calibration_history.csv",
        "stage3_config": "configs/stage3_planner.yaml",
        "relation_val": "artifacts/interaction_labels/group_a_relations_val.jsonl",
        "pair_prior": "artifacts/interaction_labels/pair_prior.json",
        "global_priority": "artifacts/interaction_labels/global_priority.json",
        "stage1_checkpoint": "artifacts/checkpoints/stage1/best_ema.pth",
        "pre_extension_run_contract": (
            "artifacts/migrations/stage3_extension_12000_to_18000_v1/run_contract.json"
        ),
        "pre_extension_last_checkpoint": (
            "artifacts/migrations/stage3_extension_12000_to_18000_v1/last.pth"
        ),
        "pre_extension_best_checkpoint": (
            "artifacts/migrations/stage3_extension_12000_to_18000_v1/best_ema.pth"
        ),
    }
)

EXPECTED_ABSOLUTE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "user_instruction": (
            "/root/.codex/attachments/e03c2857-f7a9-4774-9d9a-102b0ffce048/"
            "pasted-text.txt"
        ),
        "target_contract": (
            "/root/autodl-tmp/graphed/"
            "GUARDED_GRAPHRESTORE_FINAL_V7_1_AGENTICIR_CODEX_PROMPT.md"
        ),
        "primary_val_manifest": (
            "/root/autodl-tmp/graph/training_data/manifests/primary_val.jsonl"
        ),
    }
)

IMMUTABLE_BINDINGS = frozenset(
    {
        "run_contract",
        "abandoned_last_checkpoint",
        "pre_extension_run_contract",
        "pre_extension_last_checkpoint",
        "pre_extension_best_checkpoint",
    }
)


class Stage3FinalizationContractError(RuntimeError):
    """The permanent Stage3-finalization contract is absent or inconsistent."""


@dataclass(frozen=True)
class Stage3RevocationAuthorization:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    bindings: Mapping[str, Mapping[str, str]]

    def provenance_binding(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


def _fail(message: str) -> NoReturn:
    raise Stage3FinalizationContractError(message)


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be a strict integer")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _canonical_absolute(path: str | Path, *, field: str) -> Path:
    raw = Path(path)
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if not raw.is_absolute() or str(raw) != str(absolute):
        _fail(f"{field} path must be absolute and lexically canonical")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            _fail(f"symlink is forbidden in {field} path: {current}")
    if absolute.resolve(strict=False) != absolute:
        _fail(f"{field} path is not canonical")
    return absolute


def _regular_file(path: Path, *, field: str, immutable: bool = False) -> None:
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        _fail(f"{field} is missing: {path}")
    if not stat.S_ISREG(mode):
        _fail(f"{field} is not a regular file: {path}")
    if immutable and stat.S_IMODE(mode) != 0o444:
        _fail(f"{field} must be mode 0444")


def _expected_binding_path(root: Path, logical: str) -> Path:
    if logical in EXPECTED_ABSOLUTE_PATHS:
        return Path(EXPECTED_ABSOLUTE_PATHS[logical])
    relative = EXPECTED_RELATIVE_PATHS.get(logical)
    if relative is None:
        _fail(f"no canonical path is frozen for binding {logical}")
    return root / relative


def canonical_stage3_revocation_binding_paths(
    project_root: str | Path,
) -> dict[str, Path]:
    """Return the one canonical physical path for each revocation binding."""

    root = _canonical_absolute(Path(project_root).resolve(), field="project_root")
    return {
        logical: _canonical_absolute(
            _expected_binding_path(root, logical), field=f"expected.{logical}"
        )
        for logical in sorted(BINDING_KEYS)
    }


def _verify_binding(
    root: Path,
    logical: str,
    value: object,
) -> dict[str, str]:
    binding = _mapping(value, field=f"bindings.{logical}")
    if set(binding) != {"path", "sha256"}:
        _fail(f"binding {logical} must contain exactly path and sha256")
    raw_path = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(raw_path, str) or not is_sha256(digest):
        _fail(f"binding {logical} is malformed")
    path = _canonical_absolute(raw_path, field=f"bindings.{logical}")
    expected_path = _canonical_absolute(
        _expected_binding_path(root, logical), field=f"expected.{logical}"
    )
    if path != expected_path:
        _fail(f"binding {logical} path drifted: {path} != {expected_path}")
    _regular_file(
        path, field=f"bindings.{logical}", immutable=logical in IMMUTABLE_BINDINGS
    )
    expected_digest = AUDITED_BINDING_SHA256.get(logical)
    if expected_digest is None or digest != expected_digest:
        _fail(f"binding {logical} does not match its audited SHA256")
    before = sha256_file(path)
    if before != digest or sha256_file(path) != before:
        _fail(f"binding {logical} changed or failed its physical SHA256")
    return {"path": str(path), "sha256": digest}


def _load_checkpoint(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        value = torch.load(path, map_location="cpu", weights_only=False)
    return _mapping(value, field=field)


def _checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    *,
    field: str,
    step: int,
    role: str,
    resumable: bool,
    pending: int | None,
) -> Mapping[str, Any]:
    if (
        checkpoint.get("schema_version") != "graphrestore-checkpoint-v1"
        or checkpoint.get("stage") != "stage3"
        or _strict_int(checkpoint.get("step"), field=f"{field}.step") != step
        or checkpoint.get("model_role") != role
        or checkpoint.get("resumable") is not resumable
        or checkpoint.get("pending_validation_step") != pending
        or checkpoint.get("optimizer_transaction_active") is not False
    ):
        _fail(f"{field} header drifted")
    return _mapping(checkpoint.get("provenance"), field=f"{field}.provenance")


def _validate_history(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    planner_steps: list[int] = []
    for row in rows:
        if row.get("planner_macro_f1", "").strip():
            try:
                planner_steps.append(int(row["step"]))
            except (KeyError, ValueError) as exc:
                raise Stage3FinalizationContractError(
                    "calibration history contains an invalid Stage3 row"
                ) from exc
    if planner_steps != [2_000, 4_000, 6_000, 8_000, 10_000, 12_000]:
        _fail("calibration history does not freeze exactly six Stage3 validations")


def _validate_semantic_map(value: object, *, field: str) -> dict[str, str]:
    mapping = _mapping(value, field=field)
    result: dict[str, str] = {}
    for raw_path, digest in mapping.items():
        if not isinstance(raw_path, str) or not raw_path or not is_sha256(digest):
            _fail(f"{field} contains an invalid source binding")
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            _fail(f"{field} contains a non-project source path")
        result[raw_path] = str(digest)
    if len(result) != HISTORICAL_SEMANTIC_SOURCE_COUNT:
        _fail(
            f"{field} must contain exactly {HISTORICAL_SEMANTIC_SOURCE_COUNT} sources"
        )
    return dict(sorted(result.items()))


def _hash_historical_source_set(
    root: Path, historical: Mapping[str, str]
) -> dict[str, str]:
    current: dict[str, str] = {}
    for relative in historical:
        path = _canonical_absolute(root / relative, field=f"semantic source {relative}")
        try:
            path.relative_to(root)
        except ValueError:
            _fail(f"semantic source escaped project root: {relative}")
        _regular_file(path, field=f"semantic source {relative}")
        current[relative] = sha256_file(path)
    return dict(sorted(current.items()))


def _validate_extension_lineage(
    bindings: Mapping[str, Mapping[str, str]],
    provenance: Mapping[str, Any],
) -> None:
    approval = _mapping(
        load_json(bindings["stage3_approval"]["path"]), field="stage3 approval"
    )
    required = _mapping(
        load_json(bindings["approval_required"]["path"]), field="approval required"
    )
    approval_bindings = _mapping(approval.get("bindings"), field="approval.bindings")
    if (
        len(approval_bindings) != 22
        or required.get("bindings") != approval_bindings
        or approval.get("approved") is not True
        or required.get("approved") is not False
        or approval.get("approval_required_sha256")
        != bindings["approval_required"]["sha256"]
    ):
        _fail("base Stage3 approval lineage drifted")
    for logical in (
        "config_stage3",
        "primary_val_manifest",
        "relation_val",
        "pair_prior",
        "global_priority",
        "stage1_checkpoint",
    ):
        revocation_name = "stage3_config" if logical == "config_stage3" else logical
        if approval_bindings.get(logical) != bindings[revocation_name]:
            _fail(f"base approval binding drifted for {logical}")

    extension = _mapping(
        load_json(bindings["historical_extension_authorization"]["path"]),
        field="historical extension authorization",
    )
    extension_binding = _mapping(
        provenance.get("stage3_extension"), field="provenance.stage3_extension"
    )
    expected_extension = {
        "path": bindings["historical_extension_authorization"]["path"],
        "sha256": bindings["historical_extension_authorization"]["sha256"],
        "cycles": 3,
        "base_step": 12_000,
        "target_step": 18_000,
        "validation_every_steps": 2_000,
        "validation_steps": [14_000, 16_000, 18_000],
        "schedule_horizon_steps": 12_000,
        "min_lr": 2.0e-6,
        "lr_policy": "hold_original_cosine_floor_after_schedule_horizon",
    }
    if (
        extension_binding != expected_extension
        or extension.get("approved") is not True
        or extension.get("authorized_pipeline") != ["stage3_extension", "stage4"]
        or extension.get("formal_mio100_authorized") is not False
    ):
        _fail("historical extension lineage drifted")
    receipt = _mapping(
        load_json(bindings["historical_extension_migration_receipt"]["path"]),
        field="historical extension migration receipt",
    )
    new = _mapping(receipt.get("new"), field="historical receipt.new")
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("cpu_only") is not True
        or receipt.get("three_live_artifacts_share_exact_provenance") is not True
        or new.get("extension_approval")
        != bindings["historical_extension_authorization"]["sha256"]
        or new.get("run_contract") != bindings["run_contract"]["sha256"]
        or new.get("best_checkpoint") != bindings["selected_checkpoint"]["sha256"]
    ):
        _fail("historical extension migration receipt drifted")


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Mapping[str, str]]:
    if set(payload) != REVOCATION_KEYS:
        _fail("Stage3 revocation top-level fields drifted")
    for key, expected in _FIXED_VALUES.items():
        actual = payload.get(key)
        if isinstance(expected, bool):
            valid = isinstance(actual, bool) and actual is expected
        elif isinstance(expected, int):
            valid = (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and actual == expected
            )
        else:
            valid = actual == expected
        if not valid:
            _fail(f"Stage3 revocation field {key} drifted")
    created = payload.get("created_utc")
    if (
        not isinstance(created, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created) is None
    ):
        _fail("Stage3 revocation created_utc is invalid")

    raw_bindings = _mapping(payload.get("bindings"), field="bindings")
    if set(raw_bindings) != BINDING_KEYS:
        _fail("Stage3 revocation binding set drifted")
    bindings = {
        logical: _verify_binding(project_root, logical, raw_bindings[logical])
        for logical in sorted(BINDING_KEYS)
    }

    immutable_inodes: set[tuple[int, int]] = set()
    for logical in sorted(IMMUTABLE_BINDINGS):
        info = Path(bindings[logical]["path"]).stat(follow_symlinks=False)
        identity = (info.st_dev, info.st_ino)
        if identity in immutable_inodes:
            _fail("immutable Stage3 evidence files must have distinct inodes")
        immutable_inodes.add(identity)

    # The archive is durable evidence, but the user also froze the canonical
    # live Stage3 state byte-for-byte.  Verify both live files independently so
    # a correct archive can never mask mutation of the finalizer's real inputs.
    for logical, relative in (
        ("run_contract", "artifacts/checkpoints/stage3/run_contract.json"),
        ("abandoned_last_checkpoint", "artifacts/checkpoints/stage3/last.pth"),
    ):
        live_path = _canonical_absolute(
            project_root / relative, field=f"live.{logical}"
        )
        _regular_file(live_path, field=f"live.{logical}")
        live_info = live_path.stat(follow_symlinks=False)
        if (live_info.st_dev, live_info.st_ino) in immutable_inodes:
            _fail(f"live {logical} must not share an inode with immutable evidence")
        expected = bindings[logical]["sha256"]
        before = sha256_file(live_path)
        if before != expected or sha256_file(live_path) != before:
            _fail(f"canonical live {logical} changed after revocation")

    run_contract = _mapping(
        load_json(bindings["run_contract"]["path"]), field="run contract"
    )
    if run_contract.get("schema_version") != "graphrestore-stage3-runtime-v1":
        _fail("Stage3 run contract schema drifted")
    run_provenance = _mapping(run_contract.get("provenance"), field="run provenance")
    abandoned = _load_checkpoint(
        Path(bindings["abandoned_last_checkpoint"]["path"]),
        field="abandoned checkpoint",
    )
    abandoned_provenance = _checkpoint_provenance(
        abandoned,
        field="abandoned checkpoint",
        step=ABANDONED_STEP,
        role="raw_training_state",
        resumable=True,
        pending=ABANDONED_STEP,
    )
    selected = _load_checkpoint(
        Path(bindings["selected_checkpoint"]["path"]), field="selected checkpoint"
    )
    selected_provenance = _checkpoint_provenance(
        selected,
        field="selected checkpoint",
        step=SELECTED_STEP,
        role="ema_selection",
        resumable=False,
        pending=None,
    )
    if run_provenance != abandoned_provenance or run_provenance != selected_provenance:
        _fail("run/abandoned/selected provenance is not exactly shared")
    runtime = _mapping(run_provenance.get("runtime"), field="provenance.runtime")
    if (
        _strict_int(runtime.get("max_steps"), field="runtime.max_steps") != 12_000
        or _strict_int(
            runtime.get("training_target_step"), field="runtime.training_target_step"
        )
        != 18_000
    ):
        _fail("historical Stage3 runtime horizon drifted")
    _validate_extension_lineage(bindings, run_provenance)

    selected_validation = _mapping(
        load_json(bindings["selected_validation"]["path"]),
        field="selected validation",
    )
    planner = _mapping(selected_validation.get("planner"), field="selected planner")
    graph = _mapping(selected_validation.get("graph"), field="selected graph")
    if (
        selected_validation.get("protocol_id") != PROTOCOL_ID
        or planner.get("sample_count") != 1_600
        or graph.get("sample_count") != 1_600
        or selected_validation.get("checkpoint_presence_threshold") != 0.5
    ):
        _fail("selected validation evidence drifted")
    _validate_history(Path(bindings["calibration_history"]["path"]))

    historical = _validate_semantic_map(
        payload.get("historical_semantic_source_sha256"),
        field="historical_semantic_source_sha256",
    )
    if historical != _validate_semantic_map(
        run_provenance.get("semantic_source_sha256"),
        field="provenance.semantic_source_sha256",
    ):
        _fail("historical semantic source map differs from frozen provenance")
    current = _validate_semantic_map(
        payload.get("current_semantic_source_sha256"),
        field="current_semantic_source_sha256",
    )
    if set(current) != set(historical) or current != _hash_historical_source_set(
        project_root, historical
    ):
        _fail("current semantic source map failed physical re-hash")
    allowed = payload.get("allowed_semantic_source_drift")
    if (
        not isinstance(allowed, list)
        or any(not isinstance(item, str) for item in allowed)
        or allowed != sorted(set(allowed))
        or allowed != list(ALLOWED_SEMANTIC_SOURCE_DRIFT)
    ):
        _fail("allowed_semantic_source_drift differs from the frozen exact set")
    actual_drift = sorted(
        path for path in historical if historical[path] != current[path]
    )
    if allowed != actual_drift:
        _fail("semantic source drift differs from the explicit allowlist")
    finalizer = _mapping(
        payload.get("finalizer_semantic_source_sha256"),
        field="finalizer_semantic_source_sha256",
    )
    expected_finalizer = semantic_source_hashes(
        project_root, entrypoints=(FINALIZER_ENTRYPOINT,)
    )
    if dict(finalizer) != expected_finalizer:
        _fail("finalizer semantic closure failed physical re-hash")

    # Close the read transaction: all large checkpoints and evidence files may
    # have been parsed after their first hash, so bind them again at return.
    for logical, binding in bindings.items():
        if sha256_file(binding["path"]) != binding["sha256"]:
            _fail(f"binding {logical} changed while validating")
    for logical, relative in (
        ("run_contract", "artifacts/checkpoints/stage3/run_contract.json"),
        ("abandoned_last_checkpoint", "artifacts/checkpoints/stage3/last.pth"),
    ):
        if sha256_file(project_root / relative) != bindings[logical]["sha256"]:
            _fail(f"canonical live {logical} changed while validating")
    return bindings


def build_stage3_extension_revocation_payload(
    *,
    project_root: str | Path,
    allowed_semantic_source_drift: Sequence[str],
    binding_paths: Mapping[str, str | Path] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build, but never publish, the exact permanent-revocation payload."""

    root = _canonical_absolute(Path(project_root).resolve(), field="project_root")
    if binding_paths is None:
        binding_paths = canonical_stage3_revocation_binding_paths(root)
    if set(binding_paths) != BINDING_KEYS:
        _fail("builder binding path set drifted")
    bindings: dict[str, dict[str, str]] = {}
    for logical in sorted(BINDING_KEYS):
        path = _canonical_absolute(binding_paths[logical], field=f"builder.{logical}")
        bindings[logical] = {
            "path": str(path),
            "sha256": AUDITED_BINDING_SHA256[logical],
        }
    run_contract = _mapping(
        load_json(bindings["run_contract"]["path"]), field="run contract"
    )
    provenance = _mapping(run_contract.get("provenance"), field="run provenance")
    historical = _validate_semantic_map(
        provenance.get("semantic_source_sha256"),
        field="provenance.semantic_source_sha256",
    )
    current = _hash_historical_source_set(root, historical)
    finalizer = semantic_source_hashes(root, entrypoints=(FINALIZER_ENTRYPOINT,))
    payload: dict[str, Any] = {
        **dict(_FIXED_VALUES),
        "created_utc": created_utc or utc_now_iso(),
        "bindings": bindings,
        "historical_semantic_source_sha256": historical,
        "current_semantic_source_sha256": current,
        "finalizer_semantic_source_sha256": finalizer,
        "allowed_semantic_source_drift": list(allowed_semantic_source_drift),
    }
    _validate_payload(payload, project_root=root)
    return payload


def validate_stage3_extension_revocation(
    path: str | Path,
    *,
    project_root: str | Path,
    require_present: bool = True,
) -> Stage3RevocationAuthorization | None:
    """Validate the canonical revocation and every physical authorization leaf."""

    root = _canonical_absolute(Path(project_root).resolve(), field="project_root")
    canonical = _canonical_absolute(root / REVOCATION_RELATIVE_PATH, field="revocation")
    supplied = _canonical_absolute(path, field="revocation")
    if supplied != canonical:
        _fail("revocation authorization must use its canonical path")
    if not isinstance(require_present, bool):
        _fail("require_present must be a strict boolean")
    if not os.path.lexists(canonical):
        if require_present:
            _fail("revocation authorization is missing")
        return None
    _regular_file(canonical, field="revocation authorization")
    before = sha256_file(canonical)
    payload = _mapping(load_json(canonical), field="revocation authorization")
    bindings = _validate_payload(payload, project_root=root)
    if sha256_file(canonical) != before:
        _fail("revocation authorization changed while validating")
    frozen_payload = MappingProxyType(dict(payload))
    frozen_bindings = MappingProxyType(
        {key: MappingProxyType(dict(value)) for key, value in bindings.items()}
    )
    return Stage3RevocationAuthorization(
        path=canonical,
        sha256=before,
        payload=frozen_payload,
        bindings=frozen_bindings,
    )


def refuse_stage3_training_if_revoked(project_root: str | Path) -> None:
    """Refuse training if any object occupies the permanent tombstone path.

    This intentionally does not validate the object.  A malformed file,
    directory, or dangling symlink must never turn revocation into permission.
    """

    root = Path(project_root).resolve()
    tombstone = root / REVOCATION_RELATIVE_PATH
    if os.path.lexists(tombstone):
        _fail(
            "Stage3 training is permanently disabled by the canonical extension "
            f"revocation tombstone: {tombstone}"
        )


__all__ = [
    "BINDING_KEYS",
    "REVOCATION_KEYS",
    "REVOCATION_RELATIVE_PATH",
    "Stage3FinalizationContractError",
    "Stage3RevocationAuthorization",
    "build_stage3_extension_revocation_payload",
    "canonical_stage3_revocation_binding_paths",
    "refuse_stage3_training_if_revoked",
    "validate_stage3_extension_revocation",
]
