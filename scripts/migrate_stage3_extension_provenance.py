#!/usr/bin/env python3
"""Publish the one-off Stage3 12k -> 18k extension authorization.

This migration is deliberately CPU-only and independent of the older Stage3
migration implementations.  It creates three immutable pre-extension backups,
publishes the exact 20-field extension authorization, and changes the live
Stage3 run contract plus raw/EMA checkpoints only inside ``provenance``.

The permitted provenance changes are exactly:

* the caller-frozen semantic-source SHA256 old/new map;
* ``runtime.training_target_step = 18000``;
* the exact ten-field ``stage3_extension`` authorization binding.

Without ``--execute`` the command only builds and reloads temporary candidates.
Interrupted ``PREPARED`` transactions are recovered with ``--recover-prepared``
and a different exact confirmation token.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import stat
import struct
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

# This assignment must precede every import that can transitively import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.provenance import semantic_source_hashes  # noqa: E402
from src.utils.hashing import is_sha256, sha256_file, sha256_json  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_write_json,
    fsync_directory,
    load_json,
    utc_now_iso,
)


PROTOCOL_ID = "graphrestore-v7.1-agenticir-locked"
CHECKPOINT_SCHEMA = "graphrestore-checkpoint-v1"
STAGE3_RUNTIME_SCHEMA = "graphrestore-stage3-runtime-v1"
BASE_APPROVAL_SCHEMA = "graphrestore-stage3-approval-v1"
EXTENSION_APPROVAL_SCHEMA = "graphrestore-stage3-extension-approval-v1"
RECEIPT_SCHEMA = "graphrestore-stage3-extension-provenance-migration-v1"
MIGRATION_KIND = "stage3_12000_to_18000_extension_authorization_and_provenance"

BASE_STEP = 12_000
TARGET_STEP = 18_000
VALIDATION_EVERY_STEPS = 2_000
VALIDATION_STEPS = (14_000, 16_000, 18_000)
CYCLES = 3
SCHEDULE_HORIZON_STEPS = 12_000
MIN_LR = 2.0e-6
LR_POLICY = "hold_original_cosine_floor_after_schedule_horizon"
EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT = 20
EXPECTED_UNCHANGED_CHECKPOINT_TOP_LEVEL_COUNT = 19
EXPECTED_BASE_BINDING_COUNT = 22

ENTRYPOINTS = (
    "scripts/train_stage3_planner.py",
    "scripts/eval_guard_diagnostics.py",
)
BACKUP_DIR_NAME = "stage3_extension_12000_to_18000_v1"
EXTENSION_APPROVAL_NAME = "STAGE3_EXTENSION_APPROVED.json"
RECEIPT_NAME = "MIGRATION_RECEIPT.json"

GUARD_BACKUP_DIR_NAME = "stage3_guard_alignment_pending2000_v1"
GUARD_RECEIPT_SCHEMA = "graphrestore-stage3-guard-alignment-migration-v1"
GUARD_MIGRATION_KIND = "stage3_pending_2000_guard_alignment_provenance_only"
EMA_BACKUP_DIR_NAME = "stage3_ema_device_pending2000_v1"
EMA_RECEIPT_SCHEMA = "graphrestore-stage3-ema-device-migration-v1"
EMA_MIGRATION_KIND = "stage3_pending_2000_ema_device_provenance_only"

CONFIRMATION_TOKEN = "MIGRATE_STAGE3_12000_TO_18000_EXTENSION_PROVENANCE"
RECOVERY_CONFIRMATION_TOKEN = (
    "RECOVER_PREPARED_STAGE3_12000_TO_18000_EXTENSION_PROVENANCE"
)

# Frozen canonical anchors.  Semantic source maps are intentionally supplied
# separately because their final values are frozen only after all source edits.
AUDITED_RUN_CONTRACT_SHA256 = (
    "d98b7493b41a0ace9fcb228c50b3acbdf855f092bb2ddc9c9f479730cecf053f"
)
AUDITED_LAST_CHECKPOINT_SHA256 = (
    "39733371064c282e46e858aaf50df7b0d4a9fdf3c49c5bc8838798b4958e2438"
)
AUDITED_BEST_CHECKPOINT_SHA256 = (
    "b26ebca987fae140bbaff8a7b530692f7a4e0113bdeea863547b6aaec8958b20"
)
AUDITED_STATE_SHA256 = (
    "876a3fffada00db1ad9c87891f94a23d751fb626005c9b7e5818a5a2e31b888d"
)
AUDITED_BASE_APPROVAL_SHA256 = (
    "7b351c0958aa681dc1f65114e801c58e3a5bc4bb7cc73c06507c0b647e51a08b"
)
AUDITED_APPROVAL_REQUIRED_SHA256 = (
    "33be4aba2c4229175ac33edef7a5914a48a249b8c733d86338c64a8662072825"
)
AUDITED_STAGE3_CONFIG_SHA256 = (
    "9ccf41bb3ce6ee859ec553c7b805250020445a8947019e0158aa7f6f693fa01e"
)
AUDITED_GUARD_RECEIPT_SHA256 = (
    "449bd49b3e31a430eed1d4c6e217c4299084beb272d9845648ded95b7f8718e6"
)
AUDITED_EMA_RECEIPT_SHA256 = (
    "9848708c1a2dc91a99230a68ebf630c8574c64b6cbc8bad97700b5846efc21cb"
)

# Root freezes these maps immediately before the canonical dry-run/execute.
AUDITED_OLD_SEMANTIC_SOURCE_SHA256: dict[str, str] = {
    "scripts/eval_guard_diagnostics.py": "b461ce62ce233505f32239d364f13a406d40dbcbea333b2dc034b32510780eec",
    "scripts/train_stage3_planner.py": "3d498fcea7cdc52480e6ff8e3e2d85596d2bde94ed289f14deeee66f9d9beabc",
    "src/__init__.py": "457b63d50d01f3c33d60220d67dfa8c4717085a136db419c6a584b3408e44669",
    "src/data/__init__.py": "ef568ed92708ff8ce9693b058c282ddd62203442b8dcfcfd5bf817eaf30afea7",
    "src/data/agenticir_degradations.py": "7a7d160aa6ca228b031bb00e65f8f31c61d734ef8a572881e95e0ee42ba89e54",
    "src/data/episode_dataset.py": "867519c6c5f22f5b0926b3af47d5938a8f1ebe6ca336dcedbb74f5d29bae24cb",
    "src/data/manifests.py": "7319d67dc34f15ddb7f2e607fc3d8c0f0010c693756ae3f198c4acf89953e710",
    "src/data/samplers.py": "4e79abc547bd413d8fd4787a5c499892d75b46de7396e80a0f8f3da52d9b3ea4",
    "src/data/scale_canonicalizer.py": "3e66a83c4bd09679741a38d7fae0b0876dbe8c1357d4108193ebfc8e58c8fe8f",
    "src/data/subset_targets.py": "7521871e29bb58b71b7f6abf9a8e69b1b52208befce239b8976c9f71e5f68405",
    "src/losses/__init__.py": "e2c87fe8637f928278ceef6f2f94ff81c596474eace3444f89c0b4e10d6c611e",
    "src/losses/cycle_consistency.py": "c4c45e5ac01e07ca5d68cf8f8fa4d8b90a50a5c4e969f626231a21ca700f4bf7",
    "src/losses/guard_losses.py": "34ece30ee865defe580ae4e0a2795e143df4da2f6f06cde6c42ddd2cd80b7a51",
    "src/losses/planner_losses.py": "a2a0fa419272a840846ba00411e05dd12ad39bbf62b140459398fc3523a6492e",
    "src/losses/restoration.py": "c8a6a5bf0ffd96032dfbdc272d53ed6a516479714275c5271b9b24e15cde7294",
    "src/metrics/__init__.py": "24fd1a749ea4dac6603c70c2636c69af96c366d3cc0d56d2c2c5594ecea442ae",
    "src/metrics/agenticir_official.py": "1bc92d924c2825233d989b48add4d4e8af749747368273a3d2d5615546152219",
    "src/net/__init__.py": "f255aab130fe8a2f63a695dedb08e2fcfe712abf538fb529c45a441ee34d1eab",
    "src/net/cooperative_executor.py": "10d6d3ceb69bb7e517df4a2d2b970d62f4ce15097a99943c2af35a1b192c07fd",
    "src/net/graph_compiler.py": "49529ab6e04cbd63dd392b0795da8bf06ee9d64b43b151b64cf303d4dfe14d65",
    "src/net/graphrestore.py": "2bdb139b94a82818e6a16dde45e2d2fee066a03cd4eca447715d310d1fecc6d8",
    "src/net/latent_skill_bank.py": "161878c2c38db929eb8c6f00e5a5a7829ab1544da2cb946341a6508bbaf3d71f",
    "src/net/mio_stagea.py": "6d00daae0b4f1f6a66ee2a6fe72f8077443df2623f86c391c7e0001af5b131e9",
    "src/net/program_planner.py": "c395c41b5b85dec8e178d3a4bf9216ad5acf28b56c5a4abcb12db4ba3ab829a7",
    "src/net/restormer_blocks.py": "0fd43c1dbf2ccec239e3ee5eded1c643ace3e68d90dda82b1f184e626eab501d",
    "src/net/skill_adapter.py": "93e1b79e07832219cb3f37d3f6d453e8c8ffc86f2f507ebea4d49e65d416dfd3",
    "src/net/trace_pyramid.py": "f8c55d10bba8bc10829a73bc1ae2bb87b89b4d62d99d2bd0e2b798dc2f4ee62f",
    "src/training/__init__.py": "eab12a7cec3421440b882b27532ba321503e47b780eb42b1b605d930f99a930b",
    "src/training/checkpointing.py": "063e31657922ad2006f45a6e31677aa9953b2b95d38dbaf6207bcc20bb47f072",
    "src/training/ema.py": "7c11d3857f194a406fc6bdd6b2d2b96adc4f1e8bfe1cc8f1acf0ecb3154234d4",
    "src/training/optimization.py": "7d7b4486822697e96c17ce24e86377023444c512d5c6328547612619022494e7",
    "src/training/orchestration.py": "7979ae0feedc1677a02fe2bd2ac76432185881a75b80122dc8bcd936b9cbff1f",
    "src/training/provenance.py": "d49f2de6d65e52bf4bddb4a0049ed88058879d7e877069cc61ed5e8508735020",
    "src/training/relation_supervision.py": "82c94341cb0a009f15e9e91c5b96dce3b36ef45ee11731505c757d1f40b68699",
    "src/training/runtime.py": "47e550c55e5684150fc6cffee90d1874a8ca2afd0d2652345462261a644c3dfe",
    "src/training/selection.py": "8a6c75e19913168dd876bf9af9c0d7702599b4e59131426d0136315712310166",
    "src/training/stage0_engine.py": "9357c67de16c2fc3eaa79c0b9687e77d841cec22c0bbb605835b47ec7759182a",
    "src/training/stage1_engine.py": "ab76a61422532d0deac8d7c01da69dae9f1d8154a277a4c0c26658b936c36f6c",
    "src/training/stage2_distillation.py": "4cd37e4a1e5725e9b948c758eab893ff9a95f1c23357a2cdea3215df93d1b06c",
    "src/training/stage3_engine.py": "908bcd7ff829aabba8376ec949156890983f51924aaa7e2313e013648d817b49",
    "src/training/stage4_engine.py": "e2fbfbc2ee580b90cb92c48e6b289d6bc6d3d4651c42d34295ce07fc664814b6",
    "src/utils/__init__.py": "f665f064cf3c517389fe1cea13dd6632ba6a3bf17bedde46d60bff743289524c",
    "src/utils/audit.py": "ba50200e5a88afd3ed97ef3ce82a3f4003bd90cd90f0f83ac2269e9e1acc772a",
    "src/utils/git.py": "1fe4fd00a7f3d8db39cdbbd0233a424da5c81dd9a345b3a5e3e62b6a92197a8b",
    "src/utils/hashing.py": "173749dc7552576f047c08fc8fdccbec252cec3cef3c1e78d9a43784d90d9980",
    "src/utils/io.py": "321384d77f0d85ffae0da2d8b63ff03afe2afafc51a320d50543aab8afaad61f",
    "src/utils/paths.py": "c739fa180d6857972bd193500825076d782d00f28ef26beefd94493a86fef337",
}
AUDITED_NEW_SEMANTIC_SOURCE_SHA256: dict[str, str] = {
    "scripts/eval_guard_diagnostics.py": "b461ce62ce233505f32239d364f13a406d40dbcbea333b2dc034b32510780eec",
    "scripts/train_stage3_planner.py": "1e7db4c46f640d62501e91eb50862073f8c6473b9090018771b41fe1bdfc4b9d",
    "src/__init__.py": "457b63d50d01f3c33d60220d67dfa8c4717085a136db419c6a584b3408e44669",
    "src/data/__init__.py": "ef568ed92708ff8ce9693b058c282ddd62203442b8dcfcfd5bf817eaf30afea7",
    "src/data/agenticir_degradations.py": "7a7d160aa6ca228b031bb00e65f8f31c61d734ef8a572881e95e0ee42ba89e54",
    "src/data/episode_dataset.py": "867519c6c5f22f5b0926b3af47d5938a8f1ebe6ca336dcedbb74f5d29bae24cb",
    "src/data/manifests.py": "7319d67dc34f15ddb7f2e607fc3d8c0f0010c693756ae3f198c4acf89953e710",
    "src/data/samplers.py": "4e79abc547bd413d8fd4787a5c499892d75b46de7396e80a0f8f3da52d9b3ea4",
    "src/data/scale_canonicalizer.py": "3e66a83c4bd09679741a38d7fae0b0876dbe8c1357d4108193ebfc8e58c8fe8f",
    "src/data/subset_targets.py": "7521871e29bb58b71b7f6abf9a8e69b1b52208befce239b8976c9f71e5f68405",
    "src/losses/__init__.py": "e2c87fe8637f928278ceef6f2f94ff81c596474eace3444f89c0b4e10d6c611e",
    "src/losses/cycle_consistency.py": "c4c45e5ac01e07ca5d68cf8f8fa4d8b90a50a5c4e969f626231a21ca700f4bf7",
    "src/losses/guard_losses.py": "34ece30ee865defe580ae4e0a2795e143df4da2f6f06cde6c42ddd2cd80b7a51",
    "src/losses/planner_losses.py": "a2a0fa419272a840846ba00411e05dd12ad39bbf62b140459398fc3523a6492e",
    "src/losses/restoration.py": "c8a6a5bf0ffd96032dfbdc272d53ed6a516479714275c5271b9b24e15cde7294",
    "src/metrics/__init__.py": "24fd1a749ea4dac6603c70c2636c69af96c366d3cc0d56d2c2c5594ecea442ae",
    "src/metrics/agenticir_official.py": "1bc92d924c2825233d989b48add4d4e8af749747368273a3d2d5615546152219",
    "src/net/__init__.py": "f255aab130fe8a2f63a695dedb08e2fcfe712abf538fb529c45a441ee34d1eab",
    "src/net/cooperative_executor.py": "10d6d3ceb69bb7e517df4a2d2b970d62f4ce15097a99943c2af35a1b192c07fd",
    "src/net/graph_compiler.py": "49529ab6e04cbd63dd392b0795da8bf06ee9d64b43b151b64cf303d4dfe14d65",
    "src/net/graphrestore.py": "2bdb139b94a82818e6a16dde45e2d2fee066a03cd4eca447715d310d1fecc6d8",
    "src/net/latent_skill_bank.py": "161878c2c38db929eb8c6f00e5a5a7829ab1544da2cb946341a6508bbaf3d71f",
    "src/net/mio_stagea.py": "6d00daae0b4f1f6a66ee2a6fe72f8077443df2623f86c391c7e0001af5b131e9",
    "src/net/program_planner.py": "c395c41b5b85dec8e178d3a4bf9216ad5acf28b56c5a4abcb12db4ba3ab829a7",
    "src/net/restormer_blocks.py": "0fd43c1dbf2ccec239e3ee5eded1c643ace3e68d90dda82b1f184e626eab501d",
    "src/net/skill_adapter.py": "93e1b79e07832219cb3f37d3f6d453e8c8ffc86f2f507ebea4d49e65d416dfd3",
    "src/net/trace_pyramid.py": "f8c55d10bba8bc10829a73bc1ae2bb87b89b4d62d99d2bd0e2b798dc2f4ee62f",
    "src/training/__init__.py": "eab12a7cec3421440b882b27532ba321503e47b780eb42b1b605d930f99a930b",
    "src/training/checkpointing.py": "063e31657922ad2006f45a6e31677aa9953b2b95d38dbaf6207bcc20bb47f072",
    "src/training/ema.py": "7c11d3857f194a406fc6bdd6b2d2b96adc4f1e8bfe1cc8f1acf0ecb3154234d4",
    "src/training/optimization.py": "7d7b4486822697e96c17ce24e86377023444c512d5c6328547612619022494e7",
    "src/training/orchestration.py": "8691c56fafafd6f5f2b37d53ab01009b092ca0395735a69ab71ab97f34a9b622",
    "src/training/provenance.py": "d49f2de6d65e52bf4bddb4a0049ed88058879d7e877069cc61ed5e8508735020",
    "src/training/relation_supervision.py": "82c94341cb0a009f15e9e91c5b96dce3b36ef45ee11731505c757d1f40b68699",
    "src/training/runtime.py": "47e550c55e5684150fc6cffee90d1874a8ca2afd0d2652345462261a644c3dfe",
    "src/training/selection.py": "8a6c75e19913168dd876bf9af9c0d7702599b4e59131426d0136315712310166",
    "src/training/stage0_engine.py": "9357c67de16c2fc3eaa79c0b9687e77d841cec22c0bbb605835b47ec7759182a",
    "src/training/stage1_engine.py": "ab76a61422532d0deac8d7c01da69dae9f1d8154a277a4c0c26658b936c36f6c",
    "src/training/stage2_distillation.py": "4cd37e4a1e5725e9b948c758eab893ff9a95f1c23357a2cdea3215df93d1b06c",
    "src/training/stage3_engine.py": "7c65d89f9778dd3f49250774fcfaa4f3f6209d62ac6e9f9f507991fe22427e0a",
    "src/training/stage4_engine.py": "518b10b49320fd24879febc3483d30f7a8b28e96037588102ddb65f89a958845",
    "src/utils/__init__.py": "f665f064cf3c517389fe1cea13dd6632ba6a3bf17bedde46d60bff743289524c",
    "src/utils/audit.py": "ba50200e5a88afd3ed97ef3ce82a3f4003bd90cd90f0f83ac2269e9e1acc772a",
    "src/utils/git.py": "1fe4fd00a7f3d8db39cdbbd0233a424da5c81dd9a345b3a5e3e62b6a92197a8b",
    "src/utils/hashing.py": "173749dc7552576f047c08fc8fdccbec252cec3cef3c1e78d9a43784d90d9980",
    "src/utils/io.py": "321384d77f0d85ffae0da2d8b63ff03afe2afafc51a320d50543aab8afaad61f",
    "src/utils/paths.py": "c739fa180d6857972bd193500825076d782d00f28ef26beefd94493a86fef337",
}
AUDITED_SOURCE_TRANSITION_SHA256 = (
    "27d1df126529545bec0d84c15c386c9a7a9b24e64f93d13f85a32a4ba8eea3b3"
)
AUDITED_CHANGED_SOURCE_PATHS = (
    "scripts/train_stage3_planner.py",
    "src/training/orchestration.py",
    "src/training/stage3_engine.py",
    "src/training/stage4_engine.py",
)


class Stage3ExtensionMigrationError(RuntimeError):
    """The requested migration does not satisfy the frozen transaction."""


def _fail(message: str) -> NoReturn:
    raise Stage3ExtensionMigrationError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _assert_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        _fail("migration must remain CPU-only with CUDA_VISIBLE_DEVICES empty")


def _absolute_lexical(path: str | Path) -> Path:
    raw = Path(path)
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if not raw.is_absolute() or str(raw) != str(absolute):
        _fail(f"path must be absolute and lexically canonical: {raw}")
    return absolute


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            _fail(f"symlink is forbidden in {label} path: {current}")


def _canonical_path(path: str | Path, *, label: str) -> Path:
    absolute = _absolute_lexical(path)
    _reject_symlink_chain(absolute, label=label)
    resolved = absolute.resolve(strict=False)
    if str(resolved) != str(absolute):
        _fail(f"{label} path is not canonical: {absolute}")
    return resolved


def _validate_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not is_sha256(value):
        _fail(f"{field} must be a lowercase SHA256")
    return value


def _deterministic_json_file_sha256(value: object) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_source_map(value: Mapping[str, str], *, field: str) -> dict[str, str]:
    result = dict(value)
    if not result:
        _fail(f"{field} must not be empty")
    if any(
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or path != Path(path).as_posix()
        or not isinstance(digest, str)
        or not is_sha256(digest)
        for path, digest in result.items()
    ):
        _fail(f"{field} contains an invalid path/SHA256 entry")
    return dict(sorted(result.items()))


def _validate_frozen_source_transition(
    old_value: Mapping[str, str],
    new_value: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Require the complete, immutable 47-entry production transition.

    Caller-provided maps remain explicit inputs for auditability, but they may
    not widen or reverse the user-authorized source transition.  The independent
    transition digest also detects accidental edits to either built-in map.
    """

    old = _validate_source_map(old_value, field="expected old source map")
    new = _validate_source_map(new_value, field="expected new source map")
    built_in_old = _validate_source_map(
        AUDITED_OLD_SEMANTIC_SOURCE_SHA256,
        field="audited old source map",
    )
    built_in_new = _validate_source_map(
        AUDITED_NEW_SEMANTIC_SOURCE_SHA256,
        field="audited new source map",
    )
    built_in_digest = sha256_json({"old": built_in_old, "new": built_in_new})
    if built_in_digest != AUDITED_SOURCE_TRANSITION_SHA256:
        _fail("built-in audited source-transition values drifted")
    if len(old) != 47 or len(new) != 47 or old.keys() != new.keys():
        _fail("source transition must contain the same exact 47 paths")
    changed = tuple(path for path in old if old[path] != new[path])
    if changed != AUDITED_CHANGED_SOURCE_PATHS:
        _fail(
            "source transition changed-path set drifted: "
            f"expected={list(AUDITED_CHANGED_SOURCE_PATHS)}, actual={list(changed)}"
        )
    if old != built_in_old or new != built_in_new:
        _fail("caller source maps differ from the frozen audited transition")
    return old, new


@contextmanager
def _single_writer_lock(migrations_directory: Path) -> Iterator[None]:
    """Lock the stable migrations directory across every live-file replace."""

    descriptor = os.open(
        migrations_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Stage3ExtensionMigrationError(
                "another Stage3 extension migration writer holds the lock"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _qualified_type(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _tensor_bytes(value: torch.Tensor) -> bytes:
    if value.layout is not torch.strided:
        _fail(f"unsupported tensor layout: {value.layout}")
    flat = value.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _assert_bit_exact(before: object, after: object, *, path: str) -> None:
    if type(before) is not type(after):
        _fail(
            f"type mutation at {path}: {_qualified_type(before)} != "
            f"{_qualified_type(after)}"
        )
    if isinstance(before, torch.Tensor):
        assert isinstance(after, torch.Tensor)
        metadata_before = (
            before.dtype,
            before.layout,
            tuple(before.shape),
            tuple(before.stride()),
            before.storage_offset(),
            before.requires_grad,
        )
        metadata_after = (
            after.dtype,
            after.layout,
            tuple(after.shape),
            tuple(after.stride()),
            after.storage_offset(),
            after.requires_grad,
        )
        if metadata_before != metadata_after or _tensor_bytes(before) != _tensor_bytes(
            after
        ):
            _fail(f"tensor mutation at {path}")
        return
    if isinstance(before, np.ndarray):
        assert isinstance(after, np.ndarray)
        if (
            before.dtype != after.dtype
            or before.shape != after.shape
            or before.strides != after.strides
            or before.tobytes(order="A") != after.tobytes(order="A")
        ):
            _fail(f"numpy mutation at {path}")
        return
    if isinstance(before, Mapping):
        assert isinstance(after, Mapping)
        if list(before) != list(after):
            _fail(f"mapping key/order mutation at {path}")
        for key in before:
            _assert_bit_exact(before[key], after[key], path=f"{path}.{key}")
        return
    if isinstance(before, (list, tuple)):
        assert isinstance(after, (list, tuple))
        if len(before) != len(after):
            _fail(f"sequence length mutation at {path}")
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            _assert_bit_exact(old, new, path=f"{path}[{index}]")
        return
    if isinstance(before, float):
        if struct.pack(">d", before) != struct.pack(">d", after):
            _fail(f"float mutation at {path}")
        return
    if before != after:
        _fail(f"value mutation at {path}: {before!r} != {after!r}")


def _update_fingerprint(
    digest: Any,
    value: object,
    counts: Counter[str],
) -> None:
    kind = _qualified_type(value)
    counts[kind] += 1
    encoded_kind = kind.encode()
    digest.update(struct.pack(">Q", len(encoded_kind)))
    digest.update(encoded_kind)
    if isinstance(value, torch.Tensor):
        metadata = repr(
            (
                str(value.dtype),
                str(value.layout),
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                value.requires_grad,
            )
        ).encode()
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        raw = _tensor_bytes(value)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    elif isinstance(value, np.ndarray):
        metadata = repr((str(value.dtype), value.shape, value.strides)).encode()
        raw = value.tobytes(order="A")
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    elif isinstance(value, Mapping):
        digest.update(struct.pack(">Q", len(value)))
        for key, child in value.items():
            _update_fingerprint(digest, key, counts)
            _update_fingerprint(digest, child, counts)
    elif isinstance(value, (list, tuple)):
        digest.update(struct.pack(">Q", len(value)))
        for child in value:
            _update_fingerprint(digest, child, counts)
    elif value is None:
        digest.update(b"none")
    elif isinstance(value, bool):
        digest.update(b"true" if value else b"false")
    elif isinstance(value, int):
        digest.update(str(value).encode())
    elif isinstance(value, float):
        digest.update(struct.pack(">d", value))
    elif isinstance(value, str):
        raw = value.encode()
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    else:
        _fail(f"unsupported fingerprint value: {kind}")


def _fingerprint(value: object) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    _update_fingerprint(digest, value, counts)
    return {"sha256": digest.hexdigest(), "counts": dict(sorted(counts.items()))}


def _walk_finite(value: object, *, path: str = "checkpoint") -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            _fail(f"non-finite tensor at {path}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(
            np.isfinite(value).all()
        ):
            _fail(f"non-finite numpy value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_finite(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_finite(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        _fail(f"non-finite scalar at {path}")


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        raise Stage3ExtensionMigrationError(
            f"could not load checkpoint on CPU: {type(exc).__name__}: {exc}"
        ) from exc
    return _mapping(payload, field=f"checkpoint {path}")


def _validate_checkpoint(payload: Mapping[str, Any], *, role: str) -> None:
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": "stage3",
        "step": BASE_STEP,
        "model_role": role,
        "resumable": role == "raw_training_state",
        "pending_validation_step": None,
        "optimizer_transaction_active": False,
        "executor_frozen": True,
        "trainable_prefixes": ["planner."],
        "amp": {"dtype": "bfloat16", "scaler_required": False},
        "scaler": None,
    }
    for key, expected_value in expected.items():
        if payload.get(key, object()) != expected_value:
            _fail(f"{role} checkpoint header drifted at {key}")
    if len(payload) != EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT:
        _fail(f"{role} checkpoint must have exactly 20 top-level sections")
    metrics = _mapping(payload.get("metrics"), field=f"{role}.metrics")
    if (
        metrics.get("validation_step") != BASE_STEP
        or metrics.get("best_step") != BASE_STEP
    ):
        _fail(f"{role} checkpoint is not the selected step-12000 validation")
    sampler = _mapping(payload.get("sampler_state"), field=f"{role}.sampler_state")
    if (
        sampler.get("consumed_optimizer_step") != BASE_STEP
        or sampler.get("sample_cursor") != BASE_STEP * 8
    ):
        _fail(f"{role} sampler is not the exact step-12000 boundary")
    ema = _mapping(payload.get("ema"), field=f"{role}.ema")
    if (
        ema.get("num_updates") != BASE_STEP
        or ema.get("scope") != "planner_parameters_only_executor_bitwise_frozen"
    ):
        _fail(f"{role} EMA policy/boundary drifted")
    _walk_finite(payload)


def _assert_cross_role_tensor_state_equal(
    before: object,
    after: object,
    *,
    path: str,
) -> None:
    """Compare cross-role state tensors without equating container classes.

    Canonical Stage3 stores ``model`` as ``OrderedDict`` and EMA ``shadow`` as
    ``dict``.  This helper is used only for their semantic tensor equality.
    Old-to-candidate serialization preservation remains strictly type-sensitive
    in ``_checkpoint_section_evidence`` via ``_assert_bit_exact``.
    """

    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        _fail(f"{path} must compare two tensor mappings")
    if list(before) != list(after):
        _fail(f"cross-role tensor-state key/order mutation at {path}")
    for key in before:
        old_tensor, new_tensor = before[key], after[key]
        if not torch.is_tensor(old_tensor) or not torch.is_tensor(new_tensor):
            _fail(f"cross-role non-tensor state entry at {path}.{key}")
        _assert_bit_exact(old_tensor, new_tensor, path=f"{path}.{key}")


def _validate_checkpoint_pair(last: Mapping[str, Any], best: Mapping[str, Any]) -> None:
    _validate_checkpoint(last, role="raw_training_state")
    _validate_checkpoint(best, role="ema_selection")
    if last.get("provenance") != best.get("provenance"):
        _fail("last/best provenance differs")
    last_ema = _mapping(last.get("ema"), field="last.ema")
    best_ema = _mapping(best.get("ema"), field="best.ema")
    _assert_cross_role_tensor_state_equal(
        last_ema.get("shadow"), best.get("model"), path="last_ema.best"
    )
    _assert_cross_role_tensor_state_equal(
        best.get("model"), best_ema.get("shadow"), path="best.model_ema"
    )


def _checkpoint_section_evidence(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    if list(old) != list(new):
        _fail("checkpoint top-level key/order changed")
    evidence: dict[str, Any] = {}
    unchanged = 0
    for key in old:
        old_fingerprint = _fingerprint(old[key])
        new_fingerprint = _fingerprint(new[key])
        if key != "provenance":
            _assert_bit_exact(old[key], new[key], path=f"checkpoint.{key}")
            unchanged += 1
        evidence[key] = {
            "old": old_fingerprint,
            "new": new_fingerprint,
            "bit_exact": old_fingerprint == new_fingerprint,
        }
    if unchanged != EXPECTED_UNCHANGED_CHECKPOINT_TOP_LEVEL_COUNT:
        _fail("checkpoint bit-exact outside-provenance section count drifted")
    return evidence


def _validate_failed_state(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        _fail("orchestration state SHA256 drifted")
    state = _mapping(load_json(path), field="orchestration state")
    command = state.get("last_command")
    if (
        state.get("schema_version") != "graphrestore-orchestration-v1"
        or state.get("protocol_id") != PROTOCOL_ID
        or state.get("status") != "FAILED"
        or state.get("current_stage") != "FAILED"
        or state.get("gpu") != "released"
        or state.get("last_exit_code") != 1
        or state.get("next_command")
        != "python scripts/orchestrate.py --resume_post_approval_pipeline"
        or not isinstance(command, list)
        or "scripts/train_stage3_planner.py" not in command
        or "--resume" not in command
    ):
        _fail("orchestration state is not the exact failed step-12000 boundary")
    return dict(state)


def _validate_base_approval(
    approval_path: Path,
    required_path: Path,
    config_path: Path,
    *,
    expected_approval_sha256: str,
    expected_required_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    for label, path, expected in (
        ("base approval", approval_path, expected_approval_sha256),
        ("approval-required", required_path, expected_required_sha256),
        ("Stage3 config", config_path, expected_config_sha256),
    ):
        if sha256_file(path) != expected:
            _fail(f"{label} SHA256 drifted")
    approval = _mapping(load_json(approval_path), field="base Stage3 approval")
    required = _mapping(load_json(required_path), field="Stage3 approval-required")
    bindings = _mapping(approval.get("bindings"), field="base approval.bindings")
    if (
        approval.get("schema_version") != BASE_APPROVAL_SCHEMA
        or approval.get("kind") != "stage3_approval"
        or approval.get("protocol_id") != PROTOCOL_ID
        or approval.get("approved") is not True
        or approval.get("approval_required_sha256") != expected_required_sha256
        or required.get("schema_version") != BASE_APPROVAL_SCHEMA
        or required.get("kind") != "stage3_approval_required"
        or required.get("protocol_id") != PROTOCOL_ID
        or required.get("approved") is not False
        or required.get("bindings") != bindings
        or len(bindings) != EXPECTED_BASE_BINDING_COUNT
    ):
        _fail("base approval/approval-required 22-binding contract drifted")
    verified: dict[str, str] = {}
    for logical, raw_binding in bindings.items():
        binding = _mapping(raw_binding, field=f"approval.bindings.{logical}")
        if set(binding) != {"path", "sha256"}:
            _fail(f"base approval binding fields drifted: {logical}")
        raw_path = binding.get("path")
        digest = _validate_sha(binding.get("sha256"), field=f"binding {logical}")
        if not isinstance(raw_path, str):
            _fail(f"base approval binding path is invalid: {logical}")
        path = _canonical_path(raw_path, label=f"binding {logical}")
        if not path.is_file() or sha256_file(path) != digest:
            _fail(f"base approval binding changed: {logical}")
        verified[str(logical)] = digest
    config_binding = _mapping(bindings.get("config_stage3"), field="config_stage3")
    if config_binding != {
        "path": str(config_path),
        "sha256": expected_config_sha256,
    }:
        _fail("base approval config_stage3 binding drifted")
    return {
        "approval_sha256": expected_approval_sha256,
        "approval_required_sha256": expected_required_sha256,
        "config_sha256": expected_config_sha256,
        "binding_count": len(bindings),
        "binding_sha256": verified,
    }


def _verify_prior_backup(
    receipt_path: Path,
    label: str,
    raw_evidence: object,
) -> dict[str, Any]:
    evidence = _mapping(raw_evidence, field=f"prior receipt backup.{label}")
    raw_path = evidence.get("path")
    digest = _validate_sha(evidence.get("sha256"), field=f"prior backup {label}")
    inode, device = evidence.get("inode"), evidence.get("device")
    if (
        not isinstance(raw_path, str)
        or not isinstance(inode, int)
        or not isinstance(device, int)
    ):
        _fail(f"prior backup evidence is malformed: {label}")
    path = _canonical_path(raw_path, label=f"prior backup {label}")
    info = path.stat() if path.is_file() else None
    if (
        path.parent != receipt_path.parent
        or path.is_symlink()
        or info is None
        or sha256_file(path) != digest
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_ino != inode
        or info.st_dev != device
        or evidence.get("hard_link_verified") is not True
    ):
        _fail(f"prior COMPLETE backup drifted: {label}")
    return {
        "path": str(path),
        "sha256": digest,
        "mode": 0o444,
        "inode": inode,
        "device": device,
    }


def _validate_prior_complete_receipt(
    *,
    receipt_path: Path,
    expected_sha256: str,
    expected_schema: str,
    expected_migration: str,
    expected_backup_labels: set[str],
) -> dict[str, Any]:
    if sha256_file(receipt_path) != expected_sha256:
        _fail(f"prior COMPLETE receipt SHA256 drifted: {receipt_path}")
    receipt = _mapping(load_json(receipt_path), field="prior COMPLETE receipt")
    backups = _mapping(receipt.get("backup"), field="prior COMPLETE backups")
    if (
        receipt.get("schema_version") != expected_schema
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("migration") != expected_migration
        or receipt.get("status") != "COMPLETE"
        or receipt.get("backup_read_only_after_publication") is not True
        or set(backups) != expected_backup_labels
    ):
        _fail(f"prior COMPLETE receipt semantic contract drifted: {receipt_path}")
    verified = {
        str(label): _verify_prior_backup(receipt_path, str(label), raw)
        for label, raw in backups.items()
    }
    return {
        "path": str(receipt_path),
        "sha256": expected_sha256,
        "schema_version": expected_schema,
        "migration": expected_migration,
        "status": "COMPLETE",
        "backup": verified,
        "protected_unchanged": True,
    }


def _validate_prior_migrations(
    *,
    project_root: Path,
    guard_receipt: Path,
    ema_receipt: Path,
    expected_guard_sha256: str,
    expected_ema_sha256: str,
) -> dict[str, Any]:
    guard_expected = (
        project_root / "artifacts/migrations" / GUARD_BACKUP_DIR_NAME / RECEIPT_NAME
    )
    ema_expected = (
        project_root / "artifacts/migrations" / EMA_BACKUP_DIR_NAME / RECEIPT_NAME
    )
    if guard_receipt != guard_expected or ema_receipt != ema_expected:
        _fail("prior COMPLETE receipt canonical path drifted")
    return {
        "guard_alignment": _validate_prior_complete_receipt(
            receipt_path=guard_receipt,
            expected_sha256=expected_guard_sha256,
            expected_schema=GUARD_RECEIPT_SCHEMA,
            expected_migration=GUARD_MIGRATION_KIND,
            expected_backup_labels={"run_contract", "checkpoint"},
        ),
        "ema_device": _validate_prior_complete_receipt(
            receipt_path=ema_receipt,
            expected_sha256=expected_ema_sha256,
            expected_schema=EMA_RECEIPT_SCHEMA,
            expected_migration=EMA_MIGRATION_KIND,
            expected_backup_labels={"run_contract", "checkpoint"},
        ),
    }


def _validate_provenance_anchor(
    provenance: Mapping[str, Any],
    approval: Mapping[str, Any],
    *,
    approval_sha256: str,
    required_sha256: str,
    config_sha256: str,
) -> None:
    recorded = _mapping(provenance.get("stage3_approval"), field="stage3_approval")
    runtime = _mapping(provenance.get("runtime"), field="provenance.runtime")
    if (
        provenance.get("schema_version") != STAGE3_RUNTIME_SCHEMA
        or provenance.get("protocol_id") != PROTOCOL_ID
        or provenance.get("config_sha256") != config_sha256
        or recorded.get("sha256") != approval_sha256
        or recorded.get("approval_required_sha256") != required_sha256
        or provenance.get("bindings") != approval.get("bindings")
        or runtime.get("max_steps") != SCHEDULE_HORIZON_STEPS
        or "stage3_extension" in provenance
        or runtime.get("training_target_step", BASE_STEP) != BASE_STEP
    ):
        _fail("pre-extension provenance identity/schedule drifted")


def _build_extension_approval(
    *,
    approval_path: Path,
    required_path: Path,
    config_path: Path,
    backup_paths: Mapping[str, Path],
    expected_approval_sha256: str,
    expected_required_sha256: str,
    expected_config_sha256: str,
    expected_old_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = {
        "schema_version": EXTENSION_APPROVAL_SCHEMA,
        "kind": "stage3_extension_approval",
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "cycles": CYCLES,
        "base_step": BASE_STEP,
        "target_step": TARGET_STEP,
        "validation_every_steps": VALIDATION_EVERY_STEPS,
        "validation_steps": list(VALIDATION_STEPS),
        "schedule_horizon_steps": SCHEDULE_HORIZON_STEPS,
        "min_lr": MIN_LR,
        "lr_policy": LR_POLICY,
        "formal_mio100_authorized": False,
        "authorized_pipeline": ["stage3_extension", "stage4"],
        "base_stage3_approval": {
            "path": str(approval_path),
            "sha256": expected_approval_sha256,
        },
        "base_approval_required": {
            "path": str(required_path),
            "sha256": expected_required_sha256,
        },
        "base_stage3_config": {
            "path": str(config_path),
            "sha256": expected_config_sha256,
        },
        "pre_extension_run_contract": {
            "path": str(backup_paths["run_contract"]),
            "sha256": expected_old_hashes["run_contract"],
        },
        "pre_extension_last_checkpoint": {
            "path": str(backup_paths["last_checkpoint"]),
            "sha256": expected_old_hashes["last_checkpoint"],
        },
        "pre_extension_best_checkpoint": {
            "path": str(backup_paths["best_checkpoint"]),
            "sha256": expected_old_hashes["best_checkpoint"],
        },
    }
    if len(payload) != 20:
        _fail("extension approval is not the exact flat 20-field schema")
    return payload


def _extension_provenance_binding(path: Path, digest: str) -> dict[str, Any]:
    binding = {
        "path": str(path),
        "sha256": digest,
        "cycles": CYCLES,
        "base_step": BASE_STEP,
        "target_step": TARGET_STEP,
        "validation_every_steps": VALIDATION_EVERY_STEPS,
        "validation_steps": list(VALIDATION_STEPS),
        "schedule_horizon_steps": SCHEDULE_HORIZON_STEPS,
        "min_lr": MIN_LR,
        "lr_policy": LR_POLICY,
    }
    if len(binding) != 10:
        _fail("Stage3 extension provenance is not the exact ten-field schema")
    return binding


def _build_new_provenance(
    old: Mapping[str, Any],
    *,
    old_semantic: Mapping[str, str],
    new_semantic: Mapping[str, str],
    extension_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded_semantic = _mapping(
        old.get("semantic_source_sha256"), field="old semantic_source_sha256"
    )
    if dict(recorded_semantic) != dict(old_semantic):
        _fail("old provenance semantic-source map differs from frozen old map")
    changed = sorted(
        path for path in old_semantic if old_semantic[path] != new_semantic[path]
    )
    if old_semantic.keys() != new_semantic.keys() or not changed:
        _fail("semantic-source maps must share keys and contain a real change")
    runtime_old = _mapping(old.get("runtime"), field="old provenance.runtime")
    if runtime_old.get("training_target_step", BASE_STEP) != BASE_STEP:
        _fail("old runtime training target is not the original 12000 boundary")

    new = copy.deepcopy(dict(old))
    new["semantic_source_sha256"] = dict(new_semantic)
    runtime_new = copy.deepcopy(dict(runtime_old))
    runtime_new["training_target_step"] = TARGET_STEP
    new["runtime"] = runtime_new
    new["stage3_extension"] = dict(extension_binding)

    allowed_top = {"semantic_source_sha256", "runtime", "stage3_extension"}
    for key in old:
        if key not in allowed_top:
            _assert_bit_exact(old[key], new[key], path=f"provenance.{key}")
    if set(new) != {*old, "stage3_extension"}:
        _fail("unexpected top-level provenance field change")
    for key in runtime_old:
        _assert_bit_exact(
            runtime_old[key], runtime_new[key], path=f"provenance.runtime.{key}"
        )
    if set(runtime_new) != {*runtime_old, "training_target_step"}:
        _fail("unexpected runtime provenance field change")
    if dict(new["stage3_extension"]) != dict(extension_binding):
        _fail("extension provenance binding changed while installing")

    changes = {
        "semantic_source_leaf_diffs": [
            {
                "path": f"semantic_source_sha256.{path}",
                "old": old_semantic[path],
                "new": new_semantic[path],
            }
            for path in changed
        ],
        "runtime_training_target_step": {
            "path": "runtime.training_target_step",
            "old_present": "training_target_step" in runtime_old,
            "old": runtime_old.get("training_target_step"),
            "new": TARGET_STEP,
        },
        "added_stage3_extension": dict(extension_binding),
    }
    return new, changes


def _make_candidate(parent: Path, name: str, suffix: str) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=f".{name}.", suffix=suffix, dir=parent)
    os.close(descriptor)
    candidate = Path(raw)
    candidate.unlink()
    return candidate


def _replace_and_fsync(candidate: Path, destination: Path) -> None:
    os.replace(candidate, destination)
    fsync_directory(destination.parent)


def _copy_backup(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        _fail(f"refusing existing extension backup: {destination}")
    mode = stat.S_IMODE(source.stat().st_mode)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with (
            source.open("rb") as reader,
            os.fdopen(descriptor, "wb", closefd=False) as writer,
        ):
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        os.close(descriptor)
    os.chmod(destination, 0o444, follow_symlinks=False)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    fsync_directory(destination.parent)
    source_info, backup_info = source.stat(), destination.stat()
    if (
        source_info.st_dev != backup_info.st_dev
        or source_info.st_ino == backup_info.st_ino
        or destination.is_symlink()
        or stat.S_IMODE(backup_info.st_mode) != 0o444
        or sha256_file(source) != sha256_file(destination)
    ):
        _fail(
            f"extension backup is not a distinct immutable same-disk copy: {destination}"
        )
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "source_mode": mode,
        "mode": 0o444,
        "device": backup_info.st_dev,
        "inode": backup_info.st_ino,
        "distinct_inode_from_live_at_creation": True,
    }


def _verify_extension_backups(
    *,
    backup_dir: Path,
    raw_evidence: Mapping[str, Any],
    expected_old_hashes: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    names = {
        "run_contract": "run_contract.json",
        "last_checkpoint": "last.pth",
        "best_checkpoint": "best_ema.pth",
    }
    if set(raw_evidence) != set(names):
        _fail("extension receipt backup labels drifted")
    identities: set[tuple[int, int]] = set()
    result: dict[str, dict[str, Any]] = {}
    for label, basename in names.items():
        evidence = _mapping(raw_evidence[label], field=f"extension backup {label}")
        expected_path = backup_dir / basename
        if evidence.get("path") != str(expected_path):
            _fail(f"extension backup path drifted: {label}")
        path = _canonical_path(expected_path, label=f"extension backup {label}")
        info = path.stat() if path.is_file() else None
        source_mode = evidence.get("source_mode")
        if (
            info is None
            or path.is_symlink()
            or not isinstance(source_mode, int)
            or evidence.get("sha256") != expected_old_hashes[label]
            or sha256_file(path) != expected_old_hashes[label]
            or stat.S_IMODE(info.st_mode) != 0o444
            or evidence.get("mode") != 0o444
            or evidence.get("device") != info.st_dev
            or evidence.get("inode") != info.st_ino
            or evidence.get("distinct_inode_from_live_at_creation") is not True
        ):
            _fail(f"extension backup content/identity drifted: {label}")
        identity = (info.st_dev, info.st_ino)
        if identity in identities:
            _fail("extension backups alias the same inode")
        identities.add(identity)
        result[label] = dict(evidence)
    return result


def _restore_backup(backup: Path, destination: Path, *, mode: int) -> None:
    candidate = _make_candidate(destination.parent, destination.name, ".rollback")
    try:
        shutil.copyfile(backup, candidate)
        os.chmod(candidate, mode, follow_symlinks=False)
        with candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        _replace_and_fsync(candidate, destination)
    finally:
        candidate.unlink(missing_ok=True)


def _remove_extension_approval(path: Path, *, expected_new_sha256: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    _reject_symlink_chain(path, label="extension approval rollback")
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != expected_new_sha256
    ):
        _fail("cannot remove unknown extension approval during rollback")
    path.unlink()
    fsync_directory(path.parent)


def _resolve_paths(project_root: str | Path) -> dict[str, Path]:
    root = _canonical_path(project_root, label="project root")
    paths = {
        "project_root": root,
        "run_contract": root / "artifacts/checkpoints/stage3/run_contract.json",
        "last_checkpoint": root / "artifacts/checkpoints/stage3/last.pth",
        "best_checkpoint": root / "artifacts/checkpoints/stage3/best_ema.pth",
        "state": root / "artifacts/orchestration/state.json",
        "approval": root / "artifacts/approvals/STAGE3_APPROVED.json",
        "approval_required": root / "artifacts/approvals/STAGE3_APPROVAL_REQUIRED.json",
        "config": root / "configs/stage3_planner.yaml",
        "guard_receipt": root
        / "artifacts/migrations"
        / GUARD_BACKUP_DIR_NAME
        / RECEIPT_NAME,
        "ema_receipt": root
        / "artifacts/migrations"
        / EMA_BACKUP_DIR_NAME
        / RECEIPT_NAME,
        "backup_dir": root / "artifacts/migrations" / BACKUP_DIR_NAME,
        "extension_approval": root / "artifacts/approvals" / EXTENSION_APPROVAL_NAME,
    }
    for label, path in paths.items():
        _reject_symlink_chain(path, label=label)
    return paths


def _validate_live_paths(
    paths: Mapping[str, Path], *, extension_must_be_absent: bool
) -> None:
    for label in (
        "run_contract",
        "last_checkpoint",
        "best_checkpoint",
        "state",
        "approval",
        "approval_required",
        "config",
        "guard_receipt",
        "ema_receipt",
    ):
        path = paths[label]
        if path.is_symlink() or not path.is_file():
            _fail(f"missing or symlinked canonical {label}: {path}")
    if extension_must_be_absent and (
        paths["extension_approval"].exists() or paths["extension_approval"].is_symlink()
    ):
        _fail("canonical extension approval already exists")


def _protected_hashes(
    *,
    state: str,
    approval: str,
    required: str,
    config: str,
    guard: str,
    ema: str,
) -> dict[str, str]:
    return {
        "state": state,
        "approval": approval,
        "approval_required": required,
        "config": config,
        "guard_receipt": guard,
        "ema_receipt": ema,
    }


def _verify_hash_set(paths: Mapping[str, Path], expected: Mapping[str, str]) -> None:
    for label, digest in expected.items():
        if sha256_file(paths[label]) != digest:
            _fail(f"protected artifact changed: {label}")


def migrate_stage3_extension_provenance(
    *,
    project_root: str | Path,
    expected_run_contract_sha256: str,
    expected_last_checkpoint_sha256: str,
    expected_best_checkpoint_sha256: str,
    expected_state_sha256: str,
    expected_base_approval_sha256: str,
    expected_approval_required_sha256: str,
    expected_stage3_config_sha256: str,
    expected_guard_receipt_sha256: str,
    expected_ema_receipt_sha256: str,
    expected_old_source_map: Mapping[str, str],
    expected_new_source_map: Mapping[str, str],
    execute: bool = False,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Build or publish the exact extension approval and three-way provenance."""

    _assert_cpu_only()
    old_semantic, new_semantic = _validate_frozen_source_transition(
        expected_old_source_map,
        expected_new_source_map,
    )
    for field, digest in (
        ("run contract", expected_run_contract_sha256),
        ("last checkpoint", expected_last_checkpoint_sha256),
        ("best checkpoint", expected_best_checkpoint_sha256),
        ("state", expected_state_sha256),
        ("base approval", expected_base_approval_sha256),
        ("approval-required", expected_approval_required_sha256),
        ("Stage3 config", expected_stage3_config_sha256),
        ("guard receipt", expected_guard_receipt_sha256),
        ("EMA receipt", expected_ema_receipt_sha256),
    ):
        _validate_sha(digest, field=field)
    if execute and confirmation_token != CONFIRMATION_TOKEN:
        _fail("execution requires the exact extension migration confirmation token")

    paths = _resolve_paths(project_root)
    _validate_live_paths(paths, extension_must_be_absent=True)
    if paths["backup_dir"].exists() or paths["backup_dir"].is_symlink():
        _fail("dedicated extension migration directory already exists")
    expected_old_hashes = {
        "run_contract": expected_run_contract_sha256,
        "last_checkpoint": expected_last_checkpoint_sha256,
        "best_checkpoint": expected_best_checkpoint_sha256,
    }
    protected = _protected_hashes(
        state=expected_state_sha256,
        approval=expected_base_approval_sha256,
        required=expected_approval_required_sha256,
        config=expected_stage3_config_sha256,
        guard=expected_guard_receipt_sha256,
        ema=expected_ema_receipt_sha256,
    )

    with _single_writer_lock(paths["backup_dir"].parent):
        _verify_hash_set(paths, expected_old_hashes | protected)
        state_evidence = _validate_failed_state(paths["state"], expected_state_sha256)
        approval_evidence = _validate_base_approval(
            paths["approval"],
            paths["approval_required"],
            paths["config"],
            expected_approval_sha256=expected_base_approval_sha256,
            expected_required_sha256=expected_approval_required_sha256,
            expected_config_sha256=expected_stage3_config_sha256,
        )
        prior_evidence = _validate_prior_migrations(
            project_root=paths["project_root"],
            guard_receipt=paths["guard_receipt"],
            ema_receipt=paths["ema_receipt"],
            expected_guard_sha256=expected_guard_receipt_sha256,
            expected_ema_sha256=expected_ema_receipt_sha256,
        )
        physical_semantic = dict(
            semantic_source_hashes(paths["project_root"], entrypoints=ENTRYPOINTS)
        )
        if physical_semantic != new_semantic:
            _fail("physical semantic-source map differs from frozen new map")

        contract = _mapping(load_json(paths["run_contract"]), field="run contract")
        if contract.get("schema_version") != STAGE3_RUNTIME_SCHEMA:
            _fail("Stage3 run-contract schema drifted")
        old_provenance = _mapping(contract.get("provenance"), field="run provenance")
        last = _load_checkpoint(paths["last_checkpoint"])
        best = _load_checkpoint(paths["best_checkpoint"])
        _verify_hash_set(paths, expected_old_hashes)
        _validate_checkpoint_pair(last, best)
        if last.get("provenance") != old_provenance:
            _fail("run-contract/last/best provenance differs before migration")
        base_approval = _mapping(load_json(paths["approval"]), field="base approval")
        _validate_provenance_anchor(
            old_provenance,
            base_approval,
            approval_sha256=expected_base_approval_sha256,
            required_sha256=expected_approval_required_sha256,
            config_sha256=expected_stage3_config_sha256,
        )
        if (
            dict(
                _mapping(
                    old_provenance.get("semantic_source_sha256"), field="old semantic"
                )
            )
            != old_semantic
        ):
            _fail("run-contract source map differs from frozen old map")

        backup_paths = {
            "run_contract": paths["backup_dir"] / "run_contract.json",
            "last_checkpoint": paths["backup_dir"] / "last.pth",
            "best_checkpoint": paths["backup_dir"] / "best_ema.pth",
        }
        extension_payload = _build_extension_approval(
            approval_path=paths["approval"],
            required_path=paths["approval_required"],
            config_path=paths["config"],
            backup_paths=backup_paths,
            expected_approval_sha256=expected_base_approval_sha256,
            expected_required_sha256=expected_approval_required_sha256,
            expected_config_sha256=expected_stage3_config_sha256,
            expected_old_hashes=expected_old_hashes,
        )
        candidates = {
            "extension_approval": _make_candidate(
                paths["extension_approval"].parent,
                paths["extension_approval"].name,
                ".extension.candidate.json",
            ),
            "run_contract": _make_candidate(
                paths["run_contract"].parent,
                paths["run_contract"].name,
                ".extension.candidate.json",
            ),
            "last_checkpoint": _make_candidate(
                paths["last_checkpoint"].parent,
                paths["last_checkpoint"].name,
                ".extension.candidate.pth",
            ),
            "best_checkpoint": _make_candidate(
                paths["best_checkpoint"].parent,
                paths["best_checkpoint"].name,
                ".extension.candidate.pth",
            ),
        }
        backups: dict[str, Any] = {}
        receipt_path = paths["backup_dir"] / RECEIPT_NAME
        try:
            atomic_write_json(candidates["extension_approval"], extension_payload)
            if load_json(candidates["extension_approval"]) != extension_payload:
                _fail("extension approval candidate failed JSON round trip")
            extension_sha = sha256_file(candidates["extension_approval"])
            extension_binding = _extension_provenance_binding(
                paths["extension_approval"], extension_sha
            )
            new_provenance, provenance_changes = _build_new_provenance(
                old_provenance,
                old_semantic=old_semantic,
                new_semantic=new_semantic,
                extension_binding=extension_binding,
            )
            new_contract = copy.deepcopy(dict(contract))
            new_contract["provenance"] = new_provenance
            _assert_bit_exact(
                {key: value for key, value in contract.items() if key != "provenance"},
                {
                    key: value
                    for key, value in new_contract.items()
                    if key != "provenance"
                },
                path="run_contract.outside_provenance",
            )
            new_last = copy.copy(last)
            new_last["provenance"] = new_provenance
            new_best = copy.copy(best)
            new_best["provenance"] = new_provenance

            atomic_write_json(candidates["run_contract"], new_contract)
            atomic_torch_save(new_last, candidates["last_checkpoint"])
            atomic_torch_save(new_best, candidates["best_checkpoint"])
            reloaded_contract = _mapping(
                load_json(candidates["run_contract"]), field="candidate run contract"
            )
            reloaded_last = _load_checkpoint(candidates["last_checkpoint"])
            reloaded_best = _load_checkpoint(candidates["best_checkpoint"])
            _validate_checkpoint_pair(reloaded_last, reloaded_best)
            if (
                reloaded_contract.get("provenance") != new_provenance
                or reloaded_last.get("provenance") != new_provenance
                or reloaded_best.get("provenance") != new_provenance
            ):
                _fail("candidate three-way provenance identity failed")
            section_evidence = {
                "last_checkpoint": _checkpoint_section_evidence(last, reloaded_last),
                "best_checkpoint": _checkpoint_section_evidence(best, reloaded_best),
            }
            _assert_bit_exact(
                {key: value for key, value in contract.items() if key != "provenance"},
                {
                    key: value
                    for key, value in reloaded_contract.items()
                    if key != "provenance"
                },
                path="candidate_contract.outside_provenance",
            )
            new_hashes = {
                label: sha256_file(path) for label, path in candidates.items()
            }
            receipt: dict[str, Any] = {
                "schema_version": RECEIPT_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "migration": MIGRATION_KIND,
                "status": "PREPARED" if execute else "DRY_RUN",
                "created_utc": utc_now_iso(),
                "cpu_only": True,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "base_step": BASE_STEP,
                "target_step": TARGET_STEP,
                "validation_steps": list(VALIDATION_STEPS),
                "schedule_horizon_steps": SCHEDULE_HORIZON_STEPS,
                "lr_policy": LR_POLICY,
                "old": {
                    label: {"path": str(paths[label]), "sha256": digest}
                    for label, digest in expected_old_hashes.items()
                }
                | {"provenance_json_sha256": sha256_json(dict(old_provenance))},
                "new": new_hashes
                | {"provenance_json_sha256": sha256_json(new_provenance)},
                "provenance_changes": provenance_changes,
                "semantic_source_count": len(old_semantic),
                "semantic_source_changed_count": len(
                    provenance_changes["semantic_source_leaf_diffs"]
                ),
                "extension_approval_field_count": len(extension_payload),
                "extension_provenance_field_count": len(extension_binding),
                "checkpoint_top_level_count": EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT,
                "checkpoint_bit_exact_outside_provenance_count": (
                    EXPECTED_UNCHANGED_CHECKPOINT_TOP_LEVEL_COUNT
                ),
                "checkpoint_section_fingerprints": section_evidence,
                "both_checkpoints_bit_exact_outside_provenance": True,
                "run_contract_bit_exact_outside_provenance": True,
                "three_live_artifacts_share_exact_provenance": True,
                "base_approval_and_22_bindings_unchanged": approval_evidence,
                "prior_complete_migrations": prior_evidence,
                "orchestration_state": {
                    "path": str(paths["state"]),
                    "sha256": expected_state_sha256,
                    **{
                        key: state_evidence.get(key)
                        for key in (
                            "status",
                            "current_stage",
                            "gpu",
                            "last_exit_code",
                            "next_command",
                        )
                    },
                },
                "protected_hashes": protected,
                "execution_confirmation_token_sha256": (
                    hashlib.sha256(CONFIRMATION_TOKEN.encode()).hexdigest()
                    if execute
                    else None
                ),
                "backup": backups,
                "migration_script_sha256": sha256_file(Path(__file__).resolve()),
            }
            if not execute:
                _assert_cpu_only()
                return receipt

            _verify_hash_set(paths, expected_old_hashes | protected)
            if (
                dict(
                    semantic_source_hashes(
                        paths["project_root"], entrypoints=ENTRYPOINTS
                    )
                )
                != new_semantic
            ):
                _fail("semantic sources changed before publication")
            migrations = paths["backup_dir"].parent
            if not migrations.is_dir() or migrations.is_symlink():
                _fail("canonical migrations directory is missing or symlinked")
            paths["backup_dir"].mkdir(parents=False, exist_ok=False)
            fsync_directory(migrations)
            devices = {
                paths["backup_dir"].stat().st_dev,
                paths["run_contract"].stat().st_dev,
                paths["last_checkpoint"].stat().st_dev,
                paths["best_checkpoint"].stat().st_dev,
            }
            if len(devices) != 1:
                _fail("extension backups are not on the live-artifact filesystem")
            for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
                backups[label] = _copy_backup(paths[label], backup_paths[label])
            if len({(row["device"], row["inode"]) for row in backups.values()}) != 3:
                _fail("extension backups do not have three distinct inode identities")
            receipt["backup"] = backups
            atomic_write_json(receipt_path, receipt)
            fsync_directory(paths["backup_dir"])

            # Approval is published first.  Any interruption after this point is
            # a PREPARED old/new mixture recoverable from the immutable backups.
            for label in (
                "extension_approval",
                "best_checkpoint",
                "last_checkpoint",
                "run_contract",
            ):
                destination = (
                    paths["extension_approval"]
                    if label == "extension_approval"
                    else paths[label]
                )
                _replace_and_fsync(candidates[label], destination)
                if sha256_file(destination) != new_hashes[label]:
                    _fail(f"published {label} hash differs from verified candidate")

            published_contract = _mapping(
                load_json(paths["run_contract"]), field="published run contract"
            )
            published_last = _load_checkpoint(paths["last_checkpoint"])
            published_best = _load_checkpoint(paths["best_checkpoint"])
            _validate_checkpoint_pair(published_last, published_best)
            if (
                published_contract.get("provenance") != new_provenance
                or published_last.get("provenance") != new_provenance
                or published_best.get("provenance") != new_provenance
            ):
                _fail("published three-way provenance identity failed")
            _checkpoint_section_evidence(last, published_last)
            _checkpoint_section_evidence(best, published_best)
            _assert_bit_exact(
                {key: value for key, value in contract.items() if key != "provenance"},
                {
                    key: value
                    for key, value in published_contract.items()
                    if key != "provenance"
                },
                path="published_contract.outside_provenance",
            )
            _verify_hash_set(paths, protected)
            _validate_prior_migrations(
                project_root=paths["project_root"],
                guard_receipt=paths["guard_receipt"],
                ema_receipt=paths["ema_receipt"],
                expected_guard_sha256=expected_guard_receipt_sha256,
                expected_ema_sha256=expected_ema_receipt_sha256,
            )
            if (
                dict(
                    semantic_source_hashes(
                        paths["project_root"], entrypoints=ENTRYPOINTS
                    )
                )
                != new_semantic
            ):
                _fail("semantic sources changed during publication")
            _verify_extension_backups(
                backup_dir=paths["backup_dir"],
                raw_evidence=backups,
                expected_old_hashes=expected_old_hashes,
            )
            receipt["status"] = "COMPLETE"
            receipt["completed_utc"] = utc_now_iso()
            receipt["backup_read_only_after_publication"] = True
            receipt["protected_artifacts_unchanged_after_publication"] = True
            _assert_cpu_only()
            atomic_write_json(receipt_path, receipt)
            return receipt
        except BaseException as original_error:
            if execute and backups:
                rollback_errors: list[str] = []
                for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
                    try:
                        evidence = _mapping(backups[label], field=f"backup {label}")
                        _restore_backup(
                            Path(str(evidence["path"])),
                            paths[label],
                            mode=int(evidence["source_mode"]),
                        )
                        if sha256_file(paths[label]) != expected_old_hashes[label]:
                            _fail(f"rollback hash mismatch: {label}")
                    except BaseException as exc:
                        rollback_errors.append(f"{label}: {type(exc).__name__}: {exc}")
                try:
                    _remove_extension_approval(
                        paths["extension_approval"],
                        expected_new_sha256=new_hashes["extension_approval"],
                    )
                except BaseException as exc:
                    rollback_errors.append(
                        f"extension_approval: {type(exc).__name__}: {exc}"
                    )
                if receipt_path.parent.is_dir():
                    atomic_write_json(
                        receipt_path,
                        {
                            "schema_version": RECEIPT_SCHEMA,
                            "protocol_id": PROTOCOL_ID,
                            "migration": MIGRATION_KIND,
                            "status": "ROLLBACK_FAILED"
                            if rollback_errors
                            else "ROLLED_BACK",
                            "rolled_back_utc": utc_now_iso(),
                            "old": expected_old_hashes,
                            "backup": backups,
                            "rollback_errors": rollback_errors,
                        },
                    )
                if rollback_errors:
                    raise Stage3ExtensionMigrationError(
                        "publication failed and rollback was incomplete: "
                        + "; ".join(rollback_errors)
                    ) from original_error
            raise
        finally:
            for candidate in candidates.values():
                candidate.unlink(missing_ok=True)


def recover_prepared_stage3_extension_provenance(
    *,
    project_root: str | Path,
    expected_run_contract_sha256: str,
    expected_last_checkpoint_sha256: str,
    expected_best_checkpoint_sha256: str,
    expected_state_sha256: str,
    expected_base_approval_sha256: str,
    expected_approval_required_sha256: str,
    expected_stage3_config_sha256: str,
    expected_guard_receipt_sha256: str,
    expected_ema_receipt_sha256: str,
    expected_old_source_map: Mapping[str, str],
    expected_new_source_map: Mapping[str, str],
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Roll any valid PREPARED old/new publication mixture back exactly."""

    if confirmation_token != RECOVERY_CONFIRMATION_TOKEN:
        _fail("PREPARED recovery requires the exact recovery confirmation token")
    _assert_cpu_only()
    old_semantic, new_semantic = _validate_frozen_source_transition(
        expected_old_source_map,
        expected_new_source_map,
    )
    paths = _resolve_paths(project_root)
    _validate_live_paths(paths, extension_must_be_absent=False)
    receipt_path = paths["backup_dir"] / RECEIPT_NAME
    if (
        paths["backup_dir"].is_symlink()
        or not paths["backup_dir"].is_dir()
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
    ):
        _fail("PREPARED recovery requires the dedicated non-symlink receipt")
    expected_old_hashes = {
        "run_contract": expected_run_contract_sha256,
        "last_checkpoint": expected_last_checkpoint_sha256,
        "best_checkpoint": expected_best_checkpoint_sha256,
    }
    protected = _protected_hashes(
        state=expected_state_sha256,
        approval=expected_base_approval_sha256,
        required=expected_approval_required_sha256,
        config=expected_stage3_config_sha256,
        guard=expected_guard_receipt_sha256,
        ema=expected_ema_receipt_sha256,
    )
    with _single_writer_lock(paths["backup_dir"].parent):
        _verify_hash_set(paths, protected)
        _validate_failed_state(paths["state"], expected_state_sha256)
        approval_evidence = _validate_base_approval(
            paths["approval"],
            paths["approval_required"],
            paths["config"],
            expected_approval_sha256=expected_base_approval_sha256,
            expected_required_sha256=expected_approval_required_sha256,
            expected_config_sha256=expected_stage3_config_sha256,
        )
        prior_evidence = _validate_prior_migrations(
            project_root=paths["project_root"],
            guard_receipt=paths["guard_receipt"],
            ema_receipt=paths["ema_receipt"],
            expected_guard_sha256=expected_guard_receipt_sha256,
            expected_ema_sha256=expected_ema_receipt_sha256,
        )
        if (
            dict(semantic_source_hashes(paths["project_root"], entrypoints=ENTRYPOINTS))
            != new_semantic
        ):
            _fail("semantic-source map drifted before PREPARED recovery")

        receipt = _mapping(load_json(receipt_path), field="extension migration receipt")
        old = _mapping(receipt.get("old"), field="receipt.old")
        new = _mapping(receipt.get("new"), field="receipt.new")
        raw_backups = _mapping(receipt.get("backup"), field="receipt.backup")
        if (
            receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("migration") != MIGRATION_KIND
            or receipt.get("status") not in {"PREPARED", "ROLLED_BACK_FROM_PREPARED"}
            or receipt.get("migration_script_sha256")
            != sha256_file(Path(__file__).resolve())
            or receipt.get("execution_confirmation_token_sha256")
            != hashlib.sha256(CONFIRMATION_TOKEN.encode()).hexdigest()
            or receipt.get("base_approval_and_22_bindings_unchanged")
            != approval_evidence
            or receipt.get("prior_complete_migrations") != prior_evidence
            or receipt.get("checkpoint_top_level_count")
            != EXPECTED_CHECKPOINT_TOP_LEVEL_COUNT
            or receipt.get("checkpoint_bit_exact_outside_provenance_count")
            != EXPECTED_UNCHANGED_CHECKPOINT_TOP_LEVEL_COUNT
            or receipt.get("extension_approval_field_count") != 20
            or receipt.get("extension_provenance_field_count") != 10
            or receipt.get("protected_hashes") != protected
        ):
            _fail("PREPARED receipt contract drifted")
        backups = _verify_extension_backups(
            backup_dir=paths["backup_dir"],
            raw_evidence=raw_backups,
            expected_old_hashes=expected_old_hashes,
        )
        new_hashes: dict[str, str] = {}
        recovered_from: dict[str, str | None] = {}
        for label, old_sha in expected_old_hashes.items():
            old_entry = _mapping(old.get(label), field=f"receipt.old.{label}")
            new_sha = _validate_sha(new.get(label), field=f"receipt.new.{label}")
            if old_entry != {"path": str(paths[label]), "sha256": old_sha}:
                _fail(f"PREPARED old anchor drifted: {label}")
            live_sha = sha256_file(paths[label])
            if live_sha not in {old_sha, new_sha}:
                _fail(f"PREPARED live artifact is neither old nor new: {label}")
            new_hashes[label] = new_sha
            recovered_from[label] = live_sha
        extension_new_sha = _validate_sha(
            new.get("extension_approval"), field="receipt.new.extension_approval"
        )
        if paths["extension_approval"].exists():
            if (
                paths["extension_approval"].is_symlink()
                or not paths["extension_approval"].is_file()
                or sha256_file(paths["extension_approval"]) != extension_new_sha
            ):
                _fail("PREPARED extension approval is not the verified new artifact")
            recovered_from["extension_approval"] = extension_new_sha
        else:
            recovered_from["extension_approval"] = None

        backup_contract = _mapping(
            load_json(Path(backups["run_contract"]["path"])),
            field="backup run contract",
        )
        backup_last = _load_checkpoint(Path(backups["last_checkpoint"]["path"]))
        backup_best = _load_checkpoint(Path(backups["best_checkpoint"]["path"]))
        _validate_checkpoint_pair(backup_last, backup_best)
        backup_provenance = _mapping(
            backup_contract.get("provenance"), field="backup provenance"
        )
        if (
            backup_contract.get("schema_version") != STAGE3_RUNTIME_SCHEMA
            or backup_last.get("provenance") != backup_provenance
            or backup_best.get("provenance") != backup_provenance
            or dict(
                _mapping(
                    backup_provenance.get("semantic_source_sha256"),
                    field="backup semantic sources",
                )
            )
            != old_semantic
        ):
            _fail("PREPARED immutable backups do not share the old provenance")
        _validate_provenance_anchor(
            backup_provenance,
            _mapping(load_json(paths["approval"]), field="base approval"),
            approval_sha256=expected_base_approval_sha256,
            required_sha256=expected_approval_required_sha256,
            config_sha256=expected_stage3_config_sha256,
        )

        # Reconstruct every deterministic part of the PREPARED publication
        # from immutable backups and protected anchors.  A PREPARED receipt is
        # intentionally not self-hashing, so recovery must not trust its new
        # hashes or provenance-diff narrative without this independent check.
        backup_paths = {
            label: Path(str(backups[label]["path"]))
            for label in ("run_contract", "last_checkpoint", "best_checkpoint")
        }
        expected_extension_payload = _build_extension_approval(
            approval_path=paths["approval"],
            required_path=paths["approval_required"],
            config_path=paths["config"],
            backup_paths=backup_paths,
            expected_approval_sha256=expected_base_approval_sha256,
            expected_required_sha256=expected_approval_required_sha256,
            expected_config_sha256=expected_stage3_config_sha256,
            expected_old_hashes=expected_old_hashes,
        )
        expected_extension_sha = _deterministic_json_file_sha256(
            expected_extension_payload
        )
        if extension_new_sha != expected_extension_sha:
            _fail("PREPARED extension approval hash is not deterministic from anchors")
        expected_extension_binding = _extension_provenance_binding(
            paths["extension_approval"], expected_extension_sha
        )
        expected_new_provenance, expected_changes = _build_new_provenance(
            backup_provenance,
            old_semantic=old_semantic,
            new_semantic=new_semantic,
            extension_binding=expected_extension_binding,
        )
        expected_new_contract = copy.deepcopy(dict(backup_contract))
        expected_new_contract["provenance"] = expected_new_provenance
        if (
            old.get("provenance_json_sha256") != sha256_json(dict(backup_provenance))
            or new.get("provenance_json_sha256") != sha256_json(expected_new_provenance)
            or new.get("run_contract")
            != _deterministic_json_file_sha256(expected_new_contract)
            or receipt.get("provenance_changes") != expected_changes
            or receipt.get("semantic_source_count") != len(old_semantic)
            or receipt.get("semantic_source_changed_count")
            != len(expected_changes["semantic_source_leaf_diffs"])
        ):
            _fail("PREPARED deterministic provenance/contract evidence drifted")
        if paths["extension_approval"].exists() and (
            load_json(paths["extension_approval"]) != expected_extension_payload
        ):
            _fail("PREPARED live extension approval payload drifted")

        live_contract_sha = sha256_file(paths["run_contract"])
        if live_contract_sha == new_hashes["run_contract"]:
            live_contract = _mapping(
                load_json(paths["run_contract"]), field="PREPARED new run contract"
            )
            if live_contract != expected_new_contract:
                _fail("PREPARED new run contract content drifted")
        for label, backup_payload in (
            ("last_checkpoint", backup_last),
            ("best_checkpoint", backup_best),
        ):
            if sha256_file(paths[label]) != new_hashes[label]:
                continue
            live_payload = _load_checkpoint(paths[label])
            if live_payload.get("provenance") != expected_new_provenance:
                _fail(f"PREPARED new {label} provenance drifted")
            _checkpoint_section_evidence(backup_payload, live_payload)

        if receipt.get("status") == "ROLLED_BACK_FROM_PREPARED":
            if (
                receipt.get("recovery_confirmation_token_sha256")
                != hashlib.sha256(RECOVERY_CONFIRMATION_TOKEN.encode()).hexdigest()
                or paths["extension_approval"].exists()
                or any(
                    sha256_file(paths[label]) != digest
                    for label, digest in expected_old_hashes.items()
                )
            ):
                _fail("finalized PREPARED recovery drifted")
            return dict(receipt)

        for label in ("run_contract", "last_checkpoint", "best_checkpoint"):
            _restore_backup(
                Path(backups[label]["path"]),
                paths[label],
                mode=int(backups[label]["source_mode"]),
            )
            if sha256_file(paths[label]) != expected_old_hashes[label]:
                _fail(f"PREPARED rollback output mismatch: {label}")
        _remove_extension_approval(
            paths["extension_approval"], expected_new_sha256=extension_new_sha
        )
        _verify_hash_set(paths, expected_old_hashes | protected)
        if (
            dict(semantic_source_hashes(paths["project_root"], entrypoints=ENTRYPOINTS))
            != new_semantic
        ):
            _fail("semantic sources changed during PREPARED recovery")
        recovered = dict(receipt)
        recovered["status"] = "ROLLED_BACK_FROM_PREPARED"
        recovered["recovered_utc"] = utc_now_iso()
        recovered["recovered_from_live_sha256"] = recovered_from
        recovered["recovery_confirmation_token_sha256"] = hashlib.sha256(
            RECOVERY_CONFIRMATION_TOKEN.encode()
        ).hexdigest()
        recovered["backup_read_only_after_recovery"] = True
        recovered["protected_artifacts_unchanged_after_recovery"] = True
        _assert_cpu_only()
        atomic_write_json(receipt_path, recovered)
        return recovered


def _parse_source_map(raw: str, *, field: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"{field} must be a JSON object")
    try:
        return _validate_source_map(value, field=field)
    except Stage3ExtensionMigrationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--expected-run-contract-sha256", default=AUDITED_RUN_CONTRACT_SHA256
    )
    parser.add_argument(
        "--expected-last-checkpoint-sha256", default=AUDITED_LAST_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--expected-best-checkpoint-sha256", default=AUDITED_BEST_CHECKPOINT_SHA256
    )
    parser.add_argument("--expected-state-sha256", default=AUDITED_STATE_SHA256)
    parser.add_argument(
        "--expected-base-approval-sha256", default=AUDITED_BASE_APPROVAL_SHA256
    )
    parser.add_argument(
        "--expected-approval-required-sha256",
        default=AUDITED_APPROVAL_REQUIRED_SHA256,
    )
    parser.add_argument(
        "--expected-stage3-config-sha256", default=AUDITED_STAGE3_CONFIG_SHA256
    )
    parser.add_argument(
        "--expected-guard-receipt-sha256", default=AUDITED_GUARD_RECEIPT_SHA256
    )
    parser.add_argument(
        "--expected-ema-receipt-sha256", default=AUDITED_EMA_RECEIPT_SHA256
    )
    parser.add_argument(
        "--expected-old-source-map-json",
        default=json.dumps(AUDITED_OLD_SEMANTIC_SOURCE_SHA256, sort_keys=True),
    )
    parser.add_argument(
        "--expected-new-source-map-json",
        default=json.dumps(AUDITED_NEW_SEMANTIC_SOURCE_SHA256, sort_keys=True),
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--recover-prepared", action="store_true")
    parser.add_argument("--confirmation-token")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        old_map = _parse_source_map(
            arguments.expected_old_source_map_json, field="expected old source map"
        )
        new_map = _parse_source_map(
            arguments.expected_new_source_map_json, field="expected new source map"
        )
        common = {
            "project_root": arguments.project_root,
            "expected_run_contract_sha256": arguments.expected_run_contract_sha256,
            "expected_last_checkpoint_sha256": arguments.expected_last_checkpoint_sha256,
            "expected_best_checkpoint_sha256": arguments.expected_best_checkpoint_sha256,
            "expected_state_sha256": arguments.expected_state_sha256,
            "expected_base_approval_sha256": arguments.expected_base_approval_sha256,
            "expected_approval_required_sha256": (
                arguments.expected_approval_required_sha256
            ),
            "expected_stage3_config_sha256": arguments.expected_stage3_config_sha256,
            "expected_guard_receipt_sha256": arguments.expected_guard_receipt_sha256,
            "expected_ema_receipt_sha256": arguments.expected_ema_receipt_sha256,
            "expected_old_source_map": old_map,
            "expected_new_source_map": new_map,
        }
        if arguments.recover_prepared:
            receipt = recover_prepared_stage3_extension_provenance(
                **common, confirmation_token=arguments.confirmation_token
            )
        else:
            receipt = migrate_stage3_extension_provenance(
                **common,
                execute=arguments.execute,
                confirmation_token=arguments.confirmation_token,
            )
    except (Stage3ExtensionMigrationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
