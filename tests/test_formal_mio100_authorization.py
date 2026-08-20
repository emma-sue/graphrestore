from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import authorize_formal_mio100 as authorizer
from src.evaluation import formal_inventory
from src.evaluation.formal_inventory import (
    FormalInventoryError,
    REQUIRED_AUTHORIZATION_BINDINGS,
    build_formal_authorization_payload,
    build_formal_data_inventory,
    load_formal_data_inventory,
    validate_lightweight_authorization,
    write_new_read_only_json,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _small_inventory_fixture(tmp_path: Path) -> dict[str, object]:
    native_paths = []
    target = (tmp_path / "gt/000001.png").resolve()
    target.parent.mkdir(parents=True)
    target.write_bytes(b"target-bytes")
    rows = []
    for index in range(2):
        native = (tmp_path / f"native/{index:06d}.png").resolve()
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_bytes(f"native-{index}".encode())
        native_paths.append(native)
        rows.append(
            {
                "schema_version": "graphrestore.agenticir_online_canonical.v1",
                "sample_id": f"test/A/rain+haze/{index:06d}",
                "group": "A",
                "degradations": ["rain", "haze"],
                "source": "AgenticIR",
                "split": "test",
                "input_mode": "agenticir_online_canonical",
                "native_lq_path": str(native),
                "input_path": str(native),
                "gt_path": str(target),
            }
        )
    manifest = (tmp_path / formal_inventory.FORMAL_MANIFEST_FILENAME).resolve()
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    protocol = (tmp_path / "FORMAL_PROTOCOL.md").resolve()
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    protocol.chmod(0o444)
    kwargs = {
        "expected_manifest_sha256": _sha(manifest),
        "expected_authorization_protocol_sha256": _sha(protocol),
        "expected_row_count": 2,
        "expected_group_counts": {"A": 2},
        "expected_combination_counts": {"rain+haze": 2},
        "expected_unique_native_count": 2,
        "expected_unique_target_count": 1,
        "expected_unique_file_count": 3,
    }
    return {
        "manifest": manifest,
        "protocol": protocol,
        "native_paths": native_paths,
        "target": target,
        "kwargs": kwargs,
    }


def _load_small_inventory(
    inventory: Path,
    fixture: dict[str, object],
    *,
    verify_file_bytes: bool = True,
):
    kwargs = dict(fixture["kwargs"])
    protocol_sha = kwargs.pop("expected_authorization_protocol_sha256")
    return load_formal_data_inventory(
        inventory,
        expected_manifest_path=fixture["manifest"],
        expected_authorization_protocol_path=fixture["protocol"],
        expected_authorization_protocol_sha256=protocol_sha,
        verify_file_bytes=verify_file_bytes,
        **kwargs,
    )


def test_hash_only_inventory_is_strict_sorted_and_no_clobber(tmp_path: Path) -> None:
    fixture = _small_inventory_fixture(tmp_path)
    payload = build_formal_data_inventory(
        fixture["manifest"],
        authorization_protocol=fixture["protocol"],
        **fixture["kwargs"],
    )
    inventory = (tmp_path / "formal_data_inventory.json").resolve()
    write_new_read_only_json(inventory, payload)
    validated = _load_small_inventory(inventory, fixture)
    assert len(validated.rows) == 2
    assert len(validated.files) == 3
    assert inventory.stat().st_mode & 0o777 == 0o444
    assert [str(path) for path in validated.files] == sorted(
        str(path) for path in validated.files
    )
    with pytest.raises(FormalInventoryError, match="refusing to overwrite"):
        write_new_read_only_json(inventory, payload)


def test_inventory_rejects_future_file_and_inventory_tamper(tmp_path: Path) -> None:
    fixture = _small_inventory_fixture(tmp_path)
    payload = build_formal_data_inventory(
        fixture["manifest"],
        authorization_protocol=fixture["protocol"],
        **fixture["kwargs"],
    )
    inventory = (tmp_path / "formal_data_inventory.json").resolve()
    write_new_read_only_json(inventory, payload)
    native = fixture["native_paths"][1]
    native.write_bytes(b"future-drift")
    with pytest.raises(FormalInventoryError, match="identity drifted"):
        _load_small_inventory(inventory, fixture)

    native.write_bytes(b"native-1")
    inventory.chmod(0o644)
    changed = json.loads(inventory.read_text(encoding="utf-8"))
    changed["rows"][0]["sample_id"] = "tampered"
    inventory.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    inventory.chmod(0o444)
    with pytest.raises(FormalInventoryError, match="rows digest drifted"):
        _load_small_inventory(inventory, fixture, verify_file_bytes=False)


def test_inventory_rejects_unique_count_drift(tmp_path: Path) -> None:
    fixture = _small_inventory_fixture(tmp_path)
    kwargs = dict(fixture["kwargs"])
    kwargs["expected_unique_target_count"] = 2
    with pytest.raises(FormalInventoryError, match="GT unique count drifted"):
        build_formal_data_inventory(
            fixture["manifest"],
            authorization_protocol=fixture["protocol"],
            **kwargs,
        )


def test_approval_requires_exact_28_bindings_and_mode_0444(tmp_path: Path) -> None:
    paths = {}
    for index, name in enumerate(REQUIRED_AUTHORIZATION_BINDINGS):
        path = (tmp_path / "bindings" / f"{index:02d}-{name}.txt").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        paths[name] = path
    payload = build_formal_authorization_payload(paths)
    approval = (tmp_path / "FORMAL_MIO100_APPROVED.json").resolve()
    write_new_read_only_json(approval, payload)
    accepted = validate_lightweight_authorization(
        approval, expected_binding_paths=paths
    )
    assert len(accepted["bindings"]) == 28
    assert approval.stat().st_mode & 0o777 == 0o444

    approval.chmod(0o644)
    with pytest.raises(FormalInventoryError, match="exact mode 0444"):
        validate_lightweight_authorization(approval, expected_binding_paths=paths)
    missing = dict(paths)
    missing.pop("formal_data_inventory")
    with pytest.raises(FormalInventoryError, match="binding keys drifted"):
        build_formal_authorization_payload(missing)


def test_authorizer_fails_before_stage4_complete_or_with_active_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authorizer, "assert_standard_library_only", lambda: None)

    def empty_gpu(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="")

    with pytest.raises(FormalInventoryError, match="missing Stage4 completion"):
        authorizer.run_inventory_phase(
            execute_token=authorizer.INVENTORY_EXECUTE_TOKEN,
            manifest=(tmp_path / "missing-manifest").resolve(),
            inventory_path=(tmp_path / "inventory.json").resolve(),
            authorization_protocol=(tmp_path / "missing-protocol").resolve(),
            stage4_complete=(tmp_path / "missing-complete.json").resolve(),
            checkpoint=(tmp_path / "missing-best.pth").resolve(),
            diagnostics=(tmp_path / "missing-diagnostics.json").resolve(),
            approval_path=(tmp_path / "approval.json").resolve(),
            output_root=(tmp_path / "output").resolve(),
            gpu_runner=empty_gpu,
        )

    def active_gpu(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="99123\n")

    with pytest.raises(FormalInventoryError, match="GPU is not released"):
        authorizer.run_inventory_phase(
            execute_token=authorizer.INVENTORY_EXECUTE_TOKEN,
            manifest=(tmp_path / "missing-manifest").resolve(),
            inventory_path=(tmp_path / "inventory.json").resolve(),
            authorization_protocol=(tmp_path / "missing-protocol").resolve(),
            stage4_complete=(tmp_path / "missing-complete.json").resolve(),
            checkpoint=(tmp_path / "missing-best.pth").resolve(),
            diagnostics=(tmp_path / "missing-diagnostics.json").resolve(),
            approval_path=(tmp_path / "approval.json").resolve(),
            output_root=(tmp_path / "output").resolve(),
            gpu_runner=active_gpu,
        )


def test_authorizer_inventory_phase_publishes_only_hash_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _small_inventory_fixture(tmp_path)
    monkeypatch.setattr(authorizer, "assert_standard_library_only", lambda: None)
    monkeypatch.setattr(
        authorizer, "validate_stage4_ready_without_torch", lambda *args, **kwargs: {}
    )

    def empty_gpu(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="")

    inventory = (tmp_path / "published_inventory.json").resolve()
    receipt = authorizer.run_inventory_phase(
        execute_token=authorizer.INVENTORY_EXECUTE_TOKEN,
        manifest=fixture["manifest"],
        inventory_path=inventory,
        authorization_protocol=fixture["protocol"],
        stage4_complete=(tmp_path / "synthetic-complete.json").resolve(),
        checkpoint=(tmp_path / "synthetic-best.pth").resolve(),
        diagnostics=(tmp_path / "synthetic-diagnostics.json").resolve(),
        approval_path=(tmp_path / "approval.json").resolve(),
        output_root=(tmp_path / "formal-output").resolve(),
        gpu_runner=empty_gpu,
        inventory_builder_kwargs=fixture["kwargs"],
    )
    assert receipt["status"] == "FORMAL_DATA_INVENTORY_COMPLETE"
    assert receipt["row_count"] == 2
    assert receipt["unique_file_count"] == 3
    assert inventory.stat().st_mode & 0o777 == 0o444
