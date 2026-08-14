from __future__ import annotations

from types import SimpleNamespace

from scripts.probe_validation_vram import (
    ACTIVE_SKILLS,
    CODE_PATHS,
    evaluate_gate,
    select_largest_clean_record,
)


def test_largest_clean_record_uses_maximum_area_and_manifest_tie_order() -> None:
    records = (
        SimpleNamespace(clean_id="small", width=2040, height=1356),
        SimpleNamespace(clean_id="first_max", width=2040, height=2040),
        SimpleNamespace(clean_id="second_max", width=2040, height=2040),
    )
    selected, ties = select_largest_clean_record(records)  # type: ignore[arg-type]
    assert selected.clean_id == "first_max"
    assert selected.width * selected.height == 2040 * 2040
    assert ties == 2


def _passing_probes() -> list[dict[str, object]]:
    common = {
        "passed": True,
        "finite": True,
        "shape_matches_input": True,
        "peak_reserved_fraction": 0.50,
    }
    return [
        {"model": "stage0_mio_stagea", **common},
        {
            "model": "expanded_guarded_skill_restormer",
            **common,
            "active_skill_count": 2,
            "active_skills": list(ACTIVE_SKILLS),
            "guard_shape": [1, 8, 2040, 2040],
        },
    ]


def test_gate_requires_both_exact_models_finite_shapes_two_guards_and_ceiling() -> None:
    probes = _passing_probes()
    assert evaluate_gate(probes, maximum_peak_fraction=0.90)

    probes[1]["peak_reserved_fraction"] = 0.900001
    assert not evaluate_gate(probes, maximum_peak_fraction=0.90)
    probes = _passing_probes()
    probes[1]["active_skill_count"] = 1
    assert not evaluate_gate(probes, maximum_peak_fraction=0.90)
    probes = _passing_probes()
    probes[0]["shape_matches_input"] = False
    assert not evaluate_gate(probes, maximum_peak_fraction=0.90)


def test_probe_sha_bindings_cover_pure_and_expanded_execution_code() -> None:
    required = {
        "scripts/probe_validation_vram.py",
        "src/net/mio_stagea.py",
        "src/net/graphrestore.py",
        "src/net/latent_skill_bank.py",
        "src/net/cooperative_executor.py",
        "src/training/stage0_engine.py",
    }
    assert required.issubset(CODE_PATHS)
