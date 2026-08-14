from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _load(relative: str) -> dict[str, Any]:
    value = load_yaml(CONFIGS / relative)
    assert isinstance(value, dict)
    return value


def _all_scalar_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            result.append(str(key))
            result.extend(_all_scalar_strings(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_all_scalar_strings(child))
        return result
    return [value] if isinstance(value, str) else []


def test_all_yaml_files_parse_with_unique_keys() -> None:
    paths = sorted(CONFIGS.rglob("*.yaml"))
    required = {
        "resolved_paths.yaml",
        "stage0_mio_stagea.yaml",
        "stage1_skill_bank.yaml",
        "stage2_interaction_distill.yaml",
        "stage3_planner.yaml",
        "stage4_graphrestore_e2e.yaml",
        "baselines/total_order.yaml",
        "baselines/parallel_only.yaml",
        "baselines/global_guard.yaml",
        "baselines/compute_matched_one_shot.yaml",
    }
    assert required.issubset({str(path.relative_to(CONFIGS)) for path in paths})
    for path in paths:
        assert isinstance(load_yaml(path), dict), path


def test_resolved_identities_are_frozen() -> None:
    config = _load("resolved_paths.yaml")
    expected = config["expected_identity"]
    assert expected["agenticir_commit"] == "9640a291480dee3ba8f2974125d4ee9e3440f3d6"
    assert expected["mioir_commit"] == "4d5f6ca0235cf2c307319673242d5722ee35d73f"
    assert expected["stage_a_parent_sha256"] == "66e056ff3537ea99416aeb119173e90fbcafc9e9f809db169ef7381cc93f77b8"
    assert len(expected["agenticir_files"]) == 6
    for digest in expected["agenticir_files"].values():
        assert len(digest) == 64
    for name in (
        "clean_train",
        "clean_val",
        "primary_train",
        "primary_val",
        "primary_all",
        "mio100_test_1440",
    ):
        assert len(expected["manifests"][name]) == 64
    expected_core_manifests = {
        "clean_train": "00247444a3b7304fe83a4783cae694181e6796253c6915d2491009def03df257",
        "clean_val": "88276445c7cc1166ace77904276dbeb61f3a049572e3b23fd1aad2b5f831947d",
        "primary_train": "83da30d0b8445d5bb427c336b125214ee62f2a0ec3a5bab61ca7119703044071",
        "primary_val": "af89bb22896a3744eab5e4b6414f5ee1b19770ce11e372e27b798afd9583a21b",
        "primary_all": "f4080efc2572ce2377646a8acabcbebe092e4a3feeabafc4200984b716c8e8eb",
        "mio100_test_1440": "5a53c28ad93d49a70d3632bfbff008a78309543bb6710921ab2a01b9bdb10950",
    }
    assert expected_core_manifests.items() <= expected["manifests"].items()


def test_stage0_exact_budget_curriculum_and_metric_order() -> None:
    config = _load("stage0_mio_stagea.yaml")
    assert config["training"]["max_steps"] == 60000
    assert config["data"]["crop_size"] == 192
    assert config["training"]["effective_batch_size"] == 8
    assert config["data"]["curriculum"] == [
        {
            "start_step": 0,
            "end_step_exclusive": 10000,
            "single_probability": 0.60,
            "group_a_probability": 0.40,
        },
        {
            "start_step": 10000,
            "end_step_exclusive": 60000,
            "single_probability": 0.30,
            "group_a_probability": 0.70,
        },
    ]
    assert config["loss"]["ssim"]["start_step"] == 12000
    assert config["validation"]["protocol"] == "agenticir_official_parity"
    assert config["hard_guards"]["allow_mio100_formal"] is False


def test_stage1_exact_episodes_adapter_layout_and_identity_gate() -> None:
    config = _load("stage1_skill_bank.yaml")
    assert sum(config["data"]["episode_sampling"].values()) == pytest.approx(1.0)
    assert config["training"]["max_steps"] == 30000
    assert config["model"]["adapters"]["levels"]["decoder_level3"] == {
        "blocks": 6,
        "channels": 192,
        "bottleneck": 24,
    }
    assert config["model"]["adapters"]["up_projection_zero_init"] is True
    assert config["hard_guards"]["require_zero_guard_identity_fp32_max_abs_lt"] == pytest.approx(1e-7)


def test_stage2_is_group_a_only_and_mandatorily_pauses() -> None:
    config = _load("stage2_interaction_distill.yaml")
    assert config["data"]["allowed_groups"] == ["single", "A"]
    assert config["data"]["allow_mio100_exploration"] is False
    assert config["data"]["allow_mio100_formal"] is False
    assert config["data"]["sampling"]["seed"] == 2027
    assert config["data"]["sampling"]["interaction_train_per_group_a_pair_max"] == 512
    assert config["data"]["sampling"]["interaction_val_per_group_a_pair_max"] == 128
    ambiguous = config["labeling"]["ambiguous"]
    assert ambiguous["head_class_count"] == 3
    assert ambiguous["supervision"] == "serial_mass_partial_label"
    assert ambiguous["serial_mass_weight"] == 0.25
    assert ambiguous["prohibit_double_weighting"] is True
    assert config["pause"]["mandatory_after_stage2"] is True
    assert config["pause"]["release_gpu"] is True
    assert config["pause"]["process_exit_code"] == 0


def test_stage3_uses_fixed_selection_threshold_then_one_time_calibration() -> None:
    config = _load("stage3_planner.yaml")
    assert config["training"]["max_steps"] == 12000
    assert config["model"]["executor"] == "frozen_stage1_ema"
    assert config["validation"]["checkpoint_presence_threshold"] == 0.50
    calibration = config["threshold_calibration"]
    assert (calibration["minimum"], calibration["maximum"], calibration["step"]) == (
        0.20,
        0.80,
        0.02,
    )
    assert calibration["mio100_forbidden"] is True
    assert config["hard_guards"]["require_explicit_stage3_approval"] is True


def test_stage4_exact_sampling_program_and_optimization() -> None:
    config = _load("stage4_graphrestore_e2e.yaml")
    assert config["training"]["max_steps"] == 40000
    assert sum(config["data"]["sampling"].values()) == pytest.approx(1.0)
    assert config["data"]["crop_size"] == 160
    assert config["data"]["minimum_crop_after_oom"] == 128
    assert config["data"]["effective_batch_size"] == 4
    assert config["program"]["compile_relations_once_at_t0"] is True
    assert config["skills"]["allow_skill_reentry"] is False
    assert config["data"]["counterfactual"]["wrong_skill_misuse"]["forced_execution_presence_override"] == 1.0
    assert config["model"]["decoder_output_contract"] == "delta_only"
    schedule = config["teacher_forcing"]["schedule"]
    assert schedule[1]["probability_end"] == 0.5
    assert schedule[2]["probability_start"] == 0.25
    assert config["optimization"]["gradient_clip_norm"] == 0.5
    assert config["runtime"]["torch_compile"] is False


def test_baseline_overlays_cover_all_required_modes_and_fair_retraining() -> None:
    expected = {
        "total_order.yaml": "forced_total_order",
        "parallel_only.yaml": "parallel_only",
        "global_guard.yaml": "full_partial_order",
        "compute_matched_one_shot.yaml": "one_shot",
    }
    orders = []
    for filename, compiler_mode in expected.items():
        config = _load(f"baselines/{filename}")
        assert config["requires_new_training"] is True
        assert config["enabled_only_after_full_effective_on_primary_val"] is True
        assert config["override"]["program"]["compiler_mode"] == compiler_mode
        assert config["hard_guards"]["allow_mio100_during_training"] is False
        orders.append(config["fair_retrain_order"])
    assert orders == [1, 2, 3, 4]
    one_shot = _load("baselines/compute_matched_one_shot.yaml")
    assert one_shot["required_before_submission"] is True
    assert one_shot["hard_guards"]["require_frozen_full_compute_profile"] is True


def test_training_configs_do_not_reference_forbidden_external_data_paths() -> None:
    forbidden = ("/rar/", "pir_tar", "sidd", "gopro", "reside", "rain200l", "div2k", "flickr2k")
    for path in sorted(CONFIGS.glob("stage*.yaml")):
        strings = [value.lower().replace("\\", "/") for value in _all_scalar_strings(load_yaml(path))]
        for value in strings:
            assert not any(marker in value for marker in forbidden), (path, value)
