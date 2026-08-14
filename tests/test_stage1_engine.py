from __future__ import annotations

import csv
from dataclasses import dataclass

import pytest
import torch

from src.data.samplers import StatefulEpisodeSampler
from src.net import GuardedSkillRestormer, MiOStageA
from src.net.skill_adapter import SKILLS
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import WarmupCosineScheduler
from src.training.stage1_engine import (
    append_stage1_calibration_history,
    build_stage1_optimizer,
    load_stage0_best_ema_backbone,
    resume_stage1_checkpoint,
    save_stage1_checkpoint,
    set_stage1_trainability,
    stage1_parameter_role,
    train_stage1_optimizer_step,
)
from src.training.stage0_engine import CALIBRATION_COLUMNS


def _tiny_guarded() -> GuardedSkillRestormer:
    return GuardedSkillRestormer(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        refinement=1,
        heads=(1, 1, 1, 1),
        expansion=2.0,
        skill_bottlenecks={
            "level3": 2,
            "level2": 2,
            "level1": 2,
            "refinement": 2,
        },
    )


def _tiny_parent() -> MiOStageA:
    return MiOStageA(
        dim=8,
        encoder_blocks=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        refinement=1,
        heads=(1, 1, 1, 1),
        expansion=2.0,
    )


def _optimizer(model: GuardedSkillRestormer) -> torch.optim.AdamW:
    return build_stage1_optimizer(
        model,
        skill_lr=1.0e-4,
        decoder_lr=1.0e-5,
        encoder34_lr=2.0e-6,
        weight_decay=1.0e-4,
        fused_if_supported=False,
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    image = torch.rand(1, 3, 16, 16, generator=generator)
    target = torch.rand(1, 3, 16, 16, generator=generator)
    guards = torch.zeros(1, len(SKILLS), 4, 4)
    guards[:, 0] = 0.75
    active = torch.zeros(1, len(SKILLS), dtype=torch.bool)
    active[:, 0] = True
    return {
        "input": image,
        "target": target,
        "guard_targets": guards,
        "active_mask": active,
    }


def test_stage1_freeze_boundary_and_optimizer_roles() -> None:
    model = _tiny_guarded()
    optimizer = _optimizer(model)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    set_stage1_trainability(model, 4999)
    for name, parameter in model.named_parameters():
        role = stage1_parameter_role(name)
        assert (id(parameter) in optimizer_ids) is (role is not None)
        assert parameter.requires_grad is (role == "skills_mixers")

    set_stage1_trainability(model, 5000)
    for name, parameter in model.named_parameters():
        role = stage1_parameter_role(name)
        assert parameter.requires_grad is (role is not None)
        if name.startswith(("encoder.level1.", "encoder.level2.")):
            assert not parameter.requires_grad
        if name.startswith(("encoder.patch.", "encoder.down12.")):
            assert not parameter.requires_grad
        if name.startswith(("encoder.down23.", "encoder.down34.")):
            assert parameter.requires_grad

    role_to_lr = {str(group["role"]): float(group["initial_lr"]) for group in optimizer.param_groups}
    assert role_to_lr == {
        "skills_mixers": pytest.approx(1.0e-4),
        "decoder_refine_head": pytest.approx(1.0e-5),
        "encoder34": pytest.approx(2.0e-6),
    }


def test_stage1_forward_backward_and_active_skill_gradient() -> None:
    torch.manual_seed(23)
    model = _tiny_guarded()
    optimizer = _optimizer(model)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=1,
        max_steps=4,
        min_lr=1.0e-6,
    )
    ema = ExponentialMovingAverage(model, decay=0.9)
    result = train_stage1_optimizer_step(
        model,
        [_batch()],
        optimizer,
        scheduler,
        ema,
        step=0,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        use_bf16=False,
        audit_first_backward=True,
    )
    assert result.samples == 1
    assert result.loss > 0.0
    assert result.lambda_ssim == 0.0
    assert result.grad_norm > 0.0
    for level in ("level3", "level2", "level1", "refinement"):
        block = model.decoder.skill_bank.adapters[level][0]
        assert torch.count_nonzero(block["noise"].up.weight).item() > 0
        for inactive in SKILLS[1:]:
            assert torch.count_nonzero(block[inactive].up.weight).item() == 0


@dataclass(frozen=True)
class _Record:
    operator_order: tuple[str, ...]


class _SamplerDataset:
    def __init__(self) -> None:
        singles = [
            ("noise",),
            ("motion blur",),
            ("defocus blur",),
            ("jpeg compression artifact",),
            ("rain",),
            ("haze",),
            ("dark",),
            ("low resolution",),
        ]
        pairs = [
            ("rain", "haze"),
            ("motion blur", "low resolution"),
            ("dark", "noise"),
            ("defocus blur", "jpeg compression artifact"),
            ("noise", "jpeg compression artifact"),
            ("rain", "low resolution"),
            ("motion blur", "dark"),
            ("defocus blur", "haze"),
        ]
        self.records = tuple(_Record(item) for item in (*singles, *pairs))

    def __len__(self) -> int:
        return len(self.records)

    def set_worker_seed(self, seed: int) -> None:
        del seed


def _sampler() -> StatefulEpisodeSampler:
    return StatefulEpisodeSampler(
        _SamplerDataset(),
        num_samples=80,
        stage="stage1",
        effective_batch_size=8,
        base_seed=2027,
        start_step=0,
    )


def test_stage1_strict_parent_ema_load_and_exact_resume(tmp_path) -> None:
    torch.manual_seed(29)
    parent = _tiny_parent()
    parent_state = {name: value.detach().clone() for name, value in parent.state_dict().items()}
    parent_payload = {
        "schema_version": "graphrestore-checkpoint-v1",
        "stage": "stage0",
        "model_role": "ema_selection",
        "resumable": False,
        "model": parent_state,
        "ema": {
            "decay": 0.9999,
            "num_updates": 3,
            "shadow": {name: value.detach().clone() for name, value in parent_state.items()},
        },
    }
    model = _tiny_guarded()
    report = load_stage0_best_ema_backbone(
        model,
        parent_payload,
        reference_model=_tiny_parent(),
    )
    assert report.missing_keys
    assert all(name.startswith("decoder.skill_bank.") for name in report.missing_keys)
    for name, expected in parent_state.items():
        torch.testing.assert_close(model.state_dict()[name], expected)

    optimizer = _optimizer(model)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=1,
        max_steps=4,
        min_lr=1.0e-6,
    )
    ema = ExponentialMovingAverage(model, decay=0.9)
    train_stage1_optimizer_step(
        model,
        [_batch()],
        optimizer,
        scheduler,
        ema,
        step=0,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    sampler = _sampler()
    provenance = {"config_sha256": "abc", "parent": {"sha256": "def"}}
    checkpoint = tmp_path / "last.pth"
    save_stage1_checkpoint(
        checkpoint,
        step=1,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        metrics={"group_a_psnr": 20.0},
    )
    best_checkpoint = tmp_path / "best_ema.pth"
    save_stage1_checkpoint(
        best_checkpoint,
        step=1,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        provenance=provenance,
        metrics={"group_a_psnr": 20.0},
        model_as_ema=True,
    )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    assert best_payload["stage"] == "stage1"
    assert best_payload["model_role"] == "ema_selection"
    assert best_payload["resumable"] is False
    assert best_payload["model"].keys() == best_payload["ema"]["shadow"].keys()
    for name, expected in best_payload["ema"]["shadow"].items():
        torch.testing.assert_close(best_payload["model"][name], expected, rtol=0, atol=0)
    expected_random = torch.rand(4)

    restored = _tiny_guarded()
    restored_optimizer = _optimizer(restored)
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer,
        warmup_steps=1,
        max_steps=4,
        min_lr=1.0e-6,
    )
    restored_ema = ExponentialMovingAverage(restored, decay=0.9)
    restored_sampler = _sampler()
    payload = resume_stage1_checkpoint(
        checkpoint,
        model=restored,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=restored_sampler,
        expected_provenance=provenance,
    )
    assert payload["step"] == 1
    assert payload["model_role"] == "raw_training_state"
    assert payload["resumable"] is True
    assert restored_sampler.state_dict()["sample_cursor"] == 8
    torch.testing.assert_close(torch.rand(4), expected_random)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    for name, expected in ema.shadow.items():
        torch.testing.assert_close(restored_ema.shadow[name], expected)
        assert restored_ema.shadow[name].device == expected.device
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_stage1_appends_the_shared_full_calibration_schema(tmp_path) -> None:
    path = tmp_path / "calibration_history.csv"
    summary = {
        "episodes": {"single_skill": {"psnr": 31.25, "ssim": 0.9125}},
        "group_a_equal_combination_mean": {"psnr": 28.75, "ssim": 0.8875},
    }
    append_stage1_calibration_history(path, step=3000, summary=summary)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CALIBRATION_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["step"] == "3000"
    assert float(rows[0]["single_psnr"]) == pytest.approx(31.25)
    assert float(rows[0]["group_a_ssim"]) == pytest.approx(0.8875)
    assert rows[0]["planner_macro_f1"] == ""
    assert rows[0]["mean_program_levels"] == ""
