from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from src.data.samplers import StatefulEpisodeSampler
from src.net import GuardedSkillRestormer, MiOStageA
from src.net.skill_adapter import SKILLS
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import WarmupCosineScheduler
from src.training.stage1_engine import (
    STAGE1_EMA_SCOPE,
    Stage1ContractError,
    Stage1PhaseAwareEMA,
    append_stage1_calibration_history,
    build_stage1_ema,
    build_stage1_optimizer,
    choose_micro_batch,
    load_stage0_best_ema_backbone,
    resume_stage1_checkpoint,
    save_stage1_checkpoint,
    set_stage1_trainability,
    stage1_fidelity_loss,
    stage1_ema_policy_metadata,
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


class _PhaseAwareEMAProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.trainable_alias = self.trainable
        self.frozen = torch.nn.Parameter(
            torch.tensor([0.375, -0.625]), requires_grad=False
        )
        self.register_buffer("float_buffer", torch.tensor([0.125, 0.875]))
        self.register_buffer("integer_buffer", torch.tensor([3, 7], dtype=torch.int64))


class _BoundaryStage1Model(torch.nn.Module):
    """Minimal Stage1-shaped module for an exact step-5000 resume test."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.skill_bank = torch.nn.Module()
        self.decoder.skill_bank.adapter = torch.nn.Parameter(torch.tensor(0.01))
        self.decoder.refinement = torch.nn.Parameter(torch.tensor(0.02))
        self.encoder = torch.nn.Module()
        self.encoder.level3 = torch.nn.Module()
        self.encoder.level3.weight = torch.nn.Parameter(torch.tensor(0.03))
        self.encoder.level1 = torch.nn.Module()
        self.encoder.level1.weight = torch.nn.Parameter(torch.tensor(0.04))

    def forward(
        self,
        image: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        guards: torch.Tensor,
    ) -> torch.Tensor:
        del active_mask, guards
        residual = (
            self.decoder.skill_bank.adapter
            + self.decoder.refinement
            + self.encoder.level3.weight
            + self.encoder.level1.weight
        )
        return image + residual


def _boundary_optimizer(model: _BoundaryStage1Model) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=0.0)


def test_stage1_checkpoint_keeps_empty_sparse_optimizer_ledger(
    tmp_path: Path,
) -> None:
    model = _BoundaryStage1Model()
    optimizer = _boundary_optimizer(model)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=1,
        max_steps=4,
        min_lr=1.0e-6,
    )
    ema = build_stage1_ema(model, decay=0.9)
    checkpoint = tmp_path / "empty_state.pth"
    save_stage1_checkpoint(
        checkpoint,
        step=0,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=_sampler(),
        provenance={"ema_policy": stage1_ema_policy_metadata(0.9)},
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["optimizer"]["state"] == {}
    assert "optimizer_state_name_ledger" in payload
    assert payload["optimizer_state_name_ledger"] == {}


def test_stage1_phase_aware_ema_update_policy_and_dynamic_unfreeze() -> None:
    model = _PhaseAwareEMAProbe()
    ema = build_stage1_ema(model, decay=0.75)
    initial_trainable = ema.shadow["trainable"].clone()

    with torch.no_grad():
        model.trainable.copy_(torch.tensor([2.0, 4.0]))
        model.frozen.copy_(torch.tensor([0.1, -0.2]))
        model.float_buffer.copy_(torch.tensor([0.3, 0.7]))
        model.integer_buffer.copy_(torch.tensor([11, 13]))
    ema.update(model)

    expected_trainable = initial_trainable.mul(0.75).add(model.trainable, alpha=0.25)
    torch.testing.assert_close(
        ema.shadow["trainable"], expected_trainable, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        ema.shadow["trainable_alias"], expected_trainable, rtol=0.0, atol=0.0
    )
    assert torch.equal(ema.shadow["frozen"], model.frozen)
    assert torch.equal(ema.shadow["float_buffer"], model.float_buffer)
    assert torch.equal(ema.shadow["integer_buffer"], model.integer_buffer)

    # Simulate the internal step=5000 boundary.  The pre-boundary copied
    # shadow is retained and receives its first ordinary EMA update.
    previous_frozen_shadow = ema.shadow["frozen"].clone()
    model.frozen.requires_grad_(True)
    with torch.no_grad():
        model.frozen.copy_(torch.tensor([0.9, -0.8]))
    expected_first_unfrozen = previous_frozen_shadow.mul(0.75).add(
        model.frozen, alpha=0.25
    )
    ema.update(model)
    torch.testing.assert_close(
        ema.shadow["frozen"], expected_first_unfrozen, rtol=0.0, atol=0.0
    )
    assert not torch.equal(ema.shadow["frozen"], model.frozen)
    assert ema.num_updates == 2
    state = ema.state_dict()
    assert state["scope"] == STAGE1_EMA_SCOPE
    assert state["policy"] == stage1_ema_policy_metadata(0.75)
    assert state["policy"]["optimizer_step_indexing"] == "zero_based_internal_step"
    assert state["policy"]["phase0_end_step_exclusive"] == 5000


def test_stage1_phase0_frozen_backbone_shadow_remains_bit_exact() -> None:
    torch.manual_seed(21)
    model = _tiny_guarded()
    _optimizer(model)
    set_stage1_trainability(model, 4999)
    ema = build_stage1_ema(model, decay=0.9)
    frozen_names = {
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    assert frozen_names

    for _ in range(3000):
        ema.update(model)
    model_state = model.state_dict()
    for name in frozen_names:
        assert torch.equal(ema.shadow[name], model_state[name]), name

    newly_unfrozen_name = next(
        name
        for name, parameter in model.named_parameters()
        if stage1_parameter_role(name) == "decoder_refine_head"
        and not parameter.requires_grad
    )
    previous_shadow = ema.shadow[newly_unfrozen_name].clone()
    set_stage1_trainability(model, 5000)
    newly_unfrozen = dict(model.named_parameters())[newly_unfrozen_name]
    with torch.no_grad():
        newly_unfrozen.add_(0.25)
    expected = previous_shadow.mul(0.9).add(newly_unfrozen, alpha=0.1)
    ema.update(model)
    torch.testing.assert_close(
        ema.shadow[newly_unfrozen_name], expected, rtol=0.0, atol=0.0
    )


def test_stage1_probe_and_formal_runner_share_phase_aware_ema_factory() -> None:
    from scripts.train_stage1_skills import run

    assert "build_stage1_ema" in choose_micro_batch.__code__.co_names
    assert "build_stage1_ema" in run.__code__.co_names
    assert isinstance(build_stage1_ema(_PhaseAwareEMAProbe()), Stage1PhaseAwareEMA)


def test_stage1_optimizer_step_rejects_generic_ema() -> None:
    model = _tiny_guarded()
    optimizer = _optimizer(model)
    generic_ema = ExponentialMovingAverage(model, decay=0.9)
    with pytest.raises(Stage1ContractError, match="phase-aware EMA"):
        train_stage1_optimizer_step(
            model,
            [_batch()],
            optimizer,
            scheduler=None,
            ema=generic_ema,  # type: ignore[arg-type]
            step=0,
            device=torch.device("cpu"),
            use_bf16=False,
        )


def test_stage1_step5000_resume_performs_first_unfrozen_ema_without_reset(
    tmp_path: Path,
) -> None:
    source = _BoundaryStage1Model()
    set_stage1_trainability(source, 4999)
    source_optimizer = _boundary_optimizer(source)
    source_scheduler = WarmupCosineScheduler(
        source_optimizer,
        warmup_steps=1,
        max_steps=6000,
        min_lr=1.0e-6,
    )
    source_ema = build_stage1_ema(source, decay=0.9)
    for _ in range(5000):
        source.decoder.skill_bank.adapter.grad = torch.ones_like(
            source.decoder.skill_bank.adapter
        )
        source_optimizer.step()
        source_optimizer.zero_grad(set_to_none=True)
        source_scheduler.step()
        source_ema.update(source)
    assert source_scheduler.last_epoch == 5000
    assert source_scheduler._step_count == 5001
    assert set(source_optimizer.state) == {source.decoder.skill_bank.adapter}
    assert (
        int(source_optimizer.state[source.decoder.skill_bank.adapter]["step"].item())
        == 5000
    )
    provenance = {"ema_policy": stage1_ema_policy_metadata(0.9)}

    with pytest.raises(Stage1ContractError, match="step/EMA update count mismatch"):
        save_stage1_checkpoint(
            tmp_path / "mismatched.pth",
            step=4999,
            model=source,
            ema=source_ema,
            optimizer=source_optimizer,
            scheduler=source_scheduler,
            sampler=_sampler(),
            provenance=provenance,
        )

    inconsistent_optimizer = _boundary_optimizer(source)
    inconsistent_scheduler = WarmupCosineScheduler(
        inconsistent_optimizer,
        warmup_steps=1,
        max_steps=6000,
        min_lr=1.0e-6,
    )
    with pytest.raises(Stage1ContractError, match="scheduler.last_epoch"):
        save_stage1_checkpoint(
            tmp_path / "bad_scheduler.pth",
            step=5000,
            model=source,
            ema=source_ema,
            optimizer=inconsistent_optimizer,
            scheduler=inconsistent_scheduler,
            sampler=_sampler(),
            provenance=provenance,
        )
    inconsistent_scheduler.last_epoch = 5000
    inconsistent_scheduler._step_count = 5000
    with pytest.raises(Stage1ContractError, match="scheduler._step_count"):
        save_stage1_checkpoint(
            tmp_path / "bad_scheduler_count.pth",
            step=5000,
            model=source,
            ema=source_ema,
            optimizer=inconsistent_optimizer,
            scheduler=inconsistent_scheduler,
            sampler=_sampler(),
            provenance=provenance,
        )

    checkpoint = tmp_path / "last.pth"
    save_stage1_checkpoint(
        checkpoint,
        step=5000,
        model=source,
        ema=source_ema,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        sampler=_sampler(),
        provenance=provenance,
    )

    restored = _BoundaryStage1Model()
    restored_optimizer = _boundary_optimizer(restored)
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer,
        warmup_steps=1,
        max_steps=6000,
        min_lr=1.0e-6,
    )
    restored_ema = build_stage1_ema(restored, decay=0.9)
    payload = resume_stage1_checkpoint(
        checkpoint,
        model=restored,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=_sampler(),
        expected_provenance=provenance,
        expected_validation_every=3000,
        expected_max_steps=6000,
    )
    assert payload["step"] == restored_ema.num_updates == 5000
    assert restored_scheduler.last_epoch == 5000
    assert restored_scheduler._step_count == 5001
    assert set(restored_optimizer.state) == {restored.decoder.skill_bank.adapter}
    assert (
        int(
            restored_optimizer.state[restored.decoder.skill_bank.adapter]["step"].item()
        )
        == 5000
    )
    newly_unfrozen = dict(restored.named_parameters())["decoder.refinement"]
    assert newly_unfrozen.requires_grad
    previous_shadow = restored_ema.shadow["decoder.refinement"].clone()

    train_stage1_optimizer_step(
        restored,
        [_batch()],
        restored_optimizer,
        restored_scheduler,
        restored_ema,
        step=5000,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        use_bf16=False,
    )
    expected = previous_shadow.mul(0.9).add(newly_unfrozen, alpha=0.1)
    torch.testing.assert_close(
        restored_ema.shadow["decoder.refinement"], expected, rtol=0.0, atol=0.0
    )
    assert not torch.equal(restored_ema.shadow["decoder.refinement"], newly_unfrozen)
    assert restored_ema.num_updates == 5001
    assert (
        int(
            restored_optimizer.state[restored.decoder.skill_bank.adapter]["step"].item()
        )
        == 5001
    )
    assert (
        int(restored_optimizer.state[restored.decoder.refinement]["step"].item()) == 1
    )
    assert (
        int(restored_optimizer.state[restored.encoder.level3.weight]["step"].item())
        == 1
    )
    assert restored.encoder.level1.weight not in restored_optimizer.state
    assert torch.equal(
        restored_ema.shadow["encoder.level1.weight"],
        restored.encoder.level1.weight,
    )

    phase_checkpoint = tmp_path / "phase_local_state.pth"
    save_stage1_checkpoint(
        phase_checkpoint,
        step=5001,
        model=restored,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=_sampler(),
        provenance=provenance,
    )
    phase_payload = torch.load(
        phase_checkpoint, map_location="cpu", weights_only=False
    )
    decoder_state_id = next(
        serialized_id
        for serialized_id, name in phase_payload[
            "optimizer_state_name_ledger"
        ].items()
        if name == "decoder.refinement"
    )
    phase_payload["optimizer"]["state"][decoder_state_id]["step"] = torch.tensor(
        2.0
    )
    invalid_phase_checkpoint = tmp_path / "invalid_phase_local_state.pth"
    torch.save(phase_payload, invalid_phase_checkpoint)
    phase_victim = _BoundaryStage1Model()
    phase_victim_optimizer = _boundary_optimizer(phase_victim)
    phase_victim_scheduler = WarmupCosineScheduler(
        phase_victim_optimizer,
        warmup_steps=1,
        max_steps=6000,
        min_lr=1.0e-6,
    )
    phase_victim_ema = build_stage1_ema(phase_victim, decay=0.9)
    before_phase_victim = {
        name: value.detach().clone()
        for name, value in phase_victim.state_dict().items()
    }
    with pytest.raises(Stage1ContractError, match="phase-local maximum"):
        resume_stage1_checkpoint(
            invalid_phase_checkpoint,
            model=phase_victim,
            ema=phase_victim_ema,
            optimizer=phase_victim_optimizer,
            scheduler=phase_victim_scheduler,
            sampler=_sampler(),
            expected_provenance=provenance,
            expected_validation_every=3000,
            expected_max_steps=6000,
        )
    for name, expected in before_phase_victim.items():
        assert torch.equal(phase_victim.state_dict()[name], expected)
    assert phase_victim_optimizer.state_dict()["state"] == {}
    assert phase_victim_ema.num_updates == 0


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
    ema = build_stage1_ema(model, decay=0.9)
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


def test_stage1_ssim_loss_stays_fp32_under_bf16_autocast() -> None:
    prediction = torch.full(
        (1, 3, 16, 16), 0.5, dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.full(prediction.shape, 0.5, dtype=torch.float32)
    target[..., ::2, ::2] += 0.01
    reference = stage1_fidelity_loss(prediction.detach().float(), target, step=6000)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = stage1_fidelity_loss(prediction, target, step=6000)

    torch.testing.assert_close(result.ssim, reference.ssim, rtol=0.0, atol=0.0)
    assert result.ssim.dtype == torch.float32
    assert result.total.dtype == torch.float32
    assert bool(torch.isfinite(result.total))
    assert float(result.ssim) >= 0.0
    result.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


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


def test_stage1_strict_parent_ema_load_and_exact_resume(tmp_path: Path) -> None:
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
    ema = build_stage1_ema(model, decay=0.9)
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
    provenance = {
        "config_sha256": "abc",
        "parent": {"sha256": "def"},
        "ema_policy": stage1_ema_policy_metadata(0.9),
    }
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
        metrics={},
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
        metrics={},
        model_as_ema=True,
    )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert best_payload["stage"] == "stage1"
    assert best_payload["model_role"] == "ema_selection"
    assert best_payload["resumable"] is False
    assert best_payload["ema"]["scope"] == STAGE1_EMA_SCOPE
    assert best_payload["ema"]["policy"] == stage1_ema_policy_metadata(0.9)
    assert best_payload["model"].keys() == best_payload["ema"]["shadow"].keys()
    for saved_payload in (raw_payload, best_payload):
        ledger = saved_payload["optimizer_state_name_ledger"]
        optimizer_state = saved_payload["optimizer"]
        assert ledger
        assert set(ledger) == set(optimizer_state["state"])
        assert all(stage1_parameter_role(name) == "skills_mixers" for name in ledger.values())
        optimizer_parameter_count = sum(
            len(group["params"]) for group in optimizer_state["param_groups"]
        )
        assert len(ledger) < optimizer_parameter_count
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
    restored_ema = build_stage1_ema(restored, decay=0.9)
    restored_sampler = _sampler()
    payload = resume_stage1_checkpoint(
        checkpoint,
        model=restored,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        sampler=restored_sampler,
        expected_provenance=provenance,
        expected_validation_every=3,
        expected_max_steps=4,
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
    assert restored_ema.state_dict()["scope"] == STAGE1_EMA_SCOPE
    assert restored_ema.state_dict()["policy"] == stage1_ema_policy_metadata(0.9)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_stage1_optimizer_state_ledger_is_fail_closed_before_mutation(
    tmp_path: Path,
) -> None:
    torch.manual_seed(41)
    source = _tiny_guarded()
    source_optimizer = _optimizer(source)
    source_scheduler = WarmupCosineScheduler(
        source_optimizer,
        warmup_steps=1,
        max_steps=4,
        min_lr=1.0e-6,
    )
    source_ema = build_stage1_ema(source, decay=0.9)
    train_stage1_optimizer_step(
        source,
        [_batch()],
        source_optimizer,
        source_scheduler,
        source_ema,
        step=0,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    provenance = {"ema_policy": stage1_ema_policy_metadata(0.9)}
    valid_path = tmp_path / "ledger_valid.pth"
    save_stage1_checkpoint(
        valid_path,
        step=1,
        model=source,
        ema=source_ema,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        sampler=_sampler(),
        provenance=provenance,
    )
    valid_payload = torch.load(valid_path, map_location="cpu", weights_only=False)
    valid_ledger = valid_payload["optimizer_state_name_ledger"]
    assert valid_ledger
    assert set(valid_ledger) == set(valid_payload["optimizer"]["state"])

    forbidden_parameter = next(
        parameter
        for name, parameter in source.named_parameters()
        if stage1_parameter_role(name) == "decoder_refine_head"
    )
    source_optimizer.state[forbidden_parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(forbidden_parameter),
        "exp_avg_sq": torch.zeros_like(forbidden_parameter),
    }
    forbidden_save_path = tmp_path / "forbidden_save.pth"
    with pytest.raises(Stage1ContractError, match="ledger role is illegal"):
        save_stage1_checkpoint(
            forbidden_save_path,
            step=1,
            model=source,
            ema=source_ema,
            optimizer=source_optimizer,
            scheduler=source_scheduler,
            sampler=_sampler(),
            provenance=provenance,
        )
    assert not forbidden_save_path.exists()
    del source_optimizer.state[forbidden_parameter]

    ledger_id = next(iter(valid_ledger))
    allowed_wrong_name = next(
        name
        for name, _parameter in source.named_parameters()
        if stage1_parameter_role(name) == "skills_mixers"
        and name != valid_ledger[ledger_id]
    )
    forbidden_name = next(
        name
        for name, _parameter in source.named_parameters()
        if stage1_parameter_role(name) == "decoder_refine_head"
    )
    maximum_optimizer_id = max(
        parameter_id
        for group in valid_payload["optimizer"]["param_groups"]
        for parameter_id in group["params"]
    )
    cases = (
        ("missing_ledger", "lacks optimizer state-name ledger"),
        ("deleted_state", "ledger keys differ"),
        ("wrong_name", "ledger name drifted"),
        ("wrong_id", "ledger keys differ"),
        ("forbidden_role", "ledger role is illegal"),
    )
    for label, error_pattern in cases:
        bad_payload = torch.load(valid_path, map_location="cpu", weights_only=False)
        ledger = bad_payload["optimizer_state_name_ledger"]
        if label == "missing_ledger":
            bad_payload.pop("optimizer_state_name_ledger")
        elif label == "deleted_state":
            bad_payload["optimizer"]["state"].pop(ledger_id)
        elif label == "wrong_name":
            ledger[ledger_id] = allowed_wrong_name
        elif label == "wrong_id":
            ledger[maximum_optimizer_id + 1] = ledger.pop(ledger_id)
        else:
            ledger[ledger_id] = forbidden_name
        bad_path = tmp_path / f"ledger_{label}.pth"
        torch.save(bad_payload, bad_path)

        victim = _tiny_guarded()
        victim_optimizer = _optimizer(victim)
        victim_scheduler = WarmupCosineScheduler(
            victim_optimizer,
            warmup_steps=1,
            max_steps=4,
            min_lr=1.0e-6,
        )
        victim_ema = build_stage1_ema(victim, decay=0.9)
        victim_sampler = _sampler()
        before_model = {
            name: value.detach().clone() for name, value in victim.state_dict().items()
        }
        before_ema = {
            name: value.detach().clone() for name, value in victim_ema.shadow.items()
        }
        before_optimizer = victim_optimizer.state_dict()
        before_scheduler = victim_scheduler.state_dict()
        before_sampler = victim_sampler.state_dict()
        before_torch_rng = torch.get_rng_state().clone()
        with pytest.raises(Stage1ContractError, match=error_pattern):
            resume_stage1_checkpoint(
                bad_path,
                model=victim,
                ema=victim_ema,
                optimizer=victim_optimizer,
                scheduler=victim_scheduler,
                sampler=victim_sampler,
                expected_provenance=provenance,
                expected_validation_every=3,
                expected_max_steps=4,
            )
        for name, expected in before_model.items():
            assert torch.equal(victim.state_dict()[name], expected)
        for name, expected in before_ema.items():
            assert torch.equal(victim_ema.shadow[name], expected)
        assert victim_ema.num_updates == 0
        assert victim_optimizer.state_dict() == before_optimizer
        assert victim_scheduler.state_dict() == before_scheduler
        assert victim_sampler.state_dict() == before_sampler
        assert torch.equal(torch.get_rng_state(), before_torch_rng)


@pytest.mark.parametrize(
    ("tampered_field", "error_pattern"),
    [
        ("scope", "scope"),
        ("policy", "policy"),
        ("num_updates", "update count"),
        ("dtype", "dtypes"),
        ("scheduler_last_epoch", "scheduler.last_epoch"),
        ("scheduler_step_count", "scheduler._step_count"),
        ("fixed_shadow", "fixed parameter/buffer"),
        ("sampler", "sampler effective_batch_size"),
        ("sampler_consumed_bool", "sampler consumed step"),
        ("sampler_cursor_bool", "sampler cursor"),
        ("optimizer", "optimizer parameter ID order"),
        ("rng", "RNG state"),
        ("pending_type", "pending_validation_step must be an integer or null"),
        ("pending_mismatch", "pending_validation_step differs"),
        ("pending_boundary", "not a validation boundary"),
        ("max_steps", "exceeds expected max_steps"),
        ("optimizer_initial_lr", "optimizer static field drifted: initial_lr"),
        ("optimizer_betas", "optimizer static field drifted: betas"),
        ("optimizer_lr", "optimizer/scheduler LR trajectory"),
        ("optimizer_lr_coordinated", "optimizer/scheduler LR trajectory"),
        ("optimizer_lr_nonfinite", "optimizer lr is non-finite"),
        ("scheduler_warmup", "scheduler warmup_steps drifted"),
        ("scheduler_max_steps", "scheduler max_steps drifted"),
        ("scheduler_min_lr", "scheduler min_lr drifted"),
        ("scheduler_base_lrs", "scheduler base_lrs drifted"),
        ("scheduler_last_lr", "optimizer/scheduler LR trajectory"),
        ("scheduler_last_lr_nonfinite", "scheduler LR state is non-finite"),
        ("metrics_type", "metrics must be a mapping"),
        ("metrics_partial_best", "partial best fields"),
        ("metrics_partial_current", "partial current fields"),
        ("metrics_nonfinite", "best_group_a_psnr is non-finite"),
        ("metrics_step_type", "best_step is invalid"),
        ("metrics_step_future", "best_step is invalid"),
        ("scaler", "resumable raw Stage1 training checkpoint"),
        ("cuda_rng_missing", "RNG state"),
        ("cuda_rng_shape", "RNG state"),
        ("cuda_rng_dtype", "RNG state"),
    ],
)
def test_stage1_resume_rejects_ema_contract_before_model_mutation(
    tmp_path: Path,
    tampered_field: str,
    error_pattern: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    source = _tiny_guarded()
    source_optimizer = _optimizer(source)
    source_scheduler = WarmupCosineScheduler(
        source_optimizer,
        warmup_steps=1,
        max_steps=4,
        min_lr=1.0e-6,
    )
    source_ema = build_stage1_ema(source, decay=0.9)
    provenance = {"ema_policy": stage1_ema_policy_metadata(0.9)}
    valid_path = tmp_path / "valid.pth"
    save_stage1_checkpoint(
        valid_path,
        step=0,
        model=source,
        ema=source_ema,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        sampler=_sampler(),
        provenance=provenance,
    )
    bad_payload = torch.load(valid_path, map_location="cpu", weights_only=False)
    if tampered_field == "scope":
        bad_payload["ema"]["scope"] = "generic_all_floating_state_ema"
    elif tampered_field == "policy":
        bad_payload["ema"]["policy"] = {
            **bad_payload["ema"]["policy"],
            "buffer_update": "standard_ema",
        }
    elif tampered_field == "num_updates":
        bad_payload["ema"]["num_updates"] = 1
    elif tampered_field == "dtype":
        shadow = bad_payload["ema"]["shadow"]
        name = next(key for key, value in shadow.items() if value.is_floating_point())
        shadow[name] = shadow[name].double()
    elif tampered_field == "scheduler_last_epoch":
        bad_payload["scheduler"]["last_epoch"] = 1
    elif tampered_field == "scheduler_step_count":
        bad_payload["scheduler"]["_step_count"] = 2
    elif tampered_field == "fixed_shadow":
        shadow = dict(bad_payload["ema"]["shadow"])
        name = next(key for key in shadow if key.startswith("encoder.level1."))
        shadow[name] = shadow[name] + 0.25
        bad_payload["ema"]["shadow"] = shadow
    elif tampered_field == "sampler":
        bad_payload["sampler_state"]["effective_batch_size"] = 4
    elif tampered_field == "sampler_consumed_bool":
        bad_payload["sampler_state"]["consumed_optimizer_step"] = False
    elif tampered_field == "sampler_cursor_bool":
        bad_payload["sampler_state"]["sample_cursor"] = False
    elif tampered_field == "optimizer":
        parameter_ids = bad_payload["optimizer"]["param_groups"][0]["params"]
        parameter_ids[:2] = reversed(parameter_ids[:2])
    elif tampered_field == "rng":
        bad_payload["rng_states"]["torch_cpu"] = torch.tensor([1], dtype=torch.uint8)
    elif tampered_field == "pending_type":
        bad_payload["pending_validation_step"] = "0"
    elif tampered_field == "pending_mismatch":
        bad_payload["pending_validation_step"] = 1
    elif tampered_field == "pending_boundary":
        bad_payload["step"] = 1
        bad_payload["pending_validation_step"] = 1
    elif tampered_field == "max_steps":
        bad_payload["step"] = 5
    elif tampered_field == "optimizer_initial_lr":
        bad_payload["optimizer"]["param_groups"][0]["initial_lr"] = 2.0e-4
    elif tampered_field == "optimizer_betas":
        bad_payload["optimizer"]["param_groups"][0]["betas"] = (0.8, 0.999)
    elif tampered_field == "optimizer_lr":
        bad_payload["optimizer"]["param_groups"][0]["lr"] = 2.0e-4
    elif tampered_field == "optimizer_lr_coordinated":
        bad_payload["optimizer"]["param_groups"][0]["lr"] = 2.0e-4
        bad_payload["scheduler"]["_last_lr"][0] = 2.0e-4
    elif tampered_field == "optimizer_lr_nonfinite":
        bad_payload["optimizer"]["param_groups"][0]["lr"] = float("nan")
    elif tampered_field == "scheduler_warmup":
        bad_payload["scheduler"]["warmup_steps"] = 2
    elif tampered_field == "scheduler_max_steps":
        bad_payload["scheduler"]["max_steps"] = 5
    elif tampered_field == "scheduler_min_lr":
        bad_payload["scheduler"]["min_lr"] = 2.0e-6
    elif tampered_field == "scheduler_base_lrs":
        bad_payload["scheduler"]["base_lrs"][0] = 2.0e-4
    elif tampered_field == "scheduler_last_lr":
        bad_payload["scheduler"]["_last_lr"][0] = 2.0e-4
    elif tampered_field == "scheduler_last_lr_nonfinite":
        bad_payload["scheduler"]["_last_lr"][0] = float("nan")
    elif tampered_field == "metrics_type":
        bad_payload["metrics"] = []
    elif tampered_field == "metrics_partial_best":
        bad_payload["metrics"] = {"best_group_a_psnr": 1.0}
    elif tampered_field == "metrics_partial_current":
        bad_payload["metrics"] = {"group_a_psnr": 1.0}
    elif tampered_field in {"metrics_nonfinite", "metrics_step_type", "metrics_step_future"}:
        bad_payload["metrics"] = {
            "best_group_a_psnr": 1.0,
            "best_group_a_ssim": 0.5,
            "best_single_psnr": 1.0,
            "best_single_ssim": 0.5,
            "best_step": 0,
        }
        if tampered_field == "metrics_nonfinite":
            bad_payload["metrics"]["best_group_a_psnr"] = float("nan")
        elif tampered_field == "metrics_step_type":
            bad_payload["metrics"]["best_step"] = 0.0
        else:
            bad_payload["metrics"]["best_step"] = 1
    elif tampered_field == "scaler":
        bad_payload["scaler"] = {"scale": 1.0}
    elif tampered_field == "cuda_rng_shape":
        bad_payload["rng_states"]["torch_cuda_all"] = [
            torch.zeros(5, dtype=torch.uint8)
        ]
    elif tampered_field == "cuda_rng_dtype":
        bad_payload["rng_states"]["torch_cuda_all"] = [torch.zeros(10)]
    bad_path = tmp_path / f"bad_{tampered_field}.pth"
    torch.save(bad_payload, bad_path)

    victim = _tiny_guarded()
    victim_optimizer = _optimizer(victim)
    victim_scheduler = WarmupCosineScheduler(
        victim_optimizer,
        warmup_steps=1,
        max_steps=4,
        min_lr=1.0e-6,
    )
    victim_ema = build_stage1_ema(victim, decay=0.9)
    victim_sampler = _sampler()
    before_model = {
        name: value.detach().clone() for name, value in victim.state_dict().items()
    }
    before_ema = {
        name: value.detach().clone() for name, value in victim_ema.shadow.items()
    }
    before_optimizer = victim_optimizer.state_dict()
    before_scheduler = victim_scheduler.state_dict()
    before_sampler = victim_sampler.state_dict()
    before_torch_rng = torch.get_rng_state().clone()
    if tampered_field.startswith("cuda_rng_"):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(
            torch.cuda,
            "get_rng_state_all",
            lambda: [torch.zeros(10, dtype=torch.uint8)],
        )
    with pytest.raises(Stage1ContractError, match=error_pattern):
        resume_stage1_checkpoint(
            bad_path,
            model=victim,
            ema=victim_ema,
            optimizer=victim_optimizer,
            scheduler=victim_scheduler,
            sampler=victim_sampler,
            expected_provenance=provenance,
            expected_validation_every=3,
            expected_max_steps=4,
        )
    for name, expected in before_model.items():
        assert torch.equal(victim.state_dict()[name], expected)
    for name, expected in before_ema.items():
        assert torch.equal(victim_ema.shadow[name], expected)
    assert victim_ema.num_updates == 0
    assert victim_optimizer.state_dict() == before_optimizer
    assert victim_scheduler.state_dict() == before_scheduler
    assert victim_sampler.state_dict() == before_sampler
    assert torch.equal(torch.get_rng_state(), before_torch_rng)


def test_stage1_resume_rejects_invalid_adam_state_before_mutation(
    tmp_path: Path,
) -> None:
    torch.manual_seed(37)
    source = _tiny_guarded()
    source_optimizer = _optimizer(source)
    source_scheduler = WarmupCosineScheduler(
        source_optimizer,
        warmup_steps=1,
        max_steps=4,
        min_lr=1.0e-6,
    )
    source_ema = build_stage1_ema(source, decay=0.9)
    train_stage1_optimizer_step(
        source,
        [_batch()],
        source_optimizer,
        source_scheduler,
        source_ema,
        step=0,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    provenance = {"ema_policy": stage1_ema_policy_metadata(0.9)}
    valid_path = tmp_path / "adam_valid.pth"
    save_stage1_checkpoint(
        valid_path,
        step=1,
        model=source,
        ema=source_ema,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        sampler=_sampler(),
        provenance=provenance,
    )

    cases = (
        ("missing_key", "Adam state fields"),
        ("shape", "Adam tensor state"),
        ("dtype", "Adam tensor state"),
        ("nonfinite", "Adam tensor state"),
        ("step_nonfinite", "Adam step"),
        ("step_future", "Adam step"),
    )
    for label, error_pattern in cases:
        bad_payload = torch.load(valid_path, map_location="cpu", weights_only=False)
        optimizer_states = bad_payload["optimizer"]["state"]
        parameter_id = next(
            key for key, value in optimizer_states.items() if value["exp_avg"].ndim >= 2
        )
        adam_state = optimizer_states[parameter_id]
        if label == "missing_key":
            adam_state.pop("exp_avg_sq")
        elif label == "shape":
            adam_state["exp_avg"] = adam_state["exp_avg"].flatten()
        elif label == "dtype":
            adam_state["exp_avg"] = adam_state["exp_avg"].double()
        elif label == "nonfinite":
            adam_state["exp_avg"].flatten()[0] = float("nan")
        elif label == "step_nonfinite":
            adam_state["step"] = torch.tensor(float("nan"))
        else:
            adam_state["step"] = torch.tensor(2.0)
        bad_path = tmp_path / f"adam_{label}.pth"
        torch.save(bad_payload, bad_path)

        victim = _tiny_guarded()
        victim_optimizer = _optimizer(victim)
        victim_scheduler = WarmupCosineScheduler(
            victim_optimizer,
            warmup_steps=1,
            max_steps=4,
            min_lr=1.0e-6,
        )
        victim_ema = build_stage1_ema(victim, decay=0.9)
        victim_sampler = _sampler()
        before = {
            name: value.detach().clone() for name, value in victim.state_dict().items()
        }
        with pytest.raises(Stage1ContractError, match=error_pattern):
            resume_stage1_checkpoint(
                bad_path,
                model=victim,
                ema=victim_ema,
                optimizer=victim_optimizer,
                scheduler=victim_scheduler,
                sampler=victim_sampler,
                expected_provenance=provenance,
                expected_validation_every=3,
                expected_max_steps=4,
            )
        for name, expected in before.items():
            assert torch.equal(victim.state_dict()[name], expected)
        assert victim_optimizer.state_dict()["state"] == {}
        assert victim_ema.num_updates == 0
        assert victim_sampler.state_dict()["consumed_optimizer_step"] == 0


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
