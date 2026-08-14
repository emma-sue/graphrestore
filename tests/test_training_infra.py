from __future__ import annotations

import torch
from torch import nn

from src.training.ema import ExponentialMovingAverage
from src.training.optimization import parameter_groups
from src.training.selection import ValidationScore, is_better_checkpoint


def test_optimizer_groups_cover_parameters_once() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
    groups = parameter_groups(
        model,
        [(('0.', '1.', '2.'), 1e-4)],
        weight_decay=1e-4,
        weight_decay_norm_bias=0.0,
    )
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for _ in model.parameters())


def test_ema_context_restores_original_weights() -> None:
    model = nn.Linear(2, 1)
    ema = ExponentialMovingAverage(model, decay=0.9)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    ema.update(model)
    changed = {name: value.clone() for name, value in model.state_dict().items()}
    with ema.apply_to(model):
        assert any(not torch.equal(model.state_dict()[name], changed[name]) for name in changed)
    for name, value in changed.items():
        torch.testing.assert_close(model.state_dict()[name], value)
    assert original.keys() == changed.keys()


def test_ema_load_preserves_destination_dtype_and_device() -> None:
    destination_model = nn.Linear(3, 2).to(dtype=torch.float64)
    destination = ExponentialMovingAverage(destination_model, decay=0.9)
    source_model = nn.Linear(3, 2).float()
    source = ExponentialMovingAverage(source_model, decay=0.8).state_dict()
    destination.load_state_dict(source)
    assert destination.decay == 0.8
    assert all(value.dtype == torch.float32 for value in destination.shadow.values())
    assert all(
        value.device == destination_model.weight.device
        for value in destination.shadow.values()
    )


def test_checkpoint_selection_uses_ssim_inside_psnr_tolerance() -> None:
    incumbent = ValidationScore(30.0, 0.90, 31.0, step=100)
    candidate = ValidationScore(30.01, 0.91, 30.0, step=200)
    assert is_better_checkpoint(candidate, incumbent)
    clearly_better = ValidationScore(30.03, 0.89, 29.0, step=300)
    assert is_better_checkpoint(clearly_better, candidate)
