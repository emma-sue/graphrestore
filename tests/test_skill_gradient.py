from __future__ import annotations

import torch

from src.net.graphrestore import GuardedSkillRestormer
from src.net.skill_adapter import SKILLS


def _tiny_model() -> GuardedSkillRestormer:
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


def test_active_adapter_up_has_first_backward_gradient() -> None:
    torch.manual_seed(7)
    model = _tiny_model().train()
    image = torch.rand(1, 3, 16, 16)
    guards = torch.zeros(1, len(SKILLS), 4, 4)
    guards[:, 0] = 1.0
    active = torch.zeros(1, len(SKILLS), dtype=torch.bool)
    active[:, 0] = True

    output = model(image, active_mask=active, guards=guards)
    output.square().mean().backward()

    for level in ("level3", "level2", "level1", "refinement"):
        for block in model.decoder.skill_bank.adapters[level]:
            gradient = block["noise"].up.weight.grad
            assert gradient is not None
            assert float(gradient.abs().sum()) > 0.0
            for inactive_skill in SKILLS[1:]:
                inactive_gradient = block[inactive_skill].up.weight.grad
                assert inactive_gradient is None or torch.count_nonzero(
                    inactive_gradient
                ) == 0


def test_zero_init_adapter_does_not_change_initial_decoder_candidate() -> None:
    torch.manual_seed(11)
    model = _tiny_model().eval()
    image = torch.rand(1, 3, 16, 16)
    padded = image
    features = model.encode(padded)
    guards = torch.ones(1, len(SKILLS), 4, 4)
    active = torch.zeros(1, len(SKILLS), dtype=torch.bool)
    active[:, :2] = True
    with torch.inference_mode():
        guarded = model.decode_delta(features, guards=guards, active_mask=active)
        inactive = model.decode_delta(
            features,
            guards=guards,
            active_mask=torch.zeros_like(active),
        )
    assert torch.equal(guarded, inactive)
