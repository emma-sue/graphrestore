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
    ).eval()


def test_all_zero_guard_is_exact_identity_fp32() -> None:
    torch.manual_seed(19)
    model = _tiny_model()
    image = torch.rand(2, 3, 17, 19, dtype=torch.float32)
    guards = torch.zeros(2, len(SKILLS), 5, 5, dtype=torch.float32)
    active = torch.ones(2, len(SKILLS), dtype=torch.bool)
    with torch.inference_mode():
        output = model(image, active_mask=active, guards=guards)
    assert float((output - image).abs().max()) < 1e-7
    assert torch.equal(output, image)


def test_no_active_skill_is_identity_even_with_nonzero_guards() -> None:
    torch.manual_seed(23)
    model = _tiny_model()
    image = torch.rand(1, 3, 16, 16, dtype=torch.float32)
    guards = torch.ones(1, len(SKILLS), 4, 4, dtype=torch.float32)
    active = torch.zeros(1, len(SKILLS), dtype=torch.bool)
    with torch.inference_mode():
        result = model(
            image,
            active_mask=active,
            guards=guards,
            return_trace=True,
        )
    assert torch.equal(result.final, image)
    assert bool(result.execution.identity_mask.all())
