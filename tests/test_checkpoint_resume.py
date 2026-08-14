from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from src.training.checkpointing import (
    CheckpointProvenanceError,
    atomic_torch_save,
    checkpoint_payload,
    load_checkpoint,
)


def test_checkpoint_restores_model_optimizer_rng_and_provenance(tmp_path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    provenance = {"config_sha256": "abc", "manifests": {"train": "def"}}
    payload = checkpoint_payload(
        stage="stage0",
        step=1,
        model=model,
        ema_state=None,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        sampler_state={"cursor": 3},
        provenance=provenance,
    )
    checkpoint = tmp_path / "last.pth"
    atomic_torch_save(payload, checkpoint)
    expected_next = (random.random(), float(np.random.rand()), float(torch.rand(())))

    restored = nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    loaded = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        expected_provenance=provenance,
    )
    actual_next = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual_next == pytest.approx(expected_next)
    assert loaded["sampler_state"] == {"cursor": 3}
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_rejects_provenance_drift(tmp_path) -> None:
    model = nn.Linear(2, 1)
    checkpoint = tmp_path / "last.pth"
    atomic_torch_save(
        checkpoint_payload(
            stage="stage0",
            step=0,
            model=model,
            ema_state=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            sampler_state=None,
            provenance={"config_sha256": "old"},
        ),
        checkpoint,
    )
    with pytest.raises(CheckpointProvenanceError):
        load_checkpoint(checkpoint, expected_provenance={"config_sha256": "new"})


def test_checkpoint_rejects_selection_snapshot_before_mutating_state(tmp_path) -> None:
    torch.manual_seed(41)
    source = nn.Linear(3, 2)
    payload = checkpoint_payload(
        stage="stage0",
        step=5,
        model=source,
        ema_state=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        sampler_state=None,
        provenance={"config_sha256": "bound"},
    )
    payload["model_role"] = "ema_selection"
    payload["resumable"] = False
    checkpoint = tmp_path / "best_ema.pth"
    atomic_torch_save(payload, checkpoint)

    victim = nn.Linear(3, 2)
    before = {name: value.detach().clone() for name, value in victim.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(CheckpointProvenanceError, match="resumable flag mismatch"):
        load_checkpoint(
            checkpoint,
            model=victim,
            expected_provenance={"config_sha256": "bound"},
            require_resumable=True,
            expected_model_role="raw_training_state",
        )
    for name, value in victim.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
