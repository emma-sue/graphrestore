from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.episode_dataset import GraphRestoreEpisodeDataset
from src.data.manifests import ALLOWED_GROUP_A, ALLOWED_SINGLE
from src.data.samplers import EpisodeRequest
from src.training.ema import ExponentialMovingAverage
from src.training.runtime import MicroBatchTrial, select_micro_batch
from src.training.stage0_engine import (
    Stage0RestorationDataset,
    Stage0StepEngine,
    aggregate_stage0_metric_records,
    build_stage0_optimizer,
    evaluate_primary_val,
    load_and_validate_stage0_config,
    save_stage0_checkpoint,
    stage0_lambda_ssim,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/stage0_mio_stagea.yaml"
TRAIN_MANIFEST = Path(
    "/root/autodl-tmp/graph/training_data/manifests/primary_train.jsonl"
)


class _TinyStage0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.patch = nn.Conv2d(3, 3, 1)
        self.encoder.level1 = nn.Sequential(nn.Conv2d(3, 3, 1), nn.GELU())
        self.encoder.down12 = nn.Conv2d(3, 3, 1)
        self.encoder.level2 = nn.Sequential(nn.Conv2d(3, 3, 1), nn.GELU())
        self.encoder.down23 = nn.Conv2d(3, 3, 1)
        self.encoder.level3 = nn.Sequential(nn.Conv2d(3, 3, 1), nn.GELU())
        self.encoder.down34 = nn.Conv2d(3, 3, 1)
        self.encoder.level4 = nn.Sequential(nn.Conv2d(3, 3, 1), nn.GELU())
        self.decoder = nn.Conv2d(3, 3, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = self.encoder.patch(image)
        value = self.encoder.level1(value)
        value = self.encoder.down12(value)
        value = self.encoder.level2(value)
        value = self.encoder.down23(value)
        value = self.encoder.level3(value)
        value = self.encoder.down34(value)
        value = self.encoder.level4(value)
        return image + self.decoder(value)


def _trial(micro: int, throughput: float, fraction: float) -> MicroBatchTrial:
    return MicroBatchTrial(
        micro_batch=micro,
        crop_size=192,
        gradient_checkpointing=False,
        consecutive_optimizer_steps=10,
        consecutive_forward_backward=max(10, 80 // micro),
        images_per_second=throughput,
        peak_reserved_bytes=int(fraction * 1000),
        total_memory_bytes=1000,
        peak_reserved_fraction=fraction,
        finite=True,
    )


def test_stage0_loss_boundary_and_fastest_valid_micro_selection() -> None:
    assert stage0_lambda_ssim(0) == 0.0
    assert stage0_lambda_ssim(11_999) == 0.0
    assert stage0_lambda_ssim(12_000) == 0.05
    selection = select_micro_batch(
        (
            _trial(8, 20.0, 0.95),
            _trial(4, 18.0, 0.82),
            _trial(2, 19.0, 0.70),
            _trial(1, 10.0, 0.60),
        )
    )
    assert selection.micro_batch == 2
    assert selection.accumulation_steps == 4


def test_stage0_step_keeps_frozen_parameters_in_optimizer_then_unfreezes() -> None:
    config, _ = load_and_validate_stage0_config(CONFIG)
    model = _TinyStage0()
    all_parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimizer, scheduler = build_stage0_optimizer(model, config)
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_parameter_ids == all_parameter_ids
    assert not any(parameter.requires_grad for parameter in model.encoder.patch.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.level1.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.down12.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.level2.parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.down23.parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.level3.parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.down34.parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.level4.parameters())
    ema = ExponentialMovingAverage(model, decay=0.9)
    engine = Stage0StepEngine(
        model,
        optimizer,
        scheduler,
        ema,
        device=torch.device("cpu"),
        accumulation_steps=2,
        micro_batch=4,
    )
    generator = torch.Generator().manual_seed(9)
    batches = [
        {
            "input": torch.rand(4, 3, 16, 16, generator=generator),
            "target": torch.rand(4, 3, 16, 16, generator=generator),
        }
        for _ in range(2)
    ]
    result = engine.train_optimizer_step(batches, step=0)
    assert result.images == 8
    assert result.lambda_ssim == 0.0
    assert torch.isfinite(torch.tensor(result.loss))
    engine.train_optimizer_step(batches, step=2000)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_stage0_metric_aggregation_is_task_equal() -> None:
    rows = []
    for task_index, order in enumerate(ALLOWED_SINGLE):
        rows.append(
            {
                "task": order,
                "group": "single",
                "psnr": 20.0 + task_index,
                "ssim": 0.70 + task_index * 0.01,
            }
        )
    for task_index, order in enumerate(ALLOWED_GROUP_A):
        # Deliberately duplicate one task: image-weighting would change the result.
        repeats = 2 if task_index == 0 else 1
        for _ in range(repeats):
            rows.append(
                {
                    "task": order,
                    "group": "A",
                    "psnr": 30.0 + task_index,
                    "ssim": 0.80 + task_index * 0.01,
                }
            )
    result = aggregate_stage0_metric_records(rows, expected_per_task=None)
    assert result.single_psnr == pytest.approx(sum(20.0 + i for i in range(8)) / 8)
    assert result.group_a_psnr == pytest.approx(sum(30.0 + i for i in range(8)) / 8)


class _SyntheticPrimaryVal(Dataset):
    training = False
    crop_size = None

    def __init__(self) -> None:
        self.tasks = (*ALLOWED_SINGLE, *ALLOWED_GROUP_A)

    def __len__(self) -> int:
        return 1600

    def __getitem__(self, index: int) -> dict[str, object]:
        task_index = index // 100
        order = self.tasks[task_index]
        value = task_index / 20.0
        return {
            "input": torch.full((3, 12, 12), value),
            "target": torch.zeros(3, 12, 12),
            "sample_id": f"sample-{index}",
            "operator_order": " + ".join(order),
            "group": "single" if len(order) == 1 else "A",
        }


def test_primary_val_batch1_loader_preserves_order_and_all_1600(monkeypatch) -> None:
    dataset = _SyntheticPrimaryVal()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def fake_metric(prediction, target, *, quantize):
        assert prediction.shape[0] == 1
        assert quantize is True
        value = prediction.mean().reshape(1)
        return SimpleNamespace(psnr=value + 20.0, ssim=value / 10.0 + 0.8)

    monkeypatch.setattr("src.training.stage0_engine.official_psnr_ssim", fake_metric)
    result = evaluate_primary_val(
        nn.Identity(),
        dataset,
        device=torch.device("cpu"),
        dataloader=loader,
    )
    assert result.image_count == 1600
    assert len(result.task_means) == 16
    assert all(row["count"] == 100 for row in result.task_means.values())


def test_best_checkpoint_keeps_standard_payload_and_ema_shadow(tmp_path: Path) -> None:
    model = nn.Conv2d(3, 3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    live = {name: value.clone() for name, value in model.state_dict().items()}
    destination = tmp_path / "best_ema.pth"
    save_stage0_checkpoint(
        destination,
        step=4,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=None,
        sampler_state={"consumed_optimizer_step": 4},
        provenance={"protocol_id": "test"},
        metrics={},
        model_as_ema=True,
    )
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "graphrestore-checkpoint-v1"
    assert payload["stage"] == "stage0"
    assert payload["model_role"] == "ema_selection"
    assert payload["resumable"] is False
    assert payload["ema"]["shadow"]
    for name, shadow in payload["ema"]["shadow"].items():
        torch.testing.assert_close(payload["model"][name], shadow)
        torch.testing.assert_close(model.state_dict()[name], live[name])


@pytest.mark.skipif(not TRAIN_MANIFEST.is_file(), reason="frozen training data unavailable")
def test_fast_stage0_view_matches_general_both_clean_path_without_subset_fields() -> None:
    _, resolved = load_and_validate_stage0_config(CONFIG)
    common = dict(
        manifest_path=TRAIN_MANIFEST,
        training_data_root=Path(str(resolved["training_data_root"])),
        depth_compat_root=PROJECT_ROOT / "artifacts/cache/mioir_depth_compat",
        crop_size=32,
        training=True,
        stage="stage0",
        base_seed=2027,
        agenticir_repo=Path(str(resolved["agenticir_repo"])),
        mioir_repo=Path(str(resolved["mioir_repo"])),
    )
    general = GraphRestoreEpisodeDataset(**common)
    fast = Stage0RestorationDataset(**common)
    index = next(
        position
        for position, record in enumerate(fast.records)
        if record.group == "A" and record.contains_low_resolution
    )
    request = EpisodeRequest(
        index=index,
        episode_type="stage0_restoration",
        absolute_step=41,
        sample_cursor=328,
    )
    expected = general[request]
    actual = fast[request]
    torch.testing.assert_close(actual["input"], expected["input"], rtol=0, atol=0)
    torch.testing.assert_close(actual["target"], expected["target"], rtol=0, atol=0)
    assert not {"only_i", "only_j", "guard_targets", "presence_target"}.intersection(actual)
