"""Soft three-cycle penalty for the shared 28-pair relation head."""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor


def _pair_index(num_skills: int) -> dict[tuple[int, int], int]:
    return {pair: index for index, pair in enumerate(combinations(range(num_skills), 2))}


def cycle_consistency_loss(relation_logits: Tensor, num_skills: int = 8) -> Tensor:
    expected_pairs = num_skills * (num_skills - 1) // 2
    if relation_logits.ndim != 3 or relation_logits.shape[1:] != (expected_pairs, 3):
        raise ValueError(f"expected Bx{expected_pairs}x3 relation logits")
    probabilities = relation_logits.softmax(dim=-1)
    indices = _pair_index(num_skills)

    def directed(source: int, destination: int) -> Tensor:
        low, high = sorted((source, destination))
        pair = probabilities[:, indices[(low, high)]]
        return pair[:, 0] if source == low else pair[:, 1]

    penalties = []
    for i, j, k in combinations(range(num_skills), 3):
        clockwise = directed(i, j) * directed(j, k) * directed(k, i)
        counter = directed(j, i) * directed(k, j) * directed(i, k)
        penalties.append(clockwise + counter)
    if not penalties:
        return relation_logits.new_zeros(())
    return torch.stack(penalties, dim=1).mean()
