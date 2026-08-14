"""Restormer building blocks used by the checkpoint-compatible host.

The parameterized layers are mechanically retained from ProVIR's clean
prompt-free Restormer host.  Keeping their nesting and names unchanged is a
hard warm-start invariant; GraphRestore extensions live outside these blocks.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, norm_type: str):
        super().__init__()
        body = BiasFreeLayerNorm if norm_type == "BiasFree" else WithBiasLayerNorm
        self.body = body(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), height, width)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float, bias: bool):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, bias: bool):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            3,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query, key, value = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        pattern = "b (head c) h w -> b head c (h w)"
        query = rearrange(query, pattern, head=self.heads)
        key = rearrange(key, pattern, head=self.heads)
        value = rearrange(value, pattern, head=self.heads)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (
            (query @ key.transpose(-2, -1)) * self.temperature
        ).softmax(dim=-1)
        output = rearrange(
            attention @ value,
            "b head c (h w) -> b (head c) h w",
            head=self.heads,
            h=height,
            w=width,
        )
        if output.shape != (batch, channels, height, width):
            raise RuntimeError("attention reshape drift")
        return self.project_out(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        expansion: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, norm_type)
        self.attn = Attention(dim, heads, bias)
        self.norm2 = LayerNorm(dim, norm_type)
        self.ffn = FeedForward(dim, expansion, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class BlockStack(nn.Sequential):
    def __init__(self, *blocks: nn.Module, gradient_checkpointing: bool = False):
        super().__init__(*blocks)
        self.gradient_checkpointing = gradient_checkpointing

    def run_block(self, index: int, x: torch.Tensor) -> torch.Tensor:
        """Run one block with the stack's checkpointing policy.

        The helper lets the guarded decoder insert an adapter after every
        original block without wrapping that block and changing its state key.
        """

        block = self[index]
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index in range(len(self)):
            x = self.run_block(index, x)
        return x


def make_blocks(
    dim: int,
    count: int,
    heads: int,
    expansion: float = 2.66,
    bias: bool = False,
    norm_type: str = "WithBias",
    gradient_checkpointing: bool = False,
) -> BlockStack:
    return BlockStack(
        *[
            TransformerBlock(dim, heads, expansion, bias, norm_type)
            for _ in range(count)
        ],
        gradient_checkpointing=gradient_checkpointing,
    )


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = 8,
) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (height, width)


def crop_to_shape(x: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    height, width = shape
    return x[..., :height, :width]
