# SPDX-License-Identifier: Apache-2.0

"""Expansion and mean contraction of flattened residual streams."""

from __future__ import annotations

import torch


def hc_expand(x: torch.Tensor, streams: int) -> torch.Tensor:
    """将每个 token 的 embedding 复制到 S 条 residual stream。

    输入 x: [T, H]，streams: 标量 S；T = tokens，H = hidden size。
    输出: [T, S * H]，dtype/device 与 x 相同，逻辑形状为 [T, S, H]。
    output.view(T, S, H)[t, s, h] = x[t, h]，沿 stream 维复制完整的 hidden 向量。
    """

    assert x.ndim == 2
    return x.unsqueeze(1).expand(-1, streams, -1).reshape(x.shape[0], -1)


def hc_contract(x: torch.Tensor, streams: int) -> torch.Tensor:
    """沿 stream 维取均值，将模型末尾的多条残差收缩成一个 hidden 向量。

    输入 x: [T, S * H]，streams: 标量 S；T = tokens，H = 单条 stream 的 hidden size。
    输出: [T, H]，dtype/device 与 x 相同，即 x.view(T, S, H).mean(dim=1)。
    此操作不含可学习权重；模型若使用 hc_head，应调用其专用的加权收缩算子。
    """

    assert x.ndim == 2 and x.shape[-1] % streams == 0
    return x.view(x.shape[0], streams, -1).mean(dim=1)
