# SPDX-License-Identifier: Apache-2.0

"""mHC post-mixing of sublayer outputs into residual streams."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_post_4stream_kernel(
    layer_output,  # 输入 [T, H]，激活 dtype。
    residual,  # 输入 [T, 4 * H]，激活 dtype；当前子层 pre-mixing 前的残差。
    residual_mix,  # 输入 [T, 4, 4]，fp32；最后两维为输入、输出 stream。
    post_mix,  # 输入 [T, 4]，fp32。
    output,  # 输出 [T, 4 * H]，与 layer_output 相同 dtype。
    hidden: tl.constexpr,
    layer_stride_m: tl.constexpr,
    residual_stride_m: tl.constexpr,
    mix_stride_m: tl.constexpr,
    post_stride_m: tl.constexpr,
    out_stride_m: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """将子层输出和原始残差混合为四条新 stream：[T, H] + [T, 4, H] -> [T, 4, H]。

    T = tokens，H = hidden。每个 program 同时计算四条 stream 的 BLOCK_H 个
    hidden 元素，沿 residual_mix 的输入 stream 维归约；fp32 累加后写入 output。
    输出按 [T, 4 * H] 展平存储，不返回 Tensor。
    """
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden

    # Compute all four outputs in one program so the layer output and
    # residual streams are read only once, reducing prefill memory traffic.
    layer_value = tl.load(
        layer_output + token * layer_stride_m + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    residual_base = residual + token * residual_stride_m + hidden_offsets
    residual_0 = tl.load(residual_base, mask=hidden_mask, other=0.0).to(tl.float32)
    residual_1 = tl.load(residual_base + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
    residual_2 = tl.load(residual_base + 2 * hidden, mask=hidden_mask, other=0.0).to(tl.float32)
    residual_3 = tl.load(residual_base + 3 * hidden, mask=hidden_mask, other=0.0).to(tl.float32)

    post_base = post_mix + token * post_stride_m
    mix_base = residual_mix + token * mix_stride_m
    accumulator_0 = layer_value * tl.load(post_base)
    accumulator_1 = layer_value * tl.load(post_base + 1)
    accumulator_2 = layer_value * tl.load(post_base + 2)
    accumulator_3 = layer_value * tl.load(post_base + 3)

    accumulator_0 += residual_0 * tl.load(mix_base)
    accumulator_0 += residual_1 * tl.load(mix_base + 4)
    accumulator_0 += residual_2 * tl.load(mix_base + 8)
    accumulator_0 += residual_3 * tl.load(mix_base + 12)
    accumulator_1 += residual_0 * tl.load(mix_base + 1)
    accumulator_1 += residual_1 * tl.load(mix_base + 5)
    accumulator_1 += residual_2 * tl.load(mix_base + 9)
    accumulator_1 += residual_3 * tl.load(mix_base + 13)
    accumulator_2 += residual_0 * tl.load(mix_base + 2)
    accumulator_2 += residual_1 * tl.load(mix_base + 6)
    accumulator_2 += residual_2 * tl.load(mix_base + 10)
    accumulator_2 += residual_3 * tl.load(mix_base + 14)
    accumulator_3 += residual_0 * tl.load(mix_base + 3)
    accumulator_3 += residual_1 * tl.load(mix_base + 7)
    accumulator_3 += residual_2 * tl.load(mix_base + 11)
    accumulator_3 += residual_3 * tl.load(mix_base + 15)

    output_base = output + token * out_stride_m + hidden_offsets
    tl.store(output_base, accumulator_0, mask=hidden_mask)
    tl.store(output_base + hidden, accumulator_1, mask=hidden_mask)
    tl.store(output_base + 2 * hidden, accumulator_2, mask=hidden_mask)
    tl.store(output_base + 3 * hidden, accumulator_3, mask=hidden_mask)


def hc_post(
    layer_output: torch.Tensor,
    residual: torch.Tensor,
    residual_mix: torch.Tensor,
    post_mix: torch.Tensor,
    streams: int,
) -> torch.Tensor:
    """用一次 Triton launch 将子层输出混回残差 streams。

    T = tokens，S = streams，H = 单条 stream 的 hidden size。

    Args:
        layer_output: [T, H]，连续的 attention / FFN 子层输出。
        residual: [T, S * H]，连续的、该子层 pre-mixing 前保存的原始残差。
        residual_mix: [T, S, S]，fp32；来自 hc_pre_norm。
        post_mix: [T, S]，fp32；来自同一次 pre 调用。
        streams: 残差 stream 数 S；当前 Triton 入口要求 S = 4。

    Returns:
        新残差 [T, S * H]，dtype 与 layer_output 相同；输入张量保持不变。
        逻辑输出 [T, S, H] 以 fp32 计算：
        out[t, j, h] = post_mix[t, j] * layer_output[t, h]
                       + sum_i residual_mix[t, i, j] * residual.view(T, S, H)[t, i, h]。
        矩阵第一个 stream 维 i 是输入，第二个 stream 维 j 是输出。
    """

    tokens, hidden = layer_output.shape
    assert streams == 4, "the fused mHC kernel is specialized for four streams"
    assert layer_output.is_contiguous() and residual.is_contiguous()
    output = torch.empty(
        (tokens, streams * hidden),
        dtype=layer_output.dtype,
        device=layer_output.device,
    )
    block_h = min(triton.next_power_of_2(hidden), 1024)
    _hc_post_4stream_kernel[(tokens, triton.cdiv(hidden, block_h))](
        layer_output,
        residual,
        residual_mix,
        post_mix,
        output,
        hidden=hidden,
        layer_stride_m=layer_output.stride(0),
        residual_stride_m=residual.stride(0),
        mix_stride_m=residual_mix.stride(0),
        post_stride_m=post_mix.stride(0),
        out_stride_m=output.stride(0),
        BLOCK_H=block_h,
        num_warps=8,
    )
    return output
