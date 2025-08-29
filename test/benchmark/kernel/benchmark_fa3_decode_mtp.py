# This file is adapted from tile-ai/tilelang:
# https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/example_mla_decode_paged.py
# The original code and this file are licensed under the Apache License, Version 2.0.
#
# Copyright (c) sgl-project and other contributors.
# Modifications Copyright (c) LightLLM contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# type: ignore
import torch
import argparse
import math
from typing import Callable, Optional, List, Literal, Union
from lightllm.common.flash_attn import flash_attn_with_kvcache_mtp
from lightllm.utils.bench_utils import do_bench


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(
    q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype
):
    # q: [b, s_q, h_q, d]
    # block_table: [b, max_seqlen_pad // block_size]
    # blocked_k: [b * max_seqlen_pad // block_size, block_size, h_kv, d]
    # cache_seqlens: [b]
    blocked_v = blocked_k[..., :dv]

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=q.device)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q.device)
        for i in range(b):
            seq_len = cache_seqlens[i // 2] - ((i + 1) % 2)
            kv_indices = block_table[i // 2, :seq_len]  # 获取前seq_len个block索引
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[kv_indices].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[kv_indices].transpose(0, 1),
                h_q,
                h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out.to(dtype), lse.to(dtype)

    out_torch, _ = ref_mla()
    return out_torch


def run_fa3_mla_mtp(
    mtp_size,
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
    dtype,
):

    assert d > dv, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv].contiguous(), q[..., dv:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[..., dv:].contiguous()

    dpe = d - dv

    batch_mtp = b // mtp_size
    cu_seqlens_q = torch.arange(0, batch_mtp + 1, step=s_q, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.cumsum(cache_seqlens, dim=0)
    cu_seqlens_k = torch.cat([torch.tensor([0]).to(cu_seqlens_k), cu_seqlens_k])
    scale = (1.0 / (dv + dpe)) ** 0.5  # log2(e)
    k_descale, v_descale = None, None
    BLOCK_H = h_q * mtp_size

    def flash_mla_fa3():
        out = flash_attn_with_kvcache_mtp(
            q=q_pe.view(-1, BLOCK_H, dpe),
            k=blocked_k_pe,
            v=blocked_k_nope,
            q_v=q_nope.view(-1, BLOCK_H, dv),
            page_table=block_table,
            seqused_k=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=1,
            softmax_scale=scale,
            is_causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            mtp_step=1,
        )
        return out.view([b, s_q, h_q, dv])

    out_flash = flash_mla_fa3()
    t = do_bench(flash_mla_fa3)

    out_ref = run_torch_mla(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype
    )

    # 计算相对绝对误差
    def print_error(a, b, name=""):
        max_absolute_error = torch.abs(a - b).max()
        relative_abs_error = torch.abs(a - b) / (torch.abs(a) + 1e-4)
        max_relative_abs_error = relative_abs_error.max()
        mean_relative_abs_error = relative_abs_error.mean()

        print(f"{name}: Maximum absolute difference: {max_absolute_error:.6e}")
        print(f"Maximum relative absolute error: {max_relative_abs_error:.6e}")
        print(f"Mean relative absolute error: {mean_relative_abs_error:.6e}")

    print_error(out_flash, out_ref, "out_flash, out_ref")
    torch.testing.assert_close(out_flash, out_ref, rtol=0.001, atol=0.001)
    print("All close")
    return out_flash, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--h_q", type=int, default=16, help="q heads number")
    parser.add_argument("--h_kv", type=int, default=1, help="kv heads number")
    parser.add_argument("--cache_seqlen", type=int, default=8192, help="kv cache context length")
    parser.add_argument("--d", type=int, default=576, help="query/key head dim, d = dv + dpe")
    parser.add_argument("--dv", type=int, default=512, help="value head dim")
    parser.add_argument("--mtp_size", type=int, default=2, help="Specifies the number of tokens per prediction.")
    args = parser.parse_args()
    b, h_q, h_kv, cache_seqlen, d, dv = args.batch, args.h_q, args.h_kv, args.cache_seqlen, args.d, args.dv
    mtp_size = args.mtp_size

    device = "cuda"
    dtype = torch.float16

    s_q = 1  # for decode, s_q = 1
    block_size = 1
    batch_mtp = b // mtp_size
    cache_seqlens = torch.tensor([cache_seqlen + i for i in range(batch_mtp)], dtype=torch.int32, device=device)
    # print(cache_seqlens[-1])
    dpe = d - dv
    causal = True

    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / 256) * 256  # ?为什么对齐256

    total_flops = s_q * (total_seqlens * 2 - batch_mtp) * h_q * (d + dv) * 2

    q = torch.randn(b, s_q, h_q, d, dtype=dtype, device=device)
    block_table = torch.arange(batch_mtp * max_seqlen_pad, dtype=torch.int32, device=device).view(
        batch_mtp, max_seqlen_pad
    )

    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=dtype, device=device)
    out_flash, latency = run_fa3_mla_mtp(
        mtp_size,
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
        dtype,
    )

    print("Tile-lang: {:.3f} ms".format(latency))
    print("Tile-lang: {:.3f} TFlops".format(total_flops / latency * 1e-9))
