# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Triton implementation of the DSV4 math adapted from SGLang commit
# 8cea0473ea5299bc04885f8f6ba71269415a39b5.

import torch
import triton
import triton.language as tl

from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import _butterfly_stage
from lightllm.utils.custom_kernel_utis import get_pdl_launch_kwargs


@triton.jit
def _q_norm_rope_kernel(
    q_input,
    q_output,
    freqs_cis,
    position_ids,
    q_input_stride,
    q_output_stride,
    num_rows,
    NUM_HEADS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    dim = tl.arange(0, 512)
    batch = row // NUM_HEADS
    head = row % NUM_HEADS
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(
        q_input + batch[:, None] * q_input_stride + head[:, None] * 512 + dim[None, :],
        mask=row[:, None] < num_rows,
        other=0,
    ).to(tl.float32)
    norm = tl.rsqrt(tl.sum(x * x, axis=1) / 512 + EPS)
    # Main Q rounds RMSNorm to BF16 before applying RoPE.
    x = (x * norm[:, None]).to(tl.bfloat16).to(tl.float32)
    out_base = q_output + batch[:, None] * q_output_stride + head[:, None] * 512
    tl.store(out_base + dim[None, :], x, mask=(row[:, None] < num_rows) & (dim[None, :] < 448))
    pair = tl.arange(0, 32)
    rope_dim = tl.arange(0, 64)
    rope_base = q_input + batch[:, None] * q_input_stride + head[:, None] * 512 + 448
    rope = tl.load(rope_base + rope_dim[None, :], mask=row[:, None] < num_rows, other=0).to(tl.float32)
    rope = (rope * norm[:, None]).to(tl.bfloat16).to(tl.float32)
    real, imag = tl.split(tl.reshape(rope, (BLOCK_R, 32, 2)))
    pos = tl.load(position_ids + batch, mask=row < num_rows, other=0)
    freq = freqs_cis + pos[:, None] * 64 + pair[None, :] * 2
    mask = row[:, None] < num_rows
    cos = tl.load(freq, mask=mask, other=1)
    sin = tl.load(freq + 1, mask=mask, other=0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    rope = tl.reshape(tl.join(real * cos - imag * sin, real * sin + imag * cos), (BLOCK_R, 64))
    tl.store(out_base + 448 + rope_dim[None, :], rope, mask=mask)


@triton.jit
def _kv_norm_rope_to_swa_kernel(
    kv,
    kv_norm_weight,
    freqs_cis,
    position_ids,
    swa_write_slots,
    swa_runtime_write_slots,
    swa_kv_buffer,
    req_to_swa,
    kv_stride,
    EPS: tl.constexpr,
    SWA_PAGE_TOKEN_NUM: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    WRITE_SWA_RUNTIME: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    dim = tl.arange(0, 512)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(kv + row * kv_stride + dim).to(tl.float32)
    w = tl.load(kv_norm_weight + dim).to(tl.float32)
    norm = tl.rsqrt(tl.sum(x * x, axis=0) / 512 + EPS)
    x = x * norm * w
    groups = tl.arange(0, 8)
    cols = tl.arange(0, 64)
    grouped = tl.reshape(x, (8, 64))
    scale = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1e-4) / 448
    bits = scale.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    inv_scale = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    quantized = tl.clamp(grouped * inv_scale[:, None], -448, 448)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    real, imag = tl.split(tl.reshape(x, (8, 32, 2)))
    pair = tl.arange(0, 32)
    pos = tl.load(position_ids + row)
    cos = tl.load(freqs_cis + pos * 64 + pair * 2)
    sin = tl.load(freqs_cis + pos * 64 + pair * 2 + 1)
    rope = tl.reshape(
        tl.join(real * cos[None, :] - imag * sin[None, :], real * sin[None, :] + imag * cos[None, :]), (8, 64)
    )
    for destination in tl.static_range(2 if WRITE_SWA_RUNTIME else 1):
        slot = tl.load(swa_write_slots + row) if destination == 0 else tl.load(swa_runtime_write_slots + row)
        slot = slot.to(tl.int64)
        valid_slot = True if destination == 0 else slot >= 0
        page, offset = slot // SWA_PAGE_TOKEN_NUM, slot % SWA_PAGE_TOKEN_NUM
        value_base = page * PAGE_BYTES + offset * 576
        buffer = swa_kv_buffer if destination == 0 else req_to_swa
        tl.store(
            buffer.to(tl.pointer_type(tl.float8e4nv)) + value_base + groups[:, None] * 64 + cols[None, :],
            quantized,
            mask=(groups[:, None] < 7) & valid_slot,
        )
        tl.store(
            buffer + page * PAGE_BYTES + SWA_PAGE_TOKEN_NUM * 576 + offset * 8 + groups,
            exponent,
            mask=(groups < 7) & valid_slot,
        )
        rope_ptr = tl.broadcast_to(
            buffer.to(tl.pointer_type(tl.bfloat16)) + (value_base + 448) // 2 + cols[None, :], (8, 64)
        )
        tl.store(rope_ptr, rope, mask=(groups[:, None] == 7) & valid_slot)


@triton.jit
def _indexer_rope_hadamard_quant_kernel(
    q_input,
    weight,
    freqs_cis,
    position_ids,
    q_fp8,
    weights_out,
    num_rows,
    NUM_HEADS: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    dim = tl.arange(0, 128)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(q_input + row[:, None] * 128 + dim[None, :], mask=row[:, None] < num_rows, other=0).to(tl.float32)
    real, imag = tl.split(tl.reshape(x, (BLOCK_R, 64, 2)))
    pair = tl.arange(0, 64)
    pos = tl.load(position_ids + row // NUM_HEADS, mask=row < num_rows, other=0)
    freq = freqs_cis + pos[:, None] * 64 + (pair[None, :] - 32) * 2
    mask = (row[:, None] < num_rows) & (pair[None, :] >= 32)
    cos = tl.load(freq, mask=mask, other=1)
    sin = tl.load(freq + 1, mask=mask, other=0)
    x = tl.reshape(tl.join(real * cos - imag * sin, real * sin + imag * cos), (BLOCK_R, 128))
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    for stage in tl.static_range(7):
        x = _butterfly_stage(x, 64 >> stage, 1 << stage, BLOCK_R, 128)
    x = x * 0.08838834764831845
    scale = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4) / 448
    tl.store(
        q_fp8 + row[:, None] * 128 + dim[None, :], tl.clamp(x / scale[:, None], -448, 448), mask=row[:, None] < num_rows
    )
    weight = tl.load(weight + row, mask=row < num_rows, other=0).to(tl.float32)
    tl.store(weights_out + row, weight * WEIGHT_SCALE * scale, mask=row < num_rows)


@torch.no_grad()
def fused_q_norm_rope(q_input, q_output, eps, freqs_cis, position_ids):
    """RMSNorm/RoPE for BF16 Q [T, heads, 512], allowing padded head strides.

    freqs_cis is contiguous complex64 [max_position, 32]; position_ids is int32/int64 [T].
    RMSNorm rounds to BF16 before interleaved RoPE on the last 64 dimensions.
    """
    rows = q_input.shape[0] * q_input.shape[1]
    if rows:
        block_r, warps = (1, 1) if rows <= 256 else (4, 2)
        _q_norm_rope_kernel[(triton.cdiv(rows, block_r),)](
            q_input,
            q_output,
            torch.view_as_real(freqs_cis).flatten(-2),
            position_ids,
            q_input.stride(0),
            q_output.stride(0),
            rows,
            NUM_HEADS=q_input.shape[1],
            EPS=float(eps),
            BLOCK_R=block_r,
            num_warps=warps,
            num_stages=1,
            **get_pdl_launch_kwargs(q_input.device),
        )


@torch.no_grad()
def fused_kv_norm_rope_to_swa(
    kv,
    kv_norm_weight,
    eps,
    freqs_cis,
    position_ids,
    swa_write_slots,
    swa_kv_buffer,
    swa_page_token_num,
    swa_runtime_write_slots=None,
    req_to_swa=None,
):
    """将 BF16 KV [T, 512] 归一化、旋转并写入 packed SWA。

    swa_write_slots 对应当前 attention 使用的位置；swa_runtime_write_slots 将
    长 prefill 的尾部同时保留到 req_to_swa，值为 -1 时跳过该 token。

    Each uint8 page stores 576 data bytes per slot followed by 8 scale bytes per
    slot. NoPE uses FP8 with UE8M0 scales; the last 64 dimensions use BF16 RoPE.
    """
    if kv.shape[0]:
        flat = swa_kv_buffer.view(-1)
        _kv_norm_rope_to_swa_kernel[(kv.shape[0],)](
            kv,
            kv_norm_weight,
            torch.view_as_real(freqs_cis).flatten(-2),
            position_ids,
            swa_write_slots,
            swa_runtime_write_slots,
            flat,
            req_to_swa,
            kv.stride(0),
            EPS=float(eps),
            SWA_PAGE_TOKEN_NUM=swa_page_token_num,
            PAGE_BYTES=swa_kv_buffer.shape[-1],
            WRITE_SWA_RUNTIME=swa_runtime_write_slots is not None,
            num_warps=1 if kv.shape[0] >= 512 else 8,
            num_stages=1,
            **get_pdl_launch_kwargs(kv.device),
        )


@torch.no_grad()
def fused_q_indexer_rope_hadamard_quant(
    q_input,
    weight,
    weight_scale,
    freqs_cis,
    position_ids,
    q_fp8=None,
    weights_out=None,
):
    """RoPE/Hadamard-128 and FP8 quantization for contiguous BF16 indexer Q.

    q_input: [T, heads, 128], weight: [T, heads]. Return FP8 Q and FP32
    weights [T, heads, 1] folded with weight_scale and the quantization scale.
    """
    if q_fp8 is None:
        q_fp8 = torch.empty_like(q_input, dtype=torch.float8_e4m3fn)
    if weights_out is None:
        weights_out = torch.empty((*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device)
    rows = q_input.shape[0] * q_input.shape[1]
    if rows:
        block_r = 1 if rows <= 256 else min(32, triton.next_power_of_2(max(1, rows // 1024)))
        _indexer_rope_hadamard_quant_kernel[(triton.cdiv(rows, block_r),)](
            q_input,
            weight,
            torch.view_as_real(freqs_cis).flatten(-2),
            position_ids,
            q_fp8,
            weights_out,
            rows,
            NUM_HEADS=q_input.shape[1],
            WEIGHT_SCALE=float(weight_scale),
            BLOCK_R=block_r,
            num_warps=1,
            num_stages=1,
            **get_pdl_launch_kwargs(q_input.device),
        )
    return q_fp8, weights_out
