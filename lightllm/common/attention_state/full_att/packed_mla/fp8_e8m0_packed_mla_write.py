"""group64/E8M0 packed MLA 的定制 KV 量化写入，不依赖具体模型。"""

import torch
import triton
import triton.language as tl


@torch.no_grad()
def write_fp8_e8m0_packed_kv(
    kv: torch.Tensor,
    token_indices: torch.Tensor,
    packed_kv: torch.Tensor,
) -> None:
    assert kv.shape[1:] == (1, 512)
    assert packed_kv.dtype == torch.uint8 and packed_kv.shape[1] == 1
    assert packed_kv.shape[2] >= 584 and packed_kv.stride(2) == 1
    _write_fp8_e8m0_packed_kv[(token_indices.numel(),)](
        KV=kv,
        TOKEN_INDICES=token_indices.contiguous(),
        PACKED_KV=packed_kv,
        kv_stride_s=kv.stride(0),
        kv_stride_d=kv.stride(2),
        packed_stride_s=packed_kv.stride(0),
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _write_fp8_e8m0_packed_kv(
    KV,
    TOKEN_INDICES,
    PACKED_KV,
    kv_stride_s,
    kv_stride_d,
    packed_stride_s,
):
    token = tl.program_id(0)
    mem_index = tl.load(TOKEN_INDICES + token).to(tl.int64)
    packed_row = PACKED_KV + mem_index * packed_stride_s

    # 每个 token 按 group64 并行量化七组特征；末尾 64 维单独保留 BF16。
    groups = tl.arange(0, 8)
    offsets = groups[:, None] * 64 + tl.arange(0, 64)[None, :]
    group_mask = groups < 7
    values = tl.load(KV + token * kv_stride_s + offsets * kv_stride_d, mask=group_mask[:, None], other=0.0).to(
        tl.float32
    )
    absmax = tl.max(tl.abs(values), axis=1)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(absmax / 448.0, 1e-4))))
    quantized = tl.clamp(values / scale[:, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(packed_row + offsets, quantized.to(tl.uint8, bitcast=True), mask=group_mask[:, None])
    tl.store(packed_row + 576 + groups, (scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8), mask=group_mask)

    tail_offsets = tl.arange(0, 64)
    tail_values = tl.load(KV + token * kv_stride_s + (448 + tail_offsets) * kv_stride_d)
    tl.store((packed_row + 448).to(tl.pointer_type(tl.bfloat16)) + tail_offsets, tail_values)
    tl.store(packed_row + 583, 0)
