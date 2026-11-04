import torch
import triton
import triton.language as tl


@triton.jit
def _build_compressed_kv_indices_kernel(
    token_req_idx,
    position_ids,
    req_to_token_indexs,
    req_to_token_stride,
    compressed_kv_indices,
    compressed_kv_indices_stride,
    compressed_kv_lengths,
    compressed_kv_capacity,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_idx = tl.load(token_req_idx + token_idx).to(tl.int64)
    pos = tl.load(position_ids + token_idx).to(tl.int64)
    raw_len = (pos + 1) // COMPRESS_RATIO

    compressed_token_idx = block_idx * BLOCK_E + tl.arange(0, BLOCK_E)
    capacity_mask = compressed_token_idx < compressed_kv_capacity
    valid = (compressed_token_idx < raw_len) & capacity_mask
    # 从每个压缩组末尾 token 的 mem_index 得到压缩 KV 槽位。
    end_pos = compressed_token_idx * COMPRESS_RATIO + (COMPRESS_RATIO - 1)
    safe_pos = tl.where(valid, end_pos, 0)
    mem_index = tl.load(req_to_token_indexs + req_idx * req_to_token_stride + safe_pos, mask=valid, other=0).to(
        tl.int64
    )
    compressed_kv_slot = tl.where(valid, mem_index // COMPRESS_RATIO, -1).to(tl.int32)
    tl.store(
        compressed_kv_indices + token_idx * compressed_kv_indices_stride + compressed_token_idx,
        compressed_kv_slot,
        mask=capacity_mask,
    )

    if block_idx == 0:
        tl.store(compressed_kv_lengths + token_idx, tl.maximum(raw_len, 1).to(tl.int32))


def build_compressed_kv_indices(
    token_req_idx: torch.Tensor,
    position_ids: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    compress_ratio: int,
    compressed_kv_indices: torch.Tensor,
    compressed_kv_lengths: torch.Tensor,
):
    """Derive compressed entries from group-end full slots in aligned allocation pages.

    Only entries closed at the logical query position are visible. Reserved
    capacity beyond that position is masked to -1, regardless of its contents.
    """
    token_num = position_ids.shape[0]
    compressed_kv_capacity = compressed_kv_indices.shape[1]
    if token_num == 0:
        return compressed_kv_indices, compressed_kv_lengths
    BLOCK_E = 256
    grid = (token_num, triton.cdiv(compressed_kv_capacity, BLOCK_E))
    _build_compressed_kv_indices_kernel[grid](
        token_req_idx,
        position_ids,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        compressed_kv_indices,
        compressed_kv_indices.stride(0),
        compressed_kv_lengths,
        compressed_kv_capacity,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_E=BLOCK_E,
        num_warps=4,
    )
    return compressed_kv_indices, compressed_kv_lengths
