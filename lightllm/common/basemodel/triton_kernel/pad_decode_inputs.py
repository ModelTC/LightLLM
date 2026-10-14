from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_decode_inputs_kernel(
    input_ids,
    b_req_idx,
    b_mtp_index,
    b_seq_len,
    b_position_delta,
    b_shared_seq_len,
    b_shared_radix_node_id,
    padded_input_ids,
    padded_b_req_idx,
    padded_b_mtp_index,
    padded_b_seq_len,
    padded_b_position_delta,
    padded_b_shared_seq_len,
    padded_b_shared_radix_node_id,
    batch_size,
    new_batch_size,
    hold_request_id,
    HAS_B_POSITION_DELTA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    source_mask = offsets < batch_size
    output_mask = offsets < new_batch_size

    # 保留已有请求，新增 HOLD 请求使用 token id 1、seq_len 2。
    token_ids = tl.load(input_ids + offsets, mask=source_mask, other=1)
    tl.store(padded_input_ids + offsets, token_ids, mask=output_mask)

    req_idx = tl.load(b_req_idx + offsets, mask=source_mask, other=hold_request_id)
    tl.store(padded_b_req_idx + offsets, req_idx, mask=output_mask)

    mtp_index = tl.load(b_mtp_index + offsets, mask=source_mask, other=0)
    tl.store(padded_b_mtp_index + offsets, mtp_index, mask=output_mask)

    seq_len = tl.load(b_seq_len + offsets, mask=source_mask, other=2)
    tl.store(padded_b_seq_len + offsets, seq_len, mask=output_mask)

    # 位置偏移按需补齐；HOLD 请求不携带位置偏移和共享缓存。
    if HAS_B_POSITION_DELTA:
        position_delta = tl.load(b_position_delta + offsets, mask=source_mask, other=0)
        tl.store(padded_b_position_delta + offsets, position_delta, mask=output_mask)

    shared_seq_len = tl.load(b_shared_seq_len + offsets, mask=source_mask, other=0)
    tl.store(padded_b_shared_seq_len + offsets, shared_seq_len, mask=output_mask)

    shared_radix_node_id = tl.load(b_shared_radix_node_id + offsets, mask=source_mask, other=-1)
    tl.store(padded_b_shared_radix_node_id + offsets, shared_radix_node_id, mask=output_mask)


@torch.no_grad()
def pad_decode_inputs(
    input_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_position_delta: Optional[torch.Tensor],
    b_shared_seq_len: torch.Tensor,
    b_shared_radix_node_id: torch.Tensor,
    new_batch_size: int,
    hold_request_id: int,
):
    """一次 kernel 调用将 decode Token 和请求元数据补齐到 new_batch_size。

    调用方保证输入为同一 CUDA 设备上的连续一维 Tensor，各字段按请求对齐，
    且 new_batch_size 大于现有请求数。b_position_delta 为 None 时保持 None。
    返回顺序与七个 Tensor 输入一致。
    """
    batch_size = b_req_idx.shape[0]

    padded_input_ids = torch.empty((new_batch_size,), dtype=input_ids.dtype, device=input_ids.device)
    padded_b_req_idx = torch.empty((new_batch_size,), dtype=b_req_idx.dtype, device=b_req_idx.device)
    padded_b_mtp_index = torch.empty((new_batch_size,), dtype=b_mtp_index.dtype, device=b_mtp_index.device)
    padded_b_seq_len = torch.empty((new_batch_size,), dtype=b_seq_len.dtype, device=b_seq_len.device)
    padded_b_position_delta = None
    if b_position_delta is not None:
        padded_b_position_delta = torch.empty(
            (new_batch_size,), dtype=b_position_delta.dtype, device=b_position_delta.device
        )
    padded_b_shared_seq_len = torch.empty(
        (new_batch_size,), dtype=b_shared_seq_len.dtype, device=b_shared_seq_len.device
    )
    padded_b_shared_radix_node_id = torch.empty(
        (new_batch_size,), dtype=b_shared_radix_node_id.dtype, device=b_shared_radix_node_id.device
    )

    block_size = 256
    grid = (triton.cdiv(new_batch_size, block_size),)
    _pad_decode_inputs_kernel[grid](
        input_ids=input_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        b_position_delta=b_position_delta,
        b_shared_seq_len=b_shared_seq_len,
        b_shared_radix_node_id=b_shared_radix_node_id,
        padded_input_ids=padded_input_ids,
        padded_b_req_idx=padded_b_req_idx,
        padded_b_mtp_index=padded_b_mtp_index,
        padded_b_seq_len=padded_b_seq_len,
        padded_b_position_delta=padded_b_position_delta,
        padded_b_shared_seq_len=padded_b_shared_seq_len,
        padded_b_shared_radix_node_id=padded_b_shared_radix_node_id,
        batch_size=batch_size,
        new_batch_size=new_batch_size,
        hold_request_id=hold_request_id,
        HAS_B_POSITION_DELTA=b_position_delta is not None,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return (
        padded_input_ids,
        padded_b_req_idx,
        padded_b_mtp_index,
        padded_b_seq_len,
        padded_b_position_delta,
        padded_b_shared_seq_len,
        padded_b_shared_radix_node_id,
    )
