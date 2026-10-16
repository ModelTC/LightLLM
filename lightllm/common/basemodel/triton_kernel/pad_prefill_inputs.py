import torch
import triton
import triton.language as tl


@triton.jit
def _pad_prefill_inputs_kernel(
    input_ids,
    b_req_idx,
    b_mtp_index,
    b_seq_len,
    b_ready_cache_len,
    b_is_decode_req,
    padded_input_ids,
    padded_b_req_idx,
    padded_b_mtp_index,
    padded_b_seq_len,
    padded_b_ready_cache_len,
    padded_b_is_decode_req,
    handle_token_num,
    padded_token_num,
    batch_size,
    hold_request_id,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Token 按实际补齐数量扩展，新增位置填入 token id 1。
    token_ids = tl.load(input_ids + offsets, mask=offsets < handle_token_num, other=1)
    tl.store(padded_input_ids + offsets, token_ids, mask=offsets < handle_token_num + padded_token_num)

    # 请求元数据只追加一个 HOLD 请求；后续 Token block 不再处理元数据。
    if block_start < batch_size + 1:
        source_mask = offsets < batch_size
        output_mask = offsets < batch_size + 1

        req_idx = tl.load(b_req_idx + offsets, mask=source_mask, other=hold_request_id)
        tl.store(padded_b_req_idx + offsets, req_idx, mask=output_mask)

        mtp_index = tl.load(b_mtp_index + offsets, mask=source_mask, other=0)
        tl.store(padded_b_mtp_index + offsets, mtp_index, mask=output_mask)

        seq_len = tl.load(b_seq_len + offsets, mask=source_mask, other=padded_token_num)
        tl.store(padded_b_seq_len + offsets, seq_len, mask=output_mask)

        ready_cache_len = tl.load(b_ready_cache_len + offsets, mask=source_mask, other=0)
        tl.store(padded_b_ready_cache_len + offsets, ready_cache_len, mask=output_mask)

        is_decode_req = tl.load(b_is_decode_req + offsets, mask=source_mask, other=False)
        tl.store(padded_b_is_decode_req + offsets, is_decode_req, mask=output_mask)


@torch.no_grad()
def pad_prefill_inputs(
    input_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    b_is_decode_req: torch.Tensor,
    new_handle_token_num: int,
    hold_request_id: int,
):
    """一次 kernel 调用补齐 prefill Token，并追加一个 HOLD 请求。

    调用方保证输入为同一 CUDA 设备上的连续一维 Tensor，元数据按请求对齐，
    且 new_handle_token_num 大于现有 Token 数。返回顺序与六个 Tensor 输入一致。
    """
    handle_token_num = input_ids.shape[0]
    padded_token_num = new_handle_token_num - handle_token_num
    batch_size = b_req_idx.shape[0]
    new_batch_size = batch_size + 1

    padded_input_ids = torch.empty((new_handle_token_num,), dtype=input_ids.dtype, device=input_ids.device)
    padded_b_req_idx = torch.empty((new_batch_size,), dtype=b_req_idx.dtype, device=b_req_idx.device)
    padded_b_mtp_index = torch.empty((new_batch_size,), dtype=b_mtp_index.dtype, device=b_mtp_index.device)
    padded_b_seq_len = torch.empty((new_batch_size,), dtype=b_seq_len.dtype, device=b_seq_len.device)
    padded_b_ready_cache_len = torch.empty(
        (new_batch_size,), dtype=b_ready_cache_len.dtype, device=b_ready_cache_len.device
    )
    padded_b_is_decode_req = torch.empty((new_batch_size,), dtype=b_is_decode_req.dtype, device=b_is_decode_req.device)

    block_size = 256
    grid = (triton.cdiv(max(new_handle_token_num, new_batch_size), block_size),)
    _pad_prefill_inputs_kernel[grid](
        input_ids=input_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        b_is_decode_req=b_is_decode_req,
        padded_input_ids=padded_input_ids,
        padded_b_req_idx=padded_b_req_idx,
        padded_b_mtp_index=padded_b_mtp_index,
        padded_b_seq_len=padded_b_seq_len,
        padded_b_ready_cache_len=padded_b_ready_cache_len,
        padded_b_is_decode_req=padded_b_is_decode_req,
        handle_token_num=handle_token_num,
        padded_token_num=padded_token_num,
        batch_size=batch_size,
        hold_request_id=hold_request_id,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return (
        padded_input_ids,
        padded_b_req_idx,
        padded_b_mtp_index,
        padded_b_seq_len,
        padded_b_ready_cache_len,
        padded_b_is_decode_req,
    )
