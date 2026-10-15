import torch
import triton
import triton.language as tl


@triton.jit
def _gather_window_hidden(
    target_hidden,
    b_req_idx,
    b_start_loc,
    b_ready_cache_len,
    b_q_seq_len,
    out_hidden,
    out_position_ids,
    out_mem_index,
    req_to_seq_len,
    stride_hidden_bs,
    stride_hidden_d,
    stride_out_bs,
    stride_out_d,
    hidden_dim: tl.constexpr,
    max_q_seq_len: tl.constexpr,
    window_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cur_token = tl.program_id(0)
    cur_batch = tl.program_id(1)
    cur_req_idx = tl.load(b_req_idx + cur_batch).to(tl.int64)
    cur_ready_cache_len = tl.load(b_ready_cache_len + cur_batch)
    cur_q_seq_len = tl.load(b_q_seq_len + cur_batch)
    cur_seq_len = cur_ready_cache_len + cur_q_seq_len
    cur_start_loc = tl.load(b_start_loc + cur_batch)

    # A chunk larger than the ring only needs its most recent positions.
    cur_window_start = tl.maximum(cur_ready_cache_len, cur_seq_len - window_size)
    cur_position = cur_window_start + cur_token
    token_mask = cur_position < cur_seq_len
    src_index = cur_start_loc + cur_position - cur_ready_cache_len
    out_index = cur_batch * max_q_seq_len + cur_token
    offs_d = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    hidden = tl.load(
        target_hidden + src_index.to(tl.int64) * stride_hidden_bs + offs_d * stride_hidden_d,
        mask=token_mask & (offs_d < hidden_dim),
        other=0,
    )
    tl.store(out_hidden + out_index * stride_out_bs + offs_d * stride_out_d, hidden, mask=offs_d < hidden_dim)
    if tl.program_id(2) == 0:
        tl.store(out_position_ids + out_index, tl.where(token_mask, cur_position, -1))
        # This is a logical token index into the standard [token, head, dim] KV view.
        mem_index = cur_req_idx * window_size + cur_position % window_size
        tl.store(out_mem_index + out_index, tl.where(token_mask, mem_index, -1))
        if cur_token == 0:
            tl.store(req_to_seq_len + cur_req_idx, cur_seq_len)


def gather_window_hidden(
    target_hidden,
    b_req_idx,
    b_start_loc,
    b_ready_cache_len,
    b_q_seq_len,
    req_to_seq_len,
    max_q_seq_len,
    window_size,
    alloc_func=torch.empty,
):
    max_q_seq_len = min(max_q_seq_len, window_size)
    token_num = b_req_idx.numel() * max_q_seq_len
    out_hidden = alloc_func((token_num, target_hidden.shape[1]), dtype=target_hidden.dtype, device=target_hidden.device)
    position_ids = alloc_func((token_num,), dtype=torch.int64, device=target_hidden.device)
    mem_index = alloc_func((token_num,), dtype=torch.int32, device=target_hidden.device)
    if token_num:
        _gather_window_hidden[(max_q_seq_len, b_req_idx.numel(), triton.cdiv(target_hidden.shape[1], 512))](
            target_hidden,
            b_req_idx,
            b_start_loc,
            b_ready_cache_len,
            b_q_seq_len,
            out_hidden,
            position_ids,
            mem_index,
            req_to_seq_len,
            stride_hidden_bs=target_hidden.stride(0),
            stride_hidden_d=target_hidden.stride(1),
            stride_out_bs=out_hidden.stride(0),
            stride_out_d=out_hidden.stride(1),
            hidden_dim=target_hidden.shape[1],
            max_q_seq_len=max_q_seq_len,
            window_size=window_size,
            BLOCK=512,
        )
    return out_hidden, position_ids, mem_index


@triton.jit
def _pack_window_kv(
    kv_buffer,
    noise_kv,
    b_req_idx,
    req_to_seq_len,
    out_kv,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_noise_bs,
    stride_noise_h,
    stride_noise_d,
    stride_out_b,
    stride_out_s,
    stride_out_h,
    stride_out_d,
    window_size: tl.constexpr,
    block_size: tl.constexpr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_token = tl.program_id(0)
    cur_batch = tl.program_id(1)
    cur_req_idx = tl.load(b_req_idx + cur_batch).to(tl.int64)
    cur_seq_len = tl.load(req_to_seq_len + cur_req_idx)
    cur_cache_len = tl.minimum(cur_seq_len, window_size)
    cur_position = tl.maximum(0, cur_seq_len - window_size) + cur_token
    cur_mem_index = cur_req_idx * window_size + cur_position % window_size
    cur_noise_index = cur_batch * block_size + cur_token - cur_cache_len
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    head_mask = (offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim)

    kv = tl.load(
        kv_buffer + cur_mem_index * stride_kv_bs + offs_h[:, None] * stride_kv_h + offs_d[None, :] * stride_kv_d,
        mask=(cur_token < cur_cache_len) & head_mask,
        other=0,
    )
    noise = tl.load(
        noise_kv
        + cur_noise_index.to(tl.int64) * stride_noise_bs
        + offs_h[:, None] * stride_noise_h
        + offs_d[None, :] * stride_noise_d,
        mask=(cur_token >= cur_cache_len) & (cur_token < cur_cache_len + block_size) & head_mask,
        other=0,
    )
    out_ptr = (
        out_kv
        + cur_batch * stride_out_b
        + cur_token * stride_out_s
        + offs_h[:, None] * stride_out_h
        + offs_d[None, :] * stride_out_d
    )
    tl.store(out_ptr, tl.where(cur_token < cur_cache_len, kv, noise), mask=head_mask)


def pack_window_kv(kv_buffer, noise_kv, b_req_idx, req_to_seq_len, window_size, block_size, alloc_func=torch.empty):
    head_num, head_dim = kv_buffer.shape[-2:]
    out_kv = alloc_func(
        (b_req_idx.numel(), window_size + block_size, head_num, head_dim), dtype=noise_kv.dtype, device=noise_kv.device
    )
    if b_req_idx.numel():
        _pack_window_kv[(window_size + block_size, b_req_idx.numel())](
            kv_buffer,
            noise_kv,
            b_req_idx,
            req_to_seq_len,
            out_kv,
            stride_kv_bs=kv_buffer.stride(0),
            stride_kv_h=kv_buffer.stride(1),
            stride_kv_d=kv_buffer.stride(2),
            stride_noise_bs=noise_kv.stride(0),
            stride_noise_h=noise_kv.stride(1),
            stride_noise_d=noise_kv.stride(2),
            stride_out_b=out_kv.stride(0),
            stride_out_s=out_kv.stride(1),
            stride_out_h=out_kv.stride(2),
            stride_out_d=out_kv.stride(3),
            window_size=window_size,
            block_size=block_size,
            head_num=head_num,
            head_dim=head_dim,
            BLOCK_HEAD=triton.next_power_of_2(head_num),
            BLOCK_DMODEL=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
    return out_kv
