import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_window_write_indices(
    b_req_idx,
    b_start_loc,
    b_ready_cache_len,
    b_commit_len,
    out_mem_index,
    req_to_seq_len,
    window_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_batch).to(tl.int64)
    cur_ready_cache_len = tl.load(b_ready_cache_len + cur_batch)
    cur_commit_len = tl.load(b_commit_len + cur_batch)
    cur_seq_len = cur_ready_cache_len + cur_commit_len
    cur_start_loc = tl.load(b_start_loc + cur_batch)

    # Only the retained tail may write: older rows would collide in the ring.
    tail_start = tl.maximum(0, cur_commit_len - window_size)
    for start in range(tail_start, cur_commit_len, BLOCK):
        row = start + tl.arange(0, BLOCK)
        position = cur_ready_cache_len + row
        mem_index = cur_req_idx * window_size + position % window_size
        tl.store(out_mem_index + cur_start_loc + row, mem_index, mask=row < cur_commit_len)
    # Publish the new length now; the caller must order all layer KV writes before any state reads.
    tl.store(req_to_seq_len + cur_req_idx, cur_seq_len)


def prepare_window_write_indices(
    b_req_idx,
    b_start_loc,
    b_ready_cache_len,
    b_commit_len,
    req_to_seq_len,
    window_size,
    token_num,
    alloc_func=torch.empty,
):
    """Return one destination per source token; rejected and expired rows stay -1."""
    mem_index = alloc_func((token_num,), dtype=torch.int32, device=b_req_idx.device)
    mem_index.fill_(-1)
    if b_req_idx.numel():
        _prepare_window_write_indices[(b_req_idx.numel(),)](
            b_req_idx,
            b_start_loc,
            b_ready_cache_len,
            b_commit_len,
            mem_index,
            req_to_seq_len,
            window_size=window_size,
            BLOCK=256,
        )
    return mem_index


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
