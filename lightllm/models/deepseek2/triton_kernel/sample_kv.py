import torch

import triton
import triton.language as tl

TESLA = "Tesla" in torch.cuda.get_device_name(0)
CUDA_CAPABILITY = torch.cuda.get_device_capability()


@triton.jit
def _sample_kv_kernel(
    KV_input,
    KV_nope,
    KV_rope,
    B_start_loc,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    stride_input_dim,
    stride_nope_dim,
    stride_rope_dim,
    stride_req_to_tokens_b,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    start_m = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_loc = tl.load(B_start_loc + cur_batch)

    offs_nope_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    block_end_loc = tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)

    kv_loc = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_m,
        mask=offs_m < block_end_loc,
        other=0,
    )
    off_kv_nope = kv_loc[:, None] * stride_input_dim + offs_nope_d[None, :]
    off_kv_rope = kv_loc[:, None] * stride_input_dim + (offs_rope_d + BLOCK_DMODEL)[None, :]
    kv_nope = tl.load(KV_input + off_kv_nope, mask=offs_m[:, None] < block_end_loc, other=0.0)
    kv_rope = tl.load(KV_input + off_kv_rope, mask=offs_m[:, None] < block_end_loc, other=0.0)
    off_nope = (offs_m + cur_batch_start_loc)[:, None] * stride_nope_dim + offs_nope_d[None, :]
    off_rope = (offs_m + cur_batch_start_loc)[:, None] * stride_rope_dim + offs_rope_d[None, :]
    nope_ptrs = KV_nope + off_nope
    rope_ptrs = KV_rope + off_rope
    tl.store(nope_ptrs, kv_nope, mask=offs_m[:, None] < block_end_loc)
    tl.store(rope_ptrs, kv_rope, mask=offs_m[:, None] < block_end_loc)
    return


@torch.no_grad()
def sample_kv(
    kv_input,
    kv_nope,
    kv_rope,
    b_req_idx,
    b_seq_len,
    req_to_token_indexs,
):
    BLOCK = 128 if not TESLA else 64

    nope_dim = kv_nope.shape[-1]
    rope_dim = kv_rope.shape[-1]
    if nope_dim >= 512:
        BLOCK = 64 if not TESLA else 32
    else:
        BLOCK = 128 if not TESLA else 64

    batch = b_seq_len.shape[0]

    max_input_len = b_seq_len.max()
    grid = (
        batch,
        triton.cdiv(max_input_len, BLOCK),
    )
    num_warps = 4 if nope_dim <= 64 else 8

    b_start_loc = torch.cat([torch.zeros([1], device=b_seq_len.device, dtype=b_seq_len.dtype), b_seq_len[1:].cumsum(0)])
    _sample_kv_kernel[grid](
        kv_input,
        kv_nope,
        kv_rope,
        b_start_loc,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        kv_input.stride(0),
        kv_nope.stride(0),
        kv_rope.stride(0),
        req_to_token_indexs.stride(0),
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=nope_dim,
        BLOCK_ROPE_DMODEL=rope_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
