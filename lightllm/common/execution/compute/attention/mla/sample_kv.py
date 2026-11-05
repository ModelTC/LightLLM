import torch

import triton
import triton.language as tl

from lightllm.utils.device_utils import is_tesla


@triton.jit
def _sample_kv_kernel(
    all_compressed_kv,
    stride_all_s,
    stride_all_d,
    sampled_compressed_kv_nope,
    stride_nope_s,
    stride_nope_d,
    sampled_k_rope,
    stride_rope_s,
    stride_rope_d,
    b_kv_start_loc,
    b_seq_len,
    req_to_token_indexs,
    stride_req_to_tokens_b,
    b_req_idx,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_NOPE_DIM: tl.constexpr,
    BLOCK_ROPE_DIM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    start_m = tl.program_id(1)

    cur_batch_seq_len = tl.load(b_seq_len + cur_batch)
    cur_batch_req_idx = tl.load(b_req_idx + cur_batch)
    cur_batch_start_loc = tl.load(b_kv_start_loc + cur_batch)

    offs_nope_d = tl.arange(0, BLOCK_NOPE_DIM)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DIM)
    offs_m = (start_m * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)) % cur_batch_seq_len

    if start_m * BLOCK_SEQ > cur_batch_seq_len:
        return

    kv_loc = tl.load(
        req_to_token_indexs + stride_req_to_tokens_b * cur_batch_req_idx + offs_m,
    ).to(tl.int64)
    off_kv_nope = kv_loc[:, None] * stride_all_s + offs_nope_d[None, :]
    off_kv_rope = kv_loc[:, None] * stride_all_s + (offs_rope_d + BLOCK_NOPE_DIM)[None, :]
    kv_nope = tl.load(all_compressed_kv + off_kv_nope)
    kv_rope = tl.load(all_compressed_kv + off_kv_rope)
    off_nope = (offs_m + cur_batch_start_loc)[:, None] * stride_nope_s + offs_nope_d[None, :]
    off_rope = (offs_m + cur_batch_start_loc)[:, None] * stride_rope_s + offs_rope_d[None, :]
    nope_ptrs = sampled_compressed_kv_nope + off_nope
    rope_ptrs = sampled_k_rope + off_rope
    tl.store(nope_ptrs, kv_nope)
    tl.store(rope_ptrs, kv_rope)
    return


@torch.no_grad()
def sample_kv(
    all_compressed_kv: torch.Tensor,
    sampled_compressed_kv_nope: torch.Tensor,
    sampled_k_rope: torch.Tensor,
    b_req_idx: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_kv_start_loc: torch.Tensor,
    max_kv_seq_len: int,
):
    nope_dim = sampled_compressed_kv_nope.shape[-1]
    rope_dim = sampled_k_rope.shape[-1]
    assert rope_dim == 64
    batch = b_seq_len.shape[0]

    BLOCK = 64 if not is_tesla() else 32
    num_warps = 8
    grid = (
        batch,
        triton.cdiv(max_kv_seq_len, BLOCK),
    )

    all_compressed_kv = all_compressed_kv.view(all_compressed_kv.shape[0], all_compressed_kv.shape[2])
    sampled_compressed_kv_nope = sampled_compressed_kv_nope.view(sampled_compressed_kv_nope.shape[0], nope_dim)
    sampled_k_rope = sampled_k_rope.view(sampled_k_rope.shape[0], rope_dim)
    assert triton.next_power_of_2(nope_dim) == nope_dim
    assert triton.next_power_of_2(rope_dim) == rope_dim

    _sample_kv_kernel[grid](
        all_compressed_kv=all_compressed_kv,
        stride_all_s=all_compressed_kv.stride(0),
        stride_all_d=all_compressed_kv.stride(1),
        sampled_compressed_kv_nope=sampled_compressed_kv_nope,
        stride_nope_s=sampled_compressed_kv_nope.stride(0),
        stride_nope_d=sampled_compressed_kv_nope.stride(1),
        sampled_k_rope=sampled_k_rope,
        stride_rope_s=sampled_k_rope.stride(0),
        stride_rope_d=sampled_k_rope.stride(1),
        b_kv_start_loc=b_kv_start_loc,
        b_seq_len=b_seq_len,
        req_to_token_indexs=req_to_token_indexs,
        stride_req_to_tokens_b=req_to_token_indexs.stride(0),
        b_req_idx=b_req_idx,
        BLOCK_SEQ=BLOCK,
        BLOCK_NOPE_DIM=nope_dim,
        BLOCK_ROPE_DIM=rope_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
