import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K,
    Dest_loc,
    Out,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    head_num,
    head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = (tl.arange(0, BLOCK_DMODEL)) % head_dim

    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)

    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    tl.store(o_ptrs, k, mask=offs_h[:, None] < head_num)
    return


@triton.jit
def _fwd_kernel_destindex_copy_kv_fused(
    KV,
    Dest_loc,
    OutK,
    OutV,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_ok_bs,
    stride_ok_h,
    stride_ok_d,
    stride_ov_bs,
    stride_ov_h,
    stride_ov_d,
    head_num,
    head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = (tl.arange(0, BLOCK_DMODEL)) % head_dim

    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)
    mask = offs_h[:, None] < head_num

    kv_base = KV + cur_index * stride_kv_bs + offs_h[:, None] * stride_kv_h + offs_d[None, :] * stride_kv_d
    v_base = KV + cur_index * stride_kv_bs + (head_num + offs_h[:, None]) * stride_kv_h + offs_d[None, :] * stride_kv_d
    k_out = OutK + dest_index * stride_ok_bs + offs_h[:, None] * stride_ok_h + offs_d[None, :] * stride_ok_d
    v_out = OutV + dest_index * stride_ov_bs + offs_h[:, None] * stride_ov_h + offs_d[None, :] * stride_ov_d

    tl.store(k_out, tl.load(kv_base, mask=mask, other=0.0), mask=mask)
    tl.store(v_out, tl.load(v_base, mask=mask, other=0.0), mask=mask)
    return


@torch.no_grad()
def destindex_copy_kv(K, DestLoc, Out):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv[grid](
        K,
        DestLoc,
        Out,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        head_num,
        head_dim,
        BLOCK_DMODEL=triton.next_power_of_2(head_dim),
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@torch.no_grad()
def destindex_copy_kv_fused(KV, DestLoc, OutK, OutV):
    """Copy packed K/V rows to separate storage buffers with one kernel launch."""
    seq_len = DestLoc.shape[0]
    head_num = OutK.shape[1]
    head_dim = OutK.shape[2]
    assert KV.ndim == 3 and KV.shape[0] == seq_len
    assert OutK.shape == OutV.shape
    assert KV.shape[1] == 2 * head_num and KV.shape[2] == head_dim

    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)

    _fwd_kernel_destindex_copy_kv_fused[grid](
        KV,
        DestLoc,
        OutK,
        OutV,
        KV.stride(0),
        KV.stride(1),
        KV.stride(2),
        OutK.stride(0),
        OutK.stride(1),
        OutK.stride(2),
        OutV.stride(0),
        OutV.stride(1),
        OutV.stride(2),
        head_num,
        head_dim,
        BLOCK_DMODEL=triton.next_power_of_2(head_dim),
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=1,
        num_stages=1,
    )
    return
