import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv_split(
    KV,
    DestLoc,
    OutK,
    OutV,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_index,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_v_bs,
    stride_v_h,
    stride_v_d,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    token = tl.program_id(0)
    heads = tl.arange(0, BLOCK_HEAD)[:, None]
    dims = tl.arange(0, BLOCK_DMODEL)[None, :]
    mask = (heads < HEAD_NUM) & (dims < HEAD_DIM)
    dest = tl.load(DestLoc + token * stride_index).to(tl.int64)
    source = KV + token * stride_kv_bs + heads * stride_kv_h + dims * stride_kv_d
    k = tl.load(source, mask=mask, other=0)
    v = tl.load(source + HEAD_NUM * stride_kv_h, mask=mask, other=0)
    tl.store(OutK + dest * stride_k_bs + heads * stride_k_h + dims * stride_k_d, k, mask=mask)
    tl.store(OutV + dest * stride_v_bs + heads * stride_v_h + dims * stride_v_d, v, mask=mask)


@torch.no_grad()
def destindex_copy_kv_split(KV, DestLoc, OutK, OutV):
    head_num, head_dim = OutK.shape[1:]
    assert OutK.shape == OutV.shape
    assert KV.shape == (DestLoc.numel(), 2 * head_num, head_dim)
    if not DestLoc.numel():
        return
    _fwd_kernel_destindex_copy_kv_split[(DestLoc.numel(),)](
        KV,
        DestLoc,
        OutK,
        OutV,
        *KV.stride(),
        DestLoc.stride(0),
        *OutK.stride(),
        *OutV.stride(),
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        BLOCK_HEAD=triton.next_power_of_2(head_num),
        BLOCK_DMODEL=triton.next_power_of_2(head_dim),
        num_warps=1,
        num_stages=1,
    )


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
