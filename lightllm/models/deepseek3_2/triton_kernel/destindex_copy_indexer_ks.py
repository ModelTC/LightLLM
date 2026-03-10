import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_indexer_ks(
    K_fp8,
    K_scale,
    DestLoc,
    stride_k_bs,
    stride_k_d,
    stride_scale_bs,
    stride_scale_d,
    O_fp8,
    stride_o_bs,
    stride_o_d,
    O_fp8_scale,
    stride_o_scale_bs,
    stride_o_scale_d,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load destination index for this thread
    dest_index = tl.load(DestLoc + cur_index).to(tl.int64)

    # Load K_fp8 (128 values) and K_scale (1 value) from source
    k_fp8_ptrs = K_fp8 + cur_index * stride_k_bs + stride_k_d * offs_d
    k_fp8 = tl.load(k_fp8_ptrs)

    k_scale = tl.load(K_scale + cur_index * stride_scale_bs + stride_scale_d * 0)

    o_k_ptrs = O_fp8 + dest_index * stride_o_bs + stride_o_d * offs_d
    tl.store(o_k_ptrs, k_fp8)

    o_scale_ptr = O_fp8_scale + dest_index * stride_o_scale_bs + stride_o_scale_d * 0
    tl.store(o_scale_ptr, k_scale)
    return


@torch.no_grad()
def destindex_copy_indexer_ks(
    K_fp8: torch.Tensor, K_scale: torch.Tensor, DestLoc: torch.Tensor, O_buffer: torch.Tensor
):
    seq_len = DestLoc.shape[0]
    head_dim = K_fp8.shape[1]

    assert head_dim == 128, f"Expected head_dim=128, got {head_dim}"
    assert O_buffer.shape[2] == 132, f"Expected O_buffer last dim=132, got {O_buffer.shape[2]}"
    assert K_fp8.shape[0] == seq_len, f"Expected K_fp8 shape[0]={seq_len}, got {K_fp8.shape[0]}"
    K_fp8 = K_fp8.view(-1, head_dim)
    K_scale = K_scale.view(-1, 1)

    assert K_fp8.shape[0] == seq_len, f"Expected K_fp8 shape[0]={seq_len}, got {K_fp8.shape[0]}"
    assert K_scale.shape[0] == seq_len, f"Expected K_scale shape[0]={seq_len}, got {K_scale.shape[0]}"

    O_fp8 = O_buffer[:, :, :128].view(dtype=torch.float8_e4m3fn).view(-1, head_dim)
    O_fp8_scale = O_buffer[:, :, 128:132].view(dtype=torch.float32).view(-1, 1)

    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_indexer_ks[grid](
        K_fp8=K_fp8,
        K_scale=K_scale,
        DestLoc=DestLoc,
        stride_k_bs=K_fp8.stride(0),
        stride_k_d=K_fp8.stride(1),
        stride_scale_bs=K_scale.stride(0),
        stride_scale_d=K_scale.stride(1),
        O_fp8=O_fp8,
        stride_o_bs=O_fp8.stride(0),
        stride_o_d=O_fp8.stride(1),
        O_fp8_scale=O_fp8_scale,
        stride_o_scale_bs=O_fp8_scale.stride(0),
        stride_o_scale_d=O_fp8_scale.stride(1),
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
