import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_indexer_ks(
    K_fp8,
    K_scale,
    DestLoc,
    O_buffer,
    stride_k_bs,
    stride_k_d,
    stride_scale_bs,
    stride_scale_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load destination index for this thread
    dest_index = tl.load(DestLoc + cur_index).to(tl.int64)

    # Load K_fp8 (128 values) and K_scale (1 value) from source
    k_fp8_ptrs = K_fp8 + cur_index * stride_k_bs + stride_k_d * offs_d
    k_fp8 = tl.load(k_fp8_ptrs)

    k_scale = tl.load(K_scale + cur_index * stride_scale_bs)

    # Store K_fp8 to O_buffer[:, 0, :128]
    # Convert fp8 to uint8 through bitcast for storage in uint8 buffer
    o_k_ptrs = O_buffer + dest_index * stride_o_bs + stride_o_d * offs_d
    k_fp8_as_uint8 = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(o_k_ptrs, k_fp8_as_uint8)

    # Store K_scale to O_buffer[:, 0, 128:132] (4 bytes for float32)
    # Convert float32 scale to 4 uint8 bytes using bitcast and bit manipulation
    o_scale_ptr = O_buffer + dest_index * stride_o_bs + BLOCK_DMODEL * stride_o_d
    scale_as_uint32 = k_scale.to(tl.float32, bitcast=True).to(tl.uint32, bitcast=True)

    # Store each byte of the float32 scale (little-endian)
    for i in range(4):
        byte_val = ((scale_as_uint32 >> (i * 8)) & 0xFF).to(tl.uint8)
        tl.store(o_scale_ptr + i * stride_o_d, byte_val)

    return


@torch.no_grad()
def destindex_copy_indexer_ks(
    K_fp8: torch.Tensor, K_scale: torch.Tensor, DestLoc: torch.Tensor, O_buffer: torch.Tensor
):
    seq_len = DestLoc.shape[0]
    head_dim = K_fp8.shape[1]

    assert head_dim == 128, f"Expected head_dim=128, got {head_dim}"

    # Handle cases where tensor lengths don't match (e.g., during prefix cache)
    actual_seq_len = min(K_scale.shape[0], seq_len)
    if actual_seq_len < seq_len:
        K_fp8 = K_fp8[:actual_seq_len]
        K_scale = K_scale[:actual_seq_len]
        DestLoc = DestLoc[:actual_seq_len]

    assert O_buffer.shape[2] == 132, f"Expected O_buffer last dim=132, got {O_buffer.shape[2]}"

    grid = (actual_seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_indexer_ks[grid](
        K_fp8,
        K_scale,
        DestLoc,
        O_buffer,
        K_fp8.stride(0),
        K_fp8.stride(1),
        K_scale.stride(0),
        K_scale.stride(1),
        O_buffer.stride(0),
        O_buffer.stride(1),
        O_buffer.stride(2),
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
