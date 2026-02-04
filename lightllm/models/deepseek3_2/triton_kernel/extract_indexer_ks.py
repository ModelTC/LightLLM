import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_extract_indexer_ks(
    I_buffer,  # Input buffer [large_size, 1, 132] uint8
    SrcLoc,  # Source indices [req_size] int32/int64
    O_fp8,  # Output FP8 [req_size, 128] float8_e4m3fn
    O_scale,  # Output scale [req_size] float32
    stride_i_bs,
    stride_i_h,
    stride_i_d,
    stride_o_fp8_bs,
    stride_o_fp8_d,
    stride_o_scale_bs,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    src_index = tl.load(SrcLoc + cur_index).to(tl.int64)

    i_k_ptrs = I_buffer + src_index * stride_i_bs + stride_i_d * offs_d
    k_fp8_as_uint8 = tl.load(i_k_ptrs)

    k_fp8 = k_fp8_as_uint8.to(tl.float8e4nv, bitcast=True)

    o_k_ptrs = O_fp8 + cur_index * stride_o_fp8_bs + stride_o_fp8_d * offs_d
    tl.store(o_k_ptrs, k_fp8)

    i_scale_base_ptr = I_buffer + src_index * stride_i_bs + BLOCK_DMODEL * stride_i_d

    byte0 = tl.load(i_scale_base_ptr + 0 * stride_i_d).to(tl.uint32)
    byte1 = tl.load(i_scale_base_ptr + 1 * stride_i_d).to(tl.uint32)
    byte2 = tl.load(i_scale_base_ptr + 2 * stride_i_d).to(tl.uint32)
    byte3 = tl.load(i_scale_base_ptr + 3 * stride_i_d).to(tl.uint32)

    scale_as_uint32 = byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)

    k_scale = scale_as_uint32.to(tl.float32, bitcast=True)

    o_scale_ptr = O_scale + cur_index * stride_o_scale_bs
    tl.store(o_scale_ptr, k_scale)

    return


@torch.no_grad()
def extract_indexer_ks(I_buffer: torch.Tensor, SrcLoc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    req_size = SrcLoc.shape[0]
    head_dim = 128

    assert I_buffer.dtype == torch.uint8, f"Expected I_buffer dtype=uint8, got {I_buffer.dtype}"
    assert I_buffer.shape[2] == 132, f"Expected I_buffer last dim=132, got {I_buffer.shape[2]}"

    # Allocate output tensors
    O_fp8 = torch.empty((req_size, head_dim), dtype=torch.float8_e4m3fn, device=I_buffer.device)
    O_scale = torch.empty((req_size,), dtype=torch.float32, device=I_buffer.device)

    grid = (req_size,)
    num_warps = 1

    _fwd_kernel_extract_indexer_ks[grid](
        I_buffer,
        SrcLoc,
        O_fp8,
        O_scale,
        I_buffer.stride(0),
        I_buffer.stride(1),
        I_buffer.stride(2),
        O_fp8.stride(0),
        O_fp8.stride(1),
        O_scale.stride(0),
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )

    return O_fp8, O_scale
