import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_extract_indexer_ks(
    in_fp8,
    stride_in_fp8_bs,
    stride_in_fp8_h,
    stride_in_fp8_d,
    in_fp8_scale,
    stride_in_scale_bs,
    stride_in_scale_h,
    stride_in_scale_d,
    req_to_token_indexs,
    stride_req_to_token_m,
    stride_req_to_token_n,
    b_seq_len,
    b_req_idx,
    O_fp8,
    stride_o_fp8_bs,
    stride_o_fp8_d,
    O_scale,
    stride_o_scale_bs,
    stride_o_scale_d,
    mtp_step,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ_LEN: tl.constexpr,
):
    origin_cur_req_index = tl.program_id(0)
    cur_req_index = (origin_cur_req_index + 1) * (mtp_step + 1) - 1
    token_start_index = tl.program_id(1)
    cur_req_idx = tl.load(b_req_idx + cur_req_index)
    cur_seq_len = tl.load(b_seq_len + cur_req_index)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    b_seq_len = tl.load(
        b_seq_len + (tl.arange(0, BLOCK_SEQ_LEN) + 1) * (mtp_step + 1) - 1,
        mask=tl.arange(0, BLOCK_SEQ_LEN) < origin_cur_req_index,
        other=0,
    )
    store_start_index = tl.sum(b_seq_len)

    for i in range(token_start_index, cur_seq_len, tl.num_programs(1)):
        mem_index = tl.load(req_to_token_indexs + cur_req_idx * stride_req_to_token_m + i * stride_req_to_token_n)

        in_fp8_ptrs = in_fp8 + mem_index * stride_in_fp8_bs + 0 * stride_in_fp8_h + stride_in_fp8_d * offs_d
        kv_fp8 = tl.load(in_fp8_ptrs)

        in_scale_ptrs = in_fp8_scale + mem_index * stride_in_scale_bs + 0 * stride_in_scale_h + 0 * stride_in_scale_d
        kv_scale = tl.load(in_scale_ptrs)

        o_fp8_ptrs = O_fp8 + (store_start_index + i) * stride_o_fp8_bs + stride_o_fp8_d * offs_d
        tl.store(o_fp8_ptrs, kv_fp8)

        o_scale_ptr = O_scale + (store_start_index + i) * stride_o_scale_bs
        tl.store(o_scale_ptr, kv_scale)

    return


@torch.no_grad()
def extract_indexer_ks(
    I_buffer: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_req_idx: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    out_token_num: int,
    max_kv_seq_len: int,
    mtp_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = 128

    assert I_buffer.dtype == torch.uint8, f"Expected I_buffer dtype=uint8, got {I_buffer.dtype}"
    assert I_buffer.shape[2] == 132, f"Expected I_buffer last dim=132, got {I_buffer.shape[2]}"
    in_fp8 = I_buffer[:, :, 0:128].view(dtype=torch.float8_e4m3fn)
    in_fp8_scale = I_buffer[:, :, 128:132].view(dtype=torch.float32)

    # Allocate output tensors
    O_fp8 = torch.empty((out_token_num // (mtp_step + 1), head_dim), dtype=torch.float8_e4m3fn, device=I_buffer.device)
    O_scale = torch.empty((out_token_num // (mtp_step + 1), 1), dtype=torch.float32, device=I_buffer.device)

    assert b_seq_len.shape[0] % (mtp_step + 1) == 0
    grid = (b_seq_len.shape[0] // (mtp_step + 1), min(256, max_kv_seq_len))
    num_warps = 1

    _fwd_kernel_extract_indexer_ks[grid](
        in_fp8,
        stride_in_fp8_bs=in_fp8.stride(0),
        stride_in_fp8_h=in_fp8.stride(1),
        stride_in_fp8_d=in_fp8.stride(2),
        in_fp8_scale=in_fp8_scale,
        stride_in_scale_bs=in_fp8_scale.stride(0),
        stride_in_scale_h=in_fp8_scale.stride(1),
        stride_in_scale_d=in_fp8_scale.stride(2),
        req_to_token_indexs=req_to_token_indexs,
        stride_req_to_token_m=req_to_token_indexs.stride(0),
        stride_req_to_token_n=req_to_token_indexs.stride(1),
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        O_fp8=O_fp8,
        stride_o_fp8_bs=O_fp8.stride(0),
        stride_o_fp8_d=O_fp8.stride(1),
        O_scale=O_scale,
        stride_o_scale_bs=O_scale.stride(0),
        stride_o_scale_d=O_scale.stride(1),
        mtp_step=mtp_step,
        BLOCK_DMODEL=head_dim,
        BLOCK_SEQ_LEN=triton.next_power_of_2(b_seq_len.shape[0] // (mtp_step + 1)),
        num_warps=num_warps,
        num_stages=1,
    )

    return O_fp8, O_scale
