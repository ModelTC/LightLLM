import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_scatter(
    next_token_ids,
    req_to_next_token_ids,
    b_req_idx,
    b_mtp_index,
    b_has_out,
    req_to_next_token_ids_stride,
    req_to_next_token_ids_stride_1,
    num_size,
    HAS_OUT_IS_NONE: tl.constexpr,
    BLOCK: tl.constexpr,
    OLD_VERSION_TRITON: tl.constexpr,
):
    block_index = tl.program_id(0)
    block_range = block_index * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_range < num_size

    cur_req_idx = tl.load(b_req_idx + block_range, mask=block_mask)
    cur_mtp_index = tl.load(b_mtp_index + block_range, mask=block_mask)
    cur_next_token_id = tl.load(next_token_ids + block_range, mask=block_mask)

    if not HAS_OUT_IS_NONE:
        cur_has_out = tl.load(b_has_out + block_range, mask=block_mask, other=False)
        if OLD_VERSION_TRITON:
            cur_has_out = cur_has_out != 0
        tl.store(
            req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + cur_mtp_index,
            cur_next_token_id,
            mask=cur_has_out & block_mask,
        )
    else:
        tl.store(
            req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + cur_mtp_index,
            cur_next_token_id,
            mask=block_mask,
        )

    return


@torch.no_grad()
def scatter_token(
    next_token_ids: torch.Tensor,
    req_to_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    b_has_out: torch.Tensor = None,
):
    """
    This function is used to scatter the token_info(GPU tensor) to the req_to_token_info(CPU tensor).
    Args:
        next_token_ids: (batch_size,)
        req_to_next_token_ids: (max_req_num, max_mtp_step)
        b_req_idx: (batch_size,)
        b_mtp_index: (batch_size,)
    """
    assert (
        next_token_ids.shape[0] == b_req_idx.shape[0]
    ), f"batch size not match, {next_token_ids.shape[0]} != {b_req_idx.shape[0]}"
    batch_size = b_req_idx.shape[0]
    BLOCK = 256

    grid = (triton.cdiv(batch_size, BLOCK),)
    num_warps = 1

    _fwd_kernel_scatter[grid](
        next_token_ids=next_token_ids,
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_has_out=b_has_out,
        req_to_next_token_ids_stride=req_to_next_token_ids.stride(0),
        req_to_next_token_ids_stride_1=req_to_next_token_ids.stride(1),
        num_size=batch_size,
        HAS_OUT_IS_NONE=b_has_out is None,
        BLOCK=BLOCK,
        OLD_VERSION_TRITON=triton.__version__ < "3.2.0",
        num_warps=num_warps,
        num_stages=1,
    )
    return
