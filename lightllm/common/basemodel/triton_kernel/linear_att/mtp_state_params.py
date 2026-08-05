import torch
import triton
import triton.language as tl


@triton.jit
def _build_dynamic_mtp_linear_att_state_params_kernel(
    b_req_idx,
    b_mtp_index,
    req_to_mtp_state_index,
    out_cu_q_seq_len,
    out_conv_buffer_idx,
    out_num_accepted_tokens,
    batch_size,
    hold_req_id,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    token_mask = offsets < batch_size
    cu_mask = offsets <= batch_size

    req_idx = tl.load(b_req_idx + offsets, mask=token_mask, other=hold_req_id)
    mtp_index = tl.load(b_mtp_index + offsets, mask=token_mask, other=0)

    # CUDA-graph warmup uses HOLD_REQUEST_ID for every row but still supplies
    # a real 0..K MTP layout. Runtime graph padding, by contrast, appends HOLD
    # rows whose mtp_index is always zero. Preserve warmup work while excluding
    # runtime padding from the final real sequence.
    has_hold_draft_row = tl.sum(
        tl.where(token_mask & (req_idx == hold_req_id) & (mtp_index > 0), 1, 0),
        axis=0,
    ) > 0
    valid_row = token_mask & ((req_idx != hold_req_id) | has_hold_draft_row)
    actual_token_num = tl.sum(tl.where(valid_row, 1, 0), axis=0)

    # Keep tensor shapes dependent only on the target graph batch size. The
    # unused tail represents zero-length sequences, so one captured graph can
    # replay arbitrary compact per-request widths at the same token batch size.
    tl.store(out_cu_q_seq_len + offsets, actual_token_num, mask=cu_mask)
    tl.store(out_conv_buffer_idx + offsets, hold_req_id, mask=token_mask)
    tl.store(out_num_accepted_tokens + offsets, 1, mask=token_mask)
    tl.debug_barrier()

    is_start = valid_row & (mtp_index == 0)
    sequence_index = tl.cumsum(tl.where(is_start, 1, 0), axis=0) - 1
    accepted_state_index = tl.load(
        req_to_mtp_state_index + req_idx,
        mask=is_start,
        other=0,
    )

    tl.store(out_cu_q_seq_len + sequence_index, offsets, mask=is_start)
    tl.store(out_conv_buffer_idx + sequence_index, req_idx, mask=is_start)
    tl.store(out_num_accepted_tokens + sequence_index, accepted_state_index + 1, mask=is_start)


def build_dynamic_mtp_linear_att_state_params(
    *,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    req_to_mtp_state_index: torch.Tensor,
    hold_req_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build graph-stable varlen GDN parameters for compact MTP verify rows.

    The compact input remains request-major and each request contributes a
    prefix beginning at ``b_mtp_index == 0``. Outputs are padded to the token
    batch size; trailing entries describe zero-length sequences.
    """

    assert b_req_idx.is_cuda and b_mtp_index.is_cuda and req_to_mtp_state_index.is_cuda
    assert b_req_idx.ndim == 1 and b_req_idx.shape == b_mtp_index.shape
    assert b_req_idx.dtype == torch.int32 and b_mtp_index.dtype == torch.int32
    batch_size = b_req_idx.shape[0]
    assert batch_size > 0

    b1_cu_q_seq_len = torch.empty((batch_size + 1,), dtype=torch.int32, device=b_req_idx.device)
    b_conv_buffer_idx = torch.empty_like(b_req_idx)
    b_num_accepted_tokens = torch.empty_like(b_req_idx)

    _build_dynamic_mtp_linear_att_state_params_kernel[(1,)](
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        req_to_mtp_state_index=req_to_mtp_state_index,
        out_cu_q_seq_len=b1_cu_q_seq_len,
        out_conv_buffer_idx=b_conv_buffer_idx,
        out_num_accepted_tokens=b_num_accepted_tokens,
        batch_size=batch_size,
        hold_req_id=int(hold_req_id),
        BLOCK_SIZE=triton.next_power_of_2(batch_size + 1),
        num_warps=8,
        num_stages=1,
    )
    return b1_cu_q_seq_len, b_conv_buffer_idx, b_num_accepted_tokens
