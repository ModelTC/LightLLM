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

    valid_row = token_mask & (req_idx != hold_req_id)
    actual_token_num = tl.sum(tl.where(valid_row, 1, 0), axis=0)

    # Keep tensor shapes dependent only on the target graph batch size. The
    # unused tail represents zero-length sequences, so one captured graph can
    # replay arbitrary compact per-request widths at the same token batch size.
    tl.store(out_cu_q_seq_len + offsets, actual_token_num, mask=cu_mask)
    tl.store(out_conv_buffer_idx + offsets, hold_req_id, mask=token_mask)
    tl.store(out_num_accepted_tokens + offsets, 1, mask=token_mask)
    tl.debug_barrier()

    # mtp_index restarts at zero on the first row of every request group.
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
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    req_to_mtp_state_index: torch.Tensor,
    hold_req_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert compact MTP verify rows to variable-length GDN sequences.

    Rows remain request-major after dynamic verification trimming, but each request can
    contribute a different number of rows. ``b_mtp_index`` starts at zero for
    every request and then increases within that request.

    For example, let ``H`` denote ``hold_req_id`` and suppose::

        b_req_idx   = [7, 7, 7, 4, 9, 9, H, H]
        b_mtp_index = [0, 1, 2, 0, 0, 1, 0, 0]

    The first six rows contain three query sequences::

        request 7 -> rows [0, 3), query length 3
        request 4 -> rows [3, 4), query length 1
        request 9 -> rows [4, 6), query length 2

    Runtime ``H`` rows are graph padding and do not form query sequences. If
    ``req_to_mtp_state_index`` stores ``{7: 2, 4: 0, 9: 1}``, the fixed-shape
    outputs are::

        b1_cu_q_seq_len       = [0, 3, 4, 6, 6, 6, 6, 6, 6]
        b_conv_buffer_idx     = [7, 4, 9, H, H, H, H, H]
        b_num_accepted_tokens = [3, 1, 2, 1, 1, 1, 1, 1]

    ``b_conv_buffer_idx`` therefore changes from one request id per query row
    to one request id per GDN sequence. Repeated cumulative lengths describe
    zero-length tail sequences. Keeping every output shape dependent only on
    the padded input batch allows the same CUDA Graph to replay different
    per-request query lengths.
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
