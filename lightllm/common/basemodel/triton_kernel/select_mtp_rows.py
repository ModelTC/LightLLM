"""MTP verification row-selection kernels."""

from typing import NamedTuple

import torch
import triton
import triton.language as tl


class SelectedMtpRows(NamedTuple):
    input_ids: torch.Tensor
    hidden: torch.Tensor
    b_req_idx: torch.Tensor
    b_mtp_index: torch.Tensor
    b_seq_len: torch.Tensor
    mem_indexes: torch.Tensor
    b_shared_seq_len: torch.Tensor
    b_shared_radix_node_id: torch.Tensor
    b_position_delta: torch.Tensor


@triton.jit
def _select_accepted_tail_rows_kernel(
    b_req_mtp_start_loc,
    accept_len,
    input_ids,
    input_ids_stride,
    hidden,
    hidden_stride_0,
    hidden_stride_1,
    b_req_idx,
    b_req_idx_stride,
    b_mtp_index,
    b_mtp_index_stride,
    b_seq_len,
    b_seq_len_stride,
    mem_indexes,
    mem_indexes_stride,
    b_shared_seq_len,
    b_shared_seq_len_stride,
    b_shared_radix_node_id,
    b_shared_radix_node_id_stride,
    b_position_delta,
    b_position_delta_stride,
    out_input_ids,
    out_hidden,
    out_hidden_stride_0,
    out_hidden_stride_1,
    out_b_req_idx,
    out_b_mtp_index,
    out_b_seq_len,
    out_mem_indexes,
    out_b_shared_seq_len,
    out_b_shared_radix_node_id,
    out_b_position_delta,
    hidden_size,
    BLOCK_HIDDEN: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    out_row = tl.program_id(0)
    src_row = tl.load(b_req_mtp_start_loc + out_row) + tl.load(accept_len + out_row) - 1

    tl.store(out_input_ids + out_row, tl.load(input_ids + src_row * input_ids_stride))
    tl.store(out_b_req_idx + out_row, tl.load(b_req_idx + src_row * b_req_idx_stride))
    tl.store(out_b_mtp_index + out_row, tl.load(b_mtp_index + src_row * b_mtp_index_stride))
    tl.store(out_b_seq_len + out_row, tl.load(b_seq_len + src_row * b_seq_len_stride))
    tl.store(out_mem_indexes + out_row, tl.load(mem_indexes + src_row * mem_indexes_stride))
    tl.store(
        out_b_shared_seq_len + out_row,
        tl.load(b_shared_seq_len + src_row * b_shared_seq_len_stride),
    )
    tl.store(
        out_b_shared_radix_node_id + out_row,
        tl.load(b_shared_radix_node_id + src_row * b_shared_radix_node_id_stride),
    )
    tl.store(
        out_b_position_delta + out_row,
        tl.load(b_position_delta + src_row * b_position_delta_stride),
    )

    hidden_block_offsets = tl.arange(0, BLOCK_HIDDEN)
    for hidden_start in tl.range(0, hidden_size, BLOCK_HIDDEN, num_stages=PIPELINE_STAGES):
        hidden_offsets = hidden_start + hidden_block_offsets
        hidden_mask = hidden_offsets < hidden_size
        hidden_values = tl.load(
            hidden + src_row * hidden_stride_0 + hidden_offsets * hidden_stride_1,
            mask=hidden_mask,
            other=0,
        )
        tl.store(
            out_hidden + out_row * out_hidden_stride_0 + hidden_offsets * out_hidden_stride_1,
            hidden_values,
            mask=hidden_mask,
        )


@torch.no_grad()
def select_accepted_tail_rows(
    b_req_mtp_start_loc: torch.Tensor,
    accept_len: torch.Tensor,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    b_seq_len: torch.Tensor,
    mem_indexes: torch.Tensor,
    b_shared_seq_len: torch.Tensor,
    b_shared_radix_node_id: torch.Tensor,
    b_position_delta: torch.Tensor,
) -> SelectedMtpRows:
    """Select one accepted-tail row per request in a single CUDA kernel."""

    req_num = b_req_mtp_start_loc.shape[0]
    assert input_ids.is_cuda
    assert hidden.ndim == 2 and hidden.shape[0] == input_ids.shape[0]
    assert hidden.shape[1] > 0
    tensors = (
        b_req_mtp_start_loc,
        accept_len,
        hidden,
        b_req_idx,
        b_mtp_index,
        b_seq_len,
        mem_indexes,
        b_shared_seq_len,
        b_shared_radix_node_id,
        b_position_delta,
    )
    assert all(tensor.is_cuda and tensor.device == input_ids.device for tensor in tensors)

    selected = SelectedMtpRows(
        input_ids=input_ids.new_empty((req_num,)),
        hidden=hidden.new_empty((req_num, hidden.shape[1])),
        b_req_idx=b_req_idx.new_empty((req_num,)),
        b_mtp_index=b_mtp_index.new_empty((req_num,)),
        b_seq_len=b_seq_len.new_empty((req_num,)),
        mem_indexes=mem_indexes.new_empty((req_num,)),
        b_shared_seq_len=b_shared_seq_len.new_empty((req_num,)),
        b_shared_radix_node_id=b_shared_radix_node_id.new_empty((req_num,)),
        b_position_delta=b_position_delta.new_empty((req_num,)),
    )
    if req_num == 0:
        return selected

    block_hidden = 1024
    pipeline_stages = 3
    grid = (req_num,)
    _select_accepted_tail_rows_kernel[grid](
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        accept_len=accept_len,
        input_ids=input_ids,
        input_ids_stride=input_ids.stride(0),
        hidden=hidden,
        hidden_stride_0=hidden.stride(0),
        hidden_stride_1=hidden.stride(1),
        b_req_idx=b_req_idx,
        b_req_idx_stride=b_req_idx.stride(0),
        b_mtp_index=b_mtp_index,
        b_mtp_index_stride=b_mtp_index.stride(0),
        b_seq_len=b_seq_len,
        b_seq_len_stride=b_seq_len.stride(0),
        mem_indexes=mem_indexes,
        mem_indexes_stride=mem_indexes.stride(0),
        b_shared_seq_len=b_shared_seq_len,
        b_shared_seq_len_stride=b_shared_seq_len.stride(0),
        b_shared_radix_node_id=b_shared_radix_node_id,
        b_shared_radix_node_id_stride=b_shared_radix_node_id.stride(0),
        b_position_delta=b_position_delta,
        b_position_delta_stride=b_position_delta.stride(0),
        out_input_ids=selected.input_ids,
        out_hidden=selected.hidden,
        out_hidden_stride_0=selected.hidden.stride(0),
        out_hidden_stride_1=selected.hidden.stride(1),
        out_b_req_idx=selected.b_req_idx,
        out_b_mtp_index=selected.b_mtp_index,
        out_b_seq_len=selected.b_seq_len,
        out_mem_indexes=selected.mem_indexes,
        out_b_shared_seq_len=selected.b_shared_seq_len,
        out_b_shared_radix_node_id=selected.b_shared_radix_node_id,
        out_b_position_delta=selected.b_position_delta,
        hidden_size=hidden.shape[1],
        BLOCK_HIDDEN=block_hidden,
        PIPELINE_STAGES=pipeline_stages,
        num_warps=8,
        num_stages=pipeline_stages,
    )
    return selected
