"""仅动态 MTP verify 使用的选行与 ModelInput 压缩算子。"""

from typing import Optional

import triton
import triton.language as tl
import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.mtp_manager import MtpManager


# 动态 verify 行选择。
@triton.jit
def _fwd_kernel_cumprod_scores(
    req_to_next_token_scores,
    req_to_next_token_scores_stride,
    b_req_idx,
    max_draft_step,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index * (max_draft_step + 1))
    base_ptr = req_to_next_token_scores + cur_req_idx * req_to_next_token_scores_stride
    tl.store(base_ptr, 1.0)

    offset = tl.arange(0, BLOCK_SIZE)
    store_mask = offset < (max_draft_step + 1)

    scores = tl.load(base_ptr + offset, mask=store_mask, other=0.0)
    # offset 0 是 target sample，本轮恒接受；只有 draft 条件接受概率需要 clamp。
    scores = tl.where(offset == 0, 1.0, scores)
    # Clamp draft scheduling scores before converting them to prefix-survival scores.
    # This makes each request's cumulative acceptance probabilities monotonic,
    # so global top-k selection cannot pick a later draft row without its prefix.
    scores = tl.where((offset != 0) & (scores >= 0.99), 0.99, scores)
    scores = tl.where((offset != 0) & (scores <= 0.01), 0.01, scores)

    cumulative_scores = tl.cumprod(scores, axis=0)

    tl.store(base_ptr + offset, cumulative_scores, mask=store_mask)
    return


def sample_dynamic_mtp_row_mask(
    dynamic_batch_size: int,
    b_req_idx: torch.Tensor,
    req_to_next_token_scores: torch.Tensor,
    max_draft_step: int,
    pre_draft_step: int = None,
) -> torch.Tensor:
    dynamic_batch_size = int(dynamic_batch_size)
    max_draft_step = int(max_draft_step)
    pre_draft_step = max_draft_step if pre_draft_step is None else int(pre_draft_step)
    assert 0 <= pre_draft_step <= max_draft_step
    assert b_req_idx.shape[0] % (max_draft_step + 1) == 0
    assert req_to_next_token_scores.is_cuda
    assert dynamic_batch_size <= b_req_idx.shape[0]
    req_num = len(b_req_idx) // (max_draft_step + 1)
    valid_row_num = req_num * (pre_draft_step + 1)
    assert dynamic_batch_size <= valid_row_num

    # Convert each request's conditional scheduling scores to prefix survival scores.
    _fwd_kernel_cumprod_scores[(req_num,)](
        req_to_next_token_scores=req_to_next_token_scores,
        req_to_next_token_scores_stride=req_to_next_token_scores.stride(0),
        b_req_idx=b_req_idx,
        max_draft_step=max_draft_step,
        BLOCK_SIZE=triton.next_power_of_2(max_draft_step + 1),
        num_warps=1,
        num_stages=1,
    )

    request_ids = b_req_idx[:: max_draft_step + 1].long()
    scores = req_to_next_token_scores.index_select(0, request_ids)[:, : pre_draft_step + 1].flatten()
    compact_ids = torch.topk(scores, k=dynamic_batch_size, sorted=False).indices
    request_offsets = compact_ids // (pre_draft_step + 1)
    step_offsets = compact_ids % (pre_draft_step + 1)
    selected_ids = request_offsets * (max_draft_step + 1) + step_offsets

    selected_row_mask = torch.zeros((len(b_req_idx),), dtype=torch.int32, device=b_req_idx.device)
    selected_row_mask.scatter_(0, selected_ids, 1)
    return selected_row_mask


# 动态 ModelInput 行压缩。
@triton.jit
def _fwd_kernel_compact_dynamic_mtp_model_input(
    input_ids,
    out_input_ids,
    b_req_idx,
    out_b_req_idx,
    b_mtp_index,
    out_b_mtp_index,
    b_seq_len,
    out_b_seq_len,
    b_position_delta,
    out_b_position_delta,
    b_shared_seq_len,
    out_b_shared_seq_len,
    b_shared_radix_node_id,
    out_b_shared_radix_node_id,
    selected_mask,
    selected_dst_pos,
    batch_size,
    HAS_INPUT_IDS: tl.constexpr,
    HAS_B_POSITION_DELTA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    selected_i32 = tl.load(selected_mask + offsets, mask=mask, other=0)
    selected = selected_i32 != 0
    dst_pos = tl.cumsum(selected_i32, axis=0) - 1
    write_mask = mask & selected

    cur_b_req_idx = tl.load(b_req_idx + offsets, mask=mask, other=0)
    cur_b_mtp_index = tl.load(b_mtp_index + offsets, mask=mask, other=0)
    cur_b_seq_len = tl.load(b_seq_len + offsets, mask=mask, other=0)

    tl.store(selected_dst_pos + offsets, dst_pos, mask=mask)
    tl.store(out_b_req_idx + dst_pos, cur_b_req_idx, mask=write_mask)
    tl.store(out_b_mtp_index + dst_pos, cur_b_mtp_index, mask=write_mask)
    tl.store(out_b_seq_len + dst_pos, cur_b_seq_len, mask=write_mask)

    if HAS_INPUT_IDS:
        input_id = tl.load(input_ids + offsets, mask=mask, other=0)
        tl.store(out_input_ids + dst_pos, input_id, mask=write_mask)

    if HAS_B_POSITION_DELTA:
        position_delta = tl.load(b_position_delta + offsets, mask=mask, other=0)
        tl.store(out_b_position_delta + dst_pos, position_delta, mask=write_mask)

    shared_seq_len = tl.load(b_shared_seq_len + offsets, mask=mask, other=0)
    shared_radix_node_id = tl.load(b_shared_radix_node_id + offsets, mask=mask, other=-1)
    tl.store(out_b_shared_seq_len + dst_pos, shared_seq_len, mask=write_mask)
    tl.store(out_b_shared_radix_node_id + dst_pos, shared_radix_node_id, mask=write_mask)

    return


@triton.jit
def _fwd_kernel_pack_selected_rows_2d(
    src,
    src_stride_0,
    src_stride_1,
    dst,
    dst_stride_0,
    dst_stride_1,
    selected_mask,
    selected_dst_pos,
    batch_size,
    hidden_size,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_block_id = tl.program_id(1)
    col_offsets = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_id < batch_size

    selected_val = tl.load(selected_mask + row_id, mask=row_mask, other=0)
    dst_row = tl.load(selected_dst_pos + row_id, mask=row_mask, other=0)
    write_row = row_mask & (selected_val != 0)
    col_mask = col_offsets < hidden_size
    src_ptrs = src + row_id * src_stride_0 + col_offsets * src_stride_1
    dst_ptrs = dst + dst_row * dst_stride_0 + col_offsets * dst_stride_1
    vals = tl.load(src_ptrs, mask=write_row & col_mask, other=0)
    tl.store(dst_ptrs, vals, mask=write_row & col_mask)


def _pack_selected_hidden(
    hidden: torch.Tensor,
    selected_row_mask: torch.Tensor,
    selected_dst_pos: torch.Tensor,
    dynamic_batch_size: int,
):
    assert hidden.is_cuda
    assert hidden.ndim == 2
    assert selected_row_mask.is_cuda
    assert selected_dst_pos.is_cuda
    assert hidden.shape[0] == selected_row_mask.shape[0]

    selected_row_mask = selected_row_mask.to(torch.int32)
    hidden_size = hidden.shape[1]
    dst = torch.empty((dynamic_batch_size, hidden_size), dtype=hidden.dtype, device=hidden.device)
    grid = (hidden.shape[0], triton.cdiv(hidden_size, 128))
    _fwd_kernel_pack_selected_rows_2d[grid](
        src=hidden,
        src_stride_0=hidden.stride(0),
        src_stride_1=hidden.stride(1),
        dst=dst,
        dst_stride_0=dst.stride(0),
        dst_stride_1=dst.stride(1),
        selected_mask=selected_row_mask,
        selected_dst_pos=selected_dst_pos,
        batch_size=hidden.shape[0],
        hidden_size=hidden_size,
        BLOCK_N=128,
        num_warps=4,
        num_stages=1,
    )
    return dst


def _compact_decode_model_input(
    model_input: ModelInput,
    selected_row_mask: torch.Tensor,
    dynamic_batch_size: int,
) -> ModelInput:
    assert not model_input.is_prefill
    assert selected_row_mask.is_cuda
    assert model_input.b_req_idx.is_cuda
    assert model_input.b_mtp_index.is_cuda
    assert model_input.b_seq_len.is_cuda
    assert model_input.b_shared_seq_len.is_cuda
    assert model_input.b_shared_radix_node_id.is_cuda

    # Dynamic scheduling guarantees exactly dynamic_batch_size selected rows.
    selected_row_mask = selected_row_mask.to(torch.int32)
    old_batch_size = model_input.b_req_idx.shape[0]
    selected_dst_pos = torch.empty((old_batch_size,), dtype=torch.int32, device=model_input.b_req_idx.device)

    out_input_ids = None
    if model_input.input_ids is not None:
        assert model_input.input_ids.is_cuda
        out_input_ids = torch.empty(
            (dynamic_batch_size,), dtype=model_input.input_ids.dtype, device=model_input.input_ids.device
        )

    out_b_shared_seq_len = torch.empty(
        (dynamic_batch_size,), dtype=model_input.b_shared_seq_len.dtype, device=model_input.b_shared_seq_len.device
    )
    out_b_shared_radix_node_id = torch.empty(
        (dynamic_batch_size,),
        dtype=model_input.b_shared_radix_node_id.dtype,
        device=model_input.b_shared_radix_node_id.device,
    )

    out_b_req_idx = torch.empty(
        (dynamic_batch_size,), dtype=model_input.b_req_idx.dtype, device=model_input.b_req_idx.device
    )
    out_b_mtp_index = torch.empty(
        (dynamic_batch_size,), dtype=model_input.b_mtp_index.dtype, device=model_input.b_mtp_index.device
    )
    out_b_seq_len = torch.empty(
        (dynamic_batch_size,), dtype=model_input.b_seq_len.dtype, device=model_input.b_seq_len.device
    )
    out_b_position_delta = None
    if model_input.b_position_delta is not None:
        assert model_input.b_position_delta.is_cuda
        out_b_position_delta = torch.empty(
            (dynamic_batch_size,),
            dtype=model_input.b_position_delta.dtype,
            device=model_input.b_position_delta.device,
        )

    dummy_1d = model_input.b_req_idx
    BLOCK_SIZE = triton.next_power_of_2(old_batch_size)
    grid = (1,)
    _fwd_kernel_compact_dynamic_mtp_model_input[grid](
        input_ids=model_input.input_ids if model_input.input_ids is not None else dummy_1d,
        out_input_ids=out_input_ids if out_input_ids is not None else dummy_1d,
        b_req_idx=model_input.b_req_idx,
        out_b_req_idx=out_b_req_idx,
        b_mtp_index=model_input.b_mtp_index,
        out_b_mtp_index=out_b_mtp_index,
        b_seq_len=model_input.b_seq_len,
        out_b_seq_len=out_b_seq_len,
        b_position_delta=model_input.b_position_delta if model_input.b_position_delta is not None else dummy_1d,
        out_b_position_delta=out_b_position_delta if out_b_position_delta is not None else dummy_1d,
        b_shared_seq_len=model_input.b_shared_seq_len,
        out_b_shared_seq_len=out_b_shared_seq_len,
        b_shared_radix_node_id=model_input.b_shared_radix_node_id,
        out_b_shared_radix_node_id=out_b_shared_radix_node_id,
        selected_mask=selected_row_mask,
        selected_dst_pos=selected_dst_pos,
        batch_size=old_batch_size,
        HAS_INPUT_IDS=model_input.input_ids is not None,
        HAS_B_POSITION_DELTA=model_input.b_position_delta is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )

    model_input.input_ids = out_input_ids
    model_input.b_req_idx = out_b_req_idx
    model_input.b_mtp_index = out_b_mtp_index
    model_input.b_seq_len = out_b_seq_len
    model_input.b_position_delta = out_b_position_delta
    model_input.b_shared_seq_len = out_b_shared_seq_len
    model_input.b_shared_radix_node_id = out_b_shared_radix_node_id

    if model_input.mtp_draft_input_hiddens is not None:
        assert model_input.mtp_draft_input_hiddens.is_cuda
        model_input.mtp_draft_input_hiddens = _pack_selected_hidden(
            model_input.mtp_draft_input_hiddens,
            selected_row_mask,
            selected_dst_pos,
            dynamic_batch_size,
        )
    model_input.batch_size = dynamic_batch_size

    return model_input


def prepare_dynamic_mtp_model_input(
    model_input: ModelInput,
    req_num: int,
    dynamic_batch_size: int,
    req_to_next_token_scores: torch.Tensor,
    pre_draft_step: Optional[int] = None,
):
    req_num = int(req_num)
    dynamic_batch_size = int(dynamic_batch_size)
    assert not model_input.is_prefill, "prepare_dynamic_mtp_model_input only supports decode inputs"
    assert req_to_next_token_scores is not None
    assert dynamic_batch_size >= req_num
    assert dynamic_batch_size <= model_input.batch_size
    max_draft_step = MtpManager.get_instance().get_decode_draft_step(is_draft_model=False)
    pre_draft_step = max_draft_step if pre_draft_step is None else int(pre_draft_step)
    assert 0 <= pre_draft_step <= max_draft_step
    assert model_input.batch_size == req_num * (max_draft_step + 1)
    assert dynamic_batch_size <= req_num * (pre_draft_step + 1)

    # All compaction work stays on the current CUDA stream and needs no host sync.
    model_input.to_cuda()
    assert model_input.mem_indexes.shape[0] == dynamic_batch_size

    selected_row_mask = sample_dynamic_mtp_row_mask(
        dynamic_batch_size=dynamic_batch_size,
        b_req_idx=model_input.b_req_idx,
        req_to_next_token_scores=req_to_next_token_scores,
        max_draft_step=max_draft_step,
        pre_draft_step=pre_draft_step,
    )

    model_input = _compact_decode_model_input(
        model_input=model_input,
        selected_row_mask=selected_row_mask,
        dynamic_batch_size=dynamic_batch_size,
    )
    # Decode and draft-cache commit use the compacted b_position_delta, so
    # placeholder multimodal metadata only needs to keep ModelInput shapes
    # consistent.
    if model_input.multimodal_params is not None:
        # Read-only placeholders: avoid rebuilding hundreds of nested Python
        # objects on every compacted decode iteration.
        empty_multimodal_params = {"images": [], "audios": []}
        model_input.multimodal_params = [empty_multimodal_params] * dynamic_batch_size

    model_input.max_q_seq_len = 1
    return model_input, selected_row_mask
