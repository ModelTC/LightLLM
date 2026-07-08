from typing import Optional
import triton
import triton.language as tl
import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.mtp_utils1 import sample_dynamic_mtp_req_mask
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size, get_env_start_args


@triton.jit
def _fwd_kernel_mtp_verify(
    req_to_next_token_ids,
    req_to_next_token_ids_stride,
    new_next_token_ids,
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    accepted_index,
    req_mtp_all_num,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    req_nums = tl.num_programs(axis=0)

    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    req_start_end = tl.load(b_req_mtp_start_loc + cur_index + 1, mask=cur_index + 1 < req_nums, other=req_mtp_all_num)
    req_mtp_num = req_start_end - req_start_loc
    cur_req_idx = tl.load(b_req_idx + req_start_loc)

    offset = tl.arange(0, BLOCK_SIZE)
    req_offset = req_start_loc + offset

    cur_next_token_id = tl.load(
        req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + offset + 1,
        mask=offset + 1 < req_mtp_num,
        other=-1,
    )
    cur_new_next_token_id = tl.load(new_next_token_ids + req_offset, mask=offset + 1 < req_mtp_num, other=-2)

    match_mask = cur_next_token_id == cur_new_next_token_id
    mismatch_positions = tl.where(match_mask, BLOCK_SIZE, offset)
    first_mismatch_pos = tl.min(mismatch_positions)
    accept_len = first_mismatch_pos + 1
    tl.store(mtp_accept_len + cur_index, accept_len)
    accpeted_index = tl.where((offset < accept_len), 1, 0)
    tl.store(accepted_index + req_offset, accpeted_index, mask=offset < req_mtp_num)
    return


def mtp_verify(
    req_to_next_token_ids: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    new_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
):
    """
    This function is used to verify the accept_len.
    Args:
        req_to_next_token_ids: (max_req_num, max_mtp_step)
        b_req_mtp_start_loc: (num_reqs,)
        new_next_token_ids: (batch_size,)
        b_req_idx: (batch_size,)
    Returns:
        mtp_accept_len: (num_reqs,)
        accepted_index: (batch_size,)
        accepted_index: [1, 0, 1, 1, 0], 0 means the token is not accepted, 1 means the token is accepted.
    """
    max_mtp_step = req_to_next_token_ids.shape[1]
    BLOCK_SIZE = 16
    assert max_mtp_step <= BLOCK_SIZE, f"max_mtp_step must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    req_mtp_all_num = b_req_idx.shape[0]
    mtp_accept_len = torch.empty((num_reqs,), dtype=torch.int32, device=req_to_next_token_ids.device)
    accepted_index = torch.empty((req_mtp_all_num,), dtype=torch.int32, device=req_to_next_token_ids.device)

    grid = (num_reqs,)
    num_warps = 1
    _fwd_kernel_mtp_verify[grid](
        req_to_next_token_ids=req_to_next_token_ids,
        req_to_next_token_ids_stride=req_to_next_token_ids.stride(0),
        new_next_token_ids=new_next_token_ids,
        mtp_accept_len=mtp_accept_len,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        b_req_idx=b_req_idx,
        accepted_index=accepted_index,
        req_mtp_all_num=req_mtp_all_num,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return mtp_accept_len, accepted_index


@triton.jit
def _fwd_kernel_mtp_scatter_next_token_ids(
    req_to_next_token_ids,
    req_to_next_token_ids_stride,
    all_next_token_ids,
    all_next_token_ids_stride,
    req_to_next_token_probs,
    req_to_next_token_probs_stride,
    all_next_token_probs,
    all_next_token_probs_stride,
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    mtp_step,
    HAS_HAS_NEXT_TOKEN_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    accept_len = tl.load(mtp_accept_len + cur_index)
    cur_req_idx = tl.load(b_req_idx + req_start_loc)
    offset = tl.arange(0, BLOCK_SIZE)

    if HAS_HAS_NEXT_TOKEN_PROBS:
        cur_next_token_probs = tl.load(
            all_next_token_probs + (req_start_loc + accept_len - 1) * all_next_token_probs_stride + offset,
            mask=offset < mtp_step,
            other=0.0,
        )
        tl.store(
            req_to_next_token_probs + cur_req_idx * req_to_next_token_probs_stride + offset,
            cur_next_token_probs,
            mask=offset < mtp_step,
        )
    scatter_next_token_ids = tl.load(
        all_next_token_ids + (req_start_loc + accept_len - 1) * all_next_token_ids_stride + offset,
        mask=offset < mtp_step,
        other=0,
    )
    tl.store(
        req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + offset,
        scatter_next_token_ids,
        mask=offset < mtp_step,
    )
    return


def mtp_scatter_next_token_ids(
    req_to_next_token_ids: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    all_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    mtp_accept_len: torch.Tensor,
    req_to_next_token_probs: Optional[torch.Tensor] = None,
    all_next_token_probs: Optional[torch.Tensor] = None,
):
    max_mtp_step = req_to_next_token_ids.shape[1]
    BLOCK_SIZE = 16
    assert max_mtp_step <= BLOCK_SIZE, f"max_mtp_step must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    mtp_step = all_next_token_ids.shape[1]
    if req_to_next_token_probs is not None:
        assert all_next_token_probs is not None
        assert all_next_token_probs.shape == all_next_token_ids.shape

    HAS_HAS_NEXT_TOKEN_PROBS = req_to_next_token_probs is not None
    # Triton launch 参数阶段不能直接传 None；不开动态 MTP 时这里传一个不会被实际使用的 dummy tensor 即可。
    req_to_next_token_probs_arg = (
        req_to_next_token_probs if req_to_next_token_probs is not None else req_to_next_token_ids
    )
    req_to_next_token_probs_stride = (
        req_to_next_token_probs.stride(0) if req_to_next_token_probs is not None else req_to_next_token_ids.stride(0)
    )
    all_next_token_probs_arg = all_next_token_probs if all_next_token_probs is not None else all_next_token_ids
    all_next_token_probs_stride = (
        all_next_token_probs.stride(0) if all_next_token_probs is not None else all_next_token_ids.stride(0)
    )

    grid = (num_reqs,)
    num_warps = 1
    _fwd_kernel_mtp_scatter_next_token_ids[grid](
        req_to_next_token_ids=req_to_next_token_ids,
        req_to_next_token_ids_stride=req_to_next_token_ids.stride(0),
        all_next_token_ids=all_next_token_ids,
        all_next_token_ids_stride=all_next_token_ids.stride(0),
        req_to_next_token_probs=req_to_next_token_probs_arg,
        req_to_next_token_probs_stride=req_to_next_token_probs_stride,
        all_next_token_probs=all_next_token_probs_arg,
        all_next_token_probs_stride=all_next_token_probs_stride,
        mtp_accept_len=mtp_accept_len,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        b_req_idx=b_req_idx,
        mtp_step=mtp_step,
        HAS_HAS_NEXT_TOKEN_PROBS=HAS_HAS_NEXT_TOKEN_PROBS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )


@triton.jit
def _fwd_kernel_trim_dynamic_mtp_model_input(
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
    mem_indexes,
    out_mem_indexes,
    b_shared_seq_len,
    out_b_shared_seq_len,
    selected_mask,
    selected_dst_pos,
    batch_size,
    HAS_INPUT_IDS: tl.constexpr,
    HAS_B_POSITION_DELTA: tl.constexpr,
    HAS_MEM_INDEXES: tl.constexpr,
    HAS_B_SHARED_SEQ_LEN: tl.constexpr,
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

    if HAS_MEM_INDEXES:
        mem_index = tl.load(mem_indexes + offsets, mask=mask, other=0)
        tl.store(out_mem_indexes + dst_pos, mem_index, mask=write_mask)

    if HAS_B_SHARED_SEQ_LEN:
        shared_seq_len = tl.load(b_shared_seq_len + offsets, mask=mask, other=0)
        tl.store(out_b_shared_seq_len + dst_pos, shared_seq_len, mask=write_mask)

    return


@triton.jit
def _fwd_kernel_rebuild_trimmed_mtp_b_mark_shared_group(
    b_req_idx,
    out_b_mark_shared_group,
    batch_size,
    max_batch_shared_group_size: tl.constexpr,
    MAX_RUN_SCAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    cur_req_idx = tl.load(b_req_idx + offsets, mask=mask, other=-1)

    prev_same_count = tl.full((BLOCK_SIZE,), 0, tl.int32)
    for scan_offset in tl.static_range(1, MAX_RUN_SCAN + 1):
        prev_offsets = offsets - scan_offset
        prev_mask = mask & (prev_offsets >= 0)
        prev_req_idx = tl.load(b_req_idx + prev_offsets, mask=prev_mask, other=-2)
        prev_same_count += tl.where(prev_mask & (prev_req_idx == cur_req_idx), 1, 0)

    next_offsets = offsets + 1
    next_req_idx = tl.load(b_req_idx + next_offsets, mask=next_offsets < batch_size, other=-2)
    group_pos = prev_same_count % max_batch_shared_group_size
    is_group_end = mask & (
        (next_offsets == batch_size) | (next_req_idx != cur_req_idx) | (group_pos == max_batch_shared_group_size - 1)
    )
    mark_value = tl.where(is_group_end, group_pos + 1, 0)
    tl.store(out_b_mark_shared_group + offsets, mark_value, mask=mask)
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


def _pack_selected_rows_2d(
    src: Optional[torch.Tensor],
    selected_mask_gpu: torch.Tensor,
    selected_dst_pos: torch.Tensor,
    dynamic_batch_size: int,
):
    if src is None:
        return None

    assert src.is_cuda
    assert src.ndim == 2
    assert selected_mask_gpu.is_cuda
    assert selected_dst_pos.is_cuda
    assert src.shape[0] == selected_mask_gpu.shape[0]

    selected_mask_gpu = selected_mask_gpu.to(torch.int32)
    hidden_size = src.shape[1]
    dst = torch.empty((dynamic_batch_size, hidden_size), dtype=src.dtype, device=src.device)
    grid = (src.shape[0], triton.cdiv(hidden_size, 128))
    _fwd_kernel_pack_selected_rows_2d[grid](
        src=src,
        src_stride_0=src.stride(0),
        src_stride_1=src.stride(1),
        dst=dst,
        dst_stride_0=dst.stride(0),
        dst_stride_1=dst.stride(1),
        selected_mask=selected_mask_gpu,
        selected_dst_pos=selected_dst_pos,
        batch_size=src.shape[0],
        hidden_size=hidden_size,
        BLOCK_N=128,
        num_warps=4,
        num_stages=1,
    )
    return dst


def _rebuild_trimmed_mtp_b_mark_shared_group_from_b_req_idx(b_req_idx: torch.Tensor) -> torch.Tensor:
    assert b_req_idx.is_cuda
    batch_size = b_req_idx.shape[0]
    max_batch_shared_group_size = int(get_diverse_max_batch_shared_group_size())
    assert max_batch_shared_group_size > 0
    if batch_size == 0:
        return torch.empty((0,), dtype=torch.int32, device=b_req_idx.device)

    b_mark_shared_group = torch.empty((batch_size,), dtype=torch.int32, device=b_req_idx.device)
    BLOCK_SIZE = triton.next_power_of_2(batch_size)
    _fwd_kernel_rebuild_trimmed_mtp_b_mark_shared_group[(1,)](
        b_req_idx=b_req_idx,
        out_b_mark_shared_group=b_mark_shared_group,
        batch_size=batch_size,
        max_batch_shared_group_size=max_batch_shared_group_size,
        MAX_RUN_SCAN=16,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    return b_mark_shared_group


def _trim_decode_model_input_inplace(
    model_input: ModelInput,
    selected_mask_gpu: torch.Tensor,
    dynamic_batch_size: int,
) -> ModelInput:
    assert not model_input.is_prefill
    assert selected_mask_gpu.is_cuda
    assert model_input.b_req_idx.is_cuda
    assert model_input.b_mtp_index.is_cuda
    assert model_input.b_seq_len.is_cuda

    # 动态 MTP 采样阶段已经保证 selected_mask_gpu 恰好选出 dynamic_batch_size 个位置。
    selected_mask_gpu = selected_mask_gpu.to(torch.int32)
    old_batch_size = model_input.b_req_idx.shape[0]
    selected_dst_pos = torch.empty((old_batch_size,), dtype=torch.int32, device=model_input.b_req_idx.device)

    out_input_ids = None
    if model_input.input_ids is not None:
        assert model_input.input_ids.is_cuda
        out_input_ids = torch.empty(
            (dynamic_batch_size,), dtype=model_input.input_ids.dtype, device=model_input.input_ids.device
        )

    out_b_shared_seq_len = None
    if model_input.b_shared_seq_len is not None:
        assert model_input.b_shared_seq_len.is_cuda
        out_b_shared_seq_len = torch.empty(
            (dynamic_batch_size,), dtype=model_input.b_shared_seq_len.dtype, device=model_input.b_shared_seq_len.device
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

    out_mem_indexes = None
    if model_input.mem_indexes is not None:
        assert model_input.mem_indexes.is_cuda
        out_mem_indexes = torch.empty(
            (dynamic_batch_size,),
            dtype=model_input.mem_indexes.dtype,
            device=model_input.mem_indexes.device,
        )

    dummy_1d = model_input.b_req_idx
    BLOCK_SIZE = triton.next_power_of_2(old_batch_size)
    grid = (1,)
    _fwd_kernel_trim_dynamic_mtp_model_input[grid](
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
        mem_indexes=model_input.mem_indexes if model_input.mem_indexes is not None else dummy_1d,
        out_mem_indexes=out_mem_indexes if out_mem_indexes is not None else dummy_1d,
        b_shared_seq_len=model_input.b_shared_seq_len if model_input.b_shared_seq_len is not None else dummy_1d,
        out_b_shared_seq_len=out_b_shared_seq_len if out_b_shared_seq_len is not None else dummy_1d,
        selected_mask=selected_mask_gpu,
        selected_dst_pos=selected_dst_pos,
        batch_size=old_batch_size,
        HAS_INPUT_IDS=model_input.input_ids is not None,
        HAS_B_POSITION_DELTA=model_input.b_position_delta is not None,
        HAS_MEM_INDEXES=model_input.mem_indexes is not None,
        HAS_B_SHARED_SEQ_LEN=model_input.b_shared_seq_len is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )

    model_input.input_ids = out_input_ids
    model_input.b_req_idx = out_b_req_idx
    model_input.b_mtp_index = out_b_mtp_index
    model_input.b_seq_len = out_b_seq_len
    model_input.b_position_delta = out_b_position_delta
    model_input.mem_indexes = out_mem_indexes
    model_input.b_shared_seq_len = out_b_shared_seq_len
    model_input.b_mark_shared_group = _rebuild_trimmed_mtp_b_mark_shared_group_from_b_req_idx(out_b_req_idx)

    if model_input.input_ids is not None:
        assert model_input.input_ids.shape[0] == dynamic_batch_size
    if model_input.mtp_draft_input_hiddens is not None:
        assert model_input.mtp_draft_input_hiddens.is_cuda
        model_input.mtp_draft_input_hiddens = _pack_selected_rows_2d(
            model_input.mtp_draft_input_hiddens,
            selected_mask_gpu,
            selected_dst_pos,
            dynamic_batch_size,
        )
    if model_input.b_position_delta is not None:
        assert model_input.b_position_delta.shape[0] == dynamic_batch_size
    if model_input.mem_indexes is not None:
        assert model_input.mem_indexes.shape[0] == dynamic_batch_size
    model_input.batch_size = dynamic_batch_size

    return model_input


def prepare_dynamic_mtp_model_input(
    model_input: ModelInput,
    req_num: int,
    dynamic_batch_size: int,
    req_to_next_token_ids: torch.Tensor,
    req_to_next_token_probs: Optional[torch.Tensor] = None,
):
    if req_to_next_token_probs is None:
        selected_mask = torch.ones((model_input.batch_size,), dtype=torch.int32, device="cuda")
        return model_input, selected_mask

    req_num = int(req_num)
    dynamic_batch_size = int(dynamic_batch_size)
    assert not model_input.is_prefill, "trim_dynamic_mtp_model_input only supports decode inputs"
    assert dynamic_batch_size >= req_num
    assert dynamic_batch_size <= model_input.batch_size
    mtp_step = int(get_env_start_args().mtp_step)
    assert model_input.batch_size == req_num * (mtp_step + 1)

    # ! 在一个CUDA流上面的GPU操作会自动串行化，因此不需要额外同步
    # ! model_input必须在GPU上，才能高效进行trim操作
    model_input.to_cuda()

    selected_mask_gpu = sample_dynamic_mtp_req_mask(
        dynamic_batch_size=dynamic_batch_size,
        b_req_idx=model_input.b_req_idx,
        req_to_next_token_probs=req_to_next_token_probs,
        mtp_step=mtp_step,
    )

    selected_mask_cpu = selected_mask_gpu.detach().cpu().to(dtype=torch.bool)
    assert int(selected_mask_cpu.sum().item()) == dynamic_batch_size

    if model_input.mem_indexes_cpu is not None and len(model_input.mem_indexes_cpu) > 0:
        release_mem_indexes_cpu = model_input.mem_indexes_cpu[~selected_mask_cpu]
        if len(release_mem_indexes_cpu) > 0:
            g_infer_context.req_manager.mem_manager.free(release_mem_indexes_cpu)
        model_input.mem_indexes_cpu = model_input.mem_indexes_cpu[selected_mask_cpu]

    if model_input.multimodal_params is not None:
        model_input.multimodal_params = [
            params for params, selected in zip(model_input.multimodal_params, selected_mask_cpu.tolist()) if selected
        ]

    model_input = _trim_decode_model_input_inplace(
        model_input=model_input,
        selected_mask_gpu=selected_mask_gpu,
        dynamic_batch_size=dynamic_batch_size,
    )

    # ! 现在这些值没有实际作用，同时修改这些值还会导致阻塞操作，因此暂时不修改这些值了
    # model_input.total_token_num = int(model_input.b_seq_len.sum().item())
    # model_input.max_kv_seq_len = int(model_input.b_seq_len.max().item())
    model_input.max_q_seq_len = 1
    return model_input, selected_mask_gpu


@triton.jit
def _fwd_kernel_gen_b_req_mtp_start_loc(
    b_mtp_index,
    b_req_mtp_start_loc,
    num_reqs: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_SIZE)
    cur_mtp_index = tl.load(b_mtp_index + offset, mask=offset < batch_size, other=-1)
    non_zero_mask = tl.where(cur_mtp_index == 0, 1, 0)  # 1 0 1 0 0
    output_offset = tl.cumsum(non_zero_mask) - 1
    tl.store(b_req_mtp_start_loc + output_offset, offset, mask=non_zero_mask == 1)
    return


def gen_b_req_mtp_start_loc(b_mtp_index: torch.Tensor, num_reqs: int):
    b_req_mtp_start_loc = torch.empty((num_reqs,), dtype=torch.int32, device=b_mtp_index.device)
    BLOCK_SIZE = triton.next_power_of_2(b_mtp_index.shape[0])
    batch_size = b_mtp_index.shape[0]
    grid = (1,)
    _fwd_kernel_gen_b_req_mtp_start_loc[grid](
        b_mtp_index=b_mtp_index,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        num_reqs=num_reqs,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return b_req_mtp_start_loc


def test_mtp_verify():
    req_to_next_token_ids = torch.tensor(
        [[1, 2, -2, -1, -1], [1, 2, 0, -1, -1], [1, 3, 4, 4, 5]], dtype=torch.int32, device="cuda"
    )
    b_req_idx = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int32, device="cuda")
    b_req_mtp_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    new_next_token_ids = torch.tensor([1, 4, 2, 4, 13], dtype=torch.int32, device="cuda")
    all_next_token_ids = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=torch.int32, device="cuda"
    )
    mtp_accept_len, accepted_index = mtp_verify(
        req_to_next_token_ids, b_req_mtp_start_loc, new_next_token_ids, b_req_idx
    )
    mtp_scatter_next_token_ids(
        req_to_next_token_ids, b_req_mtp_start_loc, all_next_token_ids, b_req_idx, mtp_accept_len
    )
    print(mtp_accept_len)
    print(req_to_next_token_ids)
    print(accepted_index)


def test_gen_b_req_mtp_start_loc():
    b_mtp_index = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int32, device="cuda")
    gt_output = torch.where(b_mtp_index == 0)[0]
    b_req_mtp_start_loc = gen_b_req_mtp_start_loc(b_mtp_index, 2)
    print(b_req_mtp_start_loc, gt_output)


def test_sample_dynamic_mtp_req_mask():
    torch.manual_seed(1234)

    req_to_next_token_ids = torch.tensor(
        [
            [100, 101, 102, 103, 0, 0],
            [200, 201, 202, 203, 0, 0],
            [300, 301, 302, 303, 0, 0],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    req_to_next_token_probs = torch.tensor(
        [
            [1.0, 0.95, 0.90, 0.10, 0.0, 0.0],
            [1.0, 0.20, 0.80, 0.80, 0.0, 0.0],
            [1.0, 0.99, 0.99, 0.99, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    # 3 个请求，每个请求展开成 4 个 decode row
    b_req_idx = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        dtype=torch.int32,
        device="cuda",
    )
    b_mtp_index = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        dtype=torch.int32,
        device="cuda",
    )

    dynamic_batch_size = 8
    selected_mask = sample_dynamic_mtp_req_mask(
        dynamic_batch_size=dynamic_batch_size,
        b_req_idx=b_req_idx,
        req_to_next_token_probs=req_to_next_token_probs,
        mtp_step=3,
    )

    print("==== test_sample_dynamic_mtp_req_mask ====")
    print("req_to_next_token_ids:")
    print(req_to_next_token_ids.cpu())
    print("req_to_next_token_probs:")
    print(req_to_next_token_probs.cpu())
    print("b_req_idx:")
    print(b_req_idx.cpu())
    print("b_mtp_index:")
    print(b_mtp_index.cpu())
    print("selected_mask:")
    print(selected_mask.cpu())
    print("selected rows [req_idx, mtp_index]:")
    selected_pos = torch.where(selected_mask == 1)[0]
    print(torch.stack([b_req_idx[selected_pos], b_mtp_index[selected_pos]], dim=-1).cpu())


def test_trim_dynamic_mtp_model_input():
    torch.manual_seed(1234)
    origin_b_position_delta = torch.arange(12, dtype=torch.int32, device="cuda")
    origin_mem_indexes = torch.arange(12, dtype=torch.int32, device="cuda")

    model_input = ModelInput(
        batch_size=12,
        total_token_num=54,
        max_q_seq_len=1,
        max_kv_seq_len=6,
        input_ids=None,
        b_req_idx=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6], dtype=torch.int32, device="cuda"),
        b_position_delta=origin_b_position_delta,
        b_shared_seq_len=None,
        b_mark_shared_group=torch.tensor([0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4], dtype=torch.int32, device="cuda"),
        mem_indexes=origin_mem_indexes,
        mem_indexes_cpu=torch.arange(12, dtype=torch.int32, device="cpu"),
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []} for _ in range(12)],
    )
    req_to_next_token_probs = torch.tensor(
        [
            [1.0, 0.95, 0.90, 0.10, 0.0, 0.0],
            [1.0, 0.20, 0.80, 0.80, 0.0, 0.0],
            [1.0, 0.99, 0.99, 0.99, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    selected_mask = sample_dynamic_mtp_req_mask(
        dynamic_batch_size=8,
        b_req_idx=model_input.b_req_idx,
        req_to_next_token_probs=req_to_next_token_probs,
        mtp_step=3,
    )
    model_input = _trim_decode_model_input_inplace(
        model_input=model_input,
        selected_mask_gpu=selected_mask,
        dynamic_batch_size=8,
    )
    selected_pos = torch.where(selected_mask == 1)[0]
    assert torch.equal(model_input.b_position_delta, origin_b_position_delta[selected_pos])
    assert torch.equal(model_input.mem_indexes, origin_mem_indexes[selected_pos])

    print("==== test_trim_dynamic_mtp_model_input ====")
    print("selected_mask:")
    print(selected_mask.cpu())
    print("batch_size:", model_input.batch_size)
    print("total_token_num:", model_input.total_token_num)
    print("max_kv_seq_len:", model_input.max_kv_seq_len)
    print("b_req_idx:")
    print(model_input.b_req_idx.cpu())
    print("b_mtp_index:")
    print(model_input.b_mtp_index.cpu())
    print("b_seq_len:")
    print(model_input.b_seq_len.cpu())
    print("b_position_delta:")
    print(model_input.b_position_delta.cpu())
    print("b_mark_shared_group:")
    print(model_input.b_mark_shared_group.cpu())
    print("mem_indexes:")
    print(model_input.mem_indexes.cpu())
    print("mem_indexes_cpu:")
    print(model_input.mem_indexes_cpu)


if __name__ == "__main__":
    # test_mtp_verify()
    # test_gen_b_req_mtp_start_loc()
    test_trim_dynamic_mtp_model_input()
    # test_trim_dynamic_mtp_model_input()
