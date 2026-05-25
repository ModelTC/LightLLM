import random
from typing import Optional, Tuple
import triton
import triton.language as tl
import torch

from lightllm.common.basemodel.batch_objs import ModelInput


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
    req_to_next_token_probs_arg = req_to_next_token_probs if req_to_next_token_probs is not None else req_to_next_token_ids
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
def _fwd_kernel_sample_dynamic_mtp_steps(
    req_to_next_token_probs,
    req_to_next_token_probs_stride,
    req_indices,
    sampled_steps,
    rand_seed,
    MAX_MTP_STEP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(req_indices + cur_index)

    sampled_step = 1
    prefix_ok = 1

    for step in range(1, BLOCK_SIZE):
        if step < MAX_MTP_STEP:
            cur_prob = tl.load(req_to_next_token_probs + cur_req_idx * req_to_next_token_probs_stride + step)
            cur_rand = tl.rand(rand_seed, cur_index * BLOCK_SIZE + step)
            cur_accept = (cur_prob > cur_rand) & (prefix_ok == 1)
            sampled_step += tl.where(cur_accept, 1, 0)
            prefix_ok = tl.where(cur_accept, 1, 0)

    tl.store(sampled_steps + cur_index, sampled_step)
    return


def sample_dynamic_mtp_req_mask(
    dynamic_batch_size: int,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    req_to_next_token_probs: torch.Tensor,
    req_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert b_req_idx.shape == b_mtp_index.shape
    assert b_req_idx.is_cuda
    assert b_mtp_index.is_cuda
    assert req_to_next_token_probs.is_cuda

    batch_size = b_req_idx.shape[0]
    req_start_mask = b_mtp_index == 0
    assert dynamic_batch_size >= req_num
    assert dynamic_batch_size <= batch_size

    # 这里 req_start_mask 恰好有 req_num 个 1，直接做固定容量 pack，避免布尔索引带来的动态 shape 开销。
    req_indices_gpu = _pack_selected_rows_1d(b_req_idx.to(dtype=torch.int32), req_start_mask.to(torch.int32), req_num)

    max_mtp_step = req_to_next_token_probs.shape[1]
    BLOCK_SIZE = 16
    assert max_mtp_step <= BLOCK_SIZE, f"max_mtp_step must be less than or equal to {BLOCK_SIZE}"

    # 首先调用算子对每个请求进行动态采样，得到每一步需要参与verify的请求个数
    sampled_steps = torch.empty((req_num,), dtype=torch.int32, device="cuda")
    rand_seed = random.randint(0, 2**31 - 1)
    
    grid = (req_num,)
    _fwd_kernel_sample_dynamic_mtp_steps[grid](
        req_to_next_token_probs=req_to_next_token_probs,
        req_to_next_token_probs_stride=req_to_next_token_probs.stride(0),
        req_indices=req_indices_gpu,
        sampled_steps=sampled_steps,
        rand_seed=rand_seed,
        MAX_MTP_STEP=max_mtp_step,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
        num_stages=1,
    )

    # * dynamic_batch_size 必定大于等于 req_num，因此每个请求至少保留一个主行(mtp_index == 0)
    # * 随机采样命中的位置如果过多，就截断到 dynamic_batch_size，但必须保留所有主行
    # * 随机采样命中的位置如果不足，就从未命中的位置里按 batch 顺序补齐，直到恰好填满 dynamic_batch_size
    req_order = torch.cumsum(req_start_mask.to(torch.int32), dim=0) - 1
    row_sampled_steps = sampled_steps[req_order.long()]
    sampled_mask_gpu = (b_mtp_index < row_sampled_steps).to(torch.int32)

    req_start_mask_i32 = req_start_mask.to(torch.int32)
    sampled_extra_mask = sampled_mask_gpu & (1 - req_start_mask_i32)
    extra_budget = dynamic_batch_size - req_num
    extra_budget_tensor = torch.full((), extra_budget, dtype=torch.int32, device=sampled_mask_gpu.device)

    # 先在所有采样命中的 draft rows 中按顺序保留前 extra_budget 个。
    sampled_extra_rank = torch.cumsum(sampled_extra_mask, dim=0)
    kept_sampled_extra_mask = sampled_extra_mask & (sampled_extra_rank <= extra_budget_tensor)

    # 如果采样命中的 draft rows 不够，再从剩余未选中的 rows 中按顺序补齐。
    kept_sampled_extra_total = sampled_extra_rank[-1].clamp_max(extra_budget)
    remaining_extra_budget = extra_budget_tensor - kept_sampled_extra_total
    fill_candidates = (1 - req_start_mask_i32) & (1 - kept_sampled_extra_mask)
    fill_rank = torch.cumsum(fill_candidates, dim=0)
    fill_mask = fill_candidates & (fill_rank <= remaining_extra_budget)

    final_mask_gpu = req_start_mask_i32 | kept_sampled_extra_mask | fill_mask

    return final_mask_gpu, sampled_steps


def _rebuild_trimmed_mtp_b_mark_shared_group_from_b_mtp_index(b_mtp_index: torch.Tensor) -> torch.Tensor:
    assert b_mtp_index.is_cuda
    batch_size = b_mtp_index.shape[0]
    if batch_size == 0:
        return torch.empty((0,), dtype=torch.int32, device=b_mtp_index.device)

    # prepare_decode_inputs 中已经保证了每个请求的 decode rows 是按
    # b_mtp_index = [0, 1, 2, ...] 的前缀顺序展开的；动态 trim 只会保留
    # 每个请求的前缀，因此 compact 之后每个请求块的末尾满足：
    # 1. 是最后一个元素；或
    # 2. 下一个元素的 b_mtp_index 重新回到 0
    group_end_mask = torch.ones((batch_size,), dtype=torch.bool, device=b_mtp_index.device)
    if batch_size > 1:
        group_end_mask[:-1] = b_mtp_index[1:] == 0

    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device=b_mtp_index.device)
    b_mark_shared_group[group_end_mask] = (b_mtp_index[group_end_mask] + 1).to(torch.int32)
    return b_mark_shared_group


@triton.jit
def _fwd_kernel_pack_selected_rows_1d(
    src,
    dst,
    selected_mask,
    selected_dst_pos,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    src_vals = tl.load(src + offsets, mask=mask, other=0)
    selected_vals = tl.load(selected_mask + offsets, mask=mask, other=0)
    dst_pos_vals = tl.load(selected_dst_pos + offsets, mask=mask, other=0)
    write_mask = mask & (selected_vals != 0)
    tl.store(dst + dst_pos_vals, src_vals, mask=write_mask)


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


def _pack_selected_rows_1d(src: Optional[torch.Tensor], selected_mask_gpu: torch.Tensor, dynamic_batch_size: int):
    if src is None:
        return None

    assert src.is_cuda
    assert src.ndim == 1
    assert selected_mask_gpu.is_cuda
    assert src.shape[0] == selected_mask_gpu.shape[0]

    selected_mask_gpu = selected_mask_gpu.to(torch.int32)
    selected_dst_pos = torch.cumsum(selected_mask_gpu, dim=0, dtype=torch.int32) - 1
    dst = torch.empty((dynamic_batch_size,), dtype=src.dtype, device=src.device)
    grid = (triton.cdiv(src.shape[0], 256),)
    _fwd_kernel_pack_selected_rows_1d[grid](
        src=src,
        dst=dst,
        selected_mask=selected_mask_gpu,
        selected_dst_pos=selected_dst_pos,
        batch_size=src.shape[0],
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=1,
    )
    return dst


def _pack_selected_rows_2d(src: Optional[torch.Tensor], selected_mask_gpu: torch.Tensor, dynamic_batch_size: int):
    if src is None:
        return None

    assert src.is_cuda
    assert src.ndim == 2
    assert selected_mask_gpu.is_cuda
    assert src.shape[0] == selected_mask_gpu.shape[0]

    selected_mask_gpu = selected_mask_gpu.to(torch.int32)
    selected_dst_pos = torch.cumsum(selected_mask_gpu, dim=0, dtype=torch.int32) - 1
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
    assert model_input.mem_indexes is not None and model_input.mem_indexes.is_cuda

    # 动态 MTP 采样阶段已经保证 selected_mask_gpu 恰好选出 dynamic_batch_size 个位置，
    # 因此这里可以直接做固定容量 pack，不再需要 nonzero / numel 之类的动态 compact 同步。
    if model_input.input_ids is not None:
        assert model_input.input_ids.is_cuda
        model_input.input_ids = _pack_selected_rows_1d(model_input.input_ids, selected_mask_gpu, dynamic_batch_size)
    model_input.b_req_idx = _pack_selected_rows_1d(model_input.b_req_idx, selected_mask_gpu, dynamic_batch_size)
    model_input.b_mtp_index = _pack_selected_rows_1d(model_input.b_mtp_index, selected_mask_gpu, dynamic_batch_size)
    model_input.b_seq_len = _pack_selected_rows_1d(model_input.b_seq_len, selected_mask_gpu, dynamic_batch_size)

    if model_input.mem_indexes is not None:
        assert model_input.mem_indexes.is_cuda
        model_input.mem_indexes = _pack_selected_rows_1d(model_input.mem_indexes, selected_mask_gpu, dynamic_batch_size)
    if model_input.b_shared_seq_len is not None:
        assert model_input.b_shared_seq_len.is_cuda
        model_input.b_shared_seq_len = _pack_selected_rows_1d(
            model_input.b_shared_seq_len, selected_mask_gpu, dynamic_batch_size
        )
    if model_input.mtp_draft_input_hiddens is not None:
        assert model_input.mtp_draft_input_hiddens.is_cuda
        model_input.mtp_draft_input_hiddens = _pack_selected_rows_2d(
            model_input.mtp_draft_input_hiddens, selected_mask_gpu, dynamic_batch_size
        )
    # ! multimodal_params 是 Python list，而且在语言模型测试中用不到，这里直接硬截取前dynamic_batch_size个元素
    if model_input.multimodal_params is not None:
        model_input.multimodal_params = model_input.multimodal_params[:dynamic_batch_size]

    model_input.b_mark_shared_group = _rebuild_trimmed_mtp_b_mark_shared_group_from_b_mtp_index(
        model_input.b_mtp_index
    )
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

    assert not model_input.is_prefill, "trim_dynamic_mtp_model_input only supports decode inputs"
    
    # ! 在一个CUDA流上面的GPU操作会自动串行化，因此不需要额外同步
    # ! model_input必须在GPU上，才能高效进行trim操作
    model_input.to_cuda()
    # torch.cuda.current_stream().synchronize()

    selected_mask_gpu, _ = sample_dynamic_mtp_req_mask(
        req_num=req_num,
        dynamic_batch_size=dynamic_batch_size,
        b_req_idx=model_input.b_req_idx,
        b_mtp_index=model_input.b_mtp_index,
        req_to_next_token_probs=req_to_next_token_probs,
    )
    model_input = _trim_decode_model_input_inplace(
        model_input=model_input, 
        selected_mask_gpu=selected_mask_gpu,
        dynamic_batch_size=dynamic_batch_size,
    )
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
    selected_mask, sampled_steps = sample_dynamic_mtp_req_mask(
        dynamic_batch_size=dynamic_batch_size,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        req_to_next_token_probs=req_to_next_token_probs,
        req_num=3,
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
    print("sampled_steps per req:")
    print(sampled_steps.cpu())
    print("selected_mask:")
    print(selected_mask.cpu())
    print("selected rows [req_idx, mtp_index]:")
    selected_pos = torch.where(selected_mask == 1)[0]
    print(torch.stack([b_req_idx[selected_pos], b_mtp_index[selected_pos]], dim=-1).cpu())


def test_trim_dynamic_mtp_model_input():
    torch.manual_seed(1234)

    model_input = ModelInput(
        batch_size=12,
        total_token_num=54,
        max_q_seq_len=1,
        max_kv_seq_len=6,
        input_ids=None,
        b_req_idx=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6], dtype=torch.int32, device="cuda"),
        b_shared_seq_len=None,
        b_mark_shared_group=torch.tensor([0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4], dtype=torch.int32, device="cuda"),
        mem_indexes=torch.arange(12, dtype=torch.int32, device="cuda"),
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

    model_input, selected_mask = trim_dynamic_mtp_model_input(
        model_input=model_input,
        dynamic_batch_size=8,
        req_to_next_token_ids=torch.empty((0,), dtype=torch.int64, device="cuda"),
        req_to_next_token_probs=req_to_next_token_probs,
    )

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
    print("b_mark_shared_group:")
    print(model_input.b_mark_shared_group.cpu())
    print("mem_indexes:")
    print(model_input.mem_indexes.cpu())
    print("mem_indexes_cpu:")
    print(model_input.mem_indexes_cpu)


if __name__ == "__main__":
    # test_mtp_verify()
    # test_gen_b_req_mtp_start_loc()
    test_sample_dynamic_mtp_req_mask()
    # test_trim_dynamic_mtp_model_input()
