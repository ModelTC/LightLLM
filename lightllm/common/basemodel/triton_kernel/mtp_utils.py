"""固定布局与动态布局共同使用的 MTP Triton 算子。"""

from typing import Optional

import triton
import triton.language as tl
import torch

from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size


@triton.jit
def _fwd_kernel_build_mtp_shared_group_markers(
    b_req_idx,
    b_mark_mtp_shared_group,
    batch_size,
    hold_req_id,
    MAX_GROUP_SIZE: tl.constexpr,
    SCAN_BLOCK_SIZE: tl.constexpr,
):
    current_row = tl.program_id(axis=0)
    current_req_idx = tl.load(b_req_idx + current_row)

    # HOLD rows never participate in request grouping. Each one directly forms
    # an independent one-row group, even when multiple HOLD rows are adjacent.
    has_next_row = current_row + 1 < batch_size
    next_req_idx = tl.load(
        b_req_idx + current_row + 1,
        mask=has_next_row,
        other=-1,
    )
    is_hold_row = current_req_idx == hold_req_id
    if is_hold_row:
        tl.store(b_mark_mtp_shared_group + current_row, 1)
        return

    # A normal request only needs work on the final row of its consecutive run.
    is_request_run_end = (~has_next_row) | (next_req_idx != current_req_idx)
    if not is_request_run_end:
        return

    # ModelInput keeps all MTP rows of one real request together and emits each
    # request only once. Count the rows matching the current request, then use
    # that count to recover the request's first row.
    backward_offsets = tl.arange(0, SCAN_BLOCK_SIZE)
    scanned_rows = current_row - backward_offsets
    valid_scanned_rows = scanned_rows >= 0
    scanned_req_idx = tl.load(
        b_req_idx + scanned_rows,
        mask=valid_scanned_rows,
        other=-1,
    )
    belongs_to_current_request = valid_scanned_rows & (scanned_req_idx == current_req_idx)
    request_row_count = tl.sum(belongs_to_current_request, axis=0)
    request_start_row = current_row - request_row_count + 1

    # Split the request run from left to right into groups of MAX_GROUP_SIZE.
    # Each loop iteration finds one group's final row and writes its size.
    for group_start_offset in tl.range(0, request_row_count, MAX_GROUP_SIZE):
        group_size = tl.minimum(MAX_GROUP_SIZE, request_row_count - group_start_offset)
        group_end_row = request_start_row + group_start_offset + group_size - 1
        tl.store(b_mark_mtp_shared_group + group_end_row, group_size)


def build_mtp_shared_group_markers(b_req_idx: torch.Tensor, hold_req_id: int) -> torch.Tensor:
    """Build MTP group markers from consecutive request indexes on GPU.

    Only the final row of each group stores its size. Long request runs are
    split by ``max_batch_shared_group_size``. Every ``hold_req_id`` row forms
    an independent one-row group. For example, with limit 3 and ``H`` denoting
    ``hold_req_id``::

        b_req_idx:               [7, 7, 7, 7, 11, 11, H, H]
        b_mark_mtp_shared_group: [0, 0, 3, 1,  0,  2,  1,  1]
    """

    assert b_req_idx.is_cuda
    batch_size = b_req_idx.shape[0]
    if batch_size == 0:
        return torch.empty((0,), dtype=torch.int32, device=b_req_idx.device)

    max_group_size = int(get_diverse_max_batch_shared_group_size())
    assert max_group_size > 0
    b_mark_mtp_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device=b_req_idx.device)
    _fwd_kernel_build_mtp_shared_group_markers[(batch_size,)](
        b_req_idx=b_req_idx,
        b_mark_mtp_shared_group=b_mark_mtp_shared_group,
        batch_size=batch_size,
        hold_req_id=hold_req_id,
        MAX_GROUP_SIZE=max_group_size,
        SCAN_BLOCK_SIZE=triton.next_power_of_2(batch_size),
        num_warps=8,
        num_stages=1,
    )
    return b_mark_mtp_shared_group


@triton.jit
def _fwd_kernel_mtp_verify(
    req_to_next_token_ids,
    req_to_next_token_ids_stride,
    new_next_token_ids,
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    accepted_index,
    verify_batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    req_nums = tl.num_programs(axis=0)

    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    req_start_end = tl.load(
        b_req_mtp_start_loc + cur_index + 1,
        mask=cur_index + 1 < req_nums,
        other=verify_batch_size,
    )
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
        req_to_next_token_ids: (max_req_num, verify_width)
        b_req_mtp_start_loc: (num_reqs,)
        new_next_token_ids: (verify_batch_size,)
        b_req_idx: (verify_batch_size,)
    Returns:
        mtp_accept_len: (num_reqs,)
        accepted_index: (verify_batch_size,)
        accepted_index: [1, 0, 1, 1, 0], 0 means the token is not accepted, 1 means the token is accepted.
    """
    verify_width = req_to_next_token_ids.shape[1]
    BLOCK_SIZE = 16
    assert verify_width <= BLOCK_SIZE, f"verify_width must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    verify_batch_size = b_req_idx.shape[0]
    assert new_next_token_ids.shape == b_req_idx.shape
    mtp_accept_len = torch.empty((num_reqs,), dtype=torch.int32, device=req_to_next_token_ids.device)
    accepted_index = torch.empty((verify_batch_size,), dtype=torch.int32, device=req_to_next_token_ids.device)

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
        verify_batch_size=verify_batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return mtp_accept_len, accepted_index


@triton.jit
def _fwd_kernel_mtp_scatter_next_token_ids(
    req_to_next_token_ids,
    req_to_next_token_ids_stride,
    target_next_token_ids,
    draft_token_ids,
    draft_token_ids_stride,
    req_to_next_token_scores,
    req_to_next_token_scores_stride,
    schedule_scores,
    schedule_scores_stride,
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    draft_step,
    verify_width,
    HAS_NEXT_TOKEN_SCORES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    cur_index = tl.program_id(0)
    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    accept_len = tl.load(mtp_accept_len + cur_index)
    cur_req_idx = tl.load(b_req_idx + req_start_loc)
    offset = tl.arange(0, BLOCK_SIZE)
    selected_row = req_start_loc + accept_len - 1

    # 第 0 列写入当前请求最终接受位置对应的 target token。
    target_token_id = tl.load(target_next_token_ids + selected_row)
    tl.store(
        req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride,
        target_token_id,
    )

    # 从第 1 列开始写入 draft token；固定宽度 buffer 的未使用列填 1。
    draft_token_id = tl.load(
        draft_token_ids + cur_index * draft_token_ids_stride + offset,
        mask=offset < draft_step,
        other=1,
    )
    tl.store(
        req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + offset + 1,
        draft_token_id,
        mask=offset + 1 < verify_width,
    )

    if HAS_NEXT_TOKEN_SCORES:
        # Target token 必然接受，因此第 0 列分数固定为 1.0。
        tl.store(
            req_to_next_token_scores + cur_req_idx * req_to_next_token_scores_stride,
            1.0,
        )

        draft_scores = tl.load(
            schedule_scores + cur_index * schedule_scores_stride + offset,
            mask=offset < draft_step,
            other=0.0,
        )
        tl.store(
            req_to_next_token_scores + cur_req_idx * req_to_next_token_scores_stride + offset + 1,
            draft_scores,
            mask=offset + 1 < verify_width,
        )
    return


def mtp_scatter_next_token_ids(
    req_to_next_token_ids: torch.Tensor,  # [max_req_num, verify_width]
    b_req_mtp_start_loc: torch.Tensor,  # [req_num]
    target_next_token_ids: torch.Tensor,  # [verify_batch_size]
    draft_token_ids: torch.Tensor,  # [req_num, draft_step]
    b_req_idx: torch.Tensor,  # [verify_batch_size]
    mtp_accept_len: torch.Tensor,  # [req_num]
    req_to_next_token_scores: Optional[torch.Tensor] = None,  # [max_req_num, verify_width]
    schedule_scores: Optional[torch.Tensor] = None,  # [req_num, draft_step]
):
    """将 target token 和 draft proposal 写入请求级 MTP buffer。"""

    verify_width = req_to_next_token_ids.shape[1]
    BLOCK_SIZE = 16
    assert verify_width <= BLOCK_SIZE, f"verify_width must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    draft_step = draft_token_ids.shape[1]
    assert draft_token_ids.shape[0] == num_reqs
    assert draft_step < verify_width
    if req_to_next_token_scores is not None:
        assert schedule_scores is not None
        assert schedule_scores.shape == draft_token_ids.shape

    HAS_NEXT_TOKEN_SCORES = req_to_next_token_scores is not None
    # Triton launch arguments cannot be None; static verification uses an unused placeholder.
    req_to_next_token_scores_arg = (
        req_to_next_token_scores if req_to_next_token_scores is not None else req_to_next_token_ids
    )
    req_to_next_token_scores_stride = (
        req_to_next_token_scores.stride(0) if req_to_next_token_scores is not None else req_to_next_token_ids.stride(0)
    )
    has_schedule_scores = schedule_scores is not None and schedule_scores.numel() > 0
    schedule_scores_arg = schedule_scores if has_schedule_scores else req_to_next_token_scores_arg
    schedule_scores_stride = schedule_scores.stride(0) if has_schedule_scores else 0

    grid = (num_reqs,)
    num_warps = 1
    _fwd_kernel_mtp_scatter_next_token_ids[grid](
        req_to_next_token_ids=req_to_next_token_ids,
        req_to_next_token_ids_stride=req_to_next_token_ids.stride(0),
        target_next_token_ids=target_next_token_ids,
        draft_token_ids=draft_token_ids,
        draft_token_ids_stride=draft_token_ids.stride(0),
        req_to_next_token_scores=req_to_next_token_scores_arg,
        req_to_next_token_scores_stride=req_to_next_token_scores_stride,
        schedule_scores=schedule_scores_arg,
        schedule_scores_stride=schedule_scores_stride,
        mtp_accept_len=mtp_accept_len,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        b_req_idx=b_req_idx,
        draft_step=draft_step,
        verify_width=verify_width,
        HAS_NEXT_TOKEN_SCORES=HAS_NEXT_TOKEN_SCORES,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )


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


@triton.jit
def _fwd_kernel_linear_att_mtp_state_index_update(
    req_to_mtp_state_index,
    b_req_mtp_start_loc,
    b_req_idx,
    b_mtp_index,
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

    cur_mtp_index = tl.load(b_mtp_index + req_offset, mask=offset < req_mtp_num, other=-1)
    cur_accepted = tl.load(accepted_index + req_offset, mask=offset < req_mtp_num, other=0)

    valid_mtp_index = tl.where(cur_accepted == 1, cur_mtp_index, -1)
    max_mtp_index = tl.max(valid_mtp_index, axis=0)

    tl.store(req_to_mtp_state_index + cur_req_idx, max_mtp_index)
    return


def linear_att_mtp_state_index_update(
    req_to_mtp_state_index: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    accepted_index: torch.Tensor,
    verify_width: int,
):
    """
    Update req_to_mtp_state_index with the max b_mtp_index among accepted tokens per request.
    Args:
        req_to_mtp_state_index: (max_req_num + 1,)
        b_req_mtp_start_loc: (num_reqs,)
        b_req_idx: (batch_size,)
        b_mtp_index: (batch_size,)
        accepted_index: (batch_size,), 1 means accepted, 0 means not accepted.
        verify_width: maximum verify width per request, including the target row.
    """
    BLOCK_SIZE = 16
    assert verify_width <= BLOCK_SIZE, f"verify_width must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    req_mtp_all_num = b_req_idx.shape[0]

    assert len(b_req_idx) == len(b_mtp_index) == len(accepted_index)

    grid = (num_reqs,)
    num_warps = 1
    _fwd_kernel_linear_att_mtp_state_index_update[grid](
        req_to_mtp_state_index=req_to_mtp_state_index,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        accepted_index=accepted_index,
        req_mtp_all_num=req_mtp_all_num,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )


def test_mtp_verify():
    req_to_next_token_ids = torch.tensor(
        [[1, 2, -2, -1, -1], [1, 2, 0, -1, -1], [1, 3, 4, 4, 5]], dtype=torch.int64, device="cuda"
    )
    b_req_idx = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int32, device="cuda")
    b_req_mtp_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    new_next_token_ids = torch.tensor([1, 4, 2, 4, 13], dtype=torch.int64, device="cuda")
    draft_token_ids = torch.tensor([[2, 3], [8, 9]], dtype=torch.int64, device="cuda")
    mtp_accept_len, accepted_index = mtp_verify(
        req_to_next_token_ids, b_req_mtp_start_loc, new_next_token_ids, b_req_idx
    )
    mtp_scatter_next_token_ids(
        req_to_next_token_ids,
        b_req_mtp_start_loc,
        new_next_token_ids,
        draft_token_ids,
        b_req_idx,
        mtp_accept_len,
    )
    print(mtp_accept_len)
    print(req_to_next_token_ids)
    print(accepted_index)


def test_gen_b_req_mtp_start_loc():
    b_mtp_index = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int32, device="cuda")
    gt_output = torch.where(b_mtp_index == 0)[0]
    b_req_mtp_start_loc = gen_b_req_mtp_start_loc(b_mtp_index, 2)
    print(b_req_mtp_start_loc, gt_output)


if __name__ == "__main__":
    test_mtp_verify()
    # test_gen_b_req_mtp_start_loc()
