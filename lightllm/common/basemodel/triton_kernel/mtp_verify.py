import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_kernel_mtp_verify(
    req_to_next_token_ids,
    req_to_next_token_ids_stride,
    new_next_token_ids,
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    b_mtp_index,
    accepted_index,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    cur_req_idx = tl.load(b_req_idx + req_start_loc)
    offset = tl.arange(0, BLOCK_SIZE)
    req_offset = req_start_loc + offset
    cur_mtp_index = tl.load(b_mtp_index + req_offset, mask=req_offset < batch_size)

    mask = cur_mtp_index == offset

    cur_next_token_id = tl.load(
        req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + offset + 1, mask=mask, other=-1
    )
    cur_new_next_token_id = tl.load(new_next_token_ids + req_offset, mask=mask, other=-2)

    match_mask = cur_next_token_id == cur_new_next_token_id

    first_false = tl.where(~match_mask, offset, BLOCK_SIZE - 1)
    accept_len = tl.min(first_false)
    tl.store(mtp_accept_len + cur_index, accept_len)
    accpeted_index = tl.where((offset < accept_len + 1), 1, 0)
    tl.store(accepted_index + req_offset, accpeted_index, mask=mask)
    return


def mtp_verify(
    req_to_next_token_ids: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    new_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
):
    """
    This function is used to verify the accept_len.
    Args:
        req_to_next_token_ids: (max_req_num, max_mtp_step)
        b_req_mtp_start_loc: (num_reqs,)
        new_next_token_ids: (batch_size,)
        b_req_idx: (batch_size,)
        b_mtp_index: (batch_size,)
    Returns:
        mtp_accept_len: (num_reqs,)
        accepted_index: (batch_size,)
        accepted_index: [1, 0, 1, 1, 0], 0 means the token is not accepted, 1 means the token is accepted.
    """
    max_mtp_step = req_to_next_token_ids.shape[1]
    BLOCK_SIZE = 16
    assert max_mtp_step <= BLOCK_SIZE, f"max_mtp_step must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    batch_size = b_req_idx.shape[0]
    mtp_accept_len = torch.empty((num_reqs,), dtype=torch.int32, device=req_to_next_token_ids.device)
    accepted_index = torch.empty((batch_size,), dtype=torch.int32, device=req_to_next_token_ids.device)

    grid = (num_reqs,)
    num_warps = 1
    _fwd_kernel_mtp_verify[grid](
        req_to_next_token_ids,
        req_to_next_token_ids.stride(0),
        new_next_token_ids,
        mtp_accept_len,
        b_req_mtp_start_loc,
        b_req_idx,
        b_mtp_index,
        accepted_index,
        batch_size,
        BLOCK_SIZE,
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
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    b_mtp_index,
    mtp_step: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    cur_index = tl.program_id(0)
    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    accept_len = tl.load(mtp_accept_len + cur_index)
    cur_req_idx = tl.load(b_req_idx + req_start_loc)
    offset = tl.arange(0, BLOCK_SIZE)
    req_offset = req_start_loc + offset
    cur_mtp_index = tl.load(b_mtp_index + req_offset, mask=req_offset < batch_size)

    mask = cur_mtp_index == offset
    scatter_next_token_ids = tl.load(
        all_next_token_ids + (req_start_loc + accept_len) * all_next_token_ids_stride + offset,
        mask=offset < mtp_step,
        other=0,
    )
    scatter_next_token_ids = tl.where(mask, scatter_next_token_ids, -1)
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
    b_mtp_index: torch.Tensor,
    mtp_accept_len: torch.Tensor,
):
    max_mtp_step = req_to_next_token_ids.shape[1]
    BLOCK_SIZE = 16
    assert max_mtp_step <= BLOCK_SIZE, f"max_mtp_step must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    batch_size = b_req_idx.shape[0]
    mtp_step = all_next_token_ids.shape[1]
    grid = (num_reqs,)
    num_warps = 1
    _fwd_kernel_mtp_scatter_next_token_ids[grid](
        req_to_next_token_ids,
        req_to_next_token_ids.stride(0),
        all_next_token_ids,
        all_next_token_ids.stride(0),
        mtp_accept_len,
        b_req_mtp_start_loc,
        b_req_idx,
        b_mtp_index,
        mtp_step,
        batch_size,
        BLOCK_SIZE,
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


def test_mtp_verify():
    req_to_next_token_ids = torch.tensor(
        [[1, 2, -2, -1, -1], [1, 2, 0, -1, -1], [1, 3, 4, 4, 5]], dtype=torch.int32, device="cuda"
    )
    b_req_idx = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int32, device="cuda")
    b_mtp_index = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int32, device="cuda")
    b_req_mtp_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    new_next_token_ids = torch.tensor([1, 4, 3, 4, 13], dtype=torch.int32, device="cuda")
    all_next_token_ids = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=torch.int32, device="cuda"
    )
    mtp_accept_len, accepted_index = mtp_verify(
        req_to_next_token_ids, b_req_mtp_start_loc, new_next_token_ids, b_req_idx, b_mtp_index
    )
    mtp_scatter_next_token_ids(
        req_to_next_token_ids, b_req_mtp_start_loc, all_next_token_ids, b_req_idx, b_mtp_index, mtp_accept_len
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
    # test_mtp_verify()
    test_gen_b_req_mtp_start_loc()
