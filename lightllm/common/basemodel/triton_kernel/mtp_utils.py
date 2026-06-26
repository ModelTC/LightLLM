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
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    mtp_step,
    BLOCK_SIZE: tl.constexpr,
):

    cur_index = tl.program_id(0)
    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    accept_len = tl.load(mtp_accept_len + cur_index)
    cur_req_idx = tl.load(b_req_idx + req_start_loc)
    offset = tl.arange(0, BLOCK_SIZE)

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
):
    max_mtp_step = req_to_next_token_ids.shape[1]
    BLOCK_SIZE = 16
    assert max_mtp_step <= BLOCK_SIZE, f"max_mtp_step must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    mtp_step = all_next_token_ids.shape[1]
    grid = (num_reqs,)
    num_warps = 1
    _fwd_kernel_mtp_scatter_next_token_ids[grid](
        req_to_next_token_ids=req_to_next_token_ids,
        req_to_next_token_ids_stride=req_to_next_token_ids.stride(0),
        all_next_token_ids=all_next_token_ids,
        all_next_token_ids_stride=all_next_token_ids.stride(0),
        mtp_accept_len=mtp_accept_len,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        b_req_idx=b_req_idx,
        mtp_step=mtp_step,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )


@triton.jit
def _fwd_kernel_scatter_accept_len(
    req_to_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    mtp_accept_len,
):
    cur_index = tl.program_id(0)
    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    cur_req_idx = tl.load(b_req_idx + req_start_loc)
    accept_len = tl.load(mtp_accept_len + cur_index)
    tl.store(req_to_accept_len + cur_req_idx, accept_len)
    return


def scatter_mtp_accept_len(
    req_to_accept_len: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    b_req_idx: torch.Tensor,
    mtp_accept_len: torch.Tensor,
):
    """
    将本步每个真实请求(组首)的 accept 数量写入 GPU 常驻的 req_to_accept_len[req_idx]。
    融合 `req_to_accept_len[b_req_idx[b_req_mtp_start_loc]] = mtp_accept_len` 的 gather+scatter
    为单次 launch、无中间张量。每个 program 处理一个真实请求。
    Args:
        req_to_accept_len: (max_req_num + 1,)
        b_req_mtp_start_loc: (num_reqs,)  每组首行在 batch 中的偏移
        b_req_idx: (batch_size,)          grouped 布局的 req_idx（组首即该请求的 req_idx）
        mtp_accept_len: (num_reqs,)
    """
    num_reqs = mtp_accept_len.shape[0]
    if num_reqs == 0:
        return
    grid = (num_reqs,)
    _fwd_kernel_scatter_accept_len[grid](
        req_to_accept_len=req_to_accept_len,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        b_req_idx=b_req_idx,
        mtp_accept_len=mtp_accept_len,
        num_warps=1,
        num_stages=1,
    )


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


if __name__ == "__main__":
    test_mtp_verify()
