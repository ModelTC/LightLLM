import torch
import triton
import triton.language as tl


@triton.jit
def _build_chained_mtp_decode_input_kernel(
    input_ids,
    draft_token_ids,
    b_req_mtp_start_loc,
    accept_len,
    BLOCK_SIZE: tl.constexpr,
):
    req_index = tl.program_id(0)
    req_start = tl.load(b_req_mtp_start_loc + req_index)
    req_accept_len = tl.load(accept_len + req_index)

    offsets = tl.arange(0, BLOCK_SIZE)
    # tail 行需要保留当前级 draft model 新生成的 token；只覆盖 tail 之前
    # 的已接受行，使它们使用上一层输入中右侧相邻的真实 token。
    mask = offsets < req_accept_len - 1
    shifted_token_ids = tl.load(input_ids + req_start + offsets + 1, mask=mask, other=0)
    tl.store(draft_token_ids + req_start + offsets, shifted_token_ids, mask=mask)


@torch.no_grad()
def build_chained_mtp_decode_input_inplace(
    input_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    accept_len: torch.Tensor,
) -> torch.Tensor:
    """将完整 verify token 串与当前 draft token 级联为下一层输入。

    对请求 ``i``，令 ``start=b_req_mtp_start_loc[i]``、
    ``tail=start+accept_len[i]-1``。本函数原地执行：

    ``draft_token_ids[start:tail] = input_ids[start + 1:tail + 1]``

    ``draft_token_ids[tail]`` 保持不变，因此它仍是当前 draft model 在最后
    接受行生成的新 token。未接受行也保持原 draft 输出，确保所有 token id
    都合法。返回值与传入的 ``draft_token_ids`` 是同一个 tensor。

    例如，某请求已接受的输入为 ``[m0, m1, m2]``，当前级在 tail 行生成
    ``d0``，覆盖后该请求的下一层输入就是 ``[m1, m2, d0]``。
    """

    assert input_ids.is_cuda
    assert draft_token_ids.is_cuda
    assert b_req_mtp_start_loc.is_cuda
    assert accept_len.is_cuda
    assert input_ids.shape == draft_token_ids.shape
    assert b_req_mtp_start_loc.shape == accept_len.shape
    assert input_ids.dtype == draft_token_ids.dtype
    assert input_ids.device == draft_token_ids.device == b_req_mtp_start_loc.device == accept_len.device
    assert input_ids.is_contiguous()
    assert draft_token_ids.is_contiguous()
    assert b_req_mtp_start_loc.is_contiguous()
    assert accept_len.is_contiguous()

    req_num = int(b_req_mtp_start_loc.shape[0])
    if req_num == 0:
        return draft_token_ids

    _build_chained_mtp_decode_input_kernel[(req_num,)](
        input_ids=input_ids,
        draft_token_ids=draft_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        accept_len=accept_len,
        BLOCK_SIZE=16,
        num_warps=1,
        num_stages=1,
    )
    return draft_token_ids
