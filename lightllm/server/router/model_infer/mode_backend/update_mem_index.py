import torch
import triton
import triton.language as tl

import torch
import triton
import triton.language as tl

@triton.jit
def update_eagle_mem_indexes_kernel(
    old_ptr,            # [N]
    b_mtp_ptr,          # [N]
    req_ids_ptr,        # [N], 每个位置所属第几个主请求
    new_step_ptr,       # [num_reqs]
    out_ptr,            # [N]
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # 当前/下一个 mtp_index
    cur_mtp = tl.load(b_mtp_ptr + offs, mask=mask, other=0)
    next_offs = offs + 1
    next_in_range = next_offs < N
    next_mtp = tl.load(b_mtp_ptr + next_offs, mask=next_in_range, other=0)

    # 组尾判断：越界 或 下一个是新组起点
    is_last = (~next_in_range) | (next_mtp == 0)

    # 非组尾：左移 old[i+1]
    shifted = tl.load(old_ptr + next_offs, mask=next_in_range, other=0)

    # 组尾：new_step[req_id]
    req_id = tl.load(req_ids_ptr + offs, mask=mask, other=0)
    new_val = tl.load(new_step_ptr + req_id, mask=mask, other=0)

    out = tl.where(is_last, new_val, shifted)
    tl.store(out_ptr + offs, out, mask=mask)
    

def update_eagle_mem_indexes_triton(old_indexes, new_step_indexes, b_mtp_index):
    """
    old_indexes:      [N] CUDA Tensor
    new_step_indexes: [num_reqs] CUDA Tensor
    b_mtp_index:      [N] CUDA Tensor, 如 [0,1,0,1,2,0,1,2,3]
    """
    # 固定 shape 的 GPU 计算，不用 nonzero/where 动态输出
    is_start = (b_mtp_index == 0).to(torch.int32)          # [N]
    req_ids = torch.cumsum(is_start, dim=0) - 1            # [N], int32/int64 均可
    req_ids = req_ids.to(torch.int32)
    out = torch.empty_like(old_indexes)
    N = old_indexes.numel()
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    update_eagle_mem_indexes_kernel[grid](
        old_indexes,
        b_mtp_index,
        req_ids,
        new_step_indexes,
        out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out