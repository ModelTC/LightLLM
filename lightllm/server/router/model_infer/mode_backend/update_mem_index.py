import torch
import triton
import triton.language as tl


@triton.jit
def update_eagle_mem_indexes_kernel(
    old_mems_ptr,  # [N]
    new_step_mems_ptr,  # [num_reqs]
    b_req_mtp_start_loc,  # [num_reqs]
    out_mems_ptr,  # [N]
    req_all_num,
    BLOCK_SIZE: tl.constexpr,
):
    cur_req_idx = tl.program_id(0)
    origin_req_num = tl.num_programs(0)

    offs = tl.arange(0, BLOCK_SIZE)
    start_loc = tl.load(b_req_mtp_start_loc + cur_req_idx)
    end_loc = tl.load(b_req_mtp_start_loc + cur_req_idx + 1, mask=cur_req_idx + 1 < origin_req_num, other=req_all_num)

    req_mtp_num = end_loc - start_loc
    old_mems = tl.load(old_mems_ptr + start_loc + offs + 1, mask=offs + 1 < req_mtp_num, other=0)
    tl.store(out_mems_ptr + start_loc + offs, old_mems, mask=offs + 1 < req_mtp_num)
    new_step_mems = tl.load(new_step_mems_ptr + cur_req_idx)
    tl.store(out_mems_ptr + end_loc - 1, new_step_mems)


def update_eagle_mem_indexes_triton(
    old_mem_indexes: torch.Tensor, new_step_mem_indexes: torch.Tensor, b_req_mtp_start_loc: torch.Tensor
):
    """
    old_mem_indexes:      [N] CUDA Tensor
    new_step_mem_indexes: [num_reqs] CUDA Tensor
    """
    out = torch.empty_like(old_mem_indexes)
    BLOCK_SIZE = 32
    original_num_reqs = b_req_mtp_start_loc.shape[0]
    assert original_num_reqs == new_step_mem_indexes.shape[0]
    req_all_num = old_mem_indexes.shape[0]
    grid = (original_num_reqs,)
    update_eagle_mem_indexes_kernel[grid](
        old_mems_ptr=old_mem_indexes,
        new_step_mems_ptr=new_step_mem_indexes,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        out_mems_ptr=out,
        req_all_num=req_all_num,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
