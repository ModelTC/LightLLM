import torch
import triton
import triton.language as tl


@triton.jit
def _redundancy_topk_ids_repair_kernel(
    topk_ids_ptr,
    topk_total_num,
    ep_expert_num,
    redundancy_expert_num,
    global_rank,
    redundancy_expert_ids_ptr,
    expert_counter_ptr,
    BLOCK_SIZE: tl.constexpr,
    ENABLE_COUNTER: tl.constexpr,
):
    block_index = tl.program_id(0)
    offs_d = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_d < topk_total_num
    current_topk_ids = tl.load(topk_ids_ptr + offs_d, mask=mask, other=0)

    if ENABLE_COUNTER:
        tl.atomic_add(expert_counter_ptr + current_topk_ids, 1, mask=mask)
    
    # Remap original expert IDs to a new space that accounts for redundant expert slots.
    new_current_topk_ids = (current_topk_ids // ep_expert_num) * redundancy_expert_num + current_topk_ids

    for i in tl.range(0, redundancy_expert_num, step=1, num_stages=3):
        cur_redundancy_expert_id = tl.load(redundancy_expert_ids_ptr + i)
        cur_redundancy_expert_id = (
            cur_redundancy_expert_id // ep_expert_num
        ) * redundancy_expert_num + cur_redundancy_expert_id
        new_current_topk_ids = tl.where(
            new_current_topk_ids == cur_redundancy_expert_id,
            (ep_expert_num + redundancy_expert_num) * (global_rank) + ep_expert_num + i,
            new_current_topk_ids,
        )

    tl.store(topk_ids_ptr + offs_d, new_current_topk_ids, mask=mask)
    return


@torch.no_grad()
def redundancy_topk_ids_repair(
    topk_ids: torch.Tensor,
    redundancy_expert_ids: torch.Tensor,
    ep_expert_num: int,
    global_rank: int,
    expert_counter: torch.Tensor = None,
    enable_counter: bool = False,
):
    assert topk_ids.is_contiguous()
    assert len(topk_ids.shape) == 2
    assert redundancy_expert_ids is not None
    redundancy_expert_num = redundancy_expert_ids.shape[0]
    BLOCK_SIZE = 512
    grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
    num_warps = 4

    _redundancy_topk_ids_repair_kernel[grid](
        topk_ids_ptr=topk_ids,
        topk_total_num=topk_ids.numel(),
        ep_expert_num=ep_expert_num,
        redundancy_expert_num=redundancy_expert_num,
        global_rank=global_rank,
        redundancy_expert_ids_ptr=redundancy_expert_ids,
        expert_counter_ptr=expert_counter,
        BLOCK_SIZE=BLOCK_SIZE,
        ENABLE_COUNTER=enable_counter,
        num_warps=num_warps,
        num_stages=3,
    )
    return


@triton.jit
def _expert_id_counter_kernel(
    topk_ids_ptr,
    topk_total_num,
    expert_counter_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block_index = tl.program_id(0)
    offs_d = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_d < topk_total_num
    current_topk_ids = tl.load(topk_ids_ptr + offs_d, mask=mask, other=0)
    tl.atomic_add(expert_counter_ptr + current_topk_ids, 1, mask=mask)
    return


@torch.no_grad()
def expert_id_counter(
    topk_ids: torch.Tensor,
    expert_counter: torch.Tensor,
):
    assert topk_ids.is_contiguous()
    assert len(topk_ids.shape) == 2
    BLOCK_SIZE = 512
    grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
    num_warps = 4

    _expert_id_counter_kernel[grid](
        topk_ids_ptr=topk_ids,
        topk_total_num=topk_ids.numel(),
        expert_counter_ptr=expert_counter,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return
