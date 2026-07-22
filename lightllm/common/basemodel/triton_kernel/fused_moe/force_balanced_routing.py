import torch
import triton
import triton.language as tl


@triton.jit
def _force_balanced_routing_kernel(
    topk_ids_ptr,
    replace_token_num,
    global_rank: tl.constexpr,
    world_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    assignment_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = assignment_offsets < replace_token_num * world_size
    token_offsets = assignment_offsets // world_size
    destination_ranks = assignment_offsets % world_size
    local_experts = (token_offsets + global_rank) % experts_per_rank
    expert_ids = destination_ranks * experts_per_rank + local_experts
    tl.store(topk_ids_ptr + assignment_offsets, expert_ids, mask=mask)


def force_balanced_routing(
    topk_ids: torch.Tensor,
    ratio: float,
    *,
    expert_num: int,
    global_rank: int,
    world_size: int,
) -> None:
    """Replace a local token prefix with independently rank-balanced rows."""
    assert topk_ids.is_cuda
    assert topk_ids.is_contiguous()
    assert topk_ids.ndim == 2
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be between 0.0 and 1.0, got {ratio}")
    assert world_size > 0
    assert 0 <= global_rank < world_size
    assert expert_num > 0
    assert expert_num % world_size == 0
    if topk_ids.shape[1] != world_size:
        raise ValueError("force-balanced routing requires top_k == world_size")
    if ratio == 0.0 or topk_ids.numel() == 0:
        return

    token_num, top_k = topk_ids.shape
    replace_token_num = int((global_rank + 1) * token_num * ratio) - int(global_rank * token_num * ratio)
    if replace_token_num == 0:
        return

    block_size = 256
    _force_balanced_routing_kernel[(triton.cdiv(replace_token_num * top_k, block_size),)](
        topk_ids,
        replace_token_num,
        world_size=world_size,
        experts_per_rank=expert_num // world_size,
        global_rank=global_rank,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
