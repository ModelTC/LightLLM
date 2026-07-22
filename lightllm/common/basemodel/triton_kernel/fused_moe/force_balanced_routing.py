from typing import Dict, Tuple

import torch
import triton
import triton.language as tl


_PERMUTATION_SEED = 20260722
_permutation_cache: Dict[Tuple[int, int], torch.Tensor] = {}


@triton.jit
def _force_balanced_routing_kernel(
    topk_ids_ptr,
    permutation_ptr,
    total_assignment_num,
    replace_token_num,
    top_k: tl.constexpr,
    expert_num: tl.constexpr,
    permutation_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token_ids = offsets // top_k
    mask = (offsets < total_assignment_num) & (token_ids < replace_token_num)
    permutation_indices = (offsets + permutation_offset) % expert_num
    balanced_expert_ids = tl.load(permutation_ptr + permutation_indices, mask=mask)
    tl.store(topk_ids_ptr + offsets, balanced_expert_ids, mask=mask)


def _get_expert_permutation(expert_num: int, device: torch.device) -> torch.Tensor:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device_index, expert_num)
    permutation = _permutation_cache.get(key)
    if permutation is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_PERMUTATION_SEED)
        permutation = torch.randperm(expert_num, generator=generator, dtype=torch.int64).to(device)
        _permutation_cache[key] = permutation
    return permutation


@torch.no_grad()
def force_balanced_routing(
    topk_ids: torch.Tensor,
    ratio: float,
    *,
    expert_num: int,
    global_rank: int,
    world_size: int,
) -> torch.Tensor:
    """Replace a fraction of complete token rows with a deterministic balanced expert assignment."""
    assert topk_ids.is_cuda and topk_ids.is_contiguous() and topk_ids.ndim == 2
    assert 0.0 <= ratio <= 1.0
    assert topk_ids.shape[1] <= expert_num
    if ratio == 0.0 or topk_ids.numel() == 0:
        return topk_ids

    replace_token_num = int(topk_ids.shape[0] * ratio)
    if replace_token_num == 0:
        return topk_ids

    permutation = _get_expert_permutation(expert_num, topk_ids.device)
    experts_per_rank = max(1, expert_num // world_size)
    permutation_offset = (global_rank * experts_per_rank) % expert_num
    total_assignment_num = topk_ids.numel()
    block_size = 256
    _force_balanced_routing_kernel[(triton.cdiv(total_assignment_num, block_size),)](
        topk_ids,
        permutation,
        total_assignment_num,
        replace_token_num,
        top_k=topk_ids.shape[1],
        expert_num=expert_num,
        permutation_offset=permutation_offset,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return topk_ids
