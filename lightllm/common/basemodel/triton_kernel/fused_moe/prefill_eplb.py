import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_eplb_map_kernel(
    topk_ids_ptr,
    logical_to_physical_ptr,
    logical_replica_count_ptr,
    expert_counter_ptr,
    total_assignment_num,
    topk_num: tl.constexpr,
    map_slots: tl.constexpr,
    RECORD_LOAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_assignment_num
    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int64)

    if RECORD_LOAD:
        tl.atomic_add(expert_counter_ptr + logical_id, 1, mask=mask)

    replica_count = tl.load(
        logical_replica_count_ptr + logical_id,
        mask=mask,
        other=1,
    )
    token_index = (offsets // topk_num).to(tl.int64)
    hashed = (token_index * 2654435769) & 0xFFFFFFFF
    replica_index = hashed % replica_count
    physical_id = tl.load(
        logical_to_physical_ptr + logical_id * map_slots + replica_index,
        mask=mask,
        other=-1,
    )
    tl.store(topk_ids_ptr + offsets, physical_id, mask=mask)


@triton.jit
def _logical_to_primary_physical_kernel(
    topk_ids_ptr,
    total_assignment_num,
    experts_per_rank: tl.constexpr,
    redundant_experts_per_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_assignment_num
    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    physical_id = logical_id + (logical_id // experts_per_rank) * redundant_experts_per_rank
    tl.store(topk_ids_ptr + offsets, physical_id, mask=mask)


@torch.no_grad()
def prefill_eplb_map(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_counter: torch.Tensor,
    *,
    record_load: bool,
):
    """Map logical expert IDs to physical replicas and optionally count load."""
    assert topk_ids.is_cuda and topk_ids.is_contiguous() and topk_ids.ndim == 2
    assert logical_to_physical_map.is_cuda
    assert logical_replica_count.is_cuda
    assert expert_counter.is_cuda
    total_assignment_num = topk_ids.numel()
    if total_assignment_num == 0:
        return
    block_size = 256
    _prefill_eplb_map_kernel[(triton.cdiv(total_assignment_num, block_size),)](
        topk_ids,
        logical_to_physical_map,
        logical_replica_count,
        expert_counter,
        total_assignment_num,
        topk_num=topk_ids.shape[1],
        map_slots=logical_to_physical_map.shape[1],
        RECORD_LOAD=record_load,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


@torch.no_grad()
def logical_to_primary_physical(
    topk_ids: torch.Tensor,
    *,
    experts_per_rank: int,
    redundant_experts_per_rank: int,
):
    """Map logical IDs to fixed primary slots, bypassing all EPLB replicas."""
    assert topk_ids.is_cuda and topk_ids.is_contiguous() and topk_ids.ndim == 2
    total_assignment_num = topk_ids.numel()
    if total_assignment_num == 0:
        return
    block_size = 256
    _logical_to_primary_physical_kernel[(triton.cdiv(total_assignment_num, block_size),)](
        topk_ids,
        total_assignment_num,
        experts_per_rank=experts_per_rank,
        redundant_experts_per_rank=redundant_experts_per_rank,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
