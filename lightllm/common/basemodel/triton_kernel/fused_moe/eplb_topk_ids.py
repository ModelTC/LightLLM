import torch
import triton
import triton.language as tl


@triton.jit
def _replica_index(token_index, logical_id, replica_count):
    token_hash = token_index.to(tl.uint32) * 2654435769
    expert_hash = logical_id.to(tl.uint32) * 2246822519
    return (token_hash + expert_hash) % replica_count.to(tl.uint32)


@triton.jit
def _eplb_repair_topk_ids_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    topk_id_count,
    topk,
    logical_to_physical_map_ptr,
    logical_replica_count_ptr,
    expert_counter_ptr,
    sample_index,
    MAP_SLOTS: tl.constexpr,
    COUNTER_NUM_EXPERTS: tl.constexpr,
    RECORD_LOAD: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < topk_id_count
    logical_ids = tl.load(logical_topk_ids_ptr + offsets, mask=mask, other=0)

    if RECORD_LOAD:
        tl.atomic_add(
            expert_counter_ptr + sample_index * COUNTER_NUM_EXPERTS + logical_ids,
            1,
            mask=mask,
            sem="relaxed",
        )

    if SINGLE_TOKEN:
        replica_indices = tl.zeros((BLOCK_SIZE,), tl.int32)
    else:
        replica_counts = tl.load(logical_replica_count_ptr + logical_ids, mask=mask, other=1)
        token_indices = offsets // topk
        replica_indices = _replica_index(token_indices, logical_ids, replica_counts)

    physical_ids = tl.load(
        logical_to_physical_map_ptr + logical_ids * MAP_SLOTS + replica_indices,
        mask=mask,
        other=-1,
    )
    tl.store(physical_topk_ids_ptr + offsets, physical_ids, mask=mask)


@torch.no_grad()
def eplb_repair_topk_ids(
    logical_topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_counter: torch.Tensor,
    sample_index: int,
    record_load: bool,
) -> torch.Tensor:
    """Map logical top-k IDs to the current EPLB physical expert layout."""
    assert logical_topk_ids.is_contiguous()
    assert logical_topk_ids.ndim == 2
    physical_topk_ids = torch.empty_like(logical_topk_ids)
    if logical_topk_ids.numel() == 0:
        return physical_topk_ids

    block_size = 512
    _eplb_repair_topk_ids_kernel[(triton.cdiv(logical_topk_ids.numel(), block_size),)](
        logical_topk_ids,
        physical_topk_ids,
        logical_topk_ids.numel(),
        logical_topk_ids.shape[1],
        logical_to_physical_map,
        logical_replica_count,
        expert_counter,
        sample_index,
        MAP_SLOTS=logical_to_physical_map.shape[1],
        COUNTER_NUM_EXPERTS=expert_counter.shape[1],
        RECORD_LOAD=record_load,
        SINGLE_TOKEN=logical_topk_ids.shape[0] == 1,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return physical_topk_ids
