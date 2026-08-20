import torch
import triton
import triton.language as tl


@triton.jit
def eplb_replica_index(token_index, logical_id, replica_count):
    """Choose a replica with independent phases for a token's top-k experts."""
    token_hash = token_index.to(tl.uint32) * 2654435769
    expert_hash = logical_id.to(tl.uint32) * 2246822519
    return (token_hash + expert_hash) % replica_count.to(tl.uint32)


@triton.jit
def _eplb_push_copy_kernel(
    src_ptrs_ptr,
    dst_ptrs_ptr,
    bytes_per_descriptor,
    BLOCK_SIZE: tl.constexpr,
    ITEMS_PER_PROGRAM: tl.constexpr,
):
    descriptor_index = tl.program_id(1)
    offsets = tl.program_id(0) * (BLOCK_SIZE * ITEMS_PER_PROGRAM) + tl.arange(0, BLOCK_SIZE)
    src_ptr = tl.load(src_ptrs_ptr + descriptor_index).to(tl.pointer_type(tl.uint64))
    dst_ptr = tl.load(dst_ptrs_ptr + descriptor_index).to(tl.pointer_type(tl.uint64))
    word_count = bytes_per_descriptor // 8
    for item_index in tl.static_range(0, ITEMS_PER_PROGRAM):
        word_offsets = offsets + item_index * BLOCK_SIZE
        mask = word_offsets < word_count
        values = tl.load(src_ptr + word_offsets, mask=mask, cache_modifier=".cg")
        tl.store(dst_ptr + word_offsets, values, mask=mask, cache_modifier=".cs")


@torch.no_grad()
def eplb_push_copy(src_ptrs: torch.Tensor, dst_ptrs: torch.Tensor, bytes_per_descriptor: int) -> None:
    """Copy 16-byte-aligned expert rows from source to destination pointers."""
    if bytes_per_descriptor <= 64 * 1024:
        block_size = 128
        num_warps = 4
    elif bytes_per_descriptor >= 4 * 1024 * 1024 and src_ptrs.numel() > 1:
        block_size = 512
        num_warps = 8
    else:
        block_size = 256
        num_warps = 4
    items_per_program = 4
    words_per_program = block_size * items_per_program
    _eplb_push_copy_kernel[(triton.cdiv(bytes_per_descriptor // 8, words_per_program), src_ptrs.numel())](
        src_ptrs,
        dst_ptrs,
        bytes_per_descriptor,
        BLOCK_SIZE=block_size,
        ITEMS_PER_PROGRAM=items_per_program,
        num_warps=num_warps,
    )


@triton.jit
def _eplb_map_kernel(
    topk_ids_ptr,
    logical_to_physical_ptr,
    logical_replica_count_ptr,
    expert_counter_ptr,
    record_load_ptr,
    sample_index,
    total_assignment_num,
    counter_num_experts: tl.constexpr,
    topk_num: tl.constexpr,
    map_slots: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_assignment_num
    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)

    record_mask = mask & (tl.load(record_load_ptr) != 0)
    tl.atomic_add(
        expert_counter_ptr + sample_index * counter_num_experts + logical_id,
        1,
        mask=record_mask,
        sem="relaxed",
    )

    if SINGLE_TOKEN:
        replica_index = 0
    else:
        replica_count = tl.load(
            logical_replica_count_ptr + logical_id,
            mask=mask,
            other=1,
        )
        token_index = offsets // topk_num
        replica_index = eplb_replica_index(token_index, logical_id, replica_count)
    physical_id = tl.load(
        logical_to_physical_ptr + logical_id * map_slots + replica_index,
        mask=mask,
        other=-1,
    )
    tl.store(topk_ids_ptr + offsets, physical_id, mask=mask)


@triton.jit
def _eplb_map_no_record_kernel(
    topk_ids_ptr,
    logical_to_physical_ptr,
    logical_replica_count_ptr,
    total_assignment_num,
    topk_num: tl.constexpr,
    map_slots: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Map logical ids without carrying any sampling state in the ABI."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_assignment_num
    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    if SINGLE_TOKEN:
        replica_index = 0
    else:
        replica_count = tl.load(logical_replica_count_ptr + logical_id, mask=mask, other=1)
        token_index = offsets // topk_num
        replica_index = eplb_replica_index(token_index, logical_id, replica_count)
    physical_id = tl.load(
        logical_to_physical_ptr + logical_id * map_slots + replica_index,
        mask=mask,
        other=-1,
    )
    tl.store(topk_ids_ptr + offsets, physical_id, mask=mask)


@triton.jit
def _eplb_map_record_kernel(
    topk_ids_ptr,
    logical_to_physical_ptr,
    logical_replica_count_ptr,
    expert_counter_ptr,
    sample_index,
    total_assignment_num,
    counter_num_experts: tl.constexpr,
    topk_num: tl.constexpr,
    map_slots: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Map logical ids and unconditionally record the sampled load."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_assignment_num
    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    tl.atomic_add(
        expert_counter_ptr + sample_index * counter_num_experts + logical_id,
        1,
        mask=mask,
        sem="relaxed",
    )
    if SINGLE_TOKEN:
        replica_index = 0
    else:
        replica_count = tl.load(logical_replica_count_ptr + logical_id, mask=mask, other=1)
        token_index = offsets // topk_num
        replica_index = eplb_replica_index(token_index, logical_id, replica_count)
    physical_id = tl.load(
        logical_to_physical_ptr + logical_id * map_slots + replica_index,
        mask=mask,
        other=-1,
    )
    tl.store(topk_ids_ptr + offsets, physical_id, mask=mask)


@triton.jit
def _eplb_map_record_histogram_kernel(
    topk_ids_ptr,
    logical_to_physical_ptr,
    logical_replica_count_ptr,
    expert_counter_ptr,
    sample_index,
    total_assignment_num,
    counter_num_experts: tl.constexpr,
    histogram_bins: tl.constexpr,
    topk_num: tl.constexpr,
    map_slots: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_assignment_num
    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)

    histogram = tl.histogram(logical_id, histogram_bins)
    bins = tl.arange(0, histogram_bins)
    valid_count = tl.minimum(total_assignment_num - tl.program_id(0) * BLOCK_SIZE, BLOCK_SIZE)
    histogram -= tl.where(bins == 0, BLOCK_SIZE - valid_count, 0)
    tl.atomic_add(
        expert_counter_ptr + sample_index * counter_num_experts + bins,
        histogram,
        mask=(bins < counter_num_experts) & (histogram != 0),
        sem="relaxed",
    )

    if SINGLE_TOKEN:
        replica_index = 0
    else:
        replica_count = tl.load(logical_replica_count_ptr + logical_id, mask=mask, other=1)
        token_index = offsets // topk_num
        replica_index = eplb_replica_index(token_index, logical_id, replica_count)
    physical_id = tl.load(
        logical_to_physical_ptr + logical_id * map_slots + replica_index,
        mask=mask,
        other=-1,
    )
    tl.store(topk_ids_ptr + offsets, physical_id, mask=mask)


def _map_launch_config(total_assignment_num: int):
    block_size = min(256, triton.next_power_of_2(total_assignment_num))
    num_warps = 1 if block_size <= 32 else 2 if block_size <= 64 else 4
    return block_size, num_warps


def _record_histogram_block_size(total_assignment_num: int, counter_num_experts: int) -> int:
    if total_assignment_num < 16 * counter_num_experts:
        return 0
    if total_assignment_num < 16 * 1024:
        return 128
    if total_assignment_num < 32 * 1024:
        return 256
    return 512


@torch.no_grad()
def eplb_map(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_counter: torch.Tensor,
    record_load_tensor: torch.Tensor,
    sample_index: int,
):
    """Public dynamic-flag EPLB map, including CUDA-graph compatible recording."""
    assert topk_ids.is_cuda and topk_ids.is_contiguous() and topk_ids.ndim == 2
    assert logical_to_physical_map.is_cuda
    assert logical_replica_count.is_cuda
    assert expert_counter.is_cuda and expert_counter.ndim == 2
    assert 0 <= sample_index < expert_counter.shape[0]
    assert record_load_tensor.is_cuda and record_load_tensor.numel() == 1
    total_assignment_num = topk_ids.numel()
    if total_assignment_num == 0:
        return
    block_size, num_warps = _map_launch_config(total_assignment_num)
    _eplb_map_kernel[(triton.cdiv(total_assignment_num, block_size),)](
        topk_ids,
        logical_to_physical_map,
        logical_replica_count,
        expert_counter,
        record_load_tensor,
        sample_index,
        total_assignment_num,
        counter_num_experts=expert_counter.shape[1],
        topk_num=topk_ids.shape[1],
        map_slots=logical_to_physical_map.shape[1],
        SINGLE_TOKEN=topk_ids.shape[0] == 1,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


@torch.no_grad()
def eplb_map_fast(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_counter: torch.Tensor,
    sample_index: int,
    *,
    record_load: bool,
) -> None:
    """Service fast path selected by a host recording state."""
    total_assignment_num = topk_ids.numel()
    if total_assignment_num == 0:
        return
    if record_load:
        block_size = _record_histogram_block_size(total_assignment_num, expert_counter.shape[1])
        if block_size:
            _eplb_map_record_histogram_kernel[(triton.cdiv(total_assignment_num, block_size),)](
                topk_ids,
                logical_to_physical_map,
                logical_replica_count,
                expert_counter,
                sample_index,
                total_assignment_num,
                counter_num_experts=expert_counter.shape[1],
                histogram_bins=triton.next_power_of_2(expert_counter.shape[1]),
                topk_num=topk_ids.shape[1],
                map_slots=logical_to_physical_map.shape[1],
                SINGLE_TOKEN=topk_ids.shape[0] == 1,
                BLOCK_SIZE=block_size,
                num_warps=8 if block_size == 512 else 4,
            )
        else:
            block_size, num_warps = _map_launch_config(total_assignment_num)
            _eplb_map_record_kernel[(triton.cdiv(total_assignment_num, block_size),)](
                topk_ids,
                logical_to_physical_map,
                logical_replica_count,
                expert_counter,
                sample_index,
                total_assignment_num,
                counter_num_experts=expert_counter.shape[1],
                topk_num=topk_ids.shape[1],
                map_slots=logical_to_physical_map.shape[1],
                SINGLE_TOKEN=topk_ids.shape[0] == 1,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
    else:
        block_size, num_warps = _map_launch_config(total_assignment_num)
        _eplb_map_no_record_kernel[(triton.cdiv(total_assignment_num, block_size),)](
            topk_ids,
            logical_to_physical_map,
            logical_replica_count,
            total_assignment_num,
            topk_num=topk_ids.shape[1],
            map_slots=logical_to_physical_map.shape[1],
            SINGLE_TOKEN=topk_ids.shape[0] == 1,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
