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
