import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_routing_topk_to_cpu(
    topk_ids,
    mem_indexes,
    routing_buffer_ptr,
    total_size,
    moe_layer_index: tl.constexpr,
    layer_topk_size: tl.constexpr,
    topk: tl.constexpr,
    dtype_id: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_size

    token_offsets = offsets // topk
    topk_offsets = offsets - token_offsets * topk
    mem_index = tl.load(mem_indexes + token_offsets, mask=mask, other=-1).to(tl.int64)
    data = tl.load(topk_ids + offsets, mask=mask, other=0)

    dst_offsets = mem_index * layer_topk_size + moe_layer_index * topk + topk_offsets
    if dtype_id == 1:
        dst_ptr = tl.load(routing_buffer_ptr).to(tl.pointer_type(tl.uint8))
        tl.store(dst_ptr + dst_offsets, data.to(tl.uint8), mask=mask)
    else:
        dst_ptr = tl.load(routing_buffer_ptr).to(tl.pointer_type(tl.int16))
        tl.store(dst_ptr + dst_offsets, data.to(tl.int16), mask=mask)


def scatter_routing_topk_to_cpu(
    topk_ids: torch.Tensor,
    mem_indexes: torch.Tensor,
    routing_buffer_ptr: torch.Tensor,
    moe_layer_index: int,
    num_moe_layers: int,
    topk: int,
    dtype_id: int,
):
    assert topk_ids.is_cuda
    assert mem_indexes.is_cuda
    assert mem_indexes.is_contiguous()
    assert routing_buffer_ptr.is_cuda
    assert routing_buffer_ptr.dtype == torch.uint64
    assert routing_buffer_ptr.numel() == 1
    assert topk_ids.dim() == 2
    assert topk_ids.shape[1] == topk
    assert topk_ids.is_contiguous()
    assert 0 <= moe_layer_index < num_moe_layers

    num_tokens = topk_ids.shape[0]
    layer_topk_size = num_moe_layers * topk
    total_size = num_tokens * topk
    if total_size == 0:
        return

    BLOCK = 1024
    grid = (triton.cdiv(total_size, BLOCK),)
    _scatter_routing_topk_to_cpu[grid](
        topk_ids=topk_ids,
        mem_indexes=mem_indexes,
        routing_buffer_ptr=routing_buffer_ptr,
        total_size=total_size,
        moe_layer_index=moe_layer_index,
        layer_topk_size=layer_topk_size,
        topk=topk,
        dtype_id=dtype_id,
        BLOCK=BLOCK,
    )
