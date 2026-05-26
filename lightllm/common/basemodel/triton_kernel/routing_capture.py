import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_routing_capture_to_cpu(
    capture_buffer,
    mem_indexes,
    routing_buffer,
    total_size,
    layer_topk_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_size

    token_offsets = offsets // layer_topk_size
    inner_offsets = offsets - token_offsets * layer_topk_size
    mem_index = tl.load(mem_indexes + token_offsets, mask=mask, other=-1).to(tl.int64)
    data = tl.load(capture_buffer + offsets, mask=mask, other=0)

    dst_offsets = mem_index * layer_topk_size + inner_offsets
    tl.store(routing_buffer + dst_offsets, data, mask=mask & (mem_index >= 0))


def scatter_routing_capture_to_cpu(
    capture_buffer: torch.Tensor,
    mem_indexes: torch.Tensor,
    routing_buffer: torch.Tensor,
    num_tokens: int,
    num_moe_layers: int,
    topk: int,
):
    assert capture_buffer.is_cuda
    assert mem_indexes.is_cuda
    assert routing_buffer.device.type == "cpu" and routing_buffer.is_pinned()
    assert capture_buffer.is_contiguous()
    assert routing_buffer.is_contiguous()

    layer_topk_size = num_moe_layers * topk
    total_size = num_tokens * layer_topk_size
    if total_size == 0:
        return

    BLOCK = 1024
    grid = (triton.cdiv(total_size, BLOCK),)
    _scatter_routing_capture_to_cpu[grid](
        capture_buffer=capture_buffer,
        mem_indexes=mem_indexes,
        routing_buffer=routing_buffer,
        total_size=total_size,
        layer_topk_size=layer_topk_size,
        BLOCK=BLOCK,
    )
