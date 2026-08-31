import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_weight_scale_kernel(
    weights,
    q_scale,
    size,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    value = tl.load(weights + offsets, mask=mask).to(tl.float32)
    query_scale = tl.load(q_scale + offsets, mask=mask).to(tl.float32)
    tl.store(weights + offsets, value * scale * query_scale, mask=mask)


def scale_indexer_weights_(weights: torch.Tensor, q_scale: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply the indexer query/head scales in one in-place kernel."""

    assert weights.dtype == torch.float32
    assert weights.is_contiguous() and q_scale.is_contiguous()
    assert weights.numel() == q_scale.numel()
    size = weights.numel()
    if size == 0:
        return weights
    block_size = 256
    _indexer_weight_scale_kernel[(triton.cdiv(size, block_size),)](
        weights,
        q_scale,
        size,
        scale,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return weights
