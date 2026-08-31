"""Local greedy statistics for distributed vocabulary shards."""

import torch
import triton
import triton.language as tl


@triton.jit
def _greedy_sample_stage1_kernel(
    logits,
    partial_max,
    partial_sum,
    partial_argmax,
    stride_row,
    stride_col,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        logits + row * stride_row + offsets * stride_col,
        mask=offsets < vocab_size,
        other=-float("inf"),
    )
    values = values.to(tl.float32)

    block_max = tl.max(values, axis=0)
    block_sum = tl.sum(tl.exp(values - block_max), axis=0)
    block_argmax = tl.argmax(values, axis=0) + block * BLOCK_SIZE
    output_offset = row * num_blocks + block
    tl.store(partial_max + output_offset, block_max)
    tl.store(partial_sum + output_offset, block_sum)
    tl.store(partial_argmax + output_offset, block_argmax)


@triton.jit
def _greedy_sample_stage2_stats_kernel(
    partial_max,
    partial_sum,
    partial_argmax,
    output_stats,
    output_argmax,
    num_blocks: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    input_offset = row * num_blocks + offsets
    block_max = tl.load(partial_max + input_offset, mask=mask, other=-float("inf"))
    block_sum = tl.load(partial_sum + input_offset, mask=mask, other=0.0)
    block_argmax = tl.load(partial_argmax + input_offset, mask=mask, other=0x7FFFFFFF)

    global_max = tl.max(block_max, axis=0)
    global_sum = tl.sum(block_sum * tl.exp(block_max - global_max), axis=0)
    candidate_ids = tl.where(block_max == global_max, block_argmax, 0x7FFFFFFF)
    global_argmax = tl.min(candidate_ids, axis=0)
    tl.store(output_stats + row, global_max)
    tl.store(output_stats + batch_size + row, global_max + tl.log(global_sum))
    tl.store(output_argmax + row, global_argmax)


def _launch_stage1(logits: torch.Tensor, scratch: torch.Tensor, block_size: int, num_blocks: int) -> None:
    batch_size, vocab_size = logits.shape
    _greedy_sample_stage1_kernel[(batch_size, num_blocks)](
        logits,
        scratch[0],
        scratch[1],
        scratch[2],
        logits.stride(0),
        logits.stride(1),
        vocab_size=vocab_size,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


@torch.no_grad()
def greedy_sample_local_stats(logits: torch.Tensor, alloc_func=torch.empty) -> torch.Tensor:
    """Return local max, logsumexp and argmax rows for distributed greedy sampling."""

    assert logits.ndim == 2 and logits.is_cuda and logits.is_contiguous()
    batch_size, vocab_size = logits.shape
    block_size = 4096
    num_blocks = triton.cdiv(vocab_size, block_size)
    scratch = alloc_func((3, batch_size, num_blocks), dtype=torch.float32, device=logits.device)
    # The third FP32 row carries INT32 argmax bits. Keeping one fixed-size
    # payload gives the distributed reducer a single collective without losing
    # token-id precision through a numeric int-to-float conversion.
    output_stats = alloc_func((3, batch_size), dtype=torch.float32, device=logits.device)

    _launch_stage1(logits, scratch, block_size, num_blocks)
    _greedy_sample_stage2_stats_kernel[(batch_size,)](
        scratch[0],
        scratch[1],
        scratch[2],
        output_stats,
        output_stats[2].view(torch.int32),
        num_blocks=num_blocks,
        batch_size=batch_size,
        BLOCK_SIZE=triton.next_power_of_2(num_blocks),
        num_warps=4,
    )
    return output_stats
