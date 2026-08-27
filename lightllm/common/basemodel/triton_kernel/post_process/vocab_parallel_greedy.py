"""Greedy sampling directly from tensor-parallel vocabulary shards."""

import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.post_process.greedy_sample import (
    greedy_sample_local_stats,
)
from lightllm.common.basemodel.triton_kernel.transpose_convert import (
    transpose_convert_2d,
)
from lightllm.distributed.communication_op import all_gather_into_tensor


@triton.jit
def _combine_vocab_parallel_stats_kernel(
    gathered_stats,
    gathered_argmax,
    output_logits,
    output_token_ids,
    output_logsumexp,
    token_num,
    vocab_size: tl.constexpr,
    tp_world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token_mask = token_offsets < token_num
    rank_stride = 3 * token_num

    global_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    global_id = tl.full((BLOCK_SIZE,), 0x7FFFFFFF, tl.int32)
    for rank in tl.static_range(tp_world_size):
        rank_base = rank * rank_stride
        local_max = tl.load(
            gathered_stats + rank_base + token_offsets,
            mask=token_mask,
            other=-float("inf"),
        )
        local_id = tl.load(
            gathered_argmax + rank_base + 2 * token_num + token_offsets,
            mask=token_mask,
            other=0x7FFFFFFF,
        )
        local_id += (rank * vocab_size) // tp_world_size
        wins = (local_max > global_max) | ((local_max == global_max) & (local_id < global_id))
        global_max = tl.where(wins, local_max, global_max)
        global_id = tl.where(wins, local_id, global_id)

    global_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
    for rank in tl.static_range(tp_world_size):
        rank_base = rank * rank_stride
        local_lse = tl.load(
            gathered_stats + rank_base + token_num + token_offsets,
            mask=token_mask,
            other=-float("inf"),
        )
        global_sum += tl.exp(local_lse - global_max)

    tl.store(output_logits + token_offsets, global_max, mask=token_mask)
    tl.store(output_token_ids + token_offsets, global_id, mask=token_mask)
    tl.store(output_logsumexp + token_offsets, global_max + tl.log(global_sum), mask=token_mask)


@torch.no_grad()
def vocab_parallel_greedy(
    local_logits: torch.Tensor,
    *,
    vocab_size: int,
    tp_world_size: int,
    group,
    alloc_func,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact sparse logits, global token ids and full-vocab logsumexp."""

    assert local_logits.ndim == 2 and local_logits.is_cuda and local_logits.is_contiguous()
    local_vocab_size, token_num = local_logits.shape
    assert local_vocab_size in {
        vocab_size // tp_world_size,
        (vocab_size + tp_world_size - 1) // tp_world_size,
    }

    transposed_logits = alloc_func(
        (token_num, local_vocab_size),
        dtype=local_logits.dtype,
        device=local_logits.device,
    )
    transpose_convert_2d(local_logits, transposed_logits)
    local_stats = greedy_sample_local_stats(transposed_logits, alloc_func=alloc_func)

    if tp_world_size == 1:
        gathered_stats = local_stats.view(1, 3, token_num)
    else:
        gathered_stats = alloc_func((tp_world_size, 3, token_num), dtype=torch.float32, device=local_logits.device)
        all_gather_into_tensor(
            output_=gathered_stats,
            input_=local_stats,
            group=group,
            async_op=False,
        )

    output_logits = alloc_func((token_num, 1), dtype=torch.float32, device=local_logits.device)
    output_token_ids = alloc_func((token_num, 1), dtype=torch.int64, device=local_logits.device)
    output_logsumexp = alloc_func((token_num,), dtype=torch.float32, device=local_logits.device)
    _combine_vocab_parallel_stats_kernel[(triton.cdiv(token_num, 256),)](
        gathered_stats,
        gathered_stats.view(torch.int32),
        output_logits,
        output_token_ids,
        output_logsumexp,
        token_num,
        vocab_size=vocab_size,
        tp_world_size=tp_world_size,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output_logits, output_token_ids, output_logsumexp
