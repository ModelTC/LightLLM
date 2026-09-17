import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _greedy_partial(
    logits,
    temperatures,
    partial_stats,
    vocab_size: tl.constexpr,
    stride_batch,
    stride_vocab,
    num_chunks: tl.constexpr,
    BLOCK: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    ids = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + row * stride_batch + ids * stride_vocab, ids < vocab_size, other=-float("inf"))
    if APPLY_TEMPERATURE:
        values = tl.div_rn(values, tl.load(temperatures + row))
        # 保留原地温度处理，后续 token rank 计算仍使用同一份 logits。
        tl.store(logits + row * stride_batch + ids * stride_vocab, values, ids < vocab_size)
    # 与 torch.argmax 一致：优先返回第一个 NaN，否则同分时取最小 token ID。
    nan_id = tl.min(tl.where(values != values, ids, 0x7FFFFFFF), 0)
    maximum = tl.max(tl.where(values != values, -float("inf"), values), 0)
    token_id = tl.min(tl.where((ids < vocab_size) & (values == maximum), ids, 0x7FFFFFFF), 0)
    token_id = tl.where(nan_id != 0x7FFFFFFF, nan_id, token_id)
    maximum = tl.where(nan_id != 0x7FFFFFFF, float("nan"), maximum)
    denominator = tl.sum(tl.exp(values - maximum), 0)
    # 整块被屏蔽时贡献为零，不能让该块的 -inf - -inf 污染其他有效块。
    denominator = tl.where(maximum == -float("inf"), 0.0, denominator)
    offset = row * num_chunks + chunk
    tl.store(partial_stats + offset * 3, maximum)
    tl.store(partial_stats + offset * 3 + 1, denominator)
    # 第三个槽位按 int32 读写索引，复用同一缓冲区并保留全部索引位。
    tl.store(partial_stats.to(tl.pointer_type(tl.int32)) + offset * 3 + 2, token_id)


@triton.jit
def _greedy_finish(
    partial_stats,
    output_ids,
    output_logprobs,
    num_chunks: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    chunks = tl.arange(0, BLOCK)
    offset = row * num_chunks + chunks
    maxima = tl.load(partial_stats + offset * 3, chunks < num_chunks, other=-float("inf"))
    denominators = tl.load(partial_stats + offset * 3 + 1, chunks < num_chunks, other=0.0)
    ids = tl.load(partial_stats.to(tl.pointer_type(tl.int32)) + offset * 3 + 2, chunks < num_chunks, other=0x7FFFFFFF)
    nan_id = tl.min(tl.where(maxima != maxima, ids, 0x7FFFFFFF), 0)
    maximum = tl.max(tl.where(maxima != maxima, -float("inf"), maxima), 0)
    token_id = tl.min(tl.where(maxima == maximum, ids, 0x7FFFFFFF), 0)
    token_id = tl.where(nan_id != 0x7FFFFFFF, nan_id, token_id)
    terms = denominators * tl.exp(maxima - maximum)
    denominator = tl.sum(tl.where(maxima == -float("inf"), 0.0, terms), 0)
    # 直接计算最大项的 logprob，避免先加回巨大偏移再相减造成精度损失。
    logprob = -tl.log(denominator)
    logprob = tl.where((maximum == -float("inf")) | (nan_id != 0x7FFFFFFF), float("nan"), logprob)
    tl.store(output_ids + row, token_id)
    tl.store(output_logprobs + row, logprob)


def greedy_sample(
    logits: torch.Tensor, temperatures: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """对已完成惩罚的 FP32 logits 执行可选的原地温度处理，并返回 greedy token 和 logprob。"""
    # 小张量的 Triton 调度成本高于收益；按完整 sample 的测量保留较快的 PyTorch 路径。
    if logits.numel() < 16 * 1024 * 1024:
        if temperatures is not None:
            logits.div_(temperatures.view(-1, 1))
        token_ids = logits.argmax(dim=-1)
        logprobs = torch.log_softmax(logits, dim=-1).gather(1, token_ids.view(-1, 1)).view(-1)
        return token_ids, logprobs
    return _fused_greedy_sample(logits, temperatures)


def _fused_greedy_sample(
    logits: torch.Tensor, temperatures: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    assert logits.is_cuda and logits.dtype == torch.float32 and logits.ndim == 2
    batch_size, vocab_size = logits.shape
    assert 0 < vocab_size < 0x7FFFFFFF
    block = 4096
    num_chunks = triton.cdiv(vocab_size, block)
    partial_stats = torch.empty((batch_size, num_chunks, 3), device=logits.device, dtype=torch.float32)
    output_ids = torch.empty((batch_size,), device=logits.device, dtype=torch.int64)
    output_logprobs = torch.empty((batch_size,), device=logits.device, dtype=torch.float32)
    if batch_size == 0:
        return output_ids, output_logprobs
    _greedy_partial[(batch_size, num_chunks)](
        logits,
        temperatures,
        partial_stats,
        vocab_size,
        *logits.stride(),
        num_chunks,
        block,
        temperatures is not None,
        num_warps=4,
    )
    _greedy_finish[(batch_size,)](
        partial_stats,
        output_ids,
        output_logprobs,
        num_chunks,
        triton.next_power_of_2(num_chunks),
        num_warps=4,
    )
    return output_ids, output_logprobs
