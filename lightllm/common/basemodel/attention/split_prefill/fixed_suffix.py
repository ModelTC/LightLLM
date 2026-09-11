from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fixed_suffix_softmax_kernel(
    scores,
    lse,
    stride_sb,
    stride_sr,
    stride_sk,
    stride_lb,
    stride_lr,
    scale,
    ROWS: tl.constexpr,
    KEYS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    FIRST_SUFFIX_TOKEN_ISOLATED: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,
):
    batch = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    keys = tl.arange(0, BLOCK_KEYS)
    valid_rows = rows < ROWS
    valid_keys = keys < KEYS
    if FIRST_SUFFIX_TOKEN_ISOLATED:
        # The first suffix query sees only the first suffix key. All remaining
        # suffix queries see the complete fixed-size suffix.
        valid_keys = valid_keys[None, :] & (
            (rows[:, None] >= QUERY_HEADS) | (keys[None, :] == 0)
        )
    else:
        valid_keys = valid_keys[None, :]

    pointers = (
        scores
        + batch * stride_sb
        + rows[:, None] * stride_sr
        + keys[None, :] * stride_sk
    )
    mask = valid_rows[:, None] & valid_keys
    values = tl.load(pointers, mask=mask, other=-float("inf")).to(tl.float32)
    values *= scale
    row_max = tl.max(values, axis=1)
    values = tl.exp(values - row_max[:, None])
    denominator = tl.sum(values, axis=1)
    values /= denominator[:, None]
    tl.store(pointers, values, mask=valid_rows[:, None] & (keys[None, :] < KEYS))
    tl.store(
        lse + batch * stride_lb + rows * stride_lr,
        row_max + tl.log(denominator),
        mask=valid_rows,
    )


def _allocate_lse(
    batch_size: int,
    rows: int,
    query: torch.Tensor,
    alloc_func,
) -> torch.Tensor:
    if alloc_func is torch.empty:
        return torch.empty(
            (batch_size, rows),
            dtype=torch.float32,
            device=query.device,
        )
    return alloc_func((batch_size, rows), torch.float32)


def fixed_suffix_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    batch_size: int,
    suffix_length: int,
    first_suffix_token_isolated: bool,
    alloc_func=torch.empty,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend packed queries to a fixed suffix without materializing a mask."""

    token_count, query_heads, head_dim = query.shape
    if token_count != batch_size * suffix_length:
        raise ValueError("packed query tokens do not match the suffix shape")
    if key.shape != value.shape:
        raise ValueError("suffix K/V shapes differ")
    if key.shape != (token_count, 1, head_dim):
        raise ValueError("fixed suffix attention currently requires one K/V head")
    if query.device.type != "cuda" or key.device != query.device:
        raise ValueError("fixed suffix attention requires colocated CUDA tensors")
    if value.device != query.device:
        raise ValueError("fixed suffix attention requires colocated CUDA tensors")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("fixed suffix Q/K/V dtypes must match")

    query = query.view(batch_size, suffix_length, query_heads, head_dim)
    key = key.view(batch_size, suffix_length, head_dim)
    value = value.view(batch_size, suffix_length, head_dim)
    rows = suffix_length * query_heads
    scores = torch.bmm(
        query.reshape(batch_size, rows, head_dim),
        key.transpose(1, 2),
    )
    lse = _allocate_lse(batch_size, rows, query, alloc_func)
    block_rows = 4
    _fixed_suffix_softmax_kernel[(triton.cdiv(rows, block_rows), batch_size)](
        scores,
        lse,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        lse.stride(0),
        lse.stride(1),
        head_dim**-0.5,
        ROWS=rows,
        KEYS=suffix_length,
        QUERY_HEADS=query_heads,
        FIRST_SUFFIX_TOKEN_ISOLATED=first_suffix_token_isolated,
        BLOCK_ROWS=block_rows,
        BLOCK_KEYS=triton.next_power_of_2(suffix_length),
        num_warps=8,
    )
    output = torch.bmm(scores, value)
    return (
        output.view(token_count, query_heads, head_dim),
        lse.view(token_count, query_heads),
    )
