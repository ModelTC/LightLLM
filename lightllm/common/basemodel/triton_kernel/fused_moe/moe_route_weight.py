"""Apply GLM routing probabilities at the BF16 W2 input boundary."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _multiply_rows(X, P, ROWS: tl.constexpr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * COLS + cols, cols < COLS, 0).to(tl.float32)
    p = tl.load(P + row)
    tl.store(X + row * COLS + cols, x * p, cols < COLS)


def multiply_routing_weights(x, weights):
    """x is already rounded to BF16 by SwiGLU; round x.float()*p to BF16 again."""
    assert x.is_contiguous() and weights.is_contiguous()
    assert x.numel() // x.shape[-1] == weights.numel()
    _multiply_rows[(weights.numel(), triton.cdiv(x.shape[-1], 256))](
        x, weights, weights.numel(), x.shape[-1], 256, num_warps=4
    )


@triton.jit
def _sort_tp_weights(Blocks, Counts, Weights, Out, STRIDE: tl.constexpr, BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    expert = tl.load(Blocks + block * 3)
    if expert >= 0:
        tile = tl.load(Blocks + block * 3 + 1)
        start = tl.load(Blocks + block * 3 + 2)
        lanes = tl.arange(0, BLOCK_M)
        count = tl.load(Counts + expert)
        p = tl.load(Weights + expert * STRIDE + tile * BLOCK_M + lanes, tile * BLOCK_M + lanes < count, 0)
        tl.store(Out + start + lanes, p, tile * BLOCK_M + lanes < count)


def sorted_tp_routing_weights(topk_weights, expert_weights, counts, mblock_infos):
    """Reorder p exactly like the TMA W13 output, including non-aligned expert tails."""
    out = torch.empty_like(topk_weights).view(-1)
    blocks, block_m = mblock_infos
    _sort_tp_weights[(blocks.shape[0],)](
        blocks, counts, expert_weights, out, expert_weights.stride(0), block_m, num_warps=4
    )
    return out


@triton.jit
def _gather_low_latency_weights(
    Routes,
    SrcInfo,
    Layout,
    Out,
    NUM_RANKS: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    expert, src_rank = tl.program_id(0), tl.program_id(1)
    layout = tl.load(Layout + expert * NUM_RANKS + src_rank)
    count = (layout & 0xFFFFFFFF).to(tl.int32)
    start = (layout >> 32).to(tl.int32)
    k = tl.arange(0, BLOCK_TOPK)
    for offset in range(0, count, BLOCK):
        rows = offset + tl.arange(0, BLOCK)
        valid = rows < count
        dst = expert * (NUM_RANKS * MAX_TOKENS) + start + rows
        src_token = tl.load(SrcInfo + dst, valid, 0)
        ptr = Routes + ((src_rank * MAX_TOKENS + src_token[:, None]) * TOPK + k[None, :]) * 2
        ids = tl.load(ptr, valid[:, None] & (k[None, :] < TOPK), -1).to(tl.int32)
        weights = tl.load(ptr + 1, valid[:, None] & (k[None, :] < TOPK), 0)
        p = tl.sum(tl.where(ids == RANK * LOCAL_EXPERTS + expert, weights, 0), 1)
        tl.store(Out + dst, p, valid)


def gather_low_latency_routing_weights(buffer, topk_idx, topk_weights, handle):
    """Recover p using DeepEP's actual receive order, without CPU count synchronization.

    Legacy dispatch transports X but not p. All-gather only the small padded
    FP32 (expert id, p) table, then use its src_info/layout_range handle. All
    ranks participate, including empty/HOLD ranks, so CUDA graph replay has a
    fixed collective shape. Expert ids are exactly representable in FP32.
    """
    src_info, layout, max_tokens, _, num_experts = handle
    ranks, topk = buffer.group_size, topk_idx.shape[1]
    assert num_experts < 2**24 and topk_idx.shape[0] <= max_tokens
    assert topk_idx.shape == topk_weights.shape and topk_weights.dtype == torch.float32
    assert src_info.shape == (num_experts // ranks, ranks * max_tokens)
    assert layout.shape == (num_experts // ranks, ranks)
    assert src_info.dtype == torch.int32 and layout.dtype == torch.int64
    routes = torch.zeros((max_tokens, topk, 2), dtype=torch.float32, device=topk_weights.device)
    routes[:, :, 0].fill_(-1)
    routes[: topk_idx.shape[0], :, 0].copy_(topk_idx)
    routes[: topk_idx.shape[0], :, 1].copy_(topk_weights)
    if ranks == 1:
        all_routes = routes
    else:
        all_routes = torch.empty((ranks * max_tokens, topk, 2), dtype=routes.dtype, device=routes.device)
        dist.all_gather_into_tensor(all_routes, routes, group=buffer.group)
    out = torch.zeros(src_info.shape, dtype=torch.float32, device=topk_weights.device)
    _gather_low_latency_weights[(num_experts // ranks, ranks)](
        all_routes,
        src_info,
        layout,
        out,
        ranks,
        max_tokens,
        topk,
        num_experts // ranks,
        buffer.rank,
        32,
        triton.next_power_of_2(topk),
        num_warps=4,
    )
    return out
