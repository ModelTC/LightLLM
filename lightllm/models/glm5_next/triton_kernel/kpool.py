import torch
import triton
import triton.language as tl

from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import _butterfly_stage


@triton.jit
def _compress_pools(Raw, Packed, Ape, Lengths, Starts, Ragged, RAW_STRIDE: tl.constexpr, PACKED_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(Lengths + row)
    if length % 4 == 0 and length > 0:
        start = tl.load(Starts + row)
        pool = tl.arange(0, 4)
        cols = tl.arange(0, 128)
        locs = tl.load(Ragged + start + length - 4 + pool)
        raw = tl.load(Raw + locs[:, None] * RAW_STRIDE + cols[None, :]).to(tl.float32)
        score = tl.load(Raw + locs[:, None] * RAW_STRIDE + 128 + cols[None, :]).to(tl.float32)
        score += tl.load(Ape + pool[:, None] * 128 + cols[None, :])
        score = tl.exp(score - tl.max(score, 0)[None, :])
        weights = score / tl.sum(score, 0)[None, :]
        key = tl.sum(raw * weights, 0).to(tl.bfloat16).to(tl.float32).reshape(1, 128)
        for step in tl.static_range(7):
            key = _butterfly_stage(key, 64 >> step, 1 << step, 1, 128)
        key = (key * (128 ** -0.5)).to(tl.bfloat16).to(tl.float32)
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(key), 1), 1e-4) / 448.0)))
        key = tl.minimum(tl.maximum(key / scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
        loc = tl.load(Ragged + start + length - 1)
        dest = Packed + loc * PACKED_STRIDE
        tl.store(dest + cols, key.reshape(128).to(tl.uint8, bitcast=True))
        tl.store((dest + 128).to(tl.pointer_type(tl.float32)), tl.sum(scale, 0))


def compress_pools(raw_buffer, packed_buffer, ape, lengths, starts, ragged):
    _compress_pools[(lengths.numel(),)](
        raw_buffer,
        packed_buffer,
        ape,
        lengths,
        starts,
        ragged,
        raw_buffer.stride(0),
        packed_buffer.stride(0),
        num_warps=4,
    )


@triton.jit
def _gather_pools(
    Packed,
    ReqTable,
    ReqIdx,
    SeqLen,
    K,
    Scale,
    PACKED_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    POOLS: tl.constexpr,
):
    batch, pool = tl.program_id(0), tl.program_id(1)
    length = tl.load(SeqLen + batch)
    req = tl.load(ReqIdx + batch)
    valid = pool < length // 4
    loc = tl.load(ReqTable + req * REQ_STRIDE + pool * 4 + 3, valid, 0)
    cols = tl.arange(0, 128)
    packed = tl.load(Packed + loc * PACKED_STRIDE + cols, valid, 0)
    scale = tl.load((Packed + loc * PACKED_STRIDE + 128).to(tl.pointer_type(tl.float32)), valid, 1.0)
    tl.store(K + (batch * POOLS + pool) * 128 + cols, packed.to(tl.float8e4nv, bitcast=True))
    tl.store(Scale + batch * POOLS + pool, scale)


def gather_pools(packed_buffer, req_table, req_idx, seq_len, max_pools):
    keys = torch.empty((req_idx.numel() * max_pools, 128), device=packed_buffer.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((keys.shape[0],), device=keys.device, dtype=torch.float32)
    _gather_pools[(req_idx.numel(), max_pools)](
        packed_buffer,
        req_table,
        req_idx,
        seq_len,
        keys,
        scales,
        packed_buffer.stride(0),
        req_table.stride(0),
        max_pools,
        num_warps=4,
    )
    return keys, scales


@triton.jit
def _expand_topk(
    Groups,
    Lengths,
    Starts,
    Ragged,
    Out,
    Relative,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    DENSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    length = tl.load(Lengths + row)
    start = tl.load(Starts + row)
    if DENSE:
        token = lane
        valid = lane < length
    else:
        closed_tokens = tl.minimum(length // 4 * 4, TOPK)
        group = tl.load(Groups + row * (TOPK // 4) + lane // 4, lane < closed_tokens, -1)
        token = tl.where(lane < closed_tokens, group * 4 + lane % 4, length // 4 * 4 + lane - closed_tokens)
        valid = (lane < closed_tokens + length % 4) & (token >= 0)
    mem = tl.load(Ragged + start + token, valid & (lane < WIDTH), -1)
    tl.store(Out + row * WIDTH + lane, mem, lane < WIDTH)
    tl.store(Relative + row * WIDTH + lane, tl.where(valid, token, -1), lane < WIDTH)


def expand_topk(groups, lengths, starts, ragged, topk, dense=False):
    width = triton.cdiv(topk + 3, 128) * 128
    out = torch.empty((lengths.numel(), width), dtype=torch.int32, device=lengths.device)
    relative = torch.empty_like(out)
    _expand_topk[(lengths.numel(),)](
        groups,
        lengths,
        starts,
        ragged,
        out,
        relative,
        topk,
        width,
        dense,
        triton.next_power_of_2(width),
        num_warps=4,
    )
    return out, relative
