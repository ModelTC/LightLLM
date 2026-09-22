"""Stream-ordered copies for token KV and arbitrary pinned checkpoint tensors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_checkpoint_pages(
    SRC,
    DST,
    SRC_IDS,
    DST_IDS,
    READIES,
    SRC_STRIDE,
    DST_STRIDE,
    WIDTH,
    PAGE_NUM,
    SKIP_READY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for page in range(PAGE_NUM):
        src_id = tl.load(SRC_IDS + page).to(tl.int64)
        dst_id = tl.load(DST_IDS + page).to(tl.int64)
        valid = (src_id >= 0) & (dst_id >= 0)
        if SKIP_READY:
            valid = valid & ~tl.load(READIES + page)
        if valid:
            for block in range(tl.program_id(0), tl.cdiv(WIDTH, BLOCK), tl.num_programs(0)):
                offsets = block * BLOCK + tl.arange(0, BLOCK)
                data = tl.load(SRC + src_id * SRC_STRIDE + offsets, offsets < WIDTH, other=0)
                tl.store(DST + dst_id * DST_STRIDE + offsets, data, offsets < WIDTH, cache_modifier=".wt")


def copy_checkpoint_pages(sources, destinations, source_ids, destination_ids, page_readies=None):
    """Copy slot-indexed, contiguous rows; preserve each tensor's byte representation."""
    assert len(sources) == len(destinations)
    assert len(source_ids) == len(destination_ids)
    for source, destination in zip(sources, destinations):
        source = source.reshape(source.shape[0], -1).view(torch.uint8)
        destination = destination.reshape(destination.shape[0], -1).view(torch.uint8)
        assert source.shape[1] == destination.shape[1]
        assert source.stride(1) == destination.stride(1) == 1
        _copy_checkpoint_pages[(12,)](
            source,
            destination,
            source_ids,
            destination_ids,
            page_readies,
            source.stride(0),
            destination.stride(0),
            source.shape[1],
            len(source_ids),
            SKIP_READY=page_readies is not None,
            BLOCK=4096,
        )


@triton.jit
def _copy_checkpoint_tensor(SRC, DST, SIZE, BLOCK: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(SIZE, BLOCK), tl.num_programs(0)):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        data = tl.load(SRC + offsets, offsets < SIZE, other=0)
        tl.store(DST + offsets, data, offsets < SIZE, cache_modifier=".wt")


def copy_checkpoint_state(sources, destinations):
    """Copy a tail checkpoint on the CUDA stream, including CPU-to-CPU copies.

    A host-side tensor copy would race prior asynchronous saves and transfers.
    """
    assert len(sources) == len(destinations)
    for source, destination in zip(sources, destinations):
        assert source.shape == destination.shape and source.dtype == destination.dtype
        assert source.is_contiguous() and destination.is_contiguous()
        source = source.reshape(-1).view(torch.uint8)
        destination = destination.reshape(-1).view(torch.uint8)
        _copy_checkpoint_tensor[(12,)](source, destination, source.numel(), BLOCK=4096)


@triton.jit
def _copy_full_att_pages(
    KV,
    CACHE,
    MEM_IDS,
    PAGE_IDS,
    READIES,
    KV_STRIDE_L,
    KV_STRIDE_T,
    PAGE_STRIDE,
    RANK_OFFSET,
    WIDTH: tl.constexpr,
    LAYERS: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    PAGE_NUM,
    WRITE_RANK: tl.constexpr,
    OFFLOAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page_width = PAGE_TOKENS * LAYERS * WIDTH
    for page in range(PAGE_NUM):
        page_id = tl.load(PAGE_IDS + page).to(tl.int64)
        valid = page_id >= 0
        if OFFLOAD:
            valid = valid & ~tl.load(READIES + page) & WRITE_RANK
        if valid:
            for block in range(tl.program_id(0), tl.cdiv(page_width, BLOCK), tl.num_programs(0)):
                offsets = block * BLOCK + tl.arange(0, BLOCK)
                token = offsets // (LAYERS * WIDTH)
                layer = offsets // WIDTH % LAYERS
                dim = offsets % WIDTH
                mem_id = tl.load(MEM_IDS + page * PAGE_TOKENS + token, offsets < page_width, other=-1).to(tl.int64)
                mask = (offsets < page_width) & (mem_id >= 0)
                kv_ptr = KV + layer.to(tl.int64) * KV_STRIDE_L + mem_id * KV_STRIDE_T + dim
                cache_ptr = CACHE + page_id * PAGE_STRIDE + RANK_OFFSET + offsets
                if OFFLOAD:
                    value = tl.load(kv_ptr, mask, other=0)
                    tl.store(cache_ptr, value, mask, cache_modifier=".wt")
                else:
                    value = tl.load(cache_ptr, mask, other=0)
                    tl.store(kv_ptr, value, mask)


def copy_full_att_pages(kv, cpu_cache, mem_indexes, page_indexes, config, tp_rank, page_tokens, page_readies=None):
    """Pack ordinary target KV as [page, TP shard, token, layer, KV bytes]."""
    kv = kv.view(kv.shape[0], kv.shape[1], -1).view(torch.uint8)
    pages = cpu_cache.view(cpu_cache.shape[0], -1).view(torch.uint8)
    world_size = config.tp_world_size
    heads = config.full_att_all_num_kv_heads
    if heads >= world_size:
        assert heads % world_size == 0
        replicas = 1
    else:
        assert world_size % heads == 0
        replicas = world_size // heads
    shard_bytes = page_tokens * kv.shape[0] * kv.shape[2]
    assert shard_bytes * (world_size // replicas) == config.get_cpu_cache_full_att_bytes()
    assert len(mem_indexes) == len(page_indexes) * page_tokens
    _copy_full_att_pages[(12,)](
        kv,
        pages,
        mem_indexes,
        page_indexes,
        page_readies,
        kv.stride(0),
        kv.stride(1),
        pages.stride(0),
        (tp_rank // replicas) * shard_bytes,
        kv.shape[2],
        kv.shape[0],
        page_tokens,
        len(page_indexes),
        WRITE_RANK=tp_rank % replicas == 0,
        OFFLOAD=page_readies is not None,
        BLOCK=4096,
    )
