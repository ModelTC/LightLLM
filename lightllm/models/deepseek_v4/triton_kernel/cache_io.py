"""Model-side packed views and token-indexed cache transfers."""

import triton
import triton.language as tl

from lightllm.common.state_cache_manager.deepseek4 import (
    DSV4_CACHE_BLOCK_SIZE,
    DSV4_C4_PAGE_BYTES,
    DSV4_C4_INDEXER_PAGE_BYTES,
    DSV4_C128_PAGE_BYTES,
)


def view_compressed_kv(buffer, ratio):
    if buffer is None:
        return None
    pages = buffer.view(-1, DSV4_CACHE_BLOCK_SIZE * buffer.shape[-1])
    width = DSV4_C4_PAGE_BYTES if ratio == 4 else DSV4_C128_PAGE_BYTES
    return pages[:, :width]


def get_compressed_kv_buffer(manager, layer):
    return view_compressed_kv(manager.get_att_input_params(layer), manager.cache_config.compress_rates[layer])


def get_indexer_k_buffer(manager, layer):
    buffer = manager.get_att_input_params(layer)
    pages = buffer.view(-1, DSV4_CACHE_BLOCK_SIZE * buffer.shape[-1])
    return pages[:, DSV4_C4_PAGE_BYTES : DSV4_C4_PAGE_BYTES + DSV4_C4_INDEXER_PAGE_BYTES]


@triton.jit
def _copy_tokens(
    storage,
    pages,
    indexes,
    page_ids,
    readies,
    layer_offset,
    TOKEN_NUM: tl.constexpr,
    COPY_NUM: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    TO_PAGES: tl.constexpr,
    HAS_READIES: tl.constexpr,
    ROUND_TAIL: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    storage = storage.to(tl.pointer_type(tl.uint8))
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    slot = tl.load(indexes + rows, rows < TOKEN_NUM, -1).to(tl.int64)
    if ROUND_TAIL:
        # Packed page tails include scales for earlier rows. PD also carries
        # the unused token slots of its final allocated compression block.
        last = tl.load(indexes + TOKEN_NUM - 1).to(tl.int64)
        slot = tl.where(rows < TOKEN_NUM, slot, last + rows - TOKEN_NUM + 1)
    page = rows // PAGE_TOKENS
    valid = (rows < COPY_NUM) & (slot >= 0)
    if HAS_READIES:
        valid &= tl.load(readies + page, rows < COPY_NUM, 1) == 0
    cpu_page = tl.load(page_ids + page, rows < COPY_NUM, 0).to(tl.int64)
    columns = tl.arange(0, BLOCK_WIDTH)
    gpu = storage + slot[:, None] * WIDTH + columns[None, :]
    cpu = (
        pages
        + cpu_page[:, None] * PAGE_STRIDE
        + layer_offset
        + (rows % PAGE_TOKENS)[:, None] * WIDTH
        + columns[None, :]
    )
    mask = valid[:, None] & (columns[None, :] < WIDTH)
    if TO_PAGES:
        tl.store(cpu, tl.load(gpu, mask, 0), mask)
    else:
        tl.store(gpu, tl.load(cpu, mask, 0), mask)


def copy_history_pages(
    manager, indexes, page_ids, pages, *, to_pages, readies=None, page_tokens=None, round_tail=False
):
    if not indexes.numel():
        return
    page_tokens = manager.cache_config.cpu_cache_token_page_size if page_tokens is None else page_tokens
    copy_num = (
        triton.cdiv(indexes.numel(), DSV4_CACHE_BLOCK_SIZE) * DSV4_CACHE_BLOCK_SIZE if round_tail else indexes.numel()
    )
    offset = 0
    for buffer in manager.kv_buffer:
        width = buffer.shape[-1]
        _copy_tokens[(triton.cdiv(copy_num, 32),)](
            buffer.data_ptr(),
            pages,
            indexes,
            page_ids,
            readies,
            offset,
            TOKEN_NUM=indexes.numel(),
            COPY_NUM=copy_num,
            PAGE_TOKENS=page_tokens,
            PAGE_STRIDE=pages.stride(0),
            WIDTH=width,
            TO_PAGES=to_pages,
            HAS_READIES=readies is not None,
            ROUND_TAIL=round_tail,
            BLOCK_WIDTH=triton.next_power_of_2(width),
            num_warps=4,
        )
        offset += page_tokens * width


@triton.jit
def _copy_checkpoints(
    pages,
    page_ids,
    readies,
    pointers,
    STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    NBYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    if tl.load(readies + page):
        return
    source = tl.load(pointers + page).to(tl.pointer_type(tl.uint8))
    target = pages + tl.load(page_ids + page).to(tl.int64) * STRIDE + OFFSET
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(target + offset, tl.load(source + offset, offset < NBYTES, 0), offset < NBYTES)


def copy_checkpoints_to_pages(manager, page_ids, readies, pointers, pages):
    nbytes = manager.big_page_buffers.buffer.shape[-1]
    _copy_checkpoints[(page_ids.numel(), triton.cdiv(nbytes, 8192))](
        pages,
        page_ids,
        readies,
        pointers,
        STRIDE=pages.stride(0),
        OFFSET=manager.cache_config.get_cpu_cache_full_att_bytes(),
        NBYTES=nbytes,
        BLOCK=8192,
        num_warps=4,
    )
