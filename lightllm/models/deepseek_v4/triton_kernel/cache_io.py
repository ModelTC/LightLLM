"""Model-side packed views and token-indexed cache transfers."""

import triton
import triton.language as tl


def view_swa_kv(buffer, config):
    return buffer.view(-1, config.packed_kv_bytes(config.swa_token_alignment))


def get_compressed_kv_buffer(manager, layer):
    buffer = manager.get_att_input_params(layer)
    if buffer is None:
        return None
    config = manager.cache_config
    ratio = config.compress_rates[layer]
    storage = buffer.view(-1, config.history_token_alignment * buffer.shape[-1])
    return storage[:, : config.packed_kv_bytes(config.history_token_alignment // ratio)]


def get_indexer_k_buffer(manager, layer):
    config = manager.cache_config
    buffer = manager.get_att_input_params(layer)
    storage = buffer.view(-1, config.history_token_alignment * buffer.shape[-1])
    entries = config.history_token_alignment // config.short_compress_ratio
    begin = config.packed_kv_bytes(entries)
    return storage[:, begin : begin + entries * config.indexer_bytes]


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
    tokens = tl.program_id(0) * 32 + tl.arange(0, 32)
    slot = tl.load(indexes + tokens, tokens < TOKEN_NUM, -1).to(tl.int64)
    if ROUND_TAIL:
        # Packed page tails include scales for earlier tokens. PD also carries
        # the unused token slots of its final allocated compression block.
        last = tl.load(indexes + TOKEN_NUM - 1).to(tl.int64)
        slot = tl.where(tokens < TOKEN_NUM, slot, last + tokens - TOKEN_NUM + 1)
    page = tokens // PAGE_TOKENS
    valid = (tokens < COPY_NUM) & (slot >= 0)
    if HAS_READIES:
        valid &= tl.load(readies + page, tokens < COPY_NUM, 1) == 0
    cpu_page = tl.load(page_ids + page, tokens < COPY_NUM, 0).to(tl.int64)
    columns = tl.arange(0, BLOCK_WIDTH)
    gpu = storage + slot[:, None] * WIDTH + columns[None, :]
    cpu = (
        pages
        + cpu_page[:, None] * PAGE_STRIDE
        + layer_offset
        + (tokens % PAGE_TOKENS)[:, None] * WIDTH
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
        triton.cdiv(indexes.numel(), manager.cache_config.history_token_alignment)
        * manager.cache_config.history_token_alignment
        if round_tail
        else indexes.numel()
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
    checkpoints,
    checkpoint_ids,
    pages,
    page_ids,
    readies,
    CHECKPOINT_STRIDE: tl.constexpr,
    STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    NBYTES: tl.constexpr,
    TO_PAGES: tl.constexpr,
    HAS_READIES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    if HAS_READIES:
        if tl.load(readies + page):
            return
    checkpoint = checkpoints + tl.load(checkpoint_ids + page).to(tl.int64) * CHECKPOINT_STRIDE
    cpu_page = pages + tl.load(page_ids + page).to(tl.int64) * STRIDE + OFFSET
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if TO_PAGES:
        tl.store(cpu_page + offset, tl.load(checkpoint + offset, offset < NBYTES, 0), offset < NBYTES)
    else:
        tl.store(checkpoint + offset, tl.load(cpu_page + offset, offset < NBYTES, 0), offset < NBYTES)


def copy_checkpoint_pages(manager, checkpoint_ids, page_ids, pages, *, to_pages, readies=None):
    checkpoints = manager.big_page_buffers.buffer
    assert checkpoint_ids.numel() == page_ids.numel()
    nbytes = checkpoints.shape[-1]
    _copy_checkpoints[(page_ids.numel(), triton.cdiv(nbytes, 8192))](
        checkpoints,
        checkpoint_ids,
        pages,
        page_ids,
        readies,
        CHECKPOINT_STRIDE=checkpoints.stride(0),
        STRIDE=pages.stride(0),
        OFFSET=manager.cache_config.get_cpu_cache_full_att_bytes(),
        NBYTES=nbytes,
        TO_PAGES=to_pages,
        HAS_READIES=readies is not None,
        BLOCK=8192,
        num_warps=4,
    )
