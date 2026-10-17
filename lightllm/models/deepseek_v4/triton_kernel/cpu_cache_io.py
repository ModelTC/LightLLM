"""Copy ordinary history pages without repacking their GPU representation."""

import triton
import triton.language as tl


@triton.jit
def _copy_history(
    history,
    pages,
    indexes,
    page_ids,
    readies,
    TOKEN_NUM: tl.constexpr,
    CPU_BLOCKS: tl.constexpr,
    GPU_BLOCKS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    C4_OFFSET: tl.constexpr,
    INDEXER_OFFSET: tl.constexpr,
    C128_OFFSET: tl.constexpr,
    C4_BYTES: tl.constexpr,
    INDEXER_BYTES: tl.constexpr,
    C128_BYTES: tl.constexpr,
    C4_LAYERS: tl.constexpr,
    C128_LAYERS: tl.constexpr,
    TO_PAGES: tl.constexpr,
    HAS_READIES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    history = history.to(tl.pointer_type(tl.uint8))
    job, layer, kind = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    page, local_block = job // CPU_BLOCKS, job % CPU_BLOCKS
    if job * 256 >= TOKEN_NUM:
        return
    if HAS_READIES:
        if tl.load(readies + page):
            return
    slot = tl.load(indexes + job * 256).to(tl.int64)
    if slot < 0:
        return
    gpu_block = slot // 256
    cpu_page = tl.load(page_ids + page).to(tl.int64)
    if kind == 0:
        gpu_offset, page_offset, width, layers = C4_OFFSET, 0, C4_BYTES, C4_LAYERS
    elif kind == 1:
        gpu_offset, page_offset, width, layers = (
            INDEXER_OFFSET,
            C4_LAYERS * CPU_BLOCKS * C4_BYTES,
            INDEXER_BYTES,
            C4_LAYERS,
        )
    else:
        gpu_offset = C128_OFFSET
        page_offset = C4_LAYERS * CPU_BLOCKS * (C4_BYTES + INDEXER_BYTES)
        width, layers = C128_BYTES, C128_LAYERS
    if layer < layers:
        gpu = history + gpu_offset + (layer * GPU_BLOCKS + gpu_block) * width
        cpu = pages + cpu_page * PAGE_STRIDE + page_offset + (layer * CPU_BLOCKS + local_block) * width
        for start in range(0, width, BLOCK):
            offset = start + tl.arange(0, BLOCK)
            if TO_PAGES:
                tl.store(cpu + offset, tl.load(gpu + offset, offset < width, 0), offset < width)
            else:
                tl.store(gpu + offset, tl.load(cpu + offset, offset < width, 0), offset < width)


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


def copy_history_pages(manager, indexes, page_ids, pages, *, to_pages, readies=None, page_tokens=None):
    page_tokens = manager.cache_config.cpu_cache_token_page_size if page_tokens is None else page_tokens
    if indexes.numel() and max(manager.n_c4, manager.n_c128):
        _copy_history[(triton.cdiv(indexes.numel(), 256), max(manager.n_c4, manager.n_c128), 3)](
            manager.history_buffer.data_ptr(),
            pages,
            indexes,
            page_ids,
            readies,
            TOKEN_NUM=indexes.numel(),
            CPU_BLOCKS=page_tokens // 256,
            GPU_BLOCKS=(manager.size + manager.page_size) // 256,
            PAGE_STRIDE=pages.stride(0),
            C4_OFFSET=manager.history_offsets[0],
            INDEXER_OFFSET=manager.history_offsets[1],
            C128_OFFSET=manager.history_offsets[2],
            C4_BYTES=manager.c4_kv_buffer.shape[-1],
            INDEXER_BYTES=manager.c4_indexer_k_buffer.shape[-1],
            C128_BYTES=manager.c128_kv_buffer.shape[-1],
            C4_LAYERS=manager.n_c4,
            C128_LAYERS=manager.n_c128,
            TO_PAGES=to_pages,
            HAS_READIES=readies is not None,
            BLOCK=8192,
            num_warps=4,
        )


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
