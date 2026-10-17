"""DP copies compressed history only; request state travels as a CPU checkpoint."""

import triton
import triton.language as tl


@triton.jit
def _copy_history(
    source_buffers,
    destination,
    source_slots,
    source_dp,
    destination_slots,
    TP_SIZE: tl.constexpr,
    RANK_IN_DP: tl.constexpr,
    C4_OFFSET: tl.constexpr,
    INDEXER_OFFSET: tl.constexpr,
    C128_OFFSET: tl.constexpr,
    C4_PAGE_BYTES: tl.constexpr,
    INDEXER_PAGE_BYTES: tl.constexpr,
    C128_PAGE_BYTES: tl.constexpr,
    HISTORY_PAGES: tl.constexpr,
    C4_LAYERS: tl.constexpr,
    C128_LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    job = tl.program_id(0)
    layer = tl.program_id(1)
    source_rank = tl.load(source_dp + job * 256) * TP_SIZE + RANK_IN_DP
    source = tl.load(source_buffers + source_rank).to(tl.pointer_type(tl.uint8))
    src_page = tl.load(source_slots + job * 256).to(tl.int64) // 256
    dst_page = tl.load(destination_slots + job * 256).to(tl.int64) // 256
    lanes = tl.arange(0, BLOCK)
    for kind in tl.static_range(3):
        if kind == 0:
            base, page_bytes, layers = C4_OFFSET, C4_PAGE_BYTES, C4_LAYERS
        elif kind == 1:
            base, page_bytes, layers = INDEXER_OFFSET, INDEXER_PAGE_BYTES, C4_LAYERS
        else:
            base, page_bytes, layers = C128_OFFSET, C128_PAGE_BYTES, C128_LAYERS
        if layer < layers:
            src = source + base + (layer * HISTORY_PAGES + src_page) * page_bytes
            dst = destination + base + (layer * HISTORY_PAGES + dst_page) * page_bytes
            for start in range(0, page_bytes, BLOCK):
                offsets = start + lanes
                value = tl.load(src + offsets, offsets < page_bytes, 0)
                tl.store(dst + offsets, value, offsets < page_bytes)


def copy_dsv4_dp_history(source_buffers, manager, source_slots, source_dp, destination_slots, tp_size, rank_in_dp):
    """Copy complete 256-token blocks from one base pointer per source rank."""
    layers = max(manager.n_c4, manager.n_c128)
    if not destination_slots.numel() or not layers:
        return
    c4, indexer, c128 = manager.c4_kv_buffer, manager.c4_indexer_k_buffer, manager.c128_kv_buffer
    _copy_history[(destination_slots.numel() // 256, layers)](
        source_buffers,
        manager.history_buffer,
        source_slots,
        source_dp,
        destination_slots,
        TP_SIZE=tp_size,
        RANK_IN_DP=rank_in_dp,
        C4_OFFSET=manager.history_offsets[0],
        INDEXER_OFFSET=manager.history_offsets[1],
        C128_OFFSET=manager.history_offsets[2],
        C4_PAGE_BYTES=c4.shape[-1],
        INDEXER_PAGE_BYTES=indexer.shape[-1],
        C128_PAGE_BYTES=c128.shape[-1],
        HISTORY_PAGES=(manager.size + manager.page_size) // 256,
        C4_LAYERS=manager.n_c4,
        C128_LAYERS=manager.n_c128,
        BLOCK=8192,
        num_warps=4,
        num_stages=1,
    )
