import torch
import triton
import triton.language as tl

from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_PROMPT_CACHE_PAGE_SIZE


_C4_RATIO = 4
_C128_RATIO = 128
_BYTE_BLOCK = 8192


@triton.jit
def _copy_dsv4_dp_cache_kernel(
    src_full_slots,
    dst_full_slots,
    src_full_to_c4,
    dst_full_to_c4,
    src_c4_pool,
    src_c4_pool_stride0,
    src_c4_pool_stride1,
    dst_c4_pool,
    dst_c4_pool_stride0,
    dst_c4_pool_stride1,
    src_c4_indexer_pool,
    src_c4_indexer_pool_stride0,
    src_c4_indexer_pool_stride1,
    dst_c4_indexer_pool,
    dst_c4_indexer_pool_stride0,
    dst_c4_indexer_pool_stride1,
    src_full_to_c128,
    dst_full_to_c128,
    src_c128_pool,
    src_c128_pool_stride0,
    src_c128_pool_stride1,
    dst_c128_pool,
    dst_c128_pool_stride0,
    dst_c128_pool_stride1,
    src_full_to_swa,
    dst_full_to_swa,
    src_swa_pool,
    src_swa_pool_stride0,
    src_swa_pool_stride1,
    dst_swa_pool,
    dst_swa_pool_stride0,
    dst_swa_pool_stride1,
    src_c4_state,
    src_c4_state_stride0,
    src_c4_state_stride1,
    dst_c4_state,
    dst_c4_state_stride0,
    dst_c4_state_stride1,
    src_c4_indexer_state,
    src_c4_indexer_state_stride0,
    src_c4_indexer_state_stride1,
    dst_c4_indexer_state,
    dst_c4_indexer_state_stride0,
    dst_c4_indexer_state_stride1,
    token_num,
    history_program_num,
    history_block_size: tl.constexpr,
    history_layer_num: tl.constexpr,
    c4_ratio: tl.constexpr,
    c4_pool_page_size: tl.constexpr,
    c4_layer_num: tl.constexpr,
    c4_pool_page_nbytes: tl.constexpr,
    c4_indexer_pool_page_nbytes: tl.constexpr,
    c4_copy_nbytes: tl.constexpr,
    c128_ratio: tl.constexpr,
    c128_layer_num: tl.constexpr,
    c128_pool_page_size: tl.constexpr,
    c128_data_nbytes: tl.constexpr,
    c128_scale_nbytes: tl.constexpr,
    c128_scale_offset: tl.constexpr,
    swa_pool_page_size: tl.constexpr,
    swa_pool_page_nbytes: tl.constexpr,
    swa_program_num: tl.constexpr,
    src_c4_state_ring: tl.constexpr,
    dst_c4_state_ring: tl.constexpr,
    c4_state_row_nbytes: tl.constexpr,
    c4_indexer_state_row_nbytes: tl.constexpr,
    c4_state_copy_nbytes: tl.constexpr,
    HAS_HISTORY: tl.constexpr,
    HAS_C4: tl.constexpr,
    HAS_C128: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)

    if HAS_HISTORY:
        if pid < history_program_num:
            history_block = pid // history_layer_num
            layer = pid % history_layer_num
            history_block_i64 = history_block.to(tl.int64)
            layer_i64 = layer.to(tl.int64)

            if HAS_C4:
                if layer < c4_layer_num:
                    full_offset = history_block_i64 * history_block_size + c4_ratio - 1
                    src_full_slot = tl.load(src_full_slots + full_offset).to(tl.int64)
                    dst_full_slot = tl.load(dst_full_slots + full_offset).to(tl.int64)
                    src_pool_slot = tl.load(src_full_to_c4 + src_full_slot).to(tl.int64)
                    dst_pool_slot = tl.load(dst_full_to_c4 + dst_full_slot).to(tl.int64)
                    src_page = src_pool_slot // c4_pool_page_size
                    dst_page = dst_pool_slot // c4_pool_page_size

                    src_c4_page = src_c4_pool + layer_i64 * src_c4_pool_stride0 + src_page * src_c4_pool_stride1
                    dst_c4_page = dst_c4_pool + layer_i64 * dst_c4_pool_stride0 + dst_page * dst_c4_pool_stride1
                    src_indexer_page = (
                        src_c4_indexer_pool
                        + layer_i64 * src_c4_indexer_pool_stride0
                        + src_page * src_c4_indexer_pool_stride1
                    )
                    dst_indexer_page = (
                        dst_c4_indexer_pool
                        + layer_i64 * dst_c4_indexer_pool_stride0
                        + dst_page * dst_c4_indexer_pool_stride1
                    )
                    for byte_start in tl.range(0, c4_copy_nbytes, BLOCK):
                        offsets = byte_start + lanes
                        offsets_i64 = offsets.to(tl.int64)
                        c4_mask = offsets < c4_pool_page_nbytes
                        tl.store(
                            dst_c4_page + offsets_i64,
                            tl.load(src_c4_page + offsets_i64, mask=c4_mask),
                            mask=c4_mask,
                        )
                        indexer_mask = offsets < c4_indexer_pool_page_nbytes
                        tl.store(
                            dst_indexer_page + offsets_i64,
                            tl.load(src_indexer_page + offsets_i64, mask=indexer_mask),
                            mask=indexer_mask,
                        )

            if HAS_C128:
                if layer < c128_layer_num:
                    for row in tl.static_range(0, 2):
                        full_offset = history_block_i64 * history_block_size + (row + 1) * c128_ratio - 1
                        src_full_slot = tl.load(src_full_slots + full_offset).to(tl.int64)
                        dst_full_slot = tl.load(dst_full_slots + full_offset).to(tl.int64)
                        src_pool_slot = tl.load(src_full_to_c128 + src_full_slot).to(tl.int64)
                        dst_pool_slot = tl.load(dst_full_to_c128 + dst_full_slot).to(tl.int64)
                        src_page = src_pool_slot // c128_pool_page_size
                        dst_page = dst_pool_slot // c128_pool_page_size
                        src_token = src_pool_slot % c128_pool_page_size
                        dst_token = dst_pool_slot % c128_pool_page_size

                        src_page_ptr = (
                            src_c128_pool + layer_i64 * src_c128_pool_stride0 + src_page * src_c128_pool_stride1
                        )
                        dst_page_ptr = (
                            dst_c128_pool + layer_i64 * dst_c128_pool_stride0 + dst_page * dst_c128_pool_stride1
                        )
                        offsets_i64 = lanes.to(tl.int64)
                        data_mask = lanes < c128_data_nbytes
                        src_data = src_page_ptr + src_token * c128_data_nbytes + offsets_i64
                        dst_data = dst_page_ptr + dst_token * c128_data_nbytes + offsets_i64
                        tl.store(dst_data, tl.load(src_data, mask=data_mask), mask=data_mask)

                        scale_mask = lanes < c128_scale_nbytes
                        src_scale = src_page_ptr + c128_scale_offset + src_token * c128_scale_nbytes + offsets_i64
                        dst_scale = dst_page_ptr + c128_scale_offset + dst_token * c128_scale_nbytes + offsets_i64
                        tl.store(dst_scale, tl.load(src_scale, mask=scale_mask), mask=scale_mask)

    swa_pid = pid - history_program_num
    if (swa_pid >= 0) & (swa_pid < swa_program_num):
        page = swa_pid % 2
        layer = swa_pid // 2
        page_i64 = page.to(tl.int64)
        layer_i64 = layer.to(tl.int64)
        full_offset = token_num - history_block_size + page_i64 * swa_pool_page_size
        src_full_slot = tl.load(src_full_slots + full_offset).to(tl.int64)
        dst_full_slot = tl.load(dst_full_slots + full_offset).to(tl.int64)
        src_pool_slot = tl.load(src_full_to_swa + src_full_slot).to(tl.int64)
        dst_pool_slot = tl.load(dst_full_to_swa + dst_full_slot).to(tl.int64)
        src_page = src_pool_slot // swa_pool_page_size
        dst_page = dst_pool_slot // swa_pool_page_size
        src_page_ptr = src_swa_pool + layer_i64 * src_swa_pool_stride0 + src_page * src_swa_pool_stride1
        dst_page_ptr = dst_swa_pool + layer_i64 * dst_swa_pool_stride0 + dst_page * dst_swa_pool_stride1
        for byte_start in tl.range(0, swa_pool_page_nbytes, BLOCK):
            offsets = byte_start + lanes
            offsets_i64 = offsets.to(tl.int64)
            mask = offsets < swa_pool_page_nbytes
            tl.store(dst_page_ptr + offsets_i64, tl.load(src_page_ptr + offsets_i64, mask=mask), mask=mask)

    if HAS_C4:
        state_pid = swa_pid - swa_program_num
        if state_pid >= 0:
            row = state_pid % 4
            layer = state_pid // 4
            row_i64 = row.to(tl.int64)
            layer_i64 = layer.to(tl.int64)
            full_offset = token_num - 4 + row_i64
            src_full_slot = tl.load(src_full_slots + full_offset).to(tl.int64)
            dst_full_slot = tl.load(dst_full_slots + full_offset).to(tl.int64)
            src_swa_slot = tl.load(src_full_to_swa + src_full_slot).to(tl.int64)
            dst_swa_slot = tl.load(dst_full_to_swa + dst_full_slot).to(tl.int64)
            src_state_row = (src_swa_slot // swa_pool_page_size) * src_c4_state_ring + src_swa_slot % src_c4_state_ring
            dst_state_row = (dst_swa_slot // swa_pool_page_size) * dst_c4_state_ring + dst_swa_slot % dst_c4_state_ring

            src_state_row_ptr = src_c4_state + layer_i64 * src_c4_state_stride0 + src_state_row * src_c4_state_stride1
            dst_state_row_ptr = dst_c4_state + layer_i64 * dst_c4_state_stride0 + dst_state_row * dst_c4_state_stride1
            src_indexer_row_ptr = (
                src_c4_indexer_state
                + layer_i64 * src_c4_indexer_state_stride0
                + src_state_row * src_c4_indexer_state_stride1
            )
            dst_indexer_row_ptr = (
                dst_c4_indexer_state
                + layer_i64 * dst_c4_indexer_state_stride0
                + dst_state_row * dst_c4_indexer_state_stride1
            )
            for byte_start in tl.range(0, c4_state_copy_nbytes, BLOCK):
                offsets = byte_start + lanes
                offsets_i64 = offsets.to(tl.int64)
                state_mask = offsets < c4_state_row_nbytes
                tl.store(
                    dst_state_row_ptr + offsets_i64,
                    tl.load(src_state_row_ptr + offsets_i64, mask=state_mask),
                    mask=state_mask,
                )
                indexer_mask = offsets < c4_indexer_state_row_nbytes
                tl.store(
                    dst_indexer_row_ptr + offsets_i64,
                    tl.load(src_indexer_row_ptr + offsets_i64, mask=indexer_mask),
                    mask=indexer_mask,
                )


def copy_dsv4_dp_cache(
    src_mem_manager,
    dst_mem_manager,
    src_full_slots: torch.Tensor,
    dst_full_slots: torch.Tensor,
) -> None:
    """Copy one 256-token-aligned DP radix suffix directly between DSV4 GPU pools."""
    src_full_slots = src_full_slots.reshape(-1)
    dst_full_slots = dst_full_slots.reshape(-1)
    token_num = src_full_slots.numel()
    assert (
        token_num == dst_full_slots.numel()
        and token_num >= DSV4_PROMPT_CACHE_PAGE_SIZE
        and token_num % DSV4_PROMPT_CACHE_PAGE_SIZE == 0
    )

    has_c4 = dst_mem_manager.c4_pool is not None
    has_c128 = dst_mem_manager.c128_pool is not None
    history_layer_num = max(dst_mem_manager.n_c4, dst_mem_manager.n_c128)
    history_program_num = token_num // DSV4_PROMPT_CACHE_PAGE_SIZE * history_layer_num
    swa_program_num = dst_mem_manager.layer_num * 2
    c4_state_program_num = dst_mem_manager.n_c4 * 4

    src_c4_pool = src_mem_manager.c4_pool.buffer if has_c4 else None
    dst_c4_pool = dst_mem_manager.c4_pool.buffer if has_c4 else None
    src_c4_indexer_pool = src_mem_manager.c4_indexer_pool.buffer if has_c4 else None
    dst_c4_indexer_pool = dst_mem_manager.c4_indexer_pool.buffer if has_c4 else None
    src_c128_pool = src_mem_manager.c128_pool.buffer if has_c128 else None
    dst_c128_pool = dst_mem_manager.c128_pool.buffer if has_c128 else None
    src_swa_pool = src_mem_manager.swa_pool.buffer
    dst_swa_pool = dst_mem_manager.swa_pool.buffer
    src_c4_state = src_mem_manager.c4_state_buffer.view(torch.uint8) if has_c4 else None
    dst_c4_state = dst_mem_manager.c4_state_buffer.view(torch.uint8) if has_c4 else None
    src_c4_indexer_state = src_mem_manager.c4_indexer_state_buffer.view(torch.uint8) if has_c4 else None
    dst_c4_indexer_state = dst_mem_manager.c4_indexer_state_buffer.view(torch.uint8) if has_c4 else None

    c4_page_nbytes = dst_mem_manager.c4_pool.bytes_per_page if has_c4 else 0
    c4_indexer_page_nbytes = dst_mem_manager.c4_indexer_pool.bytes_per_page if has_c4 else 0
    c4_state_row_nbytes = dst_c4_state.shape[-1] if has_c4 else 0
    c4_indexer_state_row_nbytes = dst_c4_indexer_state.shape[-1] if has_c4 else 0

    _copy_dsv4_dp_cache_kernel[(history_program_num + swa_program_num + c4_state_program_num,)](
        src_full_slots,
        dst_full_slots,
        src_mem_manager.full_to_c4_indexs if has_c4 else None,
        dst_mem_manager.full_to_c4_indexs if has_c4 else None,
        src_c4_pool,
        src_c4_pool.stride(0) if has_c4 else 0,
        src_c4_pool.stride(1) if has_c4 else 0,
        dst_c4_pool,
        dst_c4_pool.stride(0) if has_c4 else 0,
        dst_c4_pool.stride(1) if has_c4 else 0,
        src_c4_indexer_pool,
        src_c4_indexer_pool.stride(0) if has_c4 else 0,
        src_c4_indexer_pool.stride(1) if has_c4 else 0,
        dst_c4_indexer_pool,
        dst_c4_indexer_pool.stride(0) if has_c4 else 0,
        dst_c4_indexer_pool.stride(1) if has_c4 else 0,
        src_mem_manager.full_to_c128_indexs if has_c128 else None,
        dst_mem_manager.full_to_c128_indexs if has_c128 else None,
        src_c128_pool,
        src_c128_pool.stride(0) if has_c128 else 0,
        src_c128_pool.stride(1) if has_c128 else 0,
        dst_c128_pool,
        dst_c128_pool.stride(0) if has_c128 else 0,
        dst_c128_pool.stride(1) if has_c128 else 0,
        src_mem_manager.full_to_swa_indexs,
        dst_mem_manager.full_to_swa_indexs,
        src_swa_pool,
        src_swa_pool.stride(0),
        src_swa_pool.stride(1),
        dst_swa_pool,
        dst_swa_pool.stride(0),
        dst_swa_pool.stride(1),
        src_c4_state,
        src_c4_state.stride(0) if has_c4 else 0,
        src_c4_state.stride(1) if has_c4 else 0,
        dst_c4_state,
        dst_c4_state.stride(0) if has_c4 else 0,
        dst_c4_state.stride(1) if has_c4 else 0,
        src_c4_indexer_state,
        src_c4_indexer_state.stride(0) if has_c4 else 0,
        src_c4_indexer_state.stride(1) if has_c4 else 0,
        dst_c4_indexer_state,
        dst_c4_indexer_state.stride(0) if has_c4 else 0,
        dst_c4_indexer_state.stride(1) if has_c4 else 0,
        token_num,
        history_program_num,
        history_block_size=DSV4_PROMPT_CACHE_PAGE_SIZE,
        history_layer_num=history_layer_num,
        c4_ratio=_C4_RATIO,
        c4_pool_page_size=dst_mem_manager.c4_pool.page_size if has_c4 else 0,
        c4_layer_num=dst_mem_manager.n_c4,
        c4_pool_page_nbytes=c4_page_nbytes,
        c4_indexer_pool_page_nbytes=c4_indexer_page_nbytes,
        c4_copy_nbytes=max(c4_page_nbytes, c4_indexer_page_nbytes),
        c128_ratio=_C128_RATIO,
        c128_layer_num=dst_mem_manager.n_c128,
        c128_pool_page_size=dst_mem_manager.c128_pool.page_size if has_c128 else 0,
        c128_data_nbytes=dst_mem_manager.c128_pool.data_bytes_per_token if has_c128 else 0,
        c128_scale_nbytes=dst_mem_manager.c128_pool.scale_bytes_per_token if has_c128 else 0,
        c128_scale_offset=dst_mem_manager.c128_pool.scale_offset_in_page if has_c128 else 0,
        swa_pool_page_size=dst_mem_manager.swa_pool.page_size,
        swa_pool_page_nbytes=dst_mem_manager.swa_pool.bytes_per_page,
        swa_program_num=swa_program_num,
        src_c4_state_ring=src_mem_manager.c4_state_ring,
        dst_c4_state_ring=dst_mem_manager.c4_state_ring,
        c4_state_row_nbytes=c4_state_row_nbytes,
        c4_indexer_state_row_nbytes=c4_indexer_state_row_nbytes,
        c4_state_copy_nbytes=max(c4_state_row_nbytes, c4_indexer_state_row_nbytes),
        HAS_HISTORY=history_layer_num > 0,
        HAS_C4=has_c4,
        HAS_C128=has_c128,
        BLOCK=_BYTE_BLOCK,
        num_warps=4,
        num_stages=1,
    )
