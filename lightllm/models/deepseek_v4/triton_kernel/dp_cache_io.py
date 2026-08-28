import torch
import triton
import triton.language as tl

from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_PROMPT_CACHE_PAGE_SIZE


_C4_RATIO = 4
_C128_RATIO = 128
_BYTE_BLOCK = 8192
# Per-source row: c4 map/data/indexer, c128 map/data, SWA map/data, c4 state/indexer state.
_SOURCE_POOL_PTR_COUNT = 9
# Per-task row: source manager index, token count, source full-slot pointer, destination full-slot pointer.
_TASK_META_WIDTH = 4


@triton.jit
def _copy_dsv4_dp_caches_kernel(
    source_pool_ptrs,
    task_meta,
    history_meta,
    dst_full_to_c4,
    dst_c4_pool,
    dst_c4_pool_stride0,
    dst_c4_pool_stride1,
    dst_c4_indexer_pool,
    dst_c4_indexer_pool_stride0,
    dst_c4_indexer_pool_stride1,
    dst_full_to_c128,
    dst_c128_pool,
    dst_c128_pool_stride0,
    dst_c128_pool_stride1,
    dst_full_to_swa,
    dst_swa_pool,
    dst_swa_pool_stride0,
    dst_swa_pool_stride1,
    dst_c4_state,
    dst_c4_state_stride0,
    dst_c4_state_stride1,
    dst_c4_indexer_state,
    dst_c4_indexer_state_stride0,
    dst_c4_indexer_state_stride1,
    history_program_num,
    task_num,
    source_pool_ptr_count: tl.constexpr,
    task_meta_width: tl.constexpr,
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
    c4_state_ring: tl.constexpr,
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
            history_index = pid // history_layer_num
            layer = pid % history_layer_num
            task = tl.load(history_meta + history_index * 2).to(tl.int64)
            history_block = tl.load(history_meta + history_index * 2 + 1).to(tl.int64)
            task_row = task_meta + task * task_meta_width
            source_manager = tl.load(task_row).to(tl.int64)
            src_full_slots = tl.load(task_row + 2).to(tl.pointer_type(tl.int32))
            dst_full_slots = tl.load(task_row + 3).to(tl.pointer_type(tl.int32))
            source_ptr_row = source_pool_ptrs + source_manager * source_pool_ptr_count
            layer_i64 = layer.to(tl.int64)

            if HAS_C4:
                if layer < c4_layer_num:
                    src_full_to_c4 = tl.load(source_ptr_row).to(tl.pointer_type(tl.int32))
                    src_c4_pool = tl.load(source_ptr_row + 1).to(tl.pointer_type(tl.uint8))
                    src_c4_indexer_pool = tl.load(source_ptr_row + 2).to(tl.pointer_type(tl.uint8))
                    full_offset = history_block * history_block_size + c4_ratio - 1
                    src_full_slot = tl.load(src_full_slots + full_offset).to(tl.int64)
                    dst_full_slot = tl.load(dst_full_slots + full_offset).to(tl.int64)
                    src_pool_slot = tl.load(src_full_to_c4 + src_full_slot).to(tl.int64)
                    dst_pool_slot = tl.load(dst_full_to_c4 + dst_full_slot).to(tl.int64)
                    src_page = src_pool_slot // c4_pool_page_size
                    dst_page = dst_pool_slot // c4_pool_page_size

                    src_c4_page = src_c4_pool + layer_i64 * dst_c4_pool_stride0 + src_page * dst_c4_pool_stride1
                    dst_c4_page = dst_c4_pool + layer_i64 * dst_c4_pool_stride0 + dst_page * dst_c4_pool_stride1
                    src_indexer_page = (
                        src_c4_indexer_pool
                        + layer_i64 * dst_c4_indexer_pool_stride0
                        + src_page * dst_c4_indexer_pool_stride1
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
                    src_full_to_c128 = tl.load(source_ptr_row + 3).to(tl.pointer_type(tl.int32))
                    src_c128_pool = tl.load(source_ptr_row + 4).to(tl.pointer_type(tl.uint8))
                    for row in tl.static_range(0, 2):
                        full_offset = history_block * history_block_size + (row + 1) * c128_ratio - 1
                        src_full_slot = tl.load(src_full_slots + full_offset).to(tl.int64)
                        dst_full_slot = tl.load(dst_full_slots + full_offset).to(tl.int64)
                        src_pool_slot = tl.load(src_full_to_c128 + src_full_slot).to(tl.int64)
                        dst_pool_slot = tl.load(dst_full_to_c128 + dst_full_slot).to(tl.int64)
                        src_page = src_pool_slot // c128_pool_page_size
                        dst_page = dst_pool_slot // c128_pool_page_size
                        src_token = src_pool_slot % c128_pool_page_size
                        dst_token = dst_pool_slot % c128_pool_page_size

                        src_page_ptr = (
                            src_c128_pool + layer_i64 * dst_c128_pool_stride0 + src_page * dst_c128_pool_stride1
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

    tail_pid = pid - history_program_num
    if tail_pid >= 0:
        task = (tail_pid % task_num).to(tl.int64)
        task_pid = tail_pid // task_num
        task_row = task_meta + task * task_meta_width
        source_manager = tl.load(task_row).to(tl.int64)
        token_num = tl.load(task_row + 1).to(tl.int64)
        src_full_slots = tl.load(task_row + 2).to(tl.pointer_type(tl.int32))
        dst_full_slots = tl.load(task_row + 3).to(tl.pointer_type(tl.int32))
        source_ptr_row = source_pool_ptrs + source_manager * source_pool_ptr_count
        src_full_to_swa = tl.load(source_ptr_row + 5).to(tl.pointer_type(tl.int32))
        src_swa_pool = tl.load(source_ptr_row + 6).to(tl.pointer_type(tl.uint8))

        if task_pid < swa_program_num:
            page = task_pid % 2
            layer = task_pid // 2
            page_i64 = page.to(tl.int64)
            layer_i64 = layer.to(tl.int64)
            full_offset = token_num - history_block_size + page_i64 * swa_pool_page_size
            src_full_slot = tl.load(src_full_slots + full_offset).to(tl.int64)
            dst_full_slot = tl.load(dst_full_slots + full_offset).to(tl.int64)
            src_pool_slot = tl.load(src_full_to_swa + src_full_slot).to(tl.int64)
            dst_pool_slot = tl.load(dst_full_to_swa + dst_full_slot).to(tl.int64)
            src_page = src_pool_slot // swa_pool_page_size
            dst_page = dst_pool_slot // swa_pool_page_size
            src_page_ptr = src_swa_pool + layer_i64 * dst_swa_pool_stride0 + src_page * dst_swa_pool_stride1
            dst_page_ptr = dst_swa_pool + layer_i64 * dst_swa_pool_stride0 + dst_page * dst_swa_pool_stride1
            for byte_start in tl.range(0, swa_pool_page_nbytes, BLOCK):
                offsets = byte_start + lanes
                offsets_i64 = offsets.to(tl.int64)
                mask = offsets < swa_pool_page_nbytes
                tl.store(dst_page_ptr + offsets_i64, tl.load(src_page_ptr + offsets_i64, mask=mask), mask=mask)

        if HAS_C4:
            state_pid = task_pid - swa_program_num
            if state_pid >= 0:
                row = state_pid % 4
                layer = state_pid // 4
                row_i64 = row.to(tl.int64)
                layer_i64 = layer.to(tl.int64)
                src_c4_state = tl.load(source_ptr_row + 7).to(tl.pointer_type(tl.uint8))
                src_c4_indexer_state = tl.load(source_ptr_row + 8).to(tl.pointer_type(tl.uint8))
                full_offset = token_num - 4 + row_i64
                src_full_slot = tl.load(src_full_slots + full_offset).to(tl.int64)
                dst_full_slot = tl.load(dst_full_slots + full_offset).to(tl.int64)
                src_swa_slot = tl.load(src_full_to_swa + src_full_slot).to(tl.int64)
                dst_swa_slot = tl.load(dst_full_to_swa + dst_full_slot).to(tl.int64)
                src_state_row = (src_swa_slot // swa_pool_page_size) * c4_state_ring + src_swa_slot % c4_state_ring
                dst_state_row = (dst_swa_slot // swa_pool_page_size) * c4_state_ring + dst_swa_slot % c4_state_ring

                src_state_row_ptr = (
                    src_c4_state + layer_i64 * dst_c4_state_stride0 + src_state_row * dst_c4_state_stride1
                )
                dst_state_row_ptr = (
                    dst_c4_state + layer_i64 * dst_c4_state_stride0 + dst_state_row * dst_c4_state_stride1
                )
                src_indexer_row_ptr = (
                    src_c4_indexer_state
                    + layer_i64 * dst_c4_indexer_state_stride0
                    + src_state_row * dst_c4_indexer_state_stride1
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


def copy_dsv4_dp_caches(
    source_pool_ptrs: torch.Tensor,
    dst_mem_manager,
    task_meta: torch.Tensor,
    history_meta: torch.Tensor,
) -> None:
    """Copy aligned DP suffixes; history_meta rows are (task index, local 256-token block)."""
    task_num = task_meta.numel() // _TASK_META_WIDTH
    history_layer_num = max(dst_mem_manager.n_c4, dst_mem_manager.n_c128)
    history_program_num = history_meta.numel() // 2 * history_layer_num
    swa_program_num = dst_mem_manager.layer_num * 2
    c4_state_program_num = dst_mem_manager.n_c4 * 4

    has_c4 = dst_mem_manager.c4_pool is not None
    has_c128 = dst_mem_manager.c128_pool is not None
    dst_c4_pool = dst_mem_manager.c4_pool.buffer if has_c4 else None
    dst_c4_indexer_pool = dst_mem_manager.c4_indexer_pool.buffer if has_c4 else None
    dst_c128_pool = dst_mem_manager.c128_pool.buffer if has_c128 else None
    dst_swa_pool = dst_mem_manager.swa_pool.buffer
    dst_c4_state = dst_mem_manager.c4_state_buffer.view(torch.uint8) if has_c4 else None
    dst_c4_indexer_state = dst_mem_manager.c4_indexer_state_buffer.view(torch.uint8) if has_c4 else None

    c4_page_nbytes = dst_mem_manager.c4_pool.bytes_per_page if has_c4 else 0
    c4_indexer_page_nbytes = dst_mem_manager.c4_indexer_pool.bytes_per_page if has_c4 else 0
    c4_state_row_nbytes = dst_c4_state.shape[-1] if has_c4 else 0
    c4_indexer_state_row_nbytes = dst_c4_indexer_state.shape[-1] if has_c4 else 0
    program_num = history_program_num + task_num * (swa_program_num + c4_state_program_num)

    _copy_dsv4_dp_caches_kernel[(program_num,)](
        source_pool_ptrs,
        task_meta,
        history_meta,
        dst_mem_manager.full_to_c4_indexs if has_c4 else None,
        dst_c4_pool,
        dst_c4_pool.stride(0) if has_c4 else 0,
        dst_c4_pool.stride(1) if has_c4 else 0,
        dst_c4_indexer_pool,
        dst_c4_indexer_pool.stride(0) if has_c4 else 0,
        dst_c4_indexer_pool.stride(1) if has_c4 else 0,
        dst_mem_manager.full_to_c128_indexs if has_c128 else None,
        dst_c128_pool,
        dst_c128_pool.stride(0) if has_c128 else 0,
        dst_c128_pool.stride(1) if has_c128 else 0,
        dst_mem_manager.full_to_swa_indexs,
        dst_swa_pool,
        dst_swa_pool.stride(0),
        dst_swa_pool.stride(1),
        dst_c4_state,
        dst_c4_state.stride(0) if has_c4 else 0,
        dst_c4_state.stride(1) if has_c4 else 0,
        dst_c4_indexer_state,
        dst_c4_indexer_state.stride(0) if has_c4 else 0,
        dst_c4_indexer_state.stride(1) if has_c4 else 0,
        history_program_num,
        task_num,
        source_pool_ptr_count=_SOURCE_POOL_PTR_COUNT,
        task_meta_width=_TASK_META_WIDTH,
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
        c4_state_ring=dst_mem_manager.c4_state_ring,
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
