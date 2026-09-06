import torch
import triton
import triton.language as tl

from .cache_staging_io import (
    BYTE_BLOCK as _BYTE_BLOCK,
    STATE_BLOCK as _STATE_BLOCK,
    copy_pool_pages,
)

_C4_RATIO = 4
_C4_POOL_PAGE_SIZE = 64
_C4_TOKEN_BLOCK = _C4_RATIO * _C4_POOL_PAGE_SIZE
_C128_RATIO = 128
_SWA_PAGE_SIZE = 128


@triton.jit
def _pd_tail_kernel(
    full_slots,
    full_to_swa,
    swa_pool,
    swa_pool_stride0,
    swa_pool_stride1,
    c4_state,
    c4_state_stride0,
    c4_state_stride1,
    c4_indexer_state,
    c4_indexer_state_stride0,
    c4_indexer_state_stride1,
    c128_state,
    c128_state_stride0,
    c128_state_stride1,
    staging,
    staging_f32,
    swa_program_num,
    c4_program_num,
    swa_page_num,
    swa_first_full_offset,
    swa_section_gpu_page_start,
    c4_row_num,
    c4_first_full_offset,
    c4_section_row_start,
    c128_row_num,
    c128_first_position,
    c128_section_row_start,
    req_idx,
    swa_page_size: tl.constexpr,
    swa_page_nbytes: tl.constexpr,
    swa_section_offset: tl.constexpr,
    swa_section_layer_nbytes: tl.constexpr,
    swa_blocks_per_page: tl.constexpr,
    c4_state_ring: tl.constexpr,
    c4_state_width: tl.constexpr,
    c4_indexer_state_width: tl.constexpr,
    c4_state_section_offset_f32: tl.constexpr,
    c4_state_section_layer_elems: tl.constexpr,
    c4_indexer_state_section_offset_f32: tl.constexpr,
    c4_indexer_state_section_layer_elems: tl.constexpr,
    c4_blocks_per_row: tl.constexpr,
    c128_state_ring: tl.constexpr,
    c128_state_width: tl.constexpr,
    c128_state_section_offset_f32: tl.constexpr,
    c128_state_section_layer_elems: tl.constexpr,
    c128_blocks_per_row: tl.constexpr,
    HAS_C4_STATE: tl.constexpr,
    HAS_C128_STATE: tl.constexpr,
    MODE: tl.constexpr,
    BYTE_BLOCK: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < swa_program_num:
        byte_block = pid % swa_blocks_per_page
        job = pid // swa_blocks_per_page
        gpu_page = job % swa_page_num
        layer = job // swa_page_num
        gpu_page_i64 = gpu_page.to(tl.int64)
        layer_i64 = layer.to(tl.int64)
        full_slot = tl.load(full_slots + swa_first_full_offset + gpu_page_i64 * swa_page_size).to(tl.int64)
        swa_slot = tl.load(full_to_swa + full_slot).to(tl.int64)
        physical_page = swa_slot // swa_page_size
        offsets = byte_block * BYTE_BLOCK + tl.arange(0, BYTE_BLOCK)
        offsets_i64 = offsets.to(tl.int64)
        mask = offsets < swa_page_nbytes
        pool_ptr = swa_pool + layer_i64 * swa_pool_stride0 + physical_page * swa_pool_stride1 + offsets_i64
        staging_ptr = (
            staging
            + swa_section_offset
            + layer_i64 * swa_section_layer_nbytes
            + (swa_section_gpu_page_start + gpu_page_i64) * swa_page_nbytes
            + offsets_i64
        )
        if MODE == 0:
            tl.store(staging_ptr, tl.load(pool_ptr, mask=mask), mask=mask)
        else:
            tl.store(pool_ptr, tl.load(staging_ptr, mask=mask), mask=mask)
    else:
        state_pid = pid - swa_program_num
        if HAS_C4_STATE:
            if state_pid < c4_program_num:
                state_block = state_pid % c4_blocks_per_row
                job = state_pid // c4_blocks_per_row
                row = job % c4_row_num
                layer = job // c4_row_num
                row_i64 = row.to(tl.int64)
                layer_i64 = layer.to(tl.int64)
                full_slot = tl.load(full_slots + c4_first_full_offset + row_i64).to(tl.int64)
                swa_slot = tl.load(full_to_swa + full_slot).to(tl.int64)
                state_row = (swa_slot // swa_page_size) * c4_state_ring + swa_slot % c4_state_ring
                offsets = state_block * STATE_BLOCK + tl.arange(0, STATE_BLOCK)
                offsets_i64 = offsets.to(tl.int64)
                c4_mask = offsets < c4_state_width
                c4_state_ptr = c4_state + layer_i64 * c4_state_stride0 + state_row * c4_state_stride1 + offsets_i64
                c4_staging_ptr = (
                    staging_f32
                    + c4_state_section_offset_f32
                    + layer_i64 * c4_state_section_layer_elems
                    + (c4_section_row_start + row_i64) * c4_state_width
                    + offsets_i64
                )
                indexer_mask = offsets < c4_indexer_state_width
                indexer_state_ptr = (
                    c4_indexer_state
                    + layer_i64 * c4_indexer_state_stride0
                    + state_row * c4_indexer_state_stride1
                    + offsets_i64
                )
                indexer_staging_ptr = (
                    staging_f32
                    + c4_indexer_state_section_offset_f32
                    + layer_i64 * c4_indexer_state_section_layer_elems
                    + (c4_section_row_start + row_i64) * c4_indexer_state_width
                    + offsets_i64
                )
                if MODE == 0:
                    tl.store(c4_staging_ptr, tl.load(c4_state_ptr, mask=c4_mask), mask=c4_mask)
                    tl.store(
                        indexer_staging_ptr,
                        tl.load(indexer_state_ptr, mask=indexer_mask),
                        mask=indexer_mask,
                    )
                else:
                    tl.store(c4_state_ptr, tl.load(c4_staging_ptr, mask=c4_mask), mask=c4_mask)
                    tl.store(
                        indexer_state_ptr,
                        tl.load(indexer_staging_ptr, mask=indexer_mask),
                        mask=indexer_mask,
                    )

        if HAS_C128_STATE:
            c128_pid = state_pid - c4_program_num
            if c128_pid >= 0:
                state_block = c128_pid % c128_blocks_per_row
                job = c128_pid // c128_blocks_per_row
                row = job % c128_row_num
                layer = job // c128_row_num
                row_i64 = row.to(tl.int64)
                layer_i64 = layer.to(tl.int64)
                state_row = req_idx * c128_state_ring + (c128_first_position + row_i64) % c128_state_ring
                offsets = state_block * STATE_BLOCK + tl.arange(0, STATE_BLOCK)
                offsets_i64 = offsets.to(tl.int64)
                mask = offsets < c128_state_width
                state_ptr = c128_state + layer_i64 * c128_state_stride0 + state_row * c128_state_stride1 + offsets_i64
                staging_ptr = (
                    staging_f32
                    + c128_state_section_offset_f32
                    + layer_i64 * c128_state_section_layer_elems
                    + (c128_section_row_start + row_i64) * c128_state_width
                    + offsets_i64
                )
                if MODE == 0:
                    tl.store(staging_ptr, tl.load(state_ptr, mask=mask), mask=mask)
                else:
                    tl.store(state_ptr, tl.load(staging_ptr, mask=mask), mask=mask)


def _copy_pd_tail(
    mode,
    mem_manager,
    layout,
    full_slots,
    staging,
    start_kv_index,
    end_kv_index,
    request_kv_len,
    req_idx,
    swa_tail_start,
):
    swa_intersection_start = max(start_kv_index, swa_tail_start)
    swa_intersection_end = min(end_kv_index, request_kv_len)
    swa_page_start = swa_intersection_start // _SWA_PAGE_SIZE * _SWA_PAGE_SIZE
    swa_page_end = triton.cdiv(swa_intersection_end, _SWA_PAGE_SIZE) * _SWA_PAGE_SIZE
    swa_page_num = (swa_page_end - swa_page_start) // _SWA_PAGE_SIZE

    c4_state = None
    c4_indexer_state = None
    c4_row_num = 0
    c4_first_full_offset = 0
    c4_section_row_start = 0
    if mem_manager.c4_pool is not None:
        c4_remainder = request_kv_len % _C4_RATIO
        c4_required_rows = _C4_RATIO if c4_remainder == 0 else _C4_RATIO + c4_remainder
        c4_state_start = max(0, request_kv_len - c4_required_rows)
        c4_intersection_start = max(start_kv_index, c4_state_start)
        c4_intersection_end = min(end_kv_index, request_kv_len)
        if c4_intersection_start < c4_intersection_end:
            c4_state = mem_manager.c4_state_buffer
            c4_indexer_state = mem_manager.c4_indexer_state_buffer
            c4_row_num = c4_intersection_end - c4_intersection_start
            c4_first_full_offset = c4_intersection_start - start_kv_index
            c4_section_row_start = c4_intersection_start - c4_state_start

    c128_state = None
    c128_row_num = 0
    c128_first_position = 0
    c128_section_row_start = 0
    c128_partial_row_num = request_kv_len % _C128_RATIO
    if mem_manager.c128_pool is not None and c128_partial_row_num:
        c128_state_start = request_kv_len - c128_partial_row_num
        c128_intersection_start = max(start_kv_index, c128_state_start)
        c128_intersection_end = min(end_kv_index, request_kv_len)
        if c128_intersection_start < c128_intersection_end:
            c128_state = mem_manager.c128_state_buffer
            c128_row_num = c128_intersection_end - c128_intersection_start
            c128_first_position = c128_intersection_start
            c128_section_row_start = c128_intersection_start - c128_state_start

    swa_pool = mem_manager.swa_pool.buffer
    swa_blocks_per_page = triton.cdiv(swa_pool.shape[-1], _BYTE_BLOCK)
    swa_program_num = mem_manager.layer_num * swa_page_num * swa_blocks_per_page

    c4_state_width = c4_state.shape[-1] if c4_state is not None else 0
    c4_indexer_state_width = c4_indexer_state.shape[-1] if c4_indexer_state is not None else 0
    c4_blocks_per_row = (
        triton.cdiv(max(c4_state_width, c4_indexer_state_width), _STATE_BLOCK) if c4_state is not None else 0
    )
    c4_program_num = c4_state.shape[0] * c4_row_num * c4_blocks_per_row if c4_state is not None else 0

    c128_state_width = c128_state.shape[-1] if c128_state is not None else 0
    c128_blocks_per_row = triton.cdiv(c128_state_width, _STATE_BLOCK) if c128_state is not None else 0
    c128_program_num = c128_state.shape[0] * c128_row_num * c128_blocks_per_row if c128_state is not None else 0

    _pd_tail_kernel[(swa_program_num + c4_program_num + c128_program_num,)](
        full_slots,
        mem_manager.full_to_swa_indexs,
        swa_pool,
        swa_pool.stride(0),
        swa_pool.stride(1),
        c4_state,
        c4_state.stride(0) if c4_state is not None else 0,
        c4_state.stride(1) if c4_state is not None else 0,
        c4_indexer_state,
        c4_indexer_state.stride(0) if c4_indexer_state is not None else 0,
        c4_indexer_state.stride(1) if c4_indexer_state is not None else 0,
        c128_state,
        c128_state.stride(0) if c128_state is not None else 0,
        c128_state.stride(1) if c128_state is not None else 0,
        staging,
        staging.view(torch.float32),
        swa_program_num,
        c4_program_num,
        swa_page_num,
        swa_page_start - start_kv_index,
        (swa_page_start - swa_tail_start) // _SWA_PAGE_SIZE,
        c4_row_num,
        c4_first_full_offset,
        c4_section_row_start,
        c128_row_num,
        c128_first_position,
        c128_section_row_start,
        req_idx,
        swa_page_size=mem_manager.swa_pool.page_size,
        swa_page_nbytes=swa_pool.shape[-1],
        swa_section_offset=layout.swa_offset,
        swa_section_layer_nbytes=layout.swa_layer_nbytes,
        swa_blocks_per_page=swa_blocks_per_page,
        c4_state_ring=mem_manager.c4_state_ring,
        c4_state_width=c4_state_width,
        c4_indexer_state_width=c4_indexer_state_width,
        c4_state_section_offset_f32=layout.c4_state_offset // 4,
        c4_state_section_layer_elems=layout.c4_state_rows * c4_state_width,
        c4_indexer_state_section_offset_f32=layout.c4_indexer_state_offset // 4,
        c4_indexer_state_section_layer_elems=layout.c4_state_rows * c4_indexer_state_width,
        c4_blocks_per_row=c4_blocks_per_row,
        c128_state_ring=mem_manager.c128_state_ring,
        c128_state_width=c128_state_width,
        c128_state_section_offset_f32=layout.c128_state_offset // 4,
        c128_state_section_layer_elems=layout.c128_state_layer_nbytes // 4,
        c128_blocks_per_row=c128_blocks_per_row,
        HAS_C4_STATE=c4_state is not None,
        HAS_C128_STATE=c128_state is not None,
        MODE=0 if mode == "pack" else 1,
        BYTE_BLOCK=_BYTE_BLOCK,
        STATE_BLOCK=_STATE_BLOCK,
        num_warps=4,
    )


def _copy_pd_cache_page(
    mode,
    mem_manager,
    layout,
    mem_indexes: torch.Tensor,
    staging: torch.Tensor,
    start_kv_index: int,
    request_kv_len: int,
    req_idx: int,
) -> int:
    full_slots = mem_indexes.reshape(-1)
    end_kv_index = start_kv_index + full_slots.numel()
    staging = staging.reshape(-1)
    c4_entry_num = end_kv_index // _C4_RATIO - start_kv_index // _C4_RATIO
    c4_page_num = triton.cdiv(c4_entry_num, _C4_POOL_PAGE_SIZE) if c4_entry_num else 0
    c128_row_num = end_kv_index // _C128_RATIO - start_kv_index // _C128_RATIO
    c4_pool = mem_manager.c4_pool
    c128_pool = mem_manager.c128_pool
    c4_work = c4_pool is not None and c4_page_num > 0
    c128_work = c128_pool is not None and c128_row_num > 0
    if c4_work or c128_work:
        c4_indexer_pool = mem_manager.c4_indexer_pool if c4_work else None
        copy_pool_pages(
            mode,
            full_slots=full_slots,
            mapping=mem_manager.full_to_c4_indexs if c4_work else None,
            pool=c4_pool.buffer if c4_work else None,
            staging=staging,
            page_num=1,
            gpu_page_num=c4_page_num if c4_work else 0,
            layer_num=c4_pool.layer_num if c4_work else 0,
            full_slots_page_stride=0,
            staging_page_stride=0,
            first_full_offset=_C4_RATIO - 1,
            full_offset_per_gpu_page=_C4_TOKEN_BLOCK,
            pool_page_size=c4_pool.page_size if c4_work else 0,
            section_offset=layout.c4_offset,
            section_layer_nbytes=layout.c4_layer_nbytes,
            paired_pool=c4_indexer_pool.buffer if c4_work else None,
            paired_section_offset=layout.c4_indexer_offset,
            paired_section_layer_nbytes=layout.c4_indexer_layer_nbytes,
            c128_mapping=mem_manager.full_to_c128_indexs if c128_work else None,
            c128_pool=c128_pool if c128_work else None,
            c128_row_num=c128_row_num,
            c128_first_full_offset=_C128_RATIO - 1,
            c128_full_offset_per_row=_C128_RATIO,
            c128_row_nbytes=layout.c128_row_nbytes,
            c128_section_offset=layout.c128_offset,
            c128_section_layer_nbytes=layout.c128_layer_nbytes,
        )

    swa_tail_start = max(0, request_kv_len // _C4_TOKEN_BLOCK * _C4_TOKEN_BLOCK - _C4_TOKEN_BLOCK)
    if end_kv_index <= swa_tail_start:
        return layout.swa_offset

    _copy_pd_tail(
        mode,
        mem_manager,
        layout,
        full_slots,
        staging,
        start_kv_index,
        end_kv_index,
        request_kv_len,
        req_idx,
        swa_tail_start,
    )
    return layout.page_nbytes


def pack_pd_cache_page(
    mem_manager,
    layout,
    mem_indexes: torch.Tensor,
    staging: torch.Tensor,
    start_kv_index: int,
    request_kv_len: int,
    req_idx: int,
) -> int:
    return _copy_pd_cache_page(
        "pack",
        mem_manager,
        layout,
        mem_indexes,
        staging,
        start_kv_index,
        request_kv_len,
        req_idx,
    )


def unpack_pd_cache_page(
    mem_manager,
    layout,
    mem_indexes: torch.Tensor,
    staging: torch.Tensor,
    start_kv_index: int,
    request_kv_len: int,
    req_idx: int,
) -> None:
    _copy_pd_cache_page(
        "unpack",
        mem_manager,
        layout,
        mem_indexes,
        staging,
        start_kv_index,
        request_kv_len,
        req_idx,
    )
