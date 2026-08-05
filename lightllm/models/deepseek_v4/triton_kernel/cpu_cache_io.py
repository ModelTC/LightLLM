import torch
import triton
import triton.language as tl

from .cache_staging_io import BYTE_BLOCK as _BYTE_BLOCK

_HISTORY_BLOCK_SIZE = 256
_C128_RATIO = 128


@triton.jit
def _scatter_staging_to_cpu_kernel(
    staging,
    staging_stride0,
    cpu_pages,
    cpu_stride0,
    page_indexes,
    page_nbytes: tl.constexpr,
    blocks_per_page: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    byte_block = pid % blocks_per_page
    logical_page = pid // blocks_per_page
    logical_page_i64 = logical_page.to(tl.int64)
    cpu_page = tl.load(page_indexes + logical_page_i64).to(tl.int64)

    offsets = byte_block * BLOCK + tl.arange(0, BLOCK)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < page_nbytes
    source = staging + logical_page_i64 * staging_stride0 + offsets_i64
    target = cpu_pages + cpu_page * cpu_stride0 + offsets_i64
    tl.store(target, tl.load(source, mask=mask), mask=mask, cache_modifier=".wt")


@triton.jit
def _pack_gpu_cache_to_staging_kernel(
    full_slots,
    full_to_c4,
    c4_pool,
    c4_pool_stride0,
    c4_pool_stride1,
    c4_indexer_pool,
    c4_indexer_pool_stride0,
    c4_indexer_pool_stride1,
    full_to_c128,
    c128_pool,
    c128_pool_stride0,
    c128_pool_stride1,
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
    staging,
    staging_stride0,
    programs_per_page: tl.constexpr,
    c4_program_num: tl.constexpr,
    c128_program_num: tl.constexpr,
    swa_program_num: tl.constexpr,
    token_page_size: tl.constexpr,
    history_block_size: tl.constexpr,
    c128_ratio: tl.constexpr,
    c4_layer_num: tl.constexpr,
    c4_gpu_page_num: tl.constexpr,
    c4_pool_page_size: tl.constexpr,
    c4_gpu_page_nbytes: tl.constexpr,
    c4_section_offset: tl.constexpr,
    c4_section_layer_nbytes: tl.constexpr,
    c4_indexer_gpu_page_nbytes: tl.constexpr,
    c4_indexer_section_offset: tl.constexpr,
    c4_indexer_section_layer_nbytes: tl.constexpr,
    c4_blocks_per_gpu_page: tl.constexpr,
    c128_layer_num: tl.constexpr,
    c128_row_num: tl.constexpr,
    c128_pool_page_size: tl.constexpr,
    c128_data_nbytes: tl.constexpr,
    c128_scale_nbytes: tl.constexpr,
    c128_scale_offset: tl.constexpr,
    c128_row_nbytes: tl.constexpr,
    c128_section_offset: tl.constexpr,
    c128_section_layer_nbytes: tl.constexpr,
    layer_num: tl.constexpr,
    swa_gpu_page_num: tl.constexpr,
    swa_pool_page_size: tl.constexpr,
    swa_gpu_page_nbytes: tl.constexpr,
    swa_section_offset: tl.constexpr,
    swa_section_layer_nbytes: tl.constexpr,
    swa_blocks_per_gpu_page: tl.constexpr,
    c4_state_ring: tl.constexpr,
    c4_state_row_nbytes: tl.constexpr,
    c4_state_section_offset: tl.constexpr,
    c4_indexer_state_row_nbytes: tl.constexpr,
    c4_indexer_state_section_offset: tl.constexpr,
    c4_state_blocks_per_row: tl.constexpr,
    HAS_C4: tl.constexpr,
    HAS_C128: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    logical_page = pid // programs_per_page
    page_pid = pid % programs_per_page
    logical_page_i64 = logical_page.to(tl.int64)
    offsets_base = tl.arange(0, BLOCK)

    if HAS_C4:
        if page_pid < c4_program_num:
            byte_block = page_pid % c4_blocks_per_gpu_page
            job = page_pid // c4_blocks_per_gpu_page
            gpu_page = job % c4_gpu_page_num
            layer = job // c4_gpu_page_num
            layer_i64 = layer.to(tl.int64)
            gpu_page_i64 = gpu_page.to(tl.int64)
            full_slot = tl.load(
                full_slots + logical_page_i64 * token_page_size + 3 + gpu_page_i64 * history_block_size
            ).to(tl.int64)
            pool_slot = tl.load(full_to_c4 + full_slot).to(tl.int64)
            physical_page = pool_slot // c4_pool_page_size
            offsets = byte_block * BLOCK + offsets_base
            offsets_i64 = offsets.to(tl.int64)

            c4_mask = offsets < c4_gpu_page_nbytes
            c4_source = c4_pool + layer_i64 * c4_pool_stride0 + physical_page * c4_pool_stride1 + offsets_i64
            c4_target = (
                staging
                + logical_page_i64 * staging_stride0
                + c4_section_offset
                + layer_i64 * c4_section_layer_nbytes
                + gpu_page_i64 * c4_gpu_page_nbytes
                + offsets_i64
            )
            tl.store(c4_target, tl.load(c4_source, mask=c4_mask), mask=c4_mask)

            indexer_mask = offsets < c4_indexer_gpu_page_nbytes
            indexer_source = (
                c4_indexer_pool
                + layer_i64 * c4_indexer_pool_stride0
                + physical_page * c4_indexer_pool_stride1
                + offsets_i64
            )
            indexer_target = (
                staging
                + logical_page_i64 * staging_stride0
                + c4_indexer_section_offset
                + layer_i64 * c4_indexer_section_layer_nbytes
                + gpu_page_i64 * c4_indexer_gpu_page_nbytes
                + offsets_i64
            )
            tl.store(indexer_target, tl.load(indexer_source, mask=indexer_mask), mask=indexer_mask)

    c128_pid = page_pid - c4_program_num
    if HAS_C128:
        if (c128_pid >= 0) & (c128_pid < c128_program_num):
            row = c128_pid % c128_row_num
            layer = c128_pid // c128_row_num
            row_i64 = row.to(tl.int64)
            layer_i64 = layer.to(tl.int64)
            full_slot = tl.load(
                full_slots + logical_page_i64 * token_page_size + c128_ratio - 1 + row_i64 * c128_ratio
            ).to(tl.int64)
            pool_slot = tl.load(full_to_c128 + full_slot).to(tl.int64)
            physical_page = pool_slot // c128_pool_page_size
            token_in_page = pool_slot % c128_pool_page_size
            offsets_i64 = offsets_base.to(tl.int64)
            target = (
                staging
                + logical_page_i64 * staging_stride0
                + c128_section_offset
                + layer_i64 * c128_section_layer_nbytes
                + row_i64 * c128_row_nbytes
                + offsets_i64
            )
            pool_page = c128_pool + layer_i64 * c128_pool_stride0 + physical_page * c128_pool_stride1
            data_mask = offsets_base < c128_data_nbytes
            data_source = pool_page + token_in_page * c128_data_nbytes + offsets_i64
            tl.store(target, tl.load(data_source, mask=data_mask), mask=data_mask)

            scale_local = offsets_i64 - c128_data_nbytes
            scale_mask = (offsets_base >= c128_data_nbytes) & (offsets_base < c128_row_nbytes)
            scale_source = pool_page + c128_scale_offset + token_in_page * c128_scale_nbytes + scale_local
            tl.store(target, tl.load(scale_source, mask=scale_mask), mask=scale_mask)

    swa_pid = c128_pid - c128_program_num
    if (swa_pid >= 0) & (swa_pid < swa_program_num):
        byte_block = swa_pid % swa_blocks_per_gpu_page
        job = swa_pid // swa_blocks_per_gpu_page
        gpu_page = job % swa_gpu_page_num
        layer = job // swa_gpu_page_num
        layer_i64 = layer.to(tl.int64)
        gpu_page_i64 = gpu_page.to(tl.int64)
        full_slot = tl.load(
            full_slots
            + logical_page_i64 * token_page_size
            + token_page_size
            - history_block_size
            + gpu_page_i64 * swa_pool_page_size
        ).to(tl.int64)
        pool_slot = tl.load(full_to_swa + full_slot).to(tl.int64)
        physical_page = pool_slot // swa_pool_page_size
        offsets = byte_block * BLOCK + offsets_base
        offsets_i64 = offsets.to(tl.int64)
        mask = offsets < swa_gpu_page_nbytes
        source = swa_pool + layer_i64 * swa_pool_stride0 + physical_page * swa_pool_stride1 + offsets_i64
        target = (
            staging
            + logical_page_i64 * staging_stride0
            + swa_section_offset
            + layer_i64 * swa_section_layer_nbytes
            + gpu_page_i64 * swa_gpu_page_nbytes
            + offsets_i64
        )
        tl.store(target, tl.load(source, mask=mask), mask=mask)

    state_pid = swa_pid - swa_program_num
    if HAS_C4:
        if state_pid >= 0:
            byte_block = state_pid % c4_state_blocks_per_row
            job = state_pid // c4_state_blocks_per_row
            row = job % 4
            layer = job // 4
            row_i64 = row.to(tl.int64)
            layer_i64 = layer.to(tl.int64)
            full_slot = tl.load(full_slots + logical_page_i64 * token_page_size + token_page_size - 4 + row_i64).to(
                tl.int64
            )
            swa_slot = tl.load(full_to_swa + full_slot).to(tl.int64)
            state_row = (swa_slot // swa_pool_page_size) * c4_state_ring + swa_slot % c4_state_ring
            offsets = byte_block * BLOCK + offsets_base
            offsets_i64 = offsets.to(tl.int64)

            state_mask = offsets < c4_state_row_nbytes
            state_source = c4_state + layer_i64 * c4_state_stride0 + state_row * c4_state_stride1 + offsets_i64
            state_target = (
                staging
                + logical_page_i64 * staging_stride0
                + c4_state_section_offset
                + (layer_i64 * 4 + row_i64) * c4_state_row_nbytes
                + offsets_i64
            )
            tl.store(state_target, tl.load(state_source, mask=state_mask), mask=state_mask)

            indexer_mask = offsets < c4_indexer_state_row_nbytes
            indexer_source = (
                c4_indexer_state
                + layer_i64 * c4_indexer_state_stride0
                + state_row * c4_indexer_state_stride1
                + offsets_i64
            )
            indexer_target = (
                staging
                + logical_page_i64 * staging_stride0
                + c4_indexer_state_section_offset
                + (layer_i64 * 4 + row_i64) * c4_indexer_state_row_nbytes
                + offsets_i64
            )
            tl.store(indexer_target, tl.load(indexer_source, mask=indexer_mask), mask=indexer_mask)


@triton.jit
def _unpack_cpu_cache_to_gpu_kernel(
    history_c4_slots,
    history_c128_slots,
    resume_swa_slots,
    c4_pool,
    c4_pool_stride0,
    c4_pool_stride1,
    c4_indexer_pool,
    c4_indexer_pool_stride0,
    c4_indexer_pool_stride1,
    c128_pool,
    c128_pool_stride0,
    c128_pool_stride1,
    swa_pool,
    swa_pool_stride0,
    swa_pool_stride1,
    c4_state,
    c4_state_stride0,
    c4_state_stride1,
    c4_indexer_state,
    c4_indexer_state_stride0,
    c4_indexer_state_stride1,
    cpu_pages,
    cpu_stride0,
    cpu_page_indexes,
    cpu_page_num,
    first_history_block,
    c4_program_num,
    c128_program_num,
    c4_layer_num: tl.constexpr,
    c4_pool_page_size: tl.constexpr,
    c4_gpu_page_nbytes: tl.constexpr,
    c4_section_offset: tl.constexpr,
    c4_section_layer_nbytes: tl.constexpr,
    c4_indexer_gpu_page_nbytes: tl.constexpr,
    c4_indexer_section_offset: tl.constexpr,
    c4_indexer_section_layer_nbytes: tl.constexpr,
    c4_blocks_per_gpu_page: tl.constexpr,
    c128_layer_num: tl.constexpr,
    c128_pool_page_size: tl.constexpr,
    c128_data_nbytes: tl.constexpr,
    c128_scale_nbytes: tl.constexpr,
    c128_scale_offset: tl.constexpr,
    c128_row_nbytes: tl.constexpr,
    c128_section_offset: tl.constexpr,
    c128_section_layer_nbytes: tl.constexpr,
    blocks_per_cpu_page: tl.constexpr,
    layer_num: tl.constexpr,
    swa_program_num: tl.constexpr,
    swa_gpu_page_num: tl.constexpr,
    swa_pool_page_size: tl.constexpr,
    swa_gpu_page_nbytes: tl.constexpr,
    swa_section_offset: tl.constexpr,
    swa_section_layer_nbytes: tl.constexpr,
    swa_blocks_per_gpu_page: tl.constexpr,
    c4_state_ring: tl.constexpr,
    c4_state_row_nbytes: tl.constexpr,
    c4_state_section_offset: tl.constexpr,
    c4_indexer_state_row_nbytes: tl.constexpr,
    c4_indexer_state_section_offset: tl.constexpr,
    c4_state_blocks_per_row: tl.constexpr,
    HAS_C4: tl.constexpr,
    HAS_C128: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_base = tl.arange(0, BLOCK)

    if HAS_C4:
        if pid < c4_program_num:
            byte_block = pid % c4_blocks_per_gpu_page
            job = pid // c4_blocks_per_gpu_page
            layer = job % c4_layer_num
            history_block = job // c4_layer_num
            history_block_i64 = history_block.to(tl.int64)
            layer_i64 = layer.to(tl.int64)
            absolute_block = history_block_i64 + first_history_block
            cpu_page_list_index = absolute_block // blocks_per_cpu_page
            block_in_cpu_page = absolute_block % blocks_per_cpu_page
            cpu_page = tl.load(cpu_page_indexes + cpu_page_list_index).to(tl.int64)
            pool_slot = tl.load(history_c4_slots + history_block_i64 * c4_pool_page_size).to(tl.int64)
            physical_page = pool_slot // c4_pool_page_size
            offsets = byte_block * BLOCK + offsets_base
            offsets_i64 = offsets.to(tl.int64)

            c4_mask = offsets < c4_gpu_page_nbytes
            c4_source = (
                cpu_pages
                + cpu_page * cpu_stride0
                + c4_section_offset
                + layer_i64 * c4_section_layer_nbytes
                + block_in_cpu_page * c4_gpu_page_nbytes
                + offsets_i64
            )
            c4_target = c4_pool + layer_i64 * c4_pool_stride0 + physical_page * c4_pool_stride1 + offsets_i64
            tl.store(c4_target, tl.load(c4_source, mask=c4_mask), mask=c4_mask)

            indexer_mask = offsets < c4_indexer_gpu_page_nbytes
            indexer_source = (
                cpu_pages
                + cpu_page * cpu_stride0
                + c4_indexer_section_offset
                + layer_i64 * c4_indexer_section_layer_nbytes
                + block_in_cpu_page * c4_indexer_gpu_page_nbytes
                + offsets_i64
            )
            indexer_target = (
                c4_indexer_pool
                + layer_i64 * c4_indexer_pool_stride0
                + physical_page * c4_indexer_pool_stride1
                + offsets_i64
            )
            tl.store(indexer_target, tl.load(indexer_source, mask=indexer_mask), mask=indexer_mask)

    c128_pid = pid - c4_program_num
    if HAS_C128:
        if (c128_pid >= 0) & (c128_pid < c128_program_num):
            row = c128_pid % 2
            job = c128_pid // 2
            layer = job % c128_layer_num
            history_block = job // c128_layer_num
            history_block_i64 = history_block.to(tl.int64)
            layer_i64 = layer.to(tl.int64)
            row_i64 = row.to(tl.int64)
            absolute_block = history_block_i64 + first_history_block
            cpu_page_list_index = absolute_block // blocks_per_cpu_page
            block_in_cpu_page = absolute_block % blocks_per_cpu_page
            cpu_page = tl.load(cpu_page_indexes + cpu_page_list_index).to(tl.int64)
            pool_slot = tl.load(history_c128_slots + history_block_i64 * 2 + row_i64).to(tl.int64)
            physical_page = pool_slot // c128_pool_page_size
            token_in_page = pool_slot % c128_pool_page_size
            offsets_i64 = offsets_base.to(tl.int64)
            cpu_row = block_in_cpu_page * 2 + row_i64
            source = (
                cpu_pages
                + cpu_page * cpu_stride0
                + c128_section_offset
                + layer_i64 * c128_section_layer_nbytes
                + cpu_row * c128_row_nbytes
                + offsets_i64
            )
            pool_page = c128_pool + layer_i64 * c128_pool_stride0 + physical_page * c128_pool_stride1
            data_mask = offsets_base < c128_data_nbytes
            data_target = pool_page + token_in_page * c128_data_nbytes + offsets_i64
            tl.store(data_target, tl.load(source, mask=data_mask), mask=data_mask)

            scale_local = offsets_i64 - c128_data_nbytes
            scale_mask = (offsets_base >= c128_data_nbytes) & (offsets_base < c128_row_nbytes)
            scale_target = pool_page + c128_scale_offset + token_in_page * c128_scale_nbytes + scale_local
            tl.store(scale_target, tl.load(source, mask=scale_mask), mask=scale_mask)

    swa_pid = c128_pid - c128_program_num
    if (swa_pid >= 0) & (swa_pid < swa_program_num):
        byte_block = swa_pid % swa_blocks_per_gpu_page
        job = swa_pid // swa_blocks_per_gpu_page
        gpu_page = job % swa_gpu_page_num
        layer = job // swa_gpu_page_num
        layer_i64 = layer.to(tl.int64)
        gpu_page_i64 = gpu_page.to(tl.int64)
        pool_slot = tl.load(resume_swa_slots + gpu_page_i64 * swa_pool_page_size).to(tl.int64)
        physical_page = pool_slot // swa_pool_page_size
        cpu_page = tl.load(cpu_page_indexes + cpu_page_num - 1).to(tl.int64)
        offsets = byte_block * BLOCK + offsets_base
        offsets_i64 = offsets.to(tl.int64)
        mask = offsets < swa_gpu_page_nbytes
        source = (
            cpu_pages
            + cpu_page * cpu_stride0
            + swa_section_offset
            + layer_i64 * swa_section_layer_nbytes
            + gpu_page_i64 * swa_gpu_page_nbytes
            + offsets_i64
        )
        target = swa_pool + layer_i64 * swa_pool_stride0 + physical_page * swa_pool_stride1 + offsets_i64
        tl.store(target, tl.load(source, mask=mask), mask=mask)

    state_pid = swa_pid - swa_program_num
    if HAS_C4:
        if state_pid >= 0:
            byte_block = state_pid % c4_state_blocks_per_row
            job = state_pid // c4_state_blocks_per_row
            row = job % 4
            layer = job // 4
            row_i64 = row.to(tl.int64)
            layer_i64 = layer.to(tl.int64)
            swa_slot = tl.load(resume_swa_slots + 2 * swa_pool_page_size - 4 + row_i64).to(tl.int64)
            state_row = (swa_slot // swa_pool_page_size) * c4_state_ring + swa_slot % c4_state_ring
            cpu_page = tl.load(cpu_page_indexes + cpu_page_num - 1).to(tl.int64)
            offsets = byte_block * BLOCK + offsets_base
            offsets_i64 = offsets.to(tl.int64)

            state_mask = offsets < c4_state_row_nbytes
            state_source = (
                cpu_pages
                + cpu_page * cpu_stride0
                + c4_state_section_offset
                + (layer_i64 * 4 + row_i64) * c4_state_row_nbytes
                + offsets_i64
            )
            state_target = c4_state + layer_i64 * c4_state_stride0 + state_row * c4_state_stride1 + offsets_i64
            tl.store(state_target, tl.load(state_source, mask=state_mask), mask=state_mask)

            indexer_mask = offsets < c4_indexer_state_row_nbytes
            indexer_source = (
                cpu_pages
                + cpu_page * cpu_stride0
                + c4_indexer_state_section_offset
                + (layer_i64 * 4 + row_i64) * c4_indexer_state_row_nbytes
                + offsets_i64
            )
            indexer_target = (
                c4_indexer_state
                + layer_i64 * c4_indexer_state_stride0
                + state_row * c4_indexer_state_stride1
                + offsets_i64
            )
            tl.store(indexer_target, tl.load(indexer_source, mask=indexer_mask), mask=indexer_mask)


def pack_gpu_cache_to_staging(mem_manager, source_mem_indexes: torch.Tensor, staging: torch.Tensor) -> None:
    """Pack a compact batch of complete CPU checkpoint pages into caller-owned CUDA staging."""
    layout = mem_manager.cpu_cache_layout
    assert source_mem_indexes.is_cuda and staging.is_cuda
    assert source_mem_indexes.ndim == 2 and source_mem_indexes.shape[1] == layout.token_page_size
    assert staging.dtype == torch.uint8 and staging.is_contiguous()
    page_num = source_mem_indexes.shape[0]
    assert staging.shape == (page_num, layout.page_nbytes)

    full_slots = source_mem_indexes.reshape(-1)
    has_c4 = mem_manager.c4_pool is not None
    has_c128 = mem_manager.c128_pool is not None
    c4_pool = mem_manager.c4_pool.buffer if has_c4 else None
    c4_indexer_pool = mem_manager.c4_indexer_pool.buffer if has_c4 else None
    c128_pool = mem_manager.c128_pool.buffer if has_c128 else None
    swa_pool = mem_manager.swa_pool.buffer
    c4_state = mem_manager.c4_state_buffer.view(torch.uint8) if has_c4 else None
    c4_indexer_state = mem_manager.c4_indexer_state_buffer.view(torch.uint8) if has_c4 else None

    c4_blocks_per_gpu_page = (
        triton.cdiv(max(layout.c4_gpu_page_nbytes, layout.c4_indexer_gpu_page_nbytes), _BYTE_BLOCK) if has_c4 else 0
    )
    c4_program_num = mem_manager.n_c4 * layout.c4_gpu_pages_per_page * c4_blocks_per_gpu_page
    c128_program_num = mem_manager.n_c128 * layout.c128_rows_per_page
    swa_blocks_per_gpu_page = triton.cdiv(layout.swa_gpu_page_nbytes, _BYTE_BLOCK)
    swa_program_num = mem_manager.layer_num * layout.swa_gpu_pages_per_page * swa_blocks_per_gpu_page
    c4_state_blocks_per_row = (
        triton.cdiv(max(layout.c4_state_row_nbytes, layout.c4_indexer_state_row_nbytes), _BYTE_BLOCK) if has_c4 else 0
    )
    c4_state_program_num = mem_manager.n_c4 * layout.c4_state_rows * c4_state_blocks_per_row
    programs_per_page = c4_program_num + c128_program_num + swa_program_num + c4_state_program_num

    _pack_gpu_cache_to_staging_kernel[(page_num * programs_per_page,)](
        full_slots,
        mem_manager.full_to_c4_indexs if has_c4 else None,
        c4_pool,
        c4_pool.stride(0) if has_c4 else 0,
        c4_pool.stride(1) if has_c4 else 0,
        c4_indexer_pool,
        c4_indexer_pool.stride(0) if has_c4 else 0,
        c4_indexer_pool.stride(1) if has_c4 else 0,
        mem_manager.full_to_c128_indexs if has_c128 else None,
        c128_pool,
        c128_pool.stride(0) if has_c128 else 0,
        c128_pool.stride(1) if has_c128 else 0,
        mem_manager.full_to_swa_indexs,
        swa_pool,
        swa_pool.stride(0),
        swa_pool.stride(1),
        c4_state,
        c4_state.stride(0) if has_c4 else 0,
        c4_state.stride(1) if has_c4 else 0,
        c4_indexer_state,
        c4_indexer_state.stride(0) if has_c4 else 0,
        c4_indexer_state.stride(1) if has_c4 else 0,
        staging,
        staging.stride(0),
        programs_per_page=programs_per_page,
        c4_program_num=c4_program_num,
        c128_program_num=c128_program_num,
        swa_program_num=swa_program_num,
        token_page_size=layout.token_page_size,
        history_block_size=_HISTORY_BLOCK_SIZE,
        c128_ratio=_C128_RATIO,
        c4_layer_num=mem_manager.n_c4,
        c4_gpu_page_num=layout.c4_gpu_pages_per_page,
        c4_pool_page_size=mem_manager.c4_pool.page_size if has_c4 else 0,
        c4_gpu_page_nbytes=layout.c4_gpu_page_nbytes,
        c4_section_offset=layout.c4_offset,
        c4_section_layer_nbytes=layout.c4_layer_nbytes,
        c4_indexer_gpu_page_nbytes=layout.c4_indexer_gpu_page_nbytes,
        c4_indexer_section_offset=layout.c4_indexer_offset,
        c4_indexer_section_layer_nbytes=layout.c4_indexer_layer_nbytes,
        c4_blocks_per_gpu_page=c4_blocks_per_gpu_page,
        c128_layer_num=mem_manager.n_c128,
        c128_row_num=layout.c128_rows_per_page,
        c128_pool_page_size=mem_manager.c128_pool.page_size if has_c128 else 0,
        c128_data_nbytes=mem_manager.c128_pool.data_bytes_per_token if has_c128 else 0,
        c128_scale_nbytes=mem_manager.c128_pool.scale_bytes_per_token if has_c128 else 0,
        c128_scale_offset=mem_manager.c128_pool.scale_offset_in_page if has_c128 else 0,
        c128_row_nbytes=layout.c128_row_nbytes,
        c128_section_offset=layout.c128_offset,
        c128_section_layer_nbytes=layout.c128_layer_nbytes,
        layer_num=mem_manager.layer_num,
        swa_gpu_page_num=layout.swa_gpu_pages_per_page,
        swa_pool_page_size=mem_manager.swa_pool.page_size,
        swa_gpu_page_nbytes=layout.swa_gpu_page_nbytes,
        swa_section_offset=layout.swa_offset,
        swa_section_layer_nbytes=layout.swa_layer_nbytes,
        swa_blocks_per_gpu_page=swa_blocks_per_gpu_page,
        c4_state_ring=mem_manager.c4_state_ring,
        c4_state_row_nbytes=layout.c4_state_row_nbytes,
        c4_state_section_offset=layout.c4_state_offset,
        c4_indexer_state_row_nbytes=layout.c4_indexer_state_row_nbytes,
        c4_indexer_state_section_offset=layout.c4_indexer_state_offset,
        c4_state_blocks_per_row=c4_state_blocks_per_row,
        HAS_C4=has_c4,
        HAS_C128=has_c128,
        BLOCK=_BYTE_BLOCK,
        num_warps=4,
    )


def scatter_staging_to_cpu_pages(
    staging: torch.Tensor,
    cpu_pages: torch.Tensor,
    page_indexes: torch.Tensor,
) -> None:
    """Copy packed staging rows to their pinned, mapped shared-memory pages."""
    assert staging.is_cuda and page_indexes.is_cuda
    assert staging.dtype == torch.uint8 and staging.is_contiguous() and cpu_pages.is_contiguous()
    page_num, page_nbytes = staging.shape
    assert page_indexes.numel() == page_num
    assert cpu_pages.ndim == 2 and cpu_pages.shape[1] == page_nbytes
    blocks_per_page = triton.cdiv(page_nbytes, _BYTE_BLOCK)
    _scatter_staging_to_cpu_kernel[(page_num * blocks_per_page,)](
        staging,
        staging.stride(0),
        cpu_pages,
        cpu_pages.stride(0),
        page_indexes,
        page_nbytes=page_nbytes,
        blocks_per_page=blocks_per_page,
        BLOCK=_BYTE_BLOCK,
        num_warps=4,
    )


def unpack_cpu_cache_to_gpu(
    mem_manager,
    load_plan,
    cpu_pages: torch.Tensor,
    cpu_page_indexes: torch.Tensor,
    first_page_history_offset_tokens: int = 0,
) -> None:
    """Restore missing 256-token history blocks and the final 256-token resume state.

    ``cpu_page_indexes`` names consecutive large CPU checkpoint pages. The first
    page may overlap an existing radix prefix; ``first_page_history_offset_tokens``
    selects its first missing 256-token block without copying the overlapped bytes.
    """
    layout = mem_manager.cpu_cache_layout
    assert load_plan.history_full_slots.is_cuda and cpu_page_indexes.is_cuda
    assert load_plan.history_full_slots.ndim == 2
    assert load_plan.history_full_slots.shape[1] == _HISTORY_BLOCK_SIZE
    assert load_plan.resume_swa_slots.shape == (_HISTORY_BLOCK_SIZE,)
    assert 0 <= first_page_history_offset_tokens < layout.token_page_size
    assert first_page_history_offset_tokens % _HISTORY_BLOCK_SIZE == 0
    assert cpu_pages.ndim == 2 and cpu_pages.shape[1] == layout.page_nbytes and cpu_pages.is_contiguous()
    assert cpu_page_indexes.numel() > 0

    history_block_num = load_plan.history_full_slots.shape[0]
    assert history_block_num > 0
    assert load_plan.loaded_end - load_plan.loaded_start == history_block_num * _HISTORY_BLOCK_SIZE
    assert load_plan.loaded_start % layout.token_page_size == first_page_history_offset_tokens
    blocks_per_cpu_page = layout.history_block_num
    first_history_block = first_page_history_offset_tokens // _HISTORY_BLOCK_SIZE
    expected_cpu_page_num = triton.cdiv(first_history_block + history_block_num, blocks_per_cpu_page)
    assert cpu_page_indexes.numel() == expected_cpu_page_num

    has_c4 = mem_manager.c4_pool is not None
    has_c128 = mem_manager.c128_pool is not None
    if has_c4:
        assert load_plan.history_c4_slots.shape == (history_block_num, mem_manager.c4_pool.page_size)
    if has_c128:
        assert load_plan.history_c128_slots.shape == (history_block_num, 2)

    c4_pool = mem_manager.c4_pool.buffer if has_c4 else None
    c4_indexer_pool = mem_manager.c4_indexer_pool.buffer if has_c4 else None
    c128_pool = mem_manager.c128_pool.buffer if has_c128 else None
    swa_pool = mem_manager.swa_pool.buffer
    c4_state = mem_manager.c4_state_buffer.view(torch.uint8) if has_c4 else None
    c4_indexer_state = mem_manager.c4_indexer_state_buffer.view(torch.uint8) if has_c4 else None

    c4_blocks_per_gpu_page = (
        triton.cdiv(max(layout.c4_gpu_page_nbytes, layout.c4_indexer_gpu_page_nbytes), _BYTE_BLOCK) if has_c4 else 0
    )
    c4_program_num = history_block_num * mem_manager.n_c4 * c4_blocks_per_gpu_page
    c128_program_num = history_block_num * mem_manager.n_c128 * 2
    swa_blocks_per_gpu_page = triton.cdiv(layout.swa_gpu_page_nbytes, _BYTE_BLOCK)
    swa_program_num = mem_manager.layer_num * layout.swa_gpu_pages_per_page * swa_blocks_per_gpu_page
    c4_state_blocks_per_row = (
        triton.cdiv(max(layout.c4_state_row_nbytes, layout.c4_indexer_state_row_nbytes), _BYTE_BLOCK) if has_c4 else 0
    )
    c4_state_program_num = mem_manager.n_c4 * layout.c4_state_rows * c4_state_blocks_per_row

    _unpack_cpu_cache_to_gpu_kernel[(c4_program_num + c128_program_num + swa_program_num + c4_state_program_num,)](
        load_plan.history_c4_slots if has_c4 else None,
        load_plan.history_c128_slots if has_c128 else None,
        load_plan.resume_swa_slots,
        c4_pool,
        c4_pool.stride(0) if has_c4 else 0,
        c4_pool.stride(1) if has_c4 else 0,
        c4_indexer_pool,
        c4_indexer_pool.stride(0) if has_c4 else 0,
        c4_indexer_pool.stride(1) if has_c4 else 0,
        c128_pool,
        c128_pool.stride(0) if has_c128 else 0,
        c128_pool.stride(1) if has_c128 else 0,
        swa_pool,
        swa_pool.stride(0),
        swa_pool.stride(1),
        c4_state,
        c4_state.stride(0) if has_c4 else 0,
        c4_state.stride(1) if has_c4 else 0,
        c4_indexer_state,
        c4_indexer_state.stride(0) if has_c4 else 0,
        c4_indexer_state.stride(1) if has_c4 else 0,
        cpu_pages,
        cpu_pages.stride(0),
        cpu_page_indexes,
        cpu_page_indexes.numel(),
        first_history_block,
        c4_program_num,
        c128_program_num,
        c4_layer_num=mem_manager.n_c4,
        c4_pool_page_size=mem_manager.c4_pool.page_size if has_c4 else 0,
        c4_gpu_page_nbytes=layout.c4_gpu_page_nbytes,
        c4_section_offset=layout.c4_offset,
        c4_section_layer_nbytes=layout.c4_layer_nbytes,
        c4_indexer_gpu_page_nbytes=layout.c4_indexer_gpu_page_nbytes,
        c4_indexer_section_offset=layout.c4_indexer_offset,
        c4_indexer_section_layer_nbytes=layout.c4_indexer_layer_nbytes,
        c4_blocks_per_gpu_page=c4_blocks_per_gpu_page,
        c128_layer_num=mem_manager.n_c128,
        c128_pool_page_size=mem_manager.c128_pool.page_size if has_c128 else 0,
        c128_data_nbytes=mem_manager.c128_pool.data_bytes_per_token if has_c128 else 0,
        c128_scale_nbytes=mem_manager.c128_pool.scale_bytes_per_token if has_c128 else 0,
        c128_scale_offset=mem_manager.c128_pool.scale_offset_in_page if has_c128 else 0,
        c128_row_nbytes=layout.c128_row_nbytes,
        c128_section_offset=layout.c128_offset,
        c128_section_layer_nbytes=layout.c128_layer_nbytes,
        blocks_per_cpu_page=blocks_per_cpu_page,
        layer_num=mem_manager.layer_num,
        swa_program_num=swa_program_num,
        swa_gpu_page_num=layout.swa_gpu_pages_per_page,
        swa_pool_page_size=mem_manager.swa_pool.page_size,
        swa_gpu_page_nbytes=layout.swa_gpu_page_nbytes,
        swa_section_offset=layout.swa_offset,
        swa_section_layer_nbytes=layout.swa_layer_nbytes,
        swa_blocks_per_gpu_page=swa_blocks_per_gpu_page,
        c4_state_ring=mem_manager.c4_state_ring,
        c4_state_row_nbytes=layout.c4_state_row_nbytes,
        c4_state_section_offset=layout.c4_state_offset,
        c4_indexer_state_row_nbytes=layout.c4_indexer_state_row_nbytes,
        c4_indexer_state_section_offset=layout.c4_indexer_state_offset,
        c4_state_blocks_per_row=c4_state_blocks_per_row,
        HAS_C4=has_c4,
        HAS_C128=has_c128,
        BLOCK=_BYTE_BLOCK,
        num_warps=4,
    )
