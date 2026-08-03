import torch
import triton
import triton.language as tl


_BYTE_BLOCK = 1024
_STATE_BLOCK = 256
_HISTORY_BLOCK_SIZE = 256


@triton.jit
def _pack_pool_gpu_pages_kernel(
    full_slots,
    full_to_pool,
    pool,
    pool_stride0,
    pool_stride1,
    staging,
    staging_stride0,
    token_page_size: tl.constexpr,
    layer_num: tl.constexpr,
    gpu_pages_per_cpu_page: tl.constexpr,
    first_full_offset: tl.constexpr,
    full_offset_per_gpu_page: tl.constexpr,
    pool_page_size: tl.constexpr,
    gpu_page_nbytes: tl.constexpr,
    section_offset: tl.constexpr,
    section_layer_nbytes: tl.constexpr,
    byte_blocks_per_gpu_page: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    byte_block = pid % byte_blocks_per_gpu_page
    job = pid // byte_blocks_per_gpu_page
    gpu_page = job % gpu_pages_per_cpu_page
    job = job // gpu_pages_per_cpu_page
    layer = job % layer_num
    logical_page = job // layer_num

    logical_page_i64 = logical_page.to(tl.int64)
    layer_i64 = layer.to(tl.int64)
    gpu_page_i64 = gpu_page.to(tl.int64)
    full_offset = logical_page_i64 * token_page_size + first_full_offset + gpu_page_i64 * full_offset_per_gpu_page
    full_slot = tl.load(full_slots + full_offset).to(tl.int64)
    pool_slot = tl.load(full_to_pool + full_slot).to(tl.int64)
    physical_page = pool_slot // pool_page_size

    offsets = byte_block * BLOCK + tl.arange(0, BLOCK)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < gpu_page_nbytes
    pool_ptr = pool + layer_i64 * pool_stride0 + physical_page * pool_stride1 + offsets_i64
    staging_ptr = (
        staging
        + logical_page_i64 * staging_stride0
        + section_offset
        + layer_i64 * section_layer_nbytes
        + gpu_page_i64 * gpu_page_nbytes
        + offsets_i64
    )
    tl.store(staging_ptr, tl.load(pool_ptr, mask=mask), mask=mask)


@triton.jit
def _pack_c128_rows_kernel(
    full_slots,
    full_to_c128,
    pool,
    pool_stride0,
    pool_stride1,
    staging,
    staging_stride0,
    token_page_size: tl.constexpr,
    layer_num: tl.constexpr,
    rows_per_page: tl.constexpr,
    pool_page_size: tl.constexpr,
    data_nbytes: tl.constexpr,
    scale_nbytes: tl.constexpr,
    scale_offset: tl.constexpr,
    row_nbytes: tl.constexpr,
    section_offset: tl.constexpr,
    section_layer_nbytes: tl.constexpr,
    BLOCK: tl.constexpr,
):
    job = tl.program_id(0)
    row = job % rows_per_page
    job = job // rows_per_page
    layer = job % layer_num
    logical_page = job // layer_num

    logical_page_i64 = logical_page.to(tl.int64)
    layer_i64 = layer.to(tl.int64)
    row_i64 = row.to(tl.int64)
    full_offset = logical_page_i64 * token_page_size + (row_i64 + 1) * 128 - 1
    full_slot = tl.load(full_slots + full_offset).to(tl.int64)
    pool_slot = tl.load(full_to_c128 + full_slot).to(tl.int64)
    physical_page = pool_slot // pool_page_size
    token_in_page = pool_slot % pool_page_size

    offsets = tl.arange(0, BLOCK)
    offsets_i64 = offsets.to(tl.int64)
    staging_ptr = (
        staging
        + logical_page_i64 * staging_stride0
        + section_offset
        + layer_i64 * section_layer_nbytes
        + row_i64 * row_nbytes
        + offsets_i64
    )
    pool_page = pool + layer_i64 * pool_stride0 + physical_page * pool_stride1
    data_mask = offsets < data_nbytes
    data_ptr = pool_page + token_in_page * data_nbytes + offsets_i64
    tl.store(staging_ptr, tl.load(data_ptr, mask=data_mask), mask=data_mask)

    scale_local = offsets_i64 - data_nbytes
    scale_mask = (offsets >= data_nbytes) & (offsets < row_nbytes)
    scale_ptr = pool_page + scale_offset + token_in_page * scale_nbytes + scale_local
    tl.store(staging_ptr, tl.load(scale_ptr, mask=scale_mask), mask=scale_mask)


@triton.jit
def _pack_c4_tail_states_kernel(
    full_slots,
    full_to_swa,
    state,
    state_stride0,
    state_stride1,
    staging_f32,
    staging_page_stride,
    token_page_size: tl.constexpr,
    layer_num: tl.constexpr,
    swa_page_size: tl.constexpr,
    state_ring: tl.constexpr,
    state_width: tl.constexpr,
    section_offset_f32: tl.constexpr,
    section_layer_elems: tl.constexpr,
    blocks_per_row: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    state_block = pid % blocks_per_row
    job = pid // blocks_per_row
    tail_row = job % 4
    job = job // 4
    layer = job % layer_num
    logical_page = job // layer_num

    logical_page_i64 = logical_page.to(tl.int64)
    layer_i64 = layer.to(tl.int64)
    tail_row_i64 = tail_row.to(tl.int64)
    tail_full_offset = logical_page_i64 * token_page_size + token_page_size - 4 + tail_row_i64
    tail_full_slot = tl.load(full_slots + tail_full_offset).to(tl.int64)
    tail_swa_slot = tl.load(full_to_swa + tail_full_slot).to(tl.int64)
    state_row = (tail_swa_slot // swa_page_size) * state_ring + tail_swa_slot % state_ring

    offsets = state_block * BLOCK + tl.arange(0, BLOCK)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < state_width
    state_ptr = state + layer_i64 * state_stride0 + state_row * state_stride1 + offsets_i64
    staging_ptr = (
        staging_f32
        + logical_page_i64 * staging_page_stride
        + section_offset_f32
        + layer_i64 * section_layer_elems
        + tail_row_i64 * state_width
        + offsets_i64
    )
    tl.store(staging_ptr, tl.load(state_ptr, mask=mask), mask=mask)


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
def _unpack_history_gpu_pages_kernel(
    pool_slots,
    pool,
    pool_stride0,
    pool_stride1,
    cpu_pages,
    cpu_stride0,
    cpu_page_indexes,
    first_history_block,
    layer_num: tl.constexpr,
    blocks_per_cpu_page: tl.constexpr,
    pool_slots_per_block: tl.constexpr,
    pool_page_size: tl.constexpr,
    gpu_page_nbytes: tl.constexpr,
    section_offset: tl.constexpr,
    section_layer_nbytes: tl.constexpr,
    byte_blocks_per_gpu_page: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    byte_block = pid % byte_blocks_per_gpu_page
    job = pid // byte_blocks_per_gpu_page
    layer = job % layer_num
    history_block = job // layer_num

    history_block_i64 = history_block.to(tl.int64)
    layer_i64 = layer.to(tl.int64)
    absolute_block = history_block_i64 + first_history_block
    cpu_page_list_index = absolute_block // blocks_per_cpu_page
    block_in_cpu_page = absolute_block % blocks_per_cpu_page
    cpu_page = tl.load(cpu_page_indexes + cpu_page_list_index).to(tl.int64)

    pool_slot = tl.load(pool_slots + history_block_i64 * pool_slots_per_block).to(tl.int64)
    physical_page = pool_slot // pool_page_size

    offsets = byte_block * BLOCK + tl.arange(0, BLOCK)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < gpu_page_nbytes
    source = (
        cpu_pages
        + cpu_page * cpu_stride0
        + section_offset
        + layer_i64 * section_layer_nbytes
        + block_in_cpu_page * gpu_page_nbytes
        + offsets_i64
    )
    target = pool + layer_i64 * pool_stride0 + physical_page * pool_stride1 + offsets_i64
    tl.store(target, tl.load(source, mask=mask), mask=mask)


@triton.jit
def _unpack_c128_rows_kernel(
    c128_slots,
    pool,
    pool_stride0,
    pool_stride1,
    cpu_pages,
    cpu_stride0,
    cpu_page_indexes,
    first_history_block,
    layer_num: tl.constexpr,
    blocks_per_cpu_page: tl.constexpr,
    pool_page_size: tl.constexpr,
    data_nbytes: tl.constexpr,
    scale_nbytes: tl.constexpr,
    scale_offset: tl.constexpr,
    row_nbytes: tl.constexpr,
    section_offset: tl.constexpr,
    section_layer_nbytes: tl.constexpr,
    BLOCK: tl.constexpr,
):
    job = tl.program_id(0)
    row = job % 2
    job = job // 2
    layer = job % layer_num
    history_block = job // layer_num

    history_block_i64 = history_block.to(tl.int64)
    layer_i64 = layer.to(tl.int64)
    row_i64 = row.to(tl.int64)
    absolute_block = history_block_i64 + first_history_block
    cpu_page_list_index = absolute_block // blocks_per_cpu_page
    block_in_cpu_page = absolute_block % blocks_per_cpu_page
    cpu_page = tl.load(cpu_page_indexes + cpu_page_list_index).to(tl.int64)

    pool_slot = tl.load(c128_slots + history_block_i64 * 2 + row_i64).to(tl.int64)
    physical_page = pool_slot // pool_page_size
    token_in_page = pool_slot % pool_page_size

    offsets = tl.arange(0, BLOCK)
    offsets_i64 = offsets.to(tl.int64)
    cpu_row = block_in_cpu_page * 2 + row_i64
    source = (
        cpu_pages
        + cpu_page * cpu_stride0
        + section_offset
        + layer_i64 * section_layer_nbytes
        + cpu_row * row_nbytes
        + offsets_i64
    )
    pool_page = pool + layer_i64 * pool_stride0 + physical_page * pool_stride1
    data_mask = offsets < data_nbytes
    data_target = pool_page + token_in_page * data_nbytes + offsets_i64
    tl.store(data_target, tl.load(source, mask=data_mask), mask=data_mask)

    scale_local = offsets_i64 - data_nbytes
    scale_mask = (offsets >= data_nbytes) & (offsets < row_nbytes)
    scale_target = pool_page + scale_offset + token_in_page * scale_nbytes + scale_local
    tl.store(scale_target, tl.load(source, mask=scale_mask), mask=scale_mask)


@triton.jit
def _unpack_resume_gpu_pages_kernel(
    resume_swa_slots,
    pool,
    pool_stride0,
    pool_stride1,
    cpu_pages,
    cpu_stride0,
    cpu_page_indexes,
    cpu_page_num,
    layer_num: tl.constexpr,
    gpu_pages_per_cpu_page: tl.constexpr,
    pool_page_size: tl.constexpr,
    gpu_page_nbytes: tl.constexpr,
    section_offset: tl.constexpr,
    section_layer_nbytes: tl.constexpr,
    byte_blocks_per_gpu_page: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    byte_block = pid % byte_blocks_per_gpu_page
    job = pid // byte_blocks_per_gpu_page
    gpu_page = job % gpu_pages_per_cpu_page
    layer = (job // gpu_pages_per_cpu_page) % layer_num

    layer_i64 = layer.to(tl.int64)
    gpu_page_i64 = gpu_page.to(tl.int64)
    pool_slot = tl.load(resume_swa_slots + gpu_page_i64 * pool_page_size).to(tl.int64)
    physical_page = pool_slot // pool_page_size
    cpu_page = tl.load(cpu_page_indexes + cpu_page_num - 1).to(tl.int64)

    offsets = byte_block * BLOCK + tl.arange(0, BLOCK)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < gpu_page_nbytes
    source = (
        cpu_pages
        + cpu_page * cpu_stride0
        + section_offset
        + layer_i64 * section_layer_nbytes
        + gpu_page_i64 * gpu_page_nbytes
        + offsets_i64
    )
    target = pool + layer_i64 * pool_stride0 + physical_page * pool_stride1 + offsets_i64
    tl.store(target, tl.load(source, mask=mask), mask=mask)


@triton.jit
def _unpack_c4_tail_states_kernel(
    resume_swa_slots,
    state,
    state_stride0,
    state_stride1,
    cpu_pages_f32,
    cpu_page_stride,
    cpu_page_indexes,
    cpu_page_num,
    layer_num: tl.constexpr,
    swa_page_size: tl.constexpr,
    state_ring: tl.constexpr,
    state_width: tl.constexpr,
    section_offset_f32: tl.constexpr,
    section_layer_elems: tl.constexpr,
    blocks_per_row: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    state_block = pid % blocks_per_row
    job = pid // blocks_per_row
    tail_row = job % 4
    layer = (job // 4) % layer_num

    layer_i64 = layer.to(tl.int64)
    tail_row_i64 = tail_row.to(tl.int64)
    tail_swa_slot = tl.load(resume_swa_slots + 2 * swa_page_size - 4 + tail_row_i64).to(tl.int64)
    state_row = (tail_swa_slot // swa_page_size) * state_ring + tail_swa_slot % state_ring
    cpu_page = tl.load(cpu_page_indexes + cpu_page_num - 1).to(tl.int64)

    offsets = state_block * BLOCK + tl.arange(0, BLOCK)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < state_width
    source = (
        cpu_pages_f32
        + cpu_page * cpu_page_stride
        + section_offset_f32
        + layer_i64 * section_layer_elems
        + tail_row_i64 * state_width
        + offsets_i64
    )
    target = state + layer_i64 * state_stride0 + state_row * state_stride1 + offsets_i64
    tl.store(target, tl.load(source, mask=mask), mask=mask)


def _launch_pack_pool_gpu_pages(
    *,
    full_slots,
    mapping,
    pool,
    pool_page_size,
    staging,
    page_num,
    token_page_size,
    layer_num,
    gpu_pages_per_cpu_page,
    first_full_offset,
    full_offset_per_gpu_page,
    section_offset,
    section_layer_nbytes,
):
    gpu_page_nbytes = pool.shape[-1]
    byte_blocks_per_gpu_page = triton.cdiv(gpu_page_nbytes, _BYTE_BLOCK)
    _pack_pool_gpu_pages_kernel[(page_num * layer_num * gpu_pages_per_cpu_page * byte_blocks_per_gpu_page,)](
        full_slots,
        mapping,
        pool,
        pool.stride(0),
        pool.stride(1),
        staging,
        staging.stride(0),
        token_page_size=token_page_size,
        layer_num=layer_num,
        gpu_pages_per_cpu_page=gpu_pages_per_cpu_page,
        first_full_offset=first_full_offset,
        full_offset_per_gpu_page=full_offset_per_gpu_page,
        pool_page_size=pool_page_size,
        gpu_page_nbytes=gpu_page_nbytes,
        section_offset=section_offset,
        section_layer_nbytes=section_layer_nbytes,
        byte_blocks_per_gpu_page=byte_blocks_per_gpu_page,
        BLOCK=_BYTE_BLOCK,
        num_warps=4,
    )


def _pack_c4_states(mem_manager, full_slots, state, staging, page_num, section_offset):
    layout = mem_manager.cpu_cache_layout
    state_width = state.shape[-1]
    blocks_per_row = triton.cdiv(state_width, _STATE_BLOCK)
    _pack_c4_tail_states_kernel[(page_num * mem_manager.n_c4 * 4 * blocks_per_row,)](
        full_slots,
        mem_manager.full_to_swa_indexs,
        state,
        state.stride(0),
        state.stride(1),
        staging.view(torch.float32),
        staging.stride(0) // 4,
        token_page_size=layout.token_page_size,
        layer_num=mem_manager.n_c4,
        swa_page_size=mem_manager.swa_pool.page_size,
        state_ring=mem_manager.c4_state_ring,
        state_width=state_width,
        section_offset_f32=section_offset // 4,
        section_layer_elems=4 * state_width,
        blocks_per_row=blocks_per_row,
        BLOCK=_STATE_BLOCK,
        num_warps=4,
    )


def pack_gpu_cache_to_staging(mem_manager, source_mem_indexes: torch.Tensor, staging: torch.Tensor) -> None:
    """Pack a compact batch of complete CPU checkpoint pages into caller-owned CUDA staging."""
    layout = mem_manager.cpu_cache_layout
    assert source_mem_indexes.is_cuda and staging.is_cuda
    assert source_mem_indexes.ndim == 2 and source_mem_indexes.shape[1] == layout.token_page_size
    assert staging.dtype == torch.uint8 and staging.is_contiguous()
    page_num = source_mem_indexes.shape[0]
    assert staging.shape == (page_num, layout.page_nbytes)

    full_slots = source_mem_indexes.reshape(-1)
    if mem_manager.n_c4:
        for pool, section_offset, section_layer_nbytes in (
            (mem_manager.c4_pool, layout.c4_offset, layout.c4_layer_nbytes),
            (
                mem_manager.c4_indexer_pool,
                layout.c4_indexer_offset,
                layout.c4_indexer_layer_nbytes,
            ),
        ):
            _launch_pack_pool_gpu_pages(
                full_slots=full_slots,
                mapping=mem_manager.full_to_c4_indexs,
                pool=pool.buffer,
                pool_page_size=pool.page_size,
                staging=staging,
                page_num=page_num,
                token_page_size=layout.token_page_size,
                layer_num=mem_manager.n_c4,
                gpu_pages_per_cpu_page=layout.c4_gpu_pages_per_cpu_page,
                first_full_offset=3,
                full_offset_per_gpu_page=_HISTORY_BLOCK_SIZE,
                section_offset=section_offset,
                section_layer_nbytes=section_layer_nbytes,
            )

    if mem_manager.n_c128:
        pool = mem_manager.c128_pool
        _pack_c128_rows_kernel[(page_num * mem_manager.n_c128 * layout.c128_rows_per_page,)](
            full_slots,
            mem_manager.full_to_c128_indexs,
            pool.buffer,
            pool.buffer.stride(0),
            pool.buffer.stride(1),
            staging,
            staging.stride(0),
            token_page_size=layout.token_page_size,
            layer_num=mem_manager.n_c128,
            rows_per_page=layout.c128_rows_per_page,
            pool_page_size=pool.page_size,
            data_nbytes=pool.data_bytes_per_token,
            scale_nbytes=pool.scale_bytes_per_token,
            scale_offset=pool.scale_offset_in_page,
            row_nbytes=layout.c128_row_nbytes,
            section_offset=layout.c128_offset,
            section_layer_nbytes=layout.c128_layer_nbytes,
            BLOCK=triton.next_power_of_2(layout.c128_row_nbytes),
            num_warps=4,
        )

    _launch_pack_pool_gpu_pages(
        full_slots=full_slots,
        mapping=mem_manager.full_to_swa_indexs,
        pool=mem_manager.swa_pool.buffer,
        pool_page_size=mem_manager.swa_pool.page_size,
        staging=staging,
        page_num=page_num,
        token_page_size=layout.token_page_size,
        layer_num=mem_manager.layer_num,
        gpu_pages_per_cpu_page=layout.swa_gpu_pages_per_cpu_page,
        first_full_offset=layout.token_page_size - _HISTORY_BLOCK_SIZE,
        full_offset_per_gpu_page=mem_manager.swa_pool.page_size,
        section_offset=layout.swa_offset,
        section_layer_nbytes=layout.swa_layer_nbytes,
    )
    if mem_manager.n_c4:
        _pack_c4_states(
            mem_manager,
            full_slots,
            mem_manager.c4_state_buffer,
            staging,
            page_num,
            layout.c4_state_offset,
        )
        _pack_c4_states(
            mem_manager,
            full_slots,
            mem_manager.c4_indexer_state_buffer,
            staging,
            page_num,
            layout.c4_indexer_state_offset,
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


def _unpack_history_gpu_pages(
    *,
    pool_slots,
    pool,
    cpu_pages,
    cpu_page_indexes,
    first_history_block,
    history_block_num,
    blocks_per_cpu_page,
    layer_num,
    pool_slots_per_block,
    section_offset,
    section_layer_nbytes,
):
    gpu_page_nbytes = pool.buffer.shape[-1]
    byte_blocks_per_gpu_page = triton.cdiv(gpu_page_nbytes, _BYTE_BLOCK)
    _unpack_history_gpu_pages_kernel[(history_block_num * layer_num * byte_blocks_per_gpu_page,)](
        pool_slots,
        pool.buffer,
        pool.buffer.stride(0),
        pool.buffer.stride(1),
        cpu_pages,
        cpu_pages.stride(0),
        cpu_page_indexes,
        first_history_block,
        layer_num=layer_num,
        blocks_per_cpu_page=blocks_per_cpu_page,
        pool_slots_per_block=pool_slots_per_block,
        pool_page_size=pool.page_size,
        gpu_page_nbytes=gpu_page_nbytes,
        section_offset=section_offset,
        section_layer_nbytes=section_layer_nbytes,
        byte_blocks_per_gpu_page=byte_blocks_per_gpu_page,
        BLOCK=_BYTE_BLOCK,
        num_warps=4,
    )


def _unpack_c4_states(mem_manager, resume_swa_slots, state, cpu_pages, page_indexes, section_offset):
    state_width = state.shape[-1]
    blocks_per_row = triton.cdiv(state_width, _STATE_BLOCK)
    _unpack_c4_tail_states_kernel[(mem_manager.n_c4 * 4 * blocks_per_row,)](
        resume_swa_slots,
        state,
        state.stride(0),
        state.stride(1),
        cpu_pages.view(torch.float32),
        cpu_pages.stride(0) // 4,
        page_indexes,
        page_indexes.numel(),
        layer_num=mem_manager.n_c4,
        swa_page_size=mem_manager.swa_pool.page_size,
        state_ring=mem_manager.c4_state_ring,
        state_width=state_width,
        section_offset_f32=section_offset // 4,
        section_layer_elems=4 * state_width,
        blocks_per_row=blocks_per_row,
        BLOCK=_STATE_BLOCK,
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

    if mem_manager.n_c4:
        assert load_plan.history_c4_slots.shape == (history_block_num, mem_manager.c4_pool.page_size)
        for pool, section_offset, section_layer_nbytes in (
            (mem_manager.c4_pool, layout.c4_offset, layout.c4_layer_nbytes),
            (
                mem_manager.c4_indexer_pool,
                layout.c4_indexer_offset,
                layout.c4_indexer_layer_nbytes,
            ),
        ):
            _unpack_history_gpu_pages(
                pool_slots=load_plan.history_c4_slots,
                pool=pool,
                cpu_pages=cpu_pages,
                cpu_page_indexes=cpu_page_indexes,
                first_history_block=first_history_block,
                history_block_num=history_block_num,
                blocks_per_cpu_page=blocks_per_cpu_page,
                layer_num=mem_manager.n_c4,
                pool_slots_per_block=mem_manager.c4_pool.page_size,
                section_offset=section_offset,
                section_layer_nbytes=section_layer_nbytes,
            )

    if mem_manager.n_c128:
        assert load_plan.history_c128_slots.shape == (history_block_num, 2)
        pool = mem_manager.c128_pool
        _unpack_c128_rows_kernel[(history_block_num * mem_manager.n_c128 * 2,)](
            load_plan.history_c128_slots,
            pool.buffer,
            pool.buffer.stride(0),
            pool.buffer.stride(1),
            cpu_pages,
            cpu_pages.stride(0),
            cpu_page_indexes,
            first_history_block,
            layer_num=mem_manager.n_c128,
            blocks_per_cpu_page=blocks_per_cpu_page,
            pool_page_size=pool.page_size,
            data_nbytes=pool.data_bytes_per_token,
            scale_nbytes=pool.scale_bytes_per_token,
            scale_offset=pool.scale_offset_in_page,
            row_nbytes=layout.c128_row_nbytes,
            section_offset=layout.c128_offset,
            section_layer_nbytes=layout.c128_layer_nbytes,
            BLOCK=triton.next_power_of_2(layout.c128_row_nbytes),
            num_warps=4,
        )

    pool = mem_manager.swa_pool
    byte_blocks_per_gpu_page = triton.cdiv(pool.buffer.shape[-1], _BYTE_BLOCK)
    _unpack_resume_gpu_pages_kernel[
        (mem_manager.layer_num * layout.swa_gpu_pages_per_cpu_page * byte_blocks_per_gpu_page,)
    ](
        load_plan.resume_swa_slots,
        pool.buffer,
        pool.buffer.stride(0),
        pool.buffer.stride(1),
        cpu_pages,
        cpu_pages.stride(0),
        cpu_page_indexes,
        cpu_page_indexes.numel(),
        layer_num=mem_manager.layer_num,
        gpu_pages_per_cpu_page=layout.swa_gpu_pages_per_cpu_page,
        pool_page_size=pool.page_size,
        gpu_page_nbytes=pool.buffer.shape[-1],
        section_offset=layout.swa_offset,
        section_layer_nbytes=layout.swa_layer_nbytes,
        byte_blocks_per_gpu_page=byte_blocks_per_gpu_page,
        BLOCK=_BYTE_BLOCK,
        num_warps=4,
    )
    if mem_manager.n_c4:
        _unpack_c4_states(
            mem_manager,
            load_plan.resume_swa_slots,
            mem_manager.c4_state_buffer,
            cpu_pages,
            cpu_page_indexes,
            layout.c4_state_offset,
        )
        _unpack_c4_states(
            mem_manager,
            load_plan.resume_swa_slots,
            mem_manager.c4_indexer_state_buffer,
            cpu_pages,
            cpu_page_indexes,
            layout.c4_indexer_state_offset,
        )
