import triton
import triton.language as tl


BYTE_BLOCK = 1024
STATE_BLOCK = 256


@triton.jit
def _pool_pages_kernel(
    full_slots,
    full_to_pool,
    pool,
    pool_stride0,
    pool_stride1,
    paired_pool,
    paired_pool_stride0,
    paired_pool_stride1,
    full_to_c128,
    c128_pool,
    c128_pool_stride0,
    c128_pool_stride1,
    staging,
    grid_layer_num: tl.constexpr,
    pool_layer_num: tl.constexpr,
    pool_gpu_page_num: tl.constexpr,
    full_slots_page_stride: tl.constexpr,
    staging_page_stride: tl.constexpr,
    first_full_offset,
    full_offset_per_gpu_page: tl.constexpr,
    pool_page_size: tl.constexpr,
    gpu_page_nbytes: tl.constexpr,
    section_offset: tl.constexpr,
    section_layer_nbytes: tl.constexpr,
    paired_gpu_page_nbytes: tl.constexpr,
    paired_section_offset: tl.constexpr,
    paired_section_layer_nbytes: tl.constexpr,
    c128_row_num,
    c128_layer_num: tl.constexpr,
    c128_first_full_offset: tl.constexpr,
    c128_full_offset_per_row: tl.constexpr,
    c128_pool_page_size: tl.constexpr,
    c128_data_nbytes: tl.constexpr,
    c128_scale_nbytes: tl.constexpr,
    c128_scale_offset: tl.constexpr,
    c128_row_nbytes: tl.constexpr,
    c128_section_offset: tl.constexpr,
    c128_section_layer_nbytes: tl.constexpr,
    section_gpu_page_start,
    HAS_POOL: tl.constexpr,
    HAS_PAIRED_POOL: tl.constexpr,
    HAS_C128: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    job = tl.program_id(0)
    gpu_page = tl.program_id(1)
    byte_block = tl.program_id(2)
    layer = job % grid_layer_num
    logical_page = job // grid_layer_num

    logical_page_i64 = logical_page.to(tl.int64)
    layer_i64 = layer.to(tl.int64)
    gpu_page_i64 = gpu_page.to(tl.int64)
    block_offsets = tl.arange(0, BLOCK)
    offsets = byte_block * BLOCK + block_offsets
    offsets_i64 = offsets.to(tl.int64)

    if HAS_POOL:
        if (layer < pool_layer_num) & (gpu_page < pool_gpu_page_num):
            full_slot = tl.load(
                full_slots
                + logical_page_i64 * full_slots_page_stride
                + first_full_offset
                + gpu_page_i64 * full_offset_per_gpu_page
            ).to(tl.int64)
            pool_slot = tl.load(full_to_pool + full_slot).to(tl.int64)
            physical_page = pool_slot // pool_page_size
            mask = offsets < gpu_page_nbytes
            pool_ptr = pool + layer_i64 * pool_stride0 + physical_page * pool_stride1 + offsets_i64
            staging_ptr = (
                staging
                + logical_page_i64 * staging_page_stride
                + section_offset
                + layer_i64 * section_layer_nbytes
                + (section_gpu_page_start + gpu_page_i64) * gpu_page_nbytes
                + offsets_i64
            )
            if MODE == 0:
                tl.store(staging_ptr, tl.load(pool_ptr, mask=mask), mask=mask)
            else:
                tl.store(pool_ptr, tl.load(staging_ptr, mask=mask), mask=mask)

            if HAS_PAIRED_POOL:
                paired_mask = offsets < paired_gpu_page_nbytes
                paired_pool_ptr = (
                    paired_pool + layer_i64 * paired_pool_stride0 + physical_page * paired_pool_stride1 + offsets_i64
                )
                paired_staging_ptr = (
                    staging
                    + logical_page_i64 * staging_page_stride
                    + paired_section_offset
                    + layer_i64 * paired_section_layer_nbytes
                    + (section_gpu_page_start + gpu_page_i64) * paired_gpu_page_nbytes
                    + offsets_i64
                )
                if MODE == 0:
                    tl.store(
                        paired_staging_ptr,
                        tl.load(paired_pool_ptr, mask=paired_mask),
                        mask=paired_mask,
                    )
                else:
                    tl.store(
                        paired_pool_ptr,
                        tl.load(paired_staging_ptr, mask=paired_mask),
                        mask=paired_mask,
                    )

    if HAS_C128:
        if (byte_block < 2) & (layer < c128_layer_num):
            c128_row = gpu_page * 2 + byte_block
            if c128_row < c128_row_num:
                c128_row_i64 = c128_row.to(tl.int64)
                c128_full_slot = tl.load(
                    full_slots
                    + logical_page_i64 * full_slots_page_stride
                    + c128_first_full_offset
                    + c128_row_i64 * c128_full_offset_per_row
                ).to(tl.int64)
                c128_pool_slot = tl.load(full_to_c128 + c128_full_slot).to(tl.int64)
                c128_physical_page = c128_pool_slot // c128_pool_page_size
                c128_token_in_page = c128_pool_slot % c128_pool_page_size
                c128_pool_page = c128_pool + layer_i64 * c128_pool_stride0 + c128_physical_page * c128_pool_stride1
                block_offsets_i64 = block_offsets.to(tl.int64)
                c128_staging_ptr = (
                    staging
                    + logical_page_i64 * staging_page_stride
                    + c128_section_offset
                    + layer_i64 * c128_section_layer_nbytes
                    + c128_row_i64 * c128_row_nbytes
                    + block_offsets_i64
                )
                c128_data_mask = block_offsets < c128_data_nbytes
                c128_data_ptr = c128_pool_page + c128_token_in_page * c128_data_nbytes + block_offsets_i64
                if MODE == 0:
                    tl.store(
                        c128_staging_ptr,
                        tl.load(c128_data_ptr, mask=c128_data_mask),
                        mask=c128_data_mask,
                    )
                else:
                    tl.store(
                        c128_data_ptr,
                        tl.load(c128_staging_ptr, mask=c128_data_mask),
                        mask=c128_data_mask,
                    )

                c128_scale_local = block_offsets_i64 - c128_data_nbytes
                c128_scale_mask = (block_offsets >= c128_data_nbytes) & (block_offsets < c128_row_nbytes)
                c128_scale_ptr = (
                    c128_pool_page + c128_scale_offset + c128_token_in_page * c128_scale_nbytes + c128_scale_local
                )
                if MODE == 0:
                    tl.store(
                        c128_staging_ptr,
                        tl.load(c128_scale_ptr, mask=c128_scale_mask),
                        mask=c128_scale_mask,
                    )
                else:
                    tl.store(
                        c128_scale_ptr,
                        tl.load(c128_staging_ptr, mask=c128_scale_mask),
                        mask=c128_scale_mask,
                    )


def copy_pool_pages(
    mode,
    *,
    full_slots,
    mapping,
    pool,
    staging,
    page_num,
    gpu_page_num,
    layer_num,
    full_slots_page_stride,
    staging_page_stride,
    first_full_offset,
    full_offset_per_gpu_page,
    pool_page_size,
    section_offset,
    section_layer_nbytes,
    section_gpu_page_start=0,
    paired_pool=None,
    paired_section_offset=0,
    paired_section_layer_nbytes=0,
    c128_mapping=None,
    c128_pool=None,
    c128_row_num=0,
    c128_first_full_offset=0,
    c128_full_offset_per_row=0,
    c128_row_nbytes=0,
    c128_section_offset=0,
    c128_section_layer_nbytes=0,
):
    """Copy the present packed pools in one launch."""
    has_pool = pool is not None
    has_c128 = c128_pool is not None and c128_row_num > 0
    c128_buffer = c128_pool.buffer if has_c128 else None
    c128_layer_num = c128_pool.layer_num if has_c128 else 0
    c128_pool_page_size = c128_pool.page_size if has_c128 else 0
    c128_data_nbytes = c128_pool.data_bytes_per_token if has_c128 else 0
    c128_scale_nbytes = c128_pool.scale_bytes_per_token if has_c128 else 0
    c128_scale_offset = c128_pool.scale_offset_in_page if has_c128 else 0
    grid_layer_num = max(layer_num, c128_layer_num)
    grid_gpu_page_num = max(gpu_page_num, triton.cdiv(c128_row_num, 2) if has_c128 else 0)
    gpu_page_nbytes = pool.shape[-1] if has_pool else 0
    paired_gpu_page_nbytes = paired_pool.shape[-1] if paired_pool is not None else 0
    byte_blocks_per_gpu_page = triton.cdiv(
        max(gpu_page_nbytes, paired_gpu_page_nbytes, 2 * BYTE_BLOCK if has_c128 else 0),
        BYTE_BLOCK,
    )
    _pool_pages_kernel[(page_num * grid_layer_num, grid_gpu_page_num, byte_blocks_per_gpu_page)](
        full_slots,
        mapping,
        pool,
        pool.stride(0) if has_pool else 0,
        pool.stride(1) if has_pool else 0,
        paired_pool,
        paired_pool.stride(0) if paired_pool is not None else 0,
        paired_pool.stride(1) if paired_pool is not None else 0,
        c128_mapping,
        c128_buffer,
        c128_buffer.stride(0) if has_c128 else 0,
        c128_buffer.stride(1) if has_c128 else 0,
        staging,
        grid_layer_num=grid_layer_num,
        pool_layer_num=layer_num,
        pool_gpu_page_num=gpu_page_num,
        full_slots_page_stride=full_slots_page_stride,
        staging_page_stride=staging_page_stride,
        first_full_offset=first_full_offset,
        full_offset_per_gpu_page=full_offset_per_gpu_page,
        pool_page_size=pool_page_size,
        gpu_page_nbytes=gpu_page_nbytes,
        section_offset=section_offset,
        section_layer_nbytes=section_layer_nbytes,
        paired_gpu_page_nbytes=paired_gpu_page_nbytes,
        paired_section_offset=paired_section_offset,
        paired_section_layer_nbytes=paired_section_layer_nbytes,
        c128_row_num=c128_row_num,
        c128_layer_num=c128_layer_num,
        c128_first_full_offset=c128_first_full_offset,
        c128_full_offset_per_row=c128_full_offset_per_row,
        c128_pool_page_size=c128_pool_page_size,
        c128_data_nbytes=c128_data_nbytes,
        c128_scale_nbytes=c128_scale_nbytes,
        c128_scale_offset=c128_scale_offset,
        c128_row_nbytes=c128_row_nbytes,
        c128_section_offset=c128_section_offset,
        c128_section_layer_nbytes=c128_section_layer_nbytes,
        section_gpu_page_start=section_gpu_page_start,
        HAS_POOL=has_pool,
        HAS_PAIRED_POOL=paired_pool is not None,
        HAS_C128=has_c128,
        MODE=0 if mode == "pack" else 1,
        BLOCK=BYTE_BLOCK,
        num_warps=4,
    )
