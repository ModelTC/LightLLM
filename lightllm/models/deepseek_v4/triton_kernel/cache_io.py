"""Model-side packed views and token-indexed cache transfers."""

import triton
import triton.language as tl

from lightllm.utils.envs_utils import get_env_start_args


def view_swa_kv(swa_buffer, config):
    return swa_buffer.view(-1, config.get_packed_kv_bytes(config.swa_page_token_num))


def get_compressed_kv_buffer(mem_manager, layer_index):
    kv_buffer = mem_manager.get_att_input_params(layer_index)
    if kv_buffer is None:
        return None
    config = mem_manager.cache_config
    compress_ratio = config.compress_rates[layer_index]
    packed_kv_pages = kv_buffer.view(-1, config.compressed_kv_token_alignment * kv_buffer.shape[-1])
    return packed_kv_pages[:, : config.get_packed_kv_bytes(config.compressed_kv_token_alignment // compress_ratio)]


def get_indexer_k_buffer(mem_manager, layer_index):
    config = mem_manager.cache_config
    kv_buffer = mem_manager.get_att_input_params(layer_index)
    packed_kv_pages = kv_buffer.view(-1, config.compressed_kv_token_alignment * kv_buffer.shape[-1])
    entries = config.compressed_kv_token_alignment // config.short_compress_ratio
    begin = config.get_packed_kv_bytes(entries)
    return packed_kv_pages[:, begin : begin + entries * config.indexer_bytes]


@triton.jit(do_not_specialize=["TOKEN_NUM", "COPY_NUM"])
def _copy_compressed_kv_kernel(
    kv_buffers,
    kv_move_buffer,
    mem_indexes,
    page_indexes,
    TOKEN_NUM,
    COPY_NUM,
    PAGE_TOKENS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    SLOT_BYTES: tl.constexpr,
    TO_PAGES: tl.constexpr,
    ROUND_TAIL: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    layer = tl.program_id(1)
    kv_address = tl.full((), 0, tl.int64)
    slot_bytes = tl.full((), 0, tl.int32)
    layer_byte_offset = tl.full((), 0, tl.int64)
    offset = 0
    for i in tl.static_range(len(SLOT_BYTES)):
        kv_address = tl.where(layer == i, kv_buffers[i].to(tl.int64), kv_address)
        slot_bytes = tl.where(layer == i, SLOT_BYTES[i], slot_bytes)
        layer_byte_offset = tl.where(layer == i, offset, layer_byte_offset)
        offset += PAGE_TOKENS * SLOT_BYTES[i]
    kv_buffer = kv_address.to(tl.pointer_type(tl.uint8))
    tokens = tl.program_id(0) * 32 + tl.arange(0, 32)
    slot = tl.load(mem_indexes + tokens, tokens < TOKEN_NUM, -1).to(tl.int64)
    if ROUND_TAIL:
        # Packed page tails include scales for earlier tokens. PD also carries
        # the unused token slots of its final allocated compression block.
        last = tl.load(mem_indexes + TOKEN_NUM - 1).to(tl.int64)
        slot = tl.where(tokens < TOKEN_NUM, slot, last + tokens - TOKEN_NUM + 1)
    page = tokens // PAGE_TOKENS
    valid = (tokens < COPY_NUM) & (slot >= 0)
    page_id = tl.load(page_indexes + page, tokens < COPY_NUM, 0).to(tl.int64)
    columns = tl.arange(0, BLOCK_BYTES)
    kv_ptr = kv_buffer + slot[:, None] * slot_bytes + columns[None, :]
    page_ptr = (
        kv_move_buffer
        + page_id[:, None] * PAGE_STRIDE
        + layer_byte_offset
        + (tokens % PAGE_TOKENS)[:, None] * slot_bytes
        + columns[None, :]
    )
    mask = valid[:, None] & (columns[None, :] < slot_bytes)
    if TO_PAGES:
        tl.store(page_ptr, tl.load(kv_ptr, mask, 0), mask)
    else:
        tl.store(kv_ptr, tl.load(page_ptr, mask, 0), mask)


def copy_compressed_kv_pages(
    mem_manager, mem_indexes, page_indexes, kv_move_buffer, *, to_pages, page_size, round_tail=False
):
    if not mem_indexes.numel():
        return
    config = mem_manager.cache_config
    copy_num = (
        triton.cdiv(mem_indexes.numel(), config.compressed_kv_token_alignment) * config.compressed_kv_token_alignment
        if round_tail
        else mem_indexes.numel()
    )
    slot_bytes = config.compressed_kv_slot_bytes
    _copy_compressed_kv_kernel[(triton.cdiv(copy_num, 32), len(slot_bytes))](
        tuple(mem_manager.kv_buffer),
        kv_move_buffer,
        mem_indexes,
        page_indexes,
        TOKEN_NUM=mem_indexes.numel(),
        COPY_NUM=copy_num,
        PAGE_TOKENS=page_size,
        PAGE_STRIDE=kv_move_buffer.stride(0),
        SLOT_BYTES=slot_bytes,
        TO_PAGES=to_pages,
        ROUND_TAIL=round_tail,
        BLOCK_BYTES=triton.next_power_of_2(max(slot_bytes)),
        num_warps=4,
    )


@triton.jit(do_not_specialize=["TOKEN_NUM"])
def _copy_cpu_cache_kernel(
    kv_buffer,
    big_page_state_buffer,
    small_page_state_buffer,
    mem_indexes,
    big_page_buffer_ids,
    page_indexes,
    page_readies,
    cpu_cache_tensor,
    TOKEN_NUM,
    PAGE_TOKENS: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    STATE_BYTES: tl.constexpr,
    SLOT_BYTES: tl.constexpr,
    TO_CPU_CACHE: tl.constexpr,
    HAS_READIES: tl.constexpr,
    HAS_SMALL_STATES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    if HAS_READIES:
        if tl.load(page_readies + page):
            return

    cpu_page = cpu_cache_tensor + tl.load(page_indexes + page).to(tl.int64) * PAGE_BYTES
    block_start, block_step = tl.program_id(1), tl.num_programs(1)
    layer_offset = 0
    # CPU 页前半段按层保存压缩 KV 和 indexer K，碎页只复制实际存在的 token 槽位。
    for layer_index in tl.static_range(len(SLOT_BYTES)):
        slot_bytes = SLOT_BYTES[layer_index]
        layer_bytes = PAGE_TOKENS * slot_bytes
        for block in range(block_start, tl.cdiv(layer_bytes, BLOCK), block_step):
            offsets = block * BLOCK + tl.arange(0, BLOCK)
            tokens = page * PAGE_TOKENS + offsets // slot_bytes
            mask = (offsets < layer_bytes) & (tokens < TOKEN_NUM)
            slots = tl.load(mem_indexes + tokens, mask, -1).to(tl.int64)
            mask &= slots >= 0
            gpu = kv_buffer[layer_index] + slots * slot_bytes + offsets % slot_bytes
            cpu = cpu_page + layer_offset + offsets
            if TO_CPU_CACHE:
                tl.store(cpu, tl.load(gpu, mask, 0), mask)
            else:
                tl.store(gpu, tl.load(cpu, mask, 0), mask)
        layer_offset += layer_bytes

    # 页尾的 SWA、C4 和 indexer 状态已连续存储，作为一整块复制。
    state_id = tl.load(big_page_buffer_ids + page).to(tl.int64)
    state = big_page_state_buffer + state_id * STATE_BYTES
    if HAS_SMALL_STATES:
        if state_id < 0:
            state = small_page_state_buffer + (-state_id - 1) * STATE_BYTES
    for block in range(block_start, tl.cdiv(STATE_BYTES, BLOCK), block_step):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < STATE_BYTES
        cpu = cpu_page + layer_offset + offsets
        if TO_CPU_CACHE:
            tl.store(cpu, tl.load(state + offsets, mask, 0), mask)
        else:
            tl.store(state + offsets, tl.load(cpu, mask, 0), mask)


def _copy_cpu_cache_pages(
    mem_manager,
    mem_indexes,
    big_page_buffer_ids,
    page_indexes,
    cpu_cache_tensor,
    *,
    to_cpu_cache,
    page_readies=None,
    small_page_state_buffer=None,
):
    assert len(big_page_buffer_ids) == len(page_indexes)
    cpu_cache_pages = cpu_cache_tensor.view(cpu_cache_tensor.shape[0], -1)
    _copy_cpu_cache_kernel[(len(page_indexes), 12)](
        tuple(mem_manager.kv_buffer),
        mem_manager.big_page_buffers.buffer,
        small_page_state_buffer,
        mem_indexes,
        big_page_buffer_ids,
        page_indexes,
        page_readies,
        cpu_cache_pages,
        TOKEN_NUM=len(mem_indexes),
        PAGE_TOKENS=get_env_start_args().cpu_cache_token_page_size,
        PAGE_BYTES=cpu_cache_pages.stride(0),
        STATE_BYTES=mem_manager.big_page_buffers.buffer.shape[-1],
        SLOT_BYTES=mem_manager.cache_config.compressed_kv_slot_bytes,
        TO_CPU_CACHE=to_cpu_cache,
        HAS_READIES=page_readies is not None,
        HAS_SMALL_STATES=small_page_state_buffer is not None,
        BLOCK=4096,
        num_warps=4,
    )


def copy_kv_buffer_to_cpu_cache(
    mem_manager,
    mem_indexes,
    page_indexes,
    page_readies,
    big_page_buffer_ids,
    cpu_cache_tensor,
    small_page_state_buffer=None,
):
    _copy_cpu_cache_pages(
        mem_manager,
        mem_indexes,
        big_page_buffer_ids,
        page_indexes,
        cpu_cache_tensor,
        to_cpu_cache=True,
        page_readies=page_readies,
        small_page_state_buffer=small_page_state_buffer,
    )


def copy_cpu_cache_to_kv_buffer(mem_manager, mem_indexes, big_page_buffer_ids, page_indexes, cpu_cache_tensor):
    _copy_cpu_cache_pages(
        mem_manager, mem_indexes, big_page_buffer_ids, page_indexes, cpu_cache_tensor, to_cpu_cache=False
    )


@triton.jit
def _scatter_cpu_cache_pages_kernel(staging, page_indexes, cpu_pages, PAGE_BYTES: tl.constexpr, BLOCK: tl.constexpr):
    page = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    destination = tl.load(page_indexes + page).to(tl.int64)
    mask = offsets < PAGE_BYTES
    values = tl.load(staging + page * PAGE_BYTES + offsets, mask, 0)
    tl.store(cpu_pages + destination * PAGE_BYTES + offsets, values, mask)


def scatter_cpu_cache_pages(staging, page_indexes, cpu_cache_tensor):
    page_bytes = staging.shape[1]
    assert cpu_cache_tensor[0].numel() == page_bytes
    _scatter_cpu_cache_pages_kernel[(len(page_indexes), triton.cdiv(page_bytes, 4096))](
        staging, page_indexes, cpu_cache_tensor, PAGE_BYTES=page_bytes, BLOCK=4096, num_warps=4
    )


@triton.jit
def _copy_state_cache_kernel(src_state, dst_state, NBYTES: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NBYTES
    tl.store(dst_state + offsets, tl.load(src_state + offsets, mask, 0), mask)


def copy_state_cache(src_state, dst_state):
    """在当前 CUDA stream 上整块复制 pinned 状态，保持与 offload/load 的执行顺序。"""
    assert src_state.shape == dst_state.shape
    _copy_state_cache_kernel[(triton.cdiv(src_state.numel(), 4096),)](
        src_state, dst_state, NBYTES=src_state.numel(), BLOCK=4096, num_warps=4
    )
