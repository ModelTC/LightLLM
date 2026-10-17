import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_indexer_k_dsv4(
    indexer_k,
    mem_index,
    positions,
    indexer_k_fp8,
    indexer_k_scale,
    indexer_k_stride0,
    indexer_k_stride1,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AMAX_MIN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    position = tl.load(positions + cur_index)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    full_slot = tl.load(mem_index + cur_index).to(tl.int64)
    dest_index = full_slot // COMPRESS_RATIO

    page = dest_index // PAGE_SIZE
    token_in_page = dest_index % PAGE_SIZE

    offs_d = tl.arange(0, HEAD_DIM)
    vals = tl.load(indexer_k + cur_index * indexer_k_stride0 + offs_d * indexer_k_stride1).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(vals), axis=0), AMAX_MIN)
    # Each indexer key carries one FP32 quantization scale in the page tail.
    scale = amax / FP8_MAX
    k_fp8 = tl.clamp(vals / scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)

    data_base = page * BYTES_PER_PAGE + token_in_page * HEAD_DIM
    tl.store(indexer_k_fp8 + data_base + offs_d, k_fp8)
    scale_idx = (page * BYTES_PER_PAGE + PAGE_SIZE * HEAD_DIM) // 4 + token_in_page
    tl.store(indexer_k_scale + scale_idx, scale)
    return


@torch.no_grad()
def destindex_copy_indexer_k_dsv4(
    indexer_k: torch.Tensor,
    mem_index: torch.Tensor,
    positions: torch.Tensor,
    indexer_k_cache: torch.Tensor,
    page_size: int,
):
    """Packed indexer-indexer_k page-slab writer (DeepSeek-V4 c4/CSA layers).

    indexer_k: [T, 128] bf16 unquantized indexer keys.
    mem_index: [T] int — full-token slots for the current rows.
    positions: [T] int — logical token positions; only c4 group-end rows are written.
    C4 slots are group-end full-token slots divided by four.
    indexer_k_cache: [num_pages, bytes_per_page] uint8 — one layer's slab from the c4 indexer
        Packed pages (128B fp8 data region + 4B fp32 scale tail per token).

    """
    seq_len = mem_index.shape[0]
    if seq_len == 0:
        return
    head_dim = 128

    indexer_k = indexer_k.reshape(-1, head_dim)
    bytes_per_page = indexer_k_cache.shape[-1]

    flat = indexer_k_cache.view(-1)
    _fwd_kernel_destindex_copy_indexer_k_dsv4[(seq_len,)](
        indexer_k,
        mem_index,
        positions,
        flat.view(torch.float8_e4m3fn),
        flat.view(torch.float32),
        indexer_k.stride(0),
        indexer_k.stride(1),
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        AMAX_MIN=1e-4,
        HEAD_DIM=head_dim,
        COMPRESS_RATIO=4,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        num_warps=1,
        num_stages=1,
    )
    return
