import torch

import triton
import triton.language as tl
from .cache_io import INDEXER_PAGE_SIZE


@triton.jit
def _destindex_copy_indexer_k_kernel(
    indexer_k,
    mem_index,
    position_ids,
    indexer_k_fp8,
    indexer_k_scale,
    indexer_k_stride0,
    indexer_k_stride1,
    freqs_cis,
    APPLY_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AMAX_MIN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SLAB_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    position = tl.load(position_ids + cur_index)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    mem_slot = tl.load(mem_index + cur_index).to(tl.int64)
    compressed_kv_slot = mem_slot // COMPRESS_RATIO

    page = compressed_kv_slot // PAGE_SIZE
    token_in_page = compressed_kv_slot % PAGE_SIZE

    offs_d = tl.arange(0, HEAD_DIM)
    vals = tl.load(indexer_k + cur_index * indexer_k_stride0 + offs_d * indexer_k_stride1).to(tl.float32)
    if APPLY_ROPE:
        pair = tl.arange(0, HEAD_DIM // 2)
        real, imag = tl.split(tl.reshape(vals, (HEAD_DIM // 2, 2)))
        pos = position - COMPRESS_RATIO + 1
        freq = freqs_cis + pos * 64 + (pair - (HEAD_DIM - 64) // 2) * 2
        cos = tl.load(freq, pair >= (HEAD_DIM - 64) // 2, 1.0)
        sin = tl.load(freq + 1, pair >= (HEAD_DIM - 64) // 2, 0.0)
        vals = tl.reshape(tl.join(real * cos - imag * sin, real * sin + imag * cos), (HEAD_DIM,))
        vals = vals.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(vals), axis=0), AMAX_MIN)
    # Each indexer key carries one FP32 quantization scale in the page tail.
    scale = amax / FP8_MAX
    k_fp8 = tl.clamp(vals / scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)

    slab = token_in_page // SLAB_SIZE
    offset = token_in_page % SLAB_SIZE
    base = page * BYTES_PER_PAGE + slab * SLAB_SIZE * (HEAD_DIM + 4)
    data_base = base + offset * HEAD_DIM
    tl.store(indexer_k_fp8 + data_base + offs_d, k_fp8)
    scale_idx = (base + SLAB_SIZE * HEAD_DIM) // 4 + offset
    tl.store(indexer_k_scale + scale_idx, scale)
    return


@torch.no_grad()
def destindex_copy_indexer_k(
    indexer_k: torch.Tensor,
    mem_index: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_k_buffer: torch.Tensor,
    page_size: int,
    compress_ratio: int = 4,
    freqs_cis: torch.Tensor = None,
):
    """Write closed compression groups to packed FP8 indexer K pages.

    indexer_k: [T, head_dim] BF16; indexer_k_buffer: [page, bytes] uint8.
    Each page contains 64-key slabs, with FP8 data followed by FP32 scales.
    V4.1 passes freqs_cis to apply RoPE before quantization.
    """
    token_num = mem_index.shape[0]
    if token_num == 0:
        return
    head_dim = indexer_k.shape[-1]

    indexer_k = indexer_k.reshape(-1, head_dim)
    bytes_per_page = indexer_k_buffer.stride(0)
    _destindex_copy_indexer_k_kernel[(token_num,)](
        indexer_k,
        mem_index,
        position_ids,
        indexer_k_buffer.view(torch.float8_e4m3fn),
        indexer_k_buffer.view(torch.float32),
        indexer_k.stride(0),
        indexer_k.stride(1),
        None if freqs_cis is None else torch.view_as_real(freqs_cis).flatten(-2),
        APPLY_ROPE=freqs_cis is not None,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        AMAX_MIN=1e-4,
        HEAD_DIM=head_dim,
        COMPRESS_RATIO=compress_ratio,
        PAGE_SIZE=page_size,
        SLAB_SIZE=INDEXER_PAGE_SIZE,
        BYTES_PER_PAGE=bytes_per_page,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=freqs_cis is None,
    )
    return
