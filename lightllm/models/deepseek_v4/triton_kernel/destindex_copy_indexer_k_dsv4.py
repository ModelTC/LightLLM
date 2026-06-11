import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_indexer_k_dsv4(
    K,
    Dest_loc,
    O_fp8,
    O_f32,
    stride_k_bs,
    stride_k_d,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    SCALE_MIN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)
    # negative dest (unmapped slot) is a no-op, not an OOB write into a neighboring page.
    if dest_index < 0:
        return

    page = dest_index // PAGE_SIZE
    token_in_page = dest_index % PAGE_SIZE

    offs_d = tl.arange(0, HEAD_DIM)
    vals = tl.load(K + cur_index * stride_k_bs + offs_d * stride_k_d).to(tl.float32)
    amax = tl.max(tl.abs(vals), axis=0)
    # per-token plain fp32 scale (not ue8m0), matching DeepseekV4MemoryManager._pack_indexer_k
    scale = tl.maximum(amax / FP8_MAX, SCALE_MIN)
    k_fp8 = tl.clamp(vals / scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)

    data_base = page * BYTES_PER_PAGE + token_in_page * HEAD_DIM
    tl.store(O_fp8 + data_base + offs_d, k_fp8)
    scale_idx = (page * BYTES_PER_PAGE + PAGE_SIZE * HEAD_DIM) // 4 + token_in_page
    tl.store(O_f32 + scale_idx, scale)
    return


@torch.no_grad()
def destindex_copy_indexer_k_dsv4(
    K: torch.Tensor,
    DestLoc: torch.Tensor,
    O_buffer: torch.Tensor,
    page_size: int,
):
    """Packed indexer-K page-slab writer (DeepSeek-V4 c4/CSA layers).

    K: [T, 128] bf16 unquantized indexer keys.
    DestLoc: [T] int — c4-pool-local token slots; must already be allocated by the caller.
        Negative slots (unmapped) are skipped.
    O_buffer: [num_pages, bytes_per_page] uint8 — one layer's slab from the c4 indexer
        PackedPagePool (128B fp8 data region + 4B fp32 scale tail per token).

    Bit-compatible with DeepseekV4MemoryManager._pack_indexer_k + PackedPagePool.write.
    """
    seq_len = DestLoc.shape[0]
    if seq_len == 0:
        return
    head_dim, scale_bytes = 128, 4

    K = K.reshape(-1, head_dim)
    assert K.shape[0] == seq_len, f"Expected K shape[0]={seq_len}, got {K.shape[0]}"
    assert K.dtype == torch.bfloat16, f"Expected bf16 indexer K, got {K.dtype}"
    bytes_per_page = O_buffer.shape[-1]
    assert O_buffer.dtype == torch.uint8 and O_buffer.is_contiguous()
    assert bytes_per_page % 4 == 0
    assert bytes_per_page >= page_size * (head_dim + scale_bytes)

    flat = O_buffer.view(-1)
    _fwd_kernel_destindex_copy_indexer_k_dsv4[(seq_len,)](
        K,
        DestLoc,
        flat.view(torch.float8_e4m3fn),
        flat.view(torch.float32),
        K.stride(0),
        K.stride(1),
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        SCALE_MIN=1e-4,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        num_warps=1,
        num_stages=1,
    )
    return
