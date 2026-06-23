import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fwd_kernel_destindex_copy_kv_flashmla_dsv4(
    KV,
    Dest_loc,
    O_fp8,
    O_bf16,
    O_u8,
    stride_kv_bs,
    stride_kv_d,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    SCALE_MIN: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)
    # negative dest (unmapped slot, e.g. full_to_c* rows that never closed a group) is a no-op,
    # not an OOB write into a neighboring page.
    if dest_index < 0:
        return

    page = dest_index // PAGE_SIZE
    token_in_page = dest_index % PAGE_SIZE
    data_base = page * BYTES_PER_PAGE + token_in_page * (NOPE_DIM + ROPE_DIM * 2)
    scale_base = page * BYTES_PER_PAGE + PAGE_SIZE * (NOPE_DIM + ROPE_DIM * 2) + token_in_page * SCALE_BYTES

    # nope: per-group ue8m0 quant. SCALE_BYTES(=NUM_GROUPS+1) lanes cover the exponent bytes
    # plus the trailing zero pad byte in one store. libdevice.log2 (not tl.log2, which is the
    # approx instruction) and the bit-packed 2**e keep this bit-exact with the torch oracle
    # DeepseekV4MemoryManager._pack_mla_kv.
    offs_g = tl.arange(0, SCALE_BYTES)
    offs_e = tl.arange(0, GROUP_SIZE)
    group_mask = offs_g < NUM_GROUPS
    kv_ptrs = KV + cur_index * stride_kv_bs + (offs_g[:, None] * GROUP_SIZE + offs_e[None, :]) * stride_kv_d
    vals = tl.load(kv_ptrs, mask=group_mask[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(vals), axis=1)
    scale_exp = tl.ceil(libdevice.log2(tl.maximum(amax / FP8_MAX, SCALE_MIN))).to(tl.int32)
    scale = ((scale_exp + 127) << 23).to(tl.float32, bitcast=True)
    kv_fp8 = tl.clamp(vals / scale[:, None], min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)
    tl.store(O_fp8 + data_base + offs_g[:, None] * GROUP_SIZE + offs_e[None, :], kv_fp8, mask=group_mask[:, None])
    scale_bytes = tl.where(group_mask, scale_exp + 127, 0).to(tl.uint8)
    tl.store(O_u8 + scale_base + offs_g, scale_bytes)

    # rope: bf16 passthrough into the data region right after the nope bytes
    offs_r = tl.arange(0, ROPE_DIM)
    rope = tl.load(KV + cur_index * stride_kv_bs + (NOPE_DIM + offs_r) * stride_kv_d)
    tl.store(O_bf16 + (data_base + NOPE_DIM) // 2 + offs_r, rope)
    return


@torch.no_grad()
def destindex_copy_kv_flashmla_dsv4(
    KV: torch.Tensor,
    DestLoc: torch.Tensor,
    O_buffer: torch.Tensor,
    page_size: int,
):
    """fp8_ds_mla packed page-slab writer (DeepSeek-V4 ABI, all latent pools).

    KV: [T, 512] bf16 — 448 normed-latent dims + 64 rope'd dims per token.
    DestLoc: [T] int — pool-local token slots (page = slot // page_size); the pool HOLD slot is
        a valid in-bounds row, negative slots (unmapped) are skipped. Slots must already be
        resolved/allocated by the caller.
    O_buffer: [num_pages, bytes_per_page] uint8 — one layer's slab from PackedPagePool
        (swa page=128 / c4 page=64 / c128 page=2 all share this kernel).

    Per token: 448B fp8(e4m3) in 7x64 ue8m0 groups + 128B bf16 rope in the page data region;
    7 exponent bytes (e+127) + 1 zero pad at the page scale tail. Bit-compatible with
    DeepseekV4MemoryManager._pack_mla_kv + PackedPagePool.write.
    """
    seq_len = DestLoc.shape[0]
    if seq_len == 0:
        return
    nope_dim, rope_dim, group_size = 448, 64, 64
    head_dim = nope_dim + rope_dim
    scale_bytes = nope_dim // group_size + 1

    KV = KV.reshape(-1, head_dim)
    assert KV.shape[0] == seq_len, f"Expected KV shape[0]={seq_len}, got {KV.shape[0]}"
    assert KV.dtype == torch.bfloat16, f"Expected bf16 KV (rope bytes are stored as-is), got {KV.dtype}"
    bytes_per_page = O_buffer.shape[-1]
    assert O_buffer.dtype == torch.uint8 and O_buffer.is_contiguous()
    assert bytes_per_page % 2 == 0
    assert bytes_per_page >= page_size * (nope_dim + rope_dim * 2 + scale_bytes)

    flat = O_buffer.view(-1)
    _fwd_kernel_destindex_copy_kv_flashmla_dsv4[(seq_len,)](
        KV,
        DestLoc,
        flat.view(torch.float8_e4m3fn),
        flat.view(torch.bfloat16),
        flat,
        KV.stride(0),
        KV.stride(1),
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        SCALE_MIN=1e-4,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        GROUP_SIZE=group_size,
        NUM_GROUPS=nope_dim // group_size,
        SCALE_BYTES=scale_bytes,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        num_warps=4,
        num_stages=1,
    )
    return
