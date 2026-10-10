import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv_flashmla_fp8(
    KV,
    MemIndex,
    PackedKV,
    kv_stride_s,
    kv_stride_d,
    packed_stride_s,
):
    token = tl.program_id(0)
    mem_index = tl.load(MemIndex + token).to(tl.int64)
    packed_row = PackedKV + mem_index * packed_stride_s

    # The V4 BF16 tail stores actual NoPE values, without an extra RoPE region.
    for group in range(7):
        offsets = group * 64 + tl.arange(0, 64)
        values = tl.load(KV + token * kv_stride_s + offsets * kv_stride_d).to(tl.float32)
        absmax = tl.max(tl.abs(values), 0)
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(absmax / 448.0, 1e-4))))
        quantized = tl.clamp(values / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(packed_row + offsets, quantized.to(tl.uint8, bitcast=True))
        tl.store(packed_row + 576 + group, (scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8))

    tail_offsets = tl.arange(0, 64)
    tail_values = tl.load(KV + token * kv_stride_s + (448 + tail_offsets) * kv_stride_d)
    tl.store((packed_row + 448).to(tl.pointer_type(tl.bfloat16)) + tail_offsets, tail_values)
    tl.store(packed_row + 583, 0)


@torch.no_grad()
def destindex_copy_kv_flashmla_fp8(kv: torch.Tensor, mem_index: torch.Tensor, packed_kv: torch.Tensor):
    assert kv.shape[1:] == (1, 512)
    assert packed_kv.dtype == torch.uint8 and packed_kv.shape[1] == 1
    assert packed_kv.shape[2] >= 584 and packed_kv.stride(2) == 1
    _fwd_kernel_destindex_copy_kv_flashmla_fp8[(mem_index.numel(),)](
        kv,
        mem_index.contiguous(),
        packed_kv,
        kv.stride(0),
        kv.stride(2),
        packed_kv.stride(0),
        num_warps=4,
        num_stages=1,
    )
