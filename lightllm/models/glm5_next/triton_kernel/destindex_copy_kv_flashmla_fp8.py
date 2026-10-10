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

    # FlashMLA layout: 512 FP8 values, four FP32 scales, 64 zero BF16 RoPE values.
    for group in range(4):
        offsets = group * 128 + tl.arange(0, 128)
        values = tl.load(KV + token * kv_stride_s + offsets * kv_stride_d)
        absmax = tl.max(tl.abs(values), 0)
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(absmax / 448.0, 1e-4))))
        quantized = tl.clamp(values / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(packed_row + offsets, quantized.to(tl.uint8, bitcast=True))
        tl.store((packed_row + 512).to(tl.pointer_type(tl.float32)) + group, scale)

    rope_offsets = tl.arange(0, 64)
    tl.store((packed_row + 528).to(tl.pointer_type(tl.bfloat16)) + rope_offsets, 0)


@torch.no_grad()
def destindex_copy_kv_flashmla_fp8(kv: torch.Tensor, mem_index: torch.Tensor, packed_kv: torch.Tensor):
    assert kv.shape[1:] == (1, 512)
    assert packed_kv.dtype == torch.uint8 and packed_kv.shape[1] == 1
    assert packed_kv.shape[2] >= 656 and packed_kv.stride(2) == 1
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
