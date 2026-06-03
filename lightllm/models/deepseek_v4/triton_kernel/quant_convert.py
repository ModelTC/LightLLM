import torch

# DeepSeek-V4-Flash ships weights in two quantized formats:
#   * non-expert linears: FP8 e4m3 with block-[128,128] scales stored as float8_e8m0fnu (ue8m0)
#   * routed experts:      FP4 e2m1 packed 2-per-byte (stored as int8) with group-32 ue8m0 scales
# Hopper (H200) has no native SM100 MegaMoE path. Non-expert FP8 weights can run directly through
# DeepGEMM. Routed FP4 experts are converted blockwise to FP8, avoiding a full bf16 expansion.

# OCP E2M1 magnitude table for the 3 low bits (sign = bit 3). torch.float4_e2m1fn_x2 packs two
# such codes per byte, low nibble = lower (even) logical index.
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """float8_e8m0fnu encodes 2**(byte-127); torch decodes it correctly on .to(float32)."""
    return scale.to(torch.float32)


def dequant_fp8_block_to_bf16(weight_e4m3: torch.Tensor, scale_e8m0: torch.Tensor, block_size: int = 128):
    """De-quantize an FP8 e4m3 weight [out, in] with block-[bs,bs] ue8m0 scale to bf16."""
    from lightllm.models.deepseek2.triton_kernel.weight_dequant import weight_dequant

    w = weight_e4m3.cuda().contiguous()
    s = e8m0_to_fp32(scale_e8m0).cuda().contiguous()
    # weight_dequant runs with torch default dtype for the output; force bf16 result.
    return weight_dequant(w, s, block_size)


def cast_e2m1fn_to_e4m3fn(weight_int8: torch.Tensor, scale_e8m0: torch.Tensor):
    """Cast packed FP4 e2m1 expert weights to FP8 e4m3 with block-128 fp32 scales.

    This follows the DeepSeek-V4 reference converter, but returns the scale in fp32 because
    LightLLM's DeepGEMM FP8 weight pack stores block scales as fp32.
    """
    assert weight_int8.dtype == torch.int8
    assert weight_int8.ndim == 2
    out_dim, packed_in = weight_int8.shape
    in_dim = packed_in * 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale_e8m0.shape[0] == out_dim
    assert scale_e8m0.shape[1] == in_dim // fp4_block_size

    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=weight_int8.device,
    )
    packed = weight_int8.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    vals = torch.stack([table[low.long()], table[high.long()]], dim=-1).reshape(out_dim, in_dim)

    # 6.0 * 2**6 fits in e4m3fn (384 < 448), while 6.0 * 2**7 would overflow.
    max_offset_bits = 6
    block_out = out_dim // fp8_block_size
    block_in = in_dim // fp8_block_size

    vals = vals.view(block_out, fp8_block_size, block_in, fp8_block_size).transpose(1, 2)
    scale = scale_e8m0.float().view(block_out, fp8_block_size, block_in, -1).transpose(1, 2).flatten(2)
    block_scale = scale.amax(dim=-1, keepdim=True) / (2**max_offset_bits)
    offset = scale / block_scale
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    vals = (vals * offset).transpose(1, 2).reshape(out_dim, in_dim)
    block_scale = block_scale.squeeze(-1).to(torch.float8_e8m0fnu).to(torch.float32)
    return vals.to(torch.float8_e4m3fn), block_scale


def dequant_fp4_group_to_bf16(weight_int8: torch.Tensor, scale_e8m0: torch.Tensor, group_size: int = 32):
    """De-quantize an int8-packed FP4 e2m1 weight to bf16.

    weight_int8: [out, in // 2] int8 (two e2m1 codes per byte, low nibble = even index).
    scale_e8m0:  [out, in // group_size] ue8m0 (one scale per group_size logical elements along K).
    returns:     [out, in] bf16.
    """
    w = weight_int8.cuda()
    out, packed_in = w.shape
    in_dim = packed_in * 2
    b = w.to(torch.int32).bitwise_and(0xFF)
    lut = torch.tensor(_E2M1_MAG, dtype=torch.float32, device=w.device)

    def _decode(nib: torch.Tensor) -> torch.Tensor:
        mag = lut[nib.bitwise_and(0x7)]
        neg = nib.bitwise_and(0x8).bool()
        return torch.where(neg, -mag, mag)

    lo = _decode(b.bitwise_and(0xF))
    hi = _decode(b.bitwise_right_shift(4).bitwise_and(0xF))
    vals = torch.stack([lo, hi], dim=-1).reshape(out, in_dim)  # [out, in]
    s = e8m0_to_fp32(scale_e8m0).cuda()  # [out, in//group_size]
    s = s.repeat_interleave(group_size, dim=1)[:, :in_dim]
    return (vals * s).to(torch.bfloat16)
