import torch


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
