import torch


def quantize_fp8(inputs, group_size, dtype):
    from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
        lightllm_per_token_group_quant_fp8,
    )

    encoded = torch.empty_like(inputs, dtype=dtype)
    scale = torch.empty((inputs.shape[0], inputs.shape[1] // group_size), dtype=torch.float32, device=inputs.device)
    lightllm_per_token_group_quant_fp8(inputs, group_size, encoded, scale)
    return encoded, scale
