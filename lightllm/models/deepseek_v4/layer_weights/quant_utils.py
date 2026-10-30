"""Checkpoint FP8 format conversion owned by the weight loaders."""

import torch


def _dequant_fp8_block_to_bf16(weight_e4m3: torch.Tensor, scale_e8m0: torch.Tensor, block_size: int = 128):
    """De-quantize an FP8 e4m3 weight [out, in] with block-[bs,bs] ue8m0 scale to bf16."""
    from lightllm.models.deepseek2.triton_kernel.weight_dequant import weight_dequant

    w = weight_e4m3.cuda().contiguous()
    s = scale_e8m0.to(dtype=torch.float32, device=w.device).contiguous()
    return weight_dequant(w, s, block_size)


def prepare_fp8_weights(weights: dict, matrix_weights, data_type: torch.dtype, block_size: int = 128) -> None:
    """Prepare checkpoint FP8 weights and scales for each matrix's runtime format in place."""
    for matrix_weight in matrix_weights:
        weight_names = matrix_weight.weight_names
        # BMM weights have no quantization scale names and always load dense values.
        scale_names = (
            [None] * len(weight_names) if matrix_weight.quant_method is None else matrix_weight.weight_scale_names
        )
        for weight_name, scale_name in zip(weight_names, scale_names):
            checkpoint_scale_name = weight_name.removesuffix(".weight") + ".scale"
            if checkpoint_scale_name not in weights:
                continue
            if checkpoint_scale_name == scale_name:
                continue
            scale = weights[checkpoint_scale_name]
            if scale_name is None:
                weights[weight_name] = _dequant_fp8_block_to_bf16(weights[weight_name], scale, block_size).to(data_type)
            else:
                # Quantized paths can load scales before the weight's shard arrives.
                weights[scale_name] = scale.to(torch.float32)
            del weights[checkpoint_scale_name]
