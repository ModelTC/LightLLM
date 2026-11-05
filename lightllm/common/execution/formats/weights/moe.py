import torch


def validate_moe_weights(
    offer, gate_up, down, gate_up_scale, down_scale, correction_bias, expert_scale, gate_up_bias=None, down_bias=None
):
    spec, format, device = offer.spec, offer.format, gate_up.device
    count = spec.experts + offer.fused_shared_experts
    width = spec.intermediate_size // format.shard[1]
    for tensor, shape in (
        (gate_up, (count, 2 * width, spec.hidden_size)),
        (down, (count, spec.hidden_size, width)),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != format.dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError("MoE native weight view does not match the selected format")
    for scale, weight in ((gate_up_scale, gate_up), (down_scale, down)):
        if format.encoding == "plain":
            if scale is not None:
                raise ValueError("plain MoE must not have quantization scales")
        elif (
            scale is None
            or scale.shape != (count, weight.shape[1] // 128, weight.shape[2] // 128)
            or scale.dtype != torch.float32
            or scale.device != device
            or not scale.is_contiguous()
        ):
            raise ValueError("MoE block scale view does not match the weight")
    for tensor in (correction_bias, expert_scale):
        if tensor is not None and (
            tensor.shape != (spec.experts,)
            or tensor.device != device
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not tensor.is_contiguous()
        ):
            raise ValueError("MoE routing parameter must have one floating-point value per routed expert")
    if spec.scoring == "topk_softmax" and correction_bias is not None:
        raise ValueError("top-k softmax does not support correction bias")
    biased = "gate_up_bias" in format.fields
    for tensor, shape in ((gate_up_bias, (count, 2 * width)), (down_bias, (count, spec.hidden_size))):
        if not biased:
            if tensor is not None:
                raise ValueError("expert bias was not declared by the weight format")
        elif (
            tensor is None
            or tensor.shape != shape
            or tensor.dtype not in (spec.dtype, torch.float32)
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError("expert bias does not match the bound weight format")
