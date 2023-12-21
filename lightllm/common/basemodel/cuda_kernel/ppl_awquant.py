import torch

def CONTIGUOUS_TENSOR(tensor: torch.Tensor):
    """ Helper function """
    if tensor.is_contiguous(): return tensor
    else: return tensor.contiguous()

def skiprmsnorm_ppl(x, weight, skip=None, eps=1e-6):
    from lightllm_ppl_int8_kernel import SkipRmsNormForward_fp16_i8
    if skip is None: skip = torch.zeros_like(x)
    return SkipRmsNormForward_fp16_i8(
        CONTIGUOUS_TENSOR(x), CONTIGUOUS_TENSOR(weight), 
        CONTIGUOUS_TENSOR(skip), eps)

def gatesilu_i32_i8_ppl(x, y, x_scale, y_scale, token_scale):
    from lightllm_ppl_int8_kernel import GateSilu_i32_i8
    return GateSilu_i32_i8(
        CONTIGUOUS_TENSOR(x), CONTIGUOUS_TENSOR(y),
        CONTIGUOUS_TENSOR(token_scale), CONTIGUOUS_TENSOR(x_scale),
        CONTIGUOUS_TENSOR(y_scale))

def matmul_i8_i32_ppl(
    A: torch.Tensor,
    B: torch.Tensor,
    selected_algo: int = -1,
    split_k_slices: int = 1,
) -> torch.Tensor:
    from lightllm_ppl_int8_kernel import GemmForward_i8_i32
    return GemmForward_i8_i32(
        CONTIGUOUS_TENSOR(A), CONTIGUOUS_TENSOR(B), selected_algo, split_k_slices)

def dynamic_channelwise_quant_fp16_i8_ppl(x: torch.Tensor, channel_idx=0, tp_rank=8):
    x = x.transpose(0, 1).to(dtype=torch.float16).cuda(tp_rank)
    from lightllm_ppl_int8_kernel import QuantizeTensor_LG
    assert channel_idx < x.ndim, "channel index out of range"
    # reorder channel to first dimension, then invoke group quantize impl.
    num_of_channel = x.shape[channel_idx]
    element_per_channel = x.numel() // num_of_channel
    _ = x.transpose(channel_idx, 0)
    qt, scale = QuantizeTensor_LG(_, element_per_channel)
    qt = qt.view_as(_).transpose(0, channel_idx)
    return qt, scale

def channel_token_dequant_i32_fp16_ppl(x: torch.Tensor, scale_tokenwise: torch.Tensor, scale_channelwise: torch.Tensor):
    from lightllm_ppl_int8_kernel import PerTokenPerChannelDequant_i32_fp16
    return PerTokenPerChannelDequant_i32_fp16(CONTIGUOUS_TENSOR(x), CONTIGUOUS_TENSOR(scale_tokenwise), CONTIGUOUS_TENSOR(scale_channelwise))
