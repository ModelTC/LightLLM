import torch
from typing import Tuple

def CONTIGUOUS_TENSOR(tensor: torch.Tensor):
    """ Helper function """
    if tensor.is_contiguous(): return tensor
    else: return tensor.contiguous()

def int4Matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    from lightllm_quik_kernel.matmul import int4Matmul
    return int4Matmul(
        CONTIGUOUS_TENSOR(input), CONTIGUOUS_TENSOR(weight))

def int8Matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    from lightllm_quik_kernel.matmul import int8Matmul
    return int8Matmul(
        CONTIGUOUS_TENSOR(input), CONTIGUOUS_TENSOR(weight))

def asym_quantize(
    src: torch.Tensor,
    int_indices: torch.Tensor,
    fp_indices: torch.Tensor,
    bits: int,
)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    asymmetric quantization for activations of QUIK

    Returns:
        quantized_weights: int8 tensor of shape (rows, len(int_indices)) for bits=8 or (rows, len(int_indices)/2) for bits=4
        meta: float16 tensor of shape (2*rows,), layout is [scale, zero, scale, zero ...]
        float_weights: float16 tensor of shape (rows, len(fp_indices))
    """
    from lightllm_quik_kernel.asymmetric import quantize

    return quantize(
        CONTIGUOUS_TENSOR(src),
        CONTIGUOUS_TENSOR(int_indices),
        CONTIGUOUS_TENSOR(fp_indices),
        bits)

def asym_dequantize(
    int_result: torch.Tensor,
    act_meta: torch.Tensor,
    weight_scale: torch.Tensor,
    wReduced: torch.Tensor,
    fp_result: torch.Tensor,
    bits: int
)->torch.Tensor:
    """
    asymmetric dequantization for activations of QUIK

    Args:
        int_result: the result of matmul(q_act, q_weight)
        act_meta: the packed tensor of activation scale and zero, layout is [scale, zero, ...]
        weight_scale: the tensor of weight scales
        wReduced: the constant term for dequantization
        fp_result: the result of matmul(fp_act, fp_weight)
        bits: 4 or 8
    Returns:
        the tensor of dequantization result
    """
    from lightllm_quik_kernel.asymmetric import dequantize

    return dequantize(
        CONTIGUOUS_TENSOR(int_result),
        CONTIGUOUS_TENSOR(act_meta),
        CONTIGUOUS_TENSOR(weight_scale),
        CONTIGUOUS_TENSOR(wReduced),
        CONTIGUOUS_TENSOR(fp_result),
        bits)

def sym_quantize(
    src: torch.Tensor,
    scale: torch.Tensor,
    bits: int
)->torch.Tensor:
    """
    symmetric quantization for activations of QUIK

    Args:
        src: the tensor to be quantized
        scale: the scale tensor calculated externally using S = (fp_max - fp_min)/ (q_max - q_min)
            Example: (torch.max(torch.abs(x), dim=1)[0].unsqueeze(1) / (1 << (bits - 1) - 1)).to(torch.float16)
        bits: 4 or 8
    Returns:
        the quantized tensor
    """
    from lightllm_quik_kernel.symmetric import quantize

    return quantize(
        CONTIGUOUS_TENSOR(src),
        CONTIGUOUS_TENSOR(scale),
        bits)

def sym_dequantize(
    int_result: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    fp_result: torch.Tensor,
    bits: int
)->torch.Tensor:
    """
    symmetric dequantization for activations of QUIK

    Args:
        int_result: the result of matmul(q_act, q_weight)
        act_scale: the tensor of weight scales
        weight_scale: the tensor of weight scales
        fp_result: the result of matmul(fp_act, fp_weight)
        bits: 4 or 8

    Return:
        the dequantized result
    """
    from lightllm_quik_kernel.symmetric import dequantize

    return dequantize(
        CONTIGUOUS_TENSOR(int_result),
        CONTIGUOUS_TENSOR(act_scale),
        CONTIGUOUS_TENSOR(weight_scale),
        CONTIGUOUS_TENSOR(fp_result),
        bits)