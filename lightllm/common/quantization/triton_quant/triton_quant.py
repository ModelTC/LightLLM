import os
import torch
import torch.nn.functional as F
from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.common.quantization.registry import QUANTMETHODS
from .fp8.fp8w8a8_block_gemm_kernel import w8a8_block_fp8_matmul
from .fp8.fp8act_quant_kernel import per_token_group_quant_fp8


class TritonBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor):
        """ """
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        pass


@QUANTMETHODS.register(["triton-fp8w8a8-block128"])
class TritonFP8w8a8QuantizationMethod(TritonBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.is_moe = False
        self.block_size = 128

    def quantize(self, weight: torch.Tensor):
        # TODO block-wise quant kernel
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True):
        qweight, weight_scale, input_scale = weights
        m, k = input_tensor.shape
        n = qweight.shape[1]
        alloc_func = torch.empty if not use_custom_tensor_mananger else self.cache_manager.empty
        if input_scale is None:
            input_tensor_q, input_scale = per_token_group_quant_fp8(
                input_tensor, self.block_size, dtype=qweight.dtype, alloc_func=alloc_func
            )
        else:
            # TODO
            raise "statci input scale is not supported by triton fp8 block gemm kernel."
        m = input_tensor.shape[0]
        n = qweight.shape[1]
        if out is None:
            out = alloc_func((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        w8a8_block_fp8_matmul(
            input_tensor_q,
            qweight,
            input_scale,
            weight_scale,
            out,
            (self.block_size, self.block_size),
            dtype=input_tensor.dtype,
        )
        return out
