import os
import torch
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
import torch.nn.functional as F
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops, cutlass_scaled_mm
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops

if HAS_VLLM:
    awq_dequantize = vllm_ops.awq_dequantize
    awq_gemm = vllm_ops.awq_gemm


class AWQBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert HAS_VLLM, "vllm are not installed, you can't use quant api of them."
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor):
        """ """
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        pass


@QUANTMETHODS.register("awq")
class AWQW4A16QuantizationMethod(AWQBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.pack_factor = 8
        self.weight_scale_suffix = "scales"
        self.weight_zero_point_suffix = "qzeros"
        self.weight_suffix = "qweight"

    def get_name(self):
        return "awq"

    def quantize(self, weight: torch.Tensor):
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True):
        qweight, weight_scale, qzeros = weights

        NEED_DEQUANT_WEIGHT = input_tensor.shape[:-1].numel() >= 256
        if NEED_DEQUANT_WEIGHT:
            fpweight = awq_dequantize(qweight, weight_scale, qzeros, 0, 0, 0)
            out = torch.matmul(input_tensor, fpweight)
        else:
            out = awq_gemm(input_tensor, qweight, weight_scale, qzeros, self.pack_factor)

        if bias is not None:
            out.add_(bias)
        return out
