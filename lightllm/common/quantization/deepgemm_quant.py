import os
import torch
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
import torch.nn.functional as F
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
    tma_align_input_scale,
)

try:
    HAS_DEEPGEMM = True
    import deep_gemm
except:
    HAS_DEEPGEMM = False


class DeepGEMMBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager
        assert HAS_DEEPGEMM, "deepgemm is not installed, you can't use quant api of it"

    def quantize(self, weight: torch.Tensor):
        """ """
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        pass


@QUANTMETHODS.register(["deepgemm-fp8w8a8-b128"])
class DeepGEMMFP8w8a8B128QuantizationMethod(DeepGEMMBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.block_size = 128
        self.weight_scale_suffix = "weight_scale_inv"
        self.act_scale_suffix = None  # no support for static input tensor scale for ds model.

    def quantize(self, weight: torch.Tensor):
        from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_block_quant_kernel import weight_quant

        return weight_quant(weight, self.block_size)

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True):
        if len(weights) == 3:
            qweight, weight_scale, input_scale = weights
        else:
            qweight, weight_scale = weights
            input_scale = None
        alloc_func = torch.empty if not use_custom_tensor_mananger else self.cache_manager.empty
        m, k = input_tensor.shape
        n = weights[0].shape[1]
        if input_scale is None:
            qinput_tensor, input_scale = per_token_group_quant_fp8(
                input_tensor,
                self.block_size,
                dtype=qweight.dtype,
                column_major_scales=True,
                scale_tma_aligned=True,
                alloc_func=alloc_func,
            )

        if out is None:
            out = alloc_func((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        _deepgemm_fp8_nt((qinput_tensor, input_scale), (qweight.t(), weight_scale.t()), out)
        return out


def _deepgemm_fp8_nt(a_tuple, b_tuple, out):
    if HAS_DEEPGEMM:
        if hasattr(deep_gemm, "gemm_fp8_fp8_bf16_nt"):
            return deep_gemm.gemm_fp8_fp8_bf16_nt([a_tuple[0], a_tuple[1]], [b_tuple[0], b_tuple[1]], out)
        if hasattr(deep_gemm, "fp8_gemm_nt"):
            return deep_gemm.fp8_gemm_nt((a_tuple[0], a_tuple[1]), (b_tuple[0], b_tuple[1]), out)
    raise RuntimeError("deep_gemm does not provide fp8 NT GEMM kernel in this version")
