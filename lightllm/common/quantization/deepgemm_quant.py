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

        raise Exception("Not implemented")

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True):
        qweight, weight_scale, input_scale = weights
        m, k = input_tensor.shape
        n = weights[0].shape[1]
        if input_scale is None:
            input_scale = torch.empty((m, k // self.block_size), dtype=torch.float32, device=input_tensor.device)
            qinput_tensor = self.cache_manager.alloc_tensor(
                (m, k), qweight.dtype, device=qweight.device, is_graph_out=False
            )
            per_token_group_quant_fp8(input_tensor, self.block_size, qinput_tensor, input_scale)
            input_scale = tma_align_input_scale(input_scale)

        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor(
                    (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
                )
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        deep_gemm.gemm_fp8_fp8_bf16_nt([qinput_tensor, input_scale], [qweight.t(), weight_scale.t()], out)
        return out
