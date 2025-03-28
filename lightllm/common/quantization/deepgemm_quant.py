import os
import torch
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
import torch.nn.functional as F
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_block_gemm_kernel import w8a8_block_fp8_matmul

try:
    HAS_DEEPGEMM = True
    import deep_gemm
    from deep_gemm import ceil_div, get_col_major_tma_aligned_tensor
except:
    HAS_DEEPGEMM = False


# copy from
# https://github.com/deepseek-ai/DeepGEMM/blob/bd2a77552886b98c205af12f8d7d2d61247c4b27/deep_gemm/jit_kernels/utils.py#L58
def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


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
            padded_m = get_tma_aligned_size(m, 4)  # the dtype of input_scale is torch.float32
            input_scale = torch.empty(
                (k // self.block_size, padded_m), dtype=torch.float32, device=input_tensor.device
            ).t()
            qinput_tensor = self.cache_manager.alloc_tensor(
                (m, k), qweight.dtype, device=qweight.device, is_graph_out=False
            )
            per_token_group_quant_fp8(input_tensor, self.block_size, qinput_tensor, input_scale)
            input_scale = input_scale[:m, :]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor(
                    (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
                )
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        if n % 128 != 0:
            w8a8_block_fp8_matmul(
                qinput_tensor,
                qweight,
                input_scale,
                weight_scale,
                out,
                (self.block_size, self.block_size),
                dtype=input_tensor.dtype,
            )
        else:
            deep_gemm.gemm_fp8_fp8_bf16_nt([qinput_tensor, input_scale], [qweight.t(), weight_scale.t()], out)
        return out
