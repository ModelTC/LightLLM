import os
import torch
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
import torch.nn.functional as F

try:
    HAS_VLLM = True
    from lightllm.common.vllm_kernel import _custom_ops as ops
except:
    HAS_VLLM = False


class vLLMBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert HAS_VLLM, "vllm is not installed, you can't use quant api of it"
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor):
        """ """
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        pass


@QUANTMETHODS.register(["vllm-w8a8"])
class vLLMw8a8QuantizationMethod(vLLMBaseQuantizationMethod):
    def __init__(self):
        super().__init__()

    def quantize(self, weight: torch.Tensor):
        if isinstance(weight, tuple):
            return (weight[0].transpose(0, 1).cuda(self.device_id_),) + weight[1:]
        weight = weight.float()
        scale = weight.abs().max(dim=-1)[0] / 127
        weight = weight.transpose(0, 1) / scale.reshape(1, -1)
        weight = torch.round(weight.clamp(min=-128, max=127)).to(dtype=torch.int8)
        return weight.cuda(self.device_id_), scale.cuda(self.device_id_)

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True):
        input_scale = None
        if len(weights) == 3:
            qweight, weight_scale, input_scale = weights
        elif len(weights) == 2:
            qweight, weight_scale = weights
        else:
            raise ValueError("vllm-quant Weights must be a tuple of length 2 or 3.")

        x_q, x_scale, x_zp = ops.scaled_int8_quant(input_tensor, scale=input_scale, azp=None, symmetric=True)
        m = input_tensor.shape[0]
        n = qweight.shape[1]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor(
                    (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
                )
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        torch.ops._C.cutlass_scaled_mm(out, x_q, qweight, x_scale, weight_scale, bias)
        return out


@QUANTMETHODS.register(["vllm-fp8w8a8"])
class vLLMFP8w8a8QuantizationMethod(vLLMBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.is_moe = False
        # PINGPONG_FP8_GEMM is per tensor quant way.
        self.use_pingpong_fp8_gemm = os.getenv("ENABLE_PINGPONG_FP8_GEMM", "0").upper() in ["ON", "TRUE", "1"]

        if self.use_pingpong_fp8_gemm:
            self.quantize = self.quantize_pingpong_fp8
            self.apply = self.apply_pingpong_fp8
        else:
            self.quantize = self.quantize_scaled_mm_fp8
            self.apply = self.apply_scaled_mm_fp8

    def quantize(self, weight: torch.Tensor):
        raise Exception("This function needs to be bound.")

    def quantize_scaled_mm_fp8(self, weight: torch.Tensor):
        if self.is_moe:
            return self.quantize_moe(weight)
        qweight, weight_scale = ops.scaled_fp8_quant(
            weight.contiguous().cuda(self.device_id_), scale=None, use_per_token_if_dynamic=True
        )
        return qweight.transpose(0, 1), weight_scale

    def quantize_pingpong_fp8(self, weight: torch.Tensor):
        if self.is_moe:
            return self.quantize_moe(weight)
        qweight, weight_scale = ops.scaled_fp8_quant(
            weight.contiguous().cuda(), scale=None, use_per_token_if_dynamic=False
        )
        return qweight.transpose(0, 1), weight_scale

    def quantize_moe(self, weight):
        num_experts = weight.shape[0]
        qweights = []
        weight_scales = []
        qweights = torch.empty_like(weight, dtype=torch.float8_e4m3fn).cuda(self.device_id_)
        for i in range(num_experts):
            qweight, weight_scale = ops.scaled_fp8_quant(
                weight[i].contiguous().cuda(self.device_id_), scale=None, use_per_token_if_dynamic=False
            )
            qweights[i] = qweight
            weight_scales.append(weight_scale)
        weight_scale = torch.cat(weight_scales, dim=0).reshape(-1)
        return qweights, weight_scale

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True):
        raise Exception("This function needs to be bound.")

    def apply_scaled_mm_fp8(
        self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True
    ):
        x_q, x_scale = ops.scaled_fp8_quant(input_tensor, scale=None, scale_ub=None, use_per_token_if_dynamic=True)
        m = input_tensor.shape[0]
        n = weights[0].shape[1]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor(
                    (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
                )
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        torch.ops._C.cutlass_scaled_mm(out, x_q, weights[0], x_scale, weights[1], bias)
        return out

    def apply_pingpong_fp8(
        self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True
    ):
        x_q, x_scale = ops.scaled_fp8_quant(input_tensor, scale=None, scale_ub=None, use_per_token_if_dynamic=False)
        assert bias is None
        m = input_tensor.shape[0]
        n = weights[0].shape[1]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor(
                    (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
                )
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)

        from fp8_pingpong_gemm import cutlass_scaled_mm

        return cutlass_scaled_mm(x_q, weights[0], x_scale, weights[1], out)
