import os
import torch

from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_scaled_mm_per_token_kernel import fp8_scaled_mm_per_token
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_block_gemm_kernel import w8a8_block_fp8_matmul
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops, cutlass_scaled_mm
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops

if TYPE_CHECKING:
    from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightPack

if HAS_LIGHTLLM_KERNEL:

    def scaled_fp8_quant(tensor, *args, **kwargs):
        return light_ops.per_token_quant_bf16_fp8(tensor)

else:
    if HAS_VLLM:
        scaled_fp8_quant = vllm_ops.scaled_fp8_quant

LIGHTLLM_USE_TRITON_FP8_SCALED_MM = os.getenv("LIGHTLLM_USE_TRITON_FP8_SCALED_MM", "False").upper() in [
    "ON",
    "TRUE",
    "1",
]


class BaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert HAS_VLLM, "vllm are not installed, you can't use quant api of them."
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor):
        pass

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError("Not implemented")

    @property
    def method_name(self):
        return "w8a8-base"


@QUANTMETHODS.register(["vllm-w8a8", "w8a8"])
class w8a8QuantizationMethod(BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def quantize(self, weight: torch.Tensor):
        if isinstance(weight, tuple):
            return (weight[0].transpose(0, 1).cuda(self.device_id_),) + weight[1:]
        weight = weight.float()
        scale = weight.abs().max(dim=-1)[0] / 127
        weight = weight.transpose(0, 1) / scale.reshape(1, -1)
        weight = torch.round(weight.clamp(min=-128, max=127)).to(dtype=torch.int8)
        return weight.cuda(self.device_id_), scale.cuda(self.device_id_), None

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        input_scale = None
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        bias = weight_pack.bias
        input_scale = None  # dynamic quantization for input tensor
        x_q, x_scale, x_zp = vllm_ops.scaled_int8_quant(input_tensor, scale=input_scale, azp=None, symmetric=True)
        m = input_tensor.shape[0]
        n = qweight.shape[1]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor(
                    (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
                )
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        cutlass_scaled_mm(out, x_q, qweight, x_scale, weight_scale, bias)
        return out

    @property
    def method_name(self):
        return "vllm-w8a8"


@QUANTMETHODS.register(["vllm-fp8w8a8", "fp8w8a8"])
class FP8w8a8QuantizationMethod(BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.is_moe = False
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def quantize(self, weight: torch.Tensor):
        if self.is_moe:
            return self.quantize_moe(weight)
        qweight, weight_scale = scaled_fp8_quant(
            weight.contiguous().cuda(self.device_id_), scale=None, use_per_token_if_dynamic=True
        )
        return qweight.transpose(0, 1), weight_scale, None

    def quantize_moe(self, weight: torch.Tensor):
        num_experts = weight.shape[0]
        qweights = []
        weight_scales = []
        qweights = torch.empty_like(weight, dtype=torch.float8_e4m3fn).cuda(self.device_id_)
        for i in range(num_experts):
            qweight, weight_scale = scaled_fp8_quant(
                weight[i].contiguous().cuda(self.device_id_), scale=None, use_per_token_if_dynamic=True
            )
            qweights[i] = qweight
            weight_scales.append(weight_scale)
        weight_scale = torch.stack(weight_scales, dim=0).contiguous()
        return qweights, weight_scale, None

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        bias = weight_pack.bias
        x_q, x_scale = scaled_fp8_quant(input_tensor, scale=None, scale_ub=None, use_per_token_if_dynamic=True)
        m = input_tensor.shape[0]
        n = qweight.shape[1]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor(
                    (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
                )
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        if LIGHTLLM_USE_TRITON_FP8_SCALED_MM:
            out = fp8_scaled_mm_per_token(x_q, qweight, x_scale, weight_scale, input_tensor.dtype, out)
        else:
            cutlass_scaled_mm(out, x_q, qweight, x_scale, weight_scale, bias)
        return out

    @property
    def method_name(self):
        return "vllm-fp8w8a8"


@QUANTMETHODS.register(["vllm-fp8w8a8-b128", "fp8w8a8-b128"])
class FP8w8a8B128QuantizationMethod(BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.block_size = 128
        self.weight_scale_suffix = "weight_scale_inv"
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def quantize(self, weight: torch.Tensor):

        raise Exception("Not implemented")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        bias = weight_pack.bias
        input_scale = None  # dynamic quantization for input tensor
        m, k = input_tensor.shape
        n = qweight.shape[1]
        alloc_func = torch.empty if not use_custom_tensor_mananger else self.cache_manager.empty
        if input_scale is None:
            qinput_tensor, input_scale = per_token_group_quant_fp8(
                input_tensor, self.block_size, dtype=qweight.dtype, alloc_func=alloc_func
            )
        if out is None:
            out = alloc_func((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
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
            input_scale = input_scale.t().contiguous().t()
            cutlass_scaled_mm(out, qinput_tensor, qweight, input_scale, weight_scale, bias)
        return out

    @property
    def method_name(self):
        return "vllm-fp8w8a8-b128"
