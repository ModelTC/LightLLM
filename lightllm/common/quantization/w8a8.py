import torch
from typing import Optional

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS
from lightllm.common.basemodel.layer_weights.meta_weights.platform_op import PlatformAwareOp
from lightllm.common.basemodel.triton_kernel.quantization.scaled_mm_per_token_kernel import (
    fp8_scaled_mm_per_token,
    int8_scaled_mm_per_token,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# Conditional imports for optional backends
try:
    from lightllm.utils.vllm_utils import HAS_VLLM

    if HAS_VLLM:
        from lightllm.utils.vllm_utils import vllm_ops, cutlass_scaled_mm
    else:
        vllm_ops = None
        cutlass_scaled_mm = None
except ImportError:
    HAS_VLLM = False
    vllm_ops = None
    cutlass_scaled_mm = None


try:
    from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops

    if HAS_LIGHTLLM_KERNEL:

        def scaled_fp8_quant(tensor, *args, **kwargs):
            return light_ops.per_token_quant_bf16_fp8(tensor)

    else:
        if HAS_VLLM:
            scaled_fp8_quant = vllm_ops.scaled_fp8_quant
        else:
            scaled_fp8_quant = None
except ImportError:
    HAS_LIGHTLLM_KERNEL = False
    if HAS_VLLM:
        scaled_fp8_quant = vllm_ops.scaled_fp8_quant
    else:
        scaled_fp8_quant = None


@QUANTMETHODS.register(["w8a8", "vllm-w8a8"])
class W8A8Quantization(QuantizationMethod, PlatformAwareOp):
    def __init__(self):
        super().__init__()
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "w8a8"

    def create_weight(
        self, out_dim: int, in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> WeightPack:
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.int8).cuda(device_id)
        weight_scale = torch.empty(expert_prefix + (out_dim,), dtype=torch.float32).cuda(device_id)
        return WeightPack(weight=weight, weight_scale=weight_scale)

    def quantize(self, weight: torch.Tensor, output: WeightPack, offset: int = 0) -> None:
        weight = weight.float().cuda(self.device_id_)
        scale = weight.abs().max(dim=-1)[0] / 127
        weight = weight / scale.reshape(-1, 1)
        weight = torch.round(weight.clamp(min=-128, max=127)).to(dtype=torch.int8)
        output.weight[offset : offset + weight.shape[0]].copy_(weight)
        output.weight_scale[offset : offset + weight.shape[0]].copy_(scale)
        return

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward(
            input_tensor=input_tensor,
            weight_pack=weight_pack,
            out=out,
            workspace=workspace,
            use_custom_tensor_mananger=use_custom_tensor_mananger,
            bias=bias,
        )

    def _triton_forward(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        use_custom_tensor_mananger: bool,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:

        qweight = weight_pack.weight.t()
        weight_scale = weight_pack.weight_scale

        # TODO: support fp8 quantization triton

        x_q, x_scale = scaled_fp8_quant(input_tensor, scale=None, scale_ub=None, use_per_token_if_dynamic=True)

        m = input_tensor.shape[0]
        n = qweight.shape[1]

        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor((m, n), input_tensor.dtype, device=input_tensor.device)
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)

        out = int8_scaled_mm_per_token(x_q, qweight, x_scale, weight_scale, input_tensor.dtype, out)

        if bias is not None:
            out.add_(bias)
        return out

    def _cuda_forward(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        use_custom_tensor_mananger: bool,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qweight = weight_pack.weight.t()
        weight_scale = weight_pack.weight_scale

        x_q, x_scale, x_zp = vllm_ops.scaled_int8_quant(input_tensor, scale=None, azp=None, symmetric=True)

        m = input_tensor.shape[0]
        n = qweight.shape[1]

        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor((m, n), input_tensor.dtype, device=input_tensor.device)
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)

        cutlass_scaled_mm(out, x_q, qweight, x_scale, weight_scale, bias)
        return out


class Fp8W8A8Quantization(QuantizationMethod, PlatformAwareOp):
    def __init__(self):
        super().__init__()
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager
        self.is_moe = False
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "f8w8a8"

    def quantize(self, weight: torch.Tensor, output: WeightPack, offset: int = 0) -> None:
        """Quantize weights using per-token FP8 quantization."""
        qweight, weight_scale = scaled_fp8_quant(
            weight.cuda(self.device_id_), scale=None, use_per_token_if_dynamic=True
        )
        output.weight[offset : offset + qweight.shape[0], :].copy_(qweight)
        output.weight_scale[offset : offset + weight_scale.shape[0]].copy_(weight_scale.view(-1))
        return

    def create_weight(
        self, out_dim: int, in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> WeightPack:
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.float8_e4m3fn).cuda(device_id)
        if self.is_moe:
            assert num_experts > 1, "Number of experts must be greater than 1 for MOE"
            # per-tensor weight quantization for moe
            weight_scale = torch.empty((num_experts,), dtype=torch.float32).cuda(device_id)
        else:
            weight_scale = torch.empty(expert_prefix + (out_dim,), dtype=torch.float32).cuda(device_id)
        return WeightPack(weight=weight, weight_scale=weight_scale)

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward(input_tensor, weight_pack, out, use_custom_tensor_mananger, bias)

    def _cuda_forward(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor],
        use_custom_tensor_mananger: bool,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qweight = weight_pack.weight.t()
        weight_scale = weight_pack.weight_scale

        x_q, x_scale = scaled_fp8_quant(input_tensor, scale=None, scale_ub=None, use_per_token_if_dynamic=True)

        m = input_tensor.shape[0]
        n = qweight.shape[1]

        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor((m, n), input_tensor.dtype, device=input_tensor.device)
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)

        cutlass_scaled_mm(out, x_q, qweight, x_scale, weight_scale, bias)
        return out

    def _apply_triton(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor],
        use_custom_tensor_mananger: bool,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qweight = weight_pack.weight.t()
        weight_scale = weight_pack.weight_scale

        # TODO: support fp8 quantization triton

        x_q, x_scale = scaled_fp8_quant(input_tensor, scale=None, scale_ub=None, use_per_token_if_dynamic=True)

        m = input_tensor.shape[0]
        n = qweight.shape[1]

        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor((m, n), input_tensor.dtype, device=input_tensor.device)
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)

        out = fp8_scaled_mm_per_token(x_q, qweight, x_scale, weight_scale, input_tensor.dtype, out)

        if bias is not None:
            out.add_(bias)
        return out
