import torch
from typing import Optional

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS
from lightllm.common.quantization.backend import QUANT_BACKEND, BackendType
from lightllm.common.basemodel.triton_kernel.quantization.scaled_mm_per_token_kernel import fp8_scaled_mm_per_token
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

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


@QUANTMETHODS.register(["fp8-per-token", "fp8w8a8"])
class FP8PerTokenQuantization(QuantizationMethod):
    def __init__(self):
        super().__init__()
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager
        self.is_moe = False
        self.has_weight_scale = True
        self.has_weight_zero_point = False
        self._backend = QUANT_BACKEND.get_backend("fp8-per-token")
        logger.info(f"FP8PerTokenQuantization using backend: {self._backend.name}")

    @property
    def method_name(self):
        return "fp8-per-token"

    def quantize(self, weight: torch.Tensor, output: WeightPack, offset: int = 0) -> None:
        """Quantize weights using per-token FP8 quantization."""
        if self.is_moe:
            return self._quantize_moe(weight, output, offset)

        if scaled_fp8_quant is None:
            raise RuntimeError("No FP8 quantization kernel available. Install vLLM or lightllm-kernel.")

        qweight, weight_scale = scaled_fp8_quant(
            weight.cuda(self.device_id_), scale=None, use_per_token_if_dynamic=True
        )
        output.weight[offset : offset + qweight.shape[0], :].copy_(qweight)
        output.weight_scale[offset : offset + weight_scale.shape[0]].copy_(weight_scale.view(-1))
        return

    def _quantize_moe(self, weight: torch.Tensor, output: WeightPack, offset: int) -> None:
        if scaled_fp8_quant is None:
            raise RuntimeError("No FP8 quantization kernel available. Install vLLM or lightllm-kernel.")

        num_experts = weight.shape[0]
        qweights = torch.empty_like(weight, dtype=torch.float8_e4m3fn).cuda(self.device_id_)
        weight_scales = []
        for i in range(num_experts):
            qweight, weight_scale = scaled_fp8_quant(
                weight[i].contiguous().cuda(self.device_id_), scale=None, use_per_token_if_dynamic=True
            )
            qweights[i] = qweight
            weight_scales.append(weight_scale)
        weight_scale = torch.stack(weight_scales, dim=0).contiguous()
        output.weight.copy_(qweights)
        output.weight_scale.copy_(weight_scale)
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
        if self._backend == BackendType.TRITON:
            return self._apply_triton(input_tensor, weight_pack, out, use_custom_tensor_mananger, bias)
        else:
            return self._apply_vllm(input_tensor, weight_pack, out, use_custom_tensor_mananger, bias)

    def _apply_vllm(
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

        if scaled_fp8_quant is None:
            raise RuntimeError("No FP8 quantization kernel available. Install vLLM or lightllm-kernel.")

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

    def create_weight(
        self, out_dim: int, in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> WeightPack:
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.float8_e4m3fn).cuda(device_id)
        weight_scale = torch.empty(expert_prefix + (out_dim,), dtype=torch.float32).cuda(device_id)
        return WeightPack(weight=weight, weight_scale=weight_scale)
