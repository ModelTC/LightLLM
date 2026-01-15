import torch
from typing import Optional

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS
from lightllm.common.quantization.backend import QUANT_BACKEND, BackendType
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_block_gemm_kernel import w8a8_block_fp8_matmul
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    import deep_gemm

    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False

try:
    from lightllm.utils.vllm_utils import HAS_VLLM

    if HAS_VLLM:
        from lightllm.utils.vllm_utils import cutlass_scaled_mm
    else:
        cutlass_scaled_mm = None
except ImportError:
    HAS_VLLM = False
    cutlass_scaled_mm = None


def _deepgemm_fp8_nt(a_tuple, b_tuple, out):
    if hasattr(deep_gemm, "gemm_fp8_fp8_bf16_nt"):
        return deep_gemm.gemm_fp8_fp8_bf16_nt([a_tuple[0], a_tuple[1]], [b_tuple[0], b_tuple[1]], out)
    if hasattr(deep_gemm, "fp8_gemm_nt"):
        return deep_gemm.fp8_gemm_nt((a_tuple[0], a_tuple[1]), (b_tuple[0], b_tuple[1]), out)
    raise RuntimeError("deep_gemm does not provide fp8 NT GEMM kernel in this version")


@QUANTMETHODS.register(["fp8-block128"])
class FP8Block128Quantization(QuantizationMethod):
    def __init__(self):
        super().__init__()
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager
        self.block_size = 128
        self.weight_scale_suffix = "weight_scale_inv"
        self.has_weight_scale = True
        self.has_weight_zero_point = False

        self._backend = QUANT_BACKEND.get_backend("fp8-block128")
        logger.info(f"FP8Block128Quantization using backend: {self._backend.name}")

    @property
    def method_name(self):
        return "fp8-block128"

    def quantize(self, weight: torch.Tensor, output: WeightPack, offset: int = 0) -> None:
        from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_block_quant_kernel import weight_quant

        device = output.weight.device
        weight, scale = weight_quant(weight.cuda(device), self.block_size)
        output.weight[offset : offset + weight.shape[0], :].copy_(weight)
        output.weight_scale[offset // self.block_size : offset + weight.shape[0] // self.block_size].copy_(scale)
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
        alloc_func = torch.empty if not use_custom_tensor_mananger else self.cache_manager.empty
        m, k = input_tensor.shape

        if self._backend == BackendType.DEEPGEMM:
            return self._apply_deepgemm(input_tensor, weight_pack, out, alloc_func, bias)
        elif self._backend == BackendType.VLLM:
            return self._apply_vllm(input_tensor, weight_pack, out, alloc_func, bias)
        else:
            return self._apply_triton(input_tensor, weight_pack, out, alloc_func, bias)

    def _apply_deepgemm(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor],
        alloc_func,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        m, k = input_tensor.shape
        n = qweight.shape[0]

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

        _deepgemm_fp8_nt((qinput_tensor, input_scale), (qweight, weight_scale), out)

        if bias is not None:
            out.add_(bias)
        return out

    def _apply_vllm(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor],
        alloc_func,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qweight = weight_pack.weight.t()
        weight_scale = weight_pack.weight_scale.t()
        m, k = input_tensor.shape
        n = qweight.shape[1]

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

        if bias is not None:
            out.add_(bias)
        return out

    def _apply_triton(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor],
        alloc_func,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        m, k = input_tensor.shape
        n = qweight.shape[1]

        qinput_tensor, input_scale = per_token_group_quant_fp8(
            input_tensor, self.block_size, dtype=qweight.dtype, alloc_func=alloc_func
        )

        if out is None:
            out = alloc_func((m, n), dtype=input_tensor.dtype, device=input_tensor.device)

        w8a8_block_fp8_matmul(
            qinput_tensor,
            qweight,
            input_scale,
            weight_scale,
            out,
            (self.block_size, self.block_size),
            dtype=input_tensor.dtype,
        )

        if bias is not None:
            out.add_(bias)
        return out

    def create_weight(
        self, out_dim: int, in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> WeightPack:
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.float8_e4m3fn).cuda(device_id)
        weight_scale = torch.empty(
            expert_prefix + (out_dim // self.block_size, in_dim // self.block_size), dtype=torch.float32
        ).cuda(device_id)
        return WeightPack(weight=weight, weight_scale=weight_scale)

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack, start_idx: int) -> None:
        weight_pack.weight[start_idx : start_idx + weight.shape[0]].copy_(weight)
        return

    def load_weight_scale(self, weight_scale: torch.Tensor, weight_pack: WeightPack, start_idx: int) -> None:
        weight_pack.weight_scale[
            start_idx // self.block_size : start_idx + weight_scale.shape[0] // self.block_size
        ].copy_(weight_scale)
        return

    def load_weight_zero_point(self, weight_zero_point: torch.Tensor, weight_pack: WeightPack, start_idx: int) -> None:
        if weight_pack.weight_zero_point is not None:
            weight_pack.weight_zero_point[
                start_idx // self.block_size : start_idx + weight_zero_point.shape[0] // self.block_size
            ].copy_(weight_zero_point)
        return
