import os
import torch

from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops, cutlass_scaled_mm
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops

if TYPE_CHECKING:
    from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightPack

if HAS_VLLM:
    scaled_fp4_quant = vllm_ops.scaled_fp4_quant

# Use flashinfer for FP4 GEMM (more stable on Blackwell GPUs)
try:
    from flashinfer import mm_fp4 as flashinfer_mm_fp4

    HAS_FLASHINFER_FP4 = True
except ImportError:
    flashinfer_mm_fp4 = None
    HAS_FLASHINFER_FP4 = False

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
        return "w4a4-base"


@QUANTMETHODS.register(["vllm-fp4w4a4-b16", "fp4w4a4-b16"])
class FP4w4a4B16QuantizationMethod(BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.block_size = 16
        self.weight_scale_suffix = "weight_scale_inv"
        self.has_weight_scale = True
        self.has_weight_zero_point = True  # Used to store weight_global_scale (alpha)

    def quantize(self, weight: torch.Tensor):
        """
        Quantize weight tensor to FP4.

        Args:
            weight: Weight tensor of shape (N, K), dtype BF16/FP16

        Returns:
            qweight: FP4 quantized weight (N, K//2), dtype uint8
            weight_scale: Swizzled block scale for weight, dtype float8_e4m3fn
            weight_global_scale: Global scale for weight, used as alpha in GEMM
        """
        # Compute global scale: 2688.0 / max(|weight|)
        # 2688.0 = 6 (FP4 max) * 448 (FP8 max)
        weight_global_scale = (2688.0 / torch.amax(weight.abs().flatten())).to(torch.float32)
        # scaled_fp4_quant returns (N, K//2) and swizzled scale (N, K//16)
        # Note: the returned scale is already in swizzled layout!
        qweight, weight_scale = scaled_fp4_quant(weight.contiguous().cuda(self.device_id_), weight_global_scale)
        return qweight.t(), weight_scale.t().view(torch.uint8), weight_global_scale

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        """
        Apply FP4 quantized matrix multiplication.

        Args:
            input_tensor: Input tensor of shape (m, k), dtype BF16/FP16
            weight_pack: Weight pack containing:
                - weight: FP4 quantized weight (n, k//2), dtype uint8
                - weight_scale: Block scale for weight (n, k//block_size), dtype float8_e4m3fn
                - weight_zero_point: Global scale for weight (alpha), dtype float32
                - bias: Optional bias tensor
        """
        qweight = weight_pack.weight  # (n, k//2), uint8
        weight_scale = weight_pack.weight_scale  # swizzled, float8_e4m3fn
        weight_global_scale = weight_pack.weight_zero_point  # float32, used as alpha
        bias = weight_pack.bias

        output_dtype = input_tensor.dtype

        # Ensure input is contiguous
        input_tensor = input_tensor.contiguous()

        # Compute input global scale dynamically: 2688.0 / max(|input|)
        # 2688.0 = 6 (FP4 max) * 448 (FP8 max)
        input_amax = torch.amax(input_tensor.abs())
        input_global_scale = (2688.0 / input_amax).to(torch.float32)

        # Quantize input to FP4 with interleaved block scale
        # scaled_fp4_quant returns swizzled block scale for input
        x_fp4, x_blockscale = scaled_fp4_quant(input_tensor, input_global_scale)

        # Compute alpha for dequantization
        # alpha = 1 / (input_global_scale * weight_global_scale)
        if weight_global_scale is not None:
            alpha = (1.0 / (input_global_scale * weight_global_scale)).to(torch.float32)
        else:
            alpha = torch.tensor(1.0, dtype=torch.float32, device=input_tensor.device)

        # Validate dtypes
        assert x_fp4.dtype == torch.uint8, f"Expected x_fp4 dtype uint8, got {x_fp4.dtype}"
        assert qweight.dtype == torch.uint8, f"Expected weight dtype uint8, got {qweight.dtype}"
        assert (
            x_blockscale.dtype == torch.float8_e4m3fn
        ), f"Expected x_blockscale dtype float8_e4m3fn, got {x_blockscale.dtype}"

        # Call flashinfer FP4 GEMM with cutlass backend
        # Note: flashinfer.mm_fp4 expects:
        #   - b to be transposed
        #   - scales as uint8
        #   - weight_scale to be transposed
        out = flashinfer_mm_fp4(
            x_fp4,
            qweight,  # transpose weight
            x_blockscale.view(torch.uint8),
            weight_scale,  # transpose and view as uint8
            alpha,
            out_dtype=output_dtype,
            backend="cutlass",
        )

        if bias is not None:
            out = out + bias

        return out

    @property
    def method_name(self):
        return "vllm-fp4w4a4-b16"
