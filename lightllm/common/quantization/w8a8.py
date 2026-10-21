import os
import threading
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from .quantize_method import QuantizationMethod, WeightPack
from .registry import QUANTMETHODS
from lightllm.common.basemodel.triton_kernel.quantization.scaled_mm_per_token_kernel import fp8_scaled_mm_per_token
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
    lightllm_per_token_group_quant_fp8,
)
from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_gemm_kernel import w8a8_block_fp8_matmul
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops, cutlass_scaled_mm
from lightllm.utils.sgl_utils import HAS_SGL_KERNEL, sgl_ops

_HAS_SGL_FP8 = HAS_SGL_KERNEL and sgl_ops is not None and hasattr(sgl_ops, "fp8_scaled_mm")


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
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        raise NotImplementedError("Not implemented")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Not implemented")

    @property
    def method_name(self):
        return "w8a8-base"

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        raise NotImplementedError("Not implemented")


@QUANTMETHODS.register(["w8a8-vllm", "w8a8"], platform="cuda")
class w8a8QuantizationMethod(BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        weight = weight.float().cuda(self.device_id_)
        scale = weight.abs().max(dim=-1)[0] / 127
        weight = weight / scale.reshape(-1, 1)
        weight = torch.round(weight.clamp(min=-127, max=127)).to(dtype=torch.int8)
        output.weight.copy_(weight)
        output.weight_scale.copy_(scale)
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
        input_scale = None
        qweight = weight_pack.weight.t()
        weight_scale = weight_pack.weight_scale
        input_scale = None  # dynamic quantization for input tensor
        x_q, x_scale, x_zp = vllm_ops.scaled_int8_quant(input_tensor, scale=input_scale, azp=None, symmetric=True)
        m = input_tensor.shape[0]
        n = qweight.shape[1]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor((m, n), input_tensor.dtype, device=input_tensor.device)
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        cutlass_scaled_mm(out, x_q, qweight, x_scale, weight_scale, bias)
        return out

    @property
    def method_name(self):
        return "w8a8-vllm"

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims) if isinstance(out_dims, list) else out_dims
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.int8).cuda(device_id)
        weight_scale = torch.empty(expert_prefix + (out_dim,), dtype=torch.float32).cuda(device_id)
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale)
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
            weight_scale_out_dims=out_dims,
            weight_scale_split_dim=-1,
        )
        return mm_param, mm_param_list


@QUANTMETHODS.register(["fp8w8a8-vllm", "fp8w8a8"], platform="cuda")
class FP8w8a8QuantizationMethod(BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        qweight, weight_scale = scaled_fp8_quant(
            weight.cuda(self.device_id_), scale=None, use_per_token_if_dynamic=True
        )
        output.weight.copy_(qweight)
        output.weight_scale.copy_(weight_scale.view(-1))
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
        if LIGHTLLM_USE_TRITON_FP8_SCALED_MM:
            out = fp8_scaled_mm_per_token(x_q, qweight, x_scale, weight_scale, input_tensor.dtype, out)
            assert bias is None, "Bias addition is not supported in fp8w8a8 quantization method for now"
        else:
            cutlass_scaled_mm(out, x_q, qweight, x_scale, weight_scale, bias)
        return out

    @property
    def method_name(self):
        return "fp8w8a8-vllm"

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims) if isinstance(out_dims, list) else out_dims
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.float8_e4m3fn).cuda(device_id)
        weight_scale = torch.empty(expert_prefix + (out_dim,), dtype=torch.float32).cuda(device_id)
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale)

        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
            weight_scale_out_dims=out_dims,
            weight_scale_split_dim=-1,
        )
        return mm_param, mm_param_list


class FP8w8a8PerTensorQuantizationMethod(BaseQuantizationMethod):
    """Common weight and activation quantization for FP8 per-tensor GEMMs.

    A fused dense projection may be exposed as several ``WeightPack`` objects
    during checkpoint loading (for example, the gate and up projections). The
    packs are views of one fused runtime weight and therefore must be quantized
    together with one scale. Private ``_ptq_*`` attributes coordinate that
    deferred loading path until every source tensor is available.
    """

    def __init__(self):
        super().__init__()
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def _fp8_ptq_quant(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a complete weight tensor with one FP8 E4M3 scale."""
        weight = weight.float().cuda(self.device_id_)
        fp8_e4m3_max = torch.finfo(torch.float8_e4m3fn).max
        # Map the largest absolute weight to the largest finite E4M3 value. The
        # lower bound keeps an all-zero tensor from producing a zero divisor.
        scale = weight.abs().max() / fp8_e4m3_max
        scale = torch.clamp(scale, min=torch.finfo(torch.float32).tiny)
        qweight = (weight / scale).clamp(min=-fp8_e4m3_max, max=fp8_e4m3_max).to(dtype=torch.float8_e4m3fn)
        return qweight, scale.reshape(-1)

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        qweight, weight_scale = self._fp8_ptq_quant(weight)
        output.weight.copy_(qweight)
        # The logical scale is always a single value. Triton and Cutlass store it
        # as one element, whereas the SGL GEMM API requires an [N] scale buffer.
        # Tensor.copy_ broadcasts the scalar into that backend-specific storage
        # without changing the per-tensor quantization semantics.
        output.weight_scale.copy_(weight_scale)
        return

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack) -> None:
        # A pack without staging metadata represents a complete weight, so the
        # standard loader can quantize it immediately. Split packs instead defer
        # quantization through their shared parent pack.
        parent_pack = getattr(weight_pack, "_ptq_parent_pack", None)
        if parent_pack is None:
            super().load_weight(weight, weight_pack)
            return

        # Checkpoint loaders may populate sibling projections concurrently. The
        # parent lock protects the loaded flags and guarantees that exactly one
        # thread performs the final fused quantization and metadata cleanup.
        with parent_pack._ptq_staged_lock:
            if parent_pack._ptq_finalized:
                return
            staged_view = weight_pack._ptq_staged_view
            staged_view.copy_(weight.to(dtype=staged_view.dtype))
            weight_pack._ptq_staged_loaded = True
            if all(child_pack._ptq_staged_loaded for child_pack in parent_pack._ptq_child_packs):
                # Quantizing each child independently would produce different
                # scales for slices of the same fused GEMM weight. Quantize the
                # assembled buffer once so every output channel uses the scale
                # stored on the parent pack.
                self.quantize(parent_pack._ptq_staging_buffer, parent_pack)
                parent_pack.load_ok = [True, True, True]
                for child_pack in parent_pack._ptq_child_packs:
                    child_pack.load_ok = [True, True, True]
                parent_pack._ptq_finalized = True

                # The full-precision CPU staging allocation is needed only while
                # checkpoint shards are arriving. Drop all parent/child metadata
                # after quantization so it cannot retain memory for inference.
                child_packs = parent_pack._ptq_child_packs
                del parent_pack._ptq_staging_buffer
                del parent_pack._ptq_child_packs
                for child_pack in child_packs:
                    del child_pack._ptq_parent_pack
                    del child_pack._ptq_staged_view
                    del child_pack._ptq_staged_loaded
        return

    def _dynamic_quant_input(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        qweight = weight_pack.weight.t()
        weight_scale = weight_pack.weight_scale
        m = input_tensor.shape[0]
        k = input_tensor.shape[-1]
        n = qweight.shape[1]
        # Per-token activation quantization is the group-quantization case where
        # group_size equals the full K dimension. Call the LightLLM Triton kernel
        # directly because the generic wrapper may dispatch to the SGL kernel,
        # which rejects group_size == K.
        alloc_func = self.cache_manager.empty if use_custom_tensor_mananger else torch.empty
        x_q = alloc_func((m, k), dtype=torch.float8_e4m3fn, device=input_tensor.device)
        x_scale = alloc_func((m, 1), dtype=torch.float32, device=input_tensor.device)
        lightllm_per_token_group_quant_fp8(input_tensor, k, x_q, x_scale)
        assert bias is None, f"Bias addition is not supported in {self.method_name} for now"
        return qweight, weight_scale, x_q, x_scale, m, n

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        if num_experts != 1:
            raise NotImplementedError("FP8 per-tensor quantization currently supports dense weights only")
        if isinstance(out_dims, int):
            out_dims = [out_dims]
        out_dim = sum(out_dims)
        weight = torch.empty((out_dim, in_dim), dtype=torch.float8_e4m3fn, device=f"cuda:{device_id}")
        weight_scale = self._create_weight_scale(out_dim, device_id)
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale)
        weight_splits = torch.split(weight, out_dims, dim=-2)
        mm_param_list = [WeightPack(weight=split_weight, weight_scale=weight_scale) for split_weight in weight_splits]
        if len(mm_param_list) > 1:
            # Checkpoints store fused projections as separate tensors, but this
            # method needs one scale for the concatenated runtime weight. Give
            # each child a view into a full-precision CPU staging buffer; the last
            # child loaded triggers one quantization of the complete parent pack.
            staging_buffer = torch.empty((out_dim, in_dim), dtype=dtype, device="cpu")
            mm_param._ptq_staging_buffer = staging_buffer
            mm_param._ptq_child_packs = mm_param_list
            mm_param._ptq_staged_lock = threading.Lock()
            mm_param._ptq_finalized = False
            staged_views = torch.split(staging_buffer, out_dims, dim=-2)
            for child_pack, staged_view in zip(mm_param_list, staged_views):
                child_pack._ptq_parent_pack = mm_param
                child_pack._ptq_staged_view = staged_view
                child_pack._ptq_staged_loaded = False
        return mm_param, mm_param_list

    def _create_weight_scale(self, out_dim: int, device_id: int) -> torch.Tensor:
        # Triton and Cutlass consume the logical per-tensor scale directly, so
        # their weight packs reserve only one element regardless of output size.
        return torch.empty((1,), dtype=torch.float32, device=f"cuda:{device_id}")


@QUANTMETHODS.register("fp8w8a8-pt-vllm", platform="cuda")
class FP8w8a8PerTensorVllmQuantizationMethod(FP8w8a8PerTensorQuantizationMethod):
    def __init__(self):
        super().__init__()
        if not HAS_VLLM:
            raise RuntimeError("fp8w8a8-pt-vllm requires vllm with cutlass_scaled_mm support")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight, weight_scale, x_q, x_scale, _, _ = self._dynamic_quant_input(
            input_tensor, weight_pack, use_custom_tensor_mananger, bias
        )
        result = vllm_ops.cutlass_scaled_mm(
            x_q, qweight, x_scale, weight_scale.reshape(1, 1).to(torch.float32), input_tensor.dtype
        )
        return out.copy_(result) if out is not None else result

    @property
    def method_name(self):
        return "fp8w8a8-pt-vllm"


@QUANTMETHODS.register("fp8w8a8-pt-sgl", platform="cuda")
class FP8w8a8PerTensorSglQuantizationMethod(FP8w8a8PerTensorQuantizationMethod):
    def __init__(self):
        super().__init__()
        if not _HAS_SGL_FP8:
            raise RuntimeError("fp8w8a8-pt-sgl requires sgl_kernel.fp8_scaled_mm support")

    def _create_weight_scale(self, out_dim: int, device_id: int) -> torch.Tensor:
        # SGL's scaled GEMM accepts only per-output-channel weight scales. Reserve
        # a contiguous [N] buffer; quantize() fills every entry with the same
        # scalar so this remains mathematically equivalent to per-tensor scaling.
        return torch.empty((out_dim,), dtype=torch.float32, device=f"cuda:{device_id}")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight, weight_scale, x_q, x_scale, _, _ = self._dynamic_quant_input(
            input_tensor, weight_pack, use_custom_tensor_mananger, bias
        )
        result = sgl_ops.fp8_scaled_mm(x_q, qweight, x_scale, weight_scale, input_tensor.dtype)
        return out.copy_(result) if out is not None else result

    @property
    def method_name(self):
        return "fp8w8a8-pt-sgl"


@QUANTMETHODS.register(["fp8w8a8-pt-triton", "fp8w8a8-pt"], platform="cuda")
class FP8w8a8PerTensorTritonQuantizationMethod(FP8w8a8PerTensorQuantizationMethod):
    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight, weight_scale, x_q, x_scale, m, n = self._dynamic_quant_input(
            input_tensor, weight_pack, use_custom_tensor_mananger, bias
        )
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor((m, n), input_tensor.dtype, device=input_tensor.device)
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        return fp8_scaled_mm_per_token(
            x_q,
            qweight,
            x_scale,
            weight_scale,
            input_tensor.dtype,
            out,
        )

    @property
    def method_name(self):
        return "fp8w8a8-pt-triton"


class FP8w8a8B128QuantizationMethod(BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.block_size = 128
        self.weight_scale_suffix = "weight_scale_inv"
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_quant_kernel import weight_quant

        device = output.weight.device
        weight, scale = weight_quant(weight.cuda(device), self.block_size)
        output.weight.copy_(weight)
        output.weight_scale.copy_(scale)
        return

    def _dynamic_quant_input(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qweight = weight_pack.weight.t()
        weight_scale = weight_pack.weight_scale.t()
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
        return qinput_tensor, qweight, input_scale, weight_scale, out

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims) if isinstance(out_dims, list) else out_dims
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.float8_e4m3fn).cuda(device_id)
        weight_scale = torch.empty(
            expert_prefix + (out_dim // self.block_size, in_dim // self.block_size), dtype=torch.float32
        ).cuda(device_id)
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale)
        weight_scale_out_dims = [_out_dim // self.block_size for _out_dim in out_dims]
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
            weight_scale_out_dims=weight_scale_out_dims,
            weight_scale_split_dim=-2,
        )
        return mm_param, mm_param_list


@QUANTMETHODS.register("fp8w8a8-b128-vllm", platform="cuda")
class FP8w8a8B128VllmQuantizationMethod(FP8w8a8B128QuantizationMethod):
    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qinput_tensor, qweight, input_scale, weight_scale, out = self._dynamic_quant_input(
            input_tensor, weight_pack, out, use_custom_tensor_mananger
        )
        if qweight.shape[1] % self.block_size != 0:
            raise ValueError(
                "fp8w8a8-b128-vllm requires the output dimension to be divisible by 128; "
                "use fp8w8a8-b128-triton instead"
            )
        input_scale = input_scale.t().contiguous().t()
        cutlass_scaled_mm(out, qinput_tensor, qweight, input_scale, weight_scale, bias)
        return out

    @property
    def method_name(self):
        return "fp8w8a8-b128-vllm"


@QUANTMETHODS.register(["fp8w8a8-b128-triton", "fp8w8a8-b128"], platform="cuda")
class FP8w8a8B128TritonQuantizationMethod(FP8w8a8B128QuantizationMethod):
    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qinput_tensor, qweight, input_scale, weight_scale, out = self._dynamic_quant_input(
            input_tensor, weight_pack, out, use_custom_tensor_mananger
        )
        w8a8_block_fp8_matmul(
            qinput_tensor,
            qweight,
            input_scale,
            weight_scale,
            out,
            (self.block_size, self.block_size),
            dtype=input_tensor.dtype,
        )
        assert bias is None, f"Bias addition is not supported in {self.method_name} for now"
        return out

    @property
    def method_name(self):
        return "fp8w8a8-b128-triton"
