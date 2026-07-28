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


@QUANTMETHODS.register(["vllm-w8a8", "w8a8"], platform="cuda")
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
        return "vllm-w8a8"

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


@QUANTMETHODS.register(["vllm-fp8w8a8", "fp8w8a8"], platform="cuda")
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
        return "vllm-fp8w8a8"

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
    def __init__(self):
        super().__init__()
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def _fp8_per_tensor_quant(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = weight.float().cuda(self.device_id_)
        fp8_e4m3_max = torch.finfo(torch.float8_e4m3fn).max
        if weight.ndim == 3:
            scale = weight.abs().amax(dim=(-1, -2)) / fp8_e4m3_max
        else:
            scale = weight.abs().max() / fp8_e4m3_max
        scale = torch.clamp(scale, min=torch.finfo(torch.float32).tiny)
        scale_view = scale.reshape(-1, 1, 1) if weight.ndim == 3 else scale
        qweight = (weight / scale_view).clamp(min=-fp8_e4m3_max, max=fp8_e4m3_max).to(dtype=torch.float8_e4m3fn)
        return qweight, scale.reshape(-1)

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        if weight.ndim == 3 and output.weight_scale is not None and output.weight_scale.numel() == weight.shape[0]:
            for expert_idx in range(weight.shape[0]):
                qweight, weight_scale = self._fp8_per_tensor_quant(weight[expert_idx])
                output.weight[expert_idx].copy_(qweight)
                output.weight_scale[expert_idx : expert_idx + 1].copy_(weight_scale.reshape(-1))
            return

        qweight, weight_scale = self._fp8_per_tensor_quant(weight)
        output.weight.copy_(qweight)
        output.weight_scale.copy_(weight_scale)
        return

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack) -> None:
        parent_pack = weight_pack.per_tensor_parent_pack
        # Single-part weights do not need CPU staging; the base loader will call this class's quantize().
        if parent_pack is None:
            super().load_weight(weight, weight_pack)
            return

        with parent_pack._per_tensor_staged_lock:
            if parent_pack._per_tensor_finalized:
                return
            child_idx = weight_pack.per_tensor_child_index
            expert_idx = weight_pack.per_tensor_expert_index
            # Copy this shard into the full CPU weight before doing per-tensor quantization.
            staged_weight = parent_pack._per_tensor_staged_weight
            if staged_weight.ndim == 3:
                staged_weight = staged_weight[expert_idx]
            staged_weight = torch.split(staged_weight, parent_pack._per_tensor_out_dims, dim=-2)[child_idx]
            staged_weight.copy_(weight.to(dtype=staged_weight.dtype))
            parent_pack._per_tensor_staged_loaded[expert_idx][child_idx] = True
            self._try_finalize_staged_weight(parent_pack)
        return

    def _try_finalize_staged_weight(self, parent_pack: WeightPack) -> bool:
        if parent_pack._per_tensor_finalized:
            return True
        staged_loaded = parent_pack._per_tensor_staged_loaded
        all_loaded = all(all(expert_loaded) for expert_loaded in staged_loaded)
        if not all_loaded:
            return False

        # Quantize once after all split shards are present, so the scale is computed on the full tensor.
        self.quantize(parent_pack._per_tensor_staged_weight, parent_pack)
        parent_pack.load_ok = [True, True, True]
        parent_pack._per_tensor_finalized = True
        self._clear_staged_weight_state(parent_pack)
        return True

    def _clear_staged_weight_state(self, parent_pack: WeightPack) -> None:
        child_packs = parent_pack._per_tensor_child_packs
        # Drop the temporary CPU tensor and reset child packs to normal GPU views after finalization.
        del parent_pack._per_tensor_staged_weight
        del parent_pack._per_tensor_staged_loaded
        del parent_pack._per_tensor_child_packs
        del parent_pack._per_tensor_out_dims
        for child_pack in child_packs:
            child_pack.load_ok = [True, True, True]
            child_pack.per_tensor_parent_pack = None
            child_pack.per_tensor_child_index = None
            child_pack.per_tensor_expert_index = 0
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
        # direct triton call: the per_token_group_quant_fp8 wrapper picks sgl, which rejects group_size == k
        alloc_func = self.cache_manager.empty if use_custom_tensor_mananger else torch.empty
        x_q = alloc_func((m, k), dtype=torch.float8_e4m3fn, device=input_tensor.device)
        x_scale = alloc_func((m, 1), dtype=torch.float32, device=input_tensor.device)
        lightllm_per_token_group_quant_fp8(input_tensor, k, x_q, x_scale)
        assert bias is None, f"Bias addition is not supported in {self.method_name} for now"
        return qweight, weight_scale, x_q, x_scale, m, n

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        if isinstance(out_dims, int):
            out_dims = [out_dims]
        out_dim = sum(out_dims)
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.float8_e4m3fn).cuda(device_id)

        weight_scale = torch.empty(expert_prefix or (1,), dtype=torch.float32, device=f"cuda:{device_id}")
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale)
        weight_splits = torch.split(weight, out_dims, dim=-2)
        mm_param_list = [WeightPack(weight=weight, weight_scale=weight_scale) for weight in weight_splits]

        if len(out_dims) > 1:
            # Split weights share one final GPU tensor, but load into CPU first to get one per-tensor scale.
            staged_weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=dtype, device="cpu")
            mm_param._per_tensor_staged_weight = staged_weight
            mm_param._per_tensor_staged_loaded = [[False] * len(mm_param_list) for _ in range(num_experts)]
            mm_param._per_tensor_child_packs = mm_param_list
            mm_param._per_tensor_out_dims = out_dims
            mm_param._per_tensor_finalized = False
            mm_param._per_tensor_staged_lock = threading.Lock()
            for idx, child_pack in enumerate(mm_param_list):
                child_pack.per_tensor_parent_pack = mm_param
                child_pack.per_tensor_child_index = idx
        return mm_param, mm_param_list


@QUANTMETHODS.register("fp8w8a8-pt-cutlass", platform="cuda")
class FP8w8a8PerTensorCutlassQuantizationMethod(FP8w8a8PerTensorQuantizationMethod):
    def __init__(self):
        super().__init__()
        if not HAS_VLLM:
            raise RuntimeError("fp8w8a8-pt-cutlass requires vllm with cutlass_scaled_mm support")

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
        return "fp8w8a8-pt-cutlass"


@QUANTMETHODS.register("fp8w8a8-pt-sgl", platform="cuda")
class FP8w8a8PerTensorSglQuantizationMethod(FP8w8a8PerTensorQuantizationMethod):
    def __init__(self):
        super().__init__()
        if not _HAS_SGL_FP8:
            raise RuntimeError("fp8w8a8-pt-sgl requires sgl_kernel.fp8_scaled_mm support")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight, weight_scale, x_q, x_scale, _, n = self._dynamic_quant_input(
            input_tensor, weight_pack, use_custom_tensor_mananger, bias
        )
        # sgl needs a per-channel weight scale [N]; expand the per-tensor scalar once, cache it.
        b_scale = getattr(weight_pack, "_fp8_sgl_bscale", None)
        if b_scale is None or b_scale.numel() != n:
            b_scale = weight_scale.reshape(1).to(torch.float32).expand(n).contiguous()
            weight_pack._fp8_sgl_bscale = b_scale
        result = sgl_ops.fp8_scaled_mm(x_q, qweight, x_scale, b_scale, input_tensor.dtype)
        return out.copy_(result) if out is not None else result

    @property
    def method_name(self):
        return "fp8w8a8-pt-sgl"


@QUANTMETHODS.register(["fp8w8a8-pt", "fp8w8a8-pt-triton"], platform="cuda")
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


@QUANTMETHODS.register(["vllm-fp8w8a8-b128", "fp8w8a8-b128", "fp8w8a8-b128-cutlass"], platform="cuda")
class FP8w8a8B128CutlassQuantizationMethod(FP8w8a8B128QuantizationMethod):
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
                "fp8w8a8-b128-cutlass requires the output dimension to be divisible by 128; "
                "use fp8w8a8-b128-triton instead"
            )
        input_scale = input_scale.t().contiguous().t()
        cutlass_scaled_mm(out, qinput_tensor, qweight, input_scale, weight_scale, bias)
        return out

    @property
    def method_name(self):
        return "fp8w8a8-b128-cutlass"


@QUANTMETHODS.register("fp8w8a8-b128-triton", platform="cuda")
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
