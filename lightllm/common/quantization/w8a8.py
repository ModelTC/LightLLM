import os
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

# fp8 GEMM backend: LIGHTLLM_FP8_GEMM = auto | cutlass | sgl | triton (auto: cutlass > sgl > triton).
_HAS_SGL_FP8 = HAS_SGL_KERNEL and sgl_ops is not None and hasattr(sgl_ops, "fp8_scaled_mm")
_FP8_GEMM_BACKEND = os.getenv("LIGHTLLM_FP8_GEMM", "auto").lower()
if _FP8_GEMM_BACKEND in ("cutlass", "sgl", "triton"):
    _FP8_BACKEND = _FP8_GEMM_BACKEND
else:  # auto: Cutlass > sgl_kernel > triton, by availability
    _FP8_BACKEND = "cutlass" if HAS_VLLM else ("sgl" if _HAS_SGL_FP8 else "triton")


if HAS_VLLM:
    scaled_fp8_quant = vllm_ops.scaled_fp8_quant

LIGHTLLM_USE_TRITON_FP8_SCALED_MM = os.getenv("LIGHTLLM_USE_TRITON_FP8_SCALED_MM", "False").upper() in [
    "ON",
    "TRUE",
    "1",
]

FP8_E4M3_MAX = 448.0


def _fp8_per_tensor_quant(weight: torch.Tensor, device_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = weight.float().cuda(device_id)
    if weight.ndim == 3:
        scale = weight.abs().amax(dim=(-1, -2)) / FP8_E4M3_MAX
    else:
        scale = weight.abs().max() / FP8_E4M3_MAX
    scale = torch.clamp(scale, min=torch.finfo(torch.float32).tiny)
    scale_view = scale.reshape(-1, 1, 1) if weight.ndim == 3 else scale
    qweight = _fp8_quant_with_scale(weight, scale_view)
    return qweight, scale.reshape(-1)


def _fp8_quant_with_scale(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (weight / scale).clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX).to(dtype=torch.float8_e4m3fn)


def _copy_scale_with_broadcast(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.numel() == src.numel():
        dst.copy_(src.reshape_as(dst))
    elif src.numel() == 1:
        if dst.dim() == 0:
            dst.copy_(src.reshape(()))
        else:
            dst.copy_(src.reshape(1).expand_as(dst))
    else:
        raise ValueError(f"can not copy scale with shape {tuple(src.shape)} to {tuple(dst.shape)}")


class BaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        # assert HAS_VLLM, "vllm are not installed, you can't use quant api of them."
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


@QUANTMETHODS.register(
    ["triton-fp8w8a8-pertensor", "fp8w8a8-pertensor", "triton-fp8w8a8-pt", "fp8w8a8-pt"],
    platform="cuda",
)
class TritonFP8w8a8PerTensorQuantizationMethod(BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        if weight.ndim == 3 and output.weight_scale is not None and output.weight_scale.numel() == weight.shape[0]:
            for expert_idx in range(weight.shape[0]):
                qweight, weight_scale = _fp8_per_tensor_quant(weight[expert_idx], self.device_id_)
                output.weight[expert_idx].copy_(qweight)
                output.weight_scale[expert_idx].copy_(weight_scale.reshape(()))
            return

        qweight, weight_scale = _fp8_per_tensor_quant(weight, self.device_id_)
        output.weight.copy_(qweight)
        _copy_scale_with_broadcast(output.weight_scale, weight_scale)
        return

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack) -> None:
        parent_pack = getattr(weight_pack, "_fp8_pt_parent_pack", None)
        if parent_pack is None:
            super().load_weight(weight, weight_pack)
            return

        staged_weight = weight_pack._fp8_pt_staged_weight
        staged_weight.copy_(weight.to(device=staged_weight.device, dtype=staged_weight.dtype, non_blocking=True))
        loaded_index = weight_pack._fp8_pt_child_index
        if hasattr(weight_pack, "_fp8_pt_expert_index"):
            loaded_index = (weight_pack._fp8_pt_expert_index, loaded_index)
        parent_pack._fp8_pt_staged_loaded[loaded_index] = True
        self._try_finalize_deferred_weight(parent_pack)
        return

    def _try_finalize_deferred_weight(self, parent_pack: WeightPack) -> bool:
        if getattr(parent_pack, "_fp8_pt_finalized", False):
            return True
        staged_loaded = parent_pack._fp8_pt_staged_loaded
        if isinstance(staged_loaded, torch.Tensor):
            all_loaded = bool(staged_loaded.all().item())
        else:
            all_loaded = all(staged_loaded)
        if not all_loaded:
            return False

        self.quantize(parent_pack._fp8_pt_staged_weight, parent_pack)
        parent_pack.load_ok = [True, True, True]
        parent_pack._fp8_pt_finalized = True
        parent_pack._fp8_pt_staged_weight = None
        for child_pack in parent_pack._fp8_pt_child_packs:
            child_pack.load_ok = [True, True, True]
            child_pack._fp8_pt_staged_weight = None
            for expert_child_pack in getattr(child_pack, "_fp8_pt_expert_child_packs", []):
                expert_child_pack.load_ok = [True, True, True]
                expert_child_pack._fp8_pt_staged_weight = None
        return True

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
        m = input_tensor.shape[0]
        k = input_tensor.shape[-1]
        n = qweight.shape[1]
        # direct triton call: the per_token_group_quant_fp8 wrapper picks sgl, which rejects group_size == k
        alloc_func = self.cache_manager.empty if use_custom_tensor_mananger else torch.empty
        x_q = alloc_func((m, k), dtype=torch.float8_e4m3fn, device=input_tensor.device)
        x_scale = alloc_func((m, 1), dtype=torch.float32, device=input_tensor.device)
        lightllm_per_token_group_quant_fp8(input_tensor, k, x_q, x_scale)
        assert bias is None, "Bias addition is not supported in triton-fp8w8a8-pertensor for now"
        if _FP8_BACKEND == "cutlass":
            cu_out = vllm_ops.cutlass_scaled_mm(
                x_q, qweight, x_scale, weight_scale.reshape(1, 1).to(torch.float32), input_tensor.dtype
            )
            return out.copy_(cu_out) if out is not None else cu_out
        if _FP8_BACKEND == "sgl":
            # sgl needs a per-channel weight scale [N]; expand the per-tensor scalar once, cache it.
            b_scale = getattr(weight_pack, "_fp8_sgl_bscale", None)
            if b_scale is None or b_scale.numel() != n:
                b_scale = weight_scale.reshape(1).to(torch.float32).expand(n).contiguous()
                weight_pack._fp8_sgl_bscale = b_scale
            sgl_out = sgl_ops.fp8_scaled_mm(x_q, qweight, x_scale, b_scale, input_tensor.dtype)
            return out.copy_(sgl_out) if out is not None else sgl_out
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
        return "triton-fp8w8a8-pertensor"

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
            staged_weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=dtype, device="cpu")
            staged_splits = torch.split(staged_weight, out_dims, dim=-2)
            mm_param._fp8_pt_staged_weight = staged_weight
            if num_experts > 1:
                mm_param._fp8_pt_staged_loaded = torch.zeros(
                    (num_experts, len(mm_param_list)), dtype=torch.bool, device="cpu"
                )
            else:
                mm_param._fp8_pt_staged_loaded = [False] * len(mm_param_list)
            mm_param._fp8_pt_child_packs = mm_param_list
            mm_param._fp8_pt_finalized = False
            for idx, (child_pack, staged_split) in enumerate(zip(mm_param_list, staged_splits)):
                child_pack._fp8_pt_parent_pack = mm_param
                child_pack._fp8_pt_child_index = idx
                child_pack._fp8_pt_staged_weight = staged_split
                if num_experts > 1:
                    child_pack._fp8_pt_expert_child_packs = []
                    child_pack._fp8_pt_get_expert = child_pack.get_expert

                    def _get_deferred_expert(expert_idx, _child_pack=child_pack):
                        expert_child_pack = _child_pack._fp8_pt_get_expert(expert_idx)
                        expert_child_pack._fp8_pt_parent_pack = _child_pack._fp8_pt_parent_pack
                        expert_child_pack._fp8_pt_child_index = _child_pack._fp8_pt_child_index
                        expert_child_pack._fp8_pt_expert_index = expert_idx
                        expert_child_pack._fp8_pt_staged_weight = _child_pack._fp8_pt_staged_weight[expert_idx]
                        _child_pack._fp8_pt_expert_child_packs.append(expert_child_pack)
                        return expert_child_pack

                    child_pack.get_expert = _get_deferred_expert
        return mm_param, mm_param_list


@QUANTMETHODS.register(["vllm-fp8w8a8-b128", "fp8w8a8-b128"], platform="cuda")
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
            assert bias is None, "Bias addition is not supported in fp8w8a8-b128 quantization method for now"
        else:
            input_scale = input_scale.t().contiguous().t()
            cutlass_scaled_mm(out, qinput_tensor, qweight, input_scale, weight_scale, bias)
        return out

    @property
    def method_name(self):
        return "vllm-fp8w8a8-b128"

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
