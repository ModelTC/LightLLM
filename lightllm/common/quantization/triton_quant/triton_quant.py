import os
import torch
import torch.nn.functional as F
from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.common.quantization.registry import QUANTMETHODS
from .fp8.fp8w8a8_block_gemm_kernel import w8a8_block_fp8_matmul
from .fp8.fp8act_quant_kernel import per_token_group_quant_fp8
from typing import TYPE_CHECKING, Optional

from lightllm.common.quantization.quantize_method import WeightPack


class TritonBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor, output: WeightPack, offset: int = 0) -> WeightPack:
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


@QUANTMETHODS.register(["triton-fp8w8a8-block128"])
class TritonFP8w8a8QuantizationMethod(TritonBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.is_moe = False
        self.block_size = 128
        self.weight_suffix = None
        self.weight_zero_point_suffix = None
        self.weight_scale_suffix = "weight_scale_inv"
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    def quantize(self, weight: torch.Tensor, output: WeightPack, offset: int = 0) -> None:
        # TODO block-wise quant kernel
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
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        input_scale = None
        m, k = input_tensor.shape
        n = qweight.shape[1]
        alloc_func = torch.empty if not use_custom_tensor_mananger else self.cache_manager.empty
        if input_scale is None:
            input_tensor_q, input_scale = per_token_group_quant_fp8(
                input_tensor, self.block_size, dtype=qweight.dtype, alloc_func=alloc_func
            )
        else:
            # TODO
            raise "statci input scale is not supported by triton fp8 block gemm kernel."
        m = input_tensor.shape[0]
        n = qweight.shape[1]
        if out is None:
            out = alloc_func((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        w8a8_block_fp8_matmul(
            input_tensor_q,
            qweight,
            input_scale,
            weight_scale,
            out,
            (self.block_size, self.block_size),
            dtype=input_tensor.dtype,
        )
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
        weight_pack.weight_zero_point[
            start_idx // self.block_size : start_idx + weight_zero_point.shape[0] // self.block_size
        ].copy_(weight_zero_point)
        return
