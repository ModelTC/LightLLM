import torch
from typing import Optional, List, Union, Tuple
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS

from .quantize_method import WeightPack


class _BaseQuantizationMethod(QuantizationMethod):
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
        return "w8a8gx-base"

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        raise NotImplementedError("Not implemented")


@QUANTMETHODS.register(["triton-fp8w8a8g128", "fp8w8a8g128"], platform="cuda")
class FP8w8a8g128QuantizationMethod(_BaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.has_weight_scale = True
        self.has_weight_zero_point = False
        self.act_quant_group_size = 128

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        # per channel quantization for weight, per token group quantization for input activation
        from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_perchannel_quant_kernel import weight_quant

        qweight, weight_scale = weight_quant(weight.cuda(self.device_id_))
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
        from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
            lightllm_per_token_group_quant_fp8,
        )

        assert (
            input_tensor.shape[-1] % self.act_quant_group_size == 0
        ), "Input feature dimension must be divisible by act_quant_group_size"
        if use_custom_tensor_mananger:
            x_q = self.cache_manager.alloc_tensor(input_tensor.shape, dtype=qweight.dtype, device=input_tensor.device)
            x_scale = self.cache_manager.alloc_tensor(
                input_tensor.shape[:-1] + (input_tensor.shape[-1] // self.act_quant_group_size,),
                dtype=torch.float32,
                device=input_tensor.device,
            )
        else:
            x_q = torch.empty(input_tensor.shape, dtype=qweight.dtype, device=input_tensor.device)
            x_scale = torch.empty(
                input_tensor.shape[:-1] + (input_tensor.shape[-1] // self.act_quant_group_size,),
                dtype=torch.float32,
                device=input_tensor.device,
            )

        lightllm_per_token_group_quant_fp8(x=input_tensor, group_size=self.act_quant_group_size, x_q=x_q, x_s=x_scale)

        m = input_tensor.shape[0]
        n = qweight.shape[1]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor((m, n), input_tensor.dtype, device=input_tensor.device)
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)

        from lightllm.common.basemodel.triton_kernel.quantization.scaled_mm_per_token_group_quant_kernel import (
            scaled_mm_act_per_group_w_perchannel,
        )

        assert bias is None, "Bias addition is not supported in fp8w8a8g128 quantization method for now"
        out = scaled_mm_act_per_group_w_perchannel(
            A=x_q,
            B=qweight,
            Ascale=x_scale,
            Bscale=weight_scale,
            act_quant_group_size=self.act_quant_group_size,
            out=out,
        )
        return out

    @property
    def method_name(self):
        return f"triton-fp8w8a8g{self.act_quant_group_size}"

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


@QUANTMETHODS.register(["triton-fp8w8a8g64", "fp8w8a8g64"], platform="cuda")
class FP8w8a8g64QuantizationMethod(FP8w8a8g128QuantizationMethod):
    def __init__(self):
        super().__init__()
        self.act_quant_group_size = 64
