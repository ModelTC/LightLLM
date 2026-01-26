import torch
from typing import Optional

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS


@QUANTMETHODS.register("none", platform="musa")
@QUANTMETHODS.register("none", platform="cuda")
class NoQuantization(QuantizationMethod):
    """No quantization - uses full precision weights."""

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        weight = weight_pack.weight.t()
        if out is None:
            shape = (input_tensor.shape[0], weight.shape[1])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if bias is None:
            return torch.mm(input_tensor, weight, out=out)
        return torch.addmm(bias, input_tensor, weight, out=out)

    def create_weight(
        self, out_dim: int, in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> WeightPack:
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=dtype).cuda(device_id)
        return WeightPack(weight=weight, weight_scale=None, weight_zero_point=None)

    def weight_need_quanted(self, weight: torch.Tensor) -> bool:
        return False

    def quantize(self, weight: torch.Tensor, output: WeightPack, offset: int = 0) -> None:
        return

    @property
    def method_name(self):
        return "none"

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack, start_idx: int = 0) -> None:
        if weight is None:
            return
        weight_pack.weight.copy_(weight)
        return
