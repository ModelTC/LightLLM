import torch
from typing import List, Optional, Tuple

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS


@QUANTMETHODS.register("mxfp4", platform="cuda")
class MXFP4QuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        self.weight_suffix = "weight_packed"
        self.weight_scale_suffix = "weight_scale"
        self.has_weight_scale = True
        self.group_size = 32
        self.pack_factor = 2

    @property
    def method_name(self):
        return "mxfp4"

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        raise NotImplementedError("MXFP4 online quantization is not supported")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("MXFP4 is currently supported only for fused MoE weights")

    def _create_weight(
        self,
        out_dims: List[int],
        in_dim: int,
        dtype: torch.dtype,
        device_id: int,
        num_experts: int = 1,
    ) -> Tuple[WeightPack, List[WeightPack]]:
        del dtype
        if in_dim % self.group_size != 0:
            raise ValueError(f"MXFP4 input dimension {in_dim} must be divisible by {self.group_size}")

        out_dim = sum(out_dims)
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        device = f"cuda:{device_id}"
        weight = torch.empty(
            expert_prefix + (out_dim, in_dim // self.pack_factor),
            dtype=torch.uint8,
            device=device,
        )
        weight_scale = torch.empty(
            expert_prefix + (out_dim, in_dim // self.group_size),
            dtype=torch.uint8,
            device=device,
        )
        weight_pack = WeightPack(weight=weight, weight_scale=weight_scale)
        split_packs = self._split_weight_pack(
            weight_pack,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
            weight_scale_out_dims=out_dims,
            weight_scale_split_dim=-2,
        )
        return weight_pack, split_packs

    def _check_weight_need_quanted(self, weight: torch.Tensor) -> bool:
        return False
