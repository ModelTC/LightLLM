import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from lightllm.utils.dist_utils import get_current_device_id
from typing import Optional, List, Tuple


@dataclass
class WeightPack:
    weight: Optional[torch.Tensor] = None
    weight_scale: Optional[torch.Tensor] = None
    weight_zero_point: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.load_ok = [False, self.weight_scale is None, self.weight_zero_point is None]

    def get_expert(self, expert_idx: int):
        assert self.weight.ndim == 3, f"weight must be a 3D tensor, but got {self.weight.ndim}"
        weight = self.weight[expert_idx]
        weight_scale = self.weight_scale[expert_idx] if self.weight_scale is not None else None
        weight_zero_point = self.weight_zero_point[expert_idx] if self.weight_zero_point is not None else None
        return WeightPack(weight=weight, weight_scale=weight_scale, weight_zero_point=weight_zero_point)


class QuantizationMethod(ABC):
    def __init__(self):
        super().__init__()
        self.device_id_ = get_current_device_id()
        self.weight_suffix = "weight"
        self.weight_scale_suffix = None
        self.weight_zero_point_suffix = None
        self.act_scale_suffix = None
        self.has_weight_scale: bool = None
        self.has_weight_zero_point: bool = None
        self.group_size: int = -1  # -1表示不分组即per-channel量化，其他表示分组大小
        self.pack_factor: int = 1
        self.block_size: int = 1

        # 一些量化模式需要用到的额外量化参数，如awq量化
        self.hf_quantization_config = None

    @abstractmethod
    def quantize(
        self,
        weight: torch.Tensor,
        output: WeightPack,
    ) -> None:
        pass

    @abstractmethod
    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "WeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def method_name(self):
        pass

    def create_weight(
        self, out_dims: List[int], in_dim: int, dtype: torch.dtype, device_id: int
    ) -> Tuple[WeightPack, List[WeightPack]]:
        return self._create_weight(
            out_dims=out_dims,
            in_dim=in_dim,
            dtype=dtype,
            device_id=device_id,
        )

    def create_moe_weight(
        self, out_dims: List[int], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int
    ) -> Tuple[WeightPack, List[WeightPack]]:
        return self._create_weight(
            out_dims=out_dims,
            in_dim=in_dim,
            dtype=dtype,
            device_id=device_id,
            num_experts=num_experts,
        )

    def weight_need_quanted(self, weight: torch.Tensor) -> bool:
        if weight is None:
            return False
        # 判断一个 weight 是否需要进行量化操作。
        return weight.dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack) -> None:
        if weight is None:
            return
        weight_pack.weight.copy_(weight)
        return

    def load_weight_scale(self, weight_scale: torch.Tensor, weight_pack: WeightPack) -> None:
        if weight_scale is None:
            return
        weight_pack.weight_scale.copy_(weight_scale)
        return

    def load_weight_zero_point(self, weight_zero_point: torch.Tensor, weight_pack: WeightPack) -> None:
        if weight_zero_point is None:
            return
        weight_pack.weight_zero_point.copy_(weight_zero_point)
        return

    def _create_weight(
        self, out_dims: List[int], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        pass

    def _split_weight_pack(
        self,
        weight_pack: WeightPack,
        out_dims: List[int],
        weight_split_dim: Optional[int],
        weight_scale_split_dim: Optional[int] = None,
    ) -> List[WeightPack]:
        # only support per-channel or block-wise quantization for now.
        mm_param_list: List[WeightPack] = []
        packed_out_dims = [dim // self.pack_factor for dim in out_dims]
        scale_out_dims = [dim // self.block_size for dim in out_dims]
        weight = torch.split(weight_pack.weight, packed_out_dims, dim=weight_split_dim)
        weight_scale = (
            [None] * len(out_dims)
            if weight_pack.weight_scale is None
            else (torch.split(weight_pack.weight_scale, scale_out_dims, dim=weight_scale_split_dim))
        )
        # the ndim of weight_zero_point is the same as weight_scale.
        weight_zero_point = (
            [None] * len(out_dims)
            if weight_pack.weight_zero_point is None
            else (torch.split(weight_pack.weight_zero_point, packed_out_dims, dim=weight_scale_split_dim))
        )
        for weight, weight_scale, weight_zero_point in zip(weight, weight_scale, weight_zero_point):
            mm_param_list.append(
                WeightPack(weight=weight, weight_scale=weight_scale, weight_zero_point=weight_zero_point)
            )
        return mm_param_list
