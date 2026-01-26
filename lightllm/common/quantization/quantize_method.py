import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from lightllm.utils.dist_utils import get_current_device_id
from typing import Optional, Tuple


@dataclass
class WeightPack:
    weight: Optional[torch.Tensor] = None
    weight_scale: Optional[torch.Tensor] = None
    weight_zero_point: Optional[torch.Tensor] = None
    fused_dim: Optional[int] = 0

    def get_expert(self, expert_idx: int):
        assert self.weight.ndim == 3, f"weight must be a 3D tensor, but got {self.weight.ndim}"
        weight = self.weight[expert_idx]
        weight_scale = self.weight_scale[expert_idx] if self.weight_scale is not None else None
        weight_zero_point = self.weight_zero_point[expert_idx] if self.weight_zero_point is not None else None
        return WeightPack(
            weight=weight, weight_scale=weight_scale, weight_zero_point=weight_zero_point, fused_dim=self.fused_dim
        )

    def create_cpu_buffer(self, weight_num: int):
        self.weight_cpu_buffer = [None] * weight_num
        self.weight_scale_cpu_buffer = [None] * weight_num
        self.weight_zero_point_cpu_buffer = [None] * weight_num
        self.load_ok = [False, self.weight_scale is None, self.weight_zero_point is None]
        return

    def get_fused_weight_part(self, weight_type) -> Optional[torch.Tensor]:
        buffer_map = {
            "weight": ("weight_cpu_buffer", 0),
            "weight_scale": ("weight_scale_cpu_buffer", 1),
            "weight_zero_point": ("weight_zero_point_cpu_buffer", 2),
        }
        buffer_name, index = buffer_map.get(weight_type)
        if buffer_name is None:
            raise ValueError(f"unknown weight type: {weight_type}")
        cpu_buffer = getattr(self, buffer_name)
        if None not in cpu_buffer:
            try:
                fused = torch.cat(cpu_buffer, dim=self.fused_dim)
            except Exception as e:
                print(len(cpu_buffer), self.fused_dim)
                for buff in cpu_buffer:
                    print(buff.shape)
                raise e
            setattr(self, buffer_name, [None] * len(cpu_buffer))
            self.load_ok[index] = True
            return fused
        return None


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
        self, out_dim: int, in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> WeightPack:
        pass

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
