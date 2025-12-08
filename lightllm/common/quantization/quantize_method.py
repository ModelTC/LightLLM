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

        # 一些量化模式需要用到的额外量化参数，如awq量化
        self.hf_quantization_config = None

    @abstractmethod
    def quantize(
        self,
        weight: torch.Tensor,
        output: WeightPack,
        offset: int = 0,
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
        # 判断一个 weight 是否需要进行量化操作。
        return weight.dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack, start_idx: int) -> None:
        raise NotImplementedError(
            f"quantization method {self.method_name} is not supported to load offline quantized weight"
        )

    def load_weight_scale(self, weight_scale: torch.Tensor, weight_pack: WeightPack, start_idx: int) -> None:
        raise NotImplementedError(
            f"quantization method {self.method_name} is not supported to load offline quantized weight scale"
        )

    def load_weight_zero_point(self, weight_zero_point: torch.Tensor, weight_pack: WeightPack, start_idx: int) -> None:
        raise NotImplementedError(
            f"quantization method {self.method_name} is not supported to load offline quantized weight zero point"
        )
