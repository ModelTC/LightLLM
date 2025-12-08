import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from lightllm.utils.dist_utils import get_current_device_id
from typing import Optional, Tuple, TYPE_CHECKING, List


@dataclass
class WeightPack:
    weight: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None


@dataclass
class TensorMeta:
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class QuantizedMetadata:
    weight: Optional[TensorMeta] = None
    scale: Optional[TensorMeta] = None
    zero_point: Optional[TensorMeta] = None

    def create_weight_pack(self, device=None) -> WeightPack:
        device = f"cuda:{get_current_device_id()}" if device is None else device
        return WeightPack(
            weight=torch.empty(self.weight.shape, dtype=self.weight.dtype, device=device),
            scale=torch.empty(self.scale.shape, dtype=self.scale.dtype, device=device),
            zero_point=torch.empty(self.zero_point.shape, dtype=self.zero_point.dtype, device=device),
        )


class QuantizationMethod(ABC):
    def __init__(self):
        super().__init__()
        self.device_id_ = get_current_device_id()
        self.weight_suffix = None
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
        weights: torch.Tensor,
    ) -> WeightPack:
        pass

    @abstractmethod
    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "WeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def method_name(self):
        pass

    @abstractmethod
    def get_metadata(self, in_dim: int, out_dims: List[int], data_type: torch.dtype) -> QuantizedMetadata:
        # 针对一个数据类型和形状，返回量化后的元数据
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
