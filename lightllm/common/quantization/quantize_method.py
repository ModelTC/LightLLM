import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from lightllm.utils.dist_utils import get_current_device_id
from typing import Optional, Tuple, TYPE_CHECKING


@dataclass
class WeightPack:
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    weight_scale: Optional[torch.Tensor] = None
    weight_zero_point: Optional[torch.Tensor] = None


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
        output: WeightPack = None,
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
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def method_name(self):
        pass

    # weight 不需要传递dtype，因为每个量化算法，量化后的weight的dtype是确定的
    # scale/zero_point暂时保留外部dtype接口，兼容部分根据weight dtype来动态确定scale/zero_point的dtype的情况
    @abstractmethod
    def create_weight(self, out_dim: int, in_dim: int, dtype: torch.dtype, device_id: int) -> WeightPack:
        pass

    def weight_need_quanted(self, weight: torch.Tensor) -> bool:
        # 判断一个 weight 是否需要进行量化操作。
        return weight.dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]

    def params_need_repack(self) -> bool:
        """
        用于说明是否需要对量化后的权重进行repack操作，目前只有awq支持
        """
        return False

    def params_repack(
        self, weight: torch.Tensor, weight_scale: torch.Tensor, weight_zero_point: torch.Tensor, dtype_type: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        一些量化方法在将参数完成量化后，为了加速性能，还需要将参数进行重拍，使算子性能达到最优，如awq方法。
        """
        return weight, weight_scale, weight_zero_point
