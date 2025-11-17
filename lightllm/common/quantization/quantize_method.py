import torch
from abc import ABC, abstractmethod
from lightllm.utils.dist_utils import get_current_device_id
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightPack


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
        # 一些量化模式需要用到的额外量化参数，如awq量化
        self.hf_quantization_config = None

    @abstractmethod
    def quantize(self, weights: torch.Tensor):
        pass

    @abstractmethod
    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        bias: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def method_name(self):
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
