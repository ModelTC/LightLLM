import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    SingleMMWeightTpl,
    DeepGemmFP8W8A8B128MMWeight,
    AWQMMWeightTpl,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional
from .mm_slicer import ColSliceMixin, QuantizedRowSliceMixin, QuantizedColSliceMixin


class UnquantizedCOLMMWeight(SingleMMWeightTpl):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            weight_name=weight_name,
            data_type=data_type,
            bias_name=bias_name,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self.param_slicer = ColSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class DeepGemmFP8W8A8B128COLMMWeight(DeepGemmFP8W8A8B128MMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            weight_name=weight_name,
            data_type=data_type,
            bias_name=bias_name,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self.param_slicer = QuantizedColSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class AWQCOLMMWeight(AWQMMWeightTpl):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            weight_name=weight_name,
            data_type=data_type,
            bias_name=bias_name,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        # 注意这里不是错误，因为awq的weight是按inxout存的
        self.param_slicer = QuantizedRowSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class AWQMARLINCOLMMWeight(AWQCOLMMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            weight_name=weight_name,
            data_type=data_type,
            bias_name=bias_name,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )

    def _process_weight(self, weight: torch.Tensor) -> torch.Tensor:
        new_weight = self.quant_method._process_weight_after_loading(weight.cuda(get_current_device_id()))
        self.mm_param.weight = new_weight
        return

    def _process_weight_scale(self, weight_scale: torch.Tensor) -> torch.Tensor:
        new_weight_scale = self.quant_method._process_weight_scale_after_loading(
            weight_scale.cuda(get_current_device_id()).to(self.data_type_)
        )
        self.mm_param.weight_scale = new_weight_scale
        return

    def _process_weight_zero_point(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        new_weight_zero_point = self.quant_method._process_weight_zero_point_after_loading(
            weight_zero_point.cuda(get_current_device_id())
        )
        self.mm_param.weight_zero_point = new_weight_zero_point
        return


COLMM_WEIGHT_CLS_MAP = {
    "deepgemm-fp8w8a8-b128": DeepGemmFP8W8A8B128COLMMWeight,
    "awq": AWQCOLMMWeight,
    "awq_marlin": AWQMARLINCOLMMWeight,
}
