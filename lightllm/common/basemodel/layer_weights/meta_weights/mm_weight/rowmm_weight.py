import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    SingleMMWeightTpl,
    MultiMMWeightTpl,
    DeepGemmFP8W8A8B128MMWeight,
    DeepGemmFP8W8A8B128MultiMMWeight,
    AWQMMWeightTpl,
    AWQMultiMMWeightTpl,
    BMMWeightTpl,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional
from .mm_slicer import RowSliceMixin, QuantizedRowSliceMixin, QuantizedColSliceMixin


class UnquantizedROWMMWeight(SingleMMWeightTpl):
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
            bias_name=bias_name,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self.param_slicer = RowSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class UnquantizedMultiROWMMWeight(MultiMMWeightTpl):
    def __init__(
        self,
        weight_names: str,
        data_type: torch.dtype,
        bias_names: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            weight_names=weight_names,
            data_type=data_type,
            bias_names=bias_names,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self.param_slicer = RowSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class DeepGemmFP8W8A8B128ROWMMWeight(DeepGemmFP8W8A8B128MMWeight):
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
        self.param_slicer = QuantizedRowSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)
        return


class DeepGemmFP8W8A8B128MultiROWMMWeight(DeepGemmFP8W8A8B128MultiMMWeight):
    def __init__(
        self,
        weight_names: str,
        data_type: torch.dtype,
        bias_names: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            weight_names=weight_names,
            data_type=data_type,
            bias_names=bias_names,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self.param_slicer = QuantizedRowSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class UnquantizedROWBMMWeight(BMMWeightTpl):
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
        self.param_slicer = RowSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class AWQROWMMWeight(AWQMMWeightTpl):
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
        self.param_slicer = QuantizedColSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class AWQMultiROWMMWeight(AWQMultiMMWeightTpl):
    def __init__(
        self,
        weight_names: List[str],
        data_type: torch.dtype,
        bias_names: Optional[List[str]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            weight_names=weight_names,
            data_type=data_type,
            bias_names=bias_names,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        # 注意这里不是错误，因为awq的weight是按inxout存的
        self.param_slicer = QuantizedColSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class AWQMARLINROWMMWeight(AWQROWMMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(weight_name, data_type, bias_name, quant_method, tp_rank, tp_world_size)

    def _process_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self.quant_method._process_weight_after_loading(weight.cuda(get_current_device_id()))

    def _process_weight_scale(self, weight_scale: torch.Tensor) -> torch.Tensor:
        return self.quant_method._process_weight_scale_after_loading(
            weight_scale.cuda(get_current_device_id()).to(self.data_type_)
        )

    def _process_weight_zero_point(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        return self.quant_method._process_weight_zero_point_after_loading(
            weight_zero_point.cuda(get_current_device_id())
        )


class AWQMARLINMultiROWMMWeight(AWQMultiROWMMWeight):
    def __init__(
        self,
        weight_names: List[str],
        data_type: torch.dtype,
        bias_names: Optional[List[str]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(weight_names, data_type, bias_names, quant_method, tp_rank, tp_world_size)

    def _process_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self.quant_method._process_weight_after_loading(weight.cuda(get_current_device_id()))

    def _process_weight_scale(self, weight_scale: torch.Tensor) -> torch.Tensor:
        return self.quant_method._process_weight_scale_after_loading(
            weight_scale.cuda(get_current_device_id()).to(self.data_type_)
        )

    def _process_weight_zero_point(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        return self.quant_method._process_weight_zero_point_after_loading(
            weight_zero_point.cuda(get_current_device_id())
        )


ROWMM_WEIGHT_CLS_MAP = {
    "deepgemm-fp8w8a8-b128": DeepGemmFP8W8A8B128ROWMMWeight,
    "awq": AWQROWMMWeight,
    "awq_marlin": AWQMARLINROWMMWeight,
}

MULTI_ROWMM_WEIGHT_CLS_MAP = {
    "deepgemm-fp8w8a8-b128": DeepGemmFP8W8A8B128MultiROWMMWeight,
    "awq": AWQMultiROWMMWeight,
    "awq_marlin": AWQMARLINMultiROWMMWeight,
}
