import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeightTpl,
    DeepGemmFP8W8A8B128MMWeight,
    AWQMMWeightTpl,
    BMMWeightTpl,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional, Union
from .mm_slicer import RowSliceMixin, QuantizedRowSliceMixin, QuantizedColSliceMixin


class StandardROWMMWeight(MMWeightTpl):
    def __init__(
        self,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            weight_names=weight_names,
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self.param_slicer = RowSliceMixin(tp_rank=tp_rank, tp_world_size=tp_world_size)


class DeepGemmFP8W8A8B128ROWMMWeight(DeepGemmFP8W8A8B128MMWeight):
    def __init__(
        self,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
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
        return


class UnquantizedROWBMMWeight(BMMWeightTpl):
    def __init__(
        self,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
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


class AWQROWMMWeight(AWQMMWeightTpl):
    def __init__(
        self,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
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
        self.param_slicer = QuantizedColSliceMixin(
            tp_rank=tp_rank, tp_world_size=tp_world_size, bias_div_world_size=False
        )


class AWQMARLINROWMMWeight(AWQROWMMWeight):
    def __init__(
        self,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
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


ROWMM_WEIGHT_CLS_MAP = {
    "deepgemm-fp8w8a8-b128": DeepGemmFP8W8A8B128ROWMMWeight,
    "awq": AWQROWMMWeight,
    "awq_marlin": AWQMARLINROWMMWeight,
}
