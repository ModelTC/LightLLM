import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeightTpl,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional, Union
from .mm_slicer import get_col_slice_mixin


class COLMMWeight(MMWeightTpl):
    def __init__(
        self,
        in_dim: int,
        out_dims: Optional[Union[int, List[int]]],
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dims=out_dims,
            weight_names=weight_names,
            data_type=data_type,
            bias_names=bias_names,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self.param_slicer = get_col_slice_mixin(quant_method.method_name, tp_rank=tp_rank, tp_world_size=tp_world_size)
