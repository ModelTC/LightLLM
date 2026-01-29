import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl, BMMWeightTpl
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional, Union
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from .mm_slicer import get_row_slice_mixin


class ROWMMWeight(MMWeightTpl):
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
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        out_dims = [self._get_tp_dim(out_dim) for out_dim in out_dims]
        super().__init__(
            in_dim=in_dim,
            out_dims=out_dims,
            weight_names=weight_names,
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
        )
        self.param_slicer = get_row_slice_mixin(
            self.quant_method.method_name, tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_
        )


class KVROWNMMWeight(MMWeightTpl):
    def __init__(
        self,
        in_dim: int,
        kv_head_num: int,
        head_dim: int,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.repeat_times = 1
        assert kv_head_num % self.tp_world_size_ == 0 or self.tp_world_size_ % kv_head_num == 0, (
            f"kv_head_num must be divisible by tp_world_size_ or "
            f"tp_world_size_ must be divisible by kv_head_num, "
            f"but found: {kv_head_num} % {self.tp_world_size_}"
        )
        kv_hidden_size = self._get_tp_padded_head_num(kv_head_num) * head_dim
        out_dims = [kv_hidden_size, kv_hidden_size]
        super().__init__(
            in_dim=in_dim,
            out_dims=out_dims,
            weight_names=weight_names,
            data_type=data_type,
            bias_names=bias_names,
            quant_method=quant_method,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
        )
        self.param_slicer = get_row_slice_mixin(
            self.quant_method.method_name,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
            repeat_times=self.repeat_times,
        )

    def _get_tp_padded_head_num(self, head_num: int):
        if head_num % self.tp_world_size_ == 0:
            return head_num // self.tp_world_size_
        elif self.tp_world_size_ % head_num == 0:
            self.repeat_times = self.tp_world_size_ // head_num
            return self.repeat_times * head_num // self.tp_world_size_
        else:
            raise ValueError(
                f"head_num must be divisible by tp_world_size_ or "
                f"tp_world_size_ must be divisible by head_num, "
                f"but found: {head_num} % {self.tp_world_size_}"
            )


class ROWBMMWeight(BMMWeightTpl):
    def __init__(
        self,
        dim0: int,
        dim1: int,
        dim2: int,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        assert (
            dim0 % self.tp_world_size_ == 0
        ), f"dim0 of bmm must be divisible by tp_world_size_, but found: {dim0} % {self.tp_world_size_}"
        dim0 = dim0 // self.tp_world_size_
        super().__init__(
            dim0=dim0,
            dim1=dim1,
            dim2=dim2,
            weight_names=weight_names,
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
        )
        self.param_slicer = get_row_slice_mixin(
            quant_method_name="none", tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_
        )
