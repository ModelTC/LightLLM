import os
import torch
import threading
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union, Type
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.log_utils import init_logger
from .mm_slicer import SliceMixinTpl

logger = init_logger(__name__)


class MMWeightTpl(BaseWeightTpl):
    def __init__(
        self,
        in_dim: int,
        out_dims: Optional[Union[int, List[int]]],
        weight_names: Union[str, List[str]],
        bias_names: Optional[Union[str, List[str]]],
        data_type: torch.dtype,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(tp_rank, tp_world_size, data_type)

        self.in_dim = in_dim
        if isinstance(out_dims, int):
            out_dims = [out_dims]
        self.out_dims = out_dims

        if isinstance(weight_names, str):
            weight_names = [weight_names]
        if isinstance(bias_names, str):
            bias_names = [bias_names]

        # 过滤输入的bias_names 是list， 但是内容全是None的情况
        if isinstance(bias_names, list):
            if bias_names[0] is None:
                bias_names = None

        # 同时存在 weight_names 和 quanted_weight_names 是为了兼容在线和离线两种加载方案
        self.weight_names = weight_names
        self.bias_names = bias_names
        self.quant_method: QuantizationMethod = NoQuantization() if quant_method is None else quant_method
        self.param_slicer: SliceMixinTpl = None
        self._create_weight()
        self.gen_weight_quant_param_names(quant_method=quant_method)

    def mm(
        self, input_tensor: torch.Tensor, out: Optional[torch.Tensor] = None, use_custom_tensor_mananger: bool = True
    ) -> torch.Tensor:
        return self.quant_method.apply(
            input_tensor, self.mm_param, out, use_custom_tensor_mananger=use_custom_tensor_mananger, bias=self.bias
        )

    def gen_weight_quant_param_names(self, quant_method: Optional[QuantizationMethod]):
        if quant_method.method_name == "none":
            self.weight_zero_point_names = None
            self.weight_scale_names = None
            return

        quanted_weight_names = []
        weight_scale_names = []
        weight_zero_point_names = []

        for weight_name in self.weight_names:
            if quant_method.weight_scale_suffix is not None:
                weight_scale_name = weight_name.replace("weight", quant_method.weight_scale_suffix)
                weight_scale_names.append(weight_scale_name)
            if quant_method.weight_zero_point_suffix is not None:
                weight_zero_point_name = weight_name.replace("weight", quant_method.weight_zero_point_suffix)
                weight_zero_point_names.append(weight_zero_point_name)
            if quant_method.weight_suffix is not None:
                weight_name = weight_name.replace("weight", quant_method.weight_suffix)
                quanted_weight_names.append(weight_name)

        if len(quanted_weight_names) != 0:
            self.quanted_weight_names = quanted_weight_names

        if len(weight_scale_names) != 0:
            self.weight_scale_names = weight_scale_names
        else:
            self.weight_scale_names = None

        if len(weight_zero_point_names) != 0:
            self.weight_zero_point_names = weight_zero_point_names
        else:
            self.weight_zero_point_names = None
        return

    def load_hf_weights(self, weights):

        for sub_child_index, param_name in enumerate(self.weight_names):
            self._load_weight(param_name=param_name, weights=weights, sub_child_index=sub_child_index)

        if self.bias_names is not None:
            for sub_child_index, param_name in enumerate(self.bias_names):
                self._load_bias(param_name=param_name, weights=weights, sub_child_index=sub_child_index)
        if self.weight_scale_names is not None:
            for sub_child_index, param_name in enumerate(self.weight_scale_names):
                self._load_weight_scale(param_name=param_name, weights=weights, sub_child_index=sub_child_index)
        if self.weight_zero_point_names is not None:
            for sub_child_index, param_name in enumerate(self.weight_zero_point_names):
                self._load_weight_zero_point(param_name=param_name, weights=weights, sub_child_index=sub_child_index)

    def _create_weight(self):
        self.bias = None
        if self.bias_names is not None:
            self.bias = torch.empty(sum(self.out_dims), dtype=self.data_type_).cuda(get_current_device_id())
            # bias_list shares storage with bias for each output shard
            self.bias_list = torch.split(self.bias, self.out_dims, dim=0)
            for sub_bias in self.bias_list:
                sub_bias.load_ok = False
        self.mm_param: WeightPack = None
        self.mm_param_list: List[WeightPack] = None
        self.mm_param, self.mm_param_list = self.quant_method.create_weight(
            in_dim=self.in_dim, out_dims=self.out_dims, dtype=self.data_type_, device_id=get_current_device_id()
        )
        return

    # 执行顺序
    def _load_weight(
        self, param_name: Union[str, List[str]], weights: Dict[str, torch.Tensor], sub_child_index: int
    ) -> None:
        if param_name in weights:
            weight = self.param_slicer._slice_weight(weights[param_name])
            self.quant_method.load_weight(weight, self.mm_param_list[sub_child_index])
        return

    def _load_bias(
        self, param_name: Union[str, List[str]], weights: Dict[str, torch.Tensor], sub_child_index: int
    ) -> None:
        if param_name in weights:
            bias = self.param_slicer._slice_bias(weights[param_name])
            self.bias_list[sub_child_index].copy_(bias)
            self.bias_list[sub_child_index].load_ok = True
        return

    def _load_weight_scale(
        self, param_name: Union[str, List[str]], weights: Dict[str, torch.Tensor], sub_child_index: int
    ) -> None:
        if param_name in weights:
            weight_scale = self.param_slicer._slice_weight_scale(weights[param_name])
            self.quant_method.load_weight_scale(weight_scale, self.mm_param_list[sub_child_index])
        return

    def _load_weight_zero_point(
        self, param_name: Union[str, List[str]], weights: Dict[str, torch.Tensor], sub_child_index: int
    ) -> None:
        if param_name in weights:
            weight_zero_point = self.param_slicer._slice_weight_zero_point(weights[param_name])
            self.quant_method.load_weight_zero_point(weight_zero_point, self.mm_param_list[sub_child_index])
        return

    def verify_load(self):
        mm_param_load_ok = all(all(_mm_param.load_ok) for _mm_param in self.mm_param_list)
        bias_load_ok = True if self.bias is None else all(sub_bias.load_ok for sub_bias in self.bias_list)
        if not (mm_param_load_ok and bias_load_ok):
            logger.warning(f"mm_param_load_ok: {self.mm_param_list[0].load_ok}")
        return mm_param_load_ok and bias_load_ok

    def _get_tp_dim(self, dim: int) -> int:
        assert (
            dim % self.tp_world_size_ == 0
        ), f"dim must be divisible by tp_world_size_, but found: {dim} % {self.tp_world_size_}"
        return dim // self.tp_world_size_


class BMMWeightTpl(BaseWeightTpl):
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
        super().__init__(tp_rank, tp_world_size, data_type)
        if isinstance(weight_names, str):
            weight_names = [weight_names]
        self.weight_names = weight_names
        self.bias_names = bias_names
        assert bias_names is None, "bmm not support bias"
        if isinstance(bias_names, list):
            assert all(bias_name is None for bias_name in bias_names), "bmm not support bias"
        assert quant_method is None, "bmm not support quantized weight"
        self.quant_method = quant_method
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self._create_weight()
        return

    def _create_weight(self):
        self.weight = torch.empty(self.dim0, self.dim1, self.dim2, dtype=self.data_type_).cuda(get_current_device_id())
        self.weight.load_ok = False
        return

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        for weight_name in self.weight_names:
            if weight_name in weights:
                weight = self.param_slicer._slice_weight(weights[weight_name])
                self.weight.copy_(weight)
                self.weight.load_ok = True
        return

    def verify_load(self):
        return self.weight.load_ok

    def bmm(
        self, input_tensor: torch.Tensor, out: Optional[torch.Tensor] = None, use_custom_tensor_mananger: bool = True
    ) -> torch.Tensor:
        # 目前 bmm 不支持量化运算操作
        fpweight = self.weight
        if out is None:
            shape = (input_tensor.shape[0], input_tensor.shape[1], fpweight.shape[2])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        return torch.bmm(input_tensor, fpweight, out=out)
