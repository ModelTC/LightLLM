import os
import torch
import threading
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union, Type
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.quantization import Quantcfg
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
        self.lock = threading.Lock()

        self.in_dim = in_dim
        if isinstance(out_dims, int):
            out_dims = [out_dims]
        self.out_dims = out_dims
        self.cusum_out_dims = [0]
        for out_dim in out_dims[:-1]:
            self.cusum_out_dims.append(self.cusum_out_dims[-1] + out_dim)

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

        self.gen_weight_quant_param_names(quant_method=quant_method)
        self.quant_method = quant_method
        self.param_slicer: SliceMixinTpl = None

        self.weight_fused_dim = 0
        self.bias_fused_dim = 0
        self.weight_scale_and_zero_point_fused_dim = 0
        self.load_finished: bool = False
        self._create_weight()

    def _create_weight(self):
        self.weight_pack = self.quant_method.create_weight(self)

        if self.bias_names is not None:
            self.bias = torch.empty(sum(self.out_dims), dtype=self.data_type_).cuda(get_current_device_id())
        return

    def mm(
        self, input_tensor: torch.Tensor, out: Optional[torch.Tensor] = None, use_custom_tensor_mananger: bool = True
    ) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(
                input_tensor, self.mm_param, out, use_custom_tensor_mananger=use_custom_tensor_mananger
            )
        if out is None:
            shape = (input_tensor.shape[0], self.mm_param.weight.shape[1])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.mm_param.bias is None:
            return torch.mm(input_tensor, self.mm_param.weight, out=out)
        return torch.addmm(self.mm_param.bias, input_tensor, self.mm_param.weight, out=out)

    def gen_weight_quant_param_names(self, quant_method: Optional[QuantizationMethod]):
        if quant_method is None:
            self.quanted_weight_names = None
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
        else:
            self.quanted_weight_names = None

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

        if self.quanted_weight_names is not None:
            for sub_child_index, param_name in enumerate(self.quanted_weight_names):
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

    def verify_load(self) -> bool:
        return self.mm_param.is_ready()

    # 执行顺序
    def _load_weight(
        self, param_name: Union[str, List[str]], weights: Dict[str, torch.Tensor], sub_child_index: int
    ) -> None:
        if param_name in weights:
            weight = self.param_slicer._slice_weight(weights[param_name])
            start_idx = self.cusum_out_dims[sub_child_index]
            end_idx = start_idx + weight.shape[0]
            if self.quant_method is not None and self.quant_method.weight_need_quanted(weight):
                qweight, weight_scale, weight_zero_point = self.quant_method.quantize(weight)
                self.mm_param.weight[start_idx:end_idx, :].copy_(qweight)
                self.mm_param.weight_scale[start_idx:end_idx].copy_(weight_scale)
                if self.mm_param.has_weight_zero_point:
                    self.mm_param.weight_zero_point[start_idx:end_idx].copy_(weight_zero_point)
            else:
                self.mm_param.weight[:, start_idx:end_idx].copy_(weight.t())
        return

    def _load_bias(
        self, param_name: Union[str, List[str]], weights: Dict[str, torch.Tensor], sub_child_index: int
    ) -> None:
        if param_name in weights:
            bias = self.param_slicer._slice_bias(weights[param_name])
            start_idx = self.cusum_out_dims[sub_child_index]
            end_idx = start_idx + bias.shape[0]
            self.mm_param.bias[start_idx:end_idx].copy_(bias)
        return

    def _load_weight_scale(
        self, param_name: Union[str, List[str]], weights: Dict[str, torch.Tensor], sub_child_index: int
    ) -> None:
        if param_name in weights:
            weight_scale = self.param_slicer._slice_weight_scale(weights[param_name])
            self.sub_child_mm_params[sub_child_index].weight_scale = weight_scale
        return

    def _load_weight_zero_point(
        self, param_name: Union[str, List[str]], weights: Dict[str, torch.Tensor], sub_child_index: int
    ) -> None:
        if param_name in weights:
            weight_zero_point = self.param_slicer._slice_weight_zero_point(weights[param_name])
            self.sub_child_mm_params[sub_child_index].weight_zero_point = weight_zero_point
        return


class BMMWeightTpl(MMWeightTpl):
    def mm(
        self, input_tensor: torch.Tensor, out: Optional[torch.Tensor] = None, use_custom_tensor_mananger: bool = True
    ) -> torch.Tensor:
        raise RuntimeError("use bmm not mm")

    def bmm(
        self, input_tensor: torch.Tensor, out: Optional[torch.Tensor] = None, use_custom_tensor_mananger: bool = True
    ) -> torch.Tensor:
        # 目前 bmm 不支持量化运算操作
        fpweight = self.mm_param.weight
        if out is None:
            shape = (input_tensor.shape[0], input_tensor.shape[1], fpweight.shape[2])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.mm_param.bias is None:
            return torch.bmm(input_tensor, fpweight, out=out)
        return torch.addbmm(self.mm_param.bias, input_tensor, fpweight, out=out)

    def _to_gpu_device(self) -> None:
        if self.mm_param.weight is not None:
            if self.quant_method is not None:
                self.mm_param.weight = self.mm_param.weight.cuda(get_current_device_id())
            else:
                # bmm 不需要 transpose 操作
                self.mm_param.weight = self.mm_param.weight.to(self.data_type_).cuda(get_current_device_id())
        if self.mm_param.weight_scale is not None:
            self.mm_param.weight_scale = self.mm_param.weight_scale.cuda(get_current_device_id())
        if self.mm_param.weight_zero_point is not None:
            self.mm_param.weight_zero_point = self.mm_param.weight_zero_point.cuda(get_current_device_id())
        if self.mm_param.bias is not None:
            # TODO 是不是所有的bias都需要转换为全局设置的数据类型吗，会不会影响精度
            self.mm_param.bias = self.mm_param.bias.to(self.data_type_).cuda(get_current_device_id())
        return


class DeepGemmFP8W8A8B128MMWeight(MMWeightTpl):
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
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )

    def _to_gpu_device(self) -> None:
        if self.mm_param.weight is not None:
            self.mm_param.weight = self.mm_param.weight.cuda(get_current_device_id()).transpose(0, 1)
        if self.mm_param.weight_scale is not None:
            self.mm_param.weight_scale = self.mm_param.weight_scale.cuda(get_current_device_id()).transpose(0, 1)

        assert self.mm_param.has_weight_zero_point is False

        if self.mm_param.bias is not None:
            # TODO 是不是所有的bias都需要转换为全局设置的数据类型吗，会不会影响精度
            self.mm_param.bias = self.mm_param.bias.to(self.data_type_).cuda(get_current_device_id())
        return


class AWQMMWeightTpl(MMWeightTpl):
    def __init__(
        self,
        in_dim: int,
        out_dims: Optional[Union[int, List[int]]],
        weight_names: Union[str, List[str]],
        bias_names: Optional[Union[str, List[str]]] = None,
        data_type: torch.dtype = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dims=out_dims,
            weight_names=weight_names,
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        self.weight_fused_dim = 1
        self.bias_fused_dim = 0
        self.weight_scale_and_zero_point_fused_dim = 1

    def _to_gpu_device(self) -> None:
        if self.mm_param.weight is not None:
            self.mm_param.weight = self.mm_param.weight.cuda(get_current_device_id())
        if self.mm_param.weight_scale is not None:
            self.mm_param.weight_scale = self.mm_param.weight_scale.to(self.data_type_).cuda(get_current_device_id())
        if self.mm_param.weight_zero_point is not None:
            self.mm_param.weight_zero_point = self.mm_param.weight_zero_point.cuda(get_current_device_id())
        if self.mm_param.bias is not None:
            # TODO 是不是所有的bias都需要转换为全局设置的数据类型吗，会不会影响精度
            self.mm_param.bias = self.mm_param.bias.to(self.data_type_).cuda(get_current_device_id())
        return
