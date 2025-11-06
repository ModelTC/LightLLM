import os
import torch
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


@dataclass
class MMWeightPack:
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    weight_scale: Optional[torch.Tensor] = None
    weight_zero_point: Optional[torch.Tensor] = None

    has_bias: bool = False
    has_weight_scale: bool = False
    has_weight_zero_point: bool = False

    def is_ready(self) -> bool:
        return (
            self.weight is not None
            and (not self.has_bias or (self.has_bias and self.bias is not None))
            and (not self.has_weight_scale or (self.has_weight_scale and self.weight_scale is not None))
            and (not self.has_weight_zero_point or (self.has_weight_zero_point and self.weight_zero_point is not None))
        )


class MMWeightTpl(BaseWeightTpl):
    def __init__(
        self,
        data_type: torch.dtype,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
        has_bias: bool = False,
        has_weight_scale: bool = False,
        has_weight_zero_point: bool = False,
    ) -> None:
        super().__init__(tp_rank, tp_world_size, data_type)
        self.quant_method = quant_method
        self.mm_param: MMWeightPack = MMWeightPack(
            has_bias=has_bias,
            has_weight_scale=has_weight_scale,
            has_weight_zero_point=has_weight_zero_point,
        )
        self.param_slicer: SliceMixinTpl = None

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

    def load_hf_weights(self, weights):
        raise NotImplementedError("load_hf_weights must implement this method")

    def verify_load(self) -> bool:
        return self.mm_param.is_ready()

    def _process_weight(self, weight: torch.Tensor) -> None:
        # 由于所有的量化算法，都会产生一个scale，所以只要没有scale，就说明需要在线对weight进行量化
        if self.quant_method is not None and not self.mm_param.has_weight_scale:
            quantized_weight, weight_scale, weight_zero_point = self.quant_method.quantize(
                weight.to(self.data_type_).cuda(get_current_device_id())
            )
            self.mm_param.weight = quantized_weight
            self.mm_param.weight_scale = weight_scale
            self.mm_param.weight_zero_point = weight_zero_point
            return
        # 让 k dim 更连续，大多数split k 算法的算子可能能更快
        self.mm_param.weight = weight.to(self.data_type_).cuda(get_current_device_id()).transpose(0, 1)
        return

    def _process_bias(self, bias: torch.Tensor) -> None:
        self.mm_param.bias = bias.to(self.data_type_).cuda(get_current_device_id())
        return

    def _process_weight_scale(self, weight_scale: torch.Tensor) -> None:
        raise NotImplementedError("process_weight_scale must implement this method")

    def _process_weight_zero_point(self, weight_zero_point: torch.Tensor) -> None:
        raise NotImplementedError("process_weight_zero_point must implement this method")

    def _load_weight(self, weights: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError("load_weight_scale must implement this method")

    def _load_bias(self, weights: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError("load_bias must implement this method")

    def _load_weight_scale(self, weights: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError("load_weight_scale must implement this method")

    def _load_weight_zero_point(self, weights: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError("load_weight_zero_point must implement this method")

    def _fuse_weights(self, dim: int = 0) -> None:
        raise NotImplementedError("fuse_weights must implement this method")


class SingleMMWeightTpl(MMWeightTpl):
    def __init__(
        self,
        weight_name: str,
        bias_name: Optional[str] = None,
        data_type: torch.dtype = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
        has_weight_scale: bool = False,
        has_weight_zero_point: bool = False,
    ) -> None:
        super().__init__(
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            has_bias=bias_name is not None,
            has_weight_scale=has_weight_scale,
            has_weight_zero_point=has_weight_zero_point,
        )
        self.weight_name = weight_name
        self.bias_name = bias_name
        return

    def _load_weight(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_name in weights:
            weight = weights[self.weight_name]
            weight = self.param_slicer._slice_weight(weight)
            self._process_weight(weight)
        return

    def _load_bias(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.bias_name in weights:
            bias = self.param_slicer._slice_bias(weights[self.bias_name])
            self._process_bias(bias)
        return

    def load_hf_weights(self, weights):
        self._load_weight(weights)
        self._load_bias(weights)
        return


class MultiMMWeightTpl(MMWeightTpl):
    def __init__(
        self,
        weight_names: List[str],
        bias_names: Optional[List[str]] = None,
        data_type: torch.dtype = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
        has_weight_scale: bool = False,
        has_weight_zero_point: bool = False,
    ) -> None:
        has_bias = bias_names is not None and any(b is not None for b in bias_names)
        super().__init__(
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            has_bias=has_bias,
            has_weight_scale=has_weight_scale,
            has_weight_zero_point=has_weight_zero_point,
        )
        self.weight_names = weight_names
        self.bias_names = bias_names
        self.mm_params: List[MMWeightPack] = [
            MMWeightPack(
                weight=None,
                bias=None,
                weight_scale=None,
                weight_zero_point=None,
                has_bias=has_bias,
                has_weight_scale=has_weight_scale,
                has_weight_zero_point=has_weight_zero_point,
            )
            for _ in range(len(weight_names))
        ]

    def _load_weight(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.weight_names)):
            if self.weight_names[i] in weights:
                weight_i = weights[self.weight_names[i]]
                weight_i = self.param_slicer._slice_weight(weight_i)
                self.mm_params[i].weight = weight_i
        return

    def _load_bias(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.bias_names)):
            if self.bias_names[i] in weights:
                bias_i = weights[self.bias_names[i]]
                bias_i = self.param_slicer._slice_bias(bias_i)
                self.mm_params[i].bias = bias_i.to(self.data_type_)
        return

    def _fuse_weights(self, dim: int = 0) -> None:
        if self.mm_param.weight is None and all(p.weight is not None for p in self.mm_params):
            weight = torch.cat([p.weight for p in self.mm_params], dim=dim)
            self._process_weight(weight)
            for p in self.mm_params:
                p.weight = None

        if self.mm_param.has_bias and self.mm_param.bias is None and all(p.bias is not None for p in self.mm_params):
            bias = torch.cat([p.bias for p in self.mm_params], dim=dim)
            self._process_bias(bias)
            for p in self.mm_params:
                p.bias = None
        return

    def load_hf_weights(self, weights):
        self._load_weight(weights)
        self._load_bias(weights)
        self._fuse_weights(dim=0)
        return


class BMMWeightTpl(SingleMMWeightTpl):
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

    def _process_weight(self, weight) -> None:
        self.mm_param.weight = weight.cuda(get_current_device_id())


class SingleQuantizedMMWeightTpl(SingleMMWeightTpl):
    def __init__(
        self,
        weight_name: str,
        bias_name: Optional[str] = None,
        data_type: torch.dtype = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
        has_weight_scale: bool = True,
        has_weight_zero_point: bool = False,  # 目前较多的是对称量化，所以默认没有zero_point
    ) -> None:
        super().__init__(
            weight_name=weight_name,
            bias_name=bias_name,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            has_weight_scale=has_weight_scale,
            has_weight_zero_point=has_weight_zero_point,
        )
        assert quant_method is not None, "quant_method is not set"
        assert quant_method.weight_scale_suffix is not None, "weight_scale_suffix is not set"
        self.weight_scale_name = weight_name.replace("weight", quant_method.weight_scale_suffix)
        if has_weight_zero_point:
            assert quant_method.weight_zero_point_suffix is not None, "weight_zero_point_suffix is not set"
            self.weight_zero_point_name = weight_name.replace("weight", quant_method.weight_zero_point_suffix)
        if quant_method.weight_suffix is not None:
            self.weight_name = weight_name.replace("weight", quant_method.weight_suffix)
        return

    def _load_weight_scale(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_scale_name is not None and self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name]
            weight_scale = self.param_slicer._slice_weight_scale(weight_scale)
            self._process_weight_scale(weight_scale)

    def _load_weight_zero_point(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_zero_point_name is not None and self.weight_zero_point_name in weights:
            weight_zero_point = weights[self.weight_zero_point_name]
            weight_zero_point = self.param_slicer._slice_weight_zero_point(weight_zero_point)
            self._process_weight_zero_point(weight_zero_point)

    def load_hf_weights(self, weights):
        self._load_weight(weights)
        self._load_bias(weights)
        self._load_weight_scale(weights)
        self._load_weight_zero_point(weights)
        return

    # 不同的量化算法，往往需要不同的处理方式，所以强制要求实现这些方法
    def _process_weight(self, weight: torch.Tensor) -> None:
        raise NotImplementedError("Quantized weight process_weight must implement this method")

    def _process_weight_scale(self, weight_scale: torch.Tensor) -> None:
        raise NotImplementedError("Quantized weight process_weight_scale must implement this method")

    def _process_weight_zero_point(self, weight_zero_point: torch.Tensor) -> None:
        raise NotImplementedError("Quantized weight process_weight_zero_point must implement this method")


class MultiQuantizedMMWeightTpl(MultiMMWeightTpl):
    def __init__(
        self,
        weight_names: List[str],
        bias_names: Optional[List[str]] = None,
        data_type: torch.dtype = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
        has_weight_scale: bool = True,
        has_weight_zero_point: bool = False,
    ) -> None:
        super().__init__(
            weight_names=weight_names,
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            has_weight_scale=has_weight_scale,
            has_weight_zero_point=has_weight_zero_point,
        )
        assert quant_method is not None, "quant_method is not set"
        assert quant_method.weight_scale_suffix is not None, "weight_scale_suffix is not set"
        self.weight_scale_names = [
            weight_name.replace("weight", quant_method.weight_scale_suffix) for weight_name in weight_names
        ]
        if has_weight_zero_point:
            assert quant_method.weight_zero_point_suffix is not None, "weight_zero_point_suffix is not set"
            self.weight_zero_point_names = [
                weight_name.replace("weight", quant_method.weight_zero_point_suffix) for weight_name in weight_names
            ]
        if quant_method.weight_suffix is not None:
            self.weight_names = [
                weight_name.replace("weight", quant_method.weight_suffix) for weight_name in weight_names
            ]
        return

    def _load_weight(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.weight_names)):
            if self.weight_names[i] in weights:
                weight = weights[self.weight_names[i]]
                weight = self.param_slicer._slice_weight(weight)
                self.mm_params[i].weight = weight

    def _load_weight_scale(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.weight_names)):
            if self.weight_scale_names[i] is not None and self.weight_scale_names[i] in weights:
                weight_scale = weights[self.weight_scale_names[i]]
                weight_scale = self.param_slicer._slice_weight_scale(weight_scale)
                self.mm_params[i].weight_scale = weight_scale.to(self.data_type_)

    def _load_weight_zero_point(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.weight_names)):
            if self.weight_zero_point_names[i] is not None and self.weight_zero_point_names[i] in weights:
                weight_zero_point = weights[self.weight_zero_point_names[i]]
                weight_zero_point = self.param_slicer._slice_weight_zero_point(weight_zero_point)
                self.mm_params[i].weight_zero_point = weight_zero_point
        return

    def _fuse_weights(self, dim: int = 0) -> None:
        super()._fuse_weights(dim=dim)
        if self.mm_param.weight_scale is None and (None not in [p.weight_scale for p in self.mm_params]):
            # awq 保存的量化参数，weight shape 是 in x out。所以这里的cat dim 是 1
            weight_scale = torch.cat([p.weight_scale for p in self.mm_params], dim=dim).cuda(get_current_device_id())
            self._process_weight_scale(weight_scale)
            for p in self.mm_params:
                p.weight_scale = None

        if self.mm_param.weight_zero_point is None and (None not in [p.weight_zero_point for p in self.mm_params]):
            weight_zero_point = torch.cat([p.weight_zero_point for p in self.mm_params], dim=dim)
            self._process_weight_zero_point(weight_zero_point)
            for p in self.mm_params:
                p.weight_zero_point = None
        torch.cuda.empty_cache()
        return

    # 不同的量化算法，往往需要不同的处理方式，所以强制要求实现这些方法
    def _process_weight(self, weight: torch.Tensor) -> None:
        raise NotImplementedError("Quantized weight process_weight must implement this method")

    def _process_weight_scale(self, weight_scale: torch.Tensor) -> None:
        raise NotImplementedError("Quantized weight process_weight_scale must implement this method")

    def _process_weight_zero_point(self, weight_zero_point: torch.Tensor) -> None:
        raise NotImplementedError("Quantized weight process_weight_zero_point must implement this method")

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        self._load_weight(weights)
        self._load_bias(weights)
        self._load_weight_scale(weights)
        self._load_weight_zero_point(weights)
        self._fuse_weights(dim=0)
        return


class DeepGemmFP8W8A8B128MMWeight(SingleQuantizedMMWeightTpl):
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
            has_weight_scale=True,
            has_weight_zero_point=False,
        )

    def _process_weight_scale(self, weight_scale) -> None:
        self.mm_param.weight_scale = weight_scale.to(torch.float).cuda(get_current_device_id()).transpose(0, 1)
        return

    def _process_weight(self, weight) -> None:
        self.mm_param.weight = weight.cuda(get_current_device_id()).transpose(0, 1)
        return


class DeepGemmFP8W8A8B128MultiMMWeight(MultiQuantizedMMWeightTpl):
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
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            has_weight_scale=True,
            has_weight_zero_point=False,
        )

    def _process_weight_scale(self, weight_scale) -> None:
        self.mm_param.weight_scale = weight_scale.cuda(get_current_device_id()).transpose(0, 1)
        return

    def _process_weight(self, weight) -> None:
        self.mm_param.weight = weight.cuda(get_current_device_id()).transpose(0, 1)
        return


class AWQMMWeightTpl(SingleQuantizedMMWeightTpl):
    def __init__(
        self,
        weight_name: str,
        bias_name: Optional[str] = None,
        data_type: torch.dtype = None,
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
            has_weight_scale=True,
            has_weight_zero_point=True,
        )

    def _process_weight(self, weight: torch.Tensor) -> None:
        self.mm_param.weight = weight.cuda(get_current_device_id())
        return

    def _process_weight_scale(self, weight_scale: torch.Tensor) -> None:
        self.mm_param.weight_scale = weight_scale.to(self.data_type_).cuda(get_current_device_id())
        return

    def _process_weight_zero_point(self, weight_zero_point: torch.Tensor) -> None:
        self.mm_param.weight_zero_point = weight_zero_point.cuda(get_current_device_id())
        return


class AWQMultiMMWeightTpl(MultiQuantizedMMWeightTpl):
    def __init__(
        self,
        weight_names: List[str],
        bias_names: Optional[List[str]] = None,
        data_type: torch.dtype = None,
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
            has_weight_scale=True,
            has_weight_zero_point=True,
        )

    def _process_weight(self, weight: torch.Tensor) -> None:
        self.mm_param.weight = weight.cuda(get_current_device_id())
        return

    def _process_weight_scale(self, weight_scale: torch.Tensor) -> None:
        self.mm_param.weight_scale = weight_scale.to(self.data_type_).cuda(get_current_device_id())
        return

    def _process_weight_zero_point(self, weight_zero_point: torch.Tensor) -> None:
        self.mm_param.weight_zero_point = weight_zero_point.cuda(get_current_device_id())
        return

    def load_hf_weights(self, weights):
        self._load_weight(weights)
        self._load_bias(weights)
        self._load_weight_scale(weights)
        self._load_weight_zero_point(weights)
        # 由于awq的储存格式是inxout，所以拼接dim是 1
        self._fuse_weights(dim=1)
        return
