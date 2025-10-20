import os
import torch
from abc import abstractmethod
from typing import Optional, Tuple, List, Dict, Union, Type
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def generate_scale_name(name, weight_scale_suffix, act_scale_suffix):
    weight_scale_name = None
    act_scale_name = None
    if weight_scale_suffix is not None:
        weight_scale_name = ".".join(name.split(".")[:-1] + [weight_scale_suffix])
    if act_scale_suffix is not None:
        act_scale_name = ".".join(name.split(".")[:-1] + [act_scale_suffix])
    return weight_scale_name, act_scale_name


class MMWeightTpl(BaseWeightTpl):
    def __init__(
        self,
        data_type: torch.dtype,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(tp_rank, tp_world_size, data_type)
        self.quant_method = quant_method
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        # quantized_weight 用于标记加载的权重是已经量化的权重格式
        # 不需要做在线量化
        self.quantized_weight: bool = False
        # 标记是否存在 bias， 由子类初始化
        self.has_bias: bool = None

    def mm(
        self, input_tensor: torch.Tensor, out: Optional[torch.Tensor] = None, use_custom_tensor_mananger: bool = True
    ) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(
                input_tensor, self.weight, self.bias, out, use_custom_tensor_mananger=use_custom_tensor_mananger
            )
        if out is None:
            shape = (input_tensor.shape[0], self.weight.shape[1])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.bias is None:
            return torch.mm(input_tensor, self.weight, out=out)
        return torch.addmm(self.bias, input_tensor, self.weight, out=out)

    def verify_load(self) -> bool:
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.has_bias:
            load_ok = load_ok and self.bias is not None
        return load_ok

    def _process_weight(self, weight) -> None:
        if self.quant_method is not None and not self.quantized_weight:
            self.weight = self.quant_method.quantize(weight.to(self.data_type_).cuda(get_current_device_id()))
            return
        # 让 k dim 更连续，大多数split k 算法的算子可能能更快
        self.weight = weight.cuda(get_current_device_id()).transpose(0, 1)

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_name in weights:
            weight = self._slice_weight(weights[self.weight_name])
            self._process_weight(weight)

        if self.bias_name in weights:
            self.bias = self._slice_bias(weights[self.bias_name]).cuda(get_current_device_id())
        return


class MultiMMWeightTpl(MMWeightTpl):
    def __init__(
        self,
        weight_names: List[str],
        data_type: torch.dtype,
        bias_names: Optional[List[str]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(data_type, quant_method, tp_rank, tp_world_size)

        self.weight_names = weight_names
        self.bias_names = bias_names
        self.weights = [None] * len(self.weight_names)
        if self.bias_names is not None:
            self.biases = [None] * len(self.bias_names)
            self.has_bias = all(b is not None for b in self.bias_names) and len(bias_names) > 0
        else:
            self.biases = None
            self.has_bias = False

    def _pre_porcess_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.weight_names)):
            if self.weight_names[i] in weights:
                weight = weights[self.weight_names[i]]
                self.weights[i] = self._slice_weight(weight)
            if self.has_bias and self.bias_names[i] in weights:
                bias = weights[self.bias_names[i]]
                self.biases[i] = self._slice_bias(bias)

    def _fuse_weights(self) -> None:
        if self.weight is None and (None not in self.weights):
            weight = torch.cat(self.weights, dim=0)
            self._process_weight(weight)
            delattr(self, "weights")

        if self.has_bias and self.bias is None and (None not in self.biases):
            self.bias = torch.cat(self.biases, dim=0).cuda(get_current_device_id())
            delattr(self, "biases")
        return self

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        self._pre_porcess_weights(weights)

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        self._fuse_weights()
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
        fpweight = self.weight
        if out is None:
            shape = (input_tensor.shape[0], input_tensor.shape[1], fpweight.shape[2])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.bias is None:
            return torch.bmm(input_tensor, fpweight, out=out)
        return torch.addbmm(self.bias, input_tensor, fpweight, out=out)

    def _process_weight(self, weight) -> None:
        self.weight = weight.cuda(get_current_device_id())


class AWQMMWeightTpl(MMWeightTpl):
    def __init__(
        self,
        data_type: torch.dtype,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(data_type, quant_method, tp_rank, tp_world_size)
        self.weight = [None, None, None]

    def verify_load(self) -> bool:
        load_ok = True
        # Verify weight. The weight must be not None.
        weight_ok = all(w is not None for w in self.weight)
        load_ok = load_ok and weight_ok
        # Verify bias. If bias_name is set, it must be not None.
        if self.has_bias:
            load_ok = load_ok and self.bias is not None
        return load_ok

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_name is not None and self.weight_name in weights:
            weight = weights[self.weight_name]
            weight = self._slice_weight(weight)
            self.weight[0] = weight.cuda(get_current_device_id())
        if self.bias_name is not None and self.bias_name in weights:
            bias = weights[self.bias_name]
            bias = self._slice_bias(bias)
            self.bias = bias.cuda(get_current_device_id())

    def _load_scales(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_scale_name is not None and self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name]
            weight_scale = self._slice_weight_scale(weight_scale)
            self.weight[1] = weight_scale.cuda(get_current_device_id())

    def _load_zero_points(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_zero_point_name is not None and self.weight_zero_point_name in weights:
            weight_zero_point = weights[self.weight_zero_point_name]
            weight_zero_point = self._slice_weight_zero_point(weight_zero_point)
            self.weight[2] = weight_zero_point.cuda(get_current_device_id())


class AWQMultiMMWeightTpl(AWQMMWeightTpl):
    def __init__(
        self,
        weight_names: List[str],
        data_type: torch.dtype,
        bias_names: Optional[List[str]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(data_type, quant_method, tp_rank, tp_world_size)
        self.weight_names = [weight.replace("weight", quant_method.weight_suffix) for weight in weight_names]
        self.weight_scale_names = [
            weight.replace("weight", quant_method.weight_scale_suffix) for weight in weight_names
        ]
        self.weight_zero_point_names = [
            weight.replace("weight", quant_method.weight_zero_point_suffix) for weight in weight_names
        ]
        self.bias_names = bias_names
        self.weights = [None] * len(self.weight_names)
        self.weight_scales = [None] * len(self.weight_names)
        self.weight_zero_points = [None] * len(self.weight_names)
        if self.bias_names is not None:
            self.biases = [None] * len(self.bias_names)
            self.has_bias = all(b is not None for b in self.bias_names) and len(bias_names) > 0
        else:
            self.biases = None
            self.has_bias = False

    def _fuse(self) -> None:
        if self.weight[0] is None and (None not in self.weights):
            weight = torch.cat(self.weights, dim=1)
            self.weight[0] = weight.cuda(get_current_device_id())
            delattr(self, "weights")

        if self.weight[1] is None and (None not in self.weight_scales):
            # awq 保存的量化参数，weight shape 是 in x out。所以这里的cat dim 是 1
            weight_scale = torch.cat(self.weight_scales, dim=1).cuda(get_current_device_id())
            self.weight[1] = weight_scale.cuda(get_current_device_id())
            delattr(self, "weight_scales")

        if self.weight[2] is None and (None not in self.weight_zero_points):
            weight_zero_point = torch.cat(self.weight_zero_points, dim=1)
            self.weight[2] = weight_zero_point.cuda(get_current_device_id())
            print("weight_zero_point", self.weight[2].dtype)
            delattr(self, "weight_zero_points")

        if self.has_bias and self.bias is None and (None not in self.biases):
            self.bias = torch.cat(self.biases, dim=0).cuda(get_current_device_id())
            delattr(self, "biases")
        return self

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.weight_names)):
            if self.weight_names[i] is not None and self.weight_names[i] in weights:
                weight = weights[self.weight_names[i]]
                weight = self._slice_weight(weight)
                self.weights[i] = weight.cuda(get_current_device_id())

    def _load_scales(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.weight_names)):
            if self.weight_scale_names[i] is not None and self.weight_scale_names[i] in weights:
                weight_scale = weights[self.weight_scale_names[i]]
                weight_scale = self._slice_weight_scale(weight_scale)
                self.weight_scales[i] = weight_scale.cuda(get_current_device_id())

    def _load_zero_points(self, weights: Dict[str, torch.Tensor]) -> None:
        for i in range(len(self.weight_names)):
            if self.weight_zero_point_names[i] is not None and self.weight_zero_point_names[i] in weights:
                weight_zero_point = weights[self.weight_zero_point_names[i]]
                weight_zero_point = self._slice_weight_zero_point(weight_zero_point)
                self.weight_zero_points[i] = weight_zero_point.cuda(get_current_device_id())

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        self._fuse()
        return


class MMWeight:
    def __new__(cls, **kwargs):
        quant_cfg = kwargs.pop("quant_cfg", None)
        layer_num_ = kwargs.pop("layer_num", None)
        name = kwargs.pop("name", None)
        quant_method, quantized_weight = cls._get_quant_method(quant_cfg, layer_num_, name)
        kwargs["quant_method"] = quant_method
        mmcls = cls._get_mmcls(quant_method, quantized_weight)
        return mmcls(**kwargs)

    @classmethod
    def _get_quant_method(cls, quant_cfg: Quantcfg, layer_num_: int, name: str) -> QuantizationMethod:
        if quant_cfg is None:
            return None, False
        quant_method = quant_cfg.get_quant_method(layer_num_, name)
        quant_method.hf_quantization_method = quant_cfg.hf_quantization_method
        quantized_weight = quant_cfg.quantized_weight
        return quant_method, quantized_weight

    @classmethod
    def _get_mmcls(
        cls, quant_method: QuantizationMethod, quantized_weight: bool
    ) -> Type[Union[MMWeightTpl, MultiMMWeightTpl, BMMWeightTpl]]:
        raise NotImplementedError("Subclasses must implement _get_mmcls method")
