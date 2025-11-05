import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeight,
    MMWeightTpl,
    BMMWeightTpl,
    MultiMMWeightTpl,
    AWQMMWeightTpl,
    AWQMultiMMWeightTpl,
    generate_scale_name,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional


class ROWMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedROWMMWeight

        return ROWBMM_WEIGHT_CLS_MAP[quant_method.method_name]


class MultiROWMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedMultiROWMMWeight

        return MULTI_ROWBMM_WEIGHT_CLS_MAP[quant_method.method_name]


class ROWBMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedROWBMMWeight
        else:
            return W8A8B128ROWBMMWeight
        # TODO: Implement more quantization weight
        return None


class UnquantizedROWMMWeight(MMWeightTpl):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.has_bias = bias_name is not None
        super().__init__(data_type, quant_method, tp_rank, tp_world_size)

    def _slice_weight(self, weight: torch.Tensor):
        assert weight.shape[0] % self.tp_world_size_ == 0, f"tp slice error {weight.shape[0]} % {self.tp_world_size_}"
        tp_size = weight.shape[0] // self.tp_world_size_
        return weight[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)].to(self.data_type_)

    def _slice_bias(self, bias):
        assert bias.shape[0] % self.tp_world_size_ == 0, f"tp slice error {bias.shape[0]} % {self.tp_world_size_}"
        tp_size = bias.shape[0] // self.tp_world_size_
        return bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)].to(self.data_type_)


class W8A8B128ROWMMWeight(UnquantizedROWMMWeight):
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

        self.weight_scale_name, _ = generate_scale_name(
            weight_name, quant_method.weight_scale_suffix, quant_method.act_scale_suffix
        )
        self.weight_scale: Optional[torch.Tensor] = None
        self.quantized_weight = True

    def _slice_weight(self, weight: torch.Tensor):
        assert weight.shape[0] % self.tp_world_size_ == 0, f"tp slice error {weight.shape[0]} % {self.tp_world_size_}"
        tp_size = weight.shape[0] // self.tp_world_size_
        return weight[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_bias(self, bias):
        assert bias.shape[0] % self.tp_world_size_ == 0, f"tp slice error {bias.shape[0]} % {self.tp_world_size_}"
        tp_size = bias.shape[0] // self.tp_world_size_
        return bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        assert (
            weight_scale.shape[0] % self.tp_world_size_ == 0
        ), f"tp slice error {weight_scale.shape[0]} % {self.tp_world_size_}"
        tp_size = weight_scale.shape[0] // self.tp_world_size_
        scale_start = tp_size * self.tp_rank_
        scale_end = tp_size * (self.tp_rank_ + 1)
        return weight_scale.to(torch.float)[scale_start:scale_end]

    def _process_weight_scale(self, weight_scale) -> None:
        self.weight_scale = weight_scale.cuda(get_current_device_id()).transpose(0, 1)

    def _process_weight(self, weight) -> None:
        self.weight = weight.cuda(get_current_device_id()).transpose(0, 1)

    def _load_scales(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name]
            weight_scale = self._slice_weight_scale(weight_scale)
            self._process_weight_scale(weight_scale)

        if self.weight_scale is not None and isinstance(self.weight, torch.Tensor):
            self.weight = [
                self.weight,
                self.weight_scale,
                None,  # placeholder for input scale
            ]
        return


class UnquantizedMultiROWMMWeight(MultiMMWeightTpl):
    _slice_weight = UnquantizedROWMMWeight._slice_weight
    _slice_bias = UnquantizedROWMMWeight._slice_bias

    def __init__(
        self,
        weight_names: str,
        data_type: torch.dtype,
        bias_names: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(weight_names, data_type, bias_names, quant_method, tp_rank, tp_world_size)


class W8A8B128MultiROWMMWeight(UnquantizedMultiROWMMWeight):
    _slice_weight = W8A8B128ROWMMWeight._slice_weight
    _slice_bias = W8A8B128ROWMMWeight._slice_bias
    _slice_weight_scale = W8A8B128ROWMMWeight._slice_weight_scale

    def __init__(
        self,
        weight_names: str,
        data_type: torch.dtype,
        bias_names: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(weight_names, data_type, bias_names, quant_method, tp_rank, tp_world_size)
        self.weight_scale_names = []
        self.weight_scale: Optional[torch.Tensor] = None
        self.weight_scales = [None] * len(self.weight_names)
        for weight_name in weight_names:
            weight_scale_name, act_scale_name = generate_scale_name(
                weight_name, quant_method.weight_scale_suffix, quant_method.act_scale_suffix
            )
            self.weight_scale_names.append(weight_scale_name)
        self.quantized_weight = True

    def _load_scales(self, weights):
        for i in range(len(self.weight_names)):
            if self.weight_scale_names[i] in weights:
                weight_scale = weights[self.weight_scale_names[i]]
                weight_scale = self._slice_weight_scale(weight_scale)
                self.weight_scales[i] = weight_scale

    def _process_weight_scale(self, weight_scale) -> None:
        self.weight_scale = weight_scale.cuda(get_current_device_id()).transpose(0, 1)

    def _process_weight(self, weight) -> None:
        self.weight = weight.cuda(get_current_device_id()).transpose(0, 1)

    def _fuse_weights(self) -> None:
        super()._fuse_weights()
        if self.weight_scale is None and (None not in self.weight_scales):
            weight_scale = torch.cat(self.weight_scales, dim=0).cuda(get_current_device_id())
            self._process_weight_scale(weight_scale)
            delattr(self, "weight_scales")

        if self.weight_scale is not None and isinstance(self.weight, torch.Tensor):
            self.weight = [
                self.weight,
                self.weight_scale,
                None,
            ]


class UnquantizedROWBMMWeight(BMMWeightTpl):
    _slice_weight = UnquantizedROWMMWeight._slice_weight
    _slice_bias = UnquantizedROWMMWeight._slice_bias

    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.has_bias = bias_name is not None
        super().__init__(data_type, quant_method, tp_rank, tp_world_size)


class W8A8B128ROWBMMWeight(UnquantizedROWBMMWeight):
    _slice_weight = W8A8B128ROWMMWeight._slice_weight
    _slice_bias = W8A8B128ROWMMWeight._slice_bias

    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
        weight_scale_suffix: Optional[str] = None,
        act_scale_suffix: Optional[str] = None,
    ) -> None:
        super().__init__(weight_name, data_type, bias_name, quant_method, tp_rank, tp_world_size)
        self.weight_scale_name, self.act_scale_name = generate_scale_name(
            weight_name, weight_scale_suffix, act_scale_suffix
        )
        self.weight_scale: Optional[torch.Tensor] = None
        self.quantized_weight = True

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        tp_size = weight_scale.shape[0] // self.tp_world_size_
        scale_start = tp_size * self.tp_rank_
        scale_end = tp_size * (self.tp_rank_ + 1)
        return weight_scale[scale_start:scale_end].to(torch.float)

    def _load_scales(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_scale_name is not None and self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name]
            weight_scale = self._slice_weight_scale(weight_scale)

        if self.weight_scale is not None and isinstance(self.weight, torch.Tensor):
            self.weight = [
                self.weight,
                self.weight_scale,
                None,
            ]


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
        super().__init__(data_type, quant_method, tp_rank, tp_world_size)
        self.weight_name = weight_name.replace("weight", quant_method.weight_suffix)
        self.weight_scale_name = weight_name.replace("weight", quant_method.weight_scale_suffix)
        self.weight_zero_point_name = weight_name.replace("weight", quant_method.weight_zero_point_suffix)
        self.bias_name = bias_name
        self.weight_scale: Optional[torch.Tensor] = None
        self.quantized_weight = True
        self.weight = [None, None, None]

    def _slice_weight(self, weight: torch.Tensor):
        assert weight.shape[1] % self.tp_world_size_ == 0, f"tp slice error {weight.shape[1]} % {self.tp_world_size_}"
        tp_size = weight.shape[1] // self.tp_world_size_
        return weight[:, tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_bias(self, bias):
        assert bias.shape[1] % self.tp_world_size_ == 0, f"tp slice error {bias.shape[1]} % {self.tp_world_size_}"
        tp_size = bias.shape[1] // self.tp_world_size_
        return bias[:, tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        tp_size = weight_scale.shape[1] // self.tp_world_size_
        scale_start = tp_size * self.tp_rank_
        scale_end = tp_size * (self.tp_rank_ + 1)
        return weight_scale[:, scale_start:scale_end].to(torch.half)

    def _slice_weight_zero_point(self, weight_zero_point: torch.Tensor):
        tp_size = weight_zero_point.shape[1] // self.tp_world_size_
        zero_point_start = tp_size * self.tp_rank_
        zero_point_end = tp_size * (self.tp_rank_ + 1)
        return weight_zero_point[:, zero_point_start:zero_point_end]


class AWQMultiROWMMWeight(AWQMultiMMWeightTpl):
    _slice_weight = AWQROWMMWeight._slice_weight
    _slice_bias = AWQROWMMWeight._slice_bias
    _slice_weight_scale = AWQROWMMWeight._slice_weight_scale
    _slice_weight_zero_point = AWQROWMMWeight._slice_weight_zero_point

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


ROWBMM_WEIGHT_CLS_MAP = {
    "deepgemm-fp8w8a8-b128": W8A8B128ROWMMWeight,
    "awq": AWQROWMMWeight,
    "awq_marlin": AWQMARLINROWMMWeight,
}

MULTI_ROWBMM_WEIGHT_CLS_MAP = {
    "deepgemm-fp8w8a8-b128": W8A8B128MultiROWMMWeight,
    "awq": AWQMultiROWMMWeight,
    "awq_marlin": AWQMARLINMultiROWMMWeight,
}
