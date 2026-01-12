import torch
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size


class SliceMixinBase(ABC):
    """切片操作的Mixin基类"""

    def __init__(self, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1):
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        # this param is used to slice the weight when tp_world_size_ is divisible by the kv_head_num
        # for example, if tp_world_size_ is 8 and kv_head_num is 4, then repeat_times_ is 2
        self.repeat_times_ = repeat_times

    @abstractmethod
    def _slice_weight(self, weight: torch.Tensor):
        pass

    @abstractmethod
    def _slice_bias(self, bias):
        pass

    def _get_slice_start_end(self, size: int) -> Tuple[int, int]:
        tp_size = size * self.repeat_times_ // self.tp_world_size_
        start = tp_size * (self.tp_rank_ // self.repeat_times_)
        end = start + tp_size
        return start, end


class SliceMixinTpl(SliceMixinBase):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1):
        super().__init__(tp_rank, tp_world_size, repeat_times)

    def _slice_weight(self, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("slice_weight must implement this method")

    def _slice_bias(self, bias) -> torch.Tensor:
        raise NotImplementedError("slice_bias must implement this method")

    def _slice_weight_scale(self, weight_scale: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("slice_weight_scale must implement this method")

    def _slice_weight_zero_point(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("slice_weight_zero_point must implement this method")


# 默认weight 的shape是 outxin，这也是目前最通用的约定。
# 所以row-wise是沿着dim=0进行切分，col-wise是沿着dim=1进行切分。
class RowSliceMixin(SliceMixinTpl):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1):
        super().__init__(tp_rank, tp_world_size, repeat_times)

    def _slice_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert (
            weight.shape[0] * self.repeat_times_ % self.tp_world_size_ == 0
        ), f"tp slice error {weight.shape[0] * self.repeat_times_} % {self.tp_world_size_}"
        start, end = self._get_slice_start_end(weight.shape[0])
        return weight[start:end, :]

    def _slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        assert (
            bias.shape[0] * self.repeat_times_ % self.tp_world_size_ == 0
        ), f"tp slice error {bias.shape[0] * self.repeat_times_} % {self.tp_world_size_}"
        start, end = self._get_slice_start_end(bias.shape[0])
        return bias[start:end]


# 量化切片默认实现方式是group-wise的量化，所以weight_scale 和weight_zero_point ndims跟weight一样。
# 后续按需要，扩展per-tensor、per-channel的量化方式。
class QuantizedRowSliceMixin(RowSliceMixin):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1):
        super().__init__(tp_rank, tp_world_size, repeat_times)

    def _slice_weight_scale(self, weight_scale: torch.Tensor) -> torch.Tensor:
        assert (
            weight_scale.shape[0] % self.tp_world_size_ == 0
        ), f"tp slice error {weight_scale.shape[0]} % {self.tp_world_size_}"
        start, end = self._get_slice_start_end(weight_scale.shape[0])
        return weight_scale[start:end]

    def _slice_weight_zero_point(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        assert (
            weight_zero_point.shape[0] % self.tp_world_size_ == 0
        ), f"tp slice error {weight_zero_point.shape[0]} % {self.tp_world_size_}"
        start, end = self._get_slice_start_end(weight_zero_point.shape[0])
        return weight_zero_point[start:end]


class ColSliceMixin(SliceMixinTpl):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1):
        super().__init__(tp_rank, tp_world_size, repeat_times)

    def _slice_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert (
            weight.shape[1] * self.repeat_times_ % self.tp_world_size_ == 0
        ), f"tp slice error {weight.shape[1] * self.repeat_times_ } % {self.tp_world_size_}"
        start, end = self._get_slice_start_end(weight.shape[1])
        return weight[:, start:end]

    def _slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        return bias / self.tp_world_size_ * self.repeat_times_


class QuantizedColSliceMixin(ColSliceMixin):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1):
        super().__init__(tp_rank, tp_world_size, repeat_times)

    def _slice_weight_scale(self, weight_scale: torch.Tensor) -> torch.Tensor:
        assert (
            weight_scale.shape[1] * self.repeat_times_ % self.tp_world_size_ == 0
        ), f"tp slice error {weight_scale.shape[1] * self.repeat_times_ } % {self.tp_world_size_}"
        start, end = self._get_slice_start_end(weight_scale.shape[1])
        return weight_scale[:, start:end]

    def _slice_weight_zero_point(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        assert (
            weight_zero_point.shape[1] * self.repeat_times_ % self.tp_world_size_ == 0
        ), f"tp slice error {weight_zero_point.shape[1] * self.repeat_times_ } % {self.tp_world_size_}"
        start, end = self._get_slice_start_end(weight_zero_point.shape[1])
        return weight_zero_point[:, start:end]


# awq 的量化权重是inxout存储格式，需要定制实现。
class AwqQuantizedRowSliceMixin(QuantizedColSliceMixin):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1):
        super().__init__(tp_rank, tp_world_size, repeat_times)

    def _slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        assert (
            bias.shape[0] * self.repeat_times_ % self.tp_world_size_ == 0
        ), f"tp slice error {bias.shape[0] * self.repeat_times_ } % {self.tp_world_size_}"
        start, end = self._get_slice_start_end(bias.shape[0])
        return bias[start:end]


class AwqQuantizedColSliceMixin(QuantizedRowSliceMixin):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1):
        super().__init__(tp_rank, tp_world_size, repeat_times)

    def _slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        return bias / self.tp_world_size_ * self.repeat_times_


def get_row_slice_mixin(
    quant_method_name: str, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1
) -> SliceMixinTpl:
    if quant_method_name.startswith("awq"):
        return AwqQuantizedRowSliceMixin(tp_rank, tp_world_size, repeat_times)
    elif quant_method_name == "none":
        return RowSliceMixin(tp_rank, tp_world_size, repeat_times)
    else:
        return QuantizedRowSliceMixin(tp_rank, tp_world_size, repeat_times)


def get_col_slice_mixin(
    quant_method_name: str, tp_rank: int = None, tp_world_size: int = None, repeat_times: int = 1
) -> SliceMixinTpl:
    if quant_method_name.startswith("awq"):
        return AwqQuantizedColSliceMixin(tp_rank, tp_world_size, repeat_times)
    elif quant_method_name == "none":
        return ColSliceMixin(tp_rank, tp_world_size, repeat_times)
    else:
        return QuantizedColSliceMixin(tp_rank, tp_world_size, repeat_times)
