import torch
from typing import Optional
from abc import ABC, abstractmethod
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size


class SliceMixinBase(ABC):
    """切片操作的Mixin基类"""

    def __init__(self, tp_rank: int = None, tp_world_size: int = None, bias_div_world_size: bool = False):
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.bias_div_world_size_ = bias_div_world_size

    @abstractmethod
    def _slice_weight(self, weight: torch.Tensor):
        pass

    @abstractmethod
    def _slice_bias(self, bias):
        pass


class SliceMixinTpl(SliceMixinBase):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, bias_div_world_size: bool = False):
        super().__init__(tp_rank, tp_world_size, bias_div_world_size)

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
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, bias_div_world_size: bool = False):
        super().__init__(tp_rank, tp_world_size, bias_div_world_size)

    def _slice_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.shape[0] % self.tp_world_size_ == 0, f"tp slice error {weight.shape[0]} % {self.tp_world_size_}"
        tp_size = weight.shape[0] // self.tp_world_size_
        return weight[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_bias(self, bias) -> torch.Tensor:
        assert bias.shape[0] % self.tp_world_size_ == 0, f"tp slice error {bias.shape[0]} % {self.tp_world_size_}"
        tp_size = bias.shape[0] // self.tp_world_size_
        if self.bias_div_world_size_:
            return bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)] / self.tp_world_size_
        return bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]


# 量化切片默认实现方式是group-wise的量化，所以weight_scale 和weight_zero_point ndims跟weight一样。
# 后续按需要，扩展per-tensor、per-channel的量化方式。
class QuantizedRowSliceMixin(RowSliceMixin):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, bias_div_world_size: bool = False):
        super().__init__(tp_rank, tp_world_size, bias_div_world_size)

    def _slice_weight_scale(self, weight_scale: torch.Tensor) -> torch.Tensor:
        assert (
            weight_scale.shape[0] % self.tp_world_size_ == 0
        ), f"tp slice error {weight_scale.shape[0]} % {self.tp_world_size_}"
        tp_size = weight_scale.shape[0] // self.tp_world_size_
        scale_start = tp_size * self.tp_rank_
        scale_end = tp_size * (self.tp_rank_ + 1)
        return weight_scale[scale_start:scale_end]

    def _slice_weight_zero_point(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        assert (
            weight_zero_point.shape[0] % self.tp_world_size_ == 0
        ), f"tp slice error {weight_zero_point.shape[0]} % {self.tp_world_size_}"
        tp_size = weight_zero_point.shape[0] // self.tp_world_size_
        zero_point_start = tp_size * self.tp_rank_
        zero_point_end = tp_size * (self.tp_rank_ + 1)
        return weight_zero_point[zero_point_start:zero_point_end]


class ColSliceMixin(SliceMixinTpl):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, bias_div_world_size: bool = True):
        super().__init__(tp_rank, tp_world_size, bias_div_world_size)

    def _slice_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.shape[1] % self.tp_world_size_ == 0, f"tp slice error {weight.shape[1]} % {self.tp_world_size_}"
        tp_size = weight.shape[1] // self.tp_world_size_
        return weight[:, tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_bias(self, bias) -> torch.Tensor:
        assert bias.shape[0] % self.tp_world_size_ == 0, f"tp slice error {bias.shape[0]} % {self.tp_world_size_}"
        tp_size = bias.shape[0] // self.tp_world_size_
        if self.bias_div_world_size_:
            return bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)] / self.tp_world_size_
        return bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]


class QuantizedColSliceMixin(ColSliceMixin):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, bias_div_world_size: bool = True):
        super().__init__(tp_rank, tp_world_size, bias_div_world_size)

    def _slice_weight_scale(self, weight_scale: torch.Tensor) -> torch.Tensor:
        assert (
            weight_scale.shape[1] % self.tp_world_size_ == 0
        ), f"tp slice error {weight_scale.shape[1]} % {self.tp_world_size_}"
        tp_size = weight_scale.shape[1] // self.tp_world_size_
        scale_start = tp_size * self.tp_rank_
        scale_end = tp_size * (self.tp_rank_ + 1)
        return weight_scale[:, scale_start:scale_end]

    def _slice_weight_zero_point(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        assert (
            weight_zero_point.shape[1] % self.tp_world_size_ == 0
        ), f"tp slice error {weight_zero_point.shape[1]} % {self.tp_world_size_}"
        tp_size = weight_zero_point.shape[1] // self.tp_world_size_
        zero_point_start = tp_size * self.tp_rank_
        zero_point_end = tp_size * (self.tp_rank_ + 1)
        return weight_zero_point[:, zero_point_start:zero_point_end]
