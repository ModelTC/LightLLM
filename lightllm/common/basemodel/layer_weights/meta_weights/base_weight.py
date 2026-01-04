import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp, get_current_device_id


class BaseWeight(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_hf_weights(self, weights):
        pass

    @abstractmethod
    def verify_load(self) -> bool:
        pass


class BaseWeightTpl(BaseWeight):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, data_type: torch.dtype = None):
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.device_id_ = get_current_device_id()
        self.data_type_ = data_type

    def load_hf_weights(self, weights):
        raise NotImplementedError("load_hf_weights must implement this method")

    def verify_load(self) -> bool:
        raise NotImplementedError("verify_load must implement this method")

    def _get_head_tp_split_params(self, weight: torch.Tensor) -> Tuple[int, int]:
        """
        Docstring for _get_head_tp_split_params,
        一个常用的tp 划分head获取head_index 范围的功能函数, 一些继承类可能会使用。
        :param self: Description
        :param weight: Description
        :type weight: torch.Tensor
        :return: Description
        :rtype: Tuple[int, int]
        """
        assert weight.ndim == 2

        all_head_num = weight.shape[0]
        tp_head_num = all_head_num // self.tp_world_size_

        if tp_head_num > 0:
            start_head_index = self.tp_rank_ * tp_head_num
            end_head_index = (self.tp_rank_ + 1) * tp_head_num
        else:
            # 当 tp_world_size 大于 all_head_num 时的特殊处理
            scale_size = self.tp_world_size_ // all_head_num
            assert self.tp_world_size_ % all_head_num == 0
            start_head_index = self.tp_rank_ // scale_size
            end_head_index = start_head_index + 1

        return start_head_index, end_head_index
