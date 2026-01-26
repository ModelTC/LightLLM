import torch
from typing import Dict, Tuple
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id


class TpAttSinkWeight(BaseWeightTpl):
    def __init__(self, all_kv_head_num: int, head_dim: int, weight_name: str, data_type):
        super().__init__()
        self.all_kv_head_num = all_kv_head_num
        self.head_dim = head_dim
        self.weight_name = weight_name
        self.data_type_ = data_type
        self._start_head_index, self._end_head_index = self._get_head_tp_split_params(all_head_num=self.all_kv_head_num)
        self._create_weight()

    def _create_weight(self):
        self.weight = torch.empty((self.all_kv_head_num, self.head_dim), dtype=self.data_type_, device="cuda")
        self.weight.load_ok = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name not in weights or self.weight is not None:
            return

        t_weight = weights[self.weight_name]
        self.weight = (
            t_weight[self._start_head_index : self._end_head_index].to(self.data_type_).cuda(get_current_device_id())
        )
        self.weight.load_ok = True

    def verify_load(self):
        return self.weight.load_ok

    def _get_head_tp_split_params(self, all_head_num: int) -> Tuple[int, int]:
        """
        Docstring for _get_head_tp_split_params,
        一个常用的tp 划分head获取head_index 范围的功能函数, 一些继承类可能会使用。
        :param self: Description
        :param weight: Description
        :type weight: torch.Tensor
        :return: Description
        :rtype: Tuple[int, int]
        """
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
