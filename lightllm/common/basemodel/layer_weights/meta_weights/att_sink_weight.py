import torch
from typing import Dict
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id


class TpAttSinkWeight(BaseWeightTpl):
    def __init__(self, weight_name: str, data_type):
        super().__init__()
        self.weight_name = weight_name
        self.data_type_ = data_type
        self.weight: torch.Tensor = None

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name not in weights or self.weight is not None:
            return

        t_weight = weights[self.weight_name]
        all_head_num = t_weight.shape[0]
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

        self.weight = (
            weights[self.weight_name][start_head_index:end_head_index].to(self.data_type_).cuda(get_current_device_id())
        )
