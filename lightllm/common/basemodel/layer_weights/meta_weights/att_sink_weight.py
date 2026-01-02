import torch
from typing import Dict
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id


class TpAttSinkWeight(BaseWeightTpl):
    def __init__(self, weight_name: str, data_type, tp_head_num: int):
        super().__init__()
        self.weight_name = weight_name
        self.data_type_ = data_type
        self.weight: torch.Tensor = None
        self.tp_head_num = tp_head_num
        assert self.tp_head_num > 0

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        start_head_index = self.tp_head_num * self.tp_rank_
        end_head_index = self.tp_head_num * (self.tp_rank_ + 1)

        if self.weight_name in weights:
            self.weight = (
                weights[self.weight_name][start_head_index:end_head_index]
                .to(self.data_type_)
                .cuda(get_current_device_id())
            )
