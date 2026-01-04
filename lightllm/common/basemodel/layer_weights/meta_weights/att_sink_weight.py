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
        start_head_index, end_head_index = self._get_head_tp_split_params(weight=t_weight)
        self.weight = t_weight[start_head_index:end_head_index].to(self.data_type_).cuda(get_current_device_id())

    def verify_load(self):
        return self.weight is not None
