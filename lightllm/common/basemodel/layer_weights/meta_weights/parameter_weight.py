import torch
from typing import Dict
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id


class ParameterWeight(BaseWeightTpl):
    def __init__(self, weight_name: str, data_type: torch.dtype, bias_name: str = None):
        super().__init__()
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.data_type_ = data_type
        self.weight = None
        self.bias = None

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_name in weights:
            self.weight = weights[self.weight_name].to(self.data_type_).cuda(get_current_device_id())
        if self.bias_name in weights:
            self.bias = weights[self.bias_name].to(self.data_type_).cuda(get_current_device_id())

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.bias_name is not None:
            load_ok = load_ok and self.bias is not None
        return load_ok


class TpParameterWeight(ParameterWeight):
    def __init__(self, weight_name: str, data_type: torch.dtype, split_n_embed: int, bias_name: str = None):
        super().__init__(weight_name, data_type, bias_name)
        self.split_n_embed = split_n_embed

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        start = self.split_n_embed * self.tp_rank_
        end = self.split_n_embed * (self.tp_rank_ + 1)

        if self.weight_name in weights:
            self.weight = weights[self.weight_name][start:end].to(self.data_type_).cuda(get_current_device_id())
        if self.bias_name in weights:
            self.bias = weights[self.bias_name][start:end].to(self.data_type_).cuda(get_current_device_id())
