import torch
from abc import ABC, abstractmethod
from typing import Dict
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp, get_current_device_id


class BaseWeight(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_hf_weights(self, weights):
        pass

    @abstractmethod
    def verify_load(self):
        pass


class BaseWeightTpl(BaseWeight):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, data_type: torch.dtype = None):
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.device_id_ = get_current_device_id()
        self.data_type_ = data_type

    def _slice_weight(self, weight: torch.Tensor):
        # slice weight
        return weight.to(self.data_type_)

    def _slice_bias(self, bias: torch.Tensor):
        # slice bias
        return bias.to(self.data_type_)

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        # slice weight scale and zero point
        return weight_scale

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        # load weight
        pass

    def _load_scales(self, weights: Dict[str, torch.Tensor]) -> None:
        # load quantization scale
        pass

    def load_hf_weights(self, weights):
        self._load_weights(weights)
        self._load_scales(weights)
        return

    def verify_load(self):
        pass
