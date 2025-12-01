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
