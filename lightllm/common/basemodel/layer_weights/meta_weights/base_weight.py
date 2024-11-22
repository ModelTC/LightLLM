import torch
from abc import ABC, abstractmethod
from lightllm.utils.dist_utils import get_world_size, get_rank


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
    def __init__(self):
        self.world_size_ = get_world_size()
        self.tp_rank_ = get_rank()

    def load_hf_weights(self, weights):
        pass

    def verify_load(self):
        pass
