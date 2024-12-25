import torch
from abc import ABC, abstractmethod
from lightllm.utils.device_utils import get_current_device_id

class QuantizationMethod(ABC):
    def __init__(self):
        super().__init__()
        self.device_id_ = get_current_device_id()

    @abstractmethod
    def quantize(self, weights: torch.Tensor):
        pass

    @abstractmethod
    def apply(self, input_tensor, weight, bias=None, out=None, use_custom_tensor_mananger=True):
        pass
