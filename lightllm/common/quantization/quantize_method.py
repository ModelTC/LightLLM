import torch
from abc import ABC, abstractmethod


class QuantizationMethod(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def quantize(self, weights: torch.Tensor):
        pass

    @abstractmethod
    def apply(self, input_tensor, weight, bias=None, out=None):
        pass
