from typing import Tuple
import torch
import numpy as np
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class LayerCache:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        shape: Tuple[int, ...],
        layer_num: int,
        device: torch.device,
        size_first: bool = False,
    ):
        self.size_first = size_first
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.layer_num = layer_num
        self.device = device
        if not self.size_first:
            if device == "cpu":
                self.buffer = torch.zeros((self.layer_num, size, *shape), dtype=dtype, device="cpu", pin_memory=True)
            else:
                self.buffer = torch.zeros((self.layer_num, size, *shape), dtype=dtype, device=device)
        else:
            if device == "cpu":
                self.buffer = torch.zeros((size, self.layer_num, *shape), dtype=dtype, device="cpu", pin_memory=True)
            else:
                self.buffer = torch.zeros((size, self.layer_num, *shape), dtype=dtype, device=device)

    def get_cell_size(self):
        return np.prod(self.shape) * self.layer_num * torch._utils._element_size(self.dtype)
