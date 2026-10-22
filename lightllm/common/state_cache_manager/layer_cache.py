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
        shm_name: str = None,
        shm_create: bool = False,
    ):
        self.size_first = size_first
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.layer_num = layer_num
        self.device = device
        if shm_name is not None:
            assert device == "cpu" and size_first
            self.init_share_memory(shm_name, size, shm_create)
        elif not self.size_first:
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

    def init_share_memory(self, name: str, size: int, create: bool):
        """Service-owned, CUDA-mapped CPU storage, attached separately in each DP process."""
        import ctypes
        from lightllm.utils.shm_utils import create_or_link_shm
        from lightllm.utils.kv_cache_utils import register_shm_ptr_to_pin

        size_bytes = int(size * self.get_cell_size())
        self.shm = create_or_link_shm(name, size_bytes, force_mode="create" if create else "link")
        host_ptr = ctypes.addressof(ctypes.c_uint8.from_buffer(self.shm.buf))
        # CUDA memcpy cannot span independently registered host allocations.
        mapped_ptr = register_shm_ptr_to_pin(host_ptr, size_bytes, registration_alignment=int(self.get_cell_size()))
        array = np.frombuffer((ctypes.c_uint8 * size_bytes).from_address(mapped_ptr), dtype=np.uint8)
        self.buffer = torch.from_numpy(array).view(self.dtype).view(size, self.layer_num, *self.shape)
        self.size = size
        if create:
            self.buffer.zero_()
