import torch
import threading
import collections
from typing import List, Dict


class PinMemTensorManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.key_to_tensor_list: Dict[str, List[torch.Tensor]] = collections.defaultdict(list)
        self.key_to_alloc_index: Dict[str, int] = {}
        self.buffer_size = 4

    def alloc_pin_tensor(self, key: str, size: int, dtype: torch.dtype) -> torch.Tensor:
        """
        利用 buffer_size buffer的 pin mem的cache，加速对pin mem的申请和释放操作。
        """

        with self.lock:
            if key not in self.key_to_tensor_list:
                self.key_to_tensor_list[key] = [
                    torch.empty(size=(max(2048, size),), dtype=dtype, device="cpu", pin_memory=True)
                    for _ in range(self.buffer_size)
                ]
                self.key_to_alloc_index[key] = 0

            alloc_index = self.key_to_alloc_index[key]
            buff_tensor = self.key_to_tensor_list[key][alloc_index]
            if buff_tensor.numel() < size:
                self.key_to_tensor_list[key][alloc_index] = torch.empty(
                    size=(size,), dtype=dtype, device="cpu", pin_memory=True
                )
                buff_tensor = self.key_to_tensor_list[key][alloc_index]
            self.key_to_alloc_index[key] = (alloc_index + 1) % self.buffer_size
            return buff_tensor[0:size]

    def gen_from_list(self, key: str, data: List, dtype: torch.dtype) -> torch.Tensor:
        size = len(data)
        pin_mem = self.alloc_pin_tensor(key, size=size, dtype=dtype)
        pin_mem.numpy()[:] = data
        return pin_mem

    def async_copy_from_gpu_tensor(self, key: str, gpu_tensor: torch.Tensor) -> torch.Tensor:
        size = gpu_tensor.numel()
        pin_mem = self.alloc_pin_tensor(key, size=size, dtype=gpu_tensor.dtype)
        pin_mem.copy_(gpu_tensor.view(-1), non_blocking=True)
        return pin_mem.view(gpu_tensor.shape)


g_pin_mem_manager = PinMemTensorManager()
