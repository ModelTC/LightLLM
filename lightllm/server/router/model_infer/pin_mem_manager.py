import torch
import threading
import collections
from typing import List, Dict


class PinMemTensorManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.key_to_tensor_list: Dict[str, List[torch.Tensor]] = collections.defaultdict(list)
        self.key_to_alloc_index: Dict[str, int] = {}

    def alloc_pin_tensor(self, key: str, size: int, dtype: torch.dtype):
        """
        利用4 buffer的 pin mem的cache，加速对pin mem的申请和释放操作。
        """
        with self.lock:
            if key not in self.key_to_tensor_list:
                self.key_to_tensor_list[key].append(
                    torch.empty(size=(size,), dtype=dtype, device="cpu", pin_memory=True)
                )
                self.key_to_tensor_list[key].append(
                    torch.empty(size=(size,), dtype=dtype, device="cpu", pin_memory=True)
                )
                self.key_to_tensor_list[key].append(
                    torch.empty(size=(size,), dtype=dtype, device="cpu", pin_memory=True)
                )
                self.key_to_tensor_list[key].append(
                    torch.empty(size=(size,), dtype=dtype, device="cpu", pin_memory=True)
                )
                self.key_to_alloc_index[key] = 0

            alloc_index = self.key_to_alloc_index[key]
            buff_tensor = self.key_to_tensor_list[key][alloc_index]
            if buff_tensor.numel() < size:
                self.key_to_tensor_list[key][alloc_index] = torch.empty(
                    size=(size,), dtype=dtype, device="cpu", pin_memory=True
                )
                buff_tensor = self.key_to_tensor_list[key][alloc_index]
            self.key_to_alloc_index[key] = (alloc_index + 1) % 4
            return buff_tensor[0:size]


g_pin_mem_manager = PinMemTensorManager()
