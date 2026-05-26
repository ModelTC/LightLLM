import ctypes
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from lightllm.utils.kv_cache_utils import attach_shm_kv_cache_ptr, create_shm_kv_cache_ptr, register_shm_ptr_to_pin


@dataclass(frozen=True)
class CpuCacheTensorSpec:
    shm_key: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int


class CpuCacheCreator:
    def __init__(self, tensor_spec: CpuCacheTensorSpec):
        self.tensor_spec = tensor_spec

    def create_or_attach(self, init_shm_data: bool, pin: bool) -> torch.Tensor:
        if init_shm_data:
            shm_ptr = create_shm_kv_cache_ptr(key=self.tensor_spec.shm_key, size=self.tensor_spec.size_bytes)
        else:
            shm_ptr = attach_shm_kv_cache_ptr(key=self.tensor_spec.shm_key, size=self.tensor_spec.size_bytes)

        if pin:
            device_ptr = register_shm_ptr_to_pin(shm_ptr=shm_ptr, size=self.tensor_spec.size_bytes)
            cpu_cache_tensor = self._build_tensor_view(shm_ptr=device_ptr)
            assert device_ptr == cpu_cache_tensor.data_ptr()
        else:
            cpu_cache_tensor = self._build_tensor_view(shm_ptr=shm_ptr)
            assert shm_ptr == cpu_cache_tensor.data_ptr()

        return cpu_cache_tensor

    def _build_tensor_view(self, shm_ptr: int) -> torch.Tensor:
        numpy_array = np.frombuffer(
            memoryview((ctypes.c_uint8 * self.tensor_spec.size_bytes).from_address(shm_ptr)),
            dtype=np.uint8,
        )
        return torch.from_numpy(numpy_array).view(dtype=self.tensor_spec.dtype).view(self.tensor_spec.shape)
