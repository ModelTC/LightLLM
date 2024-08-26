import os
import torch
import collections
import dataclasses
import numpy as np
import torch._C
from typing import Dict, Iterable, Literal, Tuple, Union, List
from torch.storage import UntypedStorage
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_disable_gpu_tensor_cache = os.getenv("DISABLE_GPU_TENSOR_CACHE", None) is not None

if torch.__version__ >= "2.1.0" and (not _disable_gpu_tensor_cache):
    logger.info("USE_GPU_TENSOR_CACHE is On")

    @dataclasses.dataclass
    class BufNode:
        inner_tensor: torch.Tensor
        storage_weak_ptr: int

        def __del__(self):
            UntypedStorage._free_weak_ref(self.storage_weak_ptr)
            return

    class CacheTensorManager:
        def __init__(self):
            self.pool: Dict[Tuple, List[BufNode]] = collections.defaultdict(list)
            self.ready_tensor_list: List[Tuple[BufNode, Union[torch.Size, Iterable[int]]]] = []
            from torch._C import _storage_Use_Count as use_count

            # use_count 函数可以用于获取有多少 tensor 真正引用了这片显存 tensor
            self.use_count = use_count

        def mark_cache_alloc_start(self):
            self.alloc_tensor = self._pre_alloc_cache_tensor

        def mark_cache_alloc_end(self):
            self.pool.clear()
            self.alloc_tensor = self._ready_alloc_cache_tensor
            self.ready_index = 0
            self.ready_tensor_list = [
                buf_tensor.inner_tensor.view(shape) for (buf_tensor, shape) in self.ready_tensor_list
            ]
            self.ready_tensor_len = len(self.ready_tensor_list)

        def alloc_tensor(
            self, shape: Union[torch.Size, Iterable[int]], data_type: torch.dtype, device: str = "cuda"
        ) -> torch.Tensor:
            pass

        def _pre_alloc_cache_tensor(
            self, shape: Union[torch.Size, Iterable[int]], data_type: torch.dtype, device: str = "cuda"
        ) -> torch.Tensor:
            size = np.prod(shape)
            key = (size, data_type)
            buf_node_list = self.pool[key]
            for buf_node in buf_node_list:
                count = self.use_count(buf_node.storage_weak_ptr)
                if count == 1:
                    self.ready_tensor_list.append((buf_node, shape))
                    return buf_node.inner_tensor.view(shape)

            buf_tensor = torch.empty(shape, dtype=data_type, device=device, requires_grad=False)
            storage_weak_ptr = buf_tensor.untyped_storage()._weak_ref()
            new_buf_node = BufNode(buf_tensor, storage_weak_ptr)
            buf_node_list.append(new_buf_node)
            self.ready_tensor_list.append((new_buf_node, shape))
            # 这里必须用view(shape) 生成一个新的tensor, 否则会导致引用计数的判断条件不正常
            return buf_tensor.view(shape)

        def _ready_alloc_cache_tensor(
            self, shape: Union[torch.Size, Iterable[int]], data_type: torch.dtype, device: str = "cuda"
        ) -> torch.Tensor:
            alloc_tensor = self.ready_tensor_list[self.ready_index]
            self.ready_index = (self.ready_index + 1) % self.ready_tensor_len
            return alloc_tensor

        def release_all_caches(self):
            self.ready_tensor_list.clear()

else:
    logger.info("USE_GPU_TENSOR_CACHE is OFF")

    class CacheTensorManager:
        def __init__(self):
            pass

        def mark_cache_alloc_start(self):
            pass

        def mark_cache_alloc_end(self):
            pass

        def alloc_tensor(
            self, shape: Union[torch.Size, Iterable[int]], data_type: torch.dtype, device: str = "cuda"
        ) -> torch.Tensor:
            return torch.empty(shape, dtype=data_type, device=device, requires_grad=False)

        def release_all_caches(self):
            pass


g_cache_manager = CacheTensorManager()
