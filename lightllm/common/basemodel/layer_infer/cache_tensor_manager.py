import os
import torch
import collections
import dataclasses
import numpy as np
import torch._C
from typing import Dict, Iterable, Literal, Tuple, Union, List, Set
from torch.storage import UntypedStorage
from dataclasses import field
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_disable_gpu_tensor_cache = os.getenv("DISABLE_GPU_TENSOR_CACHE", None) is not None

if torch.__version__ >= "2.1.0" and (not _disable_gpu_tensor_cache):
    logger.info("USE_GPU_TENSOR_CACHE is On")

    # 用于进行引用计数调整和判断
    def custom_del(self: torch.Tensor):
        global g_cache_manager
        if hasattr(self, "storage_weak_ptr"):
            storage_weak_ptr = self.storage_weak_ptr
        else:
            storage_weak_ptr = self.untyped_storage()._weak_ref()
            UntypedStorage._free_weak_ref(storage_weak_ptr)
        if storage_weak_ptr in g_cache_manager.ptr_to_bufnode:
            g_cache_manager.changed_ptr.add(storage_weak_ptr)
        return

    @dataclasses.dataclass
    class BufNode:
        inner_tensor: torch.Tensor
        shape_key: Tuple[int, torch.dtype]
        storage_weak_ptr: int
        shape_to_tensor: Dict[Union[torch.Size, Iterable[int]], torch.Tensor] = field(default_factory=dict)

        def __del__(self):
            UntypedStorage._free_weak_ref(self.storage_weak_ptr)
            return

    class CacheTensorManager:
        def __init__(self):
            self.ptr_to_bufnode: Dict[int, BufNode] = {}
            self.free_shape_dtype_to_bufs: Dict[Tuple, List[BufNode]] = collections.defaultdict(list)
            self.calcu_shape_cache: Dict[torch.Size, int] = {}
            self.changed_ptr: Set[int] = set()
            from torch._C import _storage_Use_Count as use_count

            # use_count 函数可以用于获取有多少 tensor 真正引用了这片显存 tensor
            self.use_count = use_count
            self.managed_total_tensor_bytes = 0
            # 防止误用导致显存泄露，添加标记变量。
            # 当使用者没有合法的调用 cache_env_in 和 cache_env_out 的时候
            # 如果调用了alloc_tensor 接口，则退化为 torch.empty 申请方式。
            self.cache_env_ok = False

        def cache_env_in(self):
            self.managed_total_tensor_bytes = 0
            setattr(torch.Tensor, "__del__", custom_del)
            self.cache_env_ok = True
            return

        def cache_env_out(self):
            delattr(torch.Tensor, "__del__")
            self.ptr_to_bufnode.clear()
            self.free_shape_dtype_to_bufs.clear()
            self.calcu_shape_cache.clear()
            self.changed_ptr.clear()
            self.cache_env_ok = False
            return

        def empty(
            self,
            shape: Union[torch.Size, Iterable[int]],
            dtype: torch.dtype,
            device: str = "cuda",
        ) -> torch.Tensor:
            return self.alloc_tensor(
                shape=shape,
                data_type=dtype,
                device=device,
            )

        def alloc_tensor(
            self,
            shape: Union[torch.Size, Tuple[int, ...]],
            data_type: torch.dtype,
            device: str = "cuda",
        ) -> torch.Tensor:
            # shape 类型转换
            if isinstance(shape, list):
                shape = torch.Size(shape)

            # cache manager 没有被正常使用时
            if not self.cache_env_ok:
                return torch.empty(shape, dtype=data_type, device=device, requires_grad=False)

            # 回收可能消亡的 tensor
            for ptr in self.changed_ptr:
                t_buf_node = self.ptr_to_bufnode[ptr]
                if self.use_count(ptr) == 1 + len(t_buf_node.shape_to_tensor):
                    self.free_shape_dtype_to_bufs[t_buf_node.shape_key].append(t_buf_node)
            self.changed_ptr.clear()

            if shape not in self.calcu_shape_cache:
                size = np.prod(shape)
                self.calcu_shape_cache[shape] = size
            else:
                size = self.calcu_shape_cache[shape]

            key = (size, data_type)
            buf_node_list = self.free_shape_dtype_to_bufs[key]
            if buf_node_list:
                buf_node = buf_node_list.pop()
                if shape not in buf_node.shape_to_tensor:
                    mark_tensor = buf_node.inner_tensor.view(shape)
                    buf_node.shape_to_tensor[shape] = mark_tensor
                else:
                    mark_tensor = buf_node.shape_to_tensor[shape]
                ans = mark_tensor.data  # 返回一个新的引用, 否则引用计数会无法判断
                ans.storage_weak_ptr = buf_node.storage_weak_ptr
                return ans

            buf_tensor = torch.empty((size,), dtype=data_type, device=device, requires_grad=False)
            # 用于调试显存占用的重要日志
            # self.managed_total_tensor_bytes +=  buf_tensor.element_size() * buf_tensor.numel()
            # logger.info(f"gpu cache managed_total_tensor_bytes: {self.managed_total_tensor_bytes}")
            storage_weak_ptr = buf_tensor.untyped_storage()._weak_ref()
            buf_node = BufNode(buf_tensor, key, storage_weak_ptr)
            self.ptr_to_bufnode[storage_weak_ptr] = buf_node
            if shape not in buf_node.shape_to_tensor:
                buf_node.shape_to_tensor[shape] = buf_node.inner_tensor.view(shape)
            mark_tensor = buf_node.shape_to_tensor[shape]
            ans = mark_tensor.data  # 返回一个新的引用, 否则引用计数会无法判断
            ans.storage_weak_ptr = buf_node.storage_weak_ptr
            return ans

else:
    logger.info("USE_GPU_TENSOR_CACHE is OFF")

    class CacheTensorManager:
        def __init__(self):
            pass

        def cache_env_in(self):
            return

        def cache_env_out(self):
            return

        def empty(
            self,
            shape: Union[torch.Size, Iterable[int]],
            dtype: torch.dtype,
            device: str = "cuda",
        ) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

        def alloc_tensor(
            self,
            shape: Union[torch.Size, Iterable[int]],
            data_type: torch.dtype,
            device: str = "cuda",
        ) -> torch.Tensor:
            return torch.empty(shape, dtype=data_type, device=device, requires_grad=False)


global g_cache_manager
g_cache_manager = CacheTensorManager()
