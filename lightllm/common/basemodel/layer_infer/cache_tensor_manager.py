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

    # 实现 batch 1 到 cuda_graph_max_batch_size 共享内部torch.Tensor的特制cache manager.
    # cuda graph cache 可以不实现得那么高效，因为只会捕获一次，后续都是在 graph 图中run。
    class CudaGraphCacheTensorManager:
        def __init__(self, cuda_graph_max_batch_size: int):
            self.bufnodes: List[BufNode] = []
            self.cuda_graph_max_batch_size = cuda_graph_max_batch_size
            from torch._C import _storage_Use_Count as use_count

            self.graph_out_tensor: torch.Tensor = None

            self.use_count = use_count
            return

        def alloc_tensor_for_cuda_graph(
            self,
            cur_batch_size: int,
            shape: Union[torch.Size, Iterable[int]],
            data_type: torch.dtype,
            device: str = "cuda",
            is_graph_out: bool = False,
        ) -> torch.Tensor:
            """
            cuda graph 对应的分配需要先将shape扩大到最大batch_size 对应的shape, 然后再slice回来。
            """
            size = np.prod(shape)
            assert size % cur_batch_size == 0
            max_size = size // cur_batch_size * self.cuda_graph_max_batch_size
            key = (max_size, data_type)
            # graph out tensor, 只有一个， 不需要进行引用计数管理
            if is_graph_out:
                if self.graph_out_tensor is None:
                    self.graph_out_tensor = torch.empty(
                        (max_size,), dtype=data_type, device=device, requires_grad=False
                    )
                    logger.info(
                        f"pid {os.getpid()} cuda graph alloc graph out mem {shape} {data_type} {size} {max_size}"
                    )
                return self.graph_out_tensor[0:size].view(shape)

            for buf_node in self.bufnodes:
                if buf_node.shape_key == key and self.use_count(buf_node.storage_weak_ptr) == 1:
                    ans = buf_node.inner_tensor[0:size].view(shape)
                    return ans

            buf_tensor = torch.empty((max_size,), dtype=data_type, device=device, requires_grad=False)
            logger.info(f"pid {os.getpid()} cuda graph alloc mem {shape} {data_type} {size} {max_size}")
            storage_weak_ptr = buf_tensor.untyped_storage()._weak_ref()
            buf_node = BufNode(buf_tensor, key, storage_weak_ptr)
            self.bufnodes.append(buf_node)
            ans = buf_node.inner_tensor[0:size].view(shape)
            return ans

    class CacheTensorManager:
        def __init__(self):
            self.ptr_to_bufnode: Dict[int, BufNode] = {}
            self.free_shape_dtype_to_bufs: Dict[Tuple, List[BufNode]] = collections.defaultdict(list)
            self.calcu_shape_cache: Dict[torch.Size, int] = {}
            self.changed_ptr: Set[int] = set()
            from torch._C import _storage_Use_Count as use_count

            # use_count 函数可以用于获取有多少 tensor 真正引用了这片显存 tensor
            self.use_count = use_count
            self.inner_cuda_graph_manager = None
            self.cuda_graph_cur_batch_size = None
            self.is_cuda_graph = False

        def cache_env_in(
            self, is_cuda_graph: bool = False, cur_batch_size: int = 0, cuda_graph_max_batch_size: int = 0
        ):
            setattr(torch.Tensor, "__del__", custom_del)
            self.is_cuda_graph = is_cuda_graph
            if self.is_cuda_graph:
                if self.inner_cuda_graph_manager is None:
                    self.inner_cuda_graph_manager = CudaGraphCacheTensorManager(cuda_graph_max_batch_size)
                else:
                    assert self.inner_cuda_graph_manager.cuda_graph_max_batch_size == cuda_graph_max_batch_size
                self.cuda_graph_cur_batch_size = cur_batch_size
                assert cur_batch_size != 0
            return

        def cache_env_out(self):
            delattr(torch.Tensor, "__del__")
            self.ptr_to_bufnode.clear()
            self.free_shape_dtype_to_bufs.clear()
            self.calcu_shape_cache.clear()
            self.changed_ptr.clear()
            return

        def alloc_tensor(
            self,
            shape: Union[torch.Size, Tuple[int, ...]],
            data_type: torch.dtype,
            device: str = "cuda",
            is_graph_out: bool = False,
        ) -> torch.Tensor:
            # shape 类型转换
            if isinstance(shape, list):
                shape = torch.Size(shape)
            # 是 cuda graph的时候，由cuda graph manager 接管
            if self.is_cuda_graph:
                return self.inner_cuda_graph_manager.alloc_tensor_for_cuda_graph(
                    self.cuda_graph_cur_batch_size, shape, data_type, device, is_graph_out
                )

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

        def cache_env_in(
            self, is_cuda_graph: bool = False, cur_batch_size: int = 0, cuda_graph_max_batch_size: int = 0
        ):
            return

        def cache_env_out(self):
            return

        def alloc_tensor(
            self,
            shape: Union[torch.Size, Iterable[int]],
            data_type: torch.dtype,
            device: str = "cuda",
            is_graph_out: bool = False,
        ) -> torch.Tensor:
            return torch.empty(shape, dtype=data_type, device=device, requires_grad=False)


global g_cache_manager
g_cache_manager = CacheTensorManager()
