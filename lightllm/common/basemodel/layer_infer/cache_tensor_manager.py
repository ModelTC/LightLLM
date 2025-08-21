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

    # cuda graph cache 中，对于输出 tensor （申请参数中is_graph_out = True），实现跨不同
    # batch size 的 cuda graph 的tensor共享。对于 非输出 tensor， 退化为使用 torch.empty 进行显存的分配。
    class CudaGraphCacheTensorManager:
        def __init__(self, cuda_graph_max_batch_size: int):
            self.cuda_graph_max_batch_size = cuda_graph_max_batch_size
            # Dict[graph_out_key, Dict[microbatch_index, tensor_chache]]
            self.graph_out_tensor_dict: Dict[int, Dict[int, torch.Tensor]] = collections.defaultdict(dict)
            self.managed_total_tensor_bytes = 0
            return

        def alloc_tensor_for_cuda_graph(
            self,
            cur_batch_size: int,
            shape: Union[torch.Size, Iterable[int]],
            data_type: torch.dtype,
            device: str = "cuda",
            is_graph_out: bool = False,
            microbatch_index: int = 0,
            graph_out_key: int = 0,
        ) -> torch.Tensor:
            assert microbatch_index in [0, 1]
            if not is_graph_out:
                return torch.empty(shape, dtype=data_type, device=device)

            size = np.prod(shape)
            assert size % cur_batch_size == 0
            max_size = size // cur_batch_size * self.cuda_graph_max_batch_size

            # graph out tensor, 只有一个， 不需要进行引用计数管理
            microbatch_index_to_tensor_cache = self.graph_out_tensor_dict[graph_out_key]

            if microbatch_index not in microbatch_index_to_tensor_cache:
                graph_out_tensor = torch.empty((max_size,), dtype=data_type, device=device, requires_grad=False)
                logger.info(f"pid {os.getpid()} cuda graph alloc graph out mem {shape} {data_type} {size} {max_size}")
                self.managed_total_tensor_bytes += graph_out_tensor.element_size() * graph_out_tensor.numel()
                logger.info(f"cuda graph managed_total_tensor_bytes: {self.managed_total_tensor_bytes}")
                microbatch_index_to_tensor_cache[microbatch_index] = graph_out_tensor

            return self.graph_out_tensor_dict[graph_out_key][microbatch_index][0:size].view(shape)

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
            self.managed_total_tensor_bytes = 0
            # 防止误用导致显存泄露，添加标记变量。
            # 当使用者没有合法的调用 cache_env_in 和 cache_env_out 的时候
            # 如果调用了alloc_tensor 接口，则退化为 torch.empty 申请方式。
            self.cache_env_ok = False

        def cache_env_in(
            self, is_cuda_graph: bool = False, cur_batch_size: int = 0, cuda_graph_max_batch_size: int = 0
        ):
            self.managed_total_tensor_bytes = 0
            setattr(torch.Tensor, "__del__", custom_del)
            self.is_cuda_graph = is_cuda_graph
            if self.is_cuda_graph:
                if self.inner_cuda_graph_manager is None:
                    self.inner_cuda_graph_manager = CudaGraphCacheTensorManager(cuda_graph_max_batch_size)
                else:
                    assert self.inner_cuda_graph_manager.cuda_graph_max_batch_size == cuda_graph_max_batch_size
                self.cuda_graph_cur_batch_size = cur_batch_size
                assert cur_batch_size != 0
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
            is_graph_out: bool = False,
            microbatch_index: int = 0,
        ) -> torch.Tensor:
            return self.alloc_tensor(
                shape=shape,
                data_type=dtype,
                device=device,
                is_graph_out=is_graph_out,
                microbatch_index=microbatch_index,
            )

        def alloc_tensor(
            self,
            shape: Union[torch.Size, Tuple[int, ...]],
            data_type: torch.dtype,
            device: str = "cuda",
            is_graph_out: bool = False,
            microbatch_index: int = 0,
            graph_out_key: int = 0,
        ) -> torch.Tensor:
            # shape 类型转换
            if isinstance(shape, list):
                shape = torch.Size(shape)

            # cache manager 没有被正常使用时
            if not self.cache_env_ok:
                return torch.empty(shape, dtype=data_type, device=device, requires_grad=False)

            # 是 cuda graph的时候，由cuda graph manager 接管
            if self.is_cuda_graph:
                return self.inner_cuda_graph_manager.alloc_tensor_for_cuda_graph(
                    self.cuda_graph_cur_batch_size,
                    shape,
                    data_type,
                    device,
                    is_graph_out,
                    microbatch_index,
                    graph_out_key,
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

        def cache_env_in(
            self, is_cuda_graph: bool = False, cur_batch_size: int = 0, cuda_graph_max_batch_size: int = 0
        ):
            return

        def cache_env_out(self):
            return

        def empty(
            self,
            shape: Union[torch.Size, Iterable[int]],
            dtype: torch.dtype,
            device: str = "cuda",
            is_graph_out: bool = False,
            microbatch_index: int = 0,
        ) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

        def alloc_tensor(
            self,
            shape: Union[torch.Size, Iterable[int]],
            data_type: torch.dtype,
            device: str = "cuda",
            is_graph_out: bool = False,
            microbatch_index: int = 0,
            graph_out_key: int = 0,
        ) -> torch.Tensor:
            return torch.empty(shape, dtype=data_type, device=device, requires_grad=False)


global g_cache_manager
g_cache_manager = CacheTensorManager()
