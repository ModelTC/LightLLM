import torch
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..mem_manager import MemoryManager
    from lightllm.server.router.model_infer.infer_batch import InferReq


# 定义一个抽象基类
class BaseMemManagerOperator(ABC):
    supports_dp_cache_fetch = False

    def __init__(self, mem_manager: "MemoryManager") -> None:
        super().__init__()
        self.mem_manager = mem_manager

    @abstractmethod
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        pass

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        src_mem_index = src_mem_index.cuda(non_blocking=True)
        dst_mem_index = dst_mem_index.cuda(non_blocking=True)
        copy_kv_buffer_to_kv_buffer(src_mem_index, dst_mem_index, self.mem_manager.kv_buffer)
        if hasattr(self.mem_manager, "scale_buffer"):
            copy_kv_buffer_to_kv_buffer(src_mem_index, dst_mem_index, self.mem_manager.scale_buffer)

    # cpu cache 的相关操作接口
    def load_cpu_cache_to_gpu(
        self, mem_indexes: torch.Tensor, page_indexes: torch.Tensor, cpu_cache_client, req: "InferReq"
    ) -> None:
        raise NotImplementedError()

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        page_readies: torch.Tensor,
        cpu_cache_client,
        req: "InferReq",
    ):
        raise NotImplementedError()

    # dp 间共享 kv 的操作接口, 提升dp 模式下的kv 共享效率，降低调度的难度
    def copy_kv_from_other_dp_ranks(
        self,
        kv_views: List,
        move_token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        mem_indexes: torch.Tensor,
        dp_size_in_node: int,
        rank_in_dp: int,
    ):
        raise NotImplementedError()
