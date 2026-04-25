import torch
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..mem_manager import MemoryManager
    from lightllm.server.router.model_infer.infer_batch import InferReq

# 定义一个抽象基类
class BaseMemManagerOperator(ABC):
    def __init__(self, mem_manager: "MemoryManager") -> None:
        super().__init__()
        self.mem_manager = mem_manager

    @abstractmethod
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        pass

    # cpu cache 的相关操作接口
    def load_cpu_cache_to_gpu(
        self, mem_indexes: torch.Tensor, page_indexes: torch.Tensor, cpu_cache_client, req: "InferReq"
    ):
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
        mem_managers: List["MemoryManager"],
        move_token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        mem_indexes: torch.Tensor,
        dp_size_in_node: int,
        rank_in_dp: int,
    ):
        raise NotImplementedError()
