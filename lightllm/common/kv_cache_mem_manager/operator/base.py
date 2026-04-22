import torch
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..mem_manager import MemoryManager

# 定义一个抽象基类
class BaseMemManagerOperator(ABC):
    def __init__(self, mem_manager: "MemoryManager") -> None:
        super().__init__()
        self.mem_manager = mem_manager

    # cpu cache 的相关操作接口

    @abstractmethod
    def load_cpu_kv_to_gpu(mem_indexes: torch.Tensor, page_indexes: torch.Tensor, cpu_cache_client):
        """子类必须实现此方法用于处理数据"""
        pass

    @abstractmethod
    def offload_gpu_kv_to_cpu(
        mem_indexes: torch.Tensor, page_indexes: torch.Tensor, page_readies: torch.Tensor, cpu_cache_client
    ):
        """子类必须实现此方法用于处理数据"""
        pass

    # dp 间共享 kv 的操作接口, 提升dp 模式下的kv 共享效率，降低调度的难度
    @abstractmethod
    def copy_kv_from_other_dp_ranks(
        self,
        mem_managers: List["MemoryManager"],
        move_token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        mem_indexes: torch.Tensor,
        dp_size_in_node: int,
        rank_in_dp: int,
    ):
        pass
