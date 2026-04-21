import torch
from abc import ABC, abstractmethod

# 定义一个抽象基类
class BaseMemManagerOperator(ABC):
    def __init__(self, mem_manager) -> None:
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
