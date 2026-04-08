from typing import Optional

import torch

from .hybrid_kv_buffer import HybridKvBuffer
from .kv_buffer_adapter import KvBufferAdapter


class HybridKvBufferAdapter(KvBufferAdapter):
    def __init__(self, kv_buffer: HybridKvBuffer):
        super().__init__(kv_buffer)

    def write_to_page_buffer(
        self, mem_indexes: torch.Tensor, page_tensor: torch.Tensor, tp_index: int, tp_world_size: int
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support paged kv write")

    def read_from_page_buffer(
        self, mem_indexes: torch.Tensor, page_tensor: torch.Tensor, tp_index: int, tp_world_size: int
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support paged kv read")

    def write_from_mla_page_buffer(self, mem_indexes: torch.Tensor, page_tensor: torch.Tensor) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support mla paged kv write")

    def read_from_mla_page_buffer(self, mem_indexes: torch.Tensor, page_tensor: torch.Tensor) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support mla paged kv read")

    def load_from_cpu_cache(
        self,
        gpu_mem_indexes: torch.Tensor,
        cpu_kv_cache: torch.Tensor,
        cpu_kv_cache_scale: Optional[torch.Tensor],
        page_indexes: torch.Tensor,
        tp_index: int,
        tp_world_size: int,
        grid_num: int,
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support cpu cache load")

    def offload_to_cpu_cache(
        self,
        token_indexes: torch.Tensor,
        cpu_kv_cache: torch.Tensor,
        cpu_kv_cache_scale: Optional[torch.Tensor],
        page_indexes: torch.Tensor,
        page_readies: torch.Tensor,
        tp_index: int,
        tp_world_size: int,
        grid_num: int,
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support cpu cache offload")

    def copy_kv_from_other_dp_ranks(
        self,
        mem_managers,
        move_token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        mem_indexes: torch.Tensor,
        dp_size_in_node: int,
        rank_in_dp: int,
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support dp kv copy")
