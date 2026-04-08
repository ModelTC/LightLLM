from typing import Optional

import torch

from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu, load_cpu_kv_to_gpu
from lightllm.common.kv_trans_kernel.kv_trans_v2 import kv_trans_for_dp
from lightllm.common.kv_trans_kernel.nixl_kv_trans import mla_page_io, page_io

from .kv_buffer import KvBuffer


class KvBufferAdapter:
    """与 kv buffer 相关的业务适配类。

    这个类的职责是承接 page io、cpu cache、dp 传输等业务流程。
    这些能力会使用 kv buffer，但 kv buffer 只是业务函数的输入之一，
    并不是这里唯一的关注对象；方法通常还会组合 page tensor、索引、
    tp/dp 上下文等额外业务参数一起完成操作。
    """

    def __init__(self, kv_buffer: KvBuffer):
        self.kv_buffer = kv_buffer
        # dp copy 路径会缓存远端 kv buffer 的底层地址，避免重复构建。
        self._mem_ptrs_tensor = None

    def write_to_page_buffer(
        self,
        mem_indexes: torch.Tensor,
        page_tensor: torch.Tensor,
        tp_index: int = 0,
        tp_world_size: int = 1,
        is_mla: bool = False,
    ) -> None:
        if is_mla:
            mla_page_io(
                mem_indexes=mem_indexes,
                page_tensor=page_tensor,
                kv_buffer=self.kv_buffer.get_storage_tensor(),
                mode="write",
            )
        else:
            page_io(
                mem_indexes=mem_indexes,
                page_tensor=page_tensor,
                kv_buffer=self.kv_buffer.get_storage_tensor(),
                tp_index=tp_index,
                tp_world_size=tp_world_size,
                mode="write",
            )

    def read_from_page_buffer(
        self,
        mem_indexes: torch.Tensor,
        page_tensor: torch.Tensor,
        tp_index: int = 0,
        tp_world_size: int = 1,
        is_mla: bool = False,
    ) -> None:
        if is_mla:
            mla_page_io(
                mem_indexes=mem_indexes,
                page_tensor=page_tensor,
                kv_buffer=self.kv_buffer.get_storage_tensor(),
                mode="read",
            )
        else:
            page_io(
                mem_indexes=mem_indexes,
                page_tensor=page_tensor,
                kv_buffer=self.kv_buffer.get_storage_tensor(),
                tp_index=tp_index,
                tp_world_size=tp_world_size,
                mode="read",
            )

    def copy_kv_from_other_dp_ranks(
        self,
        mem_managers,
        move_token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        mem_indexes: torch.Tensor,
        dp_size_in_node: int,
        rank_in_dp: int,
    ) -> None:
        if self._mem_ptrs_tensor is None:
            mems_ptr_list = [mem.kv_buffer.get_storage_data_ptr() for mem in mem_managers]
            self._mem_ptrs_tensor = torch.tensor(mems_ptr_list, dtype=torch.uint64, device="cpu", pin_memory=True)

        kv_trans_for_dp(
            input_mems=self._mem_ptrs_tensor.cuda(non_blocking=True),
            input_idx=move_token_indexes,
            input_dp_idx=token_dp_indexes,
            output=self.kv_buffer.get_storage_tensor(),
            output_idx=mem_indexes,
            dp_size_in_node=dp_size_in_node,
            rank_in_dp=rank_in_dp,
        )

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
        load_cpu_kv_to_gpu(
            gpu_mem_indexes=gpu_mem_indexes,
            gpu_kv_cache=self.kv_buffer.get_storage_tensor(),
            gpu_kv_cache_scale=self.kv_buffer.get_scale_buffer(),
            cpu_kv_cache=cpu_kv_cache,
            cpu_kv_cache_scale=cpu_kv_cache_scale,
            page_indexes=page_indexes,
            tp_index=tp_index,
            tp_world_size=tp_world_size,
            grid_num=grid_num,
        )

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
        offload_gpu_kv_to_cpu(
            token_indexes=token_indexes,
            gpu_kv_cache=self.kv_buffer.get_storage_tensor(),
            gpu_kv_cache_scale=self.kv_buffer.get_scale_buffer(),
            cpu_kv_cache=cpu_kv_cache,
            cpu_kv_cache_scale=cpu_kv_cache_scale,
            page_indexes=page_indexes,
            page_readies=page_readies,
            tp_index=tp_index,
            tp_world_size=tp_world_size,
            grid_num=grid_num,
        )
