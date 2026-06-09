import torch
import os
import torch.distributed as dist
from .mem_manager import MemoryManager
from typing import List, Union, Any
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_trans_kernel.nixl_kv_trans import mla_page_io
from .operator import Deepseek2MemOperator


logger = init_logger(__name__)


class Deepseek2MemoryManager(MemoryManager):

    operator_class = Deepseek2MemOperator

    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

    def get_att_input_params(self, layer_index: int) -> Any:
        kv = self.kv_buffer[layer_index]
        return kv

    def get_cell_size(self):
        return self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty((layer_num, size + 1, head_num, head_dim), dtype=dtype, device="cuda")

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        self.kv_move_buffer = torch.empty(
            (page_num, page_size, self.layer_num, self.head_num, self.head_dim), dtype=self.dtype, device="cuda"
        )
        self._buffer_mem_indexes_tensors = [
            torch.empty((page_size,), dtype=torch.int64, device="cpu", pin_memory=True) for _ in range(page_num)
        ]
        return self.kv_move_buffer

    def write_mem_to_page_kv_move_buffer(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
        page_kind: str = "kv",
        req_idx: int = None,
    ):
        assert page_kind == "kv", f"{type(self).__name__} does not support page_kind={page_kind}"
        cur_page = self.kv_move_buffer[page_index]
        pin_mem_indexes = self._buffer_mem_indexes_tensors[page_index][0 : len(mem_indexes)]
        pin_mem_indexes.numpy()[:] = mem_indexes
        mem_indexes_gpu = pin_mem_indexes.cuda(non_blocking=True)
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        mla_page_io(
            mem_indexes=mem_indexes_gpu,
            page_tensor=cur_page,
            kv_buffer=dp_mems[0].kv_buffer,
            mode="write",
        )
        return

    def read_page_kv_move_buffer_to_mem(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
        page_kind: str = "kv",
        req_idx: int = None,
    ):
        assert page_kind == "kv", f"{type(self).__name__} does not support page_kind={page_kind}"
        cur_page = self.kv_move_buffer[page_index]
        pin_mem_indexes = self._buffer_mem_indexes_tensors[page_index][0 : len(mem_indexes)]
        pin_mem_indexes.numpy()[:] = mem_indexes
        mem_indexes_gpu = pin_mem_indexes.cuda(non_blocking=True)
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        for mem in dp_mems:
            mla_page_io(
                mem_indexes=mem_indexes_gpu,
                page_tensor=cur_page,
                kv_buffer=mem.kv_buffer,
                mode="read",
            )
