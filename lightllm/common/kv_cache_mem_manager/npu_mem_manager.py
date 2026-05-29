
from lightllm.common.kv_cache_mem_manager.operator.base import BaseMemManagerOperator
import torch
from typing import Any, List, Tuple

from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.envs_utils import get_page_size
from lightllm.utils.log_utils import init_logger

from .mem_manager import MemoryManager


logger = init_logger(__name__)


class NPUOperator(BaseMemManagerOperator):

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        kb, vb = self.mem_manager.k_buffer[layer_index], self.mem_manager.v_buffer[layer_index]
        k_src, v_src = kv[:, : self.mem_manager.head_num, :], kv[:, self.mem_manager.head_num :, :]
        assert kv.shape[0] == mem_index.shape[0], (kv.shape, mem_index.shape)
        assert k_src.shape[1] == kb.shape[1] and k_src.shape[2] == kb.shape[2], (k_src.shape, kb.shape)
        assert v_src.shape[1] == vb.shape[1] and v_src.shape[2] == vb.shape[2], (v_src.shape, vb.shape)
        self.mem_manager.k_buffer[layer_index].index_copy_(0, mem_index, k_src)
        self.mem_manager.v_buffer[layer_index].index_copy_(0, mem_index, v_src) 


class NPUMemoryManager(MemoryManager):
    operator_class = NPUOperator

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        return self.k_buffer[layer_index], self.v_buffer[layer_index]

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        page_size = get_page_size()
        alloc_size = ((size // page_size) + 1) * page_size if page_size > 1 else size + 1
        logger.info(f"Total page blocks allocated: {alloc_size // page_size} for page_size: {page_size}")
        self.k_buffer = torch.empty((layer_num, alloc_size, head_num, head_dim), dtype=dtype, device=self.target_device)
        self.v_buffer = torch.empty((layer_num, alloc_size, head_num, head_dim), dtype=dtype, device=self.target_device)
        self.kv_buffer = self.k_buffer

    def _free_buffers(self):
        self.k_buffer = None
        self.v_buffer = None
        self.kv_buffer = None

    def get_index_kv_buffer(self, index):
        return {
            "kv_buffer": torch.cat([self.k_buffer[:, index], self.v_buffer[:, index]], dim=1),
        }

    def load_index_kv_buffer(self, index, load_tensor_dict):
        t = load_tensor_dict["kv_buffer"]
        self.k_buffer[:, index].copy_(t[:, : self.head_num])
        self.v_buffer[:, index].copy_(t[:, self.head_num :])

    def alloc_kv_move_buffer(self, max_req_total_len):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated alloc_kv_move_buffer")

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        raise NotImplementedError("NPUMemoryManager does not support PD-separated alloc_paged_kv_move_buffer")

    def write_mem_to_page_kv_move_buffer(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
    ):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated write_mem_to_page_kv_move_buffer")

    def read_page_kv_move_buffer_to_mem(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
    ):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated read_page_kv_move_buffer_to_mem")

    def send_to_decode_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm,
    ):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated send_to_decode_node")

    def receive_from_prefill_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm,
    ):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated receive_from_prefill_node")

    def send_to_decode_node_p2p(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm,
    ):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated send_to_decode_node_p2p")

    def receive_from_prefill_node_p2p(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm,
    ):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated receive_from_prefill_node_p2p")

    def copy_kv_from_other_dp_ranks(
        self,
        mem_managers: List["MemoryManager"],
        move_token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        mem_indexes: torch.Tensor,
        dp_size_in_node: int,
        rank_in_dp: int,
    ):
        raise NotImplementedError(
            "NPUMemoryManager does not support copy_kv_from_other_dp_ranks (needs split-kv kernel)"
        )
