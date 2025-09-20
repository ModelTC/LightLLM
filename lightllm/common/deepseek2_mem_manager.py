import torch
import os
import torch.distributed as dist
from lightllm.server.pd_io_struct import KVMoveTask
from .mem_manager import MemoryManager
from typing import List, Union
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_trans_kernel.kv_trans import kv_trans
from lightllm.common.kv_trans_kernel.kv_trans_v2 import kv_trans_v2_for_d_node, kv_trans_v2_for_p_node
from lightllm.distributed.pynccl import PyNcclCommunicator
from lightllm.common.kv_trans_kernel.nixl_kv_trans import mla_page_io

logger = init_logger(__name__)


class Deepseek2MemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

    def get_cell_size(self):
        return self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty((layer_num, size + 1, head_num, head_dim), dtype=dtype, device="cuda")

        # todo, etp or edp use the same work buffer here
        # also it can be used for any kernels for work buffer witout save info only
        if os.environ.get("ETP_MODE_ENABLED") == "true":
            self.work_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")
            self.work_buffer.share_memory_()

    def alloc_kv_move_buffer(self, max_req_total_len):
        self.kv_move_buffer = torch.empty(
            (1, max_req_total_len + 8, self.head_num, self.head_dim), dtype=self.dtype, device="cuda"
        )
        self.kv_move_buf_indexes = torch.arange(0, max_req_total_len + 8, dtype=torch.int64, device="cuda")
        self.token_dim_size = self.kv_move_buffer.shape[-1] * self.kv_move_buffer.shape[-2]
        return

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
    ):
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
    ):
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

    def send_to_decode_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["Deepseek2MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1

        # 先将数据发送到指定的一张卡上的buffer，再发送。
        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])

        cur_device_index = self.kv_buffer.get_device()
        cur_mem = mem_managers[cur_device_index]
        for layer_index in range(cur_mem.layer_num):
            move_buffer = cur_mem._get_kv_move_data(move_token_indexes, layer_index)
            nccl_comm.send(move_buffer, dst=1)
        return

    def _get_kv_move_data(self, token_indexes: List[int], layer_index: int):
        move_size = self.token_dim_size * len(token_indexes)
        move_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(
            1, len(token_indexes), self.head_num, self.head_dim
        )
        move_buffer[:, :, :, :] = self.kv_buffer[layer_index, token_indexes, :, :]
        return move_buffer

    def receive_from_prefill_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1

        # 先将数据接受到指定的一张卡上的buffer，再复制到其他的卡上。
        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])

        cur_device_index = self.kv_buffer.get_device()
        token_num = len(move_token_indexes)
        move_size = self.token_dim_size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(1, token_num, self.head_num, self.head_dim)
        for layer_index in range(self.layer_num):
            nccl_comm.recv(recive_buffer, src=0)
            for i, mem in enumerate(mem_managers):
                if i == cur_device_index:
                    mem._write_kv_move_data(move_token_indexes, recive_buffer, layer_index)
                else:
                    new_recive_buffer = mem.kv_move_buffer.view(-1)[0:move_size].view(recive_buffer.shape)
                    from torch.cuda import comm

                    comm.broadcast(recive_buffer, out=[new_recive_buffer])
                    mem._write_kv_move_data(move_token_indexes, new_recive_buffer, layer_index)
        return

    def _write_kv_move_data(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index):
        self.kv_buffer[layer_index : layer_index + 1, token_indexes, :, :] = buffer_tensor
        return

    def send_to_decode_node_p2p(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        """
        使用 p2p triton kernel 进行数据复制和传输的实现方式。
        """
        if not hasattr(self, "mem_ptrs_dict"):
            self.mem_ptrs_dict = {}
            for layer_index in range(self.layer_num):
                mems_ptr = []
                for i in range(0, len(mem_managers), len(mem_managers) // dp_size_in_node):
                    mems_ptr.append(mem_managers[i].kv_buffer[layer_index, :, :, :].data_ptr())
                mems_ptr = torch.tensor(mems_ptr, dtype=torch.uint64, device="cuda")
                self.mem_ptrs_dict[layer_index] = mems_ptr

        move_token_indexes = []
        token_dp_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])
                token_dp_indexes.extend([task.prefill_dp_index for _ in range(task.move_kv_len)])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        token_dp_indexes = torch.tensor(token_dp_indexes, dtype=torch.int32, device="cuda")
        for layer_index in range(self.layer_num):
            move_buffer = self._get_kv_move_data_p2p(
                move_token_indexes, token_dp_indexes, layer_index, self.kv_move_buffer, dp_size_in_node
            )
            nccl_comm.send(move_buffer, dst=1)
        return

    def _get_kv_move_data_p2p(
        self,
        token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        layer_index: int,
        kv_move_buffer: torch.Tensor,
        dp_size_in_node: int,
    ):
        move_token_num = len(token_indexes)
        move_size = self.token_dim_size * move_token_num
        move_buffer = kv_move_buffer.view(-1)[0:move_size].view(move_token_num, self.head_num, self.head_dim)
        kv_trans_v2_for_p_node(
            input_mems=self.mem_ptrs_dict[layer_index],
            input_idx=token_indexes,
            input_dp_idx=token_dp_indexes,
            output=move_buffer,
            output_idx=self.kv_move_buf_indexes[0:move_token_num],
            dp_size_in_node=dp_size_in_node,
        )
        return move_buffer

    def receive_from_prefill_node_p2p(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        if not hasattr(self, "mem_ptrs_dict"):
            self.mem_ptrs_dict = {}
            for layer_index in range(self.layer_num):
                mems_ptr = []
                for i in range(0, len(mem_managers)):
                    mems_ptr.append(mem_managers[i].kv_buffer[layer_index, :, :, :].data_ptr())
                mems_ptr = torch.tensor(mems_ptr, dtype=torch.uint64, device="cuda")
                self.mem_ptrs_dict[layer_index] = mems_ptr

        move_token_indexes = []
        token_dp_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])
                token_dp_indexes.extend([task.decode_dp_index for _ in range(task.move_kv_len)])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        token_dp_indexes = torch.tensor(token_dp_indexes, dtype=torch.int32, device="cuda")

        token_num = len(move_token_indexes)
        move_size = self.token_dim_size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(token_num, self.head_num, self.head_dim)
        for layer_index in range(self.layer_num):
            nccl_comm.recv(recive_buffer, src=0)
            self._write_kv_move_data_p2p(
                move_token_indexes, token_dp_indexes, recive_buffer, layer_index, dp_size_in_node
            )
        return

    def _write_kv_move_data_p2p(
        self,
        token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        buffer_tensor: torch.Tensor,
        layer_index,
        dp_size_in_node: int,
    ):
        move_token_num = len(token_indexes)
        kv_trans_v2_for_d_node(
            output_mems=self.mem_ptrs_dict[layer_index],
            output_idx=token_indexes,
            output_dp_idx=token_dp_indexes,
            input=buffer_tensor,
            input_idx=self.kv_move_buf_indexes[0:move_token_num],
            dp_size_in_node=dp_size_in_node,
        )
        return
