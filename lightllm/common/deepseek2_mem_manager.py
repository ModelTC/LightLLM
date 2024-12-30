import torch
import os
import torch.distributed as dist
from lightllm.server.pd_io_struct import KVMoveTask
from .mem_manager import MemoryManager
from typing import List
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_trans_kernel.kv_trans import kv_trans

logger = init_logger(__name__)


class Deepseek2MemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]
        self.holding_size = 1 if self.enable_dp else 0
        self.mem_state[0 : self.holding_size] = 1
        self.can_use_mem_size -= self.holding_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

    def get_cell_size(self):
        return self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty((layer_num, size, head_num, head_dim), dtype=dtype, device="cuda")
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
        return

    def send_to_decode_node(
        self, move_tasks: List[KVMoveTask], mem_managers: List["Deepseek2MemoryManager"], dp_size: int
    ):
        assert dp_size == 1

        # 先将数据发送到指定的一张卡上的buffer，再发送。
        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])

        cur_device_index = self.kv_buffer.get_device()
        cur_mem = mem_managers[cur_device_index]
        for layer_index in range(cur_mem.layer_num):
            move_buffer = cur_mem._get_kv_move_data(move_token_indexes, layer_index)
            dist.send(move_buffer, dst=1)
        return

    def _get_kv_move_data(self, token_indexes: List[int], layer_index: int):
        move_size = self.kv_buffer.numel() // self.layer_num // self.size * len(token_indexes)
        move_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(
            1, len(token_indexes), self.head_num, self.head_dim
        )
        move_buffer[:, :, :, :] = self.kv_buffer[layer_index, token_indexes, :, :]
        return move_buffer

    def receive_from_prefill_node(
        self, move_tasks: List[KVMoveTask], mem_managers: List["MemoryManager"], dp_size: int
    ):
        assert dp_size == 1

        # 先将数据接受到指定的一张卡上的buffer，再复制到其他的卡上。
        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])

        cur_device_index = self.kv_buffer.get_device()
        token_num = len(move_token_indexes)
        move_size = self.kv_buffer.numel() // self.layer_num // self.size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(1, token_num, self.head_num, self.head_dim)
        for layer_index in range(self.layer_num):
            dist.recv(recive_buffer, src=0)
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

    def send_to_decode_node_p2p(self, move_tasks: List[KVMoveTask], mem_managers: List["MemoryManager"], dp_size: int):
        """
        使用 p2p triton kernel 进行数据复制和传输的实现方式。
        """
        assert dp_size == 1

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        for layer_index in range(self.layer_num):
            move_buffer = self._get_kv_move_data_p2p(move_token_indexes, layer_index, self.kv_move_buffer)
            dist.send(move_buffer, dst=1)
        return

    def _get_kv_move_data_p2p(self, token_indexes: torch.Tensor, layer_index: int, kv_move_buffer: torch.Tensor):
        move_token_num = len(token_indexes)
        move_size = self.kv_buffer.numel() // self.layer_num // self.size * move_token_num
        move_buffer = kv_move_buffer.view(-1)[0:move_size].view(move_token_num, self.head_num, self.head_dim)
        kv_trans(
            self.kv_buffer[layer_index, :, :, :], token_indexes, move_buffer, self.kv_move_buf_indexes[0:move_token_num]
        )
        return move_buffer

    def receive_from_prefill_node_p2p(
        self, move_tasks: List[KVMoveTask], mem_managers: List["MemoryManager"], dp_size: int
    ):
        assert dp_size == 1

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")

        token_num = len(move_token_indexes)
        move_size = self.kv_buffer.numel() // self.layer_num // self.size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(token_num, self.head_num, self.head_dim)
        for layer_index in range(self.layer_num):
            dist.recv(recive_buffer, src=0)
            for i, mem in enumerate(mem_managers):
                mem._write_kv_move_data_p2p(move_token_indexes, recive_buffer, layer_index)
        return

    def _write_kv_move_data_p2p(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index):
        move_token_num = len(token_indexes)
        kv_trans(buffer_tensor, self.kv_move_buf_indexes[0:move_token_num], self.kv_buffer[layer_index], token_indexes)
        return

    @torch.no_grad()
    def free_all(self):
        self.can_use_mem_size = len(self.mem_state) - self.holding_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self.mem_state[:] = 0
        self.mem_state[0 : self.holding_size] = 1

    @torch.no_grad()
    def free(self, free_index):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        free_index = free_index.long()
        self.decrease_refs(free_index)
        if self.can_use_mem_size + self.holding_size == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
            if self.holding_size > 0:
                logger.debug(f"holding gpu mem size {self.holding_size} for dp")
        return
