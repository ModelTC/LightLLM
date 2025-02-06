import re
import os
import torch
import torch.distributed as dist
from typing import List, Union
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory
from lightllm.common.kv_trans_kernel.kv_trans import kv_trans

logger = init_logger(__name__)


class MemoryManager:
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        self.size = size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.always_copy = always_copy
        self.dtype = dtype
        # profile the max total token num if the size is None
        self.profile_size(mem_fraction)

        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_start = 0
        self.mark_end = self.size

        self.can_use_mem_size = self.size

        # 用共享内存进行共享，router 模块读取进行精确的调度估计, nccl port 作为一个单机中单实列的标记。防止冲突。
        from torch.distributed.distributed_c10d import _default_pg_init_method

        nccl_port = re.search(r":(\d+)$", _default_pg_init_method).group(1)
        assert nccl_port is not None
        logger.info(f"mem manger get nccl port: {str(nccl_port)}")

        rank_id = dist.get_rank()
        self.shared_can_use_token_num = SharedInt(f"{str(nccl_port)}_mem_manger_can_use_token_num_{rank_id}")

        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self._init_buffers(
            self.size,
            dtype,
            head_num,
            head_dim,
            layer_num,
        )

    def get_cell_size(self):
        return 2 * self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def profile_size(self, mem_fraction):
        if self.size is not None:
            return
        import torch.distributed as dist

        tp_rank = dist.get_rank()
        world_size = dist.get_world_size()
        total_memory = get_total_gpu_memory()
        available_memory = get_available_gpu_memory(tp_rank, world_size) - total_memory * (1 - mem_fraction)
        cell_size = self.get_cell_size()
        self.size = int(available_memory * 1024 ** 3 / cell_size)
        logger.info(
            f"{str(available_memory)} GB space is available after load the model weight\n"
            f"{str(cell_size / 1024 ** 2)} MB is the size of one token kv cache\n"
            f"{self.size} is the profiled max_total_token_num with the mem_fraction {mem_fraction}\n"
        )
        return

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty((layer_num, size, 2 * head_num, head_dim), dtype=dtype, device="cuda")

    def alloc_kv_move_buffer(self, max_req_total_len):
        """
        pd 分离模式使用的特殊接口
        """
        if isinstance(self, MemoryManager) and type(self) != MemoryManager:
            raise NotImplementedError("subclass need reimpl this method")
        self.kv_move_buffer = torch.empty(
            (1, max_req_total_len + 8, 2 * self.head_num, self.head_dim), dtype=self.dtype, device="cuda"
        )
        self.kv_move_buf_indexes = torch.arange(0, max_req_total_len + 8, dtype=torch.int64, device="cuda")
        return

    def send_to_decode_node(self, move_tasks: List[KVMoveTask], mem_managers: List["MemoryManager"], dp_size: int):
        """
        dp_size 是为 deepseekv2 类型，可以 dp 和 tp 混合模式运行的模型定制的参数，
        普通tp模式下, dp_size 一定等于 1, dp_index 一定等于 0, 同时普通模式下, 这两个参数并不会
        被真正使用
        """
        assert dp_size == 1

        # 先将数据发送到指定的一张卡上的buffer，再发送。
        import torch.distributed as dist

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])

        cur_device_index = self.kv_buffer.get_device()
        cur_mem = mem_managers[cur_device_index]
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                move_buffer = mem._get_kv_move_data(move_token_indexes, layer_index)
                if i == cur_device_index:
                    dist.send(move_buffer, dst=1)
                else:
                    move_size = move_buffer.numel()
                    new_move_buffer = cur_mem.kv_move_buffer.view(-1)[0:move_size].view(move_buffer.shape)
                    from torch.cuda import comm

                    comm.broadcast(move_buffer, out=[new_move_buffer])
                    dist.send(new_move_buffer, dst=1)
        return

    def _get_kv_move_data(self, token_indexes: List[int], layer_index: int):
        move_size = self.kv_buffer.numel() // self.layer_num // self.size * len(token_indexes)
        move_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(
            1, len(token_indexes), 2 * self.head_num, self.head_dim
        )
        move_buffer[:, :, :, :] = self.kv_buffer[layer_index, token_indexes, :, :]
        return move_buffer

    def receive_from_prefill_node(
        self, move_tasks: List[KVMoveTask], mem_managers: List["MemoryManager"], dp_size: int
    ):
        """
        dp_size 是为 deepseekv2 类型，可以 dp 和 tp 混合模式运行的模型定制的参数，
        普通tp模式下, dp_size 一定等于 1, 同时普通模式下, 这两个参数并不会
        被真正使用
        """
        assert dp_size == 1

        # 先将数据接受到指定的一张卡上的buffer，再复制到其他的卡上。
        import torch.distributed as dist

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])

        cur_device_index = self.kv_buffer.get_device()
        token_num = len(move_token_indexes)
        move_size = self.kv_buffer.numel() // self.layer_num // self.size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(1, token_num, 2 * self.head_num, self.head_dim)
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                dist.recv(recive_buffer, src=0)
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

        # 先将数据发送到指定的一张卡上的buffer，再发送。
        import torch.distributed as dist

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                move_buffer = mem._get_kv_move_data_p2p(move_token_indexes, layer_index, self.kv_move_buffer)
                dist.send(move_buffer, dst=1)
        return

    def _get_kv_move_data_p2p(self, token_indexes: torch.Tensor, layer_index: int, kv_move_buffer: torch.Tensor):
        move_token_num = len(token_indexes)
        move_size = self.kv_buffer.numel() // self.layer_num // self.size * move_token_num
        move_buffer = kv_move_buffer.view(-1)[0:move_size].view(move_token_num, 2 * self.head_num, self.head_dim)
        kv_trans(
            self.kv_buffer[layer_index, :, :, :], token_indexes, move_buffer, self.kv_move_buf_indexes[0:move_token_num]
        )
        return move_buffer

    def receive_from_prefill_node_p2p(
        self, move_tasks: List[KVMoveTask], mem_managers: List["MemoryManager"], dp_size: int
    ):
        assert dp_size == 1

        # 先将数据接受到指定的一张卡上的buffer，再复制到其他的卡上。
        import torch.distributed as dist

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")

        token_num = len(move_token_indexes)
        move_size = self.kv_buffer.numel() // self.layer_num // self.size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(token_num, 2 * self.head_num, self.head_dim)
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                dist.recv(recive_buffer, src=0)
                mem._write_kv_move_data_p2p(move_token_indexes, recive_buffer, layer_index)
        return

    def _write_kv_move_data_p2p(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index):
        move_token_num = len(token_indexes)
        kv_trans(buffer_tensor, self.kv_move_buf_indexes[0:move_token_num], self.kv_buffer[layer_index], token_indexes)
        return

    def _free_buffers(self):
        self.kv_buffer = None

    def alloc(self, need_size) -> torch.Tensor:
        if need_size > self.mark_end - self.mark_start:
            logger.error(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            assert False, "error alloc state"

        start = self.mark_start
        end = self.mark_start + need_size
        ans = self.mem_state[start:end]
        self.mark_start += need_size

        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return ans

    def free(self, free_index: Union[torch.Tensor, List[int]]):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """

        end = self.mark_start
        start = self.mark_start - len(free_index)
        assert start >= 0, f"error free state start: {self.mark_start} free len {len(free_index)}"

        if isinstance(free_index, list):
            self.mem_state.numpy()[start:end] = free_index
        else:
            self.mem_state[start:end] = free_index

        self.mark_start -= len(free_index)

        self.can_use_mem_size += len(free_index)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        if self.can_use_mem_size == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return

    def free_all(self):
        self.can_use_mem_size = len(self.mem_state)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self.mem_state.numpy()[:] = list(range(0, len(self.mem_state)))
        self.mark_start = 0
        self.mark_end = len(self.mem_state)

    def resize_mem(self, new_size):
        """
        just for test code
        """
        size = new_size
        dtype = self.dtype
        head_num = self.head_num
        head_dim = self.head_dim
        layer_num = self.layer_num

        self.size = new_size
        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_start = 0
        self.mark_end = self.size
        self.can_use_mem_size = self.size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self._free_buffers()
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
        return


class ReadOnlyStaticsMemoryManager:
    """
    读取一些统计信息
    """

    def __init__(self, nccl_port, tp_size) -> None:
        self.shared_tp_infos = [
            SharedInt(f"{str(nccl_port)}_mem_manger_can_use_token_num_{tp_index}") for tp_index in range(tp_size)
        ]

    def get_unrefed_token_num(self, tp_index: int):
        return self.shared_tp_infos[tp_index].get_value()
