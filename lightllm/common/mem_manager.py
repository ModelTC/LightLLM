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
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from lightllm.distributed.pynccl import PyNcclCommunicator
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.config_utils import get_num_key_value_heads
from lightllm.common.kv_trans_kernel.nixl_kv_trans import page_io

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
        self._mem_state_return = torch.arange(
            0, self.size * 3, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self._return_start = 0
        self.mark_start = 0
        self.mark_end = self.size

        self.can_use_mem_size = self.size

        # 用共享内存进行共享，router 模块读取进行精确的调度估计, nccl port 作为一个单机中单实列的标记。防止冲突。
        from lightllm.utils.envs_utils import get_unique_server_name

        rank_in_node = get_current_rank_in_node()
        self.shared_can_use_token_num = SharedInt(
            f"{get_unique_server_name()}_mem_manger_can_use_token_num_{rank_in_node}"
        )

        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self._init_buffers(
            self.size,
            dtype,
            head_num,
            head_dim,
            layer_num,
        )
        self.HOLD_TOKEN_MEMINDEX = self.size

    def get_cell_size(self):
        return 2 * self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def profile_size(self, mem_fraction):
        if self.size is not None:
            return

        world_size = dist.get_world_size()
        total_memory = get_total_gpu_memory()
        available_memory = get_available_gpu_memory(world_size) - total_memory * (1 - mem_fraction)
        cell_size = self.get_cell_size()
        self.size = int(available_memory * 1024 ** 3 / cell_size)
        if world_size > 1:
            tensor = torch.tensor(self.size, dtype=torch.int64, device=f"cuda:{get_current_device_id()}")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            self.size = tensor.item()
        logger.info(
            f"{str(available_memory)} GB space is available after load the model weight\n"
            f"{str(cell_size / 1024 ** 2)} MB is the size of one token kv cache\n"
            f"{self.size} is the profiled max_total_token_num with the mem_fraction {mem_fraction}\n"
        )
        return

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        # 在初始化 kv_buffer 的时候，每层多初始化了一个 token，这个 token 永远不会被真的被对外
        # 分配，内部实际也没有管理，这个token是预留来对一些特殊的运行模式，如多dp下，overlap microbatch
        # 等模式下 padding 一些请求，使推理过程可以正常运行采用的，其索引值为size，存储在HOLD_TOKEN_MEMINDEX
        # 成员变量中，其与 req_manager 中的HOLD_REQUEST_ID具有类似的作用和意义。
        self.kv_buffer = torch.empty((layer_num, size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda")

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
        self.token_dim_size = self.kv_move_buffer.shape[-2] * self.kv_move_buffer.shape[-1]
        return

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        if isinstance(self, MemoryManager) and type(self) != MemoryManager:
            raise NotImplementedError("subclass need reimpl this method")

        num_kv_head = get_num_key_value_heads(get_env_start_args().model_dir)
        self.kv_move_buffer = torch.empty(
            (page_num, page_size, self.layer_num, 2 * num_kv_head, self.head_dim), dtype=self.dtype, device="cuda"
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
        repeat_count = dp_world_size * self.kv_buffer.shape[2] // self.kv_move_buffer.shape[3]
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        for tp_index in range(dp_world_size):
            if tp_index % repeat_count == 0:
                page_io(
                    mem_indexes=mem_indexes_gpu,
                    page_tensor=cur_page,
                    kv_buffer=dp_mems[tp_index].kv_buffer,
                    tp_index=tp_index,
                    tp_world_size=dp_world_size,
                    mode="write",
                )
        # keep for debug
        # logger.info(f"src token tensor {self.kv_buffer[:, mem_indexes[0], 0, 0]}")
        # logger.info(f"src page token tensor {cur_page[0, :, 0, 0]}")
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
        mem_indexes_gpu = torch.tensor(mem_indexes, dtype=torch.int64, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        for tp_index in range(dp_world_size):
            page_io(
                mem_indexes=mem_indexes_gpu,
                page_tensor=cur_page,
                kv_buffer=dp_mems[tp_index].kv_buffer,
                tp_index=tp_index,
                tp_world_size=dp_world_size,
                mode="read",
            )
        # keep for debug
        # logger.info(f"dst token tensor {self.kv_buffer[:, mem_indexes[0], 0, 0]}")
        # logger.info(f"dst page token tensor {cur_page[0, :, 0, 0]}")

    def send_to_decode_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
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
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                move_buffer = mem._get_kv_move_data(move_token_indexes, layer_index)
                if i == cur_device_index:
                    nccl_comm.send(move_buffer, dst=1)
                else:
                    move_size = move_buffer.numel()
                    new_move_buffer = cur_mem.kv_move_buffer.view(-1)[0:move_size].view(move_buffer.shape)
                    from torch.cuda import comm

                    comm.broadcast(move_buffer, out=[new_move_buffer])
                    nccl_comm.send(new_move_buffer, dst=1)
        return

    def _get_kv_move_data(self, token_indexes: List[int], layer_index: int):
        move_size = self.token_dim_size * len(token_indexes)
        move_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(
            1, len(token_indexes), 2 * self.head_num, self.head_dim
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
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(1, token_num, 2 * self.head_num, self.head_dim)
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                nccl_comm.recv(recive_buffer, src=0)
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
        assert dp_size_in_node == 1

        # 先将数据发送到指定的一张卡上的buffer，再发送。

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                move_buffer = mem._get_kv_move_data_p2p(move_token_indexes, layer_index, self.kv_move_buffer)
                nccl_comm.send(move_buffer, dst=1)
        return

    def _get_kv_move_data_p2p(self, token_indexes: torch.Tensor, layer_index: int, kv_move_buffer: torch.Tensor):
        move_token_num = len(token_indexes)
        move_size = self.token_dim_size * move_token_num
        move_buffer = kv_move_buffer.view(-1)[0:move_size].view(move_token_num, 2 * self.head_num, self.head_dim)
        kv_trans(
            self.kv_buffer[layer_index, :, :, :], token_indexes, move_buffer, self.kv_move_buf_indexes[0:move_token_num]
        )
        return move_buffer

    def receive_from_prefill_node_p2p(
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

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")

        token_num = len(move_token_indexes)
        move_size = self.token_dim_size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(token_num, 2 * self.head_num, self.head_dim)
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                nccl_comm.recv(recive_buffer, src=0)
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
        self.mark_start += need_size

        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        # 利用缓冲区返回，避免异步情况下的内存竞争
        if self._return_start + need_size > self._mem_state_return.shape[0]:
            self._return_start = 0
        ans = self._mem_state_return[self._return_start : self._return_start + need_size]
        ans.copy_(self.mem_state[start:end])
        self._return_start += need_size
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
            # 从 gpu 到 cpu 的拷贝操作是流内阻塞操作
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

    def get_index_kv_buffer(self, index):
        return {"kv_buffer": self.kv_buffer[:, index]}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        self.kv_buffer[:, index].copy_(load_tensor_dict["kv_buffer"])


class ReadOnlyStaticsMemoryManager:
    """
    读取一些统计信息
    """

    def __init__(self) -> None:
        args = get_env_start_args()
        self.global_world_size = args.tp
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = self.global_world_size // args.dp
        # 兼容多机 dp size=1 纯 tp 模式的情况
        self.is_multinode_tp = args.dp == 1 and args.nnodes > 1
        self.shared_tp_infos = [
            SharedInt(f"{get_unique_server_name()}_mem_manger_can_use_token_num_{rank_in_node}")
            for rank_in_node in range(0, self.node_world_size, self.dp_world_size)
        ]

    def get_unrefed_token_num(self, dp_rank_in_node: int):
        if self.is_multinode_tp:
            return self.shared_tp_infos[0].get_value()
        return self.shared_tp_infos[dp_rank_in_node].get_value()
