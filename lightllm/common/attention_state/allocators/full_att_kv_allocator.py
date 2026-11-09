"""Full attention KV 分配器：内部按页保存空闲索引，对外只按 token 申请和释放。"""

from __future__ import annotations

import torch

from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class FullAttKVAllocator:
    """按 token 申请和释放。页表只在内部使用。

    公开的是 ``size``、``page_size``、``can_use_mem_size``、``hold_token_indices``、
    ``alloc``、``free`` 和 ``free_all``。
    """

    def __init__(self, size: int, page_size: int = 1, pin_memory: bool = True) -> None:
        assert size >= 0 and page_size > 0 and size % page_size == 0, "KV capacity must contain complete pages"
        assert size + page_size - 1 <= torch.iinfo(torch.int32).max, "KV token indices including HOLD must fit in int32"
        self.page_size = page_size
        self.size = size
        self._pin_memory = pin_memory
        self._page_offsets = torch.arange(page_size, dtype=torch.int32, device="cpu")
        # 每项保存一个物理页的首 token 索引；[_mark_start, _mark_end) 是空闲页区间。
        self._mem_state = torch.arange(0, size, page_size, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        self._mark_start = 0
        self._mark_end = self._mem_state.numel()
        self._return_buffer = None
        rank = get_current_rank_in_node()
        self._shared_can_use_token_num = SharedInt(f"kv_available_tokens_{rank}")
        self._shared_can_use_token_num.set_value(self.can_use_mem_size)

    @property
    def can_use_mem_size(self) -> int:
        return (self._mark_end - self._mark_start) * self.page_size

    @property
    def hold_token_indices(self) -> tuple[int, ...]:
        """One reserved page after the ordinary allocation domain."""
        return tuple(range(self.size, self.size + self.page_size))

    def alloc(self, need_size: int) -> torch.Tensor:
        assert need_size >= 0 and need_size % self.page_size == 0, "allocation must contain complete KV pages"
        need_page_num = need_size // self.page_size
        if need_page_num > self._mark_end - self._mark_start:
            logger.error(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            assert False, f"need {need_size} KV tokens, only {self.can_use_mem_size} available"

        page_start = self._mark_start
        page_end = page_start + need_page_num
        dst = self._alloc_return_tensor(need_size)
        page_starts = self._mem_state[page_start:page_end]
        if self.page_size == 1:
            dst.copy_(page_starts)
        else:
            # 将每个页首索引展开为页内连续的 token 索引，直接写入返回值。
            torch.add(page_starts[:, None], self._page_offsets, out=dst.view(-1, self.page_size))
        self._mark_start = page_end
        self._shared_can_use_token_num.set_value(self.can_use_mem_size)
        return dst

    def free(self, free_index) -> None:
        """接收按完整页展开的 token 索引，每页只回收页首索引。"""
        free_token_num = len(free_index)
        if not free_token_num:
            return
        assert free_token_num % self.page_size == 0, "free must contain complete KV pages"
        free_page_num = free_token_num // self.page_size
        page_end = self._mark_start
        page_start = page_end - free_page_num
        assert page_start >= 0, f"free {free_token_num} KV tokens exceeds allocated capacity"

        page_starts = free_index[:: self.page_size]
        destination = self._mem_state[page_start:page_end]
        if isinstance(page_starts, torch.Tensor):
            # GPU 输入只拷贝页首索引；返回时 CPU 空闲页表已可重新分配。
            destination.copy_(page_starts)
        else:
            destination.numpy()[:] = page_starts
        self._mark_start = page_start
        self._shared_can_use_token_num.set_value(self.can_use_mem_size)
        if self.can_use_mem_size == self.size:
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")

    def free_all(self) -> None:
        torch.arange(0, self.size, self.page_size, out=self._mem_state)
        self._mark_start = 0
        self._mark_end = self._mem_state.numel()
        self._shared_can_use_token_num.set_value(self.can_use_mem_size)

    def _alloc_return_tensor(self, need_size: int) -> torch.Tensor:
        """返回后续申请不会覆写的区间，只保留 slab 尚未使用的尾部。"""
        buffer_token_num = 1024 * 1024
        if need_size == 0 or need_size > buffer_token_num:
            return torch.empty(need_size, dtype=torch.int32, device="cpu", pin_memory=self._pin_memory)
        if self._return_buffer is None or self._return_buffer.numel() < need_size:
            self._return_buffer = torch.empty(
                buffer_token_num, dtype=torch.int32, device="cpu", pin_memory=self._pin_memory
            )
        dst = self._return_buffer[:need_size]
        self._return_buffer = self._return_buffer[need_size:]
        return dst


class KVCapacityReader:
    """Read each local DP group's allocator capacity without modifying it."""

    def __init__(self):
        args = get_env_start_args()
        node_world_size = args.tp // args.nnodes
        tp_group_size = args.tp // args.dp
        self.is_multinode_tp = args.dp == 1 and args.nnodes > 1
        self.counters = [SharedInt(f"kv_available_tokens_{rank}") for rank in range(0, node_world_size, tp_group_size)]

    def get_available_tokens(self, dp_rank_in_node: int) -> int:
        return self.counters[0 if self.is_multinode_tp else dp_rank_in_node].get_value()
