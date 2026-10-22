import torch
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.log_utils import init_logger
from typing import Union, List

logger = init_logger(__name__)


class KvCacheAllocator:
    """内部按页维护空闲索引，对外的容量和申请、释放参数均以 token 为单位。"""

    def __init__(self, size: int, page_size: int) -> None:
        assert page_size >= 1, "page_size must be positive"
        self.page_size = page_size
        self.page_offsets = torch.arange(page_size, dtype=torch.int32, device="cpu", requires_grad=False)
        self._init_buffers(size)

        rank_in_node = get_current_rank_in_node()
        # 用共享内存进行共享，router 模块读取进行精确的调度估计；基础层会统一添加服务前缀以防止实例冲突。
        self.shared_can_use_token_num = SharedInt(f"mem_manger_can_use_token_num_{rank_in_node}")
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return

    def alloc(self, need_size: int) -> torch.Tensor:
        assert need_size >= 0 and need_size % self.page_size == 0, "KV cache allocations must cover complete pages"
        need_page_num = need_size // self.page_size
        if need_page_num > self.mark_end - self.mark_start:
            logger.error(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            assert False, "error alloc state"

        page_start = self.mark_start
        page_end = page_start + need_page_num

        dst = self._alloc_return_tensor(need_size)
        page_starts = self.mem_state[page_start:page_end]
        if self.page_size == 1:
            dst.copy_(page_starts)
        else:
            # 将每个 page 的页首索引展开为页内连续的 token 级别索引，直接写入 dst。
            torch.add(page_starts[:, None], self.page_offsets, out=dst.view(-1, self.page_size))
        self.mark_start = page_end
        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return dst

    def free(self, free_index: Union[torch.Tensor, List[int]]):
        """接收按完整页展开的 token 索引，每页仅回收一个页首索引。"""
        free_token_num = len(free_index)
        assert free_token_num % self.page_size == 0, "KV cache releases must cover complete pages"
        free_page_num = free_token_num // self.page_size
        page_end = self.mark_start
        page_start = page_end - free_page_num
        assert page_start >= 0, f"error free state start: {self.mark_start} free len {len(free_index)}"

        page_starts = free_index[:: self.page_size]
        if isinstance(free_index, list):
            self.mem_state.numpy()[page_start:page_end] = page_starts
        else:
            # GPU 输入在默认流按页取首索引后阻塞 D2H；返回时 CPU 空闲页表已可重新分配。
            self.mem_state[page_start:page_end].copy_(page_starts)

        self.mark_start = page_start
        self.can_use_mem_size += free_token_num
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        if self.can_use_mem_size == self.size:
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return

    def free_all(self):
        self.mem_state = torch.arange(0, self.size, self.page_size, out=self.mem_state, requires_grad=False)
        self.mark_start = 0
        self.mark_end = self.mem_state.numel()
        self.can_use_mem_size = self.size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return

    def resize(self, new_size: int) -> None:
        """仅供测试：按新的 token 容量重新初始化空闲页表。"""
        self._init_buffers(new_size)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

    def _init_buffers(self, size: int) -> None:
        assert size >= 0 and size % self.page_size == 0, "KV cache size must cover complete pages"
        self.size = size
        # 每项保存一个物理页的首 token 索引；[mark_start, mark_end) 是空闲页区间。
        self.mem_state = torch.arange(
            0, size, self.page_size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_start = 0
        self.mark_end = self.mem_state.numel()
        self._return_buffer = None
        self.can_use_mem_size = size

    def _alloc_return_tensor(self, need_size: int) -> torch.Tensor:
        """返回后续申请不会覆写的 pinned 区间，仅保留 slab 尚未使用的尾部。"""
        buffer_token_num = 1024 * 1024
        if need_size == 0 or need_size > buffer_token_num:
            return torch.empty(need_size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True)
        if self._return_buffer is None or self._return_buffer.numel() < need_size:
            self._return_buffer = torch.empty(
                buffer_token_num, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
            )
        # 只保留未使用的尾部，已返回的切片不会被后续分配覆盖。
        dst = self._return_buffer[:need_size]
        self._return_buffer = self._return_buffer[need_size:]
        return dst
