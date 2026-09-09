import collections
from abc import ABC, abstractmethod
from typing import List, Optional

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class StateCacheManager(ABC):
    """CPU checkpoint 槽位池；大小页分别实例化，具体状态布局由子类负责。

    尾部 keep_num 个槽位保留给 CPU cache 碎页传输，不参与普通分配。
    本类不管理 GPU 运行态，也不判断页面边界、前缀匹配或淘汰策略。
    """

    def __init__(self, size: int, keep_num: int = 0):
        self.size = size
        self.keep_num = keep_num
        assert 0 <= keep_num <= size, f"invalid keep_num {keep_num} for size {size}"
        self.free_list = collections.deque(range(size - keep_num))

    @abstractmethod
    def get_state_cache(self, buffer_idx: int):
        """返回指定槽位的状态视图；可以是单个 Tensor 或多个 Tensor。"""

    def alloc_one_state_cache(self) -> Optional[int]:
        return None if not self.free_list else self.free_list.popleft()

    def alloc_state_cache(self, need_size: int) -> Optional[List[int]]:
        if need_size > len(self.free_list):
            logger.error(f"warn no enough cache need_size {need_size} free_size {len(self.free_list)}")
            return None
        return [self.free_list.popleft() for _ in range(need_size)]

    def free_state_cache(self, free_indexes: List[int]):
        alloc_upper_bound = self.size - self.keep_num
        for idx in free_indexes:
            assert 0 <= idx < alloc_upper_bound, (
                f"free index {idx} out of alloc range [0, {alloc_upper_bound}), " f"reserved tail num {self.keep_num}"
            )
        self.free_list.extend(free_indexes)
        assert (
            len(self.free_list) <= alloc_upper_bound
        ), f"free cache num {len(self.free_list)} should not be larger than alloc size {alloc_upper_bound}"

    def get_free_cache_num(self):
        return len(self.free_list)

    def get_used_cache_num(self):
        # Preserve the existing accounting: reserved slots count as used.
        return self.size - len(self.free_list)

    def clear_to_init_state(self):
        """重置空闲槽位；子类同时清零自身的 checkpoint buffer。"""
        self.free_list = collections.deque(range(self.size - self.keep_num))
