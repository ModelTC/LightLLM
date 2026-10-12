import numpy as np
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import List, Optional

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class StateCacheManager(ABC):
    """CPU checkpoint 槽位池；大小页分别实例化，具体状态布局由子类负责。

    尾部 keep_num 个槽位保留给 CPU cache 碎页传输，不参与普通分配。
    本类不管理 GPU 运行态，也不判断页面边界、前缀匹配或淘汰策略。

    本地和共享池使用同一 FIFO 环形队列和引用计数布局。假设普通容量为 3：
    metadata = [
        3,          # 当前空闲数
        0,          # 队头位置
        0, 1, 2,    # 空闲编号表
        0, 0, 0,    # 槽位 0、1、2 的引用计数
    ]
    """

    def __init__(self, size: int, keep_num: int = 0, shm_name: Optional[str] = None, shm_create: bool = False):
        self.size = size
        self.keep_num = keep_num
        assert 0 <= keep_num <= size, f"invalid keep_num {keep_num} for size {size}"
        capacity = size - keep_num
        assert capacity > 0
        self.metadata_shm = None
        if shm_name is None:
            self.metadata = np.empty((2 + 2 * capacity,), dtype=np.int64)
            self.lock = nullcontext()
        else:
            from lightllm.utils.shm_utils import create_or_link_shm
            from lightllm.server.core.objs.atomic_lock import AtomicShmLock

            self.metadata_shm = create_or_link_shm(
                shm_name + "_metadata", (2 + 2 * capacity) * 8, force_mode="create" if shm_create else "link"
            )
            self.metadata = np.ndarray((2 + 2 * capacity,), dtype=np.int64, buffer=self.metadata_shm.buf)
            self.lock = AtomicShmLock(shm_name + "_lock")
        self.free_ids = self.metadata[2 : 2 + capacity]
        self.refs = self.metadata[2 + capacity :]
        if shm_name is None or shm_create:
            self.metadata[0] = capacity
            self.metadata[1] = 0
            self.free_ids[:] = np.arange(capacity)
            self.refs.fill(0)

    @abstractmethod
    def get_state_cache(self, buffer_idx: int):
        """返回指定槽位的状态视图；可以是单个 Tensor 或多个 Tensor。"""

    def alloc_one_state_cache(self) -> Optional[int]:
        indexes = self.alloc_state_cache(1)
        return None if indexes is None else indexes[0]

    def alloc_state_cache(self, need_size: int) -> Optional[List[int]]:
        if need_size == 0:
            return []
        with self.lock:
            free_num = int(self.metadata[0])
            if need_size > free_num:
                logger.error(f"warn no enough cache need_size {need_size} free_size {free_num}")
                return None
            capacity = self.size - self.keep_num
            head = int(self.metadata[1])
            first_size = min(need_size, capacity - head)
            indexes = self.free_ids[head : head + first_size].tolist()
            indexes.extend(self.free_ids[: need_size - first_size].tolist())
            self.metadata[0] = free_num - need_size
            self.metadata[1] = (head + need_size) % capacity
            self.refs[indexes] = 1
            return indexes

    def free_state_cache(self, free_indexes: List[int]):
        alloc_upper_bound = self.size - self.keep_num
        for idx in free_indexes:
            assert 0 <= idx < alloc_upper_bound, (
                f"free index {idx} out of alloc range [0, {alloc_upper_bound}), " f"reserved tail num {self.keep_num}"
            )
        with self.lock:
            for idx in free_indexes:
                assert self.refs[idx] > 0, f"checkpoint {idx} has no owner"
                self.refs[idx] -= 1
                if self.refs[idx] == 0:
                    free_num = int(self.metadata[0])
                    tail = (int(self.metadata[1]) + free_num) % alloc_upper_bound
                    self.free_ids[tail] = idx
                    self.metadata[0] = free_num + 1

    def get_free_cache_num(self):
        with self.lock:
            return int(self.metadata[0])

    def retain_state_cache(self, indexes: List[int]):
        """Adopt checkpoints while their source radix path is still pinned."""
        with self.lock:
            for idx in indexes:
                self.refs[idx] += 1

    def get_used_cache_num(self):
        # Preserve the existing accounting: reserved slots count as used.
        return self.size - self.get_free_cache_num()
