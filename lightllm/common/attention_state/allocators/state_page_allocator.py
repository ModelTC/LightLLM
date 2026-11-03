"""状态页槽位分配器，仅管理槽位，不持有状态 buffer 或缓存策略。"""

from __future__ import annotations

from typing import Callable, List
from collections import deque


class StatePageAllocator:
    def __init__(self, size: int, reserved_slots: int = 0):
        self.size = size
        self.reserved_slots = reserved_slots
        if not 0 <= self.reserved_slots <= self.size:
            raise ValueError("reserved_slots must be between zero and capacity")
        self.capacity = self.size - self.reserved_slots
        self._free = deque(range(self.capacity))
        self._allocated = bytearray(self.capacity)
        self._retained = {}
        self.on_allocation_pressure: Callable[[StatePageAllocator, int], None] | None = None

    @property
    def free_count(self) -> int:
        return len(self._free)

    def alloc(self, num_pages: int) -> list[int] | None:
        assert num_pages >= 0, "num_pages must be nonnegative"
        if num_pages > self.free_count and self.on_allocation_pressure is not None:
            self.on_allocation_pressure(self, num_pages)
        if num_pages > self.free_count:
            return None
        slots = [self._free.popleft() for _ in range(num_pages)]
        for slot in slots:
            self._allocated[slot] = 1
        return slots

    def free(self, page_slot_ids: List[int]) -> None:
        assert isinstance(page_slot_ids, list), "slot indices must be a list"
        assert len(set(page_slot_ids)) == len(page_slot_ids), "cannot free duplicate slots"
        assert all(0 <= slot < self.capacity for slot in page_slot_ids), "slot outside the allocatable domain"
        assert all(self._allocated[slot] for slot in page_slot_ids), "cannot free an unallocated slot"
        for slot in page_slot_ids:
            if slot in self._retained:
                self._retained[slot] -= 1
                if self._retained[slot] == 0:
                    del self._retained[slot]
                continue
            self._allocated[slot] = 0
            self._free.append(slot)

    def retain(self, page_slot_ids: List[int]) -> None:
        """异步复制持有独立引用，原请求释放后仍不能复用对应的 pinned 状态页。"""
        assert len(set(page_slot_ids)) == len(page_slot_ids), "cannot retain duplicate slots"
        assert all(0 <= slot < self.capacity and self._allocated[slot] for slot in page_slot_ids)
        for slot in page_slot_ids:
            self._retained[slot] = self._retained.get(slot, 0) + 1
