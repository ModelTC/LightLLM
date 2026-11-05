"""状态页槽位分配器，仅管理槽位，不持有状态 buffer 或缓存策略。"""

from __future__ import annotations

from typing import List
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

    @property
    def free_count(self) -> int:
        return len(self._free)

    def alloc(self, num_pages: int) -> list[int] | None:
        assert num_pages >= 0, "num_pages must be nonnegative"
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
            self._allocated[slot] = 0
        self._free.extend(page_slot_ids)
