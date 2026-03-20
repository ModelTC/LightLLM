from typing import Optional

from sortedcontainers import SortedSet


class MemoryBlock:
    """内存块类，表示一个连续的内存区域"""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def size(self):
        return self.end - self.start

    def __repr__(self):
        return f"Block(start={self.start}, end={self.end})"

    def can_merge(self, block: "MemoryBlock"):
        return (self.start == block.end) or (block.start == self.end)


class MemoryManager:
    def __init__(self, total_size):
        """
        初始化内存管理器
        :param total_size: 总内存大小
        """
        self.total_size = total_size
        self.mem_set_by_start = SortedSet(key=lambda x: (x.start, x.size()))
        self.mem_set_by_size = SortedSet(key=lambda x: (x.size(), x.start))
        total = MemoryBlock(0, total_size)
        self.__add(total)

    def alloc(self, need_size: int) -> Optional[MemoryBlock]:
        assert need_size > 0

        if len(self.mem_set_by_size) == 0:
            return None

        key = MemoryBlock(start=-1, end=-1 + need_size)
        find_index = self.mem_set_by_size.bisect_left(key)
        if find_index < len(self.mem_set_by_size):
            finded_mem_block: MemoryBlock = self.mem_set_by_size[find_index]
            self.__del(finded_mem_block)
            ret_mem_block = MemoryBlock(
                start=finded_mem_block.start,
                end=finded_mem_block.start + need_size,
            )
            left_block = MemoryBlock(
                start=finded_mem_block.start + need_size,
                end=finded_mem_block.end,
            )
            if left_block.size() > 0:
                self.__add(left_block)

            return ret_mem_block
        else:
            return None

    def release(self, block: MemoryBlock):
        if block is None:
            return
        if len(self.mem_set_by_size) == 0:
            self.__add(block)
            return

        finded_index = self.mem_set_by_start.bisect_left(block)
        for index in [finded_index - 1, finded_index, finded_index + 1]:
            if index < len(self.mem_set_by_start):
                sub_block: MemoryBlock = self.mem_set_by_start[index]
                # merge
                if block.can_merge(sub_block):
                    self.__del(sub_block)
                    merge_block = MemoryBlock(
                        start=min(block.start, sub_block.start),
                        end=max(block.end, sub_block.end),
                    )
                    self.release(merge_block)
                    return
        # 无法merge时，直接add
        self.__add(block)
        return

    def __add(self, block):
        self.mem_set_by_start.add(block)
        self.mem_set_by_size.add(block)

    def __del(self, block):
        self.mem_set_by_start.remove(block)
        self.mem_set_by_size.remove(block)
