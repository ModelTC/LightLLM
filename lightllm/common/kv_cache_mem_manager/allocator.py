import numpy as np
import torch
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.dist_utils import get_current_rank_in_node


class KvCacheAllocator:
    """A stack of physical page bases; returned token indexes own their storage."""

    def __init__(self, size: int, page_size: int = 1) -> None:
        self.page_size = page_size
        rank = get_current_rank_in_node()
        self.shared_can_use_token_num = SharedInt(f"mem_manger_can_use_token_num_{rank}")
        self.resize(size)

    def _alloc_pages(self, need_size: int):
        """Borrow page bases until the next allocator mutation; callers snapshot before DMA."""
        assert need_size % self.page_size == 0
        assert 0 <= need_size <= self.can_use_mem_size, "error alloc state"
        start = (self.size - self.can_use_mem_size) // self.page_size
        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return self.free_pages[start : start + need_size // self.page_size]

    def alloc(self, need_size: int) -> torch.Tensor:
        pages = self._alloc_pages(need_size)
        indexes = torch.empty(need_size, dtype=torch.int32, device="cpu", pin_memory=True)
        self.expand_pages(pages, indexes.numpy())
        return indexes

    def expand_pages(self, pages, out=None):
        if out is None:
            out = np.empty(len(pages) * self.page_size, dtype=np.int32)
        if self.page_size == 1:
            out[:] = pages
        else:
            # Short pages expand faster along the page-base axis than in short token runs.
            np.add(
                pages,
                self.page_offsets[:, None],
                out=out.reshape(-1, self.page_size).T,
                order="C" if self.page_size < 8 else "F",
            )
        return out

    def free(self, free_index):
        size = len(free_index)
        assert size % self.page_size == 0
        values = free_index.cpu().numpy() if isinstance(free_index, torch.Tensor) else np.asarray(free_index)
        self.release_pages(values[:: self.page_size])

    def release_pages(self, pages):
        size = len(pages) * self.page_size
        assert size <= self.size - self.can_use_mem_size, "error free state"
        end = (self.size - self.can_use_mem_size) // self.page_size
        self.free_pages[end - len(pages) : end] = pages
        self.can_use_mem_size += size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

    def free_all(self):
        self.free_pages[:] = np.arange(0, self.size, self.page_size, dtype=np.int32)
        self.can_use_mem_size = self.size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

    def resize(self, new_size: int) -> None:
        assert new_size % self.page_size == 0
        self.size = new_size
        self.free_pages = np.arange(0, new_size, self.page_size, dtype=np.int32)
        self.page_offsets = np.arange(self.page_size, dtype=np.int32)
        self.can_use_mem_size = new_size
        self.shared_can_use_token_num.set_value(new_size)
