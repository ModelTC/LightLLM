import numpy as np
import torch
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.dist_utils import get_current_rank_in_node


class KvCacheAllocator:
    """Allocate token indexes; consecutive pages use compressed storage when page_size > 1."""

    def __init__(self, size: int, page_size: int = 1) -> None:
        self.page_size = page_size
        rank = get_current_rank_in_node()
        self.shared_can_use_token_num = SharedInt(f"mem_manger_can_use_token_num_{rank}")
        self.resize(size)

    def _alloc_token_index_blocks(self, need_size: int):
        """Borrow index storage until the next mutation; P > 1 stores each block's first token index."""
        assert 0 <= need_size <= self.can_use_mem_size, "error alloc state"
        start, size = self.size - self.can_use_mem_size, need_size
        if self.page_size != 1:
            assert need_size % self.page_size == 0
            start, size = start // self.page_size, size // self.page_size
        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return self._free_token_index_blocks[start : start + size]

    def alloc(self, need_size: int) -> torch.Tensor:
        index_blocks = self._alloc_token_index_blocks(need_size)
        out = torch.empty(need_size, dtype=torch.int32, device="cpu", pin_memory=True)
        if self.page_size == 1:
            out.numpy()[:] = index_blocks
        else:
            self.expand_token_index_blocks(index_blocks, out.numpy())
        return out

    def expand_token_index_blocks(self, index_blocks, out=None):
        if out is None:
            out = np.empty(len(index_blocks) * self.page_size, dtype=np.int32)
        if self.page_size == 1:
            out[:] = index_blocks
        else:
            # Short pages expand faster along the page-base axis than in short token runs.
            np.add(
                index_blocks,
                self.page_offsets[:, None],
                out=out.reshape(-1, self.page_size).T,
                order="C" if self.page_size < 8 else "F",
            )
        return out

    def free(self, free_index):
        size = len(free_index)
        assert size % self.page_size == 0
        values = free_index.cpu().numpy() if isinstance(free_index, torch.Tensor) else np.asarray(free_index)
        self.free_token_index_blocks(values[:: self.page_size])

    def free_token_index_blocks(self, index_blocks):
        size = len(index_blocks) * self.page_size
        assert size <= self.size - self.can_use_mem_size, "error free state"
        end = (self.size - self.can_use_mem_size) // self.page_size
        self._free_token_index_blocks[end - len(index_blocks) : end] = index_blocks
        self.can_use_mem_size += size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

    def free_all(self):
        self._free_token_index_blocks[:] = np.arange(0, self.size, self.page_size, dtype=np.int32)
        self.can_use_mem_size = self.size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

    def resize(self, new_size: int) -> None:
        assert new_size % self.page_size == 0
        self.size = new_size
        self._free_token_index_blocks = np.arange(0, new_size, self.page_size, dtype=np.int32)
        self.page_offsets = np.arange(self.page_size, dtype=np.int32)
        self.can_use_mem_size = new_size
        self.shared_can_use_token_num.set_value(new_size)
