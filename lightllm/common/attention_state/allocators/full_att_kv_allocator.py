"""Full attention KV 分配器：接口使用 token 数量和索引，分配与回收须满足整页约束。"""

from __future__ import annotations

import numpy as np
import torch

from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_env_start_args


class FullAttKVAllocator:
    """Allocate token indexes; consecutive pages use compressed storage when page_size > 1."""

    def __init__(self, size: int, page_size: int = 1, pin_memory: bool = True) -> None:
        assert size >= 0 and page_size > 0 and size % page_size == 0, "KV capacity must contain complete pages"
        assert size + page_size - 1 <= np.iinfo(np.int32).max, "KV token indices including HOLD must fit in int32"
        self.page_size = page_size
        self.pin_memory = pin_memory
        rank = get_current_rank_in_node()
        self.shared_can_use_token_num = SharedInt(f"kv_available_tokens_{rank}")
        self.resize(size)

    @property
    def hold_token_indices(self) -> tuple[int, ...]:
        """One reserved page after the ordinary allocation domain."""
        return tuple(range(self.size, self.size + self.page_size))

    def _alloc_token_index_blocks(self, need_size: int):
        """Borrow index storage until the next mutation; P > 1 stores each block's first token index."""
        assert need_size >= 0, "allocation must be nonnegative"
        assert need_size <= self.can_use_mem_size, f"need {need_size} KV tokens, only {self.can_use_mem_size} available"
        start, size = self.size - self.can_use_mem_size, need_size
        assert need_size % self.page_size == 0, "allocation must contain complete KV pages"
        if self.page_size > 1:
            start, size = start // self.page_size, size // self.page_size
        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return self._free_token_index_blocks[start : start + size]

    def alloc(self, need_size: int) -> torch.Tensor:
        index_blocks = self._alloc_token_index_blocks(need_size)
        out = torch.empty(need_size, dtype=torch.int32, device="cpu", pin_memory=self.pin_memory)
        if self.page_size > 1:
            self.expand_token_index_blocks(index_blocks, out.numpy())
        else:
            out.numpy()[:] = index_blocks
        return out

    def expand_token_index_blocks(self, index_blocks, out=None):
        assert self.page_size > 1
        if out is None:
            out = np.empty(len(index_blocks) * self.page_size, dtype=np.int32)
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
        if not size:
            return
        assert size % self.page_size == 0, "free must contain complete KV pages"
        values = free_index.cpu().numpy() if isinstance(free_index, torch.Tensor) else np.asarray(free_index)
        self.free_token_index_blocks(values[:: self.page_size])

    def free_token_index_blocks(self, index_blocks):
        size = len(index_blocks) * self.page_size
        assert size <= self.size - self.can_use_mem_size, f"free {size} KV tokens exceeds allocated capacity"
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
