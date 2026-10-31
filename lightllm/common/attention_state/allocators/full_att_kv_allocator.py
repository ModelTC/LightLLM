"""Full attention KV 分配器：接口使用 token 数量和索引，分配与回收须满足整页约束。"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import torch

from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = init_logger(__name__)


class FullAttKVAllocator:
    def __init__(self, size: int, page_size: int = 1, pin_memory: bool = True):
        assert size >= 0 and page_size > 0 and size % page_size == 0, "KV capacity must contain complete pages"
        assert size + page_size - 1 <= np.iinfo(np.int32).max, "KV token indices including HOLD must fit in int32"
        self.size = size
        self.page_size = page_size
        self.mem_state = torch.arange(
            end=self.size,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.mark_start = 0
        self.mark_end = self.size
        self._mem_state_return = torch.empty(
            size=(self.size * 3,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._return_start = 0
        self.can_use_mem_size = self.size
        rank_in_node = get_current_rank_in_node()
        self.shared_can_use_token_num = SharedInt(f"kv_available_tokens_{rank_in_node}")
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

    @property
    def hold_token_indices(self) -> tuple[int, ...]:
        """One reserved page after the ordinary allocation domain."""
        return tuple(range(self.size, self.size + self.page_size))

    def alloc(self, need_size: int) -> torch.Tensor:
        assert need_size >= 0 and need_size % self.page_size == 0, "allocation must contain complete KV pages"
        assert need_size <= self.mark_end - self.mark_start, (
            f"need {need_size} KV tokens, only {self.can_use_mem_size} available"
        )
        start = self.mark_start
        end = start + need_size
        self.mark_start = end
        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        # Reuse a rolling buffer so frees do not overwrite indices
        # still being consumed by asynchronous H2D copies.
        if self._return_start + need_size > self._mem_state_return.shape[0]:
            self._return_start = 0
        ans = self._mem_state_return[self._return_start : self._return_start + need_size]
        ans.copy_(self.mem_state[start:end])
        self._return_start += need_size
        return ans

    def free(self, token_indices: Sequence[int] | torch.Tensor) -> None:
        num_tokens = len(token_indices)
        if num_tokens == 0:
            return
        assert num_tokens % self.page_size == 0, "free must contain complete KV pages"
        end = self.mark_start
        start = end - num_tokens
        assert start >= 0, f"free {num_tokens} KV tokens exceeds allocated {end}"
        if isinstance(token_indices, torch.Tensor):
            self.mem_state[start:end].copy_(token_indices)
        else:
            self.mem_state.numpy()[start:end] = token_indices
        self.mark_start = start
        self.can_use_mem_size += num_tokens
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        if self.can_use_mem_size == self.size:
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")

    def free_all(self) -> None:
        """Reset after all users of this allocation domain have finished."""
        self.mem_state.numpy()[:] = np.arange(self.size, dtype=np.int32)
        self.mark_start = 0
        self.mark_end = self.size
        self.can_use_mem_size = self.size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)


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
