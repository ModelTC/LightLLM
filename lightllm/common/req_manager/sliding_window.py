import collections
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import (
    commit_sliding_window_state,
    prepare_sliding_window_indexes,
)

from .hybrid_att import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager import HybridSlidingMemoryManager
    from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig
    from lightllm.server.router.model_infer.infer_batch import InferReq


class SlidingWindowStateCacheManager:
    """Pinned host storage for request-level sliding-window checkpoints."""

    def __init__(self, size: int, sliding_config: "SlidingWindowCacheConfig", keep_num: int = 0):
        self.size = size
        self.keep_num = keep_num
        self.sliding_config = sliding_config
        assert 0 <= keep_num <= size
        self.state_cache = torch.empty(
            (size, *sliding_config.get_state_shape()),
            dtype=sliding_config.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.clear_to_init_state()

    def get_state_cache(self, buffer_idx: int):
        return self.state_cache[buffer_idx]

    def alloc_one_state_cache(self) -> Optional[int]:
        return None if not self.free_list else self.free_list.popleft()

    def alloc_state_cache(self, need_size: int) -> Optional[List[int]]:
        if need_size > len(self.free_list):
            return None
        return [self.free_list.popleft() for _ in range(need_size)]

    def free_state_cache(self, free_indexes: List[int]):
        alloc_upper_bound = self.size - self.keep_num
        assert all(0 <= idx < alloc_upper_bound for idx in free_indexes)
        self.free_list.extend(free_indexes)
        assert len(self.free_list) <= alloc_upper_bound

    def get_free_cache_num(self):
        return len(self.free_list)

    def get_used_cache_num(self):
        return self.size - len(self.free_list)

    def clear_to_init_state(self):
        self.state_cache.zero_()
        self.free_list = collections.deque(range(self.size - self.keep_num))


class ReqManagerForSlidingWindow(HybridAttentionReqManager):
    """Token-granular virtual addresses plus request-granular sliding KV."""

    def __init__(
        self,
        max_request_num: int,
        max_sequence_length: int,
        mem_manager: Optional["HybridSlidingMemoryManager"],
        sliding_config: "SlidingWindowCacheConfig",
        scratch_token_num: int,
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.sliding_config = sliding_config
        self.sliding_window = sliding_config.sliding_window
        self.scratch_token_num = scratch_token_num
        self.runtime_token_num = (max_request_num + 1) * self.sliding_window
        self.scratch_start = self.runtime_token_num
        self.req_to_sliding_window = torch.zeros(
            (
                sliding_config.sliding_layer_num,
                self.runtime_token_num + scratch_token_num,
                2 * sliding_config.sliding_head_num,
                sliding_config.sliding_head_dim,
            ),
            dtype=sliding_config.dtype,
            device="cuda",
        )
        self.req_to_sliding_window_indexs = torch.zeros(
            (max_request_num + 1, max_sequence_length),
            dtype=torch.int32,
            device="cuda",
        )

    def create_state_cache_manager(self, size: int):
        return SlidingWindowStateCacheManager(size=size, sliding_config=self.sliding_config)

    def init_hybrid_attention_state(self, req: "InferReq"):
        start = req.req_idx * self.sliding_window
        self.req_to_sliding_window[:, start : start + self.sliding_window].zero_()

    def restore_big_page_state(self, big_page_buffer_idx: int, req: "InferReq"):
        self._restore_state(req.req_idx, self.mem_manager.hybrid_att_big_page_buffers, big_page_buffer_idx)

    def restore_small_page_state(self, req: "InferReq", small_page_buffers):
        self._restore_state(req.req_idx, small_page_buffers, req.shared_kv_node.small_page_buffer_idx)

    def _restore_state(self, req_idx: int, state_cache_manager, buffer_idx: int):
        start = req_idx * self.sliding_window
        self.req_to_sliding_window[:, start : start + self.sliding_window].copy_(
            state_cache_manager.get_state_cache(buffer_idx),
            non_blocking=True,
        )

    def copy_runtime_state_to_cache(
        self,
        req_indexes: Union[List[int], torch.Tensor],
        buffer_indexes: List[int],
        state_cache_manager: SlidingWindowStateCacheManager,
    ):
        assert len(req_indexes) == len(buffer_indexes)
        if isinstance(req_indexes, torch.Tensor):
            req_indexes = req_indexes.tolist()
        for req_idx, buffer_idx in zip(req_indexes, buffer_indexes):
            if buffer_idx == -1:
                continue
            start = req_idx * self.sliding_window
            state_cache_manager.get_state_cache(buffer_idx).copy_(
                self.req_to_sliding_window[:, start : start + self.sliding_window],
                non_blocking=True,
            )

    def prepare_sliding_window(self, infer_state):
        q_token_num = infer_state.input_ids.shape[0]
        assert q_token_num <= self.scratch_token_num
        prepare_sliding_window_indexes(
            req_to_sliding_window_indexs=self.req_to_sliding_window_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            b_q_seq_len=infer_state.b_q_seq_len,
            b_q_start_loc=infer_state.b_q_start_loc,
            sliding_window=self.sliding_window,
            scratch_start=self.scratch_start,
            max_q_seq_len=infer_state.max_q_seq_len,
        )
        infer_state.sliding_window_mem_index = torch.arange(
            self.scratch_start,
            self.scratch_start + q_token_num,
            dtype=torch.int64,
            device="cuda",
        )

    def get_layer_kv(self, layer_index: int):
        local_layer = self.sliding_config.get_sliding_layer_index(layer_index)
        layer_buffer = self.req_to_sliding_window[local_layer]
        head_num = self.sliding_config.sliding_head_num
        return layer_buffer[:, :head_num], layer_buffer[:, head_num:]

    def commit_layer_state(self, layer_index: int, infer_state):
        local_layer = self.sliding_config.get_sliding_layer_index(layer_index)
        commit_sliding_window_state(
            layer_buffer=self.req_to_sliding_window[local_layer],
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            b_q_seq_len=infer_state.b_q_seq_len,
            b_q_start_loc=infer_state.b_q_start_loc,
            sliding_window=self.sliding_window,
            scratch_start=self.scratch_start,
            max_q_seq_len=infer_state.max_q_seq_len,
        )
