from typing import TYPE_CHECKING, List, Optional

import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import commit_sliding_window_state
from lightllm.common.sliding_window_cache_manager import SlidingWindowStateCacheManager

from .hybrid_att import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager import HybridSlidingMemoryManager
    from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForSlidingWindow(HybridAttentionReqManager):
    """按请求保存窗口运行态，并在大小页边界保存和恢复 checkpoint。"""

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
        self.scratch_start = (max_request_num + 1) * self.sliding_window
        # Attention reads history and current-chunk KV from one buffer. The
        # request state is a view of its ring region, not a second allocation.
        self.sliding_kv_buffer = torch.zeros(
            (
                sliding_config.sliding_layer_num,
                self.scratch_start + scratch_token_num,
                2 * sliding_config.sliding_head_num,
                sliding_config.sliding_head_dim,
            ),
            dtype=sliding_config.dtype,
            device="cuda",
        )
        self.req_to_sliding_window = self.sliding_kv_buffer[:, : self.scratch_start].view(
            sliding_config.sliding_layer_num,
            max_request_num + 1,
            self.sliding_window,
            2 * sliding_config.sliding_head_num,
            sliding_config.sliding_head_dim,
        )

    def create_state_cache_manager(self, size: int):
        # Allocated with full KV and big pages, within the same GPU budget.
        return self.mem_manager.sliding_small_page_buffers

    def init_hybrid_attention_state(self, req: "InferReq"):
        self.req_to_sliding_window[:, req.req_idx].zero_()

    def restore_big_page_state(self, big_page_buffer_idx: int, req: "InferReq"):
        self._restore_state(req.req_idx, self.mem_manager.linear_att_big_page_buffers, big_page_buffer_idx)

    def restore_small_page_state(self, req: "InferReq", small_page_buffers):
        self._restore_state(req.req_idx, small_page_buffers, req.shared_kv_node.small_page_buffer_idx)

    def _restore_state(self, req_idx: int, state_cache_manager, buffer_idx: int):
        self.req_to_sliding_window[:, req_idx].copy_(
            state_cache_manager.get_state_cache(buffer_idx),
            non_blocking=True,
        )

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        assert len(req_indexes) == len(buffer_indexes)
        for req_idx, buffer_idx in zip(req_indexes, buffer_indexes):
            if buffer_idx == -1:
                continue
            self.save_small_page_state(req_idx, buffer_idx, self.mem_manager.linear_att_big_page_buffers)

    def save_small_page_state(self, req_idx: int, buffer_idx: int, small_page_buffers: SlidingWindowStateCacheManager):
        small_page_buffers.get_state_cache(buffer_idx).copy_(
            self.req_to_sliding_window[:, req_idx],
            non_blocking=True,
        )

    def prepare_sliding_window(self, infer_state):
        q_token_num = infer_state.input_ids.shape[0]
        assert q_token_num <= self.scratch_token_num
        infer_state.sliding_window_mem_index = torch.arange(
            self.scratch_start,
            self.scratch_start + q_token_num,
            dtype=torch.int64,
            device="cuda",
        )

    def get_layer_kv(self, layer_index: int):
        local_layer = self.sliding_config.get_sliding_layer_index(layer_index)
        layer_buffer = self.sliding_kv_buffer[local_layer]
        head_num = self.sliding_config.sliding_head_num
        return layer_buffer[:, :head_num], layer_buffer[:, head_num:]

    def commit_layer_state(self, layer_index: int, infer_state):
        local_layer = self.sliding_config.get_sliding_layer_index(layer_index)
        commit_sliding_window_state(
            layer_buffer=self.sliding_kv_buffer[local_layer],
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            b_q_seq_len=infer_state.b_q_seq_len,
            b_q_start_loc=infer_state.b_q_start_loc,
            sliding_window=self.sliding_window,
            scratch_start=self.scratch_start,
            max_q_seq_len=infer_state.max_q_seq_len,
        )
