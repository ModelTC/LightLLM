from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import copy_sliding_window_checkpoint
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.sliding_window_cache_manager import SlidingWindowStateCacheManager
from lightllm.utils.dist_utils import get_dp_world_size
from lightllm.utils.envs_utils import get_env_start_args

from .hybrid_base import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager import HybridSlidingMemoryManager
    from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForSlidingWindow(HybridAttentionReqManager):
    """管理请求的窗口索引与 checkpoint；私有 MemoryManager 负责 GPU KV 和物理槽位。"""

    def __init__(
        self,
        max_request_num: int,
        max_sequence_length: int,
        mem_manager: Optional["HybridSlidingMemoryManager"],
        sliding_config: "SlidingWindowCacheConfig",
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.sliding_config = sliding_config
        self.sliding_window = sliding_config.sliding_window
        # Like linear attention, reserve runtime state before mem_manager profiles full KV capacity.
        self._init_runtime_buffer()

    def _init_runtime_buffer(self):
        args = get_env_start_args()
        # Both microbatches share batch_max_tokens; allow TP/dummy padding for each.
        max_forward_tokens = max(args.batch_max_tokens, self.max_request_num) + 2 * get_dp_world_size()
        self.sliding_mem_manager = MemoryManager(
            size=self.max_request_num * self.sliding_window + max_forward_tokens,
            dtype=self.sliding_config.dtype,
            head_num=self.sliding_config.sliding_head_num,
            head_dim=self.sliding_config.sliding_head_dim,
            layer_num=self.sliding_config.sliding_layer_num,
            publish_usage=False,
        )
        # Absolute-token addressing matches full attention, using a separate physical pool.
        self.req_to_sliding_window = torch.zeros_like(self.req_to_token_indexs)
        self.req_to_sliding_window[self.HOLD_REQUEST_ID].fill_(self.sliding_mem_manager.HOLD_TOKEN_MEMINDEX)
        self._sliding_req_indexes = [torch.empty(0, dtype=torch.int32) for _ in range(self.max_request_num)]
        self._sliding_seq_lens = [0] * self.max_request_num

    def init_hybrid_attention_state(self, req: "InferReq"):
        # A cache miss owns no history slots; the forward allocates only its new tokens.
        self._release_sliding_window(req.req_idx)

    def update_sliding_window(self, req_idx: int, seq_len: int, new_indexes: torch.Tensor):
        """所有层读取后只保留最后 W 个槽位，无需移动 KV 或清空过期映射。"""
        indexes = torch.cat((self._sliding_req_indexes[req_idx], new_indexes))
        expired = max(0, indexes.numel() - self.sliding_window)
        if expired:
            self.sliding_mem_manager.free(indexes[:expired])
        # Retain only the suffix, not the whole forward's pinned index buffer.
        self._sliding_req_indexes[req_idx] = indexes[expired:].clone()
        self._sliding_seq_lens[req_idx] = seq_len

    def _release_sliding_window(self, req_idx: int):
        self.sliding_mem_manager.free(self._sliding_req_indexes[req_idx])
        self._sliding_req_indexes[req_idx] = torch.empty(0, dtype=torch.int32)
        self._sliding_seq_lens[req_idx] = 0

    def free_req(self, free_req_index: int):
        self._release_sliding_window(free_req_index)
        super().free_req(free_req_index)

    def free_all(self):
        self.sliding_mem_manager.free_all()
        self._sliding_req_indexes = [torch.empty(0, dtype=torch.int32) for _ in range(self.max_request_num)]
        self._sliding_seq_lens = [0] * self.max_request_num
        self.req_to_sliding_window.zero_()
        self.req_to_sliding_window[self.HOLD_REQUEST_ID].fill_(self.sliding_mem_manager.HOLD_TOKEN_MEMINDEX)
        super().free_all()

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = SlidingWindowStateCacheManager(size=size, sliding_config=self.sliding_config)
        return self.small_page_buffers

    def restore_state(self, req: "InferReq", state_cache_manager, buffer_idx: int):
        cache_len = req.cur_kv_len
        if req.shared_kv_node is not None:
            # GPU small-page matching restores before updating cur_kv_len;
            # a subsequent CPU-cache load can extend beyond this shared node.
            cache_len = max(cache_len, req.shared_kv_node.node_prefix_total_len)
        self._release_sliding_window(req.req_idx)
        window_len = min(cache_len, self.sliding_window)
        # Own the indices: MemoryManager.alloc() returns a reusable staging-buffer view.
        indexes = torch.empty(window_len, dtype=torch.int32, device="cpu", pin_memory=True)
        indexes.copy_(self.sliding_mem_manager.alloc(window_len))
        self._sliding_req_indexes[req.req_idx] = indexes
        self._sliding_seq_lens[req.req_idx] = cache_len
        self.req_to_sliding_window[req.req_idx, cache_len - window_len : cache_len].copy_(indexes, non_blocking=True)
        copy_sliding_window_checkpoint(
            self.sliding_mem_manager.kv_buffer,
            self.req_to_sliding_window,
            cache_len,
            req.req_idx,
            state_cache_manager.get_state_cache(buffer_idx),
            restore=True,
        )

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: SlidingWindowStateCacheManager):
        copy_sliding_window_checkpoint(
            self.sliding_mem_manager.kv_buffer,
            self.req_to_sliding_window,
            self._sliding_seq_lens[req_idx],
            req_idx,
            state_cache_manager.get_state_cache(buffer_idx),
        )
