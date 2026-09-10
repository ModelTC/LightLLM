from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import copy_sliding_window_checkpoint
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.state_cache_manager import SlidingWindowStateCacheManager
from lightllm.utils.dist_utils import get_dp_world_size
from lightllm.utils.envs_utils import get_env_start_args

from .hybrid_base import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager import HybridSlidingMemoryManager
    from lightllm.common.state_cache_manager import SlidingWindowCacheConfig
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
        # Each request owns W slots, addressed by absolute position % W on the CPU.
        # Prefill may exchange their physical indices; decode reuses them in place.
        self._sliding_req_indexes = torch.empty(
            (self.max_request_num, self.sliding_window), dtype=torch.int32, device="cpu"
        )
        self._sliding_seq_lens = [0] * self.max_request_num

    def alloc(self):
        req_idx = super().alloc()
        if req_idx is not None:
            self._sliding_req_indexes[req_idx].copy_(self.sliding_mem_manager.alloc(self.sliding_window))
        return req_idx

    def init_hybrid_attention_state(self, req: "InferReq"):
        # The request already owns its window; a cache miss has no valid history.
        self._sliding_seq_lens[req.req_idx] = 0

    def alloc_sliding_window_indexes(self, req_idx: int, token_num: int):
        """先复用请求窗口的空闲槽，再借用本轮额外槽位；返回 CPU 索引片段供 batch 合并。"""
        seq_len = self._sliding_seq_lens[req_idx]
        # The first query needs at most W-1 history tokens, so at least one slot is reusable.
        reuse_num = min(token_num, self.sliding_window - min(seq_len, self.sliding_window - 1))
        ring_start = seq_len % self.sliding_window
        indexes = [self._sliding_req_indexes[req_idx, ring_start : ring_start + reuse_num]]
        if token_num > reuse_num:
            indexes.append(self.sliding_mem_manager.alloc(token_num - reuse_num))
        return indexes

    def update_sliding_window(self, req_idx: int, seq_len: int, new_indexes: torch.Tensor):
        """所有层读取后将借用的尾部槽纳入请求窗口，归还被替换和过期的槽；不移动 KV。"""
        token_num = new_indexes.numel()
        old_seq_len = seq_len - token_num
        reuse_num = min(token_num, self.sliding_window - min(old_seq_len, self.sliding_window - 1))
        if token_num > reuse_num:
            # Retain only borrowed tokens in the final W positions, replacing their old ring slots.
            retain_start = max(reuse_num, token_num - self.sliding_window)
            ring_positions = torch.arange(old_seq_len + retain_start, seq_len, device="cpu") % self.sliding_window
            ring = self._sliding_req_indexes[req_idx]
            expired_indexes = torch.cat((ring[ring_positions], new_indexes[reuse_num:retain_start]))
            self.sliding_mem_manager.free(expired_indexes)
            ring[ring_positions] = new_indexes[retain_start:]
        # Decode only advances the position: no allocator call or window-index copy.
        self._sliding_seq_lens[req_idx] = seq_len

    def free_req(self, free_req_index: int):
        self.sliding_mem_manager.free(self._sliding_req_indexes[free_req_index])
        self._sliding_seq_lens[free_req_index] = 0
        super().free_req(free_req_index)

    def free_all(self):
        self.sliding_mem_manager.free_all()
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
        window_len = min(cache_len, self.sliding_window)
        # Restore into the window reserved by alloc(), ordered by absolute token position.
        ring_positions = torch.arange(cache_len - window_len, cache_len, device="cpu") % self.sliding_window
        indexes = torch.empty(window_len, dtype=torch.int32, device="cpu", pin_memory=True)
        indexes.copy_(self._sliding_req_indexes[req.req_idx, ring_positions])
        self._sliding_seq_lens[req.req_idx] = cache_len
        self.req_to_sliding_window[req.req_idx, cache_len - window_len : cache_len].copy_(indexes, non_blocking=True)
        copy_sliding_window_checkpoint(
            gpu_sliding_kv_buffer=self.sliding_mem_manager.kv_buffer,
            req_to_sliding_window=self.req_to_sliding_window,
            cache_len=cache_len,
            req_idx=req.req_idx,
            cpu_kv_sliding_state=state_cache_manager.get_state_cache(buffer_idx),
            restore=True,
        )

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: SlidingWindowStateCacheManager):
        copy_sliding_window_checkpoint(
            gpu_sliding_kv_buffer=self.sliding_mem_manager.kv_buffer,
            req_to_sliding_window=self.req_to_sliding_window,
            cache_len=self._sliding_seq_lens[req_idx],
            req_idx=req_idx,
            cpu_kv_sliding_state=state_cache_manager.get_state_cache(buffer_idx),
            restore=False,
        )
