import torch
import triton

from lightllm.common.sliding_window_cache_manager import SlidingWindowStateCacheManager
from lightllm.utils.dist_utils import get_dp_world_size
from lightllm.utils.envs_utils import get_env_start_args

from .mem_manager import MemoryManager
from .operator.hybrid_sliding import HybridSlidingMemOperator


class HybridSlidingMemoryManager(MemoryManager):
    """Token-granular full KV plus request-granular sliding-window KV."""

    operator_class = HybridSlidingMemOperator

    def __init__(self, size, sliding_config, always_copy=False, mem_fraction=0.9):
        args = get_env_start_args()
        self.sliding_config = sliding_config
        self.sliding_prefill_start = (args.running_max_req_size + 1) * sliding_config.sliding_window
        # Both microbatches share batch_max_tokens; allow TP/dummy padding for each.
        self.max_sliding_prefill_tokens = args.batch_max_tokens + 2 * get_dp_world_size()
        self._sliding_prefill_used = 0
        self._sliding_prefill_batches = 0
        # One layer-first pool: fixed request rings followed by this batch's new KV.
        # Reserve it before profiling how much memory can be given to full attention.
        self.sliding_kv_buffer = torch.zeros(
            (
                sliding_config.sliding_layer_num,
                self.sliding_prefill_start + self.max_sliding_prefill_tokens,
                2 * sliding_config.sliding_head_num,
                sliding_config.sliding_head_dim,
            ),
            dtype=sliding_config.dtype,
            device="cuda",
        )
        self.big_page_token_num = args.linear_att_page_block_num * args.linear_att_hash_page_size
        super().__init__(
            size=size,
            dtype=sliding_config.dtype,
            head_num=sliding_config.full_head_num,
            head_dim=sliding_config.full_head_dim,
            layer_num=sliding_config.full_layer_num,
            always_copy=always_copy,
            mem_fraction=mem_fraction,
        )

    def alloc_sliding_prefill(self, token_num: int) -> torch.Tensor:
        # Overlapping microbatches lease disjoint ranges of the same token budget.
        assert (
            self._sliding_prefill_used + token_num <= self.max_sliding_prefill_tokens
        ), "sliding prefill pool exhausted"
        start = self.sliding_prefill_start + self._sliding_prefill_used
        indexes = torch.arange(start, start + token_num, dtype=torch.int32, device="cuda")
        self._sliding_prefill_used += token_num
        self._sliding_prefill_batches += 1
        return indexes

    def free_sliding_prefill(self):
        assert self._sliding_prefill_batches > 0
        self._sliding_prefill_batches -= 1
        if self._sliding_prefill_batches == 0:
            self._sliding_prefill_used = 0

    def free_all(self):
        super().free_all()
        # Also discard leases when warmup/error cleanup resets all requests.
        self._sliding_prefill_used = 0
        self._sliding_prefill_batches = 0

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        # Match linear attention: CPU checkpoints plus two reserved tail-transfer slots.
        self.linear_att_big_page_buffers = SlidingWindowStateCacheManager(
            size=triton.cdiv(size, self.big_page_token_num) + 2,
            sliding_config=self.sliding_config,
            keep_num=2,
        )
        self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 2
        self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 1

    def write_to_shm(self, req_manager):
        # As in Qwen3NextMemManager, keep pickling from replacing pinned CPU
        # checkpoints with ordinary shared storage inaccessible to Triton.
        big_page_buffers = self.linear_att_big_page_buffers
        self.linear_att_big_page_buffers = None
        try:
            return super().write_to_shm(req_manager)
        finally:
            self.linear_att_big_page_buffers = big_page_buffers

    def get_att_input_params(self, layer_index: int):
        if layer_index in self.sliding_config.sliding_layer_to_cache_index:
            layer_buffer = self.sliding_kv_buffer[self.sliding_config.sliding_layer_to_cache_index[layer_index]]
            head_num = self.sliding_config.sliding_head_num
            return layer_buffer[:, :head_num], layer_buffer[:, head_num:]
        return super().get_att_input_params(self.sliding_config.full_layer_to_cache_index[layer_index])

    def _free_buffers(self):
        super()._free_buffers()
        self.linear_att_big_page_buffers = None
