import torch
import triton

from lightllm.common.state_cache_manager import SlidingWindowStateCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .mem_manager import MemoryManager
from .operator.hybrid_sliding import HybridSlidingMemOperator


class HybridSlidingMemoryManager(MemoryManager):
    """管理 token 粒度的 full KV 和大页 checkpoint，向 attention 提供窗口运行池的引用。"""

    operator_class = HybridSlidingMemOperator
    # Bound by the model to req_manager's runtime pool; no allocation or request-slot ownership here.
    sliding_kv_buffer: torch.Tensor

    def __init__(self, size, sliding_config, always_copy=False, mem_fraction=0.9):
        args = get_env_start_args()
        self.sliding_config = sliding_config
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

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        # Match linear attention: CPU checkpoints plus two reserved tail-transfer slots.
        self.big_page_buffers = SlidingWindowStateCacheManager(
            size=triton.cdiv(size, self.big_page_token_num) + 2,
            sliding_config=self.sliding_config,
            keep_num=2,
        )
        self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 2
        self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 1

    def write_to_shm(self, req_manager):
        # As in Qwen3NextMemManager, keep pickling from replacing pinned CPU
        # checkpoints with ordinary shared storage inaccessible to Triton.
        big_page_buffers = self.big_page_buffers
        self.big_page_buffers = None
        try:
            return super().write_to_shm(req_manager)
        finally:
            self.big_page_buffers = big_page_buffers

    def get_att_input_params(self, layer_index: int):
        if layer_index in self.sliding_config.sliding_layer_to_cache_index:
            layer_buffer = self.sliding_kv_buffer[self.sliding_config.sliding_layer_to_cache_index[layer_index]]
            head_num = self.sliding_config.sliding_head_num
            return layer_buffer[:, :head_num], layer_buffer[:, head_num:]
        return super().get_att_input_params(self.sliding_config.full_layer_to_cache_index[layer_index])

    def _free_buffers(self):
        super()._free_buffers()
        self.big_page_buffers = None
