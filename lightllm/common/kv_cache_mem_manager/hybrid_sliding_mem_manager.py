import triton

from lightllm.common.req_manager.sliding_window import SlidingWindowStateCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .mem_manager import MemoryManager
from .operator.hybrid_sliding import HybridSlidingMemOperator


class HybridSlidingMemoryManager(MemoryManager):
    """Token-granular full KV plus request-granular sliding-window KV."""

    operator_class = HybridSlidingMemOperator

    def __init__(self, size, sliding_config, always_copy=False, mem_fraction=0.9):
        self.sliding_config = sliding_config
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
        big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        self.hybrid_att_big_page_buffers = SlidingWindowStateCacheManager(
            size=max(1, triton.cdiv(self.size, big_page_token_num)),
            sliding_config=self.sliding_config,
        )
        # Compatibility with the unchanged big/small-page radix implementation.
        self.linear_att_big_page_buffers = self.hybrid_att_big_page_buffers

    def get_att_input_params(self, layer_index: int):
        return super().get_att_input_params(self.sliding_config.get_full_layer_index(layer_index))

    def get_full_cache_layer_index(self, layer_index: int):
        return self.sliding_config.get_full_layer_index(layer_index)

    def _free_buffers(self):
        super()._free_buffers()
        self.hybrid_att_big_page_buffers = None
        self.linear_att_big_page_buffers = None

    def write_to_shm(self, req_manager):
        # Host-side checkpoints are local to the inference process. Excluding
        # them also preserves their pinned allocation during serialization.
        big_page_buffers = self.hybrid_att_big_page_buffers
        legacy_big_page_buffers = self.linear_att_big_page_buffers
        self.hybrid_att_big_page_buffers = None
        self.linear_att_big_page_buffers = None
        try:
            return super().write_to_shm(req_manager)
        finally:
            self.hybrid_att_big_page_buffers = big_page_buffers
            self.linear_att_big_page_buffers = legacy_big_page_buffers
