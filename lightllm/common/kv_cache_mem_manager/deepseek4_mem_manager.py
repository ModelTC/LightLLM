import math

import torch

from .mem_manager import MemoryManager
from .operator import DeepseekV4MemOperator
from lightllm.common.state_cache_manager.deepseek4 import (
    DeepseekV4CacheConfig,
)
from lightllm.utils.envs_utils import get_env_start_args


class DeepseekV4MemoryManager(MemoryManager):
    """One token-slot capacity, with one byte buffer per history layer.

    SWA and compressor working state belong to DeepseekV4StateManager.
    Layer widths include packed KV, indexer K where present, and alignment.
    Model operators expose the physical views needed by their kernels.
    """

    kv_is_replicated = True

    operator_class = DeepseekV4MemOperator

    def __init__(
        self,
        size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        cache_config: DeepseekV4CacheConfig,
        always_copy=False,
        mem_fraction=0.9,
    ):
        assert head_num == 1 and cache_config.layer_num == layer_num and cache_config.head_dim == head_dim
        self.cache_config = cache_config
        super().__init__(
            size,
            torch.uint8,
            head_num,
            max(cache_config.history_widths, default=0),
            len(cache_config.history_widths),
            always_copy,
            mem_fraction,
        )

    def get_cell_size(self):
        return sum(self.cache_config.history_widths)

    def profile_size(self, mem_fraction):
        automatic = self.size is None
        super().profile_size(mem_fraction)
        if automatic:
            self.size -= get_env_start_args().page_size
            if self.size <= 0:
                raise RuntimeError("Insufficient memory for compressed KV history and its HOLD page")

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = [
            torch.zeros((size + self.page_size, head_num, width), dtype=dtype, device="cuda")
            for width in self.cache_config.history_widths
        ]

    def _free_buffers(self):
        self.kv_buffer = None

    def get_att_input_params(self, layer_index):
        index = self.cache_config.layer_to_history_idx.get(layer_index)
        return self.kv_buffer[index] if index is not None else None

    def get_index_kv_buffer(self, index):
        return {f"layer_{layer}": buffer[index].unsqueeze(0) for layer, buffer in enumerate(self.kv_buffer)}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        for layer, buffer in enumerate(self.kv_buffer):
            buffer[index] = load_tensor_dict[f"layer_{layer}"][0].to(buffer.device)

    def get_transfer_view(self, *, include_page_layout=False):
        from lightllm.common.deepseek4_cache_transfer import DeepseekV4KVTransferView

        return DeepseekV4KVTransferView(tuple(self.kv_buffer), self.cache_config)

    def get_paged_kv_move_buffer_shape(self, page_num, page_size):
        from lightllm.common.deepseek4_cache_transfer import get_pd_page_nbytes

        return (page_num, 1, 1, 1, get_pd_page_nbytes(self.cache_config, page_size))

    def get_pd_kv_move_buffer_size(self):
        args = get_env_start_args()
        if args.run_mode not in ("prefill", "decode"):
            return 0
        return math.prod(self.get_paged_kv_move_buffer_shape(args.pd_kv_page_num, args.pd_kv_page_size))
