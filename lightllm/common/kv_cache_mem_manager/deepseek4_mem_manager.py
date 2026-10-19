import math

import torch
import triton

from .mem_manager import MemoryManager
from .operator import DeepseekV4MemOperator
from lightllm.common.state_cache_manager.deepseek4 import (
    DeepseekV4CacheConfig,
    DeepseekV4StateCacheManager,
)
from lightllm.utils.envs_utils import get_env_start_args


class DeepseekV4MemoryManager(MemoryManager):
    """One token-slot capacity, with one byte buffer per history layer.

    SWA and compressor working state belong to ReqManager, like linear state.
    Layer widths include packed KV, indexer K where present, and alignment.
    Model operators expose the physical views needed by their kernels.
    """

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
        args = get_env_start_args()
        big_page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        # Reserve the same CPU-cache load/offload staging slots as the other hybrid models.
        self.big_page_buffers = DeepseekV4StateCacheManager(
            triton.cdiv(size, big_page_tokens) + 2, self.cache_config, keep_num=2
        )
        self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 2
        self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 1

    def _free_buffers(self):
        self.kv_buffer = self.big_page_buffers = None

    def free_all(self):
        super().free_all()
        self.big_page_buffers.clear_to_init_state()

    def get_att_input_params(self, layer_index):
        index = self.cache_config.layer_to_history_idx.get(layer_index)
        return self.kv_buffer[index] if index is not None else None

    def get_index_kv_buffer(self, index):
        return {f"layer_{layer}": buffer[index].unsqueeze(0) for layer, buffer in enumerate(self.kv_buffer)}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        for layer, buffer in enumerate(self.kv_buffer):
            buffer[index] = load_tensor_dict[f"layer_{layer}"][0].to(buffer.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["big_page_buffers"] = None  # Keep process-local pinned checkpoints out of CUDA IPC.
        return state

    def write_to_shm(self, req_manager):
        # Transport receives GPU views, just as Qwen's mem manager exports linear state.
        for name in (
            "swa_buffer",
            "swa_capacity_tokens",
            "HOLD_SWA_SLOT",
            "HOLD_REQUEST_ID",
            "req_to_c4_state",
            "req_to_c4_indexer_state",
            "req_to_c128_state",
        ):
            setattr(self, name, getattr(req_manager, name))
        return super().write_to_shm(req_manager)

    def get_paged_kv_move_buffer_shape(self, page_num, page_size):
        from lightllm.models.deepseek_v4.pd_cache import get_pd_page_nbytes

        return (page_num, 1, 1, 1, get_pd_page_nbytes(self.cache_config, page_size))

    def get_pd_kv_move_buffer_size(self):
        args = get_env_start_args()
        if args.run_mode not in ("prefill", "decode"):
            return 0
        return math.prod(self.get_paged_kv_move_buffer_shape(args.pd_kv_page_num, args.pd_kv_page_size))

    def alloc_paged_kv_move_buffer(self, page_num, page_size):
        self.pd_kv_page_size = page_size
        self.kv_move_buffer = torch.empty(
            self.get_paged_kv_move_buffer_shape(page_num, page_size), dtype=torch.uint8, device="cuda"
        )
        self._buffer_mem_indexes_tensors = [
            torch.empty((page_size,), dtype=torch.int64, device="cpu", pin_memory=True) for _ in range(page_num)
        ]
        return self.kv_move_buffer
