import math

import torch
import triton

from .mem_manager import MemoryManager
from .operator import DeepseekV4MemOperator
from lightllm.common.state_cache_manager.deepseek4 import (
    DSV4_CACHE_BLOCK_SIZE,
    DSV4_C4_PAGE_BYTES,
    DSV4_C4_INDEXER_PAGE_BYTES,
    DSV4_C128_PAGE_BYTES,
    DeepseekV4CacheConfig,
    DeepseekV4StateCacheManager,
)
from lightllm.utils.envs_utils import get_env_start_args


class DeepseekV4MemoryManager(MemoryManager):
    """One ordinary allocator and capacity for all compressed history buffers.

    SWA and compressor working state belong to ReqManager, like linear state.
    The C4, indexer and C128 tensors only differ in their packed byte widths.
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
        self.n_c4, self.n_c128 = self.cache_config.n_c4, self.cache_config.n_c128
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

    def get_cell_size(self):
        return self.cache_config.history_block_nbytes / DSV4_CACHE_BLOCK_SIZE

    def profile_size(self, mem_fraction):
        automatic = self.size is None
        super().profile_size(mem_fraction)
        if automatic:
            self.size -= get_env_start_args().page_size
            if self.size <= 0:
                raise RuntimeError("Insufficient memory for compressed KV history and its HOLD page")

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        blocks = (size + self.page_size) // DSV4_CACHE_BLOCK_SIZE
        shapes = (
            (self.n_c4, blocks, DSV4_C4_PAGE_BYTES),
            (self.n_c4, blocks, DSV4_C4_INDEXER_PAGE_BYTES),
            (self.n_c128, blocks, DSV4_C128_PAGE_BYTES),
        )
        section_bytes = tuple(math.prod(shape) for shape in shapes)
        self.history_buffer = torch.zeros(sum(section_bytes), dtype=torch.uint8, device="cuda")
        self.history_offsets = (0, section_bytes[0], section_bytes[0] + section_bytes[1])
        self.kv_buffer = tuple(
            buffer.view(shape) for buffer, shape in zip(self.history_buffer.split(section_bytes), shapes)
        )
        self.c4_kv_buffer, self.c4_indexer_k_buffer, self.c128_kv_buffer = self.kv_buffer
        self.HOLD_TOKEN_MEMINDEX = size
        args = get_env_start_args()
        big_page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        self.big_page_buffers = DeepseekV4StateCacheManager(triton.cdiv(size, big_page_tokens), self.cache_config)

    def _free_buffers(self):
        self.history_buffer = self.kv_buffer = self.big_page_buffers = None
        self.c4_kv_buffer = self.c4_indexer_k_buffer = self.c128_kv_buffer = None

    def free_all(self):
        super().free_all()
        self.big_page_buffers.clear_to_init_state()

    def get_compressed_kv_buffer(self, layer_index):
        rate = self.cache_config.compress_rates[layer_index]
        if rate == 4:
            return self.c4_kv_buffer[self.cache_config.layer_to_c4_idx[layer_index]]
        if rate == 128:
            return self.c128_kv_buffer[self.cache_config.layer_to_c128_idx[layer_index]]
        return None

    def get_att_input_params(self, layer_index):
        kv = self.get_compressed_kv_buffer(layer_index)
        return kv, kv

    def get_indexer_k_buffer(self, layer_index):
        return self.c4_indexer_k_buffer[self.cache_config.layer_to_c4_idx[layer_index]]

    def pack_indexer_k_to_cache(self, layer_index, mem_index, positions, indexer_k):
        from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k_dsv4 import (
            destindex_copy_indexer_k_dsv4,
        )

        if indexer_k.numel():
            destindex_copy_indexer_k_dsv4(
                indexer_k.reshape(-1, self.cache_config.indexer_head_dim),
                mem_index.reshape(-1),
                positions.reshape(-1),
                self.get_indexer_k_buffer(layer_index),
                DSV4_CACHE_BLOCK_SIZE // 4,
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["big_page_buffers"] = None  # Keep process-local pinned checkpoints out of CUDA IPC.
        return state

    def write_to_shm(self, req_manager):
        # Transport receives GPU views, just as Qwen's mem manager exports linear state.
        for name in (
            "swa_buffer",
            "c4_state_buffer",
            "c4_indexer_state_buffer",
            "c128_state_buffer",
            "c4_state_ring",
            "c128_state_ring",
            "swa_pages_per_req",
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
