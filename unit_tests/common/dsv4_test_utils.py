"""旧 V4 行为用例的测试夹具；数据操作委托新的 State，不向生产代码添加兼容接口。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import triton

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.attention_state import AttentionStateManager
from lightllm.models.deepseek_v4.config import DSV4AttentionConfig
from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.common.attention_state.bounded_att import FP8E8M0SlidingWindowState, WindowCompressorState
from lightllm.common.attention_state.bounded_att.window_pages_pool import WindowPagesPool
from lightllm.common.attention_state.bounded_att.utils import create_page_pool_tensor
from lightllm.common.attention_state.full_att.compressed_att_state import (
    CompressedAttentionConfig,
    CompressedAttentionState,
)
from lightllm.common.attention_state.allocators import FullAttKVAllocator, StatePageAllocator
from lightllm.common.req_manager import ReqManager

DSV4_COMPRESSED_KV_TOKEN_ALIGNMENT = 256


@dataclass
class DeepseekV4CacheConfig:
    """V4 KV 布局和大小页状态缓存配置。"""

    compress_ratios: tuple[int, ...]
    head_dim: int
    rope_head_dim: int
    indexer_head_dim: int
    sliding_window: int

    # Packed FP8 KV format shared by Flash and Pro.
    quant_group_dim = 64
    scale_alignment_bytes = 8
    # 每个 SWA 物理页的 token 数，用于 KV/scale 打包和 ring 容量对齐。
    swa_page_token_num = 128
    compressed_kv_token_alignment = DSV4_COMPRESSED_KV_TOKEN_ALIGNMENT
    short_compress_ratio = 4
    long_compress_ratio = 128

    def __post_init__(self):
        self.compress_ratios = tuple(self.compress_ratios)
        self.layer_num = len(self.compress_ratios)
        c4_layers = [layer for layer, rate in enumerate(self.compress_ratios) if rate == self.short_compress_ratio]
        c128_layers = [layer for layer, rate in enumerate(self.compress_ratios) if rate == self.long_compress_ratio]
        self.layer_to_c4_idx = {layer: slot for slot, layer in enumerate(c4_layers)}
        self.layer_to_c128_idx = {layer: slot for slot, layer in enumerate(c128_layers)}
        self.n_c4, self.n_c128 = len(self.layer_to_c4_idx), len(self.layer_to_c128_idx)
        compressed_kv_layers = [layer for layer, rate in enumerate(self.compress_ratios) if rate]
        self.layer_to_compressed_kv_idx = {layer: slot for slot, layer in enumerate(compressed_kv_layers)}
        self.kv_data_bytes = self.head_dim + self.rope_head_dim
        self.kv_scale_bytes = (
            triton.cdiv((self.head_dim - self.rope_head_dim) // self.quant_group_dim, self.scale_alignment_bytes)
            * self.scale_alignment_bytes
        )
        self.kv_bytes = self.kv_data_bytes + self.kv_scale_bytes
        self.indexer_bytes = self.indexer_head_dim + torch.float32.itemsize
        self.swa_slot_bytes = self.get_packed_kv_bytes(self.swa_page_token_num) // self.swa_page_token_num
        self.compressed_kv_slot_bytes = self._get_compressed_kv_slot_bytes()

    @classmethod
    def from_model_config(cls, config: Any, args: Any):
        from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num

        layer_num = config["num_hidden_layers"]
        compress_ratios = list(config["compress_ratios"])
        if args.mtp_mode is not None:
            layer_num += get_added_mtp_kv_layer_num()
            # Runtime-added draft layers use sliding-window attention.
            compress_ratios.extend([0] * (layer_num - len(compress_ratios)))
        return cls(
            compress_ratios=tuple(compress_ratios[:layer_num]),
            head_dim=config["head_dim"],
            rope_head_dim=config["qk_rope_head_dim"],
            indexer_head_dim=config["index_head_dim"],
            sliding_window=config["sliding_window"],
        )

    def get_packed_kv_bytes(self, token_count: int):
        return triton.cdiv(token_count * self.kv_bytes, self.kv_data_bytes) * self.kv_data_bytes

    def get_swa_state_shape(self):
        return (self.layer_num, self.sliding_window, self.kv_bytes)

    def get_c4_state_shape(self):
        return (self.n_c4, self.short_compress_ratio, 4 * self.head_dim)

    def get_indexer_state_shape(self):
        return (self.n_c4, self.short_compress_ratio, 4 * self.indexer_head_dim)

    def get_runtime_state_shapes(self, mtp_step: int):
        """每个请求的完整 ring；P/D 使用相同布局，可直接复制槽位。"""
        swa_tokens = triton.cdiv(self.sliding_window + 2 * mtp_step, self.swa_page_token_num) * self.swa_page_token_num
        c4_tokens = 2 * self.short_compress_ratio + mtp_step
        c128_tokens = self.long_compress_ratio + mtp_step
        return (
            (self.layer_num, swa_tokens, self.swa_slot_bytes),
            (self.n_c4, c4_tokens, 4 * self.head_dim),
            (self.n_c4, c4_tokens, 4 * self.indexer_head_dim),
            (self.n_c128, c128_tokens, 2 * self.head_dim),
        )

    def get_cpu_cache_full_att_bytes(self):
        args = get_env_start_args()
        return args.cpu_cache_token_page_size * sum(self.compressed_kv_slot_bytes)

    def get_cpu_cache_big_page_bytes(self):
        # MLA KV and compressor state are replicated across TP ranks.
        swa_bytes = math.prod(self.get_swa_state_shape())
        c4_bytes = math.prod(self.get_c4_state_shape()) * torch.float32.itemsize
        indexer_bytes = math.prod(self.get_indexer_state_shape()) * torch.float32.itemsize
        return self.get_cpu_cache_full_att_bytes() + swa_bytes + c4_bytes + indexer_bytes

    def _get_compressed_kv_slot_bytes(self):
        block_token_num = self.compressed_kv_token_alignment
        # 整块按 KV 数据行对齐，同时能均分到 token 槽位。
        block_bytes_alignment = math.lcm(block_token_num, self.kv_data_bytes)
        slot_bytes_list = []
        for ratio in self.compress_ratios:
            if ratio == 0:
                continue
            compressed_token_num = block_token_num // ratio
            block_bytes = self.get_packed_kv_bytes(compressed_token_num)
            if ratio == self.short_compress_ratio:
                block_bytes += compressed_token_num * self.indexer_bytes
            block_bytes = triton.cdiv(block_bytes, block_bytes_alignment) * block_bytes_alignment
            slot_bytes_list.append(block_bytes // block_token_num)
        return tuple(slot_bytes_list)


def state_config(config: Any):
    return DSV4AttentionConfig(
        logical_layer_ids=tuple(range(config.layer_num)),
        compress_ratios=config.compress_ratios,
        head_dim=config.head_dim,
        rope_head_dim=config.rope_head_dim,
        indexer_head_dim=config.indexer_head_dim,
        sliding_window=config.sliding_window,
    )


class DeepseekV4StateCacheManager:
    def __init__(self, size: int, cache_config: DeepseekV4CacheConfig, keep_num: int = 0):
        # 仅供旧用例保留原联合页视图；生产原子 State 分别拥有自己的页池。
        shapes = (
            cache_config.get_swa_state_shape(),
            cache_config.get_c4_state_shape(),
            cache_config.get_indexer_state_shape(),
        )
        lengths = [
            math.prod(shape) * dtype.itemsize
            for shape, dtype in zip(shapes, (torch.uint8, torch.float32, torch.float32))
        ]
        self.buffer = create_page_pool_tensor(size=(size, sum(lengths)), dtype=torch.uint8)
        offset = 0
        # manager 按规范 State 名排序，联合页依次为 indexer、KV compressor、SWA。
        for name, index, dtype in (
            ("indexer_state", 2, torch.float32),
            ("c4_state", 1, torch.float32),
            ("swa_state", 0, torch.uint8),
        ):
            length = lengths[index]
            setattr(self, name, self.buffer[:, offset : offset + length].view(dtype).view(size, *shapes[index]))
            offset += length
        self.size = size
        self.allocator = StatePageAllocator(size=size, reserved_slots=keep_num)
        self.keep_num = keep_num

    def alloc_one_state_cache(self):
        slots = self.allocator.alloc(1)
        return slots[0] if slots is not None else None

    def alloc_state_cache(self, count: int):
        return self.allocator.alloc(count)

    def free_state_cache(self, slots: list[int]):
        self.allocator.free(slots)

    def get_free_cache_num(self):
        return self.allocator.free_count

    def get_state_cache(self, slot: int):
        return self.swa_state[slot], self.c4_state[slot], self.indexer_state[slot]


class DeepseekV4MemoryManager:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        cache_config: DeepseekV4CacheConfig,
    ):
        self.cache_config = cache_config
        self.page_size = get_env_start_args().page_size
        self.size = size = size // self.page_size * self.page_size
        self.allocator = FullAttKVAllocator(size=size, page_size=self.page_size)
        self.manager = AttentionStateManager(request_capacity=3)
        self.full_states = {}
        self.kv_buffer = []
        config = state_config(cache_config)
        for layer, ratio in enumerate(config.compress_ratios):
            if not ratio:
                continue
            state = CompressedAttentionState(
                layout_config=CompressedAttentionConfig(
                    logical_layer_groups=((layer,),),
                    compress_ratio=ratio,
                    kv_dim=config.head_dim,
                    rope_dim=config.rope_head_dim,
                    block_token_size=config.compressed_kv_token_alignment,
                    indexer_dim=config.indexer_head_dim if ratio == 4 else 0,
                    tp_rank=config.tp_rank,
                    tp_world_size=config.tp_world_size,
                ),
                page_size=self.page_size,
            )
            state.allocate_storage(size)
            self.manager.register_state("main", state)
            self.full_states[layer] = state
            self.kv_buffer.append(state.storage[0])
        big_tokens = get_env_start_args().linear_att_hash_page_size * get_env_start_args().linear_att_page_block_num
        count = triton.cdiv(size, big_tokens)
        self.big_page_buffers = DeepseekV4StateCacheManager(count + 2, cache_config, keep_num=2)
        self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = count
        self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = count + 1
        self.operator = SimpleNamespace(copy_mem_to_mem=self.copy_mem_to_mem)

    def alloc(self, count: int):
        return self.allocator.alloc(count)

    def free(self, indices: torch.Tensor):
        self.allocator.free(indices)

    def get_cell_size(self):
        return sum(state.get_token_bytes() for state in self.full_states.values())

    def get_att_input_params(self, layer: int):
        index = self.cache_config.layer_to_compressed_kv_idx.get(layer)
        return self.kv_buffer[index] if index is not None else None

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        for state in self.full_states.values():
            state.copy_full_att_tokens(src_mem_index, dst_mem_index)

    def alloc_paged_kv_move_buffer(self, count: int, page_size: int):
        from copy import copy

        self.manager = copy(self.manager)
        for original in self.manager.get_bounded_attention_states():
            state = copy(original)
            if isinstance(state, FP8E8M0SlidingWindowState):
                if hasattr(self, "req_to_swa"):
                    state.swa_buffer = self.req_to_swa
            else:
                field = (
                    "req_to_c4_indexer_state"
                    if state.config.target == "indexer"
                    else ("req_to_c4_state" if state.config.window_size == 8 else "req_to_c128_state")
                )
                if hasattr(self, field):
                    state.buffer = getattr(self, field)
            setattr(self.manager, state.state_name, state)
        self.manager._get_states.cache_clear()
        get_env_start_args().pd_kv_page_size = page_size
        self.manager.get_pd_token_page_layouts.cache_clear()
        self.manager.get_pd_runtime_page_layouts.cache_clear()
        self.manager.get_pd_page_bytes.cache_clear()
        self.kv_move_buffer = torch.empty((count, self.manager.get_pd_page_bytes()), dtype=torch.uint8, device="cuda")
        return self.kv_move_buffer

    def get_pd_kv_move_buffer_size(self):
        return self.manager.get_pd_page_bytes() * get_env_start_args().pd_kv_page_num

    def get_paged_kv_move_buffer_shape(self, count: int, page_size: int):
        return (count, self.manager.get_pd_page_bytes())

    def write_mem_to_page_kv_move_buffer(
        self,
        indices: torch.Tensor,
        page: int,
        dp_index: int,
        managers: list[DeepseekV4MemoryManager],
        tp_size: int,
        kind: str = "kv",
        req_idx: int | None = None,
    ):
        for manager in managers[dp_index * tp_size : dp_index * tp_size + 1]:
            manager.manager.write_to_pd_page(
                pd_page_buffers=self.kv_move_buffer,
                pd_page_index=page,
                kind=kind,
                layouts=manager.manager.get_pd_token_page_layouts()
                if kind == "kv"
                else manager.manager.get_pd_runtime_page_layouts(),
                token_indices=torch.tensor(indices, device="cuda", dtype=torch.int64),
                req_idx=req_idx,
                state_token_len=getattr(manager, "state_token_len", 0),
            )

    def read_page_kv_move_buffer_to_mem(
        self,
        indices: torch.Tensor,
        page: int,
        dp_index: int,
        managers: list[DeepseekV4MemoryManager],
        tp_size: int,
        kind: str = "kv",
        req_idx: int | None = None,
    ):
        for manager in managers[dp_index * tp_size : (dp_index + 1) * tp_size]:
            manager.manager.read_from_pd_page(
                pd_page_buffers=self.kv_move_buffer,
                pd_page_index=page,
                kind=kind,
                layouts=manager.manager.get_pd_token_page_layouts()
                if kind == "kv"
                else manager.manager.get_pd_runtime_page_layouts(),
                token_indices=torch.tensor(indices, device="cuda", dtype=torch.int64),
                req_idx=req_idx,
                state_token_len=getattr(manager, "state_token_len", 0),
            )


class DeepseekV4ReqManager(ReqManager):
    def __init__(
        self, count: int, length: int, memory: DeepseekV4MemoryManager | None, config: Any, mtp_step: int = 0
    ):
        super().__init__(count, length)
        self.cache_config = config
        self.states = DeepseekV4TpPartModel.create_bounded_att_states(
            model_config=dict(
                compress_ratios=config.compress_ratios,
                head_dim=config.head_dim,
                qk_rope_head_dim=config.rope_head_dim,
                index_head_dim=config.indexer_head_dim,
                sliding_window=config.sliding_window,
            ),
            args=SimpleNamespace(mtp_step=mtp_step),
            model_logical_layer_ids=tuple(range(config.layer_num)),
            request_capacity=count + 1,
            device="cuda",
        )
        self.swa_state = next(state for state in self.states if isinstance(state, FP8E8M0SlidingWindowState))
        self.config = self.swa_state.config
        self.swa_buffer = self.req_to_swa = self.swa_state.swa_buffer
        shapes = config.get_runtime_state_shapes(mtp_step)
        buffers = []
        for window_size, target, shape in ((8, "kv", shapes[1]), (8, "indexer", shapes[2]), (128, "kv", shapes[3])):
            state = next(
                (
                    item
                    for item in self.states
                    if isinstance(item, WindowCompressorState)
                    and item.config.window_size == window_size
                    and item.config.target == target
                ),
                None,
            )
            buffers.append(
                state.buffer if state is not None else torch.zeros((shape[0], count + 1, *shape[1:]), device="cuda")
            )
        self.req_to_c4_state, self.req_to_c4_indexer_state, self.req_to_c128_state = buffers
        self.window = SimpleNamespace(
            swa_buffer=self.swa_buffer,
            c4_buffer=buffers[0],
            indexer_buffer=buffers[1],
            c128_buffer=buffers[2],
            request_capacity=count + 1,
            swa_capacity_tokens=self.swa_state.swa_capacity_tokens,
            HOLD_SWA_SLOT=self.swa_state.HOLD_SWA_SLOT,
        )
        self.c4_state_tokens = self.req_to_c4_state.shape[2]
        self.c128_state_tokens = self.req_to_c128_state.shape[2]
        self.sliding_window = config.sliding_window
        self.swa_capacity_tokens, self.HOLD_SWA_SLOT = self.window.swa_capacity_tokens, self.window.HOLD_SWA_SLOT
        if memory is not None:
            self.bind_mem_manager(memory)

    def bind_mem_manager(self, memory: DeepseekV4MemoryManager | None):
        self.memory = memory
        self.bind_hold_page(memory.allocator.hold_token_indices)
        if not memory.manager._state_registration_finished:
            for state in self.states:
                memory.manager.register_state("main", state)
            memory.manager.finish_state_registration()
        memory.manager.full_att_allocator = memory.allocator
        memory.manager.bounded_att_big_page_allocator = memory.big_page_buffers.allocator
        memory.manager.bounded_att_cpu_load_big_page_slot = memory.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID
        self._bind_page_views(memory.big_page_buffers, "big")

    def _bind_page_views(self, pool, kind):
        for state in self.states:
            if isinstance(state, WindowCompressorState) and not state.config.overlap_tokens:
                continue
            view = (
                pool.swa_state
                if isinstance(state, FP8E8M0SlidingWindowState)
                else (pool.indexer_state if state.config.target == "indexer" else pool.c4_state)
            )
            atomic_pool = object.__new__(WindowPagesPool)
            atomic_pool.buffer = view
            setattr(state, f"{kind}_pages_pool", atomic_pool)

    def get_compressor_state(self, logical_layer_id, target="kv"):
        return next(
            state
            for state in self.states
            if isinstance(state, WindowCompressorState)
            and state.config.target == target
            and state.contains_logical_layer(logical_layer_id)
        )

    def get_compressed_kv_state(self, logical_layer_id):
        return self.memory.full_states[logical_layer_id]

    def get_compressed_kv_view(self, logical_layer_id: int):
        return self.memory.full_states[logical_layer_id].get_layer_view(logical_layer_id)

    def get_att_input_params(self, layer: int):
        compressed = self.get_compressed_kv_view(layer).kv if layer in self.memory.full_states else None
        return (
            self.req_to_swa[layer].view(-1, self.config.get_packed_kv_bytes(self.config.page_token_size)),
            compressed,
        )

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = DeepseekV4StateCacheManager(size, self.cache_config)
        self._bind_page_views(self.small_page_buffers, "small")
        self.memory.manager.bounded_att_small_page_allocator = self.small_page_buffers.allocator
        return self.small_page_buffers

    def save_state(self, req: Any, slot: int, pool: DeepseekV4StateCacheManager):
        for state in self.states:
            if state.big_pages_pool is not None:
                external = (
                    pool.swa_state
                    if isinstance(state, FP8E8M0SlidingWindowState)
                    else (pool.indexer_state if state.config.target == "indexer" else pool.c4_state)
                )
                state._copy_window(
                    req_idx=req.req_idx,
                    external=external[slot],
                    state_token_len=req.get_chuncked_input_token_len(),
                    write=True,
                )

    def restore_state(self, req: Any, pool: DeepseekV4StateCacheManager, slot: int):
        for state in self.states:
            if state.big_pages_pool is not None:
                external = (
                    pool.swa_state
                    if isinstance(state, FP8E8M0SlidingWindowState)
                    else (pool.indexer_state if state.config.target == "indexer" else pool.c4_state)
                )
                state._copy_window(
                    req_idx=req.req_idx, external=external[slot], state_token_len=req.cur_kv_len, write=False
                )

    def free(self, reqs: list[int], indices: torch.Tensor = None):
        super().free(reqs)
        if indices is not None:
            self.memory.free(indices)
