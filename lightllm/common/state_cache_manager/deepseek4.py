import math
from dataclasses import dataclass

import torch
import triton

from lightllm.utils.envs_utils import get_env_start_args

from .base import StateCacheManager


DSV4_COMPRESSED_KV_TOKEN_ALIGNMENT = 256


@dataclass
class DeepseekV4CacheConfig:
    """V4 KV 布局和大小页状态缓存配置。"""

    compress_rates: tuple[int, ...]
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
        self.compress_rates = tuple(self.compress_rates)
        self.layer_num = len(self.compress_rates)
        c4_layers = [layer for layer, rate in enumerate(self.compress_rates) if rate == self.short_compress_ratio]
        c128_layers = [layer for layer, rate in enumerate(self.compress_rates) if rate == self.long_compress_ratio]
        self.layer_to_c4_idx = {layer: slot for slot, layer in enumerate(c4_layers)}
        self.layer_to_c128_idx = {layer: slot for slot, layer in enumerate(c128_layers)}
        self.n_c4, self.n_c128 = len(self.layer_to_c4_idx), len(self.layer_to_c128_idx)
        compressed_kv_layers = [layer for layer, rate in enumerate(self.compress_rates) if rate]
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
    def from_model_config(cls, config, args):
        from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num

        layer_num = config["num_hidden_layers"]
        compress_rates = list(config["compress_ratios"])
        if args.mtp_mode is not None:
            layer_num += get_added_mtp_kv_layer_num()
            # Runtime-added draft layers use sliding-window attention.
            compress_rates.extend([0] * (layer_num - len(compress_rates)))
        return cls(
            compress_rates=tuple(compress_rates[:layer_num]),
            head_dim=config["head_dim"],
            rope_head_dim=config["qk_rope_head_dim"],
            indexer_head_dim=config["index_head_dim"],
            sliding_window=config["sliding_window"],
        )

    def get_packed_kv_bytes(self, token_count):
        return triton.cdiv(token_count * self.kv_bytes, self.kv_data_bytes) * self.kv_data_bytes

    def _get_compressed_kv_slot_bytes(self):
        block_token_num = self.compressed_kv_token_alignment
        # 整块按 KV 数据行对齐，同时能均分到 token 槽位。
        block_bytes_alignment = math.lcm(block_token_num, self.kv_data_bytes)
        slot_bytes_list = []
        for ratio in self.compress_rates:
            if ratio == 0:
                continue
            compressed_token_num = block_token_num // ratio
            block_bytes = self.get_packed_kv_bytes(compressed_token_num)
            if ratio == self.short_compress_ratio:
                block_bytes += compressed_token_num * self.indexer_bytes
            block_bytes = triton.cdiv(block_bytes, block_bytes_alignment) * block_bytes_alignment
            slot_bytes_list.append(block_bytes // block_token_num)
        return tuple(slot_bytes_list)

    def get_swa_state_shape(self):
        return (self.layer_num, self.sliding_window, self.kv_bytes)

    def get_c4_state_shape(self):
        return (self.n_c4, self.short_compress_ratio, 4 * self.head_dim)

    def get_indexer_state_shape(self):
        return (self.n_c4, self.short_compress_ratio, 4 * self.indexer_head_dim)

    def get_runtime_state_shapes(self, mtp_step):
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


class DeepseekV4StateCacheManager(StateCacheManager):
    """Pinned continuation checkpoints; compressed KV stays in token pages."""

    def __init__(self, size, cache_config: DeepseekV4CacheConfig, keep_num=0, shm_name=None, shm_create=False):
        super().__init__(size, keep_num, shm_name, shm_create)
        self.cache_config = cache_config
        self.swa_shape = cache_config.get_swa_state_shape()
        self.c4_shape = cache_config.get_c4_state_shape()
        self.indexer_shape = cache_config.get_indexer_state_shape()
        self.swa_nbytes = math.prod(self.swa_shape)
        self.c4_nbytes = math.prod(self.c4_shape) * torch.float32.itemsize
        indexer_nbytes = math.prod(self.indexer_shape) * torch.float32.itemsize
        state_nbytes = self.swa_nbytes + self.c4_nbytes + indexer_nbytes
        if shm_name is None:
            self.buffer = torch.empty((size, state_nbytes), dtype=torch.uint8, pin_memory=True)
        else:
            from .layer_cache import LayerCache

            self.layer_cache = LayerCache(
                size=size,
                dtype=torch.uint8,
                shape=(state_nbytes,),
                layer_num=1,
                device="cpu",
                size_first=True,
                shm_name=shm_name + "_states",
                shm_create=shm_create,
            )
            self.buffer = self.layer_cache.buffer[:, 0]

    def get_state_cache(self, buffer_idx):
        data = self.buffer[buffer_idx]
        swa = data[: self.swa_nbytes].view(self.swa_shape)
        indexer_start = self.swa_nbytes + self.c4_nbytes
        c4 = data[self.swa_nbytes : indexer_start].view(torch.float32).view(self.c4_shape)
        indexer = data[indexer_start:].view(torch.float32).view(self.indexer_shape)
        return swa, c4, indexer
