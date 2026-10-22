import math
from dataclasses import dataclass

import torch
import triton

from .base import StateCacheManager


DSV4_HISTORY_TOKEN_ALIGNMENT = 256


@dataclass
class DeepseekV4CacheConfig:
    """Layer mapping and checkpoint shapes for the ordinary hybrid cache path."""

    compress_rates: tuple[int, ...]
    cpu_cache_token_page_size: int = 2048
    head_dim: int = 512
    indexer_head_dim: int = 128
    sliding_window: int = 128
    rope_head_dim: int = 64

    # FlashMLA packing constraints; request capacities and indexes remain token-based.
    swa_token_alignment = 128
    quant_group_dim = 64
    scale_alignment_bytes = 8
    history_token_alignment = DSV4_HISTORY_TOKEN_ALIGNMENT
    short_compress_ratio = 4
    long_compress_ratio = 128

    def __post_init__(self):
        if self.cpu_cache_token_page_size <= 0 or self.cpu_cache_token_page_size % DSV4_HISTORY_TOKEN_ALIGNMENT:
            raise ValueError("DeepSeek-V4 cache pages must be positive multiples of 256 tokens")
        if (self.head_dim, self.rope_head_dim, self.indexer_head_dim) != (512, 64, 128):
            raise ValueError(
                "DeepSeek-V4 FlashMLA packing requires head_dim=512, rope_head_dim=64 and indexer_head_dim=128"
            )
        if self.sliding_window <= 0:
            raise ValueError("DeepSeek-V4 sliding_window must be positive")
        if any(rate not in (0, self.short_compress_ratio, self.long_compress_ratio) for rate in self.compress_rates):
            raise ValueError("DeepSeek-V4 compression ratios must be 0, 4 or 128")
        self.compress_rates = tuple(self.compress_rates)
        self.layer_num = len(self.compress_rates)
        c4_layers = [layer for layer, rate in enumerate(self.compress_rates) if rate == self.short_compress_ratio]
        c128_layers = [layer for layer, rate in enumerate(self.compress_rates) if rate == self.long_compress_ratio]
        self.layer_to_c4_idx = {layer: slot for slot, layer in enumerate(c4_layers)}
        self.layer_to_c128_idx = {layer: slot for slot, layer in enumerate(c128_layers)}
        self.n_c4, self.n_c128 = len(self.layer_to_c4_idx), len(self.layer_to_c128_idx)
        history_layers = [layer for layer, rate in enumerate(self.compress_rates) if rate]
        self.layer_to_history_idx = {layer: slot for slot, layer in enumerate(history_layers)}
        self.kv_data_bytes = self.head_dim + self.rope_head_dim
        self.kv_scale_bytes = (
            triton.cdiv((self.head_dim - self.rope_head_dim) // self.quant_group_dim, self.scale_alignment_bytes)
            * self.scale_alignment_bytes
        )
        self.kv_bytes = self.kv_data_bytes + self.kv_scale_bytes
        self.indexer_bytes = self.indexer_head_dim + torch.float32.itemsize
        self.swa_storage_width = self.packed_kv_bytes(self.swa_token_alignment) // self.swa_token_alignment
        # A token buffer must also admit a packed view with data-row-aligned strides.
        width_alignment = self.kv_data_bytes // math.gcd(self.kv_data_bytes, self.history_token_alignment)
        widths = []
        for layer in history_layers:
            ratio = self.compress_rates[layer]
            entries = self.history_token_alignment // ratio
            nbytes = self.packed_kv_bytes(entries)
            if ratio == self.short_compress_ratio:
                nbytes += entries * self.indexer_bytes
            widths.append(triton.cdiv(nbytes, self.history_token_alignment * width_alignment) * width_alignment)
        self.history_widths = tuple(widths)

    def packed_kv_bytes(self, token_count):
        return triton.cdiv(token_count * self.kv_bytes, self.kv_data_bytes) * self.kv_data_bytes

    @property
    def state_cache_shapes(self):
        return dict(
            swa_shape=(self.layer_num, self.sliding_window, self.kv_bytes),
            c4_shape=(self.n_c4, self.short_compress_ratio, 4 * self.head_dim),
            indexer_shape=(self.n_c4, self.short_compress_ratio, 4 * self.indexer_head_dim),
        )

    def get_cpu_cache_full_att_bytes(self):
        return self.cpu_cache_token_page_size * sum(self.history_widths)

    def get_cpu_cache_big_page_bytes(self):
        # MLA KV and compressor state are replicated across TP ranks.
        state = self.state_cache_shapes
        checkpoint = math.prod(state["swa_shape"]) + torch.float32.itemsize * (
            math.prod(state["c4_shape"]) + math.prod(state["indexer_shape"])
        )
        return self.get_cpu_cache_full_att_bytes() + checkpoint

    @classmethod
    def from_model_config(cls, config, args):
        from lightllm.utils.config_utils import get_deepseek_v4_compress_rates
        from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num

        layers = config.get("n_layer", config.get("num_hidden_layers")) + get_added_mtp_kv_layer_num()
        return cls(
            get_deepseek_v4_compress_rates(config, layers),
            args.cpu_cache_token_page_size or 2048,
            config["head_dim"],
            config["index_head_dim"],
            config["sliding_window"],
            config["qk_rope_head_dim"],
        )

    @classmethod
    def load_from_args(cls):
        from lightllm.utils.config_utils import get_config_json
        from lightllm.utils.envs_utils import get_env_start_args

        args = get_env_start_args()
        return cls.from_model_config(get_config_json(args.model_dir), args)


class DeepseekV4StateCacheManager(StateCacheManager):
    """Pinned continuation checkpoints; compressed history stays in token pages."""

    def __init__(self, size, cache_config: DeepseekV4CacheConfig, keep_num=0):
        super().__init__(size, keep_num)
        shapes = cache_config.state_cache_shapes
        swa_shape, c4_shape, indexer_shape = shapes["swa_shape"], shapes["c4_shape"], shapes["indexer_shape"]
        self.swa_shape = swa_shape
        self.c4_shape = c4_shape
        self.indexer_shape = indexer_shape
        self.swa_nbytes = math.prod(swa_shape)
        self.c4_nbytes = math.prod(c4_shape) * torch.float32.itemsize
        indexer_nbytes = math.prod(indexer_shape) * torch.float32.itemsize
        self.buffer = torch.empty(
            (size, self.swa_nbytes + self.c4_nbytes + indexer_nbytes), dtype=torch.uint8, pin_memory=True
        )

    def get_state_cache(self, buffer_idx):
        return self.view(self.buffer[buffer_idx])

    def view(self, data):
        swa = data[: self.swa_nbytes].view(self.swa_shape)
        indexer_start = self.swa_nbytes + self.c4_nbytes
        c4 = data[self.swa_nbytes : indexer_start].view(torch.float32).view(self.c4_shape)
        indexer = data[indexer_start:].view(torch.float32).view(self.indexer_shape)
        return swa, c4, indexer
