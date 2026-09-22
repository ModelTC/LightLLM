"""Window state configuration and checkpoints shared by hybrid attention caches."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from lightllm.utils.log_utils import init_logger
from lightllm.common.build_utils import repair_config
from .base import StateCacheManager
from .layer_cache import LayerCache


logger = init_logger(__name__)


@dataclass(frozen=True)
class WindowStateCacheConfig:
    layers: int
    kv_heads: int
    head_dim: int
    window: int

    @classmethod
    def from_model_config(cls, config, window, tp_world_size):
        return cls(
            layers=config["n_layer"],
            kv_heads=config["num_key_value_heads"] // tp_world_size,
            head_dim=config.get("head_dim", config["n_embed"] // config["num_attention_heads"]),
            window=window,
        )

    def get_state_shape(self):
        return (self.window, 2 * self.kv_heads, self.head_dim)

    def get_state_bytes(self, element_size):
        return self.layers * self.window * 2 * self.kv_heads * self.head_dim * element_size + 12


def load_window_state_config(args):
    config = json.loads((Path(args.mtp_draft_model_dir[0]) / "config.json").read_text())
    # Match the existing draft initialization order: base aliases, Llama KV
    # defaults, then the overrides applied by Qwen3.5 adapters.
    # Do not normalize again after merging: runtime also retains those aliases.
    merge_nested = config.get("model_type") in ("qwen3_5", "qwen3_5_text")
    repair_config(config, same_names=["num_attention_heads", "n_head"])
    repair_config(config, same_names=["hidden_size", "n_embd", "n_embed"])
    repair_config(config, same_names=["num_hidden_layers", "n_layer"])
    config.setdefault("num_key_value_heads", config["num_attention_heads"])
    if merge_nested:
        config.update(config.get("dflash_config", {}))
    return WindowStateCacheConfig.from_model_config(config, args.mtp_draft_window_size, args.tp // args.dp)


class WindowStateBuffers:
    """Pinned CPU payload; the containing StateCacheManager owns all slot IDs."""

    def __init__(self, size: int, config: WindowStateCacheConfig, dtype: torch.dtype):
        self.config = config
        self.kv = LayerCache(
            size=size,
            dtype=dtype,
            shape=config.get_state_shape(),
            layer_num=config.layers,
            device="cpu",
            size_first=True,
        ).buffer
        self.ends = torch.zeros(size, dtype=torch.int64, device="cpu", pin_memory=True)
        self.counts = torch.zeros(size, dtype=torch.int32, device="cpu", pin_memory=True)
        logger.info(
            f"draft window checkpoint slots={size}, pinned_bytes={size * config.get_state_bytes(dtype.itemsize)}"
        )

    def get_state_cache(self, buffer_idx: int):
        return self.kv[buffer_idx], self.ends[buffer_idx : buffer_idx + 1], self.counts[buffer_idx : buffer_idx + 1]

    def clear_to_init_state(self):
        self.kv.zero_()
        self.ends.zero_()
        self.counts.zero_()


class WindowStateCacheManager(StateCacheManager):
    def __init__(self, size: int, config: WindowStateCacheConfig, dtype: torch.dtype, keep_num: int = 0):
        super().__init__(size, keep_num)
        self.draft_window = WindowStateBuffers(size, config, dtype)

    def get_state_cache(self, buffer_idx: int):
        return self.draft_window.get_state_cache(buffer_idx)

    def clear_to_init_state(self):
        self.draft_window.clear_to_init_state()
        super().clear_to_init_state()


@dataclass
class WindowedMTPCacheConfig:
    """CPU pages contain target KV/state followed by one draft window per TP rank."""

    window_config: WindowStateCacheConfig
    dtype: torch.dtype
    tp_world_size: int
    full_att_all_num_kv_heads: int
    full_att_head_dim: int
    full_att_layer_num: int
    linear_config: object = None

    @classmethod
    def load_from_args(cls, linear_config=None):
        from lightllm.utils.config_utils import get_num_key_value_heads, get_head_dim, get_layer_num
        from lightllm.utils.envs_utils import get_env_start_args
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        args = get_env_start_args()
        return cls(
            window_config=load_window_state_config(args),
            dtype=get_torch_dtype(args.data_type),
            tp_world_size=args.tp // args.dp,
            full_att_all_num_kv_heads=get_num_key_value_heads(args.model_dir),
            full_att_head_dim=get_head_dim(args.model_dir),
            full_att_layer_num=(
                linear_config.get_full_att_kv_layer_num_with_draft_model()
                if linear_config is not None
                else get_layer_num(args.model_dir)
            ),
            linear_config=linear_config,
        )

    def get_cpu_cache_full_att_bytes(self):
        if self.linear_config is not None:
            return self.linear_config.get_cpu_cache_full_att_bytes()
        from lightllm.utils.envs_utils import get_env_start_args

        args = get_env_start_args()
        page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        assert page_tokens == args.cpu_cache_token_page_size
        return (
            page_tokens
            * self.full_att_layer_num
            * 2
            * self.full_att_all_num_kv_heads
            * self.full_att_head_dim
            * self.dtype.itemsize
        )

    def get_cpu_cache_window_offset(self):
        if self.linear_config is not None:
            return self.linear_config.get_cpu_cache_big_page_bytes()
        return (self.get_cpu_cache_full_att_bytes() + 15) // 16 * 16

    def get_cpu_cache_window_rank_bytes(self):
        kv_bytes = self.window_config.get_state_bytes(self.dtype.itemsize) - 12
        # Align int64 ends and each rank's payload; counts is int32.
        return (((kv_bytes + 7) // 8 * 8 + 12) + 15) // 16 * 16

    def get_cpu_cache_big_page_bytes(self):
        return self.get_cpu_cache_window_offset() + self.tp_world_size * self.get_cpu_cache_window_rank_bytes()

    def get_window_views(self, cpu_cache_tensor, tp_rank):
        assert 0 <= tp_rank < self.tp_world_size
        pages = cpu_cache_tensor.view(cpu_cache_tensor.shape[0], -1).view(torch.uint8)
        assert pages.shape[1] == self.get_cpu_cache_big_page_bytes()
        start = self.get_cpu_cache_window_offset() + tp_rank * self.get_cpu_cache_window_rank_bytes()
        kv_bytes = self.window_config.get_state_bytes(self.dtype.itemsize) - 12
        end_offset = start + (kv_bytes + 7) // 8 * 8
        return (
            pages[:, start : start + kv_bytes].view(self.dtype),
            pages[:, end_offset : end_offset + 8].view(torch.int64),
            pages[:, end_offset + 8 : end_offset + 12].view(torch.int32),
        )
