import torch
import dataclasses
import triton

from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.torch_dtype_utils import get_torch_dtype

from .base import StateCacheManager
from .layer_cache import LayerCache


logger = init_logger(__name__)


@dataclasses.dataclass
class LinearAttCacheConfig:
    tp_world_size: int
    # full att 的参数
    full_att_all_num_kv_heads: int
    full_att_dtype: torch.dtype
    full_att_num_kv_heads: int  # 这个是 tp 后的head头数量
    full_att_head_dim: int

    # linear att 的参数
    global_linear_k_heads: int
    global_linear_v_heads: int
    num_linear_k_heads: int
    num_linear_v_heads: int
    head_linear_k_dim: int
    head_linear_v_dim: int
    conv_kernel_size: int
    conv_state_dtype: torch.dtype
    ssm_state_dtype: torch.dtype
    layer_types: tuple[str, ...]
    draft_full_att_kv_layer_num: int = 0
    linear_layer_num: int = dataclasses.field(init=False)
    all_layer_num: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.layer_types = tuple(self.layer_types)
        indices = []
        full_count = linear_count = 0
        for kind in self.layer_types:
            if kind == "linear_attention":
                indices.append(linear_count)
                linear_count += 1
            else:
                indices.append(full_count)
                full_count += 1
        # MTP draft layers only use full-attention KV slots after the main model.
        indices.extend(range(full_count, full_count + self.draft_full_att_kv_layer_num))
        self._layer_indices = tuple(indices)
        self.linear_layer_num = linear_count
        self.all_layer_num = len(self.layer_types)

    def get_linear_state_layer_index(self, layer_index: int) -> int:
        return self._layer_indices[layer_index]

    def get_conv_dim(self):
        # 第一项对应q的参数，第二项对应k的参数，第三项对应v的参数
        # 由于 k_dim = q_dim, k_heads = q_heads, 所以第一项和第二项的计算
        # 形式相同，但是实际内在含义是不同的。
        return (
            self.head_linear_k_dim * self.num_linear_k_heads
            + self.head_linear_k_dim * self.num_linear_k_heads
            + self.head_linear_v_dim * self.num_linear_v_heads
        )

    def get_main_model_full_att_layer_num(self):
        full_att_layer_num = self.all_layer_num - self.linear_layer_num
        return full_att_layer_num

    def get_full_att_kv_layer_num_with_draft_model(self):
        return self.get_main_model_full_att_layer_num() + self.draft_full_att_kv_layer_num

    def get_full_att_kv_layer_index(self, layer_index: int) -> int:
        """Map a global target/draft layer index to the packed full-attention cache."""

        return self._layer_indices[layer_index]

    def get_conv_state_shape(self):
        # Base committed sliding-window state, without speculative MTP tail.
        return (self.get_conv_dim(), self.conv_kernel_size - 1)

    def get_mtp_conv_state_shape(self, mtp_step: int):
        # Working state with room for S speculative tokens before acceptance.
        return (self.get_conv_dim(), (self.conv_kernel_size - 1) + mtp_step)

    def get_ssm_state_shape(self):
        return (self.num_linear_v_heads, self.head_linear_k_dim, self.head_linear_v_dim)

    def get_conv_state_bytes_per_layer(self):
        return self.get_conv_dim() * (self.conv_kernel_size - 1) * self.conv_state_dtype.itemsize

    def get_ssm_state_bytes_per_layer(self):
        return self.num_linear_v_heads * self.head_linear_k_dim * self.head_linear_v_dim * self.ssm_state_dtype.itemsize

    def get_cpu_cache_big_page_bytes(self):
        a = self.get_cpu_cache_full_att_bytes()
        b = self.get_cpu_cache_conv_bytes()
        c = self.get_cpu_cache_ssm_bytes()

        return triton.cdiv(a + b + c, 16) * 16

    def get_cpu_cache_full_att_bytes(self):
        big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        assert big_page_token_num == get_env_start_args().cpu_cache_token_page_size
        full_att_bytes = 2 * self.full_att_all_num_kv_heads * self.full_att_head_dim * self.full_att_dtype.itemsize
        a = full_att_bytes * self.get_full_att_kv_layer_num_with_draft_model() * big_page_token_num
        return a

    def get_cpu_cache_conv_bytes(self):
        b = self.get_conv_state_bytes_per_layer() * self.linear_layer_num * self.tp_world_size
        return b

    def get_cpu_cache_ssm_bytes(self):
        c = self.get_ssm_state_bytes_per_layer() * self.linear_layer_num * self.tp_world_size
        return c

    @staticmethod
    def load_from_args() -> "LinearAttCacheConfig":
        args = get_env_start_args()
        model_path = args.model_dir
        from transformers.configuration_utils import PretrainedConfig

        model_cfg, _ = PretrainedConfig.get_config_dict(model_path)
        llm_config = model_cfg.get("text_config", model_cfg)
        layer_types = llm_config["layer_types"]
        if model_cfg["model_type"] == "sensenova_loop":
            from lightllm.models.sensenova_loop.config import get_loop_layer_types

            layer_types = get_loop_layer_types(model_cfg["loop_settings"], llm_config)
        tp_world_size = args.tp // args.dp
        full_att_dtype = (
            torch.uint8 if args.llm_kv_type in {"fp8kv_sph", "fp8kv_spt"} else get_torch_dtype(args.data_type)
        )
        return LinearAttCacheConfig(
            tp_world_size=tp_world_size,
            full_att_all_num_kv_heads=llm_config["num_key_value_heads"],
            full_att_dtype=full_att_dtype,
            full_att_num_kv_heads=max(1, llm_config["num_key_value_heads"] // tp_world_size),
            full_att_head_dim=llm_config["head_dim"],
            global_linear_k_heads=llm_config["linear_num_key_heads"],
            global_linear_v_heads=llm_config["linear_num_value_heads"],
            num_linear_k_heads=max(1, llm_config["linear_num_key_heads"] // tp_world_size),
            num_linear_v_heads=max(1, llm_config["linear_num_value_heads"] // tp_world_size),
            head_linear_k_dim=llm_config["linear_key_head_dim"],
            head_linear_v_dim=llm_config["linear_value_head_dim"],
            conv_kernel_size=llm_config["linear_conv_kernel_dim"],
            conv_state_dtype=get_torch_dtype(args.data_type),
            ssm_state_dtype=get_torch_dtype(args.linear_att_ssm_data_type),
            layer_types=layer_types,
            draft_full_att_kv_layer_num=get_added_mtp_kv_layer_num(),
        )


class LinearAttCacheManager(StateCacheManager):
    """CPU pinned conv/SSM checkpoint，两个 buffer 均保持 size-first 布局。"""

    def __init__(
        self,
        size: int,
        linear_config: LinearAttCacheConfig,
        keep_num: int = 0,  # 用于记录需要保留的缓存数量，用于支持含有 linear_att 的如qwen3.5 模型的cpu cache的碎页处理。
    ):
        super().__init__(size, keep_num)
        self.linear_config = linear_config
        # init the layer cache
        self.conv_state_cache = LayerCache(
            size=self.size,
            dtype=self.linear_config.conv_state_dtype,
            shape=self.linear_config.get_conv_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cpu",
            size_first=True,
        )
        self.ssm_state_cache = LayerCache(
            size=self.size,
            dtype=self.linear_config.ssm_state_dtype,
            shape=self.linear_config.get_ssm_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cpu",
            size_first=True,
        )
        self.clear_to_init_state()
        return

    def get_state_cache(self, buffer_idx: int):
        return self.conv_state_cache.buffer[buffer_idx, ...], self.ssm_state_cache.buffer[buffer_idx, ...]

    def clear_to_init_state(self):
        self.conv_state_cache.buffer.zero_()
        self.ssm_state_cache.buffer.zero_()
        super().clear_to_init_state()
        return
