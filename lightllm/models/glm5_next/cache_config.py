import dataclasses

from lightllm.common.state_cache_manager import LinearAttCacheConfig
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.torch_dtype_utils import get_torch_dtype


@dataclasses.dataclass
class Glm5NextCacheConfig(LinearAttCacheConfig):
    """Replicated MLA/index KV plus TP-sharded KDA checkpoints."""

    MLA_PADDING = 64  # Existing FlashMLA/FA3 kernels consume a 576-wide MLA key.
    INDEX_PADDING_BYTES = 144

    @classmethod
    def from_model_config(cls, config, args):
        config = config.get("text_config", config)
        linear = config["linear_attn_config"]
        tp = args.tp // args.dp
        assert linear["num_heads"] % tp == 0
        layers = config["num_hidden_layers"]
        assert config["layer_types"] == [
            "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)
        ]
        dtype = get_torch_dtype(args.data_type)
        # Keep raw index keys and compression scores with each token, so a
        # pool spanning a prefill chunk/cache hit never needs hidden tail state.
        packed_dim = (
            config["kv_lora_rank"]
            + cls.MLA_PADDING
            + 2 * config["index_head_dim"]
            + cls.INDEX_PADDING_BYTES // dtype.itemsize
        )
        return cls(
            tp_world_size=tp,
            full_att_all_num_kv_heads=1,
            full_att_dtype=dtype,
            full_att_num_kv_heads=1,
            full_att_head_dim=packed_dim,
            global_linear_k_heads=linear["num_heads"],
            global_linear_v_heads=linear["num_heads"],
            num_linear_k_heads=linear["num_heads"] // tp,
            num_linear_v_heads=linear["num_heads"] // tp,
            head_linear_k_dim=linear["head_dim"],
            head_linear_v_dim=linear["head_dim"],
            conv_kernel_size=linear["short_conv_kernel_size"],
            linear_layer_num=len(linear["kda_layers"]),
            conv_state_dtype=dtype,
            ssm_state_dtype=get_torch_dtype(args.linear_att_ssm_data_type),
            full_attention_interval=4,
            all_layer_num=layers,
        )

    def get_cpu_cache_full_att_bytes(self):
        args = get_env_start_args()
        page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        assert page_tokens == args.cpu_cache_token_page_size
        return (
            self.full_att_head_dim
            * self.full_att_dtype.itemsize
            * self.get_main_model_full_att_layer_num()
            * page_tokens
        )
