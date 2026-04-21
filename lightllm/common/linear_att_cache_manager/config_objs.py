import torch
import dataclasses
import triton
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.torch_dtype_utils import get_torch_dtype

logger = init_logger(__name__)


@dataclasses.dataclass
class LinearAttCacheConfig:
    tp_world_size: int
    # full att 的参数
    full_att_dtype: torch.dtype
    full_att_num_kv_heads: int
    full_att_head_dim: int

    # linear att 的参数
    num_linear_k_heads: int
    num_linear_v_heads: int
    head_linear_k_dim: int
    head_linear_v_dim: int
    conv_kernel_size: int
    linear_layer_num: int
    conv_state_dtype: torch.dtype
    ssm_state_dtype: torch.dtype
    full_attention_interval: int
    all_layer_num: int  # 包括 linear att 和 full att 的层加起来的层数

    def get_conv_dim(self):
        return self.head_linear_k_dim * self.num_linear_k_heads * 2 + self.head_linear_v_dim * self.num_linear_v_heads

    def get_conv_state_shape(self):
        return (self.get_conv_dim(), self.conv_kernel_size - 1)

    def get_ssm_state_shape(self):
        return (self.num_linear_v_heads, self.head_linear_k_dim, self.head_linear_v_dim)

    def get_conv_state_bytes(self):
        return self.get_conv_dim() * (self.conv_kernel_size - 1) * self.conv_state_dtype.itemsize

    def get_ssm_state_bytes(self):
        return self.num_linear_v_heads * self.head_linear_k_dim * self.head_linear_v_dim * self.ssm_state_dtype.itemsize

    def get_full_att_bytes(self):
        return 2 * self.full_att_num_kv_heads * self.full_att_head_dim * self.full_att_dtype.itemsize

    def get_cpu_cache_big_page_bytes(self):
        big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        assert big_page_token_num == get_env_start_args().cpu_cache_token_page_size
        a = self.get_full_att_bytes() * (self.all_layer_num - self.linear_layer_num) * big_page_token_num
        b = self.get_conv_state_bytes() * self.linear_layer_num * big_page_token_num
        c = self.get_ssm_state_bytes() * self.linear_layer_num * big_page_token_num

        return triton.cdiv(a + b + c, 16) * 16

    @staticmethod
    def load_from_args() -> "LinearAttCacheConfig":
        args = get_env_start_args()
        model_path = args.model_dir
        from transformers.configuration_utils import PretrainedConfig

        model_cfg, _ = PretrainedConfig.get_config_dict(model_path)
        model_type = model_cfg["model_type"]
        assert model_type in ["qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"]
        llm_config = model_cfg
        try:
            llm_config = llm_config["text_config"]
        except:
            pass

        n_layer = llm_config["num_hidden_layers"]

        tp_world_size = get_env_start_args().tp // get_env_start_args().dp
        return LinearAttCacheConfig(
            tp_world_size=tp_world_size,
            full_att_dtype=get_torch_dtype(args.data_type),
            full_att_num_kv_heads=llm_config["num_attention_heads"] // tp_world_size,
            full_att_head_dim=llm_config["head_dim"],
            num_linear_k_heads=llm_config["linear_num_key_heads"] // tp_world_size,
            num_linear_v_heads=llm_config["linear_num_value_heads"] // tp_world_size,
            head_linear_k_dim=llm_config["linear_key_head_dim"],
            head_linear_v_dim=llm_config["linear_value_head_dim"],
            conv_kernel_size=llm_config["linear_conv_kernel_dim"],
            linear_layer_num=n_layer - (n_layer // llm_config["full_attention_interval"]),
            conv_state_dtype=get_torch_dtype(args.data_type),
            ssm_state_dtype=get_torch_dtype(args.linear_att_ssm_data_type),
            full_attention_interval=llm_config["full_attention_interval"],
            all_layer_num=n_layer,
        )
