from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.config_utils import get_config_json

logger = init_logger(__name__)


def calcu_cpu_cache_page_num() -> int:
    args = get_env_start_args()
    assert args.enable_cpu_cache
    model_config = get_config_json(args.model_dir)
    item_size = 2
    head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"]
    layer_num = model_config["num_hidden_layers"]

    one_token_byte_size = layer_num * num_key_value_heads * head_dim * item_size
    cpu_cache_page_num = int((args.cpu_cache_storage_size * 1024 * 1024 * 1024) / one_token_byte_size)
    return cpu_cache_page_num
