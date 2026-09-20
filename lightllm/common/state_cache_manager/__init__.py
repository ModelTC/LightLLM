from .base import StateCacheManager
from .layer_cache import LayerCache
from .linear_att import LinearAttCacheConfig, LinearAttCacheManager
from .deepseek4 import DeepseekV4StateCacheManager


def get_hybrid_cache_config():
    """Return the model-specific layout used by hybrid CPU/disk cache pages."""
    from lightllm.utils.config_utils import is_linear_att_mixed_model, get_model_type
    from lightllm.utils.envs_utils import get_env_start_args

    if get_model_type(get_env_start_args().model_dir) == "deepseek_v4":
        from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4CpuCacheLayout

        return DeepseekV4CpuCacheLayout.load_from_args()
    if is_linear_att_mixed_model(get_env_start_args().model_dir):
        return LinearAttCacheConfig.load_from_args()
    raise ValueError("No hybrid state-cache layout registered for this model")
