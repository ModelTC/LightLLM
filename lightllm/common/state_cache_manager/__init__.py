from .base import StateCacheManager
from .layer_cache import LayerCache
from .linear_att import LinearAttCacheConfig, LinearAttCacheManager
from .glm5_next import Glm5NextCacheConfig


def get_hybrid_cache_config():
    """Return the model-specific layout used by hybrid CPU/disk cache pages."""
    from transformers.configuration_utils import PretrainedConfig
    from lightllm.utils.envs_utils import get_env_start_args

    args = get_env_start_args()
    model_cfg, _ = PretrainedConfig.get_config_dict(args.model_dir)
    model_type = model_cfg["model_type"]
    if model_type in ("glm5_next", "glm5_next_text"):
        return Glm5NextCacheConfig.from_model_config(model_cfg, args)
    if model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"):
        return LinearAttCacheConfig.load_from_args()
    raise ValueError("No hybrid state-cache layout registered for this model")
