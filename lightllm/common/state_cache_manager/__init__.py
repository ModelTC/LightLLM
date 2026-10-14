from .base import StateCacheManager
from .layer_cache import LayerCache
from .linear_att import LinearAttCacheConfig, LinearAttCacheManager
from .glm5_next import Glm5NextCacheConfig
from .windowed_mtp import WindowStateCacheManager


def get_hybrid_cache_config(linear_config=None):
    """Return the model-specific layout used by hybrid CPU/disk cache pages."""
    from transformers.configuration_utils import PretrainedConfig
    from lightllm.utils.envs_utils import get_env_start_args
    from .windowed_mtp import WindowedMTPCacheConfig

    args = get_env_start_args()
    if linear_config is None:
        model_cfg, _ = PretrainedConfig.get_config_dict(args.model_dir)
        model_type = model_cfg["model_type"]
        if model_type in ("glm5_next", "glm5_next_text"):
            linear_config = Glm5NextCacheConfig.from_model_config(model_cfg, args)
        elif model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"):
            linear_config = LinearAttCacheConfig.load_from_args()
    if args.mtp_draft_kv_mode == "window":
        return WindowedMTPCacheConfig.load_from_args(linear_config)
    if linear_config is not None:
        return linear_config
    raise ValueError("No hybrid state-cache layout registered for this model")
