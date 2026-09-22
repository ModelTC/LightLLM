from .base import StateCacheManager
from .layer_cache import LayerCache
from .linear_att import LinearAttCacheConfig, LinearAttCacheManager
from .windowed_mtp import WindowStateCacheManager


def get_hybrid_cache_config(linear_config=None):
    """Return the model-specific layout used by hybrid CPU/disk cache pages."""
    from lightllm.utils.config_utils import is_linear_att_mixed_model
    from lightllm.utils.envs_utils import get_env_start_args
    from .windowed_mtp import WindowedMTPCacheConfig

    args = get_env_start_args()
    if linear_config is None and is_linear_att_mixed_model(args.model_dir):
        linear_config = LinearAttCacheConfig.load_from_args()
    if args.mtp_draft_kv_mode == "window":
        return WindowedMTPCacheConfig.load_from_args(linear_config)
    if linear_config is not None:
        return linear_config
    raise ValueError("No hybrid state-cache layout registered for this model")
