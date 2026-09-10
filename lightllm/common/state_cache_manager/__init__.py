from .base import StateCacheManager
from .layer_cache import LayerCache
from .linear_att import LinearAttCacheConfig, LinearAttCacheManager
from .sliding_window import SlidingWindowStateCacheManager
from .sliding_window_config import SlidingWindowCacheConfig


def get_hybrid_cache_config():
    """Return the model-specific layout used by hybrid CPU/disk cache pages."""
    from lightllm.utils.config_utils import is_linear_att_mixed_model, is_sliding_att_mixed_model
    from lightllm.utils.envs_utils import get_env_start_args

    model_dir = get_env_start_args().model_dir
    if is_linear_att_mixed_model(model_dir):
        return LinearAttCacheConfig.load_from_args()
    if is_sliding_att_mixed_model(model_dir):
        return SlidingWindowCacheConfig.load_from_args()
    raise ValueError("No hybrid state-cache layout registered for this model")
