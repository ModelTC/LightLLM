from functools import lru_cache
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


@lru_cache(maxsize=None)
def get_llm_model_class():
    from lightllm.models import get_model_class
    from lightllm.utils.config_utils import get_model_config, get_model_paths

    model_cfg = get_model_config(get_model_paths())
    model_class = get_model_class(model_cfg=model_cfg)
    return model_class
