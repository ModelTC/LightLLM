from functools import lru_cache
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


@lru_cache(maxsize=None)
def get_llm_model_class():
    from lightllm.models import get_model_class
    from lightllm.utils.config_utils import get_start_args_model_config

    model_cfg = get_start_args_model_config()
    model_class = get_model_class(model_cfg=model_cfg)
    return model_class
