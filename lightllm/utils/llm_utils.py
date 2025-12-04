from functools import lru_cache
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


@lru_cache(maxsize=None)
def get_llm_model_class():
    from transformers.configuration_utils import PretrainedConfig

    model_cfg, _ = PretrainedConfig.get_config_dict(get_env_start_args().model_dir)
    from lightllm.models import get_model_class

    model_class = get_model_class(model_cfg=model_cfg)
    return model_class
