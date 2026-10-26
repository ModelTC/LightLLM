from . import (
    MemoryManager,
    PPLINT8KVMemoryManager,
    PPLINT4KVMemoryManager,
    FP8StaticPerHeadQuantMemManager,
    FP8StaticPerTensorQuantMemManager,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from functools import lru_cache

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def select_mem_manager_class():
    kv_type = get_env_start_args().llm_kv_type
    managers = {
        "int8kv": PPLINT8KVMemoryManager,
        "int4kv": PPLINT4KVMemoryManager,
        "fp8kv_sph": FP8StaticPerHeadQuantMemManager,
        "fp8kv_spt": FP8StaticPerTensorQuantMemManager,
        "None": MemoryManager,
    }
    try:
        memory_manager_class = managers[kv_type]
    except KeyError:
        raise ValueError(f"Unsupported KV cache type: {kv_type}") from None
    logger.info(f"Model kv cache using mem_manager class: {memory_manager_class}")
    return memory_manager_class
