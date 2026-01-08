from . import (
    MemoryManager,
    CalibrationFP8KVMemoryManager,
    ExportCalibrationMemoryManager,
    PPLINT8KVMemoryManager,
    PPLINT4KVMemoryManager,
    Deepseek2MemoryManager,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.llm_utils import get_llm_model_class
from functools import lru_cache

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def select_mem_manager_class():
    # case 1
    # 先判断是否是 deepseek 系列的模型
    model_class = get_llm_model_class()
    from lightllm.models import Deepseek2TpPartModel

    if issubclass(model_class, Deepseek2TpPartModel):
        mem_class = Deepseek2MemoryManager
        logger.info(f"Model kv cache using default, mem_manager class: {mem_class}")
        return mem_class

    # case normal
    logger.info(f"mode setting params: {get_env_start_args().llm_kv_type}")
    if get_env_start_args().llm_kv_type == "int8kv":
        memory_manager_class = PPLINT8KVMemoryManager
    elif get_env_start_args().llm_kv_type == "int4kv":
        memory_manager_class = PPLINT4KVMemoryManager
    elif get_env_start_args().llm_kv_type == "fp8kv":
        memory_manager_class = ExportCalibrationMemoryManager
    elif get_env_start_args().llm_kv_type == "None":
        memory_manager_class = MemoryManager

    logger.info(f"Model kv cache using mem_manager class: {memory_manager_class}")
    return memory_manager_class


@lru_cache(maxsize=None)
def used_mem_manager_has_scale() -> bool:
    mem_class = select_mem_manager_class()
    return mem_class in [PPLINT8KVMemoryManager, PPLINT4KVMemoryManager]
