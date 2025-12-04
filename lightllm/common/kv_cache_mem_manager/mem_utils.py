from . import (
    MemoryManager,
    INT8KVMemoryManager,
    CalibrationFP8KVMemoryManager,
    ExportCalibrationMemoryManager,
    PPLINT8KVMemoryManager,
    PPLINT4KVMemoryManager,
    Deepseek2MemoryManager,
    Deepseek2FP8KVMemoryManager,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.llm_utils import get_llm_model_class
from functools import lru_cache

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def select_mem_manager_class():
    mode = get_env_start_args().mode

    # case 1
    # 先判断是否是 deepseek 系列的模型
    model_class = get_llm_model_class()
    from lightllm.models import Deepseek2TpPartModel

    if issubclass(model_class, Deepseek2TpPartModel):
        mem_class = Deepseek2MemoryManager
        if "triton_fp8kv" in mode:
            mem_class = Deepseek2FP8KVMemoryManager

        logger.info(f"Model kv cache using mode {mode}, mem_manager class: {mem_class}")
        return mem_class

    # case normal
    logger.info(f"mode setting params: {mode}")
    if "ppl_int8kv" in mode or "ppl_int8kv_flashdecoding" in mode or "ppl_int8kv_flashdecoding_diverse" in mode:
        memory_manager_class = PPLINT8KVMemoryManager
        logger.info(f"Model kv cache using mode {mode}")
    elif "ppl_int4kv_flashdecoding" in mode:
        memory_manager_class = PPLINT4KVMemoryManager
        logger.info(f"Model kv cache using mode {mode}")
    elif "triton_int8kv" in mode:
        memory_manager_class = INT8KVMemoryManager
        logger.info("Model kv cache using mode triton int8kv")
    elif "triton_fp8kv" in mode:
        raise Exception("currently only for deepseek")
    elif "offline_calibration_fp8kv" in mode:
        memory_manager_class = CalibrationFP8KVMemoryManager
        logger.info("Model kv cache using mode offline calibration fp8kv")
    elif "export_fp8kv_calibration" in mode:
        memory_manager_class = ExportCalibrationMemoryManager
        logger.info("Using mode export fp8kv calibration")
    else:
        memory_manager_class = MemoryManager
        logger.info("Model kv cache using mode normal")
    return memory_manager_class


@lru_cache(maxsize=None)
def used_mem_manager_has_scale() -> bool:
    mem_class = select_mem_manager_class()
    return mem_class in [PPLINT8KVMemoryManager, PPLINT4KVMemoryManager, INT8KVMemoryManager]
