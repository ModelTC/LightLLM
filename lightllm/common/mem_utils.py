from lightllm.common.mem_manager import MemoryManager
from lightllm.common.int8kv_mem_manager import INT8KVMemoryManager
from lightllm.common.calibration_fp8kv_mem_manager import CalibrationFP8KVMemoryManager
from lightllm.common.export_calibration_mem_manager import ExportCalibrationMemoryManager
from lightllm.common.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager
from lightllm.common.ppl_int4kv_mem_manager import PPLINT4KVMemoryManager
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def select_mem_manager_class(mode):
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
