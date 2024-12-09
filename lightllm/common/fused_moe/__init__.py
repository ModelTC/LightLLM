from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    from vllm.model_executor.layers.fused_moe import *
except ImportError:
    try:
        from lightllm.common.fused_moe.fused_moe import *
    except ImportError:
        logger.error("vllm or lightllm_kernel is not installed, you can't use fused_moe")
