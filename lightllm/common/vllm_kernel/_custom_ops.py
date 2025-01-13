from lightllm.utils.log_utils import init_logger
from typing import Callable, List, Optional, Tuple

logger = init_logger(__name__)

try:
    from vllm._custom_ops import *
    from vllm.model_executor.layers.fused_moe import FusedMoE

    select_experts = FusedMoE.select_experts
except ImportError:
    try:
        from lightllm.common.vllm_kernel._ops import *

    except ImportError:
        logger.error("vllm or lightllm_kernel is not installed, you can't use custom ops")

try:
    from lightllm.common.vllm_kernel._ops import init_custom_gather_ar
    from lightllm.common.vllm_kernel._ops import all_gather
    from lightllm.common.vllm_kernel._ops import allgather_dispose
    from lightllm.common.vllm_kernel._ops import allgather_register_buffer
    from lightllm.common.vllm_kernel._ops import allgather_get_graph_buffer_ipc_meta
    from lightllm.common.vllm_kernel._ops import allgather_register_graph_buffers
except ImportError:
    logger.error("lightllm_kernel is not installed, you can't use custom allgather")
