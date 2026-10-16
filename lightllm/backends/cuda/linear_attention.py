from lightllm.common.basemodel.attention.linear.flashqla import FlashQlaLinearAttBackend
from lightllm.common.basemodel.attention.linear.triton import TritonLinearAttBackend
from lightllm.utils.backend_validator import validate
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

linear_prefill_att_backend_classes = {
    "flashqla": FlashQlaLinearAttBackend,
    "triton": TritonLinearAttBackend,
}

linear_decode_att_backend_classes = {
    "triton": TritonLinearAttBackend,
}


def get_qwen35_linear_prefill_att_backend_class(index=1, priority_list=("flashqla", "triton")):
    args = get_env_start_args()
    backend_str = _get_backend_str(args.llm_prefill_att_backend, index)
    if backend_str != "auto":
        return linear_prefill_att_backend_classes[backend_str]

    for backend_name in priority_list:
        backend_class = linear_prefill_att_backend_classes[backend_name]
        if backend_name == "triton":
            logger.info("Linear prefill attention backend: triton.")
            return backend_class
        if validate(backend_name):
            logger.info(f"Linear prefill attention backend: {backend_name} (validated).")
            return backend_class

    logger.warning("No linear prefill attention backend validation succeeded, falling back to Triton.")
    return TritonLinearAttBackend


def get_qwen35_linear_decode_att_backend_class(index=1, priority_list=("triton",)):
    args = get_env_start_args()
    backend_str = _get_backend_str(args.llm_decode_att_backend, index)
    if backend_str != "auto":
        return linear_decode_att_backend_classes[backend_str]

    for backend_name in priority_list:
        backend_class = linear_decode_att_backend_classes[backend_name]
        if backend_name == "triton":
            logger.info("Linear decode attention backend: triton.")
            return backend_class
        if validate(backend_name):
            logger.info(f"Linear decode attention backend: {backend_name} (validated).")
            return backend_class

    logger.warning("No linear decode attention backend validation succeeded, falling back to Triton.")
    return TritonLinearAttBackend


def _get_backend_str(backend_args, index):
    assert index == 1, "linear attention backend index must be 1"
    return backend_args[index] if len(backend_args) > index else "auto"
