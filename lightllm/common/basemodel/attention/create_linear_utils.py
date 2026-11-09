from .import_utils import load_backend_class
from lightllm.utils.backend_validator import validate
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

linear_prefill_att_backend_classes = {
    "flashqla": "linear.flashqla:FlashQlaLinearAttBackend",
    "triton": "linear.triton:TritonLinearAttBackend",
}

linear_decode_att_backend_classes = {
    "triton": "linear.triton:TritonLinearAttBackend",
}


def get_qwen35_linear_prefill_att_backend_class(index=1, priority_list=("flashqla", "triton")):
    args = get_env_start_args()
    backend_str = _get_backend_str(args.llm_prefill_att_backend, index)
    if backend_str != "auto":
        return _linear_backend_class(linear_prefill_att_backend_classes[backend_str], backend_str)

    for backend_name in priority_list:
        backend_class = linear_prefill_att_backend_classes[backend_name]
        if backend_name == "triton":
            logger.info("Linear prefill attention backend: triton.")
            return _linear_backend_class(backend_class, backend_name)
        if validate(backend_name):
            logger.info(f"Linear prefill attention backend: {backend_name} (validated).")
            return _linear_backend_class(backend_class, backend_name)

    logger.warning("No linear prefill attention backend validation succeeded, falling back to Triton.")
    return _linear_backend_class("linear.triton:TritonLinearAttBackend", "triton")


def get_qwen35_linear_decode_att_backend_class(index=1, priority_list=("triton",)):
    args = get_env_start_args()
    backend_str = _get_backend_str(args.llm_decode_att_backend, index)
    if backend_str != "auto":
        return _linear_backend_class(linear_decode_att_backend_classes[backend_str], backend_str)

    for backend_name in priority_list:
        backend_class = linear_decode_att_backend_classes[backend_name]
        if backend_name == "triton":
            logger.info("Linear decode attention backend: triton.")
            return _linear_backend_class(backend_class, backend_name)
        if validate(backend_name):
            logger.info(f"Linear decode attention backend: {backend_name} (validated).")
            return _linear_backend_class(backend_class, backend_name)

    logger.warning("No linear decode attention backend validation succeeded, falling back to Triton.")
    return _linear_backend_class("linear.triton:TritonLinearAttBackend", "triton")


def _linear_backend_class(native_path: str, backend_name: str):
    """Native 仍按路径加载。replay 把 kernel 名字交给独立的 GDN replay 入口。"""

    if getattr(get_env_start_args(), "ssm_state_mode", "native") != "replay":
        return load_backend_class(native_path)
    from lightllm.common.basemodel.attention.linear.replay_gdn import replay_gdn_backend_class

    return replay_gdn_backend_class(backend_name)


def _get_backend_str(backend_args, index):
    assert index == 1, "linear attention backend index must be 1"
    return backend_args[index] if len(backend_args) > index else "auto"
