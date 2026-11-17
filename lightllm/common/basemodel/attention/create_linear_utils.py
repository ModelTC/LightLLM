from lightllm.common.basemodel.attention.linear.flashqla import FlashQlaLinearAttBackend
from lightllm.common.basemodel.attention.linear.kda import KDALinearAttBackend
from lightllm.common.basemodel.attention.linear.replay_gdn import (
    ReplayGDNFlashQlaAttBackend,
    ReplayGDNTritonAttBackend,
)
from lightllm.common.basemodel.attention.linear.replay_kda import ReplayKDALinearAttBackend
from lightllm.common.basemodel.attention.linear.triton import TritonLinearAttBackend
from lightllm.utils.backend_validator import validate
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

gdn_prefill_backends = {
    "flashqla": FlashQlaLinearAttBackend,
    "triton": TritonLinearAttBackend,
}

gdn_replay_prefill_backends = {
    "flashqla": ReplayGDNFlashQlaAttBackend,
    "triton": ReplayGDNTritonAttBackend,
}

gdn_decode_backends = {
    "triton": TritonLinearAttBackend,
}

gdn_replay_decode_backends = {
    "triton": ReplayGDNTritonAttBackend,
}

kda_backends = {
    "triton": KDALinearAttBackend,
}

kda_replay_backends = {
    "triton": ReplayKDALinearAttBackend,
}


def get_gdn_linear_prefill_att_backend_class(index=1, priority_list=("flashqla", "triton")):
    args = get_env_start_args()
    classes = gdn_replay_prefill_backends if args.ssm_state_mode == "replay" else gdn_prefill_backends
    return _select_backend(classes, args.llm_prefill_att_backend, index, priority_list, "prefill")


def get_gdn_linear_decode_att_backend_class(index=1, priority_list=("triton",)):
    args = get_env_start_args()
    classes = gdn_replay_decode_backends if args.ssm_state_mode == "replay" else gdn_decode_backends
    return _select_backend(classes, args.llm_decode_att_backend, index, priority_list, "decode")


def get_kda_linear_att_backend_class():
    args = get_env_start_args()
    classes = kda_replay_backends if args.ssm_state_mode == "replay" else kda_backends
    return classes["triton"]


def _select_backend(classes, backend_args, index, priority_list, stage):
    backend_str = _get_backend_str(backend_args, index)
    if backend_str != "auto":
        return classes[backend_str]

    for backend_name in priority_list:
        if backend_name == "triton" or validate(backend_name):
            if backend_name == "triton":
                logger.info(f"Linear {stage} attention backend: triton.")
            else:
                logger.info(f"Linear {stage} attention backend: {backend_name} (validated).")
            return classes[backend_name]

    logger.warning(f"No linear {stage} attention backend validation succeeded, falling back to Triton.")
    return classes["triton"]


def _get_backend_str(backend_args, index):
    assert index == 1, "linear attention backend index must be 1"
    return backend_args[index] if len(backend_args) > index else "auto"
