import torch
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.backend_validator import validate_vit
from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend
from lightllm.common.basemodel.attention_vit.fa3.fp import Fa3VitAttBackend
from lightllm.common.basemodel.attention_vit.triton.fp import TritonVitAttBackend
from lightllm.common.basemodel.attention_vit.sdpa.fp import SdpaVitAttBackend
from lightllm.common.basemodel.attention_vit.xformers.fp import XformersVitAttBackend

logger = init_logger(__name__)


vit_att_backend = {
    "triton": TritonVitAttBackend,
    "sdpa": SdpaVitAttBackend,
    "fa3": Fa3VitAttBackend,
    "xformers": XformersVitAttBackend,
}


def get_vit_att_backend_class(
    index=0, priority_list: list = ["fa3", "xformers", "sdpa", "triton"]
) -> BaseVitAttBackend:
    args = get_env_start_args()
    backend_str = args.vit_att_backend[index]
    if backend_str != "auto":
        logger.info(f"Selected {backend_str} backend for VIT")
        return vit_att_backend[backend_str]
    else:
        return _select_vit_backend(priority_list=priority_list)


def _select_vit_backend(priority_list: list = ["fa3", "xformers", "sdpa", "triton"]) -> type:
    """Auto-select the best available backend with validation for VIT.

    Priority: FA3 > Xformers > Sdpa > Triton
    Each backend is validated in a subprocess with ground truth checks.
    """
    backend_map = vit_att_backend

    for backend_name in priority_list:
        if validate_vit(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated) for VIT")
            return backend_map[backend_name]

    # Fallback to triton without validation (should not happen)
    logger.warning("No backend validation succeeded, falling back to triton")
    return backend_map["triton"]
