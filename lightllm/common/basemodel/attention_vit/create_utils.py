import torch
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.backend_validator import _validate
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


def get_vit_att_backend_class(backend_name: str) -> BaseVitAttBackend:
    vit_att_backend_class = vit_att_backend[backend_name]
    return vit_att_backend_class


def init_vit_att_backend(index=0, priority_list: list = ["fa3", "xformers", "sdpa", "triton"]) -> str:
    args = get_env_start_args()
    backend_name = args.vit_att_backend[index]
    supported = _platform_supported_backends()
    if backend_name != "auto":
        if supported is not None and backend_name not in supported:
            raise ValueError(
                f"VIT attention backend {backend_name} is not supported on this platform, "
                f"supported backends: {list(supported)}"
            )
        logger.info(f"Selected {backend_name} backend for VIT")
        return backend_name
    elif supported is not None:
        # The subprocess validation only checks cuda kernels, so pick the first backend the platform supports.
        for backend_name in priority_list:
            if backend_name in supported:
                logger.info(f"Auto-selected {backend_name} backend (platform supported) for VIT")
                return backend_name
        raise ValueError(f"No VIT attention backend in {priority_list} is supported on this platform")
    else:
        return _select_vit_backend(priority_list=priority_list)


def _platform_supported_backends():
    """Backend names the hardware platform can run, None means no restriction (cuda)."""
    from lightllm.platform import get_hardware_backend

    return get_hardware_backend().runtime.supported_att_backends


def _select_vit_backend(priority_list: list = ["fa3", "xformers", "sdpa", "triton"]) -> str:
    """Auto-select the best available backend with validation for VIT.

    Priority: FA3 > Xformers > Sdpa > Triton
    Each backend is validated in a subprocess with ground truth checks.
    """

    for backend_name in priority_list:
        if _validate(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated) for VIT")
            return backend_name

    # Fallback to triton without validation (should not happen)
    logger.warning("No backend validation succeeded, falling back to triton")
    return "triton"
