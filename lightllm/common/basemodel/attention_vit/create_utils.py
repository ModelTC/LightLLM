from importlib import import_module

from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.backend_validator import _validate
from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend

logger = init_logger(__name__)


vit_att_backend = {
    "triton": "triton.fp:TritonVitAttBackend",
    "sdpa": "sdpa.fp:SdpaVitAttBackend",
    "fa3": "fa3.fp:Fa3VitAttBackend",
    "xformers": "xformers.fp:XformersVitAttBackend",
}


def get_vit_att_backend_class(backend_name: str) -> BaseVitAttBackend:
    module_name, class_name = vit_att_backend[backend_name].split(":")
    return getattr(import_module(f"{__package__}.{module_name}"), class_name)


def init_vit_att_backend(index=0, priority_list: list = ["fa3", "xformers", "sdpa", "triton"]) -> str:
    args = get_env_start_args()
    backend_name = args.vit_att_backend[index]
    if backend_name != "auto":
        logger.info(f"Selected {backend_name} backend for VIT")
        return backend_name
    return _select_vit_backend(priority_list=priority_list)


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
