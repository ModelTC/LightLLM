import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    from flash_attn.cute import flash_attn_varlen_func

    HAS_FA4 = True
except Exception:
    flash_attn_varlen_func = None
    HAS_FA4 = False
    logger.warning("flash-attn-4 is not installed")


def is_fa4_supported_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major in (9, 10, 11, 12)


def ensure_fa4_available() -> None:
    if not HAS_FA4 or flash_attn_varlen_func is None:
        raise ImportError(
            "flash-attn-4 is unavailable. Install it first, e.g. `pip install flash-attn-4`, "
            "or install from the local flash-attention repo."
        )


def ensure_fa4_supported_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("FA4 backend requires CUDA, but CUDA is not available.")
    major, minor = torch.cuda.get_device_capability()
    if major not in (9, 10, 11, 12):
        raise RuntimeError(
            f"FA4 backend requires Hopper/Blackwell-class GPUs (SM90/SM100/SM110/SM120). "
            f"Current device capability is {major}.{minor}."
        )


def unwrap_fa4_output(output):
    return output[0] if isinstance(output, tuple) else output
