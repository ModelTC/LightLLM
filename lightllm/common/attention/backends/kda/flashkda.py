# SPDX-License-Identifier: Apache-2.0

"""Load FlashKDA's prefill kernel for the KDA prefill step builder, or fall back to Triton."""

from typing import Callable, Optional

import torch

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


def import_flash_kda_fwd() -> Callable:
    """Import FlashKDA's fwd; the ImportError says how to install it."""
    try:
        from flash_kda import fwd
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The flashkda prefill backend requires FlashKDA. Install it with "
            "`pip install --no-build-isolation git+https://github.com/MoonshotAI/FlashKDA.git` "
            "and build it for the current GPU and PyTorch/CUDA versions."
        ) from exc
    return fwd


def load_flash_kda_fwd(requested: str, *, dtype: torch.dtype, ssm_dtype: torch.dtype) -> Optional[Callable]:
    """Return FlashKDA's fwd for the requested prefill provider, or None to use Triton.

    requested is "auto", "flashkda" or "triton". "auto" uses FlashKDA only for BF16 models with a
    BF16 SSM state on SM90 when it loads; "flashkda" raises if it cannot be loaded.
    """
    if requested == "triton":
        return None
    if requested not in ("auto", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {requested}; use auto, triton, or flashkda")
    if requested == "auto" and (
        dtype != torch.bfloat16
        or ssm_dtype != torch.bfloat16
        or torch.version.hip is not None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        return None
    try:
        fwd = import_flash_kda_fwd()
    except ImportError as exc:
        if requested == "auto":
            logger.info("KDA prefill uses Triton because FlashKDA could not be loaded: %s", exc.__cause__)
            return None
        raise
    logger.info(
        "KDA prefill attention backend: %s.", "automatic FlashKDA/Triton" if requested == "auto" else "flashkda"
    )
    return fwd
