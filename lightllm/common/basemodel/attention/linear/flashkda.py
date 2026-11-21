# SPDX-License-Identifier: Apache-2.0

"""Choose the KDA prefill kernel: FlashKDA when requested or automatically usable, else Triton."""

import torch

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


def get_flash_kda_fwd(model):
    backends = model.args.llm_prefill_att_backend
    backend = backends[1] if len(backends) > 1 else "auto"
    if backend == "triton":
        return None
    if backend not in ("auto", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}; use auto, triton, or flashkda")

    if backend == "auto" and (
        model.data_type != torch.bfloat16
        or model.args.linear_att_ssm_data_type != "bfloat16"
        or torch.version.hip is not None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        return None

    try:
        from flash_kda import fwd
    except (ImportError, OSError) as exc:
        if backend == "auto":
            logger.info("KDA prefill uses Triton because FlashKDA could not be loaded: %s", exc)
            return None
        raise ImportError(
            "The flashkda prefill backend requires FlashKDA. Install it with "
            "`pip install --no-build-isolation git+https://github.com/MoonshotAI/FlashKDA.git` "
            "and build it for the current GPU and PyTorch/CUDA versions."
        ) from exc
    logger.info("KDA prefill attention backend: %s.", "automatic FlashKDA/Triton" if backend == "auto" else "flashkda")
    return fwd
