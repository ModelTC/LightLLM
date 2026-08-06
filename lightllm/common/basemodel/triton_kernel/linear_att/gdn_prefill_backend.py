import functools

import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
)
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@functools.lru_cache(maxsize=1)
def get_gdn_prefill_chunk_fn():
    backend = getattr(get_env_start_args(), "gdn_prefill_backend", "fla")
    if backend != "flashqla":
        return _fla_chunk_gated_delta_rule

    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:
        logger.warning(
            f"gdn_prefill_backend=flashqla requires Hopper (SM90+), got SM{capability[0]}{capability[1]}; "
            "falling back to the FLA triton kernel."
        )
        return _fla_chunk_gated_delta_rule

    try:
        from flash_qla import chunk_gated_delta_rule as flashqla_chunk_gated_delta_rule
    except Exception as exc:
        logger.warning(
            f"gdn_prefill_backend=flashqla but importing flash_qla failed ({exc!r}); "
            "falling back to the FLA triton kernel. "
            "Install FlashQLA (https://github.com/QwenLM/FlashQLA)."
        )
        return _fla_chunk_gated_delta_rule

    flashqla_initialized = False
    use_flashqla = True

    def flashqla_chunk(q, k, v, **kwargs):
        nonlocal flashqla_initialized, use_flashqla

        if not use_flashqla:
            return _fla_chunk_gated_delta_rule(q=q, k=k, v=v, **kwargs)

        flashqla_q = q.contiguous()
        flashqla_k = k.contiguous()
        flashqla_v = v.contiguous()
        try:
            result = flashqla_chunk_gated_delta_rule(
                q=flashqla_q,
                k=flashqla_k,
                v=flashqla_v,
                **kwargs,
            )
        except Exception as exc:
            if flashqla_initialized:
                raise
            use_flashqla = False
            logger.warning(
                f"FlashQLA failed during its first invocation ({exc!r}); " "falling back to the FLA triton kernel."
            )
            return _fla_chunk_gated_delta_rule(q=q, k=k, v=v, **kwargs)

        flashqla_initialized = True
        return result

    logger.info("GDN chunked-prefill backend: FlashQLA (TileLang, Hopper).")
    return flashqla_chunk
