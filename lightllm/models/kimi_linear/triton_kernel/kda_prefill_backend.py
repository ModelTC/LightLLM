import functools

import torch

from lightllm.models.kimi_linear.triton_kernel.fla.ops import (
    chunk_kda_with_fused_gate as _fla_chunk_kda,
)
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

FLASH_KDA_GATE_LOWER_BOUND = -5.0


def _flashkda_cuda_version_supported():
    cuda_version = torch.version.cuda
    if cuda_version is None:
        return False
    try:
        major, minor = (int(part) for part in cuda_version.split(".")[:2])
    except (TypeError, ValueError):
        return False
    return (major, minor) >= (12, 9)


def _triton_kda_chunk(**kwargs):
    return _fla_chunk_kda(**kwargs)


@torch.no_grad()
def _flash_kda_chunk(
    *,
    flash_kda,
    q,
    k,
    v,
    raw_g,
    beta,
    A_log,
    g_bias,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
    cu_seqlens=None,
    **kwargs,
):
    if not use_qk_l2norm_in_kernel:
        raise ValueError("FlashKDA requires use_qk_l2norm_in_kernel=True")
    if not use_beta_sigmoid_in_kernel:
        raise ValueError("FlashKDA requires raw beta logits and use_beta_sigmoid_in_kernel=True")
    if not safe_gate or lower_bound is None:
        raise ValueError("FlashKDA requires safe_gate=True and a lower_bound")
    if g_bias is None:
        raise ValueError("FlashKDA requires the KDA gate bias")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    raw_g = raw_g.contiguous()
    beta = beta.contiguous()
    A_log = A_log.reshape(-1).contiguous()
    g_bias = g_bias.reshape(q.shape[2], q.shape[3]).contiguous()
    initial_state = initial_state.contiguous() if initial_state is not None else None
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(dtype=torch.long).contiguous()

    if scale is None:
        scale = q.shape[-1] ** -0.5

    output = torch.empty_like(v)
    final_state = None
    if output_final_state:
        num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        state_dtype = torch.float32 if initial_state is None else initial_state.dtype
        final_state = torch.empty(
            num_sequences,
            v.shape[2],
            v.shape[3],
            q.shape[3],
            dtype=state_dtype,
            device=q.device,
        )

    flash_kda.fwd(
        q,
        k,
        v,
        raw_g,
        beta,
        scale,
        output,
        A_log=A_log,
        dt_bias=g_bias,
        lower_bound=lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
    return output, final_state


@functools.lru_cache(maxsize=4)
def get_kda_prefill_chunk_fn(head_dim: int, data_type: torch.dtype):
    backend = getattr(get_env_start_args(), "kda_prefill_backend", "auto")
    if backend == "fla":
        return _triton_kda_chunk

    if head_dim != 128:
        logger.warning(f"FlashKDA requires head_dim=128, got {head_dim}; falling back to the FLA Triton kernel.")
        return _triton_kda_chunk
    if data_type != torch.bfloat16:
        logger.warning(f"FlashKDA requires bfloat16, got {data_type}; falling back to the FLA Triton kernel.")
        return _triton_kda_chunk
    if not torch.cuda.is_available():
        logger.warning("FlashKDA requires CUDA; falling back to the FLA Triton kernel.")
        return _triton_kda_chunk
    if not _flashkda_cuda_version_supported():
        logger.warning(
            f"FlashKDA requires CUDA 12.9+, but PyTorch uses CUDA {torch.version.cuda}; "
            "falling back to the FLA Triton kernel."
        )
        return _triton_kda_chunk

    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:
        logger.warning(
            f"FlashKDA requires SM90+, got SM{capability[0]}{capability[1]}; " "falling back to the FLA Triton kernel."
        )
        return _triton_kda_chunk

    try:
        import flash_kda
    except Exception as exc:
        logger.warning(
            f"Importing flash_kda failed ({exc!r}); falling back to the FLA Triton kernel. "
            "Install FlashKDA from https://github.com/MoonshotAI/FlashKDA."
        )
        return _triton_kda_chunk

    logger.info("KDA chunked-prefill backend: FlashKDA (CUTLASS, SM90+).")
    return functools.partial(_flash_kda_chunk, flash_kda=flash_kda)
