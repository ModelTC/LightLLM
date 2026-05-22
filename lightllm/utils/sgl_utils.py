import torch
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    import sgl_kernel

    sgl_ops = sgl_kernel
    sgl_allreduce_ops = sgl_ops.allreduce
    HAS_SGL_KERNEL = True
except:
    sgl_ops = None
    sgl_allreduce_ops = None
    HAS_SGL_KERNEL = False
    logger.warning(
        "sgl_kernel is not installed, you can't use the api of it. \
                   You can solve it by running `pip install sgl_kernel`."
    )


def sgl_scaled_fp8_quant_per_token(x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
    """vllm-compatible per-token (per-row) fp8 dynamic quant via sgl_kernel.

    Mirrors ``vllm._custom_ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)``:
    returns (x_q, x_s) with x_s shape (M, 1).
    """
    assert HAS_SGL_KERNEL, "sgl_kernel is required for sgl_scaled_fp8_quant_per_token"
    assert x.ndim == 2, f"expected 2D input, got shape {tuple(x.shape)}"
    x = x.contiguous()
    M = x.shape[0]
    x_q = torch.empty_like(x, dtype=dtype)
    x_s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    sgl_ops.sgl_per_token_quant_fp8(x, x_q, x_s)
    return x_q, x_s


try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    flash_attn_varlen_func = flash_attn_varlen_func
    flash_attn_with_kvcache = flash_attn_with_kvcache
    merge_state_v2 = sgl_ops.merge_state_v2
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    merge_state_v2 = None
    logger.warning(
        "sgl_kernel is not installed, or the installed version did not support fa3. \
        Try to upgrade it."
    )
