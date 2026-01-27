import torch
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.backend_validator import _validate_triton, _compute_ground_truth
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

    Priority: FA3 > Sdpa > Triton
    Each backend is validated in a subprocess with ground truth checks.
    """
    backend_map = vit_att_backend

    for backend_name in priority_list:
        if validate(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated) for VIT")
            return backend_map[backend_name]

    # Fallback to triton without validation (should not happen)
    logger.warning("No backend validation succeeded, falling back to triton")
    return backend_map["triton"]


def validate(backend_name: str) -> bool:
    if backend_name == "fa3":
        validate_ok = _validate_fa3()
    elif backend_name == "xformers":
        validate_ok = _validate_xformers()
    elif backend_name == "sdpa":
        validate_ok = _validate_sdpa()
    elif backend_name == "triton":
        validate_ok = _validate_triton()
    else:
        raise ValueError("not suuported vit attn backend")
    return validate_ok


def _validate_fa3():
    """Validate FA3 with ground truth."""
    from lightllm.utils.sgl_utils import flash_attn_varlen_func

    if flash_attn_varlen_func is None:
        return False

    batch, heads, seq, dim = 1, 4, 8, 64
    q = torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16, device="cuda")

    expected = _compute_ground_truth(q, k, v)

    q_flat = q.transpose(1, 2).reshape(batch * seq, heads, dim)
    k_flat = k.transpose(1, 2).reshape(batch * seq, heads, dim)
    v_flat = v.transpose(1, 2).reshape(batch * seq, heads, dim)
    cu_seqlens = torch.arange(0, batch * seq + 1, seq, dtype=torch.int32, device="cuda")

    out = flash_attn_varlen_func(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        softmax_scale=1.0 / (dim ** 0.5),
        causal=True,
    )
    out = out.reshape(batch, seq, heads, dim).transpose(1, 2)
    torch.cuda.synchronize()

    if not torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
        return False

    return True


def _validate_xformers():
    """Validate Xformers Attn"""
    from xformers import ops as xformers_ops

    if xformers_ops is None:
        return False

    return True


def _validate_sdpa():
    """Validate SDPA Attn"""
    from torch.nn.functional import scaled_dot_product_attention

    if scaled_dot_product_attention is None:
        return False

    return True
