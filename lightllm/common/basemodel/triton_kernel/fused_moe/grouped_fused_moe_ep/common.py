import torch
from typing import Any

from lightllm.distributed import dist_group_manager
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
)
from lightllm.utils.device_utils import is_sm100_gpu

from .fp4_mega import use_sm100_mega_moe

SUPPORTED_EP_EXPERT_DTYPES = ("fp8w8a8-b128-deepgemm", "fp4fp8-b32-deepgemm")


def get_ep_num_sms(overlap_with_compute: bool) -> int:
    """根据通信是否与计算重叠，返回对应的 DeepEP SM 配额。"""
    attribute = "ep_num_sms" if overlap_with_compute else "ep_non_overlap_num_sms"
    return getattr(dist_group_manager, attribute, None) or 0


def check_ep_expert_dtype(quant_method: Any):
    expert_dtype = getattr(quant_method, "method_name", None)
    if expert_dtype not in SUPPORTED_EP_EXPERT_DTYPES:
        raise ValueError(
            "EP MoE requires --expert_dtype to be one of ['fp8', 'fp4'], "
            f"but the resolved fused_moe quant method is `{expert_dtype}`. "
            "Please start with --expert_dtype fp8 or --expert_dtype fp4. "
            "Note that --expert_dtype fp4 is only supported on SM100 GPUs."
        )
    if expert_dtype == "fp4fp8-b32-deepgemm" and not is_sm100_gpu():
        raise RuntimeError(
            "--expert_dtype fp4 requires an SM100 GPU for EP MoE; " "please use --expert_dtype fp8 on non-SM100 GPUs."
        )


def quantize_fused_experts_input(
    hidden_states: torch.Tensor,
    w13: Any,
    quant_method: Any,
):
    check_ep_expert_dtype(quant_method)
    if use_sm100_mega_moe(quant_method):
        from deep_gemm.utils import per_token_cast_to_fp8

        return per_token_cast_to_fp8(
            hidden_states,
            use_ue8m0=True,
            gran_k=quant_method.block_size,
            use_packed_ue8m0=True,
        )

    block_size_k = 0
    if w13.weight.ndim == 3:
        block_size_k = w13.weight.shape[2] // w13.weight_scale.shape[2]
    assert block_size_k == 128, "block_size_k must be 128"
    return per_token_group_quant_fp8(hidden_states, block_size_k, dtype=w13.weight.dtype)
