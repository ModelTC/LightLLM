"""
Fused QKV projection and GDN gating computation.

This kernel fuses:
1. Linear projection (matmul with weight)
2. Output reorganization (split and reshape)
3. Gating computation (g and beta from a, b)

This reduces kernel launches from 3 to 1 for the QKV+gating path.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _fused_gdn_gating_only_kernel(
    # Output pointers
    g_ptr,
    beta_ptr,
    # Input pointers
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    # Dimensions
    batch_size,
    num_heads,
    # Constants
    beta_const: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    """
    Fused kernel for GDN gating computation with better memory access patterns.

    Computes:
    - g = -exp(A_log) * softplus(a + dt_bias)
    - beta = sigmoid(b)
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)

    batch_offs = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    head_offs = pid_head * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)

    batch_mask = batch_offs < batch_size
    head_mask = head_offs < num_heads
    mask = batch_mask[:, None] & head_mask[None, :]

    # Load A_log and dt_bias (broadcast across batch)
    A_log = tl.load(A_log_ptr + head_offs, mask=head_mask, other=0.0)
    dt_bias = tl.load(dt_bias_ptr + head_offs, mask=head_mask, other=0.0)

    # Load a and b
    offs = batch_offs[:, None] * num_heads + head_offs[None, :]
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)

    # Compute g = -exp(A_log) * softplus(a + dt_bias)
    x = a.to(tl.float32) + dt_bias.to(tl.float32)
    softplus_x = tl.where(beta_const * x <= threshold, (1.0 / beta_const) * tl.log(1.0 + tl.exp(beta_const * x)), x)
    g = -tl.exp(A_log.to(tl.float32)) * softplus_x

    # Compute beta = sigmoid(b)
    beta_out = tl.sigmoid(b.to(tl.float32))

    # Store outputs with layout [1, batch, num_heads]
    out_offs = batch_offs[:, None] * num_heads + head_offs[None, :]
    tl.store(g_ptr + out_offs, g.to(g_ptr.dtype.element_ty), mask=mask)
    tl.store(beta_ptr + out_offs, beta_out.to(beta_ptr.dtype.element_ty), mask=mask)


def _get_fused_gating_configs():
    """Generate autotuning configurations."""
    configs = []
    for block_batch in [1, 4, 8, 16]:
        for block_heads in [8, 16, 32]:
            for num_warps in [2, 4, 8]:
                configs.append(
                    {
                        "BLOCK_BATCH": block_batch,
                        "BLOCK_HEADS": block_heads,
                        "num_warps": num_warps,
                    }
                )
    return configs


def _get_fused_gating_static_key(a: torch.Tensor):
    return {"dtype": str(a.dtype), "num_heads": a.shape[1]}


def _get_fused_gating_run_key(a: torch.Tensor):
    return a.shape[0]


@autotune(
    kernel_name="fused_gdn_gating_v2:v1",
    configs_gen_func=_get_fused_gating_configs,
    static_key_func=_get_fused_gating_static_key,
    run_key_func=_get_fused_gating_run_key,
    mutates_args=["g", "beta"],
)
def fused_gdn_gating_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    beta_const: float = 1.0,
    threshold: float = 20.0,
    run_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized GDN gating with pre-allocated output tensors.

    Args:
        a: Input tensor [batch, num_heads]
        b: Input tensor [batch, num_heads]
        A_log: Log of A parameter [num_heads]
        dt_bias: Bias for dt [num_heads]
        g: Output tensor [1, batch, num_heads] (pre-allocated)
        beta: Output tensor [1, batch, num_heads] (pre-allocated)
        beta_const: Beta constant for softplus (default: 1.0)
        threshold: Threshold for softplus approximation (default: 20.0)
        run_config: Optional autotuning configuration

    Returns:
        Tuple of (g, beta) - same tensors passed in, now filled
    """
    batch_size, num_heads = a.shape

    if run_config is None:
        run_config = {"BLOCK_BATCH": 8, "BLOCK_HEADS": 16, "num_warps": 4}

    grid = (
        triton.cdiv(batch_size, run_config["BLOCK_BATCH"]),
        triton.cdiv(num_heads, run_config["BLOCK_HEADS"]),
    )

    _fused_gdn_gating_only_kernel[grid](
        g,
        beta,
        a,
        b,
        A_log,
        dt_bias,
        batch_size,
        num_heads,
        beta_const,
        threshold,
        run_config["BLOCK_BATCH"],
        run_config["BLOCK_HEADS"],
        num_warps=run_config["num_warps"],
    )

    return g, beta
