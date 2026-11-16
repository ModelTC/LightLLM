"""Temporary legacy import shim for the attention migration.

New code imports lightllm.common.attention.backends directly. Removed with the
runtime integration in PR #1721 (5/5), after remaining model callers migrate.
"""

from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn
from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import causal_conv1d_update
from lightllm.common.attention.backends.gdn.triton.kernels.fused_gdn_gating import fused_gdn_gating
from lightllm.common.attention.backends.gdn.triton.kernels.gdn_decode_pack import conv_pack_gdn_decode_inputs
from lightllm.common.attention.backends.gdn.triton.kernels.mtp_fused_recurrent import (
    mtp_fused_recurrent_gated_delta_rule,
)

from .fla.ops import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "fused_gdn_gating",
    "conv_pack_gdn_decode_inputs",
    "mtp_fused_recurrent_gated_delta_rule",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
