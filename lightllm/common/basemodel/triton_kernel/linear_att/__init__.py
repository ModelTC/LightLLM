"""Linear-attention / GDN triton kernels shared across hybrid models."""

from .causal_conv1d import causal_conv1d_fn
from .causal_conv1d_spec import causal_conv1d_update
from .fused_gdn_gating import fused_gdn_gating
from .gdn_decode_pack import conv_pack_gdn_decode_inputs
from .mtp_fused_recurrent import mtp_fused_recurrent_gated_delta_rule
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
