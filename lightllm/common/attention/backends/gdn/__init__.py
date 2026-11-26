"""Gated DeltaNet (GDN) linear attention: a causal depthwise conv followed by the gated delta rule.

Unlike softmax attention, GDN keeps per-request recurrent state instead of a KV cache: a conv state
holding the last conv inputs and an SSM state matrix per value head. A backend reads and updates that
state in place; the caller owns it and supplies each layer's views. A forward must therefore run
exactly once per step and layer.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .. import require_positive


@dataclass(frozen=True)
class GDNLayerConfig:
    """Static configuration of GDN layers, per rank.

    Queries and keys have num_k_heads heads of head_k_dim; values have num_v_heads (a multiple of
    num_k_heads) heads of head_v_dim. conv_kernel_dim is the conv width and activation its activation.
    state_mtp_step is the speculative depth the state layout reserves: conv states keep that many extra
    trailing columns, and native SSM states keep state_mtp_step + 1 slots per request. replay=True
    selects replay state: one SSM slot per request, and decode appends history to
    GDNLayerInputs.replay_state instead of updating SSM states.
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_dim: int
    activation: str
    dtype: torch.dtype
    state_mtp_step: int = 0
    replay: bool = False

    def __post_init__(self) -> None:
        require_positive(self, "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim", "conv_kernel_dim")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be a multiple of num_k_heads")
        if self.state_mtp_step < 0:
            raise ValueError("state_mtp_step must be >= 0")


@dataclass(frozen=True)
class GDNLayerInputs:
    """One layer's operands, in token (prefill) or row (decode) order.

    mixed_qkv is [tokens, 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim] before the conv;
    a and b are the raw gate inputs [tokens, num_v_heads] and z the output gate [tokens, num_v_heads,
    head_v_dim], which decode kernels may read. conv_weight/conv_bias, A_log and dt_bias are the
    layer's parameters. conv_states and ssm_states are the layer's state views, indexed by request
    slot, which forward updates in place. A replay-mode decode reads and appends history through
    replay_state and its storage layer for layer_num instead of ssm_states.
    """

    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    z: torch.Tensor
    conv_weight: torch.Tensor
    conv_bias: Optional[torch.Tensor]
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    conv_states: torch.Tensor
    ssm_states: Optional[torch.Tensor] = None
    replay_state: Optional[object] = None
    layer_num: Optional[int] = None


def _split_qkv(layer_config: GDNLayerConfig, mixed_qkv: torch.Tensor):
    key_dim = layer_config.num_k_heads * layer_config.head_k_dim
    value_dim = layer_config.num_v_heads * layer_config.head_v_dim
    return torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
