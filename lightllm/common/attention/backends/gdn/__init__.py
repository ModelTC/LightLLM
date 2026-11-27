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

    Queries and keys have num_k_heads heads of k_head_dim; values have num_v_heads (a multiple of
    num_k_heads) heads of v_head_dim. conv_kernel_size is the conv width and activation its activation.
    state_draft_capacity is the speculative depth the state layout reserves: conv states keep that many extra
    trailing columns, and native SSM states keep state_draft_capacity + 1 slots per request. use_ssm_replay=True
    selects SSM replay state: one SSM slot per request, and decode appends history to
    GDNLayerInputs.replay_state instead of updating SSM states.
    """

    num_k_heads: int
    num_v_heads: int
    k_head_dim: int
    v_head_dim: int
    conv_kernel_size: int
    activation: str
    dtype: torch.dtype
    state_draft_capacity: int = 0
    use_ssm_replay: bool = False

    def __post_init__(self) -> None:
        require_positive(self, "num_k_heads", "num_v_heads", "k_head_dim", "v_head_dim", "conv_kernel_size")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be a multiple of num_k_heads")
        if self.state_draft_capacity < 0:
            raise ValueError("state_draft_capacity must be >= 0")


@dataclass(frozen=True)
class GDNLayerInputs:
    """One layer's operands, in token (prefill) or row (decode) order.

    mixed_qkv is [tokens, 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim] before the conv;
    a and b are the raw gate inputs [tokens, num_v_heads] and z the output gate [tokens, num_v_heads,
    v_head_dim], which decode kernels may read. conv_weight/conv_bias, A_log and dt_bias are the
    layer's parameters. conv_states and ssm_states are the layer's state views, indexed by request
    slot, which forward updates in place. A SSM replay decode reads and appends history through
    replay_state and its storage layer for logical_layer_id instead of ssm_states.
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
    logical_layer_id: Optional[int] = None


def _split_qkv(layer_config: GDNLayerConfig, mixed_qkv: torch.Tensor):
    key_dim = layer_config.num_k_heads * layer_config.k_head_dim
    value_dim = layer_config.num_v_heads * layer_config.v_head_dim
    return torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
