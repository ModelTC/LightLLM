"""Kimi Delta Attention (KDA) linear attention: a causal depthwise conv followed by a delta rule with
per-channel gates (GLM-5-Next's linear layers).

Like GDN, KDA keeps per-request recurrent state (conv and SSM) that backends update in place; the caller
owns it and supplies each layer's views, so a forward runs exactly once per step and layer.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .. import require_positive

PREFILL_KERNELS = ("triton", "flashkda", "auto")


@dataclass(frozen=True)
class KDALayerConfig:
    """Static configuration of KDA layers, per rank.

    q, k and v each have num_heads heads of head_dim; conv_kernel_size is the conv width. Decay gates
    are clamped below at gate_lower_bound. state_draft_capacity is the speculative depth of the state layout
    (native SSM states keep state_draft_capacity + 1 slots per request). use_ssm_replay=True selects SSM replay state:
    one SSM slot per request, and decode appends history to KDALayerInputs.replay_state. ssm_dtype is
    the dtype of the SSM state.

    prefill_kernel chooses FlashKDA for prefill steps: "triton" never uses it, "flashkda" always does
    and fails if it cannot load, and "auto" uses it for BF16 layers with a BF16 SSM state on SM90,
    for steps no longer than the lengths it was validated on.
    """

    num_heads: int
    head_dim: int
    conv_kernel_size: int
    gate_lower_bound: float
    dtype: torch.dtype
    state_draft_capacity: int = 0
    use_ssm_replay: bool = False
    ssm_dtype: torch.dtype = torch.float32
    prefill_kernel: str = "triton"

    def __post_init__(self) -> None:
        require_positive(self, "num_heads", "head_dim", "conv_kernel_size")
        if self.state_draft_capacity < 0:
            raise ValueError("state_draft_capacity must be >= 0")
        if self.prefill_kernel not in PREFILL_KERNELS:
            raise ValueError(f"prefill_kernel must be one of {PREFILL_KERNELS}, got {self.prefill_kernel!r}")


@dataclass(frozen=True)
class KDALayerInputs:
    """One layer's operands in token (prefill) or row (decode) order.

    mixed_qkv is [tokens, 3 * num_heads * head_dim] before the conv; raw_gate [tokens, num_heads *
    head_dim] and raw_beta [tokens, num_heads] are the raw decay and beta inputs. conv_weight is the
    merged q/k/v conv weight, A_log and dt_bias the gate parameters. conv_states and ssm_states are the
    layer's state views, updated in place; a SSM replay decode appends history through replay_state
    and its storage layer for logical_layer_id instead.
    """

    mixed_qkv: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
    conv_weight: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    conv_states: torch.Tensor
    ssm_states: Optional[torch.Tensor] = None
    replay_state: Optional[object] = None
    logical_layer_id: Optional[int] = None
