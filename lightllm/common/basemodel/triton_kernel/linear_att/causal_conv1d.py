# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/mamba/causal_conv1d.py

from typing import Optional

import torch


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
    **kwargs,
):
    """Run causal depthwise convolution with SGL Kernel or a local Triton fallback.

    x is [batch, channels, tokens] or packed [channels, total_tokens]. The Triton
    path requires packed inputs and all four request/state tensors below.
    weight is [channels, width]; optional bias is [channels].

    query_start_loc: cumulative token offsets [batch + 1], starting at zero.
    cache_indices: maps each request to a slot in conv_states; pad_slot_id
        entries keep their input and state unchanged.
    has_initial_state: one bool per request; false means a zero-filled prefix.
    conv_states: [slots, channels, width - 1], updated in place with input tails.

    activation is None, "silu" or "swish". The optional max_seqlen keyword bounds
    the longest packed query for Triton without reading GPU lengths on the CPU.
    Returns the same shape as x. SGL may overwrite x; Triton allocates an output.
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    try:
        from sgl_kernel import causal_conv1d_fwd
    except ImportError:
        from .causal_conv1d_triton import causal_conv1d_fn as triton_causal_conv1d_fn

        return triton_causal_conv1d_fn(
            x,
            weight,
            bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            conv_states=conv_states,
            activation=activation,
            pad_slot_id=pad_slot_id,
            **kwargs,
        )

    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ["silu", "swish"],
        pad_slot_id,
    )
    return x


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
):
    """Decode with SGL Kernel when available, otherwise the local Triton kernel.

    x is [batch, channels] or [batch, channels, tokens]. conv_state is
    [slots, channels, state_len], where state_len >= width - 1, and is updated
    in place. weight is [channels, width]; optional bias is [channels].

    cache_seqlens supplies one cursor per request for circular state buffers;
    new tokens are written starting at cursor % state_len. Without cursors,
    states shift left and append the new tokens.
    conv_state_indices optionally maps requests to cache slots. Entries equal
    to pad_slot_id keep their input and state unchanged.

    activation is None, "silu" or "swish". Returns the same shape as x.
    SGL updates x in place; Triton allocates an output.
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(f"activation must be None, silu, or swish, actual: {activation}")
    try:
        from sgl_kernel import causal_conv1d_update as causal_conv1d_update_kernel
    except ImportError:
        from .causal_conv1d_triton import causal_conv1d_update as triton_causal_conv1d_update

        return triton_causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            activation=activation,
            cache_seqlens=cache_seqlens,
            conv_state_indices=conv_state_indices,
            pad_slot_id=pad_slot_id,
        )

    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    causal_conv1d_update_kernel(
        x,
        conv_state,
        weight,
        bias,
        activation_val,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
    if unsqueeze:
        x = x.squeeze(-1)
    return x
