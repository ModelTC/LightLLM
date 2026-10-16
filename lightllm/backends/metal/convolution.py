"""Convolution over the existing request-indexed Metal state cache."""
from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _library(
    data_type,
    state_type,
    weight_type,
    bias_type,
    index_type,
    batch,
    channels,
    x_row,
    x_col,
    state_row,
    state_channel,
    state_col,
    state_count,
    weight_row,
    weight_col,
    width,
    activation,
    has_bias,
):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void conv_decode(
    const device $T* x [[buffer(0)]],
    device $S* state [[buffer(1)]],
    const device $W* weight [[buffer(2)]],
    const device $B* bias [[buffer(3)]],
    const device $I* indices [[buffer(4)]],
    device $T* output [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if(index >= $BATCH*$CHANNELS) return;
    const uint first=index/$CHANNELS;
    const uint channel=index%$CHANNELS;
    int slot=int(indices[first]);if(slot<0) slot+=$STATES;
    for(uint prev=0;prev<first;++prev) {
        int previous=int(indices[prev]);if(previous<0) previous+=$STATES;
        if(previous==slot) return;
    }
    float history[$HISTORY];
    for(uint i=0;i<$HISTORY;++i)
        history[i]=float(state[slot*$SR+channel*$SC+i*$SD]);
    for(uint row=first;row<$BATCH;++row) {
        int current=int(indices[row]);if(current<0) current+=$STATES;
        if(current!=slot) continue;
        const float value=float(x[row*$XR+channel*$XC]);
        float result=0.0f;
        for(uint i=0;i<$HISTORY;++i)
            result+=history[i]*float(weight[channel*$WR+i*$WC]);
        result+=value*float(weight[channel*$WR+$HISTORY*$WC]);
        if($BIAS) result+=float(bias[channel]);
        if($ACTIVATE) result=result/(1.0f+exp(-result));
        output[row*$CHANNELS+channel]=$T(result);
        for(uint i=0;i+1<$HISTORY;++i) history[i]=history[i+1];
        // Cast through the state dtype between aliased input rows, matching
        // the original sequential cache updates.
        history[$HISTORY-1]=float($S(value));
    }
    for(uint i=0;i<$HISTORY;++i)
        state[slot*$SR+channel*$SC+i*$SD]=$S(history[i]);
}
"""
    values = dict(
        T=data_type,
        S=state_type,
        W=weight_type,
        B=bias_type,
        I=index_type,
        BATCH=batch,
        CHANNELS=channels,
        XR=x_row,
        XC=x_col,
        SR=state_row,
        SC=state_channel,
        SD=state_col,
        STATES=state_count,
        WR=weight_row,
        WC=weight_col,
        HISTORY=width - 1,
        ACTIVATE=int(activation),
        BIAS=int(has_bias),
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def decode(mixed_qkv, state, weight, bias, indices, activation, width):
    if not hasattr(torch.mps, "compile_shader") or mixed_qkv.device.type != "mps":
        return None
    types = {torch.float16: "half", torch.float32: "float"}
    if any(t.dtype not in types for t in (mixed_qkv, state, weight)):
        return None
    if bias is not None and (bias.dtype not in types or bias.stride() != (1,)):
        return None
    if (
        indices.device != mixed_qkv.device
        or indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.stride() != (1,)
    ):
        return None
    if width < 2 or width > 8 or activation not in (None, "silu", "swish"):
        return None
    batch, channels = mixed_qkv.shape
    if state.ndim != 3 or weight.shape != (channels, width) or state.shape[1] != channels or state.shape[2] < width - 1:
        return None
    output = torch.empty((batch, channels), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    if not batch:
        return output
    lib = _library(
        types[mixed_qkv.dtype],
        types[state.dtype],
        types[weight.dtype],
        types[bias.dtype] if bias is not None else types[weight.dtype],
        "long" if indices.dtype == torch.int64 else "int",
        batch,
        channels,
        *mixed_qkv.stride(),
        *state.stride(),
        state.shape[0],
        *weight.stride(),
        width,
        activation in ("silu", "swish"),
        bias is not None,
    )
    lib.conv_decode(
        mixed_qkv,
        state,
        weight,
        bias if bias is not None else weight,
        indices,
        output,
        threads=batch * channels,
        group_size=min(256, batch * channels),
    )
    return output


@lru_cache(maxsize=128)
def _prefill_library(types, strides, batch, channels, width, pad_slot_id, state_count, activation, has_bias):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void conv_prefill(
    device half* input [[buffer(0)]],
    device $S* state [[buffer(1)]],
    const device $W* weight [[buffer(2)]],
    const device $B* bias [[buffer(3)]],
    const device $I* indices [[buffer(4)]],
    const device $L* boundaries [[buffer(5)]],
    const device bool* has_initial [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if(index >= $BATCH*$CHANNELS) return;
    const uint first=index/$CHANNELS, channel=index%$CHANNELS;
    long slot=long(indices[first*$IS]);
    if(slot==$PAD) return;
    if(slot<0) slot+=$STATES;
    // Each channel of a cache slot has one owner. Aliased request rows
    // are processed in input order, including per-row state reset flags.
    for(uint previous=0;previous<first;++previous) {
        long candidate=long(indices[previous*$IS]);
        if(candidate==$PAD) continue;
        if(candidate<0) candidate+=$STATES;
        if(candidate==slot) return;
    }
    float history[$HISTORY];
    for(uint row=first;row<$BATCH;++row) {
        long current=long(indices[row*$IS]);
        if(current==$PAD) continue;
        if(current<0) current+=$STATES;
        if(current!=slot) continue;
        const long begin=long(boundaries[row*$LS]), end=long(boundaries[(row+1)*$LS]);
        if(begin==end) continue;
        for(uint i=0;i<$HISTORY;++i)
            history[i]=has_initial[row*$FS] ? float(state[slot*$SR+channel*$SC+i*$SD]) : 0.0f;
        for(long token=begin;token<end;++token) {
            const float raw=float(input[channel*$XC+token*$XT]);
            float result=0.0f;
            for(uint i=0;i<$HISTORY;++i)
                result+=history[i]*float(weight[channel*$WR+i*$WC]);
            result+=raw*float(weight[channel*$WR+$HISTORY*$WC]);
            if($BIAS) result+=float(bias[channel*$BS]);
            if($ACTIVATE) result=result/(1.0f+exp(-result));
            input[channel*$XC+token*$XT]=half(result);
            for(uint i=0;i+1<$HISTORY;++i) history[i]=history[i+1];
            history[$HISTORY-1]=raw;
        }
        for(uint i=0;i<$HISTORY;++i) state[slot*$SR+channel*$SC+i*$SD]=$S(history[i]);
    }
}
"""
    values = dict(zip(("S", "W", "B", "I", "L"), types))
    values.update(zip(("XC", "XT", "SR", "SC", "SD", "WR", "WC", "BS", "IS", "LS", "FS"), strides))
    values.update(
        BATCH=batch,
        CHANNELS=channels,
        HISTORY=width - 1,
        PAD=pad_slot_id,
        STATES=state_count,
        ACTIVATE=int(activation),
        BIAS=int(has_bias),
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def prefill(x, weight, bias, boundaries, indices, has_initial, state, activation, pad_slot_id):
    """Update FP16 packed sequences in place without reading GPU metadata on CPU."""
    if not hasattr(torch.mps, "compile_shader") or x.device.type != "mps" or x.dtype != torch.float16:
        return False
    if any(t is None for t in (boundaries, indices, has_initial, state)):
        return False
    if x.ndim != 2 or weight.ndim != 2 or state.ndim != 3:
        return False
    channels, width = weight.shape
    if x.shape[0] != channels or state.shape[1:] != (channels, width - 1) or not 2 <= width <= 8:
        return False
    if activation not in (None, "silu", "swish"):
        return False
    types = {torch.float16: "half", torch.float32: "float"}
    ints = {torch.int32: "int", torch.int64: "long"}
    if weight.dtype not in types or state.dtype not in types:
        return False
    if boundaries.ndim != 1 or indices.ndim != 1 or has_initial.ndim != 1:
        return False
    if boundaries.dtype not in ints or indices.dtype not in ints or has_initial.dtype != torch.bool:
        return False
    batch = indices.numel()
    if boundaries.numel() != batch + 1 or has_initial.numel() != batch:
        return False
    tensors = [weight, boundaries, indices, has_initial, state]
    if bias is not None:
        if bias.dtype not in types or bias.shape != (channels,):
            return False
        tensors.append(bias)
    if any(t.device != x.device for t in tensors):
        return False
    if x.untyped_storage().data_ptr() in {t.untyped_storage().data_ptr() for t in tensors}:
        return False
    if not batch or not channels or not x.shape[1]:
        return True
    lib = _prefill_library(
        (
            types[state.dtype],
            types[weight.dtype],
            types[bias.dtype] if bias is not None else types[weight.dtype],
            ints[indices.dtype],
            ints[boundaries.dtype],
        ),
        (
            *x.stride(),
            *state.stride(),
            *weight.stride(),
            bias.stride(0) if bias is not None else 1,
            indices.stride(0),
            boundaries.stride(0),
            has_initial.stride(0),
        ),
        batch,
        channels,
        width,
        pad_slot_id,
        state.shape[0],
        activation in ("silu", "swish"),
        bias is not None,
    )
    lib.conv_prefill(
        x,
        state,
        weight,
        bias if bias is not None else weight,
        indices,
        boundaries,
        has_initial,
        threads=batch * channels,
        group_size=min(256, batch * channels),
    )
    return True
