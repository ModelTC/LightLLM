"""Single-dispatch gated products and request token-counter updates."""

from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _product_library(x_type, gate_type, width, x_stride, gate_stride, output_stride, interleaved, broadcast):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void silu_product(
    const device $X* input [[buffer(0)]],
    device $X* output [[buffer(1)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint row=index.y, col=index.x;
    const uint gate_col=$INTERLEAVED ? col*2 : col;
    const uint up_col=$INTERLEAVED ? col*2+1 : col+$WIDTH;
    const float gate=float(input[row*$XR+gate_col*$XC]);
    const float sigmoid=1.0f/(1.0f+exp(-gate));
    // Preserve the existing FP16 rounding before the product.
    const $X activated=$X(sigmoid*gate);
    output[row*$OR+col*$OC]=$X(float(activated)*float(input[row*$XR+up_col*$XC]));
}
kernel void sigmoid_product(
    device $X* input [[buffer(0)]],
    const device $G* gate [[buffer(1)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint row=index.y, col=index.x;
    const float value=float(gate[row*$GR+($BROADCAST ? 0 : col)*$GC]);
    const $G rounded=$G(1.0f/(1.0f+exp(-value)));
    input[row*$XR+col*$XC]=$X(float(input[row*$XR+col*$XC])*float(rounded));
}
"""
    values = dict(
        X=x_type,
        G=gate_type,
        WIDTH=width,
        XR=x_stride[0],
        XC=x_stride[1],
        GR=gate_stride[0],
        GC=gate_stride[1],
        OR=output_stride[0],
        OC=output_stride[1],
        INTERLEAVED=int(interleaved),
        BROADCAST=int(broadcast),
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def silu_product(x, output, layout):
    if not hasattr(torch.mps, "compile_shader") or x.device.type != "mps":
        return False
    types = {torch.float16: "half", torch.float32: "float"}
    if x.dtype not in types or output.dtype != x.dtype or output.device != x.device or x.ndim != 2 or output.ndim != 2:
        return False
    batch, width = output.shape
    if x.shape != (batch, 2 * width) or layout not in ("blocked", "interleaved"):
        return False
    if output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
        return False
    if not batch or not width:
        return True
    lib = _product_library(
        types[x.dtype], types[x.dtype], width, x.stride(), x.stride(), output.stride(), layout == "interleaved", False
    )
    lib.silu_product(x, output, threads=(width, batch, 1), group_size=(min(256, width), 1, 1))
    return True


def sigmoid_product(x, gate):
    if not hasattr(torch.mps, "compile_shader") or x.device.type != "mps":
        return False
    types = {torch.float16: "half", torch.float32: "float"}
    if x.dtype not in types or gate.dtype not in types or gate.device != x.device or x.ndim != 2 or gate.ndim != 2:
        return False
    batch, width = x.shape
    if gate.shape[0] != batch or gate.shape[1] not in (1, width):
        return False
    if x.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr():
        return False
    if not batch or not width:
        return True
    lib = _product_library(
        types[x.dtype], types[gate.dtype], width, x.stride(), gate.stride(), x.stride(), False, gate.shape[1] == 1
    )
    lib.sigmoid_product(x, gate, threads=(width, batch, 1), group_size=(min(256, width), 1, 1))
    return True


@lru_cache(maxsize=64)
def _counter_library(req_type, token_type, count, state_count, vocab_size, strides, has_mask):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void update_counter(
    const device $R* requests [[buffer(0)]],
    const device $T* tokens [[buffer(1)]],
    device atomic_int* counters [[buffer(2)]],
    const device bool* mask [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if(index >= $COUNT || ($MASK && !mask[index*$MS])) return;
    const long token=long(tokens[index*$TS]);
    if(token<0 || token >= $VOCAB) return;
    long row=long(requests[index*$RS]);
    if(row<0) row+=$STATES;
    atomic_fetch_add_explicit(counters+row*$CR+token*$CC,1,memory_order_relaxed);
}
"""
    values = dict(R=req_type, T=token_type, COUNT=count, STATES=state_count, VOCAB=vocab_size, MASK=int(has_mask))
    values.update(zip(("RS", "TS", "CR", "CC", "MS"), strides))
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def update_counter(requests, tokens, counters, mask):
    if not hasattr(torch.mps, "compile_shader") or requests.device.type != "mps":
        return False
    ints = {torch.int32: "int", torch.int64: "long"}
    if requests.dtype not in ints or tokens.dtype not in ints or counters.dtype != torch.int32:
        return False
    if requests.ndim != 1 or tokens.shape != requests.shape or counters.ndim != 2:
        return False
    if tokens.device != requests.device or counters.device != requests.device:
        return False
    tensors = [requests, tokens]
    if mask is not None:
        if mask.device != requests.device or mask.dtype != torch.bool or mask.shape != requests.shape:
            return False
        tensors.append(mask)
    if counters.untyped_storage().data_ptr() in {t.untyped_storage().data_ptr() for t in tensors}:
        return False
    if not requests.numel():
        return True
    lib = _counter_library(
        ints[requests.dtype],
        ints[tokens.dtype],
        requests.numel(),
        counters.shape[0],
        counters.shape[1],
        (requests.stride(0), tokens.stride(0), *counters.stride(), mask.stride(0) if mask is not None else 1),
        mask is not None,
    )
    lib.update_counter(
        requests,
        tokens,
        counters,
        mask if mask is not None else requests,
        threads=requests.numel(),
        group_size=min(256, requests.numel()),
    )
    return True
