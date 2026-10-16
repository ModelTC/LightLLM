"""One-dispatch RMS normalization, optionally including the output gate."""
from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _library(data_type, weight_type, gate_type_name, size, x_stride, out_stride, z_stride, eps, gate):
    group = min(256, max(32, 1 << (size - 1).bit_length()))
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void rms(
    const device $T* x [[buffer(0)]],
    const device $W* weight [[buffer(1)]],
    const device $Z* z [[buffer(2)]],
    device $T* output [[buffer(3)]],
    uint3 block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint row=block.y;
    threadgroup float partials[8];
    float sum=0.0f;
    for(uint j=tid;j<$SIZE;j+=$GROUP) {
        const float value=float(x[row*$XR+j*$XC]);sum+=value*value;
    }
    sum=simd_sum(sum);
    if(lane==0) partials[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=0.0f;
    for(uint i=0;i<$GROUP/32;++i) sum+=partials[i];
    const float scale=rsqrt(sum/float($SIZE)+float($EPS));
    for(uint j=tid;j<$SIZE;j+=$GROUP) {
        float value=(float(x[row*$XR+j*$XC])*scale)*float(weight[j]);
        if($GATE) {
            const float gate=float(z[row*$ZR+j*$ZC]);
            value*=($SILU ? gate : 1.0f)/(1.0f+exp(-gate));
        }
        output[row*$OR+j*$OC]=$T(value);
    }
}
"""
    values = dict(
        T=data_type,
        W=weight_type,
        Z=gate_type_name,
        SIZE=size,
        GROUP=group,
        XR=x_stride[0],
        XC=x_stride[1],
        OR=out_stride[0],
        OC=out_stride[1],
        ZR=z_stride[0],
        ZC=z_stride[1],
        EPS=repr(float(eps)),
        GATE=int(gate is not None),
        SILU=int(gate == "silu"),
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source), group


def rms(x, weight, eps, out=None, z=None, gate=None):
    if not hasattr(torch.mps, "compile_shader") or x.device.type != "mps":
        return None
    types = {torch.float16: "half", torch.float32: "float"}
    if x.dtype not in types or weight.dtype not in types or weight.ndim != 1 or weight.stride() != (1,):
        return None
    if weight.device != x.device or x.shape[-1] != weight.numel():
        return None
    if z is not None and (
        z.dtype not in types or z.device != x.device or z.shape != x.shape or gate not in ("silu", "sigmoid")
    ):
        return None
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device) if out is None else out
    if output.dtype != x.dtype or output.device != x.device or output.shape != x.shape:
        return None
    size = x.shape[-1]
    if not size:
        return None
    xf = x.reshape(-1, size)
    of = output.view(-1, size) if output.is_contiguous() else None
    if of is None:
        # Strided 2D output (including in-place Q/K norm views) is supported.
        if output.ndim != 2:
            return None
        of = output
    zf = z.reshape(-1, size) if z is not None else xf
    if not xf.shape[0]:
        return output
    lib, group = _library(
        types[x.dtype],
        types[weight.dtype],
        types[zf.dtype],
        size,
        xf.stride(),
        of.stride(),
        zf.stride(),
        float(eps),
        gate,
    )
    lib.rms(xf, weight, zf, of, threads=(group, xf.shape[0], 1), group_size=(group, 1, 1))
    return output
