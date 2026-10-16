"""Small-batch vector projection with FP32 accumulation on Apple GPUs."""
from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _library(data_type, size, rows, x_stride, weight_stride, out_stride, bias, vectorized=False, groups=4):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void gemv(
    const device $T* x [[buffer(0)]],
    const device $T* weight [[buffer(1)]],
    const device $T* bias [[buffer(2)]],
    device $T* out [[buffer(3)]],
    uint3 block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint row=block.x*$GROUPS+simd;
    if(row >= $ROWS) return;
    $DOT_PRODUCT
    total=simd_sum(total);
    if(lane==0) {
        if($BIAS) total+=float(bias[row]);
        out[row*$OS]=$T(total);
    }
}
"""
    scalar_dot = r"""
    float total=0.0f;
    for(uint j=lane;j<$SIZE;j+=32)
        total=fma(float(weight[row*$WR+j*$WC]),float(x[j*$XS]),total);
"""
    vector_dot = r"""
    float4 first=0.0f, second=0.0f;
    for(uint j=lane*8;j<$SIZE;j+=256) {
        first=fma(float4(*reinterpret_cast<const device half4*>(weight+row*$WR+j)),
                  float4(*reinterpret_cast<const device half4*>(x+j)),first);
        second=fma(float4(*reinterpret_cast<const device half4*>(weight+row*$WR+j+4)),
                   float4(*reinterpret_cast<const device half4*>(x+j+4)),second);
    }
    float total=dot(first,float4(1.0f))+dot(second,float4(1.0f));
"""
    source = source.replace("$DOT_PRODUCT", vector_dot if vectorized else scalar_dot)
    values = dict(
        T=data_type,
        SIZE=size,
        ROWS=rows,
        WR=weight_stride[0],
        WC=weight_stride[1],
        XS=x_stride,
        OS=out_stride,
        BIAS=int(bias),
        GROUPS=groups,
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def project(x, weight, out=None, bias=None):
    if not hasattr(torch.mps, "compile_shader") or x.device.type != "mps":
        return None
    types = {torch.float16: "half", torch.float32: "float"}
    if x.ndim != 2 or x.shape[0] != 1 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        return None
    if x.dtype not in types or weight.dtype != x.dtype or weight.device != x.device:
        return None
    rows, size = weight.shape
    if size < 256 or rows < 256 or rows > 16384:
        return None
    if bias is not None and (
        bias.dtype != x.dtype or bias.device != x.device or bias.shape != (rows,) or bias.stride() != (1,)
    ):
        return None
    if out is None:
        out = torch.empty((1, rows), dtype=x.dtype, device=x.device)
    if out.shape != (1, rows) or out.dtype != x.dtype or out.device != x.device:
        return None
    if out.untyped_storage().data_ptr() in (x.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr()):
        return None
    vectorized = (
        x.dtype == torch.float16
        and size % 256 == 0
        and x.stride(1) == weight.stride(1) == 1
        and x.storage_offset() % 4 == 0
        and weight.storage_offset() % 4 == weight.stride(0) % 4 == 0
    )
    groups = 8 if vectorized and rows >= 4096 else 4
    lib = _library(
        types[x.dtype],
        size,
        rows,
        x.stride(1),
        weight.stride(),
        out.stride(1),
        bias is not None,
        vectorized=vectorized,
        groups=groups,
    )
    lib.gemv(
        x,
        weight,
        bias if bias is not None else weight,
        out,
        threads=((rows + groups - 1) // groups * groups * 32, 1, 1),
        group_size=(groups * 32, 1, 1),
    )
    return out
