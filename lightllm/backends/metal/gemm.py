"""Small request batches with wide projections and FP32 accumulation."""

from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _library(batch, size, rows):
    source = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void gemm(
    const device half* input [[buffer(0)]],
    const device half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint m0=block.y*16, n0=block.x*32;
    const uint sm=(sg/2)*8, sn=(sg%2)*16;
    threadgroup float a[16*32];
    threadgroup float b[32*32];
    threadgroup float c[16*32];
    simdgroup_float8x8 c0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 c1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k0=0;k0<$K;k0+=32) {
        for(uint j=tid;j<16*32;j+=128) {
            const uint m=j/32, k=j%32;
            a[j]=(m0+m<$M) ? float(input[(m0+m)*$K+k0+k]) : 0.0f;
        }
        for(uint j=tid;j<32*32;j+=128) {
            const uint n=j/32, k=j%32;
            b[j]=(n0+n<$N) ? float(weight[(n0+n)*$K+k0+k]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kk=0;kk<32;kk+=8) {
            simdgroup_float8x8 aa, b0, b1;
            simdgroup_load(aa,a+sm*32+kk,32);
            simdgroup_load(b0,b+sn*32+kk,32,ulong2(0,0),true);
            simdgroup_load(b1,b+(sn+8)*32+kk,32,ulong2(0,0),true);
            simdgroup_multiply_accumulate(c0,aa,b0,c0);
            simdgroup_multiply_accumulate(c1,aa,b1,c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0,c+sm*32+sn,32);
    simdgroup_store(c1,c+sm*32+sn+8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<16*32;j+=128) {
        const uint m=j/32, n=j%32;
        if(m0+m<$M && n0+n<$N) output[(m0+m)*$N+n0+n]=half(c[j]);
    }
}
"""
    for key, value in (("$M", batch), ("$K", size), ("$N", rows)):
        source = source.replace(key, str(value))
    return torch.mps.compile_shader(source)


def project(x, weight, out=None, bias=None):
    """Specialize measured small-batch shapes; retain Torch for prefill/down projections."""
    if not hasattr(torch.mps, "compile_shader") or x.device.type != "mps":
        return None
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        return None
    batch, size = x.shape
    rows = weight.shape[0]
    if not 4 <= batch <= 16 or size != 1024 or not 4096 <= rows <= 16384:
        return None
    if x.dtype != torch.float16 or weight.dtype != x.dtype or weight.device != x.device or bias is not None:
        return None
    if not x.is_contiguous() or not weight.is_contiguous():
        return None
    if out is None:
        out = torch.empty((batch, rows), dtype=x.dtype, device=x.device)
    if out.shape != (batch, rows) or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        return None
    if out.untyped_storage().data_ptr() in (x.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr()):
        return None
    lib = _library(batch, size, rows)
    lib.gemm(x, weight, out, threads=(((rows + 31) // 32) * 128, 1, 1), group_size=(128, 1, 1))
    return out
