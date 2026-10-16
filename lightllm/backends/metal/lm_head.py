"""Large-vocabulary projection for measured FP16 Qwen request batches."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(
        r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void project(
    const device half* input [[buffer(0)]],
    const device half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint n0=block*32, sm=(sg/2)*8, sn=(sg%2)*16;
    threadgroup half a[16*32], b[32*32];
    simdgroup_float8x8 c0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 c1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k0=0;k0<1024;k0+=32) {
        for(uint j=tid;j<16*32;j+=128)
            a[j]=input[(j/32)*1024+k0+j%32];
        for(uint j=tid;j<32*32;j+=128)
            b[j]=weight[(n0+j/32)*1024+k0+j%32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kk=0;kk<32;kk+=8) {
            simdgroup_half8x8 aa,b0,b1;
            simdgroup_load(aa,a+sm*32+kk,32);
            simdgroup_load(b0,b+sn*32+kk,32,ulong2(0,0),true);
            simdgroup_load(b1,b+(sn+8)*32+kk,32,ulong2(0,0),true);
            simdgroup_multiply_accumulate(c0,aa,b0,c0);
            simdgroup_multiply_accumulate(c1,aa,b1,c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_half8x8 rounded;
    rounded.thread_elements()[0]=half(c0.thread_elements()[0]);
    rounded.thread_elements()[1]=half(c0.thread_elements()[1]);
    simdgroup_store(rounded,output+sm*248320+n0+sn,248320);
    rounded.thread_elements()[0]=half(c1.thread_elements()[0]);
    rounded.thread_elements()[1]=half(c1.thread_elements()[1]);
    simdgroup_store(rounded,output+sm*248320+n0+sn+8,248320);
}
"""
    )


def project(input, weight, out=None):
    """Return [vocabulary, requests], retaining FP16 rounding before logit casting."""
    if not hasattr(torch.mps, "compile_shader") or input.device.type != "mps":
        return None
    if input.shape != (1024, 16) or weight.shape != (248320, 1024):
        return None
    if input.dtype != torch.float16 or weight.dtype != input.dtype or weight.device != input.device:
        return None
    if input.requires_grad or weight.requires_grad or not weight.is_contiguous():
        return None
    x = input.t()
    if not x.is_contiguous() or input.storage_offset() % 4 or weight.storage_offset() % 4:
        return None
    if out is not None:
        if out.shape != (248320, 16) or out.dtype != input.dtype or out.device != input.device:
            return None
        if out.untyped_storage().data_ptr() in (
            input.untyped_storage().data_ptr(),
            weight.untyped_storage().data_ptr(),
        ):
            return None
    result = torch.empty((16, 248320), dtype=input.dtype, device=input.device)
    _library().project(x, weight, result, threads=(248320 // 32) * 128, group_size=128)
    result = result.t()
    if out is not None:
        out.copy_(result)
        return out
    return result
