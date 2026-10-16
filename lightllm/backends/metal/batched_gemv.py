"""Reuse projection weights across small request batches with FP32 accumulation."""

from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _library(tile, size, rows, x_stride, out_stride, tail):
    initialization, products, stores = [], [], []
    for b in range(tile):
        initialization.append(f"float4 first{b}=0.0f,second{b}=0.0f;")
        first = f"float4(*reinterpret_cast<const device half4*>(x+{b}*$XR+j))"
        second = f"float4(*reinterpret_cast<const device half4*>(x+{b}*$XR+j+4))"
        valid = f"batch_base+{b}<batch_count"
        if tail:
            first = f"({valid} ? {first} : float4(0.0f))"
            second = f"({valid} ? {second} : float4(0.0f))"
        products.append(f"first{b}=fma(w0,{first},first{b}); second{b}=fma(w1,{second},second{b});")
        guard = f"lane==0 && {valid}" if tail else "lane==0"
        stores.append(
            f"float sum{b}=simd_sum(dot(first{b},float4(1.0f))+dot(second{b},float4(1.0f)));"
            f"if({guard}) output[{b}*$OR+row]=half(sum{b});"
        )
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void project(
    const device half* x [[buffer(0)]],
    const device half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant long& batch_count [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row=group.x*4+sg, batch_base=group.y*$TILE;
    if(row>=$N) return;
    x+=batch_base*$XR; output+=batch_base*$OR;
    $INITIALIZE
    for(uint j=lane*8;j<$K;j+=256) {
        float4 w0=float4(*reinterpret_cast<const device half4*>(weight+row*$K+j));
        float4 w1=float4(*reinterpret_cast<const device half4*>(weight+row*$K+j+4));
        $PRODUCTS
    }
    $STORES
}
"""
    fragments = dict(INITIALIZE=initialization, PRODUCTS=products, STORES=stores)
    for key, lines in fragments.items():
        source = source.replace("$" + key, "\n".join(lines))
    params = dict(TILE=tile, XR=x_stride, OR=out_stride, K=size, N=rows)
    for key in sorted(params, key=len, reverse=True):
        source = source.replace("$" + key, str(params[key]))
    return torch.mps.compile_shader(source)


def project(x, weight, out=None, bias=None):
    """Specialize measured Qwen projections; unsupported layouts retain Torch."""
    if not hasattr(torch.mps, "compile_shader") or x.device.type != "mps" or bias is not None:
        return None
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        return None
    batch, size = x.shape
    rows = weight.shape[0]
    if not 2 <= batch <= 16:
        return None
    wide = batch <= 7 and size == 1024 and rows in (5120, 7168, 8224)
    narrow = batch <= 8 and rows == 1024 and size in (2048, 3584)
    if not (wide or narrow):
        return None
    if x.dtype != torch.float16 or weight.dtype != x.dtype or weight.device != x.device:
        return None
    if not weight.is_contiguous() or x.stride(1) != 1 or x.stride(0) < size:
        return None
    if x.storage_offset() % 4 or x.stride(0) % 4 or weight.storage_offset() % 4:
        return None
    if out is None:
        out = torch.empty((batch, rows), dtype=x.dtype, device=x.device)
    if out.shape != (batch, rows) or out.dtype != x.dtype or out.device != x.device:
        return None
    if out.stride(1) != 1 or out.stride(0) < rows:
        return None
    if out.untyped_storage().data_ptr() in (x.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr()):
        return None
    tile = min(batch, 4)
    lib = _library(tile, size, rows, x.stride(0), out.stride(0), batch % tile != 0)
    lib.project(
        x,
        weight,
        out,
        batch,
        threads=(((rows + 3) // 4) * 128, (batch + tile - 1) // tile, 1),
        group_size=(128, 1, 1),
    )
    return out
