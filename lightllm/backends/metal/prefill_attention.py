"""Causal Qwen prefill attention with FP32 online softmax on Apple Metal.

FP16 Q/K/V operands use FP32 matrix accumulators. Probabilities are split into
FP16 high and residual components before PV, avoiding single-half rounding.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _library(qs, ks, vs, os, qheads, kvheads, tileq=32):
    init = "\n".join(f"simdgroup_float8x8 o{j}=make_filled_simdgroup_matrix<float,8>(0.0f);" for j in range(16))
    rescale = "\n".join(f"o{j}.thread_elements()[0]*=alpha0; o{j}.thread_elements()[1]*=alpha1;" for j in range(16))
    pv = "\n".join(
        f"""simdgroup_half8x8 v{j};
        simdgroup_load(v{j},kv_tile+pk*256+vbase+{j * 8},256);
        simdgroup_multiply_accumulate(o{j},p,v{j},o{j});
        simdgroup_multiply_accumulate(o{j},plow,v{j},o{j});"""
        for j in range(16)
    )
    store = "\n".join(
        f"""
        if(m0+qbase+r0<qlen)
            output[batch*$OB+(m0+qbase+r0)*$OT+head*$OH+vbase+{j * 8}+c0]=
                half(o{j}.thread_elements()[0]/denom[qbase+r0]);
        if(m0+qbase+r1<qlen)
            output[batch*$OB+(m0+qbase+r1)*$OT+head*$OH+vbase+{j * 8}+c1]=
                half(o{j}.thread_elements()[1]/denom[qbase+r1]);
        """
        for j in range(16)
    )
    source = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attend(
    const device half* q [[buffer(0)]],
    const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant long& qlen [[buffer(4)]],
    constant long& kvlen [[buffer(5)]],
    constant long& start [[buffer(6)]],
    constant long& qbatch_stride [[buffer(7)]],
    constant long& kbatch_stride [[buffer(8)]],
    constant long& vbatch_stride [[buffer(9)]],
    constant long& obatch_stride [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint head=group.y, kh=head/($QHEADS/$KVHEADS), m0=group.x*$TQ, batch=group.z;
    const uint qbase=(sg/2)*8, kbase=(sg%2)*16, vbase=(sg%2)*128;
    // Ask each public matrix type for its own logical element coordinates.
    // This stays on the GPU and assumes no hardware lane-to-element layout.
    threadgroup float float_coords[64];
    threadgroup half half_coords[64];
    threadgroup half kv_tile[32*256], probabilities[$TQ*32], residuals[$TQ*32];
    threadgroup float scores[$TQ*32], maxima[$TQ], denom[$TQ], alphas[$TQ];
    if(tid<64) {float_coords[tid]=float(tid);half_coords[tid]=half(tid);}
    if(tid<$TQ) {maxima[tid]=-INFINITY;denom[tid]=0.0f;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 fc;
    simdgroup_half8x8 hc;
    simdgroup_load(fc,float_coords,8);
    simdgroup_load(hc,half_coords,8);
    const uint a=uint(fc.thread_elements()[0]),b=uint(fc.thread_elements()[1]);
    const uint ha=uint(hc.thread_elements()[0]),hb=uint(hc.thread_elements()[1]);
    const uint r0=a/8,c0=a%8,r1=b/8,c1=b%8;
    const uint hr0=ha/8,hc0=ha%8,hr1=hb/8,hc1=hb%8;
    $INIT
    const uint last_key=min(uint(kvlen),uint(start)+m0+$TQ);
    for(uint n0=0;n0<last_key;n0+=32) {
        for(uint j=tid;j<32*256/4;j+=$THREADS) {
            const uint row=j/64,d=(j%64)*4;
            const half4 vec=n0+row<kvlen ? *((const device half4*)(k+batch*$KB+(n0+row)*$KT+kh*$KH+d)) : half4(0);
            *((threadgroup half4*)(kv_tile+row*256+d))=vec;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dots0=make_filled_simdgroup_matrix<float,8>(0.0f);
        simdgroup_float8x8 dots1=make_filled_simdgroup_matrix<float,8>(0.0f);
        for(uint d=0;d<256;d+=8) {
            simdgroup_half8x8 qm,km0,km1;
            if(m0+qbase+8<=qlen) {
                simdgroup_load(qm,q+batch*$QB+(m0+qbase)*$QT+head*$QH+d,$QT);
            } else {
                qm.thread_elements()[0]=m0+qbase+hr0<qlen ? q[batch*$QB+(m0+qbase+hr0)*$QT+head*$QH+d+hc0] : half(0);
                qm.thread_elements()[1]=m0+qbase+hr1<qlen ? q[batch*$QB+(m0+qbase+hr1)*$QT+head*$QH+d+hc1] : half(0);
            }
            simdgroup_load(km0,kv_tile+kbase*256+d,256,ulong2(0,0),true);
            simdgroup_load(km1,kv_tile+(kbase+8)*256+d,256,ulong2(0,0),true);
            simdgroup_multiply_accumulate(dots0,qm,km0,dots0);
            simdgroup_multiply_accumulate(dots1,qm,km1,dots1);
        }
        simdgroup_store(dots0,scores+qbase*32+kbase,32);
        simdgroup_store(dots1,scores+qbase*32+kbase+8,32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint row=sg;row<$TQ;row+=$SGS) {
            const bool valid_query=m0+row<qlen;
            const bool allowed=valid_query && n0+lane<kvlen && n0+lane<=uint(start)+m0+row;
            const float score=allowed ? scores[row*32+lane]*0.0625f : -INFINITY;
            const float mx=simd_max(score);
            const float next_max=valid_query ? max(maxima[row],mx) : 0.0f;
            const float alpha=valid_query ? exp(maxima[row]-next_max) : 0.0f;
            const float probability=allowed ? exp(score-next_max) : 0.0f;
            const float sum=simd_sum(probability);
            const half high=half(probability);
            probabilities[row*32+lane]=high;
            residuals[row*32+lane]=half(probability-float(high));
            if(lane==0) {
                alphas[row]=alpha;
                maxima[row]=next_max;
                denom[row]=valid_query ? denom[row]*alpha+sum : 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=tid;j<32*256/4;j+=$THREADS) {
            const uint row=j/64,d=(j%64)*4;
            const half4 vec=n0+row<kvlen ? *((const device half4*)(v+batch*$VB+(n0+row)*$VT+kh*$VH+d)) : half4(0);
            *((threadgroup half4*)(kv_tile+row*256+d))=vec;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float alpha0=alphas[qbase+r0],alpha1=alphas[qbase+r1];
        $RESCALE
        for(uint pk=0;pk<32;pk+=8) {
            simdgroup_half8x8 p,plow;
            simdgroup_load(p,probabilities+qbase*32+pk,32);
            simdgroup_load(plow,residuals+qbase*32+pk,32);
            $PV
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    $STORE
}
"""
    params = dict(
        INIT=init,
        RESCALE=rescale,
        PV=pv,
        STORE=store,
        QB="qbatch_stride",
        QT=qs[0],
        QH=qs[1],
        KB="kbatch_stride",
        KT=ks[0],
        KH=ks[1],
        VB="vbatch_stride",
        VT=vs[0],
        VH=vs[1],
        OB="obatch_stride",
        OT=os[0],
        OH=os[1],
        QHEADS=qheads,
        KVHEADS=kvheads,
        TQ=tileq,
        THREADS=(tileq // 8) * 2 * 32,
        SGS=(tileq // 8) * 2,
    )
    for name in sorted(params, key=len, reverse=True):
        source = source.replace("$" + name, str(params[name]))
    return torch.mps.compile_shader(source)


def prefill(q, k, v, query_start, out=None):
    """Return grouped [B,T,H,D] output, or None before writing unsupported inputs."""
    if not hasattr(torch.mps, "compile_shader") or q.device.type != "mps":
        return None
    if any(t.ndim != 4 for t in (q, k, v)):
        return None
    if any(t.dtype != torch.float16 or t.device != q.device or t.requires_grad for t in (q, k, v)):
        return None
    batch, queries, heads, dim = q.shape
    if not 1 <= batch <= 16 or not 2 <= queries <= 2048 or (heads, dim) != (8, 256):
        return None
    if k.shape != v.shape or k.shape[0] != batch or k.shape[2:] != (2, 256):
        return None
    keys = k.shape[1]
    if not isinstance(query_start, int) or not 0 <= query_start <= keys - queries or keys > 8192:
        return None
    if any(
        t.stride(-1) != 1 or t.storage_offset() % 4 or any(s < 0 or s % 4 for s in t.stride()[:-1]) for t in (q, k, v)
    ):
        return None
    if out is not None:
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
            or out.requires_grad
            or not out.is_contiguous()
        ):
            return None
        if out.untyped_storage().data_ptr() in {t.untyped_storage().data_ptr() for t in (q, k, v)}:
            return None
    else:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    tileq = 16 if queries <= 32 else 32
    # Leading strides depend on prefix length, even at B=1. They must stay
    # runtime arguments so new cached prefixes do not trigger another JIT.
    lib = _library(q.stride()[1:3], k.stride()[1:3], v.stride()[1:3], out.stride()[1:3], heads, k.shape[2], tileq)
    count = (tileq // 8) * 2 * 32
    lib.attend(
        q,
        k,
        v,
        out,
        queries,
        keys,
        query_start,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out.stride(0),
        threads=(((queries + tileq - 1) // tileq) * count, heads, batch),
        group_size=(count, 1, 1),
    )
    return out
