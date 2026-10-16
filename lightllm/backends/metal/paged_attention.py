"""Single-query attention over LightLLM's request-to-slot mapping.

The split softmax reads K/V in place, supports GQA without head expansion,
and leaves request scheduling and all cache allocation to LightLLM.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _library(
    dtype,
    table_type,
    req_type,
    len_type,
    q_stride,
    k_stride,
    v_stride,
    table_stride,
    req_stride,
    len_stride,
    heads,
    kv_heads,
    dim,
    partitions,
):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void attention(
    const device $T* q [[buffer(0)]],
    const device $T* keys [[buffer(1)]],
    const device $T* values [[buffer(2)]],
    const device $TABLE* table [[buffer(3)]],
    const device $REQ* requests [[buffer(4)]],
    const device $LEN* lengths [[buffer(5)]],
    device float* partial [[buffer(6)]],
    device $T* output [[buffer(7)]],
    uint3 block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint partition=block.x, head=block.y, batch=block.z;
    const uint kv_head=head/($HEADS/$KVHEADS);
    const long request=long(requests[batch*$REQSTRIDE]);
    const uint length=uint(lengths[batch*$LENSTRIDE]);
    const uint first=partition*256;
    const uint row=batch*$HEADS+head;
    const uint poff=(row*$PARTITIONS+partition)*($DIM+2);
    if(first>=length) {
        if($PARTITIONS==1) {
            if(tid<$DIM) output[row*$DIM+tid]=$T(0.0f);
        } else {
            if(tid==0) {partial[poff]=-INFINITY;partial[poff+1]=0.0f;}
            if(tid<$DIM) partial[poff+2+tid]=0.0f;
        }
        return;
    }
    threadgroup float query[$DIM];
    threadgroup float scores[256];
    threadgroup float weights[256];
    threadgroup float reductions[8];
    for(uint d=tid;d<$DIM;d+=256) query[d]=float(q[batch*$QR+head*$QH+d*$QD]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint p=sg;p<256;p+=8) {
        float score=0.0f;
        if(first+p<length) {
            const long slot=long(table[request*$TR+(first+p)*$TC]);
            for(uint d=lane;d<$DIM;d+=32)
                score=fma(query[d],float(keys[slot*$KR+kv_head*$KH+d*$KD]),score);
        }
        score=simd_sum(score)*rsqrt(float($DIM));
        if(lane==0) scores[p]=(first+p<length) ? score : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=simd_max(scores[tid]);
    if(lane==0) reductions[sg]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=-INFINITY;
    for(uint i=0;i<8;++i) maximum=max(maximum,reductions[i]);
    const float weight=exp(scores[tid]-maximum);
    weights[tid]=weight;
    const float subtotal=simd_sum(weight);
    // All readers must finish using the maximum buffer before reusing it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) reductions[sg]=subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(uint i=0;i<8;++i) denominator+=reductions[i];
    const uint count=min(256u,length-first);
    for(uint d=tid;d<$DIM;d+=256) {
        float result=0.0f;
        for(uint p=0;p<count;++p) {
            const long slot=long(table[request*$TR+(first+p)*$TC]);
            result=fma(weights[p],float(values[slot*$VR+kv_head*$VH+d*$VD]),result);
        }
        if($PARTITIONS==1) output[row*$DIM+d]=$T(result/denominator);
        else partial[poff+2+d]=result;
    }
    if($PARTITIONS>1 && tid==0) {partial[poff]=maximum;partial[poff+1]=denominator;}
}

kernel void merge(
    const device float* partial [[buffer(0)]],
    device $T* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint row=index/$DIM, d=index%$DIM;
    const uint base=row*$PARTITIONS*($DIM+2);
    float maximum=-INFINITY;
    for(uint p=0;p<$PARTITIONS;++p) maximum=max(maximum,partial[base+p*($DIM+2)]);
    float result=0.0f,denominator=0.0f;
    for(uint p=0;p<$PARTITIONS;++p) {
        const uint offset=base+p*($DIM+2);
        const float scale=exp(partial[offset]-maximum);
        result=fma(partial[offset+2+d],scale,result);
        denominator=fma(partial[offset+1],scale,denominator);
    }
    output[index]=$T(denominator>0.0f ? result/denominator : 0.0f);
}
"""
    values = dict(
        T=dtype,
        TABLE=table_type,
        REQ=req_type,
        LEN=len_type,
        HEADS=heads,
        KVHEADS=kv_heads,
        DIM=dim,
        PARTITIONS=partitions,
        REQSTRIDE=req_stride,
        LENSTRIDE=len_stride,
        QR=q_stride[0],
        QH=q_stride[1],
        QD=q_stride[2],
        KR=k_stride[0],
        KH=k_stride[1],
        KD=k_stride[2],
        VR=v_stride[0],
        VH=v_stride[1],
        VD=v_stride[2],
        TR=table_stride[0],
        TC=table_stride[1],
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def decode(q, k, v, request_to_token, requests, lengths, max_length, alloc_func=torch.empty):
    if not hasattr(torch.mps, "compile_shader") or q.device.type != "mps":
        return None
    types = {torch.float16: "half", torch.float32: "float"}
    ints = {torch.int32: "int", torch.int64: "long"}
    if q.dtype not in types or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[-1] != k.shape[-1] or q.shape[-1] not in (128, 256):
        return None
    if q.shape[1] % k.shape[1] or request_to_token.ndim != 2:
        return None
    if any(t.device != q.device for t in (k, v, request_to_token, requests, lengths)):
        return None
    if any(t.dtype not in ints for t in (request_to_token, requests, lengths)):
        return None
    if requests.ndim != 1 or lengths.ndim != 1 or requests.numel() != q.shape[0] or lengths.numel() != q.shape[0]:
        return None
    if max_length is None or max_length < 1 or max_length > request_to_token.shape[1]:
        return None
    batch, heads, dim = q.shape
    output = alloc_func(q.shape, dtype=q.dtype, device=q.device)
    if not output.is_contiguous():
        return None
    if not batch:
        return output
    partitions = (max_length + 255) // 256
    partial = torch.empty(
        (batch * heads * partitions * (dim + 2)) if partitions > 1 else 1, dtype=torch.float32, device=q.device
    )
    lib = _library(
        types[q.dtype],
        ints[request_to_token.dtype],
        ints[requests.dtype],
        ints[lengths.dtype],
        q.stride(),
        k.stride(),
        v.stride(),
        request_to_token.stride(),
        requests.stride(0),
        lengths.stride(0),
        heads,
        k.shape[1],
        dim,
        partitions,
    )
    lib.attention(
        q,
        k,
        v,
        request_to_token,
        requests,
        lengths,
        partial,
        output,
        threads=(partitions * 256, heads, batch),
        group_size=(256, 1, 1),
    )
    if partitions > 1:
        lib.merge(partial, output, threads=batch * heads * dim, group_size=256)
    return output
