"""Greedy sampling with the existing LightLLM penalties and FP32 logprobs.

The vocabulary is reduced in partitions. No probability matrix is allocated;
the adjusted logits retain the same in-place contract as the generic sampler.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _library(request_type, length_type, mask_type, eos_type, vocab, states, eos_count, strides):
    source = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline bool nan_bits(float value) {return (as_type<uint>(value)&0x7fffffffu)>0x7f800000u;}
inline bool finite_bits(float value) {return (as_type<uint>(value)&0x7fffffffu)<0x7f800000u;}

kernel void partition(
    device float* logits [[buffer(0)]],
    const device $R* requests [[buffer(1)]],
    const device float* temperatures [[buffer(2)]],
    const device $L* length_penalty [[buffer(3)]],
    const device $M* mask_eos [[buffer(4)]],
    const device $E* eos_ids [[buffer(5)]],
    const device int* counts [[buffer(6)]],
    const device float* repetition [[buffer(7)]],
    const device float* frequency [[buffer(8)]],
    const device float* presence [[buffer(9)]],
    const device float* decay [[buffer(10)]],
    device float* stats [[buffer(11)]],
    device uint* ids [[buffer(12)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint part=block.x, row=block.y;
    long request=long(requests[row*$RS]);if(request<0) request+=$STATES;
    const float rep=repetition[request*$REPS], freq=frequency[request*$FREQS], pres=presence[request*$PRESS];
    const float scale=pow(decay[request*$DS],float(length_penalty[row*$LS]))-1.0f;
    const float temperature=temperatures[row*$TS];
    float values[8];
    float local_max=-INFINITY;
    uint local_best=0xffffffffu, local_nan=0xffffffffu;
    for(uint i=0;i<8;++i) {
        const uint col=part*2048+tid+i*256;
        float value=-INFINITY;
        if(col<$VOCAB) {
            value=logits[row*$LR+col*$LC];
            const int count=counts[request*$CR+col*$CC];
            if(count>0) value=value>0.0f ? value/rep : value*rep;
            value=value-float(count)*freq;
            value=value-float(count>0)*pres;
            for(uint e=0;e<$EOS;++e) {
                long eos=long(eos_ids[e*$ES]);if(eos<0) eos+=$VOCAB;
                if(long(col)==eos) {
                    value=bool(mask_eos[row*$MS]) ? -10000000.0f : value+abs(value)*scale;
                    break;
                }
            }
            value=value/temperature;
            logits[row*$LR+col*$LC]=value;
            if(nan_bits(value)) local_nan=min(local_nan,col);
            else if(value>local_max || (value==local_max && col<local_best)) {local_max=value;local_best=col;}
        }
        values[i]=value;
    }
    threadgroup float maxima[8], sums[8];
    threadgroup uint best[8], nans[8];
    const float simd_maximum=simd_max(local_max);
    const uint simd_best=simd_min(local_max==simd_maximum ? local_best : 0xffffffffu);
    const uint simd_nan=simd_min(local_nan);
    if(lane==0) {maxima[sg]=simd_maximum;best[sg]=simd_best;nans[sg]=simd_nan;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;
    for(uint i=0;i<8;++i) maximum=max(maximum,maxima[i]);
    float subtotal=0.0f;
    if(finite_bits(maximum)) {
        for(uint i=0;i<8;++i) subtotal+=exp(values[i]-maximum);
    }
    subtotal=simd_sum(subtotal);
    if(lane==0) sums[sg]=subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        uint chosen=0xffffffffu, first_nan=0xffffffffu;
        float total=0.0f;
        for(uint i=0;i<8;++i) {
            if(maxima[i]==maximum) chosen=min(chosen,best[i]);
            first_nan=min(first_nan,nans[i]);total+=sums[i];
        }
        const uint offset=(row*$PARTS+part)*2;
        stats[offset]=maximum;stats[offset+1]=total;
        ids[offset]=chosen;ids[offset+1]=first_nan;
    }
}

kernel void merge(
    const device float* stats [[buffer(0)]],
    const device uint* ids [[buffer(1)]],
    device long* tokens [[buffer(2)]],
    device float* logprobs [[buffer(3)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row=block.y, base=row*$PARTS*2;
    float local_max=-INFINITY;
    uint local_best=0xffffffffu, local_nan=0xffffffffu;
    for(uint p=tid;p<$PARTS;p+=256) {
        const uint offset=base+p*2;
        const float value=stats[offset];
        if(value>local_max || (value==local_max && ids[offset]<local_best)) {local_max=value;local_best=ids[offset];}
        local_nan=min(local_nan,ids[offset+1]);
    }
    threadgroup float maxima[8], sums[8];
    threadgroup uint best[8], nans[8];
    const float simd_maximum=simd_max(local_max);
    const uint simd_best=simd_min(local_max==simd_maximum ? local_best : 0xffffffffu);
    const uint simd_nan=simd_min(local_nan);
    if(lane==0) {maxima[sg]=simd_maximum;best[sg]=simd_best;nans[sg]=simd_nan;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;
    for(uint i=0;i<8;++i) maximum=max(maximum,maxima[i]);
    float subtotal=0.0f;
    if(finite_bits(maximum)) {
        for(uint p=tid;p<$PARTS;p+=256) subtotal+=stats[base+p*2+1]*exp(stats[base+p*2]-maximum);
    }
    subtotal=simd_sum(subtotal);
    if(lane==0) sums[sg]=subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        uint chosen=0xffffffffu, first_nan=0xffffffffu;
        float total=0.0f;
        for(uint i=0;i<8;++i) {
            if(maxima[i]==maximum) chosen=min(chosen,best[i]);
            first_nan=min(first_nan,nans[i]);total+=sums[i];
        }
        tokens[row]=long(first_nan!=0xffffffffu ? first_nan : chosen);
        logprobs[row]=(first_nan!=0xffffffffu || !finite_bits(maximum)) ? NAN : log(1.0f/total);
    }
}
"""
    values = dict(
        R=request_type,
        L=length_type,
        M=mask_type,
        E=eos_type,
        VOCAB=vocab,
        STATES=states,
        EOS=eos_count,
        PARTS=(vocab + 2047) // 2048,
    )
    values.update(zip(("LR", "LC", "RS", "TS", "LS", "MS", "ES", "CR", "CC", "REPS", "FREQS", "PRESS", "DS"), strides))
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def greedy(logits, requests, temperatures, length_penalty, mask_eos, eos_ids, manager):
    """Return None before any write when the generic sampler is required."""
    if not hasattr(torch.mps, "compile_shader") or logits.device.type != "mps":
        return None
    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or getattr(manager, "penalty_counter_mode", None) != "gpu_counter"
    ):
        return None
    batch, vocab = logits.shape
    if not 1 <= vocab < 2 ** 31:
        return None
    counts = manager.req_to_out_token_id_counter
    repetition, frequency, presence, decay = (
        manager.req_to_repetition_penalty,
        manager.req_to_frequency_penalty,
        manager.req_to_presence_penalty,
        manager.req_to_exponential_decay_length_penalty,
    )
    ints = {torch.int32: "int", torch.int64: "long"}
    masks = {torch.bool: "bool", torch.int32: "int"}
    lengths = {**ints, torch.float32: "float"}
    if (
        requests.dtype not in ints
        or eos_ids.dtype not in ints
        or length_penalty.dtype not in lengths
        or mask_eos.dtype not in masks
    ):
        return None
    if any(t.shape != (batch,) for t in (requests, temperatures, length_penalty, mask_eos)) or eos_ids.ndim != 1:
        return None
    if (
        counts.ndim != 2
        or counts.shape[1] != vocab
        or counts.dtype != torch.int32
        or temperatures.dtype != torch.float32
    ):
        return None
    if any(t.shape != (counts.shape[0],) or t.dtype != torch.float32 for t in (repetition, frequency, presence, decay)):
        return None
    inputs = (requests, temperatures, length_penalty, mask_eos, eos_ids, counts, repetition, frequency, presence, decay)
    if any(t.device != logits.device for t in inputs):
        return None
    if logits.untyped_storage().data_ptr() in {t.untyped_storage().data_ptr() for t in inputs}:
        return None
    if any(s <= 0 for s in logits.stride()):
        return None
    if batch and not (logits.stride(0) >= vocab * logits.stride(1) or logits.stride(1) >= batch * logits.stride(0)):
        return None
    tokens = torch.empty(batch, dtype=torch.int64, device=logits.device)
    logprobs = torch.empty(batch, dtype=torch.float32, device=logits.device)
    if not batch:
        return tokens, logprobs
    parts = (vocab + 2047) // 2048
    stats = torch.empty((batch, parts, 2), dtype=torch.float32, device=logits.device)
    ids = torch.empty((batch, parts, 2), dtype=torch.int32, device=logits.device)
    lib = _library(
        ints[requests.dtype],
        lengths[length_penalty.dtype],
        masks[mask_eos.dtype],
        ints[eos_ids.dtype],
        vocab,
        counts.shape[0],
        eos_ids.numel(),
        (
            *logits.stride(),
            requests.stride(0),
            temperatures.stride(0),
            length_penalty.stride(0),
            mask_eos.stride(0),
            eos_ids.stride(0),
            *counts.stride(),
            repetition.stride(0),
            frequency.stride(0),
            presence.stride(0),
            decay.stride(0),
        ),
    )
    lib.partition(
        logits,
        requests,
        temperatures,
        length_penalty,
        mask_eos,
        eos_ids,
        counts,
        repetition,
        frequency,
        presence,
        decay,
        stats,
        ids,
        threads=(parts * 256, batch, 1),
        group_size=(256, 1, 1),
    )
    lib.merge(stats, ids, tokens, logprobs, threads=(256, batch, 1), group_size=(256, 1, 1))
    return tokens, logprobs
