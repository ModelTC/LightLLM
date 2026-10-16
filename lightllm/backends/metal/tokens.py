"""Token selection for LightLLM's shared prefill/decode mixed scheduler."""

from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _library(types, strides):
    output_type, table_type, req_type, mtp_type, loc_type = types
    output_stride, table_stride, req_stride, mtp_stride, flag_stride, loc_stride = strides
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void gather_mixed(
    device $OUTPUT* input [[buffer(0)]],
    const device $TABLE* table [[buffer(1)]],
    const device $REQ* requests [[buffer(2)]],
    const device $MTP* mtp [[buffer(3)]],
    const device bool* flags [[buffer(4)]],
    const device $LOC* locations [[buffer(5)]],
    constant long& token_count [[buffer(6)]],
    constant long& request_count [[buffer(7)]],
    constant long& step_count [[buffer(8)]],
    uint row [[thread_position_in_grid]],
    uint count [[threads_per_grid]]) {
    if(!flags[row*$FLAG_STRIDE]) return;
    long position=long(locations[row*$LOC_STRIDE]);
    if(position<0) position+=$TOKENS;
    if(position<0 || position>=$TOKENS) return;
    // Scheduler positions are distinct. Also preserve last-row ownership
    // for aliased internal inputs without issuing racing GPU writes.
    for(uint later=row+1;later<count;++later) {
        if(!flags[later*$FLAG_STRIDE]) continue;
        long other=long(locations[later*$LOC_STRIDE]);
        if(other<0) other+=$TOKENS;
        if(other==position) return;
    }
    long req=long(requests[row*$REQ_STRIDE]),step=long(mtp[row*$MTP_STRIDE]);
    if(req<0) req+=$REQUESTS;
    if(step<0) step+=$STEPS;
    if(req<0 || req>=$REQUESTS || step<0 || step>=$STEPS) return;
    input[position*$OUT_STRIDE]=$OUTPUT(table[req*$TABLE_ROW+step*$TABLE_COL]);
}
"""
    values = dict(
        OUTPUT=output_type,
        TABLE=table_type,
        REQ=req_type,
        MTP=mtp_type,
        LOC=loc_type,
        OUT_STRIDE=output_stride,
        TABLE_ROW=table_stride[0],
        TABLE_COL=table_stride[1],
        REQ_STRIDE=req_stride,
        MTP_STRIDE=mtp_stride,
        FLAG_STRIDE=flag_stride,
        LOC_STRIDE=loc_stride,
        TOKENS="token_count",
        REQUESTS="request_count",
        STEPS="step_count",
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def gather_mixed(input_ids, table, requests, mtp, flags, locations):
    """Fill only decode token locations using valid request-manager indices."""
    if not hasattr(torch.mps, "compile_shader") or input_ids.device.type != "mps":
        return False
    types = {torch.int32: "int", torch.int64: "long"}
    tensors = (input_ids, table, requests, mtp, flags, locations)
    if any(t.device != input_ids.device for t in tensors) or flags.dtype != torch.bool:
        return False
    if any(t.dtype not in types for t in (input_ids, table, requests, mtp, locations)):
        return False
    if input_ids.ndim != 1 or table.ndim != 2 or any(t.ndim != 1 for t in (requests, mtp, flags, locations)):
        return False
    if any(t.numel() != requests.numel() for t in (mtp, flags, locations)):
        return False
    if input_ids.stride(0) <= 0 or any(min(t.stride()) < 0 for t in tensors):
        return False
    storage = input_ids.untyped_storage().data_ptr()
    if input_ids.numel() and any(t.untyped_storage().data_ptr() == storage for t in tensors[1:]):
        return False
    if not requests.numel():
        return True
    lib = _library(
        tuple(types[t.dtype] for t in (input_ids, table, requests, mtp, locations)),
        (input_ids.stride(0), table.stride(), requests.stride(0), mtp.stride(0), flags.stride(0), locations.stride(0)),
    )
    lib.gather_mixed(
        *tensors,
        input_ids.numel(),
        table.shape[0],
        table.shape[1],
        threads=requests.numel(),
        group_size=min(256, requests.numel()),
    )
    return True
