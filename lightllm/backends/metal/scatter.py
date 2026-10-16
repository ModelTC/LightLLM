"""Write new K/V rows without a full-cache MPSGraph index_copy operation."""
from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _library(data_type, index_type, source_stride, dest_stride, index_stride, head_dim):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void scatter_kv(
    const device $T* source [[buffer(0)]],
    const device $I* indices [[buffer(1)]],
    device $T* dest [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint head=tid.x/$DIM;
    const uint dim=tid.x%$DIM;
    const long slot=long(indices[tid.y*$IS]);
    dest[slot*$DR+head*$DH+dim*$DD] = source[tid.y*$SR+head*$SH+dim*$SD];
}
"""
    values = dict(
        T=data_type,
        I=index_type,
        DIM=head_dim,
        IS=index_stride,
        SR=source_stride[0],
        SH=source_stride[1],
        SD=source_stride[2],
        DR=dest_stride[0],
        DH=dest_stride[1],
        DD=dest_stride[2],
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def copy_kv(source, indices, dest):
    """Copy valid, distinct slots allocated by LightLLM's memory manager."""
    if not hasattr(torch.mps, "compile_shader") or source.device.type != "mps":
        return False
    types = {torch.float16: "half", torch.float32: "float"}
    if source.dtype not in types or dest.dtype != source.dtype or source.ndim != 3 or dest.ndim != 3:
        return False
    if (
        indices.device != source.device
        or dest.device != source.device
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        return False
    if indices.ndim != 1 or indices.numel() != source.shape[0] or source.shape[1:] != dest.shape[1:]:
        return False
    if source.untyped_storage().data_ptr() == dest.untyped_storage().data_ptr():
        return False
    if not source.numel():
        return True
    width = source.shape[1] * source.shape[2]
    lib = _library(
        types[source.dtype],
        "long" if indices.dtype == torch.int64 else "int",
        source.stride(),
        dest.stride(),
        indices.stride(0),
        source.shape[2],
    )
    lib.scatter_kv(source, indices, dest, threads=(width, source.shape[0], 1), group_size=(min(256, width), 1, 1))
    return True


@lru_cache(maxsize=64)
def _gather_library(data_type, index_type, k_stride, v_stride, index_stride, head_dim):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void gather_kv(
    const device $T* keys [[buffer(0)]],
    const device $T* values [[buffer(1)]],
    const device $I* indices [[buffer(2)]],
    device $T* out_keys [[buffer(3)]],
    device $T* out_values [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
    const uint head=tid.x/$DIM;
    const uint dim=tid.x%$DIM;
    const long slot=long(indices[tid.y*$IS]);
    out_keys[tid.y*grid.x+tid.x]=keys[slot*$KR+head*$KH+dim*$KD];
    out_values[tid.y*grid.x+tid.x]=values[slot*$VR+head*$VH+dim*$VD];
}
"""
    values = dict(
        T=data_type,
        I=index_type,
        DIM=head_dim,
        IS=index_stride,
        KR=k_stride[0],
        KH=k_stride[1],
        KD=k_stride[2],
        VR=v_stride[0],
        VH=v_stride[1],
        VD=v_stride[2],
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def gather_kv(keys, values, indices):
    """Gather only selected slots from strided K/V views of the shared pool."""
    if not hasattr(torch.mps, "compile_shader") or keys.device.type != "mps":
        return None
    types = {torch.float16: "half", torch.float32: "float"}
    if keys.dtype not in types or values.dtype != keys.dtype or keys.ndim != 3 or values.shape != keys.shape:
        return None
    if (
        values.device != keys.device
        or indices.device != keys.device
        or indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        return None
    output = torch.empty((2, indices.numel(), keys.shape[1], keys.shape[2]), dtype=keys.dtype, device=keys.device)
    out_keys, out_values = output[0], output[1]
    if not indices.numel():
        return out_keys, out_values
    width = keys.shape[1] * keys.shape[2]
    lib = _gather_library(
        types[keys.dtype],
        "long" if indices.dtype == torch.int64 else "int",
        keys.stride(),
        values.stride(),
        indices.stride(0),
        keys.shape[2],
    )
    lib.gather_kv(
        keys,
        values,
        indices,
        out_keys,
        out_values,
        threads=(width, indices.numel(), 1),
        group_size=(min(256, width), 1, 1),
    )
    return out_keys, out_values
