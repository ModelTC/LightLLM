"""In-place multimodal rotary embedding without CPU section reads."""
from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _library(
    data_type,
    cos_type,
    sin_type,
    index_type,
    q_stride,
    k_stride,
    cos_stride,
    sin_stride,
    q_heads,
    k_heads,
    half_dim,
    interleaved,
):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void rotary(
    device $T* q [[buffer(0)]],
    device $T* k [[buffer(1)]],
    const device $C* cosine [[buffer(2)]],
    const device $S* sine [[buffer(3)]],
    const device $I* sections [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint head=tid.x/$HALF;
    const uint dim=tid.x%$HALF;
    if(head >= $QH+$KH) return;
    uint axis;
    if($INTERLEAVED) {
        axis=(dim%3==1 && dim<=3*sections[1]) ? 1 :
             ((dim%3==2 && dim<=3*sections[2]) ? 2 : 0);
    } else {
        axis=dim<sections[0] ? 0 : (dim<sections[0]+sections[1] ? 1 : 2);
    }
    const float c=float(cosine[axis*$CA+tid.y*$CT+dim*$CD]);
    const float s=float(sine[axis*$SA+tid.y*$ST+dim*$SD]);
    if(head<$QH) {
        const uint offset=tid.y*$QT+head*$QHS+dim*$QD;
        const float first=float(q[offset]),second=float(q[offset+$HALF*$QD]);
        q[offset]=$T(first*c-second*s);
        q[offset+$HALF*$QD]=$T(first*s+second*c);
    } else {
        const uint offset=tid.y*$KT+(head-$QH)*$KHS+dim*$KD;
        const float first=float(k[offset]),second=float(k[offset+$HALF*$KD]);
        k[offset]=$T(first*c-second*s);
        k[offset+$HALF*$KD]=$T(first*s+second*c);
    }
}
"""
    values = dict(
        T=data_type,
        C=cos_type,
        S=sin_type,
        I=index_type,
        HALF=half_dim,
        QH=q_heads,
        KH=k_heads,
        INTERLEAVED=int(interleaved),
        QT=q_stride[0],
        QHS=q_stride[1],
        QD=q_stride[2],
        KT=k_stride[0],
        KHS=k_stride[1],
        KD=k_stride[2],
        CA=cos_stride[0],
        CT=cos_stride[1],
        CD=cos_stride[2],
        SA=sin_stride[0],
        ST=sin_stride[1],
        SD=sin_stride[2],
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def rotary(q, k, cos, sin, sections, interleaved, partial):
    if not hasattr(torch.mps, "compile_shader") or q.device.type != "mps":
        return False
    types = {torch.float16: "half", torch.float32: "float"}
    if q.dtype not in types or k.dtype != q.dtype or cos.dtype not in types or sin.dtype not in types:
        return False
    if any(t.device != q.device for t in (k, cos, sin, sections)):
        return False
    if sections.dtype not in (torch.int32, torch.int64) or sections.numel() != 3 or not sections.is_contiguous():
        return False
    if q.ndim != 3 or k.ndim != 3 or cos.ndim != 3 or sin.ndim != 3:
        return False
    size = int(q.shape[2] * partial)
    if not size or size % 2 or size > q.shape[2] or size > k.shape[2]:
        return False
    if (
        cos.shape[0] != 3
        or sin.shape[0] != 3
        or min(cos.shape[1], sin.shape[1]) < q.shape[0]
        or min(cos.shape[2], sin.shape[2]) < size // 2
    ):
        return False
    if not q.shape[0]:
        return True
    lib = _library(
        types[q.dtype],
        types[cos.dtype],
        types[sin.dtype],
        "long" if sections.dtype == torch.int64 else "int",
        q.stride(),
        k.stride(),
        cos.stride(),
        sin.stride(),
        q.shape[1],
        k.shape[1],
        size // 2,
        interleaved,
    )
    width = (size // 2) * (q.shape[1] + k.shape[1])
    lib.rotary(q, k, cos, sin, sections, threads=(width, q.shape[0], 1), group_size=(min(256, width), 1, 1))
    return True
