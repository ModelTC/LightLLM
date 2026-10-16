"""Native prefill GDN gates, preserving the FP16 beta rounding."""
from functools import lru_cache
import torch


@lru_cache(maxsize=64)
def _library(dtype, a_type, d_type, heads, a_strides, b_strides):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void gates(const device $A* A_log [[buffer(0)]],const device $T* a [[buffer(1)]],
    const device $T* b [[buffer(2)]],const device $D* bias [[buffer(3)]],
    device float* g [[buffer(4)]],device float* beta [[buffer(5)]],constant long& rows [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if(index>=rows*$HEADS)return;
    const uint row=index/$HEADS,head=index%$HEADS;
    const float alpha=float(a[row*$AR+head*$AC])+float(bias[head]);
    const float e=exp(alpha);
    // macOS 14 MSL has no log1p. The alternating six-term series avoids
    // cancellation below 1/16 (remainder <= e^7/7); ordinary log is safe above.
    const float small=e*(1.0f+e*(-0.5f+e*(1.0f/3.0f+e*(-0.25f+e*(0.2f-e/6.0f)))));
    const float softplus=alpha>20.0f?alpha:(e<0.0625f?small:log(1.0f+e));
    g[index]=-exp(float(A_log[head]))*softplus;
    const float raw=float(b[row*$BR+head*$BC]);
    beta[index]=float($T(1.0f/(1.0f+exp(-raw))));
}
"""
    values = dict(
        T=dtype, A=a_type, D=d_type, HEADS=heads, AR=a_strides[0], AC=a_strides[1], BR=b_strides[0], BC=b_strides[1]
    )
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    if not hasattr(torch.mps, "compile_shader") or a.device.type != "mps":
        return None
    types = {torch.float16: "half", torch.float32: "float"}
    if beta != 1.0 or threshold != 20.0 or a.ndim != 2 or b.shape != a.shape:
        return None
    heads = a.shape[1]
    if not heads or A_log.shape != (heads,) or dt_bias.shape != (heads,):
        return None
    if a.dtype not in types or b.dtype != a.dtype or A_log.dtype not in types or dt_bias.dtype not in types:
        return None
    if any(t.device != a.device or t.requires_grad for t in (a, b, A_log, dt_bias)):
        return None
    if A_log.stride() != (1,) or dt_bias.stride() != (1,):
        return None
    if any(s < 1 for t in (a, b) for s in t.stride()):
        return None
    g = torch.empty(a.shape, device=a.device, dtype=torch.float32)
    mix = torch.empty_like(g)
    if a.numel():
        _library(types[a.dtype], types[A_log.dtype], types[dt_bias.dtype], heads, a.stride(), b.stride()).gates(
            A_log, a, b, dt_bias, g, mix, a.shape[0], threads=a.numel(), group_size=256
        )
    return g, mix
