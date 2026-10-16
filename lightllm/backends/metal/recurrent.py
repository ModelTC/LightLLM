"""Fused GDN recurrence operating directly on LightLLM's MPS state tensors.

Each threadgroup keeps a 128 x 32 state tile in registers across the entire
sequence. Prefill therefore launches one kernel instead of Torch operations
for every token. Unusual layouts retain the Torch implementation in kernels.py.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _library(
    data_type,
    q_stride,
    k_stride,
    v_stride,
    g_stride,
    b_stride,
    state_stride,
    final_stride,
    out_stride,
    query_heads,
    value_heads,
    normalize,
    raw_gates,
    scale,
    indexed_decode=False,
    index_type="int",
    token_count=0,
    state_count=0,
    gpu_varlen=False,
):
    source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void recurrent(
    const device $TYPE* q [[buffer(0)]],
    const device $TYPE* k [[buffer(1)]],
    const device $TYPE* v [[buffer(2)]],
    const device $GATE* g [[buffer(3)]],
    const device $GATE* beta [[buffer(4)]],
    const device float* initial [[buffer(5)]],
    device float* final_state [[buffer(6)]],
    device $TYPE* output [[buffer(7)]],
    const device $INDEX* metadata [[buffer(8)]],
    const device float* A_log [[buffer(9)]],
    const device float* dt_bias [[buffer(10)]],
    constant bool& has_initial [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint h = group.y;
    const uint qh = h / ($VH / $QH);
    const uint col = group.x * 32 + lane;
    const uint seq = group.z;
    int begin, end, read_row, write_row;
    if ($INDEXED) {
        read_row = int(metadata[seq]);
        if (read_row < 0) read_row += $STATES;
        // A state slot can appear multiple times. Its first threadgroup
        // processes those rows in input order, preserving alias semantics.
        for (uint prev=0;prev<seq;++prev) {
            int row=int(metadata[prev]); if(row<0) row+=$STATES;
            if(row==read_row) return;
        }
        begin=int(seq);end=$TOKENS;write_row=read_row;
    } else if ($VARLEN) {
        begin=int(metadata[seq]);end=int(metadata[seq+1]);
        read_row=int(seq);write_row=int(seq);
    } else {
        begin=int(metadata[seq * 4]);end=int(metadata[seq * 4 + 1]);
        read_row=int(metadata[seq * 4 + 2]);write_row=int(metadata[seq * 4 + 3]);
    }
    threadgroup float queries[128];
    threadgroup float keys[128];
    threadgroup float norms[8];
    threadgroup float partials[128];
    float state[32];
    for (uint i = 0; i < 32; ++i) {
        const uint row = simd * 32 + i;
        state[i] = has_initial ? initial[read_row * $SS + h * 16384 + row * 128 + col] : 0.0f;
    }
    for (int token = begin; token < end; ++token) {
        if ($INDEXED) {
            int row=int(metadata[token]); if(row<0) row+=$STATES;
            if(row!=read_row) continue;
        }
        const float qr = float(q[token * $QT + qh * $QHSTR + tid * $QD]);
        const float kr = float(k[token * $KT + qh * $KHSTR + tid * $KD]);
        queries[tid] = qr;
        keys[tid] = kr;
        if ($NORMALIZE) {
            const float qsum = simd_sum(qr * qr);
            const float ksum = simd_sum(kr * kr);
            if (lane == 0) { norms[simd] = qsum; norms[simd + 4] = ksum; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float qnorm = $NORMALIZE ? rsqrt(norms[0]+norms[1]+norms[2]+norms[3]+1.0e-6f) : 1.0f;
        const float knorm = $NORMALIZE ? rsqrt(norms[4]+norms[5]+norms[6]+norms[7]+1.0e-6f) : 1.0f;
        float decay_log = float(g[token * $GT + h * $GH]);
        float beta_value = float(beta[token * $BT + h * $BH]);
        if ($RAW) {
            const float alpha = decay_log + dt_bias[h];
            decay_log = -exp(A_log[h]) * (alpha > 20.0f ? alpha : log(1.0f + exp(alpha)));
            beta_value = 1.0f / (1.0f + exp(-beta_value));
        }
        const float decay = exp(decay_log);
        float predicted = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            state[i] *= decay;
            predicted += state[i] * (keys[simd * 32 + i] * knorm);
        }
        partials[tid] = predicted;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        predicted = partials[lane] + partials[32+lane] + partials[64+lane] + partials[96+lane];
        const float delta = (float(v[token * $VT + h * $VHSTR + col * $VD]) - predicted) * beta_value;
        // Every thread must finish reading the previous reduction before any
        // thread reuses the shared buffer for the output reduction.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float result = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            const uint row = simd * 32 + i;
            state[i] += keys[row] * knorm * delta;
            result += state[i] * (queries[row] * qnorm * $SCALE);
        }
        partials[tid] = result;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            output[token * $OT + h * $OH + col * $OD] = $TYPE(
                partials[lane] + partials[32+lane] + partials[64+lane] + partials[96+lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 32; ++i) {
        final_state[write_row * $FS + h * 16384 + (simd * 32 + i) * 128 + col] = state[i];
    }
}
"""
    values = dict(
        TYPE=data_type,
        GATE=data_type if raw_gates else "float",
        QT=q_stride[0],
        QHSTR=q_stride[1],
        QD=q_stride[2],
        KT=k_stride[0],
        KHSTR=k_stride[1],
        KD=k_stride[2],
        VT=v_stride[0],
        VHSTR=v_stride[1],
        VD=v_stride[2],
        GT=g_stride[0],
        GH=g_stride[1],
        BT=b_stride[0],
        BH=b_stride[1],
        SS=state_stride,
        FS=final_stride,
        OT=out_stride[0],
        OH=out_stride[1],
        OD=out_stride[2],
        QH=query_heads,
        VH=value_heads,
        NORMALIZE=int(normalize),
        RAW=int(raw_gates),
        SCALE=f"float({scale!r})",
        INDEXED=int(indexed_decode),
        VARLEN=int(gpu_varlen),
        INDEX=index_type,
        TOKENS=token_count,
        STATES=state_count,
    )
    # Longer names first: $QH must not consume the prefix of $QHSTR.
    for key in sorted(values, key=len, reverse=True):
        source = source.replace("$" + key, str(values[key]))
    return torch.mps.compile_shader(source)


def recurrent(
    q,
    k,
    v,
    g,
    beta,
    initial,
    final,
    output,
    sequence_ranges,
    read_indices,
    write_indices,
    normalize,
    scale,
    A_log,
    dt_bias,
    a_raw,
    b_raw,
    inplace,
):
    """Return False when the reference path must preserve a broader contract."""
    if not hasattr(torch.mps, "compile_shader") or q.device.type != "mps":
        return False
    if q.dtype not in (torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if output.dtype != q.dtype or output.device != q.device:
        return False
    if q.shape[-1] != 128 or v.shape[-1] != 128 or final.dtype != torch.float32:
        return False
    if initial is not None and (initial.dtype != torch.float32 or initial.stride()[-3:] != (16384, 128, 1)):
        return False
    if final.stride()[-3:] != (16384, 128, 1):
        return False
    if read_indices is not None and read_indices.ndim != 1:
        return False
    if write_indices is not None and write_indices.ndim != 1:
        return False
    count = len(sequence_ranges)
    reads = read_indices.tolist() if read_indices is not None else list(range(count))
    writes = (write_indices.tolist() if write_indices is not None else reads) if inplace else list(range(count))
    if min(reads + writes, default=0) < 0 or len(set(writes)) != len(writes):
        return False
    if (initial is not None and max(reads, default=-1) >= initial.shape[0]) or max(writes, default=-1) >= final.shape[
        0
    ]:
        return False
    if not count:
        return True
    # Distinct threadgroups may not overwrite another sequence's input state.
    if inplace and any(w in reads and reads.index(w) != i for i, w in enumerate(writes)):
        return False
    raw = A_log is not None
    gate, mix = (a_raw, b_raw) if raw else (g, beta)
    if gate is None or mix is None:
        return False
    if not raw and (gate.dtype != torch.float32 or mix.dtype != torch.float32):
        return False
    if raw and (gate.dtype != q.dtype or mix.dtype != q.dtype):
        return False
    if gate.shape != (q.shape[0], v.shape[1]) or mix.shape != gate.shape:
        return False
    metadata = torch.tensor(
        [[begin, end, read, write] for (begin, end), read, write in zip(sequence_ranges, reads, writes)],
        dtype=torch.int32,
        device=q.device,
    )
    initial_tensor = initial if initial is not None else final
    lib = _library(
        "half" if q.dtype == torch.float16 else "float",
        q.stride(),
        k.stride(),
        v.stride(),
        gate.stride(),
        mix.stride(),
        initial_tensor.stride(0),
        final.stride(0),
        output.stride(),
        q.shape[1],
        v.shape[1],
        normalize,
        raw,
        float(scale),
    )
    lib.recurrent(
        q,
        k,
        v,
        gate,
        mix,
        initial_tensor,
        final,
        output,
        metadata,
        A_log.reshape(-1).float() if raw else final,
        dt_bias.reshape(-1).float() if raw else final,
        initial is not None,
        threads=(4 * 128, v.shape[1], count),
        group_size=(128, 1, 1),
    )
    return True


def decode(q, k, v, g, beta, initial, indices, normalize, scale, A_log, dt_bias, a_raw, b_raw, out):
    """Ordinary in-place decode without copying state indices to the CPU."""
    if not hasattr(torch.mps, "compile_shader") or q.device.type != "mps":
        return None
    if q.dtype not in (torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if q.shape[-1] != 128 or v.shape[-1] != 128 or initial.dtype != torch.float32:
        return None
    if (
        initial.stride()[-3:] != (16384, 128, 1)
        or indices.device != q.device
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.stride() != (1,)
    ):
        return None
    batch = q.shape[0]
    qf = q.reshape(batch, q.shape[2], 128)
    kf = k.reshape_as(qf)
    vf = v.reshape(batch, v.shape[2], 128)
    raw = A_log is not None
    gate, mix = (a_raw, b_raw) if raw else (g, beta)
    if gate is None or mix is None:
        return None
    gate = gate.reshape(batch, -1)
    mix = mix.reshape(batch, -1)
    if gate.shape != (batch, v.shape[2]) or mix.shape != gate.shape:
        return None
    if gate.dtype != (q.dtype if raw else torch.float32) or mix.dtype != gate.dtype:
        return None
    output = torch.empty(vf.shape, dtype=v.dtype, device=v.device) if out is None else out.reshape_as(vf)
    if output.dtype != q.dtype or output.device != q.device:
        return None
    if not batch:
        return output.view_as(v), initial
    if scale is None:
        scale = 128 ** -0.5
    lib = _library(
        "half" if q.dtype == torch.float16 else "float",
        qf.stride(),
        kf.stride(),
        vf.stride(),
        gate.stride(),
        mix.stride(),
        initial.stride(0),
        initial.stride(0),
        output.stride(),
        q.shape[2],
        v.shape[2],
        normalize,
        raw,
        float(scale),
        True,
        "long" if indices.dtype == torch.int64 else "int",
        batch,
        initial.shape[0],
    )
    lib.recurrent(
        qf,
        kf,
        vf,
        gate,
        mix,
        initial,
        initial,
        output,
        indices,
        A_log.reshape(-1).float() if raw else initial,
        dt_bias.reshape(-1).float() if raw else initial,
        True,
        threads=(4 * 128, v.shape[2], batch),
        group_size=(128, 1, 1),
    )
    return output.view_as(v), initial


def prefill(q, k, v, g, beta, initial, boundaries, output, normalize, scale):
    """Varlen prefill with GPU sequence boundaries and independent returned states."""
    if not hasattr(torch.mps, "compile_shader") or q.device.type != "mps":
        return None
    if initial is None or g is None or beta is None or boundaries.ndim != 1 or boundaries.stride() != (1,):
        return None
    if boundaries.dtype not in (torch.int32, torch.int64) or boundaries.numel() < 1:
        return None
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3 or q.shape[-1] != 128 or v.shape[-1] != 128:
        return None
    if q.shape[0] != v.shape[0] or not q.shape[1] or v.shape[1] % q.shape[1]:
        return None
    if q.dtype not in (torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    batch = boundaries.numel() - 1
    if initial.dtype != torch.float32 or initial.shape != (batch, v.shape[1], 128, 128):
        return None
    if initial.stride()[-3:] != (16384, 128, 1):
        return None
    if g.shape != (q.shape[0], v.shape[1]) or beta.shape != g.shape:
        return None
    if g.dtype != torch.float32 or beta.dtype != torch.float32 or output.dtype != q.dtype or output.shape != v.shape:
        return None
    if any(t.device != q.device for t in (k, v, g, beta, initial, boundaries, output)):
        return None
    final = torch.empty(initial.shape, dtype=initial.dtype, device=initial.device)
    if not q.shape[0]:
        final.copy_(initial)
        return output, final
    if not batch:
        return None
    lib = _library(
        "half" if q.dtype == torch.float16 else "float",
        q.stride(),
        k.stride(),
        v.stride(),
        g.stride(),
        beta.stride(),
        initial.stride(0),
        final.stride(0),
        output.stride(),
        q.shape[1],
        v.shape[1],
        normalize,
        False,
        float(scale),
        index_type="long" if boundaries.dtype == torch.int64 else "int",
        gpu_varlen=True,
    )
    lib.recurrent(
        q,
        k,
        v,
        g,
        beta,
        initial,
        final,
        output,
        boundaries,
        final,
        final,
        True,
        threads=(4 * 128, v.shape[1], batch),
        group_size=(128, 1, 1),
    )
    return output, final
