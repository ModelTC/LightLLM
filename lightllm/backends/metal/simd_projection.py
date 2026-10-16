"""Vector-loaded SIMD projections for measured Qwen FP16 shapes on Metal.

FP16 operands retain FP32 matrix accumulation and the ordinary FP16 output
rounding. The token count is a runtime argument, avoiding length-dependent JITs.
"""

from functools import lru_cache
import torch


@lru_cache(maxsize=128)
def _library(size, outputs, tm, tn, tk, groups, padding, partial_rows=False):
    m_groups = 1 if tm == 8 else 2
    n_groups = groups // m_groups
    fm, fn = tm // (m_groups * 8), tn // (n_groups * 8)
    init = "\n".join(
        f"simdgroup_float8x8 c_{i}_{j}=make_filled_simdgroup_matrix<float,8>(0.0f);"
        for i in range(fm)
        for j in range(fn)
    )
    loads = []
    for i in range(fm):
        loads.append(f"simdgroup_half8x8 a{i};simdgroup_load(a{i},as+(sm+{i * 8})*$SK+kk,$SK);")
    for j in range(fn):
        loads.append(f"simdgroup_half8x8 b{j};simdgroup_load(b{j},bs+(sn+{j * 8})*$SK+kk,$SK,ulong2(0,0),true);")
    mma = "\n".join(
        f"simdgroup_multiply_accumulate(c_{i}_{j},a{i},b{j},c_{i}_{j});" for i in range(fm) for j in range(fn)
    )
    stores = []
    for i in range(fm):
        for j in range(fn):
            store = f"""if(m0+sm+{i * 8}+8<=$M && n0+sn+{j * 8}+8<=$N) {{
                temp.thread_elements()[0]=half(c_{i}_{j}.thread_elements()[0]);
                temp.thread_elements()[1]=half(c_{i}_{j}.thread_elements()[1]);
                simdgroup_store(temp,output+(m0+sm+{i * 8})*$N+n0+sn+{j * 8},$N);
            }}"""
            if partial_rows:
                store += f"""else if(m0+sm+{i * 8}<$M && n0+sn+{j * 8}<$N) {{
                    for(uint e=0;e<2;++e) {{
                        const uint pos=uint(coordinates.thread_elements()[e]);
                        const uint m=m0+sm+{i * 8}+pos/8,n=n0+sn+{j * 8}+pos%8;
                        if(m<$M && n<$N) output[m*$N+n]=half(c_{i}_{j}.thread_elements()[e]);
                    }}
                }}"""
            stores.append(store)
    source = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void gemm(const device half* x [[buffer(0)]],const device half* w [[buffer(1)]],
                device half* output [[buffer(2)]],constant long& batch [[buffer(3)]],
                uint3 block [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint m0=block.y*$TM,n0=block.x*$TN,sm=(sg/$NG)*($TM/$MG),sn=(sg%$NG)*($TN/$NG);
    threadgroup half as[$TM*$SK],bs[$TN*$SK];
    $INITIALIZE
    simdgroup_half8x8 temp;
    $COORDINATES
    for(uint k0=0;k0<$K;k0+=$TK) {
        for(uint j=tid;j<$TM*$TK/4;j+=$THREADS) {
            const uint m=j/($TK/4),k=(j%($TK/4))*4;
            half4 val=(m0+m<$M && k0+k<$K) ? *((const device half4*)(x+(m0+m)*$K+k0+k)) : half4(0);
            *((threadgroup half4*)(as+m*$SK+k))=val;
        }
        for(uint j=tid;j<$TN*$TK/4;j+=$THREADS) {
            const uint n=j/($TK/4),k=(j%($TK/4))*4;
            half4 val=(n0+n<$N && k0+k<$K) ? *((const device half4*)(w+(n0+n)*$K+k0+k)) : half4(0);
            *((threadgroup half4*)(bs+n*$SK+k))=val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kk=0;kk<$TK;kk+=8) {
            $LOADS
            $MMA
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    $STORES
}
"""
    # Query the public matrix API for this float fragment's element coordinates.
    # Only the last partial M tile needs them; complete-row shaders retain the
    # original source and shared-memory footprint.
    coordinates = (
        """
    threadgroup float coordinate_grid[64];
    simdgroup_float8x8 coordinates;
    if(m0+$TM>$M) {
        if(tid<64) coordinate_grid[tid]=float(tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(coordinates,coordinate_grid,8);
    }
    """
        if partial_rows
        else ""
    )
    source = source.replace("$COORDINATES", coordinates)
    for key, val in [("INITIALIZE", init), ("LOADS", "\n".join(loads)), ("MMA", mma), ("STORES", "\n".join(stores))]:
        source = source.replace("$" + key, val)
    params = dict(
        TM=tm,
        TN=tn,
        TK=tk,
        SK=tk + padding,
        MG=m_groups,
        NG=n_groups,
        THREADS=groups * 32,
        M="batch",
        K=size,
        N=outputs,
    )
    for key in sorted(params, key=len, reverse=True):
        source = source.replace("$" + key, str(params[key]))
    return torch.mps.compile_shader(source)


def project(x, weight, out=None, bias=None):
    """Return output, or None without writes when this measured path cannot apply."""
    if not hasattr(torch.mps, "compile_shader") or x.device.type != "mps":
        return None
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        return None
    batch, size = x.shape
    rows = weight.shape[0]
    if not 8 <= batch <= 272:
        return None
    if (size, rows) not in ((1024, 5120), (1024, 7168), (1024, 8224), (2048, 1024), (3584, 1024)):
        return None
    if x.dtype != torch.float16 or weight.dtype != x.dtype or weight.device != x.device or bias is not None:
        return None
    if x.requires_grad or weight.requires_grad or not x.is_contiguous() or not weight.is_contiguous():
        return None
    if x.storage_offset() % 4 or weight.storage_offset() % 4:
        return None
    if out is not None:
        if out.shape != (batch, rows) or out.dtype != x.dtype or out.device != x.device or out.requires_grad:
            return None
        if not out.is_contiguous() or out.storage_offset() % 4:
            return None
        if out.untyped_storage().data_ptr() in {x.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr()}:
            return None
    else:
        out = torch.empty((batch, rows), dtype=x.dtype, device=x.device)
    if batch == 8 and size == 1024:
        config = (8, 64, 64, 8, 8) if rows == 7168 else (8, 32, 64, 4, 0)
    elif batch <= 16:
        if rows == 7168:
            config = (16, 32, 32, 4, 8)
        elif rows in (5120, 8224):
            config = (16, 32, 64, 4, 0)
        elif size == 2048:
            config = (16, 32, 32, 8, 0)
        else:
            config = (16, 32, 64, 8, 8)
    elif batch < 32 and size != 1024 and batch % 8:
        config = (16, 32, 32, 8, 0) if size == 2048 else (16, 32, 64, 8, 8)
    elif batch == 24:
        config = (32, 64, 32, 4, 8)
    elif size == 1024:
        config = (64, 64, 32, 4, 8)
    else:
        config = (32, 64, 32, 4, 8)
    tm, tn, tk, groups, padding = config
    lib = _library(size, rows, *config, batch % 8 != 0)
    threads = groups * 32
    lib.gemm(
        x,
        weight,
        out,
        batch,
        threads=(((rows + tn - 1) // tn) * threads, (batch + tm - 1) // tm, 1),
        group_size=(threads, 1, 1),
    )
    return out
