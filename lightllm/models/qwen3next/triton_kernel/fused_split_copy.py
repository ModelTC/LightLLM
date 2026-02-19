"""
Fused Split-Copy Triton Kernels for GDN Decode Path

Replaces multiple separate .copy_() calls with single kernel launches to reduce
kernel launch overhead in the decode hot path (36 GDN layers per step).

Kernel 1 (fused_split_copy_qkvzba): 4 copies → 1 kernel
    Splits GEMM output [batch, total_dim] into qkv, z, b, a destination buffers.

Kernel 2 (fused_split_copy_qkv): 3 copies → 1 kernel
    Splits conv1d output [batch, qkv_dim] into q, k, v destination buffers.
    Handles non-contiguous source (stride(0) != total_dim from column slicing).
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Kernel 1: Fused split-copy for qkv, z, b, a from GEMM output
# =============================================================================


@triton.jit
def _fused_split_copy_qkvzba_kernel(
    # Source pointer (contiguous GEMM output)
    src_ptr,
    # Destination pointers (pre-allocated contiguous buffers)
    dst_qkv_ptr,
    dst_z_ptr,
    dst_b_ptr,
    dst_a_ptr,
    # Row strides
    src_stride0,
    dst_qkv_stride0,
    dst_z_stride0,
    dst_b_stride0,
    dst_a_stride0,
    # Segment boundaries (cumulative): [0, qkv_dim) [qkv_dim, z_end) [z_end, b_end) [b_end, total_dim)
    qkv_dim,
    z_end,
    b_end,
    total_dim,
    # Block size
    BLOCK_N: tl.constexpr,
):
    """
    One program per (row, column_block). Loads a BLOCK_N chunk from the source row,
    then conditionally stores to the correct destination based on column position.

    Grid: (batch, cdiv(total_dim, BLOCK_N))
    """
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    col_start = col_block * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    mask = cols < total_dim

    # Load source chunk
    data = tl.load(src_ptr + row * src_stride0 + cols, mask=mask)

    # Store to qkv destination: columns [0, qkv_dim)
    qkv_mask = mask & (cols < qkv_dim)
    tl.store(dst_qkv_ptr + row * dst_qkv_stride0 + cols, data, mask=qkv_mask)

    # Store to z destination: columns [qkv_dim, z_end)
    z_mask = mask & (cols >= qkv_dim) & (cols < z_end)
    tl.store(dst_z_ptr + row * dst_z_stride0 + (cols - qkv_dim), data, mask=z_mask)

    # Store to b destination: columns [z_end, b_end)
    b_mask = mask & (cols >= z_end) & (cols < b_end)
    tl.store(dst_b_ptr + row * dst_b_stride0 + (cols - z_end), data, mask=b_mask)

    # Store to a destination: columns [b_end, total_dim)
    a_mask = mask & (cols >= b_end)
    tl.store(dst_a_ptr + row * dst_a_stride0 + (cols - b_end), data, mask=a_mask)


def fused_split_copy_qkvzba(
    src: torch.Tensor,
    dst_qkv: torch.Tensor,
    dst_z: torch.Tensor,
    dst_b: torch.Tensor,
    dst_a: torch.Tensor,
    qkv_dim: int,
    z_dim: int,
    b_dim: int,
    a_dim: int,
):
    """
    Fused split-copy from GEMM output into 4 contiguous destination buffers.

    Replaces:
        conv_buf.copy_(mixed_qkvzba[:, :qkv_dim])
        z_buf.view(batch, -1).copy_(mixed_qkvzba[:, qkv_dim:z_end])
        b_buf.copy_(mixed_qkvzba[:, z_end:b_end])
        a_buf.copy_(mixed_qkvzba[:, b_end:])

    Args:
        src: [batch, total_dim] contiguous source (GEMM output)
        dst_qkv: [batch, qkv_dim] contiguous destination for conv1d input
        dst_z: [batch, z_dim] contiguous destination (z_buf viewed flat)
        dst_b: [batch, b_dim] contiguous destination
        dst_a: [batch, a_dim] contiguous destination
        qkv_dim: width of qkv segment (tp_key_dim * 2 + tp_value_dim)
        z_dim: width of z segment (tp_value_dim)
        b_dim: width of b segment (tp_num_v_heads)
        a_dim: width of a segment (tp_num_v_heads)
    """
    total_dim = qkv_dim + z_dim + b_dim + a_dim
    z_end = qkv_dim + z_dim
    b_end = z_end + b_dim

    batch = src.shape[0]
    BLOCK_N = 128
    num_col_blocks = triton.cdiv(total_dim, BLOCK_N)

    grid = (batch, num_col_blocks)

    _fused_split_copy_qkvzba_kernel[grid](
        src,
        dst_qkv,
        dst_z,
        dst_b,
        dst_a,
        src.stride(0),
        dst_qkv.stride(0),
        dst_z.stride(0),
        dst_b.stride(0),
        dst_a.stride(0),
        qkv_dim,
        z_end,
        b_end,
        total_dim,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


# =============================================================================
# Kernel 2: Fused split-copy for q, k, v from conv1d output
# =============================================================================


@triton.jit
def _fused_split_copy_qkv_kernel(
    # Source pointer (may be non-contiguous column slice)
    src_ptr,
    # Destination pointers (contiguous buffers)
    dst_q_ptr,
    dst_k_ptr,
    dst_v_ptr,
    # Row strides
    src_stride0,
    dst_q_stride0,
    dst_k_stride0,
    dst_v_stride0,
    # Segment boundaries: [0, q_dim) [q_dim, qk_end) [qk_end, total_dim)
    q_dim,
    qk_end,
    total_dim,
    # Block size
    BLOCK_N: tl.constexpr,
):
    """
    One program per (row, column_block). Loads a BLOCK_N chunk from the source row,
    then conditionally stores to q, k, or v destination.

    Supports non-contiguous source via src_stride0 (stride may be > total_dim
    when source is a column slice of a larger tensor).

    Grid: (batch, cdiv(total_dim, BLOCK_N))
    """
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    col_start = col_block * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    mask = cols < total_dim

    # Load source chunk (use src_stride0 for row advancement)
    data = tl.load(src_ptr + row * src_stride0 + cols, mask=mask)

    # Store to q destination: columns [0, q_dim)
    q_mask = mask & (cols < q_dim)
    tl.store(dst_q_ptr + row * dst_q_stride0 + cols, data, mask=q_mask)

    # Store to k destination: columns [q_dim, qk_end)
    k_mask = mask & (cols >= q_dim) & (cols < qk_end)
    tl.store(dst_k_ptr + row * dst_k_stride0 + (cols - q_dim), data, mask=k_mask)

    # Store to v destination: columns [qk_end, total_dim)
    v_mask = mask & (cols >= qk_end)
    tl.store(dst_v_ptr + row * dst_v_stride0 + (cols - qk_end), data, mask=v_mask)


def fused_split_copy_qkv(
    src: torch.Tensor,
    dst_q: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    src_stride0: int,
):
    """
    Fused split-copy from conv1d output into 3 contiguous q/k/v buffers.

    Replaces:
        q_split, k_split, v_split = torch.split(mixed_qkv, [...], dim=-1)
        q_buf.view(batch, -1).copy_(q_split)
        k_buf.view(batch, -1).copy_(k_split)
        v_buf.view(batch, -1).copy_(v_split)

    Args:
        src: [batch, total_dim] source tensor (may be non-contiguous if column slice)
        dst_q: [batch, q_dim] contiguous destination
        dst_k: [batch, k_dim] contiguous destination
        dst_v: [batch, v_dim] contiguous destination
        q_dim: width of q segment (tp_key_dim)
        k_dim: width of k segment (tp_key_dim)
        v_dim: width of v segment (tp_value_dim)
        src_stride0: row stride of source (may be > q_dim+k_dim+v_dim)
    """
    total_dim = q_dim + k_dim + v_dim
    qk_end = q_dim + k_dim

    batch = src.shape[0]
    BLOCK_N = 128
    num_col_blocks = triton.cdiv(total_dim, BLOCK_N)

    grid = (batch, num_col_blocks)

    _fused_split_copy_qkv_kernel[grid](
        src,
        dst_q,
        dst_k,
        dst_v,
        src_stride0,
        dst_q.stride(0),
        dst_k.stride(0),
        dst_v.stride(0),
        q_dim,
        qk_end,
        total_dim,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


# =============================================================================
# Test / Verification
# =============================================================================


def test_fused_split_copy():
    """Verify fused kernels produce identical results to separate .copy_() calls."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    print("=" * 60)
    print("Testing fused_split_copy_qkvzba")
    print("=" * 60)

    # Typical dimensions for Qwen3-Coder-Next with TP=4
    # tp_key_dim=128, tp_value_dim=256, tp_num_v_heads=2
    qkv_dim = 128 + 128 + 256  # q + k + v = 512
    z_dim = 256
    b_dim = 2
    a_dim = 2
    total_dim = qkv_dim + z_dim + b_dim + a_dim  # 772

    for batch in [1, 4, 8, 32]:
        src = torch.randn(batch, total_dim, dtype=dtype, device=device)

        # Reference: separate copies
        ref_qkv = src[:, :qkv_dim].clone()
        ref_z = src[:, qkv_dim : qkv_dim + z_dim].clone()
        ref_b = src[:, qkv_dim + z_dim : qkv_dim + z_dim + b_dim].clone()
        ref_a = src[:, qkv_dim + z_dim + b_dim :].clone()

        # Fused kernel
        dst_qkv = torch.empty(batch, qkv_dim, dtype=dtype, device=device)
        dst_z = torch.empty(batch, z_dim, dtype=dtype, device=device)
        dst_b = torch.empty(batch, b_dim, dtype=dtype, device=device)
        dst_a = torch.empty(batch, a_dim, dtype=dtype, device=device)
        fused_split_copy_qkvzba(src, dst_qkv, dst_z, dst_b, dst_a, qkv_dim, z_dim, b_dim, a_dim)

        assert torch.equal(dst_qkv, ref_qkv), f"qkv mismatch at batch={batch}"
        assert torch.equal(dst_z, ref_z), f"z mismatch at batch={batch}"
        assert torch.equal(dst_b, ref_b), f"b mismatch at batch={batch}"
        assert torch.equal(dst_a, ref_a), f"a mismatch at batch={batch}"
        print(f"  batch={batch:3d}: PASS")

    print()
    print("=" * 60)
    print("Testing fused_split_copy_qkv")
    print("=" * 60)

    q_dim = 128
    k_dim = 128
    v_dim = 256
    qkv_dim = q_dim + k_dim + v_dim  # 512

    for batch in [1, 4, 8, 32]:
        # Test with contiguous source
        src = torch.randn(batch, qkv_dim, dtype=dtype, device=device)

        ref_q = src[:, :q_dim].clone()
        ref_k = src[:, q_dim : q_dim + k_dim].clone()
        ref_v = src[:, q_dim + k_dim :].clone()

        dst_q = torch.empty(batch, q_dim, dtype=dtype, device=device)
        dst_k = torch.empty(batch, k_dim, dtype=dtype, device=device)
        dst_v = torch.empty(batch, v_dim, dtype=dtype, device=device)
        fused_split_copy_qkv(src, dst_q, dst_k, dst_v, q_dim, k_dim, v_dim, src.stride(0))

        assert torch.equal(dst_q, ref_q), f"q mismatch at batch={batch} (contiguous)"
        assert torch.equal(dst_k, ref_k), f"k mismatch at batch={batch} (contiguous)"
        assert torch.equal(dst_v, ref_v), f"v mismatch at batch={batch} (contiguous)"
        print(f"  batch={batch:3d} (contiguous src):     PASS")

        # Test with non-contiguous source (column slice of wider tensor)
        wider = torch.randn(batch, qkv_dim + 64, dtype=dtype, device=device)
        src_nc = wider[:, :qkv_dim]  # Non-contiguous: stride(0) = qkv_dim + 64
        assert src_nc.stride(0) == qkv_dim + 64, "expected non-contiguous slice"

        ref_q = src_nc[:, :q_dim].clone()
        ref_k = src_nc[:, q_dim : q_dim + k_dim].clone()
        ref_v = src_nc[:, q_dim + k_dim :].clone()

        dst_q = torch.empty(batch, q_dim, dtype=dtype, device=device)
        dst_k = torch.empty(batch, k_dim, dtype=dtype, device=device)
        dst_v = torch.empty(batch, v_dim, dtype=dtype, device=device)
        fused_split_copy_qkv(src_nc, dst_q, dst_k, dst_v, q_dim, k_dim, v_dim, src_nc.stride(0))

        assert torch.equal(dst_q, ref_q), f"q mismatch at batch={batch} (non-contiguous)"
        assert torch.equal(dst_k, ref_k), f"k mismatch at batch={batch} (non-contiguous)"
        assert torch.equal(dst_v, ref_v), f"v mismatch at batch={batch} (non-contiguous)"
        print(f"  batch={batch:3d} (non-contiguous src): PASS")

    print()
    print("=" * 60)
    print("Testing edge cases")
    print("=" * 60)

    # Edge case: different dimension ratios (small q/k, large v)
    q_dim, k_dim, v_dim = 32, 32, 512
    qkv_dim = q_dim + k_dim + v_dim
    batch = 2
    src = torch.randn(batch, qkv_dim, dtype=dtype, device=device)

    dst_q = torch.empty(batch, q_dim, dtype=dtype, device=device)
    dst_k = torch.empty(batch, k_dim, dtype=dtype, device=device)
    dst_v = torch.empty(batch, v_dim, dtype=dtype, device=device)
    fused_split_copy_qkv(src, dst_q, dst_k, dst_v, q_dim, k_dim, v_dim, src.stride(0))

    assert torch.equal(dst_q, src[:, :q_dim])
    assert torch.equal(dst_k, src[:, q_dim : q_dim + k_dim])
    assert torch.equal(dst_v, src[:, q_dim + k_dim :])
    print("  asymmetric dims (32, 32, 512): PASS")

    # Edge case: float32 dtype
    src_f32 = torch.randn(4, 772, dtype=torch.float32, device=device)
    dst_qkv = torch.empty(4, 512, dtype=torch.float32, device=device)
    dst_z = torch.empty(4, 256, dtype=torch.float32, device=device)
    dst_b = torch.empty(4, 2, dtype=torch.float32, device=device)
    dst_a = torch.empty(4, 2, dtype=torch.float32, device=device)
    fused_split_copy_qkvzba(src_f32, dst_qkv, dst_z, dst_b, dst_a, 512, 256, 2, 2)

    assert torch.equal(dst_qkv, src_f32[:, :512])
    assert torch.equal(dst_z, src_f32[:, 512:768])
    assert torch.equal(dst_b, src_f32[:, 768:770])
    assert torch.equal(dst_a, src_f32[:, 770:])
    print("  float32 dtype: PASS")

    # Edge case: float16 dtype
    src_f16 = torch.randn(4, 772, dtype=torch.float16, device=device)
    dst_qkv = torch.empty(4, 512, dtype=torch.float16, device=device)
    dst_z = torch.empty(4, 256, dtype=torch.float16, device=device)
    dst_b = torch.empty(4, 2, dtype=torch.float16, device=device)
    dst_a = torch.empty(4, 2, dtype=torch.float16, device=device)
    fused_split_copy_qkvzba(src_f16, dst_qkv, dst_z, dst_b, dst_a, 512, 256, 2, 2)

    assert torch.equal(dst_qkv, src_f16[:, :512])
    assert torch.equal(dst_z, src_f16[:, 512:768])
    assert torch.equal(dst_b, src_f16[:, 768:770])
    assert torch.equal(dst_a, src_f16[:, 770:])
    print("  float16 dtype: PASS")

    print()
    print("All tests passed!")


if __name__ == "__main__":
    test_fused_split_copy()
