import torch

import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    Q,
    K,
    Cos,
    Sin,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_cosbs,
    stride_cosd,
    stride_sinbs,
    stride_sind,
    max_total_len,
    HEAD_Q,
    HEAD_K,  # N_CTX 代表要计算的上下文长度
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_head_index = tl.program_id(0)
    cur_seq_index = tl.program_id(1)

    cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)
    dim_range1 = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)

    off_q0 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range0[None, None, :] * stride_qd
    )
    off_q1 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range1[None, None, :] * stride_qd
    )

    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd

    q0 = tl.load(
        Q + off_q0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q),
        other=0.0,
    )
    q1 = tl.load(
        Q + off_q1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q),
        other=0.0,
    )

    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos

    tl.store(
        Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q)
    )
    tl.store(
        Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q)
    )

    off_k0 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range0[None, None, :] * stride_kd
    )
    off_k1 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range1[None, None, :] * stride_kd
    )

    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd

    k0 = tl.load(
        K + off_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
        other=0.0,
    )
    k1 = tl.load(
        K + off_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
        other=0.0,
    )
    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out_k0 = k0 * cos - k1 * sin
    out_k1 = k0 * sin + k1 * cos

    tl.store(
        K + off_k0,
        out_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
    )
    tl.store(
        K + off_k1,
        out_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
    )
    return


@torch.no_grad()
def rotary_emb_fwd(q, k, cos, sin, partial_rotary_factor=1.0):
    total_len = q.shape[0]
    head_num_q, head_num_k = q.shape[1], k.shape[1]
    head_dim = int(q.shape[2] * partial_rotary_factor)
    assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
    assert k.shape[0] == cos.shape[0] and k.shape[0] == sin.shape[0], f"k shape {k.shape} cos shape {cos.shape}"

    BLOCK_SEQ = 16
    BLOCK_HEAD = 4
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    grid = (triton.cdiv(head_num_q, BLOCK_HEAD), triton.cdiv(total_len, BLOCK_SEQ))
    _rotary_kernel[grid](
        q,
        k,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        total_len,
        head_num_q,
        head_num_k,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _rotary_kernel_fused(
    Q,
    K,
    Cos,
    Sin,
    Mem_index,
    Kv_buffer,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_cosbs,
    stride_cosd,
    stride_sinbs,
    stride_sind,
    stride_kv_buffer_bs,
    stride_kv_buffer_h,
    stride_kv_buffer_d,
    max_total_len,
    HEAD_Q,
    HEAD_K,  # N_CTX 代表要计算的上下文长度
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_head_index = tl.program_id(0)
    cur_seq_index = tl.program_id(1)

    cur_head_start = cur_head_index * BLOCK_HEAD
    cur_head_range = cur_head_start + tl.arange(0, BLOCK_HEAD)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)
    dim_range1 = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)

    off_q0 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range0[None, None, :] * stride_qd
    )
    off_q1 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range1[None, None, :] * stride_qd
    )

    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd

    q0 = tl.load(
        Q + off_q0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q),
        other=0.0,
    )
    q1 = tl.load(
        Q + off_q1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q),
        other=0.0,
    )

    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos

    tl.store(
        Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q)
    )
    tl.store(
        Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q)
    )

    if cur_head_start >= HEAD_K:
        return

    off_k0 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range0[None, None, :] * stride_kd
    )
    off_k1 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range1[None, None, :] * stride_kd
    )

    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd

    k0 = tl.load(
        K + off_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
        other=0.0,
    )
    k1 = tl.load(
        K + off_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
        other=0.0,
    )
    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out_k0 = k0 * cos - k1 * sin
    out_k1 = k0 * sin + k1 * cos

    # Load dest_index with proper mask - use a safe default value that won't cause out-of-bounds access
    # We need to ensure dest_index is only used when cur_seq_range < max_total_len
    seq_mask = cur_seq_range < max_total_len
    dest_index = tl.load(Mem_index + cur_seq_range, mask=seq_mask, other=0).to(tl.int64)
    kv_buffer_ptrs = (
        Kv_buffer + dest_index[:, None, None] * stride_kv_buffer_bs + stride_kv_buffer_h * cur_head_range[None, :, None]
    )

    tl.store(
        kv_buffer_ptrs + dim_range0[None, None, :] * stride_kv_buffer_d,
        out_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
    )
    tl.store(
        kv_buffer_ptrs + dim_range1[None, None, :] * stride_kv_buffer_d,
        out_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
    )

    # Copy V to kv_buffer (V doesn't need rotary embedding, just copy)
    # V is stored right after K in cache_kv, so access via K pointer with head offset HEAD_K
    head_k_v_mask = (cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K)
    off_v0 = (
        cur_seq_range[:, None, None] * stride_kbs
        + (HEAD_K + cur_head_range[None, :, None]) * stride_kh
        + dim_range0[None, None, :] * stride_kd
    )
    off_v1 = (
        cur_seq_range[:, None, None] * stride_kbs
        + (HEAD_K + cur_head_range[None, :, None]) * stride_kh
        + dim_range1[None, None, :] * stride_kd
    )

    v0 = tl.load(K + off_v0, mask=head_k_v_mask, other=0.0)
    v1 = tl.load(K + off_v1, mask=head_k_v_mask, other=0.0)

    kv_buffer_v_ptrs = (
        Kv_buffer
        + dest_index[:, None, None] * stride_kv_buffer_bs
        + stride_kv_buffer_h * (HEAD_K + cur_head_range[None, :, None])
    )

    tl.store(kv_buffer_v_ptrs + dim_range0[None, None, :] * stride_kv_buffer_d, v0, mask=head_k_v_mask)
    tl.store(kv_buffer_v_ptrs + dim_range1[None, None, :] * stride_kv_buffer_d, v1, mask=head_k_v_mask)
    return


@torch.no_grad()
def rotary_emb_fwd_fused(q, k, cos, sin, mem_index, kv_buffer, partial_rotary_factor=1.0):
    total_len = q.shape[0]
    head_num_q, head_num_k = q.shape[1], k.shape[1]
    head_dim = int(q.shape[2] * partial_rotary_factor)
    assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
    assert k.shape[0] == cos.shape[0] and k.shape[0] == sin.shape[0], f"k shape {k.shape} cos shape {cos.shape}"

    BLOCK_SEQ = 16
    BLOCK_HEAD = 4
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    grid = (triton.cdiv(head_num_q, BLOCK_HEAD), triton.cdiv(total_len, BLOCK_SEQ))
    _rotary_kernel_fused[grid](
        q,
        k,
        cos,
        sin,
        mem_index,
        kv_buffer,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        kv_buffer.stride(0),
        kv_buffer.stride(1),
        kv_buffer.stride(2),
        total_len,
        head_num_q,
        head_num_k,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    dim = dim // 4
    x0 = x[:, :, 0 : dim // 2]
    x1 = x[:, :, dim // 2 : dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)


def test_rotary_emb(SEQ_LEN, H, D, dtype, eps=1e-5, device="cuda"):
    # create data
    x_shape = (SEQ_LEN, H, D)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    cos_shape = (SEQ_LEN, D // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    # forward pass
    y_tri = torch_rotary_emb(x, cos, sin)
    rotary_emb_fwd(x, cos, sin)
    y_ref = x

    # compare
    print("type:", y_tri.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_tri - y_ref)))
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
