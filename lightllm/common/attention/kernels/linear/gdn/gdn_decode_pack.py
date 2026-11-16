import torch
import triton
import triton.language as tl


@triton.jit
def _conv_pack_gdn_decode_kernel(
    mixed_qkv,
    z_raw,
    a_raw,
    b_raw,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    q_out,
    k_out,
    v_out,
    z_out,
    a_out,
    b_out,
    stride_m_b: tl.constexpr,
    stride_m_d: tl.constexpr,
    stride_z_b: tl.constexpr,
    stride_z_h: tl.constexpr,
    stride_z_d: tl.constexpr,
    stride_a_b: tl.constexpr,
    stride_a_d: tl.constexpr,
    stride_b_b: tl.constexpr,
    stride_b_d: tl.constexpr,
    stride_s_b: tl.constexpr,
    stride_s_d: tl.constexpr,
    stride_s_w: tl.constexpr,
    stride_w_d: tl.constexpr,
    stride_w_w: tl.constexpr,
    q_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    gate_dim: tl.constexpr,
    conv_dim: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < conv_dim
    state_idx = tl.load(conv_state_indices + row)

    x = tl.load(mixed_qkv + row * stride_m_b + offs * stride_m_d, mask=mask, other=0.0).to(tl.float32)
    # KERNEL_SIZE is a constexpr, so Triton fully unrolls these loops for each conv size.
    y = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.static_range(0, KERNEL_SIZE - 1):
        s = tl.load(conv_state + state_idx * stride_s_b + offs * stride_s_d + i * stride_s_w, mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(conv_weight + offs * stride_w_d + i * stride_w_w, mask=mask, other=0.0).to(tl.float32)
        y += s * w

    w = tl.load(conv_weight + offs * stride_w_d + (KERNEL_SIZE - 1) * stride_w_w, mask=mask, other=0.0).to(tl.float32)
    y += x * w
    if HAS_BIAS:
        bias = tl.load(conv_bias + offs, mask=mask, other=0.0).to(tl.float32)
        y += bias
    if APPLY_SILU:
        y = y * tl.sigmoid(y)

    for i in tl.static_range(0, KERNEL_SIZE - 2):
        next_s = tl.load(
            conv_state + state_idx * stride_s_b + offs * stride_s_d + (i + 1) * stride_s_w, mask=mask, other=0.0
        )
        tl.store(conv_state + state_idx * stride_s_b + offs * stride_s_d + i * stride_s_w, next_s, mask=mask)
    tl.store(conv_state + state_idx * stride_s_b + offs * stride_s_d + (KERNEL_SIZE - 2) * stride_s_w, x, mask=mask)

    q_mask = offs < q_dim
    k_mask = (offs >= q_dim) & (offs < q_dim + k_dim)
    v_mask = (offs >= q_dim + k_dim) & (offs < conv_dim)
    tl.store(q_out + row * q_dim + offs, y, mask=q_mask)
    tl.store(k_out + row * k_dim + (offs - q_dim), y, mask=k_mask)
    tl.store(v_out + row * v_dim + (offs - q_dim - k_dim), y, mask=v_mask)

    z_mask = offs < v_dim
    z_vals = tl.load(z_raw + row * stride_z_b + offs, mask=z_mask, other=0.0)
    tl.store(z_out + row * v_dim + offs, z_vals, mask=z_mask)

    gate_mask = offs < gate_dim
    a_vals = tl.load(a_raw + row * stride_a_b + offs * stride_a_d, mask=gate_mask, other=0.0)
    b_vals = tl.load(b_raw + row * stride_b_b + offs * stride_b_d, mask=gate_mask, other=0.0)
    tl.store(a_out + row * gate_dim + offs, a_vals, mask=gate_mask)
    tl.store(b_out + row * gate_dim + offs, b_vals, mask=gate_mask)


@torch.no_grad()
def conv_pack_gdn_decode_inputs(
    mixed_qkv: torch.Tensor,
    z_raw: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state_indices: torch.Tensor,
    activation: str,
    conv_size: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
):
    batch = mixed_qkv.shape[0]
    q_dim = num_k_heads * head_k_dim
    k_dim = q_dim
    v_dim = num_v_heads * head_v_dim
    gate_dim = num_v_heads
    conv_dim = q_dim + k_dim + v_dim

    assert conv_size >= 2, f"conv kernel size must be at least 2, got {conv_size}"
    assert mixed_qkv.shape[1] == conv_dim, f"mixed_qkv shape mismatch: {mixed_qkv.shape[1]} != {conv_dim}"
    assert conv_weight.shape[0] == conv_dim, f"conv_weight shape mismatch: {conv_weight.shape[0]} != {conv_dim}"
    assert conv_weight.shape[1] == conv_size, f"conv_weight kernel mismatch: {conv_weight.shape[1]} != {conv_size}"
    assert conv_state.shape[1] == conv_dim, f"conv_state shape mismatch: {conv_state.shape[1]} != {conv_dim}"
    assert (
        conv_state.shape[2] >= conv_size - 1
    ), f"conv_state width must be at least conv_size - 1, got {conv_state.shape[2]} and {conv_size}"

    q = torch.empty((batch, 1, num_k_heads, head_k_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k = torch.empty_like(q)
    v = torch.empty((batch, 1, num_v_heads, head_v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    z = torch.empty((batch, num_v_heads, head_v_dim), dtype=z_raw.dtype, device=z_raw.device)
    a = torch.empty((batch, gate_dim), dtype=a_raw.dtype, device=a_raw.device)
    b = torch.empty((batch, gate_dim), dtype=b_raw.dtype, device=b_raw.device)

    block_size = 256
    grid = (batch, triton.cdiv(conv_dim, block_size))
    _conv_pack_gdn_decode_kernel[grid](
        mixed_qkv,
        z_raw,
        a_raw,
        b_raw,
        conv_state,
        conv_weight,
        conv_bias,
        conv_state_indices,
        q,
        k,
        v,
        z,
        a,
        b,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        z_raw.stride(0),
        z_raw.stride(1),
        z_raw.stride(2),
        a_raw.stride(0),
        a_raw.stride(1),
        b_raw.stride(0),
        b_raw.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        q_dim,
        k_dim,
        v_dim,
        gate_dim,
        conv_dim,
        conv_size,
        HAS_BIAS=conv_bias is not None,
        APPLY_SILU=activation in ["silu", "swish"],
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return q, k, v, z, a, b
