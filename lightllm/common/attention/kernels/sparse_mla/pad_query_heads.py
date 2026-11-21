import triton
import triton.language as tl


@triton.jit
def _fused_strided_qpad_kernel(
    q,
    padded_q,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    num_heads: tl.constexpr,
    padded_num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offset < padded_num_heads * head_dim
    head = offset // head_dim
    dim = offset - head * head_dim
    from_q = in_range & (head < num_heads)
    value = tl.load(q + batch * q_stride_b + head * q_stride_h + dim * q_stride_d, mask=from_q, other=0.0)
    tl.store(padded_q + batch * padded_num_heads * head_dim + offset, value, mask=in_range)


def pad_query_heads(q, is_cuda_graph_capture=False):
    """Pad query heads for FlashMLA while preserving the query feature dimension."""
    num_tokens, num_heads, head_dim = q.shape
    head_tile = 64  # FlashMLA head-tile alignment.
    padded_num_heads = triton.cdiv(num_heads, head_tile) * head_tile
    if padded_num_heads != num_heads:
        if is_cuda_graph_capture:
            padded_q = q.new_empty((num_tokens, 1, padded_num_heads, head_dim))
            block_size = triton.next_power_of_2(2 * head_dim)
            _fused_strided_qpad_kernel[(num_tokens, triton.cdiv(padded_num_heads * head_dim, block_size))](
                q,
                padded_q,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                num_heads,
                padded_num_heads,
                head_dim=head_dim,
                BLOCK_SIZE=block_size,
                num_warps=4,
            )
        else:
            padded_q = q.new_zeros((num_tokens, 1, padded_num_heads, head_dim))
            padded_q[:, 0, :num_heads] = q
    else:
        # H64 BMM-transposed views already satisfy FlashMLA's last-dimension stride contract.
        padded_q = q.unsqueeze(1) if num_heads == head_tile and q.stride(-1) == 1 else q.unsqueeze(1).contiguous()
    return padded_q
