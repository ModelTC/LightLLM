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
    BLOCK1024: tl.constexpr,
):
    batch = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK1024 + tl.arange(0, BLOCK1024)
    in_range = offset < padded_num_heads * 512
    head = offset // 512
    dim = offset - head * 512
    from_q = in_range & (head < num_heads)
    value = tl.load(q + batch * q_stride_b + head * q_stride_h + dim * q_stride_d, mask=from_q, other=0.0)
    tl.store(padded_q + batch * padded_num_heads * 512 + offset, value, mask=in_range)


def pad_query_heads(q, is_cuda_graph_capture=False):
    """Prepare GLM's 512-dimensional NoPE query for FlashMLA's head tiles."""
    num_tokens, num_heads, _ = q.shape
    padded_num_heads = ((num_heads + 63) // 64) * 64
    if padded_num_heads != num_heads:
        if is_cuda_graph_capture:
            padded_q = q.new_empty((num_tokens, 1, padded_num_heads, 512))
            _fused_strided_qpad_kernel[(num_tokens, triton.cdiv(padded_num_heads * 512, 1024))](
                q,
                padded_q,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                num_heads,
                padded_num_heads,
                BLOCK1024=1024,
                num_warps=4,
            )
        else:
            padded_q = q.new_zeros((num_tokens, 1, padded_num_heads, 512))
            padded_q[:, 0, :num_heads] = q
    else:
        # H64 BMM-transposed views already satisfy FlashMLA's last-dimension stride contract.
        padded_q = q.unsqueeze(1) if num_heads == 64 and q.stride(-1) == 1 else q.unsqueeze(1).contiguous()
    return padded_q
