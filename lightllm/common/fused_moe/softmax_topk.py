import torch
import triton
import triton.language as tl


@triton.jit
def softmax_topk_kernel(
    topk_weights_ptr,
    topk_indices_ptr,
    gating_output_ptr,
    input_row_stride,
    output_weights_row_stride,
    output_indices_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    top_k: tl.constexpr,
    NEED_MASK: tl.constexpr,
    RENORM: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_input_ptr = gating_output_ptr + row_idx * input_row_stride
    row_weights_ptr = topk_weights_ptr + row_idx * output_weights_row_stride
    row_indices_ptr = topk_indices_ptr + row_idx * output_indices_row_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_cols
        values = tl.load(row_input_ptr + offsets, mask=mask, other=-float("inf"))
    else:
        values = tl.load(row_input_ptr + offsets)

    current_max = tl.max(values, axis=0)
    values = values - current_max
    numerators = tl.exp(values)
    denom = tl.sum(numerators, axis=0)

    sum_prob = 0.0
    for i in range(top_k):
        logit = tl.max(values, axis=0)
        idx = tl.argmax(values, axis=0)

        prob = tl.exp(logit) / denom
        sum_prob += prob

        ptr_w = row_weights_ptr + i
        ptr_i = row_indices_ptr + i

        tl.store(ptr_w, prob)
        tl.store(ptr_i, idx)

        values = tl.where(offsets == idx, -float("inf"), values)

    if RENORM:
        sum_prob = tl.where(sum_prob < 1e-8, 1e-8, sum_prob)
        topk_offd = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offd < top_k
        prob = tl.load(row_weights_ptr + topk_offd, mask=topk_mask, other=0.0)
        prob = prob / sum_prob
        tl.store(row_weights_ptr + topk_offd, prob, mask=topk_mask)
    return


def softmax_topk(gating_output: torch.Tensor, topk: int, renorm: bool = False):
    assert gating_output.dim() == 2, "The dim of gating_output must be 2."
    num_tokens, num_experts = gating_output.shape
    device = gating_output.device

    if gating_output.dtype != torch.float32:
        gating_output = gating_output.to(torch.float32)

    topk_vals = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    topk_idxs = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

    BLOCK_SIZE = triton.next_power_of_2(num_experts)
    NEED_MASK = BLOCK_SIZE != num_experts

    num_warps = min(max(1, (BLOCK_SIZE // 8 // 32)), 16)

    grid = (num_tokens,)
    softmax_topk_kernel[grid](
        topk_vals,
        topk_idxs,
        gating_output,
        gating_output.stride(0),
        topk_vals.stride(0),
        topk_idxs.stride(0),
        num_tokens,
        num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_TOPK=triton.next_power_of_2(topk),
        top_k=topk,
        NEED_MASK=NEED_MASK,
        RENORM=renorm,
        num_warps=num_warps,
    )

    return topk_vals, topk_idxs
