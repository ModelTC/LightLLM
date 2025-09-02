# adopt from https://github.com/triton-lang/triton/issues/3698#issuecomment-2067681396
import torch
import triton
import triton.language as tl
from triton.language.standard import _log2, sum, zeros_like


@triton.jit
def _compare_and_swap(x, x_1, ids, flip, i: tl.core.constexpr, n_dims: tl.core.constexpr):
    n_outer: tl.core.constexpr = x.numel >> n_dims
    shape: tl.core.constexpr = [n_outer * 2 ** i, 2, 2 ** (n_dims - i - 1)]
    y = tl.core.reshape(x, shape)
    y_1 = tl.core.reshape(x_1, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.core.arange(0, 2)[None, :, None]
    left = tl.core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = tl.core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = tl.core.reshape(left, x.shape)
    right = tl.core.reshape(right, x.shape)

    left_1 = tl.core.broadcast_to(sum(y_1 * (1 - mask), 1)[:, None, :], shape)
    right_1 = tl.core.broadcast_to(sum(y_1 * mask, 1)[:, None, :], shape)
    left_1 = tl.core.reshape(left_1, x_1.shape)
    right_1 = tl.core.reshape(right_1, x_1.shape)
    # idx
    y_idx = tl.core.reshape(ids, shape)
    left_idx = tl.core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.core.reshape(left_idx, x.shape)
    right_idx = tl.core.reshape(right_idx, x.shape)

    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip

    ret = ix ^ tl.core.where(cond, ileft ^ iright, zeros_like(ix))

    new_ids = ids ^ tl.core.where(cond, left_idx ^ right_idx, zeros_like(ids))

    # swap x_1
    idtype_1 = tl.core.get_int_dtype(bitwidth=x_1.dtype.primitive_bitwidth, signed=True)
    ileft_1 = left_1.to(idtype_1, bitcast=True)
    iright_1 = right_1.to(idtype_1, bitcast=True)
    ix_1 = x_1.to(idtype, bitcast=True)
    ret_1 = ix_1 ^ tl.core.where(cond, ileft_1 ^ iright_1, zeros_like(ix_1))

    return ret.to(x.dtype, bitcast=True), ret_1.to(x_1.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, x_1, ids, stage: tl.core.constexpr, order: tl.core.constexpr, n_dims: tl.core.constexpr):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: tl.core.constexpr = x.numel >> n_dims
    tl.core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2 ** stage]
        flip = tl.core.reshape(tl.core.broadcast_to(tl.core.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.core.static_range(stage):
        x, x_1, ids = _compare_and_swap(x, x_1, ids, flip, i + (n_dims - stage), n_dims)
    return x, x_1, ids


@triton.jit
def argsort(x, x_1, ids, dim: tl.core.constexpr = None, descending: tl.core.constexpr = tl.core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: tl.core.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: tl.core.constexpr = _log2(x.shape[_dim])

    for i in tl.core.static_range(1, n_dims + 1):
        x, x_1, ids = _bitonic_merge(x, x_1, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, x_1, ids


@triton.jit
def grouped_topk_kernel(
    gating_output_ptr,
    gating_output_stride_m,
    gating_output_stride_n,
    correction_bias_ptr,
    scores_buffer_ptr,  # [token_num, total_expert_num]
    scores_stride_m,
    scores_stride_n,
    scores_stride_token_m,
    scores_stride_group,
    scores_stride_group_v,
    out_topk_weights,
    out_topk_weights_stride_m,
    out_topk_weights_stride_n,
    out_topk_ids,
    out_topk_ids_stride_m,
    out_topk_ids_stride_n,
    group_num,
    group_expert_num,
    total_expert_num,  # group_num * group_expert_num == total_expert_num
    topk_num,
    group_topk_num,
    IS_SIGMOID: tl.constexpr,
    HAS_CORRECTION_BIAS: tl.constexpr,
    EXPERT_BLOCK_SIZE: tl.constexpr,  # tl.next_power_two_of(total_expert_num)
    EXPERT_GROUP_NUM: tl.constexpr,  # tl.next_power_two_of(group_num)
    EXPERT_GROUP_SIZE: tl.constexpr,  # tl.next_power_two_of(group_expert_num)
    RENORMALIZE: tl.constexpr,
    GROUP_SCORE_USED_TOPK_NUM: tl.constexpr,
):
    token_index = tl.program_id(axis=0)
    offs_n = tl.arange(0, EXPERT_BLOCK_SIZE)
    hidden_states = tl.load(
        gating_output_ptr + token_index * gating_output_stride_m + offs_n,
        mask=offs_n < total_expert_num,
        other=-10000000.0,
    ).to(tl.float32)
    if IS_SIGMOID:
        old_scores = tl.sigmoid(hidden_states)
    else:
        old_scores = tl.softmax(hidden_states)

    if HAS_CORRECTION_BIAS:
        scores = old_scores + tl.load(correction_bias_ptr + offs_n, mask=offs_n < total_expert_num, other=-10000000.0)
    else:
        scores = old_scores

    offs_group = tl.arange(0, EXPERT_GROUP_NUM)
    offs_group_v = tl.arange(0, EXPERT_GROUP_SIZE)
    tl.store(scores_buffer_ptr + scores_stride_m * token_index + offs_n, scores, mask=offs_n < total_expert_num)
    tl.debug_barrier()
    group_scores = tl.load(
        scores_buffer_ptr
        + scores_stride_token_m * token_index
        + offs_group[:, None] * scores_stride_group
        + offs_group_v[None, :] * scores_stride_group_v,
        mask=(offs_group < group_num)[:, None] & (offs_group_v < group_expert_num)[None, :],
        other=-10000000.0,
    )  # [group, group_size]

    group_value = tl.sum(
        tl.where(
            (offs_group < group_num)[:, None] & (offs_group_v < GROUP_SCORE_USED_TOPK_NUM)[None, :],
            tl.sort(group_scores, dim=1, descending=True),
            0.0,
        ),
        axis=1,
    )

    sorted_group_value = tl.sort(group_value, descending=True)
    group_topk_value = tl.sum(tl.where(offs_group == group_topk_num - 1, sorted_group_value, 0.0))
    mask_group_scores = tl.where(
        ((group_value >= group_topk_value)[:, None]) & ((offs_group_v < group_expert_num)[None, :]),
        group_scores,
        -10000000.0,
    )

    tl.store(
        scores_buffer_ptr
        + scores_stride_token_m * token_index
        + offs_group[:, None] * scores_stride_group
        + offs_group_v[None, :] * scores_stride_group_v,
        mask_group_scores,
        mask=((offs_group < group_num)[:, None]) & ((offs_group_v < group_expert_num)[None, :]),
    )  # [group, group_size]
    tl.debug_barrier()
    mask_scores = tl.load(
        scores_buffer_ptr + scores_stride_m * token_index + offs_n, mask=offs_n < total_expert_num, other=-10000000.0
    )
    _, sorted_scores, sorted_indexes = argsort(mask_scores, old_scores, offs_n, descending=True)

    if RENORMALIZE:
        sum_scores = tl.sum(tl.where(offs_n < topk_num, sorted_scores, 0.0))
        renormlize_scores = sorted_scores / sum_scores

        tl.store(
            out_topk_weights + token_index * out_topk_weights_stride_m + offs_n,
            renormlize_scores,
            mask=offs_n < topk_num,
        )
        tl.store(out_topk_ids + token_index * out_topk_ids_stride_m + offs_n, sorted_indexes, mask=offs_n < topk_num)
    else:
        tl.store(
            out_topk_weights + token_index * out_topk_weights_stride_m + offs_n, sorted_scores, mask=offs_n < topk_num
        )
        tl.store(out_topk_ids + token_index * out_topk_ids_stride_m + offs_n, sorted_indexes, mask=offs_n < topk_num)
    return


def triton_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    group_score_used_topk_num=2,
):

    if correction_bias is not None:
        has_correction_bias = True
    else:
        has_correction_bias = False

    token_num, total_expert_num = gating_output.shape
    if gating_output.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    scores_buffer = torch.empty((token_num, total_expert_num), dtype=dtype, device="cuda")
    out_topk_weights = torch.empty((token_num, topk), dtype=torch.float32, device="cuda")
    out_topk_ids = torch.empty((token_num, topk), dtype=torch.long, device="cuda")

    assert total_expert_num % num_expert_group == 0

    if token_num <= 256:
        num_warps = 4
    else:
        num_warps = 1

    grouped_topk_kernel[(token_num,)](
        gating_output,
        *gating_output.stride(),
        correction_bias,
        scores_buffer,
        *scores_buffer.stride(),
        *scores_buffer.view(token_num, num_expert_group, -1).stride(),
        out_topk_weights,
        *out_topk_weights.stride(),
        out_topk_ids,
        *out_topk_ids.stride(),
        group_num=num_expert_group,
        group_expert_num=total_expert_num // num_expert_group,
        total_expert_num=total_expert_num,
        topk_num=topk,
        group_topk_num=topk_group,
        IS_SIGMOID=scoring_func == "sigmoid",
        HAS_CORRECTION_BIAS=has_correction_bias,
        EXPERT_BLOCK_SIZE=triton.next_power_of_2(total_expert_num),
        EXPERT_GROUP_NUM=triton.next_power_of_2(num_expert_group),
        EXPERT_GROUP_SIZE=triton.next_power_of_2(total_expert_num // num_expert_group),
        RENORMALIZE=renormalize,
        GROUP_SCORE_USED_TOPK_NUM=group_score_used_topk_num,
        num_warps=num_warps,
        num_stages=1,
    )
    return out_topk_weights, out_topk_ids
