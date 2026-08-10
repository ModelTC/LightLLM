import triton
import triton.language as tl
from triton.language.standard import _log2, sum, zeros_like
import torch


@triton.jit
def _fwd_kernel_cumprod_probs(
    req_to_next_token_probs,
    req_to_next_token_probs_stride,
    b_req_idx,
    max_draft_step,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index * (max_draft_step + 1))
    base_ptr = req_to_next_token_probs + cur_req_idx * req_to_next_token_probs_stride
    tl.store(base_ptr, 1.0)

    offset = tl.arange(0, BLOCK_SIZE)
    store_mask = offset < (max_draft_step + 1)

    probs = tl.load(base_ptr + offset, mask=store_mask, other=0.0)
    # offset 0 是 target sample，本轮恒接受；只有 draft 条件接受概率需要 clamp。
    probs = tl.where(offset == 0, 1.0, probs)
    # 对于 draft probs 中大于 0.99 的值，设置为 0.99，避免错误的值，照成后续的采样操作失败。
    # 对于 draft probs 中小于 0.01 的值，设置为 0.01，避免错误的值，照成后续的采样操作失败。
    # This makes each request's cumulative acceptance probabilities monotonic,
    # so global top-k selection cannot pick a later draft row without its prefix.
    probs = tl.where((offset != 0) & (probs >= 0.99), 0.99, probs)
    probs = tl.where((offset != 0) & (probs <= 0.01), 0.01, probs)

    cum_probs = tl.cumprod(probs, axis=0)

    tl.store(base_ptr + offset, cum_probs, mask=store_mask)
    return


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.core.constexpr, n_dims: tl.core.constexpr):
    n_outer: tl.core.constexpr = x.numel >> n_dims
    shape: tl.core.constexpr = [n_outer * 2 ** i, 2, 2 ** (n_dims - i - 1)]
    y = tl.core.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.core.arange(0, 2)[None, :, None]
    left = tl.core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = tl.core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = tl.core.reshape(left, x.shape)
    right = tl.core.reshape(right, x.shape)

    y_idx = tl.core.reshape(ids, shape)
    left_idx = tl.core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.core.reshape(left_idx, x.shape)
    right_idx = tl.core.reshape(right_idx, x.shape)

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != (flip != 0)

    ret = ix ^ tl.core.where(cond, ileft ^ iright, zeros_like(ix))
    new_ids = ids ^ tl.core.where(cond, left_idx ^ right_idx, zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: tl.core.constexpr, order: tl.core.constexpr, n_dims: tl.core.constexpr):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: tl.core.constexpr = x.numel >> n_dims
    tl.core.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2 ** stage]
        flip = tl.core.reshape(tl.core.broadcast_to(tl.core.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.core.constexpr = None, descending: tl.core.constexpr = tl.core.CONSTEXPR_0):
    _dim: tl.core.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    n_dims: tl.core.constexpr = _log2(x.shape[_dim])

    for i in tl.core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@triton.jit
def _fwd_kernel_select_dynamic_spec_rows(
    req_to_next_token_probs,
    req_to_next_token_probs_stride,
    selected_row_mask,
    b_req_idx,
    max_draft_step,
    pre_draft_step,
    req_num,
    dynamic_batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    all_num = req_num * (pre_draft_step + 1)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < all_num
    req_offset = offset // (pre_draft_step + 1)
    next_token_offset = offset % (pre_draft_step + 1)
    original_offset = req_offset * (max_draft_step + 1) + next_token_offset

    req_idx_index = tl.load(b_req_idx + original_offset, mask=mask, other=0)

    probs = tl.load(
        req_to_next_token_probs + req_idx_index * req_to_next_token_probs_stride + next_token_offset,
        mask=mask,
        other=-1.0,
    )

    sorted_probs, sorted_ids = argsort(probs, original_offset, descending=True)

    tl.store(selected_row_mask + sorted_ids, 1, mask=mask & (offset < dynamic_batch_size))
    return


def sample_dynamic_spec_row_mask(
    dynamic_batch_size: int,
    b_req_idx: torch.Tensor,
    req_to_next_token_probs: torch.Tensor,
    max_draft_step: int,
    pre_draft_step: int = None,
) -> torch.Tensor:
    dynamic_batch_size = int(dynamic_batch_size)
    max_draft_step = int(max_draft_step)
    pre_draft_step = max_draft_step if pre_draft_step is None else int(pre_draft_step)
    assert 0 <= pre_draft_step <= max_draft_step
    assert b_req_idx.shape[0] % (max_draft_step + 1) == 0
    assert req_to_next_token_probs.is_cuda
    assert dynamic_batch_size <= b_req_idx.shape[0]
    req_num = len(b_req_idx) // (max_draft_step + 1)
    valid_row_num = req_num * (pre_draft_step + 1)
    assert dynamic_batch_size <= valid_row_num

    # cumprod probs for each request
    _fwd_kernel_cumprod_probs[(req_num,)](
        req_to_next_token_probs=req_to_next_token_probs,
        req_to_next_token_probs_stride=req_to_next_token_probs.stride(0),
        b_req_idx=b_req_idx,
        max_draft_step=max_draft_step,
        BLOCK_SIZE=triton.next_power_of_2(max_draft_step + 1),
        num_warps=1,
        num_stages=1,
    )

    # 1 为选中， 0 为未选中
    selected_row_mask = torch.zeros((len(b_req_idx),), dtype=torch.int32, device="cuda")

    grid = (1,)
    _fwd_kernel_select_dynamic_spec_rows[grid](
        req_to_next_token_probs=req_to_next_token_probs,
        req_to_next_token_probs_stride=req_to_next_token_probs.stride(0),
        selected_row_mask=selected_row_mask,
        b_req_idx=b_req_idx,
        max_draft_step=max_draft_step,
        pre_draft_step=pre_draft_step,
        req_num=req_num,
        dynamic_batch_size=dynamic_batch_size,
        BLOCK_SIZE=triton.next_power_of_2(valid_row_num),
        num_warps=1,
        num_stages=1,
    )
    return selected_row_mask


@triton.jit
def _fwd_kernel_trim_post_sample_tensors(
    b_req_idx,
    out_b_req_idx,
    b_temperatures,
    out_b_temperatures,
    b_top_ps,
    out_b_top_ps,
    b_top_ks,
    out_b_top_ks,
    b_length_penalty_param,
    out_b_length_penalty_param,
    b_mask_eos_reqs,
    out_b_mask_eos_reqs,
    selected_row_mask,
    selected_dst_pos,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    selected = tl.load(selected_row_mask + offsets, mask=mask, other=0) != 0
    dst_pos = tl.load(selected_dst_pos + offsets, mask=mask, other=0)
    write_mask = mask & selected

    tl.store(
        out_b_req_idx + dst_pos,
        tl.load(b_req_idx + offsets, mask=mask, other=0),
        mask=write_mask,
    )
    tl.store(
        out_b_temperatures + dst_pos,
        tl.load(b_temperatures + offsets, mask=mask, other=0.0),
        mask=write_mask,
    )
    tl.store(
        out_b_top_ps + dst_pos,
        tl.load(b_top_ps + offsets, mask=mask, other=0.0),
        mask=write_mask,
    )
    tl.store(
        out_b_top_ks + dst_pos,
        tl.load(b_top_ks + offsets, mask=mask, other=0),
        mask=write_mask,
    )
    tl.store(
        out_b_length_penalty_param + dst_pos,
        tl.load(b_length_penalty_param + offsets, mask=mask, other=0),
        mask=write_mask,
    )
    tl.store(
        out_b_mask_eos_reqs + dst_pos,
        tl.load(b_mask_eos_reqs + offsets, mask=mask, other=0),
        mask=write_mask,
    )


def trim_post_sample_tensors(
    dynamic_batch_size: int,
    selected_row_mask: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_temperatures: torch.Tensor,
    b_top_ps: torch.Tensor,
    b_top_ks: torch.Tensor,
    b_length_penalty_param: torch.Tensor,
    b_mask_eos_reqs: torch.Tensor,
):
    assert selected_row_mask.is_cuda
    dynamic_batch_size = int(dynamic_batch_size)
    selected_row_mask = selected_row_mask.to(torch.int32)
    selected_dst_pos = torch.cumsum(selected_row_mask, dim=0, dtype=torch.int32) - 1
    batch_size = selected_row_mask.shape[0]

    output_tensors = tuple(
        torch.empty((dynamic_batch_size,), dtype=tensor.dtype, device=tensor.device)
        for tensor in (
            b_req_idx,
            b_temperatures,
            b_top_ps,
            b_top_ks,
            b_length_penalty_param,
            b_mask_eos_reqs,
        )
    )
    (
        out_b_req_idx,
        out_b_temperatures,
        out_b_top_ps,
        out_b_top_ks,
        out_b_length_penalty_param,
        out_b_mask_eos_reqs,
    ) = output_tensors

    block_size = 256
    _fwd_kernel_trim_post_sample_tensors[(triton.cdiv(batch_size, block_size),)](
        b_req_idx=b_req_idx,
        out_b_req_idx=out_b_req_idx,
        b_temperatures=b_temperatures,
        out_b_temperatures=out_b_temperatures,
        b_top_ps=b_top_ps,
        out_b_top_ps=out_b_top_ps,
        b_top_ks=b_top_ks,
        out_b_top_ks=out_b_top_ks,
        b_length_penalty_param=b_length_penalty_param,
        out_b_length_penalty_param=out_b_length_penalty_param,
        b_mask_eos_reqs=b_mask_eos_reqs,
        out_b_mask_eos_reqs=out_b_mask_eos_reqs,
        selected_row_mask=selected_row_mask,
        selected_dst_pos=selected_dst_pos,
        batch_size=batch_size,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return output_tensors
