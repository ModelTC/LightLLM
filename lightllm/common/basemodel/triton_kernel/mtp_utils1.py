import triton
import triton.language as tl
from triton.language.standard import _log2, sum, zeros_like
import torch


@triton.jit
def _fwd_kernel_cumprod_probs(
    req_to_next_token_probs,
    req_to_next_token_probs_stride,
    b_req_idx,
    mtp_step,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index * (mtp_step + 1))
    base_ptr = req_to_next_token_probs + cur_req_idx * req_to_next_token_probs_stride
    tl.store(base_ptr, 1.0)

    offset = tl.arange(0, BLOCK_SIZE)
    store_mask = offset < (mtp_step + 1)

    probs = tl.load(base_ptr + offset, mask=store_mask, other=0.0)
    # 对于 probs 中大于 1.0 的值，设置为 0.99，避免错误的值，照成后续的采样操作失败。
    probs = tl.where(probs >= 1.0, 0.99, probs)
    probs = tl.where(probs <= 0.0, 0.01, probs)

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

    cond = (left > right) ^ flip

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
def _fwd_kernel_sample_dynamic_mtp_steps(
    req_to_next_token_probs,
    req_to_next_token_probs_stride,
    select_run_reqs,
    b_req_idx,
    mtp_step,
    req_num,
    dynamic_batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    all_num = req_num * (mtp_step + 1)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < all_num
    next_token_offset = offset % (mtp_step + 1)

    req_idx_index = tl.load(b_req_idx + offset % (all_num))

    probs = tl.load(
        req_to_next_token_probs + req_idx_index * req_to_next_token_probs_stride + next_token_offset,
        mask=mask,
        other=-1.0,
    )

    sorted_probs, sorted_ids = argsort(probs, offset, descending=True)

    tl.store(select_run_reqs + sorted_ids, 1, mask=mask & offset < dynamic_batch_size)
    return


def sample_dynamic_mtp_req_mask(
    dynamic_batch_size: int,
    b_req_idx: torch.Tensor,
    req_to_next_token_probs: torch.Tensor,
    mtp_step: int,
) -> torch.Tensor:
    assert b_req_idx.shape[0] % (mtp_step + 1) == 0
    assert req_to_next_token_probs.is_cuda
    assert dynamic_batch_size <= b_req_idx.shape[0]
    req_num = len(b_req_idx) // (mtp_step + 1)

    # cumprod probs for each request
    _fwd_kernel_cumprod_probs[(req_num,)](
        req_to_next_token_probs=req_to_next_token_probs,
        req_to_next_token_probs_stride=req_to_next_token_probs.stride(0),
        b_req_idx=b_req_idx,
        mtp_step=mtp_step,
        BLOCK_SIZE=triton.next_power_of_2(mtp_step + 1),
        num_warps=1,
        num_stages=1,
    )

    # 1 为选中， 0 为未选中
    select_run_reqs = torch.zeros((len(b_req_idx),), dtype=torch.int32, device="cuda")

    grid = (1,)
    _fwd_kernel_sample_dynamic_mtp_steps[grid](
        req_to_next_token_probs=req_to_next_token_probs,
        req_to_next_token_probs_stride=req_to_next_token_probs.stride(0),
        select_run_reqs=select_run_reqs,
        b_req_idx=b_req_idx,
        mtp_step=mtp_step,
        req_num=req_num,
        dynamic_batch_size=dynamic_batch_size,
        BLOCK_SIZE=triton.next_power_of_2(len(b_req_idx)),
        num_warps=1,
        num_stages=1,
    )
    return select_run_reqs
