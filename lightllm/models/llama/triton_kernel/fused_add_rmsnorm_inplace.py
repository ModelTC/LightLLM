import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_fused_add_rmsnorm(
    original,
    residual,
    weight,
    original_stride0,
    original_stride1,
    residual_stride0,
    residual_stride1,
    N,  # number of columns in X
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    # data's base address of this block
    _original = original + block_id * original_stride0
    _residual = residual + block_id * residual_stride0

    # avoid repeat loading from gmem to smem
    # in some very large size, have better performance
    if N <= BLOCK_SIZE:
        # data's offset address of this block
        range = tl.arange(0, BLOCK_SIZE)
        _original_offset = range * original_stride1
        _residual_offset = range * residual_stride1
        _weight_offset = range

        # data's pointers of this block
        _original_ptr = _original + _original_offset
        _residual_ptr = _residual + _residual_offset
        _weight_ptr = weight + _weight_offset

        # load data from memory
        mask = range < N
        original_cache = tl.load(_original_ptr, mask=mask, other=0.0).to(tl.float32)
        residual_cache = tl.load(_residual_ptr, mask=mask, other=0.0).to(tl.float32)
        weight_cache = tl.load(_weight_ptr, mask=mask, other=0.0).to(tl.float32)

        # store (original + residual) to original
        original_cache = original_cache + residual_cache
        tl.store(_original_ptr, original_cache.to(original.dtype.element_ty), mask=mask)

        # compute variance
        var = tl.sum(original_cache * original_cache) / N
        rstd = 1 / tl.sqrt(var + eps)
        residual_cache = original_cache * rstd * weight_cache

        # store rmsnorm(original + residual) back to residual
        tl.store(_residual_ptr, residual_cache.to(residual.dtype.element_ty), mask=mask)
    else:
        sum_of_squares = tl.zeros([], dtype=tl.float32)
        for block_offset in range(0, N, BLOCK_SIZE):
            # data's offset address of this block
            range = tl.arange(0, BLOCK_SIZE) + block_offset
            _original_offset = range * original_stride1
            _residual_offset = range * residual_stride1

            # data's pointers of this block
            _original_ptr = _original + _original_offset
            _residual_ptr = _residual + _residual_offset

            # load data from memory
            mask = range < N
            original_cache = tl.load(_original_ptr, mask=mask, other=0.0).to(tl.float32)
            residual_cache = tl.load(_residual_ptr, mask=mask, other=0.0).to(tl.float32)

            # store (original + residual) to original
            original_cache = original_cache + residual_cache
            tl.store(_original_ptr, original_cache.to(original.dtype.element_ty), mask=mask)

            # compute sum_of_squares
            sum_of_squares += tl.sum(original_cache * original_cache)

        # compute variance
        var = sum_of_squares / N
        rstd = 1 / tl.sqrt(var + eps)

        for block_offset in range(0, N, BLOCK_SIZE):
            # data's offset address of this block
            range = tl.arange(0, BLOCK_SIZE) + block_offset
            _original_offset = range * original_stride1
            _residual_offset = range * residual_stride1
            _weight_offset = range

            # data's pointers of this block
            _original_ptr = _original + _original_offset
            _residual_ptr = _residual + _residual_offset
            _weight_ptr = weight + _weight_offset

            # load data from memory
            mask = range < N
            original_cache = tl.load(_original_ptr, mask=mask, other=0.0).to(tl.float32)
            weight_cache = tl.load(_weight_ptr, mask=mask, other=0.0).to(tl.float32)

            # apply rmsnorm using pre-computed rstd
            original_cache = original_cache * rstd * weight_cache

            # store rmsnorm(original) back to residual
            tl.store(_residual_ptr, original_cache.to(residual.dtype.element_ty), mask=mask)


def fused_add_rmsnorm_inplace(
    original: torch.Tensor,  # [num_tokens, hidden_size]
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    """
    Perform fused add & rmsnorm

    suppose the skip connection result is H(x) = F(x) + x,
    then F(x) is the residual, x is the original.
    Here original will be (residual + original), residual will be rmsnorm(residual + original)
    At first Layer, residual should be all zeros.
    """
    # reshape input data into 2D tensor
    original_arg = original.view(-1, original.shape[-1])
    residual_arg = residual.view(-1, residual.shape[-1])

    assert original.data_ptr() == original_arg.data_ptr()
    assert residual.data_ptr() == residual_arg.data_ptr()

    M, N = original_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // original.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 4)
    num_warps = triton.next_power_of_2(num_warps)
    if BLOCK_SIZE > 16384:
        BLOCK_SIZE = 16384

    # enqueue kernel
    _fwd_fused_add_rmsnorm[(M,)](
        original_arg,
        residual_arg,
        weight,
        original_arg.stride(0),
        original_arg.stride(1),
        residual_arg.stride(0),
        residual_arg.stride(1),
        N,  # number of columns in X
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
