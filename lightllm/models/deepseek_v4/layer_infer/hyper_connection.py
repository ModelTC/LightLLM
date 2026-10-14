"""Adapt DeepSeek-V4's deferred, 3D mHC state to the shared flat-stream kernels."""

from lightllm.common.basemodel.triton_kernel.mhc import (
    hc_head as _hc_head,
    hc_post as _hc_post,
    hc_post_pre_norm,
    hc_pre_norm,
)

HC_POST_ALPHA = 2.0


def hc_pre(residual, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters, norm_weight, norm_eps):
    """Standalone hc_pre for the first layer. residual:[T, hc, dim] ->
    (x[T,dim], residual, post_mix[T,hc,1], res_mix[T,hc,hc]); the sub-layer RMSNorm is fused via norm_weight."""
    streams = residual.shape[1]
    x, res_mix, post_mix = hc_pre_norm(
        x=residual.flatten(1),
        fn=hc_fn,
        scale=hc_scale,
        base=hc_base,
        streams=streams,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        post_multiplier=HC_POST_ALPHA,
        sinkhorn_iters=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return x, residual, post_mix.unsqueeze(-1), res_mix


def hc_fused_post_pre(
    x, residual, post_mix, res_mix, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters, norm_weight, norm_eps
):
    """hc_post of the previous sub-layer fused with hc_pre of the next one (norm fused too).
    Returns (x[T,dim], residual[T,hc,dim], post_mix, res_mix)."""
    residual_shape = residual.shape
    x, residual, res_mix, post_mix = hc_post_pre_norm(
        layer_output=x,
        residual=residual.flatten(1),
        post_mix=post_mix.view(residual_shape[0], residual_shape[1]),
        residual_mix=res_mix,
        fn=hc_fn,
        scale=hc_scale,
        base=hc_base,
        streams=residual_shape[1],
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        post_multiplier=HC_POST_ALPHA,
        sinkhorn_iters=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return x, residual.view(residual_shape), post_mix.unsqueeze(-1), res_mix


def hc_post(x, residual, post_mix, res_mix):
    """Complete the hc_post left pending by the last layer. -> streams [T, hc, dim]."""
    return _hc_post(
        x, residual.flatten(1), res_mix, post_mix.view(residual.shape[0], residual.shape[1]), residual.shape[1]
    ).view_as(residual)


def hc_head(streams, hc_fn, hc_scale, hc_base, hc_mult, dim, rms_eps, hc_eps, alloc_func):
    """Final stream collapse before the lm_head. streams:[N, hc*dim] -> [N, dim]."""
    out = alloc_func((streams.shape[0], dim), dtype=streams.dtype, device=streams.device)
    return _hc_head(
        streams.view(-1, hc_mult * dim),
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        rms_eps,
        hc_eps,
        out=out,
    )
