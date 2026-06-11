import torch

try:
    import vllm.model_executor.layers.mhc  # noqa: F401
except Exception as e:
    raise RuntimeError("DeepSeek-V4 requires vLLM mHC custom ops; failed to import vllm MHC kernels") from e


# vllm DeepseekV4DecoderLayer.hc_post_alpha
HC_POST_ALPHA = 2.0


def hc_pre(residual, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters, norm_weight, norm_eps):
    """Standalone hc_pre for the first layer. residual:[T, hc, dim] ->
    (x[T,dim], residual, post_mix[T,hc,1], res_mix[T,hc,hc]); the sub-layer RMSNorm is fused via norm_weight."""
    post_mix, res_mix, x = torch.ops.vllm.mhc_pre_tilelang(
        residual=residual,
        fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=HC_POST_ALPHA,
        sinkhorn_repeat=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return x, residual, post_mix, res_mix


def hc_fused_post_pre(
    x, residual, post_mix, res_mix, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters, norm_weight, norm_eps
):
    """hc_post of the previous sub-layer fused with hc_pre of the next one (norm fused too).
    Returns (x[T,dim], residual[T,hc,dim], post_mix, res_mix)."""
    residual, post_mix, res_mix, x = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x=x,
        residual=residual,
        post_layer_mix=post_mix,
        comb_res_mix=res_mix,
        fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=HC_POST_ALPHA,
        sinkhorn_repeat=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return x, residual, post_mix, res_mix


def hc_post(x, residual, post_mix, res_mix):
    """Complete the hc_post left pending by the last layer. -> streams [T, hc, dim]."""
    return torch.ops.vllm.mhc_post_tilelang(x, residual, post_mix, res_mix)


def hc_head(streams, hc_fn, hc_scale, hc_base, hc_mult, dim, rms_eps, hc_eps):
    """Final stream collapse before the lm_head. streams:[N, hc*dim] -> [N, dim]."""
    out = torch.empty(streams.shape[0], dim, device=streams.device, dtype=streams.dtype)
    torch.ops.vllm.hc_head_fused_kernel_tilelang(
        streams.view(-1, hc_mult, dim).contiguous(),
        hc_fn,
        hc_scale,
        hc_base,
        out,
        dim,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out
