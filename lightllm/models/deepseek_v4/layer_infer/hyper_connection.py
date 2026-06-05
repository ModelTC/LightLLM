import torch


def _ensure_vllm_mhc_ops():
    try:
        import vllm.model_executor.layers.mhc  # noqa: F401
    except Exception as e:
        raise RuntimeError("DeepSeek-V4 requires vLLM mHC custom ops; failed to import vllm MHC kernels") from e


def hc_pre(streams, hc_fn, hc_scale, hc_base, hc_mult, dim, eps, sinkhorn_iters):
    """streams:[N, hc*dim] -> (collapsed[N,dim], post[N,hc,1], comb[N,hc,hc])."""
    _ensure_vllm_mhc_ops()
    post, comb, collapsed = torch.ops.vllm.mhc_pre(
        residual=streams.view(-1, hc_mult, dim).contiguous(),
        fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=eps,
        hc_pre_eps=eps,
        hc_sinkhorn_eps=eps,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=sinkhorn_iters,
    )
    return collapsed, post, comb


def hc_post(x, residual, post, comb, hc_mult, dim):
    """x:[N,dim] sub-layer output, residual:[N, hc*dim] -> [N, hc*dim]."""
    _ensure_vllm_mhc_ops()
    out = torch.ops.vllm.mhc_post(x, residual.view(-1, hc_mult, dim).contiguous(), post, comb)
    return out.reshape(-1, hc_mult * dim)


def hc_head(streams, hc_fn, hc_scale, hc_base, hc_mult, dim, eps):
    """Final stream collapse before the lm_head. streams:[N, hc*dim] -> [N, dim]."""
    _ensure_vllm_mhc_ops()
    out = torch.empty(streams.shape[0], dim, device=streams.device, dtype=streams.dtype)
    torch.ops.vllm.hc_head_fused_kernel(
        streams.view(-1, hc_mult, dim).contiguous(),
        hc_fn,
        hc_scale,
        hc_base,
        out,
        dim,
        eps,
        eps,
        hc_mult,
    )
    return out
