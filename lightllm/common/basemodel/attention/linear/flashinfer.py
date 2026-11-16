import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.l2norm import l2norm_fwd

from .gdn import LinearAttBackend


def chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
    scale=None,
):
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as flashinfer_gdn

    if q.shape[0] != 1 or cu_seqlens is None:
        raise ValueError("FlashInfer GDN requires packed variable-length inputs")
    q, k, v = (tensor.squeeze(0).contiguous() for tensor in (q, k, v))
    if use_qk_l2norm_in_kernel:
        q, k = l2norm_fwd(q), l2norm_fwd(k)
    state = None
    if initial_state is not None:
        state = initial_state.transpose(-1, -2).to(torch.float32).contiguous()
    final = None
    if output_final_state:
        final = (
            state.clone()
            if state is not None
            else torch.zeros(
                (cu_seqlens.numel() - 1, max(q.shape[1], v.shape[1]), v.shape[2], k.shape[2]),
                device=q.device,
                dtype=torch.float32,
            )
        )
    result = flashinfer_gdn(
        q=q,
        k=k,
        v=v,
        g=g.squeeze(0).float().exp(),
        beta=beta.squeeze(0).float().contiguous(),
        scale=scale,
        initial_state=state,
        output_final_state=output_final_state,
        output_state=final,
        cu_seqlens=cu_seqlens.to(torch.int64),
        use_cp=False if q.shape[0] <= 1024 * (cu_seqlens.numel() - 1) else "auto",
        backend="flashinfer",
    )
    if output_final_state:
        output, state = result
        return output.unsqueeze(0), state.transpose(-1, -2).contiguous()
    return result.unsqueeze(0), None


class FlashInferLinearAttBackend(LinearAttBackend):
    def get_prefill_kernel(self):
        return chunk_gated_delta_rule
