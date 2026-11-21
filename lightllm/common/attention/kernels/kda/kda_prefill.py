# SPDX-License-Identifier: Apache-2.0
"""KDA chunked prefill with the Triton kernel or FlashKDA, updating the SSM states of the step's requests."""

import torch

from lightllm.common.attention.kernels.gdn.third_party_fla.ops.kda import chunk_kda_with_fused_gate
from lightllm.common.attention.kernels.kda.kda_state_io import gather_kda_state, scatter_kda_state


def kda_prefill(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    A_log,
    g_bias,
    ssm_states,
    state_indices,
    cu_seqlens,
    lower_bound,
    qk_normalized,
    flash_kda_fwd=None,
):
    if q.shape[1] == 0:
        return torch.empty_like(v, memory_format=torch.contiguous_format)
    if flash_kda_fwd is None:
        output, final_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            beta=raw_beta.float().sigmoid(),
            A_log=A_log,
            g_bias=g_bias,
            initial_state=ssm_states[state_indices],
            output_final_state=True,
            use_qk_l2norm_in_kernel=not qk_normalized,
            cu_seqlens=cu_seqlens,
            safe_gate=True,
            lower_bound=lower_bound,
        )
        ssm_states[state_indices] = final_state.to(ssm_states.dtype, copy=False)
        return output

    assert not qk_normalized, "FlashKDA requires raw Q/K because it applies L2 normalization internally"
    output = torch.empty_like(v, memory_format=torch.contiguous_format)
    assert cu_seqlens.dtype == torch.int64
    state = gather_kda_state(ssm_states, state_indices)
    final_state = torch.empty_like(state)
    flash_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        # FlashKDA needs contiguous inputs; layers pass beta as a column slice of their fused projection.
        beta=raw_beta.contiguous(),
        scale=q.shape[-1] ** -0.5,
        out=output,
        A_log=A_log.reshape(-1).float(),
        dt_bias=g_bias.reshape(q.shape[2], q.shape[3]).float(),
        lower_bound=lower_bound,
        initial_state=state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
    scatter_kda_state(final_state, ssm_states, state_indices, cu_seqlens)
    return output
