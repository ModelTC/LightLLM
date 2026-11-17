# SPDX-License-Identifier: Apache-2.0

import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda import chunk_kda_with_fused_gate
from lightllm.common.basemodel.triton_kernel.linear_att.kda_state_io import gather_kda_state, scatter_kda_state
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


def get_flash_kda_fwd(model):
    backends = model.args.llm_prefill_att_backend
    backend = backends[1] if len(backends) > 1 else "auto"
    if backend == "triton":
        return None
    if backend not in ("auto", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}; use auto, triton, or flashkda")

    if backend == "auto" and (
        model.data_type != torch.bfloat16
        or model.args.linear_att_ssm_data_type != "bfloat16"
        or torch.version.hip is not None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        return None

    try:
        from flash_kda import fwd
    except (ImportError, OSError) as exc:
        if backend == "auto":
            logger.info("KDA prefill uses Triton because FlashKDA could not be loaded: %s", exc)
            return None
        raise ImportError(
            "The flashkda prefill backend requires FlashKDA. Install it with "
            "`pip install --no-build-isolation git+https://github.com/MoonshotAI/FlashKDA.git` "
            "and build it for the current GPU and PyTorch/CUDA versions."
        ) from exc
    logger.info("KDA prefill attention backend: %s.", "automatic FlashKDA/Triton" if backend == "auto" else "flashkda")
    return fwd


def select_kda_prefill(backend, max_q_seq_len):
    backends = backend.model.args.llm_prefill_att_backend
    requested = backends[1] if len(backends) > 1 else "auto"
    if requested == "auto":
        max_flash_len = 2048 if backend.tp_num_heads < 16 else 8192
        if max_q_seq_len > max_flash_len:
            return None
    return backend.flash_kda_fwd


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
        beta=raw_beta,
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
