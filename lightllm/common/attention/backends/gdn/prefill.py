"""Shared prefill flow for GDN chunk kernels."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.backend import AttentionBackend, AttentionBuildResources, TensorAllocator
from lightllm.common.basemodel.infer_struct import InferStateInfo
from . import GDNLayerConfig, GDNLayerInputs, _split_qkv


@dataclass(frozen=True)
class GDNPrefillAttState:
    """Packed prefill: request i's tokens start at cu_seqlens[i]; its conv state is slot conv_idx[i] and
    its SSM state slot ssm_idx[i]; has_initial_state[i] says whether it continues earlier tokens."""

    conv_idx: torch.Tensor
    ssm_idx: torch.Tensor
    cu_seqlens: torch.Tensor
    has_initial_state: torch.Tensor


class GDNPrefillBackend(AttentionBackend[GDNLayerConfig, GDNPrefillAttState, GDNLayerInputs]):
    def __init__(self, layer_config: GDNLayerConfig) -> None:
        super().__init__(layer_config)
        try:
            self.chunk_gated_delta_rule()
        except Exception as e:
            raise NotImplementedError(f"{type(self).__name__} cannot load its chunk kernel: {e!r}") from e

    @staticmethod
    @abstractmethod
    def chunk_gated_delta_rule():
        """Resolve the chunk kernel at model load so auto-selection can skip unavailable backends."""

    def forward(
        self,
        att_state: GDNPrefillAttState,
        layer_inputs: GDNLayerInputs,
        *,
        out: Optional[torch.Tensor],
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Return the chunk output, copying into out if supplied, and update the conv and SSM states in place."""
        from lightllm.common.attention.kernels.gdn.causal_conv1d import causal_conv1d_fn
        from lightllm.common.attention.kernels.gdn.fused_gdn_gating import fused_gdn_gating

        cfg, inputs = self.layer_config, layer_inputs
        g, beta = fused_gdn_gating(inputs.A_log, inputs.a, inputs.b, inputs.dt_bias)
        conv_states = inputs.conv_states
        if cfg.state_mtp_step > 0:
            conv_states = conv_states[:, :, : -cfg.state_mtp_step]
        mixed_qkv = causal_conv1d_fn(
            inputs.mixed_qkv.transpose(0, 1),
            inputs.conv_weight,
            bias=inputs.conv_bias,
            query_start_loc=att_state.cu_seqlens,
            cache_indices=att_state.conv_idx,
            has_initial_state=att_state.has_initial_state,
            conv_states=conv_states,
            activation=cfg.activation,
        ).transpose(0, 1)
        query, key, value = _split_qkv(cfg, mixed_qkv)
        tokens = query.shape[0]
        ssm_states = inputs.ssm_states
        core, last_state = self.chunk_gated_delta_rule()(
            q=query.view(1, tokens, cfg.num_k_heads, cfg.head_k_dim),
            k=key.view(1, tokens, cfg.num_k_heads, cfg.head_k_dim),
            v=value.view(1, tokens, cfg.num_v_heads, cfg.head_v_dim),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=ssm_states[att_state.ssm_idx],
            output_final_state=True,
            cu_seqlens=att_state.cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        # The chunk kernel accumulates in float32; index_put does not convert dtypes implicitly.
        ssm_states[att_state.ssm_idx] = last_state.to(ssm_states.dtype, copy=False)
        output = core.view(tokens, cfg.num_v_heads, cfg.head_v_dim)
        if out is None:
            return output
        out.copy_(output)
        return out

    @classmethod
    def build(cls, metadata: InferStateInfo, ssm_slots_per_request: int) -> GDNPrefillAttState:
        req = metadata.b_req_idx
        return GDNPrefillAttState(
            conv_idx=req,
            ssm_idx=req * ssm_slots_per_request if ssm_slots_per_request != 1 else req,
            cu_seqlens=metadata.b1_cu_q_seq_len,
            has_initial_state=metadata.b_ready_cache_len > 0,
        )

    @classmethod
    def create_build_context(cls, build_key: tuple[bool, int], ctx: AttentionBuildResources) -> int:
        replay, state_mtp_step = build_key
        return 1 if replay else state_mtp_step + 1

    @classmethod
    def build_context_key(cls, layer_config: GDNLayerConfig):
        return (layer_config.replay, layer_config.state_mtp_step)
