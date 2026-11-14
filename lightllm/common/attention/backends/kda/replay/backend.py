from __future__ import annotations

from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature

import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import KDAInputs
from lightllm.common.attention.backends._shared.linear.layout import LinearDecodeContext, update_linear_decode
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend


@dataclasses.dataclass(slots=True)
class ReplayKDAPrefillContext(AttentionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class ReplayKDADecodeContext(LinearDecodeContext):
    pass


class TritonKDAReplayCommon:
    state_mode = "replay"
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    provider = "triton"

    family = AttentionFamily.KDA

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    def __init__(self, env, spec):
        super().__init__(env=env, spec=spec)
        self.head_dim = spec.head_size
        self.tp_num_heads = spec.num_q_heads
        self.tp_hidden_size = self.tp_num_heads * self.head_dim
        self.conv_kernel_size = spec.config.conv_kernel_size
        self.lower_bound = spec.config.gate_lower_bound
        self.state_slots = env.state_slots

    def split_qkv(self, mixed_qkv: torch.Tensor):
        return mixed_qkv.split(self.tp_hidden_size, dim=-1)


class TritonKDAReplayPrefillBackend(
    TritonKDAReplayCommon, AttentionBackend[ReplayKDAPrefillContext, KDAInputs, torch.Tensor]
):
    phase = AttentionPhase.PREFILL

    def allocate_context(self, capacity, *, capture, workspace) -> ReplayKDAPrefillContext:
        return ReplayKDAPrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: ReplayKDAPrefillContext):
        reqs = context.batch.b_req_idx
        context.b_conv_buffer_idx = reqs
        context.b_ssm_buffer_idx = reqs

    def forward(
        self, layer: LayerAttention, context: ReplayKDAPrefillContext, inputs: KDAInputs, *, out=None
    ) -> torch.Tensor:
        from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn
        from lightllm.common.attention.backends.kda.triton.kernels.kda import chunk_kda_with_fused_gate

        conv_states, ssm_states = layer.resources.conv, layer.resources.ssm
        conv_states = conv_states[..., : self.conv_kernel_size - 1]
        mixed_qkv = causal_conv1d_fn(
            inputs.mixed_qkv.transpose(0, 1),
            layer.resources.conv_weight,
            bias=None,
            query_start_loc=context.batch.b1_cu_q_seq_len,
            cache_indices=context.b_conv_buffer_idx,
            has_initial_state=context.batch.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation="silu",
        ).transpose(0, 1)
        q, k, v = self.split_qkv(mixed_qkv)
        q = q.view(1, -1, self.tp_num_heads, self.head_dim)
        k = k.view(1, -1, self.tp_num_heads, self.head_dim)
        v = v.view(1, -1, self.tp_num_heads, self.head_dim)
        raw_gate = inputs.raw_gate.view(1, -1, self.tp_hidden_size)
        raw_beta = inputs.raw_beta.view(1, -1, self.tp_num_heads)
        output, final_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, -1, self.tp_num_heads, self.head_dim),
            beta=raw_beta.float().sigmoid(),
            A_log=layer.resources.a_log,
            g_bias=layer.resources.dt_bias,
            initial_state=ssm_states[context.b_ssm_buffer_idx].contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=context.batch.b1_cu_q_seq_len,
            safe_gate=True,
            lower_bound=self.lower_bound,
        )
        ssm_states[context.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype, copy=False)
        return output


class TritonKDAReplayDecodeBackend(
    TritonKDAReplayCommon, AttentionBackend[ReplayKDADecodeContext, KDAInputs, torch.Tensor]
):
    phase = AttentionPhase.DECODE

    def allocate_context(self, capacity, *, capture, workspace) -> ReplayKDADecodeContext:
        return ReplayKDADecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: ReplayKDADecodeContext):
        update_linear_decode(context, self.draft_step, self.spec.dynamic_verify, replay=True)

    def forward(
        self, layer: LayerAttention, context: ReplayKDADecodeContext, inputs: KDAInputs, *, out=None
    ) -> torch.Tensor:
        from .decode import replay
        from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_update
        from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
            causal_conv1d_update as causal_conv1d_update_mtp,
        )

        state = layer.resources.replay_state
        conv_states = layer.resources.conv
        conv_kwargs = dict(bias=None, activation="silu", conv_state_indices=context.b_conv_buffer_idx)
        conv_update = causal_conv1d_update
        if self.draft_step > 0:
            conv_update = causal_conv1d_update_mtp
            conv_kwargs.update(
                mtp_step=self.draft_step,
                num_accepted_tokens=context.accepted_tokens(layer.resources.mtp_state_index),
                query_start_loc=context.b1_mtp_cu_q_seq_len,
            )
        mixed_qkv = conv_update(
            inputs.mixed_qkv,
            conv_states,
            layer.resources.conv_weight,
            **conv_kwargs,
        )
        q, k, v = self.split_qkv(mixed_qkv)
        q = q.view(-1, self.tp_num_heads, self.head_dim)
        k = k.view(-1, self.tp_num_heads, self.head_dim)
        v = v.view(-1, self.tp_num_heads, self.head_dim)
        raw_gate = inputs.raw_gate.reshape(-1, self.tp_hidden_size)
        raw_beta = inputs.raw_beta.reshape(-1, self.tp_num_heads)
        storage_layer = state.storage_layer(layer.logical_layer_id)
        output = replay(
            state_ssm=state.ssm_state[storage_layer],
            state_keys=state.keys[storage_layer],
            state_deltas=state.deltas[storage_layer],
            state_gates=state.gates[storage_layer],
            state_raw_keys=state.raw_keys[storage_layer],
            state_raw_values=state.raw_values[storage_layer],
            state_betas=state.betas[storage_layer],
            state_cursors=state.cursors,
            state_tile_sync_mark=state.tile_sync_mark,
            q=q,
            k=k,
            v=v,
            a=raw_gate,
            b=raw_beta,
            a_log=layer.resources.a_log,
            bias=layer.resources.dt_bias,
            reqs=context.b_conv_buffer_idx,
            cu_seqlens=context.b_cu_seqlens,
            state_verify_width=state.verify_width,
            state_compute_mode=state.compute_mode,
            state_gate_lower_bound=state.gate_lower_bound,
        )
        return output
