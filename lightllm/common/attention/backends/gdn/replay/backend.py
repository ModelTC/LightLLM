from __future__ import annotations

import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import GDNInputs, GDNOutput
from lightllm.common.attention.api.spec import AttentionFamily, AttentionPhase
from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn
from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
    causal_conv1d_update as causal_conv1d_update_mtp,
)
from lightllm.common.attention.backends._shared.linear.gdn import GDNMetadata
from lightllm.common.attention.backends._shared.linear.layout import LinearDecodeContext, update_linear_decode
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend
from lightllm.common.attention.backends.gdn.triton.kernels.fused_gdn_gating import fused_gdn_gating
from lightllm.common.attention.backends.gdn.triton.kernels.gdn_decode_pack import conv_pack_gdn_decode_inputs

from .decode import replay


@dataclasses.dataclass(slots=True)
class ReplayGDNPrefillContext(AttentionExecutionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class ReplayGDNDecodeContext(LinearDecodeContext):
    pass


class ReplayGDNTritonAttBackend(
    GDNMetadata, BaseAttentionBackend[ReplayGDNPrefillContext | ReplayGDNDecodeContext, GDNInputs, GDNOutput]
):
    provider = "triton"
    family = AttentionFamily.GDN
    cache_formats = frozenset({"gdn"})

    def __init__(self, model, spec):
        super().__init__(model=model, spec=spec)
        self._init_linear_layer_metadata(model.config, model.tp_world_size_)
        self.prefill_kernel = self.get_prefill_kernel()

    def get_prefill_kernel(self):
        from lightllm.common.attention.backends.gdn.triton.kernels.chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return ReplayGDNPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return ReplayGDNDecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        reqs = context.batch.b_req_idx
        context.b_conv_buffer_idx = reqs
        context.b_ssm_buffer_idx = reqs

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: GDNInputs, *, out=None) -> GDNOutput:
        conv_states, ssm_states = layer.params.conv, layer.params.ssm
        if self.mtp_step > 0:
            conv_states = conv_states[:, :, : -self.mtp_step]
        mixed_qkv, z, b, a = self._split_qkvzba(inputs.mixed_qkvzba)
        core_attn_out = self._prefill(layer, context, mixed_qkv, conv_states, ssm_states, a, b)
        return GDNOutput(core_attn_out, z)

    def _prefill(self, layer, context, mixed_qkv, conv_states, ssm_states, a, b):
        g, beta = fused_gdn_gating(layer.params.a_log, a, b, layer.params.dt_bias)
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.params.conv_weight,
            bias=layer.params.conv_bias,
            query_start_loc=context.batch.b1_cu_q_seq_len,
            cache_indices=context.b_conv_buffer_idx,
            has_initial_state=context.batch.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=self.activation,
        ).transpose(0, 1)
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv)
        core_attn_out, last_state = self.prefill_kernel(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=ssm_states[context.b_ssm_buffer_idx],
            output_final_state=True,
            cu_seqlens=context.batch.b1_cu_q_seq_len,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_states[context.b_ssm_buffer_idx] = last_state.to(ssm_states.dtype, copy=False)
        return core_attn_out

    def _update_decode(self, context):
        update_linear_decode(context, self.decode_draft_step, self.uses_dynamic_spec_verify_layout(), replay=True)

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: GDNInputs, *, out=None) -> GDNOutput:
        mixed_qkv, z, b, a = self._split_qkvzba(inputs.mixed_qkvzba)
        state = layer.params.replay_state
        conv_states = layer.params.conv
        draft_step = self.decode_draft_step
        if draft_step > 0:
            core = self._mtp(layer, context, mixed_qkv, conv_states, a, b, draft_step, state)
        else:
            core, z = self._decode(layer, context, mixed_qkv, z, conv_states, a, b, state)
        # 各层都读到了本步开始时的 cursor。最后一层返回后再按整窗全接受推进。
        state.finish_decode_token(layer.logical_layer_id, context.b_conv_buffer_idx, context.b_cu_seqlens)
        return GDNOutput(core, z)

    def _decode(self, layer, context, mixed_qkv, z, conv_states, a, b, state):
        query, key, value, z, a, b = conv_pack_gdn_decode_inputs(
            mixed_qkv,
            z,
            a,
            b,
            conv_states,
            layer.params.conv_weight,
            layer.params.conv_bias,
            context.b_conv_buffer_idx,
            self.activation,
            self.conv_kernel_dim,
            self.tp_num_k_heads,
            self.head_k_dim,
            self.tp_num_v_heads,
            self.head_v_dim,
        )
        query = query.view(query.shape[0], query.shape[-2], query.shape[-1])
        key = key.view(key.shape[0], key.shape[-2], key.shape[-1])
        value = value.view(value.shape[0], value.shape[-2], value.shape[-1])
        output = replay(
            state,
            state.storage_layer(layer.logical_layer_id),
            query,
            key,
            value,
            a,
            b,
            layer.params.a_log,
            layer.params.dt_bias,
            context.b_conv_buffer_idx,
            context.b_cu_seqlens,
        )
        return output, z

    def _mtp(self, layer, context, mixed_qkv, conv_states, a, b, draft_step, state):
        mixed_qkv = causal_conv1d_update_mtp(
            mixed_qkv,
            conv_states,
            layer.params.conv_weight,
            mtp_step=draft_step,
            bias=layer.params.conv_bias,
            activation=self.activation,
            conv_state_indices=context.b_conv_buffer_idx,
            num_accepted_tokens=context.accepted_tokens(layer.params.mtp_state_index),
            query_start_loc=context.b1_mtp_cu_q_seq_len,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
            dim=-1,
        )
        query = query.view(-1, self.tp_num_k_heads, self.head_k_dim)
        key = key.view(-1, self.tp_num_k_heads, self.head_k_dim)
        value = value.view(-1, self.tp_num_v_heads, self.head_v_dim)
        output = replay(
            state,
            state.storage_layer(layer.logical_layer_id),
            query,
            key,
            value,
            a,
            b,
            layer.params.a_log,
            layer.params.dt_bias,
            context.b_conv_buffer_idx,
            context.b_cu_seqlens,
        )
        return output


class ReplayGDNFlashQlaAttBackend(ReplayGDNTritonAttBackend):
    provider = "flashqla"
    dependency = "flash_qla"
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    supported_phases = frozenset({AttentionPhase.PREFILL})

    def get_prefill_kernel(self):
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule
