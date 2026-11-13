from __future__ import annotations
from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature

"""KDA attention backend for GLM-5-Next."""

import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import KDAInputs
from lightllm.common.attention.backends._shared.linear.layout import LinearDecodeContext, update_linear_decode
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend


@dataclasses.dataclass(slots=True)
class KDAPrefillContext(AttentionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class KDADecodeContext(LinearDecodeContext):
    pass


class TritonKDACommon:
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    provider = "triton"

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    family = AttentionFamily.KDA

    def __init__(self, env, spec):
        super().__init__(env=env, spec=spec)
        config = env.network_config["linear_attn_config"]
        self.num_heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        assert self.num_heads % env.tp_size == 0
        self.tp_num_heads = self.num_heads // env.tp_size
        self.tp_hidden_size = self.tp_num_heads * self.head_dim
        self.conv_kernel_size = config["short_conv_kernel_size"]
        self.lower_bound = config.get("gate_lower_bound", -5.0)
        self.state_slots = env.state_slots


class TritonKDAPrefillBackend(TritonKDACommon, AttentionBackend[KDAPrefillContext, KDAInputs, torch.Tensor]):
    phase = AttentionPhase.PREFILL

    def allocate_context(self, capacity, *, capture, workspace) -> KDAPrefillContext:
        return KDAPrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: KDAPrefillContext):
        context.b_conv_buffer_idx = context.batch.b_req_idx
        context.b_ssm_buffer_idx = context.batch.b_req_idx * self.state_slots

    def forward(
        self, layer: LayerAttention, context: KDAPrefillContext, inputs: KDAInputs, *, out=None
    ) -> torch.Tensor:
        from lightllm.common.attention.backends._shared.linear.causal_conv1d_qkv import causal_conv1d_qkv
        from lightllm.common.attention.backends.kda.triton.kernels.kda import chunk_kda_with_fused_gate

        mixed_qkv = inputs.mixed_qkv
        raw_gate = inputs.raw_gate
        raw_beta = inputs.raw_beta
        conv_states, ssm_states = (layer.resources.conv, layer.resources.ssm)
        conv_states = conv_states[..., : self.conv_kernel_size - 1]
        q, k, v = causal_conv1d_qkv(
            mixed_qkv,
            layer.resources.conv_weight,
            conv_states,
            context.batch.b1_cu_q_seq_len,
            context.b_conv_buffer_idx,
            context.batch.b_ready_cache_len,
            context.batch.max_q_seq_len,
        )
        raw_beta = raw_beta.view(1, -1, self.tp_num_heads)
        initial_state = ssm_states[context.b_ssm_buffer_idx].contiguous()
        output, final_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, -1, self.tp_num_heads, self.head_dim),
            beta=raw_beta.float().sigmoid(),
            A_log=layer.resources.a_log,
            g_bias=layer.resources.dt_bias,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=context.batch.b1_cu_q_seq_len,
            safe_gate=True,
            lower_bound=self.lower_bound,
        )
        ssm_states[context.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype, copy=False)
        return output


class TritonKDADecodeBackend(TritonKDACommon, AttentionBackend[KDADecodeContext, KDAInputs, torch.Tensor]):
    phase = AttentionPhase.DECODE

    def split_qkv(self, mixed_qkv: torch.Tensor):
        return mixed_qkv.split(self.tp_hidden_size, dim=-1)

    def allocate_context(self, capacity, *, capture, workspace) -> KDADecodeContext:
        return KDADecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: KDADecodeContext):
        update_linear_decode(context, self.draft_step, self.spec.dynamic_verify, replay=False)

    def forward(self, layer: LayerAttention, context: KDADecodeContext, inputs: KDAInputs, *, out=None) -> torch.Tensor:
        from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_update
        from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
            causal_conv1d_update as causal_conv1d_update_mtp,
        )
        from lightllm.common.attention.backends.kda.triton.kernels.kda_decode import fused_recurrent_kda

        mixed_qkv = inputs.mixed_qkv
        raw_gate = inputs.raw_gate
        raw_beta = inputs.raw_beta
        conv_states, ssm_states = (layer.resources.conv, layer.resources.ssm)
        conv_kwargs = dict(bias=None, activation="silu", conv_state_indices=context.b_conv_buffer_idx)
        conv_update = causal_conv1d_update
        if self.draft_step > 0:
            conv_update = causal_conv1d_update_mtp
            conv_kwargs.update(
                mtp_step=self.draft_step,
                num_accepted_tokens=context.accepted_tokens(layer.resources.mtp_state_index),
                query_start_loc=context.b1_mtp_cu_q_seq_len,
            )
        mixed_qkv = conv_update(mixed_qkv, conv_states, layer.resources.conv_weight, **conv_kwargs)
        q, k, v = self.split_qkv(mixed_qkv)
        shape = (1, -1) if self.draft_step > 0 else (-1, 1)
        q = q.view(*shape, self.tp_num_heads, self.head_dim)
        k = k.view(*shape, self.tp_num_heads, self.head_dim)
        v = v.view(*shape, self.tp_num_heads, self.head_dim)
        raw_gate = raw_gate.view(*shape, self.tp_hidden_size)
        raw_beta = raw_beta.view(*shape, self.tp_num_heads)
        output, _ = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            a_log=layer.resources.a_log,
            gate_bias=layer.resources.dt_bias,
            initial_state=ssm_states,
            lower_bound=self.lower_bound,
            inplace_final_state=True,
            ssm_state_indices=context.b_ssm_buffer_idx,
            cu_seqlens=context.b1_mtp_cu_q_seq_len,
            num_accepted_tokens=(context.accepted_tokens(layer.resources.mtp_state_index) if self.draft_step else None),
        )
        return output
